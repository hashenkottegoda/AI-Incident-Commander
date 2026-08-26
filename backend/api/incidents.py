"""`POST /api/incidents/{id}/investigate` — Phase 3's investigator endpoint,
Phase 5's full-graph endpoint, and Phase 8's read endpoints for the
dashboard.

BUILD_PLAN.md Phase 3: *"Wire to `POST /api/incidents/{id}/investigate`."*
Kept thin on purpose: load the `Incident` (404 if missing), run the
single-node ReAct investigator, return its `DiagnosisResult` as the
response body. No incident_status transition or extra persistence here —
BUILD_PLAN.md scopes Phase 3 to validating tool-calling mechanics
end-to-end; wiring `DiagnosisResult` into the `Incident` lifecycle
(`triaging -> investigating -> diagnosed -> ...`) is Phase 5's job once
the full graph exists.

`/investigate` calls `investigate_incident(db, incident)` with no
`include_rag` argument, so it runs with the default `include_rag=True` --
Experiment C's configuration (tools + historical incidents), not
Experiment B. Phase 7's eval harness drives Experiment B vs. C by calling
`investigate_incident()` directly with an explicit `include_rag=` on each
side, not through this HTTP route — see that function's docstring.
`POST /{incident_id}/investigate/graph` is the new Phase 5 route: runs the full
`StateGraph` (Triage -> Investigation loop -> RAG -> Root Cause, with the
bounded conditional re-investigation loop) via `backend.graph.
run_incident_graph`, and returns the same shared `DiagnosisResult` shape
built from the graph's final `IncidentState` — same response contract as
`/investigate`, different architecture producing it (BUILD_PLAN.md: "All
four experiments emit the same DiagnosisResult schema").

## Phase 8's read endpoints: `GET /api/incidents` and `GET /{incident_id}`

Every route above this point is an *action* -- there was previously no way
to list incidents or read one incident's current state over HTTP at all,
which the React dashboard (Phase 8) needs before any of its views (incident
list, investigation trace, evidence panel, root-cause/confidence view,
approval/execution/recovery status) can be built.

`GET /api/incidents` reads the `Incident` table directly (no LangGraph
involved) -- a plain paginated/filterable list, most-recent-first.

`GET /{incident_id}` combines three sources into one response so the
dashboard's detail views need exactly one call each incident:
1. The `Incident` row itself.
2. The graph's checkpointed `IncidentState` for this incident's real
   operational thread (`backend.graph.get_incident_thread_state`, the same
   `thread_id = str(incident.id)` convention `run_incident_graph`/
   `resume_incident_graph` use) -- evidence, hypotheses, root cause,
   confidence, recommended actions, approval decision, execution/recovery
   refs. An incident that's been injected but never had `/investigate` or
   `/investigate/graph` called on it has no checkpoint at all --
   `get_incident_thread_state` returns a `StateSnapshot` with `values == {}`
   in that case (verified directly against a real thread with no prior
   graph run, not assumed), which is a normal, expected state for a
   freshly-injected incident, not an error: `investigation` is `None` in
   the response rather than the endpoint raising or fabricating zeros.
3. `AuditEvent` rows for the incident (Phase 6's audit trail) -- the
   approve/reject buttons and execution/recovery status views need
   pending-vs-decided-vs-executed action state, in `recommended_at` order
   (matching `ix_audit_events_incident_id_recommended_at`).

Both endpoints extend this existing router rather than a new file/module --
matching this task's instruction and the fact that they're one more
`Incident`-scoped concern alongside `/investigate`.

## `GET /{incident_id}/progress` (Phase 8 step 2 of 3)

Step 1 (committed separately) added `backend.models.node_progress.
NodeProgressEvent` and instrumented every graph node
(`backend.graph._with_progress`) to write one row per invocation. This is
the read side: the ordered `node_name`/`started_at` list the dashboard
polls to render a live investigation trace, deliberately its own route
rather than a field folded into `IncidentDetail` above. `IncidentDetail` is
a heavier, multi-source payload (checkpoint state plus the full audit
trail) meant for a single per-incident detail view; a live trace is
instead something the dashboard polls repeatedly *while an investigation
might still be running*, so it gets a small, fast, single-table endpoint
of its own rather than making every poll re-fetch (and re-serialize) the
whole detail payload. 404s the same way every other `{incident_id}` route
here does if the incident itself doesn't exist; a real incident with zero
progress rows (never investigated) is a normal `200 []`, not an error --
same "empty is not an error" precedent `investigation: null` already
establishes for `GET /{incident_id}`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload

from backend.agents.investigator import investigate_incident
from backend.agents.schemas import DiagnosisResult, EvidenceItem, Hypothesis, RootCauseCategory
from backend.agents.state import IncidentState
from backend.db import get_db
from backend.graph import get_incident_thread_state as _get_incident_thread_state
from backend.graph import run_incident_graph
from backend.models import (
    AuditDecisionStatus,
    AuditEvent,
    ExecutionOutcome,
    Incident,
    IncidentStatus,
    NodeProgressEvent,
    RiskClassification,
    Severity,
)

router = APIRouter(prefix="/api/incidents", tags=["incidents"])


def _get_incident_or_404(incident_id: int, db: Session) -> Incident:
    incident = db.get(Incident, incident_id)
    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"incident {incident_id} not found",
        )
    return incident


# --- Read endpoints (Phase 8) ------------------------------------------------


class IncidentSummary(BaseModel):
    """One row of `GET /api/incidents` -- just enough for a list view, not
    the full detail shape (`IncidentDetail` below)."""

    id: int
    status: IncidentStatus
    severity: Severity
    failure_type: str
    detected_at: datetime
    service_id: int
    service_name: str


class IncidentInfo(IncidentSummary):
    """The `incident` sub-section of `GET /{incident_id}` -- `IncidentSummary`
    plus the injected ground-truth `root_cause_category`
    (`backend/models/incident.py`), useful for a dashboard that wants to
    show predicted-vs-actual. Left out of the plain list view on purpose --
    BUILD_PLAN.md's eval harness is the sanctioned ground-truth consumer;
    the list view has no need for it and this keeps that summary lean."""

    root_cause_category: str


class InvestigationState(BaseModel):
    """The `investigation` sub-section of `GET /{incident_id}` -- the subset
    of `backend.agents.state.IncidentState`'s fields the dashboard needs,
    same field names as that class. `None` at the top level (not this
    model) means no checkpoint exists yet for this incident -- see module
    docstring.

    `incident_status` here is the graph's OWN live-checkpoint phase, and is
    deliberately not the same value as `IncidentSummary.status` (the DB
    row's `Incident.status` column) -- the two can and do diverge.
    `triage_node`/`investigation_node`/`root_cause_node`/
    `response_planner_node` only ever set `incident_status` inside the
    LangGraph checkpoint dict; only `action_executor_node`,
    `recovery_check_node`, and `backend.api.approvals` write the DB column.
    So an incident that's mid-investigation, diagnosed-but-not-yet-planned,
    or awaiting approval still reads back `Incident.status == "detected"`
    from the DB, while THIS field correctly shows `"investigating"`,
    `"diagnosed"`, or `"awaiting_approval"`. A dashboard that wants the
    incident's real current phase should read `investigation.
    incident_status` when it's present (i.e. a checkpoint exists), falling
    back to `incident.status` only when `investigation` is `None`."""

    incident_status: IncidentStatus
    evidence: list[EvidenceItem]
    hypotheses: list[Hypothesis]
    root_cause: RootCauseCategory | None
    diagnostic_confidence: float
    alternative_hypotheses: list[Hypothesis]
    recommended_actions: list[dict[str, Any]]
    approval_decision: str | None
    execution_result_id: int | list[int] | None
    recovery_result: dict[str, Any] | None

    @classmethod
    def from_incident_state(cls, state: IncidentState) -> InvestigationState:
        return cls(
            incident_status=state.incident_status,
            evidence=state.evidence,
            hypotheses=state.hypotheses,
            root_cause=state.root_cause,
            diagnostic_confidence=state.diagnostic_confidence,
            alternative_hypotheses=state.alternative_hypotheses,
            recommended_actions=state.recommended_actions,
            approval_decision=state.approval_decision,
            execution_result_id=state.execution_result_id,
            recovery_result=state.recovery_result,
        )


class AuditEventSummary(BaseModel):
    """One `AuditEvent` row (`backend/models/audit.py`), same column names
    as that model."""

    id: int
    action_type: str
    risk_classification: RiskClassification
    decision_status: AuditDecisionStatus
    approver: str | None
    execution_outcome: ExecutionOutcome | None
    execution_detail: dict[str, Any] | None
    recommended_at: datetime
    decided_at: datetime | None
    executed_at: datetime | None

    @classmethod
    def from_audit_event(cls, event: AuditEvent) -> AuditEventSummary:
        return cls(
            id=event.id,
            action_type=event.action_type,
            risk_classification=event.risk_classification,
            decision_status=event.decision_status,
            approver=event.approver,
            execution_outcome=event.execution_outcome,
            execution_detail=event.execution_detail,
            recommended_at=event.recommended_at,
            decided_at=event.decided_at,
            executed_at=event.executed_at,
        )


class NodeProgressEventSummary(BaseModel):
    """One `NodeProgressEvent` row (`backend/models/node_progress.py`) --
    just `node_name`/`started_at`, matching that model's fields (`id`/
    `incident_id` are the caller's own request context, not useful in the
    response body)."""

    node_name: str
    started_at: datetime

    @classmethod
    def from_node_progress_event(cls, event: NodeProgressEvent) -> NodeProgressEventSummary:
        return cls(node_name=event.node_name, started_at=event.started_at)


class IncidentDetail(BaseModel):
    """`GET /{incident_id}`'s full response -- grouped into the same three
    sources described in the module docstring, not a flat namespace."""

    incident: IncidentInfo
    investigation: InvestigationState | None
    audit_events: list[AuditEventSummary]


def _incident_summary(incident: Incident) -> IncidentSummary:
    return IncidentSummary(
        id=incident.id,
        status=incident.status,
        severity=incident.severity,
        failure_type=incident.failure_type,
        detected_at=incident.detected_at,
        service_id=incident.service_id,
        service_name=incident.service.name,
    )


def _incident_info(incident: Incident) -> IncidentInfo:
    return IncidentInfo(
        **_incident_summary(incident).model_dump(),
        root_cause_category=incident.root_cause_category,
    )


@router.get("", response_model=list[IncidentSummary])
def list_incidents(
    status_filter: IncidentStatus | None = Query(  # noqa: B008
        default=None, alias="status", description="Filter to incidents with this lifecycle status."
    ),
    limit: int = Query(default=50, ge=1, le=200),  # noqa: B008
    offset: int = Query(default=0, ge=0),  # noqa: B008
    db: Session = Depends(get_db),  # noqa: B008
) -> list[IncidentSummary]:
    """List incidents, most-recently-detected first.

    Reads the `Incident` table directly -- no LangGraph state involved (see
    `GET /{incident_id}` for that). `?status=` lets the dashboard's landing
    page filter to open/in-progress incidents instead of showing every
    incident ever generated, including Phase 7's eval-benchmark noise
    (`alias="status"` so the query string stays `?status=...` while the
    Python parameter name doesn't shadow `fastapi.status`, used elsewhere
    in this module). `limit`/`offset` is a plain, unopinionated pagination
    convention -- no other endpoint in this codebase paginates yet, so
    there was no existing convention to match.

    Caveat: `?status=` filters on the DB `Incident.status` column, which
    only advances at `action_executor_node`/`recovery_check_node`/
    `backend.api.approvals` -- `triage_node`/`investigation_node`/
    `root_cause_node`/`response_planner_node` never write it (see
    `InvestigationState`'s docstring below for the full explanation). So
    `?status=investigating`/`diagnosed`/`awaiting_approval` will never
    match a real in-flight incident; only `detected` (everything not yet
    executed) and the post-execution terminal statuses are reliably
    filterable here. A dashboard wanting the live phase for a specific
    incident should use `GET /{incident_id}` and read `investigation.
    incident_status` instead.
    """
    query = db.query(Incident).options(joinedload(Incident.service))
    if status_filter is not None:
        query = query.filter(Incident.status == status_filter)
    incidents = query.order_by(Incident.detected_at.desc()).limit(limit).offset(offset).all()
    return [_incident_summary(incident) for incident in incidents]


@router.get("/{incident_id}", response_model=IncidentDetail)
async def get_incident(incident_id: int, db: Session = Depends(get_db)) -> IncidentDetail:  # noqa: B008
    """One incident's full current picture: the `Incident` row, the graph's
    checkpointed investigation state (if any graph run has ever happened
    for it), and its full `AuditEvent` trail. See module docstring for how
    the three are combined and why "no checkpoint yet" is a normal, not an
    error, state.
    """
    incident = _get_incident_or_404(incident_id, db)

    snapshot = await _get_incident_thread_state(db, incident)
    investigation = (
        InvestigationState.from_incident_state(IncidentState.model_validate(snapshot.values))
        if snapshot.values
        else None
    )

    audit_events = (
        db.query(AuditEvent)
        .filter(AuditEvent.incident_id == incident_id)
        .order_by(AuditEvent.recommended_at)
        .all()
    )

    return IncidentDetail(
        incident=_incident_info(incident),
        investigation=investigation,
        audit_events=[AuditEventSummary.from_audit_event(event) for event in audit_events],
    )


@router.get("/{incident_id}/progress", response_model=list[NodeProgressEventSummary])
def get_incident_progress(
    incident_id: int, db: Session = Depends(get_db)  # noqa: B008
) -> list[NodeProgressEventSummary]:
    """The ordered live-trace progress log for `incident_id` -- one row per
    graph-node invocation, oldest first, so a client can render it
    top-to-bottom as a timeline. See module docstring for why this is a
    dedicated lightweight route rather than a field on `GET /{incident_id}`,
    and why a real incident with zero rows (never investigated) is a normal
    `200 []`, not a 404. A resumed `/approve` call can legitimately produce
    a second row for the same `node_name` (e.g. two `human_approval` rows)
    -- see `backend.graph._with_progress`'s docstring -- so rows are
    returned as-is, not deduplicated by node name.
    """
    _get_incident_or_404(incident_id, db)

    events = (
        db.query(NodeProgressEvent)
        .filter(NodeProgressEvent.incident_id == incident_id)
        .order_by(NodeProgressEvent.started_at, NodeProgressEvent.id)
        .all()
    )
    return [NodeProgressEventSummary.from_node_progress_event(event) for event in events]


@router.post("/{incident_id}/investigate", response_model=DiagnosisResult)
def investigate(incident_id: int, db: Session = Depends(get_db)) -> DiagnosisResult:  # noqa: B008
    """Run the tool-using investigator (Experiment C's configuration --
    tools + RAG, since `include_rag` defaults to `True`) against
    `incident_id`."""
    incident = _get_incident_or_404(incident_id, db)
    return investigate_incident(db, incident)


@router.post("/{incident_id}/investigate/graph", response_model=DiagnosisResult)
async def investigate_graph(incident_id: int, db: Session = Depends(get_db)) -> DiagnosisResult:  # noqa: B008
    """Run Phase 5's full orchestrated graph (Triage -> Investigation loop ->
    RAG -> Root Cause) against `incident_id` and return the resulting
    `DiagnosisResult`.

    `db: Session` is a sync dependency in this async route -- FastAPI runs
    sync dependency callables in a threadpool automatically, same as every
    other route in this module; only the graph invocation itself
    (`run_incident_graph`, which needs `AsyncPostgresSaver`) is async.
    """
    incident = _get_incident_or_404(incident_id, db)
    final_state = await run_incident_graph(db, incident)
    return DiagnosisResult(
        root_cause_category=final_state.root_cause or "unknown",
        hypotheses=final_state.hypotheses,
        alternative_hypotheses=final_state.alternative_hypotheses,
        evidence=final_state.evidence,
        diagnostic_confidence=final_state.diagnostic_confidence,
    )
