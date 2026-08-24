"""`POST /api/incidents/{id}/investigate` — Phase 3's investigator endpoint,
plus Phase 5's full-graph endpoint.

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
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.agents.investigator import investigate_incident
from backend.agents.schemas import DiagnosisResult
from backend.db import get_db
from backend.graph import run_incident_graph
from backend.models import Incident

router = APIRouter(prefix="/api/incidents", tags=["incidents"])


def _get_incident_or_404(incident_id: int, db: Session) -> Incident:
    incident = db.get(Incident, incident_id)
    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"incident {incident_id} not found",
        )
    return incident


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
