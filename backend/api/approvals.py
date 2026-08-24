"""`POST /api/incidents/{id}/approve` and `POST /api/incidents/{id}/reject`
-- Phase 6's human-approval decision endpoints.

BUILD_PLAN.md's Agent Architecture section: *"HIGH-IMPACT ... -> HUMAN
APPROVAL (LangGraph `interrupt`, resumed via POST /approve|/reject)"*, and,
on idempotency: *"The approval flow is idempotent: resuming (or a duplicate
`/approve`) re-enters the node but guards on `incident_status`/
`execution_result_id` so the remediation executes at most once."* Phase 6's
own verification list adds: *"a duplicate POST /approve on an
already-resolved thread is idempotent -- it does not execute the action
twice (guard on incident_status)."*

## Who owns what write

This module is the **sole writer** of `AuditEvent.decision_status` /
`.approver` / `.decided_at` for a HIGH_IMPACT action -- not
`backend.agents.human_approval_node`, which deliberately has no database
access at all (see that node's docstring for why). The approval decision
is recorded here, durably, in the SAME request that (for an approval) goes
on to call `backend.graph.resume_incident_graph`. This ordering matters:
by the time the graph resumes past `interrupt()`, the decision this
endpoint made is already committed, so a crash or duplicate request after
that point can never lose or re-decide it.

The one gap this ordering doesn't close on its own: if the process crashes
or the resume call fails *between* the `AuditEvent` commit and
`resume_incident_graph` returning, a naive "no PENDING_APPROVAL rows left"
check would report "already decided" forever without ever actually
resuming the thread -- a permanently stuck incident with no retry path.
`_retry_stuck_resume` closes this: before reporting "already decided" on
an approve call, it checks whether the checkpointed thread is still
sitting at `human_approval` despite an APPROVED audit row existing, and
if so, retries the resume (safe to retry freely -- the decision is already
durable, only the resume itself was ever incomplete).

## Approver identity

BUILD_PLAN.md: *"even a stubbed/header-supplied user is enough (full RBAC
is deliberately cut) so 'only authorized users approve' has a real hook."*
This module takes `approver` as a plain request-body field
(`ApprovalRequest.approver`) rather than a header -- a request body field
is validated by the same Pydantic machinery as every other request in this
codebase (`FailureInjectionRequest` etc.) and needs no new header-parsing
convention; swapping it for a real `X-User` header (or real auth) later is
a one-line change to this schema, not a structural one.

## Idempotency: optimistic concurrency via `AuditEvent.version_id`

Both endpoints share `_decide_pending_actions`, which:

1. Reads every `AuditEvent` row for this incident still in
   `PENDING_APPROVAL` (a plain `SELECT`, deliberately **not**
   `SELECT ... FOR UPDATE` -- see below).
2. If none are pending, this is a no-op: either nothing was ever pending
   for this incident, or another request already decided it. Either way,
   the correct response is "here's the current state," not an error and
   not a second decision.
3. Otherwise, mutates each row's `decision_status`/`approver`/`decided_at`
   in memory and commits. `AuditEvent.version_id` (see
   `backend/models/audit.py`'s docstring) makes this commit safe under a
   **real** race: two concurrent requests can both complete step 1 seeing
   `PENDING_APPROVAL` before either commits (step 1 takes no lock, by
   design -- an explicit `FOR UPDATE` would just serialize the two
   requests and hide the race rather than proving the guard works). The
   first commit to land bumps `version_id`; the second commit's `UPDATE`
   is scoped to the *stale* `version_id` it originally read, matches zero
   rows, and SQLAlchemy raises `StaleDataError` -- caught here and treated
   exactly like "nothing was pending": re-read and return the current
   (now-decided) state rather than re-deciding or double-executing
   anything. This is chosen over an atomic conditional `UPDATE ... WHERE
   decision_status = 'pending_approval'` for the same outcome by a
   different, already-present mechanism: `version_id_col` was already
   built and documented in Phase 6's audit-model step specifically for
   this guard, so using it here is exercising existing machinery rather
   than introducing a second, parallel idempotency mechanism.

## Approve vs. reject: only approve touches the graph

`POST /approve` resumes the paused thread via
`backend.graph.resume_incident_graph` (`Command(resume=...)`) -- the
`AuditEvent` write above already happened and committed, so resuming is
safe per that function's docstring.

`POST /reject` does **not** resume the graph at all. BUILD_PLAN.md:
*"Rejected approval -> `manual_intervention_required` ... do NOT resume
toward execution."* The simplest, most literal way to guarantee a
rejection never proceeds toward execution is to never hand control back to
the graph in the first place -- `incident.status` is set to
`MANUAL_INTERVENTION_REQUIRED` directly by this endpoint, and the paused
thread is simply left interrupted (its checkpoint persists in Postgres;
nothing reads or resumes it again for this incident). This also means a
rejection can never accidentally exercise `human_approval_node`'s
approved-branch code, by construction, not by a runtime check.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from backend.db import get_db
from backend.graph import get_incident_thread_state, resume_incident_graph
from backend.models import AuditDecisionStatus, AuditEvent, Incident, IncidentStatus

router = APIRouter(prefix="/api/incidents", tags=["approvals"])

# AuditEvent rows in any of these states have already been decided (or, for
# AUTO_EXECUTED/EXECUTED, were never a HIGH_IMPACT decision in the first
# place) -- surfaced back to a caller on the idempotent "nothing to decide"
# path so the response can still report *something* meaningful about the
# incident's audit trail rather than an empty list.
_DECIDED_STATUSES = (
    AuditDecisionStatus.APPROVED,
    AuditDecisionStatus.REJECTED,
    AuditDecisionStatus.AUTO_EXECUTED,
    AuditDecisionStatus.EXECUTED,
)


class ApprovalRequest(BaseModel):
    """Body for both `POST /approve` and `POST /reject`."""

    approver: str = Field(
        ..., min_length=1, description="Stubbed/header-free approver identity (no RBAC yet)."
    )


class ApprovalResponse(BaseModel):
    """Result of an approve/reject decision (or a no-op replay of one)."""

    incident_id: int
    decision: Literal["approved", "rejected", "already_decided"]
    incident_status: IncidentStatus
    audit_event_ids: list[int]
    approver: str | None
    decided_at: datetime | None
    # True only when this call actually resumed the paused graph thread
    # (i.e. a genuine first-time approval). False for rejections (which
    # never resume the graph, by design -- see module docstring) and for
    # the idempotent "already decided" no-op path.
    resumed: bool


def _get_incident_or_404(incident_id: int, db: Session) -> Incident:
    incident = db.get(Incident, incident_id)
    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"incident {incident_id} not found",
        )
    return incident


def _already_decided_response(incident: Incident, db: Session) -> ApprovalResponse:
    """Build the idempotent no-op response: nothing was PENDING_APPROVAL,
    either because this incident never had a HIGH_IMPACT action or because
    a previous (possibly concurrent) request already decided it."""
    decided = (
        db.query(AuditEvent)
        .filter(
            AuditEvent.incident_id == incident.id,
            AuditEvent.decision_status.in_(_DECIDED_STATUSES),
        )
        .order_by(AuditEvent.id)
        .all()
    )
    last_approver = next((e.approver for e in reversed(decided) if e.approver), None)
    last_decided_at = next((e.decided_at for e in reversed(decided) if e.decided_at), None)
    return ApprovalResponse(
        incident_id=incident.id,
        decision="already_decided",
        incident_status=incident.status,
        audit_event_ids=[e.id for e in decided],
        approver=last_approver,
        decided_at=last_decided_at,
        resumed=False,
    )


async def _retry_stuck_resume(
    incident: Incident, approver: str, db: Session
) -> ApprovalResponse | None:
    """Liveness guard for a narrow but real failure window: a previous
    `/approve` call can durably commit the AuditEvent decision and then
    fail (checkpointer error, connection drop, process crash) before
    `resume_incident_graph` completes. Without this check, every
    subsequent `/approve` call would see zero PENDING_APPROVAL rows,
    conclude "already decided," and never retry the resume -- leaving the
    graph thread permanently parked at `interrupt()` with no way to
    unstick it via the API.

    Detects that specific stuck state -- an APPROVED AuditEvent exists for
    this incident, but the checkpointed thread is still sitting at
    `human_approval` -- and retries the resume. This is safe to retry
    freely: the decision itself is already durable and correct (this
    function never re-decides anything), so completing the resume now is
    exactly the action that should have finished the first time. Returns
    `None` when there is nothing stuck to retry, so the caller falls
    through to the normal `_already_decided_response` path.
    """
    has_approved = (
        db.query(AuditEvent)
        .filter(
            AuditEvent.incident_id == incident.id,
            AuditEvent.decision_status == AuditDecisionStatus.APPROVED,
        )
        .first()
        is not None
    )
    if not has_approved:
        return None

    try:
        snapshot = await get_incident_thread_state(db, incident)
    except Exception:  # noqa: BLE001 - can't confirm a stuck state, don't guess
        return None
    if "human_approval" not in (snapshot.next or ()):
        return None  # already resumed correctly; nothing stuck

    final_state = await resume_incident_graph(
        db, incident, {"decision": "approved", "approver": approver}
    )
    incident.status = final_state.incident_status
    db.commit()

    decided = (
        db.query(AuditEvent)
        .filter(
            AuditEvent.incident_id == incident.id,
            AuditEvent.decision_status.in_(_DECIDED_STATUSES),
        )
        .order_by(AuditEvent.id)
        .all()
    )
    last_approver = next((e.approver for e in reversed(decided) if e.approver), approver)
    last_decided_at = next((e.decided_at for e in reversed(decided) if e.decided_at), None)
    return ApprovalResponse(
        incident_id=incident.id,
        decision="approved",
        incident_status=incident.status,
        audit_event_ids=[e.id for e in decided],
        approver=last_approver,
        decided_at=last_decided_at,
        resumed=True,
    )


async def _decide_pending_actions(
    incident_id: int,
    approver: str,
    decision: AuditDecisionStatus,
    db: Session,
) -> ApprovalResponse:
    incident = _get_incident_or_404(incident_id, db)

    # Deliberately a plain SELECT (no FOR UPDATE) -- see module docstring
    # for why the version_id optimistic-concurrency guard below needs the
    # race window this leaves open, rather than closing it with a lock.
    pending = (
        db.query(AuditEvent)
        .filter(
            AuditEvent.incident_id == incident_id,
            AuditEvent.decision_status == AuditDecisionStatus.PENDING_APPROVAL,
        )
        .order_by(AuditEvent.id)
        .all()
    )

    if not pending:
        if decision is AuditDecisionStatus.APPROVED:
            retried = await _retry_stuck_resume(incident, approver, db)
            if retried is not None:
                return retried
        return _already_decided_response(incident, db)

    now = datetime.now(UTC)
    for event in pending:
        event.decision_status = decision
        event.approver = approver
        event.decided_at = now

    try:
        db.commit()
    except StaleDataError:
        # A concurrent request already decided at least one of these rows
        # between our SELECT and our COMMIT -- someone else won the race.
        # Treat exactly like the "nothing pending" case: roll back our
        # (now-invalid) in-memory changes and report current reality.
        db.rollback()
        return _already_decided_response(incident, db)

    audit_event_ids = [event.id for event in pending]

    if decision is AuditDecisionStatus.REJECTED:
        incident.status = IncidentStatus.MANUAL_INTERVENTION_REQUIRED
        db.commit()
        return ApprovalResponse(
            incident_id=incident_id,
            decision="rejected",
            incident_status=incident.status,
            audit_event_ids=audit_event_ids,
            approver=approver,
            decided_at=now,
            resumed=False,
        )

    # Approved: the decision is already durable above -- now resume the
    # exact paused thread. `human_approval_node`'s only remaining job is to
    # turn this resume payload into a placeholder post-approval
    # incident_status (see that node's docstring).
    final_state = await resume_incident_graph(
        db, incident, {"decision": "approved", "approver": approver}
    )
    incident.status = final_state.incident_status
    db.commit()
    return ApprovalResponse(
        incident_id=incident_id,
        decision="approved",
        incident_status=incident.status,
        audit_event_ids=audit_event_ids,
        approver=approver,
        decided_at=now,
        resumed=True,
    )


@router.post("/{incident_id}/approve", response_model=ApprovalResponse)
async def approve(
    incident_id: int,
    payload: ApprovalRequest,
    db: Session = Depends(get_db),  # noqa: B008
) -> ApprovalResponse:
    """Approve incident_id's pending HIGH_IMPACT action(s) and resume its
    paused graph thread."""
    return await _decide_pending_actions(
        incident_id, payload.approver, AuditDecisionStatus.APPROVED, db
    )


@router.post("/{incident_id}/reject", response_model=ApprovalResponse)
async def reject(
    incident_id: int,
    payload: ApprovalRequest,
    db: Session = Depends(get_db),  # noqa: B008
) -> ApprovalResponse:
    """Reject incident_id's pending HIGH_IMPACT action(s). Does NOT resume
    the graph -- see module docstring."""
    return await _decide_pending_actions(
        incident_id, payload.approver, AuditDecisionStatus.REJECTED, db
    )
