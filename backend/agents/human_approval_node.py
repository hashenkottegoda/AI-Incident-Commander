"""HUMAN APPROVAL — Phase 6's `interrupt()` gate.

BUILD_PLAN.md's Agent Architecture section, verbatim: *"HIGH-IMPACT
(rollback/restart/scale/config/disable) -> HUMAN APPROVAL (LangGraph
`interrupt`, resumed via POST /approve|/reject)"* and, on safety: *"
`interrupt()` must be side-effect safe. LangGraph re-executes the
interrupted node from its start when the graph resumes, so nothing
irreversible may happen before the `interrupt()` call -- the Action
Executor runs strictly after resume, never before."*

## Why this is its own node, not folded into `response_planner_node`

`response_planner_node` already performs this graph's first real side
effect: it creates one `AuditEvent` row per recommended action and
commits (see that module's docstring). If `interrupt()` were called from
*inside* that same node (after the `AuditEvent` writes), resuming the
graph would re-execute the node **from its start** -- re-running the LLM
call, re-classifying every action, and re-inserting a brand new set of
`AuditEvent` rows for the same recommendations, duplicating the audit
trail on every single resume. That is exactly the failure mode
BUILD_PLAN.md's warning above is describing.

Splitting `interrupt()` into this dedicated, separate node avoids the
problem entirely rather than working around it: `response_planner_node`
runs to completion and commits *before* this node ever starts, so by the
time `human_approval_node` calls `interrupt()`, the `AuditEvent` rows
already exist, are durable, and are never touched again by anything in
this module. When LangGraph replays this node from its start on resume,
there is nothing above the `interrupt()` call to re-run except reading
already-computed `IncidentState` fields (pure, no I/O) -- so replay is
trivially safe. This is "option (a)" from the task brief: a dedicated
node whose only job is to pause, chosen over making the `AuditEvent`
creation itself idempotent (option (b)), because it needs no defensive
check-then-create logic anywhere -- the correctness argument is "this
function does nothing before `interrupt()` except read state," which is
true by inspection rather than by a guard that has to keep being right as
the code evolves.

## What this node does NOT do

No database session, no LLM call, no `AuditEvent` mutation -- deciding
*what* happened (approved vs. rejected, by whom, when) and durably
recording that decision on the `AuditEvent` row(s) is entirely owned by
`backend.api.approvals` (`POST /approve` / `POST /reject`), which updates
the database in the SAME request that issues the `Command(resume=...)`
call, strictly BEFORE this node resumes past `interrupt()` (approval) --
or, for a rejection, without ever resuming this node at all (see that
module's docstring for why rejection doesn't need to touch this node).
By the time this node's code below `interrupt()` runs, the approval
decision is already durable; this node only carries the raw resume
payload into `IncidentState` so the rest of the graph (and tests
inspecting the final state) can see what happened without a second
database round trip.

## What happens below `interrupt()`

On an approved resume, this node just records `approval_decision =
"approved"` and hands control to `backend/graph.py`'s unconditional
`human_approval -> action_executor` edge -- the real Action Executor
(`backend.agents.action_executor_node`) runs immediately after, reading
the now-`APPROVED` `AuditEvent` row(s) `backend.api.approvals` committed
before issuing the resume. There is no re-execution-safety concern with
this node's own code (it does nothing but read state below `interrupt()`):
it only runs once `interrupt()` has already returned a resume value on
this pass, and (per this module's own design) this node is only ever
resumed once per incident -- see `backend.api.approvals`'s idempotency
guard for why a second `/approve` call never reaches this node again, and
`action_executor_node`'s own idempotency guard (only `APPROVED`/
`AUTO_EXECUTED` rows are actionable) for why a replayed pass through this
node can never cause a second execution either.

The defensive non-approved fallback branch below is never exercised by the
real API flow (`POST /reject` never resumes the graph at all -- see
`backend.api.approvals`'s docstring), but stays correct even if hit: it
sets `incident_status = MANUAL_INTERVENTION_REQUIRED` without ever putting
the corresponding `AuditEvent` row into `APPROVED`, so `action_executor_node`
(which only acts on `APPROVED`/`AUTO_EXECUTED` rows) finds nothing to
execute for it even though the graph's unconditional edge still visits
that node.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from backend.agents.state import IncidentState
from backend.models import IncidentStatus


def human_approval_node(state: IncidentState) -> dict[str, Any]:
    """Pause the graph until `POST /approve` resumes it.

    Not a factory (`make_human_approval_node()`) like the LLM-calling
    nodes -- this function closes over nothing (no `db`, no LLM client),
    so there is nothing request-scoped to bind ahead of time.

    The `interrupt()` payload is a snapshot of what's pending, surfaced to
    whatever inspects the paused thread (a future dashboard, or a test
    asserting the graph genuinely halted) -- it is informational only,
    never the durable record of the decision (that's the `AuditEvent`
    rows `response_planner_node` already committed).
    """
    pending_actions = [
        action
        for action in state.recommended_actions
        if action.get("decision_status") == "pending_approval"
    ]
    decision = interrupt(
        {
            "incident_id": state.incident_id,
            "pending_actions": pending_actions,
        }
    )

    decision_value = decision.get("decision") if isinstance(decision, dict) else None
    if decision_value == "approved":
        return {
            "approval_decision": "approved",
            # Real (not placeholder) transitional state now: the graph's
            # unconditional human_approval -> action_executor edge
            # (backend/graph.py) runs immediately after this, which
            # overwrites incident_status to VERIFYING (HIGH_IMPACT
            # remediation just executed, pending Recovery Check) or
            # DIAGNOSED (shouldn't happen on this branch -- reaching
            # human_approval at all implies a HIGH_IMPACT action existed).
            "incident_status": IncidentStatus.EXECUTING,
        }
    # Defensive fallback only: in the actual approval flow this graph is
    # never resumed with a rejection (see backend.api.approvals -- a
    # rejection is decided and recorded entirely by POST /reject WITHOUT
    # ever calling Command(resume=...), so this node never re-executes for
    # a rejected incident at all). This branch exists so the node stays
    # correct even if something resumes it with an unexpected/non-approval
    # payload, rather than silently defaulting to "approved".
    return {
        "approval_decision": "rejected",
        "incident_status": IncidentStatus.MANUAL_INTERVENTION_REQUIRED,
    }
