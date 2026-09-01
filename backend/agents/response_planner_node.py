"""Phase 6's RESPONSE PLANNER + (inline) RISK CLASSIFIER wiring node.

BUILD_PLAN.md's Agent Architecture section, verbatim: *"RESPONSE PLANNER
(candidate actions: risk / expected benefit / confidence) -> RISK CLASSIFIER
(deterministic, code-level rule table -- never an LLM decision) -> SAFE
(report/note/tag/gather-diagnostics) -> ACTION EXECUTOR ... HIGH-IMPACT
(rollback/restart/scale/config/disable) -> HUMAN APPROVAL."*

This single node does three things, in order, for one incident:

1. Calls the Response Planner LLM (`ChatOpenRouter.with_structured_output`
   over `backend.agents.response_schemas.ResponsePlan`) to propose one or
   more candidate actions given the diagnosed root cause + evidence.
2. For each candidate, classifies it via
   `backend.agents.risk_classifier.classify_risk` -- a pure function, no
   LLM involved in this step at all -- and writes one `AuditEvent` row per
   action (this is the FIRST node in the graph that performs a real DB
   write/commit; every prior node is read-only against Postgres).
3. Sets `IncidentState.incident_status` to a genuine transitional value
   based on the classifications: `EXECUTING` if every action is SAFE
   (routes straight to the Action Executor), `AWAITING_APPROVAL` if any
   action is HIGH_IMPACT (routes to the human-approval `interrupt()`
   gate). Both are real downstream nodes now, not placeholders -- see
   "What this node does NOT do" below.

Steps 2-3 are folded into this same node rather than split into a separate
graph node: the classification logic itself
(`backend.agents.risk_classifier.classify_risk`) is already a pure,
independently unit-tested function (see `tests/test_risk_classifier.py`),
so nothing about testability is lost by not giving it its own node -- and
keeping the "write one AuditEvent per action + decide routing" step
together in one place matches how `investigation_node`/`root_cause_node`
already each do "one LLM call + local post-processing" in a single node.

## What this node does NOT do

This node's own job stops at: propose actions, classify risk, write the
`AuditEvent` rows, and set `incident_status`. The real `interrupt()`-based
Human Approval gate (`backend.agents.human_approval_node`), the Action
Executor (`backend.agents.action_executor_node`), and the Recovery Check
(`backend.agents.recovery_check_node`) are all separate downstream nodes
wired in `backend/graph.py` -- this node never calls any of them directly.
`EXECUTING` (all-SAFE plan) is a genuine transitional state now, not a
placeholder: `backend/graph.py` routes straight from here to
`action_executor`, which immediately overwrites it to `DIAGNOSED` (nothing
left to verify) once it runs. "AUTO_EXECUTED but `executed_at` is still
NULL" is the intentionally-modeled "queued for execution, not yet
executed" state for a SAFE action at this point -- see
`backend/models/audit.py`'s docstring for why `executed_at` starting NULL
is exactly what makes the Action Executor's idempotency guard work.

## Model routing

Reuses `get_settings().rca_model` rather than introducing a distinct
"response planner model" setting. BUILD_PLAN.md's Agent Architecture
section names a model role for Triage/Investigation/Root Cause but not for
the Response Planner; picking candidate actions given an already-diagnosed
root cause is a similarly reasoning-heavy (not cheap-classification) task,
so reusing the RCA-tier model is the defensible default rather than adding
a new env var for a role BUILD_PLAN.md never asked for. Revisit if a real
need for a distinct model/tier ever shows up.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from sqlalchemy.orm import Session

from backend.agents.response_schemas import ResponsePlan
from backend.agents.risk_classifier import (
    HIGH_IMPACT_ACTION_TYPES,
    SAFE_ACTION_TYPES,
    classify_risk,
)
from backend.agents.state import IncidentState
from backend.agents.structured_output import TRANSIENT_OPENROUTER_ERRORS, invoke_structured
from backend.config import get_settings
from backend.models import AuditDecisionStatus, AuditEvent, IncidentStatus, RiskClassification

RESPONSE_PLANNER_SYSTEM_PROMPT = f"""You are the response planning step of a production incident \
investigation system.

You are given a diagnosed root cause, the ranked hypotheses that led to \
it, and the accumulated evidence. Propose one or more candidate response \
actions that address the diagnosed root cause.

Prefer an action_type from this known vocabulary rather than inventing a \
new name (a downstream deterministic classifier decides how each action is \
actually handled based on this exact name, so use it verbatim when it \
applies):

SAFE action types (non-destructive, auto-executable):
{", ".join(sorted(SAFE_ACTION_TYPES))}

HIGH_IMPACT action types (require human approval before executing):
{", ".join(sorted(HIGH_IMPACT_ACTION_TYPES))}

For each action, give a short expected_benefit, your own heuristic \
confidence (0.0-1.0, not a calibrated probability) that it addresses the \
root cause, and your own informal llm_risk_assessment of how risky it is. \
Your llm_risk_assessment is read as context only -- it does not by itself \
decide how the action is routed; do not worry about getting it "right" in \
any formal sense, just describe your honest impression.

If you are not yet confident enough to recommend a HIGH_IMPACT remediation, \
it is entirely appropriate to propose only a SAFE action instead (e.g. \
gather_additional_diagnostics or add_investigation_note) rather than \
recommending a risky action prematurely. Always propose at least one \
action.
"""


def _hypothesis_lines(state: IncidentState) -> str:
    if not state.hypotheses:
        return "(no ranked hypotheses recorded)"
    return "\n".join(
        f"- {hyp.category} (confidence={hyp.confidence}): {hyp.rationale}"
        for hyp in state.hypotheses
    )


def _evidence_lines(state: IncidentState) -> str:
    if not state.evidence:
        return "(no evidence gathered)"
    return "\n".join(f"- [{item.source_ref.tool}] {item.description}" for item in state.evidence)


def _build_planner_prompt(state: IncidentState) -> str:
    return (
        f"Incident #{state.incident_id}\n"
        f"Affected service(s): {', '.join(state.affected_services) or 'unknown'}\n"
        f"Diagnosed root cause: {state.root_cause or 'unknown'}\n"
        f"Diagnostic confidence: {state.diagnostic_confidence}\n\n"
        f"Ranked hypotheses:\n{_hypothesis_lines(state)}\n\n"
        f"Accumulated evidence:\n{_evidence_lines(state)}\n\n"
        "Propose the structured response plan now."
    )


def make_response_planner_node(db: Session):
    """Return a LangGraph node function bound to one request-scoped `db`.

    Factory pattern matches `make_investigation_node(db)`/`build_tools(db)`
    -- one `Session` per request/graph run, explicit rather than global.
    Unlike Investigation (read-only tool queries), this node commits real
    writes (`AuditEvent` rows) -- see module docstring for why that has to
    happen here rather than being deferred to a later phase.
    """

    def response_planner_node(state: IncidentState) -> dict:
        settings = get_settings()
        # No temperature/top_p override -- kept unset for consistent,
        # prompt-driven behavior; explicit max_tokens, matching
        # root_cause_node's conventions -- including its headroom rationale:
        # a model's reasoning spend bills against this same ceiling and is
        # empirically highly variable per call (see root_cause_node.py's
        # comment for the live measurements this is based on), so the
        # budget has to cover a reasoning burst AND the real output, not
        # just the output alone.
        llm = ChatOpenRouter(
            model=settings.rca_model,
            api_key=settings.openrouter_api_key,
            max_tokens=16384,
        )
        # Free-tier OpenRouter models sit behind a shared, fluctuating-capacity
        # pool -- transient 502/429 "upstream overloaded" responses are routine,
        # not exceptional, and the openrouter SDK doesn't retry them internally
        # for this response shape. Retry at the LangChain Runnable level instead,
        # narrowed to the actually-transient error subset (see
        # structured_output.TRANSIENT_OPENROUTER_ERRORS) -- a bad API key or
        # oversized prompt should fail fast, not burn 5 backoff attempts first.
        structured_llm = llm.with_structured_output(ResponsePlan).with_retry(
            retry_if_exception_type=TRANSIENT_OPENROUTER_ERRORS, stop_after_attempt=5
        )

        messages = [
            SystemMessage(content=RESPONSE_PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=_build_planner_prompt(state)),
        ]
        # Free-tier models occasionally ignore the forced tool_choice and
        # reply with plain text instead -- the parser turns that into a
        # silent `None` rather than an exception, so `.with_retry()` above
        # never fires. Retry that case here too before giving up.
        result = invoke_structured(structured_llm, messages, ResponsePlan)

        recommended_actions: list[dict] = []
        any_high_impact = False

        for action in result.actions:
            # The one and only place SAFE-vs-HIGH_IMPACT is decided --
            # deterministic, code-level, zero LLM involvement. Never read
            # action.llm_risk_assessment here (see response_schemas.py's
            # module docstring).
            risk = classify_risk(action.action_type)
            decision_status = (
                AuditDecisionStatus.AUTO_EXECUTED
                if risk is RiskClassification.SAFE
                else AuditDecisionStatus.PENDING_APPROVAL
            )
            if risk is RiskClassification.HIGH_IMPACT:
                any_high_impact = True

            event = AuditEvent(
                incident_id=state.incident_id,
                action_type=action.action_type,
                risk_classification=risk,
                decision_status=decision_status,
                # approver/executed_at/execution_outcome/execution_detail
                # all stay at their column defaults (NULL) -- no human
                # decision and no execution has happened yet, even for
                # AUTO_EXECUTED rows (see module docstring's "queued, not
                # yet executed" note).
            )
            db.add(event)
            db.flush()  # assigns event.id without ending the transaction

            recommended_actions.append(
                {
                    "audit_event_id": event.id,
                    "action_type": action.action_type,
                    "expected_benefit": action.expected_benefit,
                    "confidence": action.confidence,
                    "risk_classification": risk.value,
                    "decision_status": decision_status.value,
                }
            )

        # Committed here (not left for a caller) -- a later `/approve` or
        # `/reject` request (Phase 6's next sub-step) will read these rows
        # through a completely different request-scoped session, so they
        # must be durable before this node returns.
        db.commit()

        incident_status = (
            IncidentStatus.AWAITING_APPROVAL if any_high_impact else IncidentStatus.EXECUTING
        )

        return {
            "recommended_actions": recommended_actions,
            "incident_status": incident_status,
        }

    return response_planner_node
