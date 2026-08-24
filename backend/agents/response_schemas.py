"""`ResponseAction` / `ResponsePlan` -- Phase 6's Response Planner output schema.

BUILD_PLAN.md's Agent Architecture section, verbatim: *"RESPONSE PLANNER
(candidate actions: risk / expected benefit / confidence) -> RISK CLASSIFIER
(deterministic, code-level rule table -- never an LLM decision)."*

This module defines what the Response Planner LLM proposes; it does NOT
decide SAFE vs. HIGH_IMPACT -- that is `backend.agents.risk_classifier`'s
sole job, deliberately kept in a separate, LLM-free module.

## `llm_risk_assessment` is informational only -- READ THIS BEFORE WIRING
## THIS FIELD INTO ANY ROUTING/EXECUTION DECISION

`ResponseAction.llm_risk_assessment` is the Response Planner's own informal
opinion of how risky its proposed action is (e.g. "low risk, fully
reversible" or "high risk, could cause a brief outage"). It exists purely
as **display/explanatory context** -- something a human reviewing the
approval queue (Phase 6's later HITL step) can read alongside the action.

It has **zero bearing** on whether an action is actually routed as SAFE or
HIGH_IMPACT. That routing is decided exclusively by
`backend.agents.risk_classifier.classify_risk(action_type)` -- a pure,
deterministic, code-level rule table with no model call in it at all. An
LLM could write `llm_risk_assessment="totally safe, no concerns"` on a
`rollback_deployment` action and `classify_risk` would still (correctly)
return `HIGH_IMPACT`, because it only ever looks at `action_type` against
its fixed rule table, never at this field. Nothing downstream (graph
routing, the Action Executor's gating, the audit trail's
`risk_classification` column) may ever read `llm_risk_assessment` to make
that decision -- if a future change finds itself branching on this field's
content, that is the bug this docstring is trying to prevent.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResponseAction(BaseModel):
    """One candidate remediation/response action proposed by the Response
    Planner LLM.

    `action_type` is a plain string, not a closed `Literal`/enum, on
    purpose: the planner is prompted to pick from the known vocabulary
    (the 4 SAFE names + 5 HIGH_IMPACT names -- see
    `backend.agents.response_planner_node.RESPONSE_PLANNER_SYSTEM_PROMPT`),
    but is not hard-constrained to it. An unrecognized/typo'd name is not a
    schema validation failure here -- `risk_classifier.classify_risk()`'s
    own fail-safe default (unrecognized -> HIGH_IMPACT) is what handles
    that case safely, one layer downstream. Constraining this field to a
    `Literal` here would just move the "what if the model picks something
    outside it" problem from a graceful fail-safe default into a
    structured-output parse error.
    """

    action_type: str = Field(
        description=(
            "The proposed action's type name. Prefer one of the known SAFE "
            "names (generate_incident_report, add_investigation_note, "
            "gather_additional_diagnostics, tag_incident) or HIGH_IMPACT "
            "names (rollback_deployment, restart_service, scale_service, "
            "disable_feature_flag, increase_connection_pool) -- do not "
            "invent a new action type unless none of these genuinely fit."
        )
    )
    expected_benefit: str = Field(
        description=(
            "Short plain-language statement of what this action is expected "
            "to fix or improve, e.g. 'removes the leaking deployed code path "
            "that is exhausting the connection pool'."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "The model's own heuristic confidence (0.0-1.0) that this "
            "action addresses the diagnosed root cause. Same "
            "self-reported-estimate caveat as IncidentState.diagnostic_confidence "
            "-- not a calibrated probability, a display/tie-break signal only."
        ),
    )
    llm_risk_assessment: str = Field(
        description=(
            "The model's own informal opinion of this action's risk (e.g. "
            "'low risk, fully reversible' or 'high risk, brief downtime "
            "expected'). INFORMATIONAL ONLY -- see this module's docstring. "
            "This field NEVER determines actual SAFE vs. HIGH_IMPACT "
            "routing; only backend.agents.risk_classifier.classify_risk() "
            "does that, via its deterministic rule table over action_type."
        )
    )


class ResponsePlan(BaseModel):
    """The Response Planner node's full structured-output result: one or
    more candidate actions for the diagnosed incident.

    At least one action is required -- a plan proposing nothing would give
    the Risk Classifier / audit trail nothing to act on, and BUILD_PLAN.md's
    SAFE branch is explicit that even "do nothing destructive yet" should
    still surface as a genuine SAFE action (e.g. gather_additional_diagnostics
    or add_investigation_note), not an empty list.
    """

    actions: list[ResponseAction] = Field(
        min_length=1,
        description=(
            "One or more candidate response actions, most-recommended first. "
            "Always propose at least one action -- if no remediation is "
            "confidently recommended yet, propose a SAFE action such as "
            "gather_additional_diagnostics rather than an empty list."
        ),
    )
