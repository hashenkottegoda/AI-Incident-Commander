"""The RISK CLASSIFIER — Phase 6's deterministic SAFE-vs-HIGH_IMPACT rule table.

BUILD_PLAN.md's Agent Architecture section, verbatim: *"RISK CLASSIFIER
(deterministic, code-level rule table — never an LLM decision) -> SAFE
(report/note/tag/gather-diagnostics) -> ACTION EXECUTOR ... HIGH-IMPACT
(rollback/restart/scale/config/disable) -> HUMAN APPROVAL."*

This is the guardrail the whole approval gate depends on, so it is
deliberately the most boring code in this codebase: one pure function, one
plain rule table, zero I/O, zero randomness, zero model calls. See
`backend.agents.response_schemas`'s module docstring for why the Response
Planner LLM is allowed to *propose* its own informal risk opinion
(`ResponseAction.llm_risk_assessment`) without that opinion having any
bearing on what this function returns — this module is the sole authority
on SAFE vs HIGH_IMPACT, full stop.
"""

from __future__ import annotations

from backend.models.audit import RiskClassification
from backend.simulation.scenario_schema import ACTION_TYPES as HIGH_IMPACT_ACTION_TYPES

# The 4 SAFE action names (BUILD_PLAN.md: *"SAFE actions are things like
# generate incident report, add investigation note, gather additional
# diagnostics, tag incident — all auto-executable"*). These have no home in
# `backend.simulation.scenario_schema` — see `backend/models/audit.py`'s
# module docstring for why: the simulator only needs to know the 5
# HIGH_IMPACT action names (it produces post-action telemetry for those),
# never these.
SAFE_ACTION_TYPES: frozenset[str] = frozenset(
    {
        "generate_incident_report",
        "add_investigation_note",
        "gather_additional_diagnostics",
        "tag_incident",
    }
)

# Re-exported under a locally-meaningful name — this module's rule table is
# stated as SAFE_ACTION_TYPES vs. HIGH_IMPACT_ACTION_TYPES, not
# SAFE_ACTION_TYPES vs. "whatever scenario_schema happens to call it".
HIGH_IMPACT_ACTION_TYPES: frozenset[str] = HIGH_IMPACT_ACTION_TYPES


def classify_risk(action_type: str) -> RiskClassification:
    """The deterministic rule table itself.

    Three cases, in order:

    1. `action_type` is one of the 4 known SAFE names -> `SAFE`.
    2. `action_type` is one of the 5 known HIGH_IMPACT names
       (`backend.simulation.scenario_schema.ACTION_TYPES`) -> `HIGH_IMPACT`.
    3. Anything else (an unrecognized/typo'd action_type the Response
       Planner LLM invented despite being told the known vocabulary) ->
       `HIGH_IMPACT`.

    Case 3 is the load-bearing fail-safe default: this is intentional
    default-deny, per BUILD_PLAN.md's stated philosophy for this gate. The
    two possible mistakes here are not symmetric — auto-executing an
    unrecognized action that turns out to be destructive is unacceptable,
    while routing a genuinely-harmless-but-misspelled SAFE action through
    human approval is merely a minor inconvenience (a human sees an
    unfamiliar action name in the approval queue and can reject/investigate
    it). So an unknown name always falls to the safer of the two failure
    modes: gated behind a human, never silently auto-executed. Case 2 is
    written out explicitly (not left to "anything not in SAFE_ACTION_TYPES
    falls through to case 3") so the rule table reads as a complete,
    auditable decision table rather than relying on the reader to notice
    that cases 2 and 3 happen to produce the same output today.

    This function is intentionally free of I/O, randomness, and any LLM
    call — trivially unit-testable in isolation, with no DB/graph/mocking
    required (see `tests/test_risk_classifier.py`).
    """
    if action_type in SAFE_ACTION_TYPES:
        return RiskClassification.SAFE
    if action_type in HIGH_IMPACT_ACTION_TYPES:
        return RiskClassification.HIGH_IMPACT
    return RiskClassification.HIGH_IMPACT
