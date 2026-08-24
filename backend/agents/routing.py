"""Phase 5's conditional re-investigation loop: pure predicate functions
plus the LangGraph conditional-edge router that combines them.

BUILD_PLAN.md's Agent Architecture section: *"Confidence-gap loop is a
disjunction, not confidence-only: LLM confidence numbers are poorly
calibrated and cluster, so the re-investigation loop also fires on an
evidence-sufficiency check (did the agent gather evidence covering the
affected service + its recent deployment + its dependencies?). Bounded to
N iterations regardless."*

Every function here is a pure function of `IncidentState` — no I/O, no
LLM calls, nothing but reading fields already on the state — so each is
directly unit-testable in isolation, per this task's requirement to test
the predicates thoroughly without any mocking at all.
"""

from __future__ import annotations

from backend.agents.state import IncidentState
from backend.models.incident import IncidentStatus

# --- Confidence-gap check ---------------------------------------------------

# Threshold reasoning: `diagnostic_confidence`/`Hypothesis.confidence` are
# self-reported 0.0-1.0 heuristics (BUILD_PLAN.md is explicit these are NOT
# calibrated probabilities), and Claude's own estimates for this kind of
# task tend to cluster in the 0.5-0.85 range rather than spanning the full
# scale. 0.15 is picked as a middle ground: small enough that a genuinely
# decisive diagnosis (e.g. 0.85 vs 0.3) reliably passes without triggering
# a needless re-investigation, but large enough that two hypotheses sitting
# close together in that clustered range (e.g. 0.6 vs 0.55) are correctly
# treated as "too close to trust" rather than accepting whichever the model
# happened to rank first.
CONFIDENCE_GAP_THRESHOLD = 0.15

# BUILD_PLAN.md's evidence-sufficiency check, made concrete: "did the agent
# gather evidence covering the affected service + its recent deployment +
# its dependencies?" -- get_logs/get_metrics describe the service itself,
# so the two tool categories actually worth gating on are the two BUILD_PLAN
# calls out by name: recent deployments and downstream dependencies. This is
# exactly the pair that matters for `cascading_payment_timeout`: stopping at
# the loud get_logs/get_metrics DB symptom without checking get_dependencies
# (checkout -> payment) is the specific shortcut that scenario is designed
# to catch.
REQUIRED_EVIDENCE_TOOLS: frozenset[str] = frozenset({"get_deployments", "get_dependencies"})

# "Bounded to N iterations regardless" (BUILD_PLAN.md). N=2: enough for
# `cascading_payment_timeout` to recover from an incomplete first pass (miss
# the dependency check, get routed back, cover it on the second pass) with
# one more retry as headroom for a genuinely hard case, while still bounding
# the cost of a graph run to at most 3 total Investigation/Root-Cause visits
# (1 initial + 2 retries) -- consistent with `investigator.py`'s
# `MAX_TOOL_CALLS` being a real cost/safety control, not a decorative
# constant.
MAX_REINVESTIGATION_LOOPS = 2


def confidence_gap_below_threshold(
    state: IncidentState, threshold: float = CONFIDENCE_GAP_THRESHOLD
) -> bool:
    """True when the top-2 ranked hypotheses' confidence values are too
    close together to trust the top pick without more evidence.

    "Top-2" is read as `hypotheses[0]` (the chosen root cause) vs.
    `alternative_hypotheses[0]` (the strongest hypothesis NOT chosen) --
    `DiagnosisResult`/`IncidentState` already separate "the pick" from
    "what else was seriously considered" into these two lists, so that
    pairing is the natural reading of "top-2 hypotheses" rather than
    requiring 2+ entries inside `hypotheses` itself (BUILD_PLAN.md's
    Agent Architecture section explicitly flags this as a decision this
    phase needs to make, given `DiagnosisResult`'s existing shape).

    `hypotheses[0].confidence` falls back to `diagnostic_confidence` when
    unset (the model may only populate the overall field); a missing
    `alternative_hypotheses[0].confidence` falls back to 0.0 (treat an
    unscored alternative as maximally uncertain, which only makes the gap
    look *larger*, i.e. biases toward NOT triggering on missing data here
    -- the evidence-sufficiency check is the other half of the disjunction
    and is what actually catches "not enough was investigated").

    If there is no top hypothesis, or no alternative was recorded at all,
    there is nothing to compare a gap against -- this returns False (does
    not trigger via this path) rather than treating "no data" as
    "insufficient confidence".
    """
    if not state.hypotheses:
        return False
    if not state.alternative_hypotheses:
        return False

    top = state.hypotheses[0]
    runner_up = state.alternative_hypotheses[0]

    top_confidence = top.confidence if top.confidence is not None else state.diagnostic_confidence
    runner_up_confidence = runner_up.confidence if runner_up.confidence is not None else 0.0

    return (top_confidence - runner_up_confidence) < threshold


def evidence_sufficiency_check_failed(
    state: IncidentState, required_tools: frozenset[str] = REQUIRED_EVIDENCE_TOOLS
) -> bool:
    """True when the investigation never gathered evidence covering one of
    `required_tools` (deployments + dependencies, by default).

    Checked against `IncidentState.evidence[].source_ref.tool` rather than
    `tool_call_log_ids` (which carries no tool-name information --
    see `IncidentState`'s docstring): the Investigation node writes exactly
    one `EvidenceItem` per tool call it makes, including calls that
    returned zero records ("no recent deployment" is itself a real,
    meaningful finding) -- so tool coverage and evidence coverage are the
    same signal here, and `evidence[]` is the field that actually carries
    the tool name.
    """
    covered_tools = {item.source_ref.tool for item in state.evidence}
    return not required_tools.issubset(covered_tools)


def should_reinvestigate(state: IncidentState) -> bool:
    """The disjunction BUILD_PLAN.md specifies: confidence gap too small OR
    evidence coverage incomplete. Does NOT apply the iteration bound --
    that's `route_after_root_cause`'s job, so this function stays a pure
    read of "is the diagnosis good enough," independent of "have we already
    retried too many times."
    """
    return confidence_gap_below_threshold(state) or evidence_sufficiency_check_failed(state)


def route_after_root_cause(state: IncidentState) -> str:
    """LangGraph conditional-edge callback for the ROOT CAUSE node.

    Returns "reinvestigate" (routes back to the Investigation node) when
    `should_reinvestigate` is true AND the bounded retry budget
    (`MAX_REINVESTIGATION_LOOPS`) hasn't been exhausted; otherwise "end"
    (the graph reaches its Phase 5 terminal state with
    `incident_status == diagnosed`, per BUILD_PLAN.md's Phase 5 scope --
    Response Planner etc. is Phase 6).

    `investigation_iterations` is incremented by the Investigation node
    itself each time it runs (see `backend.agents.investigation_node`), so
    by the time this router runs it already reflects how many Investigation
    passes have happened, including the initial one.
    """
    if state.investigation_iterations > MAX_REINVESTIGATION_LOOPS:
        return "end"
    if should_reinvestigate(state):
        return "reinvestigate"
    return "end"


def route_after_response_planner(state: IncidentState) -> str:
    """LangGraph conditional-edge callback for the RESPONSE PLANNER node.

    `response_planner_node` already sets `incident_status` to
    `AWAITING_APPROVAL` (any HIGH_IMPACT action present) or `EXECUTING`
    (all-SAFE plan) -- see that module's docstring. Reusing that field
    here rather than adding a new one: it's already the exact signal this
    router needs ("does this incident have something pending a human
    decision"), set by the one place (the deterministic Risk Classifier,
    via `response_planner_node`) that's allowed to decide it.

    Returns "human_approval" (routes to the `interrupt()` gate) when any
    action was classified HIGH_IMPACT; otherwise "end" -- an all-SAFE plan
    has nothing for a human to approve, so `backend/graph.py` wires this
    "end" straight to `action_executor` (never through `human_approval`),
    per BUILD_PLAN.md's graph diagram.
    """
    if state.incident_status is IncidentStatus.AWAITING_APPROVAL:
        return "human_approval"
    return "end"


def route_after_action_executor(state: IncidentState) -> str:
    """LangGraph conditional-edge callback for the ACTION EXECUTOR node.

    `action_executor_node` sets `incident_status` to `VERIFYING` when it
    just executed at least one HIGH_IMPACT remediation (something the
    Recovery Check needs to verify against telemetry), or `DIAGNOSED` for
    an all-SAFE plan (nothing to verify -- see that module's docstring for
    why `DIAGNOSED` is the closest existing lifecycle fit). Returns
    "recovery_check" in the former case, "end" in the latter.
    """
    if state.incident_status is IncidentStatus.VERIFYING:
        return "recovery_check"
    return "end"


def route_after_recovery_check(state: IncidentState) -> str:
    """LangGraph conditional-edge callback for the RECOVERY CHECK node.

    `recovery_check_node` already applies the bounded re-investigation
    loop's own budget (`state.investigation_iterations` vs.
    `MAX_REINVESTIGATION_LOOPS` -- the SAME field/constant
    `route_after_root_cause` uses, not a second parallel bound) and sets
    `incident_status` to exactly one of `RESOLVED` (recovered),
    `MANUAL_INTERVENTION_REQUIRED` (still degraded, budget exhausted), or
    `INVESTIGATING` (still degraded, budget remains -- loop back for a
    fresh Investigation pass). This router just reads that decision, the
    same pattern `route_after_response_planner` already uses against
    `response_planner_node`'s output: "end" for the two terminal states,
    "investigation" to loop back.
    """
    if state.incident_status is IncidentStatus.INVESTIGATING:
        return "investigation"
    return "end"
