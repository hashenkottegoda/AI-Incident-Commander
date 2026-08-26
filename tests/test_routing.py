"""Unit tests for Phase 5's conditional re-investigation loop predicates
(`backend/agents/routing.py`).

Every function under test is a pure function of `IncidentState` -- no I/O,
no LLM, no mocking needed. Constructed `IncidentState` instances only.
"""

from __future__ import annotations

from backend.agents.routing import (
    CONFIDENCE_GAP_THRESHOLD,
    MAX_REINVESTIGATION_LOOPS,
    confidence_gap_below_threshold,
    evidence_sufficiency_check_failed,
    route_after_root_cause,
    should_reinvestigate,
)
from backend.agents.schemas import EvidenceItem, Hypothesis, SourceRef
from backend.agents.state import IncidentState

DB_POOL = "database_connection_pool"


def _hyp(category: str, confidence: float | None, rationale: str = "r") -> Hypothesis:
    return Hypothesis(category=category, rationale=rationale, confidence=confidence)


def _evidence(tool: str) -> EvidenceItem:
    return EvidenceItem(description=f"finding from {tool}", source_ref=SourceRef(tool=tool))


def _full_coverage_evidence() -> list[EvidenceItem]:
    tools = ("get_logs", "get_metrics", "get_deployments", "get_dependencies")
    return [_evidence(tool) for tool in tools]


# --- confidence_gap_below_threshold -----------------------------------------


def test_confidence_gap_below_threshold_true_when_hypotheses_cluster():
    state = IncidentState(
        incident_id=1,
        hypotheses=[_hyp(DB_POOL, 0.55)],
        alternative_hypotheses=[_hyp("unknown", 0.5)],
    )
    assert confidence_gap_below_threshold(state) is True


def test_confidence_gap_below_threshold_false_when_decisive():
    state = IncidentState(
        incident_id=1,
        hypotheses=[_hyp(DB_POOL, 0.9)],
        alternative_hypotheses=[_hyp("unknown", 0.2)],
    )
    assert confidence_gap_below_threshold(state) is False


def test_confidence_gap_exactly_at_threshold_is_not_below_it():
    state = IncidentState(
        incident_id=1,
        hypotheses=[_hyp(DB_POOL, 0.65)],
        alternative_hypotheses=[_hyp("unknown", 0.5)],
    )
    gap = 0.65 - 0.5
    assert abs(gap - CONFIDENCE_GAP_THRESHOLD) < 1e-9
    assert confidence_gap_below_threshold(state) is False  # strictly < threshold, not <=


def test_confidence_gap_false_when_no_hypotheses_at_all():
    state = IncidentState(incident_id=1)
    assert confidence_gap_below_threshold(state) is False


def test_confidence_gap_false_when_no_alternative_recorded():
    state = IncidentState(incident_id=1, hypotheses=[_hyp(DB_POOL, 0.9)])
    assert confidence_gap_below_threshold(state) is False


def test_confidence_gap_falls_back_to_diagnostic_confidence_when_hypothesis_confidence_unset():
    state = IncidentState(
        incident_id=1,
        diagnostic_confidence=0.9,
        hypotheses=[_hyp(DB_POOL, None)],
        alternative_hypotheses=[_hyp("unknown", 0.2)],
    )
    assert confidence_gap_below_threshold(state) is False  # 0.9 - 0.2 = 0.7, decisive


def test_confidence_gap_falls_back_to_zero_when_alternative_confidence_unset():
    state = IncidentState(
        incident_id=1,
        hypotheses=[_hyp(DB_POOL, 0.1)],
        alternative_hypotheses=[_hyp("unknown", None)],  # falls back to 0.0
    )
    assert confidence_gap_below_threshold(state) is True  # 0.1 - 0.0 = 0.1 < 0.15 threshold


def test_confidence_gap_custom_threshold():
    state = IncidentState(
        incident_id=1,
        hypotheses=[_hyp(DB_POOL, 0.6)],
        alternative_hypotheses=[_hyp("unknown", 0.5)],
    )
    assert confidence_gap_below_threshold(state, threshold=0.05) is False
    assert confidence_gap_below_threshold(state, threshold=0.5) is True


def test_confidence_gap_falls_back_to_second_hypothesis_when_alternatives_empty():
    """The RCA prompt asks the model to put exactly one entry in `hypotheses`
    and any runner-up in `alternative_hypotheses`, but `DiagnosisResult`
    doesn't enforce that shape -- if the model ranks multiple candidates
    inside `hypotheses` itself instead, `hypotheses[1]` must still be used
    as the runner-up rather than silently skipping the confidence-gap check.
    """
    state = IncidentState(
        incident_id=1,
        hypotheses=[_hyp(DB_POOL, 0.55), _hyp("unknown", 0.5)],
        alternative_hypotheses=[],
    )
    assert confidence_gap_below_threshold(state) is True  # 0.55 - 0.5 = 0.05 < 0.15


def test_confidence_gap_prefers_alternative_hypotheses_over_second_hypothesis():
    """When both are present, `alternative_hypotheses[0]` -- the intended
    runner-up field -- wins over `hypotheses[1]`, not the other way round.
    """
    state = IncidentState(
        incident_id=1,
        hypotheses=[_hyp(DB_POOL, 0.9), _hyp("unknown", 0.85)],
        alternative_hypotheses=[_hyp("unknown", 0.2)],
    )
    assert confidence_gap_below_threshold(state) is False  # 0.9 - 0.2 = 0.7, decisive


# --- evidence_sufficiency_check_failed --------------------------------------


def test_evidence_sufficiency_fails_when_deployments_and_dependencies_missing():
    state = IncidentState(incident_id=1, evidence=[_evidence("get_logs"), _evidence("get_metrics")])
    assert evidence_sufficiency_check_failed(state) is True


def test_evidence_sufficiency_fails_when_only_one_of_the_two_required_tools_covered():
    state = IncidentState(incident_id=1, evidence=[_evidence("get_deployments")])
    assert evidence_sufficiency_check_failed(state) is True


def test_evidence_sufficiency_passes_with_full_coverage():
    state = IncidentState(incident_id=1, evidence=_full_coverage_evidence())
    assert evidence_sufficiency_check_failed(state) is False


def test_evidence_sufficiency_passes_even_if_logs_and_metrics_never_called():
    # Only get_deployments/get_dependencies are required -- BUILD_PLAN.md's
    # wording specifically calls those two out, not the full tool set.
    state = IncidentState(
        incident_id=1, evidence=[_evidence("get_deployments"), _evidence("get_dependencies")]
    )
    assert evidence_sufficiency_check_failed(state) is False


def test_evidence_sufficiency_fails_on_empty_evidence():
    state = IncidentState(incident_id=1)
    assert evidence_sufficiency_check_failed(state) is True


def test_evidence_sufficiency_ignores_rag_evidence_tool_tag():
    # search_historical_incidents-tagged evidence should never count toward
    # the deployments/dependencies coverage requirement.
    state = IncidentState(
        incident_id=1,
        evidence=[
            _evidence("search_historical_incidents"),
            _evidence("get_logs"),
            _evidence("get_metrics"),
        ],
    )
    assert evidence_sufficiency_check_failed(state) is True


def test_evidence_sufficiency_custom_required_tools():
    state = IncidentState(incident_id=1, evidence=[_evidence("get_logs")])
    assert evidence_sufficiency_check_failed(state, required_tools=frozenset({"get_logs"})) is False
    assert (
        evidence_sufficiency_check_failed(state, required_tools=frozenset({"get_metrics"})) is True
    )


# --- should_reinvestigate (disjunction) -------------------------------------


def test_should_reinvestigate_true_via_evidence_path_even_with_decisive_confidence():
    """The disjunction: evidence-sufficiency alone is enough to trigger,
    independent of how decisive the confidence gap looks -- this is the
    behavior BUILD_PLAN.md calls out by name ("even if confidences
    cluster" the evidence-sufficiency path is what reliably exercises the
    loop; here we show the converse holds too -- it fires even when
    confidence alone would NOT have triggered)."""
    state = IncidentState(
        incident_id=1,
        hypotheses=[_hyp(DB_POOL, 0.95)],
        alternative_hypotheses=[_hyp("unknown", 0.05)],
        # missing get_deployments/get_dependencies coverage:
        evidence=[_evidence("get_logs"), _evidence("get_metrics")],
    )
    assert confidence_gap_below_threshold(state) is False
    assert evidence_sufficiency_check_failed(state) is True
    assert should_reinvestigate(state) is True


def test_should_reinvestigate_true_via_confidence_path_even_with_full_evidence_coverage():
    state = IncidentState(
        incident_id=1,
        hypotheses=[_hyp(DB_POOL, 0.55)],
        alternative_hypotheses=[_hyp("unknown", 0.5)],
        evidence=_full_coverage_evidence(),
    )
    assert confidence_gap_below_threshold(state) is True
    assert evidence_sufficiency_check_failed(state) is False
    assert should_reinvestigate(state) is True


def test_should_reinvestigate_false_when_both_checks_pass():
    state = IncidentState(
        incident_id=1,
        hypotheses=[_hyp(DB_POOL, 0.9)],
        alternative_hypotheses=[_hyp("unknown", 0.1)],
        evidence=_full_coverage_evidence(),
    )
    assert should_reinvestigate(state) is False


# --- route_after_root_cause (the actual conditional-edge callback) ---------


def test_route_after_root_cause_returns_reinvestigate_when_insufficient_and_under_budget():
    state = IncidentState(
        incident_id=1,
        investigation_iterations=1,
        evidence=[_evidence("get_logs")],
    )
    assert route_after_root_cause(state) == "reinvestigate"


def test_route_after_root_cause_returns_end_when_sufficient():
    state = IncidentState(
        incident_id=1,
        investigation_iterations=1,
        evidence=_full_coverage_evidence(),
        hypotheses=[_hyp(DB_POOL, 0.9)],
        alternative_hypotheses=[_hyp("unknown", 0.1)],
    )
    assert route_after_root_cause(state) == "end"


def test_route_after_root_cause_stops_once_retry_budget_exhausted_even_if_still_insufficient():
    """Bounded regardless: BUILD_PLAN.md is explicit the loop is "Bounded to
    N iterations regardless" of whether the diagnosis is actually good
    enough yet."""
    state = IncidentState(
        incident_id=1,
        investigation_iterations=MAX_REINVESTIGATION_LOOPS + 1,
        evidence=[_evidence("get_logs")],  # still insufficient
    )
    assert evidence_sufficiency_check_failed(state) is True  # would otherwise loop
    assert route_after_root_cause(state) == "end"  # but the budget is exhausted


def test_route_after_root_cause_allows_up_to_max_reinvestigation_loops():
    state_at_budget = IncidentState(
        incident_id=1,
        investigation_iterations=MAX_REINVESTIGATION_LOOPS,
        evidence=[_evidence("get_logs")],
    )
    assert route_after_root_cause(state_at_budget) == "reinvestigate"
