"""Structural tests for `IncidentState` (BUILD_PLAN.md Phase 5).

Pure Pydantic model tests -- no I/O, no LLM, no skip conditions needed.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.agents.schemas import EvidenceItem, Hypothesis, SourceRef
from backend.agents.state import IncidentState
from backend.models.incident import IncidentStatus, Severity


def test_incident_id_is_the_only_required_field():
    state = IncidentState(incident_id=1)
    assert state.incident_id == 1


def test_default_values_match_pre_graph_incident_row_state():
    state = IncidentState(incident_id=1)
    assert state.incident_status == IncidentStatus.DETECTED
    assert state.severity is None
    assert state.affected_services == []
    assert state.tool_call_log_ids == []
    assert state.evidence == []
    assert state.hypotheses == []
    assert state.root_cause is None
    assert state.diagnostic_confidence == 0.0
    assert state.alternative_hypotheses == []
    assert state.recommended_actions == []
    assert state.approval_decision is None
    assert state.execution_result_id is None
    assert state.recovery_result is None
    assert state.investigation_iterations == 0


def test_incident_id_required():
    with pytest.raises(ValidationError):
        IncidentState()  # type: ignore[call-arg]


def test_full_field_set_round_trips_through_model_dump_and_validate():
    state = IncidentState(
        incident_id=42,
        incident_status=IncidentStatus.DIAGNOSED,
        severity=Severity.P1,
        affected_services=["checkout-service", "payment-service"],
        tool_call_log_ids=[1, 2, 3],
        evidence=[
            EvidenceItem(
                description="db_connections_active rose sharply",
                source_ref=SourceRef(tool="get_metrics", record_id=7),
            )
        ],
        hypotheses=[
            Hypothesis(
                category="database_connection_pool",
                rationale="connections exhausted",
                confidence=0.8,
            )
        ],
        root_cause="database_connection_pool",
        diagnostic_confidence=0.75,
        alternative_hypotheses=[
            Hypothesis(category="unknown", rationale="insufficient data", confidence=0.2)
        ],
        recommended_actions=[{"action": "rollback_deployment", "risk": "high"}],
        approval_decision="approved",
        execution_result_id=[99, 100],
        recovery_result={"outcome": "recovered", "checked_metrics": {"error_rate": 0.004}},
        investigation_iterations=1,
    )
    dumped = state.model_dump(mode="json")
    restored = IncidentState.model_validate(dumped)
    assert restored == state


def test_diagnostic_confidence_bounds_enforced():
    with pytest.raises(ValidationError):
        IncidentState(incident_id=1, diagnostic_confidence=1.5)
    with pytest.raises(ValidationError):
        IncidentState(incident_id=1, diagnostic_confidence=-0.1)


def test_severity_and_incident_status_use_the_real_enums_not_free_strings():
    state = IncidentState(
        incident_id=1, severity=Severity.P2, incident_status=IncidentStatus.TRIAGING
    )
    assert state.severity is Severity.P2
    assert state.incident_status is IncidentStatus.TRIAGING


def test_hypothesis_confidence_is_optional_and_additive():
    """Phase 3's baseline investigator never sets Hypothesis.confidence --
    confirm it defaults to None rather than being required, so Experiment
    B's existing behavior/output shape is unaffected by this addition."""
    hyp = Hypothesis(category="unknown", rationale="no evidence")
    assert hyp.confidence is None


def test_evidence_carries_structured_source_refs_not_raw_payloads():
    item = EvidenceItem(
        description="no deployments found",
        source_ref=SourceRef(tool="get_deployments", query="service='checkout-service'"),
    )
    state = IncidentState(incident_id=1, evidence=[item])
    assert state.evidence[0].source_ref.tool == "get_deployments"
    assert state.evidence[0].source_ref.record_id is None


def test_source_ref_rescues_non_numeric_record_id_string_into_query():
    """record_id is int-typed by design (see SourceRef's docstring) --
    historical-incident citations belong in `query` as a string id like
    "hist-012". Free-tier models periodically ignore the prompt rule and
    put that string straight into record_id, which would otherwise raise
    a ValidationError and crash the whole node. Confirm the before-validator
    rescues it instead: record_id ends up None, query ends up "hist-012"."""
    ref = SourceRef(tool="search_historical_incidents", record_id="hist-012")
    assert ref.record_id is None
    assert ref.query == "hist-012"


def test_lifecycle_enum_values_match_build_plan_ordering():
    # Sanity check that the lifecycle this state's incident_status field
    # references still matches BUILD_PLAN.md's documented ordering -- a
    # regression here would silently break routing/graph assumptions.
    assert [member.value for member in IncidentStatus] == [
        "detected",
        "triaging",
        "investigating",
        "diagnosed",
        "awaiting_approval",
        "executing",
        "verifying",
        "resolved",
        "manual_intervention_required",
    ]


