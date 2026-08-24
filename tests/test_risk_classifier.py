"""Unit tests for Phase 6's deterministic Risk Classifier
(`backend/agents/risk_classifier.py`).

`classify_risk` is a pure function -- no I/O, no LLM, no mocking needed.
Every known SAFE/HIGH_IMPACT name is exercised individually (not just
"one representative of each set") since this rule table is the guardrail
the whole approval gate depends on, plus the unrecognized-name fail-safe
default case.
"""

from __future__ import annotations

import pytest

from backend.agents.risk_classifier import (
    HIGH_IMPACT_ACTION_TYPES,
    SAFE_ACTION_TYPES,
    classify_risk,
)
from backend.models.audit import RiskClassification
from backend.simulation.scenario_schema import ACTION_TYPES as SCENARIO_ACTION_TYPES


def test_safe_and_high_impact_sets_are_disjoint():
    assert SAFE_ACTION_TYPES.isdisjoint(HIGH_IMPACT_ACTION_TYPES)


def test_high_impact_action_types_matches_scenario_schema_action_types():
    """`risk_classifier.HIGH_IMPACT_ACTION_TYPES` is re-exported from
    `backend.simulation.scenario_schema.ACTION_TYPES` -- prove the two
    never silently diverge."""
    assert HIGH_IMPACT_ACTION_TYPES == SCENARIO_ACTION_TYPES


def test_safe_action_types_has_the_four_build_plan_names():
    assert SAFE_ACTION_TYPES == {
        "generate_incident_report",
        "add_investigation_note",
        "gather_additional_diagnostics",
        "tag_incident",
    }


def test_high_impact_action_types_has_the_five_build_plan_names():
    assert HIGH_IMPACT_ACTION_TYPES == {
        "rollback_deployment",
        "restart_service",
        "scale_service",
        "disable_feature_flag",
        "increase_connection_pool",
    }


@pytest.mark.parametrize(
    "action_type",
    [
        "generate_incident_report",
        "add_investigation_note",
        "gather_additional_diagnostics",
        "tag_incident",
    ],
)
def test_every_safe_action_type_classifies_safe(action_type):
    assert classify_risk(action_type) is RiskClassification.SAFE


@pytest.mark.parametrize(
    "action_type",
    [
        "rollback_deployment",
        "restart_service",
        "scale_service",
        "disable_feature_flag",
        "increase_connection_pool",
    ],
)
def test_every_high_impact_action_type_classifies_high_impact(action_type):
    assert classify_risk(action_type) is RiskClassification.HIGH_IMPACT


@pytest.mark.parametrize(
    "action_type",
    [
        "",
        "delete_database",
        "rollback_deploymen",  # typo of a real HIGH_IMPACT name
        "Tag_Incident",  # case mismatch of a real SAFE name
        "shutdown_everything",
        "unknown_action_xyz",
    ],
)
def test_unrecognized_action_type_defaults_to_high_impact(action_type):
    """The load-bearing fail-safe default: an unrecognized action_type must
    never silently classify as SAFE (auto-executable)."""
    assert classify_risk(action_type) is RiskClassification.HIGH_IMPACT


def test_classify_risk_returns_a_riskclassification_instance():
    result = classify_risk("tag_incident")
    assert isinstance(result, RiskClassification)
