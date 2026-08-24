"""Validates the 6 hand-authored `failure_scenarios/*.yaml` ground-truth
files against `backend.simulation.scenario_schema.FailureScenario`.

Pure YAML parsing + Pydantic validation — no Postgres, no live
infrastructure. This is the correctness bar for the ground-truth layer
that every later phase (tools, agents, eval harness) is scored against.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.models.incident import Severity
from backend.simulation.scenario_schema import (
    ACTION_TYPES,
    CANONICAL_SERVICES,
    SCENARIOS_DIR,
    FailureScenario,
    RemediationEffects,
    load_all_scenarios,
    load_scenario,
)

EXPECTED_FAILURE_TYPES = {
    "db_connection_exhaustion",
    "memory_leak",
    "bad_deployment",
    "dependency_failure",
    "slow_query",
    "cascading_payment_timeout",
}


def _scenario_files() -> list[Path]:
    return sorted(SCENARIOS_DIR.glob("*.yaml"))


def test_scenarios_dir_has_exactly_six_files():
    files = _scenario_files()
    assert {f.stem for f in files} == EXPECTED_FAILURE_TYPES
    assert len(files) == 6


@pytest.mark.parametrize("path", _scenario_files(), ids=lambda p: p.stem)
def test_scenario_file_parses_and_validates(path: Path):
    scenario = load_scenario(path)
    assert isinstance(scenario, FailureScenario)
    # File is named after its own failure_type (the loader/generator's
    # lookup key), e.g. `db_connection_exhaustion.yaml` -> failure_type
    # "db_connection_exhaustion".
    assert scenario.failure_type == path.stem


@pytest.mark.parametrize("path", _scenario_files(), ids=lambda p: p.stem)
def test_scenario_affected_service_is_canonical(path: Path):
    scenario = load_scenario(path)
    assert scenario.affected_service in CANONICAL_SERVICES, (
        f"{path.name}: affected_service {scenario.affected_service!r} is not one of "
        f"the 3 canonical simulated services {sorted(CANONICAL_SERVICES)}"
    )


def test_canonical_services_are_the_three_from_build_plan():
    # BUILD_PLAN.md Phase 1: "3 services (checkout, payment, inventory)",
    # named consistently with tests/test_models.py's Service.name convention.
    assert CANONICAL_SERVICES == {"checkout-service", "payment-service", "inventory-service"}


@pytest.mark.parametrize("path", _scenario_files(), ids=lambda p: p.stem)
def test_scenario_evidence_and_causal_chain_nonempty(path: Path):
    scenario = load_scenario(path)
    assert len(scenario.expected_evidence) >= 1
    assert len(scenario.causal_chain) >= 1


@pytest.mark.parametrize("path", _scenario_files(), ids=lambda p: p.stem)
def test_scenario_remediation_actions_are_known(path: Path):
    scenario = load_scenario(path)
    effects = scenario.remediation_effects
    if effects.correct_remediation is not None:
        assert effects.correct_remediation in ACTION_TYPES
        assert effects.correct_remediation not in effects.ineffective_remediations
    for action in effects.ineffective_remediations:
        assert action in ACTION_TYPES


def test_load_all_scenarios_returns_all_six_keyed_by_failure_type():
    scenarios = load_all_scenarios()
    assert set(scenarios.keys()) == EXPECTED_FAILURE_TYPES
    for failure_type, scenario in scenarios.items():
        assert scenario.failure_type == failure_type


def test_severity_literal_matches_incident_severity_enum():
    # scenario_schema.py deliberately duplicates Severity's values as a
    # Literal instead of importing the SQLAlchemy-backed enum (keeps YAML
    # parsing infra-free); this test is what keeps that duplication honest.
    literal_values = set(FailureScenario.model_fields["severity"].annotation.__args__)
    assert literal_values == {member.value for member in Severity}


def test_root_cause_categories_are_unique_across_scenarios():
    # Each scenario should carry a distinct root_cause_category so Phase
    # 7's enum-equality accuracy metric can actually distinguish them —
    # in particular cascading_payment_timeout's `upstream_dependency_timeout`
    # must differ from db_connection_exhaustion's `database_connection_pool`,
    # since conflating them is exactly the shortcut that scenario is
    # designed to catch.
    scenarios = load_all_scenarios()
    categories = [s.root_cause_category for s in scenarios.values()]
    assert len(categories) == len(set(categories))


def test_slow_query_has_no_correct_remediation():
    # The one scenario where none of the 5 standard executor actions
    # actually resolves the incident (see the YAML file's comments) —
    # ground truth is "manual_intervention_required", not a false "resolved".
    scenario = load_scenario(SCENARIOS_DIR / "slow_query.yaml")
    effects = scenario.remediation_effects
    assert effects.correct_remediation is None
    assert effects.on_correct is None
    assert set(effects.ineffective_remediations) == ACTION_TYPES


def test_cascading_payment_timeout_root_cause_is_not_a_db_category():
    scenario = load_scenario(SCENARIOS_DIR / "cascading_payment_timeout.yaml")
    assert scenario.root_cause_category != "database_connection_pool"
    assert scenario.causal_chain == [
        "payment_timeout",
        "checkout_retry_storm",
        "db_connection_growth",
        "database_overload",
        "checkout_failures",
    ]
    # Evidence must cover both the loud DB symptom and the quieter
    # payment-side root cause, or the agent has no way to find the latter.
    evidence = set(scenario.expected_evidence)
    assert {"db_connections_high", "database_overload_errors"} <= evidence
    assert {"payment_service_latency_high", "payment_service_timeout_errors"} <= evidence


def test_remediation_effects_rejects_correct_without_on_correct():
    with pytest.raises(ValidationError, match="must be both set or both null"):
        RemediationEffects(correct_remediation="restart_service", on_correct=None)


def test_remediation_effects_rejects_on_correct_without_correct():
    with pytest.raises(ValidationError, match="must be both set or both null"):
        RemediationEffects(
            correct_remediation=None, on_correct={"error_rate": "recovers_to_baseline"}
        )


def test_remediation_effects_rejects_unknown_correct_remediation():
    with pytest.raises(ValidationError, match="is not one of"):
        RemediationEffects(
            correct_remediation="reboot_the_datacenter",
            on_correct={"error_rate": "recovers_to_baseline"},
        )


def test_remediation_effects_rejects_unknown_ineffective_action():
    with pytest.raises(ValidationError, match="unknown action"):
        RemediationEffects(ineffective_remediations=["reboot_the_datacenter"])


def test_remediation_effects_rejects_correct_also_listed_as_ineffective():
    with pytest.raises(ValidationError, match="cannot also appear in"):
        RemediationEffects(
            correct_remediation="restart_service",
            on_correct={"error_rate": "recovers_to_baseline"},
            ineffective_remediations=["restart_service"],
        )


def test_remediation_effects_allows_both_null():
    effects = RemediationEffects(ineffective_remediations=list(ACTION_TYPES))
    assert effects.correct_remediation is None
    assert effects.on_correct is None


def test_db_connection_exhaustion_matches_build_plan_worked_example():
    scenario = load_scenario(SCENARIOS_DIR / "db_connection_exhaustion.yaml")
    assert scenario.root_cause_category == "database_connection_pool"
    assert scenario.affected_service == "checkout-service"
    assert scenario.severity == "P1"
    assert scenario.causal_chain == [
        "checkout_deployment_v1.8.2",
        "db_connection_growth",
        "connection_pool_exhausted",
        "checkout_failures",
    ]
    assert scenario.remediation_effects.correct_remediation == "rollback_deployment"
    assert set(scenario.remediation_effects.ineffective_remediations) == {
        "scale_service",
        "restart_service",
    }
