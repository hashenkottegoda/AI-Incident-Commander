"""Pydantic schema + loader for `failure_scenarios/*.yaml`.

BUILD_PLAN.md's "Failure Scenarios & Ground Truth" section specifies the
YAML shape informally; this module is the validated, typed version of
that shape, so the *next* build step (the telemetry generator + failure
injection engine) can load a scenario file and trust its structure
instead of re-parsing/re-checking raw dicts.

Kept deliberately independent of `backend.models`/SQLAlchemy: this module
only parses and validates YAML, it doesn't touch a database, so it (and
its tests) run with zero infrastructure. `severity` is duplicated here as
a `Literal` rather than importing `backend.models.incident.Severity`
directly — `tests/test_failure_scenarios.py` asserts the two stay in
sync, which keeps the intentional decoupling from silently drifting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# The Action Executor's fixed action palette (BUILD_PLAN.md's Agent
# Architecture section: "rollback_deployment(), restart_service(),
# scale_service(), disable_feature_flag(), increase_connection_pool()").
# `correct_remediation` and every entry in `ineffective_remediations` must
# be one of these — the simulator only knows how to produce post-action
# telemetry for these five.
ACTION_TYPES: frozenset[str] = frozenset(
    {
        "rollback_deployment",
        "restart_service",
        "scale_service",
        "disable_feature_flag",
        "increase_connection_pool",
    }
)

# The 3 services Phase 1's telemetry generator seeds (BUILD_PLAN.md Phase 1:
# "3 services (checkout, payment, inventory)"), named to match
# `tests/test_models.py`'s existing `Service.name` convention
# (`checkout-service`, `payment-service`, ...). Every scenario's
# `affected_service` must be one of these — a scenario naming a service
# the simulator never seeds would be untestable ground truth.
CANONICAL_SERVICES: frozenset[str] = frozenset(
    {"checkout-service", "payment-service", "inventory-service"}
)

# Repo layout: backend/simulation/scenario_schema.py -> backend/ -> repo root.
SCENARIOS_DIR: Path = Path(__file__).resolve().parents[2] / "failure_scenarios"


class RemediationEffects(BaseModel):
    """What fixes (or doesn't fix) a scenario — the Recovery Check's ground truth.

    `correct_remediation`/`on_correct` are both optional, and required to
    be null/non-null *together*: `slow_query` is the one scenario where no
    action in `ACTION_TYPES` actually resolves the incident (see
    `failure_scenarios/slow_query.yaml`'s comments), so the schema has to
    allow "there is no correct remediation" as a distinct, valid state
    rather than forcing a fake pick. When that's the case, Phase 6's
    Recovery Check should see every attempted action leave the incident
    degraded and the bounded re-investigation loop should exhaust into
    `manual_intervention_required`.
    """

    model_config = ConfigDict(extra="forbid")

    correct_remediation: str | None = None
    on_correct: dict[str, str] | None = None
    ineffective_remediations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_correct_and_on_correct_paired(self) -> RemediationEffects:
        has_correct = self.correct_remediation is not None
        has_on_correct = self.on_correct is not None
        if has_correct != has_on_correct:
            raise ValueError(
                "correct_remediation and on_correct must be both set or both null "
                f"(got correct_remediation={self.correct_remediation!r}, "
                f"on_correct={self.on_correct!r})"
            )
        return self

    @model_validator(mode="after")
    def _check_action_names(self) -> RemediationEffects:
        if self.correct_remediation is not None and self.correct_remediation not in ACTION_TYPES:
            raise ValueError(
                f"correct_remediation {self.correct_remediation!r} is not one of "
                f"{sorted(ACTION_TYPES)}"
            )
        unknown = set(self.ineffective_remediations) - ACTION_TYPES
        if unknown:
            raise ValueError(
                f"ineffective_remediations contains unknown action(s) {sorted(unknown)}; "
                f"must be a subset of {sorted(ACTION_TYPES)}"
            )
        return self

    @model_validator(mode="after")
    def _check_correct_not_also_ineffective(self) -> RemediationEffects:
        is_also_ineffective = self.correct_remediation in self.ineffective_remediations
        if self.correct_remediation is not None and is_also_ineffective:
            raise ValueError(
                f"correct_remediation {self.correct_remediation!r} cannot also appear in "
                "ineffective_remediations"
            )
        return self


class FailureScenario(BaseModel):
    """One `failure_scenarios/*.yaml` file, validated.

    Field order/naming matches BUILD_PLAN.md's worked example exactly
    (`db_connection_exhaustion.yaml`).
    """

    model_config = ConfigDict(extra="forbid")

    failure_type: str
    root_cause_category: str
    affected_service: str
    severity: Literal["P1", "P2", "P3", "P4"]
    # Ordered evidence tags Phase 2's get_logs/get_metrics/get_deployments
    # tools should be able to surface for this incident window; Phase 7's
    # eval harness matches these against tool-call source_refs.
    expected_evidence: list[str] = Field(min_length=1)
    # Ordered, not just the final root cause — see BUILD_PLAN.md: this is
    # what makes `cascading_payment_timeout` genuinely evaluable (did the
    # agent trace back past the loud symptom to the actual root cause?).
    causal_chain: list[str] = Field(min_length=1)
    remediation_effects: RemediationEffects


def load_scenario(path: Path | str) -> FailureScenario:
    """Read, parse, and validate one `failure_scenarios/*.yaml` file."""
    with Path(path).open("r") as f:
        data = yaml.safe_load(f)
    return FailureScenario.model_validate(data)


def load_all_scenarios(directory: Path | str = SCENARIOS_DIR) -> dict[str, FailureScenario]:
    """Load every `*.yaml` scenario in `directory`, keyed by `failure_type`."""
    return {
        scenario.failure_type: scenario
        for scenario in (load_scenario(path) for path in sorted(Path(directory).glob("*.yaml")))
    }
