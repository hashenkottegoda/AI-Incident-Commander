"""RECOVERY CHECK — Phase 6's final node: deterministic post-action verification.

BUILD_PLAN.md's Agent Architecture section, verbatim: *"RECOVERY CHECK
(re-read post-action synthetic metrics, compare to pre-incident baseline)"*
and, on determinism: *"Because `remediation_effects` is ground truth, both
the recovery decision and the remediation-eval metrics are deterministic."*
Also: *"Recovery Check is not a literal timed wait (no `sleep(10)` ... The
executor writes post-action metrics immediately; the recovery node reads
them and compares against the pre-incident baseline to decide `resolved`
vs. still-degraded."*

## Zero LLM calls

Same as `backend.agents.action_executor_node`: this is plain SQL +
arithmetic, no `ChatOpenRouter` call anywhere in this module. The comparison
below is what makes the outcome auditable/testable rather than a model's
subjective read of "does this look fixed" -- and it is genuinely
independent of `AuditEvent.execution_outcome` (already set by the Action
Executor): this node re-derives recovered-vs-degraded straight from the
real `MetricPoint` rows the executor just wrote, rather than trusting that
prior bookkeeping. Because the underlying simulation is deterministic by
construction (the executor wrote either clearly-recovered or
clearly-degraded values), the two are expected to always agree -- but the
Recovery Check computing its own answer from raw telemetry, rather than
reading a label another node already decided, is the point: it is what
makes "did the incident actually get fixed" a checkable claim about the
data instead of an assertion inherited from the Action Executor.

## Baseline sampling: earliest points, not "everything before detected_at"

`backend.simulation.injector.inject_failure` writes healthy baseline
telemetry across the *entire* pre-incident window, then overlays the
causal chain's anomaly ramp on top for whichever metrics a chain step
implicates -- so "all `MetricPoint` rows before `incident.detected_at`" is
NOT a clean healthy sample for an implicated metric (the tail of that
window already contains ramp-up values, see injector.py's own "Known
simplification" docstring note). Taking the **earliest** `BASELINE_SAMPLE_SIZE`
points instead sidesteps this: `inject_failure` always generates baseline
telemetry for the full window before it ever overlays the causal chain, and
every scenario's causal chain starts several minutes (or hours, for
`memory_leak`) before `incident_start` -- comfortably after the pre-incident
window's own start -- so the earliest points in that window are reliably
pre-anomaly for every scenario this sub-step targets
(`db_connection_exhaustion`, `cascading_payment_timeout`).

## Tolerance

`abs(post_action_mean - baseline_mean) <= max(2.0 * baseline_stddev, 0.20 *
abs(baseline_mean), ABSOLUTE_FLOOR)`:

- **2.0 baseline standard deviations** covers ordinary healthy jitter
  (baseline telemetry is itself `random.gauss(mean, stddev)`) without being
  so tight that normal noise reads as "still degraded."
- **20% of the baseline mean** is a floor for metrics whose baseline
  stddev is tiny relative to its mean (e.g. `error_rate`'s stddev is
  ~30% of its mean already) -- without a percentage floor, a
  stddev-only tolerance could be too strict for a low-variance metric
  recovering to a value that's still a normal, healthy amount off the
  sampled mean.
- **`ABSOLUTE_FLOOR` (0.01)** covers the degenerate edge case of a
  near-zero baseline mean/stddev (e.g. very few baseline samples),
  purely defensive.

This doesn't need to be more sophisticated than this: the ground truth
here is deterministic by construction (the executor wrote values either
drawn from the exact healthy baseline distribution, or anchored to the
last clearly-anomalous value), so recovered and degraded cases are never
close calls in practice -- this tolerance only has to reliably separate
"drawn from the same healthy distribution" from "anchored to a value
several multiples of baseline away," which it does with a wide margin for
every scenario in this codebase.

## Bounded re-investigation loop reuse

Per this sub-step's task brief: *"reuse/extend the existing
`investigation_iterations` bound and the existing re-investigation loop
machinery from the earlier Phase 5 conditional-edge work rather than
inventing a second, parallel bounding mechanism."* This node reads
`state.investigation_iterations` (incremented by `investigation_node`
itself each pass, see `backend.agents.routing`'s docstring) and compares
against the SAME `MAX_REINVESTIGATION_LOOPS` constant `route_after_root_cause`
already uses -- no second counter, no second constant. `route_after_recovery_check`
(`backend.agents.routing`) then simply reads the `incident_status` this
node sets, the same pattern `route_after_response_planner` already uses
against `response_planner_node`'s output.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.agents.action_executor_node import resolve_on_correct_targets
from backend.agents.routing import MAX_REINVESTIGATION_LOOPS
from backend.agents.state import IncidentState
from backend.models import (
    AuditDecisionStatus,
    AuditEvent,
    Incident,
    IncidentStatus,
    MetricPoint,
    RiskClassification,
    Service,
)
from backend.simulation.scenario_schema import load_all_scenarios

# "The earliest points in the pre-incident window are reliably pre-anomaly"
# -- see module docstring's "Baseline sampling" section.
BASELINE_SAMPLE_SIZE = 10

# See module docstring's "Tolerance" section for the reasoning behind each
# term.
BASELINE_STDDEV_MULTIPLE = 2.0
BASELINE_PERCENT_TOLERANCE = 0.20
ABSOLUTE_FLOOR = 0.01


def _get_services(db: Session, names: set[str]) -> dict[str, Service]:
    rows = db.execute(select(Service).where(Service.name.in_(names))).scalars().all()
    services = {service.name: service for service in rows}
    missing = names - services.keys()
    if missing:
        raise ValueError(f"service(s) {sorted(missing)} not found (expected canonical services)")
    return services


def _query_baseline_values(
    db: Session, service_id: int, metric_name: str, detected_at: datetime
) -> list[float]:
    """Earliest `BASELINE_SAMPLE_SIZE` `MetricPoint` values before
    `detected_at` -- see module docstring's "Baseline sampling" section for
    why earliest (not "all points before detected_at")."""
    rows = db.execute(
        select(MetricPoint.value)
        .where(
            MetricPoint.service_id == service_id,
            MetricPoint.metric_name == metric_name,
            MetricPoint.timestamp < detected_at,
        )
        .order_by(MetricPoint.timestamp.asc())
        .limit(BASELINE_SAMPLE_SIZE)
    ).scalars().all()
    return list(rows)


def _query_post_action_values(
    db: Session, service_id: int, metric_name: str, executed_at: datetime
) -> list[float]:
    """Every `MetricPoint` value at/after `executed_at` -- exactly the
    points `action_executor_node` just wrote for this `(service,
    metric)`."""
    rows = db.execute(
        select(MetricPoint.value)
        .where(
            MetricPoint.service_id == service_id,
            MetricPoint.metric_name == metric_name,
            MetricPoint.timestamp >= executed_at,
        )
        .order_by(MetricPoint.timestamp.asc())
    ).scalars().all()
    return list(rows)


def _mean_stddev(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, variance**0.5


def _compare_to_baseline(baseline_values: list[float], post_values: list[float]) -> dict[str, Any]:
    """The deterministic comparison itself -- see module docstring's
    "Tolerance" section."""
    baseline_mean, baseline_stddev = _mean_stddev(baseline_values)
    post_mean, _post_stddev = _mean_stddev(post_values)
    tolerance = max(
        BASELINE_STDDEV_MULTIPLE * baseline_stddev,
        BASELINE_PERCENT_TOLERANCE * abs(baseline_mean),
        ABSOLUTE_FLOOR,
    )
    recovered = abs(post_mean - baseline_mean) <= tolerance
    return {
        "baseline_mean": baseline_mean,
        "baseline_sample_size": len(baseline_values),
        "post_action_mean": post_mean,
        "post_action_sample_size": len(post_values),
        "tolerance": tolerance,
        "recovered": recovered,
    }


def make_recovery_check_node(db: Session):
    """Return a LangGraph node function bound to one request-scoped `db`.

    Factory pattern matches every other DB-touching node in this graph.
    Only ever reached when `action_executor_node` ran at least one
    HIGH_IMPACT remediation this pass (see `backend/graph.py`'s conditional
    edge out of `action_executor` -- a SAFE-only plan never routes here).
    """

    def recovery_check_node(state: IncidentState) -> dict[str, Any]:
        incident = db.get(Incident, state.incident_id)
        if incident is None:
            raise ValueError(f"incident {state.incident_id} not found")

        scenario = load_all_scenarios().get(incident.failure_type)
        if scenario is None:
            raise ValueError(f"no failure_scenarios/*.yaml found for {incident.failure_type!r}")

        # The most recently executed HIGH_IMPACT action -- what this pass
        # needs to verify. `action_executor_node` always runs strictly
        # before this node in the same pass (see backend/graph.py's
        # wiring), so this row is guaranteed to exist and be freshly
        # EXECUTED.
        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.incident_id == incident.id,
                AuditEvent.risk_classification == RiskClassification.HIGH_IMPACT,
                AuditEvent.decision_status == AuditDecisionStatus.EXECUTED,
            )
            .order_by(AuditEvent.executed_at.desc(), AuditEvent.id.desc())
            .first()
        )
        if event is None:
            raise ValueError(
                f"recovery_check reached for incident {incident.id} with no executed "
                "HIGH_IMPACT AuditEvent to verify -- graph wiring bug (see backend/graph.py)"
            )

        on_correct = scenario.remediation_effects.on_correct or {}
        targets = resolve_on_correct_targets(on_correct, scenario.affected_service)

        checks: dict[str, dict[str, Any]] = {}
        all_recovered = bool(targets)  # no targets at all -> never "recovered"
        if targets:
            services = _get_services(db, {service_name for _, service_name in targets})
            for metric_name, service_name in targets:
                service = services[service_name]
                baseline_values = _query_baseline_values(
                    db, service.id, metric_name, incident.detected_at
                )
                post_values = _query_post_action_values(
                    db, service.id, metric_name, event.executed_at
                )
                comparison = _compare_to_baseline(baseline_values, post_values)
                checks[f"{service_name}:{metric_name}"] = comparison
                all_recovered = all_recovered and comparison["recovered"]

        if all_recovered:
            incident.status = IncidentStatus.RESOLVED
        elif state.investigation_iterations > MAX_REINVESTIGATION_LOOPS:
            # Bound exhausted -- BUILD_PLAN.md: "back to INVESTIGATION
            # (bounded; exhausted -> MANUAL_INTERVENTION_REQUIRED)."
            incident.status = IncidentStatus.MANUAL_INTERVENTION_REQUIRED
        else:
            # Loop back to a fresh Investigation pass -- investigation_node
            # increments investigation_iterations itself on that next pass.
            incident.status = IncidentStatus.INVESTIGATING

        db.commit()

        recovery_result: dict[str, Any] = {
            "outcome": "recovered" if all_recovered else "still_degraded",
            "audit_event_id": event.id,
            "action_type": event.action_type,
            "checked_metrics": checks,
        }

        return {
            "incident_status": incident.status,
            "recovery_result": recovery_result,
        }

    return recovery_check_node
