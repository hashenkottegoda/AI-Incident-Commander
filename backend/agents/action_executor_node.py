"""ACTION EXECUTOR — Phase 6's final sub-step: simulated remediation execution.

BUILD_PLAN.md's Agent Architecture section, verbatim: *"The Action Executor
performs *simulated* remediation against the simulation layer only
(`rollback_deployment()`, `restart_service()`, `scale_service()`,
`disable_feature_flag()`, `increase_connection_pool()`) — it writes new
synthetic telemetry representing the post-action system state, it never
touches anything real."* And Phase 6's own text: *"Action Executor
implementing simulated remediation functions ... whose post-action
telemetry is driven by the scenario's `remediation_effects`: applying the
`correct_remediation` produces recovered telemetry, an
`ineffective_remediation` leaves it degraded."*

## Zero LLM calls, by design

Everything in this module is deterministic Python + SQL, no
`ChatAnthropic`, no structured-output call. BUILD_PLAN.md: *"Because
`remediation_effects` is ground truth, both the recovery decision and the
remediation-eval metrics are deterministic."* Which telemetry to write is
decided by a plain string-equality check against
`FailureScenario.remediation_effects` (loaded from the already-committed
`failure_scenarios/*.yaml` ground truth for `incident.failure_type`), the
exact same "deterministic, code-level rule table, never an LLM decision"
philosophy `backend.agents.risk_classifier.classify_risk` already uses for
SAFE-vs-HIGH_IMPACT.

## Idempotency: only ever acts on not-yet-executed `AuditEvent` rows

This node is reachable either straight from `response_planner_node` (an
all-SAFE plan, no `interrupt()` needed) or from `human_approval_node` after
a resume. BUILD_PLAN.md: *"`interrupt()` must be side-effect safe ... the
Action Executor runs strictly after resume, never before."* On a replayed
pass (LangGraph re-executing this node from its start, or a duplicate
`POST /approve` reaching it a second time via `resume_incident_graph`),
this node only ever selects `AuditEvent` rows still in `APPROVED` or
`AUTO_EXECUTED` — exactly `backend/models/audit.py`'s documented guard
(`executed_at IS NULL`, expressed here as "decision_status hasn't reached
EXECUTED yet"). A row this node already executed is `EXECUTED` and is
never selected again, so re-entry can write telemetry at most once per
action — matching `human_approval_node`'s own idempotency argument.

## What "correct" vs "ineffective" means for telemetry

For each HIGH_IMPACT action executed:

- `action_type == remediation_effects.correct_remediation` → **recovered**
  telemetry: fresh `MetricPoint` rows for every metric named in
  `remediation_effects.on_correct` (mapped to a real `(metric_name,
  service)` pair via `ON_CORRECT_METRIC_MAP` below), drawn from that
  service's own healthy distribution (`backend.simulation.baseline.
  BASELINE_METRICS`) — the same distribution the injector itself draws
  baseline telemetry from, so a genuinely-fixed incident's post-action
  telemetry looks statistically identical to pre-incident health.
- `action_type` is one of `remediation_effects.ineffective_remediations`,
  OR is simply not `correct_remediation` at all (including an
  unrecognized action_type this scenario has never heard of) → **degraded**
  telemetry, anchored to whatever the most recent pre-action `MetricPoint`
  value for that `(service, metric)` already was (i.e. "stays bad"), with
  small jitter so it isn't a single flat point. Treating "not the known
  correct fix" as ineffective by default (rather than only the explicitly
  listed `ineffective_remediations`) is the same fail-safe default-deny
  precedent `risk_classifier.classify_risk` already applies to an
  unrecognized `action_type` — an action this code doesn't recognize as
  correct is never assumed to have fixed anything.

`incident_status` key inside `on_correct` is a hint about the *overall*
scenario outcome ("resolved"), not a metric name — always skipped; see
`FailureScenario.RemediationEffects`'s docstring.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.agents.state import IncidentState
from backend.models import (
    AuditDecisionStatus,
    AuditEvent,
    ExecutionOutcome,
    Incident,
    IncidentStatus,
    MetricPoint,
    RiskClassification,
    Service,
)
from backend.simulation.baseline import BASELINE_METRICS
from backend.simulation.scenario_schema import FailureScenario, load_all_scenarios

# "A handful of points with small jitter, not a single flat point" (this
# sub-step's task brief) -- mirrors the injector's own multi-point-ramp
# philosophy (backend.simulation.injector) rather than inventing a new
# telemetry shape. 30s spacing matches injector.py's own
# _METRIC_RAMP_INTERVAL_SECONDS convention.
POST_ACTION_POINT_COUNT = 5
POST_ACTION_POINT_INTERVAL_SECONDS = 30

# Maps a scenario's `remediation_effects.on_correct` key to the real metric
# name + an optional service-name override. Built by reading every
# `on_correct` key across all 6 `failure_scenarios/*.yaml` files by hand --
# same methodology as `backend.simulation.injector`'s causal-chain
# classification tables (METRIC_RAMP_SPECS et al). `None` as the service
# override means "the scenario's own `affected_service`"; a `payment_*`-
# prefixed key means the *dependency's* service recovered too (e.g.
# `cascading_payment_timeout`'s `payment_error_rate`/`dependency_failure`'s
# `payment_latency`), mirroring injector.py's own `_resolve_target_service`
# prefix convention for routing a causal_chain entry to a specific service.
ON_CORRECT_METRIC_MAP: dict[str, tuple[str, str | None]] = {
    "error_rate": ("error_rate", None),
    "db_connections": ("db_connections_active", None),
    "memory_usage": ("memory_usage_mb", None),
    "payment_error_rate": ("error_rate", "payment-service"),
    "payment_latency": ("latency_p99_ms", "payment-service"),
}

# Not a real metric -- an outcome hint ("resolved") describing the overall
# scenario, see FailureScenario.RemediationEffects's docstring.
_NON_METRIC_ON_CORRECT_KEYS = frozenset({"incident_status"})


def _rng_for(*parts: object) -> random.Random:
    """A deterministic, per-(incident, action, metric, outcome) RNG seed --
    zero LLM calls and zero flakiness (no dependency on wall-clock or the
    global `random` module), while still varying the jitter sequence across
    different metrics/actions/incidents rather than reusing one fixed
    sequence everywhere."""
    return random.Random(":".join(str(part) for part in parts))


def resolve_on_correct_targets(
    on_correct: dict[str, str], affected_service: str
) -> list[tuple[str, str]]:
    """Translate `remediation_effects.on_correct`'s keys into concrete
    `(metric_name, service_name)` pairs to write/verify telemetry for.

    Shared with `backend.agents.recovery_check_node` (imported from here)
    so both sides of "what recovering looks like" agree on the exact same
    metric/service set by construction, rather than two independently
    maintained copies of this mapping drifting apart.

    Raises `ValueError` on an `on_correct` key this module has never seen
    before (i.e. not in `ON_CORRECT_METRIC_MAP` and not the special
    `incident_status` hint) -- a new scenario YAML introducing a new
    `on_correct` key without a corresponding code change here is a real
    configuration gap, not something to silently ignore (same "auditable,
    fails loudly" philosophy as `RemediationEffects`'s own validators).
    """
    targets: list[tuple[str, str]] = []
    for key in on_correct:
        if key in _NON_METRIC_ON_CORRECT_KEYS:
            continue
        mapping = ON_CORRECT_METRIC_MAP.get(key)
        if mapping is None:
            raise ValueError(
                f"remediation_effects.on_correct key {key!r} has no known metric mapping in "
                f"ON_CORRECT_METRIC_MAP (backend.agents.action_executor_node) -- add it there."
            )
        metric_name, service_override = mapping
        targets.append((metric_name, service_override or affected_service))
    return targets


def _get_services(db: Session, names: set[str]) -> dict[str, Service]:
    rows = db.execute(select(Service).where(Service.name.in_(names))).scalars().all()
    services = {service.name: service for service in rows}
    missing = names - services.keys()
    if missing:
        raise ValueError(f"service(s) {sorted(missing)} not found (expected canonical services)")
    return services


def _write_recovered_points(
    db: Session,
    service: Service,
    metric_name: str,
    start: datetime,
    incident_id: int,
    action_type: str,
) -> list[MetricPoint]:
    """Draw `POST_ACTION_POINT_COUNT` points from `service`'s own healthy
    baseline distribution (`BASELINE_METRICS`) -- the *correct* remediation
    was applied, so post-action telemetry should look statistically
    identical to pre-incident health, not just "close to one target
    value."""
    baseline = next(b for b in BASELINE_METRICS[service.name] if b.name == metric_name)
    rng = _rng_for(incident_id, action_type, metric_name, "recovered")
    points: list[MetricPoint] = []
    cursor = start
    for _ in range(POST_ACTION_POINT_COUNT):
        value = max(rng.gauss(baseline.mean, baseline.stddev), baseline.minimum)
        points.append(
            MetricPoint(
                service_id=service.id, timestamp=cursor, metric_name=metric_name, value=value
            )
        )
        cursor += timedelta(seconds=POST_ACTION_POINT_INTERVAL_SECONDS)
    db.add_all(points)
    return points


def _write_degraded_points(
    db: Session,
    service: Service,
    metric_name: str,
    start: datetime,
    detected_at: datetime,
    incident_id: int,
    action_type: str,
) -> list[MetricPoint]:
    """Anchor to the WORST (highest) pre-incident `MetricPoint` value for
    this `(service, metric)` -- "reuse whatever the last known degraded
    metric values were" per this sub-step's task brief -- so a still-broken
    incident's post-action telemetry stays visibly at the same anomalous
    level the injector already produced, with only small jitter so it
    isn't a single flat point.

    Anchored against `MAX(value)` over everything timestamped before
    `incident.detected_at` (not "most recent point before the execution
    timestamp `start`", and not scoped to a lookback window) -- deliberately,
    for two reasons:

    1. `inject_failure` writes healthy baseline telemetry across the
       *entire* pre-incident window, right up to and including a point
       timestamped exactly at `detected_at` itself (see
       `backend.simulation.baseline.generate_baseline_telemetry`'s
       inclusive `while cursor <= end` loop) -- *after* the causal chain's
       own anomaly ramp has already stopped (`telemetry_end = incident_start
       - detection_lag`, strictly before `detected_at`). So "most recent
       point before `start`" (where `start` is the real wall-clock
       execution time, always long after `detected_at`) would pick up that
       trailing, un-anomalous baseline point instead of the actual
       injected anomaly -- exactly backwards for "stays degraded". Every
       metric in this codebase's anomaly model is "higher is worse"
       (error_rate, db_connections_active, latency_p99_ms, memory_usage_mb,
       gc_pause_ms), so the single highest pre-`detected_at` value reliably
       identifies the true injected anomaly peak regardless of any
       baseline-vs-ramp overlap noise near the tail (injector.py's own
       documented "Known simplification").
    2. It also works correctly on a second (or third) ineffective action in
       the same incident's bounded re-investigation loop: those later
       executor passes run long after `detected_at`, so they never
       pollute what "the worst pre-incident value" means -- this anchor is
       always the original injected anomaly, not whatever a prior
       ineffective attempt happened to write.
    """
    worst = db.execute(
        select(func.max(MetricPoint.value)).where(
            MetricPoint.service_id == service.id,
            MetricPoint.metric_name == metric_name,
            MetricPoint.timestamp < detected_at,
        )
    ).scalar_one_or_none()

    baseline = next(b for b in BASELINE_METRICS[service.name] if b.name == metric_name)
    if worst is not None:
        anchor = worst
    else:
        # Defensive fallback only -- inject_failure always writes baseline
        # (and, for an implicated metric, ramp) telemetry before
        # detected_at, so a prior point should always exist in practice.
        # Anchor to a clearly-anomalous multiple of baseline mean so
        # "still degraded" stays unambiguous even in this shouldn't-happen
        # case.
        anchor = baseline.mean * 5.0 + baseline.stddev

    rng = _rng_for(incident_id, action_type, metric_name, "degraded")
    jitter_stddev = baseline.stddev * 0.5
    points: list[MetricPoint] = []
    cursor = start
    for _ in range(POST_ACTION_POINT_COUNT):
        value = max(rng.gauss(anchor, jitter_stddev), 0.0)
        points.append(
            MetricPoint(
                service_id=service.id, timestamp=cursor, metric_name=metric_name, value=value
            )
        )
        cursor += timedelta(seconds=POST_ACTION_POINT_INTERVAL_SECONDS)
    db.add_all(points)
    return points


def is_correct_remediation(scenario: FailureScenario, action_type: str) -> bool:
    """Whether `action_type` is exactly this scenario's ground-truth
    `remediation_effects.correct_remediation`.

    Shared with `backend.evaluation.scoring` (imported from here, same
    pattern as `resolve_on_correct_targets` below) so the Action Executor's
    real "did this action recover the incident" decision and Phase 7's
    operational-eval scoring can never independently drift apart on what
    "correct" means -- one definition, two call sites.
    """
    effects = scenario.remediation_effects
    return effects.correct_remediation is not None and action_type == effects.correct_remediation


def _execute_high_impact_action(
    db: Session,
    scenario: FailureScenario,
    incident: Incident,
    event: AuditEvent,
    now: datetime,
) -> ExecutionOutcome:
    """Write this action's post-action telemetry and mark `event` executed.
    Returns the `ExecutionOutcome` so the caller can decide the incident's
    overall status."""
    effects = scenario.remediation_effects
    matched_correct = is_correct_remediation(scenario, event.action_type)
    matched_known_ineffective = event.action_type in effects.ineffective_remediations

    targets = resolve_on_correct_targets(effects.on_correct or {}, scenario.affected_service)
    written_metrics: dict[str, dict[str, float | int]] = {}

    if targets:
        services = _get_services(db, {service_name for _, service_name in targets})
        for metric_name, service_name in targets:
            service = services[service_name]
            if matched_correct:
                points = _write_recovered_points(
                    db, service, metric_name, now, event.incident_id, event.action_type
                )
            else:
                points = _write_degraded_points(
                    db,
                    service,
                    metric_name,
                    now,
                    incident.detected_at,
                    event.incident_id,
                    event.action_type,
                )
            written_metrics[f"{service_name}:{metric_name}"] = {
                "count": len(points),
                "mean_value": sum(p.value for p in points) / len(points),
            }

    # No `on_correct` targets at all (e.g. slow_query, whose
    # remediation_effects has no correct_remediation/on_correct pair) means
    # there is nothing this scenario defines as "recovering" -- never
    # treated as recovered regardless of matched_correct.
    outcome = (
        ExecutionOutcome.RECOVERED
        if (matched_correct and targets)
        else ExecutionOutcome.STILL_DEGRADED
    )

    event.decision_status = AuditDecisionStatus.EXECUTED
    event.executed_at = now
    event.execution_outcome = outcome
    event.execution_detail = {
        "matched_correct_remediation": matched_correct,
        "matched_known_ineffective_remediation": matched_known_ineffective,
        "written_metrics": written_metrics,
    }
    return outcome


def _execute_safe_action(event: AuditEvent, now: datetime) -> None:
    """SAFE actions are purely informational (generate report / add note /
    gather diagnostics / tag incident) -- no telemetry effect, no recovery
    outcome to record (see `backend/models/audit.py`'s docstring: SAFE
    actions never have a meaningful `execution_outcome`)."""
    event.decision_status = AuditDecisionStatus.EXECUTED
    event.executed_at = now
    event.execution_detail = {"note": "safe/informational action, no telemetry effect"}


def make_action_executor_node(db: Session):
    """Return a LangGraph node function bound to one request-scoped `db`.

    Factory pattern matches every other DB-touching node in this graph
    (`make_investigation_node(db)`, `make_response_planner_node(db)`).
    """

    def action_executor_node(state: IncidentState) -> dict[str, Any]:
        incident = db.get(Incident, state.incident_id)
        if incident is None:
            raise ValueError(f"incident {state.incident_id} not found")

        # Idempotency guard (see module docstring): only rows still
        # APPROVED (HIGH_IMPACT, just resumed past interrupt()) or
        # AUTO_EXECUTED (SAFE, never gated) are actionable. A row already
        # EXECUTED here on a prior pass is never re-selected.
        actionable = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.incident_id == incident.id,
                AuditEvent.decision_status.in_(
                    (AuditDecisionStatus.APPROVED, AuditDecisionStatus.AUTO_EXECUTED)
                ),
            )
            .order_by(AuditEvent.id)
            .all()
        )

        if not actionable:
            # Nothing left to execute -- e.g. a replayed/duplicate entry
            # into this node after everything it would have executed was
            # already executed on a prior pass. Leave incident_status
            # alone; whatever set it (VERIFYING/DIAGNOSED) already reflects
            # reality.
            return {"execution_result_id": None}

        scenario = load_all_scenarios().get(incident.failure_type)
        if scenario is None:
            raise ValueError(f"no failure_scenarios/*.yaml found for {incident.failure_type!r}")

        now = datetime.now(UTC)
        executed_ids: list[int] = []
        any_high_impact = False

        for event in actionable:
            executed_ids.append(event.id)
            if event.risk_classification is RiskClassification.HIGH_IMPACT:
                any_high_impact = True
                _execute_high_impact_action(db, scenario, incident, event, now)
            else:
                _execute_safe_action(event, now)

        # Only a HIGH_IMPACT remediation needs the Recovery Check's
        # metric-comparison step; a SAFE-only plan has nothing to verify.
        # DIAGNOSED is the closest existing IncidentStatus fit for "a
        # purely informational action ran and nothing else is pending" --
        # BUILD_PLAN.md's lifecycle enum has no dedicated value for this
        # case, and the incident is still open/undiagnosed-to-resolution
        # after an informational action, which DIAGNOSED already conveys
        # (evidence gathered, no fix applied yet) better than any other
        # existing value.
        incident.status = IncidentStatus.VERIFYING if any_high_impact else IncidentStatus.DIAGNOSED
        db.commit()

        return {
            "execution_result_id": executed_ids[0] if len(executed_ids) == 1 else executed_ids,
            "incident_status": incident.status,
        }

    return action_executor_node
