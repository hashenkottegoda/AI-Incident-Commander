"""Failure injection engine: turns one `FailureScenario` into a temporally
coherent Postgres timeline (BUILD_PLAN.md Phase 1's core simulation piece).

`inject_failure()` is the single entry point. Given a scenario (loaded via
`scenario_schema.load_scenario`/`load_all_scenarios`), an explicit
`random.Random(seed)`, and a target `incident_start` timestamp, it:

1. Ensures the 3 canonical services exist.
2. Writes "healthy" baseline telemetry (`baseline.generate_baseline_telemetry`)
   for all 3 services across a pre-incident window, so there's normal data
   before the anomaly and a same-shaped "unaffected service" contrast.
3. Walks the scenario's ordered `causal_chain` and writes staggered,
   escalating telemetry for each step (see "Causal-chain heuristic" below) —
   this is what makes the timeline "deployment at T, connections rising at
   T+1, ..., incident triggered at T+5" rather than a single `error=true`
   flag.
4. Creates and returns the `Incident` row (status `DETECTED`), carrying the
   scenario's ground truth (`failure_type`, `root_cause_category`,
   `severity`) plus optional `scenario_seed`/`scenario_instance_index`
   provenance for Phase 7's eval harness.

Determinism: like `baseline.generate_baseline_telemetry`, this never
touches the global `random` module — every draw goes through the
caller-supplied `rng`, and pure builder functions are called in a fixed
order (sorted canonical service names for baseline, then `causal_chain`'s
own list order for the anomaly overlay) so the same seed always reproduces
byte-identical rows.

## Causal-chain heuristic

Every `causal_chain` entry across the 6 hand-authored `failure_scenarios/
*.yaml` files (22 unique strings) was read and classified by hand into one
of five telemetry "kinds":

- **DEPLOYMENT** — entry names a version (`_v<digits>` or contains
  "deployment", e.g. `checkout_deployment_v1.8.2`) -> one `Deployment` row.
- **DEPENDENCY** — entry describes a cross-service call becoming unhealthy
  (`checkout_retry_storm`, `checkout_dependency_errors`) -> a handful of
  `TraceLite` spans from the affected service to the named downstream
  service (`downstream_service_id`), with `duration_ms` ramping up, plus
  one `LogEntry` (level/message per entry — see `DEPENDENCY_ENTRIES`).
- **DISCRETE_EVENT** — a one-off state-change marker that isn't itself a
  metric trend (`payment_canary_flag_enabled`, `oom_kill_event`,
  `inventory_slow_query_detected`) -> a single `LogEntry` at an
  entry-specific level.
- **METRIC_RAMP** — entry names a climbing resource metric
  (`db_connection_growth`, `memory_usage_climbing`, `gc_pause_increase`,
  `*_latency_high`, `payment_timeout`) -> `MetricPoint` rows that
  *linearly ramp* (with small jitter) from that service's baseline mean
  toward an anomalous target across `[step_time, telemetry_end]` — not a
  step function, an actual trend.
- **ERROR_CLUSTER** — entry describes failures/errors becoming visible
  (`checkout_failures`, `http_500_spike`, `connection_pool_exhausted`,
  ...) -> both an `error_rate` `MetricPoint` ramp *and* several
  ERROR-level `LogEntry` rows clustered in the final minutes before
  `telemetry_end`.

Classification is exact-match against the tables below (`METRIC_RAMP_SPECS`,
`ERROR_CLUSTER_ENTRIES`, `DISCRETE_EVENT_SPECS`, `DEPENDENCY_ENTRIES`, plus
a `_v<digits>`/"deployment" regex check) — built by literally reading each
scenario file's `causal_chain`, not a generic NLP parser. An entry that
doesn't match any table (e.g. a future 7th scenario) falls back to a single
WARN `LogEntry` carrying the raw entry text, so an unrecognized tag never
crashes the injector; extend the tables above when adding new scenarios.

**Service routing:** most entries don't name a service explicitly (e.g.
`db_connection_growth`), so they route to the scenario's own
`affected_service`. Entries that *do* carry an explicit `payment_`/
`checkout_`/`inventory_` prefix route to that service instead — this is
what lets `cascading_payment_timeout`'s and `dependency_failure`'s chain
entries correctly land evidence on `payment-service` even though the
*incident* is filed against `checkout-service` (see
`_resolve_target_service`).

**Timing:** given an `N`-step `causal_chain`, step `i` (0-indexed) is
timestamped `incident_start - (N - i) * chain_stagger` — the last step
lands `chain_stagger` before `incident_start`, the first step
`N * chain_stagger` before it. All chain-driven telemetry (ramps, log
bursts, trace spans) is capped at `telemetry_end = incident_start -
detection_lag`, not `incident_start` itself, so the `Incident.detected_at`
timestamp sits a beat *after* the last observable anomaly signal (matching
BUILD_PLAN.md's "...HTTP 500s at T+4, incident triggered at T+5").

**Known simplification:** the pre-incident baseline (step 2 above) is
generated for the *full* `[window_start, incident_start]` window for all 3
services, and the causal-chain overlay (step 3) writes *additional*
`MetricPoint` rows on top for whichever metrics a chain step implicates.
For an implicated metric this means baseline jitter and anomaly-ramp rows
coexist at nearby timestamps near the incident — acceptable noise for a
demo simulator (arguably realistic: multiple collectors reporting), and
because the ramp interval (30s) is denser than baseline's (60s), the
ramp dominates the tail of the window and the upward trend still reads
clearly in a query.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from random import Random

from sqlalchemy.orm import Session

from backend.models import (
    Deployment,
    Incident,
    IncidentStatus,
    LogEntry,
    LogLevel,
    MetricPoint,
    Service,
    Severity,
    TraceLite,
)
from backend.simulation.baseline import (
    BASELINE_METRICS,
    generate_baseline_telemetry,
    get_or_create_canonical_services,
)
from backend.simulation.scenario_schema import CANONICAL_SERVICES, FailureScenario

# --- Timing defaults ------------------------------------------------------

DEFAULT_PRE_INCIDENT_WINDOW: timedelta = timedelta(minutes=45)
DEFAULT_CHAIN_STAGGER: timedelta = timedelta(minutes=2)
DEFAULT_DETECTION_LAG: timedelta = timedelta(seconds=60)

# Per-scenario timing overrides, applied unless the caller passes an
# explicit `chain_stagger`/`pre_incident_window`. Two scenarios document
# their own timing intent in their YAML file's header comment and that
# intent isn't honored by the uniform defaults above:
#   - bad_deployment: "the causal_chain timestamps ... should be
#     tight/back-to-back rather than spread over minutes" (a code bug
#     throws errors immediately after rollout, not a slow buildup).
#   - memory_leak: "a slow, organic leak ... that accumulates over hours."
SCENARIO_TIMING_OVERRIDES: dict[str, dict[str, timedelta]] = {
    "bad_deployment": {
        "chain_stagger": timedelta(seconds=20),
        "pre_incident_window": timedelta(minutes=10),
    },
    "memory_leak": {
        "chain_stagger": timedelta(hours=1),
        "pre_incident_window": timedelta(hours=6),
    },
}

_METRIC_RAMP_INTERVAL_SECONDS: int = 30
_ERROR_LOG_BURST_COUNT: int = 3
_ERROR_LOG_BURST_WINDOW: timedelta = timedelta(minutes=3)
_DEPENDENCY_SPAN_COUNT: int = 5

# --- Classification tables (see module docstring's "Causal-chain
# heuristic") -----------------------------------------------------------

_VERSION_RE = re.compile(r"v\d+(?:\.\d+)*")


def _looks_like_deployment(entry: str) -> bool:
    return "deployment" in entry or _VERSION_RE.search(entry) is not None


# entry -> (metric_name, target = baseline_mean * multiplier)
METRIC_RAMP_SPECS: dict[str, tuple[str, float]] = {
    "db_connection_growth": ("db_connections_active", 5.5),
    # slow_query: queueing, not exhaustion
    "db_connection_queue_growth": ("db_connections_active", 2.5),
    "memory_usage_climbing": ("memory_usage_mb", 2.2),
    "gc_pause_increase": ("gc_pause_ms", 5.0),
    "payment_service_latency_high": ("latency_p99_ms", 4.0),
    "inventory_query_latency_high": ("latency_p99_ms", 4.0),
    "payment_timeout": ("latency_p99_ms", 4.0),  # cascading's quiet root-cause signal
}

# Some METRIC_RAMP entries also need an accompanying ERROR-level log burst
# to satisfy a scenario's expected_evidence — e.g. cascading_payment_timeout
# lists `payment_service_timeout_errors` as a required evidence tag, but its
# causal_chain uses the coarser `payment_timeout` tag (per BUILD_PLAN.md's
# exact given chain). Without this, `payment_timeout` would only ever
# produce a latency ramp and the promised "quiet root-cause" log evidence
# would never actually exist to be queried. Maps a METRIC_RAMP entry to the
# ERROR_CLUSTER_MESSAGES key whose message templates it should also emit.
METRIC_RAMP_ERROR_BURST: dict[str, str] = {
    "payment_timeout": "payment_service_timeout_errors",
}

# entries that mean "failures/errors are now visible": ramp error_rate +
# write a burst of ERROR logs. Severe entries (pool/db fully overloaded)
# ramp error_rate to a harder ceiling than a generic failure spike.
NORMAL_ERROR_RATE_TARGET: float = 0.30
SEVERE_ERROR_RATE_TARGET: float = 0.40
SEVERE_ERROR_CLUSTER_ENTRIES: frozenset[str] = frozenset(
    {"connection_pool_exhausted", "database_overload"}
)

ERROR_CLUSTER_MESSAGES: dict[str, tuple[str, ...]] = {
    "connection_pool_exhausted": (
        "{service} connection pool exhausted: 0 connections available",
        "{service} request rejected: no database connections available",
    ),
    "checkout_failures": (
        "{service} checkout request failed: unable to complete order",
        "{service} returned HTTP 500 to client",
    ),
    "inventory_failures_high": (
        "{service} request failed: internal error",
        "{service} returned HTTP 500 to client",
    ),
    "error_rate_spike": ("{service} error rate spike detected",),
    "http_500_spike": ("{service} returned HTTP 500 to client",),
    "payment_request_failures": ("{service} payment authorization request failed",),
    "payment_service_timeout_errors": ("{service} upstream payment provider request timed out",),
    # dependency_failure's provider error is deliberately NOT timeout-framed
    # (contrast with payment_service_timeout_errors above, shared with
    # cascading_payment_timeout) -- see dependency_failure.yaml's header
    # comment for why the two scenarios need textually distinguishable
    # evidence, not just different root_cause_category strings.
    "payment_service_error_responses": (
        "{service} upstream payment provider returned HTTP 502 Bad Gateway",
        "{service} payment provider integration returned an unexpected error response",
    ),
    "inventory_request_timeouts": ("{service} inventory lookup request timed out",),
    "database_overload": ("{service} database overload: query queue full, requests timing out",),
}
ERROR_CLUSTER_ATTRIBUTES: dict[str, dict] = {
    "connection_pool_exhausted": {"pool_size": 20, "active_connections": 20},
}
ERROR_CLUSTER_ENTRIES: frozenset[str] = frozenset(ERROR_CLUSTER_MESSAGES)

# entry -> (level, message template, attributes)
DISCRETE_EVENT_SPECS: dict[str, tuple[LogLevel, str, dict | None]] = {
    "payment_canary_flag_enabled": (
        LogLevel.WARN,
        "{service} feature flag 'payment_v2_provider_canary' enabled",
        {"flag": "payment_v2_provider_canary", "enabled": True},
    ),
    "oom_kill_event": (
        LogLevel.ERROR,
        "{service} process terminated by OOM killer",
        {"event": "oom_kill"},
    ),
    "inventory_slow_query_detected": (
        LogLevel.WARN,
        "{service} detected slow query on hot lookup path (possible missing index)",
        {"event": "slow_query_detected"},
    ),
}

# entry -> (downstream service name, log level, message template).
# `checkout_dependency_errors` must actually be generated for
# dependency_failure.yaml, whose expected_evidence lists it: without a real
# entry here, get_dependencies would return no spans at all for that
# scenario, leaving no evidence of a downstream dependency to investigate.
DEPENDENCY_ENTRIES: dict[str, tuple[str, LogLevel, str]] = {
    "checkout_retry_storm": (
        "payment-service",
        LogLevel.WARN,
        "{service} retrying {downstream} request after timeout",
    ),
    "checkout_dependency_errors": (
        "payment-service",
        LogLevel.ERROR,
        "{service} downstream call to {downstream} failed: canary payment provider error",
    ),
}


def _resolve_target_service(
    entry: str, scenario: FailureScenario, services: dict[str, Service]
) -> Service:
    """Route a causal_chain entry to the service it describes.

    Most entries don't name a service (e.g. `db_connection_growth`) and
    route to the scenario's own `affected_service`. Entries with an
    explicit `payment_`/`checkout_`/`inventory_` prefix route there
    instead — see module docstring's "Service routing".
    """
    if entry.startswith("payment_"):
        return services["payment-service"]
    if entry.startswith("checkout_"):
        return services["checkout-service"]
    if entry.startswith("inventory_"):
        return services["inventory-service"]
    return services[scenario.affected_service]


def _parse_version(entry: str) -> str:
    match = _VERSION_RE.search(entry)
    return match.group(0) if match else "v1.0.0"


def _ramp_metric_points(
    service: Service,
    metric_name: str,
    ramp_start: datetime,
    ramp_end: datetime,
    target_value: float,
    rng: Random,
    interval_seconds: int = _METRIC_RAMP_INTERVAL_SECONDS,
) -> list[MetricPoint]:
    """Linearly interpolate `metric_name` from its baseline mean toward
    `target_value` across `[ramp_start, ramp_end]`, with small jitter
    (half the metric's normal baseline stddev, so the trend stays visible
    after jitter) — "an actual ramp ... not a step function"."""
    baseline = next(b for b in BASELINE_METRICS[service.name] if b.name == metric_name)
    if ramp_end <= ramp_start:
        ramp_end = ramp_start + timedelta(seconds=interval_seconds)
    total_seconds = (ramp_end - ramp_start).total_seconds()

    points: list[MetricPoint] = []
    cursor = ramp_start
    while cursor <= ramp_end:
        fraction = (cursor - ramp_start).total_seconds() / total_seconds
        interpolated = baseline.mean + fraction * (target_value - baseline.mean)
        jitter = rng.gauss(0.0, baseline.stddev * 0.5)
        value = max(interpolated + jitter, baseline.minimum)
        points.append(
            MetricPoint(
                service_id=service.id, timestamp=cursor, metric_name=metric_name, value=value
            )
        )
        cursor += timedelta(seconds=interval_seconds)
    return points


def _error_log_burst(
    service: Service,
    window_start: datetime,
    window_end: datetime,
    rng: Random,
    messages: tuple[str, ...],
    attributes: dict | None,
    count: int = _ERROR_LOG_BURST_COUNT,
) -> list[LogEntry]:
    """A handful of ERROR-level logs scattered across `[window_start,
    window_end]` ("clustered near incident_start")."""
    if window_end <= window_start:
        window_end = window_start + timedelta(seconds=1)
    span_seconds = (window_end - window_start).total_seconds()

    logs = [
        LogEntry(
            service_id=service.id,
            timestamp=window_start + timedelta(seconds=rng.uniform(0, span_seconds)),
            level=LogLevel.ERROR,
            message=rng.choice(messages).format(service=service.name),
            attributes=attributes,
        )
        for _ in range(count)
    ]
    logs.sort(key=lambda log_: log_.timestamp)
    return logs


def _dependency_spans(
    source: Service,
    downstream: Service,
    ramp_start: datetime,
    ramp_end: datetime,
    rng: Random,
    count: int = _DEPENDENCY_SPAN_COUNT,
    base_duration_ms: float = 250.0,
    target_duration_ms: float = 3000.0,
) -> list[TraceLite]:
    """`count` spans from `source` to `downstream`, `duration_ms` ramping
    up across `[ramp_start, ramp_end]` (a growing retry/timeout stall)."""
    if ramp_end <= ramp_start:
        ramp_end = ramp_start + timedelta(seconds=1)
    total_seconds = (ramp_end - ramp_start).total_seconds()

    spans: list[TraceLite] = []
    for i in range(count):
        fraction = i / (count - 1) if count > 1 else 1.0
        timestamp = ramp_start + timedelta(seconds=fraction * total_seconds)
        duration = base_duration_ms + fraction * (target_duration_ms - base_duration_ms)
        duration += rng.uniform(-50.0, 50.0)
        spans.append(
            TraceLite(
                service_id=source.id,
                timestamp=timestamp,
                span_name=f"{source.name}->{downstream.name}",
                duration_ms=max(duration, 1.0),
                downstream_service_id=downstream.id,
            )
        )
    return spans


def _apply_causal_chain(
    scenario: FailureScenario,
    services: dict[str, Service],
    rng: Random,
    incident_start: datetime,
    chain_stagger: timedelta,
    detection_lag: timedelta,
) -> list[Deployment | LogEntry | MetricPoint | TraceLite]:
    """Walk `scenario.causal_chain` in order and build the staggered,
    escalating telemetry rows for each step (see module docstring)."""
    chain = scenario.causal_chain
    n = len(chain)
    telemetry_end = incident_start - detection_lag

    rows: list[Deployment | LogEntry | MetricPoint | TraceLite] = []
    for i, entry in enumerate(chain):
        step_time = incident_start - (n - i) * chain_stagger
        event_time = min(step_time, telemetry_end)
        target = _resolve_target_service(entry, scenario, services)

        if _looks_like_deployment(entry):
            rows.append(
                Deployment(
                    service_id=target.id, version=_parse_version(entry), deployed_at=step_time
                )
            )
            continue

        if entry in DEPENDENCY_ENTRIES:
            downstream_name, log_level, message_template = DEPENDENCY_ENTRIES[entry]
            downstream = services[downstream_name]
            rows.extend(_dependency_spans(target, downstream, step_time, telemetry_end, rng))
            rows.append(
                LogEntry(
                    service_id=target.id,
                    timestamp=event_time,
                    level=log_level,
                    message=message_template.format(
                        service=target.name, downstream=downstream.name
                    ),
                    attributes={"downstream_service": downstream.name},
                )
            )
            continue

        if entry in DISCRETE_EVENT_SPECS:
            level, template, attrs = DISCRETE_EVENT_SPECS[entry]
            rows.append(
                LogEntry(
                    service_id=target.id,
                    timestamp=event_time,
                    level=level,
                    message=template.format(service=target.name),
                    attributes=attrs,
                )
            )
            continue

        if entry in METRIC_RAMP_SPECS:
            metric_name, multiplier = METRIC_RAMP_SPECS[entry]
            baseline = next(b for b in BASELINE_METRICS[target.name] if b.name == metric_name)
            rows.extend(
                _ramp_metric_points(
                    target, metric_name, step_time, telemetry_end, baseline.mean * multiplier, rng
                )
            )
            error_burst_key = METRIC_RAMP_ERROR_BURST.get(entry)
            if error_burst_key is not None:
                log_window_start = max(step_time, telemetry_end - _ERROR_LOG_BURST_WINDOW)
                rows.extend(
                    _error_log_burst(
                        target,
                        log_window_start,
                        telemetry_end,
                        rng,
                        ERROR_CLUSTER_MESSAGES[error_burst_key],
                        ERROR_CLUSTER_ATTRIBUTES.get(error_burst_key),
                    )
                )
            continue

        if entry in ERROR_CLUSTER_ENTRIES:
            is_severe = entry in SEVERE_ERROR_CLUSTER_ENTRIES
            error_target = SEVERE_ERROR_RATE_TARGET if is_severe else NORMAL_ERROR_RATE_TARGET
            rows.extend(
                _ramp_metric_points(
                    target, "error_rate", step_time, telemetry_end, error_target, rng
                )
            )
            log_window_start = max(step_time, telemetry_end - _ERROR_LOG_BURST_WINDOW)
            rows.extend(
                _error_log_burst(
                    target,
                    log_window_start,
                    telemetry_end,
                    rng,
                    ERROR_CLUSTER_MESSAGES[entry],
                    ERROR_CLUSTER_ATTRIBUTES.get(entry),
                )
            )
            continue

        # Fallback for an entry that doesn't match any known table (e.g. a
        # future scenario's chain tag): never crash, just log it plainly.
        rows.append(
            LogEntry(
                service_id=target.id,
                timestamp=event_time,
                level=LogLevel.WARN,
                message=f"{target.name} {entry.replace('_', ' ')}",
                attributes={"causal_chain_entry": entry},
            )
        )

    return rows


def inject_failure(
    db: Session,
    scenario: FailureScenario,
    rng: Random,
    incident_start: datetime,
    *,
    pre_incident_window: timedelta | None = None,
    chain_stagger: timedelta | None = None,
    detection_lag: timedelta = DEFAULT_DETECTION_LAG,
    scenario_seed: int | None = None,
    scenario_instance_index: int | None = None,
) -> Incident:
    """Inject one instance of `scenario` ending at `incident_start` and
    return the created `Incident`.

    `rng` must be an explicit `random.Random(seed)` — never the global
    `random` module — and every random draw here happens in a fixed order
    (baseline over `sorted(CANONICAL_SERVICES)`, then `causal_chain` in its
    own list order) so the same `(scenario, seed, incident_start)` always
    reproduces byte-identical rows.

    `pre_incident_window`/`chain_stagger` default to `None`, which resolves
    to `SCENARIO_TIMING_OVERRIDES[scenario.failure_type]` if the scenario
    has one, else the module-level `DEFAULT_*` constants — this is what
    makes `bad_deployment` tight/back-to-back and `memory_leak` span hours,
    per each scenario's own documented timing intent, without every caller
    having to know and pass scenario-specific timing. Pass an explicit
    value to override that resolution entirely.

    `scenario_seed`/`scenario_instance_index` are provenance-only fields on
    the created `Incident` (nullable, default `None` for an ad-hoc
    injection) — passing them doesn't change what gets generated, they
    just let Phase 7's eval harness map an incident back to the exact
    `--count N --seed S` batch instance that produced it.

    Flushes (not commits) so the returned `Incident` has a real `.id`; the
    caller owns the transaction boundary.
    """
    overrides = SCENARIO_TIMING_OVERRIDES.get(scenario.failure_type, {})
    resolved_pre_incident_window = (
        pre_incident_window
        if pre_incident_window is not None
        else overrides.get("pre_incident_window", DEFAULT_PRE_INCIDENT_WINDOW)
    )
    resolved_chain_stagger = (
        chain_stagger
        if chain_stagger is not None
        else overrides.get("chain_stagger", DEFAULT_CHAIN_STAGGER)
    )

    services = get_or_create_canonical_services(db)

    window_start = incident_start - resolved_pre_incident_window
    for name in sorted(CANONICAL_SERVICES):
        generate_baseline_telemetry(db, services[name], window_start, incident_start, rng)

    chain_rows = _apply_causal_chain(
        scenario, services, rng, incident_start, resolved_chain_stagger, detection_lag
    )
    db.add_all(chain_rows)

    incident = Incident(
        service_id=services[scenario.affected_service].id,
        severity=Severity(scenario.severity),
        status=IncidentStatus.DETECTED,
        failure_type=scenario.failure_type,
        root_cause_category=scenario.root_cause_category,
        detected_at=incident_start,
        scenario_seed=scenario_seed,
        scenario_instance_index=scenario_instance_index,
    )
    db.add(incident)
    db.flush()

    return incident
