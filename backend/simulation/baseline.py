"""Synthetic baseline ("healthy") telemetry generator for the 3 canonical services.

This is Phase 1's foundation layer for the simulator: before the failure
injection engine (the *next* build step) can perturb a metric series into
an anomaly, there has to be a normal, "nothing is wrong" series to perturb
from. Two things live here:

- `get_or_create_canonical_services`: idempotently seeds the 3
  `CANONICAL_SERVICES` rows (`backend.simulation.scenario_schema`).
- `generate_baseline_telemetry`: given a service, a time window, and an
  explicit `random.Random(seed)`, produces "healthy" `MetricPoint` and
  `LogEntry` rows for that window and writes them to Postgres.

Determinism is the load-bearing property here (BUILD_PLAN.md Phase 1: "the
same seed reproduces a byte-identical dataset"). The global `random` module
is never touched — every random draw goes through the caller-supplied
`random.Random` instance, in a fixed call order (timestamp-major, then a
fixed per-service metric order), so the same seed always produces the same
sequence of values regardless of dict/set iteration or any other ambient
state.

The metric set below (`error_rate`, `latency_p99_ms`, `db_connections_active`,
`memory_usage_mb`, `gc_pause_ms`) was chosen by grepping
`failure_scenarios/*.yaml`'s `expected_evidence`/`causal_chain` tags for
metric-shaped evidence: `db_connections_high`/`connection_pool_exhausted`
-> `db_connections_active`; `memory_usage_climbing` -> `memory_usage_mb`;
`gc_pause_increase` -> `gc_pause_ms`; `*_latency_high`/`inventory_query_latency_high`
-> `latency_p99_ms`; `error_rate_spike`/`http_500_spike`/`*_failures`/`*_request_failures`
-> `error_rate`. All 5 metrics are generated for all 3 services (not just
the service each scenario happens to affect) so the injection engine has a
same-shaped "unaffected service" baseline to contrast against, and so a
scenario's own metric perturbation always has real, non-flat baseline
values underneath it rather than a placeholder.

Deliberately simple: this is a portfolio/demo simulator, not a
production-grade load generator. Per-metric jitter is a single
`random.gauss(mean, stddev)` draw, clamped to a sane floor — no fancy
distributions, no autocorrelation, no diurnal patterns.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models import LogEntry, LogLevel, MetricPoint, Service
from backend.simulation.scenario_schema import CANONICAL_SERVICES

# --- Service seeding ---------------------------------------------------

SERVICE_DESCRIPTIONS: dict[str, str] = {
    "checkout-service": "Cart, order placement, and checkout flow.",
    "payment-service": "Payment authorization and charge processing.",
    "inventory-service": "Stock lookups and inventory reservation.",
}


def get_or_create_canonical_services(db: Session) -> dict[str, Service]:
    """Idempotently ensure the 3 `CANONICAL_SERVICES` rows exist.

    Safe to call repeatedly (e.g. once per `POST /api/simulation/reset`,
    the next build step): a second call finds all 3 already present and
    inserts nothing, so it never raises the `services.name` unique
    constraint and never creates duplicates. Flushes (not commits) so the
    returned `Service` rows have real `.id`s the caller can use
    immediately; the caller owns the transaction boundary.
    """
    existing = {
        service.name: service
        for service in db.execute(
            select(Service).where(Service.name.in_(CANONICAL_SERVICES))
        ).scalars()
    }
    for name in sorted(CANONICAL_SERVICES):
        if name not in existing:
            service = Service(name=name, description=SERVICE_DESCRIPTIONS.get(name))
            db.add(service)
            existing[name] = service
    db.flush()
    return existing


# --- Baseline metric config ---------------------------------------------


@dataclass(frozen=True)
class MetricBaseline:
    """One metric's "healthy" distribution: `random.gauss(mean, stddev)`,
    floored at `minimum` so jitter never produces a nonsensical negative
    connection count / latency / error rate."""

    name: str
    mean: float
    stddev: float
    minimum: float = 0.0


DEFAULT_METRIC_INTERVAL_SECONDS: int = 60
DEFAULT_AVG_LOG_INTERVAL_SECONDS: float = 45.0

# Per-service baseline constants. Values are illustrative-realistic, not
# measured from anything real (this is a synthetic simulator) — chosen so
# each service's "normal" is a plausible, slightly different operating
# point (e.g. inventory-service runs a heavier DB/memory footprint than
# payment-service, which is mostly I/O-bound against an external gateway).
BASELINE_METRICS: dict[str, tuple[MetricBaseline, ...]] = {
    "checkout-service": (
        MetricBaseline("error_rate", mean=0.005, stddev=0.0015),
        MetricBaseline("latency_p99_ms", mean=180.0, stddev=15.0),
        MetricBaseline("db_connections_active", mean=8.0, stddev=1.5),
        MetricBaseline("memory_usage_mb", mean=512.0, stddev=20.0),
        MetricBaseline("gc_pause_ms", mean=12.0, stddev=3.0),
    ),
    "payment-service": (
        MetricBaseline("error_rate", mean=0.003, stddev=0.001),
        MetricBaseline("latency_p99_ms", mean=220.0, stddev=20.0),
        MetricBaseline("db_connections_active", mean=6.0, stddev=1.2),
        MetricBaseline("memory_usage_mb", mean=420.0, stddev=18.0),
        MetricBaseline("gc_pause_ms", mean=10.0, stddev=2.5),
    ),
    "inventory-service": (
        MetricBaseline("error_rate", mean=0.004, stddev=0.0012),
        MetricBaseline("latency_p99_ms", mean=140.0, stddev=12.0),
        MetricBaseline("db_connections_active", mean=10.0, stddev=1.8),
        MetricBaseline("memory_usage_mb", mean=600.0, stddev=25.0),
        MetricBaseline("gc_pause_ms", mean=15.0, stddev=3.5),
    ),
}

# --- Baseline log config -------------------------------------------------
# Healthy-system routine logging: "{service}" is filled in with the
# service name. No error-level content here by design — manufacturing
# error logs in the *baseline* is the failure injection engine's job, not
# this one's.
INFO_LOG_TEMPLATES: tuple[str, ...] = (
    "{service} processed request successfully",
    "{service} health check passed",
    "{service} completed scheduled background job",
    "{service} cache refreshed",
    "{service} handled request within SLA",
)
WARN_LOG_TEMPLATES: tuple[str, ...] = (
    "{service} observed elevated response time, still within tolerance",
    "{service} retried a transient network blip successfully",
    "{service} connection pool utilization slightly above average",
)
WARN_LOG_PROBABILITY: float = 0.12


@dataclass
class BaselineTelemetry:
    """The rows one `generate_baseline_telemetry` call produced."""

    metrics: list[MetricPoint]
    logs: list[LogEntry]


def _clamp(value: float, minimum: float) -> float:
    return value if value >= minimum else minimum


def _metric_points_for_service(
    service: Service,
    start: datetime,
    end: datetime,
    rng: random.Random,
    interval_seconds: int,
) -> list[MetricPoint]:
    """Pure (no DB I/O) builder: one draw per `(timestamp, metric)` pair,
    timestamp-major so the rng call order — and therefore the output for a
    given seed — never depends on dict/set iteration order."""
    baselines = BASELINE_METRICS[service.name]
    step = timedelta(seconds=interval_seconds)
    points: list[MetricPoint] = []
    cursor = start
    while cursor <= end:
        for baseline in baselines:
            value = _clamp(rng.gauss(baseline.mean, baseline.stddev), baseline.minimum)
            points.append(
                MetricPoint(
                    service_id=service.id,
                    timestamp=cursor,
                    metric_name=baseline.name,
                    value=value,
                )
            )
        cursor += step
    return points


def _log_entries_for_service(
    service: Service,
    start: datetime,
    end: datetime,
    rng: random.Random,
    avg_interval_seconds: float,
) -> list[LogEntry]:
    """Pure (no DB I/O) builder for sporadic, mostly-info baseline logs.

    Gaps between log lines are `uniform(0, 2 * avg_interval_seconds)`
    (mean == `avg_interval_seconds`), which reads as irregular routine
    activity rather than a suspiciously regular heartbeat.
    """
    logs: list[LogEntry] = []
    cursor = start + timedelta(seconds=rng.uniform(0, avg_interval_seconds * 2))
    while cursor <= end:
        is_warn = rng.random() < WARN_LOG_PROBABILITY
        level = LogLevel.WARN if is_warn else LogLevel.INFO
        template = rng.choice(WARN_LOG_TEMPLATES if is_warn else INFO_LOG_TEMPLATES)
        logs.append(
            LogEntry(
                service_id=service.id,
                timestamp=cursor,
                level=level,
                message=template.format(service=service.name),
                attributes=None,
            )
        )
        cursor += timedelta(seconds=rng.uniform(0, avg_interval_seconds * 2))
    return logs


def generate_baseline_telemetry(
    db: Session,
    service: Service,
    start: datetime,
    end: datetime,
    rng: random.Random,
    *,
    metric_interval_seconds: int = DEFAULT_METRIC_INTERVAL_SECONDS,
    avg_log_interval_seconds: float = DEFAULT_AVG_LOG_INTERVAL_SECONDS,
) -> BaselineTelemetry:
    """Generate "healthy" telemetry for `service` over `[start, end]` and
    write it to Postgres via the given session.

    `rng` must be an explicit `random.Random(seed)` instance — never the
    global `random` module — so the same seed reproduces a byte-identical
    set of metric values and log messages (BUILD_PLAN.md Phase 1). `service`
    must already have a real `.id` (i.e. already flushed/committed, e.g.
    via `get_or_create_canonical_services`).

    Flushes (not commits) so newly created rows get real ids immediately;
    the caller owns the transaction boundary.
    """
    if service.name not in BASELINE_METRICS:
        raise ValueError(
            f"no baseline metric config for service {service.name!r}; "
            f"expected one of {sorted(BASELINE_METRICS)}"
        )
    if start > end:
        raise ValueError(f"start ({start!r}) must be <= end ({end!r})")

    metrics = _metric_points_for_service(service, start, end, rng, metric_interval_seconds)
    logs = _log_entries_for_service(service, start, end, rng, avg_log_interval_seconds)

    db.add_all(metrics)
    db.add_all(logs)
    db.flush()

    return BaselineTelemetry(metrics=metrics, logs=logs)
