"""Tests for Phase 1's failure injection engine.

Follows `tests/test_baseline.py`'s pattern: the whole module is skipped
when Postgres isn't reachable (`docker compose up -d postgres` to run
these for real). Tests never `db.commit()` their generated telemetry —
only `get_or_create_canonical_services` gets committed once (mirroring
`test_baseline.py`'s determinism test), so `Service.id` stays stable
across repeated runs within a test while everything else is discarded via
`db.rollback()`. This is BUILD_PLAN.md Phase 1's acceptance bar: "injected
failures produce queryable, timeline-coherent rows matching each
scenario's expected_evidence" and "the same seed reproduces a
byte-identical dataset."
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from sqlalchemy import select

from backend.config import get_settings
from backend.db import SessionLocal
from backend.models import (
    Deployment,
    IncidentStatus,
    LogEntry,
    LogLevel,
    MetricPoint,
    Service,
    Severity,
    TraceLite,
)
from backend.scripts.setup_checkpointer import to_psycopg_dsn
from backend.simulation.baseline import get_or_create_canonical_services
from backend.simulation.injector import DEFAULT_PRE_INCIDENT_WINDOW, inject_failure
from backend.simulation.scenario_schema import load_all_scenarios


def _postgres_reachable() -> bool:
    dsn = to_psycopg_dsn(get_settings().database_url)
    try:
        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Postgres not reachable at DATABASE_URL (start it with `docker compose up -d postgres`)",
)

ALL_FAILURE_TYPES = (
    "db_connection_exhaustion",
    "memory_leak",
    "bad_deployment",
    "dependency_failure",
    "slow_query",
    "cascading_payment_timeout",
)


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def scenarios():
    return load_all_scenarios()


def test_inject_failure_db_connection_exhaustion_shape(db, scenarios):
    """The 'simple' scenario: single-service, deployment-caused pool exhaustion."""
    scenario = scenarios["db_connection_exhaustion"]
    incident_start = datetime(2026, 1, 1, 15, 0, tzinfo=UTC)
    window_start = incident_start - DEFAULT_PRE_INCIDENT_WINDOW

    incident = inject_failure(db, scenario, random.Random(42), incident_start)

    # --- Incident ground truth ---
    assert incident.failure_type == "db_connection_exhaustion"
    assert incident.root_cause_category == "database_connection_pool"
    assert incident.severity == Severity.P1
    assert incident.status == IncidentStatus.DETECTED
    assert incident.detected_at == incident_start
    assert incident.scenario_seed is None
    assert incident.scenario_instance_index is None

    checkout = db.execute(select(Service).where(Service.name == "checkout-service")).scalar_one()
    assert incident.service_id == checkout.id

    # --- Deployment: the causal_chain's `checkout_deployment_v1.8.2` ---
    deployments = (
        db.execute(select(Deployment).where(Deployment.service_id == checkout.id)).scalars().all()
    )
    assert len(deployments) == 1
    assert deployments[0].version == "v1.8.2"
    assert window_start <= deployments[0].deployed_at < incident_start

    # --- db_connections_active ramps up toward incident_start, not flat ---
    points = (
        db.execute(
            select(MetricPoint)
            .where(
                MetricPoint.service_id == checkout.id,
                MetricPoint.metric_name == "db_connections_active",
            )
            .order_by(MetricPoint.timestamp)
        )
        .scalars()
        .all()
    )
    assert points
    early_values = [p.value for p in points if p.timestamp <= window_start + timedelta(minutes=10)]
    late_values = [p.value for p in points if p.timestamp >= incident_start - timedelta(minutes=3)]
    assert early_values, "expected baseline-window readings early in the window"
    assert late_values, "expected anomalous readings near incident_start"
    assert sum(early_values) / len(early_values) < 12.0  # near checkout's baseline mean of 8
    assert max(late_values) > 20.0  # well above baseline, evidence of pool growth

    # --- At least one ERROR-level log clustered near incident_start ---
    error_logs = (
        db.execute(
            select(LogEntry)
            .where(LogEntry.service_id == checkout.id, LogEntry.level == LogLevel.ERROR)
            .order_by(LogEntry.timestamp)
        )
        .scalars()
        .all()
    )
    assert error_logs
    assert all(log.timestamp <= incident_start for log in error_logs)
    assert any(log.timestamp >= incident_start - timedelta(minutes=5) for log in error_logs)


def test_inject_failure_cascading_payment_timeout_shape(db, scenarios):
    """The deliberately ambiguous, multi-service scenario."""
    scenario = scenarios["cascading_payment_timeout"]
    incident_start = datetime(2026, 2, 1, 10, 0, tzinfo=UTC)

    incident = inject_failure(db, scenario, random.Random(7), incident_start)

    assert incident.failure_type == "cascading_payment_timeout"
    assert incident.root_cause_category == "upstream_dependency_timeout"
    assert incident.severity == Severity.P1

    checkout = db.execute(select(Service).where(Service.name == "checkout-service")).scalar_one()
    payment = db.execute(select(Service).where(Service.name == "payment-service")).scalar_one()
    assert incident.service_id == checkout.id

    # --- checkout_retry_storm -> TraceLite spans checkout -> payment ---
    traces = (
        db.execute(
            select(TraceLite).where(
                TraceLite.service_id == checkout.id, TraceLite.downstream_service_id == payment.id
            )
        )
        .scalars()
        .all()
    )
    assert traces
    durations = sorted(t.duration_ms for t in traces)
    assert durations[-1] > durations[0] + 500  # duration climbs, not flat

    # --- payment_timeout (the quiet root cause) -> payment-service latency ramp ---
    payment_latency = (
        db.execute(
            select(MetricPoint)
            .where(
                MetricPoint.service_id == payment.id, MetricPoint.metric_name == "latency_p99_ms"
            )
            .order_by(MetricPoint.timestamp)
        )
        .scalars()
        .all()
    )
    assert payment_latency
    late_latency = [
        p.value for p in payment_latency if p.timestamp >= incident_start - timedelta(minutes=3)
    ]
    assert late_latency
    assert max(late_latency) > 400.0  # payment-service's baseline p99 is ~220ms

    # --- database_overload / checkout_failures -> ERROR logs on checkout-service ---
    checkout_errors = (
        db.execute(
            select(LogEntry).where(
                LogEntry.service_id == checkout.id, LogEntry.level == LogLevel.ERROR
            )
        )
        .scalars()
        .all()
    )
    assert checkout_errors
    assert any(log.timestamp >= incident_start - timedelta(minutes=5) for log in checkout_errors)

    # --- payment_timeout must ALSO produce the promised
    # payment_service_timeout_errors log evidence on payment-service, not
    # just a latency ramp (expected_evidence lists it explicitly as the
    # "quiet root-cause" signal only visible by querying payment's own
    # logs) ---
    payment_errors = (
        db.execute(
            select(LogEntry).where(
                LogEntry.service_id == payment.id, LogEntry.level == LogLevel.ERROR
            )
        )
        .scalars()
        .all()
    )
    assert payment_errors, "payment_timeout must produce ERROR-level logs on payment-service"
    assert any("timed out" in log.message for log in payment_errors)

    # No deployment in this scenario's causal_chain (external dependency fault).
    checkout_deployments = (
        db.execute(select(Deployment).where(Deployment.service_id == checkout.id)).scalars().all()
    )
    assert checkout_deployments == []


def test_inject_failure_is_deterministic_for_a_fixed_seed(db, scenarios):
    """Same seed -> byte-identical generated rows (excluding autoincrement
    ids/service_ids, which Postgres sequences don't roll back)."""
    get_or_create_canonical_services(db)
    db.commit()

    scenario = scenarios["db_connection_exhaustion"]
    incident_start = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)

    def _run(seed: int):
        incident = inject_failure(db, scenario, random.Random(seed), incident_start)
        db.flush()

        incident_tuple = (incident.failure_type, incident.severity, incident.detected_at)
        metrics = [
            (m.service_id, m.timestamp, m.metric_name, m.value)
            for m in db.execute(select(MetricPoint).order_by(MetricPoint.id)).scalars()
        ]
        logs = [
            (log_.service_id, log_.timestamp, log_.level, log_.message, log_.attributes)
            for log_ in db.execute(select(LogEntry).order_by(LogEntry.id)).scalars()
        ]
        deployments = [
            (d.service_id, d.version, d.deployed_at)
            for d in db.execute(select(Deployment).order_by(Deployment.id)).scalars()
        ]
        traces = [
            (t.service_id, t.timestamp, t.duration_ms, t.downstream_service_id)
            for t in db.execute(select(TraceLite).order_by(TraceLite.id)).scalars()
        ]

        db.rollback()  # discard this run's telemetry+incident; committed services survive
        return incident_tuple, metrics, logs, deployments, traces

    run_a = _run(4242)
    run_b = _run(4242)
    assert run_a == run_b
    assert run_a[1], "expected at least one metric point"
    assert run_a[2], "expected at least one log entry"

    run_c = _run(999)
    assert run_c[1] != run_a[1]


def test_dependency_failure_produces_dependency_span_evidence(db, scenarios):
    """Regression test: dependency_failure.yaml's expected_evidence lists
    `checkout_dependency_errors`, which used to be missing from its
    causal_chain entirely -- get_dependencies genuinely returned no spans,
    which is what led a real Phase 3 investigator run to misdiagnose this
    as `application_bug` instead of `upstream_dependency_failure`."""
    scenario = scenarios["dependency_failure"]
    assert "checkout_dependency_errors" in scenario.causal_chain

    incident_start = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    incident = inject_failure(db, scenario, random.Random(1), incident_start)

    checkout = db.execute(select(Service).where(Service.name == "checkout-service")).scalar_one()
    payment = db.execute(select(Service).where(Service.name == "payment-service")).scalar_one()
    assert incident.service_id == checkout.id

    traces = (
        db.execute(
            select(TraceLite).where(
                TraceLite.service_id == checkout.id, TraceLite.downstream_service_id == payment.id
            )
        )
        .scalars()
        .all()
    )
    assert traces, "get_dependencies must find checkout->payment spans for dependency_failure"

    dependency_errors = (
        db.execute(
            select(LogEntry).where(
                LogEntry.service_id == checkout.id, LogEntry.level == LogLevel.ERROR
            )
        )
        .scalars()
        .all()
    )
    assert any("downstream call" in log.message for log in dependency_errors)


def test_bad_deployment_causal_chain_is_tight(db, scenarios):
    """bad_deployment's own YAML comment: chain timestamps should be
    tight/back-to-back, not spread over minutes like the default stagger."""
    scenario = scenarios["bad_deployment"]
    incident_start = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)

    incident = inject_failure(db, scenario, random.Random(1), incident_start)

    payment = db.execute(select(Service).where(Service.name == "payment-service")).scalar_one()
    assert incident.service_id == payment.id

    deployment = db.execute(
        select(Deployment).where(Deployment.service_id == payment.id)
    ).scalar_one()
    # Default chain_stagger (2 min) over a 4-step chain would land the
    # deployment 8 minutes before incident_start; the override should keep
    # it well under a couple of minutes.
    assert incident_start - deployment.deployed_at < timedelta(minutes=2)


def test_memory_leak_causal_chain_spans_hours(db, scenarios):
    """memory_leak's own YAML comment: a slow, organic leak that
    accumulates over hours, not the default ~8-minute chain window."""
    scenario = scenarios["memory_leak"]
    incident_start = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    incident = inject_failure(db, scenario, random.Random(1), incident_start)

    inventory = db.execute(
        select(Service).where(Service.name == "inventory-service")
    ).scalar_one()
    assert incident.service_id == inventory.id

    memory_points = (
        db.execute(
            select(MetricPoint)
            .where(
                MetricPoint.service_id == inventory.id,
                MetricPoint.metric_name == "memory_usage_mb",
            )
            .order_by(MetricPoint.timestamp)
        )
        .scalars()
        .all()
    )
    assert memory_points
    earliest = memory_points[0].timestamp
    # Default chain_stagger (2 min) over a 4-step chain would start the
    # ramp ~8 minutes before incident_start; the override should span hours.
    assert incident_start - earliest > timedelta(hours=1)


@pytest.mark.parametrize("failure_type", ALL_FAILURE_TYPES)
def test_inject_failure_smoke_all_scenarios(db, scenarios, failure_type):
    """Every scenario file must inject without error and produce a real incident."""
    scenario = scenarios[failure_type]
    incident_start = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)

    incident = inject_failure(db, scenario, random.Random(1), incident_start)

    assert incident.id is not None
    assert incident.failure_type == failure_type
    assert incident.root_cause_category == scenario.root_cause_category
    assert incident.status == IncidentStatus.DETECTED
