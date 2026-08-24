"""Tests for Phase 1's baseline telemetry generator.

Follows `tests/test_db.py`/`tests/test_models.py`'s pattern: the whole
module is skipped when Postgres isn't reachable (`docker compose up -d
postgres` to run these for real). Each test creates its own rows and
cleans up after itself so the module can run repeatedly against a live
dev database without accumulating state.
"""

import random
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from sqlalchemy import select

from backend.config import get_settings
from backend.db import SessionLocal
from backend.models import LogEntry, LogLevel, MetricPoint, Service
from backend.scripts.setup_checkpointer import to_psycopg_dsn
from backend.simulation.baseline import (
    BASELINE_METRICS,
    generate_baseline_telemetry,
    get_or_create_canonical_services,
)
from backend.simulation.scenario_schema import CANONICAL_SERVICES


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


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def test_get_or_create_canonical_services_is_idempotent(db):
    """Calling twice must still yield exactly the 3 canonical services,
    with no duplicate-name IntegrityError on the second call."""
    first = get_or_create_canonical_services(db)
    db.commit()

    assert set(first) == CANONICAL_SERVICES
    first_ids = {name: service.id for name, service in first.items()}

    # Second call should find all 3 already present and insert nothing.
    second = get_or_create_canonical_services(db)
    db.commit()

    assert set(second) == CANONICAL_SERVICES
    assert {name: service.id for name, service in second.items()} == first_ids

    rows = db.execute(
        select(Service).where(Service.name.in_(CANONICAL_SERVICES))
    ).scalars().all()
    assert len(rows) == 3
    assert {row.name for row in rows} == CANONICAL_SERVICES


def test_generate_baseline_telemetry_shape(db):
    """One window's worth of generated telemetry has the right services,
    right time range, sane values, and zero ERROR-level logs."""
    services = get_or_create_canonical_services(db)
    db.commit()
    checkout = services["checkout-service"]

    start = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    end = start + timedelta(minutes=10)
    rng = random.Random(1234)

    try:
        result = generate_baseline_telemetry(db, checkout, start, end, rng)
        db.commit()

        expected_metric_names = {b.name for b in BASELINE_METRICS["checkout-service"]}
        expected_points_per_metric = 11  # every 60s from :00 to :10 inclusive
        assert len(result.metrics) == expected_points_per_metric * len(expected_metric_names)

        for point in result.metrics:
            assert point.service_id == checkout.id
            assert start <= point.timestamp <= end
            assert point.metric_name in expected_metric_names
            assert point.value >= 0.0

        # error_rate should be a small fraction, not a spike.
        error_rate_values = [p.value for p in result.metrics if p.metric_name == "error_rate"]
        assert error_rate_values
        assert all(0.0 <= v < 0.05 for v in error_rate_values)

        # Logs: within window, on the right service, no ERROR level.
        assert result.logs
        for log in result.logs:
            assert log.service_id == checkout.id
            assert start <= log.timestamp <= end
            assert log.level in (LogLevel.INFO, LogLevel.WARN)
            assert log.level != LogLevel.ERROR

        # Round-trip via a fresh query too.
        fetched_metrics = db.execute(
            select(MetricPoint).where(MetricPoint.service_id == checkout.id)
        ).scalars().all()
        assert len(fetched_metrics) == len(result.metrics)
    finally:
        db.query(MetricPoint).filter(MetricPoint.service_id == checkout.id).delete()
        db.query(LogEntry).filter(LogEntry.service_id == checkout.id).delete()
        db.commit()


def test_generate_baseline_telemetry_rejects_unknown_service_name(db):
    service = Service(name="not-a-canonical-service")
    db.add(service)
    db.flush()

    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=5)
    with pytest.raises(ValueError, match="no baseline metric config"):
        generate_baseline_telemetry(db, service, start, end, random.Random(1))

    db.rollback()


def test_generate_baseline_telemetry_rejects_start_after_end(db):
    services = get_or_create_canonical_services(db)
    db.commit()
    checkout = services["checkout-service"]

    start = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    end = start - timedelta(minutes=1)
    with pytest.raises(ValueError, match=r"must be <= end"):
        generate_baseline_telemetry(db, checkout, start, end, random.Random(1))

    db.rollback()


def test_generate_baseline_telemetry_is_deterministic_for_a_fixed_seed(db):
    """Same seed -> byte-identical metric values and log content.

    BUILD_PLAN.md Phase 1: 'the same seed reproduces a byte-identical
    dataset' — this is the property the eval harness depends on for a fair
    A/B/C/D comparison. Verified here by comparing full
    (timestamp, metric_name, value) / (timestamp, level, message) tuples
    across two independent runs seeded identically, not just spot-checking
    a few values.
    """
    services = get_or_create_canonical_services(db)
    db.commit()
    payment = services["payment-service"]

    start = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
    end = start + timedelta(minutes=15)

    try:
        run_a = generate_baseline_telemetry(db, payment, start, end, random.Random(777))
        db.flush()
        metrics_a = [(m.timestamp, m.metric_name, m.value) for m in run_a.metrics]
        logs_a = [(log_.timestamp, log_.level, log_.message) for log_ in run_a.logs]
        db.rollback()  # discard run A's rows before generating run B

        run_b = generate_baseline_telemetry(db, payment, start, end, random.Random(777))
        db.flush()
        metrics_b = [(m.timestamp, m.metric_name, m.value) for m in run_b.metrics]
        logs_b = [(log_.timestamp, log_.level, log_.message) for log_ in run_b.logs]

        assert metrics_a == metrics_b
        assert logs_a == logs_b
        assert len(metrics_a) > 0
        assert len(logs_a) > 0

        # A different seed must (overwhelmingly likely) produce different values.
        db.rollback()
        run_c = generate_baseline_telemetry(db, payment, start, end, random.Random(999))
        metrics_c = [(m.timestamp, m.metric_name, m.value) for m in run_c.metrics]
        assert metrics_c != metrics_a
    finally:
        db.rollback()
