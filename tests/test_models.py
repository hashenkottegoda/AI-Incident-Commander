"""Round-trip tests for Phase 1's core data model.

Follows `tests/test_db.py`'s pattern: skipped as a whole when Postgres
isn't reachable (`docker compose up -d postgres` to run these for real).
Each test creates its own rows and cleans up after itself so the module
can run repeatedly against a live dev database without accumulating state.
"""

from datetime import UTC, datetime

import psycopg
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, StatementError

from backend.config import get_settings
from backend.db import SessionLocal
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
from backend.scripts.setup_checkpointer import to_psycopg_dsn


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


def test_service_deployment_log_metric_trace_incident_round_trip(db):
    """Create one of each Phase 1 model and confirm the FK graph resolves."""
    now = datetime.now(UTC)

    checkout = Service(name="checkout-service-test", description="cart + order flow")
    payment = Service(name="payment-service-test", description="payment processing")
    db.add_all([checkout, payment])
    db.flush()

    deployment = Deployment(service_id=checkout.id, version="v1.8.2", deployed_at=now)
    log = LogEntry(
        service_id=checkout.id,
        timestamp=now,
        level=LogLevel.ERROR,
        message="connection pool exhausted",
        attributes={"pool_size": 20, "active": 20},
    )
    metric = MetricPoint(
        service_id=checkout.id,
        timestamp=now,
        metric_name="db_connections_active",
        value=20.0,
    )
    trace = TraceLite(
        service_id=checkout.id,
        timestamp=now,
        span_name="checkout.charge_card",
        duration_ms=1250.5,
        downstream_service_id=payment.id,
    )
    incident = Incident(
        service_id=checkout.id,
        severity=Severity.P1,
        status=IncidentStatus.DETECTED,
        failure_type="db_connection_exhaustion",
        root_cause_category="database_connection_pool",
        detected_at=now,
        scenario_seed=42,
        scenario_instance_index=0,
    )
    db.add_all([deployment, log, metric, trace, incident])
    db.commit()

    for obj in (deployment, log, metric, trace, incident):
        db.refresh(obj)

    # Round-trip via fresh queries (not just the in-session objects).
    fetched_incident = db.get(Incident, incident.id)
    assert fetched_incident is not None
    assert fetched_incident.service.name == "checkout-service-test"
    assert fetched_incident.severity == Severity.P1
    assert fetched_incident.status == IncidentStatus.DETECTED
    assert fetched_incident.failure_type == "db_connection_exhaustion"
    assert fetched_incident.root_cause_category == "database_connection_pool"
    assert fetched_incident.scenario_seed == 42
    assert fetched_incident.scenario_instance_index == 0

    fetched_deployment = db.get(Deployment, deployment.id)
    assert fetched_deployment.service.id == checkout.id
    assert fetched_deployment.version == "v1.8.2"

    fetched_log = db.get(LogEntry, log.id)
    assert fetched_log.level == LogLevel.ERROR
    assert fetched_log.attributes == {"pool_size": 20, "active": 20}
    assert fetched_log.service.id == checkout.id

    fetched_metric = db.get(MetricPoint, metric.id)
    assert fetched_metric.metric_name == "db_connections_active"
    assert fetched_metric.value == 20.0

    fetched_trace = db.get(TraceLite, trace.id)
    assert fetched_trace.service.id == checkout.id
    assert fetched_trace.downstream_service.id == payment.id
    assert fetched_trace.downstream_service.name == "payment-service-test"

    # Reverse relationship resolves too.
    assert deployment in checkout.deployments
    assert incident in checkout.incidents

    # Cleanup: deleting the services cascades everything created above.
    db.delete(checkout)
    db.delete(payment)
    db.commit()

    assert db.get(Service, checkout.id) is None
    assert db.get(Incident, incident.id) is None
    assert db.get(Deployment, deployment.id) is None
    assert db.get(LogEntry, log.id) is None
    assert db.get(MetricPoint, metric.id) is None
    assert db.get(TraceLite, trace.id) is None


def test_service_name_unique_constraint(db):
    db.add(Service(name="dup-service-test"))
    db.commit()

    try:
        db.add(Service(name="dup-service-test"))
        with pytest.raises(IntegrityError, match="ix_services_name"):
            db.commit()
    finally:
        db.rollback()
        db.execute(text("DELETE FROM services WHERE name = 'dup-service-test'"))
        db.commit()


def test_trace_downstream_service_set_null_on_delete(db):
    """Deleting only the *downstream* service must SET NULL, not cascade.

    The span row is owned by `service_id`; `downstream_service_id` is a
    secondary, nullable pointer to a different service. Deleting that other
    service should leave the trace row intact with the pointer cleared —
    a plain CASCADE here would silently delete unrelated, still-valid spans.
    """
    owner = Service(name="trace-owner-service-test")
    downstream = Service(name="trace-downstream-service-test")
    db.add_all([owner, downstream])
    db.flush()

    trace = TraceLite(
        service_id=owner.id,
        timestamp=datetime.now(UTC),
        span_name="checkout.call_payment",
        duration_ms=42.0,
        downstream_service_id=downstream.id,
    )
    db.add(trace)
    db.commit()
    trace_id = trace.id
    owner_id = owner.id

    db.delete(downstream)
    db.commit()

    fetched = db.get(TraceLite, trace_id)
    assert fetched is not None, "trace row must survive the downstream service's deletion"
    assert fetched.downstream_service_id is None
    assert fetched.service_id == owner_id

    db.delete(owner)
    db.commit()
    assert db.get(TraceLite, trace_id) is None


def test_incident_status_rejects_invalid_value_at_orm_level(db):
    """SQLAlchemy's Enum type validates on flush, before ever reaching the DB.

    Assigning the invalid value doesn't raise by itself (the type's
    validation runs in the bind processor at statement-execution time, not
    at attribute-set time) — the invalid value is only caught once the
    session tries to flush the INSERT.
    """
    service = Service(name="enum-orm-test-service")
    db.add(service)
    db.flush()

    bad_incident = Incident(
        service_id=service.id,
        severity=Severity.P1,
        status="not_a_real_status",
        failure_type="db_connection_exhaustion",
        root_cause_category="database_connection_pool",
        detected_at=datetime.now(UTC),
    )
    db.add(bad_incident)
    with pytest.raises(StatementError, match="not among the defined enum values"):
        db.flush()

    db.rollback()


def test_incident_status_rejects_invalid_value_at_db_level(db):
    """Bypass the ORM (raw SQL) to prove the CHECK constraint is real, not
    just an application-layer validation that a direct DB write could skip."""
    service = Service(name="enum-db-test-service")
    db.add(service)
    db.commit()
    service_id = service.id

    try:
        with pytest.raises(IntegrityError, match="incident_status"):
            db.execute(
                text(
                    "INSERT INTO incidents "
                    "(service_id, severity, status, failure_type, "
                    "root_cause_category, detected_at) "
                    "VALUES (:service_id, 'P1', 'not_a_real_status', "
                    "'db_connection_exhaustion', 'database_connection_pool', now())"
                ),
                {"service_id": service_id},
            )
            db.commit()
    finally:
        db.rollback()
        db.execute(text("DELETE FROM services WHERE id = :id"), {"id": service_id})
        db.commit()
