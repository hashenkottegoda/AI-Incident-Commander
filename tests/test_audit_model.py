"""Round-trip tests for Phase 6's audit trail data model (`backend/models/audit.py`).

Follows `tests/test_models.py`'s pattern: skipped as a whole when Postgres
isn't reachable (`docker compose up -d postgres` to run these for real).
Each test creates its own rows and cleans up after itself.

This only exercises the data model itself (round-trip + CHECK constraints)
— the logic that actually populates these rows (Response Planner, Risk
Classifier, approval endpoints, Action Executor, Recovery Check) is later
Phase 6 work owned by langgraph-agent-engineer, not built here.
"""

from datetime import UTC, datetime

import psycopg
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, StatementError

from backend.config import get_settings
from backend.db import SessionLocal
from backend.models import (
    AuditDecisionStatus,
    AuditEvent,
    ExecutionOutcome,
    Incident,
    IncidentStatus,
    RiskClassification,
    Service,
    Severity,
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


@pytest.fixture
def incident(db):
    """One Service + Incident to hang audit rows off of, cleaned up after."""
    service = Service(name="audit-test-service", description="audit trail test fixture")
    db.add(service)
    db.flush()

    inc = Incident(
        service_id=service.id,
        severity=Severity.P1,
        status=IncidentStatus.DIAGNOSED,
        failure_type="db_connection_exhaustion",
        root_cause_category="database_connection_pool",
        detected_at=datetime.now(UTC),
    )
    db.add(inc)
    db.commit()
    db.refresh(inc)

    yield inc

    db.rollback()
    db.execute(text("DELETE FROM services WHERE id = :id"), {"id": service.id})
    db.commit()


def test_safe_action_auto_executed_round_trip(db, incident):
    """SAFE actions go straight to AUTO_EXECUTED, no human approver."""
    event = AuditEvent(
        incident_id=incident.id,
        action_type="gather_additional_diagnostics",
        risk_classification=RiskClassification.SAFE,
        decision_status=AuditDecisionStatus.AUTO_EXECUTED,
        approver=None,
        executed_at=datetime.now(UTC),
        execution_outcome=None,
        execution_detail=None,
    )
    db.add(event)
    db.commit()
    db.refresh(event)

    fetched = db.get(AuditEvent, event.id)
    assert fetched is not None
    assert fetched.incident_id == incident.id
    assert fetched.action_type == "gather_additional_diagnostics"
    assert fetched.risk_classification == RiskClassification.SAFE
    assert fetched.decision_status == AuditDecisionStatus.AUTO_EXECUTED
    assert fetched.approver is None
    assert fetched.recommended_at is not None
    assert fetched.decided_at is None
    assert fetched.executed_at is not None
    assert fetched.incident.id == incident.id

    # Reverse relationship resolves too.
    db.refresh(incident)
    assert event in incident.audit_events


def test_high_impact_action_pending_then_approved_then_executed(db, incident):
    """One row, updated in place through its lifecycle — the same shape the
    future approval endpoint / Action Executor will mutate, and what makes
    the executed_at-starts-NULL idempotency guard meaningful."""
    event = AuditEvent(
        incident_id=incident.id,
        action_type="rollback_deployment",
        risk_classification=RiskClassification.HIGH_IMPACT,
        decision_status=AuditDecisionStatus.PENDING_APPROVAL,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    event_id = event.id

    # --- pending -----------------------------------------------------
    fetched = db.get(AuditEvent, event_id)
    assert fetched.decision_status == AuditDecisionStatus.PENDING_APPROVAL
    assert fetched.risk_classification == RiskClassification.HIGH_IMPACT
    assert fetched.approver is None
    assert fetched.decided_at is None
    assert fetched.executed_at is None

    # --- approved (stubbed header-supplied approver identity) --------
    fetched.decision_status = AuditDecisionStatus.APPROVED
    fetched.approver = "oncall-engineer@example.com"
    fetched.decided_at = datetime.now(UTC)
    db.commit()

    reapproved = db.get(AuditEvent, event_id)
    assert reapproved.decision_status == AuditDecisionStatus.APPROVED
    assert reapproved.approver == "oncall-engineer@example.com"
    assert reapproved.decided_at is not None
    assert reapproved.executed_at is None, "approval alone must not set executed_at"

    # --- executed (Action Executor + Recovery Check both wrote here) -
    reapproved.decision_status = AuditDecisionStatus.EXECUTED
    reapproved.executed_at = datetime.now(UTC)
    reapproved.execution_outcome = ExecutionOutcome.RECOVERED
    reapproved.execution_detail = {
        "error_rate": "recovers_to_baseline",
        "db_connections": "recovers_to_baseline",
    }
    db.commit()

    executed = db.get(AuditEvent, event_id)
    assert executed.decision_status == AuditDecisionStatus.EXECUTED
    assert executed.executed_at is not None
    assert executed.execution_outcome == ExecutionOutcome.RECOVERED
    assert executed.execution_detail == {
        "error_rate": "recovers_to_baseline",
        "db_connections": "recovers_to_baseline",
    }
    # Idempotency hook: once set, a duplicate /approve would see
    # executed_at is-not-None and refuse to re-run the executor.
    assert executed.decided_at is not None


def test_rejected_high_impact_action_round_trip(db, incident):
    event = AuditEvent(
        incident_id=incident.id,
        action_type="restart_service",
        risk_classification=RiskClassification.HIGH_IMPACT,
        decision_status=AuditDecisionStatus.PENDING_APPROVAL,
    )
    db.add(event)
    db.commit()

    event.decision_status = AuditDecisionStatus.REJECTED
    event.approver = "oncall-engineer@example.com"
    event.decided_at = datetime.now(UTC)
    db.commit()

    fetched = db.get(AuditEvent, event.id)
    assert fetched.decision_status == AuditDecisionStatus.REJECTED
    assert fetched.executed_at is None, "rejection must never set executed_at"
    assert fetched.execution_outcome is None


def test_incident_cascade_deletes_audit_events(db, incident):
    event = AuditEvent(
        incident_id=incident.id,
        action_type="tag_incident",
        risk_classification=RiskClassification.SAFE,
        decision_status=AuditDecisionStatus.AUTO_EXECUTED,
        executed_at=datetime.now(UTC),
    )
    db.add(event)
    db.commit()
    event_id = event.id

    db.delete(incident)
    db.commit()

    assert db.get(AuditEvent, event_id) is None


def test_risk_classification_rejects_invalid_value_at_orm_level(db, incident):
    bad_event = AuditEvent(
        incident_id=incident.id,
        action_type="rollback_deployment",
        risk_classification="not_a_real_classification",
        decision_status=AuditDecisionStatus.PENDING_APPROVAL,
    )
    db.add(bad_event)
    with pytest.raises(StatementError, match="not among the defined enum values"):
        db.flush()
    db.rollback()


def test_decision_status_rejects_invalid_value_at_orm_level(db, incident):
    bad_event = AuditEvent(
        incident_id=incident.id,
        action_type="rollback_deployment",
        risk_classification=RiskClassification.HIGH_IMPACT,
        decision_status="not_a_real_status",
    )
    db.add(bad_event)
    with pytest.raises(StatementError, match="not among the defined enum values"):
        db.flush()
    db.rollback()


def test_execution_outcome_rejects_invalid_value_at_orm_level(db, incident):
    bad_event = AuditEvent(
        incident_id=incident.id,
        action_type="rollback_deployment",
        risk_classification=RiskClassification.HIGH_IMPACT,
        decision_status=AuditDecisionStatus.EXECUTED,
        execution_outcome="not_a_real_outcome",
    )
    db.add(bad_event)
    with pytest.raises(StatementError, match="not among the defined enum values"):
        db.flush()
    db.rollback()


def test_concurrent_update_raises_stale_data_error(db, incident):
    """version_id (SQLAlchemy's optimistic-concurrency guard) turns a lost
    concurrent update into a loud StaleDataError instead of a silent
    double-execution -- the race executed_at-IS-NULL alone can't catch.

    Simulates two concurrent duplicate `/approve` calls: session A loads
    the row first but commits second (the loser); session B loads second
    but commits first (the winner). Session A's stale write must fail
    loudly rather than silently re-executing an already-executed action.
    """
    from sqlalchemy.orm.exc import StaleDataError

    event = AuditEvent(
        incident_id=incident.id,
        action_type="rollback_deployment",
        risk_classification=RiskClassification.HIGH_IMPACT,
        decision_status=AuditDecisionStatus.APPROVED,
    )
    db.add(event)
    db.commit()
    event_id = event.id

    session_a = SessionLocal()
    stale_copy = session_a.get(AuditEvent, event_id)

    session_b = SessionLocal()
    try:
        winner = session_b.get(AuditEvent, event_id)
        winner.decision_status = AuditDecisionStatus.EXECUTED
        winner.executed_at = datetime.now(UTC)
        session_b.commit()
    finally:
        session_b.close()

    try:
        stale_copy.decision_status = AuditDecisionStatus.EXECUTED
        stale_copy.executed_at = datetime.now(UTC)
        with pytest.raises(StaleDataError):
            session_a.commit()
    finally:
        session_a.rollback()
        session_a.close()


def test_risk_classification_rejects_invalid_value_at_db_level(db, incident):
    """Bypass the ORM (raw SQL) to prove the CHECK constraint is real, not
    just an application-layer validation a direct DB write could skip.

    Invalid value kept within risk_classification's VARCHAR(20) bound so
    the failure exercises the CHECK constraint, not a separate
    length-truncation error.
    """
    try:
        with pytest.raises(IntegrityError, match="audit_risk_classification"):
            db.execute(
                text(
                    "INSERT INTO audit_events "
                    "(incident_id, action_type, risk_classification, decision_status) "
                    "VALUES (:incident_id, 'rollback_deployment', 'not_real', "
                    "'pending_approval')"
                ),
                {"incident_id": incident.id},
            )
            db.commit()
    finally:
        db.rollback()


def test_decision_status_rejects_invalid_value_at_db_level(db, incident):
    try:
        with pytest.raises(IntegrityError, match="audit_decision_status"):
            db.execute(
                text(
                    "INSERT INTO audit_events "
                    "(incident_id, action_type, risk_classification, decision_status) "
                    "VALUES (:incident_id, 'rollback_deployment', 'high_impact', "
                    "'not_a_real_status')"
                ),
                {"incident_id": incident.id},
            )
            db.commit()
    finally:
        db.rollback()


def test_execution_outcome_rejects_invalid_value_at_db_level(db, incident):
    try:
        with pytest.raises(IntegrityError, match="audit_execution_outcome"):
            db.execute(
                text(
                    "INSERT INTO audit_events "
                    "(incident_id, action_type, risk_classification, decision_status, "
                    "execution_outcome) "
                    "VALUES (:incident_id, 'rollback_deployment', 'high_impact', 'executed', "
                    "'not_a_real_outcome')"
                ),
                {"incident_id": incident.id},
            )
            db.commit()
    finally:
        db.rollback()
