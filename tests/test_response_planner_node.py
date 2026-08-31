"""Isolated (mocked-LLM) tests for Phase 6's Response Planner + inline Risk
Classifier node (`backend/agents/response_planner_node.py`).

No test in this module makes a real OpenRouter API call --
`response_planner_node.ChatOpenRouter` is monkeypatched with a small fake
returning a canned `ResponsePlan`, following
`tests/test_graph_end_to_end.py`'s `_FakeStructuredLLM` convention.

Postgres-dependent (the node writes real `AuditEvent` rows) -- skipped
cleanly without it, same convention as the rest of this suite.
"""

from __future__ import annotations

from datetime import UTC, datetime

import psycopg
import pytest
from sqlalchemy import text

from backend.agents.response_schemas import ResponseAction, ResponsePlan
from backend.agents.state import IncidentState
from backend.config import get_settings
from backend.db import SessionLocal
from backend.models import (
    AuditDecisionStatus,
    AuditEvent,
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


# --- Fakes -------------------------------------------------------------------


class _FakeStructuredLLM:
    def __init__(self, result):
        self._result = result

    def invoke(self, messages):  # noqa: ARG002
        return self._result

    def with_retry(self, **kwargs):  # noqa: ARG002
        return self


def _make_fake_chat_openrouter(plan: ResponsePlan):
    class _FakeResponsePlannerChatOpenRouter:
        def __init__(self, *args, **kwargs):
            pass

        def with_structured_output(self, schema):  # noqa: ARG002
            return _FakeStructuredLLM(plan)

    return _FakeResponsePlannerChatOpenRouter


# --- Fixtures ------------------------------------------------------------


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
    service = Service(name="response-planner-test-service", description="test fixture")
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


def _diagnosed_state(incident: Incident) -> IncidentState:
    return IncidentState(
        incident_id=incident.id,
        incident_status=IncidentStatus.DIAGNOSED,
        severity=Severity.P1,
        affected_services=["response-planner-test-service"],
        root_cause="database_connection_pool",
        diagnostic_confidence=0.85,
    )


# --- Tests -----------------------------------------------------------------


def test_safe_only_plan_creates_auto_executed_audit_event_and_sets_executing(
    monkeypatch, db, incident
):
    import backend.agents.response_planner_node as node_module

    plan = ResponsePlan(
        actions=[
            ResponseAction(
                action_type="gather_additional_diagnostics",
                expected_benefit="collects more data before recommending a fix",
                confidence=0.6,
                llm_risk_assessment="no risk, read-only",
            )
        ]
    )
    monkeypatch.setattr(node_module, "ChatOpenRouter", _make_fake_chat_openrouter(plan))

    node = node_module.make_response_planner_node(db)
    result = node(_diagnosed_state(incident))

    assert result["incident_status"] == IncidentStatus.EXECUTING
    assert len(result["recommended_actions"]) == 1
    ref = result["recommended_actions"][0]
    assert ref["action_type"] == "gather_additional_diagnostics"
    assert ref["risk_classification"] == RiskClassification.SAFE.value
    assert ref["decision_status"] == AuditDecisionStatus.AUTO_EXECUTED.value

    event = db.get(AuditEvent, ref["audit_event_id"])
    assert event is not None
    assert event.incident_id == incident.id
    assert event.action_type == "gather_additional_diagnostics"
    assert event.risk_classification is RiskClassification.SAFE
    assert event.decision_status is AuditDecisionStatus.AUTO_EXECUTED
    assert event.approver is None
    assert event.executed_at is None, (
        "AUTO_EXECUTED at this stage means queued, not yet actually executed -- "
        "the Action Executor doesn't exist yet"
    )


def test_high_impact_action_creates_pending_approval_audit_event_and_sets_awaiting_approval(
    monkeypatch, db, incident
):
    import backend.agents.response_planner_node as node_module

    plan = ResponsePlan(
        actions=[
            ResponseAction(
                action_type="rollback_deployment",
                expected_benefit="removes the leaking deployed code path",
                confidence=0.8,
                llm_risk_assessment="moderate risk, briefly reverts a live deployment",
            )
        ]
    )
    monkeypatch.setattr(node_module, "ChatOpenRouter", _make_fake_chat_openrouter(plan))

    node = node_module.make_response_planner_node(db)
    result = node(_diagnosed_state(incident))

    assert result["incident_status"] == IncidentStatus.AWAITING_APPROVAL
    ref = result["recommended_actions"][0]
    assert ref["risk_classification"] == RiskClassification.HIGH_IMPACT.value
    assert ref["decision_status"] == AuditDecisionStatus.PENDING_APPROVAL.value

    event = db.get(AuditEvent, ref["audit_event_id"])
    assert event.risk_classification is RiskClassification.HIGH_IMPACT
    assert event.decision_status is AuditDecisionStatus.PENDING_APPROVAL
    assert event.approver is None
    assert event.decided_at is None
    assert event.executed_at is None


def test_mixed_plan_with_any_high_impact_action_routes_to_awaiting_approval(
    monkeypatch, db, incident
):
    """One SAFE action + one HIGH_IMPACT action -> the incident-level
    routing decision must be AWAITING_APPROVAL (any HIGH_IMPACT action
    present gates the whole plan), even though a row for the SAFE action
    is still independently AUTO_EXECUTED."""
    import backend.agents.response_planner_node as node_module

    plan = ResponsePlan(
        actions=[
            ResponseAction(
                action_type="tag_incident",
                expected_benefit="labels the incident for tracking",
                confidence=0.9,
                llm_risk_assessment="no risk",
            ),
            ResponseAction(
                action_type="restart_service",
                expected_benefit="clears the exhausted connection pool",
                confidence=0.4,
                llm_risk_assessment="low-moderate risk, brief downtime",
            ),
        ]
    )
    monkeypatch.setattr(node_module, "ChatOpenRouter", _make_fake_chat_openrouter(plan))

    node = node_module.make_response_planner_node(db)
    result = node(_diagnosed_state(incident))

    assert result["incident_status"] == IncidentStatus.AWAITING_APPROVAL
    assert len(result["recommended_actions"]) == 2
    by_type = {ref["action_type"]: ref for ref in result["recommended_actions"]}
    assert by_type["tag_incident"]["decision_status"] == AuditDecisionStatus.AUTO_EXECUTED.value
    assert (
        by_type["restart_service"]["decision_status"] == AuditDecisionStatus.PENDING_APPROVAL.value
    )


def test_unrecognized_action_type_still_creates_high_impact_audit_row(monkeypatch, db, incident):
    """An LLM-invented action name outside the known vocabulary must still
    be safely gated -- classify_risk's fail-safe default applies even when
    the Response Planner didn't follow instructions."""
    import backend.agents.response_planner_node as node_module

    plan = ResponsePlan(
        actions=[
            ResponseAction(
                action_type="do_something_creative",
                expected_benefit="unclear",
                confidence=0.3,
                llm_risk_assessment="the model claims this is safe",
            )
        ]
    )
    monkeypatch.setattr(node_module, "ChatOpenRouter", _make_fake_chat_openrouter(plan))

    node = node_module.make_response_planner_node(db)
    result = node(_diagnosed_state(incident))

    assert result["incident_status"] == IncidentStatus.AWAITING_APPROVAL
    ref = result["recommended_actions"][0]
    assert ref["risk_classification"] == RiskClassification.HIGH_IMPACT.value
    assert ref["decision_status"] == AuditDecisionStatus.PENDING_APPROVAL.value
