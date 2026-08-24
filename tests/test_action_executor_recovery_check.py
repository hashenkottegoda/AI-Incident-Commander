"""Phase 6's final sub-step: the Action Executor + Recovery Check.

Builds directly toward BUILD_PLAN.md Phase 6's verify criteria (c), (d),
and a metric-level re-verification of (e):

(c) an approved `rollback_deployment` on `db_connection_exhaustion` (its
    real `correct_remediation`, confirmed against
    `failure_scenarios/db_connection_exhaustion.yaml` rather than assumed)
    drives the incident all the way to `resolved`, with real post-action
    `MetricPoint` rows near baseline.
(d) an approved `scale_service` on `cascading_payment_timeout` (one of its
    real `ineffective_remediations`, likewise confirmed against the YAML)
    does NOT resolve the incident -- Recovery Check correctly detects
    still-degraded telemetry and routes back to a fresh Investigation pass
    (proven by actually driving the graph through a real second
    Investigation/RAG/Root-Cause/Response-Planner round trip, not just
    asserting a routing predicate in isolation).
(e) a duplicate `POST /approve` on an already-**executed** thread (not just
    already-*decided*) is idempotent at the telemetry level too: no second
    `MetricPoint` write, matching `tests/test_human_approval.py`'s
    AuditEvent-level idempotency coverage.

Also covers the bounded-loop-exhaustion branch
(`recovery_check_node` -> `MANUAL_INTERVENTION_REQUIRED`) directly against
the real node functions and real Postgres, without needing three full
approve round trips through mocked LLMs.

This entire sub-step makes ZERO Claude/Anthropic API calls: the Action
Executor and Recovery Check are pure deterministic simulation + metric
comparison (BUILD_PLAN.md: "Because remediation_effects is ground truth,
both the recovery decision and the remediation-eval metrics are
deterministic"). The (c)/(d) tests below still route through the full
graph (Triage/Investigation/RAG/Root Cause/Response Planner all run), so
those two reuse `tests/test_graph_response_planner_e2e.py`'s
`ChatAnthropic` fakes -- via `tests/test_human_approval.py`'s own
`_run_to_interrupt` helper for (c) -- rather than inventing a new mocking
pattern, per this sub-step's own instructions.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from fastapi.testclient import TestClient

from backend.agents.action_executor_node import make_action_executor_node
from backend.agents.recovery_check_node import make_recovery_check_node
from backend.agents.routing import MAX_REINVESTIGATION_LOOPS, route_after_recovery_check
from backend.agents.state import IncidentState
from backend.config import get_settings
from backend.db import SessionLocal
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
from backend.rag.qdrant_client import get_qdrant_client
from backend.scripts.setup_checkpointer import to_psycopg_dsn
from backend.simulation.injector import inject_failure
from backend.simulation.scenario_schema import load_all_scenarios
from tests.test_graph_response_planner_e2e import ResponseAction, ResponsePlan, _patch_all_fakes
from tests.test_human_approval import _ROLLBACK_PLAN, _run_to_interrupt


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

_SCALE_SERVICE_PLAN = ResponsePlan(
    actions=[
        ResponseAction(
            action_type="scale_service",
            expected_benefit="add capacity to absorb the DB connection pressure",
            confidence=0.5,
            llm_risk_assessment="low risk, adds replicas",
        )
    ]
)


def _inject_cascading_payment_timeout_incident(db):
    scenario = load_all_scenarios()["cascading_payment_timeout"]
    incident_start = datetime(2026, 7, 5, 12, 0, tzinfo=UTC)
    incident = inject_failure(db, scenario, random.Random(99), incident_start)
    window_start = (incident_start - timedelta(hours=1)).isoformat()
    window_end = (incident_start + timedelta(minutes=5)).isoformat()
    return incident, window_start, window_end


# --- (c) Correct remediation resolves end-to-end ----------------------------


async def test_approved_rollback_on_db_connection_exhaustion_resolves_end_to_end(monkeypatch):
    from backend.graph import get_incident_thread_state
    from backend.main import app

    scenario = load_all_scenarios()["db_connection_exhaustion"]
    assert scenario.remediation_effects.correct_remediation == "rollback_deployment"
    assert _ROLLBACK_PLAN.actions[0].action_type == "rollback_deployment"

    client = TestClient(app)
    client.post("/api/simulation/reset")

    db = SessionLocal()
    try:
        incident = await _run_to_interrupt(monkeypatch, db)

        response = client.post(
            f"/api/incidents/{incident.id}/approve", json={"approver": "oncall-recovery"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["decision"] == "approved"
        assert body["incident_status"] == IncidentStatus.RESOLVED.value

        fresh_db = SessionLocal()
        try:
            incident_row = fresh_db.get(Incident, incident.id)
            assert incident_row.status == IncidentStatus.RESOLVED

            event = (
                fresh_db.query(AuditEvent).filter(AuditEvent.incident_id == incident.id).one()
            )
            assert event.action_type == "rollback_deployment"
            assert event.decision_status is AuditDecisionStatus.EXECUTED
            assert event.execution_outcome is ExecutionOutcome.RECOVERED
            assert event.executed_at is not None
            assert event.execution_detail is not None
            assert event.execution_detail["matched_correct_remediation"] is True
            assert "written_metrics" in event.execution_detail

            # Real post-action MetricPoint rows, near this service's real
            # healthy baseline (checkout-service: error_rate mean=0.005,
            # db_connections_active mean=8.0 -- see
            # backend.simulation.baseline.BASELINE_METRICS), not the
            # injected anomaly levels (error_rate ramped toward 0.40,
            # db_connections_active ramped toward ~44).
            checkout = fresh_db.query(Service).filter(Service.name == "checkout-service").one()
            post_points = (
                fresh_db.query(MetricPoint)
                .filter(
                    MetricPoint.service_id == checkout.id,
                    MetricPoint.timestamp >= event.executed_at,
                    MetricPoint.metric_name.in_(["error_rate", "db_connections_active"]),
                )
                .all()
            )
            assert len(post_points) == 10  # 5 points x 2 metrics
            error_rate_values = [p.value for p in post_points if p.metric_name == "error_rate"]
            db_conn_values = [
                p.value for p in post_points if p.metric_name == "db_connections_active"
            ]
            assert error_rate_values and all(v < 0.05 for v in error_rate_values)
            assert db_conn_values and all(v < 20.0 for v in db_conn_values)
        finally:
            fresh_db.close()

        # The thread genuinely completed -- no pending interrupt left.
        snapshot = await get_incident_thread_state(db, incident, qdrant_client=get_qdrant_client())
        assert snapshot.next == ()
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")


# --- (d) Ineffective remediation routes back to investigating ---------------


async def test_ineffective_remediation_on_cascading_routes_back_to_investigating(monkeypatch):
    from backend.graph import get_incident_thread_state, run_incident_graph
    from backend.main import app

    scenario = load_all_scenarios()["cascading_payment_timeout"]
    assert "scale_service" in scenario.remediation_effects.ineffective_remediations

    client = TestClient(app)
    client.post("/api/simulation/reset")

    db = SessionLocal()
    try:
        incident, start, end = _inject_cascading_payment_timeout_incident(db)
        _patch_all_fakes(monkeypatch, incident.service.name, start, end, _SCALE_SERVICE_PLAN)

        pre_state = await run_incident_graph(db, incident, qdrant_client=get_qdrant_client())
        assert pre_state.incident_status == IncidentStatus.AWAITING_APPROVAL
        assert pre_state.investigation_iterations == 1

        response = client.post(
            f"/api/incidents/{incident.id}/approve", json={"approver": "oncall-loop"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["decision"] == "approved"
        assert body["resumed"] is True
        # The remediation was ineffective -- must NOT resolve. Recovery
        # Check routed back to a fresh Investigation pass (budget wasn't
        # exhausted yet), which -- driven by the same deterministic fakes
        # -- ran a full second Investigation/RAG/Root-Cause/Response-Planner
        # pass and paused at a SECOND interrupt with a fresh (still
        # ineffective) recommendation, rather than falsely reporting
        # resolved.
        assert body["incident_status"] == IncidentStatus.AWAITING_APPROVAL.value

        fresh_db = SessionLocal()
        try:
            events = (
                fresh_db.query(AuditEvent)
                .filter(AuditEvent.incident_id == incident.id)
                .order_by(AuditEvent.id)
                .all()
            )
            assert len(events) == 2  # first (executed) + second (freshly pending)

            first_event, second_event = events
            assert first_event.action_type == "scale_service"
            assert first_event.decision_status is AuditDecisionStatus.EXECUTED
            assert first_event.execution_outcome is ExecutionOutcome.STILL_DEGRADED
            assert first_event.executed_at is not None
            assert first_event.execution_detail["matched_correct_remediation"] is False
            assert first_event.execution_detail["matched_known_ineffective_remediation"] is True

            assert second_event.action_type == "scale_service"
            assert second_event.decision_status is AuditDecisionStatus.PENDING_APPROVAL
            assert second_event.executed_at is None

            incident_row = fresh_db.get(Incident, incident.id)
            assert incident_row.status == IncidentStatus.AWAITING_APPROVAL

            # Real, still-anomalous post-action telemetry (checkout-service
            # db_connections_active baseline mean=8.0; the injected ramp
            # target for this scenario is ~44) -- not near baseline.
            checkout = fresh_db.query(Service).filter(Service.name == "checkout-service").one()
            post_points = (
                fresh_db.query(MetricPoint)
                .filter(
                    MetricPoint.service_id == checkout.id,
                    MetricPoint.timestamp >= first_event.executed_at,
                    MetricPoint.metric_name == "db_connections_active",
                )
                .all()
            )
            assert post_points
            assert all(p.value > 20.0 for p in post_points)
        finally:
            fresh_db.close()

        # The graph genuinely looped back through a fresh Investigation
        # pass (not a no-op): investigation_iterations incremented, and the
        # thread is parked at a second, distinct interrupt.
        snapshot = await get_incident_thread_state(db, incident, qdrant_client=get_qdrant_client())
        assert snapshot.next == ("human_approval",)
        assert len(snapshot.interrupts) == 1
        assert snapshot.values["investigation_iterations"] == 2
        assert snapshot.values["recovery_result"]["outcome"] == "still_degraded"

        # route_after_recovery_check's own decision, confirmed directly
        # against the exact incident_status recovery_check_node produced
        # for this still-degraded, budget-not-exhausted case.
        investigating_state = IncidentState(
            incident_id=incident.id, incident_status=IncidentStatus.INVESTIGATING
        )
        assert route_after_recovery_check(investigating_state) == "investigation"
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")


# --- Bounded loop exhaustion -> MANUAL_INTERVENTION_REQUIRED ----------------


def test_recovery_check_routes_to_manual_intervention_when_budget_exhausted():
    """Direct node-level test (no LLM mocking, no interrupt/approve round
    trip needed -- this sub-step is pure deterministic simulation) for
    BUILD_PLAN.md's "back to INVESTIGATION (bounded; exhausted ->
    MANUAL_INTERVENTION_REQUIRED)" branch, reusing the SAME
    investigation_iterations bound the root-cause reinvestigation loop
    already uses (backend.agents.routing.MAX_REINVESTIGATION_LOOPS), not a
    second parallel mechanism -- see recovery_check_node's docstring."""
    from backend.main import app

    client = TestClient(app)
    client.post("/api/simulation/reset")

    db = SessionLocal()
    try:
        incident, _start, _end = _inject_cascading_payment_timeout_incident(db)

        event = AuditEvent(
            incident_id=incident.id,
            action_type="scale_service",  # a real ineffective_remediations entry
            risk_classification=RiskClassification.HIGH_IMPACT,
            decision_status=AuditDecisionStatus.APPROVED,
        )
        db.add(event)
        db.commit()

        state = IncidentState(
            incident_id=incident.id,
            incident_status=IncidentStatus.AWAITING_APPROVAL,
            investigation_iterations=MAX_REINVESTIGATION_LOOPS + 1,
        )

        executor = make_action_executor_node(db)
        execution_update = executor(state)
        assert execution_update["incident_status"] == IncidentStatus.VERIFYING
        state_after_execution = state.model_copy(update=execution_update)

        recovery = make_recovery_check_node(db)
        recovery_update = recovery(state_after_execution)

        assert recovery_update["incident_status"] == IncidentStatus.MANUAL_INTERVENTION_REQUIRED
        assert recovery_update["recovery_result"]["outcome"] == "still_degraded"

        final_state = state_after_execution.model_copy(update=recovery_update)
        assert route_after_recovery_check(final_state) == "end"

        fresh_db = SessionLocal()
        try:
            incident_row = fresh_db.get(Incident, incident.id)
            assert incident_row.status == IncidentStatus.MANUAL_INTERVENTION_REQUIRED
            refreshed_event = fresh_db.get(AuditEvent, event.id)
            assert refreshed_event.decision_status is AuditDecisionStatus.EXECUTED
            assert refreshed_event.execution_outcome is ExecutionOutcome.STILL_DEGRADED
        finally:
            fresh_db.close()
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")


# --- (e), metric-level: duplicate approve after real execution --------------


async def test_duplicate_approve_after_resolution_never_rewrites_telemetry(monkeypatch):
    """Extends `tests/test_human_approval.py::test_duplicate_approve_is_idempotent`'s
    AuditEvent-level idempotency check with the metric-level guarantee: a
    duplicate `POST /approve` on an already-**executed** thread (not merely
    already-*decided*) must not write a second round of `MetricPoint`
    rows -- i.e. must not re-run the remediation."""
    from backend.main import app

    client = TestClient(app)
    client.post("/api/simulation/reset")

    db = SessionLocal()
    try:
        incident = await _run_to_interrupt(monkeypatch, db)

        first = client.post(
            f"/api/incidents/{incident.id}/approve", json={"approver": "first-caller"}
        )
        assert first.status_code == 200
        assert first.json()["incident_status"] == IncidentStatus.RESOLVED.value

        fresh_db = SessionLocal()
        try:
            checkout = fresh_db.query(Service).filter(Service.name == "checkout-service").one()
            metric_count_after_first = (
                fresh_db.query(MetricPoint).filter(MetricPoint.service_id == checkout.id).count()
            )
        finally:
            fresh_db.close()

        second = client.post(
            f"/api/incidents/{incident.id}/approve", json={"approver": "second-caller"}
        )
        assert second.status_code == 200
        second_body = second.json()
        assert second_body["decision"] == "already_decided"
        assert second_body["resumed"] is False
        assert second_body["incident_status"] == IncidentStatus.RESOLVED.value
        assert second_body["approver"] == "first-caller"  # original decision preserved

        fresh_db2 = SessionLocal()
        try:
            events = (
                fresh_db2.query(AuditEvent).filter(AuditEvent.incident_id == incident.id).all()
            )
            assert len(events) == 1  # never re-decided/re-executed into a second row
            assert events[0].decision_status is AuditDecisionStatus.EXECUTED
            assert events[0].approver == "first-caller"

            metric_count_after_second = (
                fresh_db2.query(MetricPoint)
                .filter(MetricPoint.service_id == checkout.id)
                .count()
            )
            assert metric_count_after_second == metric_count_after_first

            incident_row = fresh_db2.get(Incident, incident.id)
            assert incident_row.status == IncidentStatus.RESOLVED
        finally:
            fresh_db2.close()
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")
