"""Phase 6's human-approval `interrupt()` gate + `POST /approve`/`/reject`.

Extends `tests/test_graph_response_planner_e2e.py`'s pattern (reusing its
fakes/fixture helpers) one step further: through the real
`backend.agents.human_approval_node` `interrupt()` and the
`backend.api.approvals` endpoints that resume/decide it.

Covers, against **real Postgres** (checkpointer + `AuditEvent` are real DB
operations, no `ChatAnthropic` mocking can substitute for that):

1. A HIGH_IMPACT plan genuinely halts the graph -- verified via the
   checkpointer's own `StateSnapshot` (`.next`/`.interrupts`), not just an
   `incident_status` field -- and `POST /approve` resumes it to the
   placeholder post-approval state, with the `AuditEvent` correctly
   `APPROVED`.
2. `POST /reject` never resumes the graph at all (the checkpoint stays
   parked exactly at `interrupt()`), sets `incident_status =
   manual_intervention_required`, and the `AuditEvent` is `REJECTED` with
   `executed_at` still null.
3. A duplicate `POST /approve` is a genuine no-op: no second `AuditEvent`
   row, no re-resume, the original approver is preserved.
4. `interrupt()`'s side-effect safety, demonstrated directly rather than
   asserted: resuming an already-resumed thread a second time (bypassing
   the API's own guard on purpose, to isolate what LangGraph itself
   guarantees) never recreates the `AuditEvent` row `response_planner_node`
   committed before the interrupt.
5. The idempotency guard's optimistic-concurrency mechanism
   (`AuditEvent.version_id`) under a **real** race: two independent DB
   sessions both read the same `PENDING_APPROVAL` row before either
   commits (simulating two concurrent `/approve` requests), and the loser
   gets `StaleDataError`, not a silent double-decision.

No test in this module makes a real Claude/Anthropic API call: `ChatAnthropic`
is monkeypatched via `tests.test_graph_response_planner_e2e`'s fakes.
"""

from __future__ import annotations

from datetime import UTC, datetime

import psycopg
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm.exc import StaleDataError

from backend.config import get_settings
from backend.db import SessionLocal
from backend.models import (
    AuditDecisionStatus,
    AuditEvent,
    Incident,
    IncidentStatus,
    RiskClassification,
)
from backend.rag.qdrant_client import get_qdrant_client
from backend.scripts.setup_checkpointer import to_psycopg_dsn
from tests.test_graph_response_planner_e2e import (
    ResponseAction,
    ResponsePlan,
    _inject_db_connection_exhaustion_incident,
    _patch_all_fakes,
)


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

_ROLLBACK_PLAN = ResponsePlan(
    actions=[
        ResponseAction(
            action_type="rollback_deployment",
            expected_benefit="removes the leaking deployed code path",
            confidence=0.85,
            llm_risk_assessment="moderate risk, briefly reverts a live deployment",
        )
    ]
)


async def _run_to_interrupt(monkeypatch, db):
    """Inject a db_connection_exhaustion incident and run the full graph
    (with all LLM nodes faked) up to the point it halts at
    `human_approval_node`'s `interrupt()`. Returns the incident."""
    from backend.graph import run_incident_graph

    incident, start, end = _inject_db_connection_exhaustion_incident(db)
    _patch_all_fakes(monkeypatch, incident.service.name, start, end, _ROLLBACK_PLAN)

    final_state = await run_incident_graph(db, incident, qdrant_client=get_qdrant_client())
    assert final_state.incident_status == IncidentStatus.AWAITING_APPROVAL
    return incident


# --- 1. Genuine halt + approve resumes ---------------------------------------


async def test_high_impact_plan_halts_graph_and_approve_resumes_it(monkeypatch):
    from backend.graph import get_incident_thread_state
    from backend.main import app

    client = TestClient(app)
    client.post("/api/simulation/reset")

    db = SessionLocal()
    try:
        incident = await _run_to_interrupt(monkeypatch, db)

        # Proof the graph genuinely halted: inspect the checkpointer's own
        # StateSnapshot, not IncidentState.incident_status (which
        # response_planner_node sets regardless of whether the graph
        # actually paused).
        snapshot = await get_incident_thread_state(db, incident, qdrant_client=get_qdrant_client())
        assert snapshot.next == ("human_approval",)
        assert len(snapshot.interrupts) == 1
        assert snapshot.interrupts[0].value["incident_id"] == incident.id

        fresh_db = SessionLocal()
        try:
            event = fresh_db.query(AuditEvent).filter(AuditEvent.incident_id == incident.id).one()
            assert event.decision_status is AuditDecisionStatus.PENDING_APPROVAL
            assert event.approver is None
        finally:
            fresh_db.close()

        response = client.post(
            f"/api/incidents/{incident.id}/approve", json={"approver": "oncall-jane"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["decision"] == "approved"
        assert body["resumed"] is True
        assert body["incident_status"] == IncidentStatus.EXECUTING.value
        assert body["approver"] == "oncall-jane"
        assert body["decided_at"] is not None
        assert body["audit_event_ids"]

        fresh_db2 = SessionLocal()
        try:
            event2 = fresh_db2.query(AuditEvent).filter(AuditEvent.incident_id == incident.id).one()
            assert event2.decision_status is AuditDecisionStatus.APPROVED
            assert event2.approver == "oncall-jane"
            assert event2.decided_at is not None
            assert event2.executed_at is None  # placeholder step never "executes" anything

            incident_row = fresh_db2.get(Incident, incident.id)
            assert incident_row.status == IncidentStatus.EXECUTING
        finally:
            fresh_db2.close()

        # The thread completed -- no pending interrupt left.
        snapshot2 = await get_incident_thread_state(db, incident, qdrant_client=get_qdrant_client())
        assert snapshot2.next == ()
        assert snapshot2.interrupts == ()
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")


# --- 2. Reject: manual intervention, never resumes ---------------------------


async def test_reject_sets_manual_intervention_and_never_resumes_graph(monkeypatch):
    from backend.graph import get_incident_thread_state
    from backend.main import app

    client = TestClient(app)
    client.post("/api/simulation/reset")

    db = SessionLocal()
    try:
        incident = await _run_to_interrupt(monkeypatch, db)

        response = client.post(
            f"/api/incidents/{incident.id}/reject", json={"approver": "oncall-bob"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["decision"] == "rejected"
        assert body["resumed"] is False
        assert body["incident_status"] == IncidentStatus.MANUAL_INTERVENTION_REQUIRED.value
        assert body["approver"] == "oncall-bob"
        assert body["decided_at"] is not None

        fresh_db = SessionLocal()
        try:
            event = fresh_db.query(AuditEvent).filter(AuditEvent.incident_id == incident.id).one()
            assert event.decision_status is AuditDecisionStatus.REJECTED
            assert event.approver == "oncall-bob"
            assert event.decided_at is not None
            assert event.executed_at is None

            incident_row = fresh_db.get(Incident, incident.id)
            assert incident_row.status == IncidentStatus.MANUAL_INTERVENTION_REQUIRED
        finally:
            fresh_db.close()

        # The graph thread was never touched -- still parked exactly where
        # it halted, proving reject didn't "resume toward execution" (it
        # didn't resume at all).
        snapshot = await get_incident_thread_state(db, incident, qdrant_client=get_qdrant_client())
        assert snapshot.next == ("human_approval",)
        assert len(snapshot.interrupts) == 1
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")


# --- 3. Duplicate /approve is idempotent -------------------------------------


async def test_duplicate_approve_is_idempotent(monkeypatch):
    from backend.main import app

    client = TestClient(app)
    client.post("/api/simulation/reset")

    db = SessionLocal()
    try:
        incident = await _run_to_interrupt(monkeypatch, db)

        first = client.post(f"/api/incidents/{incident.id}/approve", json={"approver": "first"})
        assert first.status_code == 200
        first_body = first.json()
        assert first_body["decision"] == "approved"
        assert first_body["resumed"] is True

        second = client.post(f"/api/incidents/{incident.id}/approve", json={"approver": "second"})
        assert second.status_code == 200
        second_body = second.json()
        assert second_body["decision"] == "already_decided"
        assert second_body["resumed"] is False
        # The original decision is preserved -- the second caller's
        # "second" approver never overwrites it.
        assert second_body["approver"] == "first"

        fresh_db = SessionLocal()
        try:
            events = (
                fresh_db.query(AuditEvent).filter(AuditEvent.incident_id == incident.id).all()
            )
            assert len(events) == 1  # no duplicate AuditEvent row
            assert events[0].approver == "first"
            assert events[0].decision_status is AuditDecisionStatus.APPROVED
        finally:
            fresh_db.close()
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")


# --- 3b. Liveness: a stuck resume (decision committed, resume never ran) ----
# is recovered by the next /approve call, not stranded forever ---------------


async def test_approve_recovers_a_stuck_resume(monkeypatch):
    """Simulates the exact failure window `_retry_stuck_resume` exists for:
    a previous /approve call durably committed AuditEvent -> APPROVED, then
    (crash / connection drop / any transient failure) never completed
    resume_incident_graph. The thread is left parked at interrupt() with an
    APPROVED row and no PENDING_APPROVAL rows -- exactly the state a naive
    "nothing pending -> already decided" check would treat as permanently
    done, without ever actually resuming it."""
    from backend.graph import get_incident_thread_state
    from backend.main import app

    client = TestClient(app)
    client.post("/api/simulation/reset")

    db = SessionLocal()
    try:
        incident = await _run_to_interrupt(monkeypatch, db)

        # Simulate the stuck state directly: commit the APPROVED decision
        # without ever calling resume_incident_graph (i.e. what happens if
        # the endpoint crashes between its own commit and the resume call).
        event = db.query(AuditEvent).filter(AuditEvent.incident_id == incident.id).one()
        event.decision_status = AuditDecisionStatus.APPROVED
        event.approver = "first-attempt"
        event.decided_at = datetime.now(UTC)
        db.commit()

        # Confirm the thread genuinely never resumed.
        stuck_snapshot = await get_incident_thread_state(
            db, incident, qdrant_client=get_qdrant_client()
        )
        assert stuck_snapshot.next == ("human_approval",)

        # A later /approve call (retry, or simply the client trying again
        # after a timeout) must detect and complete the stuck resume rather
        # than reporting "already_decided" forever.
        response = client.post(
            f"/api/incidents/{incident.id}/approve", json={"approver": "retry-caller"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["decision"] == "approved"
        assert body["resumed"] is True
        assert body["incident_status"] == IncidentStatus.EXECUTING.value
        # The original decision/approver is preserved -- the retry never
        # re-decides, it only completes the interrupted resume.
        assert body["approver"] == "first-attempt"

        resumed_snapshot = await get_incident_thread_state(
            db, incident, qdrant_client=get_qdrant_client()
        )
        assert resumed_snapshot.next == ()

        fresh_db = SessionLocal()
        try:
            events = fresh_db.query(AuditEvent).filter(AuditEvent.incident_id == incident.id).all()
            assert len(events) == 1  # never re-decided into a second row
            assert events[0].approver == "first-attempt"
            incident_row = fresh_db.get(Incident, incident.id)
            assert incident_row.status == IncidentStatus.EXECUTING
        finally:
            fresh_db.close()
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")


# --- 4. interrupt() side-effect safety: no duplicate AuditEvent on resume ----


async def test_resuming_an_already_resumed_thread_never_recreates_audit_rows(monkeypatch):
    """Bypasses backend.api.approvals's own guard on purpose -- calls
    `resume_incident_graph` directly, twice, to isolate what LangGraph's
    `interrupt()` mechanism itself guarantees (human_approval_node is only
    ever re-executed up to a completed interrupt() call; response_planner
    -- which already ran to completion and committed before the interrupt
    -- is never touched again), independent of the API-level idempotency
    guard tested above."""
    from backend.graph import resume_incident_graph
    from backend.main import app

    client = TestClient(app)
    client.post("/api/simulation/reset")

    db = SessionLocal()
    try:
        incident = await _run_to_interrupt(monkeypatch, db)

        qdrant_client = get_qdrant_client()
        resume_payload = {"decision": "approved", "approver": "x"}

        state_1 = await resume_incident_graph(
            db, incident, resume_payload, qdrant_client=qdrant_client
        )
        assert state_1.incident_status == IncidentStatus.EXECUTING

        state_2 = await resume_incident_graph(
            db, incident, resume_payload, qdrant_client=qdrant_client
        )
        # LangGraph itself no-ops a resume on a thread with nothing left to
        # resume -- same final state, not a second execution.
        assert state_2.incident_status == IncidentStatus.EXECUTING

        fresh_db = SessionLocal()
        try:
            events = (
                fresh_db.query(AuditEvent).filter(AuditEvent.incident_id == incident.id).all()
            )
            # Exactly the one row response_planner_node created BEFORE the
            # interrupt -- never duplicated by either resume call.
            assert len(events) == 1
            assert events[0].decision_status == AuditDecisionStatus.PENDING_APPROVAL
        finally:
            fresh_db.close()
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")


# --- 5. Real concurrent-style race on AuditEvent.version_id ------------------


def test_concurrent_approve_race_is_caught_by_version_id():
    """Two independent DB sessions both read the SAME PENDING_APPROVAL row
    before either commits -- simulating two `/approve` requests racing each
    other, exactly the scenario `backend.api.approvals`'s deliberately
    lock-free SELECT is designed to survive via
    `AuditEvent.version_id`/`version_id_col` (see that module's and
    `backend/models/audit.py`'s docstrings). Not run through the HTTP layer
    (two *sequential* TestClient calls can't reproduce a real race -- the
    second would just see the first's already-committed row) -- this drives
    the same DB-level mutation `_decide_pending_actions` performs, directly,
    with two sessions deliberately interleaved."""
    from fastapi.testclient import TestClient

    from backend.main import app

    client = TestClient(app)
    client.post("/api/simulation/reset")

    setup_db = SessionLocal()
    try:
        scenario_incident, _, _ = _inject_db_connection_exhaustion_incident(setup_db)
        setup_db.commit()  # must be committed for two other sessions to see it
        event = AuditEvent(
            incident_id=scenario_incident.id,
            action_type="rollback_deployment",
            risk_classification=RiskClassification.HIGH_IMPACT,
            decision_status=AuditDecisionStatus.PENDING_APPROVAL,
        )
        setup_db.add(event)
        setup_db.commit()
        event_id = event.id
    finally:
        setup_db.close()

    session_a = SessionLocal()
    session_b = SessionLocal()
    try:
        event_a = session_a.get(AuditEvent, event_id)
        event_b = session_b.get(AuditEvent, event_id)
        assert event_a.decision_status is AuditDecisionStatus.PENDING_APPROVAL
        assert event_b.decision_status is AuditDecisionStatus.PENDING_APPROVAL

        # First requester's decision lands first.
        event_a.decision_status = AuditDecisionStatus.APPROVED
        event_a.approver = "first-responder"
        event_a.decided_at = datetime.now(UTC)
        session_a.commit()

        # Second requester, racing from a now-stale in-memory copy, tries
        # to decide the same row differently.
        event_b.decision_status = AuditDecisionStatus.REJECTED
        event_b.approver = "second-responder"
        event_b.decided_at = datetime.now(UTC)
        with pytest.raises(StaleDataError):
            session_b.commit()
        session_b.rollback()
    finally:
        session_a.close()
        session_b.close()

    verify_db = SessionLocal()
    try:
        final_event = verify_db.get(AuditEvent, event_id)
        # The first decision won outright -- not double-decided, not
        # silently overwritten by the losing racer.
        assert final_event.decision_status is AuditDecisionStatus.APPROVED
        assert final_event.approver == "first-responder"
    finally:
        verify_db.close()
        client.post("/api/simulation/reset")
