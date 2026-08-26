"""Fast, free tests for `backend/api/incidents.py` that don't touch the LLM,
plus Phase 8's `GET /api/incidents` / `GET /{incident_id}` read endpoints.

`POST /api/incidents/{id}/investigate` calls a real Claude API when given a
valid incident -- that's covered separately (and skipped by default) in
`tests/test_investigator.py`. This module covers everything about the route
that's checkable without spending API credits: the 404 path, so a
regression here (wrong status code, broken import, route not actually
mounted) is caught by the default fast suite instead of only surfacing the
next time someone runs the billed live test.

The two Phase 8 read endpoints are covered here too:
- `GET /api/incidents`'s list/filter/pagination behavior needs no LLM at
  all -- it's a plain `Incident` table read.
- `GET /{incident_id}`'s "no checkpoint yet" path needs no LLM either (an
  injected-but-never-investigated incident).
- `GET /{incident_id}`'s "real investigation ran" and "reached response
  planning" paths reuse `tests/test_graph_end_to_end.py` and
  `tests/test_graph_response_planner_e2e.py`'s `ChatAnthropic`-faking
  fixtures (no real Claude/Anthropic API call), following
  `tests/test_human_approval.py`'s convention of importing those modules'
  fakes rather than redefining them.

Postgres-dependent throughout (every route here does real `Incident`/
`AuditEvent`/checkpointer reads) -- same skipif pattern as
`tests/test_simulation_api.py`.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from backend.config import get_settings
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

from fastapi.testclient import TestClient  # noqa: E402

from backend.db import SessionLocal  # noqa: E402
from backend.main import app  # noqa: E402
from backend.models import AuditDecisionStatus  # noqa: E402
from backend.rag.qdrant_client import get_qdrant_client  # noqa: E402
from backend.simulation.injector import inject_failure  # noqa: E402
from backend.simulation.scenario_schema import load_all_scenarios  # noqa: E402
from tests.test_graph_end_to_end import (  # noqa: E402
    _inject_cascading_incident,
)
from tests.test_graph_end_to_end import (  # noqa: E402
    _patch_all_fakes as _patch_all_fakes_e2e,
)
from tests.test_graph_response_planner_e2e import (  # noqa: E402
    ResponseAction,
    ResponsePlan,
    _inject_db_connection_exhaustion_incident,
)
from tests.test_graph_response_planner_e2e import (  # noqa: E402
    _patch_all_fakes as _patch_response_planner_fakes,
)
from tests.test_human_approval import _run_to_interrupt  # noqa: E402

client = TestClient(app)


def test_investigate_unknown_incident_returns_404():
    response = client.post("/api/incidents/999999999/investigate")

    assert response.status_code == 404
    assert "999999999" in response.json()["detail"]


# --- GET /api/incidents (list) -----------------------------------------------


def test_list_incidents_orders_most_recent_first_and_pagination_works():
    client.post("/api/simulation/reset")
    db = SessionLocal()
    try:
        scenario = load_all_scenarios()["db_connection_exhaustion"]
        base = datetime(2026, 7, 5, 12, 0, tzinfo=UTC)
        older = inject_failure(db, scenario, random.Random(1), base)
        newer = inject_failure(db, scenario, random.Random(2), base + timedelta(hours=1))
        db.commit()

        response = client.get("/api/incidents")
        assert response.status_code == 200
        body = response.json()
        ids = [row["id"] for row in body]
        assert ids.index(newer.id) < ids.index(older.id)
        for row in body:
            assert row["service_name"] == "checkout-service"

        # limit=1 returns only the most recent; offset=1 skips it and
        # returns the next one instead.
        limited = client.get("/api/incidents", params={"limit": 1}).json()
        assert len(limited) == 1
        assert limited[0]["id"] == newer.id

        offset_page = client.get("/api/incidents", params={"limit": 1, "offset": 1}).json()
        assert len(offset_page) == 1
        assert offset_page[0]["id"] == older.id
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")


def test_list_incidents_filters_by_status():
    client.post("/api/simulation/reset")
    db = SessionLocal()
    try:
        scenario = load_all_scenarios()["db_connection_exhaustion"]
        incident = inject_failure(
            db, scenario, random.Random(3), datetime(2026, 7, 5, 12, 0, tzinfo=UTC)
        )
        db.commit()

        matching = client.get("/api/incidents", params={"status": "detected"}).json()
        assert any(row["id"] == incident.id for row in matching)

        non_matching = client.get("/api/incidents", params={"status": "resolved"}).json()
        assert all(row["id"] != incident.id for row in non_matching)
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")


# --- GET /{incident_id} (detail) ---------------------------------------------


def test_get_incident_unknown_id_returns_404():
    response = client.get("/api/incidents/999999999")

    assert response.status_code == 404
    assert "999999999" in response.json()["detail"]


def test_get_incident_detail_with_no_checkpoint_has_null_investigation():
    """An incident that's been injected but never had `/investigate` or
    `/investigate/graph` called on it has no LangGraph checkpoint -- this
    is a normal state, not an error: `investigation` is null, `audit_events`
    is empty, and the incident's own fields still come through."""
    client.post("/api/simulation/reset")
    db = SessionLocal()
    try:
        scenario = load_all_scenarios()["db_connection_exhaustion"]
        incident = inject_failure(
            db, scenario, random.Random(4), datetime(2026, 7, 5, 12, 0, tzinfo=UTC)
        )
        db.commit()

        response = client.get(f"/api/incidents/{incident.id}")
        assert response.status_code == 200
        body = response.json()
        assert body["incident"]["id"] == incident.id
        assert body["incident"]["failure_type"] == "db_connection_exhaustion"
        assert body["incident"]["root_cause_category"] == "database_connection_pool"
        assert body["incident"]["service_name"] == "checkout-service"
        assert body["investigation"] is None
        assert body["audit_events"] == []
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")


async def test_get_incident_detail_after_investigation_run_includes_diagnosis(monkeypatch):
    """Runs the full graph (LLM layer faked, per `test_graph_end_to_end.py`'s
    convention) far enough to produce real evidence/hypotheses/root_cause,
    then confirms `GET /{incident_id}` surfaces them."""
    from backend.graph import run_incident_graph

    client.post("/api/simulation/reset")
    db = SessionLocal()
    try:
        incident, start, end = _inject_cascading_incident(db)
        _patch_all_fakes_e2e(monkeypatch, incident.service.name, start, end)

        await run_incident_graph(db, incident, qdrant_client=get_qdrant_client())

        response = client.get(f"/api/incidents/{incident.id}")
        assert response.status_code == 200
        body = response.json()
        investigation = body["investigation"]
        assert investigation is not None
        assert investigation["root_cause"] == "upstream_dependency_timeout"
        assert investigation["evidence"]
        assert investigation["hypotheses"]
        assert investigation["hypotheses"][0]["category"] == "upstream_dependency_timeout"
        assert investigation["alternative_hypotheses"]
        assert 0.0 <= investigation["diagnostic_confidence"] <= 1.0
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")


async def test_get_incident_detail_includes_audit_events_after_response_planning(monkeypatch):
    """Runs the full graph through Response Planner (LLM layer faked, per
    `test_graph_response_planner_e2e.py`'s convention) with a SAFE-only plan
    (auto-executed, no human approval needed) and confirms the resulting
    `AuditEvent` row is surfaced under `audit_events`."""
    from backend.graph import run_incident_graph

    client.post("/api/simulation/reset")
    db = SessionLocal()
    try:
        incident, start, end = _inject_db_connection_exhaustion_incident(db)
        plan = ResponsePlan(
            actions=[
                ResponseAction(
                    action_type="generate_incident_report",
                    expected_benefit="documents the diagnosis for the record",
                    confidence=0.7,
                    llm_risk_assessment="no risk, read-only",
                )
            ]
        )
        _patch_response_planner_fakes(monkeypatch, incident.service.name, start, end, plan)

        await run_incident_graph(db, incident, qdrant_client=get_qdrant_client())

        response = client.get(f"/api/incidents/{incident.id}")
        assert response.status_code == 200
        body = response.json()
        assert body["investigation"] is not None
        assert body["investigation"]["recommended_actions"]

        audit_events = body["audit_events"]
        assert len(audit_events) == 1
        assert audit_events[0]["action_type"] == "generate_incident_report"
        assert audit_events[0]["risk_classification"] == "safe"
        # Response Planner classifies it AUTO_EXECUTED; the real Action
        # Executor then runs it to completion (EXECUTED) -- same lifecycle
        # `test_graph_response_planner_e2e.py` verifies directly against the
        # DB row.
        assert audit_events[0]["decision_status"] == AuditDecisionStatus.EXECUTED.value
        assert audit_events[0]["recommended_at"]
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")


async def test_get_incident_detail_while_paused_at_human_approval(monkeypatch):
    """The state the approve/reject dashboard view actually depends on: a
    HIGH_IMPACT plan halted at `human_approval_node`'s `interrupt()`,
    genuinely paused (not just `Incident.status`-labeled that way -- see
    `InvestigationState`'s docstring on why the DB column and the graph's
    live phase can diverge). Reuses `test_human_approval.py`'s
    `_run_to_interrupt` fixture, the same real-halt helper that module's
    own approve/reject tests build on, rather than re-deriving this state
    here."""
    client.post("/api/simulation/reset")
    db = SessionLocal()
    try:
        incident = await _run_to_interrupt(monkeypatch, db)

        response = client.get(f"/api/incidents/{incident.id}")
        assert response.status_code == 200
        body = response.json()

        # `Incident.status` (the DB column) is still "detected" -- see the
        # caveat in `list_incidents`'s docstring -- but `investigation.
        # incident_status` (the graph's own live checkpoint) correctly
        # shows the real, paused phase.
        assert body["incident"]["status"] == "detected"
        assert body["investigation"] is not None
        assert body["investigation"]["incident_status"] == "awaiting_approval"
        assert body["investigation"]["recommended_actions"]
        assert body["investigation"]["approval_decision"] is None

        audit_events = body["audit_events"]
        assert len(audit_events) == 1
        assert audit_events[0]["risk_classification"] == "high_impact"
        assert audit_events[0]["decision_status"] == AuditDecisionStatus.PENDING_APPROVAL.value
        assert audit_events[0]["executed_at"] is None
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")
