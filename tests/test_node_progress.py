"""Round-trip test for Phase 8's live-trace write path
(`backend.graph._with_progress`, `backend/models/node_progress.py`).

BUILD_PLAN.md Phase 8's "live investigation trace" spec, verbatim:
*"Live-trace transport: persist each graph node's progress to Postgres as
it runs and have the dashboard poll that progress log (simplest, MVP)."*
This module only proves the WRITE side works end-to-end through a real
graph run -- no read API/dashboard exists yet (later, separate steps).

Follows `tests/test_graph_response_planner_e2e.py`'s pattern almost
exactly (same `db_connection_exhaustion` scenario, same single-decisive-
pass fakes so the re-investigation loop never triggers and node order is
deterministic) -- reused here rather than reinvented so this test's only
genuinely new assertions are about `NodeProgressEvent` rows, not about
control flow already covered elsewhere.

No test in this module makes a real OpenRouter API call: `ChatOpenRouter`
is monkeypatched in each LLM-calling node module, matching every other
graph-level test in this suite.

Postgres-dependent (incident injection + real `NodeProgressEvent` writes)
-- skipped cleanly without it, same convention as the rest of this suite.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from backend.agents.response_schemas import ResponseAction, ResponsePlan
from backend.agents.schemas import DiagnosisResult, Hypothesis
from backend.config import get_settings
from backend.db import SessionLocal
from backend.models import AuditDecisionStatus, AuditEvent, IncidentStatus, NodeProgressEvent
from backend.rag.qdrant_client import get_qdrant_client
from backend.scripts.setup_checkpointer import to_psycopg_dsn
from backend.simulation.injector import inject_failure
from backend.simulation.scenario_schema import load_all_scenarios
from tests.test_human_approval import _run_to_interrupt


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


# --- Fakes (same shape as tests/test_graph_response_planner_e2e.py) --------


class _FakeStructuredLLM:
    def __init__(self, result):
        self._result = result

    def invoke(self, messages):  # noqa: ARG002
        return self._result

    def with_retry(self, **kwargs):  # noqa: ARG002
        return self


class _FakeTriageChatOpenRouter:
    def __init__(self, *args, **kwargs):
        pass

    def with_structured_output(self, schema):  # noqa: ARG002
        from backend.agents.triage_node import TriageResult

        return _FakeStructuredLLM(TriageResult(affected_services=["checkout-service"]))


def _make_fake_investigation_chat_openrouter(service: str, start: str, end: str):
    """Single decisive pass: covers both required evidence tools up front,
    then stops -- never triggers the re-investigation loop, so this test's
    node-order assertion is deterministic."""

    class _FakeInvestigationLLM:
        def __init__(self):
            self._turn = 0

        def invoke(self, messages):  # noqa: ARG002
            self._turn += 1
            if self._turn > 1:
                return AIMessage(content="investigation pass complete, no further tool calls")
            tool_calls = [
                {
                    "name": "get_deployments",
                    "args": {"service": service, "start": start, "end": end},
                    "id": "call_deploy_1",
                },
                {
                    "name": "get_dependencies",
                    "args": {"service": service, "start": start, "end": end},
                    "id": "call_deps_1",
                },
            ]
            return AIMessage(content="", tool_calls=tool_calls)

        def with_retry(self, **kwargs):  # noqa: ARG002
            return self

    class _FakeInvestigationChatOpenRouter:
        def __init__(self, *args, **kwargs):
            pass

        def bind_tools(self, tools):  # noqa: ARG002
            return _FakeInvestigationLLM()

    return _FakeInvestigationChatOpenRouter


def _make_fake_root_cause_chat_openrouter():
    class _FakeRootCauseChatOpenRouter:
        def __init__(self, *args, **kwargs):
            pass

        def with_structured_output(self, schema):  # noqa: ARG002
            result = DiagnosisResult(
                root_cause_category="database_connection_pool",
                hypotheses=[
                    Hypothesis(
                        category="database_connection_pool",
                        rationale="deployment leaked db connections until the pool was exhausted",
                        confidence=0.9,
                    )
                ],
                alternative_hypotheses=[
                    Hypothesis(
                        category="upstream_dependency_timeout",
                        rationale="ruled out -- no downstream dependency involvement found",
                        confidence=0.1,
                    )
                ],
                diagnostic_confidence=0.9,
                evidence=[],
            )
            return _FakeStructuredLLM(result)

    return _FakeRootCauseChatOpenRouter


def _make_fake_response_planner_chat_openrouter(plan: ResponsePlan):
    class _FakeResponsePlannerChatOpenRouter:
        def __init__(self, *args, **kwargs):
            pass

        def with_structured_output(self, schema):  # noqa: ARG002
            return _FakeStructuredLLM(plan)

    return _FakeResponsePlannerChatOpenRouter


def _patch_all_fakes(monkeypatch, service, start, end, response_plan: ResponsePlan):
    import backend.agents.investigation_node as investigation_module
    import backend.agents.response_planner_node as response_planner_module
    import backend.agents.root_cause_node as rca_module
    import backend.agents.triage_node as triage_module

    monkeypatch.setattr(triage_module, "ChatOpenRouter", _FakeTriageChatOpenRouter)
    monkeypatch.setattr(
        investigation_module,
        "ChatOpenRouter",
        _make_fake_investigation_chat_openrouter(service, start, end),
    )
    monkeypatch.setattr(rca_module, "ChatOpenRouter", _make_fake_root_cause_chat_openrouter())
    monkeypatch.setattr(
        response_planner_module,
        "ChatOpenRouter",
        _make_fake_response_planner_chat_openrouter(response_plan),
    )


def _inject_db_connection_exhaustion_incident(db):
    scenario = load_all_scenarios()["db_connection_exhaustion"]
    incident_start = datetime(2026, 7, 5, 12, 0, tzinfo=UTC)
    incident = inject_failure(db, scenario, random.Random(7), incident_start)
    window_start = (incident_start - timedelta(hours=1)).isoformat()
    window_end = (incident_start + timedelta(minutes=5)).isoformat()
    return incident, window_start, window_end


def _progress_node_names(incident_id: int) -> list[str]:
    """Read `NodeProgressEvent` rows back through a FRESH session, ordered
    by primary key (insertion order) -- proves the writes are real, durable
    Postgres rows, not just something visible within the same in-process
    session that wrote them (same convention as
    `tests/test_graph_response_planner_e2e.py`'s `AuditEvent` checks)."""
    fresh_db = SessionLocal()
    try:
        rows = (
            fresh_db.query(NodeProgressEvent)
            .filter(NodeProgressEvent.incident_id == incident_id)
            .order_by(NodeProgressEvent.id)
            .all()
        )
        # Sanity check the ordering assumption itself: id order must match
        # started_at order (both monotonic within one committed-per-node
        # write stream), not just happen to look ordered by luck.
        started_ats = [row.started_at for row in rows]
        assert started_ats == sorted(started_ats)
        return [row.node_name for row in rows]
    finally:
        fresh_db.close()


# --- Tests ---------------------------------------------------------------


async def test_safe_plan_records_progress_rows_in_node_order(monkeypatch):
    """A SAFE-only plan runs straight through to the real Action Executor
    (no human_approval, no recovery_check -- see
    `tests/test_graph_response_planner_e2e.py`'s docstring for why) -- the
    progress log should show exactly that lifecycle, in that order, one row
    per node invocation."""
    from backend.graph import run_incident_graph
    from backend.main import app

    client = TestClient(app)
    client.post("/api/simulation/reset")

    db = SessionLocal()
    try:
        incident, start, end = _inject_db_connection_exhaustion_incident(db)
        plan = ResponsePlan(
            actions=[
                ResponseAction(
                    action_type="gather_additional_diagnostics",
                    expected_benefit="collect more evidence before recommending a fix",
                    confidence=0.6,
                    llm_risk_assessment="no risk, read-only",
                )
            ]
        )
        _patch_all_fakes(monkeypatch, incident.service.name, start, end, plan)

        final_state = await run_incident_graph(db, incident, qdrant_client=get_qdrant_client())
        assert final_state.incident_status == IncidentStatus.DIAGNOSED

        node_names = _progress_node_names(incident.id)
        assert node_names == [
            "triage",
            "investigation",
            "rag",
            "root_cause",
            "response_planner",
            "action_executor",
        ]
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")


async def test_high_impact_plan_records_progress_up_through_paused_human_approval(monkeypatch):
    """A HIGH_IMPACT plan pauses at `human_approval_node`'s `interrupt()`
    (never resumed in this test -- see
    `tests/test_graph_response_planner_e2e.py`'s equivalent control-flow
    test). The progress wrapper writes its row BEFORE calling the real node
    function (see `backend.graph._with_progress`'s docstring), so a
    `human_approval` row must exist even though that node never finishes --
    exactly the "where is this incident stuck right now" signal Phase 8's
    live trace exists to show. `action_executor`/`recovery_check` must NOT
    appear -- the graph never reached them."""
    from backend.graph import run_incident_graph
    from backend.main import app

    client = TestClient(app)
    client.post("/api/simulation/reset")

    db = SessionLocal()
    try:
        incident, start, end = _inject_db_connection_exhaustion_incident(db)
        plan = ResponsePlan(
            actions=[
                ResponseAction(
                    action_type="rollback_deployment",
                    expected_benefit="removes the leaking deployed code path",
                    confidence=0.85,
                    llm_risk_assessment="moderate risk, briefly reverts a live deployment",
                )
            ]
        )
        _patch_all_fakes(monkeypatch, incident.service.name, start, end, plan)

        final_state = await run_incident_graph(db, incident, qdrant_client=get_qdrant_client())
        assert final_state.incident_status == IncidentStatus.AWAITING_APPROVAL

        node_names = _progress_node_names(incident.id)
        assert node_names == [
            "triage",
            "investigation",
            "rag",
            "root_cause",
            "response_planner",
            "human_approval",
        ]
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")


async def test_resume_writes_a_second_human_approval_row_then_a_duplicate_resume_writes_none(
    monkeypatch,
):
    """`_with_progress`'s docstring claims a genuine `/approve` resume
    causes `human_approval_node` to be re-executed from its start (per
    `backend.agents.human_approval_node`'s own documented replay-safety
    argument, empirically confirmed by
    `tests/test_human_approval.py::test_resuming_an_already_resumed_thread_never_recreates_audit_rows`),
    so it should write a SECOND `human_approval` progress row -- but a
    SECOND, duplicate resume on an already-completed thread is a pure
    LangGraph no-op (nothing left to resume from), so it must add NO further
    rows at all, not even a third `human_approval` one. This was flagged as
    an unverified claim by code review and is asserted directly here rather
    than left as a docstring-only assumption.

    Reuses `test_human_approval.py`'s `_run_to_interrupt` fixture (same
    `db_connection_exhaustion` + `rollback_deployment` -- a correct
    remediation, so the first resume runs all the way through
    `action_executor`/`recovery_check` to `RESOLVED`) and its
    resume-twice pattern, rather than reinventing either."""
    from backend.graph import resume_incident_graph
    from backend.main import app

    client = TestClient(app)
    client.post("/api/simulation/reset")

    db = SessionLocal()
    try:
        incident = await _run_to_interrupt(monkeypatch, db)

        node_names_after_pause = _progress_node_names(incident.id)
        assert node_names_after_pause == [
            "triage",
            "investigation",
            "rag",
            "root_cause",
            "response_planner",
            "human_approval",
        ]

        # Same "mark APPROVED directly, bypass backend.api.approvals" setup
        # test_human_approval.py's own resume-twice test uses -- isolates
        # what interrupt()/this wrapper guarantee, independent of the API
        # layer's own idempotency guard (already covered elsewhere).
        event = db.query(AuditEvent).filter(AuditEvent.incident_id == incident.id).one()
        event.decision_status = AuditDecisionStatus.APPROVED
        event.approver = "x"
        event.decided_at = datetime.now(UTC)
        db.commit()

        qdrant_client = get_qdrant_client()
        resume_payload = {"decision": "approved", "approver": "x"}

        state_1 = await resume_incident_graph(
            db, incident, resume_payload, qdrant_client=qdrant_client
        )
        assert state_1.incident_status == IncidentStatus.RESOLVED

        node_names_after_first_resume = _progress_node_names(incident.id)
        assert node_names_after_first_resume == [
            "triage",
            "investigation",
            "rag",
            "root_cause",
            "response_planner",
            "human_approval",
            "human_approval",  # re-executed from its start on genuine resume
            "action_executor",
            "recovery_check",
        ]

        state_2 = await resume_incident_graph(
            db, incident, resume_payload, qdrant_client=qdrant_client
        )
        assert state_2.incident_status == IncidentStatus.RESOLVED

        # Nothing left to resume -- LangGraph no-ops the whole call, so not
        # even a third human_approval row, let alone a second
        # action_executor/recovery_check pass.
        assert _progress_node_names(incident.id) == node_names_after_first_resume
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")
