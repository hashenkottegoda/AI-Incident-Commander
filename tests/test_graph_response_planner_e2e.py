"""End-to-end control-flow test extending `tests/test_graph_end_to_end.py`'s
pattern through Phase 6's Response Planner + inline Risk Classifier node.

Proves the full graph, given a diagnosed root cause:

(a) routes an all-SAFE response plan through the real Action Executor
    (`backend.agents.action_executor_node`) to the `DIAGNOSED` terminal
    state (nothing left to verify -- see that module's docstring) and
    leaves the `AuditEvent` row `EXECUTED` (auto-executed, no human
    decision needed), and
(b) routes a plan containing a HIGH_IMPACT action (`rollback_deployment`
    for `db_connection_exhaustion`) to the `AWAITING_APPROVAL` terminal
    state (still genuinely paused at `interrupt()` -- this test never
    approves it) and creates a `PENDING_APPROVAL` `AuditEvent` row with
    `approver` still null.

No test in this module makes a real OpenRouter API call:
`ChatOpenRouter` is monkeypatched in each LLM-calling node module
(`triage_node`, `investigation_node`, `root_cause_node`,
`response_planner_node`) with small fakes, following
`tests/test_graph_end_to_end.py`'s convention. The `db_connection_exhaustion`
scenario (not the cascading one) is used here and the fake Investigation
pass covers both required evidence tools (`get_deployments`,
`get_dependencies`) up front with a decisive confidence gap in the fake
Root Cause response, so the re-investigation loop never triggers -- this
test is purely about what happens *after* Root Cause, not the loop itself
(already covered by `test_graph_end_to_end.py`).

Postgres-dependent (incident injection + real `AuditEvent` writes) --
skipped cleanly without it, same convention as the rest of this suite.

The Response Planner node commits real rows mid-graph (see
`backend.agents.response_planner_node`'s docstring for why), so
`db.rollback()` alone does not clean up after these tests the way it does
for read-only Phase 5 graph tests -- cleanup goes through
`POST /api/simulation/reset` before and after each test, matching
`tests/test_graph_end_to_end.py::test_investigate_graph_endpoint_...`'s
convention for tests that persist real committed rows.
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
from backend.models import AuditDecisionStatus, AuditEvent, IncidentStatus, RiskClassification
from backend.rag.qdrant_client import get_qdrant_client
from backend.scripts.setup_checkpointer import to_psycopg_dsn
from backend.simulation.injector import inject_failure
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


# --- Fakes -------------------------------------------------------------------


class _FakeStructuredLLM:
    def __init__(self, result):
        self._result = result

    def invoke(self, messages):  # noqa: ARG002
        return self._result


class _FakeTriageChatOpenRouter:
    def __init__(self, *args, **kwargs):
        pass

    def with_structured_output(self, schema):  # noqa: ARG002
        from backend.agents.triage_node import TriageResult

        return _FakeStructuredLLM(TriageResult(affected_services=["checkout-service"]))


def _make_fake_investigation_chat_openrouter(service: str, start: str, end: str):
    """Single decisive pass: covers both required evidence tools
    (get_deployments, get_dependencies) up front, then stops -- never
    triggers the re-investigation loop."""

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

    class _FakeInvestigationChatOpenRouter:
        def __init__(self, *args, **kwargs):
            pass

        def bind_tools(self, tools):  # noqa: ARG002
            return _FakeInvestigationLLM()

    return _FakeInvestigationChatOpenRouter


def _make_fake_root_cause_chat_openrouter():
    """One decisive diagnosis -- large confidence gap so the loop never
    triggers regardless of the (already-complete) evidence coverage."""

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


# --- Shared incident fixture ------------------------------------------------


def _inject_db_connection_exhaustion_incident(db):
    scenario = load_all_scenarios()["db_connection_exhaustion"]
    incident_start = datetime(2026, 7, 5, 12, 0, tzinfo=UTC)
    incident = inject_failure(db, scenario, random.Random(7), incident_start)
    window_start = (incident_start - timedelta(hours=1)).isoformat()
    window_end = (incident_start + timedelta(minutes=5)).isoformat()
    return incident, window_start, window_end


# --- Tests ---------------------------------------------------------------


async def test_safe_only_plan_routes_to_diagnosed_with_executed_audit_row(monkeypatch):
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

        # A SAFE-only plan never touches human_approval -- it runs straight
        # through the real Action Executor to DIAGNOSED (nothing left to
        # verify, see action_executor_node's docstring for why DIAGNOSED is
        # the closest lifecycle fit).
        assert final_state.incident_status == IncidentStatus.DIAGNOSED
        assert len(final_state.recommended_actions) == 1
        ref = final_state.recommended_actions[0]
        assert ref["action_type"] == "gather_additional_diagnostics"
        assert ref["risk_classification"] == RiskClassification.SAFE.value
        assert ref["decision_status"] == AuditDecisionStatus.AUTO_EXECUTED.value

        # Verified against real Postgres via a fresh session -- proves the
        # AuditEvent write is real and durable, not just an in-memory state
        # field.
        fresh_db = SessionLocal()
        try:
            events = (
                fresh_db.query(AuditEvent).filter(AuditEvent.incident_id == incident.id).all()
            )
            assert len(events) == 1
            event = events[0]
            assert event.action_type == "gather_additional_diagnostics"
            assert event.risk_classification is RiskClassification.SAFE
            assert event.decision_status is AuditDecisionStatus.EXECUTED
            assert event.approver is None
            assert event.executed_at is not None
            assert event.execution_outcome is None  # SAFE actions have no recovery outcome
        finally:
            fresh_db.close()
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")


async def test_unrecognized_action_type_routes_to_awaiting_approval_end_to_end(monkeypatch):
    """The fail-safe default (unrecognized action_type -> HIGH_IMPACT) must
    hold through the actual graph path (run_incident_graph -> response
    planner node -> classify_risk -> IncidentState merge), not just when
    calling the node function directly (see
    tests/test_response_planner_node.py for that narrower check)."""
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
                    action_type="reboot_the_datacenter",
                    expected_benefit="the model invented this action name",
                    confidence=0.7,
                    llm_risk_assessment="the model claims this is completely safe",
                )
            ]
        )
        _patch_all_fakes(monkeypatch, incident.service.name, start, end, plan)

        final_state = await run_incident_graph(db, incident, qdrant_client=get_qdrant_client())

        assert final_state.incident_status == IncidentStatus.AWAITING_APPROVAL
        assert len(final_state.recommended_actions) == 1
        ref = final_state.recommended_actions[0]
        assert ref["action_type"] == "reboot_the_datacenter"
        assert ref["risk_classification"] == RiskClassification.HIGH_IMPACT.value
        assert ref["decision_status"] == AuditDecisionStatus.PENDING_APPROVAL.value

        fresh_db = SessionLocal()
        try:
            events = (
                fresh_db.query(AuditEvent).filter(AuditEvent.incident_id == incident.id).all()
            )
            assert len(events) == 1
            event = events[0]
            assert event.action_type == "reboot_the_datacenter"
            assert event.risk_classification is RiskClassification.HIGH_IMPACT
            assert event.decision_status is AuditDecisionStatus.PENDING_APPROVAL
            assert event.executed_at is None
        finally:
            fresh_db.close()
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")


async def test_high_impact_action_routes_to_awaiting_approval_with_pending_audit_row(monkeypatch):
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
        assert len(final_state.recommended_actions) == 1
        ref = final_state.recommended_actions[0]
        assert ref["action_type"] == "rollback_deployment"
        assert ref["risk_classification"] == RiskClassification.HIGH_IMPACT.value
        assert ref["decision_status"] == AuditDecisionStatus.PENDING_APPROVAL.value

        fresh_db = SessionLocal()
        try:
            events = (
                fresh_db.query(AuditEvent).filter(AuditEvent.incident_id == incident.id).all()
            )
            assert len(events) == 1
            event = events[0]
            assert event.action_type == "rollback_deployment"
            assert event.risk_classification is RiskClassification.HIGH_IMPACT
            assert event.decision_status is AuditDecisionStatus.PENDING_APPROVAL
            assert event.approver is None
            assert event.decided_at is None
            assert event.executed_at is None
        finally:
            fresh_db.close()
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")
