"""End-to-end control-flow test for Phase 5's full graph
(`backend.graph.run_incident_graph`), and for the new
`POST /api/incidents/{id}/investigate/graph` API route.

BUILD_PLAN.md Phase 5's verify criteria: *"full investigation lifecycle via
API returns ranked hypotheses + alternatives; the loop actually triggers on
cascading_payment_timeout (via the evidence-sufficiency path even if
confidences cluster)."*

No test in this module makes a real OpenRouter API call: `ChatOpenRouter`
is monkeypatched separately in each of the four LLM-calling node modules
(`triage_node`, `investigation_node`, `root_cause_node`,
`response_planner_node`) with a small fake chain of responses -- Triage
confirms the service, Investigation calls tools (real tool execution
against real seeded Postgres data -- only the LLM layer is faked) then
stops, Root Cause returns a structured diagnosis, and Response Planner
(Phase 6 -- the graph now continues past Root Cause, see `backend/graph.py`)
returns a single canned SAFE action so these Phase 5-focused tests aren't
otherwise affected by Phase 6's routing -- exactly the "mock chain of fake
LLM responses" pattern this task calls for, following `tests/test_rag.py`'s
`_FakeChatOpenRouter` convention.

The fakes are deliberately pass-aware (first vs. second Investigation/Root
Cause visit) so this test can prove the conditional re-investigation loop
actually executes a second Investigation pass on `cascading_payment_timeout`
rather than just asserting a final answer.

Postgres-dependent (incident injection + the graph's checkpointer) --
skipped cleanly without it, same convention as the rest of this suite.
Qdrant is used if reachable but not required (backend.agents.rag_node
degrades gracefully -- see tests/test_rag_node.py's dedicated coverage of
that path).
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
from backend.models.incident import IncidentStatus
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
    """Generic stand-in for `<ChatOpenRouter instance>.with_structured_output(...)`'s
    return value when the result doesn't need to vary call-to-call."""

    def __init__(self, result):
        self._result = result

    def invoke(self, messages):  # noqa: ARG002
        return self._result

    def with_retry(self, **kwargs):  # noqa: ARG002
        return self


class _FakeTriageChatOpenRouter:
    """Stand-in for `triage_node.ChatOpenRouter` -- confirms the reported
    service as-is, no surprises."""

    def __init__(self, *args, **kwargs):
        pass

    def with_structured_output(self, schema):  # noqa: ARG002
        from backend.agents.triage_node import TriageResult

        return _FakeStructuredLLM(TriageResult(affected_services=["checkout-service"]))


class _InvestigationPassCounter:
    """Shared across both Investigation node visits (a fresh
    `ChatOpenRouter(...)` is constructed each node execution, so the pass
    number has to live outside any one instance)."""

    def __init__(self):
        self.pass_number = 0


def _make_fake_investigation_chat_openrouter(
    counter: _InvestigationPassCounter, service, start, end
):
    class _FakeInvestigationLLM:
        def __init__(self):
            self._turn = 0

        def invoke(self, messages):  # noqa: ARG002
            self._turn += 1
            if self._turn > 1:
                return AIMessage(content="investigation pass complete, no further tool calls")

            # Pass 1 (the incomplete pass): only touches get_logs/get_metrics
            # -- deliberately never checks get_deployments/get_dependencies,
            # so the evidence-sufficiency predicate fails regardless of how
            # decisive the confidence gap looks.
            if counter.pass_number == 1:
                tool_calls = [
                    {
                        "name": "get_logs",
                        "args": {"service": service, "start": start, "end": end},
                        "id": "call_logs_1",
                    },
                    {
                        "name": "get_metrics",
                        "args": {
                            "service": service,
                            "metric_name": "db_connections_active",
                            "start": start,
                            "end": end,
                        },
                        "id": "call_metrics_1",
                    },
                ]
            else:
                # Pass 2 (the follow-up): closes the gap by checking both
                # required tool categories.
                tool_calls = [
                    {
                        "name": "get_deployments",
                        "args": {"service": service, "start": start, "end": end},
                        "id": "call_deploy_2",
                    },
                    {
                        "name": "get_dependencies",
                        "args": {"service": service, "start": start, "end": end},
                        "id": "call_deps_2",
                    },
                ]
            return AIMessage(content="", tool_calls=tool_calls)

        def with_retry(self, **kwargs):  # noqa: ARG002
            return self

    class _FakeInvestigationChatOpenRouter:
        def __init__(self, *args, **kwargs):
            pass

        def bind_tools(self, tools):  # noqa: ARG002
            counter.pass_number += 1
            return _FakeInvestigationLLM()

    return _FakeInvestigationChatOpenRouter


class _RootCausePassCounter:
    def __init__(self):
        self.pass_number = 0


def _make_fake_root_cause_chat_openrouter(counter: _RootCausePassCounter):
    class _FakeRootCauseStructuredLLM:
        def invoke(self, messages):  # noqa: ARG002
            counter.pass_number += 1
            if counter.pass_number == 1:
                # Deliberately clustered confidences (the "even if
                # confidences cluster" case from BUILD_PLAN.md) AND the
                # loud-symptom-but-wrong category, mirroring what a
                # first-pass agent that only checked logs/metrics might
                # conclude.
                return DiagnosisResult(
                    root_cause_category="database_connection_pool",
                    hypotheses=[
                        Hypothesis(
                            category="database_connection_pool",
                            rationale="db connections climbing under load",
                            confidence=0.55,
                        )
                    ],
                    alternative_hypotheses=[
                        Hypothesis(
                            category="upstream_dependency_timeout",
                            rationale="payment gateway also looked slow",
                            confidence=0.5,
                        )
                    ],
                    diagnostic_confidence=0.55,
                    evidence=[],
                )
            # Second pass: the dependency trace closed the gap -- decisive,
            # correct diagnosis.
            return DiagnosisResult(
                root_cause_category="upstream_dependency_timeout",
                hypotheses=[
                    Hypothesis(
                        category="upstream_dependency_timeout",
                        rationale=(
                            "checkout's dependency trace shows retries against a timing-out "
                            "payment gateway; db pressure is downstream of that"
                        ),
                        confidence=0.85,
                    )
                ],
                alternative_hypotheses=[
                    Hypothesis(
                        category="database_connection_pool",
                        rationale="db pressure was a symptom, not the root cause",
                        confidence=0.2,
                    )
                ],
                diagnostic_confidence=0.85,
                evidence=[],
            )

        def with_retry(self, **kwargs):  # noqa: ARG002
            return self

    class _FakeRootCauseChatOpenRouter:
        def __init__(self, *args, **kwargs):
            pass

        def with_structured_output(self, schema):  # noqa: ARG002
            return _FakeRootCauseStructuredLLM()

    return _FakeRootCauseChatOpenRouter


class _FakeResponsePlannerChatOpenRouter:
    """Phase 6's Response Planner node now runs immediately after Root
    Cause in the full graph (see `backend/graph.py`) -- these Phase
    5-focused tests aren't about the response side, so this fake always
    proposes a single, uncontroversial SAFE action (never a live API call)."""

    def __init__(self, *args, **kwargs):
        pass

    def with_structured_output(self, schema):  # noqa: ARG002
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
        return _FakeStructuredLLM(plan)


# --- Shared incident fixture ------------------------------------------------


def _inject_cascading_incident(db):
    scenario = load_all_scenarios()["cascading_payment_timeout"]
    incident_start = datetime(2026, 7, 5, 12, 0, tzinfo=UTC)
    incident = inject_failure(db, scenario, random.Random(123), incident_start)
    window_start = (incident_start - timedelta(hours=1)).isoformat()
    window_end = (incident_start + timedelta(minutes=5)).isoformat()
    return incident, window_start, window_end


def _patch_all_fakes(monkeypatch, service, start, end):
    import backend.agents.investigation_node as investigation_module
    import backend.agents.response_planner_node as response_planner_module
    import backend.agents.root_cause_node as rca_module
    import backend.agents.triage_node as triage_module

    investigation_counter = _InvestigationPassCounter()
    rca_counter = _RootCausePassCounter()

    monkeypatch.setattr(triage_module, "ChatOpenRouter", _FakeTriageChatOpenRouter)
    monkeypatch.setattr(
        investigation_module,
        "ChatOpenRouter",
        _make_fake_investigation_chat_openrouter(investigation_counter, service, start, end),
    )
    monkeypatch.setattr(
        rca_module, "ChatOpenRouter", _make_fake_root_cause_chat_openrouter(rca_counter)
    )
    monkeypatch.setattr(
        response_planner_module, "ChatOpenRouter", _FakeResponsePlannerChatOpenRouter
    )
    return investigation_counter, rca_counter


# --- Tests ---------------------------------------------------------------


async def test_full_graph_triggers_reinvestigation_loop_on_cascading_scenario(monkeypatch):
    """Phase 6's Response Planner node commits real `AuditEvent`/incident
    rows mid-graph (see `backend.agents.response_planner_node`) -- unlike
    Phase 5, `db.rollback()` alone no longer discards everything this test
    injects, so cleanup goes through `/api/simulation/reset` before and
    after, same as `test_investigate_graph_endpoint_...` below."""
    from backend.graph import run_incident_graph
    from backend.main import app

    client = TestClient(app)
    client.post("/api/simulation/reset")

    db = SessionLocal()
    try:
        incident, start, end = _inject_cascading_incident(db)
        investigation_counter, rca_counter = _patch_all_fakes(
            monkeypatch, incident.service.name, start, end
        )

        final_state = await run_incident_graph(db, incident, qdrant_client=get_qdrant_client())

        # The loop actually triggered: Investigation ran twice (initial pass
        # + one re-investigation pass), and so did Root Cause.
        assert final_state.investigation_iterations == 2
        assert investigation_counter.pass_number == 2
        assert rca_counter.pass_number == 2

        # Pass 1's evidence-sufficiency gap (missing deployments+dependencies
        # coverage) is what forced the loop -- confirm that coverage was
        # genuinely absent after pass 1 by checking it's present now (pass 2
        # added it) rather than having been there all along.
        tools_covered = {item.source_ref.tool for item in final_state.evidence}
        assert {"get_logs", "get_metrics", "get_deployments", "get_dependencies"}.issubset(
            tools_covered
        )

        # Final diagnosis is the decisive, correct second-pass result --
        # ranked hypotheses + alternatives both populated (BUILD_PLAN.md's
        # verify criterion).
        assert final_state.root_cause == "upstream_dependency_timeout"
        assert final_state.hypotheses
        assert final_state.hypotheses[0].category == "upstream_dependency_timeout"
        assert final_state.alternative_hypotheses
        # The graph now continues past Root Cause into Phase 6's Response
        # Planner, then the real Action Executor (see backend/graph.py) --
        # the fake planner always proposes a single SAFE action, which the
        # Action Executor auto-executes with nothing left to verify, so the
        # real terminal status is DIAGNOSED (see action_executor_node's
        # docstring for why that's the closest lifecycle fit for "a purely
        # informational action ran and nothing else is pending").
        assert final_state.incident_status == IncidentStatus.DIAGNOSED
        assert len(final_state.recommended_actions) == 1
        assert final_state.recommended_actions[0]["action_type"] == "generate_incident_report"
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")


async def test_full_graph_evidence_grounded_in_real_tool_calls(monkeypatch):
    """The fakes only replace the LLM layer -- tool execution against the
    real seeded incident data is real, so evidence descriptions should
    reflect actually-returned records, not placeholder text."""
    from backend.graph import run_incident_graph
    from backend.main import app

    client = TestClient(app)
    client.post("/api/simulation/reset")

    db = SessionLocal()
    try:
        incident, start, end = _inject_cascading_incident(db)
        _patch_all_fakes(monkeypatch, incident.service.name, start, end)

        final_state = await run_incident_graph(db, incident, qdrant_client=get_qdrant_client())

        tool_call_evidence = [
            item
            for item in final_state.evidence
            if item.source_ref.tool != "search_historical_incidents"
        ]
        assert tool_call_evidence
        # At least one evidence item should be grounded in a real record id
        # (get_logs/get_metrics against real seeded telemetry) -- proves the
        # tool layer, not just the LLM mock, actually ran.
        assert any(item.source_ref.record_id is not None for item in tool_call_evidence)
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")


def test_investigate_graph_endpoint_returns_ranked_hypotheses_and_alternatives(monkeypatch):
    """`POST /api/incidents/{id}/investigate/graph` -- BUILD_PLAN.md Phase 5's
    verify criterion: "full investigation lifecycle via API returns ranked
    hypotheses + alternatives."

    This test commits the injected incident (the API route reads it through
    its own request-scoped `SessionLocal()`, a different Postgres
    transaction than this test's `db`, so an uncommitted row would be
    invisible to it) -- cleaned up via `/api/simulation/reset` before and
    after, matching `tests/test_simulation_api.py`'s convention for tests
    that persist real rows.
    """
    from backend.main import app

    client = TestClient(app)
    client.post("/api/simulation/reset")

    db = SessionLocal()
    try:
        incident, start, end = _inject_cascading_incident(db)
        db.commit()
        _patch_all_fakes(monkeypatch, incident.service.name, start, end)

        response = client.post(f"/api/incidents/{incident.id}/investigate/graph")

        assert response.status_code == 200
        body = response.json()
        assert body["root_cause_category"] == "upstream_dependency_timeout"
        assert len(body["hypotheses"]) >= 1
        assert len(body["alternative_hypotheses"]) >= 1
        assert 0.0 <= body["diagnostic_confidence"] <= 1.0
    finally:
        db.rollback()
        db.close()
        client.post("/api/simulation/reset")
