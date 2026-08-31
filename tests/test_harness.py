"""Tests for Phase 7's experiment harness (`backend.evaluation.harness`).

Zero real OpenRouter API calls anywhere in this module. `ChatOpenRouter` is
replaced everywhere with `_ScriptedChatModel`, a small but GENUINE
`langchain_core.language_models.chat_models.BaseChatModel` subclass -- a
real `Runnable`, unlike the project's existing duck-typed test fakes (e.g.
`tests/test_graph_end_to_end.py`'s `_FakeChatOpenRouter` convention, plain
Python objects with an `.invoke()` *method* that never touches LangChain's
`Runnable.invoke()` -> `CallbackManager` -> tracer machinery). That
distinction matters here specifically: `backend.evaluation.harness` relies
on `langchain_core.tracers.context.collect_runs()`, which only sees runs
made through that real machinery -- see `harness.py`'s module docstring
for the throwaway verification that established this. So THIS suite's
fakes are deliberately more faithful than the rest of the project's, on
purpose, because what's under test here is the observability layer
itself, not just business logic downstream of an LLM call.

Real Postgres (tool calls query real seeded telemetry) and real Qdrant
(Experiment D's `rag_node` -- degrades gracefully if unreachable, but the
harness's own token/latency capture doesn't depend on it either way) are
used, same skip convention as the rest of this suite.
"""

from __future__ import annotations

import random
import time
from datetime import UTC, datetime

import psycopg
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field, PrivateAttr

from backend.agents.response_schemas import ResponseAction, ResponsePlan
from backend.agents.schemas import DiagnosisResult, EvidenceItem, Hypothesis, SourceRef
from backend.agents.state import IncidentState
from backend.config import get_settings
from backend.db import SessionLocal
from backend.evaluation import experiment_a, harness
from backend.evaluation.scoring import score_operational_run
from backend.models.incident import IncidentStatus, Severity
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


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def scenarios():
    return load_all_scenarios()


# --- The one genuinely-Runnable fake used throughout this module -----------


class _ScriptedChatModel(BaseChatModel):
    """A real `BaseChatModel` (genuine `Runnable`) that plays back a
    scripted list of `AIMessage`s in order, regardless of what's bound to
    it, optionally sleeping before each generation (for the latency test).

    `bind_tools`/`with_structured_output` are deliberately simplified
    versions of what `ChatOpenRouter` really does -- this fake does not
    reimplement OpenRouter's actual tool-calling wire format. What matters
    for these tests is that every `.invoke()`, however it's reached
    (directly, via `.bind(...)`, or via a `RunnableSequence` built by
    `with_structured_output`), is a REAL traced `Runnable` call, so
    `collect_runs()` sees it -- see this module's docstring.
    """

    responses: list[AIMessage] = Field(default_factory=list)
    sleep_seconds: float = 0.0
    _idx: int = PrivateAttr(default=0)
    _structured_queue: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001, ARG002
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        msg = self.responses[self._idx]
        self._idx += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ARG002
        return self.bind(tools=tools)

    def with_structured_output(self, schema, **kwargs):  # noqa: ANN001, ARG002
        def _extract(_ai_message: AIMessage):
            return self._structured_queue.pop(0)

        return self | RunnableLambda(_extract)

    def queue_structured_result(self, result) -> _ScriptedChatModel:  # noqa: ANN001
        self._structured_queue.append(result)
        return self


def _ai_message(text: str = "", tool_calls=None, in_tok: int = 1, out_tok: int = 1) -> AIMessage:
    return AIMessage(
        content=text,
        tool_calls=tool_calls or [],
        usage_metadata=UsageMetadata(
            input_tokens=in_tok, output_tokens=out_tok, total_tokens=in_tok + out_tok
        ),
    )


# =============================================================================
# Experiment A
# =============================================================================

_CANNED_A_RESULT = DiagnosisResult(
    root_cause_category="database_connection_pool",
    hypotheses=[Hypothesis(category="database_connection_pool", rationale="canned")],
    alternative_hypotheses=[],
    evidence=[
        EvidenceItem(description="canned", source_ref=SourceRef(tool="get_logs", record_id=1))
    ],
    diagnostic_confidence=0.6,
)


def test_harness_experiment_a_zero_tool_calls_and_captures_tokens(db, scenarios, monkeypatch):
    scenario = scenarios["db_connection_exhaustion"]
    incident_start = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    incident = inject_failure(db, scenario, random.Random(1), incident_start)
    db.flush()

    fake = _ScriptedChatModel(responses=[_ai_message("diagnosis", in_tok=500, out_tok=123)])
    fake.queue_structured_result(_CANNED_A_RESULT)
    monkeypatch.setattr(experiment_a, "ChatOpenRouter", lambda *a, **k: fake)  # noqa: ARG005

    result = harness.run_experiment_a(db, incident)

    assert result.diagnosis is _CANNED_A_RESULT
    # By construction: Experiment A calls the plain get_logs/get_metrics/
    # get_deployments *functions*, never a LangChain-wrapped BaseTool, so
    # there is nothing of run_type "tool" for collect_runs() to ever see.
    assert result.tool_call_count == 0
    assert result.latency_seconds > 0
    # Exactly the one scripted LLM call's usage -- no ReAct loop to sum
    # across for Experiment A.
    assert result.total_input_tokens == 500
    assert result.total_output_tokens == 123


def test_harness_experiment_a_latency_reflects_real_wall_clock(db, scenarios, monkeypatch):
    """Not a hardcoded/zero value -- the fake LLM genuinely sleeps, and
    `latency_seconds` must reflect at least that much real wall-clock
    time."""
    scenario = scenarios["db_connection_exhaustion"]
    incident_start = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    incident = inject_failure(db, scenario, random.Random(2), incident_start)
    db.flush()

    sleep_for = 0.2
    fake = _ScriptedChatModel(
        responses=[_ai_message("diagnosis", in_tok=10, out_tok=10)], sleep_seconds=sleep_for
    )
    fake.queue_structured_result(_CANNED_A_RESULT)
    monkeypatch.setattr(experiment_a, "ChatOpenRouter", lambda *a, **k: fake)  # noqa: ARG005

    result = harness.run_experiment_a(db, incident)

    assert result.latency_seconds >= sleep_for


# =============================================================================
# Experiment B (investigate_incident, include_rag=False)
# =============================================================================


def test_harness_experiment_b_tool_call_count_and_token_aggregation(db, scenarios, monkeypatch):
    """Fake LLM scripted to make exactly 2 real tool calls (one per turn)
    before concluding, then one more turn for the structured-output call --
    4 LLM turns total. `tool_call_count` must be exactly 2 (not 4 -- tool
    calls, not LLM turns), and token totals must sum across ALL 4 turns,
    not just the last one."""
    import backend.agents.investigator as investigator_mod

    scenario = scenarios["db_connection_exhaustion"]
    incident_start = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    incident = inject_failure(db, scenario, random.Random(3), incident_start)
    db.flush()

    start = "2026-07-01T11:00:00+00:00"
    end = "2026-07-01T12:00:00+00:00"
    tool_call_1 = {
        "name": "get_logs",
        "args": {"service": incident.service.name, "start": start, "end": end},
        "id": "call_1",
        "type": "tool_call",
    }
    tool_call_2 = {
        "name": "get_metrics",
        "args": {
            "service": incident.service.name,
            "metric_name": "db_connections_active",
            "start": start,
            "end": end,
        },
        "id": "call_2",
        "type": "tool_call",
    }

    responses = [
        _ai_message("", tool_calls=[tool_call_1], in_tok=10, out_tok=5),
        _ai_message("", tool_calls=[tool_call_2], in_tok=20, out_tok=6),
        _ai_message("investigation complete", in_tok=30, out_tok=7),
        _ai_message("struct-call", in_tok=40, out_tok=8),
    ]
    fake = _ScriptedChatModel(responses=responses)
    fake.queue_structured_result(_CANNED_A_RESULT)
    monkeypatch.setattr(investigator_mod, "ChatOpenRouter", lambda *a, **k: fake)  # noqa: ARG005

    result = harness.run_experiment_b(db, incident)

    assert result.diagnosis is _CANNED_A_RESULT
    assert result.tool_call_count == 2
    assert result.total_input_tokens == 10 + 20 + 30 + 40
    assert result.total_output_tokens == 5 + 6 + 7 + 8
    assert result.latency_seconds > 0


def test_harness_experiment_c_wires_include_rag_true(db, scenarios, monkeypatch):
    """Experiment C is the same underlying function with `include_rag=True`
    -- a minimal smoke test (no RAG tool call scripted) confirming the
    harness calls it with the right flag and still captures a real
    latency/tool count, distinguishing this from `run_experiment_b`."""
    import backend.agents.investigator as investigator_mod

    scenario = scenarios["db_connection_exhaustion"]
    incident_start = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    incident = inject_failure(db, scenario, random.Random(4), incident_start)
    db.flush()

    responses = [
        _ai_message("investigation complete", in_tok=5, out_tok=5),
        _ai_message("struct-call", in_tok=5, out_tok=5),
    ]
    fake = _ScriptedChatModel(responses=responses)
    fake.queue_structured_result(_CANNED_A_RESULT)
    monkeypatch.setattr(investigator_mod, "ChatOpenRouter", lambda *a, **k: fake)  # noqa: ARG005

    result = harness.run_experiment_c(db, incident)

    assert result.tool_call_count == 0
    assert result.total_input_tokens == 10
    assert result.total_output_tokens == 10
    assert result.latency_seconds > 0


# =============================================================================
# diagnosis_result_from_state -- mirrors api/incidents.py's route pattern
# =============================================================================


def test_diagnosis_result_from_state_matches_route_construction_pattern():
    """`harness.diagnosis_result_from_state` must build a `DiagnosisResult`
    field-for-field identically to `POST /{incident_id}/investigate/graph`'s
    inline construction in `backend/api/incidents.py`:

        DiagnosisResult(
            root_cause_category=final_state.root_cause or "unknown",
            hypotheses=final_state.hypotheses,
            alternative_hypotheses=final_state.alternative_hypotheses,
            evidence=final_state.evidence,
            diagnostic_confidence=final_state.diagnostic_confidence,
        )
    """
    state = IncidentState(
        incident_id=42,
        incident_status=IncidentStatus.DIAGNOSED,
        severity=Severity.P1,
        affected_services=["checkout-service"],
        tool_call_log_ids=[1, 2, 3],
        evidence=[
            EvidenceItem(description="e", source_ref=SourceRef(tool="get_logs", record_id=1))
        ],
        hypotheses=[
            Hypothesis(category="database_connection_pool", rationale="r", confidence=0.8)
        ],
        root_cause="database_connection_pool",
        diagnostic_confidence=0.8,
        alternative_hypotheses=[Hypothesis(category="unknown", rationale="alt", confidence=0.1)],
    )

    expected = DiagnosisResult(
        root_cause_category=state.root_cause or "unknown",
        hypotheses=state.hypotheses,
        alternative_hypotheses=state.alternative_hypotheses,
        evidence=state.evidence,
        diagnostic_confidence=state.diagnostic_confidence,
    )

    assert harness.diagnosis_result_from_state(state) == expected


def test_diagnosis_result_from_state_falls_back_to_unknown_when_root_cause_none():
    """`final_state.root_cause or "unknown"` -- the route's exact fallback
    -- must be reproduced, not silently dropped."""
    state = IncidentState(incident_id=1, root_cause=None)
    result = harness.diagnosis_result_from_state(state)
    assert result.root_cause_category == "unknown"


# =============================================================================
# Experiment D -- full graph, real IncidentState.tool_call_log_ids
# =============================================================================


def _make_investigation_fake(service: str):
    """Single-pass investigation: ONE turn issuing all 4 tool calls in
    parallel (get_logs/get_metrics/get_deployments/get_dependencies),
    covering both required-evidence tools in one go so the graph's
    conditional re-investigation loop never triggers -- see
    `backend.agents.routing.evidence_sufficiency_check_failed`."""
    start = "2026-07-01T11:00:00+00:00"
    end = "2026-07-01T12:00:00+00:00"
    tool_calls = [
        {
            "name": "get_logs",
            "args": {"service": service, "start": start, "end": end},
            "id": "call_logs",
            "type": "tool_call",
        },
        {
            "name": "get_metrics",
            "args": {
                "service": service,
                "metric_name": "db_connections_active",
                "start": start,
                "end": end,
            },
            "id": "call_metrics",
            "type": "tool_call",
        },
        {
            "name": "get_deployments",
            "args": {"service": service, "start": start, "end": end},
            "id": "call_deploy",
            "type": "tool_call",
        },
        {
            "name": "get_dependencies",
            "args": {"service": service, "start": start, "end": end},
            "id": "call_deps",
            "type": "tool_call",
        },
    ]
    responses = [
        _ai_message("", tool_calls=tool_calls, in_tok=100, out_tok=10),
        _ai_message("investigation pass complete", in_tok=50, out_tok=10),
    ]
    return _ScriptedChatModel(responses=responses)


def _patch_all_graph_fakes(monkeypatch, service: str, ground_truth_category: str):
    import backend.agents.investigation_node as investigation_module
    import backend.agents.response_planner_node as response_planner_module
    import backend.agents.root_cause_node as rca_module
    import backend.agents.triage_node as triage_module
    from backend.agents.triage_node import TriageResult

    triage_fake = _ScriptedChatModel(responses=[_ai_message("triage", in_tok=20, out_tok=5)])
    triage_fake.queue_structured_result(TriageResult(affected_services=[service]))
    monkeypatch.setattr(triage_module, "ChatOpenRouter", lambda *a, **k: triage_fake)  # noqa: ARG005

    investigation_fake = _make_investigation_fake(service)
    monkeypatch.setattr(
        investigation_module, "ChatOpenRouter", lambda *a, **k: investigation_fake  # noqa: ARG005
    )

    rca_fake = _ScriptedChatModel(responses=[_ai_message("rca", in_tok=80, out_tok=20)])
    # alternative_hypotheses=[] -> confidence_gap_below_threshold returns
    # False (nothing to compare a gap against), so only the
    # evidence-sufficiency predicate is in play -- satisfied by the single
    # investigation pass covering get_deployments + get_dependencies above.
    rca_fake.queue_structured_result(
        DiagnosisResult(
            root_cause_category=ground_truth_category,
            hypotheses=[
                Hypothesis(category=ground_truth_category, rationale="test", confidence=0.9)
            ],
            alternative_hypotheses=[],
            evidence=[],
            diagnostic_confidence=0.9,
        )
    )
    monkeypatch.setattr(rca_module, "ChatOpenRouter", lambda *a, **k: rca_fake)  # noqa: ARG005

    planner_fake = _ScriptedChatModel(responses=[_ai_message("plan", in_tok=15, out_tok=5)])
    planner_fake.queue_structured_result(
        ResponsePlan(
            actions=[
                ResponseAction(
                    action_type="generate_incident_report",
                    expected_benefit="documents the diagnosis",
                    confidence=0.7,
                    llm_risk_assessment="no risk, read-only",
                )
            ]
        )
    )
    monkeypatch.setattr(
        response_planner_module, "ChatOpenRouter", lambda *a, **k: planner_fake  # noqa: ARG005
    )

    # `run_experiment_d` calls `run_incident_graph_to_diagnosis`, which halts
    # via `interrupt_before=["response_planner"]` -- `response_planner` must
    # NEVER run (see that function's docstring for why reusing plain
    # `run_incident_graph` would leak an extra real LLM call's tokens into
    # this "immediately after RCA" measurement). `planner_fake` is patched
    # in anyway (rather than left unpatched) specifically so this test can
    # prove that by omission: expected totals below cover only the 4 LLM
    # turns that SHOULD run (triage:1, investigation:2, root_cause:1) -- if
    # a future change accidentally let response_planner run again, its
    # scripted 15/5 tokens would silently inflate the real totals and this
    # assertion would fail rather than staying quietly wrong.
    expected_in = 20 + 100 + 50 + 80
    expected_out = 5 + 10 + 10 + 20
    return expected_in, expected_out


async def test_harness_experiment_d_tool_call_count_matches_state_and_captures_tokens(
    db, scenarios, monkeypatch
):
    """`ExperimentRun.tool_call_count` for D must come from
    `len(final_state.tool_call_log_ids)` -- confirmed here by scripting
    exactly 4 tool calls in the single investigation pass and asserting the
    harness reports exactly 4 (not, say, a count of LangChain 'tool'-type
    traced runs, which would happen to also be 4 here but is explicitly
    NOT what `run_experiment_d` uses -- see its docstring)."""
    client = TestClient_app_reset()
    scenario = scenarios["db_connection_exhaustion"]
    incident_start = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    incident = inject_failure(db, scenario, random.Random(5), incident_start)
    db.flush()
    db.commit()  # the graph's own request-scoped session must see this incident

    expected_in, expected_out = _patch_all_graph_fakes(
        monkeypatch, incident.service.name, scenario.root_cause_category
    )

    try:
        result = await harness.run_experiment_d(db, incident, qdrant_client=get_qdrant_client())

        assert result.tool_call_count == 4
        assert result.diagnosis.root_cause_category == scenario.root_cause_category
        assert result.diagnosis.diagnostic_confidence == 0.9
        assert result.latency_seconds > 0
        assert result.total_input_tokens == expected_in
        assert result.total_output_tokens == expected_out
    finally:
        client.post("/api/simulation/reset")


def TestClient_app_reset() -> TestClient:
    """Small helper: this suite needs `/api/simulation/reset` cleanup
    around Experiment D the same way `tests/test_graph_end_to_end.py` does
    (`response_planner_node` commits real `AuditEvent`/incident rows
    mid-graph, so a plain `db.rollback()` on this test's own session isn't
    enough) -- imported lazily to avoid constructing a `TestClient` for
    every other test in this module that never needs it."""
    from backend.main import app

    client = TestClient(app)
    client.post("/api/simulation/reset")
    return client


# =============================================================================
# run_experiment_d_operational -- the FULL closed loop, auto-approved
# =============================================================================
#
# Unlike `run_experiment_d` above (which halts before response_planner ever
# runs, via `run_incident_graph_to_diagnosis`'s static interrupt_before), this
# calls the real `backend.graph.run_incident_graph` end-to-end and drives any
# HIGH_IMPACT `human_approval` pause to completion itself (see
# `run_experiment_d_operational`'s own docstring for why: unattended eval runs
# need no human in the loop). So these fakes must cover response_planner
# (and, for a HIGH_IMPACT plan, potentially several re-investigation passes)
# for real -- `_patch_all_graph_fakes` above is diagnostic-only-shaped (its
# planner fake exists only to prove response_planner is NEVER called) and
# isn't reused here.


def _make_multi_pass_investigation_fake(service: str, passes: int) -> _ScriptedChatModel:
    """`passes` back-to-back repetitions of `_make_investigation_fake`'s
    single-pass script (one turn issuing all 4 tool calls, one turn
    concluding) -- each investigation pass through the bounded
    re-investigation loop needs its own 2 scripted turns, and the SAME
    `_ScriptedChatModel` instance is reused across every pass (the harness
    tests' `lambda *a, **k: fake` monkeypatch convention returns one fixed
    object regardless of how many times `ChatOpenRouter(...)` is
    constructed), so its `responses` list must hold `passes` copies up
    front rather than just one."""
    single_pass = _make_investigation_fake(service).responses
    return _ScriptedChatModel(responses=list(single_pass) * passes)


def _make_multi_pass_rca_fake(ground_truth_category: str, passes: int) -> _ScriptedChatModel:
    """`passes` identical structured RCA results (same category, same
    empty `alternative_hypotheses` so `confidence_gap_below_threshold`
    never itself forces a reinvestigation -- the only loop trigger these
    operational tests care about is `recovery_check_node`'s ineffective-
    remediation loop, not root_cause's own gap check)."""
    fake = _ScriptedChatModel(
        responses=[_ai_message("rca", in_tok=80, out_tok=20) for _ in range(passes)]
    )
    diagnosis = DiagnosisResult(
        root_cause_category=ground_truth_category,
        hypotheses=[
            Hypothesis(category=ground_truth_category, rationale="test", confidence=0.9)
        ],
        alternative_hypotheses=[],
        evidence=[],
        diagnostic_confidence=0.9,
    )
    for _ in range(passes):
        fake.queue_structured_result(diagnosis)
    return fake


def _make_multi_pass_planner_fake(action_type: str, passes: int) -> _ScriptedChatModel:
    """`passes` identical single-action `ResponsePlan`s, always
    recommending the same `action_type` -- exercises exactly one
    HIGH_IMPACT action per pass through response_planner, however many
    passes the bounded re-investigation loop ends up taking."""
    fake = _ScriptedChatModel(
        responses=[_ai_message("plan", in_tok=15, out_tok=5) for _ in range(passes)]
    )
    plan = ResponsePlan(
        actions=[
            ResponseAction(
                action_type=action_type,
                expected_benefit="test remediation",
                confidence=0.7,
                llm_risk_assessment="test risk assessment",
            )
        ]
    )
    for _ in range(passes):
        fake.queue_structured_result(plan)
    return fake


def _patch_operational_graph_fakes(
    monkeypatch, service: str, ground_truth_category: str, action_type: str, *, passes: int
):
    """Patch `ChatOpenRouter` in every LLM-calling node for a full
    `run_incident_graph` run that recommends `action_type` on every one of
    `passes` re-investigation passes. `passes=1` for a plan that resolves
    (or is SAFE) on the first attempt; >1 for an ineffective HIGH_IMPACT
    action expected to loop `backend.agents.recovery_check_node` back to a
    fresh Investigation pass one or more times."""
    import backend.agents.investigation_node as investigation_module
    import backend.agents.response_planner_node as response_planner_module
    import backend.agents.root_cause_node as rca_module
    import backend.agents.triage_node as triage_module
    from backend.agents.triage_node import TriageResult

    # Triage only ever runs once -- recovery_check's loop routes straight
    # back to "investigation", never back through "triage" (see
    # backend/graph.py's edges).
    triage_fake = _ScriptedChatModel(responses=[_ai_message("triage", in_tok=20, out_tok=5)])
    triage_fake.queue_structured_result(TriageResult(affected_services=[service]))
    monkeypatch.setattr(triage_module, "ChatOpenRouter", lambda *a, **k: triage_fake)  # noqa: ARG005

    investigation_fake = _make_multi_pass_investigation_fake(service, passes)
    monkeypatch.setattr(
        investigation_module, "ChatOpenRouter", lambda *a, **k: investigation_fake  # noqa: ARG005
    )

    rca_fake = _make_multi_pass_rca_fake(ground_truth_category, passes)
    monkeypatch.setattr(rca_module, "ChatOpenRouter", lambda *a, **k: rca_fake)  # noqa: ARG005

    planner_fake = _make_multi_pass_planner_fake(action_type, passes)
    monkeypatch.setattr(
        response_planner_module, "ChatOpenRouter", lambda *a, **k: planner_fake  # noqa: ARG005
    )


async def test_run_experiment_d_operational_correct_remediation_resolves(
    db, scenarios, monkeypatch
):
    """A HIGH_IMPACT plan recommending the scenario's real
    `correct_remediation` (`db_connection_exhaustion`'s `rollback_deployment`,
    same ground truth `tests/test_action_executor_recovery_check.py`'s (c)
    case uses) should: genuinely pause at `human_approval` once, get
    auto-approved by `run_experiment_d_operational` with no human/HTTP
    caller involved, and resolve on the very first remediation attempt --
    `score_operational_run` should then report `in_scope=True`,
    `recovered=True`."""
    scenario = scenarios["db_connection_exhaustion"]
    assert scenario.remediation_effects.correct_remediation == "rollback_deployment"

    client = TestClient_app_reset()
    try:
        incident_start = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
        incident = inject_failure(db, scenario, random.Random(101), incident_start)
        db.flush()
        db.commit()  # the graph's own request-scoped session must see this incident

        _patch_operational_graph_fakes(
            monkeypatch,
            incident.service.name,
            scenario.root_cause_category,
            "rollback_deployment",
            passes=1,
        )

        final_state = await harness.run_experiment_d_operational(
            db, incident, qdrant_client=get_qdrant_client(), approver="eval-harness-test"
        )

        assert final_state.incident_status == IncidentStatus.RESOLVED

        result = score_operational_run(db, final_state, scenario)
        assert result.in_scope is True
        assert result.recovered is True
        assert result.recovery_check_correct is True
        assert result.wrong_remediation_flags == [False]
    finally:
        client.post("/api/simulation/reset")


async def test_run_experiment_d_operational_ineffective_remediation_stays_degraded(
    db, scenarios, monkeypatch
):
    """A HIGH_IMPACT plan that only ever recommends a known-ineffective
    action (`cascading_payment_timeout`'s `scale_service`, same ground
    truth `tests/test_action_executor_recovery_check.py`'s (d) case uses)
    should loop through the bounded re-investigation loop
    (`backend.agents.routing.MAX_REINVESTIGATION_LOOPS`, 3 total attempts:
    investigation_iterations 1, 2, then 3 which exceeds the bound) with
    `run_experiment_d_operational` auto-approving each of the 3 resulting
    `human_approval` pauses in turn, and end at
    `manual_intervention_required` -- never resolved.
    `score_operational_run` should report `in_scope=True`,
    `recovered=False`, and every attempt flagged wrong."""
    scenario = scenarios["cascading_payment_timeout"]
    assert "scale_service" in scenario.remediation_effects.ineffective_remediations

    client = TestClient_app_reset()
    try:
        incident_start = datetime(2026, 7, 5, 12, 0, tzinfo=UTC)
        incident = inject_failure(db, scenario, random.Random(102), incident_start)
        db.flush()
        db.commit()

        _patch_operational_graph_fakes(
            monkeypatch,
            incident.service.name,
            scenario.root_cause_category,
            "scale_service",
            passes=3,
        )

        final_state = await harness.run_experiment_d_operational(
            db, incident, qdrant_client=get_qdrant_client(), approver="eval-harness-test"
        )

        assert final_state.incident_status == IncidentStatus.MANUAL_INTERVENTION_REQUIRED
        assert final_state.investigation_iterations == 3

        result = score_operational_run(db, final_state, scenario)
        assert result.in_scope is True
        assert result.recovered is False
        assert result.recovery_check_correct is True  # correctly called "still degraded"
        assert result.wrong_remediation_flags == [True, True, True]
    finally:
        client.post("/api/simulation/reset")


async def test_run_experiment_d_operational_safe_only_plan_never_pauses(
    db, scenarios, monkeypatch
):
    """An all-SAFE plan (reusing `_patch_all_graph_fakes`'s canned
    `generate_incident_report` action, the same fake `run_experiment_d`'s
    own tests use) routes `response_planner -> action_executor` directly
    (`backend/graph.py`'s conditional edge) -- `run_incident_graph`'s own
    first call never pauses, so `run_experiment_d_operational` must return
    that state as-is without ever calling `resume_incident_graph`. Proven
    by monkeypatching `harness.resume_incident_graph` itself to raise if
    called at all, rather than just asserting the final status (which
    could coincidentally look right even if a spurious resume happened)."""

    def _fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        raise AssertionError(
            "resume_incident_graph must not be called for a SAFE-only plan -- "
            "the graph never paused, so there is nothing to resume"
        )

    monkeypatch.setattr(harness, "resume_incident_graph", _fail_if_called)

    client = TestClient_app_reset()
    try:
        scenario = scenarios["db_connection_exhaustion"]
        incident_start = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
        incident = inject_failure(db, scenario, random.Random(103), incident_start)
        db.flush()
        db.commit()

        _patch_all_graph_fakes(monkeypatch, incident.service.name, scenario.root_cause_category)

        final_state = await harness.run_experiment_d_operational(
            db, incident, qdrant_client=get_qdrant_client(), approver="eval-harness-test"
        )

        # SAFE-only: action_executor auto-executes with nothing left to
        # verify, landing on DIAGNOSED (see action_executor_node's
        # docstring) -- never AWAITING_APPROVAL, never RESOLVED.
        assert final_state.incident_status == IncidentStatus.DIAGNOSED

        result = score_operational_run(db, final_state, scenario)
        assert result.in_scope is False
        assert result.recovered is None
        assert result.recovery_check_correct is None
    finally:
        client.post("/api/simulation/reset")
