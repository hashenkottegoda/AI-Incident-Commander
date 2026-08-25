"""Tests for Phase 7's Experiment A (`backend.evaluation.experiment_a`) --
the context-stuffing baseline: no tools, no ReAct loop, exactly one LLM
call over a hand-assembled dump of all telemetry in a generous window.

Follows `tests/test_graph_end_to_end.py`'s `_FakeChatAnthropic`/
`_FakeStructuredLLM` convention: `ChatAnthropic` is monkeypatched with a
small fake that records what it was asked to invoke and returns a canned
`DiagnosisResult`, so this suite makes zero real Anthropic API calls while
still exercising the real prompt-assembly code and the real tool-layer
queries against real seeded Postgres data.

Skipped cleanly without Postgres, same convention as the rest of this
suite (`tests/test_injector.py`, `tests/test_tools.py`).
"""

from __future__ import annotations

import random
from datetime import UTC, datetime

import psycopg
import pytest
from sqlalchemy import select

from backend.agents.schemas import DiagnosisResult, EvidenceItem, Hypothesis, SourceRef
from backend.config import get_settings
from backend.db import SessionLocal
from backend.evaluation import experiment_a
from backend.models import Deployment, LogEntry, MetricPoint, Service
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


_CANNED_RESULT = DiagnosisResult(
    root_cause_category="database_connection_pool",
    hypotheses=[
        Hypothesis(category="database_connection_pool", rationale="canned test hypothesis")
    ],
    alternative_hypotheses=[],
    evidence=[
        EvidenceItem(
            description="canned evidence",
            source_ref=SourceRef(tool="get_logs", record_id=1),
        )
    ],
    diagnostic_confidence=0.6,
)


class _RecordingStructuredLLM:
    """Stand-in for `<ChatAnthropic instance>.with_structured_output(...)`'s
    return value -- records every `.invoke()` call so the test can assert
    exactly one happened (no ReAct loop) and inspect exactly what was sent."""

    def __init__(self, capture: dict, result: DiagnosisResult):
        self._capture = capture
        self._result = result

    def invoke(self, messages):
        self._capture["messages"] = messages
        self._capture["invoke_count"] = self._capture.get("invoke_count", 0) + 1
        return self._result


class _RecordingChatAnthropic:
    """Fake `ChatAnthropic` -- records every construction (proving no
    ChatAnthropic instance sneaks in additional calls) and hands back a
    `_RecordingStructuredLLM` from `.with_structured_output()`."""

    instances: list[_RecordingChatAnthropic] = []

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs
        self.capture: dict = {}
        _RecordingChatAnthropic.instances.append(self)

    def with_structured_output(self, schema):  # noqa: ARG002
        return _RecordingStructuredLLM(self.capture, _CANNED_RESULT)


@pytest.fixture(autouse=True)
def _fake_chat_anthropic(monkeypatch):
    _RecordingChatAnthropic.instances = []
    monkeypatch.setattr(experiment_a, "ChatAnthropic", _RecordingChatAnthropic)
    yield


def _prompt_text(llm_instance: _RecordingChatAnthropic) -> str:
    return "\n".join(
        message.content
        for message in llm_instance.capture["messages"]
        if isinstance(message.content, str)
    )


def test_context_stuffing_baseline_dumps_real_telemetry_and_makes_one_llm_call(db, scenarios):
    scenario = scenarios["db_connection_exhaustion"]
    incident_start = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    incident = inject_failure(db, scenario, random.Random(123), incident_start)
    db.flush()

    result = experiment_a.run_context_stuffing_baseline(db, incident)

    assert result is _CANNED_RESULT

    # Exactly one LLM call: one ChatAnthropic construction, one invoke --
    # no ReAct loop, no tool-calling round trips.
    assert len(_RecordingChatAnthropic.instances) == 1
    llm_instance = _RecordingChatAnthropic.instances[0]
    assert llm_instance.capture.get("invoke_count") == 1

    prompt_text = _prompt_text(llm_instance)

    # Real telemetry, not a placeholder: the injected deployment's version
    # string (db_connection_exhaustion's causal_chain leads with
    # `checkout_deployment_v1.8.2`) appears verbatim, alongside its real
    # database id -- pulled back from Postgres rather than hardcoded, so
    # this stays correct if the injector's version-parsing ever changes.
    deployment = db.execute(
        select(Deployment).where(Deployment.service_id == incident.service_id)
    ).scalars().first()
    assert deployment is not None
    assert deployment.version in prompt_text
    assert f"[id={deployment.id}]" in prompt_text

    # A real log row's id is inline and citable (connection_pool_exhausted
    # is an ERROR_CLUSTER entry, so this scenario always writes ERROR logs).
    log_row = db.execute(
        select(LogEntry).where(LogEntry.service_id == incident.service_id)
    ).scalars().first()
    assert log_row is not None
    assert f"[id={log_row.id}]" in prompt_text
    assert log_row.message in prompt_text

    # A real metric row's id is inline too (db_connections_active ramp).
    metric_row = db.execute(
        select(MetricPoint).where(MetricPoint.service_id == incident.service_id)
    ).scalars().first()
    assert metric_row is not None
    assert f"[id={metric_row.id}]" in prompt_text


def test_window_covers_memory_leaks_full_causal_chain(db, scenarios):
    """memory_leak's causal chain spans hours
    (`backend.simulation.injector.SCENARIO_TIMING_OVERRIDES` gives it a
    6-hour `pre_incident_window` and 1-hour `chain_stagger`, vs. every
    other scenario's 45-minute default). Unlike Experiments B/C/D,
    Experiment A cannot iteratively widen its window if the first query
    comes back thin -- it must get a wide-enough window in one shot.
    Confirm the assembled telemetry actually includes evidence from early
    in the causal chain (hours before `detected_at`), not just the last
    few minutes."""
    scenario = scenarios["memory_leak"]
    incident_start = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
    incident = inject_failure(db, scenario, random.Random(456), incident_start)
    db.flush()

    experiment_a.run_context_stuffing_baseline(db, incident)

    llm_instance = _RecordingChatAnthropic.instances[0]
    prompt_text = _prompt_text(llm_instance)

    # The earliest metric point ever written for this service is the
    # start of the 6-hour baseline window -- if Experiment A's window were
    # only, say, 45 minutes wide (the default for other scenarios), this
    # row would never have been queried and couldn't appear in the prompt.
    earliest_metric = db.execute(
        select(MetricPoint)
        .where(MetricPoint.service_id == incident.service_id)
        .order_by(MetricPoint.timestamp.asc())
    ).scalars().first()
    assert earliest_metric is not None

    hours_before_incident = (incident_start - earliest_metric.timestamp).total_seconds() / 3600
    # Sanity-check the fixture itself: this really is "hours before", not
    # "a few minutes before" -- otherwise the assertion below wouldn't be
    # testing what this test claims to test.
    assert hours_before_incident > 3

    assert f"[id={earliest_metric.id}]" in prompt_text

    # memory_leak is deliberately deploy-free (see the scenario's own YAML
    # comment) -- the empty DEPLOYMENTS section is itself real, correctly
    # queried "no recent deployment" evidence, not evidence the window
    # missed.
    assert "no deployments in this window" in prompt_text


def test_context_covers_root_cause_service_not_just_affected_service(db, scenarios):
    """Regression test for a real bug caught in code review: an earlier
    version of _assemble_context queried only `incident.service.name`
    (always checkout-service, the scenario's `affected_service`). For
    cascading_payment_timeout, the actual root-cause evidence -- payment-
    service's own latency ramp and timeout/error-response logs -- lives on
    payment-service, not checkout-service; the scenario's own YAML header
    comment documents this explicitly as evidence "only visible by
    querying payment-service's own logs/metrics, not checkout's."
    Restricting Experiment A to one service structurally blinded it to
    this evidence for a reason unrelated to "no selective retrieval" (the
    actual architecture variable Phase 7 measures), biasing the A/B/C/D
    comparison for this scenario. This test fails against the buggy
    single-service version and passes once telemetry is gathered across
    all canonical services."""
    scenario = scenarios["cascading_payment_timeout"]
    incident_start = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
    incident = inject_failure(db, scenario, random.Random(789), incident_start)
    db.flush()

    experiment_a.run_context_stuffing_baseline(db, incident)

    llm_instance = _RecordingChatAnthropic.instances[0]
    prompt_text = _prompt_text(llm_instance)

    payment_service = db.execute(
        select(Service).where(Service.name == "payment-service")
    ).scalar_one()

    # Payment-service's own anomalous latency metric must be visible --
    # this is the "quiet root cause" signal, not the loud checkout-side
    # database symptom.
    payment_latency = db.execute(
        select(MetricPoint)
        .where(
            MetricPoint.service_id == payment_service.id,
            MetricPoint.metric_name == "latency_p99_ms",
        )
        .order_by(MetricPoint.value.desc())
    ).scalars().first()
    assert payment_latency is not None
    assert f"[id={payment_latency.id}]" in prompt_text
    assert "payment-service" in prompt_text

    # Payment-service's own error/timeout log burst must be visible too.
    payment_error_log = db.execute(
        select(LogEntry).where(
            LogEntry.service_id == payment_service.id, LogEntry.level == "error"
        )
    ).scalars().first()
    assert payment_error_log is not None
    assert f"[id={payment_error_log.id}]" in prompt_text
    assert payment_error_log.message in prompt_text
