"""Tests for Phase 7's diagnostic scoring functions (`backend/evaluation/scoring.py`).

Follows `tests/test_tools.py`'s skip-without-Postgres pattern: these tests
need real seeded `LogEntry`/`MetricPoint`/`Deployment`/`TraceLite` rows
(via `inject_failure`) to check id validity against, but -- unlike
`tests/test_investigator.py` -- make ZERO LLM/Claude API calls. Every
`DiagnosisResult`/`SourceRef` here is hand-constructed, not model-generated,
so this module should run fast, free, and skip only on missing Postgres
(never on a missing `ANTHROPIC_API_KEY`).
"""

from __future__ import annotations

import random
from datetime import UTC, datetime

import psycopg
import pytest

from backend.agents.schemas import DiagnosisResult, EvidenceItem, SourceRef
from backend.config import get_settings
from backend.db import SessionLocal
from backend.evaluation.scoring import (
    SourceRefVerdict,
    ToolCallEfficiency,
    classify_source_ref,
    evidence_precision,
    evidence_source_ref_is_valid,
    hallucination_rate,
    root_cause_accuracy,
    tool_call_efficiency,
)
from backend.models import Deployment, LogEntry, MetricPoint, TraceLite
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


@pytest.fixture
def seeded_incident(db, scenarios):
    """Inject `db_connection_exhaustion` -- gives real `LogEntry`/
    `MetricPoint`/`Deployment` rows. Used for id-validity checks that don't
    care about scenario specifics, just "a real row with a real id exists".
    """
    scenario = scenarios["db_connection_exhaustion"]
    incident_start = datetime(2026, 7, 1, 15, 0, tzinfo=UTC)
    return inject_failure(db, scenario, random.Random(42), incident_start)


@pytest.fixture
def seeded_multi_service_incident(db, scenarios):
    """Inject `cascading_payment_timeout` -- the multi-service scenario that
    also produces `TraceLite` rows (via `get_dependencies`), needed for the
    dependencies-tool validity case.
    """
    scenario = scenarios["cascading_payment_timeout"]
    incident_start = datetime(2026, 7, 2, 10, 0, tzinfo=UTC)
    return inject_failure(db, scenario, random.Random(7), incident_start)


def _real_ids(db, model) -> tuple[int, int]:
    """Return (a real id, a definitely-nonexistent id) for `model`."""
    row = db.query(model).first()
    assert row is not None, f"expected at least one seeded {model.__name__} row"
    fake_id = -1  # primary keys are always positive here
    return row.id, fake_id


# --------------------------------------------------------------------------
# root_cause_accuracy
# --------------------------------------------------------------------------


def _diagnosis(category: str) -> DiagnosisResult:
    return DiagnosisResult(root_cause_category=category, diagnostic_confidence=0.8)


def test_root_cause_accuracy_exact_match_true():
    result = _diagnosis("database_connection_pool")
    assert root_cause_accuracy(result, "database_connection_pool") is True


def test_root_cause_accuracy_mismatch_false():
    result = _diagnosis("memory_resource_exhaustion")
    assert root_cause_accuracy(result, "database_connection_pool") is False


def test_root_cause_accuracy_unknown_vs_real_category_false():
    result = _diagnosis("unknown")
    assert root_cause_accuracy(result, "database_connection_pool") is False


# --------------------------------------------------------------------------
# evidence_source_ref_is_valid
# --------------------------------------------------------------------------


def test_valid_source_ref_for_each_db_backed_tool_real_id(
    db, seeded_incident, seeded_multi_service_incident
):
    # `db_connection_exhaustion` (seeded_incident) has real LogEntry/
    # MetricPoint/Deployment rows (deployment -> connection-pool ramp ->
    # ERROR logs). `cascading_payment_timeout` (seeded_multi_service_incident)
    # is the multi-service scenario that also produces TraceLite rows (via
    # get_dependencies) -- neither scenario alone guarantees all 4 tables
    # are populated, so both are injected here.
    log_id, _ = _real_ids(db, LogEntry)
    metric_id, _ = _real_ids(db, MetricPoint)
    deployment_id, _ = _real_ids(db, Deployment)
    trace_id, _ = _real_ids(db, TraceLite)

    assert evidence_source_ref_is_valid(db, SourceRef(tool="get_logs", record_id=log_id)) is True
    assert (
        evidence_source_ref_is_valid(db, SourceRef(tool="get_metrics", record_id=metric_id))
        is True
    )
    assert (
        evidence_source_ref_is_valid(
            db, SourceRef(tool="get_deployments", record_id=deployment_id)
        )
        is True
    )
    assert (
        evidence_source_ref_is_valid(db, SourceRef(tool="get_dependencies", record_id=trace_id))
        is True
    )


def test_invalid_source_ref_for_each_db_backed_tool_fabricated_id(db, seeded_incident):
    fake_id = -999
    assert (
        evidence_source_ref_is_valid(db, SourceRef(tool="get_logs", record_id=fake_id)) is False
    )
    assert (
        evidence_source_ref_is_valid(db, SourceRef(tool="get_metrics", record_id=fake_id))
        is False
    )
    assert (
        evidence_source_ref_is_valid(db, SourceRef(tool="get_deployments", record_id=fake_id))
        is False
    )
    assert (
        evidence_source_ref_is_valid(db, SourceRef(tool="get_dependencies", record_id=fake_id))
        is False
    )


def test_real_id_under_wrong_tool_is_invalid(db, seeded_incident):
    """A genuinely real LogEntry.id cited under tool="get_metrics" must be
    invalid -- that id was never returned by a get_metrics call, it doesn't
    exist in MetricPoint."""
    log_id, _ = _real_ids(db, LogEntry)

    # Sanity check: this id is real for LogEntry...
    assert evidence_source_ref_is_valid(db, SourceRef(tool="get_logs", record_id=log_id)) is True
    # ...but citing it under get_metrics must fail, since MetricPoint has no
    # row with this id (or if it coincidentally did, that would be a
    # different real row -- the point is the *tool* determines the table).
    metric_row = db.get(MetricPoint, log_id)
    if metric_row is not None:
        pytest.skip("log_id coincidentally also a real MetricPoint id; not a useful case here")
    assert (
        evidence_source_ref_is_valid(db, SourceRef(tool="get_metrics", record_id=log_id))
        is False
    )


def test_unrecognized_tool_name_is_invalid_fail_safe(db, seeded_incident):
    log_id, _ = _real_ids(db, LogEntry)
    # Even citing a real id under a made-up tool name must fail closed.
    made_up_tool = "get_totally_made_up_tool"
    assert (
        evidence_source_ref_is_valid(db, SourceRef(tool=made_up_tool, record_id=log_id)) is False
    )
    assert (
        evidence_source_ref_is_valid(db, SourceRef(tool=made_up_tool, query="anything")) is False
    )


def test_real_historical_incident_id_is_valid(db):
    assert (
        evidence_source_ref_is_valid(
            db, SourceRef(tool="search_historical_incidents", query="hist-001")
        )
        is True
    )


def test_fake_historical_incident_id_is_invalid(db):
    assert (
        evidence_source_ref_is_valid(
            db, SourceRef(tool="search_historical_incidents", query="hist-999")
        )
        is False
    )


def test_historical_incident_with_record_id_set_is_invalid(db):
    """record_id is int-typed but historical incident ids are strings --
    a populated record_id here can never legitimately be real."""
    assert (
        evidence_source_ref_is_valid(
            db, SourceRef(tool="search_historical_incidents", record_id=1, query="hist-001")
        )
        is False
    )


def test_historical_incident_with_no_query_or_record_id_is_invalid(db):
    assert (
        evidence_source_ref_is_valid(db, SourceRef(tool="search_historical_incidents")) is False
    )


def test_query_only_case_for_db_backed_tool_is_unverifiable_not_valid(db, seeded_incident):
    """Documented convention: a non-empty query string under a recognized
    DB-backed tool name classifies as UNVERIFIABLE, not VALID -- this is
    the fix for a real scoring loophole a code review caught (see
    SourceRefVerdict's docstring): treating query-only citations as VALID
    let a model game hallucination_rate by never citing record_id.
    `evidence_source_ref_is_valid` (the boolean convenience wrapper) must
    therefore return False for this case."""
    ref = SourceRef(tool="get_logs", query="payment-service 12:00-12:05, zero rows returned")
    assert classify_source_ref(db, ref) is SourceRefVerdict.UNVERIFIABLE
    assert evidence_source_ref_is_valid(db, ref) is False


def test_no_record_id_and_no_query_is_invalid(db, seeded_incident):
    assert evidence_source_ref_is_valid(db, SourceRef(tool="get_logs")) is False


def test_blank_query_is_invalid(db, seeded_incident):
    assert evidence_source_ref_is_valid(db, SourceRef(tool="get_logs", query="   ")) is False


# --------------------------------------------------------------------------
# evidence_precision / hallucination_rate
# --------------------------------------------------------------------------


def _evidence_item(source_ref: SourceRef) -> EvidenceItem:
    return EvidenceItem(description="synthetic evidence item", source_ref=source_ref)


def test_evidence_precision_known_mix_of_valid_and_invalid(db, seeded_incident):
    log_id, _ = _real_ids(db, LogEntry)
    metric_id, _ = _real_ids(db, MetricPoint)
    deployment_id, _ = _real_ids(db, Deployment)

    result = DiagnosisResult(
        root_cause_category="database_connection_pool",
        diagnostic_confidence=0.9,
        evidence=[
            _evidence_item(SourceRef(tool="get_logs", record_id=log_id)),  # valid
            _evidence_item(SourceRef(tool="get_metrics", record_id=metric_id)),  # valid
            _evidence_item(SourceRef(tool="get_deployments", record_id=deployment_id)),  # valid
            _evidence_item(SourceRef(tool="get_logs", record_id=-999)),  # invalid
        ],
    )

    assert evidence_precision(db, result) == pytest.approx(0.75)


def test_evidence_precision_empty_evidence_is_zero(db):
    result = DiagnosisResult(root_cause_category="unknown", diagnostic_confidence=0.1, evidence=[])
    assert evidence_precision(db, result) == 0.0


def test_evidence_precision_all_valid_is_one(db, seeded_incident):
    log_id, _ = _real_ids(db, LogEntry)
    result = DiagnosisResult(
        root_cause_category="database_connection_pool",
        diagnostic_confidence=0.9,
        evidence=[_evidence_item(SourceRef(tool="get_logs", record_id=log_id))],
    )
    assert evidence_precision(db, result) == 1.0


def test_evidence_precision_all_invalid_is_zero(db, seeded_incident):
    result = DiagnosisResult(
        root_cause_category="database_connection_pool",
        diagnostic_confidence=0.9,
        evidence=[_evidence_item(SourceRef(tool="get_logs", record_id=-999))],
    )
    assert evidence_precision(db, result) == 0.0


def test_evidence_precision_excludes_unverifiable_from_denominator(db, seeded_incident):
    """The gaming-loophole fix, in effect: mixing verifiable and
    unverifiable citations must score based on the verifiable subset only
    -- the unverifiable ones are neither rewarded nor punished, but they
    must NOT inflate the denominator in a way that dilutes a genuinely bad
    (or good) verifiable ratio."""
    log_id, _ = _real_ids(db, LogEntry)

    # 1 valid + 1 invalid (verifiable) + 2 unverifiable -> precision must be
    # 0.5 (1/2 of the VERIFIABLE subset), not 0.25 (1/4 of everything cited).
    result = DiagnosisResult(
        root_cause_category="database_connection_pool",
        diagnostic_confidence=0.9,
        evidence=[
            _evidence_item(SourceRef(tool="get_logs", record_id=log_id)),  # valid
            _evidence_item(SourceRef(tool="get_logs", record_id=-999)),  # invalid
            _evidence_item(SourceRef(tool="get_logs", query="zero rows in this range")),
            _evidence_item(SourceRef(tool="get_metrics", query="no anomalies observed")),
        ],
    )
    assert evidence_precision(db, result) == pytest.approx(0.5)


def test_evidence_precision_all_unverifiable_is_zero_not_one(db, seeded_incident):
    """A model that cites ONLY query-only evidence (the exact gaming
    strategy this fix closes) must score 0.0 precision, not a vacuous 1.0
    from an empty valid/invalid denominator -- see evidence_precision's
    'no VERIFIABLE evidence at all' convention."""
    result = DiagnosisResult(
        root_cause_category="database_connection_pool",
        diagnostic_confidence=0.9,
        evidence=[
            _evidence_item(SourceRef(tool="get_logs", query="something happened")),
            _evidence_item(SourceRef(tool="get_metrics", query="something else happened")),
        ],
    )
    assert evidence_precision(db, result) == 0.0
    assert hallucination_rate(db, result) == 1.0


@pytest.mark.parametrize(
    "evidence_refs",
    [
        [],
        [SourceRef(tool="get_logs", record_id=-1)],
        [SourceRef(tool="get_logs", query="something")],
    ],
)
def test_hallucination_rate_is_complement_of_precision(db, seeded_incident, evidence_refs):
    result = DiagnosisResult(
        root_cause_category="database_connection_pool",
        diagnostic_confidence=0.5,
        evidence=[_evidence_item(ref) for ref in evidence_refs],
    )
    precision = evidence_precision(db, result)
    assert hallucination_rate(db, result) == pytest.approx(1.0 - precision)


def test_hallucination_rate_known_mix(db, seeded_incident):
    log_id, _ = _real_ids(db, LogEntry)
    result = DiagnosisResult(
        root_cause_category="database_connection_pool",
        diagnostic_confidence=0.9,
        evidence=[
            _evidence_item(SourceRef(tool="get_logs", record_id=log_id)),  # valid
            _evidence_item(SourceRef(tool="get_logs", record_id=-1)),  # invalid
            _evidence_item(SourceRef(tool="get_logs", record_id=-2)),  # invalid
            _evidence_item(SourceRef(tool="get_logs", record_id=-3)),  # invalid
        ],
    )
    assert hallucination_rate(db, result) == pytest.approx(0.75)


def test_hallucination_rate_empty_evidence_is_one(db):
    result = DiagnosisResult(root_cause_category="unknown", diagnostic_confidence=0.1, evidence=[])
    assert hallucination_rate(db, result) == 1.0


# --------------------------------------------------------------------------
# tool_call_efficiency
# --------------------------------------------------------------------------


def test_tool_call_efficiency_basic_ratio():
    result = tool_call_efficiency(tool_call_count=4, evidence_count=8)
    assert result == ToolCallEfficiency(tool_call_count=4, evidence_per_tool_call=2.0)


def test_tool_call_efficiency_zero_calls_ratio_is_none():
    """Experiment A makes 0 tool calls by design -- the ratio is undefined,
    not zero."""
    result = tool_call_efficiency(tool_call_count=0, evidence_count=5)
    assert result.tool_call_count == 0
    assert result.evidence_per_tool_call is None


def test_tool_call_efficiency_zero_calls_zero_evidence():
    result = tool_call_efficiency(tool_call_count=0, evidence_count=0)
    assert result == ToolCallEfficiency(tool_call_count=0, evidence_per_tool_call=None)


def test_tool_call_efficiency_zero_evidence_nonzero_calls():
    result = tool_call_efficiency(tool_call_count=6, evidence_count=0)
    assert result == ToolCallEfficiency(tool_call_count=6, evidence_per_tool_call=0.0)


def test_tool_call_efficiency_negative_tool_call_count_raises():
    with pytest.raises(ValueError, match="tool_call_count"):
        tool_call_efficiency(tool_call_count=-1, evidence_count=0)


def test_tool_call_efficiency_negative_evidence_count_raises():
    with pytest.raises(ValueError, match="evidence_count"):
        tool_call_efficiency(tool_call_count=1, evidence_count=-1)
