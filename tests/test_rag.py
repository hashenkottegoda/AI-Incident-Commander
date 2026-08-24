"""Tests for Phase 4's RAG layer (`backend/rag/`,
`backend/tools/historical_incidents.py`,
`backend/scripts/seed_historical_incidents.py`).

Follows `tests/test_tools.py`'s pattern: skip the whole module when Qdrant
isn't reachable (start it with `docker compose up -d qdrant`). One test
also needs Postgres (the investigator-wiring structural check, since it
injects a real `Incident` row) and carries an additional skip for that.

No test in this module makes a Claude/Anthropic API call. The investigator
wiring test replaces `ChatAnthropic` with an in-process fake so it can
assert `search_historical_incidents` is actually bound into
`investigate_incident`'s tool list without spending real API credits or
network calls (Phase 4's cost constraint: local sentence-transformers +
Qdrant only).
"""

from __future__ import annotations

import random
from collections import Counter
from datetime import UTC, datetime

import psycopg
import pytest
from langchain_core.messages import AIMessage

from backend.config import get_settings
from backend.rag.embeddings import (
    assemble_incident_text,
    embed_incident_summary,
    embedding_dimension,
)
from backend.rag.historical_incidents import load_historical_incidents
from backend.rag.qdrant_client import (
    COLLECTION_NAME,
    ensure_collection,
    get_qdrant_client,
    point_id_for,
)
from backend.rag.schemas import IncidentSummary
from backend.scripts.seed_historical_incidents import seed_historical_incidents
from backend.scripts.setup_checkpointer import to_psycopg_dsn
from backend.tools.historical_incidents import (
    make_search_historical_incidents_tool,
    search_historical_incidents,
)


def _qdrant_reachable() -> bool:
    client = get_qdrant_client()
    try:
        client.get_collections()
        return True
    except Exception:
        return False


def _postgres_reachable() -> bool:
    dsn = to_psycopg_dsn(get_settings().database_url)
    try:
        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not _qdrant_reachable(),
    reason="Qdrant not reachable at QDRANT_URL (start it with `docker compose up -d qdrant`)",
)


@pytest.fixture(scope="module")
def seeded_client():
    """Seed the real `historical_incidents` collection once for this
    module's tests, via the actual seed script entrypoint (not a mock) --
    exactly what `uv run python -m backend.scripts.seed_historical_incidents`
    does."""
    count = seed_historical_incidents()
    client = get_qdrant_client()
    yield client, count


# --- historical_incidents.yaml -------------------------------------------


def test_loads_20_historical_incidents_spanning_all_categories():
    incidents = load_historical_incidents()
    assert len(incidents) == 20
    assert len({incident.id for incident in incidents}) == 20  # all ids unique

    categories = Counter(incident.root_cause_category for incident in incidents)
    expected_categories = {
        "database_connection_pool",
        "memory_resource_exhaustion",
        "application_bug",
        "upstream_dependency_failure",
        "inefficient_database_query",
        "upstream_dependency_timeout",
    }
    assert set(categories) == expected_categories
    # A real mix, not one category dominating or a category with a single entry.
    assert all(count >= 2 for count in categories.values())
    assert max(categories.values()) <= 4


def test_historical_incident_summary_matches_incident_summary_shape():
    incidents = load_historical_incidents()
    summary = incidents[0].summary()
    assert isinstance(summary, IncidentSummary)
    assert summary.service == incidents[0].service
    assert summary.symptoms == incidents[0].symptoms
    assert summary.timeline == incidents[0].timeline


# --- embeddings.py ----------------------------------------------------------


def test_assemble_incident_text_exact_format():
    summary = IncidentSummary(
        service="checkout-service",
        symptoms=["db_connections_high", "checkout_failures"],
        recent_changes="v1.8.2 deployed",
        observed_dependencies=None,
        timeline="deploy then errors",
    )
    text = assemble_incident_text(summary)
    assert text == (
        "Service: checkout-service\n"
        "Symptoms: db_connections_high; checkout_failures\n"
        "Recent changes: v1.8.2 deployed\n"
        "Observed dependencies: none\n"
        "Timeline: deploy then errors"
    )


def test_embedding_dimension_matches_configured_model():
    # all-MiniLM-L6-v2 (the default EMBEDDING_MODEL) is a real, verified
    # 384-dim model -- not a guessed constant (verified directly against
    # SentenceTransformer.get_embedding_dimension() during development).
    assert get_settings().embedding_model == "all-MiniLM-L6-v2"
    assert embedding_dimension() == 384


def test_embed_incident_summary_is_deterministic_and_correct_dimension():
    summary = IncidentSummary(
        service="checkout-service", symptoms=["db_connections_high"], timeline="t"
    )
    v1 = embed_incident_summary(summary)
    v2 = embed_incident_summary(summary)
    assert len(v1) == embedding_dimension()
    assert v1 == v2  # same input -> same vector, no hidden randomness


# --- qdrant_client.py --------------------------------------------------------


def test_ensure_collection_is_idempotent():
    client = get_qdrant_client()
    ensure_collection(client)
    assert client.collection_exists(COLLECTION_NAME)
    ensure_collection(client)  # second call must not raise
    assert client.collection_exists(COLLECTION_NAME)

    info = client.get_collection(COLLECTION_NAME)
    assert info.config.params.vectors.size == embedding_dimension()


def test_point_id_for_is_deterministic():
    assert point_id_for("hist-001") == point_id_for("hist-001")
    assert point_id_for("hist-001") != point_id_for("hist-002")


# --- seed script --------------------------------------------------------------


def test_seed_script_upserts_all_20_and_is_idempotent(seeded_client):
    client, count = seeded_client
    assert count == 20

    info = client.get_collection(COLLECTION_NAME)
    assert info.points_count == 20

    # Re-running must not create duplicates -- deterministic point ids
    # (point_id_for) mean a second seed upserts (overwrites) the same 20
    # points rather than appending 20 more.
    count2 = seed_historical_incidents()
    assert count2 == 20
    info2 = client.get_collection(COLLECTION_NAME)
    assert info2.points_count == 20


# --- search_historical_incidents tool -----------------------------------------


def test_search_returns_real_similarity_scores_and_correct_top_match(seeded_client):
    client, _count = seeded_client

    matches = search_historical_incidents(
        client,
        service="checkout-service",
        symptoms=[
            "the payment gateway is timing out and responding much slower than usual",
            "checkout is retrying aggressively against it",
            "db connections on checkout are climbing as a side effect",
            "checkout requests are ultimately failing",
        ],
        observed_dependencies=(
            "checkout-service calls an external payment gateway and retries on timeout"
        ),
        timeline=(
            "payment gateway latency spikes and times out; retries drive up db "
            "connections; checkout fails"
        ),
        top_k=5,
    )

    assert matches
    # This is the deliberately tricky case (mirrors cascading_payment_timeout):
    # the surface symptom is DB connection pressure, which could pull in the
    # database_connection_pool-category incidents (hist-001..004) as false
    # top matches. A high score on one of *those* would be a retrieval
    # failure even though it "looks similar" -- the correct top match is the
    # upstream-timeout incident whose own story is the same
    # loud-symptom-hides-the-real-cause pattern.
    assert matches[0].id == "hist-018"
    assert matches[0].root_cause_category == "upstream_dependency_timeout"
    # A real, non-degenerate similarity score -- not a placeholder like 1.0
    # or 0.0 for every result.
    assert 0.0 < matches[0].similarity <= 1.0
    assert len({match.similarity for match in matches}) > 1  # scores actually vary
    # Ranked best-first.
    assert all(
        matches[i].similarity >= matches[i + 1].similarity for i in range(len(matches) - 1)
    )


def test_search_rejects_empty_symptoms_and_blank_timeline(seeded_client):
    client, _count = seeded_client
    with pytest.raises(ValueError, match="symptoms"):
        search_historical_incidents(client, service="x", symptoms=[], timeline="t")
    with pytest.raises(ValueError, match="timeline"):
        search_historical_incidents(client, service="x", symptoms=["s"], timeline="   ")


def test_tool_binding_hides_client_and_invokes(seeded_client):
    client, _count = seeded_client
    tool = make_search_historical_incidents_tool(client)
    assert "client" not in tool.args
    assert set(tool.args) >= {"service", "symptoms", "timeline"}

    result = tool.invoke(
        {
            "service": "inventory-service",
            "symptoms": ["memory usage climbing for hours", "process was oom killed"],
            "timeline": "memory climbs for hours then the process is oom killed",
        }
    )
    assert result
    # Invoked through the LangChain binding -> plain JSON-serializable dicts
    # (see historical_incidents.make_search_historical_incidents_tool),
    # each carrying a real similarity score, not just text.
    assert all(
        isinstance(record, dict) and "similarity" in record and "id" in record for record in result
    )


def test_build_rag_tools_returns_search_historical_incidents_tool(seeded_client):
    from backend.tools import build_rag_tools

    client, _count = seeded_client
    tools = build_rag_tools(client)
    assert [t.name for t in tools] == ["search_historical_incidents"]


# --- investigator wiring (structural only -- no real Claude calls) -----------


class _FakeToolBoundLLM:
    """Stand-in for `ChatAnthropic.bind_tools(...)`'s return value. Its one
    `.invoke()` immediately ends the ReAct loop (returns an `AIMessage`
    with no `tool_calls`), so `investigate_incident` never actually
    dispatches a tool call -- no Qdrant search, no Claude API call, nothing
    but confirming what tools it *was* handed."""

    def __init__(self, tools):
        self.tools = tools

    def invoke(self, messages):  # noqa: ARG002 - signature-compatible stand-in
        return AIMessage(content="stub investigation summary, no tools called")


class _FakeStructuredLLM:
    def __init__(self, result):
        self._result = result

    def invoke(self, messages):  # noqa: ARG002 - signature-compatible stand-in
        return self._result


class _FakeChatAnthropic:
    """Stand-in for `ChatAnthropic` itself. `bind_tools` records the full
    tool list `investigate_incident` assembled (what this test actually
    asserts on); `with_structured_output` returns a canned `DiagnosisResult`
    so the function completes without any network call at all."""

    last_bound_tools: list | None = None

    def __init__(self, *args, **kwargs):
        pass

    def bind_tools(self, tools):
        _FakeChatAnthropic.last_bound_tools = tools
        return _FakeToolBoundLLM(tools)

    def with_structured_output(self, schema):  # noqa: ARG002
        from backend.agents.schemas import DiagnosisResult

        stub = DiagnosisResult(root_cause_category="unknown", diagnostic_confidence=0.0)
        return _FakeStructuredLLM(stub)


@pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Postgres not reachable at DATABASE_URL (start it with `docker compose up -d postgres`)",
)
def test_investigator_tool_list_includes_search_historical_incidents(monkeypatch):
    """Structural check only (per this task's cost constraint): confirms
    `search_historical_incidents` is actually bound into
    `investigate_incident`'s tool list alongside the Phase 2 tools, without
    running a live end-to-end investigation."""
    import backend.agents.investigator as investigator_module
    from backend.db import SessionLocal
    from backend.simulation.injector import inject_failure
    from backend.simulation.scenario_schema import load_all_scenarios

    monkeypatch.setattr(investigator_module, "ChatAnthropic", _FakeChatAnthropic)

    db = SessionLocal()
    try:
        scenario = load_all_scenarios()["db_connection_exhaustion"]
        incident = inject_failure(
            db, scenario, random.Random(123), datetime(2026, 7, 5, 12, 0, tzinfo=UTC)
        )

        result = investigator_module.investigate_incident(db, incident)

        bound_names = {t.name for t in _FakeChatAnthropic.last_bound_tools}
        assert bound_names == {
            "get_logs",
            "get_metrics",
            "get_deployments",
            "get_dependencies",
            "search_historical_incidents",
        }
        assert result.root_cause_category == "unknown"  # the stub result round-tripped
    finally:
        db.rollback()
        db.close()


@pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Postgres not reachable at DATABASE_URL (start it with `docker compose up -d postgres`)",
)
def test_investigator_include_rag_false_excludes_search_historical_incidents(monkeypatch):
    """Phase 7 needs `investigate_incident(..., include_rag=False)` to be a
    genuine Experiment B run (tools only, no RAG) -- distinct from the
    default `include_rag=True` behavior tested above, which is Experiment
    C's configuration (tools + RAG). Without this parameter, B and C would
    run the identical tool set and the A/B/C/D comparison would be
    meaningless."""
    import backend.agents.investigator as investigator_module
    from backend.db import SessionLocal
    from backend.simulation.injector import inject_failure
    from backend.simulation.scenario_schema import load_all_scenarios

    monkeypatch.setattr(investigator_module, "ChatAnthropic", _FakeChatAnthropic)

    db = SessionLocal()
    try:
        scenario = load_all_scenarios()["db_connection_exhaustion"]
        incident = inject_failure(
            db, scenario, random.Random(123), datetime(2026, 7, 5, 12, 0, tzinfo=UTC)
        )

        result = investigator_module.investigate_incident(db, incident, include_rag=False)

        bound_names = {t.name for t in _FakeChatAnthropic.last_bound_tools}
        assert bound_names == {"get_logs", "get_metrics", "get_deployments", "get_dependencies"}
        assert "search_historical_incidents" not in bound_names
        assert result.root_cause_category == "unknown"  # the stub result round-tripped
    finally:
        db.rollback()
        db.close()
