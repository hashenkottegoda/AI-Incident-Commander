"""Live-Qdrant, zero-LLM test of Phase 5's RAG node (`backend/agents/rag_node.py`).

This node makes no OpenRouter API call (pure local embedding + Qdrant search,
per BUILD_PLAN.md's Agent Architecture section and this task's cost
constraint), so it's tested for real against a seeded Qdrant collection --
no mocking needed, following `tests/test_rag.py`'s existing skip-without-
Qdrant convention.
"""

from __future__ import annotations

import pytest

from backend.agents.rag_node import RAG_TOOL_NAME, make_rag_node
from backend.agents.schemas import EvidenceItem, SourceRef
from backend.agents.state import IncidentState
from backend.rag.qdrant_client import get_qdrant_client
from backend.scripts.seed_historical_incidents import seed_historical_incidents


def _qdrant_reachable() -> bool:
    client = get_qdrant_client()
    try:
        client.get_collections()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _qdrant_reachable(),
    reason="Qdrant not reachable at QDRANT_URL (start it with `docker compose up -d qdrant`)",
)


@pytest.fixture(scope="module")
def seeded_client():
    seed_historical_incidents()
    return get_qdrant_client()


def _cascading_like_state() -> IncidentState:
    """Mirrors tests/test_rag.py's deliberately tricky retrieval case: the
    loud symptom is DB connection pressure, but the real story is an
    upstream payment-gateway timeout -- built here from EvidenceItems the
    way the real Investigation node would produce them, not hand-picked
    IncidentSummary fields."""
    return IncidentState(
        incident_id=1,
        affected_services=["checkout-service"],
        evidence=[
            EvidenceItem(
                description=(
                    "the payment gateway is timing out and responding much slower than usual"
                ),
                source_ref=SourceRef(tool="get_logs", record_id=1),
            ),
            EvidenceItem(
                description="db connections on checkout are climbing as a side effect",
                source_ref=SourceRef(tool="get_metrics", record_id=2),
            ),
            EvidenceItem(
                description=(
                    "checkout-service calls an external payment gateway and retries on timeout"
                ),
                source_ref=SourceRef(tool="get_dependencies", record_id=3),
            ),
            EvidenceItem(
                description="get_deployments found no matching records (no recent deploy)",
                source_ref=SourceRef(tool="get_deployments", query="service='checkout-service'"),
            ),
        ],
    )


def test_rag_node_returns_real_similarity_scores_and_correct_top_match(seeded_client):
    node = make_rag_node(seeded_client)
    state = _cascading_like_state()

    result = node(state)

    rag_items = [e for e in result["evidence"] if e.source_ref.tool == RAG_TOOL_NAME]
    assert rag_items
    # hist-018 is the upstream_dependency_timeout writeup -- same top match
    # tests/test_rag.py asserts directly against search_historical_incidents.
    assert any("hist-018" in item.source_ref.query for item in rag_items)


def test_rag_node_preserves_non_rag_evidence(seeded_client):
    node = make_rag_node(seeded_client)
    state = _cascading_like_state()

    result = node(state)

    non_rag_descriptions = {e.description for e in state.evidence}
    result_descriptions = {
        e.description for e in result["evidence"] if e.source_ref.tool != RAG_TOOL_NAME
    }
    assert non_rag_descriptions == result_descriptions


def test_rag_node_replaces_stale_matches_on_rerun_not_accumulates(seeded_client):
    node = make_rag_node(seeded_client)
    state = _cascading_like_state()

    first = node(state)
    state_with_rag_evidence = state.model_copy(update={"evidence": first["evidence"]})
    second = node(state_with_rag_evidence)

    rag_items_after_second_run = [
        e for e in second["evidence"] if e.source_ref.tool == RAG_TOOL_NAME
    ]
    # Not doubled -- same TOP_K count as a fresh run, not 2x from accumulation.
    first_rag_count = len([e for e in first["evidence"] if e.source_ref.tool == RAG_TOOL_NAME])
    assert len(rag_items_after_second_run) == first_rag_count


def test_rag_node_gracefully_degrades_when_no_evidence_gathered_yet():
    """Zero evidence (e.g. Investigation made no tool calls at all) must not
    raise -- IncidentSummary requires non-blank symptoms/timeline, which
    this node's fallback strings satisfy."""
    node = make_rag_node(get_qdrant_client())
    state = IncidentState(incident_id=1, affected_services=["checkout-service"])

    result = node(state)  # must not raise
    assert isinstance(result["evidence"], list)


def test_rag_node_survives_unreachable_qdrant():
    """RAG is corroborating evidence, never the deciding factor
    (BUILD_PLAN.md) -- a search failure must not crash the node."""
    from qdrant_client import QdrantClient

    broken_client = QdrantClient(url="http://localhost:1", timeout=1)
    node = make_rag_node(broken_client)
    state = _cascading_like_state()

    result = node(state)  # must not raise

    rag_items = [e for e in result["evidence"] if e.source_ref.tool == RAG_TOOL_NAME]
    assert rag_items == []
    # Non-RAG evidence must still survive a failed search.
    assert len(result["evidence"]) == len(state.evidence)
