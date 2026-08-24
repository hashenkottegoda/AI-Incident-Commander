"""Recall@K measurement -- BUILD_PLAN.md Phase 4's actual acceptance bar:

*"measure Recall@K -- for a set of query incidents with a known correct
historical match, what fraction have the correct incident in the Top-K
results (report Recall@1/@3/@5). This is a real retrieval metric; 'high
similarity' is not sufficient because a high score on the wrong incident
is still a retrieval failure."*

Runs the real ground-truth query set (`backend.rag.recall_eval
.RECALL_EVAL_QUERIES`) against the real seeded Qdrant collection through
the real `search_historical_incidents` tool function, computes
Recall@1/@3/@5, and prints the numbers (`-s` to see them) -- no
placeholders, no mocked retrieval. Pure local `sentence-transformers` +
Qdrant math, zero Claude/Anthropic calls (Phase 4's cost constraint).

Skips cleanly if Qdrant isn't reachable, same pattern as `tests/test_rag.py`.
"""

from __future__ import annotations

import pytest

from backend.rag.qdrant_client import get_qdrant_client
from backend.rag.recall_eval import RECALL_EVAL_QUERIES, compute_recall_at_k
from backend.scripts.seed_historical_incidents import seed_historical_incidents
from backend.tools.historical_incidents import search_historical_incidents


def _qdrant_reachable() -> bool:
    try:
        get_qdrant_client().get_collections()
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


def test_recall_at_k_against_seeded_historical_incidents(seeded_client):
    """The actual Phase 4 measurement: run every ground-truth query, rank
    the returned historical incident ids, and compute Recall@1/@3/@5.

    Assertions are floors informed by a real run of this exact eval during
    development (Recall@1=0.8, Recall@3=1.0, Recall@5=1.0 over the 10
    ground-truth queries -- see this task's report for the full per-query
    ranking), not aspirational numbers picked before measuring. A small
    buffer below the observed values tolerates minor floating-point/library
    version drift without masking an actual retrieval regression.
    """
    ranked_ids_by_query: dict[str, list[str]] = {}
    for query in RECALL_EVAL_QUERIES:
        matches = search_historical_incidents(
            seeded_client,
            service=query.summary.service,
            symptoms=query.summary.symptoms,
            recent_changes=query.summary.recent_changes,
            observed_dependencies=query.summary.observed_dependencies,
            timeline=query.summary.timeline,
            top_k=5,
        )
        ranked_ids_by_query[query.query_id] = [match.id for match in matches]

    expected_ids_by_query = {q.query_id: q.expected_historical_id for q in RECALL_EVAL_QUERIES}
    result = compute_recall_at_k(ranked_ids_by_query, expected_ids_by_query)

    print(f"\nRecall@K over {result.num_queries} ground-truth queries:")
    print(f"  Recall@1 = {result.recall_at_1:.2f}")
    print(f"  Recall@3 = {result.recall_at_3:.2f}")
    print(f"  Recall@5 = {result.recall_at_5:.2f}")
    for query_id, rank in result.per_query_ranks.items():
        expected = expected_ids_by_query[query_id]
        print(f"    {query_id}: expected={expected} rank={rank}")

    assert result.num_queries == 10
    assert result.recall_at_1 >= 0.6
    assert result.recall_at_3 >= 0.9
    assert result.recall_at_5 >= 0.9


def test_recall_computation_is_pure_and_unit_testable():
    """`compute_recall_at_k` itself needs no Qdrant/embeddings -- exercised
    directly against hand-constructed rankings so the recall *math* has its
    own fast, infrastructure-free coverage."""
    ranked = {
        "q1": ["a", "b", "c"],  # correct id "a" at rank 1
        "q2": ["x", "b", "y"],  # correct id "b" at rank 2
        "q3": ["x", "y", "z"],  # correct id "c" not present at all
        "q4": ["c", "a", "b"],  # correct id "b" at rank 3
    }
    expected = {"q1": "a", "q2": "b", "q3": "c", "q4": "b"}

    result = compute_recall_at_k(ranked, expected)

    assert result.num_queries == 4
    assert result.recall_at_1 == 1 / 4  # only q1
    assert result.recall_at_3 == 3 / 4  # q1, q2, q4 (q3 never found)
    assert result.recall_at_5 == 3 / 4  # search never returned more than 3 anyway
    assert result.per_query_ranks == {"q1": 1, "q2": 2, "q3": None, "q4": 3}


def test_compute_recall_at_k_rejects_mismatched_query_sets():
    with pytest.raises(ValueError, match="same queries"):
        compute_recall_at_k({"q1": ["a"]}, {"q2": "a"})
