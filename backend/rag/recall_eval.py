"""Recall@K measurement -- Phase 4's actual acceptance bar.

BUILD_PLAN.md Phase 4: *"measure Recall@K -- for a set of query incidents
with a known correct historical match, what fraction have the correct
incident in the Top-K results (report Recall@1/@3/@5). This is a real
retrieval metric; 'high similarity' is not sufficient because a high score
on the wrong incident is still a retrieval failure."*

## Ground truth construction

`RECALL_EVAL_QUERIES` below pairs 10 independently-phrased query
`IncidentSummary`s with the specific `historical_incidents.yaml` id that
SHOULD be the top match. Two sources, both legitimate per BUILD_PLAN.md's
own suggested approaches:

- 6 queries (`query-scenario-*`) are paraphrases of each of the 6
  `failure_scenarios/*.yaml` files' typical presentation (their
  `expected_evidence`/`causal_chain`), each paired with one
  `historical_incidents.yaml` entry deliberately authored to match that
  scenario's story (different service names/specific wording -- a genuine
  paraphrase, not a copy of either the scenario YAML or the historical
  writeup's own text).
- 4 more queries (`query-extra-*`) are independent paraphrases of 4
  additional historical incidents not covered by the 6 scenarios, so more
  of the ~20 seeded incidents participate in the measurement and the
  denominator isn't limited to exactly 6.

Every query is deliberately worded differently from its target's own
`symptoms`/`timeline` text -- copy-pasting the target's fields as the
"query" would make retrieval trivially perfect by construction and defeat
the point of measuring anything. At the same time, several of the 20
seeded incidents share a `root_cause_category` with genuinely different
specifics (see `historical_incidents.yaml`'s category-grouped comments),
so a query's nearest neighbors are real distractors, not padding --
Recall@1 is not guaranteed to be 1.0 by construction.

## What this module does NOT do

No OpenRouter calls anywhere in this file or its test -- Recall@K
is pure local-embedding retrieval math, per this phase's cost constraint.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.rag.schemas import IncidentSummary


@dataclass(frozen=True)
class RecallQuery:
    """One ground-truth (query, correct historical incident id) pair."""

    query_id: str
    summary: IncidentSummary
    expected_historical_id: str


RECALL_EVAL_QUERIES: tuple[RecallQuery, ...] = (
    # --- derived from the 6 failure_scenarios/*.yaml (one per scenario) ---
    RecallQuery(
        query_id="query-scenario-db_connection_exhaustion",
        summary=IncidentSummary(
            service="checkout-service",
            symptoms=[
                "db connection pool fully saturated",
                "requests queuing waiting for a free db connection",
                "checkout requests starting to fail",
                "error rate climbing",
            ],
            recent_changes=(
                "a new checkout release rolled out roughly 15 minutes before symptoms "
                "started, touching how database connections are managed"
            ),
            observed_dependencies=None,
            timeline=(
                "shortly after a deployment, active db connections climb steadily; "
                "within about ten minutes the pool is exhausted and checkout starts "
                "throwing errors"
            ),
        ),
        expected_historical_id="hist-001",
    ),
    RecallQuery(
        query_id="query-scenario-memory_leak",
        summary=IncidentSummary(
            service="inventory-service",
            symptoms=[
                "memory usage climbing steadily for hours",
                "garbage collection pauses getting longer",
                "service was oom killed",
                "failure rate rising",
                "no recent deployment",
            ],
            recent_changes=None,
            observed_dependencies=None,
            timeline=(
                "over several hours, memory usage on the affected service climbs "
                "without a clear trigger, gc pauses lengthen in step, and the process "
                "is eventually killed for using too much memory"
            ),
        ),
        expected_historical_id="hist-005",
    ),
    RecallQuery(
        query_id="query-scenario-bad_deployment",
        summary=IncidentSummary(
            service="payment-service",
            symptoms=[
                "error rate spiked right after a deploy",
                "large burst of http 500s",
                "failures start within minutes of the release",
                "same stack trace on every failing request",
            ],
            recent_changes=(
                "payment-service released a new version a few minutes before the "
                "error spike began"
            ),
            observed_dependencies=None,
            timeline=(
                "errors begin almost immediately after deployment and climb quickly; "
                "failures share the same underlying exception in newly-changed code"
            ),
        ),
        expected_historical_id="hist-008",
    ),
    RecallQuery(
        query_id="query-scenario-dependency_failure",
        summary=IncidentSummary(
            service="checkout-service",
            symptoms=[
                "a newly enabled canary flag routes some traffic to a new payment provider",
                "that provider starts returning error responses, not timeouts",
                "checkout calls into payment are failing with dependency errors",
                "checkout requests failing overall",
            ],
            recent_changes=None,
            observed_dependencies=(
                "checkout-service depends on payment-service, which recently enabled "
                "a canary path to a new external provider"
            ),
            timeline=(
                "shortly after the canary flag is enabled, the new payment provider "
                "begins returning outright error responses; checkout, which depends on "
                "payment, starts failing requests as a result"
            ),
        ),
        expected_historical_id="hist-012",
    ),
    RecallQuery(
        query_id="query-scenario-slow_query",
        summary=IncidentSummary(
            service="inventory-service",
            symptoms=[
                "a specific query is detected as slow",
                "query latency for that endpoint keeps climbing",
                "database cpu usage running high",
                "requests eventually timing out",
                "queue of connections waiting to run queries growing",
            ],
            recent_changes=None,
            observed_dependencies=None,
            timeline=(
                "over time as data volume grows, a particular lookup query gets "
                "progressively slower; database cpu stays elevated and, once volume "
                "crosses a threshold, requests start timing out"
            ),
        ),
        expected_historical_id="hist-015",
    ),
    RecallQuery(
        query_id="query-scenario-cascading_payment_timeout",
        summary=IncidentSummary(
            service="checkout-service",
            symptoms=[
                "the external payment gateway is responding much slower than usual and timing out",
                "checkout is retrying aggressively against the slow gateway",
                "database connections on checkout are climbing as a side effect of the retries",
                "checkout requests ultimately failing",
            ],
            recent_changes=None,
            observed_dependencies=(
                "checkout-service calls an external payment gateway synchronously "
                "and retries on timeout"
            ),
            timeline=(
                "the payment gateway's latency rises sharply and starts timing out; "
                "checkout's retry logic amplifies call volume against it; db "
                "connection usage on checkout climbs as a downstream effect of the "
                "retry storm before checkout requests start failing outright"
            ),
        ),
        expected_historical_id="hist-018",
    ),
    # --- extra queries covering more of the seeded set -------------------
    RecallQuery(
        query_id="query-extra-long-running-transaction",
        summary=IncidentSummary(
            service="catalog-service",
            symptoms=[
                "a nightly bulk import job is holding a database transaction open far "
                "longer than usual",
                "connection pool showing saturation while the job runs",
                "other services sharing the database experiencing lock wait delays",
                "checkout latency spiking during the import window",
            ],
            recent_changes=None,
            observed_dependencies=None,
            timeline=(
                "a scheduled bulk data import begins holding a single long transaction "
                "that used to complete in seconds; while it runs, pool saturation and "
                "lock waits appear across services sharing the database"
            ),
        ),
        expected_historical_id="hist-003",
    ),
    RecallQuery(
        query_id="query-extra-silent-empty-response",
        summary=IncidentSummary(
            service="recommendation-service",
            symptoms=[
                "a new release changed how the ranking model is serialized",
                "some responses are coming back with empty recommendation lists instead of errors",
                "cpu, memory, and database metrics all look completely normal",
                "the empty responses started right after the deploy",
            ],
            recent_changes=(
                "a deployment migrated the model serialization format shortly before "
                "the empty responses began"
            ),
            observed_dependencies=None,
            timeline=(
                "immediately following a release, a subset of requests silently "
                "return empty payloads instead of failing loudly, while every "
                "resource metric remains flat at baseline the whole time"
            ),
        ),
        expected_historical_id="hist-010",
    ),
    RecallQuery(
        query_id="query-extra-carrier-outage",
        summary=IncidentSummary(
            service="fulfillment-service",
            symptoms=[
                "the external shipping carrier api is returning error responses for "
                "label creation",
                "shipment creation is failing at the same rate as the carrier errors",
                "no deployment happened on our side",
                "the carrier's own status page later confirms an outage",
            ],
            recent_changes=None,
            observed_dependencies=(
                "fulfillment-service depends on an external carrier api to create "
                "shipping labels"
            ),
            timeline=(
                "the external carrier begins failing label-creation calls with no "
                "corresponding change on our side; shipment failures track the "
                "carrier's error rate one for one until the carrier's own outage is "
                "confirmed and resolved"
            ),
        ),
        expected_historical_id="hist-014",
    ),
    RecallQuery(
        query_id="query-extra-downstream-slow-query-timeout",
        summary=IncidentSummary(
            service="search-service",
            symptoms=[
                "calls to the inventory lookup are timing out",
                "inventory's own latency is climbing over about half an hour",
                "search results are coming back with stale or missing stock counts",
                "overall search latency rising",
                "no deployment on our side",
            ],
            recent_changes=None,
            observed_dependencies=(
                "search-service calls inventory-service synchronously to enrich "
                "results with stock counts"
            ),
            timeline=(
                "inventory-service's latency climbs gradually over roughly thirty "
                "minutes due to a problem on its own side; search's synchronous calls "
                "to it begin timing out, degrading search result quality and overall "
                "latency"
            ),
        ),
        expected_historical_id="hist-020",
    ),
)


@dataclass(frozen=True)
class RecallResult:
    """Recall@K for K in {1, 3, 5} over `RECALL_EVAL_QUERIES` (or any
    caller-supplied query set), plus the raw per-query hits for inspection."""

    num_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    per_query_ranks: dict[str, int | None]
    """query_id -> 1-indexed rank of the correct id in the results, or
    None if it didn't appear at all (within whatever K the search was run
    with)."""


def compute_recall_at_k(
    ranked_ids_by_query: dict[str, list[str]],
    expected_ids_by_query: dict[str, str],
) -> RecallResult:
    """Pure recall computation: given, per query id, the ranked list of
    returned historical incident ids (best match first) and the expected
    correct id, compute Recall@1/@3/@5.

    Kept separate from the actual Qdrant search call so the recall math
    itself is unit-testable with hand-constructed rankings, independent of
    a live Qdrant instance.
    """
    if set(ranked_ids_by_query) != set(expected_ids_by_query):
        raise ValueError(
            "ranked_ids_by_query and expected_ids_by_query must cover the same queries"
        )

    per_query_ranks: dict[str, int | None] = {}
    for query_id, ranked_ids in ranked_ids_by_query.items():
        expected_id = expected_ids_by_query[query_id]
        try:
            rank = ranked_ids.index(expected_id) + 1  # 1-indexed
        except ValueError:
            rank = None
        per_query_ranks[query_id] = rank

    n = len(per_query_ranks)

    def _recall_at(k: int) -> float:
        hits = sum(1 for rank in per_query_ranks.values() if rank is not None and rank <= k)
        return hits / n if n else 0.0

    return RecallResult(
        num_queries=n,
        recall_at_1=_recall_at(1),
        recall_at_3=_recall_at(3),
        recall_at_5=_recall_at(5),
        per_query_ranks=per_query_ranks,
    )
