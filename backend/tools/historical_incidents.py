"""`search_historical_incidents` -- semantic search over Phase 4's Qdrant
`historical_incidents` collection.

Follows the exact plain-function + `make_*_tool(...)`-factory pattern
`backend/tools/logs.py` (and the rest of Phase 2's tool layer) already
establishes -- see `backend/tools/__init__.py`'s docstring for why a
closure factory is the right shape rather than `@tool` directly on a
function whose first parameter is a dependency the LLM can't fill in.

The one difference from the Postgres-backed tools: the bound dependency
here is a `QdrantClient` (`backend.rag.qdrant_client.get_qdrant_client()`),
not a per-request SQLAlchemy `Session` -- there is no request-scoped state
to isolate (Qdrant search is read-only and the client itself does no
connection pooling that needs per-request scoping the way a DB session
does), so `build_rag_tools()` below is a small parallel aggregator next to
`backend.tools.build_tools()` rather than folded into it: the two take
different dependency types and a caller (e.g. Phase 3's investigator)
composes both lists rather than one factory needing to accept either shape.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool
from qdrant_client import QdrantClient

from backend.rag.embeddings import embed_incident_summary
from backend.rag.qdrant_client import COLLECTION_NAME
from backend.rag.schemas import HistoricalIncidentMatch, IncidentSummary

DEFAULT_TOP_K = 5


def search_historical_incidents(
    client: QdrantClient,
    service: str,
    symptoms: list[str],
    recent_changes: str | None = None,
    observed_dependencies: str | None = None,
    timeline: str = "",
    top_k: int = DEFAULT_TOP_K,
) -> list[HistoricalIncidentMatch]:
    """Semantically search seeded historical incidents for the closest
    matches to a structured incident summary.

    Args:
        client: Qdrant client (not LLM-facing) -- see `make_search_historical_incidents_tool`.
        service: Primary affected service, e.g. "checkout-service".
        symptoms: Short evidence-tag-style symptom strings observed so far.
        recent_changes: Deployment/config-change context, if any.
        observed_dependencies: Downstream/upstream dependency involvement, if any.
        timeline: Short prose description of the temporal sequence of events.
        top_k: Maximum number of matches to return.

    Returns:
        Up to `top_k` `HistoricalIncidentMatch`es, best match first, each
        carrying a real cosine `similarity` score (0.0-1.0) -- never just
        retrieved text with no indication of how close the match actually
        is.

    Raises:
        ValueError: `symptoms` is empty, `timeline` is blank, or `top_k` is
            not a positive integer.
    """
    if not symptoms:
        raise ValueError("symptoms must be a non-empty list")
    if not timeline.strip():
        raise ValueError("timeline must not be blank")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k!r}")

    query_summary = IncidentSummary(
        service=service,
        symptoms=symptoms,
        recent_changes=recent_changes,
        observed_dependencies=observed_dependencies,
        timeline=timeline,
    )
    query_vector = embed_incident_summary(query_summary)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return [_match_from_point(point) for point in response.points]


def _match_from_point(point) -> HistoricalIncidentMatch:  # noqa: ANN001 - qdrant ScoredPoint
    payload = point.payload or {}
    return HistoricalIncidentMatch(
        id=payload["id"],
        root_cause_category=payload["root_cause_category"],
        # Cosine distance in Qdrant is configured (see backend.rag.qdrant_client
        # .ensure_collection) as Distance.COSINE, whose returned "score" from
        # query_points *is* cosine similarity in [-1, 1] -- clamped to
        # [0, 1] since HistoricalIncidentMatch.similarity is documented as a
        # 0-1 "percent similarity" and real incident-summary embeddings
        # never land meaningfully negative in practice.
        similarity=max(0.0, min(1.0, point.score)),
        service=payload["service"],
        symptoms=payload["symptoms"],
        recent_changes=payload.get("recent_changes"),
        observed_dependencies=payload.get("observed_dependencies"),
        timeline=payload["timeline"],
        narrative=payload["narrative"],
    )


def make_search_historical_incidents_tool(client: QdrantClient) -> BaseTool:
    """Bind `search_historical_incidents` to `client`, returning a LangChain
    tool with `client` hidden from the LLM-facing schema (only the
    structured-summary fields + `top_k` are exposed)."""

    @tool("search_historical_incidents", parse_docstring=True)
    def _search_historical_incidents_tool(
        service: str,
        symptoms: list[str],
        recent_changes: str | None = None,
        observed_dependencies: str | None = None,
        timeline: str = "",
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """Search past incident writeups for the closest historical matches
        to the incident you are currently investigating.

        Call this once you have gathered enough evidence to describe the
        incident's service, symptoms, any recent deployment/config change,
        any dependency involvement, and a short timeline -- a real
        historical match (with its similarity score and how it was
        resolved) is stronger evidence-backed context than guessing from
        first principles alone.

        Args:
            service: Primary affected service, e.g. "checkout-service".
            symptoms: Short evidence-tag-style symptom strings you have
                observed, e.g. ["db_connections_high", "checkout_failures"].
            recent_changes: Deployment/config-change context, if any recent
                change is implicated. Omit if none.
            observed_dependencies: Downstream/upstream dependency
                involvement, if any. Omit if none.
            timeline: Short description of the temporal sequence of events
                you have observed.
            top_k: Maximum number of historical matches to return (default 5).
        """
        # mode="json" for the same reason logs.make_get_logs_tool does it:
        # a list[BaseModel] would otherwise hit langchain_core's repr()
        # fallback serializer instead of clean JSON.
        return [
            match.model_dump(mode="json")
            for match in search_historical_incidents(
                client, service, symptoms, recent_changes, observed_dependencies, timeline, top_k
            )
        ]

    return _search_historical_incidents_tool


def build_rag_tools(client: QdrantClient) -> list[BaseTool]:
    """Return every RAG tool bound to one `QdrantClient`.

    Parallel to `backend.tools.build_tools(db)` -- see this module's
    docstring for why a Qdrant-backed tool set is a separate small
    aggregator rather than folded into that one. A caller wiring up an
    agent's full tool list (e.g. `backend.agents.investigator`) composes
    both: `build_tools(db) + build_rag_tools(get_qdrant_client())`.
    """
    return [make_search_historical_incidents_tool(client)]
