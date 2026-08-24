"""Embedding pipeline: structured incident summary -> deterministic text -> vector.

BUILD_PLAN.md's Agent Architecture section: *"the retrieval query is
assembled as fields (service, symptoms, recent_changes,
observed_dependencies, timeline) then embedded -- cleaner retrieval, and
it makes 'why was incident #142 considered similar?' explainable rather
than a black-box cosine score."*

## Assembly format (exact, deterministic)

`assemble_incident_text(summary)` produces exactly:

    Service: {service}
    Symptoms: {symptom_1}; {symptom_2}; ...
    Recent changes: {recent_changes or "none"}
    Observed dependencies: {observed_dependencies or "none"}
    Timeline: {timeline}

One line per field in a fixed order, `symptoms` joined with `"; "`,
`recent_changes`/`observed_dependencies` rendered literally as the string
`"none"` when absent (never an empty/missing line -- keeps every embedded
document the same shape whether or not those optional fields are set).
This function is pure and has no model dependency, so "why was X
considered similar to Y" can always be answered by diffing two calls to
it against the query and the match -- exactly the explainability
BUILD_PLAN.md asks for.

## Model loading

`get_embedding_model()` is `functools.lru_cache`d: constructing a
`SentenceTransformer` loads weights from disk/the HF cache, expensive
enough (hundreds of ms to seconds) that repeating it per call would make
every `embed_text()` call slow and every seed/search operation pay that
cost redundantly. Cached once per process -- the same pattern as
`backend.config.get_settings`.

## Swappable provider (Voyage AI / sentence-transformers)

Only `get_embedding_model()` and `embed_texts()` know about
`sentence-transformers` specifically. A future Voyage AI swap
(BUILD_PLAN.md Tech Stack: "Voyage AI documented as an optional
production-grade swap") only has to change this module's internals --
`assemble_incident_text()` and every caller (the seed script, the
`search_historical_incidents` tool) depend only on
`embed_incident_summary(summary) -> list[float]`, not on which library
computed the vector. Swapping providers is therefore a config change
(`EMBEDDING_MODEL` / a future `EMBEDDING_PROVIDER`), not a code fork.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from backend.config import get_settings
from backend.rag.schemas import IncidentSummary


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    """Load (once per process) the `sentence-transformers` model named by
    `get_settings().embedding_model`. Cached so repeated calls -- across
    the seed script, the search tool, and tests within one process -- never
    reload weights."""
    return SentenceTransformer(get_settings().embedding_model)


def embedding_dimension() -> int:
    """The configured model's real output vector size.

    Used to size the Qdrant collection (`backend.rag.qdrant_client`) --
    verified against the actual loaded model rather than hard-coded, so a
    future `EMBEDDING_MODEL` swap to a different-dimension model can't
    silently create a collection with the wrong vector size.
    """
    return get_embedding_model().get_embedding_dimension()


def assemble_incident_text(summary: IncidentSummary) -> str:
    """Deterministically render a structured `IncidentSummary` into the
    exact text that gets embedded. See this module's docstring for the
    fixed format."""
    return (
        f"Service: {summary.service}\n"
        f"Symptoms: {'; '.join(summary.symptoms)}\n"
        f"Recent changes: {summary.recent_changes or 'none'}\n"
        f"Observed dependencies: {summary.observed_dependencies or 'none'}\n"
        f"Timeline: {summary.timeline}"
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of already-assembled strings in one forward pass.

    Batched rather than one call per text so the seed script embeds all
    ~20 historical incidents (and a caller can embed many queries) without
    paying per-call model-invocation overhead N times.
    """
    if not texts:
        return []
    vectors = get_embedding_model().encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [vector.tolist() for vector in vectors]


def embed_incident_summary(summary: IncidentSummary) -> list[float]:
    """Assemble + embed one structured incident summary.

    The single function both the seed script and the live
    `search_historical_incidents` tool call, so both sides of retrieval
    always go through the identical assembly format above -- a query and a
    seeded historical incident are never embedded via different code paths.
    """
    return embed_texts([assemble_incident_text(summary)])[0]
