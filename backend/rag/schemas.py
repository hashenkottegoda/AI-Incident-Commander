"""Typed shapes shared by the RAG layer's embedding pipeline and search tool.

`IncidentSummary` is the one structured shape BUILD_PLAN.md's Agent
Architecture section requires on both sides of retrieval: *"the retrieval
query is assembled as fields (service, symptoms, recent_changes,
observed_dependencies, timeline) then embedded."* `backend.rag.embeddings`
assembles the *exact same* field set for a live query and for every seeded
historical incident (`backend.rag.historical_incidents.HistoricalIncident`
embeds one of these), so a similarity score is always comparing
like-for-like structured summaries -- never a structured query against a
raw prose blob. This is what makes "why was incident #142 considered
similar?" explainable: you can point at the two summaries' fields, not
just a cosine number.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class IncidentSummary(BaseModel):
    """A structured incident summary -- the RAG query shape.

    Same field set is used for a live investigation's query and for a
    seeded historical incident's write-up. Field order/naming matches
    BUILD_PLAN.md's Agent Architecture section verbatim.
    """

    model_config = ConfigDict(frozen=True)

    service: str = Field(description="Primary affected service, e.g. 'checkout-service'.")
    symptoms: list[str] = Field(
        min_length=1,
        description=(
            "Short evidence-tag-style symptom strings, e.g. "
            "['db_connections_high', 'connection_pool_exhausted']."
        ),
    )
    recent_changes: str | None = Field(
        default=None,
        description="Deployment/config-change context, if any recent change is implicated.",
    )
    observed_dependencies: str | None = Field(
        default=None,
        description="Downstream/upstream dependency involvement, if any.",
    )
    timeline: str = Field(description="Short prose description of the temporal sequence of events.")


class HistoricalIncidentMatch(BaseModel):
    """One `search_historical_incidents` result: a historical incident plus
    its similarity score against the query.

    Carries a real similarity score (not just retrieved text) so the
    investigation agent can cite it as "87% semantic similarity to
    incident hist-004" -- evidence-backed rather than vibes-based, per
    this project's RAG conventions.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    root_cause_category: str
    similarity: float = Field(
        ge=0.0, le=1.0, description="Cosine similarity to the query, 0.0-1.0."
    )
    service: str
    symptoms: list[str]
    recent_changes: str | None = None
    observed_dependencies: str | None = None
    timeline: str
    narrative: str = Field(
        description="Human-readable resolution/root-cause note (not part of the embedded text)."
    )
