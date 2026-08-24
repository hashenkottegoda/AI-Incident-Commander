"""Phase 5's RAG node: always runs (not left to the LLM's discretion),
makes ZERO LLM calls -- pure local embedding + Qdrant search.

BUILD_PLAN.md's Agent Architecture section: *"RAG (build a STRUCTURED
incident summary -> embed -> search_historical_incidents in Qdrant) ...
RAG query is a structured incident summary, not a raw evidence dump ...
assembled as fields (service, symptoms, recent_changes,
observed_dependencies, timeline) then embedded."*

Calls `backend.tools.historical_incidents.search_historical_incidents` --
the plain function, not the LangChain-tool-wrapped version -- because this
node always searches; there's no LLM decision about *whether* to search
for it to make, unlike Phase 3's baseline agent (which is handed
`search_historical_incidents` as a callable tool and decides for itself).

## Deriving `IncidentSummary` from `IncidentState.evidence`

`IncidentState.evidence[]` already carries a `source_ref.tool` tag per item
(see `backend.agents.investigation_node`), so the four `IncidentSummary`
fields are read straight off that tag rather than needing any extra
LLM-authored tagging step (which would cost money and contradict "ZERO LLM
calls"):

- `symptoms`      <- evidence tagged get_logs / get_metrics
- `recent_changes`      <- evidence tagged get_deployments
- `observed_dependencies` <- evidence tagged get_dependencies
- `timeline`      <- all of the above, joined in the order gathered

## Re-run behavior across re-investigation loops

RAG sits between INVESTIGATION and ROOT CAUSE in the fixed graph edges
(`backend/graph.py`), so it naturally re-runs on every loop iteration with
whatever evidence the latest Investigation pass added -- a better-informed
query on retry, for free. To avoid accumulating stale duplicate matches
across iterations, this node replaces (not appends to) any
`search_historical_incidents`-tagged evidence from a prior pass before
adding the current pass's matches.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient

from backend.agents.schemas import EvidenceItem, SourceRef
from backend.agents.state import IncidentState
from backend.rag.schemas import HistoricalIncidentMatch, IncidentSummary
from backend.tools.historical_incidents import search_historical_incidents

logger = logging.getLogger(__name__)

TOP_K = 3

RAG_TOOL_NAME = "search_historical_incidents"

_NO_SYMPTOMS_FALLBACK = "no significant symptoms identified yet"
_NO_TIMELINE_FALLBACK = "incident detected; investigation in progress, no evidence gathered yet"


def _evidence_by_tool(state: IncidentState, tool_name: str) -> list[str]:
    return [item.description for item in state.evidence if item.source_ref.tool == tool_name]


def _build_incident_summary(state: IncidentState) -> IncidentSummary:
    service = state.affected_services[0] if state.affected_services else "unknown-service"

    symptoms = _evidence_by_tool(state, "get_logs") + _evidence_by_tool(state, "get_metrics")
    recent_changes = "; ".join(_evidence_by_tool(state, "get_deployments")) or None
    observed_dependencies = "; ".join(_evidence_by_tool(state, "get_dependencies")) or None

    timeline_parts = [
        item.description for item in state.evidence if item.source_ref.tool != RAG_TOOL_NAME
    ]
    timeline = "; ".join(timeline_parts) or _NO_TIMELINE_FALLBACK

    return IncidentSummary(
        service=service,
        symptoms=symptoms or [_NO_SYMPTOMS_FALLBACK],
        recent_changes=recent_changes,
        observed_dependencies=observed_dependencies,
        timeline=timeline,
    )


def _evidence_from_match(match: HistoricalIncidentMatch) -> EvidenceItem:
    return EvidenceItem(
        description=(
            f"Historical incident {match.id} ({match.root_cause_category}) is "
            f"{match.similarity:.0%} similar: {match.narrative}"
        ),
        source_ref=SourceRef(tool=RAG_TOOL_NAME, record_id=None, query=match.id),
    )


def make_rag_node(client: QdrantClient):
    """Return a LangGraph node function bound to one `QdrantClient`.

    Factory pattern matches `backend.tools.build_rag_tools(client)` -- see
    that module's docstring for why a `QdrantClient`-bound factory is the
    right shape (no per-request session state to isolate, unlike the
    Postgres-backed nodes).
    """

    def rag_node(state: IncidentState) -> dict:
        summary = _build_incident_summary(state)
        try:
            matches = search_historical_incidents(
                client,
                summary.service,
                summary.symptoms,
                summary.recent_changes,
                summary.observed_dependencies,
                summary.timeline,
                top_k=TOP_K,
            )
        except Exception:
            # RAG is corroborating context, never the deciding factor
            # (BUILD_PLAN.md) -- an unreachable Qdrant or a not-yet-seeded
            # collection shouldn't block the rest of the investigation.
            logger.warning(
                "search_historical_incidents failed for incident %s; continuing without "
                "historical matches",
                state.incident_id,
                exc_info=True,
            )
            matches = []

        non_rag_evidence = [
            item for item in state.evidence if item.source_ref.tool != RAG_TOOL_NAME
        ]
        rag_evidence = [_evidence_from_match(match) for match in matches]
        return {"evidence": non_rag_evidence + rag_evidence}

    return rag_node
