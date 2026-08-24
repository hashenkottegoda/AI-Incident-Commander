"""Phase 4's RAG layer: Qdrant collection, local `sentence-transformers`
embedding pipeline, and the ~20 hand-authored historical incident writeups
used as retrieval seed data.

BUILD_PLAN.md Phase 4: *"Qdrant collection + local sentence-transformers
embedding pipeline. Hand-author ~20 historical incident writeups, embed
and seed them. Add search_historical_incidents tool to the Phase 3
agent."*

## Module map

- `schemas.py` -- `IncidentSummary` (the structured retrieval-query shape,
  BUILD_PLAN.md's Agent Architecture section: "service, symptoms,
  recent_changes, observed_dependencies, timeline") and
  `HistoricalIncidentMatch` (one search result, similarity score included).
- `embeddings.py` -- deterministic structured-summary -> text assembly,
  plus the `sentence-transformers` model wrapper (loaded once, cached).
  This is the one module a future Voyage AI swap would touch.
- `qdrant_client.py` -- `get_qdrant_client()` / `ensure_collection()`:
  idempotent collection setup sized to the configured embedding model's
  real output dimension (never hard-coded).
- `historical_incidents.py` -- the `HistoricalIncident` schema (a
  structured summary plus `id`/`root_cause_category`/`narrative`) and a
  loader for `historical_incidents/historical_incidents.yaml`.
- `recall_eval.py` -- the ground-truth query set + Recall@K computation
  that is Phase 4's actual acceptance bar (not "high similarity").

The `search_historical_incidents` LangChain tool itself lives in
`backend.tools.historical_incidents`, alongside the rest of Phase 2's
tool layer, following that package's plain-function +
`make_*_tool(...)`-factory convention -- not in this package, so
`backend.tools` stays the single place an agent node looks for its tool
set.
"""

from __future__ import annotations
