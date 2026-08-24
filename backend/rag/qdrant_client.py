"""Qdrant client + idempotent collection setup for the historical-incident RAG store.

BUILD_PLAN.md Phase 4: *"Qdrant collection + local sentence-transformers
embedding pipeline."* Two things live here:

- `get_qdrant_client()`: a thin, cached constructor over
  `get_settings().qdrant_url` -- `QdrantClient(...)` does no network I/O on
  construction (it's a lazy REST/gRPC wrapper), so caching it is just
  avoiding redundant client objects, not avoiding a real connection cost.
- `ensure_collection()`: idempotent collection creation, sized to the
  *actual* embedding dimension of the configured `sentence-transformers`
  model (`backend.rag.embeddings.embedding_dimension()`) -- never a
  hard-coded constant, so a future `EMBEDDING_MODEL` swap to a
  different-dimension model can't silently create a mismatched collection.

Idempotency follows the same pattern as
`backend.simulation.baseline.get_or_create_canonical_services`: check
what already exists, only create what's missing, safe to call on every
app/script startup.
"""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from backend.config import get_settings
from backend.rag.embeddings import embedding_dimension

logger = logging.getLogger(__name__)

# The one collection this RAG layer uses. A module-level constant (not a
# Settings field) because BUILD_PLAN.md Phase 4 only ever calls for one
# collection -- a second collection would be a new config surface with no
# current use.
COLLECTION_NAME = "historical_incidents"

# Qdrant point IDs must be an unsigned integer or a UUID -- a plain string
# like "hist-001" (our human-readable `HistoricalIncident.id`) is rejected
# outright. `uuid.uuid5` deterministically derives the same UUID from the
# same string every time (no randomness, no state), so re-running the seed
# script upserts (overwrites) the same points instead of creating
# duplicates -- the point-id analogue of `get_or_create_canonical_services`'s
# idempotency. The human-readable id is still stored verbatim in the
# point's payload (`payload["id"]`) for display/matching.
_POINT_ID_NAMESPACE = uuid.UUID("5e3f6c1a-8b2d-4a3e-9c1f-2d7a6b4e8f01")


def point_id_for(historical_incident_id: str) -> str:
    """Deterministically map a human-readable `HistoricalIncident.id`
    (e.g. "hist-001") to the UUID Qdrant requires as a point id."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, historical_incident_id))


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Return the process-wide cached `QdrantClient`, pointed at
    `get_settings().qdrant_url`. Cached the same way as
    `backend.config.get_settings` -- constructing `QdrantClient` performs no
    network I/O itself, so this only avoids redundant client instances, not
    a real connection cost (the actual REST calls happen lazily per method
    call)."""
    # check_compatibility=False: docker-compose.yml pins the qdrant *server*
    # image to v1.12.1 for stability, independent of whatever qdrant-client
    # *python package* version pyproject.toml happens to resolve to (newer
    # client minors routinely lead the pinned server image). The client
    # only warns on a version-skew mismatch, it doesn't refuse to work --
    # silencing that warning here avoids noisy, actionable-nothing log spam
    # on every process start.
    return QdrantClient(url=get_settings().qdrant_url, check_compatibility=False)


def ensure_collection(client: QdrantClient | None = None, *, dimension: int | None = None) -> None:
    """Idempotently ensure the `historical_incidents` Qdrant collection
    exists with the correct vector size.

    Safe to call repeatedly -- e.g. once per seed-script run and once per
    app startup, mirroring `get_or_create_canonical_services`'s pattern for
    Postgres: check first, create only if missing, never error on a
    collection that's already there.

    `dimension` defaults to the configured embedding model's real output
    size (`backend.rag.embeddings.embedding_dimension()`, verified against
    the loaded model rather than assumed) -- overridable for tests that
    want a cheap fixed dimension without loading the real model.
    """
    client = client or get_qdrant_client()
    dimension = dimension if dimension is not None else embedding_dimension()

    if client.collection_exists(COLLECTION_NAME):
        logger.info("Qdrant collection %r already exists; leaving it as-is.", COLLECTION_NAME)
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qdrant_models.VectorParams(
            size=dimension,
            distance=qdrant_models.Distance.COSINE,
        ),
    )
    logger.info(
        "Created Qdrant collection %r (dimension=%d, distance=cosine).", COLLECTION_NAME, dimension
    )
