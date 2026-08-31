"""Seed Qdrant's `historical_incidents` collection from
`historical_incidents/historical_incidents.yaml`.

Follows `backend/scripts/setup_checkpointer.py`'s conventions: a one-off,
explicitly-invoked script (not run on app startup), idempotent, safe to
re-run.

    uv run python -m backend.scripts.seed_historical_incidents

Three steps, each already idempotent on its own so the whole script is:

1. `ensure_collection()` -- creates the collection if missing, no-ops if
   it already exists (`backend.rag.qdrant_client`).
2. Embed every `HistoricalIncident.summary()` via
   `backend.rag.embeddings.embed_texts` (batched, one forward pass for all
   ~20 incidents rather than one call per incident).
3. Upsert every point with a UUID derived deterministically from the
   incident's human-readable id (`backend.rag.qdrant_client.point_id_for`)
   -- upserting the same id twice overwrites the same point rather than
   creating a duplicate, so re-running this script after editing
   `historical_incidents.yaml` converges the collection to match the file
   instead of accumulating stale points.

Pure local computation: `sentence-transformers` runs on-device and Qdrant
is a local Docker container, so this script makes zero OpenRouter
API calls (Phase 4's cost constraint).
"""

from __future__ import annotations

import logging

from qdrant_client.http import models as qdrant_models

from backend.rag.embeddings import assemble_incident_text, embed_texts
from backend.rag.historical_incidents import load_historical_incidents
from backend.rag.qdrant_client import (
    COLLECTION_NAME,
    ensure_collection,
    get_qdrant_client,
    point_id_for,
)

logger = logging.getLogger(__name__)


def seed_historical_incidents() -> int:
    """Ensure the collection exists and upsert every seeded historical
    incident. Returns the number of incidents upserted."""
    client = get_qdrant_client()
    ensure_collection(client)

    incidents = load_historical_incidents()
    texts = [assemble_incident_text(incident.summary()) for incident in incidents]
    vectors = embed_texts(texts)

    points = [
        qdrant_models.PointStruct(
            id=point_id_for(incident.id),
            vector=vector,
            payload=incident.model_dump(mode="json"),
        )
        for incident, vector in zip(incidents, vectors, strict=True)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info("Seeded %d historical incidents into %r.", len(points), COLLECTION_NAME)
    return len(points)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    count = seed_historical_incidents()
    print(f"Seeded {count} historical incidents into Qdrant collection {COLLECTION_NAME!r}.")


if __name__ == "__main__":
    main()
