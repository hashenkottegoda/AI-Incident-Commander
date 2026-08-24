"""Fast, free tests for `backend/api/incidents.py` that don't touch the LLM.

`POST /api/incidents/{id}/investigate` calls a real Claude API when given a
valid incident -- that's covered separately (and skipped by default) in
`tests/test_investigator.py`. This module covers everything about the route
that's checkable without spending API credits: the 404 path, so a
regression here (wrong status code, broken import, route not actually
mounted) is caught by the default fast suite instead of only surfacing the
next time someone runs the billed live test.

Postgres-dependent (the 404 path still does a real `db.get(Incident, ...)`
lookup) -- same skipif pattern as `tests/test_simulation_api.py`.
"""

from __future__ import annotations

import psycopg
import pytest

from backend.config import get_settings
from backend.scripts.setup_checkpointer import to_psycopg_dsn


def _postgres_reachable() -> bool:
    dsn = to_psycopg_dsn(get_settings().database_url)
    try:
        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Postgres not reachable at DATABASE_URL (start it with `docker compose up -d postgres`)",
)

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402

client = TestClient(app)


def test_investigate_unknown_incident_returns_404():
    response = client.post("/api/incidents/999999999/investigate")

    assert response.status_code == 404
    assert "999999999" in response.json()["detail"]
