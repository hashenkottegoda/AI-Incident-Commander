"""Tests for `POST /api/simulation/failure` and `POST /api/simulation/reset`.

Postgres-dependent (both endpoints hit a live DB via `backend.db.get_db`),
so the whole module is skipped when Postgres isn't reachable -- same
skipif pattern as `tests/test_injector.py`/`tests/test_baseline.py`,
evaluated via `pytestmark` before `TestClient`/`backend.main` are even
imported.
"""

from __future__ import annotations

import psycopg
import pytest
from sqlalchemy import func, select

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

from backend.db import SessionLocal  # noqa: E402
from backend.main import app  # noqa: E402
from backend.models import Deployment, Incident, LogEntry, MetricPoint, TraceLite  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clean_simulation_tables():
    """Clean slate before and after each test, via the endpoint under test."""
    client.post("/api/simulation/reset")
    yield
    client.post("/api/simulation/reset")


def test_create_failure_returns_201_with_incident_body():
    response = client.post(
        "/api/simulation/failure", json={"failure_type": "db_connection_exhaustion", "seed": 123}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["failure_type"] == "db_connection_exhaustion"
    assert body["root_cause_category"] == "database_connection_pool"
    assert body["severity"] == "P1"
    assert body["status"] == "detected"
    assert isinstance(body["id"], int)
    assert body["detected_at"]


def test_create_failure_is_reproducible_for_a_fixed_seed():
    """Same failure_type + seed -> the same injected incident shape."""
    first = client.post(
        "/api/simulation/failure", json={"failure_type": "bad_deployment", "seed": 99}
    ).json()
    client.post("/api/simulation/reset")
    second = client.post(
        "/api/simulation/failure", json={"failure_type": "bad_deployment", "seed": 99}
    ).json()

    assert first["failure_type"] == second["failure_type"]
    assert first["root_cause_category"] == second["root_cause_category"]
    assert first["severity"] == second["severity"]


def test_create_failure_unknown_failure_type_returns_404():
    response = client.post("/api/simulation/failure", json={"failure_type": "not_a_real_scenario"})

    assert response.status_code == 404


def test_reset_clears_simulation_tables_but_keeps_services():
    inject_response = client.post(
        "/api/simulation/failure", json={"failure_type": "memory_leak", "seed": 5}
    )
    assert inject_response.status_code == 201

    db = SessionLocal()
    try:
        assert db.scalar(select(func.count()).select_from(Incident)) > 0
    finally:
        db.close()

    reset_response = client.post("/api/simulation/reset")
    assert reset_response.status_code == 200
    deleted = reset_response.json()["deleted"]
    assert deleted["incidents"] > 0

    db = SessionLocal()
    try:
        assert db.scalar(select(func.count()).select_from(Incident)) == 0
        assert db.scalar(select(func.count()).select_from(LogEntry)) == 0
        assert db.scalar(select(func.count()).select_from(MetricPoint)) == 0
        assert db.scalar(select(func.count()).select_from(TraceLite)) == 0
        assert db.scalar(select(func.count()).select_from(Deployment)) == 0
    finally:
        db.close()
