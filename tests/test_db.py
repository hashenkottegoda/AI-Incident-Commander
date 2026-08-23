"""DB-connectivity tests: skipped when Postgres isn't reachable.

This module is infra (not app logic), so unlike the rest of the default
`uv run pytest` suite it needs a live Postgres. To keep the default,
no-services run fast and green, every test in this module is gated behind
a single `pytest.mark.skipif` that probes the configured `database_url`
with a short timeout at collection time. Bring `postgres` up
(`docker compose up -d postgres`) to actually exercise these.
"""

import psycopg
import pytest
from sqlalchemy import text

from backend.config import get_settings
from backend.db import SessionLocal, engine
from backend.scripts.setup_checkpointer import setup_checkpointer, to_psycopg_dsn


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


def test_engine_executes_query():
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar_one() == 1


def test_session_local_executes_query():
    db = SessionLocal()
    try:
        assert db.execute(text("SELECT 1")).scalar_one() == 1
    finally:
        db.close()


async def test_checkpointer_setup_creates_tables():
    await setup_checkpointer()

    expected_tables = {
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
        "checkpoint_migrations",
    }
    with engine.connect() as conn:
        found = set(
            conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = ANY(:names)"
                ),
                {"names": list(expected_tables)},
            ).scalars()
        )

    assert found == expected_tables
