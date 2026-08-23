"""Provision LangGraph's Postgres checkpoint tables.

BUILD_PLAN.md: "LangGraph checkpoints live in the same [Postgres] instance"
as the application data. `AsyncPostgresSaver.setup()` creates/upgrades the
checkpointer's own tables (`checkpoints`, `checkpoint_blobs`,
`checkpoint_writes`, `checkpoint_migrations`) — schema that belongs to the
`langgraph-checkpoint-postgres` library, not to any SQLAlchemy model in
`backend/db.py`, so it is intentionally provisioned separately from
Alembic rather than folded into a migration.

**Why a one-off script, not app startup:** `setup()` is idempotent (it
tracks its own applied-migration version in a `checkpoint_migrations`
table, the same pattern Alembic uses for app schema) but it still does a
DDL-touching round trip on every call. Running that implicitly on every
FastAPI process boot would mean every uvicorn worker/replica racing to run
schema DDL on startup — the same reason `alembic upgrade head` isn't
wired into app startup either. Schema provisioning is an explicit,
one-time-per-environment ops step (run manually here in dev; a deploy
step / CI job in a real environment), invoked as:

    uv run python -m backend.scripts.setup_checkpointer

**Why `AsyncPostgresSaver`, not the sync `PostgresSaver`:** the FastAPI
app is async (BUILD_PLAN.md's graph is invoked from async API routes in
later phases), so the actual checkpointer used at runtime will be
`AsyncPostgresSaver`. Using the same class here — not just for this
setup script — means the connection/DSN handling this module exercises is
the same code path the LangGraph graph will use, not a separate sync
variant that happens to also work for DDL.
"""

import asyncio
import logging

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from backend.config import get_settings

logger = logging.getLogger(__name__)


def to_psycopg_dsn(database_url: str) -> str:
    """Convert a SQLAlchemy-style URL to a plain libpq/psycopg DSN.

    `backend.config.Settings.database_url` uses SQLAlchemy's
    `dialect+driver://` form (`postgresql+psycopg://...`) so SQLAlchemy
    picks the right DBAPI. psycopg's own `Connection.connect()` (which
    `AsyncPostgresSaver.from_conn_string` calls) only understands the
    plain `postgresql://` scheme, so the `+psycopg` driver suffix has to
    be stripped before handing the URL to LangGraph.
    """
    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)


async def setup_checkpointer(database_url: str | None = None) -> None:
    """Create (or upgrade) LangGraph's Postgres checkpoint tables.

    Safe to call more than once — `AsyncPostgresSaver.setup()` tracks its
    own migration version and no-ops once up to date.
    """
    dsn = to_psycopg_dsn(database_url or get_settings().database_url)
    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        await saver.setup()
    logger.info("LangGraph checkpoint tables are up to date.")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(setup_checkpointer())


if __name__ == "__main__":
    main()
