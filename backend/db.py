"""SQLAlchemy engine/session scaffolding (Phase 0 infra only).

This module owns *only* the engine, session factory, and declarative base —
no model classes live here. Phase 1 introduces `backend/models/` with
tables (services, deployments, logs, metrics, incidents, ...) that
subclass `Base`; Alembic (`alembic/env.py`) already points its
`target_metadata` at `Base.metadata` so those models are autogenerate-ready
the moment they exist.

`engine` uses a lazy connection pool (SQLAlchemy's default) — importing
this module does not require a reachable Postgres instance, only a valid
`database_url` in settings.
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import get_settings

engine = create_engine(get_settings().database_url, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    """Declarative base for all SQLAlchemy models (subclassed starting Phase 1)."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yield a request-scoped session, always closed after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
