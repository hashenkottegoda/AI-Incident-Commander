"""Typed application settings, loaded once from the environment / .env file.

Later phases (FastAPI app, LangGraph agent nodes) should import `get_settings()`
rather than reading `os.environ` directly, so every env var is validated in one
place and model IDs stay out of code (BUILD_PLAN.md Phase 0 / Tech Stack).
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration sourced from environment variables / .env.

    See `.env.example` for a documented list of every variable and its
    dev-default value.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Ignore unrelated env vars (e.g. host machine exports) instead of
        # erroring on them.
        extra="ignore",
    )

    # --- Anthropic Claude ---------------------------------------------
    # Required, no default: fail fast at startup rather than at first API call.
    anthropic_api_key: str = Field(..., description="Anthropic Claude API key.")

    # Per-role model IDs — never hard-coded in agent code (BUILD_PLAN.md).
    # These defaults are BUILD_PLAN.md's current best guess; verify against
    # the current Anthropic model catalog before relying on them in prod.
    triage_model: str = Field(
        default="claude-haiku-4-5",
        description="Model ID for the cheap/fast Triage classification node.",
    )
    investigation_model: str = Field(
        default="claude-opus-4-8",
        description="Model ID for the reasoning-heavy Investigation node.",
    )
    rca_model: str = Field(
        default="claude-opus-4-8",
        description="Model ID for the reasoning-heavy Root Cause node.",
    )

    # --- Embeddings ------------------------------------------------------
    # MVP default is a local sentence-transformers model (no API key needed).
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model name used for local embeddings.",
    )

    # Optional production-grade embedding swap. Not required/used by the
    # MVP default embedding pipeline.
    voyage_api_key: str | None = Field(
        default=None,
        description="Optional Voyage AI API key for a production-grade embedding swap.",
    )

    # --- PostgreSQL --------------------------------------------------------
    database_url: str = Field(
        default=(
            "postgresql+psycopg://incident_commander:incident_commander"
            "@localhost:5432/incident_commander"
        ),
        description="SQLAlchemy-style Postgres connection URL.",
    )

    # --- Qdrant --------------------------------------------------------------
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant HTTP endpoint URL.",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached Settings instance.

    Cached so `.env` is parsed once per process; construct `Settings()`
    directly (bypassing the cache) in tests that need distinct env vars.
    """
    return Settings()
