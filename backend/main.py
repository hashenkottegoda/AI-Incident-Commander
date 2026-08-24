"""FastAPI application entrypoint.

Run with: `uv run uvicorn backend.main:app --reload`

Settings are read once at import time (via `get_settings()`) so the app
fails fast on a missing `ANTHROPIC_API_KEY` rather than at first use. No
DB/lifespan wiring yet — that lands with the SQLAlchemy models/session
layer in a later Phase 0 sub-step.
"""

from fastapi import FastAPI

from backend.api.approvals import router as approvals_router
from backend.api.health import router as health_router
from backend.api.incidents import router as incidents_router
from backend.api.simulation import router as simulation_router
from backend.config import get_settings

# Read + validate settings at startup (fail fast on missing required env vars).
get_settings()

app = FastAPI(title="AI Incident Commander")

app.include_router(health_router)
app.include_router(incidents_router)
app.include_router(approvals_router)
app.include_router(simulation_router)
