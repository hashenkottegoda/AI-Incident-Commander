"""FastAPI application entrypoint.

Run with: `uv run uvicorn backend.main:app --reload`

Settings are read once at import time (via `get_settings()`) so the app
fails fast on a missing `ANTHROPIC_API_KEY` rather than at first use. No
DB/lifespan wiring yet — that lands with the SQLAlchemy models/session
layer in a later Phase 0 sub-step.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.approvals import router as approvals_router
from backend.api.evaluation import router as evaluation_router
from backend.api.health import router as health_router
from backend.api.incidents import router as incidents_router
from backend.api.simulation import router as simulation_router
from backend.config import get_settings

# Read + validate settings at startup (fail fast on missing required env vars).
get_settings()

app = FastAPI(title="AI Incident Commander")

# Dev-only CORS: the Phase 8 React dashboard (Vite dev server, default
# http://localhost:5173) runs on a different origin than this API
# (http://localhost:8000), so the browser's fetch calls need CORS headers
# or every request from the dashboard fails before it ever reaches a
# route. Wide open (`allow_origins=["*"]`, all methods/headers) is fine
# here on the same "no auth at this phase" MVP basis backend/api/
# simulation.py's docstring already established for this project -- there
# is no session/cookie-based auth anywhere yet for a permissive CORS
# policy to put at risk, and BUILD_PLAN.md scopes real auth/RBAC out of
# this project entirely (see BUILD_PLAN.md's Phase 9 "deliberately
# deferred" list: "JWT/RBAC ... deliberately deferred to an optional final
# phase, not core"). Revisit (restrict origins, drop `allow_credentials`)
# before this API is ever exposed outside local dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(incidents_router)
app.include_router(approvals_router)
app.include_router(simulation_router)
app.include_router(evaluation_router)
