# Backend service image for AI Incident Commander.
#
# Single-stage build — deliberately simple for MVP-stage FastAPI app
# (BUILD_PLAN.md Phase 0). Multi-stage build complexity isn't justified yet.

FROM python:3.12-slim

# Install uv (pinned via pip so the image build is reproducible without an
# extra network fetch/installer script).
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency manifests first so dependency installation is cached
# separately from source changes. README.md is also required at this stage:
# hatchling (the build backend) reads `readme = "README.md"` from
# pyproject.toml while building this package itself.
COPY pyproject.toml uv.lock README.md ./

# Production deps only — `--no-dev` excludes the `dev` dependency group
# (pytest/ruff/httpx), and `--frozen` refuses to update uv.lock, ensuring
# the image matches the committed lockfile exactly.
RUN uv sync --frozen --no-dev

# Data directories read at runtime (scenario ground truth + RAG seed data),
# copied before the application source since they change far less often than
# backend/ — keeps this layer cached across ordinary source-only edits.
# Required at runtime, not just dev: backend/simulation/scenario_schema.py
# resolves SCENARIOS_DIR to /app/failure_scenarios and
# backend/rag/historical_incidents.py resolves its seed path to
# /app/historical_incidents/historical_incidents.yaml inside the container.
COPY failure_scenarios/ ./failure_scenarios/
COPY historical_incidents/ ./historical_incidents/

# Now copy the application source.
COPY backend/ ./backend/

EXPOSE 8000

# --no-sync: the image already has exactly the locked, --no-dev environment
# from the build step above; skip uv's runtime sync check so containers
# don't re-resolve/install dev deps (e.g. ruff) on every start.
CMD ["uv", "run", "--no-sync", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
