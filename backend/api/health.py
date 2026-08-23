"""Health check endpoint.

Used by BUILD_PLAN.md's Phase 0 verify criterion (`GET /health` returns
200) and, later, by Docker Compose / deployment liveness checks.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Response body for `GET /health`."""

    status: str = "ok"


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Report that the API process is up."""
    return HealthResponse(status="ok")
