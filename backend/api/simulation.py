"""`POST /api/simulation/failure` and `POST /api/simulation/reset`.

Thin HTTP wrapper around `backend.simulation.injector.inject_failure`: this
is a dev/demo control surface for the simulator, not a real production API
(BUILD_PLAN.md's simulation layer never talks to anything real), so there's
no auth at this phase.
"""

from __future__ import annotations

import random
import secrets
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import delete
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.models import (
    Deployment,
    Incident,
    IncidentStatus,
    LogEntry,
    MetricPoint,
    Severity,
    TraceLite,
)
from backend.simulation.injector import inject_failure
from backend.simulation.scenario_schema import load_all_scenarios

router = APIRouter(prefix="/api/simulation", tags=["simulation"])


class FailureInjectionRequest(BaseModel):
    """Body for `POST /api/simulation/failure`.

    `seed` is optional. Pass it for reproducible injections (required for
    anything eval-related); omit it for ad-hoc manual testing — in that
    case a process-random seed is generated per request (via `secrets`,
    not seeded itself), so the resulting telemetry is *not* reproducible
    across requests.
    """

    failure_type: str = Field(
        ..., description="Must match one of failure_scenarios/*.yaml's failure_type."
    )
    seed: int | None = Field(
        default=None,
        description="Deterministic seed. Omit for a non-reproducible ad-hoc injection.",
    )


class IncidentResponse(BaseModel):
    """The created incident, as returned by both simulation endpoints' callers."""

    id: int
    status: IncidentStatus
    failure_type: str
    root_cause_category: str
    severity: Severity
    detected_at: datetime


class ResetResponse(BaseModel):
    """Row counts deleted per table by `POST /api/simulation/reset`."""

    deleted: dict[str, int]


def _incident_response(incident: Incident) -> IncidentResponse:
    return IncidentResponse(
        id=incident.id,
        status=incident.status,
        failure_type=incident.failure_type,
        root_cause_category=incident.root_cause_category,
        severity=incident.severity,
        detected_at=incident.detected_at,
    )


@router.post("/failure", response_model=IncidentResponse, status_code=status.HTTP_201_CREATED)
def create_failure(
    payload: FailureInjectionRequest, db: Session = Depends(get_db)  # noqa: B008
) -> IncidentResponse:
    """Inject one instance of `payload.failure_type`, ending "now"."""
    scenarios = load_all_scenarios()
    scenario = scenarios.get(payload.failure_type)
    if scenario is None:
        known = sorted(scenarios)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown failure_type {payload.failure_type!r}; expected one of {known}",
        )

    seed = payload.seed if payload.seed is not None else secrets.randbits(31)
    rng = random.Random(seed)
    incident_start = datetime.now(UTC)

    incident = inject_failure(db, scenario, rng, incident_start)
    db.commit()

    return _incident_response(incident)


@router.post("/reset", response_model=ResetResponse)
def reset_simulation(db: Session = Depends(get_db)) -> ResetResponse:  # noqa: B008
    """Delete all simulation-generated data so the demo can replay from a
    clean slate. Leaves the 3 canonical `Service` rows in place — they're
    stable seed data, not something a demo run should have to re-seed."""
    deleted: dict[str, int] = {}
    # Order doesn't matter for FK integrity here: none of these tables
    # reference each other, they only reference `services` (untouched).
    for model in (Incident, TraceLite, LogEntry, MetricPoint, Deployment):
        result = db.execute(delete(model))
        deleted[model.__tablename__] = result.rowcount or 0
    db.commit()

    return ResetResponse(deleted=deleted)
