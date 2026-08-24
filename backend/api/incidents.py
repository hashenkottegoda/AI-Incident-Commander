"""`POST /api/incidents/{id}/investigate` — Phase 3's investigator endpoint.

BUILD_PLAN.md Phase 3: *"Wire to `POST /api/incidents/{id}/investigate`."*
Kept thin on purpose: load the `Incident` (404 if missing), run the
single-node ReAct investigator, return its `DiagnosisResult` as the
response body. No incident_status transition or extra persistence here —
BUILD_PLAN.md scopes Phase 3 to validating tool-calling mechanics
end-to-end; wiring `DiagnosisResult` into the `Incident` lifecycle
(`triaging -> investigating -> diagnosed -> ...`) is Phase 5's job once
the full graph exists.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.agents.investigator import investigate_incident
from backend.agents.schemas import DiagnosisResult
from backend.db import get_db
from backend.models import Incident

router = APIRouter(prefix="/api/incidents", tags=["incidents"])


@router.post("/{incident_id}/investigate", response_model=DiagnosisResult)
def investigate(incident_id: int, db: Session = Depends(get_db)) -> DiagnosisResult:  # noqa: B008
    """Run the Phase 3 baseline investigator against `incident_id`."""
    incident = db.get(Incident, incident_id)
    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"incident {incident_id} not found",
        )
    return investigate_incident(db, incident)
