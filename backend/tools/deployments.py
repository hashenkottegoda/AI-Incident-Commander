"""`get_deployments` — query `Deployment` rows for a service/time window.

This is how the agent discovers `checkout_deployment_v1.8.2`-style
evidence (BUILD_PLAN.md's `db_connection_exhaustion` example).
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models import Deployment
from backend.tools.common import parse_time_window, resolve_service
from backend.tools.schemas import DeploymentRecord


def get_deployments(
    db: Session,
    service: str,
    start: str,
    end: str,
) -> list[DeploymentRecord]:
    """Query deploy events for one service within a time window.

    Args:
        db: Request-scoped database session (not LLM-facing).
        service: Service name, e.g. "checkout-service".
        start: ISO 8601 timestamp, inclusive window start.
        end: ISO 8601 timestamp, exclusive window end.

    Returns:
        Matching `DeploymentRecord`s ordered by `deployed_at` ascending.
        Empty list means the query was valid but no deploy happened in
        this window (a real, meaningful "no recent deployment" finding).

    Raises:
        ValueError: unknown service, malformed timestamp, or `start >= end`.
    """
    svc = resolve_service(db, service)
    start_dt, end_dt = parse_time_window(start, end)

    stmt = (
        select(Deployment)
        .where(
            Deployment.service_id == svc.id,
            Deployment.deployed_at >= start_dt,
            Deployment.deployed_at < end_dt,
        )
        .order_by(Deployment.deployed_at)
    )

    rows = db.execute(stmt).scalars().all()
    return [
        DeploymentRecord(
            id=row.id,
            service=svc.name,
            version=row.version,
            deployed_at=row.deployed_at,
        )
        for row in rows
    ]


def make_get_deployments_tool(db: Session) -> BaseTool:
    """Bind `get_deployments` to `db` (see `logs.make_get_logs_tool` for the
    binding pattern rationale)."""

    @tool("get_deployments", parse_docstring=True)
    def _get_deployments_tool(service: str, start: str, end: str) -> list[dict]:
        """Query deploy events for one service within a time window.

        Use this to check whether a recent deploy correlates with the
        start of an incident (a common root cause).

        Args:
            service: Service name, e.g. "checkout-service".
            start: ISO 8601 timestamp, inclusive window start.
            end: ISO 8601 timestamp, exclusive window end.
        """
        # See logs.make_get_logs_tool for why this returns dicts
        # (`mode="json"`) rather than the Pydantic models directly.
        return [
            record.model_dump(mode="json") for record in get_deployments(db, service, start, end)
        ]

    return _get_deployments_tool
