"""`get_dependencies` — query `TraceLite` rows for a service/time window.

BUILD_PLAN.md's Agent Architecture section lists the Investigation node's
tool set as "logs/metrics/deployments/dependencies/db-status/config;
traces = optional secondary signal" — "dependencies" is the tool name the
agent reasons about, backed by the `TraceLite` table. Surfacing
`downstream_service`/`duration_ms` is what lets the agent discover
cross-service evidence like `cascading_payment_timeout`'s
checkout -> payment retry storm (its `checkout_retry_storm_detected`
expected evidence).
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from backend.models import TraceLite
from backend.tools.common import parse_time_window, resolve_service
from backend.tools.schemas import TraceRecord


def get_dependencies(
    db: Session,
    service: str,
    start: str,
    end: str,
) -> list[TraceRecord]:
    """Query outbound call spans for one service within a time window.

    Args:
        db: Request-scoped database session (not LLM-facing).
        service: Service name, e.g. "checkout-service". Spans are queried
            by their *originating* service, i.e. calls this service made
            outward, not calls made into it.
        start: ISO 8601 timestamp, inclusive window start.
        end: ISO 8601 timestamp, exclusive window end.

    Returns:
        Matching `TraceRecord`s ordered by timestamp ascending, each
        carrying `duration_ms` and (when the span named one)
        `downstream_service`. Empty list means the query was valid but
        this service made no recorded outbound calls in the window.

    Raises:
        ValueError: unknown service, malformed timestamp, or `start >= end`.
    """
    svc = resolve_service(db, service)
    start_dt, end_dt = parse_time_window(start, end)

    stmt = (
        select(TraceLite)
        .options(joinedload(TraceLite.downstream_service))
        .where(
            TraceLite.service_id == svc.id,
            TraceLite.timestamp >= start_dt,
            TraceLite.timestamp < end_dt,
        )
        .order_by(TraceLite.timestamp)
    )

    rows = db.execute(stmt).scalars().all()
    return [
        TraceRecord(
            id=row.id,
            service=svc.name,
            timestamp=row.timestamp,
            span_name=row.span_name,
            duration_ms=row.duration_ms,
            downstream_service=row.downstream_service.name if row.downstream_service else None,
        )
        for row in rows
    ]


def make_get_dependencies_tool(db: Session) -> BaseTool:
    """Bind `get_dependencies` to `db` (see `logs.make_get_logs_tool` for
    the binding pattern rationale)."""

    @tool("get_dependencies", parse_docstring=True)
    def _get_dependencies_tool(service: str, start: str, end: str) -> list[dict]:
        """Query outbound call spans for one service within a time window.

        Use this to check whether a service's calls to a downstream
        dependency are slow/retrying — this is the signal for
        cross-service, cascading failures (a service that looks broken may
        actually be waiting on a slow downstream dependency).

        Args:
            service: Service name, e.g. "checkout-service". Spans are
                queried by their originating service.
            start: ISO 8601 timestamp, inclusive window start.
            end: ISO 8601 timestamp, exclusive window end.
        """
        # See logs.make_get_logs_tool for why this returns dicts
        # (`mode="json"`) rather than the Pydantic models directly.
        return [
            record.model_dump(mode="json") for record in get_dependencies(db, service, start, end)
        ]

    return _get_dependencies_tool
