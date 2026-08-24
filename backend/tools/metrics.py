"""`get_metrics` — query `MetricPoint` rows for a service/metric/time window."""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models import MetricPoint
from backend.tools.common import parse_time_window, resolve_service
from backend.tools.schemas import MetricRecord


def get_metrics(
    db: Session,
    service: str,
    metric_name: str,
    start: str,
    end: str,
) -> list[MetricRecord]:
    """Query metric samples for one service/metric within a time window.

    Args:
        db: Request-scoped database session (not LLM-facing).
        service: Service name, e.g. "checkout-service".
        metric_name: Metric to query, e.g. "db_connections_active",
            "error_rate", "latency_p99_ms", "memory_usage_mb".
        start: ISO 8601 timestamp, inclusive window start.
        end: ISO 8601 timestamp, exclusive window end.

    Returns:
        Matching `MetricRecord`s ordered by timestamp ascending. Empty
        list means the query was valid but matched no samples (e.g. a
        real metric name that this service just doesn't emit).

    Raises:
        ValueError: unknown service, malformed timestamp, or `start >= end`.
    """
    svc = resolve_service(db, service)
    start_dt, end_dt = parse_time_window(start, end)

    stmt = (
        select(MetricPoint)
        .where(
            MetricPoint.service_id == svc.id,
            MetricPoint.metric_name == metric_name,
            MetricPoint.timestamp >= start_dt,
            MetricPoint.timestamp < end_dt,
        )
        .order_by(MetricPoint.timestamp)
    )

    rows = db.execute(stmt).scalars().all()
    return [
        MetricRecord(
            id=row.id,
            service=svc.name,
            timestamp=row.timestamp,
            metric_name=row.metric_name,
            value=row.value,
        )
        for row in rows
    ]


def make_get_metrics_tool(db: Session) -> BaseTool:
    """Bind `get_metrics` to `db` (see `logs.make_get_logs_tool` for the
    binding pattern rationale)."""

    @tool("get_metrics", parse_docstring=True)
    def _get_metrics_tool(service: str, metric_name: str, start: str, end: str) -> list[dict]:
        """Query metric samples for one service/metric within a time window.

        Use this to check whether a resource metric (connections, memory,
        latency, error rate) is ramping/anomalous around an incident.

        Args:
            service: Service name, e.g. "checkout-service".
            metric_name: Metric to query, e.g. "db_connections_active",
                "error_rate", "latency_p99_ms", "memory_usage_mb".
            start: ISO 8601 timestamp, inclusive window start.
            end: ISO 8601 timestamp, exclusive window end.
        """
        # See logs.make_get_logs_tool for why this returns dicts
        # (`mode="json"`) rather than the Pydantic models directly.
        return [
            record.model_dump(mode="json")
            for record in get_metrics(db, service, metric_name, start, end)
        ]

    return _get_metrics_tool
