"""Typed, JSON-serializable return shapes for Phase 2's tool layer.

Every Phase 2 tool (`backend/tools/*.py`) returns a `list[...]` of one of
these Pydantic models rather than raw SQLAlchemy ORM instances — ORM
objects aren't JSON-serializable for a tool-call result, and a plain dict
crossing the tool boundary is exactly what BUILD_PLAN.md's conventions
rule out ("every endpoint gets a Pydantic request/response schema — no raw
dicts crossing the [tool/API] boundary").

Every record below carries its own row `id`. That id, plus the tool name
that produced it (implicit — the caller knows which tool it called), is
everything a later RCA node needs to build a `source_ref` like
`("get_logs", log_id=123)` (BUILD_PLAN.md's Agent Architecture section).
Building the actual `source_ref`/evidence schema is explicitly a Phase
3+ concern — this module only guarantees the id is present.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class LogRecord(BaseModel):
    """One `LogEntry` row, as returned by `get_logs`."""

    model_config = ConfigDict(frozen=True)

    id: int
    service: str
    timestamp: datetime
    level: str
    message: str
    attributes: dict | None = None


class MetricRecord(BaseModel):
    """One `MetricPoint` row, as returned by `get_metrics`."""

    model_config = ConfigDict(frozen=True)

    id: int
    service: str
    timestamp: datetime
    metric_name: str
    value: float


class DeploymentRecord(BaseModel):
    """One `Deployment` row, as returned by `get_deployments`."""

    model_config = ConfigDict(frozen=True)

    id: int
    service: str
    version: str
    deployed_at: datetime


class TraceRecord(BaseModel):
    """One `TraceLite` row, as returned by `get_dependencies`.

    `downstream_service` is `None` when the span never named a downstream
    call (mirrors `TraceLite.downstream_service_id`'s nullability).
    """

    model_config = ConfigDict(frozen=True)

    id: int
    service: str
    timestamp: datetime
    span_name: str
    duration_ms: float
    downstream_service: str | None = None
