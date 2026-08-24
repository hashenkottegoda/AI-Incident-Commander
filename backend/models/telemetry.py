"""Telemetry tables: logs, metrics, and lightweight traces.

These are the tables Phase 2's `get_logs` / `get_metrics` tools will
query, and what Phase 1's failure-injection engine writes a temporally
coherent anomaly timeline into (BUILD_PLAN.md: "deployment at T,
connection count rising at T+1, ... incident triggered at T+5"). All three
share the same query shape — filter by service and a time range — hence
the shared `(service_id, timestamp)` index convention.

`TraceLite` is deliberately minimal: BUILD_PLAN.md calls traces-lite an
optional secondary signal, not a full distributed-tracing model. It exists
mainly to carry `downstream_service_id` so dependency-failure scenarios
(payment -> checkout) have something to point the agent at, without
modeling a full span tree / trace id / parent-span graph.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Index, String
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db import Base

if TYPE_CHECKING:
    from backend.models.service import Service


class LogLevel(enum.StrEnum):
    """Fixed log-severity taxonomy the synthetic log generator emits."""

    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class LogEntry(Base):
    """One synthetic application log line for a service at a point in time."""

    __tablename__ = "log_entries"
    __table_args__ = (
        Index("ix_log_entries_service_id_timestamp", "service_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # CASCADE: see Service's docstring — telemetry rows are meaningless
    # once their owning service is gone. No standalone index: the composite
    # (service_id, timestamp) index below covers service_id-only lookups.
    service_id: Mapped[int] = mapped_column(
        ForeignKey("services.id", ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    # `native_enum=False` -> VARCHAR + CHECK constraint (`create_constraint=True`)
    # rather than a Postgres native ENUM type. Same "reject invalid
    # values" guarantee, but altering the allowed set later is a plain
    # constraint change instead of Postgres's awkward
    # `ALTER TYPE ... ADD VALUE` dance (non-transactional on older PG,
    # unremovable values). Applied consistently to every enum-ish column
    # in this package. `values_callable` is required too: without it,
    # SQLAlchemy stores/constrains on the Python enum member *name*
    # ("INFO"), not `.value` ("info").
    level: Mapped[LogLevel] = mapped_column(
        SAEnum(
            LogLevel,
            name="log_level",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            values_callable=lambda cls: [member.value for member in cls],
            length=10,
        ),
        nullable=False,
        index=True,
    )
    message: Mapped[str] = mapped_column(String(2000), nullable=False)
    # Structured, failure-type-specific detail (e.g. {"pool_size": 20,
    # "active": 20} for a connection-pool exhaustion log) that varies
    # enough per failure type to not warrant dedicated columns.
    attributes: Mapped[dict | None] = mapped_column(JSONB, default=None)

    service: Mapped[Service] = relationship(back_populates="log_entries")

    def __repr__(self) -> str:
        return f"LogEntry(id={self.id!r}, service_id={self.service_id!r}, level={self.level!r})"


class MetricPoint(Base):
    """One synthetic metric sample (e.g. `db_connections_active`) for a service."""

    __tablename__ = "metric_points"
    __table_args__ = (
        Index("ix_metric_points_service_id_timestamp", "service_id", "timestamp"),
        # get_metrics (Phase 2) and the Recovery Check (Phase 6) both
        # filter by a specific metric_name within a service + time window,
        # not just by service + time — this composite index serves that
        # narrower, more common query shape directly.
        Index(
            "ix_metric_points_service_id_metric_name_timestamp",
            "service_id",
            "metric_name",
            "timestamp",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # No standalone index: the composite indexes below (leading with
    # service_id, and service_id+metric_name) already cover service_id-only
    # lookups via the leftmost-prefix rule.
    service_id: Mapped[int] = mapped_column(
        ForeignKey("services.id", ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    service: Mapped[Service] = relationship(back_populates="metric_points")

    def __repr__(self) -> str:
        return (
            f"MetricPoint(id={self.id!r}, metric_name={self.metric_name!r}, "
            f"value={self.value!r})"
        )


class TraceLite(Base):
    """A minimal span record — deliberately not a full tracing model.

    `downstream_service_id` is nullable and self-references `services`; it
    lets a dependency-failure scenario (e.g. payment -> checkout) record
    "this span called out to another service" without a full
    span-tree/trace-id model.
    """

    __tablename__ = "traces_lite"
    __table_args__ = (
        Index("ix_traces_lite_service_id_timestamp", "service_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # No standalone index: the composite (service_id, timestamp) index
    # below covers service_id-only lookups.
    service_id: Mapped[int] = mapped_column(
        ForeignKey("services.id", ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    span_name: Mapped[str] = mapped_column(String(200), nullable=False)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False)
    # SET NULL rather than CASCADE: this is a secondary, nullable pointer
    # to *another* service. If that downstream service row is ever
    # deleted, the span itself (owned by `service_id`) is still a real,
    # meaningful record — only the dependency link becomes unknown. A
    # CASCADE here would silently delete spans that have nothing wrong
    # with them just because an unrelated service was removed.
    downstream_service_id: Mapped[int | None] = mapped_column(
        ForeignKey("services.id", ondelete="SET NULL"), default=None, index=True
    )

    service: Mapped[Service] = relationship(back_populates="traces", foreign_keys=[service_id])
    downstream_service: Mapped[Service | None] = relationship(foreign_keys=[downstream_service_id])

    def __repr__(self) -> str:
        return f"TraceLite(id={self.id!r}, span_name={self.span_name!r})"
