"""Service — the reference entity every other table hangs off of.

Represents one simulated microservice (e.g. `checkout-service`,
`payment-service`, `inventory-service`). Phase 1's telemetry generator
seeds exactly those three; nothing here hard-codes that list — the table
is a plain lookup, generic enough for the generator to add more later
without a migration.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db import Base

if TYPE_CHECKING:
    from backend.models.deployment import Deployment
    from backend.models.incident import Incident
    from backend.models.telemetry import LogEntry, MetricPoint, TraceLite


class Service(Base):
    """A simulated microservice that telemetry/deployments/incidents reference."""

    __tablename__ = "services"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    description: Mapped[str | None] = mapped_column(String(500), default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Cascade rationale: services are static reference data (a handful of
    # rows seeded once, e.g. "checkout-service"). Deleting one is a rare,
    # deliberate operation, not an accidental click in this dev/demo
    # system — so cascading its telemetry/incidents away on delete avoids
    # orphaned rows without needing a separate cleanup step. See
    # `backend/models/telemetry.py` for the one exception
    # (`TraceLite.downstream_service_id` uses SET NULL instead).
    deployments: Mapped[list[Deployment]] = relationship(
        back_populates="service", cascade="all, delete-orphan"
    )
    log_entries: Mapped[list[LogEntry]] = relationship(
        back_populates="service", cascade="all, delete-orphan"
    )
    metric_points: Mapped[list[MetricPoint]] = relationship(
        back_populates="service", cascade="all, delete-orphan"
    )
    traces: Mapped[list[TraceLite]] = relationship(
        back_populates="service",
        foreign_keys="TraceLite.service_id",
        cascade="all, delete-orphan",
    )
    incidents: Mapped[list[Incident]] = relationship(
        back_populates="service", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"Service(id={self.id!r}, name={self.name!r})"
