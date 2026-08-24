"""Deployment — a deploy event for a service.

Represents one version rollout (e.g. `checkout-service` -> `v1.8.2`).
Referenced as a root-cause / `causal_chain` link by failure scenarios
(BUILD_PLAN.md's `bad_deployment`, `db_connection_exhaustion`, ...) and is
what Phase 6's `rollback_deployment()` action will target.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db import Base

if TYPE_CHECKING:
    from backend.models.service import Service


class Deployment(Base):
    """A deploy event: one service transitioning to one version at one time."""

    __tablename__ = "deployments"
    __table_args__ = (Index("ix_deployments_service_id_deployed_at", "service_id", "deployed_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    # CASCADE: see Service's docstring for the reasoning — deployments are
    # meaningless without their owning service. No standalone index here:
    # the composite (service_id, deployed_at) index below already covers
    # service_id-only lookups via the leftmost-prefix rule.
    service_id: Mapped[int] = mapped_column(
        ForeignKey("services.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    deployed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    service: Mapped[Service] = relationship(back_populates="deployments")

    def __repr__(self) -> str:
        return (
            f"Deployment(id={self.id!r}, service_id={self.service_id!r}, version={self.version!r})"
        )
