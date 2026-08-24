"""Incident — the top-level entity a failure injection creates.

Carries both the *injected* ground truth (`failure_type`,
`root_cause_category`, and scenario provenance) that the eval harness
(Phase 7) scores the agent against, and the lifecycle `status` the agent
graph drives forward starting Phase 3. Nothing about what the *agent*
diagnoses lives here — evidence, hypotheses, approvals, and audit events
are later-phase tables (Phase 3/5/6) once the corresponding agent/response
logic exists to populate them. Keeping the injected ground truth on this
row (rather than folded into whatever the agent later produces) is what
makes accuracy scoring a plain equality check instead of a diff against a
separate ground-truth store.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db import Base

if TYPE_CHECKING:
    from backend.models.service import Service


class IncidentStatus(enum.StrEnum):
    """Lifecycle states from BUILD_PLAN.md's Agent Architecture section.

    `detected -> triaging -> investigating -> diagnosed ->
    awaiting_approval -> executing -> verifying -> resolved |
    manual_intervention_required` (verifying can loop back to
    investigating if recovery didn't stick — that's a status transition,
    not a distinct enum value).
    """

    DETECTED = "detected"
    TRIAGING = "triaging"
    INVESTIGATING = "investigating"
    DIAGNOSED = "diagnosed"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    RESOLVED = "resolved"
    MANUAL_INTERVENTION_REQUIRED = "manual_intervention_required"


class Severity(enum.StrEnum):
    """Incident severity, matching the `severity: P1` format used in the
    example `failure_scenarios/*.yaml` scenario in BUILD_PLAN.md."""

    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class Incident(Base):
    """One failure-injection instance: the thing the agent graph investigates."""

    __tablename__ = "incidents"
    __table_args__ = (
        Index("ix_incidents_service_id_detected_at", "service_id", "detected_at"),
        # Lets Phase 7's eval harness look up "which incident did
        # (scenario X, seed S, instance i) produce" without a table scan.
        # A plain index rather than a UNIQUE constraint: scenario_seed/
        # scenario_instance_index are nullable (ad-hoc single injections
        # have neither), and Postgres treats NULLs as distinct under a
        # unique index anyway, so uniqueness wouldn't actually be enforced
        # for the common no-seed case — an index is the honest tool here.
        Index(
            "ix_incidents_scenario_provenance",
            "failure_type",
            "scenario_seed",
            "scenario_instance_index",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    # Single-service MVP per BUILD_PLAN.md: the plan doesn't demand
    # multi-service incidents. The `cascading_payment_timeout` scenario's
    # cross-service story is represented via TraceLite.downstream_service_id
    # and the scenario's ordered `causal_chain`, not a multi-FK incident row.
    # CASCADE: see Service's docstring for the reasoning. No standalone
    # index: the composite (service_id, detected_at) index below covers
    # service_id-only lookups.
    service_id: Mapped[int] = mapped_column(
        ForeignKey("services.id", ondelete="CASCADE"), nullable=False
    )

    # `native_enum=False` -> VARCHAR + CHECK constraint (`create_constraint=True`)
    # rather than a Postgres native ENUM type: altering the allowed set
    # later is a plain constraint change instead of `ALTER TYPE ... ADD
    # VALUE` (non-transactional on older PG, values can't be removed).
    # `values_callable` is required: SQLAlchemy's default for a non-native
    # Enum is to store/constrain on the Python member *name*
    # ("DETECTED"), not `.value` ("detected") — without it the DB would
    # silently diverge from the `IncidentStatus`/`Severity` string values
    # used everywhere else (API responses, `failure_scenarios/*.yaml`'s
    # `severity: P1` format).
    severity: Mapped[Severity] = mapped_column(
        SAEnum(
            Severity,
            name="incident_severity",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            values_callable=lambda cls: [member.value for member in cls],
            length=2,
        ),
        nullable=False,
    )
    status: Mapped[IncidentStatus] = mapped_column(
        SAEnum(
            IncidentStatus,
            name="incident_status",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            values_callable=lambda cls: [member.value for member in cls],
            length=32,
        ),
        nullable=False,
        default=IncidentStatus.DETECTED,
        server_default=IncidentStatus.DETECTED.value,
        index=True,
    )

    # --- Injected ground truth (what the simulator seeded this incident
    # with) — kept separate from whatever the agent later *diagnoses*
    # (a Phase 3+ concern). Plain strings, not an enum: the fixed set of
    # failure_type/root_cause_category values is defined by the
    # `failure_scenarios/*.yaml` files, which is the *next* build step,
    # not this one — hard-coding an enum here would bake in values before
    # they're authored. `root_cause_category` becoming a fixed enum (the
    # scenario categories + `unknown`) is explicitly a Phase 3+/RCA-node
    # concern (BUILD_PLAN.md's Agent Architecture section), not a data
    # model concern.
    # No standalone index on failure_type: the composite
    # ix_incidents_scenario_provenance index below leads with failure_type
    # and already covers failure_type-only lookups.
    failure_type: Mapped[str] = mapped_column(String(100), nullable=False)
    root_cause_category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # --- Scenario provenance, so Phase 7's eval harness can map an
    # incident back to the exact generated instance deterministically.
    # Nullable: an ad-hoc `POST /api/simulation/failure` injection (not
    # part of a `--count N --seed S` batch) has no seed/instance index.
    scenario_seed: Mapped[int | None] = mapped_column(Integer, default=None)
    scenario_instance_index: Mapped[int | None] = mapped_column(Integer, default=None)

    service: Mapped[Service] = relationship(back_populates="incidents")

    def __repr__(self) -> str:
        return (
            f"Incident(id={self.id!r}, failure_type={self.failure_type!r}, "
            f"status={self.status!r})"
        )
