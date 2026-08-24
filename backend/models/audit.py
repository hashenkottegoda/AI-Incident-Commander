"""AuditEvent — the Phase 6 audit trail table.

BUILD_PLAN.md's Phase 6 requirement, verbatim: *"Audit trail table logging
every approval decision and executed action, including an approver
identity — even a stubbed/header-supplied user is enough (full RBAC is
deliberately cut) so 'only authorized users approve' has a real hook."*

This module builds ONLY the data model — one row per action the Response
Planner recommends, tracking that action's whole lifecycle (recommended ->
classified -> [auto-executed | pending -> approved/rejected -> executed] ->
execution outcome) as it gets updated in place by the later Response
Planner / Risk Classifier / HITL approval endpoints / Action Executor /
Recovery Check nodes (none of which exist yet — see `backend/agents/` and
`backend/graph.py`, owned by langgraph-agent-engineer). A single table
rather than separate "recommendation" and "audit log" tables: a
recommended action and its approval/execution record are the same
lifecycle, not two independently-queried concerns, and splitting them
would need a repository layer with no second caller to justify it yet
(see `backend/models/incident.py`'s docstring for the same reasoning
applied to ground truth vs. diagnosis).

One row is created per recommended action and then updated in place as it
moves through its lifecycle — not re-inserted at each stage. This is what
makes the idempotency guard BUILD_PLAN.md calls for cheap: *"The approval
flow is idempotent: resuming (or a duplicate /approve) re-enters the node
but guards on incident_status/execution_result_id so the remediation
executes at most once."* Here, that guard is `executed_at is NULL` (or
equivalently `decision_status not in (APPROVED, PENDING_APPROVAL,
REJECTED)`) on this row before the Action Executor is allowed to run —
`executed_at` starts NULL and is set exactly once, by design.

`executed_at IS NULL` alone is a read-then-write check, not an atomic one:
two concurrent duplicate `/approve` calls could both read NULL before
either commits, and both execute. SQLAlchemy's `version_id_col` (below)
closes that gap for free — a concurrent second writer working from a
stale `version_id` gets a `StaleDataError` on commit instead of silently
executing twice. The future approval-endpoint author should still commit
the "am I the one executing" check inside the same transaction as the
`executed_at` write, but the version column means a lost-update race
fails loudly rather than double-executing a real remediation.

`action_type` is a plain `String`, not an enum tied to
`backend.simulation.scenario_schema.ACTION_TYPES`: those 5 names
(`rollback_deployment`, `restart_service`, `scale_service`,
`disable_feature_flag`, `increase_connection_pool`) are only the
HIGH_IMPACT vocabulary. SAFE actions (BUILD_PLAN.md: *"generate incident
report, add investigation note, gather additional diagnostics, tag
incident"*) are a separate, smaller vocabulary that has no home in
`scenario_schema.py` (the simulator doesn't need to know about them — they
never touch synthetic telemetry). Constraining this column to the union of
both fixed sets today would mean editing this model every time the
Response Planner's SAFE-action list changes, for a portfolio-scale system
where "a typo'd action name" isn't a realistic threat this table needs to
police at the DB layer.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, String, func, text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db import Base

if TYPE_CHECKING:
    from backend.models.incident import Incident


class RiskClassification(enum.StrEnum):
    """Output of the (future, deterministic) Risk Classifier node.

    BUILD_PLAN.md: *"RISK CLASSIFIER (deterministic, code-level rule
    table — never an LLM decision) ... SAFE (report/note/tag/gather-
    diagnostics) -> ACTION EXECUTOR ... HIGH-IMPACT (rollback/restart/
    scale/config/disable) -> HUMAN APPROVAL."*
    """

    SAFE = "safe"
    HIGH_IMPACT = "high_impact"


class AuditDecisionStatus(enum.StrEnum):
    """Where this recommended action currently sits in its approval lifecycle.

    SAFE actions go straight to `AUTO_EXECUTED` (no human decision, no
    `PENDING_APPROVAL`/`APPROVED` step — the Risk Classifier routes them
    directly to the Action Executor). HIGH_IMPACT actions start at
    `PENDING_APPROVAL`, then move to `APPROVED` or `REJECTED`; an
    `APPROVED` row only reaches the terminal `EXECUTED` state once the
    Action Executor has actually run (approval and execution are distinct
    moments per BUILD_PLAN's graph — `interrupt()` resumes into "approved",
    the executor runs strictly after that, never before).
    """

    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_EXECUTED = "auto_executed"
    EXECUTED = "executed"


class ExecutionOutcome(enum.StrEnum):
    """Set by the (future) Recovery Check once post-action telemetry is read.

    Mirrors the two outcomes BUILD_PLAN.md's Recovery Check routes on:
    `resolved` vs. still-degraded (which loops back to investigating).
    Named `RECOVERED`/`STILL_DEGRADED` here rather than reusing
    `IncidentStatus` values directly — this column describes *this specific
    action's* effect, not the incident's overall lifecycle status (an
    incident can have several audited actions across a re-investigation
    loop; only the incident's own `status` is the single source of truth
    for where the incident as a whole stands).
    """

    RECOVERED = "recovered"
    STILL_DEGRADED = "still_degraded"


class AuditEvent(Base):
    """One recommended action's full audit trail: recommendation -> risk
    classification -> approval decision (+ approver) -> execution -> outcome.

    Field-by-field reasoning is in the module docstring above; timestamps
    below are each set exactly once, in lifecycle order
    (`recommended_at` -> `decided_at` -> `executed_at`), which is what lets
    a caller cheaply tell where a row is in its lifecycle from the
    timestamps alone, independent of `decision_status`.
    """

    __tablename__ = "audit_events"
    __table_args__ = (
        # Matches the (service_id, timestamp)-style composite convention
        # used across backend/models/telemetry.py and Incident's own
        # (service_id, detected_at) index: the dashboard's "audit trail for
        # this incident" view (Phase 8) and the idempotency guard both
        # filter by incident_id and want it in recommendation order.
        Index("ix_audit_events_incident_id_recommended_at", "incident_id", "recommended_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # Optimistic-concurrency guard (see module docstring): SQLAlchemy bumps
    # this on every UPDATE and raises StaleDataError if a concurrent writer
    # already moved it, instead of silently allowing a lost update. Must be
    # defined before __mapper_args__ references it below (class bodies
    # execute top-to-bottom). server_default (not just the ORM-side
    # default=0) so a raw SQL INSERT that bypasses the ORM still gets a
    # valid starting value instead of a NOT NULL violation.
    version_id: Mapped[int] = mapped_column(nullable=False, default=0, server_default=text("0"))
    __mapper_args__ = {"version_id_col": version_id}

    # CASCADE: an audit row is meaningless without its incident, same
    # reasoning as every other incident-scoped FK in this codebase
    # (see Service's docstring, applied transitively here).
    incident_id: Mapped[int] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False
    )

    # Plain string, not an enum — see module docstring for why SAFE and
    # HIGH_IMPACT action names are two separate, independently-evolving
    # vocabularies that don't belong behind one DB-level CHECK constraint.
    action_type: Mapped[str] = mapped_column(String(100), nullable=False)

    # `native_enum=False` + `create_constraint=True` + `values_callable`:
    # same VARCHAR-plus-CHECK-constraint pattern as
    # `backend/models/incident.py`'s `IncidentStatus`/`Severity` (see that
    # module's docstring for the full rationale — non-native so the
    # allowed set can change via a plain constraint migration, and
    # `values_callable` so the DB constrains on `.value`
    # ("high_impact"), not the Python member name ("HIGH_IMPACT")).
    risk_classification: Mapped[RiskClassification] = mapped_column(
        SAEnum(
            RiskClassification,
            name="audit_risk_classification",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            values_callable=lambda cls: [member.value for member in cls],
            length=20,
        ),
        nullable=False,
        index=True,
    )

    # No column default: which initial value is correct (AUTO_EXECUTED vs.
    # PENDING_APPROVAL) depends on `risk_classification`, decided by the
    # future Risk Classifier node, not a fixed starting state the way
    # `Incident.status` always starts at DETECTED.
    decision_status: Mapped[AuditDecisionStatus] = mapped_column(
        SAEnum(
            AuditDecisionStatus,
            name="audit_decision_status",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            values_callable=lambda cls: [member.value for member in cls],
            length=20,
        ),
        nullable=False,
        index=True,
    )

    # "Even a stubbed/header-supplied user is enough" (BUILD_PLAN.md) —
    # plain string, no FK to a users table (no such table exists; full
    # RBAC is explicitly cut). NULL for AUTO_EXECUTED rows: a SAFE action
    # the Risk Classifier auto-executes never has a human approver.
    approver: Mapped[str | None] = mapped_column(String(200), default=None)

    # --- Lifecycle timestamps, each set exactly once, in order ----------
    # When the Response Planner recommended this action. server_default
    # so a row is always timestamped even if the caller forgets to pass
    # it explicitly, matching Service.created_at's convention.
    recommended_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # When a human approved/rejected (HIGH_IMPACT only) — NULL for
    # AUTO_EXECUTED rows (no human decision was made) and NULL until a
    # HIGH_IMPACT row leaves PENDING_APPROVAL.
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    # The idempotency hook BUILD_PLAN.md calls for: starts NULL, gets set
    # exactly once by the Action Executor. The future /approve endpoint
    # (and LangGraph's re-entry-on-resume behavior) checks
    # `executed_at IS NULL` before running the executor, so a duplicate
    # POST /approve or a re-executed interrupted node can't run the
    # remediation twice.
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    # --- Execution result, set by the (future) Recovery Check -----------
    execution_outcome: Mapped[ExecutionOutcome | None] = mapped_column(
        SAEnum(
            ExecutionOutcome,
            name="audit_execution_outcome",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            values_callable=lambda cls: [member.value for member in cls],
            length=20,
        ),
        default=None,
    )
    # Structured detail backing execution_outcome (e.g. which metrics were
    # compared against the pre-incident baseline and their before/after
    # values) — a JSONB blob rather than dedicated columns because this
    # varies by action_type/scenario the same way LogEntry.attributes
    # does, and a plain summary string would lose the per-metric detail
    # the eval harness (Phase 7's recovery-verification accuracy metric)
    # will want to inspect.
    execution_detail: Mapped[dict | None] = mapped_column(JSONB, default=None)

    incident: Mapped[Incident] = relationship(back_populates="audit_events")

    def __repr__(self) -> str:
        return (
            f"AuditEvent(id={self.id!r}, incident_id={self.incident_id!r}, "
            f"action_type={self.action_type!r}, decision_status={self.decision_status!r})"
        )
