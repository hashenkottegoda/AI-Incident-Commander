"""NodeProgressEvent — the live-trace progress log Phase 8's dashboard
polls.

BUILD_PLAN.md Phase 8's "live investigation trace" spec, verbatim: *"Live-
trace transport: persist each graph node's progress to Postgres as it runs
and have the dashboard poll that progress log (simplest, MVP); LangGraph
astream_events -> SSE is the optional upgrade, not required for v1."*

This module builds ONLY the data model and its write path
(`backend.graph.build_incident_graph`'s `_with_progress` wrapper, applied
once to every node) — the read side (a `GET` endpoint the dashboard polls)
and the dashboard itself are later, separate steps and deliberately do not
exist yet.

One row per node *invocation* (not per node *type*) — a re-investigation
loop revisits `investigation`/`rag`/`root_cause` more than once per
incident, and the whole point of a live trace is showing that a node ran
again, not collapsing repeated passes into one row. `started_at` alone
(no `completed_at`/status column) is enough for what BUILD_PLAN.md actually
asks for: the dashboard can infer "currently running" from the most recent
row for an incident (nothing after it yet), and "node history in order"
from the full list ordered by `started_at` — a graph node here always
either completes and hands off to the next node (whose row then appears)
or the whole run ends, so there is no "silently stuck mid-node" state this
table needs to distinguish for a v1 poll-based trace. Adding a completion
timestamp/status is a natural follow-up once the read API exists and an
actual UI need for it (e.g. per-node duration) shows up — not invented
ahead of that need here.

`node_name` is a plain `String`, not an enum tied to the fixed set of node
names wired up in `backend.graph.build_incident_graph` today
(`triage`/`investigation`/`rag`/`root_cause`/`response_planner`/
`human_approval`/`action_executor`/`recovery_check`): same reasoning as
`AuditEvent.action_type` (see `backend/models/audit.py`'s docstring) — the
graph's node set is expected to keep evolving (BUILD_PLAN.md phases keep
adding nodes), and constraining this column to today's set at the DB layer
would mean editing this model every time a node is renamed or added, for a
column that exists purely for an internal dashboard's chronological trace,
not data integrity enforcement.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db import Base

if TYPE_CHECKING:
    from backend.models.incident import Incident


class NodeProgressEvent(Base):
    """One row per graph-node invocation: "node X started running for
    incident Y at time T" — the raw material for Phase 8's polled live
    trace. See module docstring for why this is intentionally a single
    start-timestamp row per invocation rather than a start/complete pair.
    """

    __tablename__ = "node_progress_events"
    __table_args__ = (
        # Matches AuditEvent's (incident_id, timestamp) composite-index
        # convention (`ix_audit_events_incident_id_recommended_at`): the
        # dashboard's "progress for this incident, in order" poll query
        # filters by incident_id and wants chronological order.
        Index("ix_node_progress_events_incident_id_started_at", "incident_id", "started_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    # CASCADE: a progress row is meaningless without its incident, same
    # reasoning as every other incident-scoped FK in this codebase (see
    # Service's docstring, applied transitively here).
    incident_id: Mapped[int] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False
    )

    # Plain string, not an enum — see module docstring for why the graph's
    # open-ended, evolving node-name vocabulary doesn't belong behind a
    # DB-level CHECK constraint (same reasoning as AuditEvent.action_type).
    node_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # server_default so a row is always timestamped even if the caller
    # forgets to pass it explicitly, matching AuditEvent.recommended_at's
    # convention.
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    incident: Mapped[Incident] = relationship(back_populates="node_progress_events")

    def __repr__(self) -> str:
        return (
            f"NodeProgressEvent(id={self.id!r}, incident_id={self.incident_id!r}, "
            f"node_name={self.node_name!r})"
        )
