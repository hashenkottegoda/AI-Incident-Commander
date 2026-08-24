"""`IncidentState` — the Phase 5 `StateGraph`'s single source of truth.

BUILD_PLAN.md's Agent Architecture section, verbatim: *"State (`IncidentState`,
Pydantic) holds references + compact reasoning state, NOT bulk data. Large
payloads (raw logs, metric series, full tool results) live in Postgres,
keyed by id; the graph state carries the ids/refs plus the distilled
reasoning. This keeps LangGraph checkpoints small ... Fields: incident_id,
incident_status, severity, affected_services, tool_call_log_ids[],
evidence[] (structured, with source_refs — not raw payloads), hypotheses[],
root_cause, diagnostic_confidence, alternative_hypotheses[],
recommended_actions[], approval_decision, execution_result_id,
recovery_result"*

This module defines the FULL schema now (Phase 5) even though only the
fields through Root Cause are populated before Phase 6 exists —
`recommended_actions`/`approval_decision`/`execution_result_id`/
`recovery_result` are declared as optional/nullable so the shape doesn't
need reshaping when Phase 6 (Response Planner / HITL / Executor / Recovery)
starts writing to them.

## `tool_call_log_ids` — no backing table exists yet

BUILD_PLAN.md's Phase 1/2 schema never defines a `ToolCallLog` table — only
`LogEntry`/`MetricPoint`/`Deployment`/`TraceLite` (the data tools *query*),
not a record of the *queries themselves*. Since there is nothing in
Postgres for these ids to reference yet, `tool_call_log_ids` here holds a
compact, monotonically increasing per-investigation ordinal (1, 2, 3, ...)
assigned to each tool call as it happens — not a Postgres foreign key. This
keeps the field genuinely "a compact reference, not bulk data" as the
docstring above requires, and a future phase that adds a real
`ToolCallLog` table can start writing those same ordinals into it without
this schema changing shape. The evidence-sufficiency conditional-edge
check (`backend.agents.routing`) does not rely on this field — it checks
tool coverage via `evidence[].source_ref.tool` instead, since every tool
call this graph makes is turned into exactly one `EvidenceItem` (including
"found nothing" calls) at the moment it happens — see
`backend.agents.investigation_node`.

## Fields beyond BUILD_PLAN.md's literal list

`investigation_iterations` is additive: BUILD_PLAN.md requires the
re-investigation loop to be "bounded to N iterations regardless" but the
literal field list doesn't name what tracks that count. A single `int`
counter is exactly the "compact reasoning state, not bulk data" this class
already holds everything else as, so it's added here rather than smuggled
in as a side channel outside the state.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.agents.schemas import EvidenceItem, Hypothesis, RootCauseCategory
from backend.models.incident import IncidentStatus, Severity


class IncidentState(BaseModel):
    """LangGraph state schema for the Phase 5 investigation graph.

    Every node function receives the current `IncidentState` and returns a
    partial-update `dict` of the fields it changed (LangGraph's default
    "last write wins" merge for non-reducer fields) — see
    `backend/agents/*_node.py` and `backend/graph.py`.
    """

    # --- Identity + lifecycle -------------------------------------------
    incident_id: int
    incident_status: IncidentStatus = IncidentStatus.DETECTED
    severity: Severity | None = None
    affected_services: list[str] = Field(default_factory=list)

    # --- Investigation evidence (references, not bulk payloads) ---------
    tool_call_log_ids: list[int] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)

    # --- Diagnosis (Root Cause node output) ------------------------------
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    root_cause: RootCauseCategory | None = None
    diagnostic_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    alternative_hypotheses: list[Hypothesis] = Field(default_factory=list)

    # --- Phase 6+ fields: declared now, unpopulated through Phase 5 -----
    # `recommended_actions` is left as opaque dicts rather than a typed
    # `ResponseAction` schema — that schema is Phase 6's Response Planner
    # concern (safe-vs-high-impact rule table etc.), not this phase's.
    recommended_actions: list[dict[str, Any]] = Field(default_factory=list)
    approval_decision: str | None = None
    execution_result_id: int | None = None
    recovery_result: str | None = None

    # --- Re-investigation loop bookkeeping (see module docstring) -------
    investigation_iterations: int = 0
