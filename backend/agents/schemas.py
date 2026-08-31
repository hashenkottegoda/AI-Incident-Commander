"""`DiagnosisResult` — the shared output schema for Experiments A/B/C/D.

BUILD_PLAN.md's Agent Architecture section: *"All four experiments emit
the same `DiagnosisResult` schema ... Only the architecture/data-access
method that produces it changes ... Same schema in/out is what makes the
A/B/C/D comparison apples-to-apples; the scoring code is written once
against `DiagnosisResult`."* Phase 3 only builds Experiment B (the single
ReAct agent in `backend.agents.investigator`), but this schema must stay
stable across Phases 3/4/5/7 — get it right here rather than retrofitting
it later.

Two structural requirements straight from BUILD_PLAN.md's "Structured
evidence & RCA (required for deterministic eval)" note, both enforced by
types rather than convention:

- `root_cause_category` is a closed `Literal` — the six real scenario
  categories (`failure_scenarios/*.yaml`) plus `"unknown"` — not a free
  string, so Phase 7's accuracy metric is `predicted == ground_truth`
  enum equality, not fuzzy string matching.
- Every `EvidenceItem` carries a structured `source_ref` (tool name +
  the tool result's real record id, or a query description when no
  single record applies), not free prose — so Phase 7's evidence-precision
  and hallucination-rate metrics can check cited `source_ref`s against a
  real tool-call log as a set operation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# The 6 scenario `root_cause_category` values (`failure_scenarios/*.yaml`,
# cross-checked against `tests/test_injector.py`) plus the escape hatch for
# "the evidence doesn't clearly support any of them" — never leave the
# model forced to pick a wrong category just because the enum has no
# honest option.
RootCauseCategory = Literal[
    "database_connection_pool",
    "memory_resource_exhaustion",
    "application_bug",
    "upstream_dependency_failure",
    "inefficient_database_query",
    "upstream_dependency_timeout",
    "unknown",
]


class SourceRef(BaseModel):
    """Points at the exact tool call (and, where applicable, the exact
    returned record) that backs one piece of evidence.

    This is the mechanism that makes Phase 7's hallucination-rate metric
    computable later: `record_id` can be checked against the real
    `tool_call_log` for that investigation run to confirm the cited id
    actually came back from that tool, rather than being invented.
    """

    tool: str = Field(
        description=(
            "Name of the tool call this evidence is grounded in, e.g. "
            "'get_logs', 'get_metrics', 'get_deployments', 'get_dependencies'."
        )
    )
    record_id: int | None = Field(
        default=None,
        description=(
            "The exact `id` field from that tool call's JSON result that this "
            "evidence cites. Required when the evidence is backed by one "
            "specific record. Never invent an id that wasn't actually "
            "returned by a tool call you made."
        ),
    )
    query: str | None = Field(
        default=None,
        description=(
            "Short description of the tool call's arguments (service/time "
            "window/etc.), used when the evidence is about the shape or "
            "absence of a result rather than one specific record id."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _rescue_non_numeric_record_id(cls, data: Any) -> Any:
        """Free-tier models occasionally cite historical-incident string ids
        (e.g. "hist-012") via record_id instead of query, despite prompt
        instructions to the contrary. record_id is int-typed by design (see
        `SourceRef.record_id`'s docstring and qdrant_client.point_id_for's
        comment); Pydantic's normal int coercion would raise ValidationError
        on a non-numeric string and crash the whole node. Numeric strings
        ("7") and real ints still go through normal coercion untouched --
        only the unparseable case is rescued, by moving the value into
        `query` (unless a real query is already set) and clearing record_id.
        """
        if isinstance(data, dict):
            record_id = data.get("record_id")
            if isinstance(record_id, str):
                try:
                    int(record_id)
                except ValueError:
                    data = dict(data)
                    if not data.get("query"):
                        data["query"] = record_id
                    data["record_id"] = None
        return data


class EvidenceItem(BaseModel):
    """One structured evidence record — never free prose without a citation."""

    description: str = Field(
        description=(
            "Short factual claim, e.g. 'db_connections_active rose from 8 to "
            "42 in the 10 minutes before the incident'."
        )
    )
    source_ref: SourceRef


class Hypothesis(BaseModel):
    """One root-cause hypothesis: a category plus the reasoning for it."""

    category: RootCauseCategory
    rationale: str = Field(
        description="Short explanation of why the evidence points to this category."
    )
    # Optional and additive (default None) -- added for Phase 5's conditional
    # re-investigation loop, which needs a per-hypothesis confidence value to
    # compare the top-2 ranked hypotheses' confidence gap
    # (backend.agents.routing.confidence_gap_below_threshold).
    # DiagnosisResult.diagnostic_confidence is a single overall float, not
    # per-hypothesis, so there was no existing field this could reuse.
    # Additive/optional by construction: Phase 3's baseline investigator
    # (backend.agents.investigator, frozen per this task's instructions)
    # never reads or writes this field -- its STRUCTURED_OUTPUT_INSTRUCTION
    # doesn't mention it, so the model simply leaves it unset (None) and
    # Experiment B's existing behavior/output shape is unchanged.
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional per-hypothesis heuristic confidence (0.0-1.0), same "
            "self-reported-estimate caveat as diagnostic_confidence -- not a "
            "calibrated probability. Populated by Phase 5's Root Cause node "
            "so the conditional re-investigation loop can compare how close "
            "the top two ranked hypotheses are."
        ),
    )


class DiagnosisResult(BaseModel):
    """Shared output schema for Experiments A/B/C/D (BUILD_PLAN.md Phase 3/7).

    `hypotheses` is ranked, most-likely first; `hypotheses[0].category`
    should equal `root_cause_category`. `alternative_hypotheses` holds the
    categories that were considered but not chosen — both use the same
    `Hypothesis` shape rather than one being free text, per this task's
    spec ("structure this as a list of structured hypothesis objects, not
    free text").
    """

    root_cause_category: RootCauseCategory = Field(
        description=(
            "The single most likely root cause category, constrained to the "
            "fixed scenario taxonomy plus 'unknown' if the evidence doesn't "
            "clearly support any of them."
        )
    )
    hypotheses: list[Hypothesis] = Field(
        default_factory=list,
        description=(
            "Ranked root-cause hypotheses, most likely first. "
            "hypotheses[0].category should equal root_cause_category."
        ),
    )
    alternative_hypotheses: list[Hypothesis] = Field(
        default_factory=list,
        description="Hypotheses considered but not chosen as the root cause.",
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description=(
            "Structured evidence records supporting the diagnosis, each "
            "grounded in a real tool call via source_ref."
        ),
    )
    # NOTE: named `diagnostic_confidence`, not `confidence` — BUILD_PLAN.md
    # is explicit that this is a model-reported heuristic, not a calibrated
    # probability, and the field name change is deliberate so nothing
    # downstream (Phase 5's confidence-gap loop, Phase 7's eval report)
    # mistakes it for a real likelihood.
    diagnostic_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "The model's own heuristic confidence in root_cause_category, "
            "0.0-1.0. This is NOT a calibrated probability -- it is a "
            "self-reported estimate only, useful as a secondary/tie-break "
            "signal and a display value, never as a real likelihood of "
            "correctness."
        ),
    )
