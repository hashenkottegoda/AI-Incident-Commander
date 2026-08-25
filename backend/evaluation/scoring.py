"""Phase 7's diagnostic scoring functions (BUILD_PLAN.md "Diagnostic
evaluation" section) -- pure, deterministic functions over a `DiagnosisResult`
(`backend.agents.schemas`), the shared output schema all four experiments
(A/B/C/D) emit.

BUILD_PLAN.md's exact scoring definitions this module implements:

    "root-cause accuracy = predicted_category == ground_truth_category (enum
    equality); evidence precision = valid source_refs / total cited, checked
    against the tool-call log; hallucination rate = cited source_refs that
    don't correspond to any real tool output; plus tool-call efficiency,
    latency, and token cost (from Claude API usage fields)."

## No `ToolCallLog` table -- how "checked against the tool-call log" is done here

BUILD_PLAN.md's data model (`backend/models/`) never defines a table that
records *queries made* -- only `LogEntry` / `MetricPoint` / `Deployment` /
`TraceLite`, the tables tools *query*. `backend.agents.state.IncidentState`'s
docstring confirms this explicitly: `tool_call_log_ids` is "a compact,
monotonically increasing per-investigation ordinal ... not a Postgres foreign
key," precisely because no such table exists.

So "checked against the tool-call log" is implemented here as the pragmatic,
already-available equivalent: every real tool call reads real rows with real
integer primary keys, so a `source_ref.record_id` is "valid" if and only if a
row with that id genuinely exists in the table the *named* tool queries. This
is strictly checkable (it can't be gamed by citing a real id under the wrong
tool -- see `evidence_source_ref_is_valid`'s wrong-tool test case) and needs
no new schema. `search_historical_incidents` gets the analogous treatment
against the seeded `historical_incidents/historical_incidents.yaml` ids.

## What this module does NOT implement, and why

- **Latency and token cost.** BUILD_PLAN.md groups these with tool-call
  efficiency in one sentence, but both require capturing wall-clock timing
  and Claude API `usage` metadata *around an actual LLM call* -- that's the
  job of the not-yet-built experiment runner
  (`backend.evaluation.run_experiments`, a later Phase 7 sub-step), not this
  pure-scoring module, which must stay callable with zero LLM calls and zero
  network I/O so it can be unit-tested fast and free (see `tests/test_scoring.py`).
- **The A/B/C/D comparison table itself and dataset/experiment wiring.**
  Also later Phase 7 sub-steps; this module only supplies the per-run scoring
  primitives that table will aggregate.
"""

from __future__ import annotations

import enum
import functools
from typing import NamedTuple

from sqlalchemy.orm import Session

from backend.agents.schemas import DiagnosisResult, SourceRef
from backend.models import Deployment, LogEntry, MetricPoint, TraceLite
from backend.rag.historical_incidents import load_historical_incidents

# --------------------------------------------------------------------------
# root-cause accuracy
# --------------------------------------------------------------------------


def root_cause_accuracy(predicted: DiagnosisResult, ground_truth_category: str) -> bool:
    """BUILD_PLAN.md: "root-cause accuracy = predicted_category ==
    ground_truth_category (enum equality)". Plain equality on the closed
    `RootCauseCategory` literal -- no fuzzy/substring matching, so an
    `"unknown"` prediction never accidentally "matches" a real category and
    vice versa.
    """
    return predicted.root_cause_category == ground_truth_category


# --------------------------------------------------------------------------
# evidence_source_ref_is_valid
# --------------------------------------------------------------------------

# The 4 DB-backed tools and the SQLAlchemy model each one queries
# (see `backend/tools/*.py`: get_logs -> LogEntry, get_metrics -> MetricPoint,
# get_deployments -> Deployment, get_dependencies -> TraceLite). A
# `source_ref.record_id` is only "valid" if it exists as a real primary key
# in the ONE table its named tool actually reads -- citing a real id from the
# wrong table is exactly the kind of citation this metric must catch.
_DB_BACKED_TOOL_MODELS: dict[str, type] = {
    "get_logs": LogEntry,
    "get_metrics": MetricPoint,
    "get_deployments": Deployment,
    "get_dependencies": TraceLite,
}

# `search_historical_incidents` (backend.tools.historical_incidents) is not
# DB-backed (Qdrant, not Postgres) and its records use string ids
# (`hist-001`, ...), so it gets its own branch rather than a row in the dict
# above.
_HISTORICAL_INCIDENTS_TOOL = "search_historical_incidents"


@functools.lru_cache(maxsize=1)
def _historical_incident_ids() -> frozenset[str]:
    """The real seeded historical-incident ids (`historical_incidents/historical_incidents.yaml`),
    loaded once and cached -- this module makes zero LLM/network calls, but
    would otherwise re-parse the same ~20-row YAML file on every evidence
    item of every evaluated result. The file is static seed data for the
    lifetime of a process, so caching is safe.
    """
    return frozenset(incident.id for incident in load_historical_incidents())


class SourceRefVerdict(enum.Enum):
    """Three-way classification underlying `evidence_source_ref_is_valid`
    and the precision/hallucination-rate metrics.

    An earlier version of this module scored a `query`-only citation under
    a recognized DB-backed tool as unconditionally VALID ("the tool name
    is real, benefit of the doubt on the rest"). Code review caught a real
    gaming loophole in that: a model could cite every piece of fabricated
    evidence via `query="..."` instead of `record_id=<N>` and never be
    caught by `hallucination_rate`, since only `record_id` citations were
    ever checked. "Unverifiable" and "valid" are not the same thing, and
    collapsing them rewarded avoiding the one checkable field.

    The fix: a genuine third state. `UNVERIFIABLE` citations are excluded
    from both the numerator and the denominator of `evidence_precision` --
    neither rewarded as grounded evidence nor punished as a fabrication,
    since this module genuinely cannot re-execute a natural-language query
    to confirm it reproduces the same result shape. This closes the
    query-only loophole (it no longer inflates precision) without
    penalizing the legitimate "queried and found nothing" case the same as
    an outright fabrication.
    """

    VALID = "valid"
    HALLUCINATED = "hallucinated"
    UNVERIFIABLE = "unverifiable"


def _historical_source_ref_is_valid(source_ref: SourceRef) -> bool:
    """Validity rule for `tool="search_historical_incidents"` citations.

    `SourceRef.record_id` is typed `int | None` -- a field shared across
    every tool -- but every seeded historical incident id is a *string*
    like `"hist-001"` (`backend.rag.historical_incidents.HistoricalIncident.id`).
    An int can never legitimately equal one of these ids, so a populated
    `record_id` under this tool is always invalid, full stop (fail-safe:
    this is a stronger guarantee than "unverifiable", it is a structural
    type mismatch that can only mean a fabricated or misused citation).

    Because there is no int-typed id to cite, evidence from this tool must
    use `query` to name which historical incident it cites (e.g.
    `query="hist-003"`) -- this is the one case in this module where the
    `query` field is checked against real data rather than given the
    generic "unverifiable but not flatly invalid" benefit of the doubt (see
    `evidence_source_ref_is_valid`'s docstring): the task explicitly
    requires being able to tell a real seeded id apart from a fabricated
    one here, which the generic query-only leniency would not do.
    """
    if source_ref.record_id is not None:
        return False
    if source_ref.query is not None and source_ref.query.strip():
        return source_ref.query.strip() in _historical_incident_ids()
    return False


def classify_source_ref(db: Session, source_ref: SourceRef) -> SourceRefVerdict:
    """Classify `source_ref` as VALID, HALLUCINATED, or UNVERIFIABLE.

    Dispatches on `source_ref.tool`:

    1. **`search_historical_incidents`** -- see `_historical_source_ref_is_valid`.
       Always resolves to VALID or HALLUCINATED, never UNVERIFIABLE: a
       specific, checkable ground truth (the seeded id set) exists here,
       so there's no "can't verify" case the way there is for the 4
       DB-backed tools' free-text `query`.
    2. **One of the 4 DB-backed tools** (`get_logs`/`get_metrics`/
       `get_deployments`/`get_dependencies`):
       - `record_id` set -> VALID iff a row with that exact id exists in
         the table *that tool* queries (checked via `Session.get`, a direct
         primary-key lookup). Citing a genuinely real id from a *different*
         table under the wrong tool name is HALLUCINATED -- e.g. a real
         `LogEntry.id` cited under `tool="get_metrics"` fails, because that
         id was never returned by a `get_metrics` call.
       - `record_id` is `None` but `query` is a non-empty string ->
         UNVERIFIABLE -- see the "query-only case" note below.
       - Neither set -- HALLUCINATED; nothing to check against and no
         claim of unverifiable-but-real evidence being made either.
    3. **Any other/unrecognized tool name** -- always HALLUCINATED,
       regardless of `record_id`/`query`. This mirrors this codebase's
       established fail-safe default-deny philosophy
       (`backend.agents.risk_classifier.classify_risk`'s case 3,
       `backend.agents.action_executor_node`'s guard on unknown action
       types): an unrecognized tool name could be a typo'd real tool or an
       invented one, and there is no way to distinguish those, so it is
       never given the benefit of the doubt.

    ## The query-only case -> UNVERIFIABLE, not VALID

    `SourceRef.query` exists (per its docstring in `backend.agents.schemas`)
    for evidence "about the shape or absence of a result rather than one
    specific record id" -- e.g. "queried get_logs for payment-service
    12:00-12:05, found zero rows" is real, checkable-in-spirit evidence with
    no single record id to cite. This module cannot re-execute an arbitrary
    natural-language query against the DB to verify it reproduces the same
    shape, so for the 4 DB-backed tools, a non-empty `query` string under a
    *recognized* tool name resolves to UNVERIFIABLE, not VALID -- see
    `SourceRefVerdict`'s docstring for why an earlier version's "valid"
    treatment here was a real scoring loophole (it let a model game
    `hallucination_rate` by always citing `query` instead of `record_id`).
    UNVERIFIABLE is excluded from `evidence_precision`'s denominator
    entirely, rather than being counted as either grounded or fabricated.

    This does NOT extend to `search_historical_incidents`, where `query` is
    checked against real seeded ids instead (see
    `_historical_source_ref_is_valid`) -- there, a specific, checkable
    ground truth exists, so a real answer (VALID/HALLUCINATED) is always
    possible.
    """
    tool = source_ref.tool

    if tool == _HISTORICAL_INCIDENTS_TOOL:
        return (
            SourceRefVerdict.VALID
            if _historical_source_ref_is_valid(source_ref)
            else SourceRefVerdict.HALLUCINATED
        )

    model = _DB_BACKED_TOOL_MODELS.get(tool)
    if model is None:
        return SourceRefVerdict.HALLUCINATED

    if source_ref.record_id is not None:
        return (
            SourceRefVerdict.VALID
            if db.get(model, source_ref.record_id) is not None
            else SourceRefVerdict.HALLUCINATED
        )

    if source_ref.query is not None and source_ref.query.strip():
        return SourceRefVerdict.UNVERIFIABLE

    return SourceRefVerdict.HALLUCINATED


def evidence_source_ref_is_valid(db: Session, source_ref: SourceRef) -> bool:
    """`True` iff `classify_source_ref(db, source_ref)` is VALID.

    A query-only citation under a recognized DB-backed tool is
    UNVERIFIABLE, not VALID, so this returns `False` for that case -- see
    `classify_source_ref`'s "query-only case" note. Use
    `classify_source_ref` directly when the UNVERIFIABLE distinction
    matters (as `evidence_precision`/`hallucination_rate` do); this
    boolean form is for simple valid/not-valid checks.
    """
    return classify_source_ref(db, source_ref) is SourceRefVerdict.VALID


# --------------------------------------------------------------------------
# evidence_precision / hallucination_rate
# --------------------------------------------------------------------------


def evidence_precision(db: Session, result: DiagnosisResult) -> float:
    """Fraction of `result.evidence`'s VERIFIABLE citations that are valid.

    UNVERIFIABLE citations (query-only evidence under a recognized
    DB-backed tool -- see `classify_source_ref`) are excluded from both the
    numerator and the denominator: neither rewarded as grounded evidence
    nor punished as a fabrication, since this module genuinely cannot
    re-execute a natural-language query to confirm it. Excluding them (as
    opposed to an earlier version of this module that counted them as
    valid) is what prevents a model from gaming this metric by always
    citing `query` instead of `record_id` -- see `SourceRefVerdict`'s
    docstring for the full reasoning.

    ## Convention: no VERIFIABLE evidence at all -> 0.0, not 1.0 / not undefined

    This covers two cases identically: a genuinely empty `evidence` list,
    and an `evidence` list where every item is UNVERIFIABLE (so nothing
    remains after exclusion). Either way there is zero checkable grounding
    for the `root_cause_category`, which BUILD_PLAN.md's whole premise for
    this system treats as a failure of evidence-backed RCA (Agent
    Architecture: "Structured evidence & RCA (required for deterministic
    eval)"), not a neutral non-event. Scoring it as 1.0 would reward
    silence (or an all-unverifiable citation strategy) over a bad citation,
    which is backwards; scoring it as `None`/NaN would require every
    downstream aggregation (the A/B/C/D comparison table) to special-case
    missing values instead of just averaging a float column. 0.0 keeps the
    metric a total function (`Session, DiagnosisResult -> float`,
    always defined) and keeps its arithmetic complement
    (`hallucination_rate`) meaningful too -- see that function's docstring.
    """
    verdicts = [classify_source_ref(db, item.source_ref) for item in result.evidence]
    verifiable = [v for v in verdicts if v is not SourceRefVerdict.UNVERIFIABLE]
    if not verifiable:
        return 0.0
    valid_count = sum(1 for v in verifiable if v is SourceRefVerdict.VALID)
    return valid_count / len(verifiable)


def hallucination_rate(db: Session, result: DiagnosisResult) -> float:
    """BUILD_PLAN.md's hallucination rate, as its own named function (not
    inlined `1 - evidence_precision(...)` at every call site) because it is
    a separately-called-out metric in the comparison table and deserves its
    own docstring/call site rather than being an implicit detail readers
    have to notice.

    Implemented as the direct complement of `evidence_precision`:
    `1.0 - evidence_precision(db, result)`. This is a deliberate design
    choice, not an accident of the two metrics happening to agree today --
    within the VERIFIABLE subset of cited evidence, "valid" and
    "hallucinated" are an exhaustive binary partition (every non-
    UNVERIFIABLE `source_ref` is one or the other, never both, never
    neither), so precision and hallucination rate can never independently
    drift apart. Given the 0.0-for-no-verifiable-evidence convention above,
    a diagnosis with zero verifiable cited evidence (whether because it
    cited nothing, or cited only UNVERIFIABLE query-only evidence) scores a
    1.0 hallucination rate -- read as "0% of what was claimed is grounded",
    consistent with precision's "evidence-backed RCA is the whole point"
    framing, not a claim that the model literally invented something out of
    thin air.
    """
    return 1.0 - evidence_precision(db, result)


# --------------------------------------------------------------------------
# tool_call_efficiency
# --------------------------------------------------------------------------


class ToolCallEfficiency(NamedTuple):
    """Return shape for `tool_call_efficiency` -- see that function's
    docstring for the scoping decision behind these two fields."""

    tool_call_count: int
    # `None`, not `0.0`, when `tool_call_count == 0` -- "0 evidence per call"
    # would misleadingly suggest zero calls were maximally *inefficient*,
    # when a 0-call run (Experiment A, by architectural design) has no
    # calls to be efficient or inefficient *about* -- the ratio is undefined,
    # not zero. See `tool_call_efficiency`'s docstring.
    evidence_per_tool_call: float | None


def tool_call_efficiency(tool_call_count: int, evidence_count: int) -> ToolCallEfficiency:
    """Tool-call efficiency, pragmatically scoped to what is ACTUALLY
    plumbed through today.

    ## Why this takes explicit ints rather than deriving anything from `DiagnosisResult`

    `DiagnosisResult` (the shared A/B/C/D output schema) carries no record of
    how many tool calls produced it -- it is a pure diagnosis, not a trace of
    the investigation that led to it. The per-architecture reality today:

    - **Experiment A** makes zero tool calls by design (`backend.evaluation
      .experiment_a`'s whole premise is "no tools, no selective retrieval").
    - **Experiments B/C** (`backend.agents.investigator.investigate_incident`)
      run a ReAct tool loop internally but the function's return type is
      just `DiagnosisResult` -- it does not currently return a call count.
    - **Experiment D** (`backend.graph.run_incident_graph`) is the one
      architecture that *does* have something usable already:
      `IncidentState.tool_call_log_ids` (`backend.agents.state`), a
      per-investigation ordinal list assigned to each tool call as it
      happens -- `len(state.tool_call_log_ids)` is a real, already-plumbed
      call count for D.

    So this function does not try to derive a count from `DiagnosisResult`
    alone (impossible for B/C today, and would silently be wrong/undefined
    for A). Instead it accepts `tool_call_count` as an explicit parameter
    that the caller supplies per run:

    - A: the harness passes `0` (true by construction).
    - B/C: the harness will need to make `investigate_incident` surface a
      count (e.g. count `ToolMessage`s in the final message list, or have it
      return the count alongside `DiagnosisResult`) -- that plumbing is the
      not-yet-built experiment runner's job, not this module's.
    - D: the harness passes `len(final_state.tool_call_log_ids)`, already
      available today with zero new plumbing.

    ## What "efficiency" means here

    Two figures, not one, because a single ratio can't cleanly express "zero
    calls" (Experiment A) without a divide-by-zero decision that would color
    the number either way:

    - `tool_call_count`: the raw count itself. Directly comparable across
      A(=0)/B/C/D in the comparison table -- "how much investigation work did
      this architecture need to reach its diagnosis" is meaningful on its
      own, independent of any ratio.
    - `evidence_per_tool_call` = `evidence_count / tool_call_count`
      (`None` when `tool_call_count == 0`): how much cited evidence was
      extracted per call made -- distinguishes a ReAct loop that fires off
      many calls but surfaces little evidence from one that investigates
      efficiently. `None` (not `0.0`) for zero calls, since the ratio is
      undefined there, not zero -- see `ToolCallEfficiency`'s docstring.

    Raises:
        ValueError: either argument is negative.
    """
    if tool_call_count < 0:
        raise ValueError(f"tool_call_count must be >= 0, got {tool_call_count!r}")
    if evidence_count < 0:
        raise ValueError(f"evidence_count must be >= 0, got {evidence_count!r}")

    ratio = evidence_count / tool_call_count if tool_call_count > 0 else None
    return ToolCallEfficiency(tool_call_count=tool_call_count, evidence_per_tool_call=ratio)
