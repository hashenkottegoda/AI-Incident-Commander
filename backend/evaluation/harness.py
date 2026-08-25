"""Phase 7's experiment harness (BUILD_PLAN.md Phase 7) -- wraps each of the
four already-committed experiment functions/coroutines (Experiments A/B/C/D)
to capture wall-clock latency, a tool-call count, and Claude API token usage
alongside the `DiagnosisResult` each one already returns, WITHOUT modifying
any of their signatures:

- `backend.evaluation.experiment_a.run_context_stuffing_baseline` (A)
- `backend.agents.investigator.investigate_incident` (B when
  `include_rag=False`, C when `include_rag=True`)
- `backend.graph.run_incident_graph_to_diagnosis` (D -- NOT
  `run_incident_graph`; see `run_experiment_d`'s docstring below for why
  the plain end-to-end operational entry point is unsafe to reuse for
  diagnostic-only scoring)

`scoring.py`'s `tool_call_efficiency` docstring names this module directly:
*"B/C: the harness will need to make investigate_incident surface a
count ... that plumbing is the not-yet-built experiment runner's job."*
This module is that plumbing -- not by changing `investigate_incident`'s
signature (it is used directly by `POST /api/incidents/{id}/investigate`
and must keep its existing contract), but by observing it from the
outside.

## The capture mechanism: `langchain_core.tracers.context.collect_runs()`

Verified against the installed `langchain-core==1.6.0` (see `uv.lock`) with
a throwaway script before writing this module (per this task's own
instructions) -- summary of what was verified and why it works:

`collect_runs()` (`langchain_core/tracers/context.py`) pushes a
`RunCollectorCallbackHandler` onto a `ContextVar`
(`run_collector_var`) that is registered via
`register_configure_hook(run_collector_var, inheritable=False)` at import
time. Every LangChain `Runnable.invoke()`/`ainvoke()` call -- a
`ChatAnthropic` LLM call, a real `BaseTool.invoke()` call, a
`RunnableSequence` built by `.with_structured_output()` -- goes through
`ensure_config()`/the callback-manager configuration path, which reads
`_configure_hooks` and silently attaches whatever callback handler is
sitting in that context var, with ZERO change to the call site. This is
exactly "ambient tracing": nothing inside `investigate_incident`/
`run_context_stuffing_baseline`/`run_incident_graph` needs to accept or
thread through a `config`/`callbacks` parameter for this to work, so their
existing signatures are untouched.

The throwaway verification script (a real `BaseChatModel` subclass playing
back scripted `AIMessage`s carrying real `usage_metadata`, monkeypatched in
place of `ChatAnthropic`, run through the REAL `backend.agents.investigator
.investigate_incident` with REAL `BaseTool` instances from
`backend.tools.build_tools(db)`) confirmed, wrapped in `with
collect_runs() as cb:`:

- Every LLM turn of the ReAct loop shows up as its own `run_type="llm"`
  entry in `cb.traced_runs`, each carrying the exact scripted
  `usage_metadata` (input/output tokens) inside
  `run.outputs["generations"][i][j]["message"]["kwargs"]["usage_metadata"]`
  -- the same place any real `ChatAnthropic` response's `AIMessage.
  usage_metadata` (which `langchain-anthropic` populates from the real
  API's `usage` field) serializes to under tracing.
- The final `.with_structured_output(DiagnosisResult).invoke(...)` call
  shows up as a `run_type="chain"` root run (a `RunnableSequence` of the
  model call + output-parsing step) whose *nested* LLM call is reachable
  via `run.child_runs`, NOT as another top-level entry in `cb.traced_runs`
  -- `RunCollectorCallbackHandler` only stores root runs; everything
  nested is reachable by walking `child_runs`. `_all_llm_runs`/`_all_
  tool_runs` below recurse for exactly this reason.
- The single real `get_logs` `BaseTool.invoke()` call made inside the
  ReAct loop shows up as its own `run_type="tool"` root run with
  `run.name == "get_logs"` -- a genuine, ambient-traced tool call, not
  something this module has to instrument `investigator.py` to report.
- A second throwaway script confirmed the same mechanism survives an
  `async def` coroutine awaiting `Runnable.ainvoke()` from inside a freshly
  created `asyncio.Task` (mirroring how LangGraph's async Pregel executor
  runs node coroutines) -- required for Experiment D, which is async
  (`run_incident_graph`). `contextvars.Context` is captured by
  `asyncio.Task` at creation time, so a context var set before `await
  compiled.ainvoke(...)` remains visible inside every node the graph
  schedules during that call.

Both scripts are NOT part of this module or the test suite (throwaway, per
this task's instructions) -- their point was to settle "does the
callback-free `collect_runs()` approach genuinely work against the
installed version" BEFORE committing to it as this module's mechanism, per
this task's own escape hatch ("if this approach turns out not to work ...
fall back to a `BaseCallbackHandler` ... but note this WOULD require the
three existing functions to accept/thread through a `config`/`callbacks`
parameter"). It works, so that fallback is not needed and none of the
three experiment functions were touched.

## Why token totals can legitimately be `None`

`total_input_tokens`/`total_output_tokens` are `int | None`. They are
`None` only when NO `run_type in {"llm", "chat_model"}` run was found
anywhere in the collected run tree for that call -- i.e. the capture
mechanism itself found nothing to sum, not "found some usage-less calls
and treated them as zero." This should never happen for a real
`ChatAnthropic` call (every real API response carries `usage`), but stays
`None` rather than `0` for a fake/test double whose scripted `AIMessage`s
happen to omit `usage_metadata` entirely, so a genuine "no usage data
available" is never silently indistinguishable from "zero tokens used" in
the A/B/C/D comparison table.

## Tool-call counting: uniform for A/B/C, `IncidentState`-derived for D

A/B/C: `tool_call_count` is the number of `run_type == "tool"` runs found
anywhere in the collected tree -- real for B/C (their ReAct loop calls
real `BaseTool` instances), naturally `0` for A by construction (Experiment
A calls the plain `get_logs`/`get_metrics`/`get_deployments` *functions*
directly, never the LangChain-wrapped tool versions -- see
`experiment_a`'s "Tool functions, not tools" docstring section -- so there
is nothing of `run_type == "tool"` for A to ever collect).

D: `IncidentState.tool_call_log_ids` (`backend.agents.state`) already gives
Experiment D a real, purpose-built tool-call count with zero derivation
needed -- `len(final_state.tool_call_log_ids)` is used directly rather than
counted from the collected run tree, per this task's explicit guidance.
Latency and token capture are still applied uniformly via `collect_runs()`
around the whole graph invocation, for the same reason B/C's are: the
graph's nodes make real `ChatAnthropic` calls too, and there is no cheaper
way to observe their token usage from outside `backend/graph.py` without
touching its nodes' internals.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal, NamedTuple

from langchain_core.tracers.context import collect_runs

from backend.agents.investigator import investigate_incident
from backend.agents.schemas import DiagnosisResult
from backend.evaluation.experiment_a import run_context_stuffing_baseline
from backend.graph import run_incident_graph_to_diagnosis

if TYPE_CHECKING:
    from langchain_core.tracers.schemas import Run
    from qdrant_client import QdrantClient
    from sqlalchemy.orm import Session

    from backend.agents.state import IncidentState
    from backend.models import Incident

ExperimentId = Literal["A", "B", "C", "D"]

# `run_type` values LangChain's tracer assigns to a real LLM call --
# "chat_model" is the modern value, "llm" is the legacy/generic one; both
# appear across LangChain's own history and either is possible depending on
# exactly which Runnable path a given call takes, so both are checked (see
# module docstring's verification notes -- the throwaway script observed
# "llm" for `ChatAnthropic`-shaped calls against this installed version).
_LLM_RUN_TYPES = frozenset({"llm", "chat_model"})
_TOOL_RUN_TYPE = "tool"


class ExperimentRun(NamedTuple):
    """Return shape for every `run_experiment_*` function below.

    `diagnosis` is the same `DiagnosisResult` the wrapped experiment
    function/coroutine already returns -- this class only adds the
    measurement fields Phase 7's comparison table also needs (BUILD_PLAN.md:
    "tool-call efficiency, latency, and token cost (from Claude API usage
    fields)").
    """

    diagnosis: DiagnosisResult
    latency_seconds: float
    tool_call_count: int
    # See module docstring's "Why token totals can legitimately be None"
    # section -- `None` means the capture mechanism found no LLM-type run
    # at all, not "found calls with zero usage."
    total_input_tokens: int | None
    total_output_tokens: int | None


def _iter_runs(runs: list[Run]):
    """Depth-first walk of every run collected by `collect_runs()`,
    including nested runs reached only via `run.child_runs` -- see module
    docstring: `RunCollectorCallbackHandler` only stores ROOT runs in
    `traced_runs`, so a run nested inside e.g. a `.with_structured_output()`
    `RunnableSequence` is only reachable this way."""
    for run in runs:
        yield run
        yield from _iter_runs(run.child_runs)


def _run_usage_tokens(run: Run) -> tuple[int, int] | None:
    """Extract `(input_tokens, output_tokens)` from one `run_type in
    {"llm", "chat_model"}` run's serialized outputs, or `None` if this
    particular run carries no `usage_metadata` (see module docstring for
    the observed shape: `run.outputs["generations"][i][j]["message"]
    ["kwargs"]["usage_metadata"]`, the same place any AIMessage's
    `usage_metadata` attribute -- which `langchain-anthropic` populates
    from the real API's `usage` field -- serializes to under tracing)."""
    if not run.outputs:
        return None
    generations = run.outputs.get("generations")
    if not generations:
        return None
    total_in = 0
    total_out = 0
    found_any = False
    for generation_batch in generations:
        for generation in generation_batch:
            message = generation.get("message") if isinstance(generation, dict) else None
            kwargs = message.get("kwargs", {}) if isinstance(message, dict) else {}
            usage = kwargs.get("usage_metadata")
            if not usage:
                continue
            found_any = True
            total_in += usage.get("input_tokens", 0) or 0
            total_out += usage.get("output_tokens", 0) or 0
    return (total_in, total_out) if found_any else None


def _aggregate_usage(root_runs: list[Run]) -> tuple[int | None, int | None]:
    """Sum token usage across every LLM-type run anywhere in the collected
    tree (every ReAct-loop turn PLUS the final structured-output call, for
    B/C/D) -- `(None, None)` if no LLM-type run carried any usage data at
    all (see `ExperimentRun`'s docstring)."""
    input_total = 0
    output_total = 0
    found_any = False
    for run in _iter_runs(root_runs):
        if run.run_type not in _LLM_RUN_TYPES:
            continue
        usage = _run_usage_tokens(run)
        if usage is None:
            continue
        found_any = True
        input_total += usage[0]
        output_total += usage[1]
    if not found_any:
        return None, None
    return input_total, output_total


def _count_tool_runs(root_runs: list[Run]) -> int:
    """Count every `run_type == "tool"` run anywhere in the collected tree
    -- see module docstring's "Tool-call counting" section for why this is
    the right count for A/B/C but NOT used for D (D uses
    `IncidentState.tool_call_log_ids` directly instead)."""
    return sum(1 for run in _iter_runs(root_runs) if run.run_type == _TOOL_RUN_TYPE)


def run_experiment_a(db: Session, incident: Incident) -> ExperimentRun:
    """Wrap Experiment A (`run_context_stuffing_baseline`). `tool_call_count`
    is derived from run collection the same uniform way as B/C, not
    hardcoded -- see module docstring: it is naturally `0` for A "by
    construction" because Experiment A never calls a LangChain-wrapped
    tool, so this is a genuine confirmation, not an assumption."""
    start = time.perf_counter()
    with collect_runs() as collector:
        diagnosis = run_context_stuffing_baseline(db, incident)
    latency = time.perf_counter() - start

    input_tokens, output_tokens = _aggregate_usage(collector.traced_runs)
    tool_call_count = _count_tool_runs(collector.traced_runs)

    return ExperimentRun(
        diagnosis=diagnosis,
        latency_seconds=latency,
        tool_call_count=tool_call_count,
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
    )


def _run_investigator(db: Session, incident: Incident, *, include_rag: bool) -> ExperimentRun:
    """Shared implementation for `run_experiment_b`/`run_experiment_c` --
    both call the SAME `investigate_incident(db, incident, include_rag=...)`
    (per that function's own docstring: it serves as the implementation of
    both Experiments B and C, selected via the keyword-only flag), so the
    measurement wrapper only needs to exist once."""
    start = time.perf_counter()
    with collect_runs() as collector:
        diagnosis = investigate_incident(db, incident, include_rag=include_rag)
    latency = time.perf_counter() - start

    input_tokens, output_tokens = _aggregate_usage(collector.traced_runs)
    tool_call_count = _count_tool_runs(collector.traced_runs)

    return ExperimentRun(
        diagnosis=diagnosis,
        latency_seconds=latency,
        tool_call_count=tool_call_count,
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
    )


def run_experiment_b(db: Session, incident: Incident) -> ExperimentRun:
    """Experiment B: `investigate_incident(..., include_rag=False)` --
    tools, no RAG."""
    return _run_investigator(db, incident, include_rag=False)


def run_experiment_c(db: Session, incident: Incident) -> ExperimentRun:
    """Experiment C: `investigate_incident(..., include_rag=True)` --
    tools + historical incidents."""
    return _run_investigator(db, incident, include_rag=True)


def diagnosis_result_from_state(final_state: IncidentState) -> DiagnosisResult:
    """Build a `DiagnosisResult` from a graph's final `IncidentState`.

    Mirrors `backend/api/incidents.py`'s `POST /{incident_id}/investigate/
    graph` route field-for-field (that route builds this same construction
    inline since there was previously no shared helper to call) -- kept
    identical here rather than diverging, per this task's instructions.
    `api/incidents.py` itself is not modified to import this helper: Phase
    7 is scoped to the harness, not a refactor of an already-committed,
    working route.
    """
    return DiagnosisResult(
        root_cause_category=final_state.root_cause or "unknown",
        hypotheses=final_state.hypotheses,
        alternative_hypotheses=final_state.alternative_hypotheses,
        evidence=final_state.evidence,
        diagnostic_confidence=final_state.diagnostic_confidence,
    )


async def run_experiment_d(
    db: Session, incident: Incident, *, qdrant_client: QdrantClient | None = None
) -> ExperimentRun:
    """Wrap Experiment D (`run_incident_graph_to_diagnosis`) -- async,
    unlike A/B/C.

    Calls `backend.graph.run_incident_graph_to_diagnosis`, NOT
    `run_incident_graph` -- see that function's docstring for why: a plain
    `run_incident_graph` call does not stop at Root Cause (`response_planner`
    runs unconditionally right after it, before the graph can return in
    either the SAFE or HIGH_IMPACT branch), which would leak an extra real
    `ChatAnthropic` call's latency/tokens into this "immediately after RCA"
    diagnostic measurement -- exactly what BUILD_PLAN.md's diagnostic-
    evaluation section says must not happen ("scored immediately after the
    RCA stage, before any response/remediation, so response planning can't
    inflate a diagnosis score"). `run_incident_graph_to_diagnosis` compiles
    the identical graph with `interrupt_before=["response_planner"]`, a
    LangGraph static breakpoint that halts execution before
    `response_planner` ever starts (see that function's docstring for why
    `interrupt_after=["root_cause"]` was tried first and rejected -- it
    fires after every pass of the Phase 5 re-investigation loop, not just
    the final one).

    `tool_call_count` comes directly from
    `len(final_state.tool_call_log_ids)` (`backend.agents.state.
    IncidentState`'s already-plumbed per-investigation ordinal list), NOT
    from counting `run_type == "tool"` runs the way A/B/C do -- see module
    docstring's "Tool-call counting" section. Latency and token usage are
    still captured via the same `collect_runs()` mechanism wrapped around
    the whole `await run_incident_graph_to_diagnosis(...)` call, verified
    (via a second throwaway script) to survive the `asyncio.Task`
    boundaries LangGraph's async executor creates internally.
    """
    start = time.perf_counter()
    with collect_runs() as collector:
        final_state = await run_incident_graph_to_diagnosis(
            db, incident, qdrant_client=qdrant_client
        )
    latency = time.perf_counter() - start

    input_tokens, output_tokens = _aggregate_usage(collector.traced_runs)

    return ExperimentRun(
        diagnosis=diagnosis_result_from_state(final_state),
        latency_seconds=latency,
        tool_call_count=len(final_state.tool_call_log_ids),
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
    )
