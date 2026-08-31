"""Phase 7's experiment harness (BUILD_PLAN.md Phase 7) -- wraps each of the
four already-committed experiment functions/coroutines (Experiments A/B/C/D)
to capture wall-clock latency, a tool-call count, and OpenRouter API token usage
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

`collect_runs()` (`langchain_core/tracers/context.py`) pushes a
`RunCollectorCallbackHandler` onto a `ContextVar` (`run_collector_var`)
that is registered via `register_configure_hook(run_collector_var,
inheritable=False)` at import time. Every LangChain `Runnable.invoke()`/
`ainvoke()` call -- a `ChatOpenRouter` LLM call, a real `BaseTool.invoke()`
call, a `RunnableSequence` built by `.with_structured_output()` -- goes
through `ensure_config()`/the callback-manager configuration path, which
reads `_configure_hooks` and silently attaches whatever callback handler is
sitting in that context var, with ZERO change to the call site. This is
"ambient tracing": nothing inside `investigate_incident`/
`run_context_stuffing_baseline`/`run_incident_graph` needs to accept or
thread through a `config`/`callbacks` parameter for this to work, so their
existing signatures are untouched -- the alternative (a `BaseCallbackHandler`
threaded explicitly through each call) would have required changing all
three signatures, including `investigate_incident`'s, which also backs
`POST /api/incidents/{id}/investigate` directly.

Wrapped in `with collect_runs() as cb:` around a real ReAct investigation:

- Every LLM turn of the ReAct loop shows up as its own `run_type="llm"`
  entry in `cb.traced_runs`, each carrying `usage_metadata` (input/output
  tokens) inside
  `run.outputs["generations"][i][j]["message"]["kwargs"]["usage_metadata"]`
  -- the same place any LangChain chat model integration (including
  `langchain-openrouter`) populates a real `AIMessage.usage_metadata`
  from the API's `usage` field.
- The final `.with_structured_output(DiagnosisResult).invoke(...)` call
  shows up as a `run_type="chain"` root run (a `RunnableSequence` of the
  model call + output-parsing step) whose *nested* LLM call is reachable
  via `run.child_runs`, NOT as another top-level entry in `cb.traced_runs`
  -- `RunCollectorCallbackHandler` only stores root runs; everything
  nested is reachable by walking `child_runs`. `_all_llm_runs`/`_all_
  tool_runs` below recurse for exactly this reason.
- Each real `BaseTool.invoke()` call made inside the ReAct loop shows up as
  its own `run_type="tool"` root run named after the tool -- a genuine,
  ambient-traced tool call, with no need to instrument `investigator.py`
  itself to report it.
- The same mechanism survives an `async def` coroutine awaiting
  `Runnable.ainvoke()` from inside a freshly created `asyncio.Task`
  (mirroring how LangGraph's async Pregel executor runs node coroutines) --
  required for Experiment D, which is async (`run_incident_graph`).
  `contextvars.Context` is captured by `asyncio.Task` at creation time, so
  a context var set before `await compiled.ainvoke(...)` remains visible
  inside every node the graph schedules during that call.

## Why token totals can legitimately be `None`

`total_input_tokens`/`total_output_tokens` are `int | None`. They are
`None` only when NO `run_type in {"llm", "chat_model"}` run was found
anywhere in the collected run tree for that call -- i.e. the capture
mechanism itself found nothing to sum, not "found some usage-less calls
and treated them as zero." This should never happen for a real
`ChatOpenRouter` call (every real API response carries `usage`), but stays
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
graph's nodes make real `ChatOpenRouter` calls too, and there is no cheaper
way to observe their token usage from outside `backend/graph.py` without
touching its nodes' internals.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, NamedTuple

from langchain_core.tracers.context import collect_runs

from backend.agents.investigator import investigate_incident
from backend.agents.routing import MAX_REINVESTIGATION_LOOPS
from backend.agents.schemas import DiagnosisResult
from backend.evaluation.experiment_a import run_context_stuffing_baseline
from backend.graph import (
    get_incident_thread_state,
    resume_incident_graph,
    run_incident_graph,
    run_incident_graph_to_diagnosis,
)
from backend.models import AuditDecisionStatus, AuditEvent

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
# exactly which Runnable path a given call takes, so both are checked
# (`ChatOpenRouter` calls trace as "llm" against the installed langchain-core
# version -- see `uv.lock`).
_LLM_RUN_TYPES = frozenset({"llm", "chat_model"})
_TOOL_RUN_TYPE = "tool"


class ExperimentRun(NamedTuple):
    """Return shape for every `run_experiment_*` function below.

    `diagnosis` is the same `DiagnosisResult` the wrapped experiment
    function/coroutine already returns -- this class only adds the
    measurement fields Phase 7's comparison table also needs (BUILD_PLAN.md:
    "tool-call efficiency, latency, and token cost (from OpenRouter API usage
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
    `usage_metadata` attribute -- which `langchain-openrouter` populates
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
    `ChatOpenRouter` call's latency/tokens into this "immediately after RCA"
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
    the whole `await run_incident_graph_to_diagnosis(...)` call -- it
    survives the `asyncio.Task` boundaries LangGraph's async executor
    creates internally (see the module docstring's capture-mechanism
    section).
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


# =============================================================================
# Operational evaluation (D only) -- driving the FULL closed loop
# =============================================================================
#
# BUILD_PLAN.md's "Operational evaluation (D only)" section (quoted in full
# in `backend.evaluation.scoring`'s module docstring) needs the real
# Response Planner -> Risk Classifier -> HITL -> Action Executor -> Recovery
# Check loop to have actually run, not `run_experiment_d`'s diagnostic-only
# `run_incident_graph_to_diagnosis` above (which halts before any of that
# starts, on purpose -- see that function's docstring). `scoring.
# score_operational_run(db, final_state, scenario)` already knows how to
# SCORE the result of that loop; nothing before this point in the codebase
# actually DRIVES it unattended (the real driver is `POST /approve`, which
# needs a human/HTTP caller in the loop by design). This section is that
# missing driver, for eval purposes only.

# Defensive circuit breaker against a genuine bug producing an infinite
# pause/auto-approve cycle in `run_experiment_d_operational`'s loop below --
# comfortably above the real bound the graph itself enforces
# (`backend.agents.routing.MAX_REINVESTIGATION_LOOPS`, currently 2: each
# bounded re-investigation pass can produce at most one fresh HIGH_IMPACT
# recommendation needing its own approval round). Hitting this is a bug to
# surface loudly (`RuntimeError`), not a legitimate incident outcome to
# silently keep scoring.
_MAX_APPROVAL_ROUNDS = MAX_REINVESTIGATION_LOOPS + 3


def _auto_approve_pending_actions(db: Session, incident: Incident, approver: str) -> None:
    """Harness-local stand-in for the APPROVED path of
    `backend.api.approvals._decide_pending_actions` -- marks every
    still-`PENDING_APPROVAL` `AuditEvent` row for `incident` as `APPROVED`/
    `approver`/`decided_at` and commits, the same durable write that module
    makes (per its own docstring: "This module is the sole writer of
    AuditEvent.decision_status/.approver/.decided_at for a HIGH_IMPACT
    action") before it goes on to resume the graph.

    ## Why this is a harness-local rewrite, not a shared extraction out of
    ## `_decide_pending_actions`, and not a direct call into that coroutine

    Most of `_decide_pending_actions`'s real content is concurrency/replay
    machinery this single-threaded, one-incident-at-a-time eval harness
    structurally cannot exercise and so should not have to carry:

    - `AuditEvent.version_id` optimistic-concurrency handling / the
      `StaleDataError` catch -- guards against two *concurrent HTTP
      requests* racing on the same incident's pending rows (see that
      module's docstring). This harness never issues two concurrent writes
      against the same incident; there is no race to guard against.
    - `_retry_stuck_resume` -- recovers from a crash *between* the
      `AuditEvent` commit and `resume_incident_graph` completing, so a
      later duplicate `POST /approve` can still unstick the thread. A
      harness run that crashes mid-call aborts the whole run; there is no
      "later duplicate call" to retry from.
    - `_already_decided_response`/the Pydantic `ApprovalResponse` shape --
      an HTTP response contract this harness never returns to anyone. Its
      caller (`run_experiment_d_operational`) already knows, from
      `get_incident_thread_state`, that the thread is genuinely paused
      with something pending -- there is no "was this already decided by
      a different request" ambiguity here the way there is for a duplicate
      HTTP call.

    Reimplementing all of that machinery here just to reuse a few lines of
    `UPDATE`-then-commit would add API-shaped complexity a harness run
    never needs, not remove it. What IS genuinely shared and reused
    directly, not duplicated, is the actual resume call itself --
    `run_experiment_d_operational` calls the real
    `backend.graph.resume_incident_graph` with the exact same
    `Command(resume={"decision": "approved", "approver": approver})`
    payload shape `backend.api.approvals` sends -- so the one piece of
    behavior that genuinely must not drift between the real operational
    path and eval (what a resume payload looks like to
    `human_approval_node`) doesn't.
    """
    pending = (
        db.query(AuditEvent)
        .filter(
            AuditEvent.incident_id == incident.id,
            AuditEvent.decision_status == AuditDecisionStatus.PENDING_APPROVAL,
        )
        .order_by(AuditEvent.id)
        .all()
    )
    now = datetime.now(UTC)
    for event in pending:
        event.decision_status = AuditDecisionStatus.APPROVED
        event.approver = approver
        event.decided_at = now
    db.commit()


async def run_experiment_d_operational(
    db: Session,
    incident: Incident,
    *,
    qdrant_client: QdrantClient | None = None,
    approver: str = "eval-harness",
) -> IncidentState:
    """Drive Phase 6's FULL closed loop for one incident (Response Planner
    -> Risk Classifier -> HITL -> Action Executor -> Recovery Check),
    auto-approving any HIGH_IMPACT plan along the way so
    `backend.evaluation.scoring.score_operational_run(db, final_state,
    scenario)` has a genuinely-executed run to score -- see this module's
    "Operational evaluation (D only)" section above for why nothing before
    this function drove that loop unattended.

    Calls `backend.graph.run_incident_graph` -- the real, unmodified,
    end-to-end operational entry point -- NOT `run_incident_graph_to_
    diagnosis` (`run_experiment_d`'s diagnostic-only entry point above,
    which deliberately halts before Response Planner ever runs; see that
    function's docstring for why that halt is correct for the diagnostic
    comparison table and must stay that way). This function exists
    precisely because the diagnostic path structurally cannot answer "did
    the remediation work" -- it never runs one.

    ## Detecting a genuine pause

    After the initial run and after every resume, this function does NOT
    infer "paused, needs approval" from `incident_status ==
    AWAITING_APPROVAL` -- `backend.graph.get_incident_thread_state`'s own
    docstring explains why that field alone is not proof of a real halt: it
    is set by `response_planner_node` regardless of whether the graph
    actually reached `interrupt()`. Instead this calls
    `get_incident_thread_state(db, incident, qdrant_client=qdrant_client)`
    and checks `snapshot.next == ("human_approval",)`, the same check
    `tests/test_human_approval.py` uses to prove a genuine halt.

    ## Looping until the incident genuinely stops pausing

    A single `resume_incident_graph` call can itself run straight into
    ANOTHER `human_approval` pause: BUILD_PLAN.md's bounded
    re-investigation loop (`recovery_check -> investigation -> rag ->
    root_cause -> response_planner -> human_approval` again) means an
    ineffective first remediation attempt can produce a second HIGH_IMPACT
    recommendation needing its own approval before the incident finally
    resolves or exhausts its budget into `manual_intervention_required`
    (see `tests/test_action_executor_recovery_check.py`'s ineffective-
    remediation case for exactly this, driven through the real API). So
    this function loops: check for a pause, auto-approve + resume if
    paused, repeat, until `get_incident_thread_state` reports no pause left
    -- bounded by `_MAX_APPROVAL_ROUNDS` purely as a defensive circuit
    breaker (see that constant's docstring); hitting it raises
    `RuntimeError` rather than hanging an eval run forever.

    If the FIRST `run_incident_graph` call never pauses at all (an
    all-SAFE plan routes `response_planner -> action_executor` directly,
    per `backend/graph.py`'s conditional edge), the loop condition is never
    true on its first check and `run_incident_graph`'s own final state is
    returned unchanged -- nothing else to do, per this function's spec.

    `incident.status` is kept in sync with each returned `final_state`
    (`incident.status = final_state.incident_status`, followed by
    `db.commit()`) after every resume, matching what
    `backend.api.approvals._decide_pending_actions` does after its own
    resume call. It's also synced after the very first `run_incident_graph`
    call, before any pause/resume -- something the real
    `POST /investigate/graph` path does not do, since `AuditEvent.
    decision_status` (not `incident.status`) is the real gating signal
    there (see `get_incident_thread_state`'s docstring). That extra sync is
    harmless here (nothing in this module gates on `incident.status`) and
    leaves the row more current for a caller who inspects `incident`
    directly rather than only `final_state`.

    Does NOT call `score_operational_run` itself -- scoring stays a
    separate step the caller performs afterward, the same "produce a final
    state, let the caller score it" separation `run_experiment_d`/
    `diagnosis_result_from_state` already use on the diagnostic side (see
    this module's own docstring: `scoring.py` is a distinct module from
    this one throughout this codebase).
    """
    final_state = await run_incident_graph(db, incident, qdrant_client=qdrant_client)
    incident.status = final_state.incident_status
    db.commit()

    rounds = 0
    while True:
        snapshot = await get_incident_thread_state(db, incident, qdrant_client=qdrant_client)
        if snapshot.next != ("human_approval",):
            return final_state

        rounds += 1
        if rounds > _MAX_APPROVAL_ROUNDS:
            raise RuntimeError(
                f"incident {incident.id} is still paused at human_approval after "
                f"{_MAX_APPROVAL_ROUNDS} auto-approval rounds -- likely a genuine bug "
                f"producing an infinite pause/approve cycle, not a legitimate bounded-"
                f"loop outcome (the graph's own MAX_REINVESTIGATION_LOOPS is only "
                f"{MAX_REINVESTIGATION_LOOPS})"
            )

        _auto_approve_pending_actions(db, incident, approver)
        final_state = await resume_incident_graph(
            db,
            incident,
            {"decision": "approved", "approver": approver},
            qdrant_client=qdrant_client,
        )
        incident.status = final_state.incident_status
        db.commit()
