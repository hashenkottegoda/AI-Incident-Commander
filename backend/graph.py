"""Phase 5/6's `StateGraph` assembly: nodes, conditional edges, Postgres
checkpointer.

BUILD_PLAN.md Phase 5: *"Build the StateGraph: Triage -> Investigation loop
-> RAG -> Root Cause/Hypothesis -> conditional re-investigation loop back to
Investigation."* Phase 6 continues the same graph past Root Cause: *"RESPONSE
PLANNER ... -> RISK CLASSIFIER ... -> SAFE -> ACTION EXECUTOR ... HIGH-IMPACT
-> HUMAN APPROVAL (LangGraph `interrupt`, resumed via POST /approve|/reject)
... ACTION EXECUTOR -> RECOVERY CHECK ... RESOLVED | back to INVESTIGATION
(bounded; exhausted -> MANUAL_INTERVENTION_REQUIRED)."* This module now
builds the FULL Phase 6 graph, terminal states and all: Response Planner +
(inline) Risk Classifier, the real `interrupt()`-based Human Approval gate
(`backend.agents.human_approval_node`), the Action Executor
(`backend.agents.action_executor_node`), and the Recovery Check
(`backend.agents.recovery_check_node`) that closes the loop back to
`resolved` / `manual_intervention_required` / a fresh Investigation pass.

```
START -> triage -> investigation -> rag -> root_cause -> response_planner
              ^                        |
              +---- reinvestigate -----+
response_planner -> human_approval (HIGH_IMPACT) -> action_executor
response_planner -> action_executor (SAFE-only, skips human_approval)
action_executor -> END (SAFE-only, nothing to verify)
action_executor -> recovery_check (a HIGH_IMPACT remediation just ran)
recovery_check -> END (RESOLVED)
recovery_check -> investigation (still degraded, budget remains)
recovery_check -> END (MANUAL_INTERVENTION_REQUIRED, budget exhausted)
```

- `response_planner -> human_approval -> action_executor`: any HIGH_IMPACT
  action. `human_approval` calls `interrupt()`, resumed via `POST /approve`
  (`backend.api.approvals`) with `Command(resume=...)`; `action_executor`
  runs strictly after that resume, never before (`interrupt()`'s
  side-effect-safety rule -- see `human_approval_node`'s docstring).
  `POST /reject` never resumes the graph at all, so `action_executor` is
  never reachable from a rejection, by construction (see
  `backend.api.approvals`'s docstring).
- `response_planner -> action_executor` directly: an all-SAFE plan,
  nothing for a human to approve (`route_after_response_planner` routes
  this case straight past `human_approval`).
- `action_executor -> recovery_check`: only when at least one HIGH_IMPACT
  remediation was just executed (`route_after_action_executor`, reading
  `incident_status == VERIFYING`). A SAFE-only plan has nothing to verify
  and routes `action_executor -> END` (`incident_status = DIAGNOSED`)
  instead.
- `recovery_check -> END` (`RESOLVED`) when post-action telemetry matches
  the pre-incident baseline within tolerance; `recovery_check ->
  investigation` (looping back through rag/root_cause/response_planner for
  a fresh attempt) when it doesn't and the bounded re-investigation budget
  (`investigation_iterations` vs. `routing.MAX_REINVESTIGATION_LOOPS` --
  the SAME bound the root-cause loop already uses) hasn't been exhausted;
  `recovery_check -> END` (`MANUAL_INTERVENTION_REQUIRED`) once it has
  (`route_after_recovery_check`).

`build_incident_graph` returns the **uncompiled** `StateGraph` -- callers
attach whichever checkpointer fits their context: `AsyncPostgresSaver` for
the real API path (`run_incident_graph`/`resume_incident_graph` below), no
checkpointer (or an in-memory one) for structural tests that only need to
inspect nodes/edges/predicates without ever invoking the graph.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, StateSnapshot
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from backend.agents.action_executor_node import make_action_executor_node
from backend.agents.human_approval_node import human_approval_node
from backend.agents.investigation_node import make_investigation_node
from backend.agents.rag_node import make_rag_node
from backend.agents.recovery_check_node import make_recovery_check_node
from backend.agents.response_planner_node import make_response_planner_node
from backend.agents.root_cause_node import make_root_cause_node
from backend.agents.routing import (
    route_after_action_executor,
    route_after_recovery_check,
    route_after_response_planner,
    route_after_root_cause,
)
from backend.agents.state import IncidentState
from backend.agents.triage_node import make_triage_node
from backend.config import get_settings
from backend.models import IncidentStatus, NodeProgressEvent
from backend.rag.qdrant_client import get_qdrant_client
from backend.scripts.setup_checkpointer import to_psycopg_dsn

if TYPE_CHECKING:
    from backend.models import Incident


def _with_progress(
    db: Session, node_name: str, fn: Callable[[IncidentState], dict]
) -> Callable[[IncidentState], dict]:
    """Wrap a node function so it writes one `NodeProgressEvent` row before
    running it -- Phase 8's live-trace write path (BUILD_PLAN.md: *"persist
    each graph node's progress to Postgres as it runs and have the
    dashboard poll that progress log"*). Applied once, centrally, to every
    `graph.add_node(...)` call below rather than inside each node module --
    purely additive instrumentation, so no individual node's own logic
    changes.

    Every node function in this graph is a plain sync `def node(state) ->
    dict` (none are `async def` -- see `backend/agents/*_node.py`), so this
    wrapper is sync too: it preserves the exact call signature LangGraph
    already invokes (`fn(state)` in, partial-update `dict` out), it doesn't
    turn a sync node async or vice versa. `db.add(...)` + `db.commit()`
    (not `flush()`) so the row is durable immediately, independent of
    whether the real node function that follows ends up committing,
    rolling back, or raising -- a "this node started" row should survive
    even if the node itself fails.

    Interacts safely with `human_approval_node`'s `interrupt()` gate: per
    that module's docstring, LangGraph re-executes an interrupted node
    **from its start** when the graph resumes, so a resumed `/approve` call
    causes this wrapper to write a second `human_approval` progress row
    before `interrupt()` returns the resume value and the node finishes.
    That's an accurate reflection of what actually happened (the node
    genuinely started running twice -- once to pause, once to resume), and
    an extra durable log-row write is exactly the kind of side effect
    `interrupt()`'s safety rule permits (nothing *irreversible* happens
    here -- see that module's docstring for the distinction) -- unlike
    `AuditEvent` creation, there's no idempotency concern to defend against.
    """

    def wrapped(state: IncidentState) -> dict:
        db.add(NodeProgressEvent(incident_id=state.incident_id, node_name=node_name))
        db.commit()
        return fn(state)

    return wrapped


def build_incident_graph(db: Session, qdrant_client: QdrantClient | None = None) -> StateGraph:
    """Assemble Phase 5's uncompiled `StateGraph`.

    `db`/`qdrant_client` are closed over by the node factories
    (`make_investigation_node(db)`, `make_rag_node(qdrant_client)`) exactly
    like `backend.tools.build_tools(db)`/`build_rag_tools(client)` -- one
    request-scoped session and one Qdrant client per graph build, not
    global state. Building the graph performs no I/O itself (node closures
    only capture references); `qdrant_client` defaults to the process-wide
    cached client (`get_qdrant_client()`, itself connection-less until a
    node actually searches) so callers that don't need a distinct client
    (tests aside) don't have to pass one.
    """
    qdrant_client = qdrant_client or get_qdrant_client()

    graph = StateGraph(IncidentState)
    # Every node is wrapped in `_with_progress` (Phase 8's live-trace write
    # path -- see that function's docstring) -- applied here, centrally,
    # rather than inside each node module, so this is the ONLY place that
    # needs to know about the progress log at all.
    graph.add_node("triage", _with_progress(db, "triage", make_triage_node()))
    graph.add_node(
        "investigation", _with_progress(db, "investigation", make_investigation_node(db))
    )
    graph.add_node("rag", _with_progress(db, "rag", make_rag_node(qdrant_client)))
    graph.add_node("root_cause", _with_progress(db, "root_cause", make_root_cause_node()))
    graph.add_node(
        "response_planner", _with_progress(db, "response_planner", make_response_planner_node(db))
    )
    graph.add_node("human_approval", _with_progress(db, "human_approval", human_approval_node))
    graph.add_node(
        "action_executor", _with_progress(db, "action_executor", make_action_executor_node(db))
    )
    graph.add_node(
        "recovery_check", _with_progress(db, "recovery_check", make_recovery_check_node(db))
    )

    graph.add_edge(START, "triage")
    graph.add_edge("triage", "investigation")
    graph.add_edge("investigation", "rag")
    graph.add_edge("rag", "root_cause")
    graph.add_conditional_edges(
        "root_cause",
        route_after_root_cause,
        {"reinvestigate": "investigation", "end": "response_planner"},
    )
    # SAFE-only plan -> straight to action_executor (no human in the loop
    # needed). Any HIGH_IMPACT action -> the interrupt() gate; see
    # backend.agents.human_approval_node's docstring for why that gate is
    # its own dedicated node rather than folded into response_planner.
    graph.add_conditional_edges(
        "response_planner",
        route_after_response_planner,
        {"human_approval": "human_approval", "end": "action_executor"},
    )
    # human_approval pauses at interrupt() until POST /approve resumes it
    # (backend.api.approvals) -- action_executor runs strictly after that
    # resume, never before (interrupt()'s side-effect-safety rule). A
    # rejection never resumes the graph at all (see backend.api.approvals's
    # docstring), so action_executor is never reachable from that path.
    graph.add_edge("human_approval", "action_executor")
    # A SAFE-only plan has nothing to verify -> straight to END
    # (incident_status = DIAGNOSED). Any executed HIGH_IMPACT remediation
    # -> recovery_check verifies it against real post-action telemetry.
    graph.add_conditional_edges(
        "action_executor",
        route_after_action_executor,
        {"recovery_check": "recovery_check", "end": END},
    )
    # recovery_check is the graph's final decision point: RESOLVED or
    # MANUAL_INTERVENTION_REQUIRED both end the graph; still-degraded with
    # re-investigation budget remaining loops back to a fresh Investigation
    # pass (see route_after_recovery_check's docstring for the shared
    # investigation_iterations bound).
    graph.add_conditional_edges(
        "recovery_check",
        route_after_recovery_check,
        {"investigation": "investigation", "end": END},
    )
    return graph


def initial_state(incident: Incident) -> dict[str, Any]:
    """Build the initial state dict for one `Incident` row.

    `incident_status` starts at `TRIAGING`, not `DETECTED` -- BUILD_PLAN.md's
    Triage node docstring says it should "set incident_status to triaging
    then investigating." `detected` is the pre-graph `Incident` row's
    state; the graph itself starts life already in the `triaging` phase (the
    state as the graph begins, before the Triage node runs and hands off to
    `INVESTIGATING`).

    `severity` and the initial `affected_services` guess are seeded
    straight from the `Incident` row -- realistic detection context from
    the alerting system that created it, not a ground-truth leak (see
    `backend.agents.triage_node`'s docstring). The Triage node confirms or
    expands `affected_services` from there.
    """
    return {
        "incident_id": incident.id,
        "incident_status": IncidentStatus.TRIAGING,
        "severity": incident.severity,
        "affected_services": [incident.service.name],
    }


async def run_incident_graph(
    db: Session,
    incident: Incident,
    *,
    qdrant_client: QdrantClient | None = None,
    database_url: str | None = None,
) -> IncidentState:
    """Compile Phase 5's graph with a Postgres checkpointer and run it
    end-to-end for one incident, returning the final `IncidentState`.

    `thread_id = str(incident.id)` so each incident's graph execution is
    independently checkpointed/resumable -- BUILD_PLAN.md Phase 0 set up
    the checkpoint tables specifically so this works "from the start";
    Phase 6 builds the human-approval `interrupt()` on top of this without
    retrofitting checkpointer plumbing.

    `AsyncPostgresSaver.from_conn_string(dsn)` is an async context manager
    -- opened for the duration of this single call (its connection pool
    doesn't need to outlive the request: Postgres persists the checkpoint
    rows between requests, so a later `/approve`/`/reject` in Phase 6 can
    open a fresh saver against the same `thread_id` and resume from where
    this one left off).
    """
    dsn = to_psycopg_dsn(database_url or get_settings().database_url)
    graph = build_incident_graph(db, qdrant_client)

    # NOT `await saver.setup()` here -- schema provisioning is an explicit,
    # one-time-per-environment ops step (`backend.scripts.setup_checkpointer`),
    # not something run on every graph invocation. See that script's
    # docstring for why (every request racing to run DDL on the shared
    # checkpoint tables is the same reason `alembic upgrade head` isn't
    # wired into app startup either).
    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        compiled = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": str(incident.id)}}
        final_state = await compiled.ainvoke(initial_state(incident), config=config)

    return IncidentState.model_validate(final_state)


async def run_incident_graph_to_diagnosis(
    db: Session,
    incident: Incident,
    *,
    qdrant_client: QdrantClient | None = None,
    database_url: str | None = None,
) -> IncidentState:
    """Run the graph through Root Cause ONLY, halting before Response
    Planner ever executes -- Phase 7's diagnostic-only entry point
    (`backend.evaluation.harness.run_experiment_d`), not used by the real
    operational path.

    BUILD_PLAN.md's "Diagnostic evaluation" section is explicit: *"every
    experiment is scored immediately after the RCA stage, before any
    response/remediation, so response planning can't inflate a diagnosis
    score."* `run_incident_graph` above is the real end-to-end operational
    path (Phase 6) and must keep running the full graph as-is -- it backs
    the live API and Phase 6's own tests, which need Response Planner /
    Human Approval / Action Executor / Recovery Check to actually run.

    It was NOT safe to reuse for Phase 7's diagnostic eval, though, because
    a single `await run_incident_graph(...)` call does NOT stop at Root
    Cause: `response_planner` sits UNCONDITIONALLY between `root_cause` and
    the `human_approval`/`action_executor` branch (see this module's graph
    diagram above), so it always runs -- and always makes one real
    `ChatAnthropic.with_structured_output(ResponsePlan)` call and writes
    `AuditEvent` rows -- before `run_incident_graph` can return, regardless
    of whether the proposed plan turns out SAFE-only (straight through to
    `action_executor`) or HIGH_IMPACT (paused at `human_approval`'s
    `interrupt()`). That extra call's latency and token usage would
    silently leak into Phase 7's "immediately after RCA" comparison table
    right alongside A/B/C, none of which ever run anything past their own
    single diagnosis call -- a real apples-to-oranges skew against
    Experiment D's latency/token-cost columns specifically (the
    `root_cause_category`/evidence fields themselves are unaffected, since
    no later node overwrites them -- see `root_cause_node`/
    `response_planner_node`'s docstrings -- but latency and token cost are
    scored metrics too, and BUILD_PLAN.md's sentence covers the whole
    diagnostic scoring pass, not just accuracy).

    The fix: compile the exact SAME graph (`build_incident_graph` -- no
    forked nodes/edges) with `interrupt_before=["response_planner"]`,
    LangGraph's native static-breakpoint mechanism. Deliberately
    `interrupt_before=["response_planner"]`, NOT
    `interrupt_after=["root_cause"]` -- an earlier version of this function
    used the latter and it is WRONG: verified with a throwaway two-node
    loop script that a static `interrupt_after=[node]` breakpoint fires
    after *every* visit to that node, not just its final one. `root_cause`
    is revisited on each pass of Phase 5's bounded re-investigation loop
    (`route_after_root_cause` -> "reinvestigate" -> back to
    `investigation`) -- `interrupt_after=["root_cause"]` would have halted
    the graph after the FIRST root_cause pass, silently truncating the
    reinvestigation loop entirely (never re-checking `cascading_payment_
    timeout`'s evidence-sufficiency gap a second time) and scoring an
    incomplete diagnosis. `response_planner` is only ever reached once, via
    `route_after_root_cause`'s "end" branch, strictly AFTER the loop has
    already fully resolved -- so `interrupt_before=["response_planner"]`
    lets the entire bounded loop run to completion within one `ainvoke`
    call (exactly matching what `run_incident_graph`'s full run would do up
    to that point) and halts only once, at the correct boundary. Verified
    against the installed `langgraph` build with a second throwaway script
    (a looping two-node graph, `interrupt_before` on the downstream node)
    before relying on it here.

    Unlike `human_approval_node`'s dynamic `interrupt()` call (which pauses
    mid-node and requires an explicit `Command(resume=...)` to continue), a
    static `interrupt_before` boundary ahead of a not-yet-started node needs
    no resume at all for this function's purposes: `ainvoke` runs the graph
    up through the end of the reinvestigation loop, persists that
    checkpoint, and returns the state as of right after `root_cause`'s
    final pass completes -- `response_planner` never starts.

    Uses a distinct thread_id (`f"{incident.id}-diagnostic-eval"`, not
    `str(incident.id)`) so this function's checkpoint can never collide
    with `run_incident_graph`/`resume_incident_graph`'s thread for the same
    incident row -- an eval run and a real operational run against the
    same incident id stay on fully independent threads.
    """
    dsn = to_psycopg_dsn(database_url or get_settings().database_url)
    graph = build_incident_graph(db, qdrant_client)

    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        compiled = graph.compile(checkpointer=saver, interrupt_before=["response_planner"])
        config = {"configurable": {"thread_id": f"{incident.id}-diagnostic-eval"}}
        final_state = await compiled.ainvoke(initial_state(incident), config=config)

    return IncidentState.model_validate(final_state)


async def resume_incident_graph(
    db: Session,
    incident: Incident,
    resume_payload: dict[str, Any],
    *,
    qdrant_client: QdrantClient | None = None,
    database_url: str | None = None,
) -> IncidentState:
    """Resume a thread paused at `human_approval_node`'s `interrupt()` call.

    Only ever called by `POST /approve` (`backend.api.approvals`) -- never
    by `POST /reject`, which decides and durably records a rejection
    entirely on its own without touching this graph at all (see that
    module's docstring for why "do not resume toward execution" is
    implemented as "do not resume the graph," not as an in-graph branch).

    `Command(resume=resume_payload)` is LangGraph's current (1.x) resume
    API (`langgraph.types.Command`, verified against the installed
    `langgraph==1.2.11`): passed as the first positional argument to
    `ainvoke` in place of an initial-state dict, it re-enters the exact
    checkpointed thread at its paused task and supplies `resume_payload`
    as `human_approval_node`'s `interrupt()` return value. Everything else
    mirrors `run_incident_graph` exactly -- same `thread_id`, same
    freshly-opened-per-call `AsyncPostgresSaver` -- so this is genuinely
    resuming the same persisted checkpoint, not starting a new run.

    The caller (`backend.api.approvals`) is responsible for having already
    durably recorded the approval decision on the relevant `AuditEvent`
    row(s), under its own optimistic-concurrency guard, BEFORE calling
    this function -- so this function is only ever invoked once per
    genuine approval (see that module's idempotency guard).
    """
    dsn = to_psycopg_dsn(database_url or get_settings().database_url)
    graph = build_incident_graph(db, qdrant_client)

    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        compiled = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": str(incident.id)}}
        final_state = await compiled.ainvoke(Command(resume=resume_payload), config=config)

    return IncidentState.model_validate(final_state)


async def get_incident_thread_state(
    db: Session,
    incident: Incident,
    *,
    qdrant_client: QdrantClient | None = None,
    database_url: str | None = None,
) -> StateSnapshot:
    """Return the raw LangGraph `StateSnapshot` for `incident`'s thread.

    Read-only inspection of the checkpointer -- used to prove a graph run
    genuinely halted at `interrupt()` (`snapshot.next == ("human_approval",)`
    and `snapshot.interrupts` non-empty) rather than merely inferring it
    from `IncidentState.incident_status`, which the Response Planner sets
    to `AWAITING_APPROVAL` regardless of whether the graph actually paused.
    Exists mainly for tests; `backend.api.approvals` doesn't need this --
    it gates on `AuditEvent.decision_status` instead (see that module).
    """
    dsn = to_psycopg_dsn(database_url or get_settings().database_url)
    graph = build_incident_graph(db, qdrant_client)

    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        compiled = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": str(incident.id)}}
        return await compiled.aget_state(config)
