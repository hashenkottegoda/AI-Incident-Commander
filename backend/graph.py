"""Phase 5/6's `StateGraph` assembly: nodes, conditional edges, Postgres
checkpointer.

BUILD_PLAN.md Phase 5: *"Build the StateGraph: Triage -> Investigation loop
-> RAG -> Root Cause/Hypothesis -> conditional re-investigation loop back to
Investigation."* Phase 6 continues the same graph past Root Cause: *"RESPONSE
PLANNER ... -> RISK CLASSIFIER ... -> SAFE -> ACTION EXECUTOR ... HIGH-IMPACT
-> HUMAN APPROVAL (LangGraph `interrupt`, resumed via POST /approve|/reject)."*
This module now builds through the Response Planner + (inline) Risk
Classifier + the real `interrupt()`-based Human Approval gate
(`backend.agents.human_approval_node`) -- the Action Executor and Recovery
Check are the next Phase 6 sub-steps, not built here (see that node's
docstring for the placeholder it uses in the meantime).

```
START -> triage -> investigation -> rag -> root_cause -+-> response_planner -+-> human_approval
                        ^                               |                    |         |
                        +---- (reinvestigate) ----------+                    |         v
                                                                              |        END
                                                                              +-> END
```

- `response_planner -> human_approval -> END`: any HIGH_IMPACT action.
  `human_approval` calls `interrupt()`, resumed via `POST /approve` or
  `/reject` (`backend.api.approvals`).
- `response_planner -> END` directly: an all-SAFE plan, nothing for a
  human to approve. `incident_status = EXECUTING` -- the not-yet-built
  Action Executor's SAFE-branch entry point.

`build_incident_graph` returns the **uncompiled** `StateGraph` -- callers
attach whichever checkpointer fits their context: `AsyncPostgresSaver` for
the real API path (`run_incident_graph`/`resume_incident_graph` below), no
checkpointer (or an in-memory one) for structural tests that only need to
inspect nodes/edges/predicates without ever invoking the graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, StateSnapshot
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from backend.agents.human_approval_node import human_approval_node
from backend.agents.investigation_node import make_investigation_node
from backend.agents.rag_node import make_rag_node
from backend.agents.response_planner_node import make_response_planner_node
from backend.agents.root_cause_node import make_root_cause_node
from backend.agents.routing import route_after_response_planner, route_after_root_cause
from backend.agents.state import IncidentState
from backend.agents.triage_node import make_triage_node
from backend.config import get_settings
from backend.models import IncidentStatus
from backend.rag.qdrant_client import get_qdrant_client
from backend.scripts.setup_checkpointer import to_psycopg_dsn

if TYPE_CHECKING:
    from backend.models import Incident


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
    graph.add_node("triage", make_triage_node())
    graph.add_node("investigation", make_investigation_node(db))
    graph.add_node("rag", make_rag_node(qdrant_client))
    graph.add_node("root_cause", make_root_cause_node())
    graph.add_node("response_planner", make_response_planner_node(db))
    graph.add_node("human_approval", human_approval_node)

    graph.add_edge(START, "triage")
    graph.add_edge("triage", "investigation")
    graph.add_edge("investigation", "rag")
    graph.add_edge("rag", "root_cause")
    graph.add_conditional_edges(
        "root_cause",
        route_after_root_cause,
        {"reinvestigate": "investigation", "end": "response_planner"},
    )
    # SAFE-only plan -> straight to END (incident_status = EXECUTING, the
    # not-yet-built Action Executor's SAFE-branch entry point). Any
    # HIGH_IMPACT action -> the interrupt() gate; see
    # backend.agents.human_approval_node's docstring for why that gate is
    # its own dedicated node rather than folded into response_planner.
    graph.add_conditional_edges(
        "response_planner",
        route_after_response_planner,
        {"human_approval": "human_approval", "end": END},
    )
    # human_approval is Phase 6's current terminal node on the HIGH_IMPACT
    # branch: it pauses at interrupt() until POST /approve resumes it (see
    # backend.api.approvals), then sets a placeholder post-approval
    # incident_status and the graph ends there until the real Action
    # Executor / Recovery Check exist.
    graph.add_edge("human_approval", END)
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
