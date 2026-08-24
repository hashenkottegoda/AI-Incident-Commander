"""Phase 5's `StateGraph` assembly: nodes, conditional edges, Postgres
checkpointer.

BUILD_PLAN.md Phase 5: *"Build the StateGraph: Triage -> Investigation loop
-> RAG -> Root Cause/Hypothesis -> conditional re-investigation loop back to
Investigation."* This is the full investigation half of Experiment D
(response/remediation is Phase 6 and not built here).

```
START -> triage -> investigation -> rag -> root_cause -+-> END (diagnosed)
                        ^                               |
                        +---- (reinvestigate) ----------+
```

`build_incident_graph` returns the **uncompiled** `StateGraph` -- callers
attach whichever checkpointer fits their context: `AsyncPostgresSaver` for
the real API path (`run_incident_graph` below), no checkpointer (or an
in-memory one) for structural tests that only need to inspect
nodes/edges/predicates without ever invoking the graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from backend.agents.investigation_node import make_investigation_node
from backend.agents.rag_node import make_rag_node
from backend.agents.root_cause_node import make_root_cause_node
from backend.agents.routing import route_after_root_cause
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

    graph.add_edge(START, "triage")
    graph.add_edge("triage", "investigation")
    graph.add_edge("investigation", "rag")
    graph.add_edge("rag", "root_cause")
    graph.add_conditional_edges(
        "root_cause",
        route_after_root_cause,
        {"reinvestigate": "investigation", "end": END},
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
