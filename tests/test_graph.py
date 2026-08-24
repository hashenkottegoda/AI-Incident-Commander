"""Structural tests for Phase 5's `StateGraph` assembly (`backend/graph.py`).

Compiling and inspecting the graph performs no I/O (node factory closures
only capture references -- see `build_incident_graph`'s docstring), so
these run without any live Postgres/Qdrant/Claude dependency except the
one test that attaches a real `AsyncPostgresSaver` (skipped cleanly when
Postgres isn't reachable, same convention as the rest of this suite).

No test in this module makes a Claude/Anthropic API call.
"""

from __future__ import annotations

import psycopg
import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START

from backend.config import get_settings
from backend.db import SessionLocal
from backend.graph import build_incident_graph, initial_state
from backend.models.incident import IncidentStatus, Severity
from backend.rag.qdrant_client import get_qdrant_client
from backend.scripts.setup_checkpointer import to_psycopg_dsn

EXPECTED_NODES = {"triage", "investigation", "rag", "root_cause", "response_planner"}


def _postgres_reachable() -> bool:
    dsn = to_psycopg_dsn(get_settings().database_url)
    try:
        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _compiled(db):
    graph = build_incident_graph(db, get_qdrant_client())
    return graph.compile()


def test_all_expected_nodes_present(db):
    compiled = _compiled(db)
    nodes = set(compiled.get_graph().nodes.keys())
    assert EXPECTED_NODES.issubset(nodes)
    assert "__start__" in nodes
    assert "__end__" in nodes


def test_linear_edges_match_build_plan_flow(db):
    """Triage -> Investigation -> RAG -> Root Cause -> Response Planner, in
    that fixed order -- BUILD_PLAN.md's graph-flow diagram."""
    compiled = _compiled(db)
    edges = {(e.source, e.target) for e in compiled.get_graph().edges if not e.conditional}
    assert (START, "triage") in edges
    assert ("triage", "investigation") in edges
    assert ("investigation", "rag") in edges
    assert ("rag", "root_cause") in edges
    assert ("response_planner", END) in edges


def test_conditional_edges_from_root_cause_go_to_investigation_and_response_planner(db):
    compiled = _compiled(db)
    conditional_targets = {
        e.target for e in compiled.get_graph().edges if e.conditional and e.source == "root_cause"
    }
    assert conditional_targets == {"investigation", "response_planner"}


def test_investigation_is_not_a_dead_end_it_has_an_incoming_loop_edge(db):
    """Confirms the re-investigation loop is actually wired, not just the
    one-way pipeline -- an edge from root_cause back to investigation must
    exist (conditional)."""
    compiled = _compiled(db)
    loop_edges = [
        e
        for e in compiled.get_graph().edges
        if e.source == "root_cause" and e.target == "investigation"
    ]
    assert len(loop_edges) == 1
    assert loop_edges[0].conditional is True


def test_only_root_cause_has_conditional_outgoing_edges(db):
    compiled = _compiled(db)
    conditional_sources = {e.source for e in compiled.get_graph().edges if e.conditional}
    assert conditional_sources == {"root_cause"}


def test_compiled_graph_has_no_checkpointer_by_default(db):
    compiled = _compiled(db)
    assert compiled.checkpointer is None


# --- initial_state -----------------------------------------------------------


class _FakeService:
    name = "checkout-service"


class _FakeIncident:
    id = 123
    severity = Severity.P1
    service = _FakeService()


def test_initial_state_starts_in_triaging_not_detected():
    state = initial_state(_FakeIncident())
    assert state["incident_status"] == IncidentStatus.TRIAGING
    assert state["incident_id"] == 123
    assert state["severity"] == Severity.P1
    assert state["affected_services"] == ["checkout-service"]


# --- Postgres checkpointer attachment (skipped without Postgres) -----------


@pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Postgres not reachable at DATABASE_URL (start it with `docker compose up -d postgres`)",
)
async def test_postgres_checkpointer_attaches_to_compiled_graph(db):
    dsn = to_psycopg_dsn(get_settings().database_url)
    graph = build_incident_graph(db, get_qdrant_client())
    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        compiled = graph.compile(checkpointer=saver)
        assert compiled.checkpointer is saver
        assert isinstance(compiled.checkpointer, AsyncPostgresSaver)
