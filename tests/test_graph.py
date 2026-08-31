"""Structural tests for Phase 5's `StateGraph` assembly (`backend/graph.py`).

Compiling and inspecting the graph performs no I/O (node factory closures
only capture references -- see `build_incident_graph`'s docstring), so
these run without any live Postgres/Qdrant/OpenRouter dependency except the
one test that attaches a real `AsyncPostgresSaver` (skipped cleanly when
Postgres isn't reachable, same convention as the rest of this suite).

No test in this module makes an OpenRouter API call.
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

EXPECTED_NODES = {
    "triage",
    "investigation",
    "rag",
    "root_cause",
    "response_planner",
    "human_approval",
    "action_executor",
    "recovery_check",
}


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
    """Triage -> Investigation -> RAG -> Root Cause, in that fixed order --
    BUILD_PLAN.md's graph-flow diagram. Response Planner's, Action
    Executor's, and Recovery Check's own outgoing edges are all conditional
    (see the dedicated tests below), not part of this fixed linear chain --
    but Human Approval's single outgoing edge to action_executor is
    unconditional, since that node's own branching (approved vs. rejected)
    happens inside the node, not via a LangGraph conditional edge (see
    human_approval_node's docstring)."""
    compiled = _compiled(db)
    edges = {(e.source, e.target) for e in compiled.get_graph().edges if not e.conditional}
    assert (START, "triage") in edges
    assert ("triage", "investigation") in edges
    assert ("investigation", "rag") in edges
    assert ("rag", "root_cause") in edges
    assert ("human_approval", "action_executor") in edges


def test_conditional_edges_from_root_cause_go_to_investigation_and_response_planner(db):
    compiled = _compiled(db)
    conditional_targets = {
        e.target for e in compiled.get_graph().edges if e.conditional and e.source == "root_cause"
    }
    assert conditional_targets == {"investigation", "response_planner"}


def test_conditional_edges_from_response_planner_go_to_human_approval_and_action_executor(db):
    """SAFE-only plan -> action_executor directly; any HIGH_IMPACT action ->
    the interrupt() gate (backend.agents.human_approval_node) -- see
    backend.agents.routing.route_after_response_planner. Neither branch
    reaches END directly anymore: action_executor is always the next real
    step."""
    compiled = _compiled(db)
    conditional_targets = {
        e.target
        for e in compiled.get_graph().edges
        if e.conditional and e.source == "response_planner"
    }
    assert conditional_targets == {"human_approval", "action_executor"}


def test_conditional_edges_from_action_executor_go_to_recovery_check_and_end(db):
    """A HIGH_IMPACT remediation just executed -> recovery_check verifies
    it; an all-SAFE plan has nothing to verify -> END directly -- see
    backend.agents.routing.route_after_action_executor."""
    compiled = _compiled(db)
    conditional_targets = {
        e.target
        for e in compiled.get_graph().edges
        if e.conditional and e.source == "action_executor"
    }
    assert conditional_targets == {"recovery_check", END}


def test_conditional_edges_from_recovery_check_go_to_investigation_and_end(db):
    """Recovered/bound-exhausted -> END (RESOLVED or
    MANUAL_INTERVENTION_REQUIRED, decided inside the node); still degraded
    with budget remaining -> loop back to investigation -- see
    backend.agents.routing.route_after_recovery_check."""
    compiled = _compiled(db)
    conditional_targets = {
        e.target
        for e in compiled.get_graph().edges
        if e.conditional and e.source == "recovery_check"
    }
    assert conditional_targets == {"investigation", END}


def test_investigation_is_not_a_dead_end_it_has_incoming_loop_edges(db):
    """Confirms the re-investigation loop is actually wired, not just the
    one-way pipeline -- conditional edges back to investigation must exist
    from both root_cause (the confidence-gap/evidence-sufficiency loop) and
    recovery_check (the still-degraded-remediation loop)."""
    compiled = _compiled(db)
    loop_sources = {
        e.source
        for e in compiled.get_graph().edges
        if e.target == "investigation" and e.conditional
    }
    assert loop_sources == {"root_cause", "recovery_check"}


def test_only_expected_nodes_have_conditional_outgoing_edges(db):
    """human_approval's approved-vs-rejected branching happens inside the
    node body (around interrupt()), not as a LangGraph conditional edge --
    see human_approval_node's docstring -- so it must NOT appear here."""
    compiled = _compiled(db)
    conditional_sources = {e.source for e in compiled.get_graph().edges if e.conditional}
    assert conditional_sources == {
        "root_cause",
        "response_planner",
        "action_executor",
        "recovery_check",
    }


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
