"""Tests for Phase 2's tool layer (`backend/tools/`).

Follows `tests/test_injector.py`'s pattern: skip the whole module when
Postgres isn't reachable, and never `db.commit()` generated telemetry
(only `get_or_create_canonical_services` is committed once, so `Service`
ids stay stable while everything else rolls back per test).

Per BUILD_PLAN.md Phase 2's acceptance bar ("pytest per tool, asserting
correct filtering by service/time range") plus this task's spec, each
tool is checked for:

- Correct filtering by service (no cross-service leakage).
- Correct filtering by time range (rows outside `[start, end)` excluded).
- Returned records carry their row `id` (future `source_ref` use).
- At least one real-evidence hit per tool against a known injected scenario.
- Predictable, clear errors on malformed input (unknown service,
  `start >= end`, bad ISO timestamp) rather than a silent empty list.

Tools are called two ways: directly as plain functions (the primary,
agent-independent path this phase cares about) and, for at least one
case per tool, through the actual LangChain binding (`make_get_*_tool(db)`
+ `.invoke({...})`) to prove the binding itself works and hides `db` from
the LLM-facing schema.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from sqlalchemy import select

from backend.config import get_settings
from backend.db import SessionLocal
from backend.models import Service
from backend.scripts.setup_checkpointer import to_psycopg_dsn
from backend.simulation.injector import DEFAULT_PRE_INCIDENT_WINDOW, inject_failure
from backend.simulation.scenario_schema import load_all_scenarios
from backend.tools.dependencies import get_dependencies, make_get_dependencies_tool
from backend.tools.deployments import get_deployments, make_get_deployments_tool
from backend.tools.logs import get_logs, make_get_logs_tool
from backend.tools.metrics import get_metrics, make_get_metrics_tool


def _postgres_reachable() -> bool:
    dsn = to_psycopg_dsn(get_settings().database_url)
    try:
        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Postgres not reachable at DATABASE_URL (start it with `docker compose up -d postgres`)",
)


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def scenarios():
    return load_all_scenarios()


@pytest.fixture
def db_connection_exhaustion_incident(db, scenarios):
    """Inject `db_connection_exhaustion` (checkout-service, deployment ->
    connection-pool ramp -> ERROR logs) and return the window to query."""
    scenario = scenarios["db_connection_exhaustion"]
    incident_start = datetime(2026, 7, 1, 15, 0, tzinfo=UTC)
    incident = inject_failure(db, scenario, random.Random(42), incident_start)
    window_start = incident_start - DEFAULT_PRE_INCIDENT_WINDOW
    return incident, window_start, incident_start


@pytest.fixture
def cascading_payment_timeout_incident(db, scenarios):
    """Inject `cascading_payment_timeout` (checkout -> payment retry storm,
    the multi-service scenario `get_dependencies` needs)."""
    scenario = scenarios["cascading_payment_timeout"]
    incident_start = datetime(2026, 7, 2, 10, 0, tzinfo=UTC)
    incident = inject_failure(db, scenario, random.Random(7), incident_start)
    window_start = incident_start - DEFAULT_PRE_INCIDENT_WINDOW
    return incident, window_start, incident_start


def _iso(dt: datetime) -> str:
    return dt.isoformat()


# --- get_logs ---------------------------------------------------------


def test_get_logs_finds_error_evidence_on_affected_service(db, db_connection_exhaustion_incident):
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    logs = get_logs(db, "checkout-service", _iso(window_start), _iso(incident_start), level="error")

    assert logs, "expected ERROR-level logs from the pool-exhaustion causal chain"
    assert all(log.level == "error" for log in logs)
    assert all(log.service == "checkout-service" for log in logs)
    assert all(isinstance(log.id, int) for log in logs)
    assert any("connection" in log.message.lower() for log in logs)


def test_get_logs_filters_by_service(db, db_connection_exhaustion_incident):
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    checkout_errors = get_logs(
        db, "checkout-service", _iso(window_start), _iso(incident_start), level="error"
    )
    payment_errors = get_logs(
        db, "payment-service", _iso(window_start), _iso(incident_start), level="error"
    )

    assert checkout_errors, "checkout-service should have the injected ERROR logs"
    assert payment_errors == [], "payment-service is unaffected by this scenario"


def test_get_logs_filters_by_time_range(db, db_connection_exhaustion_incident):
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    # A window that ends well before the causal chain's error burst
    # (clustered in the final minutes before incident_start) should miss it.
    early_only = get_logs(
        db,
        "checkout-service",
        _iso(window_start),
        _iso(window_start + timedelta(minutes=1)),
        level="error",
    )
    full_window = get_logs(
        db, "checkout-service", _iso(window_start), _iso(incident_start), level="error"
    )

    assert early_only == []
    assert full_window
    assert all(window_start <= log.timestamp < incident_start for log in full_window)


def test_get_logs_unknown_service_raises(db, db_connection_exhaustion_incident):
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    with pytest.raises(ValueError, match="unknown service"):
        get_logs(db, "nonexistent-service", _iso(window_start), _iso(incident_start))


def test_get_logs_start_after_end_raises(db, db_connection_exhaustion_incident):
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    with pytest.raises(ValueError, match="before"):
        get_logs(db, "checkout-service", _iso(incident_start), _iso(window_start))


def test_get_logs_malformed_timestamp_raises(db, db_connection_exhaustion_incident):
    _incident, _window_start, incident_start = db_connection_exhaustion_incident

    with pytest.raises(ValueError, match="invalid ISO 8601"):
        get_logs(db, "checkout-service", "not-a-timestamp", _iso(incident_start))


def test_get_logs_unknown_level_raises(db, db_connection_exhaustion_incident):
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    with pytest.raises(ValueError, match="unknown log level"):
        get_logs(db, "checkout-service", _iso(window_start), _iso(incident_start), level="critical")


def test_get_logs_tool_binding_hides_db_and_invokes(db, db_connection_exhaustion_incident):
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    tool = make_get_logs_tool(db)
    assert "db" not in tool.args
    assert set(tool.args) == {"service", "start", "end", "level"}

    result = tool.invoke(
        {
            "service": "checkout-service",
            "start": _iso(window_start),
            "end": _iso(incident_start),
            "level": "error",
        }
    )
    assert result
    # Invoked through the LangChain binding, so results are plain
    # JSON-serializable dicts (see logs.make_get_logs_tool), not the
    # Pydantic `LogRecord`s the direct `get_logs()` call above returns.
    assert all(record["service"] == "checkout-service" for record in result)


# --- get_metrics --------------------------------------------------------


def test_get_metrics_finds_connection_ramp_evidence(db, db_connection_exhaustion_incident):
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    points = get_metrics(
        db, "checkout-service", "db_connections_active", _iso(window_start), _iso(incident_start)
    )

    assert points
    assert all(isinstance(p.id, int) for p in points)
    late_values = [p.value for p in points if p.timestamp >= incident_start - timedelta(minutes=3)]
    assert late_values
    assert max(late_values) > 20.0  # well above checkout's baseline mean of 8


def test_get_metrics_filters_by_service(db, db_connection_exhaustion_incident):
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    checkout_points = get_metrics(
        db, "checkout-service", "db_connections_active", _iso(window_start), _iso(incident_start)
    )
    payment_points = get_metrics(
        db, "payment-service", "db_connections_active", _iso(window_start), _iso(incident_start)
    )

    assert checkout_points
    assert payment_points, "payment-service still has baseline db_connections_active samples"
    # Unaffected service stays near its own baseline mean (6.0), never
    # ramps toward checkout's anomalous target.
    assert max(p.value for p in payment_points) < 15.0
    assert max(p.value for p in checkout_points) > 20.0


def test_get_metrics_filters_by_time_range(db, db_connection_exhaustion_incident):
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    early_only = get_metrics(
        db,
        "checkout-service",
        "db_connections_active",
        _iso(window_start),
        _iso(window_start + timedelta(minutes=1)),
    )
    full_window = get_metrics(
        db, "checkout-service", "db_connections_active", _iso(window_start), _iso(incident_start)
    )

    assert len(early_only) < len(full_window)
    assert all(window_start <= p.timestamp < incident_start for p in full_window)


def test_get_metrics_unknown_service_raises(db, db_connection_exhaustion_incident):
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    with pytest.raises(ValueError, match="unknown service"):
        get_metrics(
            db,
            "nonexistent-service",
            "db_connections_active",
            _iso(window_start),
            _iso(incident_start),
        )


def test_get_metrics_start_after_end_raises(db, db_connection_exhaustion_incident):
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    with pytest.raises(ValueError, match="before"):
        get_metrics(
            db,
            "checkout-service",
            "db_connections_active",
            _iso(incident_start),
            _iso(window_start),
        )


def test_get_metrics_malformed_timestamp_raises(db, db_connection_exhaustion_incident):
    _incident, _window_start, incident_start = db_connection_exhaustion_incident

    with pytest.raises(ValueError, match="invalid ISO 8601"):
        get_metrics(
            db, "checkout-service", "db_connections_active", "bad-date", _iso(incident_start)
        )


def test_get_metrics_unknown_metric_name_is_empty_not_error(db, db_connection_exhaustion_incident):
    """A valid service/window but a metric name that doesn't exist is a
    legitimate empty result, not malformed input."""
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    points = get_metrics(
        db, "checkout-service", "totally_made_up_metric", _iso(window_start), _iso(incident_start)
    )
    assert points == []


def test_get_metrics_tool_binding_hides_db_and_invokes(db, db_connection_exhaustion_incident):
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    tool = make_get_metrics_tool(db)
    assert "db" not in tool.args

    result = tool.invoke(
        {
            "service": "checkout-service",
            "metric_name": "db_connections_active",
            "start": _iso(window_start),
            "end": _iso(incident_start),
        }
    )
    assert result
    # Invoked through the LangChain binding -> plain dicts, see above.
    assert all(record["metric_name"] == "db_connections_active" for record in result)


# --- get_deployments -----------------------------------------------------


def test_get_deployments_finds_v1_8_2(db, db_connection_exhaustion_incident):
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    deployments = get_deployments(db, "checkout-service", _iso(window_start), _iso(incident_start))

    assert len(deployments) == 1
    assert deployments[0].version == "v1.8.2"
    assert isinstance(deployments[0].id, int)
    assert deployments[0].service == "checkout-service"


def test_get_deployments_filters_by_service(db, db_connection_exhaustion_incident):
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    checkout_deploys = get_deployments(
        db, "checkout-service", _iso(window_start), _iso(incident_start)
    )
    payment_deploys = get_deployments(
        db, "payment-service", _iso(window_start), _iso(incident_start)
    )

    assert checkout_deploys
    assert payment_deploys == []


def test_get_deployments_filters_by_time_range(db, db_connection_exhaustion_incident):
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    before_deploy = get_deployments(
        db, "checkout-service", _iso(window_start), _iso(window_start + timedelta(seconds=1))
    )

    assert before_deploy == []


def test_get_deployments_unknown_service_raises(db, db_connection_exhaustion_incident):
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    with pytest.raises(ValueError, match="unknown service"):
        get_deployments(db, "nonexistent-service", _iso(window_start), _iso(incident_start))


def test_get_deployments_start_after_end_raises(db, db_connection_exhaustion_incident):
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    with pytest.raises(ValueError, match="before"):
        get_deployments(db, "checkout-service", _iso(incident_start), _iso(window_start))


def test_get_deployments_tool_binding_hides_db_and_invokes(db, db_connection_exhaustion_incident):
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    tool = make_get_deployments_tool(db)
    assert "db" not in tool.args

    result = tool.invoke(
        {"service": "checkout-service", "start": _iso(window_start), "end": _iso(incident_start)}
    )
    # Invoked through the LangChain binding -> plain dicts, see above.
    assert result and result[0]["version"] == "v1.8.2"


def test_get_deployments_tool_result_is_real_json_not_python_repr(
    db, db_connection_exhaustion_incident
):
    """Regression test: LangChain's ToolMessage content serializer
    (`langchain_core.tools.base._stringify`) does `json.dumps` and falls
    back to Python `repr()` on TypeError for anything not JSON-native. A
    `list[BaseModel]` return value used to hit that fallback, so the LLM
    would see repr syntax (`datetime.datetime(...)`, single-quoted
    strings) instead of clean JSON. This exercises the exact path a real
    ToolNode uses (a full tool_call dict, not a plain-args dict) and
    asserts the resulting message content round-trips through `json.loads`.
    """
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    tool = make_get_deployments_tool(db)
    message = tool.invoke(
        {
            "name": "get_deployments",
            "args": {
                "service": "checkout-service",
                "start": _iso(window_start),
                "end": _iso(incident_start),
            },
            "id": "call_1",
            "type": "tool_call",
        }
    )

    parsed = json.loads(message.content)
    assert parsed and parsed[0]["version"] == "v1.8.2"


# --- get_dependencies -----------------------------------------------------


def test_get_dependencies_finds_checkout_to_payment_retry_storm(
    db, cascading_payment_timeout_incident
):
    _incident, window_start, incident_start = cascading_payment_timeout_incident

    spans = get_dependencies(db, "checkout-service", _iso(window_start), _iso(incident_start))

    assert spans
    assert all(isinstance(s.id, int) for s in spans)
    assert all(s.downstream_service == "payment-service" for s in spans)
    durations = sorted(s.duration_ms for s in spans)
    assert durations[-1] > durations[0] + 500  # ramping, not flat


def test_get_dependencies_filters_by_service(db, cascading_payment_timeout_incident):
    _incident, window_start, incident_start = cascading_payment_timeout_incident

    checkout_spans = get_dependencies(
        db, "checkout-service", _iso(window_start), _iso(incident_start)
    )
    payment_spans = get_dependencies(
        db, "payment-service", _iso(window_start), _iso(incident_start)
    )

    assert checkout_spans
    assert payment_spans == [], "payment-service makes no outbound calls in this scenario"


def test_get_dependencies_filters_by_time_range(db, cascading_payment_timeout_incident):
    _incident, window_start, incident_start = cascading_payment_timeout_incident

    before_spans = get_dependencies(
        db, "checkout-service", _iso(window_start), _iso(window_start + timedelta(seconds=1))
    )
    full_window = get_dependencies(db, "checkout-service", _iso(window_start), _iso(incident_start))

    assert before_spans == []
    assert full_window


def test_get_dependencies_unknown_service_raises(db, cascading_payment_timeout_incident):
    _incident, window_start, incident_start = cascading_payment_timeout_incident

    with pytest.raises(ValueError, match="unknown service"):
        get_dependencies(db, "nonexistent-service", _iso(window_start), _iso(incident_start))


def test_get_dependencies_start_after_end_raises(db, cascading_payment_timeout_incident):
    _incident, window_start, incident_start = cascading_payment_timeout_incident

    with pytest.raises(ValueError, match="before"):
        get_dependencies(db, "checkout-service", _iso(incident_start), _iso(window_start))


def test_get_dependencies_tool_binding_hides_db_and_invokes(db, cascading_payment_timeout_incident):
    _incident, window_start, incident_start = cascading_payment_timeout_incident

    tool = make_get_dependencies_tool(db)
    assert "db" not in tool.args

    result = tool.invoke(
        {"service": "checkout-service", "start": _iso(window_start), "end": _iso(incident_start)}
    )
    # Invoked through the LangChain binding -> plain dicts, see above.
    assert result and result[0]["downstream_service"] == "payment-service"


# --- build_tools aggregator -----------------------------------------------


def test_build_tools_returns_all_four_bound_to_the_same_db(db, db_connection_exhaustion_incident):
    from backend.tools import build_tools

    _incident, window_start, incident_start = db_connection_exhaustion_incident

    tools = build_tools(db)
    names = {t.name for t in tools}
    assert names == {"get_logs", "get_metrics", "get_deployments", "get_dependencies"}

    get_deployments_tool = next(t for t in tools if t.name == "get_deployments")
    result = get_deployments_tool.invoke(
        {"service": "checkout-service", "start": _iso(window_start), "end": _iso(incident_start)}
    )
    # Invoked through the LangChain binding -> plain dicts, see above.
    assert result and result[0]["version"] == "v1.8.2"


def test_all_services_are_resolvable(db, db_connection_exhaustion_incident):
    """Sanity: the known-services list quoted in a ValueError actually
    matches what's seeded (guards against the helper's error message
    silently going stale)."""
    _incident, window_start, incident_start = db_connection_exhaustion_incident

    seeded_names = sorted(s.name for s in db.execute(select(Service)).scalars())
    assert seeded_names == ["checkout-service", "inventory-service", "payment-service"]
