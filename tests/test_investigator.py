"""Phase 3 acceptance test (BUILD_PLAN.md): run the investigator against
all 6 failure types and confirm each `root_cause_category` matches ground
truth.

Makes REAL Claude API calls -- BUILD_PLAN.md Phase 3's own acceptance bar
is literally "run the agent against all six failure types," not a mocked
stand-in, so this test costs real money every time it runs. It is skipped
automatically unless a real-looking `ANTHROPIC_API_KEY` is available, so
the default fast `uv run pytest` suite never burns API credits by
accident. Run it explicitly:

    uv run pytest tests/test_investigator.py -v -s

## Why this reads `.env` directly instead of `get_settings()`

`tests/conftest.py` unconditionally does
`os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-dummy")` so every
*other* test gets a harmless placeholder without needing a real key.
Because that uses `setdefault` on the real *process* environment (not
`.env`), and pydantic-settings prioritizes a real env var over the `.env`
file, that placeholder would silently shadow the real key in `.env` for
`get_settings()` -- checking `get_settings().anthropic_api_key` here would
almost always see `"sk-test-dummy"` and skip for the wrong reason. Reading
`.env` straight via `python-dotenv` sidesteps that shadowing entirely, and
the `_real_api_key` fixture below scopes the override to just this
module's tests via `monkeypatch` (auto-restored after each test).
"""

from __future__ import annotations

import random
from datetime import UTC, datetime

import psycopg
import pytest
from dotenv import dotenv_values

from backend.agents.investigator import investigate_incident
from backend.config import get_settings
from backend.db import SessionLocal
from backend.scripts.setup_checkpointer import to_psycopg_dsn
from backend.simulation.injector import inject_failure
from backend.simulation.scenario_schema import load_all_scenarios

_DOTENV_KEY = dotenv_values(".env").get("ANTHROPIC_API_KEY")
_HAS_REAL_KEY = bool(_DOTENV_KEY) and _DOTENV_KEY.startswith("sk-ant-")


def _postgres_reachable() -> bool:
    dsn = to_psycopg_dsn(get_settings().database_url)
    try:
        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


pytestmark = [
    pytest.mark.skipif(
        not _HAS_REAL_KEY,
        reason=(
            "No real ANTHROPIC_API_KEY found in .env -- these tests make live, "
            "billed Claude API calls, so they are opt-in only."
        ),
    ),
    pytest.mark.skipif(
        not _postgres_reachable(),
        reason="Postgres not reachable at DATABASE_URL (start `docker compose up -d postgres`)",
    ),
]

ALL_FAILURE_TYPES = (
    "db_connection_exhaustion",
    "memory_leak",
    "bad_deployment",
    "dependency_failure",
    "slow_query",
    "cascading_payment_timeout",
)


@pytest.fixture(autouse=True)
def _real_api_key(monkeypatch):
    """Override conftest's placeholder key with the real one, scoped to
    this module's tests only (monkeypatch restores it after each test)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", _DOTENV_KEY or "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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


@pytest.mark.parametrize("failure_type", ALL_FAILURE_TYPES)
def test_investigator_root_cause_matches_ground_truth(db, scenarios, failure_type):
    """BUILD_PLAN.md Phase 3's acceptance bar, one scenario per test so a
    failure on e.g. `memory_leak` is reported individually rather than
    hiding behind a single aggregate pass/fail."""
    scenario = scenarios[failure_type]
    incident_start = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    incident = inject_failure(db, scenario, random.Random(123), incident_start)

    result = investigate_incident(db, incident)

    print(
        f"\n[{failure_type}] expected={scenario.root_cause_category!r} "
        f"got={result.root_cause_category!r} "
        f"confidence={result.diagnostic_confidence!r} "
        f"evidence_count={len(result.evidence)}"
    )

    assert result.root_cause_category == scenario.root_cause_category, (
        f"{failure_type}: expected {scenario.root_cause_category!r}, got "
        f"{result.root_cause_category!r}. hypotheses={result.hypotheses!r}"
    )
