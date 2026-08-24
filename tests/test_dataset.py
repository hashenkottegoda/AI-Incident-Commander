"""Tests for Phase 1's `--count N --seed S` seeded batch generator
(`backend.simulation.dataset.generate_dataset`).

Follows `tests/test_injector.py`'s pattern: the whole module is skipped
when Postgres isn't reachable (`docker compose up -d postgres` to run
these for real). This is BUILD_PLAN.md Phase 1's final acceptance bar:
"the same seed reproduces a byte-identical dataset" — the determinism
test below is the load-bearing one, checked both at the incident-summary
level and at the underlying `MetricPoint`/`LogEntry` row level (mirroring
`test_injector.py`'s own determinism test, which excludes autoincrement
ids/service_ids from the comparison since Postgres sequences don't roll
back, and pre-commits the canonical services once so `service_id` stays
stable across repeated runs).
"""

from __future__ import annotations

import psycopg
import pytest
from sqlalchemy import select

from backend.config import get_settings
from backend.db import SessionLocal
from backend.models import LogEntry, MetricPoint
from backend.scripts.setup_checkpointer import to_psycopg_dsn
from backend.simulation.baseline import get_or_create_canonical_services
from backend.simulation.dataset import generate_dataset


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

ALL_FAILURE_TYPES = frozenset(
    {
        "db_connection_exhaustion",
        "memory_leak",
        "bad_deployment",
        "dependency_failure",
        "slow_query",
        "cascading_payment_timeout",
    }
)


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def test_generate_dataset_produces_count_incidents_with_full_round_robin_coverage(db):
    incidents = generate_dataset(db, count=12, seed=42)

    assert len(incidents) == 12
    assert all(i.scenario_seed == 42 for i in incidents)

    indices = [i.scenario_instance_index for i in incidents]
    assert indices == list(range(12))
    assert len(set(indices)) == 12  # no duplicates

    # 12 = 2 full round-robin cycles over 6 scenario types -> every type
    # must appear (guards the round-robin-with-shuffled-cycles mechanism,
    # not just "hope randomness covers it").
    failure_types = {i.failure_type for i in incidents}
    assert failure_types == ALL_FAILURE_TYPES


def test_generate_dataset_scales_across_multiple_round_robin_cycles(db):
    """count=18 = exactly 3 full cycles over the 6 scenario types -> each
    type should appear exactly 3 times. Proves the round-robin mechanism
    scales cleanly to larger --count without needing to actually run
    --count 100 in the fast unit test suite."""
    incidents = generate_dataset(db, count=18, seed=5)

    assert len(incidents) == 18
    indices = [i.scenario_instance_index for i in incidents]
    assert indices == list(range(18))

    counts: dict[str, int] = {}
    for incident in incidents:
        counts[incident.failure_type] = counts.get(incident.failure_type, 0) + 1
    assert set(counts) == ALL_FAILURE_TYPES
    assert all(n == 3 for n in counts.values())


def test_generate_dataset_rejects_non_positive_count(db):
    with pytest.raises(ValueError):
        generate_dataset(db, count=0, seed=1)


def test_generate_dataset_is_deterministic_for_a_fixed_seed(db):
    """Same (count, seed) -> byte-identical incident summaries AND
    byte-identical underlying telemetry rows across two separate calls."""
    get_or_create_canonical_services(db)
    db.commit()

    def _run(seed: int):
        incidents = generate_dataset(db, count=10, seed=seed)
        db.flush()

        summary = [
            (
                i.scenario_instance_index,
                i.failure_type,
                i.detected_at,
                i.service_id,
                i.severity,
            )
            for i in incidents
        ]
        # One level deeper than the incident summary: the actual generated
        # telemetry rows for the whole batch (mirrors
        # test_injector.py's determinism test).
        metrics = [
            (m.service_id, m.timestamp, m.metric_name, m.value)
            for m in db.execute(select(MetricPoint).order_by(MetricPoint.id)).scalars()
        ]
        logs = [
            (log_.service_id, log_.timestamp, log_.level, log_.message, log_.attributes)
            for log_ in db.execute(select(LogEntry).order_by(LogEntry.id)).scalars()
        ]

        db.rollback()  # discard this run's incidents+telemetry; committed services survive
        return summary, metrics, logs

    run_a = _run(777)
    run_b = _run(777)
    assert run_a == run_b
    assert run_a[0], "expected at least one incident summary row"
    assert run_a[1], "expected at least one metric point"
    assert run_a[2], "expected at least one log entry"

    # Sanity check the determinism assertion above isn't vacuously true:
    # a different seed must actually produce different output.
    run_c = _run(888)
    assert run_c[0] != run_a[0]
    assert run_c[1] != run_a[1]
