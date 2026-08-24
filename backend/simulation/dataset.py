"""`--count N --seed S` seeded batch generation (BUILD_PLAN.md Phase 1's
final simulation sub-step).

`generate_dataset()` is the single entry point: given `count` and a
mandatory `seed`, it calls `injector.inject_failure()` once per instance
and returns the list of created `Incident` rows. `backend/scripts/
generate_dataset.py` is the thin CLI wrapper (`uv run python -m
backend.scripts.generate_dataset --count 30 --seed 42`).

## Determinism (the entire point of this feature)

A single master `random.Random(seed)` drives *every* choice in this
module — which scenario type each instance uses, its `incident_start`
jitter, and the per-instance sub-seed handed to `inject_failure`. The
global `random` module and `datetime.now()` are never touched (matching
`baseline.py`/`injector.py`'s existing determinism discipline). All draws
happen in one fixed order (scenario-assignment shuffles first, then a
fixed per-instance `[jitter, sub_seed]` draw order for `i = 0..count-1`),
so `generate_dataset(db, count=N, seed=S)` reproduces byte-identical
`(failure_type, incident_start, telemetry)` per index across separate
calls/processes — see `tests/test_dataset.py`'s determinism test, which
checks this at both the incident-summary level and one instance's actual
`MetricPoint` rows.

## Scenario assignment: round-robin with a shuffled order per cycle

Pure-random scenario selection risks a scenario never appearing in a
small `--count` (`--count 5` has a real chance of missing one of the 6
types entirely under independent random draws). Instead, this walks
`sorted(failure_types)` (the 6 `failure_scenarios/*.yaml` `failure_type`s)
in a freshly `master_rng.shuffle`d order each "cycle", and concatenates
cycles until `count` is reached, truncating the final partial cycle. This
guarantees every scenario type appears at least once as soon as
`count >= 6`, guarantees the *first* 6 instances are collectively a full
sweep (useful for a `--count 5`/`6` CI smoke run), and still avoids the
"always exactly this fixed order" staleness of a non-shuffled round-robin
— each cycle's order is its own independent, seed-derived shuffle.

## What "variant" means here (timing + telemetry shape, not service-swapping)

BUILD_PLAN.md asks for "randomized service/timing/severity variants per
scenario type." Each scenario file's `affected_service`, `root_cause_category`,
and `causal_chain` are semantically load-bearing and mutually
consistent — e.g. `db_connection_exhaustion.yaml`'s chain literally names
`checkout_deployment_v1.8.2` and is written assuming checkout-service.
Randomizing *which* service a given scenario instance affects would
silently break that internal consistency (the chain's deployment-version
string, its evidence tags, `remediation_effects`) unless the YAML
templates were also parameterized per-service — a materially larger
change, out of scope here. Likewise, severity is that scenario's own
declared P1/P2/etc. framing (e.g. `db_connection_exhaustion` is always a
P1 in its own ground truth); picking a different severity per instance
without also rewriting the scenario's narrative would make the ground
truth self-contradictory for no real diversity benefit.

So "variant" is implemented honestly as the two things that *are* safe to
vary without touching ground-truth coherence:

1. **Timing** — each instance gets its own `incident_start`, spread across
   `DATASET_ANCHOR + i * INSTANCE_SPACING` with up to `±JITTER_MINUTES`
   minutes of jitter (see `_pick_incident_start`), so instances don't
   collide and don't all land on a suspiciously round timestamp.
2. **Per-instance telemetry shape** — each instance draws its own
   sub-seed from `master_rng` and gets its own `random.Random(sub_seed)`
   passed to `inject_failure`, so two `bad_deployment` instances have
   different baseline jitter, different ramp noise, and different
   error-log message ordering/timestamps within the same structural
   chain — genuinely different telemetry, not 100 byte-identical copies
   of the same 6 fixed incidents.

`affected_service`/`severity` are deliberately **not** varied by
service-swapping in this step. If a future step wants real service/
severity diversity, the honest way to do it is to parameterize the YAML
scenario templates themselves (e.g. a `service_overrides` block per
scenario), not to bolt service-swapping onto the generator against a
fixed causal_chain string.

## Timing spacing

`INSTANCE_SPACING` (12h) with `JITTER_MINUTES` (90 min) gives a worst-case
gap of 10.5h between one instance's `incident_start` and the next
instance's nominal slot. `memory_leak`'s `SCENARIO_TIMING_OVERRIDES`
pre-incident window (the widest of any scenario) is 6h, so even in the
worst-case jitter draw there's several hours of daylight between one
instance's incident window and its neighbor's — "non-overlapping enough"
per BUILD_PLAN's intent, not a hard guarantee (overlap wouldn't corrupt
anything — each instance's rows are independent, just noisier to eyeball
manually — but keeping them apart makes the generated dataset easier to
inspect by hand during dev).
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from backend.models import Incident
from backend.simulation.injector import inject_failure
from backend.simulation.scenario_schema import load_all_scenarios

# Fixed anchor, never `datetime.now()` — determinism requires the dataset's
# timeline to depend only on `count`/`seed`, not on when the generator runs.
DATASET_ANCHOR: datetime = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
INSTANCE_SPACING: timedelta = timedelta(hours=12)
JITTER_MINUTES: float = 90.0

# Sub-seeds are drawn from this range and fed to `random.Random(...)` for
# each instance's own telemetry generation.
_SUB_SEED_UPPER_BOUND: int = 2**32


def _build_scenario_assignment(
    master_rng: random.Random, failure_types: list[str], count: int
) -> list[str]:
    """Round-robin over `failure_types`, re-shuffled (via `master_rng`)
    each full cycle, concatenated and truncated to `count` — see module
    docstring's "Scenario assignment" section."""
    assignment: list[str] = []
    while len(assignment) < count:
        cycle = list(failure_types)
        master_rng.shuffle(cycle)
        assignment.extend(cycle)
    return assignment[:count]


def generate_dataset(
    db: Session,
    count: int,
    seed: int,
    *,
    dataset_start: datetime | None = None,
) -> list[Incident]:
    """Generate `count` deterministic failure-scenario incidents, seeded by `seed`.

    Every random choice (scenario assignment, per-instance `incident_start`
    jitter, per-instance telemetry sub-seed) is drawn from a single
    `random.Random(seed)` in a fixed order, so calling this twice with the
    same `(count, seed)` — even in separate processes, even against a
    freshly reset DB — produces byte-identical `Incident` rows and
    telemetry (see module docstring's "Determinism" section and
    `tests/test_dataset.py`).

    `dataset_start` defaults to the fixed `DATASET_ANCHOR` (never
    `datetime.now()`); it's exposed as a keyword mainly so tests can pin a
    distinct, collision-free window per test without affecting the
    generator's own determinism story.

    Each created `Incident` carries `scenario_seed=seed` and
    `scenario_instance_index=i` (0-indexed) for Phase 7's eval harness
    provenance. Flushes per instance (via `inject_failure`) but does not
    commit — the caller (the CLI script, or a test) owns the transaction
    boundary.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count!r}")

    master_rng = random.Random(seed)
    scenarios = load_all_scenarios()
    failure_types = sorted(scenarios)  # fixed base order before shuffling

    assignment = _build_scenario_assignment(master_rng, failure_types, count)

    start = dataset_start if dataset_start is not None else DATASET_ANCHOR

    incidents: list[Incident] = []
    for i in range(count):
        jitter = timedelta(minutes=master_rng.uniform(-JITTER_MINUTES, JITTER_MINUTES))
        incident_start = start + i * INSTANCE_SPACING + jitter

        instance_seed = master_rng.randrange(_SUB_SEED_UPPER_BOUND)
        instance_rng = random.Random(instance_seed)

        scenario = scenarios[assignment[i]]
        incident = inject_failure(
            db,
            scenario,
            instance_rng,
            incident_start,
            scenario_seed=seed,
            scenario_instance_index=i,
        )
        incidents.append(incident)

    return incidents
