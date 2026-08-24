"""Synthetic telemetry generator + failure injection engine.

BUILD_PLAN.md Phase 1's simulation layer:

- `scenario_schema.py` — Pydantic schema + loader for the hand-authored
  `failure_scenarios/*.yaml` ground-truth files.
- `baseline.py` — canonical-service seeding + "healthy" baseline telemetry
  generation.
- `injector.py` — `inject_failure()`: walks a scenario's `causal_chain`
  into a temporally coherent anomaly timeline and creates the `Incident`
  row. `backend/api/simulation.py` is the thin HTTP wrapper around it
  (`POST /api/simulation/failure`, `POST /api/simulation/reset`).
- `dataset.py` — `generate_dataset()`: the `--count N --seed S` seeded
  batch generator that reuses `inject_failure()` in a loop, so the eval
  dataset scales from ~30 (dev) to 100+ (portfolio) deterministically.
  `backend/scripts/generate_dataset.py` is its CLI wrapper.
"""

from __future__ import annotations
