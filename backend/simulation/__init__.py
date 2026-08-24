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

Still to come: the `--count N --seed S` eval-dataset-scale batch generator
that reuses `inject_failure()` in a loop.
"""

from __future__ import annotations
