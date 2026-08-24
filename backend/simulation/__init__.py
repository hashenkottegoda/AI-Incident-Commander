"""Synthetic telemetry generator + failure injection engine.

This package is still being built out (BUILD_PLAN.md Phase 1): so far it
only holds `scenario_schema.py`, the Pydantic schema + loader for the
hand-authored `failure_scenarios/*.yaml` ground-truth files. The baseline
telemetry generator, failure injection engine, and `/api/simulation/*`
routes are separate, later sub-steps.
"""

from __future__ import annotations
