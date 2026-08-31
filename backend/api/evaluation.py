"""`GET /api/evaluation/results` — Phase 7's read-only view onto the eval
harness's persisted output.

BUILD_PLAN.md Phase 7: *"`GET /api/evaluation/results` + an A/B/C/D
diagnostic table and a separate D operational table."*

## What this endpoint IS

A thin file read. `backend.evaluation.run_experiments` (the
`uv run python -m backend.evaluation.run_experiments --count N --seed S`
CLI) is the thing that actually *runs* the four architectures against a
seeded incident dataset — real OpenRouter API calls, real wall-clock, real
spend — and persists one JSON file per run under `evaluation/results/`
(`run_seed{S}_count{N}_{timestamp}.json`, see that module's `_write_results`
for the exact shape). This endpoint does nothing but locate the most
recently generated one of those files, parse it, and hand back its
`metadata` + both aggregate comparison tables (`diagnostic_aggregate`,
keyed A/B/C/D, and `operational_aggregate`, D-only) as a typed response —
fast and side-effect-free, safe to poll from a dashboard.

## What this endpoint is NOT

It never itself invokes `run_experiments.run_all`/`main` — there is no
"trigger a new benchmark run over HTTP" surface here, deliberately. Kicking
off a `--count 100` run is a deliberate, expensive, offline CLI action
(BUILD_PLAN.md's own cost note: "a full 4x100 run is thousands of OpenRouter
calls at Opus pricing"); an HTTP endpoint that could accidentally be hit by
a dashboard poll or a curious `curl` and silently start spending real money
is the wrong shape for that action. Generating results stays a conscious,
explicit CLI invocation; this endpoint only ever reads what that
invocation already wrote to disk.

## "Most recent" = highest file mtime, not `metadata.generated_at`

Both are effectively the same instant in practice — `_write_results`
computes `generated_at` and then immediately `path.write_text(...)`s the
same payload, so the two timestamps differ by however long `json.dumps`
takes on that payload (microseconds). Using file mtime means picking the
latest run costs one `Path.glob` + `stat()` per candidate file and zero
JSON parsing; using `generated_at` would mean parsing every results file
on disk just to find the one worth parsing for real. Documented here so
the choice reads as deliberate, not accidental, if it ever needs revisiting
(e.g. if results files are ever copied/rsynced between machines, which
would perturb mtime but not `generated_at`).

## No "select a specific run" query param (yet)

BUILD_PLAN.md's own phrasing is singular — "`GET /api/evaluation/results`"
- not "list of runs" or "run by id". Multiple result files can coexist on
disk (the CLI never overwrites an older run), but nothing in this phase's
scope asks for browsing history through the API; "return the latest run"
already satisfies the stated requirement. Added surface area (a
`?filename=`/`?seed=` selector, a `GET /api/evaluation/results/list`) is
easy to bolt on later once there's an actual caller (e.g. Phase 8's
dashboard) that needs it — not building it speculatively now.

## Response model shape

`metadata` and both aggregate tables are modeled as real Pydantic v2
classes — their field sets are fixed by `run_experiments.py`'s
`_aggregate_diagnostic`/`_aggregate_operational` functions, so a typed
model here gives a real, browsable OpenAPI contract instead of an opaque
blob. `per_incident` stays `list[dict[str, Any]]`: it's a nested,
per-architecture raw dump (diagnostic cells for A/B/C/D, optionally an
`operational_d` cell) that BUILD_PLAN.md's "diagnostic table" / "operational
table" framing doesn't actually ask this endpoint to reshape — fully
modeling it would mean mirroring `_score_diagnostic_run`/`_run_operational`'s
internal cell shapes here too, which is more structure than anything in
this phase consumes. Callers that want the two comparison tables (what
BUILD_PLAN.md names) get them fully typed; callers that want the raw
per-incident detail get it as parsed JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from backend.evaluation.run_experiments import _RESULTS_DIR as _DEFAULT_RESULTS_DIR

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

# Module-level binding (not a re-derivation of "<repo_root>/evaluation/results"
# from scratch, which could silently diverge from the CLI runner's own path)
# so tests can `monkeypatch.setattr(evaluation_module, "_RESULTS_DIR", tmp_path)`
# to point this endpoint at an isolated directory -- the same
# monkeypatch-the-module-under-test convention this codebase already uses
# elsewhere (e.g. tests/test_harness.py monkeypatching `experiment_a.ChatOpenRouter`).
_RESULTS_DIR: Path = _DEFAULT_RESULTS_DIR


class EvaluationRunMetadata(BaseModel):
    """`metadata` block written by `run_experiments._write_results`."""

    seed: int
    count: int
    generated_at: str
    skip_operational: bool


class DiagnosticAggregate(BaseModel):
    """One architecture's row in the A/B/C/D diagnostic comparison table --
    field set matches `run_experiments._aggregate_diagnostic`'s return dict
    exactly. `None` fields mean every scored cell for this architecture
    errored (see that function's docstring on excluding error cells from
    the means rather than averaging in a fabricated zero)."""

    n_incidents: int
    n_ok: int
    n_errors: int
    root_cause_accuracy_rate: float | None
    mean_evidence_precision: float | None
    mean_hallucination_rate: float | None
    mean_tool_call_count: float | None
    mean_evidence_per_tool_call: float | None
    mean_latency_seconds: float | None
    mean_total_tokens: float | None


class OperationalAggregate(BaseModel):
    """D's operational comparison table -- field set matches
    `run_experiments._aggregate_operational`'s return dict exactly."""

    n_incidents: int
    n_ok: int
    n_errors: int
    n_in_scope: int
    remediation_success_rate: float | None
    recovery_verification_accuracy: float | None
    wrong_remediation_rate: float | None
    n_wrong_remediation_attempts: int


class EvaluationResultsResponse(BaseModel):
    """Response body for `GET /api/evaluation/results`: the most recently
    generated run's full persisted payload, typed where BUILD_PLAN.md
    actually asks for structure (the two comparison tables) and left as
    parsed JSON where it doesn't (`per_incident`'s raw per-cell detail)."""

    metadata: EvaluationRunMetadata
    per_incident: list[dict[str, Any]]
    diagnostic_aggregate: dict[str, DiagnosticAggregate]
    operational_aggregate: OperationalAggregate | None


def _latest_results_file(results_dir: Path) -> Path | None:
    """The `*.json` file in `results_dir` with the highest mtime, or `None`
    if the directory doesn't exist or has no result files yet."""
    if not results_dir.is_dir():
        return None
    candidates = list(results_dir.glob("*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


@router.get("/results", response_model=EvaluationResultsResponse)
def get_evaluation_results() -> EvaluationResultsResponse:
    """Return the most recently generated `run_experiments` results file.

    404 (not 500) when no results exist yet -- an empty/missing
    `evaluation/results/` directory is an expected, unremarkable state
    before the CLI runner has ever been invoked, not a server error.

    Only ever looks in `_RESULTS_DIR` (the CLI runner's own default output
    directory). A run started with an explicit `--output-dir` elsewhere
    won't be found here -- BUILD_PLAN.md's Phase 7 verify step assumes the
    default location, so this is an accepted narrow gap, not a bug.
    """
    latest = _latest_results_file(_RESULTS_DIR)
    if latest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "no evaluation results found -- run "
                "`uv run python -m backend.evaluation.run_experiments "
                "--count N --seed S` first"
            ),
        )

    try:
        payload = json.loads(latest.read_text())
    except json.JSONDecodeError as exc:
        # A results file on disk that isn't valid JSON is a genuine server-
        # side data problem (a truncated write, manual corruption), not a
        # client error -- 500 is the honest status here, unlike the 404
        # above for the "nothing generated yet" case.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"results file {latest.name} is not valid JSON: {exc}",
        ) from exc

    return EvaluationResultsResponse.model_validate(payload)
