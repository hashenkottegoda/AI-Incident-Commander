"""Phase 7's single-command experiment runner (BUILD_PLAN.md Phase 7).

    uv run python -m backend.evaluation.run_experiments --count 5 --seed 42

Generates a seeded incident dataset (`backend.simulation.dataset.
generate_dataset(db, count, seed)` -- the same call `backend.scripts.
generate_dataset`'s CLI makes, invoked directly here rather than requiring a
separate manual step, per BUILD_PLAN.md Phase 7's "Generate the eval dataset
from the Phase 1 scenario generator" phrasing), then for every incident runs:

- Experiments A/B/C (`backend.evaluation.harness.run_experiment_{a,b,c}`,
  sync) and D-diagnostic (`run_experiment_d`, async, halts before Response
  Planner) -- scored against that incident's `FailureScenario` ground truth
  via `backend.evaluation.scoring`'s deterministic diagnostic functions
  (root-cause accuracy, evidence precision, hallucination rate, tool-call
  efficiency), plus the latency/token fields already on `ExperimentRun`.
- D only, additionally: `run_experiment_d_operational` (drives the FULL
  closed loop -- Response Planner -> Risk Classifier -> HITL -> Action
  Executor -> Recovery Check -- auto-approving along the way) scored via
  `score_operational_run`, for the separate operational table.

Both the diagnostic (A/B/C/D) and operational (D-only) comparison tables are
printed to stdout and persisted, together with every per-incident raw result,
as one JSON file under `evaluation/results/` (BUILD_PLAN.md's repo-structure
section names this directory as "experiment comparison output"; created here
if missing).

## Metrics NOT covered here, and why

The parent task scoped this runner to what `backend.evaluation.harness`/
`scoring.py` already implement: root-cause accuracy, evidence precision,
hallucination rate, tool-call efficiency, latency, and token cost on the
diagnostic side; remediation success rate, recovery-verification accuracy,
and wrong-remediation rate on D's operational side. Two metrics named in
this project's broader eval framing -- **severity accuracy** and **human
override rate** -- are deliberately NOT scored here: `DiagnosisResult`
(the shared A/B/C/D schema) carries no predicted-severity field and
`scoring.py` has no corresponding function, and there is no persisted
"a human overrode the AI's recommendation" signal to read yet (every
`AuditEvent` decision in this eval harness is an unattended auto-approval,
not a real human override). Adding either would mean extending
`DiagnosisResult`/`scoring.py` first -- out of scope for this runner, which
only wires together what already exists. Flagged here rather than silently
producing a comparison table that looks complete but isn't.

## Partial-failure handling: log-and-continue, per (incident, architecture) cell

A `--count 40`/`100` run against the real OpenRouter API is real wall-clock
time and real spend. Aborting an entire run because ONE incident's ONE
architecture raised (a transient API error, a rate limit, a genuine bug that
only one scenario triggers) would throw away everything already completed
for no benefit -- the comparison table is built from independent per-cell
measurements, not a single all-or-nothing computation. So every
(incident, architecture) cell runs in its own try/except: a failure is
logged with `logger.exception` (full traceback, not swallowed silently),
the session is rolled back to a clean state for the next cell, and the cell
is recorded in the raw JSON as `{"status": "error", "error": "..."}` with
every metric field `None` -- excluded from that architecture's aggregate
statistics (denominators shrink to the cells that actually succeeded) rather
than corrupting the mean with a fabricated zero. `n_errors` on each
aggregate makes a run with real failures honestly visible in the table
itself, not hidden.

## Running both D-diagnostic and D-operational against the same incident

`run_experiment_d` (`run_incident_graph_to_diagnosis`) and
`run_experiment_d_operational` (`run_incident_graph`) are both run against
the SAME incident here, in that order. These are NOT the same LangGraph
thread: `run_incident_graph_to_diagnosis` checkpoints under
`thread_id = f"{incident.id}-diagnostic-eval"`, while `run_incident_graph`/
`resume_incident_graph`/`get_incident_thread_state` (what
`run_experiment_d_operational` drives) use `thread_id = str(incident.id)`
-- two genuinely distinct checkpoint threads for the same incident row, by
design (see `run_incident_graph_to_diagnosis`'s own docstring in
`backend/graph.py`: "so this function's checkpoint can never collide with
`run_incident_graph`/`resume_incident_graph`'s thread for the same incident
row"). So there is no shared-thread resume/replay hazard to reason about
here at all.

What actually needs checking is simpler: could D-diagnostic's run leave any
durable side effect that D-operational's subsequent run would then
double-execute or otherwise be corrupted by? No -- verified directly:
`triage_node`/`investigation_node`/`rag_node`/`root_cause_node` (the only
nodes `run_incident_graph_to_diagnosis` ever reaches, since it halts via
`interrupt_before=["response_planner"]`) make zero DB writes. Every write
in this graph (`AuditEvent` rows, `MetricPoint` rows, `incident.status =`)
happens in `response_planner_node`, `action_executor_node`, or
`recovery_check_node` -- none of which D-diagnostic's halted run ever
reaches. So D-operational's subsequent full run starts from a database
state genuinely unaffected by the diagnostic run that preceded it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.db import SessionLocal
from backend.evaluation.harness import (
    ExperimentRun,
    run_experiment_a,
    run_experiment_b,
    run_experiment_c,
    run_experiment_d,
    run_experiment_d_operational,
)
from backend.evaluation.scoring import (
    evidence_precision,
    hallucination_rate,
    root_cause_accuracy,
    score_operational_run,
    tool_call_efficiency,
)
from backend.rag.qdrant_client import get_qdrant_client
from backend.simulation.dataset import generate_dataset
from backend.simulation.scenario_schema import load_all_scenarios

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from sqlalchemy.orm import Session

    from backend.models import Incident
    from backend.simulation.scenario_schema import FailureScenario

logger = logging.getLogger(__name__)

_ARCHITECTURES: tuple[str, ...] = ("A", "B", "C", "D")

# repo root: backend/evaluation/run_experiments.py -> backend/evaluation ->
# backend -> repo root. Same `.parents[N]` pattern `scenario_schema.py`'s
# SCENARIOS_DIR already uses.
_RESULTS_DIR: Path = Path(__file__).resolve().parents[2] / "evaluation" / "results"


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 7's single-command A/B/C/D experiment runner: generates a "
            "seeded incident dataset, runs all four architectures against it, "
            "and prints + persists the diagnostic and D-operational comparison "
            "tables."
        )
    )
    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help=(
            "Number of incidents to generate (default: 30, the 'dev' dataset "
            "size). BUILD_PLAN.md: --count 5 for a CI smoke run, --count 40 for "
            "the dev loop, --count 100 for occasional portfolio benchmarking -- "
            "the latter is real wall-clock time and rate-limited OpenRouter quota."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help=(
            "Mandatory RNG seed, forwarded to generate_dataset(). --count N "
            "--seed S must always reproduce the identical N incidents so "
            "repeated runs are comparable."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_RESULTS_DIR,
        help=f"Directory to write the results JSON into (default: {_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--skip-operational",
        action="store_true",
        help=(
            "Skip D's operational closed-loop run (run_experiment_d_operational) "
            "-- diagnostic-only (A/B/C/D), faster and cheaper. The operational "
            "table is omitted from stdout and null in the JSON output."
        ),
    )
    return parser.parse_args(argv)


# =============================================================================
# Per-cell diagnostic scoring (one (incident, architecture) cell)
# =============================================================================


def _error_diagnostic_cell(error: str) -> dict[str, Any]:
    """Shape for a diagnostic cell whose experiment run raised -- every
    metric field `None` so it's unambiguous in the raw JSON and structurally
    excluded from `_aggregate_diagnostic`'s means (which filter on
    `status == "ok"`), not averaged in as a fabricated zero."""
    return {
        "status": "error",
        "error": error,
        "predicted_root_cause_category": None,
        "diagnostic_confidence": None,
        "root_cause_correct": None,
        "evidence_precision": None,
        "hallucination_rate": None,
        "evidence_count": None,
        "tool_call_count": None,
        "evidence_per_tool_call": None,
        "latency_seconds": None,
        "total_input_tokens": None,
        "total_output_tokens": None,
        "total_tokens": None,
    }


def _score_diagnostic_run(
    db: Session, run: ExperimentRun, ground_truth_category: str
) -> dict[str, Any]:
    """Score one successful `ExperimentRun` against `scoring.py`'s
    deterministic diagnostic functions plus the measurement fields already
    on `ExperimentRun` (latency, token usage)."""
    efficiency = tool_call_efficiency(run.tool_call_count, len(run.diagnosis.evidence))
    total_tokens = (
        run.total_input_tokens + run.total_output_tokens
        if run.total_input_tokens is not None and run.total_output_tokens is not None
        else None
    )
    return {
        "status": "ok",
        "error": None,
        "predicted_root_cause_category": run.diagnosis.root_cause_category,
        "diagnostic_confidence": run.diagnosis.diagnostic_confidence,
        "root_cause_correct": root_cause_accuracy(run.diagnosis, ground_truth_category),
        "evidence_precision": evidence_precision(db, run.diagnosis),
        "hallucination_rate": hallucination_rate(db, run.diagnosis),
        "evidence_count": len(run.diagnosis.evidence),
        "tool_call_count": efficiency.tool_call_count,
        "evidence_per_tool_call": efficiency.evidence_per_tool_call,
        "latency_seconds": run.latency_seconds,
        "total_input_tokens": run.total_input_tokens,
        "total_output_tokens": run.total_output_tokens,
        "total_tokens": total_tokens,
    }


def _run_sync_arch(
    label: str,
    fn,
    db: Session,
    incident: Incident,
    ground_truth_category: str,
) -> dict[str, Any]:
    """Run one of the sync experiments (A/B/C) for one incident, log-and-
    continue on failure -- see module docstring's "Partial-failure handling"
    section."""
    try:
        run = fn(db, incident)
        db.commit()
        return _score_diagnostic_run(db, run, ground_truth_category)
    except Exception as exc:  # noqa: BLE001 -- deliberate log-and-continue
        logger.exception(
            "Experiment %s raised for incident %d (%s) -- recording an error "
            "cell and continuing the run",
            label,
            incident.id,
            incident.failure_type,
        )
        db.rollback()
        return _error_diagnostic_cell(f"{type(exc).__name__}: {exc}")


async def _run_async_arch(
    label: str,
    coro_fn,
    db: Session,
    incident: Incident,
    ground_truth_category: str,
    qdrant_client: QdrantClient,
) -> dict[str, Any]:
    """Run one of the async experiments (D-diagnostic) for one incident,
    log-and-continue on failure -- see module docstring's "Partial-failure
    handling" section."""
    try:
        run = await coro_fn(db, incident, qdrant_client=qdrant_client)
        db.commit()
        return _score_diagnostic_run(db, run, ground_truth_category)
    except Exception as exc:  # noqa: BLE001 -- deliberate log-and-continue
        logger.exception(
            "Experiment %s raised for incident %d (%s) -- recording an error "
            "cell and continuing the run",
            label,
            incident.id,
            incident.failure_type,
        )
        db.rollback()
        return _error_diagnostic_cell(f"{type(exc).__name__}: {exc}")


# =============================================================================
# Per-incident operational scoring (D only)
# =============================================================================


def _error_operational_cell(error: str) -> dict[str, Any]:
    return {
        "status": "error",
        "error": error,
        "in_scope": None,
        "recovered": None,
        "recovery_check_correct": None,
        "wrong_remediation_flags": [],
    }


async def _run_operational(
    db: Session, incident: Incident, scenario: FailureScenario, qdrant_client: QdrantClient
) -> dict[str, Any]:
    """Drive D's full closed loop for one incident and score it. Log-and-
    continue on failure, same convention as the diagnostic cells -- note
    that `run_experiment_d_operational` commits real `AuditEvent`/incident
    rows as it goes (per that function's own docstring), so a failure
    partway through may leave some of those writes durably committed; the
    `db.rollback()` here only discards whatever was NOT yet committed at the
    point of failure, which is the honest behavior for a real eval run
    against a real database, not a test needing full isolation."""
    try:
        final_state = await run_experiment_d_operational(db, incident, qdrant_client=qdrant_client)
        db.commit()
        result = score_operational_run(db, final_state, scenario)
    except Exception as exc:  # noqa: BLE001 -- deliberate log-and-continue
        logger.exception(
            "Experiment D operational raised for incident %d (%s) -- "
            "recording an error cell and continuing the run",
            incident.id,
            incident.failure_type,
        )
        db.rollback()
        return _error_operational_cell(f"{type(exc).__name__}: {exc}")
    return {
        "status": "ok",
        "error": None,
        "in_scope": result.in_scope,
        "recovered": result.recovered,
        "recovery_check_correct": result.recovery_check_correct,
        "wrong_remediation_flags": result.wrong_remediation_flags,
    }


# =============================================================================
# Aggregation
# =============================================================================


def _mean(values: list[float]) -> float | None:
    """`None` for an empty list rather than raising/NaN -- keeps every
    aggregate field a clean `float | None` the JSON writer and the table
    printer can both handle uniformly."""
    return statistics.fmean(values) if values else None


def _aggregate_diagnostic(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """One architecture's per-incident diagnostic cells -> the row of the
    A/B/C/D comparison table. Means are computed over `status == "ok"`
    cells only -- see module docstring's "Partial-failure handling"."""
    ok_cells = [c for c in cells if c["status"] == "ok"]
    n = len(cells)
    n_ok = len(ok_cells)
    return {
        "n_incidents": n,
        "n_ok": n_ok,
        "n_errors": n - n_ok,
        "root_cause_accuracy_rate": _mean(
            [1.0 if c["root_cause_correct"] else 0.0 for c in ok_cells]
        ),
        "mean_evidence_precision": _mean([c["evidence_precision"] for c in ok_cells]),
        "mean_hallucination_rate": _mean([c["hallucination_rate"] for c in ok_cells]),
        "mean_tool_call_count": _mean([float(c["tool_call_count"]) for c in ok_cells]),
        "mean_evidence_per_tool_call": _mean(
            [
                c["evidence_per_tool_call"]
                for c in ok_cells
                if c["evidence_per_tool_call"] is not None
            ]
        ),
        "mean_latency_seconds": _mean([c["latency_seconds"] for c in ok_cells]),
        "mean_total_tokens": _mean(
            [c["total_tokens"] for c in ok_cells if c["total_tokens"] is not None]
        ),
    }


def _aggregate_operational(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """D's per-incident operational cells -> the operational table.

    Denominators follow `OperationalRunResult`'s own docstring exactly:
    - remediation success rate = count(recovered is True) / count(in_scope
      is True), over `status == "ok"` cells.
    - recovery-verification accuracy = count(recovery_check_correct is True)
      / count(recovery_check_correct is not None).
    - wrong-remediation rate = every `True` flag / every flag, FLATTENED
      across all incidents first (a per-attempt rate, not per-incident --
      see `wrong_remediation_flags`'s docstring).

    An `in_scope=False` incident (SAFE-only plan, or a rejected HIGH_IMPACT
    recommendation) is excluded from both rate denominators entirely, per
    `score_operational_run`'s own docstring -- never counted as a failure of
    either metric.
    """
    ok_cells = [c for c in cells if c["status"] == "ok"]
    n = len(cells)
    n_ok = len(ok_cells)
    in_scope_cells = [c for c in ok_cells if c["in_scope"]]
    recovery_scored = [c for c in ok_cells if c["recovery_check_correct"] is not None]
    all_flags = [flag for c in ok_cells for flag in c["wrong_remediation_flags"]]
    return {
        "n_incidents": n,
        "n_ok": n_ok,
        "n_errors": n - n_ok,
        "n_in_scope": len(in_scope_cells),
        "remediation_success_rate": (
            sum(1 for c in in_scope_cells if c["recovered"]) / len(in_scope_cells)
            if in_scope_cells
            else None
        ),
        "recovery_verification_accuracy": (
            sum(1 for c in recovery_scored if c["recovery_check_correct"]) / len(recovery_scored)
            if recovery_scored
            else None
        ),
        "wrong_remediation_rate": (
            sum(1 for f in all_flags if f) / len(all_flags) if all_flags else None
        ),
        "n_wrong_remediation_attempts": len(all_flags),
    }


# =============================================================================
# Driving the full dataset
# =============================================================================


async def run_all(
    db: Session,
    incidents: list[Incident],
    scenarios: dict[str, FailureScenario],
    *,
    skip_operational: bool,
) -> dict[str, Any]:
    """Run every architecture against every incident, score each cell, and
    return the raw per-incident records plus both aggregated tables."""
    qdrant_client = get_qdrant_client()
    diagnostic_cells: dict[str, list[dict[str, Any]]] = {arch: [] for arch in _ARCHITECTURES}
    operational_cells: list[dict[str, Any]] = []
    per_incident_records: list[dict[str, Any]] = []

    for i, incident in enumerate(incidents, start=1):
        scenario = scenarios.get(incident.failure_type)
        if scenario is None:
            # Should never happen -- generate_dataset only ever assigns real
            # failure_scenarios/*.yaml types -- but fail loudly rather than
            # silently skipping ground truth if it ever does.
            raise ValueError(
                f"incident {incident.id} has failure_type {incident.failure_type!r}, which "
                f"has no matching FailureScenario in load_all_scenarios()"
            )
        ground_truth_category = scenario.root_cause_category
        logger.info(
            "[%d/%d] incident %d (%s, ground truth=%s)",
            i,
            len(incidents),
            incident.id,
            incident.failure_type,
            ground_truth_category,
        )

        cell_a = _run_sync_arch("A", run_experiment_a, db, incident, ground_truth_category)
        cell_b = _run_sync_arch("B", run_experiment_b, db, incident, ground_truth_category)
        cell_c = _run_sync_arch("C", run_experiment_c, db, incident, ground_truth_category)
        cell_d = await _run_async_arch(
            "D", run_experiment_d, db, incident, ground_truth_category, qdrant_client
        )

        diagnostic_cells["A"].append(cell_a)
        diagnostic_cells["B"].append(cell_b)
        diagnostic_cells["C"].append(cell_c)
        diagnostic_cells["D"].append(cell_d)

        record: dict[str, Any] = {
            "incident_id": incident.id,
            "failure_type": incident.failure_type,
            "ground_truth_category": ground_truth_category,
            "severity": incident.severity.value,
            "diagnostic": {"A": cell_a, "B": cell_b, "C": cell_c, "D": cell_d},
        }

        if not skip_operational:
            # Run AFTER D-diagnostic on the same incident -- see module
            # docstring's "Reusing one incident's thread" section for why
            # this is safe.
            op_cell = await _run_operational(db, incident, scenario, qdrant_client)
            operational_cells.append(op_cell)
            record["operational_d"] = op_cell

        per_incident_records.append(record)

    diagnostic_aggregate = {
        arch: _aggregate_diagnostic(cells) for arch, cells in diagnostic_cells.items()
    }
    operational_aggregate = (
        None if skip_operational else _aggregate_operational(operational_cells)
    )

    return {
        "per_incident": per_incident_records,
        "diagnostic_aggregate": diagnostic_aggregate,
        "operational_aggregate": operational_aggregate,
    }


# =============================================================================
# Printing
# =============================================================================


def _fmt_pct(x: float | None) -> str:
    return "N/A" if x is None else f"{x * 100:.1f}%"


def _fmt_num(x: float | None, digits: int = 2) -> str:
    return "N/A" if x is None else f"{x:.{digits}f}"


def _print_table(headers: list[str], rows: list[list[str]], *, title: str) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    widths = [
        max(len(headers[i]), max((len(r[i]) for r in rows), default=0))
        for i in range(len(headers))
    ]

    def _fmt_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(w) for cell, w in zip(cells, widths, strict=True))

    print(_fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(_fmt_row(row))


def print_diagnostic_table(aggregate: dict[str, dict[str, Any]]) -> None:
    headers = [
        "Arch",
        "N",
        "RootCauseAcc",
        "EvidPrecision",
        "HallucRate",
        "ToolCalls",
        "Evid/Call",
        "Latency(s)",
        "MeanTokens",
        "Errors",
    ]
    rows = []
    for arch in _ARCHITECTURES:
        a = aggregate[arch]
        rows.append(
            [
                arch,
                str(a["n_incidents"]),
                _fmt_pct(a["root_cause_accuracy_rate"]),
                _fmt_num(a["mean_evidence_precision"]),
                _fmt_num(a["mean_hallucination_rate"]),
                _fmt_num(a["mean_tool_call_count"]),
                _fmt_num(a["mean_evidence_per_tool_call"]),
                _fmt_num(a["mean_latency_seconds"]),
                _fmt_num(a["mean_total_tokens"], digits=0),
                str(a["n_errors"]),
            ]
        )
    _print_table(
        headers,
        rows,
        title="Diagnostic comparison (A: context-stuffing, B: +tools, C: +RAG, D: full graph)",
    )


def print_operational_table(aggregate: dict[str, Any] | None) -> None:
    if aggregate is None:
        print("\nOperational evaluation (D only): skipped (--skip-operational)")
        return
    headers = ["Metric", "Value"]
    rows = [
        [
            "Incidents (total / in-scope / errors)",
            f"{aggregate['n_incidents']} / {aggregate['n_in_scope']} / {aggregate['n_errors']}",
        ],
        ["Remediation success rate", _fmt_pct(aggregate["remediation_success_rate"])],
        [
            "Recovery-verification accuracy",
            _fmt_pct(aggregate["recovery_verification_accuracy"]),
        ],
        [
            "Wrong-remediation rate",
            f"{_fmt_pct(aggregate['wrong_remediation_rate'])} "
            f"({aggregate['n_wrong_remediation_attempts']} remediation attempts)",
        ],
    ]
    _print_table(headers, rows, title="Operational evaluation (D only, full closed loop)")


# =============================================================================
# Persisting results
# =============================================================================


def _write_results(
    output_dir: Path, *, seed: int, count: int, skip_operational: bool, output: dict[str, Any]
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    # Filename encodes seed/count/timestamp so repeated runs (dev iteration,
    # or a later --count 100 benchmark) never clobber an earlier run's file.
    path = output_dir / f"run_seed{seed}_count{count}_{timestamp}.json"
    payload = {
        "metadata": {
            "seed": seed,
            "count": count,
            "generated_at": timestamp,
            "skip_operational": skip_operational,
        },
        **output,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


# =============================================================================
# Entry point
# =============================================================================


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    db = SessionLocal()
    try:
        incidents = generate_dataset(db, count=args.count, seed=args.seed)
        db.commit()
        logger.info(
            "Generated %d incidents (seed=%d, count=%d).", len(incidents), args.seed, args.count
        )

        scenarios = load_all_scenarios()

        output = asyncio.run(
            run_all(db, incidents, scenarios, skip_operational=args.skip_operational)
        )
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    print_diagnostic_table(output["diagnostic_aggregate"])
    print_operational_table(output["operational_aggregate"])

    result_path = _write_results(
        args.output_dir,
        seed=args.seed,
        count=args.count,
        skip_operational=args.skip_operational,
        output=output,
    )
    logger.info("\nRaw results + aggregated tables written to %s", result_path)


if __name__ == "__main__":
    main()
