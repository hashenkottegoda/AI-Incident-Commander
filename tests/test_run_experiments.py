"""Tests for Phase 7's `backend.evaluation.run_experiments` CLI runner.

Zero real OpenRouter API calls. The five `run_experiment_*` functions
(`run_experiment_a/b/c/d/d_operational`) are already exercised end-to-end
against a real, genuinely-traced `_ScriptedChatModel` by
`tests/test_harness.py` -- this suite is scoped to `run_experiments.py`'s
OWN logic (per-cell orchestration, aggregation arithmetic, log-and-continue
on a raising cell, JSON persistence, the CLI entry point), so it
monkeypatches those five functions AT MODULE LEVEL on `run_experiments`
itself with small canned/raising fakes rather than re-driving the real
LLM-tracing machinery a second time. This works because `run_all` looks
up `run_experiment_a`/etc. as module globals at call time (plain Python
name resolution), so `monkeypatch.setattr(run_experiments, "run_experiment_b",
fake)` is seen by every call `run_all` makes, not just ones written after
the patch.

A real, tiny (`--count 2`) dataset is still generated for real via
`generate_dataset` against the real test Postgres (same DB/skip convention
as the rest of this suite) -- `run_all`'s incident/scenario wiring and
`score_operational_run`'s real DB queries run for real; only the experiment
functions themselves are faked.
"""

from __future__ import annotations

import json
import sys

import psycopg
import pytest

from backend.agents.schemas import DiagnosisResult, Hypothesis
from backend.agents.state import IncidentState
from backend.config import get_settings
from backend.db import SessionLocal
from backend.evaluation import run_experiments
from backend.evaluation.harness import ExperimentRun
from backend.models.incident import IncidentStatus
from backend.scripts.setup_checkpointer import to_psycopg_dsn
from backend.simulation.dataset import generate_dataset
from backend.simulation.scenario_schema import load_all_scenarios


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


# =============================================================================
# Pure aggregation-math tests -- no DB, synthetic cells with known values
# =============================================================================


def test_aggregate_diagnostic_excludes_errors_from_means():
    cells = [
        {
            "status": "ok",
            "root_cause_correct": True,
            "evidence_precision": 1.0,
            "hallucination_rate": 0.0,
            "tool_call_count": 4,
            "evidence_per_tool_call": 0.5,
            "latency_seconds": 2.0,
            "total_tokens": 100,
        },
        {
            "status": "ok",
            "root_cause_correct": False,
            "evidence_precision": 0.0,
            "hallucination_rate": 1.0,
            "tool_call_count": 2,
            "evidence_per_tool_call": None,
            "latency_seconds": 4.0,
            "total_tokens": 200,
        },
        {
            "status": "error",
            "root_cause_correct": None,
            "evidence_precision": None,
            "hallucination_rate": None,
            "tool_call_count": None,
            "evidence_per_tool_call": None,
            "latency_seconds": None,
            "total_tokens": None,
        },
    ]

    agg = run_experiments._aggregate_diagnostic(cells)

    assert agg["n_incidents"] == 3
    assert agg["n_ok"] == 2
    assert agg["n_errors"] == 1
    # 1 correct / 2 ok cells -- the errored cell must not count as a miss.
    assert agg["root_cause_accuracy_rate"] == pytest.approx(0.5)
    assert agg["mean_evidence_precision"] == pytest.approx(0.5)
    assert agg["mean_hallucination_rate"] == pytest.approx(0.5)
    assert agg["mean_tool_call_count"] == pytest.approx(3.0)
    # Only one cell has a non-None ratio -- mean is that value alone, not
    # dragged toward 0 by the None cell.
    assert agg["mean_evidence_per_tool_call"] == pytest.approx(0.5)
    assert agg["mean_latency_seconds"] == pytest.approx(3.0)
    assert agg["mean_total_tokens"] == pytest.approx(150.0)


def test_aggregate_diagnostic_all_errors_yields_none_means_not_zero():
    cells = [
        {
            "status": "error",
            "root_cause_correct": None,
            "evidence_precision": None,
            "hallucination_rate": None,
            "tool_call_count": None,
            "evidence_per_tool_call": None,
            "latency_seconds": None,
            "total_tokens": None,
        }
    ]

    agg = run_experiments._aggregate_diagnostic(cells)

    assert agg["n_ok"] == 0
    assert agg["n_errors"] == 1
    assert agg["root_cause_accuracy_rate"] is None
    assert agg["mean_evidence_precision"] is None
    assert agg["mean_latency_seconds"] is None


def test_aggregate_operational_denominators_match_scoring_docstring():
    """Mirrors `OperationalRunResult`'s own docstring: remediation success
    rate is over `in_scope` cells only, recovery-verification accuracy is
    over cells with a non-None verdict, and wrong-remediation rate is a
    FLATTENED per-attempt rate across every incident's flag list -- not an
    average of per-incident rates."""
    cells = [
        {
            "status": "ok",
            "in_scope": True,
            "recovered": True,
            "recovery_check_correct": True,
            "wrong_remediation_flags": [False],
        },
        {
            "status": "ok",
            "in_scope": True,
            "recovered": False,
            "recovery_check_correct": True,
            "wrong_remediation_flags": [True, True],
        },
        {
            # Out of scope (SAFE-only/rejected plan) -- must be excluded
            # from both rate denominators, not counted as a miss.
            "status": "ok",
            "in_scope": False,
            "recovered": None,
            "recovery_check_correct": None,
            "wrong_remediation_flags": [],
        },
        {
            "status": "error",
            "in_scope": None,
            "recovered": None,
            "recovery_check_correct": None,
            "wrong_remediation_flags": [],
        },
    ]

    agg = run_experiments._aggregate_operational(cells)

    assert agg["n_incidents"] == 4
    assert agg["n_ok"] == 3
    assert agg["n_errors"] == 1
    assert agg["n_in_scope"] == 2
    # 1 of 2 in-scope incidents recovered.
    assert agg["remediation_success_rate"] == pytest.approx(0.5)
    # Both in-scope incidents' Recovery Check calls were correct (2/2), even
    # though one of them didn't recover -- these are different questions.
    assert agg["recovery_verification_accuracy"] == pytest.approx(1.0)
    # 3 total attempts flattened across incidents, 2 wrong -- NOT the
    # average of per-incident rates (0/1 and 2/2 averaged would be 0.5,
    # not 2/3).
    assert agg["n_wrong_remediation_attempts"] == 3
    assert agg["wrong_remediation_rate"] == pytest.approx(2 / 3)


def test_aggregate_operational_no_in_scope_incidents_yields_none_rates():
    cells = [
        {
            "status": "ok",
            "in_scope": False,
            "recovered": None,
            "recovery_check_correct": None,
            "wrong_remediation_flags": [],
        }
    ]

    agg = run_experiments._aggregate_operational(cells)

    assert agg["n_in_scope"] == 0
    assert agg["remediation_success_rate"] is None
    assert agg["recovery_verification_accuracy"] is None
    assert agg["wrong_remediation_rate"] is None


# =============================================================================
# JSON persistence shape
# =============================================================================


def test_write_results_json_shape(tmp_path):
    output = {
        "per_incident": [{"incident_id": 1, "failure_type": "memory_leak"}],
        "diagnostic_aggregate": {"A": {"n_incidents": 1}},
        "operational_aggregate": None,
    }

    path = run_experiments._write_results(
        tmp_path, seed=42, count=1, skip_operational=True, output=output
    )

    assert path.exists()
    assert path.parent == tmp_path
    assert f"seed{42}" in path.name
    assert f"count{1}" in path.name

    loaded = json.loads(path.read_text())
    assert loaded["metadata"]["seed"] == 42
    assert loaded["metadata"]["count"] == 1
    assert loaded["metadata"]["skip_operational"] is True
    assert loaded["per_incident"] == output["per_incident"]
    assert loaded["diagnostic_aggregate"] == output["diagnostic_aggregate"]
    assert loaded["operational_aggregate"] is None


def test_write_results_creates_output_dir_if_missing(tmp_path):
    output_dir = tmp_path / "nested" / "results"
    assert not output_dir.exists()

    path = run_experiments._write_results(
        output_dir, seed=1, count=1, skip_operational=False, output={"per_incident": []}
    )

    assert output_dir.exists()
    assert path.parent == output_dir


# =============================================================================
# run_all: log-and-continue on a raising cell, real dataset + real DB
# =============================================================================


def _canned_run(category: str) -> ExperimentRun:
    return ExperimentRun(
        diagnosis=DiagnosisResult(
            root_cause_category=category,
            hypotheses=[Hypothesis(category=category, rationale="canned")],
            alternative_hypotheses=[],
            evidence=[],
            diagnostic_confidence=0.5,
        ),
        latency_seconds=0.01,
        tool_call_count=2,
        total_input_tokens=10,
        total_output_tokens=5,
    )


def _ok_sync_arch(scenarios, *, fail_for: frozenset[int] = frozenset()):
    def _fn(db, incident):  # noqa: ARG001
        if incident.id in fail_for:
            raise ValueError(f"boom for incident {incident.id}")
        return _canned_run(scenarios[incident.failure_type].root_cause_category)

    return _fn


def _ok_async_d(scenarios, *, fail_for: frozenset[int] = frozenset()):
    async def _fn(db, incident, *, qdrant_client=None):  # noqa: ARG001
        if incident.id in fail_for:
            raise ValueError(f"boom D for incident {incident.id}")
        return _canned_run(scenarios[incident.failure_type].root_cause_category)

    return _fn


def _ok_async_d_operational():
    async def _fn(db, incident, *, qdrant_client=None, approver="eval-harness"):  # noqa: ARG001
        # No AuditEvent rows ever get written by this fake -- score_operational_run
        # will correctly report in_scope=False for every incident (nothing to
        # recover/verify), which is fine: this test is about run_all's own
        # orchestration/aggregation, not about exercising a real remediation.
        return IncidentState(incident_id=incident.id, incident_status=IncidentStatus.DIAGNOSED)

    return _fn


async def test_run_all_continues_after_one_cell_raises(db, monkeypatch):
    scenarios = load_all_scenarios()
    incidents = generate_dataset(db, count=2, seed=90210)
    db.commit()

    failing_incident_id = incidents[0].id

    monkeypatch.setattr(run_experiments, "run_experiment_a", _ok_sync_arch(scenarios))
    monkeypatch.setattr(
        run_experiments,
        "run_experiment_b",
        _ok_sync_arch(scenarios, fail_for=frozenset({failing_incident_id})),
    )
    monkeypatch.setattr(run_experiments, "run_experiment_c", _ok_sync_arch(scenarios))
    monkeypatch.setattr(run_experiments, "run_experiment_d", _ok_async_d(scenarios))
    monkeypatch.setattr(
        run_experiments, "run_experiment_d_operational", _ok_async_d_operational()
    )

    output = await run_experiments.run_all(db, incidents, scenarios, skip_operational=False)

    # The whole run completed -- both incidents present, not aborted after
    # the first cell's failure.
    assert len(output["per_incident"]) == 2

    first_record = next(
        r for r in output["per_incident"] if r["incident_id"] == failing_incident_id
    )
    assert first_record["diagnostic"]["B"]["status"] == "error"
    assert "boom" in first_record["diagnostic"]["B"]["error"]
    # Sibling architectures for the SAME incident still ran despite B's failure.
    assert first_record["diagnostic"]["A"]["status"] == "ok"
    assert first_record["diagnostic"]["C"]["status"] == "ok"
    assert first_record["diagnostic"]["D"]["status"] == "ok"

    second_record = next(
        r for r in output["per_incident"] if r["incident_id"] != failing_incident_id
    )
    assert second_record["diagnostic"]["B"]["status"] == "ok"

    diag_agg = output["diagnostic_aggregate"]
    assert diag_agg["A"]["n_errors"] == 0
    assert diag_agg["B"]["n_errors"] == 1
    assert diag_agg["B"]["n_ok"] == 1
    # The one surviving B cell used the real ground-truth category by
    # construction of the fake, so it must score as correct.
    assert diag_agg["B"]["root_cause_accuracy_rate"] == pytest.approx(1.0)
    assert diag_agg["C"]["n_errors"] == 0
    assert diag_agg["D"]["n_errors"] == 0

    op_agg = output["operational_aggregate"]
    assert op_agg["n_incidents"] == 2
    assert op_agg["n_ok"] == 2
    assert op_agg["n_in_scope"] == 0
    assert op_agg["remediation_success_rate"] is None


async def test_run_all_continues_after_scoring_step_raises(db, monkeypatch):
    """Regression test: a failure DURING scoring (after the experiment call
    itself already succeeded and committed) must be isolated to that one
    cell exactly like a failure inside the experiment call is -- it must
    NOT propagate out of `run_all` and abort the whole run. `_run_sync_arch`/
    `_run_async_arch` previously called `_score_diagnostic_run(...)` OUTSIDE
    their own try/except, so a scoring exception (e.g. `evidence_precision`
    raising) would escape uncaught and discard every already-completed
    incident's results -- exactly what the module's own "Partial-failure
    handling" docstring section says must never happen."""
    scenarios = load_all_scenarios()
    incidents = generate_dataset(db, count=2, seed=24680)
    db.commit()

    monkeypatch.setattr(run_experiments, "run_experiment_a", _ok_sync_arch(scenarios))
    monkeypatch.setattr(run_experiments, "run_experiment_b", _ok_sync_arch(scenarios))
    monkeypatch.setattr(run_experiments, "run_experiment_c", _ok_sync_arch(scenarios))
    monkeypatch.setattr(run_experiments, "run_experiment_d", _ok_async_d(scenarios))
    monkeypatch.setattr(
        run_experiments, "run_experiment_d_operational", _ok_async_d_operational()
    )

    real_evidence_precision = run_experiments.evidence_precision

    # Fail the first call only (incident 0's Experiment A cell, given A/B/C/D
    # run in that order per incident) -- order-independent to reason about:
    # whichever cell hits it first becomes the one error cell, every other
    # call across both incidents uses the real scoring function.
    calls = {"n": 0}

    def _fail_once_evidence_precision(db, diagnosis):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("scoring boom")
        return real_evidence_precision(db, diagnosis)

    monkeypatch.setattr(run_experiments, "evidence_precision", _fail_once_evidence_precision)

    output = await run_experiments.run_all(db, incidents, scenarios, skip_operational=True)

    # The whole run completed -- both incidents present, not aborted after
    # the first cell's scoring failure.
    assert len(output["per_incident"]) == 2

    first_incident_cells = output["per_incident"][0]["diagnostic"]
    error_cells = [c for c in first_incident_cells.values() if c["status"] == "error"]
    assert len(error_cells) == 1
    assert "scoring boom" in error_cells[0]["error"]

    # Every other cell across both incidents scored normally.
    all_cells = [
        c for record in output["per_incident"] for c in record["diagnostic"].values()
    ]
    assert sum(1 for c in all_cells if c["status"] == "ok") == len(all_cells) - 1


# =============================================================================
# main(): the actual `uv run python -m backend.evaluation.run_experiments` path
# =============================================================================


def test_main_cli_writes_json_and_prints_both_tables(tmp_path, monkeypatch, capsys):
    scenarios = load_all_scenarios()

    monkeypatch.setattr(run_experiments, "run_experiment_a", _ok_sync_arch(scenarios))
    monkeypatch.setattr(run_experiments, "run_experiment_b", _ok_sync_arch(scenarios))
    monkeypatch.setattr(run_experiments, "run_experiment_c", _ok_sync_arch(scenarios))
    monkeypatch.setattr(run_experiments, "run_experiment_d", _ok_async_d(scenarios))
    monkeypatch.setattr(
        run_experiments, "run_experiment_d_operational", _ok_async_d_operational()
    )

    argv = [
        "run_experiments.py",
        "--count",
        "2",
        "--seed",
        "13579",
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    run_experiments.main()

    captured = capsys.readouterr()
    assert "Diagnostic comparison" in captured.out
    assert "Operational evaluation" in captured.out

    written = list(tmp_path.glob("run_seed13579_count2_*.json"))
    assert len(written) == 1

    payload = json.loads(written[0].read_text())
    assert payload["metadata"]["seed"] == 13579
    assert payload["metadata"]["count"] == 2
    assert len(payload["per_incident"]) == 2
    for arch in ("A", "B", "C", "D"):
        assert payload["diagnostic_aggregate"][arch]["n_incidents"] == 2
        assert payload["diagnostic_aggregate"][arch]["n_errors"] == 0
    assert payload["operational_aggregate"] is not None
