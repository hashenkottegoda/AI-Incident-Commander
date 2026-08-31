"""Tests for `GET /api/evaluation/results`.

No Postgres/OpenRouter dependency -- this endpoint only reads a JSON file
off disk (written offline by `backend.evaluation.run_experiments`), so
unlike `tests/test_simulation_api.py` there's no skipif-on-live-Postgres
guard needed here, same as `tests/test_health.py`.

Each test points `backend.api.evaluation._RESULTS_DIR` at an isolated
`tmp_path` via `monkeypatch`, per that module's own docstring on how it
expects to be tested -- never touches the real `evaluation/results/`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

import backend.api.evaluation as evaluation_module
from backend.main import app

client = TestClient(app)


def _payload(*, seed: int, count: int, generated_at: str, skip_operational: bool = False) -> dict:
    """A minimal-but-shape-complete `run_experiments._write_results` payload."""
    diagnostic_row = {
        "n_incidents": count,
        "n_ok": count,
        "n_errors": 0,
        "root_cause_accuracy_rate": 0.8,
        "mean_evidence_precision": 0.9,
        "mean_hallucination_rate": 0.1,
        "mean_tool_call_count": 3.0,
        "mean_evidence_per_tool_call": 1.5,
        "mean_latency_seconds": 2.5,
        "mean_total_tokens": 1200.0,
    }
    operational_row = None
    if not skip_operational:
        operational_row = {
            "n_incidents": count,
            "n_ok": count,
            "n_errors": 0,
            "n_in_scope": count,
            "remediation_success_rate": 0.75,
            "recovery_verification_accuracy": 0.9,
            "wrong_remediation_rate": 0.05,
            "n_wrong_remediation_attempts": 4,
        }
    return {
        "metadata": {
            "seed": seed,
            "count": count,
            "generated_at": generated_at,
            "skip_operational": skip_operational,
        },
        "per_incident": [
            {
                "incident_id": 1,
                "failure_type": "db_connection_exhaustion",
                "ground_truth_category": "database_connection_pool",
                "severity": "P1",
                "diagnostic": {"A": {}, "B": {}, "C": {}, "D": {}},
            }
        ],
        "diagnostic_aggregate": {
            "A": diagnostic_row,
            "B": diagnostic_row,
            "C": diagnostic_row,
            "D": diagnostic_row,
        },
        "operational_aggregate": operational_row,
    }


def _write_run(
    results_dir: Path, *, seed: int, count: int, generated_at: str, mtime: float
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"run_seed{seed}_count{count}_{generated_at}.json"
    path.write_text(json.dumps(_payload(seed=seed, count=count, generated_at=generated_at)))
    os.utime(path, (mtime, mtime))
    return path


def test_get_results_returns_200_with_expected_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation_module, "_RESULTS_DIR", tmp_path)
    _write_run(tmp_path, seed=42, count=5, generated_at="20260101T000000Z", mtime=1_000_000)

    response = client.get("/api/evaluation/results")

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"] == {
        "seed": 42,
        "count": 5,
        "generated_at": "20260101T000000Z",
        "skip_operational": False,
    }
    assert set(body["diagnostic_aggregate"]) == {"A", "B", "C", "D"}
    assert body["diagnostic_aggregate"]["A"]["root_cause_accuracy_rate"] == 0.8
    assert body["operational_aggregate"]["remediation_success_rate"] == 0.75
    assert len(body["per_incident"]) == 1
    assert body["per_incident"][0]["incident_id"] == 1


def test_get_results_returns_most_recent_run_by_mtime(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation_module, "_RESULTS_DIR", tmp_path)
    # Older file has a lexicographically LARGER seed than the newer one, so
    # this would fail if the endpoint ever sorted by filename instead of
    # mtime (seed digits sort before the timestamp in the filename).
    _write_run(tmp_path, seed=99, count=5, generated_at="20260101T000000Z", mtime=1_000_000)
    _write_run(tmp_path, seed=7, count=10, generated_at="20260102T000000Z", mtime=2_000_000)

    response = client.get("/api/evaluation/results")

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["seed"] == 7
    assert body["metadata"]["count"] == 10
    assert body["metadata"]["generated_at"] == "20260102T000000Z"


def test_get_results_404_when_results_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation_module, "_RESULTS_DIR", tmp_path / "does-not-exist")

    response = client.get("/api/evaluation/results")

    assert response.status_code == 404
    assert "no evaluation results found" in response.json()["detail"]


def test_get_results_404_when_results_dir_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation_module, "_RESULTS_DIR", tmp_path)

    response = client.get("/api/evaluation/results")

    assert response.status_code == 404
