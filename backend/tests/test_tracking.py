"""
Tests for src/models/tracking.py.

The capstone requires experiment tracking. The important property is not that
MLflow specifically is used, but that every training run leaves a durable
record linking configuration to results — and, critically, that tracking can
never take down a training run.
"""
import json
from pathlib import Path

import pytest

from src.models.tracking import track_run, summarise_runs, RunRecorder


def test_a_run_records_params_metrics_and_artifacts(tmp_path, monkeypatch):
    log = tmp_path / "experiments.jsonl"
    monkeypatch.setattr("src.models.tracking.JSONL_PATH", log)
    monkeypatch.setattr("src.models.tracking.MLFLOW_AVAILABLE", False)

    artifact = tmp_path / "model.joblib"
    artifact.write_text("x")

    with track_run("unit-test-run") as run:
        run.log_params(checkpoint_fraction=0.3, primary_model="xgboost")
        run.log_metrics(prefix="held_out_", recall=0.78, precision=0.71)
        run.log_artifact(artifact)

    records = summarise_runs(log)
    assert len(records) == 1
    record = records[0]
    assert record["run_name"] == "unit-test-run"
    assert record["params"]["checkpoint_fraction"] == 0.3
    assert record["metrics"]["held_out_recall"] == 0.78
    assert str(artifact) in record["artifacts"]
    assert record["timestamp"] and record["duration_seconds"] >= 0


def test_runs_accumulate_rather_than_overwrite(tmp_path, monkeypatch):
    """History is the point — a tracker that keeps only the last run cannot
    show whether a change helped."""
    log = tmp_path / "experiments.jsonl"
    monkeypatch.setattr("src.models.tracking.JSONL_PATH", log)
    monkeypatch.setattr("src.models.tracking.MLFLOW_AVAILABLE", False)

    for i in range(3):
        with track_run(f"run-{i}") as run:
            run.log_metrics(recall=0.7 + i / 100)

    records = summarise_runs(log)
    assert [r["run_name"] for r in records] == ["run-0", "run-1", "run-2"]
    assert [r["metrics"]["recall"] for r in records] == [0.7, 0.71, 0.72]


def test_non_numeric_metrics_are_dropped_not_crashed_on():
    recorder = RunRecorder("x", use_mlflow=False)
    recorder.log_metrics(recall=0.8, note="looks good", flag=True, missing=None)
    assert recorder.metrics == {"recall": 0.8}


def test_missing_artifact_paths_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr("src.models.tracking.JSONL_PATH", tmp_path / "e.jsonl")
    monkeypatch.setattr("src.models.tracking.MLFLOW_AVAILABLE", False)
    with track_run("r") as run:
        run.log_artifact(tmp_path / "does-not-exist.png")
        assert run.artifacts == []


def test_tracking_failure_never_loses_the_run(tmp_path, monkeypatch):
    """If the tracker itself breaks, the run must still be recorded. Losing a
    trained model because the logging backend misbehaved would be a far worse
    outcome than losing the log line."""
    log = tmp_path / "experiments.jsonl"
    monkeypatch.setattr("src.models.tracking.JSONL_PATH", log)
    monkeypatch.setattr("src.models.tracking.MLFLOW_AVAILABLE", True)

    class Broken:
        def set_tracking_uri(self, *_a, **_k): raise RuntimeError("backend down")
        def set_experiment(self, *_a, **_k): raise RuntimeError("backend down")
        def start_run(self, *_a, **_k): raise RuntimeError("backend down")

    monkeypatch.setattr("src.models.tracking.mlflow", Broken())

    with track_run("resilient-run", tracking_uri="http://unreachable") as run:
        run.log_metrics(recall=0.5)

    records = summarise_runs(log)
    assert len(records) == 1
    assert records[0]["tracker"] == "jsonl"
    assert records[0]["metrics"]["recall"] == 0.5


def test_an_exception_inside_the_run_still_writes_the_record(tmp_path, monkeypatch):
    log = tmp_path / "experiments.jsonl"
    monkeypatch.setattr("src.models.tracking.JSONL_PATH", log)
    monkeypatch.setattr("src.models.tracking.MLFLOW_AVAILABLE", False)

    with pytest.raises(ValueError):
        with track_run("failing-run") as run:
            run.log_params(checkpoint_fraction=0.3)
            raise ValueError("training blew up")

    records = summarise_runs(log)
    assert len(records) == 1
    assert records[0]["params"]["checkpoint_fraction"] == 0.3


def test_summarise_runs_on_a_missing_log_returns_empty(tmp_path):
    assert summarise_runs(tmp_path / "nothing.jsonl") == []


def test_tracking_is_not_a_runtime_dependency():
    """MLflow must stay out of the deployed image — training-time concerns
    belong in requirements.txt, not requirements-api.txt."""
    runtime = Path("requirements-api.txt").read_text().lower()
    packages = {
        line.split("==")[0].split("[")[0].strip()
        for line in runtime.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert "mlflow" not in packages


def test_the_real_training_run_was_tracked():
    """Integration check against the log `make train` actually produced."""
    records = summarise_runs()
    if not records:
        pytest.skip("no training run recorded yet — run `make train`")
    latest = records[-1]
    assert latest["params"].get("checkpoint_fraction") is not None
    assert latest["metrics"].get("held_out_recall") is not None
    assert latest["metrics"].get("baseline_trivial_recall") is not None, \
        "the baseline comparison should be tracked, not just printed"
    assert latest["artifacts"], "the model artifact should be linked to the run"
