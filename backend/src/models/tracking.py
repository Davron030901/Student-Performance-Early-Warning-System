"""
Experiment tracking for the training pipeline.

The capstone requires that runs be tracked — parameters, metrics and the
resulting artifact — so a reported number can be traced back to the exact
configuration that produced it.

MLflow is used when it is installed. It is deliberately NOT in
`requirements-api.txt`: tracking is a training-time concern, and the deployed
container has no business carrying it (see the matplotlib note in explain.py
for the bug that taught us to keep those two dependency sets apart).

When MLflow is absent, this falls back to appending a JSON line per run to
`reports/experiments.jsonl`. That keeps `make train` working on a clean
checkout with only `requirements-api.txt` installed, and still leaves a
durable, diffable record of every run. The fallback is not a stub: it captures
the same fields, and `summarise_runs()` reads either source.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the install profile
    mlflow = None
    MLFLOW_AVAILABLE = False

EXPERIMENT_NAME = "edu02-early-warning"
JSONL_PATH = Path("reports/experiments.jsonl")


class RunRecorder:
    """Collects params/metrics for one run and writes them out on close."""

    def __init__(self, run_name: str, use_mlflow: bool):
        self.run_name = run_name
        self.use_mlflow = use_mlflow
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self.artifacts: list[str] = []
        self.started_at = time.time()

    def log_params(self, **params: Any) -> None:
        clean = {k: v for k, v in params.items() if v is not None}
        self.params.update(clean)
        if self.use_mlflow:
            mlflow.log_params({k: str(v) for k, v in clean.items()})

    def log_metrics(self, prefix: str = "", **metrics: Any) -> None:
        """Numeric values only — MLflow rejects anything else, and a metric
        that isn't a number isn't a metric."""
        numeric = {
            f"{prefix}{k}": float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        self.metrics.update(numeric)
        if self.use_mlflow and numeric:
            mlflow.log_metrics(numeric)

    def log_artifact(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        self.artifacts.append(str(path))
        if self.use_mlflow:
            mlflow.log_artifact(str(path))

    def to_record(self) -> dict:
        return {
            "run_name": self.run_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started_at)),
            "duration_seconds": round(time.time() - self.started_at, 1),
            "tracker": "mlflow" if self.use_mlflow else "jsonl",
            "params": self.params,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }


@contextmanager
def track_run(run_name: str, tracking_uri: str | None = None):
    """Context manager wrapping one training run.

    Tracking must never be able to fail a training run: if MLflow errors for
    any reason (no write access to mlruns/, a backend store misconfiguration),
    this degrades to the JSONL fallback rather than losing the model that was
    just trained.
    """
    use_mlflow = MLFLOW_AVAILABLE
    if use_mlflow:
        try:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(EXPERIMENT_NAME)
            active = mlflow.start_run(run_name=run_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"  [tracking] MLflow unavailable ({exc}); falling back to {JSONL_PATH}")
            use_mlflow = False
            active = None
    else:
        active = None

    recorder = RunRecorder(run_name, use_mlflow)
    try:
        yield recorder
    finally:
        record = recorder.to_record()
        JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with JSONL_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")
        if use_mlflow and active is not None:
            try:
                mlflow.end_run()
            except Exception:  # pragma: no cover
                pass
        where = "MLflow + " if use_mlflow else ""
        print(f"  [tracking] run '{run_name}' recorded ({where}{JSONL_PATH})")


def summarise_runs(path: Path = JSONL_PATH) -> list[dict]:
    """Reads the run log back. Used by the tests and handy for `mlflow ui`-less
    inspection."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
