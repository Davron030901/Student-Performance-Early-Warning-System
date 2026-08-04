"""
End-to-end test of the full training pipeline (train.main()).

Marked `slow` and skipped by default: it retrains every model and takes a few
minutes. The unit tests cover the functions main() composes; this verifies they
compose correctly and that every promised artifact is actually produced.

    pytest -m slow          # run only this
    pytest -m "not slow"    # default suite
"""
import json
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_full_training_run_produces_every_promised_artifact(tmp_path, monkeypatch):
    from src.models import train

    project_root = Path.cwd()
    reports = project_root / "reports"
    artifacts = project_root / "models" / "artifacts"

    # preserve the shipped artifacts; other tests score against them
    backup = tmp_path / "backup"
    backup.mkdir()
    for f in artifacts.glob("*"):
        shutil.copy2(f, backup / f.name)

    try:
        train.main()

        # deliverables named in the brief
        assert (artifacts / "model.joblib").exists()
        assert (artifacts / "metadata.json").exists()
        for report in ["calibration.png", "checkpoint_comparison.csv",
                       "fairness_report.csv", "shap_global_importance.png"]:
            assert (reports / report).exists(), f"missing report: {report}"

        metadata = json.loads((artifacts / "metadata.json").read_text())
        for key in ["model_version", "trained_at", "checkpoint_fraction",
                    "feature_columns", "model_feature_columns", "held_out_metrics",
                    "cv_metrics", "trivial_baseline_metrics", "reference_medians",
                    "confusion_matrix_held_out", "n_training_rows"]:
            assert key in metadata, f"metadata missing {key}"

        # the model must still beat the naive rule after a fresh run
        assert metadata["held_out_metrics"]["recall"] > metadata["trivial_baseline_metrics"]["recall"]
        assert metadata["held_out_metrics"]["roc_auc"] > 0.8

        # the checkpoint comparison must actually compare the configured checkpoints
        import pandas as pd
        comparison = pd.read_csv(reports / "checkpoint_comparison.csv")
        import yaml
        expected = yaml.safe_load(open("config/config.yaml"))["prediction"]["eval_checkpoint_fractions"]
        assert sorted(comparison["checkpoint_fraction"]) == sorted(expected)

        # fairness must be broken out by every configured attribute
        fairness = pd.read_csv(reports / "fairness_report.csv")
        assert not fairness.empty
        assert fairness["recall"].between(0, 1).all()

    finally:
        for f in backup.glob("*"):
            shutil.copy2(f, artifacts / f.name)


def test_training_is_reproducible(tmp_path):
    """Fixed seeds in config mean two runs must agree, or none of the reported
    numbers can be quoted."""
    from src.models import train
    import yaml
    from src.data.ingest import load_raw_tables

    config = yaml.safe_load(open("config/config.yaml"))
    raw = load_raw_tables(config)
    a = train.run_checkpoint_evaluation(raw, config, 0.3)
    b = train.run_checkpoint_evaluation(raw, config, 0.3)
    assert a == b
