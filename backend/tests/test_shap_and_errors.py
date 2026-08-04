"""
Covers the remaining untested paths: SHAP computation against the real model,
report plot generation, the multi-checkpoint evaluation that answers "is this
early enough to matter", and the API's internal error handling.
"""
import importlib
import sys

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.models.explain import (
    compute_shap_values, save_global_importance_plot, top_factors_for_student,
)
from src.models.train import run_checkpoint_evaluation


@pytest.fixture(scope="module")
def fitted_model():
    return joblib.load("models/artifacts/model.joblib")


@pytest.fixture(scope="module")
def scoring_frame(dataset, metadata):
    """A small slice of real data aligned to the model's expected columns."""
    from src.models.encoding import align_to_model_columns
    rows = dataset["X"][dataset["feature_columns"]].head(40)
    return align_to_model_columns(rows, metadata["model_feature_columns"])


# ── compute_shap_values ───────────────────────────────────────────────────

def test_shap_values_have_one_row_per_student_and_one_column_per_feature(fitted_model, scoring_frame):
    shap_values, explainer = compute_shap_values(fitted_model, scoring_frame)
    assert shap_values.shape == scoring_frame.shape
    assert explainer is not None


def test_shap_values_reconstruct_the_model_prediction(fitted_model, scoring_frame):
    """SHAP's additivity property: base value plus contributions equals the
    model's raw output. If this breaks, explanations are decorative rather than
    faithful, which is worse than having none."""
    shap_values, explainer = compute_shap_values(fitted_model, scoring_frame)
    raw_margin = fitted_model.predict(scoring_frame, output_margin=True)
    reconstructed = explainer.expected_value + shap_values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, raw_margin, rtol=1e-3, atol=1e-3)


def test_shap_values_are_finite(fitted_model, scoring_frame):
    shap_values, _ = compute_shap_values(fitted_model, scoring_frame)
    assert np.isfinite(shap_values).all()


def test_shap_ranks_engagement_features_highly(fitted_model, scoring_frame):
    """A sanity check on the model itself: if demographics dominated the
    explanation, the fairness story would be very different."""
    shap_values, _ = compute_shap_values(fitted_model, scoring_frame)
    importance = dict(zip(scoring_frame.columns, np.abs(shap_values).mean(axis=0)))
    top_five = sorted(importance, key=importance.get, reverse=True)[:5]
    assert any(name.startswith("vle_") or name in {"avg_early_score", "n_submitted"}
               for name in top_five), f"top features look wrong: {top_five}"


def test_end_to_end_explanation_from_the_real_model(fitted_model, scoring_frame, dataset, metadata):
    """Ties the whole explanation path together: real model, real SHAP values,
    real feature row, out to advisor-facing sentences."""
    shap_values, _ = compute_shap_values(fitted_model, scoring_frame)
    factors = top_factors_for_student(
        shap_row=shap_values[0],
        feature_row=scoring_frame.iloc[0],
        all_feature_columns=list(scoring_frame.columns),
        base_feature_names=dataset["feature_columns"],
        reference=metadata["reference_medians"],
    )
    assert factors
    assert all(isinstance(f, str) and " " in f for f in factors)


# ── save_global_importance_plot ───────────────────────────────────────────

def test_global_importance_plot_is_written(fitted_model, scoring_frame, tmp_path):
    out = tmp_path / "shap.png"
    shap_values, _ = compute_shap_values(fitted_model, scoring_frame)
    save_global_importance_plot(shap_values, scoring_frame, str(out))
    assert out.exists() and out.stat().st_size > 1000


# ── run_checkpoint_evaluation ─────────────────────────────────────────────

@pytest.mark.parametrize("fraction", [0.2, 0.5])
def test_checkpoint_evaluation_returns_valid_metrics(raw, config, fraction):
    result = run_checkpoint_evaluation(raw, config, fraction)
    assert result["checkpoint_fraction"] == fraction
    for key in ["recall", "precision", "f2", "roc_auc"]:
        assert 0.0 <= result[key] <= 1.0
    assert result["n"] > 0


def test_the_model_beats_chance_even_at_the_earliest_checkpoint(raw, config):
    """This is the finding the whole project rests on: useful signal exists
    early enough for intervention to be possible."""
    result = run_checkpoint_evaluation(raw, config, 0.2)
    assert result["roc_auc"] > 0.7, f"no early signal: ROC-AUC {result['roc_auc']}"
    assert result["recall"] > 0.5


def test_later_checkpoints_do_not_perform_worse(raw, config):
    """More information should not hurt. If a later checkpoint scored clearly
    worse, something in the cutoff logic would be wrong."""
    early = run_checkpoint_evaluation(raw, config, 0.2)
    late = run_checkpoint_evaluation(raw, config, 0.5)
    assert late["roc_auc"] >= early["roc_auc"] - 0.05


# ── API internal error handling ───────────────────────────────────────────

def _payload(**overrides):
    base = {
        "student_id": "ERR-1", "gender": "M", "region": "Wales",
        "highest_education": "A Level or Equivalent", "imd_band": "20-30%",
        "age_band": "0-35", "num_of_prev_attempts": 0, "studied_credits": 60,
        "disability": "N", "date_registration": 5, "late_registration": 1,
        "vle_total_clicks": 12, "vle_active_days": 3, "vle_distinct_sites": 2,
        "vle_click_trend": -0.4, "vle_days_since_last_click": 25,
        "n_submitted": 0, "avg_early_score": -1, "pct_on_time": 0, "avg_days_early": 0,
    }
    base.update(overrides)
    return base


@pytest.fixture
def broken_model_client(monkeypatch):
    """An API whose model raises on predict — simulating a corrupted or
    mismatched artifact reaching production."""
    sys.modules.pop("src.api.main", None)
    module = importlib.import_module("src.api.main")

    with TestClient(module.app) as client:
        class Exploding:
            def predict_proba(self, *_args, **_kwargs):
                raise RuntimeError("simulated model failure")

        monkeypatch.setitem(module._state, "model", Exploding())
        yield client
    sys.modules.pop("src.api.main", None)


def test_a_failing_model_returns_422_not_a_stack_trace(broken_model_client):
    r = broken_model_client.post("/api/v1/predict", json=_payload())
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "could not score" in detail.lower()


def test_a_failing_model_in_batch_names_the_student(broken_model_client):
    r = broken_model_client.post("/api/v1/predict/batch",
                                  json={"students": [_payload(student_id="WHO-42")]})
    assert r.status_code == 422
    assert "WHO-42" in r.json()["detail"]


def test_health_still_responds_when_prediction_is_broken(broken_model_client):
    """The platform health check must not flap because scoring has a problem."""
    assert broken_model_client.get("/api/v1/health").status_code == 200
