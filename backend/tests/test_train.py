"""
Tests for src/models/train.py.

train.main() runs the whole pipeline and is exercised end-to-end by `make
train`; what's tested here are the pure functions it's built from, where a
silent bug would corrupt every number in the report without anything crashing.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.train import (
    metrics_from_predictions, evaluate_trivial_baseline, risk_band,
    make_xgb_model, make_rf_model, cross_validate_model, HELD_OUT_PRESENTATION,
)


# ── metrics_from_predictions ──────────────────────────────────────────────

def test_metrics_on_a_perfect_classifier():
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.01, 0.02, 0.98, 0.99])
    m = metrics_from_predictions(y_true, (y_proba >= 0.5).astype(int), y_proba)
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0
    assert m["f2"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["brier"] < 0.01


def test_metrics_on_a_classifier_that_misses_everyone():
    """The failure mode that matters most here: predicting 'nobody is at risk'
    must show recall 0, not be masked by high accuracy."""
    y_true = pd.Series([0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0])
    y_proba = np.array([0.1, 0.1, 0.1, 0.4])
    m = metrics_from_predictions(y_true, y_pred, y_proba)
    assert m["recall"] == 0.0
    assert m["precision"] == 0.0   # zero_division=0, not a crash
    assert m["f2"] == 0.0


def test_f2_weights_recall_above_precision():
    """F2 is the primary metric precisely because missing a struggling student
    is worse than a needless check-in. A high-recall/low-precision model must
    score better than the mirror image, or the metric isn't doing its job."""
    y_true = pd.Series([1, 1, 1, 1, 0, 0, 0, 0])

    high_recall = np.array([1, 1, 1, 1, 1, 1, 0, 0])   # 4/4 caught, 2 false alarms
    high_precision = np.array([1, 1, 0, 0, 0, 0, 0, 0])  # 2/4 caught, 0 false alarms

    m_recall = metrics_from_predictions(y_true, high_recall, high_recall.astype(float))
    m_precision = metrics_from_predictions(y_true, high_precision, high_precision.astype(float))

    assert m_recall["recall"] > m_precision["recall"]
    assert m_recall["precision"] < m_precision["precision"]
    assert m_recall["f2"] > m_precision["f2"], "F2 must favour catching students"


def test_metrics_reports_sample_size_and_base_rate():
    y_true = pd.Series([0, 0, 0, 1])
    y_proba = np.array([0.2, 0.2, 0.2, 0.8])
    m = metrics_from_predictions(y_true, (y_proba >= 0.5).astype(int), y_proba)
    assert m["n"] == 4
    assert m["positive_rate"] == 0.25


def test_all_metrics_are_within_valid_bounds(encoded_dataset):
    y = encoded_dataset["y"]
    rng = np.random.default_rng(0)
    proba = rng.random(len(y))
    m = metrics_from_predictions(y, (proba >= 0.5).astype(int), proba)
    for key in ["recall", "precision", "f2", "roc_auc", "pr_auc", "brier"]:
        assert 0.0 <= m[key] <= 1.0, f"{key} out of bounds: {m[key]}"


# ── evaluate_trivial_baseline ─────────────────────────────────────────────

def test_trivial_baseline_flags_exactly_the_non_submitters():
    X = pd.DataFrame({"n_submitted": [0, 0, 1, 3]})
    y = pd.Series([1, 0, 1, 0])
    m = evaluate_trivial_baseline(X, y)
    # flagged rows 0 and 1; of the two true positives (rows 0 and 2) it catches one
    assert m["recall"] == 0.5
    assert m["precision"] == 0.5
    assert m["n"] == 4


def test_trivial_baseline_is_beaten_by_the_real_model(metadata):
    """The brief requires the model to clearly beat the naive rule. This asserts
    the shipped artifact actually does, rather than trusting the README."""
    trivial = metadata["trivial_baseline_metrics"]
    model = metadata["held_out_metrics"]
    assert model["recall"] > trivial["recall"]
    assert model["precision"] > trivial["precision"]
    assert model["f2"] > trivial["f2"]
    assert model["roc_auc"] > trivial["roc_auc"]


# ── risk_band ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("prob,expected", [
    (0.0, "Low"), (0.20, "Low"), (0.33, "Low"),
    (0.331, "Medium"), (0.50, "Medium"), (0.66, "Medium"),
    (0.661, "High"), (0.90, "High"), (1.0, "High"),
])
def test_risk_band_boundaries(prob, expected, config):
    assert risk_band(prob, config) == expected


def test_risk_bands_are_monotonic(config):
    """A higher probability can never map to a less urgent band."""
    order = {"Low": 0, "Medium": 1, "High": 2}
    previous = -1
    for p in np.linspace(0, 1, 101):
        current = order[risk_band(float(p), config)]
        assert current >= previous
        previous = current


# ── model factories ───────────────────────────────────────────────────────

def test_xgb_model_uses_configured_hyperparameters(config):
    model = make_xgb_model(config, scale_pos_weight=2.0)
    p = config["model"]["xgboost_params"]
    assert model.n_estimators == p["n_estimators"]
    assert model.max_depth == p["max_depth"]
    assert model.learning_rate == p["learning_rate"]
    assert model.scale_pos_weight == 2.0
    assert model.random_state == config["model"]["random_state"]


def test_rf_model_balances_classes(config):
    """Class imbalance must be handled by weighting, not left to chance —
    otherwise the model can score well by ignoring the minority class."""
    model = make_rf_model(config)
    assert model.class_weight == "balanced"
    assert model.n_estimators == config["model"]["random_forest_params"]["n_estimators"]


def test_two_models_built_from_the_same_config_are_identical(config):
    a, b = make_xgb_model(config, 1.5), make_xgb_model(config, 1.5)
    assert a.get_params() == b.get_params()


# ── cross_validate_model ──────────────────────────────────────────────────

def test_cross_validation_returns_averaged_metrics_over_all_rows(encoded_dataset, config):
    X, y = encoded_dataset["X_enc"], encoded_dataset["y"]
    groups = pd.Series(range(len(y)))  # unique group per row
    result = cross_validate_model(
        lambda: make_rf_model(config), X, y, groups, n_folds=3, seed=42
    )
    for key in ["recall", "precision", "f2", "roc_auc"]:
        assert key in result and 0.0 <= result[key] <= 1.0
    # every row appears in exactly one validation fold
    assert result["n"] == len(y)


def test_cross_validation_is_deterministic(encoded_dataset, config):
    X, y = encoded_dataset["X_enc"], encoded_dataset["y"]
    groups = pd.Series(range(len(y)))
    kwargs = dict(X_enc=X, y=y, groups=groups, n_folds=3, seed=42)
    a = cross_validate_model(lambda: make_rf_model(config), **kwargs)
    b = cross_validate_model(lambda: make_rf_model(config), **kwargs)
    assert a == b, "same seed must give the same result, or reported numbers mean nothing"


def test_model_learns_real_signal_not_noise(encoded_dataset, config):
    """Guards against the pipeline silently degrading into a coin flip — which
    would still 'pass' every structural test while being useless."""
    X, y = encoded_dataset["X_enc"], encoded_dataset["y"]
    groups = pd.Series(range(len(y)))
    result = cross_validate_model(lambda: make_rf_model(config), X, y, groups, 3, 42)
    assert result["roc_auc"] > 0.70, f"ROC-AUC collapsed to {result['roc_auc']}"
    assert result["recall"] > 0.50


# ── held-out split ────────────────────────────────────────────────────────

def test_held_out_presentation_exists_and_is_a_minority_of_the_data(dataset):
    """The generalisation test only means something if the held-out cohort is
    genuinely present and genuinely held out."""
    X = dataset["X"]
    held = (X["code_presentation"] == HELD_OUT_PRESENTATION)
    assert held.sum() > 0, f"{HELD_OUT_PRESENTATION} missing from the data"
    assert held.sum() < len(X) / 2, "held-out split should be a minority of rows"


def test_no_student_appears_in_both_train_and_held_out(dataset):
    X = dataset["X"]
    held = X["code_presentation"] == HELD_OUT_PRESENTATION
    overlap = set(X.loc[held, "id_student"]) & set(X.loc[~held, "id_student"])
    assert not overlap, f"{len(overlap)} students leak across the held-out boundary"
