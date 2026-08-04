"""
Tests for src/models/fairness.py and src/models/encoding.py.

The encoding tests matter more than they look: encode_features (training) and
align_to_model_columns (inference) must produce the same columns in the same
order, or the model silently scores garbage at serving time without raising
anything. That is a failure that would never show up in offline metrics.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.fairness import fairness_report
from src.models.encoding import sanitize_column_name, encode_features, align_to_model_columns


# ── fairness_report ───────────────────────────────────────────────────────

def test_fairness_report_computes_per_group_metrics():
    demographics = pd.DataFrame({
        "gender": ["M"] * 20 + ["F"] * 20,
        "region": ["Wales"] * 40,
    })
    y_true = pd.Series([1] * 10 + [0] * 10 + [1] * 10 + [0] * 10)
    # perfect on men, misses every at-risk woman
    y_pred = pd.Series([1] * 10 + [0] * 10 + [0] * 20)

    report = fairness_report(y_true, y_pred, demographics, ["gender"])
    by_group = report.set_index("group")

    assert by_group.loc["M", "recall"] == 1.0
    assert by_group.loc["F", "recall"] == 0.0
    assert by_group.loc["M", "n"] == 20


def test_fairness_report_surfaces_a_disparity_that_overall_metrics_hide():
    """The entire point of the audit: an aggregate recall of 0.5 can mean
    'mediocre for everyone' or 'perfect for one group, useless for another'.
    Only the breakdown distinguishes them."""
    demographics = pd.DataFrame({"gender": ["M"] * 20 + ["F"] * 20})
    y_true = pd.Series([1] * 10 + [0] * 10 + [1] * 10 + [0] * 10)
    y_pred = pd.Series([1] * 10 + [0] * 10 + [0] * 20)

    from sklearn.metrics import recall_score
    overall = recall_score(y_true, y_pred)
    report = fairness_report(y_true, y_pred, demographics, ["gender"])

    assert overall == pytest.approx(0.5)
    gap = report["recall"].max() - report["recall"].min()
    assert gap == 1.0, "a total disparity must be visible in the report"


def test_small_groups_are_excluded_to_avoid_noise():
    """A 'recall' computed on 3 students is not information, it's noise that
    would invite over-reading."""
    demographics = pd.DataFrame({"region": ["Big"] * 30 + ["Tiny"] * 3})
    y_true = pd.Series([1, 0] * 15 + [1, 0, 1])
    y_pred = pd.Series([1, 0] * 15 + [0, 0, 0])

    report = fairness_report(y_true, y_pred, demographics, ["region"])
    assert "Tiny" not in report["group"].values
    assert "Big" in report["group"].values


def test_fairness_report_covers_every_configured_attribute(config):
    n = 60
    demographics = pd.DataFrame({
        "gender": np.resize(["M", "F"], n),
        "region": np.resize(["Wales", "Scotland"], n),
        "age_band": np.resize(["0-35", "35-55"], n),
        "disability": np.resize(["Y", "N"], n),
    })
    y_true = pd.Series(np.resize([1, 0], n))
    y_pred = pd.Series(np.resize([1, 0, 0, 1], n))

    report = fairness_report(y_true, y_pred, demographics, config["fairness"]["sensitive_columns"])
    covered = set(report["sensitive_attribute"])
    assert {"gender", "region", "age_band", "disability"} <= covered


def test_missing_sensitive_column_is_skipped_not_fatal():
    demographics = pd.DataFrame({"gender": ["M"] * 20 + ["F"] * 20})
    y_true = pd.Series([1, 0] * 20)
    y_pred = pd.Series([1, 0] * 20)
    report = fairness_report(y_true, y_pred, demographics, ["gender", "does_not_exist"])
    assert set(report["sensitive_attribute"]) == {"gender"}


def test_fairness_metrics_stay_within_bounds_on_real_data(dataset, encoded_dataset, config):
    X, y = encoded_dataset["X_enc"], encoded_dataset["y"]
    rng = np.random.default_rng(1)
    y_pred = pd.Series(rng.integers(0, 2, len(y)), index=y.index)
    report = fairness_report(y, y_pred, dataset["X"], config["fairness"]["sensitive_columns"])
    assert not report.empty
    assert report["recall"].between(0, 1).all()
    assert report["precision"].between(0, 1).all()
    assert (report["n"] > 0).all()


# ── sanitize_column_name ──────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("age_band_55<=", "age_band_55lt="),
    ("feature>10", "featuregt10"),
    ("col[0]", "col(0)"),
    ("plain_name", "plain_name"),
    ("imd_band_20-30%", "imd_band_20-30%"),
])
def test_sanitize_removes_characters_xgboost_rejects(raw, expected):
    assert sanitize_column_name(raw) == expected


def test_sanitize_is_idempotent():
    once = sanitize_column_name("age_band_55<=")
    assert sanitize_column_name(once) == once


def test_no_encoded_column_contains_a_forbidden_character(encoded_dataset):
    """XGBoost raises on '[', ']', '<' in feature names. OULAD's '55<=' age band
    triggers exactly this, so it's checked against the real encoded data."""
    for col in encoded_dataset["columns"]:
        assert "<" not in col and "[" not in col and "]" not in col, col


# ── encode_features / align_to_model_columns ──────────────────────────────

def test_encoding_expands_categoricals_and_keeps_numerics(dataset):
    X, feature_columns = dataset["X"], dataset["feature_columns"]
    encoded, cols = encode_features(X, feature_columns)
    assert len(cols) > len(feature_columns), "one-hot encoding should widen the frame"
    assert "vle_total_clicks" in cols
    assert "gender" not in cols, "raw categorical should have been expanded"
    assert any(c.startswith("gender_") for c in cols)
    assert len(encoded) == len(X)


def test_encoded_frame_is_entirely_numeric(encoded_dataset):
    """A stray object column reaches the model as an error at fit time, or worse,
    silently as something meaningless."""
    non_numeric = encoded_dataset["X_enc"].select_dtypes(include=["object"]).columns
    assert list(non_numeric) == []


def test_inference_alignment_matches_training_columns_exactly(metadata):
    """The contract that keeps serving honest: a single student row must encode
    to exactly the training column set, in exactly the training order."""
    model_columns = metadata["model_feature_columns"]
    row = pd.DataFrame([{
        "gender": "M", "region": "Wales", "highest_education": "A Level or Equivalent",
        "imd_band": "20-30%", "age_band": "0-35", "disability": "N",
        "num_of_prev_attempts": 0, "studied_credits": 60, "date_registration": 5,
        "late_registration": 1, "vle_total_clicks": 12, "vle_active_days": 3,
        "vle_distinct_sites": 2, "vle_click_trend": -0.4, "vle_days_since_last_click": 25,
        "n_submitted": 0, "avg_early_score": -1, "pct_on_time": 0, "avg_days_early": 0,
    }])
    aligned = align_to_model_columns(row, model_columns)
    assert list(aligned.columns) == model_columns
    assert len(aligned) == 1
    assert aligned.isna().sum().sum() == 0


def test_unseen_category_produces_zeros_not_a_crash(metadata):
    """A new region appearing in production must not take the API down; it
    should simply contribute nothing to that feature group."""
    model_columns = metadata["model_feature_columns"]
    row = pd.DataFrame([{
        "gender": "M", "region": "A Region That Did Not Exist In Training",
        "highest_education": "A Level or Equivalent", "imd_band": "20-30%",
        "age_band": "0-35", "disability": "N", "num_of_prev_attempts": 0,
        "studied_credits": 60, "date_registration": 5, "late_registration": 1,
        "vle_total_clicks": 12, "vle_active_days": 3, "vle_distinct_sites": 2,
        "vle_click_trend": -0.4, "vle_days_since_last_click": 25, "n_submitted": 0,
        "avg_early_score": -1, "pct_on_time": 0, "avg_days_early": 0,
    }])
    aligned = align_to_model_columns(row, model_columns)
    assert list(aligned.columns) == model_columns
    region_cols = [c for c in model_columns if c.startswith("region_")]
    assert aligned[region_cols].sum(axis=1).iloc[0] == 0


def test_alignment_is_order_independent_in_the_input(metadata):
    """Callers shouldn't have to know the column order; only the output order
    is contractual."""
    model_columns = metadata["model_feature_columns"]
    base = {
        "gender": "F", "region": "Wales", "highest_education": "HE Qualification",
        "imd_band": "70-80%", "age_band": "0-35", "disability": "N",
        "num_of_prev_attempts": 0, "studied_credits": 120, "date_registration": -20,
        "late_registration": 0, "vle_total_clicks": 500, "vle_active_days": 50,
        "vle_distinct_sites": 20, "vle_click_trend": 0.1, "vle_days_since_last_click": 1,
        "n_submitted": 2, "avg_early_score": 80, "pct_on_time": 1.0, "avg_days_early": 3,
    }
    forward = align_to_model_columns(pd.DataFrame([base]), model_columns)
    shuffled = align_to_model_columns(pd.DataFrame([dict(reversed(list(base.items())))]), model_columns)
    pd.testing.assert_frame_equal(forward, shuffled)


def test_batch_alignment_preserves_row_order(metadata):
    model_columns = metadata["model_feature_columns"]
    def make(clicks):
        return {
            "gender": "M", "region": "Wales", "highest_education": "A Level or Equivalent",
            "imd_band": "20-30%", "age_band": "0-35", "disability": "N",
            "num_of_prev_attempts": 0, "studied_credits": 60, "date_registration": 0,
            "late_registration": 0, "vle_total_clicks": clicks, "vle_active_days": 5,
            "vle_distinct_sites": 3, "vle_click_trend": 0.0, "vle_days_since_last_click": 2,
            "n_submitted": 1, "avg_early_score": 60, "pct_on_time": 1.0, "avg_days_early": 1,
        }
    rows = pd.DataFrame([make(10), make(500), make(250)])
    aligned = align_to_model_columns(rows, model_columns)
    assert list(aligned["vle_total_clicks"]) == [10, 500, 250]
