"""Unit tests for feature engineering behaviour that the model depends on."""
import pandas as pd
import pytest

from src.data.cutoff import resolve_checkpoints, apply_checkpoint_cutoff
from src.data.features import (
    build_vle_features, build_assessment_features, build_enrollment_features,
    label_table, CATEGORICAL_COLUMNS,
)
from src.data.pipeline import build_dataset


def test_resolve_checkpoints_scales_with_module_length(raw):
    checkpoints = resolve_checkpoints(raw["courses"], 0.50)
    lengths = raw["courses"].set_index(["code_module", "code_presentation"])["module_presentation_length"]
    for c in checkpoints:
        expected = int(round(lengths.loc[(c.code_module, c.code_presentation)] * 0.50))
        assert c.cutoff_day == expected


def test_vle_features_are_non_negative_and_complete(raw, config):
    checkpoints = resolve_checkpoints(raw["courses"], config["prediction"]["checkpoint_fraction"])
    filtered = apply_checkpoint_cutoff(
        raw["student_vle"], raw["student_assessment"], raw["assessments"], checkpoints
    )
    feats = build_vle_features(filtered["student_vle"], checkpoints)
    assert (feats["vle_total_clicks"] >= 0).all()
    assert (feats["vle_active_days"] >= 0).all()
    assert feats["id_student"].is_unique


def test_students_with_no_early_activity_get_sentinel_values(raw, config):
    """A student with zero submissions must survive into the feature table with
    an explicit 'no submission yet' sentinel, not be dropped or silently zeroed."""
    X, y, feature_columns, meta = build_dataset(raw, config)
    no_submission = X[X["n_submitted"] == 0]
    assert len(no_submission) > 0, "test dataset should contain some non-submitters"
    assert (no_submission["avg_early_score"] == -1).all()


def test_label_is_binary_and_matches_configured_outcomes(raw, config):
    labels = label_table(raw["student_info"], config["target"]["at_risk_outcomes"])
    assert set(labels["at_risk"].unique()).issubset({0, 1})
    at_risk_rows = labels[labels["at_risk"] == 1]
    assert set(at_risk_rows["final_result"].unique()).issubset(set(config["target"]["at_risk_outcomes"]))


def test_earlier_checkpoint_yields_less_or_equal_activity(raw, config):
    """The same student cannot have more recorded clicks at 20% than at 50%."""
    X_early, _, _, _ = build_dataset(raw, config, checkpoint_fraction=0.20)
    X_late, _, _, _ = build_dataset(raw, config, checkpoint_fraction=0.50)
    merged = X_early[["id_student", "vle_total_clicks"]].merge(
        X_late[["id_student", "vle_total_clicks"]], on="id_student", suffixes=("_early", "_late")
    )
    assert (merged["vle_total_clicks_early"] <= merged["vle_total_clicks_late"]).all()


def test_categorical_columns_present_in_feature_set(raw, config):
    X, y, feature_columns, meta = build_dataset(raw, config)
    for col in CATEGORICAL_COLUMNS:
        assert col in feature_columns
