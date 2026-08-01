"""
The single most important test file in this project. It proves — not just
claims — that no feature used by the model could have come from data dated
after the prediction checkpoint, and that no forbidden/outcome-leaking column
ever reaches the feature table.
"""
import pytest
import pandas as pd
from src.data.cutoff import resolve_checkpoints, apply_checkpoint_cutoff
from src.data.pipeline import build_dataset
from src.data.features import FORBIDDEN_FEATURE_COLUMNS

# `config` and `raw` fixtures come from tests/conftest.py (session-scoped)


def test_vle_rows_never_exceed_cutoff_day(raw, config):
    checkpoints = resolve_checkpoints(raw["courses"], config["prediction"]["checkpoint_fraction"])
    result = apply_checkpoint_cutoff(
        raw["student_vle"], raw["student_assessment"], raw["assessments"], checkpoints
    )
    vle = result["student_vle"]
    cutoff_by_course = {(c.code_module, c.code_presentation): c.cutoff_day for c in checkpoints}

    violations = vle[vle.apply(
        lambda r: r["date"] > cutoff_by_course[(r["code_module"], r["code_presentation"])], axis=1
    )]
    assert len(violations) == 0, f"{len(violations)} VLE rows are dated after their course's checkpoint"


def test_assessment_rows_only_include_assessments_due_before_cutoff(raw, config):
    checkpoints = resolve_checkpoints(raw["courses"], config["prediction"]["checkpoint_fraction"])
    result = apply_checkpoint_cutoff(
        raw["student_vle"], raw["student_assessment"], raw["assessments"], checkpoints
    )
    eligible_ids = result["eligible_assessment_ids"]
    sa = result["student_assessment"]

    assert set(sa["id_assessment"]).issubset(eligible_ids)

    # cross-check independently against assessments.csv due dates (not reusing
    # the function's own internal set, to catch bugs in apply_checkpoint_cutoff itself)
    cutoff_by_course = {(c.code_module, c.code_presentation): c.cutoff_day for c in checkpoints}
    due_date = raw["assessments"].set_index("id_assessment")[["date", "code_module", "code_presentation"]]
    for aid in eligible_ids:
        row = due_date.loc[aid]
        cutoff = cutoff_by_course[(row["code_module"], row["code_presentation"])]
        assert row["date"] <= cutoff, f"assessment {aid} due after checkpoint but marked eligible"


def test_smaller_checkpoint_never_admits_more_data_than_larger_checkpoint(raw, config):
    """Sanity check on the cutoff mechanism itself: an earlier checkpoint must
    always yield a subset of what a later checkpoint yields."""
    cps_small = resolve_checkpoints(raw["courses"], 0.20)
    cps_large = resolve_checkpoints(raw["courses"], 0.50)

    result_small = apply_checkpoint_cutoff(raw["student_vle"], raw["student_assessment"], raw["assessments"], cps_small)
    result_large = apply_checkpoint_cutoff(raw["student_vle"], raw["student_assessment"], raw["assessments"], cps_large)

    assert len(result_small["student_vle"]) <= len(result_large["student_vle"])
    assert result_small["eligible_assessment_ids"].issubset(result_large["eligible_assessment_ids"])


def test_feature_table_never_contains_forbidden_columns(raw, config):
    X, y, feature_columns, meta = build_dataset(raw, config)
    leaked = set(feature_columns) & FORBIDDEN_FEATURE_COLUMNS
    assert not leaked, f"Forbidden/leaking columns found in features: {leaked}"
    # final_result and date_unregistration specifically must never appear anywhere in X
    assert "final_result" not in X.columns
    assert "date_unregistration" not in X.columns


def test_feature_table_matches_row_count_of_students(raw, config):
    X, y, feature_columns, meta = build_dataset(raw, config)
    assert len(X) == len(y)
    assert len(X) == raw["student_info"][["id_student", "code_module", "code_presentation"]].drop_duplicates().shape[0]
