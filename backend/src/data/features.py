"""
Feature engineering for the early-warning model.

IMPORTANT: every function here takes already-cutoff-filtered data (see
cutoff.py) as input. This module never touches raw, unfiltered tables, and
never touches `final_result` or `date_unregistration` — those are exactly
the columns that would leak the outcome we're trying to predict.
"""
import numpy as np
import pandas as pd
from src.data.cutoff import Checkpoint

# Columns that must NEVER be used as model features (leakage or identifiers).
# Kept here explicitly so the leakage test can assert against a single source
# of truth instead of scattered magic strings.
FORBIDDEN_FEATURE_COLUMNS = {
    "final_result", "date_unregistration", "score", "date_submitted",
    "id_student", "id_assessment", "id_site",
}

# Categorical columns that need one-hot encoding before hitting a model.
CATEGORICAL_COLUMNS = ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]

# Simple numeric aggregate features used by the Logistic Regression baseline
# only (per the brief: "Logistic Regression with basic aggregate features").
BASELINE_NUMERIC_FEATURES = ["n_submitted", "avg_early_score", "pct_on_time", "vle_total_clicks"]


def _cutoff_lookup(checkpoints: list[Checkpoint]) -> dict:
    return {(c.code_module, c.code_presentation): c.cutoff_day for c in checkpoints}


def build_vle_features(vle_filtered: pd.DataFrame, checkpoints: list[Checkpoint]) -> pd.DataFrame:
    """Cumulative-clicks, active-day, and trend features from VLE interactions
    up to the checkpoint."""
    if vle_filtered.empty:
        return pd.DataFrame(columns=["id_student", "vle_total_clicks", "vle_active_days",
                                      "vle_distinct_sites", "vle_click_trend", "vle_days_since_last_click"])

    cutoffs = _cutoff_lookup(checkpoints)

    def per_student(g: pd.DataFrame) -> pd.Series:
        cutoff_day = cutoffs.get((g["code_module"].iloc[0], g["code_presentation"].iloc[0]), 0)
        total_clicks = g["sum_click"].sum()
        active_days = g["date"].nunique()
        distinct_sites = g["id_site"].nunique()

        # trend: compare clicks in the first half of the observed window vs second half
        mid = g["date"].min() + (g["date"].max() - g["date"].min()) / 2 if len(g) else 0
        first_half = g[g["date"] <= mid]["sum_click"].sum()
        second_half = g[g["date"] > mid]["sum_click"].sum()
        trend = (second_half - first_half) / max(1, total_clicks)

        last_click_day = g["date"].max()
        days_since_last_click = cutoff_day - last_click_day

        return pd.Series({
            "vle_total_clicks": total_clicks,
            "vle_active_days": active_days,
            "vle_distinct_sites": distinct_sites,
            "vle_click_trend": trend,
            "vle_days_since_last_click": days_since_last_click,
        })

    feats = vle_filtered.groupby("id_student").apply(per_student, include_groups=False)
    feats = feats.reset_index()
    return feats


def build_assessment_features(sa_filtered: pd.DataFrame, assessments: pd.DataFrame,
                               eligible_assessment_ids: set) -> pd.DataFrame:
    """Submission-rate, timeliness, and early-score features, computed only
    over assessments that were actually due before the checkpoint."""
    eligible = assessments[assessments["id_assessment"].isin(eligible_assessment_ids)]
    n_eligible_by_course = eligible.groupby(["code_module", "code_presentation"])["id_assessment"].nunique()

    due_date_by_id = eligible.set_index("id_assessment")["date"].to_dict()

    if sa_filtered.empty:
        return pd.DataFrame(columns=["id_student", "n_submitted", "submission_rate",
                                      "avg_early_score", "pct_on_time", "avg_days_early"])

    sa = sa_filtered.copy()
    sa["due_date"] = sa["id_assessment"].map(due_date_by_id)
    sa["days_early"] = sa["due_date"] - sa["date_submitted"]
    sa["on_time"] = sa["days_early"] >= 0

    def per_student(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n_submitted": len(g),
            "avg_early_score": g["score"].mean(),
            "pct_on_time": g["on_time"].mean(),
            "avg_days_early": g["days_early"].mean(),
        })

    feats = sa.groupby("id_student").apply(per_student, include_groups=False).reset_index()
    return feats


def build_enrollment_features(student_info: pd.DataFrame, registration: pd.DataFrame) -> pd.DataFrame:
    """Features known at enrollment time — always safe regardless of checkpoint,
    since they're fixed before the course even starts."""
    df = student_info.merge(
        registration[["id_student", "code_module", "code_presentation", "date_registration"]],
        on=["id_student", "code_module", "code_presentation"], how="left"
    )
    df["late_registration"] = (df["date_registration"] > 0).astype(int)
    keep = ["id_student", "code_module", "code_presentation", "gender", "region",
            "highest_education", "imd_band", "age_band", "num_of_prev_attempts",
            "studied_credits", "disability", "date_registration", "late_registration"]
    return df[keep]


def assemble_feature_table(
    student_info: pd.DataFrame,
    registration: pd.DataFrame,
    vle_filtered: pd.DataFrame,
    sa_filtered: pd.DataFrame,
    assessments: pd.DataFrame,
    eligible_assessment_ids: set,
    checkpoints: list[Checkpoint],
    n_eligible_assessments_by_course: pd.Series | None = None,
) -> pd.DataFrame:
    """Joins all feature groups into one row-per-student table. This is the
    ONLY function that should be called from outside this module."""
    enrollment = build_enrollment_features(student_info, registration)
    vle_feats = build_vle_features(vle_filtered, checkpoints)
    assess_feats = build_assessment_features(sa_filtered, assessments, eligible_assessment_ids)

    out = enrollment.merge(vle_feats, on="id_student", how="left")
    out = out.merge(assess_feats, on="id_student", how="left")

    # students with zero early activity are a real (and important!) signal, not missing data
    numeric_fill_zero = ["vle_total_clicks", "vle_active_days", "vle_distinct_sites", "vle_click_trend",
                          "n_submitted", "pct_on_time"]
    for col in numeric_fill_zero:
        if col in out.columns:
            out[col] = out[col].fillna(0)
    # a student with zero submissions has no early score / lateness signal —
    # fill with clearly-out-of-range sentinels handled explicitly, not silently as 0
    out["avg_early_score"] = out["avg_early_score"].fillna(-1)   # -1 = "no submission yet"
    out["avg_days_early"] = out["avg_days_early"].fillna(0)
    out["vle_days_since_last_click"] = out["vle_days_since_last_click"].fillna(9999)  # "never clicked"

    return out


def label_table(student_info: pd.DataFrame, at_risk_outcomes: list[str]) -> pd.DataFrame:
    """Separate function, deliberately: the label table is the ONLY place
    `final_result` is allowed to be read. It must never be merged into the
    feature table used for inference-time prediction."""
    df = student_info[["id_student", "code_module", "code_presentation", "final_result"]].copy()
    df["at_risk"] = df["final_result"].isin(at_risk_outcomes).astype(int)
    return df
