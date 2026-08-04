"""
Tests for src/models/explain.py.

This module gets the most scrutiny in the suite, for a reason: it is the only
part of the system whose output an advisor reads as prose and takes at face
value. A wrong number in a report is embarrassing; a confidently wrong sentence
about a student ("engagement is picking up" when it is falling) actively
misleads the person deciding whether to intervene, and quietly destroys trust
in every other factor on the page.

An earlier version had exactly that bug — wording was taken from the sign of the
SHAP value, which measures how far a feature moved the prediction from the
model's baseline, not whether the underlying behaviour is good or bad. The tests
below pin down the corrected contract from both directions.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.explain import (
    _phrase, _base_feature_name, top_factors_for_student,
    behavioural_reference_medians, BEHAVIORAL_FEATURES,
)

REFERENCE = {
    "vle_total_clicks": 300.0,
    "vle_active_days": 45.0,
    "vle_distinct_sites": 24.0,
    "vle_click_trend": 0.03,
    "vle_days_since_last_click": 0.0,
    "n_submitted": 0.0,
    "avg_early_score": -1.0,
    "pct_on_time": 0.0,
    "avg_days_early": 0.0,
}

RISKY = "risky"
SAFE = "safe"


def classify(phrase: str) -> str:
    """Which direction a phrase reads as, from the advisor's point of view."""
    risky_markers = [
        "low engagement", "few active", "very few", "dropping off", "not logged in",
        "missed", "below average", "late submissions", "cutting it close",
        "later than most", "no assessment submitted",
    ]
    safe_markers = [
        "strong engagement", "consistent activity", "wide range", "holding up",
        "recently active", "submitted the early", "above average",
        "on-time submissions", "comfortably ahead", "on time",
    ]
    low = phrase.lower()
    if any(m in low for m in risky_markers):
        return RISKY
    if any(m in low for m in safe_markers):
        return SAFE
    raise AssertionError(f"Phrase not classifiable, template may have drifted: {phrase!r}")


# ── _phrase: value-driven wording, every branch ───────────────────────────

@pytest.mark.parametrize("feature,value,expected", [
    # features with a plain, non-relative meaning
    ("n_submitted", 0, RISKY),
    ("n_submitted", 1, SAFE),
    ("n_submitted", 3, SAFE),
    ("vle_click_trend", -0.5, RISKY),
    ("vle_click_trend", -0.01, RISKY),
    ("vle_click_trend", 0.0, SAFE),
    ("vle_click_trend", 0.4, SAFE),
    ("vle_days_since_last_click", 25, RISKY),
    ("vle_days_since_last_click", 11, RISKY),
    ("vle_days_since_last_click", 10, SAFE),
    ("vle_days_since_last_click", 0, SAFE),
    ("pct_on_time", 0.0, RISKY),
    ("pct_on_time", 0.79, RISKY),
    ("pct_on_time", 0.8, SAFE),
    ("pct_on_time", 1.0, SAFE),
    ("late_registration", 1, RISKY),
    ("late_registration", 0, SAFE),
    ("avg_days_early", 0, RISKY),
    ("avg_days_early", 5, SAFE),
    ("avg_early_score", 30, RISKY),
    ("avg_early_score", 49, RISKY),
    ("avg_early_score", 50, SAFE),
    ("avg_early_score", 88, SAFE),
    # cohort-relative features, compared against the reference medians
    ("vle_total_clicks", 10, RISKY),
    ("vle_total_clicks", 900, SAFE),
    ("vle_active_days", 3, RISKY),
    ("vle_active_days", 60, SAFE),
    ("vle_distinct_sites", 2, RISKY),
    ("vle_distinct_sites", 30, SAFE),
])
def test_phrase_direction_follows_the_value(feature, value, expected):
    phrase = _phrase(feature, value, REFERENCE)
    assert phrase is not None
    assert classify(phrase) == expected, f"{feature}={value} produced {phrase!r}"


def test_no_submission_sentinel_is_never_reported_as_a_low_score():
    """avg_early_score uses -1 to mean 'nothing submitted yet'. Reporting that
    as a bad mark would be factually false about the student."""
    phrase = _phrase("avg_early_score", -1, REFERENCE)
    assert "below average" not in phrase.lower()
    assert "no assessment submitted yet" in phrase.lower()


def test_phrase_declines_to_guess_without_a_reference():
    """A cohort-relative feature has no meaning without a cohort to compare to.
    Returning None (and being dropped) is correct; inventing a direction is not."""
    assert _phrase("vle_total_clicks", 50, None) is None
    assert _phrase("vle_active_days", 10, {}) is None
    # ...but features with absolute meaning still work with no reference
    assert _phrase("n_submitted", 0, None) is not None
    assert _phrase("vle_click_trend", -0.3, None) is not None


def test_every_behavioural_feature_has_a_template():
    """Guards against a new feature silently rendering as its raw column name
    (e.g. 'vle_click_trend') in front of an advisor."""
    for feature in BEHAVIORAL_FEATURES:
        phrase = _phrase(feature, 1.0, REFERENCE)
        if phrase is None:
            continue
        assert phrase != feature, f"{feature} has no human-readable template"
        assert " " in phrase, f"{feature} rendered as a bare token: {phrase!r}"


# ── _base_feature_name ────────────────────────────────────────────────────

@pytest.mark.parametrize("column,expected", [
    ("vle_total_clicks", "vle_total_clicks"),
    ("gender_M", "gender"),
    ("region_London Region", "region"),
    ("age_band_55lt=", "age_band"),
    ("imd_band_20-30%", "imd_band"),
    ("unknown_column", "unknown_column"),
])
def test_dummy_columns_map_back_to_their_base_feature(column, expected):
    base_names = ["vle_total_clicks", "gender", "region", "age_band", "imd_band"]
    assert _base_feature_name(column, base_names) == expected


# ── top_factors_for_student ───────────────────────────────────────────────

def _shap_and_row(values: dict, columns: list[str], shap_map: dict):
    row = pd.Series({c: values.get(c, 0.0) for c in columns})
    shap_row = np.array([shap_map.get(c, 0.0) for c in columns])
    return shap_row, row


COLUMNS = [
    "vle_total_clicks", "vle_active_days", "vle_distinct_sites", "vle_click_trend",
    "vle_days_since_last_click", "n_submitted", "avg_early_score", "pct_on_time",
    "avg_days_early", "late_registration", "gender_M", "region_Wales",
]
BASE_NAMES = [
    "vle_total_clicks", "vle_active_days", "vle_distinct_sites", "vle_click_trend",
    "vle_days_since_last_click", "n_submitted", "avg_early_score", "pct_on_time",
    "avg_days_early", "late_registration", "gender", "region",
]


def test_demographics_are_never_surfaced_as_a_reason():
    """The model may use demographics; the explanation must not name them.
    Telling an advisor a student's region raised their risk invites profiling
    in the human conversation that follows, and isn't actionable."""
    import re
    shap_row, row = _shap_and_row(
        {"gender_M": 1, "region_Wales": 1, "vle_total_clicks": 10},
        COLUMNS,
        # give the demographic dummies the LARGEST contributions on purpose
        {"gender_M": 5.0, "region_Wales": 4.5, "vle_total_clicks": 1.0},
    )
    factors = top_factors_for_student(shap_row, row, COLUMNS, BASE_NAMES, reference=REFERENCE)
    joined = " ".join(factors).lower()
    # whole words only — "age" must not match inside "engagement"
    for term in ["gender", "male", "female", "region", "wales", "disability", "age", "deprivation"]:
        assert not re.search(rf"\b{term}\b", joined), f"demographic term '{term}' leaked into: {factors}"


def test_factors_are_ordered_by_shap_magnitude():
    shap_row, row = _shap_and_row(
        {"vle_total_clicks": 10, "n_submitted": 0, "pct_on_time": 0.0},
        COLUMNS,
        {"vle_total_clicks": 0.2, "n_submitted": 3.0, "pct_on_time": 1.0},
    )
    factors = top_factors_for_student(shap_row, row, COLUMNS, BASE_NAMES, reference=REFERENCE)
    assert "missed" in factors[0].lower(), f"largest contributor should lead: {factors}"


def test_negative_shap_values_still_rank_by_magnitude():
    """Sign determines nothing here; a strongly protective factor is as worth
    showing as a strongly aggravating one."""
    shap_row, row = _shap_and_row(
        {"vle_total_clicks": 900, "n_submitted": 0},
        COLUMNS,
        {"vle_total_clicks": -4.0, "n_submitted": 0.5},
    )
    factors = top_factors_for_student(shap_row, row, COLUMNS, BASE_NAMES, reference=REFERENCE)
    assert "strong engagement" in factors[0].lower()


def test_top_n_is_respected():
    shap_map = {c: float(i + 1) for i, c in enumerate(COLUMNS)}
    shap_row, row = _shap_and_row({c: 1.0 for c in COLUMNS}, COLUMNS, shap_map)
    for n in [1, 2, 3, 5]:
        factors = top_factors_for_student(shap_row, row, COLUMNS, BASE_NAMES, top_n=n, reference=REFERENCE)
        assert len(factors) <= n


def test_no_duplicate_phrases():
    shap_map = {c: float(i + 1) for i, c in enumerate(COLUMNS)}
    shap_row, row = _shap_and_row({c: 1.0 for c in COLUMNS}, COLUMNS, shap_map)
    factors = top_factors_for_student(shap_row, row, COLUMNS, BASE_NAMES, top_n=6, reference=REFERENCE)
    assert len(factors) == len(set(factors))


def test_nan_feature_values_are_skipped_not_rendered():
    shap_row, row = _shap_and_row({"n_submitted": 0}, COLUMNS, {"vle_total_clicks": 9.0, "n_submitted": 1.0})
    row["vle_total_clicks"] = np.nan
    factors = top_factors_for_student(shap_row, row, COLUMNS, BASE_NAMES, reference=REFERENCE)
    joined = " ".join(factors).lower()
    assert "nan" not in joined
    assert factors, "should still return the factors it can support"


# ── the regression that started all this ──────────────────────────────────

def test_a_declining_student_is_never_told_engagement_is_improving():
    """The exact bug. SHAP sign and value direction disagreed, and the wording
    followed the wrong one."""
    shap_row, row = _shap_and_row(
        {"vle_click_trend": -0.35, "vle_days_since_last_click": 22, "n_submitted": 0},
        COLUMNS,
        # a NEGATIVE shap value on a feature whose value is clearly bad
        {"vle_click_trend": -2.0, "vle_days_since_last_click": 1.5, "n_submitted": 1.0},
    )
    factors = top_factors_for_student(shap_row, row, COLUMNS, BASE_NAMES, reference=REFERENCE)
    joined = " ".join(factors).lower()
    assert "picking up" not in joined
    assert "holding up" not in joined
    assert "dropping off" in joined, f"declining trend must read as declining: {factors}"


def test_an_improving_student_is_never_told_engagement_is_falling():
    """The mirror image, so the fix can't have simply inverted the bug."""
    shap_row, row = _shap_and_row(
        {"vle_click_trend": 0.4, "vle_days_since_last_click": 1, "n_submitted": 2,
         "vle_total_clicks": 800, "pct_on_time": 1.0, "avg_early_score": 85},
        COLUMNS,
        {"vle_click_trend": 2.0, "vle_days_since_last_click": 1.0, "n_submitted": 1.5},
    )
    factors = top_factors_for_student(shap_row, row, COLUMNS, BASE_NAMES, reference=REFERENCE)
    joined = " ".join(factors).lower()
    assert "dropping off" not in joined
    assert "has not logged in" not in joined


@pytest.mark.parametrize("shap_sign", [1.0, -1.0])
def test_wording_is_independent_of_the_shap_sign(shap_sign):
    """The core contract, stated directly: flipping every SHAP sign changes
    which factors rank highest, but must never change what a given value says."""
    values = {"vle_click_trend": -0.3, "n_submitted": 0, "pct_on_time": 0.0}
    shap_row, row = _shap_and_row(
        values, COLUMNS,
        {"vle_click_trend": shap_sign * 2.0, "n_submitted": shap_sign * 1.0,
         "pct_on_time": shap_sign * 0.5},
    )
    factors = top_factors_for_student(shap_row, row, COLUMNS, BASE_NAMES, top_n=3, reference=REFERENCE)
    for phrase in factors:
        assert classify(phrase) == RISKY, f"bad values must read as concerning: {phrase!r}"


# ── behavioural_reference_medians ─────────────────────────────────────────

def test_reference_medians_cover_the_behavioural_features(encoded_dataset, dataset):
    ref = behavioural_reference_medians(encoded_dataset["X_enc"], dataset["feature_columns"])
    assert ref, "no medians computed"
    for key in ref:
        assert key in BEHAVIORAL_FEATURES
    assert "vle_total_clicks" in ref
    assert all(isinstance(v, float) for v in ref.values())


def test_reference_medians_exclude_demographics(encoded_dataset, dataset):
    ref = behavioural_reference_medians(encoded_dataset["X_enc"], dataset["feature_columns"])
    for key in ref:
        assert not key.startswith(("gender", "region", "imd", "age", "disability", "highest"))


def test_shipped_metadata_contains_usable_medians(metadata):
    """The API depends on these being in the artifact — without them every
    cohort-relative factor is silently dropped from explanations."""
    ref = metadata.get("reference_medians")
    assert ref, "reference_medians missing from shipped metadata"
    assert ref["vle_total_clicks"] > 0
    for feature in ["vle_total_clicks", "vle_active_days", "vle_distinct_sites"]:
        assert feature in ref
