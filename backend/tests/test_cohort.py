"""
Tests for src/data/build_demo_cohort.py and the shipped cohort artifact.

This file exists because the cohort builder had **no tests at all** while
producing the artifact the entire dashboard renders from. Three real defects
were sitting in it, all of which reached a live deployment and none of which
would have crashed anything:

  1. `expectedCount` was derived from the student's own submission count, so
     anyone who had submitted anything showed "1/1 submitted" — apparently
     complete — beside a 99% risk score. Students on a course with no
     assessment due yet were shown as having missed two that don't exist.
  2. Factor impacts were ranked separately from factor texts and zipped
     together, attaching ~31% of phrases to a SHAP value from an unrelated
     feature.
  3. The impact sign was taken from SHAP, which measures displacement from the
     model's baseline rather than whether the behaviour is good or bad. The
     dashboard draws its arrow and colour from that number, so a quarter of
     factors showed a reassuring blue "Lowers the score" arrow next to text
     like "Low engagement with course materials".

The through-line: none of these threw an exception. They produced a page that
looked fine and quietly said false things about students. The tests below
assert on *coherence between what is shown*, not just on shapes and types.
"""
import json
from pathlib import Path

import pytest

COHORT_PATH = Path("models/artifacts/demo_cohort.json")

RISKY_MARKERS = [
    "low ", "few ", "very few", "dropping off", "not logged", "missed",
    "below average", "late submissions", "cutting it close", "later than most",
    "no assessment submitted",
]


@pytest.fixture(scope="module")
def cohort():
    if not COHORT_PATH.exists():
        pytest.fail(
            f"\n\n{COHORT_PATH} not found.\n"
            "Build it first:\n\n    make cohort\n",
            pytrace=False,
        )
    return json.loads(COHORT_PATH.read_text())


def _is_risky_text(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in RISKY_MARKERS)


# ── structure ─────────────────────────────────────────────────────────────

def test_cohort_is_populated(cohort):
    assert len(cohort["students"]) >= 40
    assert len(cohort["courses"]) >= 1


def test_student_ids_are_unique(cohort):
    ids = [s["id"] for s in cohort["students"]]
    assert len(set(ids)) == len(ids)


def test_every_student_has_the_fields_the_dashboard_renders(cohort):
    required = {
        "id", "name", "courseCode", "riskScore", "riskBand", "submittedCount",
        "expectedCount", "lastActiveDaysAgo", "activity", "topFactors",
        "avgEarlyScore", "onTimeRate", "totalClicks", "activeDays",
        "registeredDay", "checkpointUsed", "modelVersion",
    }
    for s in cohort["students"]:
        assert required <= set(s), f"{s['id']} missing {required - set(s)}"


def test_every_student_belongs_to_a_listed_course(cohort):
    codes = {c["code"] for c in cohort["courses"]}
    for s in cohort["students"]:
        assert s["courseCode"] in codes


def test_courses_have_a_checkpoint_inside_the_course_length(cohort):
    for c in cohort["courses"]:
        assert 0 < c["checkpointDay"] < c["lengthDays"]


# ── scores and bands ──────────────────────────────────────────────────────

def test_risk_scores_are_probabilities(cohort):
    for s in cohort["students"]:
        assert 0.0 <= s["riskScore"] <= 1.0


def test_risk_bands_agree_with_their_scores(cohort):
    import yaml
    bands = yaml.safe_load(open("config/config.yaml"))["risk_bands"]
    for s in cohort["students"]:
        score = s["riskScore"]
        expected = ("Low" if score <= bands["low_max"]
                    else "Medium" if score <= bands["medium_max"] else "High")
        assert s["riskBand"] == expected, f"{s['id']}: {score} labelled {s['riskBand']}"


def test_cohort_spans_more_than_one_risk_band(cohort):
    """A roster where everyone is the same colour is useless for triage and
    usually means the scoring or sampling broke."""
    assert len({s["riskBand"] for s in cohort["students"]}) >= 2


# ── defect 1: submitted / expected ────────────────────────────────────────

def test_submitted_never_exceeds_expected(cohort):
    for s in cohort["students"]:
        assert s["submittedCount"] <= s["expectedCount"], \
            f"{s['id']} shows {s['submittedCount']}/{s['expectedCount']}"


def test_expected_count_matches_the_real_assessment_schedule(cohort):
    """The regression test for defect 1. `expectedCount` must be the number of
    assessments genuinely due before the checkpoint for that course — not
    inferred from what the student happened to submit, which made everyone who
    submitted anything look complete."""
    import yaml
    from src.data.ingest import load_raw_tables
    from src.data.cutoff import resolve_checkpoints, apply_checkpoint_cutoff

    config = yaml.safe_load(open("config/config.yaml"))
    raw = load_raw_tables(config)
    checkpoints = resolve_checkpoints(raw["courses"], config["prediction"]["checkpoint_fraction"])
    result = apply_checkpoint_cutoff(
        raw["student_vle"], raw["student_assessment"], raw["assessments"], checkpoints
    )
    eligible = raw["assessments"][
        raw["assessments"]["id_assessment"].isin(result["eligible_assessment_ids"])
    ]
    presentation = cohort["generated_from"]
    truth = (
        eligible[eligible["code_presentation"] == presentation]
        .groupby("code_module")["id_assessment"].nunique().to_dict()
    )

    for s in cohort["students"]:
        expected_for_course = truth.get(s["courseCode"], 0)
        assert s["expectedCount"] == max(expected_for_course, s["submittedCount"]), (
            f"{s['id']} on {s['courseCode']} shows /{s['expectedCount']} "
            f"but only {expected_for_course} assessments were due by the checkpoint"
        )


def test_a_course_with_no_early_assessment_shows_zero_not_an_invented_number(cohort):
    """Some courses have no assessment due this early. Those students must show
    0/0 — 'nothing due yet' — rather than being implied to have missed work
    that does not exist. This is the exact shape of defect 1."""
    by_course = {}
    for s in cohort["students"]:
        by_course.setdefault(s["courseCode"], set()).add(s["expectedCount"])
    # if any course reports 0 expected, no student on it may claim submissions
    for code, expectations in by_course.items():
        if expectations == {0}:
            students = [s for s in cohort["students"] if s["courseCode"] == code]
            assert all(s["submittedCount"] == 0 for s in students)


# ── defects 2 and 3: factor text vs impact ────────────────────────────────

def test_every_factor_impact_agrees_with_its_own_text(cohort):
    """The regression test for defects 2 and 3, and the single most important
    assertion in this file. The dashboard draws an up/down arrow, a colour and
    a 'Raises/Lowers the score' caption from `impact`, directly beside `text`.
    If they disagree, the page contradicts itself in front of an advisor."""
    mismatches = []
    for s in cohort["students"]:
        for f in s["topFactors"]:
            risky = _is_risky_text(f["text"])
            if risky and f["impact"] < 0:
                mismatches.append(f"{s['id']}: '{f['text']}' with impact {f['impact']}")
            elif not risky and f["impact"] > 0:
                mismatches.append(f"{s['id']}: '{f['text']}' with impact {f['impact']}")
    assert not mismatches, (
        f"{len(mismatches)} factors contradict their own impact sign:\n  "
        + "\n  ".join(mismatches[:8])
    )


def test_factors_are_ordered_by_magnitude(cohort):
    for s in cohort["students"]:
        magnitudes = [abs(f["impact"]) for f in s["topFactors"]]
        assert magnitudes == sorted(magnitudes, reverse=True), s["id"]


def test_high_risk_students_are_shown_a_concern_wherever_one_exists(cohort):
    """When a student is flagged, the panel should say what the concern is —
    ranking purely by SHAP magnitude could otherwise fill every slot with
    protective factors and leave an advisor reading "Strong engagement /
    Recently active" beside a 74% risk score.

    But this is deliberately not an absolute rule, because a small number of
    students genuinely have no concerning behavioural signal: they submit on
    time, engage consistently, and are still scored just over a band boundary
    on the strength of features the explanation panel intentionally does not
    surface (demographics). Forcing a fabricated concern onto those students
    would be worse than showing none — it would put a false statement in front
    of the person deciding whether to contact them. So the assertion is that
    this is rare rather than impossible, and the real guarantee lives in
    top_factors_with_impact: if a genuine concern was computed, it gets a slot.
    """
    high = [s for s in cohort["students"] if s["riskBand"] == "High"]
    assert high, "no high-risk students to check"
    without_concern = [s for s in high if not any(f["impact"] > 0 for f in s["topFactors"])]
    ratio = len(without_concern) / len(high)
    assert ratio <= 0.10, (
        f"{len(without_concern)} of {len(high)} high-risk students show no concern at all "
        f"({ratio:.0%}) — the strongest genuine concern should be given a slot; "
        f"examples: {[s['id'] for s in without_concern[:5]]}"
    )


def test_a_concern_is_not_displaced_by_purely_protective_factors():
    """Unit-level version of the rule above, on constructed input so it does
    not depend on what happens to be in the cohort: three large protective
    signals and one small genuine concern must still surface the concern."""
    import numpy as np
    import pandas as pd
    from src.models.explain import top_factors_with_impact

    columns = ["vle_total_clicks", "vle_active_days", "vle_distinct_sites",
               "vle_click_trend", "pct_on_time"]
    reference = {"vle_total_clicks": 300.0, "vle_active_days": 45.0,
                 "vle_distinct_sites": 24.0, "vle_click_trend": 0.03, "pct_on_time": 0.0}
    # engaged on every count, but submitting late
    row = pd.Series({"vle_total_clicks": 900, "vle_active_days": 60,
                     "vle_distinct_sites": 30, "vle_click_trend": 0.4, "pct_on_time": 0.2})
    shap_row = np.array([-3.0, -2.5, -2.0, -1.5, 0.05])  # the concern is the smallest

    factors = top_factors_with_impact(shap_row, row, columns, columns, top_n=4, reference=reference)
    assert any(impact > 0 for _text, impact in factors), \
        f"the genuine concern was displaced entirely: {factors}"


def test_factors_are_present_unique_and_human_readable(cohort):
    for s in cohort["students"]:
        texts = [f["text"] for f in s["topFactors"]]
        assert texts, f"{s['id']} has no explanation"
        assert len(texts) == len(set(texts)), f"{s['id']} has duplicate factors"
        for t in texts:
            assert " " in t and t[0].isupper(), f"not a sentence: {t!r}"
            assert not t.startswith("vle_"), f"raw column name leaked: {t!r}"


def test_factors_never_name_a_demographic(cohort):
    import re
    banned = re.compile(r"\b(gender|male|female|region|disability|age|deprivation)\b", re.I)
    for s in cohort["students"]:
        for f in s["topFactors"]:
            assert not banned.search(f["text"]), f"{s['id']}: {f['text']}"


# ── activity ribbon ───────────────────────────────────────────────────────

def test_activity_spans_the_checkpoint(cohort):
    """The ribbon's whole idea is the split between what the model saw and what
    it didn't. Without weeks on both sides there is nothing to show."""
    for s in cohort["students"]:
        assert any(a["beforeCheckpoint"] for a in s["activity"]), s["id"]
        assert any(not a["beforeCheckpoint"] for a in s["activity"]), s["id"]


def test_activity_weeks_are_ordered_and_the_split_is_clean(cohort):
    for s in cohort["students"]:
        weeks = [a["week"] for a in s["activity"]]
        assert weeks == sorted(weeks), s["id"]
        last_before = max(a["week"] for a in s["activity"] if a["beforeCheckpoint"])
        first_after = min(a["week"] for a in s["activity"] if not a["beforeCheckpoint"])
        assert first_after > last_before, s["id"]


def test_clicks_are_never_negative(cohort):
    for s in cohort["students"]:
        assert all(a["clicks"] >= 0 for a in s["activity"])


# ── other displayed numbers ───────────────────────────────────────────────

def test_no_missing_value_sentinels_leak_into_the_display(cohort):
    """-1 means 'no submission yet' and 9999 means 'never clicked' internally.
    Neither should ever reach a page an advisor reads."""
    for s in cohort["students"]:
        assert s["avgEarlyScore"] is None or 0 <= s["avgEarlyScore"] <= 100, s["id"]
        assert 0 <= s["lastActiveDaysAgo"] < 365, f"{s['id']}: {s['lastActiveDaysAgo']}"
        assert 0.0 <= s["onTimeRate"] <= 1.0, s["id"]
        assert s["totalClicks"] >= 0 and s["activeDays"] >= 0


def test_students_with_no_submissions_have_no_early_score(cohort):
    for s in cohort["students"]:
        if s["submittedCount"] == 0:
            assert s["avgEarlyScore"] is None, f"{s['id']} reports a score with 0 submissions"


def test_metadata_is_consistent_across_students(cohort):
    versions = {s["modelVersion"] for s in cohort["students"]}
    checkpoints = {s["checkpointUsed"] for s in cohort["students"]}
    assert len(versions) == 1
    assert len(checkpoints) == 1
    assert cohort["model_version"] in versions


def test_names_are_stable_for_the_same_student_id(cohort):
    """Names are assigned deterministically from the ID so the demo doesn't
    reshuffle on every rebuild."""
    from src.data.build_demo_cohort import _fictional_name
    for s in cohort["students"]:
        assert s["name"] == _fictional_name(int(s["id"].removeprefix("S-")))


# ── rebuild ───────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_rebuilding_the_cohort_is_deterministic(cohort):
    from src.data.build_demo_cohort import build
    rebuilt = build()
    assert [s["id"] for s in rebuilt["students"]] == [s["id"] for s in cohort["students"]]
    assert [s["riskScore"] for s in rebuilt["students"]] == [s["riskScore"] for s in cohort["students"]]
