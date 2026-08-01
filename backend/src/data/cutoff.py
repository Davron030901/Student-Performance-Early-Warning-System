"""
This module contains the ONE function responsible for preventing data
leakage: apply_checkpoint_cutoff(). Every feature, in both training and
inference, must be derived from data that has passed through this function.

Why this matters: the whole point of an early-warning system is that it only
uses information that would genuinely be available at the prediction point.
If training accidentally uses later-course or final-outcome data, offline
metrics look great but the model is useless (or actively misleading) in
production, because that information simply won't exist yet when a real
prediction is needed.

See tests/test_leakage.py for the automated check that enforces this.
"""
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Checkpoint:
    """A resolved, per-course-presentation cutoff day, relative to course start (day 0)."""
    code_module: str
    code_presentation: str
    cutoff_day: int


def resolve_checkpoints(courses: pd.DataFrame, checkpoint_fraction: float) -> list[Checkpoint]:
    """Turn the configured fraction (e.g. 0.30) into a concrete day-offset per
    course presentation, since module lengths differ (see courses.csv)."""
    checkpoints = []
    for _, c in courses.iterrows():
        cutoff_day = int(round(c["module_presentation_length"] * checkpoint_fraction))
        checkpoints.append(Checkpoint(c["code_module"], c["code_presentation"], cutoff_day))
    return checkpoints


def apply_checkpoint_cutoff(
    student_vle: pd.DataFrame,
    student_assessment: pd.DataFrame,
    assessments: pd.DataFrame,
    checkpoints: list[Checkpoint],
) -> dict[str, pd.DataFrame]:
    """
    The core leakage-prevention filter. Returns copies of student_vle and
    student_assessment truncated so that ONLY rows dated on/before each
    course-presentation's cutoff day survive.

    - student_vle rows are filtered directly on their own `date` column.
    - student_assessment rows are filtered on the DUE DATE of the assessment
      they belong to (via assessments.date), not the submission date. This
      matters: an assessment due after the cutoff must not be used even if a
      student submitted it early, because at true inference time we would not
      yet know the score for an assessment that a real student hasn't
      reached.
    """
    cutoff_by_course = {(c.code_module, c.code_presentation): c.cutoff_day for c in checkpoints}

    vle = student_vle.copy()
    vle["_cutoff"] = vle.apply(
        lambda r: cutoff_by_course.get((r["code_module"], r["code_presentation"])), axis=1
    )
    vle_filtered = vle[vle["date"] <= vle["_cutoff"]].drop(columns=["_cutoff"])

    # assessments due before/at cutoff, per course presentation
    assessments_with_cutoff = assessments.copy()
    assessments_with_cutoff["_cutoff"] = assessments_with_cutoff.apply(
        lambda r: cutoff_by_course.get((r["code_module"], r["code_presentation"])), axis=1
    )
    eligible_assessment_ids = set(
        assessments_with_cutoff.loc[
            assessments_with_cutoff["date"] <= assessments_with_cutoff["_cutoff"], "id_assessment"
        ]
    )

    sa = student_assessment.copy()
    sa_filtered = sa[sa["id_assessment"].isin(eligible_assessment_ids)]

    return {
        "student_vle": vle_filtered.reset_index(drop=True),
        "student_assessment": sa_filtered.reset_index(drop=True),
        "eligible_assessment_ids": eligible_assessment_ids,
    }


def max_permitted_day(checkpoints: list[Checkpoint], code_module: str, code_presentation: str) -> int:
    for c in checkpoints:
        if c.code_module == code_module and c.code_presentation == code_presentation:
            return c.cutoff_day
    raise KeyError(f"No checkpoint resolved for {code_module}/{code_presentation}")
