"""
Generates a synthetic dataset that matches the OULAD (Open University Learning
Analytics Dataset) schema exactly: same tables, same columns, same value
vocabularies. This exists ONLY because this sandboxed environment's network
allowlist cannot reach analyse.kmi.open.ac.uk / UCI / Kaggle to download the
real ~32,600-student dataset (see DATASET.md).

To use the REAL dataset instead: download the CSVs from
https://analyse.kmi.open.ac.uk/open-dataset, drop them into data/raw/ with
the same filenames as in config/config.yaml, and skip this script entirely.
The rest of the pipeline (cutoff.py, features.py, train.py, api/) is written
against the schema below and does not care whether the data is real or
synthetic.

The generation has a deliberate causal structure (more early engagement +
on-time submissions + better early scores -> higher chance of Pass/Distinction)
so that a model trained on it actually has real signal to learn, rather than
just noise.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

MODULES = ["AAA", "BBB", "CCC", "DDD"]
PRESENTATIONS = ["2013B", "2013J", "2014B", "2014J"]  # B=Feb start, J=Oct start
MODULE_LENGTH_DAYS = {"AAA": 268, "BBB": 234, "CCC": 269, "DDD": 240}

GENDERS = ["M", "F"]
REGIONS = [
    "East Anglian Region", "Scotland", "North Western Region", "South Region",
    "West Midlands Region", "Wales", "London Region", "East Midlands Region",
]
EDUCATION = ["Lower Than A Level", "A Level or Equivalent", "HE Qualification"]
IMD_BANDS = ["0-10%", "10-20%", "20-30%", "40-50%", "50-60%", "70-80%", "80-90%", "90-100%"]
AGE_BANDS = ["0-35", "35-55", "55<="]


def _make_courses():
    rows = []
    for m in MODULES:
        for p in PRESENTATIONS:
            rows.append({
                "code_module": m,
                "code_presentation": p,
                "module_presentation_length": MODULE_LENGTH_DAYS[m],
            })
    return pd.DataFrame(rows)


def _make_students(courses: pd.DataFrame, n_per_presentation: int):
    rows = []
    sid = 100000
    for _, c in courses.iterrows():
        for _ in range(n_per_presentation):
            sid += 1
            # latent "ability/engagement propensity" drives everything downstream
            latent = RNG.normal(0, 1)
            rows.append({
                "code_module": c["code_module"],
                "code_presentation": c["code_presentation"],
                "id_student": sid,
                "gender": RNG.choice(GENDERS),
                "region": RNG.choice(REGIONS),
                "highest_education": RNG.choice(EDUCATION, p=[0.35, 0.45, 0.20]),
                "imd_band": RNG.choice(IMD_BANDS),
                "age_band": RNG.choice(AGE_BANDS, p=[0.60, 0.32, 0.08]),
                "num_of_prev_attempts": RNG.choice([0, 0, 0, 1, 2], ),
                "studied_credits": int(RNG.choice([60, 90, 120, 150])),
                "disability": RNG.choice(["N", "Y"], p=[0.90, 0.10]),
                "_latent": latent,  # kept for generation only, dropped before saving
            })
    return pd.DataFrame(rows)


def _make_registration(students: pd.DataFrame):
    rows = []
    for _, s in students.iterrows():
        # most register a bit before/at start (negative day); a shy minority register late
        reg_day = int(RNG.normal(-15, 10))
        reg_day = min(reg_day, 40)
        unregister = None
        rows.append({
            "code_module": s["code_module"],
            "code_presentation": s["code_presentation"],
            "id_student": s["id_student"],
            "date_registration": reg_day,
            "date_unregistration": unregister,  # filled in later for withdrawn students
        })
    return pd.DataFrame(rows)


def _make_assessments(courses: pd.DataFrame, n_per_module=5):
    rows = []
    aid = 5000
    for _, c in courses.iterrows():
        length = c["module_presentation_length"]
        # spread assessments across the course, weight sums to 100
        due_days = sorted(RNG.integers(low=int(length * 0.12), high=int(length * 0.95), size=n_per_module))
        weights = RNG.dirichlet(np.ones(n_per_module)) * 100
        for i in range(n_per_module):
            aid += 1
            rows.append({
                "code_module": c["code_module"],
                "code_presentation": c["code_presentation"],
                "id_assessment": aid,
                "assessment_type": "TMA" if i < n_per_module - 1 else "Exam",
                "date": int(due_days[i]),
                "weight": round(float(weights[i]), 1),
            })
    return pd.DataFrame(rows)


def _make_student_assessment(students: pd.DataFrame, assessments: pd.DataFrame):
    rows = []
    for _, s in students.iterrows():
        latent = s["_latent"]
        a = assessments[(assessments.code_module == s.code_module) & (assessments.code_presentation == s.code_presentation)]
        # engagement propensity: latent ability shifts submission probability and score
        submit_prob_base = 1 / (1 + np.exp(-(latent + 0.3)))
        for _, asmt in a.iterrows():
            if RNG.random() < submit_prob_base:
                # on-time-ish, with some lateness noise; better latent -> earlier & better
                lateness = max(0, int(RNG.normal(-2 - latent * 2, 4)))
                submitted_day = int(asmt["date"]) - max(0, int(RNG.normal(3, 3))) + lateness
                score = float(np.clip(RNG.normal(55 + latent * 18, 12), 0, 100))
                rows.append({
                    "id_assessment": asmt["id_assessment"],
                    "id_student": s["id_student"],
                    "date_submitted": submitted_day,
                    "is_banked": 0,
                    "score": round(score, 1),
                })
    return pd.DataFrame(rows)


def _make_student_vle(students: pd.DataFrame, courses: pd.DataFrame, max_sites=25):
    rows = []
    for _, s in students.iterrows():
        latent = s["_latent"]
        length = int(courses.loc[
            (courses.code_module == s.code_module) & (courses.code_presentation == s.code_presentation),
            "module_presentation_length"
        ].iloc[0])
        # engaged students click on more days, more times per day
        active_day_prob = np.clip(0.55 + latent * 0.20, 0.03, 0.95)
        for day in range(-10, length):
            if RNG.random() < active_day_prob * (0.6 if day < 0 else 1.0):
                n_events = max(1, int(RNG.poisson(2 + max(0, latent) * 2)))
                for _ in range(n_events):
                    rows.append({
                        "code_module": s["code_module"],
                        "code_presentation": s["code_presentation"],
                        "id_student": s["id_student"],
                        "id_site": int(RNG.integers(1, max_sites)),
                        "date": day,
                        "sum_click": int(RNG.integers(1, 6)),
                    })
    return pd.DataFrame(rows)


def _assign_final_result(students: pd.DataFrame, student_assessment: pd.DataFrame, student_vle: pd.DataFrame):
    results = []
    sa_by_student = student_assessment.groupby("id_student")["score"].mean()
    clicks_by_student = student_vle.groupby("id_student")["sum_click"].sum()
    for _, s in students.iterrows():
        latent = s["_latent"]
        avg_score = sa_by_student.get(s["id_student"], np.nan)
        total_clicks = clicks_by_student.get(s["id_student"], 0)
        engagement_z = (total_clicks - 400) / 400
        score_component = 0 if np.isnan(avg_score) else (avg_score - 50) / 20
        combined = 0.5 * latent + 0.3 * score_component + 0.2 * engagement_z + RNG.normal(0, 0.5)
        if total_clicks < 15 and (np.isnan(avg_score) or avg_score < 20):
            final = "Withdrawn"
        elif combined > 0.9:
            final = "Distinction"
        elif combined > -0.1:
            final = "Pass"
        elif combined > -0.9:
            final = "Fail"
        else:
            final = "Withdrawn"
        results.append(final)
    return results


def generate(n_per_presentation: int = 140, seed: int = 42, out_dir: str = "data/raw"):
    global RNG
    RNG = np.random.default_rng(seed)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    courses = _make_courses()
    students = _make_students(courses, n_per_presentation)
    registration = _make_registration(students)
    assessments = _make_assessments(courses)
    student_assessment = _make_student_assessment(students, assessments)
    student_vle = _make_student_vle(students, courses)

    final_results = _assign_final_result(students, student_assessment, student_vle)
    students = students.assign(final_result=final_results)

    # backfill date_unregistration for withdrawn students (a leakage-relevant column
    # that MUST be excluded from any early-checkpoint feature set)
    withdrawn_mask = students["final_result"] == "Withdrawn"
    withdrawn_ids = set(students.loc[withdrawn_mask, "id_student"])
    reg_days = registration.set_index("id_student")["date_registration"].to_dict()
    unregister_days = {}
    for sid in withdrawn_ids:
        start = reg_days.get(sid, 0)
        unregister_days[sid] = int(start + max(5, RNG.normal(60, 30)))
    registration["date_unregistration"] = registration["id_student"].map(unregister_days)

    student_info = students.drop(columns=["_latent"])

    student_info.to_csv(out_path / "studentInfo.csv", index=False)
    registration.to_csv(out_path / "studentRegistration.csv", index=False)
    courses.to_csv(out_path / "courses.csv", index=False)
    assessments.to_csv(out_path / "assessments.csv", index=False)
    student_assessment.to_csv(out_path / "studentAssessment.csv", index=False)
    student_vle.to_csv(out_path / "studentVle.csv", index=False)

    print(f"Generated synthetic OULAD-schema dataset in {out_path}/")
    print(f"  students:            {len(student_info):,}")
    print(f"  vle interactions:    {len(student_vle):,}")
    print(f"  assessment records:  {len(student_assessment):,}")
    print(f"  final_result counts: {student_info['final_result'].value_counts().to_dict()}")


if __name__ == "__main__":
    generate()
