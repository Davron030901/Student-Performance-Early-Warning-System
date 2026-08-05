"""
Builds `models/artifacts/demo_cohort.json` — the roster the dashboard reads.

Why this exists: the API scores a feature payload you send it. It has no
database and no notion of "the current cohort", but the dashboard needs a list
of students to show. Rather than inventing numbers on the frontend, this
pre-computes a real one:

  * real students from the held-out module presentation (the cohort the model
    never trained on),
  * real early-course features, cut at the configured checkpoint by the same
    `apply_checkpoint_cutoff` everything else uses,
  * scored by the real trained model,
  * explained by real SHAP values through the same `top_factors_for_student`
    the /predict endpoint calls.

The only fabricated part is the names. OULAD is anonymised — students are
integer IDs — so displaying "Student 623847" would be useless to an advisor
and displaying a real name is impossible. Fictional names are assigned
deterministically by ID so the demo is stable across rebuilds.

Run: python -m src.data.build_demo_cohort   (or `make cohort`)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.data.ingest import load_raw_tables
from src.data.cutoff import resolve_checkpoints, apply_checkpoint_cutoff
from src.data.pipeline import build_dataset
from src.models.encoding import align_to_model_columns
from src.models.explain import compute_shap_values, top_factors_with_impact

HELD_OUT_PRESENTATION = "2014J"
MAX_STUDENTS = 90          # keeps the artifact small; the dashboard paginates anyway

FIRST_NAMES = [
    "Amara", "Yusuf", "Priya", "Tomas", "Nadia", "Ewan", "Chloe", "Ravi", "Marta", "Idris",
    "Freya", "Kwame", "Sinead", "Omar", "Lena", "Hugo", "Aisha", "Callum", "Mei", "Sofia",
    "Dario", "Halima", "Jonas", "Zainab", "Rory", "Tess", "Bilal", "Anika", "Felix", "Noor",
    "Kiera", "Milos", "Sana", "Declan", "Yara", "Otto", "Nia", "Emil", "Layla", "Gus",
]
LAST_NAMES = [
    "Okafor", "Demir", "Raman", "Novak", "Haddad", "Fraser", "Bennett", "Iyer", "Kowalski",
    "Abubakar", "Lindqvist", "Mensah", "O'Rourke", "Farouk", "Vogel", "Marchetti", "Bello",
    "Reid", "Chen", "Duarte", "Ferro", "Osman", "Larsen", "Karim", "Mackay", "Whitfield",
    "Aziz", "Sharma", "Brandt", "Haq",
]

COURSE_TITLES = {
    "AAA": "Foundations of Social Policy",
    "BBB": "Introductory Statistics",
    "CCC": "Computing & IT Practice",
    "DDD": "Environmental Science",
}


def _fictional_name(student_id: int) -> str:
    """Deterministic from the ID, so the roster is stable across rebuilds."""
    return f"{FIRST_NAMES[student_id % len(FIRST_NAMES)]} {LAST_NAMES[(student_id // 7) % len(LAST_NAMES)]}"


def _risk_band(prob: float, bands: dict) -> str:
    if prob <= bands["low_max"]:
        return "Low"
    if prob <= bands["medium_max"]:
        return "Medium"
    return "High"


def _weekly_activity(vle_all: pd.DataFrame, student_id: int, code_module: str,
                      code_presentation: str, cutoff_day: int, weeks_after: int = 6) -> list[dict]:
    """Weekly click totals spanning the checkpoint. The rows AFTER the cutoff are
    deliberately included: the dashboard's engagement ribbon shows them hatched,
    as time the model could not see. They are display-only and never fed to the
    model."""
    rows = vle_all[
        (vle_all["id_student"] == student_id)
        & (vle_all["code_module"] == code_module)
        & (vle_all["code_presentation"] == code_presentation)
    ]
    if rows.empty:
        return []

    last_day = cutoff_day + weeks_after * 7
    rows = rows[rows["date"] <= last_day]
    weeks = []
    total_weeks = max(1, int(np.ceil((last_day + 10) / 7)))
    for w in range(total_weeks):
        start, end = w * 7 - 10, (w + 1) * 7 - 10
        clicks = int(rows.loc[(rows["date"] >= start) & (rows["date"] < end), "sum_click"].sum())
        weeks.append({
            "week": w + 1,
            "clicks": clicks,
            "beforeCheckpoint": bool(end <= cutoff_day),
        })
    # guarantee at least one week on each side so the ribbon always renders
    if not any(x["beforeCheckpoint"] for x in weeks) or all(x["beforeCheckpoint"] for x in weeks):
        return []
    return weeks


def build(config_path: str = "config/config.yaml") -> dict:
    config = yaml.safe_load(open(config_path))
    raw = load_raw_tables(config)
    fraction = config["prediction"]["checkpoint_fraction"]

    X, y, feature_columns, meta = build_dataset(raw, config, checkpoint_fraction=fraction)
    metadata = json.load(open(config["paths"]["metadata_artifact"]))
    model = joblib.load(config["paths"]["model_artifact"])

    held = X[X["code_presentation"] == HELD_OUT_PRESENTATION].copy()
    if held.empty:
        held = X.copy()

    # keep a spread of risk rather than the first N rows
    encoded_all = align_to_model_columns(held[feature_columns], metadata["model_feature_columns"])
    held["_score"] = model.predict_proba(encoded_all)[:, 1]
    held = held.sort_values("_score", ascending=False)
    step = max(1, len(held) // MAX_STUDENTS)
    sample = held.iloc[::step].head(MAX_STUDENTS).copy()

    encoded = align_to_model_columns(sample[feature_columns], metadata["model_feature_columns"])
    probabilities = model.predict_proba(encoded)[:, 1]
    shap_values, _ = compute_shap_values(model, encoded)

    checkpoints = resolve_checkpoints(raw["courses"], fraction)
    cutoff_by_course = {(c.code_module, c.code_presentation): c.cutoff_day for c in checkpoints}
    vle_all = raw["student_vle"]
    reference = metadata.get("reference_medians")

    # How many assessments were actually DUE before the checkpoint, per course
    # presentation. This is the real denominator for "n of m submitted".
    # Deriving it from the student's own submission count instead (an earlier
    # mistake here) makes everyone who submitted anything look complete —
    # "1/1 submitted" beside a 99% risk score — and invents assessments that
    # don't exist for courses whose first assessment falls after the checkpoint.
    cutoff_result = apply_checkpoint_cutoff(
        raw["student_vle"], raw["student_assessment"], raw["assessments"], checkpoints
    )
    eligible = raw["assessments"][
        raw["assessments"]["id_assessment"].isin(cutoff_result["eligible_assessment_ids"])
    ]
    expected_by_course = (
        eligible.groupby(["code_module", "code_presentation"])["id_assessment"]
        .nunique().to_dict()
    )

    students = []
    for i, (_, row) in enumerate(sample.iterrows()):
        sid = int(row["id_student"])
        module, presentation = row["code_module"], row["code_presentation"]
        cutoff_day = cutoff_by_course[(module, presentation)]

        activity = _weekly_activity(vle_all, sid, module, presentation, cutoff_day)
        if not activity:
            continue

        probability = float(probabilities[i])
        # phrase and its own SHAP value, together — never paired after the fact
        factors = top_factors_with_impact(
            shap_row=shap_values[i],
            feature_row=encoded.iloc[i],
            all_feature_columns=metadata["model_feature_columns"],
            base_feature_names=feature_columns,
            top_n=4,
            reference=reference,
        )

        submitted = int(round(row.get("n_submitted", 0)))
        # 0 is a legitimate value: some courses have no assessment due this
        # early, and the dashboard should say so rather than imply a miss.
        expected = int(expected_by_course.get((module, presentation), 0))
        expected = max(expected, submitted)

        last_click = row.get("vle_days_since_last_click", 0)
        last_active = int(min(last_click, 365)) if np.isfinite(last_click) else 0

        students.append({
            "id": f"S-{sid}",
            "name": _fictional_name(sid),
            "courseCode": module,
            "riskScore": round(probability, 4),
            "riskBand": _risk_band(probability, config["risk_bands"]),
            "submittedCount": submitted,
            "expectedCount": expected,
            "lastActiveDaysAgo": last_active,
            "activity": activity,
            "topFactors": [
                {"text": text, "impact": round(impact, 4)} for text, impact in factors
            ],
            "avgEarlyScore": (None if row.get("avg_early_score", -1) < 0
                              else round(float(row["avg_early_score"]), 1)),
            "onTimeRate": round(float(row.get("pct_on_time", 0)), 3),
            "totalClicks": int(row.get("vle_total_clicks", 0)),
            "activeDays": int(row.get("vle_active_days", 0)),
            "registeredDay": int(row.get("date_registration", 0)),
            "checkpointUsed": f"{fraction:.0%} of course length",
            "modelVersion": metadata["model_version"],
        })

    courses = []
    for (module, presentation), cutoff_day in cutoff_by_course.items():
        if presentation != HELD_OUT_PRESENTATION:
            continue
        length = int(raw["courses"].loc[
            (raw["courses"].code_module == module)
            & (raw["courses"].code_presentation == presentation),
            "module_presentation_length"].iloc[0])
        courses.append({
            "code": module,
            "title": COURSE_TITLES.get(module, module),
            "presentation": presentation,
            "lengthDays": length,
            "checkpointDay": cutoff_day,
        })

    cohort = {
        "generated_from": HELD_OUT_PRESENTATION,
        "checkpoint_fraction": fraction,
        "model_version": metadata["model_version"],
        "courses": courses,
        "students": students,
    }

    out = Path("models/artifacts/demo_cohort.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cohort))
    counts = {b: sum(1 for s in students if s["riskBand"] == b) for b in ["Low", "Medium", "High"]}
    print(f"Wrote {out} — {len(students)} students across {len(courses)} courses")
    print(f"  risk bands: {counts}")
    print(f"  size: {out.stat().st_size / 1024:.0f} KB")
    return cohort


if __name__ == "__main__":
    build()
