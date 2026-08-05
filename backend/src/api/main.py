"""
FastAPI service for the EDU-02 early-warning model.

Run locally:  uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Run in Docker: see Dockerfile / docker-compose.yml
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import shap
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    StudentFeatures, PredictionResponse, BatchPredictRequest, BatchPredictResponse,
    ModelInfoResponse, HealthResponse,
)
from src.models.encoding import align_to_model_columns
from src.models.explain import top_factors_for_student

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edu02-api")

# Resolve paths from the project root rather than the current working
# directory, so the app starts identically under uvicorn locally, in
# Docker (WORKDIR /app), and under Render's process manager.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

_state = {"model": None, "metadata": None, "explainer": None, "config": None, "cohort": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model artifact ONCE at startup, not per-request."""
    config = yaml.safe_load(open(CONFIG_PATH))
    _state["config"] = config
    model_path = PROJECT_ROOT / config["paths"]["model_artifact"]
    metadata_path = PROJECT_ROOT / config["paths"]["metadata_artifact"]

    if model_path.exists() and metadata_path.exists():
        _state["model"] = joblib.load(model_path)
        _state["metadata"] = json.load(open(metadata_path))
        _state["explainer"] = shap.TreeExplainer(_state["model"])
        logger.info("Loaded model %s (trained %s)",
                    _state["metadata"]["model_version"], _state["metadata"]["trained_at"])
    else:
        logger.warning("Model artifact not found at %s — run `python -m src.models.train` first.", model_path)

    # Roster served to the dashboard; independent of the model artifact, so a
    # missing cohort degrades only the dashboard endpoints, not /predict.
    cohort_path = PROJECT_ROOT / "models" / "artifacts" / "demo_cohort.json"
    if cohort_path.exists():
        _state["cohort"] = json.load(open(cohort_path))
        logger.info("Loaded cohort: %d students", len(_state["cohort"]["students"]))
    else:
        logger.warning("Cohort not found at %s — dashboard endpoints will return 503. "
                       "Run `python -m src.data.build_demo_cohort`.", cohort_path)

    yield
    _state.update({"model": None, "metadata": None, "explainer": None, "cohort": None})


app = FastAPI(
    title="EDU-02 Student Early-Warning API",
    description="Predicts a student's risk of failing/withdrawing, using only "
                 "information available at an early, configurable course checkpoint.",
    version="1.0",
    lifespan=lifespan,
)

# Allowed browser origins.
#
# Render injects no origin config of its own, so this comes from the
# CORS_ALLOW_ORIGINS env var: a comma-separated list, e.g.
#   CORS_ALLOW_ORIGINS=https://course-signals.vercel.app,https://www.example.edu
#
# Vercel gives every deployment a unique preview URL, so preview builds are
# matched by regex rather than listed one by one. Note that allow_credentials
# is False: this API has no cookies or auth, and pairing credentials with a
# permissive origin regex is exactly the combination that turns CORS into a
# real vulnerability.
_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "")
_allow_origins = [o.strip() for o in _origins_env.split(",") if o.strip()] or [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_origin_regex=os.getenv("CORS_ALLOW_ORIGIN_REGEX") or r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _require_model():
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `python -m src.models.train` first.")


def _risk_band(prob: float, config: dict) -> str:
    bands = config["risk_bands"]
    if prob <= bands["low_max"]:
        return "Low"
    if prob <= bands["medium_max"]:
        return "Medium"
    return "High"


def _predict_one(student: StudentFeatures) -> PredictionResponse:
    metadata = _state["metadata"]
    config = _state["config"]
    feature_columns = metadata["feature_columns"]

    row_dict = student.model_dump(exclude={"student_id"})
    row_df = pd.DataFrame([row_dict])[feature_columns]

    row_encoded = align_to_model_columns(row_df, metadata["model_feature_columns"])

    proba = float(_state["model"].predict_proba(row_encoded)[0, 1])
    band = _risk_band(proba, config)

    shap_values = _state["explainer"].shap_values(row_encoded)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    factors = top_factors_for_student(
        shap_row=shap_values[0],
        feature_row=row_encoded.iloc[0],
        all_feature_columns=metadata["model_feature_columns"],
        base_feature_names=feature_columns,
        top_n=4,
        reference=metadata.get("reference_medians"),
    )
    if not factors:
        factors = ["No single dominant factor — risk is spread across several small signals."]

    checkpoint_pct = metadata["checkpoint_fraction"]
    return PredictionResponse(
        student_id=student.student_id,
        risk_score=round(proba, 4),
        risk_band=band,
        checkpoint_used=f"{checkpoint_pct:.0%} of course length",
        top_factors=factors,
        model_version=metadata["model_version"],
    )


@app.get("/api/v1/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=_state["model"] is not None)


@app.get("/api/v1/model/info", response_model=ModelInfoResponse)
def model_info():
    _require_model()
    m = _state["metadata"]
    return ModelInfoResponse(
        model_version=m["model_version"],
        trained_at=m["trained_at"],
        checkpoint_fraction=m["checkpoint_fraction"],
        n_training_rows=m["n_training_rows"],
        held_out_metrics=m["held_out_metrics"],
        cv_metrics=m["cv_metrics"],
    )


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict(student: StudentFeatures):
    _require_model()
    try:
        return _predict_one(student)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=422, detail=f"Could not score student: {e}")


@app.post("/api/v1/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest):
    _require_model()
    predictions = []
    for student in payload.students:
        try:
            predictions.append(_predict_one(student))
        except Exception as e:
            logger.exception("Batch prediction failed for student %s", student.student_id)
            raise HTTPException(status_code=422, detail=f"Could not score student {student.student_id}: {e}")
    return BatchPredictResponse(predictions=predictions)


# ── Cohort endpoints ──────────────────────────────────────────────────────
#
# These serve the advisor dashboard. The prediction endpoints above score a
# payload the caller supplies; they have no notion of "the current cohort".
# The dashboard needs a roster to display, so these read a pre-computed
# artifact (models/artifacts/demo_cohort.json) built by
# `python -m src.data.build_demo_cohort`: real held-out students, real
# early-course features cut at the checkpoint, scored by this same model and
# explained with the same SHAP path as /predict. Only the names are fictional,
# because OULAD is anonymised.
#
# In a real deployment this is where an institutional student database would
# be queried instead. The response shapes would not change.

PAGE_SIZE = 12


def _require_cohort():
    if _state["cohort"] is None:
        raise HTTPException(
            status_code=503,
            detail="Cohort data not available. Run `python -m src.data.build_demo_cohort` "
                   "and redeploy, or point this endpoint at your student database.",
        )
    return _state["cohort"]


@app.get("/api/v1/courses")
def list_courses():
    return _require_cohort()["courses"]


@app.get("/api/v1/students")
def list_students(
    course: str | None = None,
    riskBand: str | None = None,
    search: str | None = None,
    page: int = 1,
    sortBy: str = "risk",
):
    cohort = _require_cohort()
    rows = list(cohort["students"])

    if course and course != "all":
        rows = [s for s in rows if s["courseCode"] == course]
    if riskBand and riskBand != "all":
        rows = [s for s in rows if s["riskBand"] == riskBand]
    if search and search.strip():
        q = search.strip().lower()
        rows = [s for s in rows if q in s["name"].lower() or q in s["id"].lower()]

    if sortBy == "name":
        rows.sort(key=lambda s: s["name"])
    elif sortBy == "lastActive":
        rows.sort(key=lambda s: s["lastActiveDaysAgo"], reverse=True)
    else:
        rows.sort(key=lambda s: s["riskScore"], reverse=True)

    page = max(1, page)
    start = (page - 1) * PAGE_SIZE
    return {
        "students": rows[start:start + PAGE_SIZE],
        "total": len(rows),
        "page": page,
        "pageSize": PAGE_SIZE,
    }


@app.get("/api/v1/students/{student_id}")
def get_student(student_id: str):
    cohort = _require_cohort()
    for student in cohort["students"]:
        if student["id"] == student_id:
            return student
    raise HTTPException(status_code=404, detail=f"No student with id {student_id}")


@app.get("/api/v1/overview")
def overview():
    cohort = _require_cohort()
    students = cohort["students"]

    counts = {"Low": 0, "Medium": 0, "High": 0}
    for s in students:
        counts[s["riskBand"]] += 1

    needs_attention = sorted(students, key=lambda s: s["riskScore"], reverse=True)[:5]

    by_course = []
    for course in cohort["courses"]:
        in_course = [s for s in students if s["courseCode"] == course["code"]]
        if not in_course:
            continue
        by_course.append({
            "course": course,
            "total": len(in_course),
            "high": sum(1 for s in in_course if s["riskBand"] == "High"),
            "medium": sum(1 for s in in_course if s["riskBand"] == "Medium"),
            "low": sum(1 for s in in_course if s["riskBand"] == "Low"),
        })

    return {
        "counts": counts,
        "total": len(students),
        "needsAttention": needs_attention,
        "byCourse": by_course,
    }
