"""
FastAPI service for the EDU-02 early-warning model.

Run locally:  uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Run in Docker: see Dockerfile / docker-compose.yml
"""
import json
import logging
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

_state = {"model": None, "metadata": None, "explainer": None, "config": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model artifact ONCE at startup, not per-request."""
    config = yaml.safe_load(open("config/config.yaml"))
    _state["config"] = config
    model_path = Path(config["paths"]["model_artifact"])
    metadata_path = Path(config["paths"]["metadata_artifact"])

    if model_path.exists() and metadata_path.exists():
        _state["model"] = joblib.load(model_path)
        _state["metadata"] = json.load(open(metadata_path))
        _state["explainer"] = shap.TreeExplainer(_state["model"])
        logger.info("Loaded model %s (trained %s)",
                    _state["metadata"]["model_version"], _state["metadata"]["trained_at"])
    else:
        logger.warning("Model artifact not found at %s — run `python -m src.models.train` first.", model_path)

    yield
    _state.update({"model": None, "metadata": None, "explainer": None})


app = FastAPI(
    title="EDU-02 Student Early-Warning API",
    description="Predicts a student's risk of failing/withdrawing, using only "
                 "information available at an early, configurable course checkpoint.",
    version="1.0",
    lifespan=lifespan,
)

# The dashboard (frontend/) runs on a different origin in development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
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
