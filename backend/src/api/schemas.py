from typing import Literal
from pydantic import BaseModel, Field


class StudentFeatures(BaseModel):
    """Early-course engineered features for one student, as of the prediction
    checkpoint. See README.md for how these are produced by the data pipeline
    (src/data/pipeline.py) from the raw OULAD-schema tables."""
    student_id: str = Field(..., examples=["12345"])

    # known at enrollment
    gender: Literal["M", "F"]
    region: str = Field(..., examples=["London Region"])
    highest_education: str = Field(..., examples=["A Level or Equivalent"])
    imd_band: str = Field(..., examples=["20-30%"])
    age_band: str = Field(..., examples=["0-35"])
    num_of_prev_attempts: int = Field(..., ge=0)
    studied_credits: int = Field(..., gt=0)
    disability: Literal["Y", "N"]
    date_registration: int = Field(..., description="Days relative to course start; negative = before start")
    late_registration: int = Field(..., ge=0, le=1)

    # engineered from early-course activity (already checkpoint-filtered)
    vle_total_clicks: float = Field(..., ge=0)
    vle_active_days: float = Field(..., ge=0)
    vle_distinct_sites: float = Field(..., ge=0)
    vle_click_trend: float
    vle_days_since_last_click: float = Field(..., ge=0)
    n_submitted: float = Field(..., ge=0)
    avg_early_score: float = Field(..., description="-1 sentinel means no submission yet")
    pct_on_time: float = Field(..., ge=0, le=1)
    avg_days_early: float


class PredictionResponse(BaseModel):
    student_id: str
    risk_score: float
    risk_band: Literal["Low", "Medium", "High"]
    checkpoint_used: str
    top_factors: list[str]
    model_version: str


class BatchPredictRequest(BaseModel):
    students: list[StudentFeatures]


class BatchPredictResponse(BaseModel):
    predictions: list[PredictionResponse]


class ModelInfoResponse(BaseModel):
    model_version: str
    trained_at: str
    checkpoint_fraction: float
    n_training_rows: int
    held_out_metrics: dict
    cv_metrics: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
