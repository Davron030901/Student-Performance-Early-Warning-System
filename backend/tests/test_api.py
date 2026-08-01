"""API endpoint tests, including invalid-input handling."""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # context manager triggers the startup event
        yield c


def test_health_reports_model_loaded(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True, "run `python -m src.models.train` before testing the API"


def test_model_info_exposes_metrics_and_version(client):
    r = client.get("/api/v1/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"]
    assert 0 < body["checkpoint_fraction"] < 1
    assert "recall" in body["held_out_metrics"]


def test_predict_returns_wellformed_response(client, sample_student_payload):
    r = client.post("/api/v1/predict", json=sample_student_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["student_id"] == sample_student_payload["student_id"]
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_band"] in {"Low", "Medium", "High"}
    assert isinstance(body["top_factors"], list) and len(body["top_factors"]) > 0
    assert body["checkpoint_used"]


def test_predict_flags_a_disengaged_student_as_elevated_risk(client, sample_student_payload):
    """Behavioural check: a student with no submissions and almost no activity
    should not come back as Low risk."""
    r = client.post("/api/v1/predict", json=sample_student_payload)
    assert r.json()["risk_band"] in {"Medium", "High"}


def test_predict_scores_an_engaged_student_lower_than_a_disengaged_one(client, sample_student_payload):
    engaged = dict(sample_student_payload)
    engaged.update({
        "student_id": "S-TEST-02", "date_registration": -20, "late_registration": 0,
        "vle_total_clicks": 520, "vle_active_days": 55, "vle_distinct_sites": 22,
        "vle_click_trend": 0.15, "vle_days_since_last_click": 1,
        "n_submitted": 2, "avg_early_score": 82, "pct_on_time": 1.0, "avg_days_early": 3.5,
    })
    disengaged_score = client.post("/api/v1/predict", json=sample_student_payload).json()["risk_score"]
    engaged_score = client.post("/api/v1/predict", json=engaged).json()["risk_score"]
    assert engaged_score < disengaged_score


def test_batch_predict_returns_one_prediction_per_student(client, sample_student_payload):
    second = dict(sample_student_payload, student_id="S-TEST-03")
    r = client.post("/api/v1/predict/batch", json={"students": [sample_student_payload, second]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2
    assert {p["student_id"] for p in preds} == {"S-TEST-01", "S-TEST-03"}


def test_missing_required_field_is_rejected(client, sample_student_payload):
    broken = dict(sample_student_payload)
    del broken["vle_total_clicks"]
    r = client.post("/api/v1/predict", json=broken)
    assert r.status_code == 422


def test_out_of_range_value_is_rejected(client, sample_student_payload):
    broken = dict(sample_student_payload, pct_on_time=5.0)  # must be 0..1
    r = client.post("/api/v1/predict", json=broken)
    assert r.status_code == 422


def test_invalid_enum_value_is_rejected(client, sample_student_payload):
    broken = dict(sample_student_payload, gender="X")
    r = client.post("/api/v1/predict", json=broken)
    assert r.status_code == 422


def test_explanations_never_contradict_the_underlying_values(client, sample_student_payload):
    """Regression test.

    An earlier version worded each factor from the sign of its SHAP value, which
    is not the same thing as whether the behaviour itself is good or bad. That
    produced explanations like "engagement has been picking up recently" for a
    student whose click trend was clearly negative — a contradiction that would
    reasonably destroy an advisor's trust in every other factor shown.

    Wording is now driven by the feature's actual value, so these pairs must
    never appear together.
    """
    declining = dict(sample_student_payload, vle_click_trend=-0.4, vle_days_since_last_click=25)
    factors = client.post("/api/v1/predict", json=declining).json()["top_factors"]
    joined = " | ".join(factors).lower()
    assert "picking up" not in joined and "holding up" not in joined, factors
    assert "recently active on the course site" not in joined, factors

    rising = dict(
        sample_student_payload,
        vle_click_trend=0.3, vle_total_clicks=600, vle_active_days=58,
        vle_distinct_sites=24, vle_days_since_last_click=1,
        n_submitted=2, avg_early_score=85, pct_on_time=1.0, avg_days_early=4,
    )
    factors = client.post("/api/v1/predict", json=rising).json()["top_factors"]
    joined = " | ".join(factors).lower()
    assert "dropping off" not in joined, factors
    assert "has not logged into" not in joined, factors
    assert "below average" not in joined, factors


def test_a_student_with_no_submissions_is_told_so_explicitly(client, sample_student_payload):
    """avg_early_score uses -1 as a 'no submission yet' sentinel. It must never
    be reported as a low score, which would be factually wrong."""
    payload = dict(sample_student_payload, n_submitted=0, avg_early_score=-1)
    factors = client.post("/api/v1/predict", json=payload).json()["top_factors"]
    joined = " | ".join(factors).lower()
    assert "scores are below average" not in joined, factors
