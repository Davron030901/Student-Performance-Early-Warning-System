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


# ══════════════════════════════════════════════════════════════════════════
# Exhaustive endpoint coverage
# ══════════════════════════════════════════════════════════════════════════

# ── contract / shape ──────────────────────────────────────────────────────

def test_predict_response_has_exactly_the_documented_fields(client, sample_student_payload):
    body = client.post("/api/v1/predict", json=sample_student_payload).json()
    assert set(body) == {
        "student_id", "risk_score", "risk_band", "checkpoint_used",
        "top_factors", "model_version",
    }


def test_risk_band_is_consistent_with_the_risk_score(client, sample_student_payload, config):
    """The band is what advisors act on; if it ever disagreed with the score the
    interface would be showing two different answers."""
    bands = config["risk_bands"]
    for clicks in [5, 80, 300, 900]:
        payload = dict(sample_student_payload, vle_total_clicks=clicks, vle_active_days=clicks // 12)
        body = client.post("/api/v1/predict", json=payload).json()
        score, band = body["risk_score"], body["risk_band"]
        expected = "Low" if score <= bands["low_max"] else "Medium" if score <= bands["medium_max"] else "High"
        assert band == expected, f"score {score} labelled {band}"


def test_checkpoint_used_matches_the_trained_configuration(client, sample_student_payload, metadata):
    body = client.post("/api/v1/predict", json=sample_student_payload).json()
    assert body["checkpoint_used"] == f"{metadata['checkpoint_fraction']:.0%} of course length"


def test_model_version_matches_the_shipped_artifact(client, sample_student_payload, metadata):
    body = client.post("/api/v1/predict", json=sample_student_payload).json()
    assert body["model_version"] == metadata["model_version"]


def test_student_id_is_echoed_verbatim(client, sample_student_payload):
    for sid in ["S-1", "abc-123", "999999", "student_with_underscores"]:
        body = client.post("/api/v1/predict", json=dict(sample_student_payload, student_id=sid)).json()
        assert body["student_id"] == sid


# ── determinism and monotonicity ──────────────────────────────────────────

def test_the_same_input_always_scores_the_same(client, sample_student_payload):
    scores = {client.post("/api/v1/predict", json=sample_student_payload).json()["risk_score"]
              for _ in range(5)}
    assert len(scores) == 1, f"non-deterministic scoring: {scores}"


def test_risk_falls_monotonically_as_engagement_rises(client, sample_student_payload):
    """Not a strict mathematical guarantee of the model, but a strong sanity
    property: piling on good signals must not increase risk."""
    previous = 1.1
    for clicks, days, submitted, score in [(5, 1, 0, -1), (150, 20, 1, 55), (400, 40, 2, 70), (800, 60, 2, 90)]:
        payload = dict(
            sample_student_payload, vle_total_clicks=clicks, vle_active_days=days,
            vle_distinct_sites=min(24, days // 2), n_submitted=submitted,
            avg_early_score=score, pct_on_time=1.0 if submitted else 0.0,
            vle_days_since_last_click=1 if clicks > 100 else 25,
        )
        current = client.post("/api/v1/predict", json=payload).json()["risk_score"]
        assert current <= previous + 1e-9, "more engagement should never raise risk"
        previous = current


def test_a_student_with_no_activity_at_all_is_high_risk(client, sample_student_payload):
    payload = dict(
        sample_student_payload, vle_total_clicks=0, vle_active_days=0, vle_distinct_sites=0,
        vle_click_trend=0, vle_days_since_last_click=60, n_submitted=0, avg_early_score=-1,
    )
    body = client.post("/api/v1/predict", json=payload).json()
    assert body["risk_band"] == "High"


# ── explanations ──────────────────────────────────────────────────────────

def test_every_prediction_returns_at_least_one_factor(client, sample_student_payload):
    for clicks in [0, 50, 400, 900]:
        body = client.post("/api/v1/predict", json=dict(sample_student_payload, vle_total_clicks=clicks)).json()
        assert len(body["top_factors"]) >= 1


def test_factors_are_capped_and_unique(client, sample_student_payload):
    body = client.post("/api/v1/predict", json=sample_student_payload).json()
    factors = body["top_factors"]
    assert len(factors) <= 4
    assert len(factors) == len(set(factors))


def test_factors_are_human_readable_not_raw_column_names(client, sample_student_payload):
    body = client.post("/api/v1/predict", json=sample_student_payload).json()
    for factor in body["top_factors"]:
        assert " " in factor, f"looks like a raw feature name: {factor!r}"
        assert not factor.startswith("vle_"), factor
        assert factor[0].isupper(), f"not a sentence: {factor!r}"


def test_factors_never_mention_demographics(client, sample_student_payload):
    import re
    payload = dict(sample_student_payload, region="Scotland", disability="Y", gender="F")
    body = client.post("/api/v1/predict", json=payload).json()
    joined = " ".join(body["top_factors"]).lower()
    for term in ["gender", "female", "male", "region", "scotland", "disability", "age", "deprivation"]:
        assert not re.search(rf"\b{term}\b", joined), f"'{term}' surfaced in {body['top_factors']}"


# ── batch endpoint ────────────────────────────────────────────────────────

def test_batch_matches_single_predictions_exactly(client, sample_student_payload):
    a = dict(sample_student_payload, student_id="B-1")
    b = dict(sample_student_payload, student_id="B-2", vle_total_clicks=700,
             vle_active_days=55, n_submitted=2, avg_early_score=85, pct_on_time=1.0)

    single = [client.post("/api/v1/predict", json=p).json()["risk_score"] for p in (a, b)]
    batch = client.post("/api/v1/predict/batch", json={"students": [a, b]}).json()["predictions"]
    assert [p["risk_score"] for p in batch] == single


def test_batch_preserves_input_order(client, sample_student_payload):
    students = [dict(sample_student_payload, student_id=f"ORDER-{i}") for i in range(6)]
    body = client.post("/api/v1/predict/batch", json={"students": students}).json()
    assert [p["student_id"] for p in body["predictions"]] == [f"ORDER-{i}" for i in range(6)]


def test_batch_handles_a_single_student(client, sample_student_payload):
    body = client.post("/api/v1/predict/batch", json={"students": [sample_student_payload]})
    assert body.status_code == 200
    assert len(body.json()["predictions"]) == 1


def test_batch_with_an_empty_list_returns_no_predictions(client):
    r = client.post("/api/v1/predict/batch", json={"students": []})
    assert r.status_code == 200
    assert r.json()["predictions"] == []


def test_batch_rejects_the_whole_request_if_any_student_is_invalid(client, sample_student_payload):
    broken = dict(sample_student_payload, student_id="BAD", gender="X")
    r = client.post("/api/v1/predict/batch", json={"students": [sample_student_payload, broken]})
    assert r.status_code == 422


def test_batch_scales_to_a_realistic_cohort(client, sample_student_payload):
    students = [dict(sample_student_payload, student_id=f"C-{i}", vle_total_clicks=i * 9)
                for i in range(60)]
    r = client.post("/api/v1/predict/batch", json={"students": students})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 60


# ── validation: every guarded field ───────────────────────────────────────

@pytest.mark.parametrize("field", [
    "student_id", "gender", "region", "highest_education", "imd_band", "age_band",
    "num_of_prev_attempts", "studied_credits", "disability", "date_registration",
    "late_registration", "vle_total_clicks", "vle_active_days", "vle_distinct_sites",
    "vle_click_trend", "vle_days_since_last_click", "n_submitted", "avg_early_score",
    "pct_on_time", "avg_days_early",
])
def test_every_required_field_is_actually_required(client, sample_student_payload, field):
    payload = dict(sample_student_payload)
    del payload[field]
    assert client.post("/api/v1/predict", json=payload).status_code == 422


@pytest.mark.parametrize("field,bad_value", [
    ("gender", "X"),
    ("disability", "maybe"),
    ("num_of_prev_attempts", -1),
    ("studied_credits", 0),
    ("studied_credits", -60),
    ("late_registration", 2),
    ("late_registration", -1),
    ("vle_total_clicks", -5),
    ("vle_active_days", -1),
    ("vle_distinct_sites", -3),
    ("vle_days_since_last_click", -1),
    ("n_submitted", -1),
    ("pct_on_time", 1.5),
    ("pct_on_time", -0.1),
])
def test_out_of_range_values_are_rejected(client, sample_student_payload, field, bad_value):
    payload = dict(sample_student_payload, **{field: bad_value})
    assert client.post("/api/v1/predict", json=payload).status_code == 422, f"{field}={bad_value} accepted"


@pytest.mark.parametrize("field", ["vle_total_clicks", "n_submitted", "studied_credits"])
def test_non_numeric_values_are_rejected(client, sample_student_payload, field):
    payload = dict(sample_student_payload, **{field: "not-a-number"})
    assert client.post("/api/v1/predict", json=payload).status_code == 422


def test_validation_errors_identify_the_offending_field(client, sample_student_payload):
    """A 422 that doesn't say what was wrong is nearly useless to a client dev."""
    payload = dict(sample_student_payload, pct_on_time=9.0)
    detail = client.post("/api/v1/predict", json=payload).json()["detail"]
    assert any("pct_on_time" in str(err.get("loc", "")) for err in detail)


def test_malformed_json_body_is_rejected(client):
    r = client.post("/api/v1/predict", content="{not valid json",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_empty_body_is_rejected(client):
    assert client.post("/api/v1/predict", json={}).status_code == 422


def test_extreme_but_valid_values_are_accepted(client, sample_student_payload):
    """Guards against over-tight validation rejecting real outliers."""
    payload = dict(sample_student_payload, vle_total_clicks=1_000_000,
                   vle_active_days=365, num_of_prev_attempts=10, studied_credits=600)
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 200
    assert 0.0 <= r.json()["risk_score"] <= 1.0


# ── model info / health ───────────────────────────────────────────────────

def test_model_info_exposes_baseline_comparison_metrics(client):
    body = client.get("/api/v1/model/info").json()
    assert set(body) == {
        "model_version", "trained_at", "checkpoint_fraction",
        "n_training_rows", "held_out_metrics", "cv_metrics",
    }
    assert body["n_training_rows"] > 0
    for key in ["recall", "precision", "f2", "roc_auc"]:
        assert 0.0 <= body["held_out_metrics"][key] <= 1.0


def test_model_info_recall_is_high_enough_to_be_useful(client):
    """A recall-first system that catches half its students isn't doing its job.
    This fails loudly if a future retrain regresses badly."""
    recall = client.get("/api/v1/model/info").json()["held_out_metrics"]["recall"]
    assert recall > 0.6, f"held-out recall regressed to {recall}"


def test_health_is_cheap_and_needs_no_body(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


# ── HTTP semantics ────────────────────────────────────────────────────────

def test_unknown_route_returns_404(client):
    assert client.get("/api/v1/does-not-exist").status_code == 404


def test_wrong_method_returns_405(client, sample_student_payload):
    assert client.get("/api/v1/predict").status_code == 405
    assert client.post("/api/v1/health", json={}).status_code == 405


def test_openapi_schema_documents_every_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    for route in ["/api/v1/predict", "/api/v1/predict/batch", "/api/v1/model/info", "/api/v1/health"]:
        assert route in paths, f"{route} missing from the OpenAPI schema"


def test_interactive_docs_are_served(client):
    assert client.get("/docs").status_code == 200
