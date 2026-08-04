# EDU-02 — Student Performance Early-Warning Model (Backend)

An ML service that flags students at risk of failing or withdrawing from a course
**early enough for staff to intervene**, using only information that would
genuinely be available at that point in the term.

Companion frontend: [`../frontend`](../frontend) — the advisor dashboard that
consumes this API.

---

## Quickstart

```bash
cd backend
pip install -r requirements.txt

make data     # generate the OULAD-schema dataset into data/raw/
make train    # build features, train baselines + model, write reports/ + artifact
make test     # 235 tests (+2 opt-in slow); see Testing below
make api      # serve on http://localhost:8000
```

Then:

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "S-9001", "gender": "M", "region": "South Region",
    "highest_education": "Lower Than A Level", "imd_band": "20-30%", "age_band": "0-35",
    "num_of_prev_attempts": 0, "studied_credits": 60, "disability": "N",
    "date_registration": 5, "late_registration": 1,
    "vle_total_clicks": 12, "vle_active_days": 3, "vle_distinct_sites": 2,
    "vle_click_trend": -0.4, "vle_days_since_last_click": 25,
    "n_submitted": 0, "avg_early_score": -1, "pct_on_time": 0, "avg_days_early": 0
  }'
```

```json
{
  "student_id": "S-9001",
  "risk_score": 0.9859,
  "risk_band": "High",
  "checkpoint_used": "30% of course length",
  "top_factors": [
    "Low engagement with course materials so far",
    "Few active days on the course site so far",
    "Interacting with very few course resources",
    "Early assessment scores are below average"
  ],
  "model_version": "xgb-v1.0"
}
```

Interactive API docs: <http://localhost:8000/docs>

With Docker: `docker compose up --build`

---

## The problem, and the decision that defines it

Staff can see attendance, LMS activity and early quiz results while a course is
running, but intervention usually happens after final grades are already in. The
useful question is therefore not "can we predict final outcomes" — that is easy
and useless — but **"how much can we predict from only the first few weeks?"**

Everything in this project follows from that:

| Decision | Choice | Why |
|---|---|---|
| Target | Binary `at_risk` = final result is Fail or Withdrawn | Actionable. A single flag maps to a single decision: reach out or don't. |
| Prediction point | **30% of course length**, configurable | Late enough that early assessments exist, early enough that ~70% of the course remains to intervene in. |
| Output | Calibrated probability → Low / Medium / High band | Advisors need triage, not a decimal. The probability is kept for anyone who wants it. |
| Primary metric | **Recall / F2** on the at-risk class | An unnecessary check-in costs one conversation. A missed student can cost them the course. The asymmetry is real, so the metric reflects it. |

## Results

Measured on a **held-out module presentation** (`2014J`) — the model never saw
that cohort during training, which mirrors real deployment on a new term.

| Model | Recall | Precision | F2 | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| Trivial rule (zero submissions → at risk) | 0.620 | 0.407 | 0.561 | 0.575 | 0.383 | 0.439 |
| Logistic regression baseline *(CV)* | 0.889 | 0.672 | 0.834 | 0.918 | 0.850 | 0.123 |
| Random forest *(CV)* | 0.815 | 0.716 | 0.793 | 0.915 | 0.843 | 0.115 |
| **XGBoost (selected)** | **0.781** | **0.714** | **0.767** | **0.911** | **0.827** | **0.118** |

Confusion matrix, held-out set (n=560, 34.3% at risk): `[[308, 60], [42, 150]]`
→ 150 struggling students caught, 42 missed, 60 flagged unnecessarily.

**A note on model selection.** The logistic regression baseline posts a higher
cross-validated recall than XGBoost. XGBoost was still selected because it holds
recall while running ~4 points better on precision and materially better on
calibration — and calibration matters here, since the probability is bucketed
into bands that advisors will read as meaning something. On the real OULAD data
this comparison should be re-run before assuming the same answer; if the linear
model still wins on F2, it is the better choice and should be used.

### Is it early enough to matter?

| Checkpoint | Recall | Precision | F2 | ROC-AUC |
|---|---|---|---|---|
| 20% of course | 0.786 | 0.686 | 0.764 | 0.900 |
| **30% of course** | 0.781 | 0.714 | 0.767 | 0.911 |
| 50% of course | 0.813 | 0.706 | 0.789 | 0.911 |

Waiting until halfway through the course buys about 3 points of recall. Given
that a student flagged at 30% has roughly twice as much course left to recover
in, 30% is the better operating point — and moving it is a one-line config change,
not a code change.

Reports and plots land in `reports/`: `calibration.png`,
`shap_global_importance.png`, `checkpoint_comparison.csv`, `fairness_report.csv`.

> The dataset shipped in this repo is synthetic but schema-identical to OULAD, for
> reasons explained in [DATASET.md](DATASET.md). These numbers are real
> measurements of this pipeline, and are evidence that it works — not a forecast
> of real-world accuracy. Drop in the real CSVs and re-run to get quotable figures.

## How leakage is prevented

This is the part of the brief that is easiest to get quietly wrong, so it is
enforced by structure and tested rather than asserted in prose:

- `src/data/cutoff.py::apply_checkpoint_cutoff()` is the **only** route event data
  takes into feature engineering, and both training and inference go through it.
- Assessments are filtered by **due date**, not submission date — submitting early
  must not reveal a score the model wouldn't have at inference time.
- `final_result`, `date_unregistration`, raw `score` and `date_submitted` are on a
  deny-list asserted in `src/data/pipeline.py`, which raises before training runs.
- `tests/test_leakage.py` checks all of it against the raw CSVs, including an
  independent recomputation that would catch a bug inside the cutoff function.

Full detail, plus the fairness findings, in [LIMITATIONS.md](LIMITATIONS.md).

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/predict` | Score one student |
| `POST /api/v1/predict/batch` | Score a cohort |
| `GET /api/v1/model/info` | Version, training date, headline metrics |
| `GET /api/v1/health` | Liveness + whether the model artifact loaded |

The model is loaded once at startup via a lifespan handler, not per request.
Malformed input returns `422` with a field-level explanation from Pydantic.

## Layout

```
backend/
├── config/config.yaml          # checkpoint %, thresholds, hyperparameters — no magic numbers in code
├── src/
│   ├── data/
│   │   ├── generate_synthetic_oulad.py   # stand-in data with the real schema
│   │   ├── ingest.py                     # CSV loading
│   │   ├── cutoff.py                     # ← the leakage-prevention mechanism
│   │   ├── features.py                   # feature engineering
│   │   └── pipeline.py                   # raw tables → (X, y), parameterised by checkpoint
│   ├── models/
│   │   ├── encoding.py                   # shared by train + inference so they cannot drift
│   │   ├── train.py                      # baselines, model, evaluation, artifacts
│   │   ├── explain.py                    # SHAP → plain-language factors
│   │   └── fairness.py                   # per-group recall/precision
│   └── api/                              # FastAPI service
├── tests/                                # 235 tests: leakage, features, models, API, deploy
├── reports/                              # generated metrics and plots
└── models/artifacts/                     # model.joblib + metadata.json
```

## Testing

```bash
make test                  # 235 tests, ~2 min
pytest -m slow             # + 2 full-training integration tests, ~1 min
pytest --cov=src           # coverage report
```

| Area | What it covers |
|---|---|
| `test_leakage.py` | The checkpoint cutoff, verified against the raw CSVs including an independent recomputation |
| `test_features.py` | Feature engineering, sentinels, checkpoint monotonicity |
| `test_train.py` | Metrics, baselines, risk banding, CV determinism, held-out split integrity |
| `test_explain.py` | Every phrase branch, and the contradiction class of bug in both directions |
| `test_fairness_and_encoding.py` | Group metrics, and the train/inference column-alignment contract |
| `test_data_generation.py` | OULAD schema fidelity, referential integrity, that the data carries real signal |
| `test_api.py` | All four endpoints: contracts, determinism, monotonicity, batch, every validation rule |
| `test_deployment.py` | 503 when the artifact is missing, CORS allow/deny, Docker and Render/Vercel config |
| `test_shap_and_errors.py` | SHAP additivity against the real model, checkpoint sweep, internal error handling |

Coverage is 85%. The uncovered remainder is almost entirely `train.main()`, the
orchestration script, which the opt-in `-m slow` tests exercise end to end.

Two properties are worth calling out because they are what stop this from being
theatre:

- **SHAP additivity** — explanations are checked to reconstruct the model's raw
  output, so they are faithful rather than decorative.
- **Monotonicity** — piling on good signals must never raise a student's risk,
  asserted through the live API.

## Reproducing

`make data && make train` regenerates everything from scratch. The random seed is
fixed in `config/config.yaml` (`model.random_state: 42`), and the synthetic data
generator is seeded, so results are deterministic.

To move the checkpoint, edit `prediction.checkpoint_fraction` in
`config/config.yaml` and re-run `make train` — no code changes.
