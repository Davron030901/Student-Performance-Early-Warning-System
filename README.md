# EDU-02 — Student Performance Early-Warning System

A capstone implementation of the EDU-02 brief: identify students who may need
academic support **early enough in a course for staff to do something about it**,
using only information that would genuinely be available at that point.

```
edu02/
├── backend/       ML pipeline + FastAPI service   (Python)
├── frontend/      Advisor dashboard               (React + TypeScript)
├── DEPLOY.md      Render+Vercel, or Vercel alone
└── vercel.json    Only used for the all-on-Vercel option — see DEPLOY.md
```

## Run it

```bash
# 1. Model + API
cd backend
pip install -r requirements.txt
make data && make train      # generate data, train, write reports + artifact
make api                     # → http://localhost:8000

# 2. Dashboard (new terminal)
cd frontend
npm install && npm run dev   # → http://localhost:5173
```

The dashboard runs on seeded mock data out of the box, so it can be reviewed
without the backend running. See `frontend/README.md` to point it at the live API.

## What was decided, and why

The brief leaves the important choices open. These are the ones that shaped
everything else:

| Question | Answer |
|---|---|
| Predict what? | Binary `at_risk` — final result is Fail or Withdrawn. One flag, one decision: reach out or don't. |
| Predict when? | **30% of the way through the course**, set in config and changeable without touching code. Late enough that early assessments exist, early enough that most of the term remains. |
| Which error is worse? | Missing a struggling student. A needless check-in costs one conversation; a missed student can cost them the course. So **recall/F2** is the primary metric, with precision as a guardrail. |
| What do staff see? | A risk band and, more importantly, *why* — in plain language, from SHAP, restricted to behavioural signals. |

## Results

Measured on a **held-out course presentation** the model never trained on:

| | Recall | Precision | F2 | ROC-AUC |
|---|---|---|---|---|
| Trivial rule (no submissions) | 0.620 | 0.407 | 0.561 | 0.575 |
| **XGBoost (selected)** | **0.781** | **0.714** | **0.767** | **0.911** |

150 struggling students caught, 42 missed, 60 flagged unnecessarily (n=560).

Predicting at 20% instead of 50% of the course costs about 3 points of recall —
which is what makes an early checkpoint defensible. Full tables, the
checkpoint comparison and the model-selection reasoning are in
[`backend/README.md`](backend/README.md).

## Leakage prevention

The requirement that is easiest to fail quietly, so it is enforced structurally
and tested rather than asserted:

- One cutoff function that both training and inference must pass through.
- Assessments filtered by **due date**, not submission date.
- `final_result` and `date_unregistration` on a deny-list that raises before
  training can start.
- Five tests that check it against the raw data, including an independent
  recomputation that would catch a bug inside the cutoff function itself.

## Tests

```bash
cd backend  && make test     # 299 tests (+4 opt-in), 85% coverage
cd frontend && npm test      # 44 tests
```

343 tests in total. The ones that carry weight: the checkpoint cutoff is
verified against the raw data with an independent recomputation; SHAP
explanations are checked to reconstruct the model's own output, so they are
faithful rather than decorative; more engagement is asserted never to raise a
student's risk, through the live API; and the deployment tests confirm the
service returns a clear 503 — rather than inventing predictions — if the
container ever ships without the model artifact.

## Honest scope

- The shipped dataset is **synthetic but schema-identical to OULAD**, because the
  development environment could not reach the dataset hosts. The metrics are real
  measurements of this pipeline and prove it works — they are not a forecast of
  real-world accuracy. [`backend/DATASET.md`](backend/DATASET.md) explains the
  swap, which is a file copy.
- The fairness audit found a **24-point recall gap across regions**. It is
  documented, not solved. [`backend/LIMITATIONS.md`](backend/LIMITATIONS.md).
- Two roster endpoints the dashboard wants don't exist in the backend yet; the
  frontend runs on mock data behind an identical interface. Noted in
  `frontend/README.md`.

## Documentation

| | |
|---|---|
| [`backend/README.md`](backend/README.md) | Setup, results, API, reproduction |
| [`backend/DATASET.md`](backend/DATASET.md) | Dataset, licence, synthetic stand-in |
| [`backend/LIMITATIONS.md`](backend/LIMITATIONS.md) | Leakage controls, fairness findings, risks |
| [`frontend/README.md`](frontend/README.md) | Setup, screens, connecting the API |
| [`frontend/DESIGN.md`](frontend/DESIGN.md) | Palette, type, the ribbon, language |
| [`DEPLOY.md`](DEPLOY.md) | Deploying to Render + Vercel, or everything on Vercel alone |
| [`EDU-02-Completed-Brief.docx`](EDU-02-Completed-Brief.docx) | The client brief with Sections 5, 6 and 11 completed |
