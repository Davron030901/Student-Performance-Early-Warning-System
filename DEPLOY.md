# Deployment

Backend → **Render** (Docker). Frontend → **Vercel**.

Deploy the backend first: the frontend needs its URL, and the backend needs the
frontend's domain for CORS. That circularity is unavoidable, so the order is
backend → frontend → come back and set one backend variable.

---

## Before you start

### 1. Commit the trained model

This is the step most likely to bite you. Render builds the image from your Git
repository and **does not run training** — training needs the dataset, which is
deliberately not committed. If `models/artifacts/model.joblib` is missing from
Git, the container starts fine, answers `/api/v1/health` with
`"model_loaded": false`, and returns **503 on every prediction**.

```bash
cd backend
make data && make train                     # produces the artifact locally

git add -f models/artifacts/model.joblib models/artifacts/metadata.json
git commit -m "Add trained model artifact for deployment"
```

`-f` is needed because `.gitignore` still lists the pattern. Both files are
required — `metadata.json` carries the feature column order and the reference
medians the explanations depend on. Together they are about 440 KB.

Verify before pushing:

```bash
git ls-files models/artifacts/
# must list BOTH model.joblib and metadata.json
```

### 2. Push to GitHub

Render and Vercel both deploy from a Git remote. Push the whole monorepo
(`backend/` and `frontend/` together) — both platforms are told which
subdirectory to build.

---

## Backend → Render

### Option A — Blueprint (recommended)

`backend/render.yaml` is included. In Render: **New → Blueprint**, pick the
repo, and it reads the file. Edit `CORS_ALLOW_ORIGINS` after the frontend
exists.

### Option B — by hand

**New → Web Service**, connect the repo, then:

| Setting | Value |
|---|---|
| Language / Runtime | **Docker** |
| Root Directory | **`backend`** |
| Dockerfile Path | `./Dockerfile` |
| Instance Type | Free (or Starter — see the note on sleeping below) |
| Health Check Path | **`/api/v1/health`** |
| Region | whichever is closest to your users |

`Root Directory = backend` is not optional. This is a monorepo; without it
Render looks for a Dockerfile at the repository root and the build fails
immediately.

### Environment variables

| Key | Value | Notes |
|---|---|---|
| `CORS_ALLOW_ORIGINS` | `https://your-app.vercel.app` | Set after the frontend deploys. Comma-separated for several domains. **No trailing slash.** |
| `PYTHONUNBUFFERED` | `1` | Logs appear in Render's console immediately instead of being buffered. |

Do **not** set `PORT`. Render injects it, and the Dockerfile's
`CMD ... --port ${PORT:-8000}` binds to whatever Render provides. Hardcoding a
port here is the second most common cause of a service that builds but never
passes its health check.

Vercel *preview* deployments get a fresh URL on every push. Rather than chasing
them, the app allows any `https://*.vercel.app` origin via a regex. To restrict
that, set `CORS_ALLOW_ORIGIN_REGEX` to something narrower, or to `$^` to
disable it entirely and rely only on the explicit list.

### Verify

```bash
curl https://YOUR-SERVICE.onrender.com/api/v1/health
# {"status":"ok","model_loaded":true}
```

`"model_loaded": false` means the artifact never made it into the image — go
back to step 1.

```bash
curl -X POST https://YOUR-SERVICE.onrender.com/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"student_id":"S-1","gender":"M","region":"Wales",
       "highest_education":"A Level or Equivalent","imd_band":"20-30%",
       "age_band":"0-35","num_of_prev_attempts":0,"studied_credits":60,
       "disability":"N","date_registration":5,"late_registration":1,
       "vle_total_clicks":12,"vle_active_days":3,"vle_distinct_sites":2,
       "vle_click_trend":-0.4,"vle_days_since_last_click":25,
       "n_submitted":0,"avg_early_score":-1,"pct_on_time":0,"avg_days_early":0}'
```

Interactive docs are live at `https://YOUR-SERVICE.onrender.com/docs`.

---

## Frontend → Vercel

**Add New → Project**, import the repo, then:

| Setting | Value |
|---|---|
| Framework Preset | **Vite** |
| Root Directory | **`frontend`** |
| Build Command | `npm run build` *(default)* |
| Output Directory | `dist` *(default)* |

### Environment variables

Set both, for **Production, Preview and Development**:

| Key | Value |
|---|---|
| `VITE_API_BASE_URL` | `https://YOUR-SERVICE.onrender.com` |
| `VITE_USE_MOCK` | `false` |

Two things worth knowing about Vite variables:

- Only names starting with `VITE_` reach the browser. `API_BASE_URL` would
  silently be `undefined`.
- They are baked in **at build time**, not read at runtime. Changing one in the
  Vercel dashboard does nothing until you **redeploy**.

Leaving `VITE_USE_MOCK` unset ships the demo cohort instead of live data — which
is a perfectly reasonable way to publish a portfolio version that works even
when the backend is asleep. Just be deliberate about which one you are shipping.

`frontend/vercel.json` is already included and handles the SPA rewrite. Without
it, opening `/students/S-10436` directly or refreshing on it returns a 404,
because Vercel looks for a file at that path and React Router never gets a
chance to resolve it.

---

## Close the loop

Back in Render → your service → **Environment**, set:

```
CORS_ALLOW_ORIGINS = https://your-app.vercel.app
```

Save; Render redeploys automatically. Until this is set, the dashboard loads but
every request fails, and the browser console shows a CORS error rather than
anything useful in the UI.

---

## Free-tier sleeping

Render's free instance **spins down after ~15 minutes of inactivity** and takes
roughly **50 seconds** to wake. The first request after a quiet period will hang
and may look broken.

This is handled rather than ignored: TanStack Query retries three times with
exponential backoff, which comfortably covers a cold start, and the error state
explains what is happening instead of blaming the user's connection.

If a demo needs to be reliably instant — an assessment, an interview — either
upgrade to Render's Starter plan (no sleeping) or deploy the frontend with
`VITE_USE_MOCK=true`, which runs entirely on seeded data with no backend at all.

Note that external ping services to keep a free instance awake are against
Render's terms; upgrading is the honest fix.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Build fails, "Dockerfile not found" | Root Directory not set | Set it to `backend` |
| `"model_loaded": false`, predictions 503 | Artifact not in Git | `git add -f models/artifacts/*` and redeploy |
| Health check fails, container "unhealthy" | Bound to the wrong port | Do not set `PORT`; let Render inject it |
| Dashboard loads, all data fails; console shows CORS | `CORS_ALLOW_ORIGINS` unset or has a trailing slash | Set the exact origin, no trailing slash |
| Refreshing `/students/S-1` gives 404 | SPA rewrite missing | Ensure `frontend/vercel.json` is committed |
| Frontend still shows demo students | `VITE_USE_MOCK` not `false`, or not rebuilt | Set it and **redeploy** — Vite inlines at build time |
| First load after idle fails | Free-tier cold start | Wait ~50s and retry; upgrade for demos |
| Out-of-memory during build/boot | Free tier is 512 MB | The image already installs `requirements-api.txt` (no matplotlib/pytest); check nothing re-added them |

To debug anything else, Render's **Logs** tab shows container stdout. The app
logs the loaded model version and training date at startup, which is the fastest
way to confirm the right artifact shipped.

---

## What is not covered

This deploys a **prototype**, matching the brief's scope. Before anything
resembling real use:

- The API has **no authentication**. It would expose student risk scores to
  anyone with the URL. Real deployment needs auth in front of it.
- There is no rate limiting, no request logging retention policy, and no
  monitoring beyond Render's health check.
- The model is baked into the image, so updating it means a rebuild. That is
  fine at this scale and wrong at any larger one.
- Running this on real student data raises consent, GDPR/FERPA and governance
  questions that are out of scope here — see
  [`backend/LIMITATIONS.md`](backend/LIMITATIONS.md).
