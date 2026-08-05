# Deployment

Two real options, both included in this repo:

- **Option A — Render + Vercel** (two platforms, two domains, CORS between them). Proven, simple, works today. Documented first below.
- **Option B — everything on Vercel** (one project, one domain, no CORS). Possible since Vercel added native Dockerfile/container support on June 30, 2026 — genuinely new, covered after Option A with the caveats that come with something this recent.

If you don't already have a reason to prefer one, skip to the comparison table at the top of Option B and decide from there.

---

## Before you start

These two steps apply whichever option you pick below — both build a container
from your Git repository rather than running training themselves.

### 1. Commit the trained model and the cohort

This is the step most likely to bite you. The platform builds the image from
your Git repository and **does not run training** — training needs the
dataset, which is deliberately not committed. If
`models/artifacts/model.joblib` is missing from Git, the container starts
fine, answers `/api/v1/health` with `"model_loaded": false`, and returns
**503 on every prediction**.

```bash
cd backend
make data && make train                     # trains, then builds the cohort

git add -f models/artifacts/model.joblib \
           models/artifacts/metadata.json \
           models/artifacts/demo_cohort.json
git commit -m "Add trained model and cohort artifacts for deployment"
```

`-f` is needed because `.gitignore` still lists the pattern. All three files
are required:

| File | Why it's needed | Symptom if missing |
|---|---|---|
| `model.joblib` | The trained model | `/health` reports `model_loaded: false`; every prediction 503s |
| `metadata.json` | Feature column order and the reference medians explanations depend on | Same as above — the API won't load the model without it |
| `demo_cohort.json` | The roster the dashboard reads | `/api/v1/students`, `/overview` and `/courses` return 503; the dashboard shows "Couldn't load the overview" |

Together they are about 590 KB.

Verify before pushing:

```bash
git ls-files models/artifacts/
# must list model.joblib, metadata.json AND demo_cohort.json
```

### 2. Push to GitHub

Both options deploy from a Git remote. Push the whole monorepo (`backend/` and
`frontend/` together) — the platform is told which subdirectory (or, for
Option B, that the repo root) to build from.

---

## Option A, step 1: Backend → Render

### Blueprint (recommended)

`backend/render.yaml` is included. In Render: **New → Blueprint**, pick the
repo, and it reads the file. Edit `CORS_ALLOW_ORIGINS` after the frontend
exists.

### Or set it up by hand

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

## Option A, step 2: Frontend → Vercel

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

## Option A, step 3: Close the loop

Back in Render → your service → **Environment**, set:

```
CORS_ALLOW_ORIGINS = https://your-app.vercel.app
```

Save; Render redeploys automatically. Until this is set, the dashboard loads but
every request fails, and the browser console shows a CORS error rather than
anything useful in the UI.

---

## Option A: free-tier sleeping

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

# Option B: everything on Vercel

Vercel added the ability to deploy a Dockerfile as a genuine long-running
container (not the old 250MB zip-based Python function) on **June 30, 2026**.
Combined with **Services** — multiple apps in one Vercel project on one domain
— this means the backend's existing Dockerfile and the frontend can now both
live on Vercel, with no CORS configuration at all, since they share an origin.

This is covered in full because the user asked for it directly, but be clear
about what it is: **a genuinely new feature, about five weeks old at the time
of writing.** Treat the rest of this section as "this is how it's documented to
work and how the pieces verified locally," not as a platform with years of
production mileage behind it.

## Option A vs. Option B

| | A: Render + Vercel | B: everything on Vercel |
|---|---|---|
| Domains | Two | One |
| CORS | Required, configured | **Not needed** — same origin |
| Platforms to manage | Two | One |
| Backend billing | Free tier (sleeps) or paid, flat | Active CPU (pay for compute time actually used) |
| Maturity | Both platforms' core product | Container support is ~5 weeks old |
| Cold starts | Yes, on Render's free tier | Fluid Compute scales to zero too; behaviour under real traffic is less documented |

Reasonable defaults: **Option A** if this needs to be reliable *today* with the
least uncertainty. **Option B** if a single domain and no CORS wiring is worth
being an early adopter of a very new capability — or if you're doing this partly
to see how it works.

## Option B: what had to be fixed to make this fit

Two real problems came out of actually testing this, not just reading the
Vercel docs — worth knowing regardless of which option you pick.

**1. The runtime dependency image was accidentally 1.2GB.** Measuring it
directly: `xgboost`'s regular PyPI wheel pulls in `nvidia-nccl-cu12` — a
~400MB multi-GPU training library this project's CPU-only inference never
touches. Swapping to the `xgboost-cpu` package (same import name, same API,
verified against this project's actual trained `model.joblib`) drops the
installed footprint to **~620MB** with identical prediction output. This is
already applied in `backend/requirements-api.txt`.

**2. The API failed to start under its own declared runtime dependencies.**
`src/models/explain.py` imported `matplotlib` at module level for a
training-only plotting function, and the API imports that module for every
prediction's explanation. `requirements-api.txt` deliberately excludes
matplotlib (it's training-only) — so the deployed API would have crashed on
startup. This had gone unnoticed because the ambient development environment
happened to already have matplotlib installed, masking it. It's fixed: the
`matplotlib` import is now lazy, inside the one function that needs it, and
`tests/test_deployment.py` has two regression tests — a static check that the
import never returns to module level, and a test that builds a venv from
`requirements-api.txt` alone and confirms the API actually starts and predicts.
This fix benefits Option A too.

## Option B: structure

One Vercel project at the **repository root** (`edu02/`, not `edu02/backend` or
`edu02/frontend`), with a single `vercel.json`:

```json
{
  "services": {
    "frontend": { "root": "frontend/" },
    "backend": { "runtime": "container", "root": "backend/", "entrypoint": "Dockerfile" }
  },
  "rewrites": [
    { "source": "/api/(.*)", "destination": { "service": "backend" } },
    { "source": "/(.*)", "destination": { "service": "frontend" } }
  ]
}
```

This file is already in the repo at `edu02/vercel.json`. Notes on it:

- **`frontend`** has no `runtime` set, so Vercel auto-detects it as a Vite app
  and builds it as a normal static site — no Dockerfile needed for the frontend.
- **`backend`** uses `"runtime": "container"` and points `entrypoint` at the
  **same `backend/Dockerfile`** already written for Render. Nothing was
  duplicated for Vercel; one Dockerfile serves both platforms.
- The **rewrites are what remove CORS from the picture**: `/api/*` is proxied
  to the backend service, everything else to the frontend, both under one
  domain. The browser only ever talks to its own origin.

## Option B: steps

**1. Import the repo as one project**, with **Root Directory left as the repo
root** — do *not* set it to `backend` or `frontend` the way Option A does. Root
must be the directory containing `vercel.json`, or Vercel won't see the
services definition.

**2. Set the backend's port.** The Dockerfile defaults to port 8000
(`ENV PORT=8000`), and its `CMD` binds to whatever `$PORT` is set to. Vercel's
container routing defaults to port 80 unless told otherwise, so in the Vercel
dashboard, set an environment variable scoped to the **backend service**:

```
PORT = 8000
```

**3. Set the frontend's build-time variables**, scoped to the **frontend
service**, for Production, Preview and Development:

```
VITE_API_BASE_URL =
VITE_USE_MOCK = false
```

`VITE_API_BASE_URL` is set to **empty, not omitted** — an unset variable falls
back to `http://localhost:8000` (see `client.ts`), while an explicit empty
string resolves every request as a relative `/api/v1/...` path against
whatever origin served the page. This was checked in a real browser against a
built bundle, watching actual outgoing requests, not just read from the source:
with the variable empty, requests went to `<origin>/api/v1/...`; with it unset,
they went to `localhost:8000`.

**4. Deploy.** Vercel builds both services, wires the rewrites, and serves both
from one domain.

## Option B: verify

```bash
curl https://YOUR-PROJECT.vercel.app/api/v1/health
# {"status":"ok","model_loaded":true}
```

If this 404s instead of reaching the backend, the rewrite isn't matching —
double check Root Directory is the repo root and `vercel.json` deployed with it
(`vercel.json` at the repo root is picked up automatically; it does not need to
be referenced anywhere).

Open `https://YOUR-PROJECT.vercel.app/` and confirm the dashboard loads real
data (not the seeded mock cohort) — that confirms `VITE_USE_MOCK=false` took
and the relative-path request actually reached the backend service.

## Option B: what to watch for, since this is new

- **Billing model is different.** Render/Option A's free tier is flat (sleeps,
  but $0). Option B's backend runs on Fluid Compute's Active CPU pricing —
  billed for compute time actually used, not wall-clock uptime. For a
  low-traffic demo this is likely cheap, but it is a different mental model
  than "free tier," and worth checking current Vercel pricing before relying
  on it.
- **The 250MB/500MB Python function size limits are documented as not applying
  the same way to container images**, which is why this fits at all at ~620MB
  — but this project did not have a live Vercel account to push a real deploy
  against and confirm that boundary empirically. If a deploy fails on image
  size, that's the first thing to check against Vercel's current docs.
- **`vercel dev` can run this locally** (requires the Docker CLI/daemon on your
  machine) if you want to test the combined setup before pushing.
- **Deep-link routing inside the frontend service is the one piece not verified
  end-to-end here.** `frontend/vercel.json` has its own rewrite so refreshing on
  `/students/S-10436` returns `index.html` instead of a 404 — that's confirmed
  working for Option A, where `frontend/` is its own top-level Vercel project.
  Under Option B, `frontend/` becomes a *service* inside the root project
  instead, and whether Vercel still picks up a `vercel.json` sitting inside a
  service's own directory isn't something the available documentation states
  plainly, and there was no live Vercel account here to push a real deploy and
  check. **Test this specifically after deploying**: open the dashboard,
  navigate to a student, and refresh the page. If it 404s, the fix is almost
  certainly moving that rewrite rule into the root `vercel.json`'s `rewrites`
  array, ordered after the `/api/*` rule and before the catch-all.

---

## Troubleshooting

**Option A (Render + Vercel):**

| Symptom | Cause | Fix |
|---|---|---|
| Build fails, "Dockerfile not found" | Root Directory not set | Set it to `backend` |
| `"model_loaded": false`, predictions 503 | Model artifact not in Git | `git add -f models/artifacts/*` and redeploy |
| Dashboard shows "Couldn't load the overview"; logs show `404` or `503` on `/api/v1/overview` and `/api/v1/students` | `demo_cohort.json` not committed, or an older image predating the cohort endpoints | `make cohort`, `git add -f models/artifacts/demo_cohort.json`, redeploy |
| Health check fails, container "unhealthy" | Bound to the wrong port | Do not set `PORT`; let Render inject it |
| Dashboard loads, all data fails; console shows CORS | `CORS_ALLOW_ORIGINS` unset or has a trailing slash | Set the exact origin, no trailing slash |
| Refreshing `/students/S-1` gives 404 | SPA rewrite missing | Ensure `frontend/vercel.json` is committed |
| Frontend still shows demo students | `VITE_USE_MOCK` not `false`, or not rebuilt | Set it and **redeploy** — Vite inlines at build time |
| First load after idle fails | Free-tier cold start | Wait ~50s and retry; upgrade for demos |
| Out-of-memory during build/boot | Free tier is 512 MB | The image already installs `requirements-api.txt` (no matplotlib/pytest); check nothing re-added them |

**Option B (everything on Vercel):**

| Symptom | Cause | Fix |
|---|---|---|
| `vercel.json` seemingly ignored, 404 on everything | Root Directory set to `backend` or `frontend` instead of the repo root | Root Directory must be the directory containing `vercel.json` — the repo root |
| `/api/*` requests 404 or hit the frontend | Rewrite order or service names don't match `vercel.json` | Confirm the `backend`/`frontend` keys under `services` match what the rewrites reference |
| Dashboard loads but shows the seeded demo cohort | `VITE_API_BASE_URL` unset (falls back to `localhost:8000`) rather than explicitly empty | Set it to an **empty string**, not omitted, scoped to the frontend service |
| Backend container won't respond / times out | `PORT` env var not set on the backend service | Set `PORT=8000` on the backend service to match the Dockerfile's default |
| Build fails on image size | Container image over the platform's current limit | Confirm `requirements-api.txt` still uses `xgboost-cpu`, not `xgboost`; check Vercel's current container size limits, since this is a new feature and limits may change |
| `vercel dev` won't start the backend locally | Docker not running | Container services need a local Docker daemon for `vercel dev`; not required for pushing to Vercel itself |

To debug anything else, Render's **Logs** tab (Option A) or Vercel's **Logs**
tab (Option B) shows container stdout. The app logs the loaded model version
and training date at startup, which is the fastest way to confirm the right
artifact shipped.

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
