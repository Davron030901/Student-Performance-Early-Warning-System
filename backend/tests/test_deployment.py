"""
Tests for the behaviours that only matter once this is deployed.

These cover the failure modes that are invisible locally but break a real
deployment: a container built without the model artifact, CORS misconfiguration
against the Vercel frontend, and path resolution when the process starts from a
working directory that isn't the project root.
"""
import importlib
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _reload_api(monkeypatch, **env):
    """Re-import the API module with a given environment, since CORS origins and
    paths are resolved at import/startup time."""
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    sys.modules.pop("src.api.main", None)
    return importlib.import_module("src.api.main")


# ── missing model artifact ────────────────────────────────────────────────

def test_service_starts_and_reports_honestly_when_the_model_is_missing(monkeypatch, tmp_path):
    """The most likely deployment failure: Render builds from git, and if the
    artifact was left gitignored the image ships without it. The service must
    start (so the platform's health check can report), say so plainly, and
    refuse to invent predictions."""
    module = _reload_api(monkeypatch)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)  # no artifacts here

    with TestClient(module.app) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200, "must still respond, or the platform sees a dead container"
        assert health.json()["model_loaded"] is False

        payload = {
            "student_id": "S-1", "gender": "M", "region": "Wales",
            "highest_education": "A Level or Equivalent", "imd_band": "20-30%",
            "age_band": "0-35", "num_of_prev_attempts": 0, "studied_credits": 60,
            "disability": "N", "date_registration": 5, "late_registration": 1,
            "vle_total_clicks": 12, "vle_active_days": 3, "vle_distinct_sites": 2,
            "vle_click_trend": -0.4, "vle_days_since_last_click": 25,
            "n_submitted": 0, "avg_early_score": -1, "pct_on_time": 0, "avg_days_early": 0,
        }
        r = client.post("/api/v1/predict", json=payload)
        assert r.status_code == 503
        assert "train" in r.json()["detail"].lower(), "the error should say how to fix it"

        assert client.post("/api/v1/predict/batch", json={"students": [payload]}).status_code == 503
        assert client.get("/api/v1/model/info").status_code == 503

    sys.modules.pop("src.api.main", None)


# ── path resolution ───────────────────────────────────────────────────────

def test_artifacts_resolve_from_the_project_root_not_the_cwd():
    """In Docker the process runs from /app; under uvicorn locally it may run
    from anywhere. Paths built from the current directory would work in one and
    silently fail in the other."""
    module = importlib.import_module("src.api.main")
    assert module.PROJECT_ROOT == PROJECT_ROOT
    assert module.CONFIG_PATH.is_absolute()
    assert module.CONFIG_PATH.exists()


def test_shipped_artifacts_exist_where_the_app_expects_them():
    """If this fails, the deployed container will 503 on every request."""
    import yaml
    config = yaml.safe_load(open(PROJECT_ROOT / "config" / "config.yaml"))
    assert (PROJECT_ROOT / config["paths"]["model_artifact"]).exists()
    assert (PROJECT_ROOT / config["paths"]["metadata_artifact"]).exists()


# ── CORS ──────────────────────────────────────────────────────────────────

PREFLIGHT = {
    "Access-Control-Request-Method": "POST",
    "Access-Control-Request-Headers": "content-type",
}


def test_configured_production_origin_is_allowed(monkeypatch):
    module = _reload_api(monkeypatch, CORS_ALLOW_ORIGINS="https://course-signals.vercel.app")
    with TestClient(module.app) as client:
        r = client.options("/api/v1/predict",
                           headers={"Origin": "https://course-signals.vercel.app", **PREFLIGHT})
        assert r.headers.get("access-control-allow-origin") == "https://course-signals.vercel.app"
    sys.modules.pop("src.api.main", None)


def test_multiple_comma_separated_origins_are_supported(monkeypatch):
    module = _reload_api(
        monkeypatch,
        CORS_ALLOW_ORIGINS="https://a.example.edu, https://b.example.edu",
    )
    with TestClient(module.app) as client:
        for origin in ["https://a.example.edu", "https://b.example.edu"]:
            r = client.options("/api/v1/predict", headers={"Origin": origin, **PREFLIGHT})
            assert r.headers.get("access-control-allow-origin") == origin, origin
    sys.modules.pop("src.api.main", None)


def test_vercel_preview_deployments_are_allowed_by_regex(monkeypatch):
    """Vercel mints a new URL per push; listing them individually is not viable."""
    module = _reload_api(monkeypatch, CORS_ALLOW_ORIGINS="https://prod.vercel.app")
    with TestClient(module.app) as client:
        preview = "https://course-signals-git-feature-branch-team.vercel.app"
        r = client.options("/api/v1/predict", headers={"Origin": preview, **PREFLIGHT})
        assert r.headers.get("access-control-allow-origin") == preview
    sys.modules.pop("src.api.main", None)


def test_unrelated_origins_are_not_allowed(monkeypatch):
    module = _reload_api(monkeypatch, CORS_ALLOW_ORIGINS="https://course-signals.vercel.app")
    with TestClient(module.app) as client:
        for origin in ["https://evil.example.com", "http://localhost:9999",
                       "https://vercel.app.evil.com"]:
            r = client.options("/api/v1/predict", headers={"Origin": origin, **PREFLIGHT})
            assert r.headers.get("access-control-allow-origin") is None, f"{origin} was allowed"
    sys.modules.pop("src.api.main", None)


def test_local_development_works_with_no_cors_env_set(monkeypatch):
    module = _reload_api(monkeypatch, CORS_ALLOW_ORIGINS=None)
    with TestClient(module.app) as client:
        r = client.options("/api/v1/predict",
                           headers={"Origin": "http://localhost:5173", **PREFLIGHT})
        assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    sys.modules.pop("src.api.main", None)


def test_credentials_are_not_enabled(monkeypatch):
    """allow_credentials combined with a permissive origin regex is the
    combination that turns CORS into a real vulnerability. This API has no auth
    or cookies, so credentials must stay off."""
    module = _reload_api(monkeypatch, CORS_ALLOW_ORIGINS="https://course-signals.vercel.app")
    with TestClient(module.app) as client:
        r = client.options("/api/v1/predict",
                           headers={"Origin": "https://course-signals.vercel.app", **PREFLIGHT})
        assert r.headers.get("access-control-allow-credentials") != "true"
    sys.modules.pop("src.api.main", None)


def test_origin_regex_can_be_overridden(monkeypatch):
    module = _reload_api(
        monkeypatch,
        CORS_ALLOW_ORIGINS="https://only-this.example.edu",
        CORS_ALLOW_ORIGIN_REGEX=r"$^",  # matches nothing
    )
    with TestClient(module.app) as client:
        r = client.options("/api/v1/predict",
                           headers={"Origin": "https://anything.vercel.app", **PREFLIGHT})
        assert r.headers.get("access-control-allow-origin") is None
    sys.modules.pop("src.api.main", None)


# ── deployment config files ───────────────────────────────────────────────

def test_dockerfile_binds_to_the_platform_provided_port():
    """Render injects $PORT and fails the health check if the app ignores it."""
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()
    assert "${PORT" in dockerfile, "Dockerfile must bind to $PORT, not a fixed port"
    assert "--host 0.0.0.0" in dockerfile


def test_dockerfile_ships_the_model_and_runs_unprivileged():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()
    assert "COPY models/" in dockerfile, "the image must contain the trained model"
    assert "USER appuser" in dockerfile, "should not run as root"
    assert "libgomp1" in dockerfile, "XGBoost needs the OpenMP runtime on slim images"


def test_runtime_requirements_exclude_training_only_packages():
    """Render's free tier is 512MB; matplotlib and pytest are dead weight in the
    serving image."""
    lines = (PROJECT_ROOT / "requirements-api.txt").read_text().splitlines()
    # only real requirement lines — comments explain *why* things are absent and
    # naturally mention the excluded package names
    packages = {
        line.split("==")[0].split("[")[0].strip().lower()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    }
    for package in ["matplotlib", "pytest"]:
        assert package not in packages, f"{package} should not be in the runtime image"
    for package in ["fastapi", "uvicorn", "xgboost-cpu", "shap", "scikit-learn", "joblib"]:
        assert package in packages, f"{package} is required at runtime"


def test_runtime_requirements_use_the_gpu_free_xgboost_build():
    """xgboost's regular PyPI wheel pulls in nvidia-nccl-cu12 (~400MB) for
    multi-GPU training this project never does. xgboost-cpu is a verified
    drop-in for serving an already-trained model: same import name, same API,
    loads this project's model.joblib unchanged, and cuts the deployed image
    roughly in half. Using the GPU build here would be ~400MB of dead weight in
    every deploy."""
    lines = (PROJECT_ROOT / "requirements-api.txt").read_text().splitlines()
    packages = {
        line.split("==")[0].split("[")[0].strip().lower()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    }
    assert "xgboost-cpu" in packages
    assert "xgboost" not in packages, "the GPU-capable build should not be in the runtime image"


def test_render_blueprint_points_at_the_backend_subdirectory():
    """This is a monorepo; without rootDir the Docker build fails immediately."""
    import yaml
    blueprint = yaml.safe_load((PROJECT_ROOT / "render.yaml").read_text())
    service = blueprint["services"][0]
    assert service["rootDir"] == "backend"
    assert service["runtime"] == "docker"
    assert service["healthCheckPath"] == "/api/v1/health"


def test_vercel_config_rewrites_all_routes_to_the_spa():
    """Without this, refreshing on /students/S-1 returns a 404 from Vercel."""
    vercel = json.loads((PROJECT_ROOT.parent / "frontend" / "vercel.json").read_text())
    assert vercel["rewrites"][0]["destination"] == "/index.html"
    assert vercel["outputDirectory"] == "dist"


# ── the bug this file exists to catch ─────────────────────────────────────

def test_api_module_does_not_import_matplotlib_at_module_level():
    """Regression test for a real bug: explain.py imported matplotlib at
    module level for a training-only plotting function. src/api/main.py
    imports explain.py for top_factors_for_student, which runs on every
    prediction — so a top-level matplotlib import there meant the API failed
    at startup in the actual deployment environment (requirements-api.txt,
    which excludes matplotlib on purpose). It only appeared to work in local
    testing because matplotlib happened to already be installed in the
    ambient dev environment, which does not reflect what actually deploys.

    This test parses the source rather than just importing the module, so it
    still catches a regression even if some other installed package happens
    to pull matplotlib in as an indirect dependency in the test environment."""
    import ast
    tree = ast.parse((PROJECT_ROOT / "src" / "models" / "explain.py").read_text())
    top_level_imports = [
        n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    for node in top_level_imports:
        names = [alias.name for alias in node.names]
        if isinstance(node, ast.Import):
            assert not any(n == "matplotlib" or n.startswith("matplotlib.") for n in names), (
                "matplotlib must not be a module-level import in explain.py — "
                "it belongs inside save_global_importance_plot() only"
            )
        else:
            assert node.module != "matplotlib" and not (node.module or "").startswith("matplotlib."), (
                "matplotlib must not be a module-level import in explain.py"
            )


@pytest.mark.slow
def test_api_starts_and_predicts_with_only_the_runtime_requirements_installed(tmp_path):
    """The test that actually would have caught the bug above: build a venv
    from requirements-api.txt alone — no dev environment, no matplotlib, no
    pytest — and confirm the API starts and serves a real prediction. This is
    the only test in the suite that reflects what the deployed container
    actually has installed."""
    import subprocess
    import sys
    import time
    import json as jsonlib
    import urllib.request

    venv_dir = tmp_path / "runtime_venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True, capture_output=True)
    pip = venv_dir / "bin" / "pip"
    python = venv_dir / "bin" / "python3"

    install = subprocess.run(
        [str(pip), "install", "--quiet", "-r", "requirements-api.txt"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert install.returncode == 0, f"pip install failed:\n{install.stderr}"

    # confirm training-only packages are genuinely absent, not just unlisted
    check = subprocess.run(
        [str(python), "-c", "import matplotlib"], capture_output=True, text=True
    )
    assert check.returncode != 0, "matplotlib should not be importable in the runtime venv"

    proc = subprocess.Popen(
        [str(python), "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8123"],
        cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        health = None
        for _ in range(30):
            time.sleep(1)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8123/api/v1/health", timeout=2) as r:
                    health = jsonlib.loads(r.read())
                    break
            except Exception:
                continue

        assert health is not None, "API never became healthy — see startup log below:\n" + (proc.stdout.read() if proc.stdout else "")
        assert health["model_loaded"] is True

        payload = jsonlib.dumps({
            "student_id": "RUNTIME-VENV-1", "gender": "M", "region": "Wales",
            "highest_education": "A Level or Equivalent", "imd_band": "20-30%",
            "age_band": "0-35", "num_of_prev_attempts": 0, "studied_credits": 60,
            "disability": "N", "date_registration": 5, "late_registration": 1,
            "vle_total_clicks": 12, "vle_active_days": 3, "vle_distinct_sites": 2,
            "vle_click_trend": -0.4, "vle_days_since_last_click": 25,
            "n_submitted": 0, "avg_early_score": -1, "pct_on_time": 0, "avg_days_early": 0,
        }).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8123/api/v1/predict", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            body = jsonlib.loads(r.read())
        assert 0.0 <= body["risk_score"] <= 1.0
        assert body["risk_band"] in {"Low", "Medium", "High"}
        assert body["top_factors"]
    finally:
        proc.terminate()
        proc.wait(timeout=10)


# ── Option B: combined Vercel deployment ──────────────────────────────────

def test_root_vercel_json_declares_both_services_correctly():
    import json
    config = json.loads((PROJECT_ROOT.parent / "vercel.json").read_text())
    assert config["services"]["frontend"]["root"] == "frontend/"
    backend = config["services"]["backend"]
    assert backend["runtime"] == "container"
    assert backend["root"] == "backend/"
    # must point at the SAME Dockerfile used for Render — no duplication
    assert backend["entrypoint"] == "Dockerfile"
    assert (PROJECT_ROOT / backend["entrypoint"]).exists()


def test_root_vercel_json_routes_api_paths_to_the_backend_service():
    import json
    config = json.loads((PROJECT_ROOT.parent / "vercel.json").read_text())
    rewrites = config["rewrites"]
    api_rule = next(r for r in rewrites if r["source"].startswith("/api"))
    assert api_rule["destination"]["service"] == "backend"
    # the catch-all must come after the /api rule, or /api never matches
    assert rewrites.index(api_rule) < len(rewrites) - 1 or len(rewrites) == 1
    catch_all = rewrites[-1]
    assert catch_all["destination"]["service"] == "frontend"


def test_frontend_env_example_documents_the_combined_deployment_case():
    """The empty-string vs. unset distinction is easy to get wrong and silent
    when wrong (falls back to localhost:8000 instead of erroring), so the
    example file must explain it, not just declare the variable."""
    env_example = (PROJECT_ROOT.parent / "frontend" / ".env.example").read_text()
    assert "VITE_API_BASE_URL" in env_example
    assert "empty" in env_example.lower(), "must explain the empty-string behaviour, not just declare the key"
