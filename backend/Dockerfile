# Production image for the EDU-02 early-warning API.
# Targets Render, but is a plain container and runs anywhere.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# libgomp1 is required by XGBoost's OpenMP runtime and is not in the slim image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Runtime-only deps (see requirements-api.txt for why this differs from
# requirements.txt). Copied first so this layer caches across code changes.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY config/ ./config/
COPY src/ ./src/
# The trained model ships inside the image — the container never trains.
COPY models/ ./models/

# Don't run as root.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Render injects its own $PORT and expects the process to bind to it, so the
# port cannot be hardcoded here. Shell form is required for $PORT to expand.
# Single worker: the model is held in memory per worker, and Render's free
# instance does not have the RAM for more than one.
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
