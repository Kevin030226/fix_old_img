# API image (plan §3.13): FastAPI gateway + Gradio UI, no GPU stack.
#
# Deliberately torch-free — the API only validates requests, persists tasks and
# serves results, so the image stays small and starts fast. Set
# FIXIMG_INLINE_WORKER=false and run the worker image alongside it.
#
#   docker build -f docker/api.Dockerfile -t fiximg-api .
#   docker run -p 9502:9502 -e FIXIMG_INLINE_WORKER=false fiximg-api
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FIXIMG_PROFILE=docker \
    FIXIMG_HOST=0.0.0.0 \
    FIXIMG_PORT=9502 \
    FIXIMG_INLINE_WORKER=false \
    FIXIMG_HAS_EXTERNAL_WORKER=true

# Runtime libraries for OpenCV/Pillow only (no build toolchain needed).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY models ./models
COPY config ./config
COPY main.py worker.py run.py ./

# The deployment transports this process can be pointed at, per docker/compose.yaml's
# `platform` profile: PostgreSQL for state, Redis for the queue and rate limiter, an S3
# object store for artifacts. The `test` extra used to be installed here instead, which
# shipped pytest/moto/fakeredis in a published image while leaving the three backends the
# document names with no driver to reach them.
RUN pip install --upgrade pip && pip install -e ".[postgres,redis,s3]"

# The JSON API is bearer-token protected: set FIXIMG_API_TOKEN to pin it,
# otherwise a token is generated on first start and persisted at
# admin_data/api_token.txt (mode 0600) inside the mounted volume.
# Admin credentials likewise come from FIXIMG_ADMIN_PASSWORD, or a random
# one-time password printed once on first boot.

EXPOSE 9502
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:9502/api/v1/health/live || exit 1

CMD ["python", "-m", "fiximg.cli.api"]
