# Old photo restoration system — all-in-one image (V3)
# Python 3.11 + PyTorch 2.7.1 cu128 (RTX 50 / sm_120) + Gradio 6 + FastAPI
#
# Build: docker build -t fixoldimg .
# Run:   docker run --gpus all -p 9502:9502 fixoldimg
#
# For a split API/worker deployment use docker/compose.yaml instead; this
# Dockerfile keeps the single-container topology working.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FIXIMG_PROFILE=docker \
    FIXIMG_HOST=0.0.0.0 \
    FIXIMG_PORT=9502

# System dependencies: Python 3.11 + dlib source build toolchain + OpenCV runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip python3-dev \
    git curl unzip bzip2 build-essential cmake ninja-build \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Python virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PyTorch cu128 (official wheel, already includes Blackwell sm_120 support)
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Project dependencies (dlib builds from source on Linux, ~5-10 minutes)
# setuptools and wheel are upgraded first: they are the image's own install
# tooling, and the base image's 2022-era copies carry CVE-2025-47273 and
# CVE-2026-24049. The same reasoning, and the same advisories, as in
# docker/api.Dockerfile, where Trivy first reported them.
RUN pip install --upgrade "setuptools>=78.1.1" "wheel>=0.46.2" && pip install -r requirements.txt

# Make the `fiximg` package importable so the `python -m fiximg.*` entry points
# (weight download, verification, migration) resolve inside the image.
RUN pip install -e . --no-deps

# Weights are not baked in by default: THIRD_PARTY_NOTICES.md records this
# distribution as "code only", two of the six artifacts are licensed for
# research/non-commercial use, and the upstream restoration host did not resolve
# from GitHub's runners on 2026-09-29 -- so a build that requires it validates
# nothing. A weights-free container still serves: the registry reports each model
# unavailable through the readiness probe. Fill it with
#   docker compose exec <service> python -m fiximg.cli.download_weights download
# or build with --build-arg FIXIMG_BAKE_WEIGHTS=true where you are entitled to them.
ARG FIXIMG_BAKE_WEIGHTS=false
RUN if [ "$FIXIMG_BAKE_WEIGHTS" = "true" ]; then \
        python3 -m fiximg.cli.download_weights download \
        && python3 -m fiximg.cli.download_weights generate \
        && python3 -m fiximg.cli.verify_weights; \
    else \
        echo "[build] weights not baked -- see the ARG above"; \
    fi

# Admin credentials are NOT baked into the image (plan §21): on first start the
# app reads FIXIMG_ADMIN_PASSWORD when provided, otherwise it generates a random
# one-time password, prints it once in the container log and stores it at
# admin_data/initial_admin_password.txt (mode 0600). Change it immediately.

# The JSON API (/api/v1/*) is bearer-token protected: set FIXIMG_API_TOKEN to
# pin the token at deploy time, otherwise the app generates one on first start,
# prints it once and stores it at admin_data/api_token.txt (mode 0600).

# The weight sources inside the container may differ from local ones; the conditional
# step above regenerates the manifest from the actual files and verifies it.

EXPOSE 9502
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:9502/api/v1/health/live || exit 1
CMD ["python3", "main.py"]

