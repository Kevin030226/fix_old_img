# Old photo restoration system — all-in-one image (V3)
# Python 3.11 + PyTorch 2.7.1 cu128 (RTX 50 / sm_120) + FastAPI + native web UI
#
# Build: docker build -t fixoldimg .
# Run:   docker run --gpus all -p 9502:9502 fixoldimg
#
# For a split API/worker deployment use docker/compose.yaml instead; this
# Dockerfile keeps the single-container topology working.
# Base image: the official python:3.11 image, NOT the CUDA image's apt packages.
#
# The CUDA image was the original base, and it made this file unbuildable: on
# Ubuntu 22.04 the only `python3.11` apt publishes is `3.11.0~rc1` (verified:
# `apt-cache policy python3.11` -> Candidate: 3.11.0~rc1-1~22.04.1). A
# pre-release interpreter does not satisfy `>=3.11` under PEP 440
# (`Version('3.11.0rc1') >= Version('3.11')` is False), so every wheel that
# declares a 3.11 floor was rejected -- `numpy==2.4.6` first, which
# requirements.txt pins. The image could not be built at all.
#
# The CUDA userspace is still required (RTX 50 / sm_120 inference), so it is
# copied in from the same nvidia/cuda image this file used to start from:
# the runtime libraries and headers only, not that image's Python. What the
# GPU actually needs at run time is the driver injected by the container
# runtime plus these userspace libraries; the wheel from the cu128 index
# carries its own CUDA runtime, and torch.cuda.is_available() was measured
# True in this image with --gpus all.
#
# Both images are pinned by tag, and the python one is pinned to a minor
# version so a rebuild cannot silently move to a new interpreter.
# The base images are ARGs so a build behind a pull-through mirror or an
# air-gapped registry can name its own copy without editing this file (the
# repo already does this in docker/local.Dockerfile). Defaults are the
# official names, which is what CI and any networked build resolves.
#
#   docker build --build-arg PYTHON_BASE_IMAGE=<mirror>/library/python:3.11-slim-bookworm \
#                --build-arg CUDA_BASE_IMAGE=<mirror>/nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 .
ARG PYTHON_BASE_IMAGE=python:3.11-slim-bookworm
ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

FROM ${PYTHON_BASE_IMAGE} AS python-base
FROM ${CUDA_BASE_IMAGE} AS cuda-runtime

FROM python-base

# CUDA userspace (libcudart, cuBLAS, cuDNN, ...) plus the driver stubs.
# `/usr/local/cuda` is what the cu128 wheels probe; the ldconfig entry makes
# the sonames resolvable without LD_LIBRARY_PATH guesswork.
COPY --from=cuda-runtime /usr/local/cuda /usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf && ldconfig

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FIXIMG_PROFILE=docker \
    FIXIMG_HOST=0.0.0.0 \
    FIXIMG_PORT=9502 \
    FIXIMG_PORT=9502

# No `python3.11` in this list on purpose: the interpreter comes from the base
# image (stable 3.11, at /usr/local/bin). Installing Ubuntu's copy alongside it
# would put a second, pre-release interpreter on the box and whichever won the
# PATH race would decide whether numpy installs. Only the build toolchain and
# the shared libraries the imaging stack dlopen()s are installed here:
#
#   build-essential/cmake/ninja  dlib is compiled from source (no cp311
#                                manylinux wheel is published)
#   libgl1/libglib2.0-0          OpenCV's runtime dependencies
#   libgomp1                     torch's OpenMP runtime
#   curl                         the HEALTHCHECK below
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl unzip bzip2 build-essential cmake ninja-build \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# No virtualenv: this base image's interpreter is already a clean, isolated
# install at /usr/local, which is what the venv existed to provide. Dropping it
# keeps one interpreter path in the image instead of two (docker/api.Dockerfile
# installs the same way), so `python3` in CMD and `pip` in the build steps
# cannot disagree about which interpreter they mean.

# PyTorch cu128 (official wheel, already includes Blackwell sm_120 support)
RUN python3 -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Project dependencies (dlib builds from source on Linux, ~5-10 minutes)
# setuptools and wheel are upgraded first: they are the image's own install
# tooling, and the base image's 2022-era copies carry CVE-2025-47273 and
# CVE-2026-24049. The same reasoning, and the same advisories, as in
# docker/api.Dockerfile, where Trivy first reported them.
# Two installs, on purpose:
#
#  1. `requirements.txt` pins the web + imaging stack this image serves with;
#  2. the `postgres,redis,s3` extras add the transport drivers.
#
# The second one is not optional. `docker-compose.yml` documents a two-service
# topology and tells the operator they may select `FIXIMG_QUEUE_BACKEND=redis`
# with `FIXIMG_REDIS_URL`, or PostgreSQL through `FIXIMG_DATABASE_URL`, or S3
# through `FIXIMG_STORAGE_BACKEND=s3` — and this image is what that file builds.
# Without the extras every one of those choices failed on the first import
# (`connection.py` raising "psycopg is required", `queue/redis.py` needing the
# client), so the image could serve only the SQLite/local default while the
# documentation offered four backends. The extras are the runtime half of the
# same declaration `docker/api.Dockerfile` installs; they are deliberately NOT
# `gpu` (that is the worker image) and NOT `test`/`dev`/`benchmark`.
RUN python3 -m pip install --upgrade "setuptools>=78.1.1" "wheel>=0.46.2" "msgpack>=1.2.1" \
    && python3 -m pip install -r requirements.txt \
    && python3 -m pip install -e ".[postgres,redis,s3]"

# `pip install -e ".[...]"` above already makes the `fiximg` package importable
# (the extras pull the project itself in as an editable install), so no separate
# `pip install -e . --no-deps` step is needed; `--no-deps` would in fact undo the
# extra resolution for anything pip had not already installed.

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
# one-time password and stores it at admin_data/initial_admin_password.txt
# (mode 0600). Outside production it is also printed once to stdout for a local
# first boot; in production the log names that file instead of the secret.
# Change it immediately.

# The JSON API (/api/v1/*) is bearer-token protected: set FIXIMG_API_TOKEN to
# pin the token at deploy time, otherwise the app generates one on first start,
# prints it once and stores it at admin_data/api_token.txt (mode 0600).

# The weight sources inside the container may differ from local ones; the conditional
# step above regenerates the manifest from the actual files and verifies it.

EXPOSE 9502
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:9502/api/v1/health/live || exit 1
# The single-container topology: `docker run --gpus all -p 9502:9502 <image>`
# must both serve and execute. `FIXIMG_PROFILE=docker` above describes the
# split deployment (docker/compose.yaml, where a separate worker service owns
# the models) and turns the inline worker OFF; with only that default, a task
# submitted to this image stayed `queued` forever because no process consumed
# the queue -- measured on the built image before this line existed.
#
# The override is applied here rather than in the ENV block above on purpose:
# an image-wide ENV is inherited by every process, including the test suite
# run inside this container, where it silently rewrote the docker profile's
# own assertion (`tests/unit/test_config_profiles.py` expects
# `inline_worker is False` for that profile). Scoping it to the service
# command keeps the image's configuration honest and its tests reproducible.
CMD ["sh", "-c", "FIXIMG_INLINE_WORKER=true FIXIMG_HAS_EXTERNAL_WORKER=false exec python3 main.py"]

