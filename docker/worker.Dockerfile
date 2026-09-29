# GPU worker image (plan §3.13): owns the models and runs inference.
#
#   docker build -f docker/worker.Dockerfile -t fiximg-worker .
#   docker run --gpus all -e FIXIMG_DEVICE=auto fiximg-worker
#
# The worker shares the database and the artifact volume with the API image, so
# both must mount the same admin_data/ and storage/ paths.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FIXIMG_PROFILE=docker \
    FIXIMG_DEVICE=auto

# Python 3.11 + dlib build toolchain + OpenCV runtime libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip python3-dev \
    git curl unzip bzip2 build-essential cmake ninja-build \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY src ./src
COPY configs ./configs
COPY models ./models
COPY config ./config
COPY Global ./Global
COPY Face_Detection ./Face_Detection
COPY Face_Enhancement ./Face_Enhancement
COPY ddcolor ./ddcolor
COPY basicsr ./basicsr
COPY main.py worker.py run.py ./

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PyTorch cu128 (official wheel; includes Blackwell sm_120 support).
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Project + GPU stack, plus the three transports the `platform` profile points a worker at
# (PostgreSQL state, Redis queue, S3 artifacts). dlib builds from source on Linux, ~5-10 min.
#
# setuptools and wheel are upgraded for the same reason as in api.Dockerfile, and
# the comment there lists the advisories. Only the API image is scanned by CI, so
# this is the untested half of the same problem: fixed on the same reasoning, to
# be confirmed by the release build rather than by a gate.
RUN pip install --upgrade setuptools wheel \
    && pip install -e ".[gpu,postgres,redis,s3]"

# Weights are deliberately NOT baked in by default.
#
#   * THIRD_PARTY_NOTICES.md records this distribution as "code only", and two of the
#     six artifacts (dlib's landmark predictor and its face-recognition ResNet) are
#     licensed for research/non-commercial use, so a published image must not carry them.
#   * The upstream host for the restoration checkpoints, facevc.blob.core.windows.net,
#     did not resolve from GitHub's runners on 2026-09-29 ("Name or service not known"),
#     so a build that depends on it cannot validate the Dockerfile at all -- that is what
#     failed both `images (worker)` jobs for v3.0.0.
#
# A weights-free worker is a supported shape: the registry reports each model
# unavailable through the readiness probe instead of crashing. Fill the volumes with
# `docker compose exec worker python -m fiximg.cli.download_weights download`, or set
# FIXIMG_BAKE_WEIGHTS=true for a private build on a machine that is entitled to them:
#
#   docker build -f docker/worker.Dockerfile --build-arg FIXIMG_BAKE_WEIGHTS=true .
ARG FIXIMG_BAKE_WEIGHTS=false
RUN if [ "$FIXIMG_BAKE_WEIGHTS" = "true" ]; then \
        python -m fiximg.cli.download_weights download \
        && python -m fiximg.cli.verify_weights; \
    else \
        echo "[build] weights not baked -- see docker/worker.Dockerfile comments"; \
    fi

CMD ["python", "-m", "fiximg.cli.worker"]
