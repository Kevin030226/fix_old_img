# GPU worker image (plan §3.13): owns the models and runs inference.
#
#   docker build -f docker/worker.Dockerfile -t fiximg-worker .
#   docker run --gpus all -e FIXIMG_DEVICE=auto fiximg-worker
#
# The worker shares the database and the artifact volume with the API image, so
# both must mount the same admin_data/ and storage/ paths.
#
# --- why this image no longer starts FROM nvidia/cuda ------------------------
#
# It used to, and that base is Ubuntu 22.04, whose only `python3.11` package is
# the release candidate `3.11.0~rc1-1~22.04.1`. Measured inside the image this
# file built: `python3 --version` -> `Python 3.11.0rc1`. A pre-release does not
# satisfy this project's `requires-python = ">=3.11"` under PEP 440
# (`Version("3.11.0rc1") >= Version("3.11")` is False), so every dependency
# resolved against an interpreter the project does not support.
#
# The root Dockerfile hit the same base and failed *loudly*: requirements.txt
# pins `numpy==2.4.6`, and no 3.11 floor accepts rc1, so the build stopped. This
# file failed *quietly*, because `.[gpu,postgres,redis,s3]` declares floors
# (`>=`) rather than pins: pip walked them down to versions the pre-release
# interpreter would accept and the build reported success with an older stack
# than the project declares. Two measured consequences:
#
#   * Trivy reported 16 HIGH against Ubuntu 22.04's 2022-era openssl and gnupg
#     packages -- nothing to do with this project, and invisible while CI only
#     scanned the API image (Debian, already clean);
#   * `redis` declares `async-timeout>=4.0.3; python_full_version < "3.11.3"`.
#     3.11.0rc1 is below that boundary, so the marker matched, pip did not
#     install the package, and `import redis` raised ModuleNotFoundError.
#     `fiximg/infrastructure/queue/redis.py` wraps that import in try/except on
#     purpose (importing the module must not require the extra), so the worker
#     booted cleanly and then could not use the Redis queue it was configured
#     with -- it fell back to SQLite while `FIXIMG_QUEUE_BACKEND=redis` said
#     otherwise. Installing `async-timeout` by hand treated the symptom, not
#     the cause.
#
# The fix is the one the root Dockerfile already proved on this GPU: take the
# interpreter from the official `python:3.11` image -- a stable, PEP 440-final
# 3.11 -- and copy the CUDA userspace in from the NVIDIA image, rather than
# taking the interpreter from the NVIDIA image. What the GPU needs at run time
# is the driver the container runtime injects plus those userspace libraries;
# the cu128 wheel carries its own CUDA runtime, so `torch.cuda.is_available()`
# is True here with `--gpus all` (measured, see the report).
#
# Both bases are ARGs for the same reason as the root Dockerfile: the default is
# the official name CI resolves, and a build behind a pull-through mirror or in
# an air-gapped registry can point at its own copy without editing this file.
ARG PYTHON_BASE_IMAGE=python:3.11-slim-bookworm
ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

FROM ${PYTHON_BASE_IMAGE} AS python-base
FROM ${CUDA_BASE_IMAGE} AS cuda-runtime

FROM python-base

# CUDA userspace (libcudart, cuBLAS, cuDNN, ...) plus the driver stubs.
# `/usr/local/cuda` is what the cu128 wheels probe; the ldconfig entry makes the
# sonames resolvable without LD_LIBRARY_PATH guesswork.
COPY --from=cuda-runtime /usr/local/cuda /usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:\${LD_LIBRARY_PATH}"
RUN echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf && ldconfig

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FIXIMG_PROFILE=docker \
    FIXIMG_DEVICE=auto

# Build toolchain for dlib (compiled from source -- no cp311 manylinux wheel is
# published) and the shared libraries OpenCV/Pillow dlopen() at run time.
#
# No `python3.11` in this list, deliberately. The interpreter comes from the base
# image at /usr/local/bin, and apt-installing Ubuntu's or Debian's copy beside it
# is exactly the defect this file was rewritten to remove: two interpreters on
# the box, and whichever wins the PATH race decides whether the dependency
# floors resolve. The base image already ships the headers dlib needs
# (/usr/local/include/python3.11/Python.h), so no -dev package is required
# either.
#
# The `--only-upgrade openssl libssl3 gnupg gnupg2` pass this file used to carry
# is gone with the old base. It existed to lift Ubuntu 22.04's 2022-era packages
# to the versions Trivy named as fixed; Debian 12 bookworm ships patched copies,
# which is why the api and root images built on it scan clean without any such
# line. The scan is the authority on whether that still holds (see the report).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git curl unzip bzip2 build-essential cmake ninja-build \
        libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender-dev \
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

# The venv stays: the worker's entry point is `python -m fiximg.cli.worker` and
# its PATH contract is /opt/venv/bin first. What changes is which interpreter
# creates it -- previously Ubuntu 22.04's 3.11.0rc1, now the base image's final
# 3.11. One interpreter installs and the same one runs.
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# The venv is not the only interpreter in the image, and Trivy scans both.
#
# The base image's own /usr/local Python ships setuptools 79.0.1, whose
# vendored copies of jaraco.context 5.3.0 (CVE-2026-23949) and wheel 0.45.1
# (CVE-2026-24049) are the two advisories the api image's Dockerfile upgrades
# setuptools/wheel to remove. Upgrading them inside /opt/venv does not touch
# /usr/local, so the scan reported:
#
#   Python (python-pkg)
#   Total: 2 (HIGH: 2, CRITICAL: 0)
#     usr/local/lib/python3.11/site-packages/setuptools/_vendor/jaraco.context-5.3.0
#     usr/local/lib/python3.11/site-packages/setuptools/_vendor/wheel-0.45.1
#
# Measured in this image before the fix: /usr/local setuptools 79.0.1 +
# wheel 0.46.3, /opt/venv setuptools 84.2.0 + wheel 0.48.0. The system
# interpreter is not what runs the worker (PATH puts /opt/venv/bin first, and
# CMD is python -m fiximg.cli.worker), but it is present, it is reachable, and
# a HIGH in it fails the CI gate -- correctly, since the gate is about the
# image, not about the entry point. Upgrade it rather than excluding it.
RUN /usr/local/bin/python3 -m pip install --upgrade --no-cache-dir \
        "setuptools>=78.1.1" "wheel>=0.46.2"

# PyTorch cu128 (official wheel; includes Blackwell sm_120 support).
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Project + GPU stack, plus the three transports the `platform` profile points a
# worker at (PostgreSQL state, Redis queue, S3 artifacts). dlib builds from
# source on Linux, ~5-10 min. This is the worker image, so the `gpu` extra is
# correct here and only here.
#
# setuptools, wheel and msgpack are upgraded for the same reason as in
# api.Dockerfile, which lists the advisories; the base image here is the same
# Debian line, so the pin moves with it.
#
# NO `async-timeout` ANY MORE. That line was the shim for the pre-release
# interpreter: redis only requires it below 3.11.3, and 3.11.0rc1 is below
# 3.11.3 while a final 3.11.17 is not. With a stable interpreter the marker is
# False, redis does not want the package, and installing it anyway would be a
# dependency nothing declares. Measured after this change: `import redis`
# succeeds and `build_redis_queue_from_settings` returns a backend.
RUN pip install --upgrade "setuptools>=78.1.1" "wheel>=0.46.2" "msgpack>=1.2.1" \
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