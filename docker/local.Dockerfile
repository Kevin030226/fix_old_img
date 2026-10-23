# Local GPU runtime for Windows + Docker Desktop (WSL2 backend).
#
# Why this file exists: the shipped images (Dockerfile, docker/worker.Dockerfile)
# start from nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04. On this host that base
# cannot be pulled at all -- registry-1.docker.io is unreachable, and the mirror
# that does answer does not carry the nvidia/ namespace (it replies 'this image
# is not in the allowlist'). Docker Desktop's WSL2 GPU support does not need it:
# the driver is injected into every container as /dev/dxg plus libcuda.so.1, and
# the cu128 wheels bring the CUDA runtime itself. Measured on this machine before
# this file was written: torch 2.7.1+cu128, cuda_available=True,
# capability (12, 0), matmul on the RTX 5060 Laptop succeeded, cudnn 90701.
#
#   docker build -f docker/local.Dockerfile -t fiximg:local-wsl .
#   docker compose -f docker/compose.wsl.yaml up -d
# The base image is an ARG because on a network where registry-1.docker.io is
# unreachable it has to name a pull-through mirror (Docker Desktop resolves the
# official name through the same dead route). Default is the official name:
#   docker build --build-arg BASE_IMAGE=docker.m.daocloud.io/library/python:3.11-slim-bookworm .
ARG BASE_IMAGE=python:3.11-slim-bookworm
FROM ${BASE_IMAGE}

# Build-time only. Override for a network where these mirrors are not the fast path:
#   docker build --build-arg APT_MIRROR=deb.debian.org \
#                --build-arg PIP_INDEX_URL=https://pypi.org/simple \
#                --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 .
ARG APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG TORCH_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FIXIMG_PROFILE=docker \
    FIXIMG_HOST=0.0.0.0 \
    FIXIMG_PORT=9502 \
    FIXIMG_DEVICE=auto \
    FIXIMG_INLINE_WORKER=true \
    FIXIMG_HAS_EXTERNAL_WORKER=false

# Toolchain for dlib (compiled from source for cp311: PyPI ships no manylinux
# wheel) plus the OpenCV/Pillow runtime libraries. One RUN so the --build-arg
# mirror is applied before the first apt-get update.
RUN sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git curl unzip bzip2 \
      libgl1 libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY src ./src
COPY configs ./configs
COPY config ./config
COPY models ./models
COPY main.py worker.py run.py ./

# setuptools/wheel floors: the same two advisories the shipped images pin up
# (CVE-2025-47273 at 70.3.0, CVE-2026-24049 at 0.45.1). The floors rather than a
# bare --upgrade so two builds of this source cannot differ in security posture.
#
# The cu128 index is a separate -i on purpose: torch 2.7.1+cu128 is not on PyPI,
# and installing it first means the project's `gpu` extra resolves against what
# is already present instead of reaching for a CPU wheel.
RUN pip install --upgrade "setuptools>=78.1.1" "wheel>=0.46.2" "msgpack>=1.2.1" \
 && pip install -i ${TORCH_INDEX_URL} torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
 && pip install -i ${PIP_INDEX_URL} -e ".[gpu,postgres,redis,s3]"

# Vendored model code, copied AFTER the pip layer so adding a tree does not
# invalidate the ~10-minute dependency install. These are imported at run time
# and are not pip packages: `from ddcolor import ...` reaches /app/ddcolor
# (model_manager.py puts the repo root on sys.path), and that imports
# basicsr.archs.*; the legacy chain runs with cwd=Global/ and cwd=Face_Enhancement/.
# The checkpoint payloads stay OUT of the image: .dockerignore drops *.pt/*.pth/
# *.dat, and compose.wsl.yaml bind-mounts the real ones read-only.
# Only the *.py half of each tree is copied. A bare `COPY Global ./Global`
# would drag in Global/checkpoints (2.1 GB): the repository's .dockerignore
# lists `*.pth` / `*.pt` / `*.dat` unanchored, and Docker matches those against
# the context root only -- nested payloads are NOT excluded, which is invisible
# in CI because a fresh checkout ships those directories empty.
# The payloads arrive through compose.wsl.yaml's read-only binds instead.
COPY Global/data ./Global/data
COPY Global/detection_models ./Global/detection_models
COPY Global/detection_util ./Global/detection_util
COPY Global/models ./Global/models
COPY Global/options ./Global/options
COPY Global/util ./Global/util
COPY Global/detection.py Global/test.py Global/train_domain_A.py Global/train_domain_B.py Global/train_mapping.py ./Global/
COPY Face_Enhancement/data ./Face_Enhancement/data
COPY Face_Enhancement/models ./Face_Enhancement/models
COPY Face_Enhancement/options ./Face_Enhancement/options
COPY Face_Enhancement/util ./Face_Enhancement/util
COPY Face_Enhancement/test_face.py ./Face_Enhancement/
COPY Face_Detection/align_warp_back_multiple_dlib.py Face_Detection/align_warp_back_multiple_dlib_HR.py Face_Detection/detect_all_dlib.py Face_Detection/detect_all_dlib_HR.py ./Face_Detection/
COPY ddcolor ./ddcolor
COPY basicsr ./basicsr

# The project's `test` extra, in its own late layer: pytest/moto/fakeredis are
# not runtime dependencies, and folding them into the install above would
# re-resolve the whole dependency set and re-download the cu128 stack. Having
# them here is what makes `docker run ... python -m pytest tests/` possible.
RUN pip install -i ${PIP_INDEX_URL} -e ".[test]"

# Weights are NOT baked in (THIRD_PARTY_NOTICES.md: code-only distribution, two
# artifacts research/non-commercial). compose.wsl.yaml bind-mounts the copies
# already on this machine, so nothing is downloaded at build or run time.
EXPOSE 9502
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:9502/api/v1/health/live || exit 1
CMD ["python", "-m", "fiximg.cli.api"]
