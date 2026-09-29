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

# Project + GPU extras (dlib builds from source on Linux, ~5-10 minutes).
RUN pip install -e ".[gpu]"

# Fetch the weights and rebuild the integrity manifest (plan §19: python
# entrypoints, resumable downloads). Cache this layer across rebuilds.
RUN python -m fiximg.cli.download_weights download \
    && python -m fiximg.cli.verify_weights

CMD ["python", "-m", "fiximg.cli.worker"]
