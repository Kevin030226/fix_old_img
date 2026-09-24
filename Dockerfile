# Old photo restoration system — modern stack Docker image
# Python 3.11 + PyTorch 2.7.1 cu128 (RTX 50 / sm_120) + Gradio 6 + FastAPI
#
# Build: docker build -t fixoldimg .
# Run: docker run --gpus all -p 9502:9502 fixoldimg
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
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
RUN pip install -r requirements.txt

# Download all weights (BOB restoration chain + DDColor colorization) and
# rebuild the integrity manifest (plan §19: python entrypoints, resume support)
RUN python3 -m scripts.download_weights download

# Admin credentials are NOT baked into the image (plan §21): on first start the
# app reads FIXIMG_ADMIN_PASSWORD when provided, otherwise it generates a random
# one-time password, prints it once in the container log and stores it at
# admin_data/initial_admin_password.txt (mode 0600). Change it immediately.

# The JSON API (/api/v1/*) is bearer-token protected: set FIXIMG_API_TOKEN to
# pin the token at deploy time, otherwise the app generates one on first start,
# prints it once and stores it at admin_data/api_token.txt (mode 0600).

# The weight sources inside the container may differ from local ones; regenerate the manifest from the actual files and verify it
RUN python3 -m scripts.download_weights generate && python3 -m scripts.verify_weights

EXPOSE 9502
CMD ["python3", "main.py"]
