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

# Download all weights (BOB restoration chain + DDColor colorization)
RUN bash scripts/download_weights.sh

# If config/users.yaml is not provided, create the default admin admin/admin123 (please change it after deployment)
RUN if [ ! -f config/users.yaml ]; then \
      python3 -c "from config.security import hash_password; import yaml; \
      yaml.safe_dump({'users': {'admin': {'password': hash_password('admin123'), 'role': 'admin'}}}, \
      open('config/users.yaml','w',encoding='utf-8'), allow_unicode=True)"; \
    fi

# The weight sources inside the container may differ from local ones; regenerate based on the actual files and verify the integrity manifest
RUN python3 -m config.weights_check generate && python3 -m config.weights_check verify

EXPOSE 9502
CMD ["python3", "main.py"]

