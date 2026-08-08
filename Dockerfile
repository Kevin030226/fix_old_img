# 旧照片修复系统 —— 现代技术栈 Docker 镜像
# Python 3.11 + PyTorch 2.7.1 cu128（RTX 50 / sm_120）+ Gradio 6 + FastAPI
#
# 构建：docker build -t fixoldimg .
# 运行：docker run --gpus all -p 9502:9502 fixoldimg
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FIXIMG_HOST=0.0.0.0 \
    FIXIMG_PORT=9502

# 系统依赖：Python 3.11 + dlib 源码构建工具链 + OpenCV 运行库
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip python3-dev \
    git curl unzip bzip2 build-essential cmake ninja-build \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Python 虚拟环境
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PyTorch cu128（官方 wheel，已内置 Blackwell sm_120 支持）
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# 项目依赖（dlib 在 Linux 上从源码构建，约 5-10 分钟）
RUN pip install -r requirements.txt

# 下载全部权重（BOB 修复链路 + DDColor 上色）
RUN bash scripts/download_weights.sh

# 若未提供 config/users.yaml，则生成默认管理员 admin/admin123（部署后请立即修改）
RUN if [ ! -f config/users.yaml ]; then \
      python3 -c "from config.security import hash_password; import yaml; \
      yaml.safe_dump({'users': {'admin': {'password': hash_password('admin123'), 'role': 'admin'}}}, \
      open('config/users.yaml','w',encoding='utf-8'), allow_unicode=True)"; \
    fi

# 容器内权重来源可能与本地不同，按实际文件重新生成并校验完整性清单
RUN python3 -m config.weights_check generate && python3 -m config.weights_check verify

EXPOSE 9502
CMD ["python3", "main.py"]

