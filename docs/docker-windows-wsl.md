# Windows + WSL2 本地 Docker 运行环境

面向使用者的说明。适用场景：Windows 10/11 + Docker Desktop（WSL2 后端）+ 单张 NVIDIA 显卡，在本机跑起本项目（API + Web UI + 内联 worker，端口 9502）。

涉及的两个新文件：

| 文件 | 作用 |
|---|---|
| `docker/local.Dockerfile` | 从 `python:3.11-slim-bookworm` 构建，用 pip 装 cu128 轮子；镜像源由 build-arg 暴露 |
| `docker/compose.wsl.yaml` | 单容器拓扑（API + 内联 worker），GPU 用 `deploy.resources.reservations.devices` 申请，权重只读挂载本机已有副本 |

---

## 0. 一句话结论

仓库自带的 Linux 镜像（`Dockerfile`、`docker/worker.Dockerfile`、`docker-compose.yml`）在本机**构建不了**，因为它们都 `FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04`，而这个基础镜像在本机拉不下来；同时 Docker Desktop 的 WSL2 GPU 透传已经足够跑 PyTorch。于是改用 `docker/local.Dockerfile`（`python:3.11-slim-bookworm` 打底）+ `docker/compose.wsl.yaml`。

---

## 1. 为什么需要这个环境

### 1.1 官方基础镜像在本机不可达

既有镜像的 FROM 都是同一个：

```dockerfile
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
```

实测两个方向都走不通：

1. **直连 Docker Hub 超时**（本机实测）：

```bash
docker manifest inspect python:3.11-slim-bookworm
# failed to configure transport: error pinging v2 registry:
#   Get "https://registry-1.docker.io/v2/": net/http: request canceled
#   while waiting for connection (Client.Timeout exceeded while awaiting headers)
```

2. **可用的 pull-through 加速源不提供 `nvidia/` 命名空间**：`docker.m.daocloud.io` 对 `nvidia/cuda` 返回 `this image is not in the allowlist`（lead 实测）。本次复核同一条命令得到：

```bash
docker manifest inspect docker.m.daocloud.io/nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
# received unexpected HTTP status: 502 Bad Gateway
```

两种返回说明同一件事：该镜像从这个加速源取不到。只要 `FROM nvidia/cuda`，本机就无法构建。所以新 Dockerfile 改成 `FROM python:3.11-slim-bookworm`。

### 1.2 WSL2 的 GPU 透传已经够用

Docker Desktop 的 WSL2 后端把宿主驱动注入**任意**容器：`/dev/dxg` 加 `libcuda.so.1`；CUDA runtime 本身由 pip 装的 cu128 轮子提供。因此**不需要** nvidia/cuda 基础镜像。

本机实测（容器 `python:3.11-slim`，RTX 5060 Laptop，sm_120，驱动 617.14，torch 2.7.1+cu128）：

| 项目 | 实测值 |
|---|---|
| `torch.cuda.is_available()` | `True` |
| `torch.cuda.get_device_capability(0)` | `(12, 0)` |
| 512x512 `matmul` | 成功 |
| `torch.backends.cudnn.version()` | `90701` |

证据脚本：`_audit/docker_env/gpu_probe.py`。

### 1.3 因此新增的两个文件

- `docker/local.Dockerfile`：`FROM python:3.11-slim-bookworm`；apt 与 pip 源用 `--build-arg` 暴露（`APT_MIRROR` / `PIP_INDEX_URL` / `TORCH_INDEX_URL`）；torch 走 cu128 索引单独安装，再 `pip install -e ".[gpu,postgres,redis,s3]"`。
- `docker/compose.wsl.yaml`：单容器，API + inline worker 一个进程，端口 9502，GPU 通过 `deploy.resources.reservations.devices` 申请，权重以**只读 bind mount** 方式挂本机已有的 29 个文件（约 3.3 GB）。

---

## 2. 前置条件与一键命令

### 2.1 前置条件

- Windows + Docker Desktop，**WSL2 后端**，GPU 支持已开启（本机 Docker 29.8.2）。
- NVIDIA 驱动已装（本机 617.14）。RTX 50 系（sm_120）必须用 **cu128** 轮子，旧 CUDA 轮子认不出这张卡。
- 仓库已 checkout，且**本机已具备全部权重**：`config/weights_manifest.json` 列出的 29 个文件，合计约 3.08 GiB（`size_bytes` 合计 3,302,055,889）。权重不烘进镜像，也不需要再下载。
- 基础镜像 `python:3.11-slim-bookworm` 能在本机解析到。本机**当前**只有带加速源前缀的副本：

```bash
docker images --format '{{.Repository}}:{{.Tag}}' | grep python
# docker.m.daocloud.io/library/python:3.11-slim
# docker.m.daocloud.io/library/python:3.11-slim-bookworm

docker image inspect python:3.11-slim-bookworm --format '{{.Id}}'
# Error response from daemon: No such image: python:3.11-slim-bookworm
```

也就是说：`docker build` 里那句 `FROM python:3.11-slim-bookworm` 会先去 Hub 拉，而 Hub 直连超时。二选一先处理掉，见 §5.1。

### 2.2 构建镜像

```bash
cd <仓库根目录>
docker build -f docker/local.Dockerfile -t fiximg:local-wsl .
```

首次构建要源码编译 dlib（PyPI 无 cp311 的 manylinux 轮子），耗时以十分钟量级计。

### 2.3 启动（推荐：compose）

```bash
docker compose -f docker/compose.wsl.yaml up -d
docker compose -f docker/compose.wsl.yaml logs -f
```

浏览器打开 <http://127.0.0.1:9502>。构建与启动一步做完：

```bash
docker compose -f docker/compose.wsl.yaml up -d --build
```

### 2.4 纯 docker run 等价命令（含 `--gpus all`）

`docker/compose.wsl.yaml` 做的每一件事，下面这条 `docker run` 都能做（把 checkout 路径换成自己的）：

```bash
# Point REPO at this checkout. On Windows use forward slashes in the mount
# source, and keep the trailing segment names exactly as shown.
REPO="$(pwd)"        # e.g. D:/code/fiximg
docker run -d --name fiximg-wsl --gpus all \
  -p 9502:9502 \
  -e FIXIMG_PROFILE=docker \
  -e FIXIMG_HOST=0.0.0.0 -e FIXIMG_PORT=9502 \
  -e FIXIMG_DEVICE=auto \
  -e FIXIMG_INLINE_WORKER=true -e FIXIMG_HAS_EXTERNAL_WORKER=false \
  -e FIXIMG_WARMUP=first-use \
  --mount type=volume,source=fiximg-wsl-db,target=/app/admin_data \
  --mount type=volume,source=fiximg-wsl-storage,target=/app/storage \
  --mount type=bind,source="$REPO/weights",target=/app/weights,readonly \
  --mount type=bind,source="$REPO/Global/checkpoints",target=/app/Global/checkpoints,readonly \
  --mount type=bind,source="$REPO/Face_Enhancement/checkpoints",target=/app/Face_Enhancement/checkpoints,readonly \
  --mount type=bind,source="$REPO/Face_Detection",target=/app/Face_Detection,readonly \
  fiximg:local-wsl
```

说明：

- `--gpus all` 与 compose 里 `deploy.resources.reservations.devices` 等价。
- bind 挂载的 `source` 写成 Windows 路径时用正斜杠（例如 `D:/code/fiximg/weights`）。
- 不加 `--gpus all` 也能起来，但容器内 `torch.cuda.is_available()` 会是 `False`，推理退回 CPU。

---

## 3. 目录、端口、卷、权重挂载

### 3.1 端口

| 项 | 值 |
|---|---|
| 容器内监听 | `9502`（`FIXIMG_PORT=9502`） |
| 发布到宿主 | `9502:9502` |
| Web UI | <http://127.0.0.1:9502/>（Gradio 挂在 `/`） |
| API 文档 | `/docs`、`/redoc`、`/openapi.json` |
| 健康探针 | `/api/v1/health/live`、`/api/v1/health/ready`（**公开，不需要令牌**） |

其余 `/api/v1/*` 路由需要 bearer 令牌（`FIXIMG_API_TOKEN`，或首启生成后落在 `admin_data/api_token.txt`）。

### 3.2 卷与挂载点

| 容器内路径 | 类型 | 内容 |
|---|---|---|
| `/app/admin_data` | 命名卷 `db`（可写） | SQLite `fixoldimg.db`、`api_token.txt`(0600)、`initial_admin_password.txt` |
| `/app/storage` | 命名卷 `storage`（可写） | 任务产物 |
| `/app/weights` | bind **只读** | `weights/ddcolor/pytorch_model.pt`、`weights/yunet/*.onnx` |
| `/app/Global/checkpoints` | bind **只读** | 24 个文件（quality / scratch / detection） |
| `/app/Face_Enhancement/checkpoints` | bind **只读** | 2 个文件 |
| `/app/Face_Detection` | bind **只读** | 代码树 + 两个 `.dat` 权重 |

要点：

- **权重不进镜像**（`THIRD_PARTY_NOTICES.md`：本仓库仅分发代码；其中两个 dlib 构件限研究/非商用），compose 直接挂本机已有副本，构建期与运行期都不再访问上游主机。
- 挂载路径与 `config/weights_manifest.json` 里的**仓库相对路径逐字对应**，因此启动自检 `verify_weights()` 能直接通过（29 个文件）。
- 权重卷一律只读：应用从不写权重。

### 3.3 与 `docker/compose.local.yaml` 的差异

| 维度 | `docker/compose.local.yaml` | `docker/compose.wsl.yaml` | 原因 |
|---|---|---|---|
| 构建的基础镜像 | `docker/worker.Dockerfile`（`FROM nvidia/cuda:...`） | `docker/local.Dockerfile`（`FROM python:3.11-slim-bookworm`） | 官方基础镜像在本机拉不到 |
| 权重来源 | 命名卷（`weights` / `global-checkpoints` / `face-restore-checkpoints`），需先跑一次 `download_weights` | 本机 checkout 的**只读 bind**，29 个文件已就位 | 本机已有权重，避免再依赖上游下载 |
| `Face_Detection` | **不挂卷**（依赖镜像内 COPY） | bind 挂载整个目录（只读） | 代码树与两个 `.dat` 同处一目录，卷会遮住脚本 |
| 端口 / 环境变量 | 9502、`FIXIMG_INLINE_WORKER=true` 等 | 同上，另加 `FIXIMG_WARMUP=first-use` | 笔记本上让启动更快 |
| GPU 申请 | `deploy.resources.reservations.devices` | 同上 | 单卡场景语义相同 |
| 可写状态 | `db` / `storage` 命名卷 | 同上 | 状态与产物都不落在宿主路径上 |

两者拓扑一致（单容器、API + 内联 worker、SQLite + 本地存储），差别只在**基础镜像与权重来源**。

---

## 4. 验证方法

### 4.1 健康检查

```bash
curl -fsS http://127.0.0.1:9502/api/v1/health/live
# {"status":"ok"}

curl -fsS http://127.0.0.1:9502/api/v1/health/ready
# status=ready，database=ok，models.cuda_available=true，
# worker.topology=inline，worker.worker_running=true，
# gpu.topology.devices=[0]，model_health 报告各 backend 可用性
```

就绪探针**永不抛异常**：依赖有问题时返回 503 加 `not_ready` 与原因。镜像自带的 HEALTHCHECK 就是打 `/api/v1/health/live`。

```bash
docker compose -f docker/compose.wsl.yaml ps
# STATUS 列出现 (healthy) 即通过
```

### 4.2 容器内 torch.cuda 检查

```bash
docker exec fiximg-wsl python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability(0));a=torch.randn(512,512,device='cuda');print(float((a@a).sum()), torch.backends.cudnn.version())"
# 期望：2.7.1+cu128 12.8 True (12, 0) <数值> 90701
```

同一段检查的脚本版在 `_audit/docker_env/gpu_probe.py`，可直接拷进容器跑。

### 4.3 权重校验（weights verify）

```bash
docker exec fiximg-wsl python -m fiximg.cli.download_weights verify
# Weight integrity check passed (29 files)
```

`python -m fiximg.cli.verify_weights` 是同一实现的别名（不带子命令时默认 `verify`）。校验内容是每个清单条目的存在性 + 大小 + sha256，任一不符即非零退出。

### 4.4 启动自检

服务进程在监听前会先跑一次权重自检，不通过就**拒绝启动**：

```bash
docker compose -f docker/compose.wsl.yaml logs | grep -i self-check
# [Self-check] Weight integrity OK (29 files)
```

### 4.5 不启动容器，先看拓扑

```bash
docker compose -f docker/compose.wsl.yaml config
```

打印最终展开的 GPU reservation、端口映射与四个 bind 的宿主绝对路径，用来确认挂载指向正确的 checkout。

---

## 5. 故障排查

### 5.1 镜像拉取失败 / 国内加速

**症状**：`docker build` 在 `FROM python:3.11-slim-bookworm` 处卡住，报 `Client.Timeout exceeded while awaiting headers`；或 `docker manifest inspect` 直接超时。

**诊断**（不改任何状态）：

```bash
docker manifest inspect python:3.11-slim-bookworm
docker images --format '{{.Repository}}:{{.Tag}}' | grep python
```

**两条出路，任选一条**：

1. **给 Docker Desktop 配 registry mirror**：Settings → Docker Engine，在 JSON 里加

```json
{ "registry-mirrors": ["https://docker.m.daocloud.io"] }
```

保存后重启 Docker。本机**当前未配置**镜像加速：`docker info` 没有 `Registry Mirrors` 段，`~/.docker/daemon.json` 里也没有 mirror 键——所以 `FROM python:...` 会真的去连 Hub。

2. **用已有的本地副本改标签**（不需要联网）：

```bash
docker tag docker.m.daocloud.io/library/python:3.11-slim-bookworm python:3.11-slim-bookworm
docker image inspect python:3.11-slim-bookworm --format '{{.Id}}'   # 有输出即成功
```

本机确实有这个副本（`docker.m.daocloud.io/library/python:3.11-slim-bookworm`，210MB），所以这条路是通的。

**注意**：无论用哪条，`nvidia/cuda` 依然拿不到（见 §1.1），不要试图回到官方基础镜像。

### 5.2 容器内 DNS 瞬时失败

**症状**：`pip install` / `apt-get` 偶发报域名解析失败。本机容器的 DNS 走 Docker Desktop 内部解析器（`192.168.65.7`），曾出现瞬时解析失败。

**处理**：直接重试，通常立刻恢复。若持续失败：

```bash
docker run --rm python:3.11-slim-bookworm python -c "import socket;print(socket.gethostbyname('pypi.tuna.tsinghua.edu.cn'))"
docker compose -f docker/compose.wsl.yaml restart
```

构建期遇到也可以换一个镜像源（见 §5.5）再构建。

### 5.3 9502 端口被占用

**症状**：`up` 时 `Bind for 0.0.0.0:9502 failed: port is already allocated`，或本机已有服务占着 9502。

**诊断**（Windows）：

```bash
netstat -ano | findstr :9502
# 末列是 PID
tasklist /FI "PID eq <PID>"
```

**处理**：停掉占用进程，或改发布端口（容器内仍是 9502）：

```bash
docker compose -f docker/compose.wsl.yaml down
docker run -d --name fiximg-wsl --gpus all -p 9503:9502 ... fiximg:local-wsl
# 然后访问 http://127.0.0.1:9503
```

改发布端口只影响宿主映射，`FIXIMG_PORT=9502` 不必改。

### 5.4 GPU 不可见

**症状**：容器内 `torch.cuda.is_available()` 为 `False`，或就绪探针里 `models.cuda_available=false`。

**按顺序查**：

```bash
# 1. 宿主驱动在不在
nvidia-smi

# 2. Docker 是否注册了 nvidia runtime（应能看到 "nvidia": nvidia-container-runtime）
docker info --format '{{json .Runtimes}}' | tr ',' '\n' | grep -i nvidia

# 3. 容器是否被申请到 GPU：WSL2 透传会注入 /dev/dxg
docker exec fiximg-wsl ls -l /dev/dxg
docker exec fiximg-wsl python -c "import torch;print(torch.cuda.is_available())"
```

要点：

- 本机 `Runtimes` 已含 `nvidia`（`docker info` 可见），说明透传链路具备。
- 容器起没起 `--gpus all` / compose 的 device reservation，决定 `/dev/dxg` 是否存在。
- 若 `is_available()` 为真但算不了：确认装的是 **cu128** 轮子（sm_120 需要），`torch.version.cuda` 应为 `12.8`。
- 驱动版本过低（本机 617.14）也会让新卡认不出来，需要升驱动。

### 5.5 换镜像源：`--build-arg`

`docker/local.Dockerfile` 的三个源都是 build-arg，默认走国内镜像：

| build-arg | 默认值 |
|---|---|
| `APT_MIRROR` | `mirrors.tuna.tsinghua.edu.cn` |
| `PIP_INDEX_URL` | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| `TORCH_INDEX_URL` | `https://mirror.sjtu.edu.cn/pytorch-wheels/cu128` |

换源重构建（例：回到官方源）：

```bash
docker build -f docker/local.Dockerfile -t fiximg:local-wsl \
  --build-arg APT_MIRROR=deb.debian.org \
  --build-arg PIP_INDEX_URL=https://pypi.org/simple \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 .
```

用 compose 构建时同样可以传（`docker compose build --build-arg`，本机 Compose v5.6.0 支持）：

```bash
docker compose -f docker/compose.wsl.yaml build --build-arg PIP_INDEX_URL=https://pypi.org/simple
docker compose -f docker/compose.wsl.yaml up -d
```

---

## 6. 边界与未验证项

诚实说明适用边界，避免把它当成已验证的生产方案：

1. **容器内已跑通 GPU 推理测试档，但未走完整「上传照片 → 出图」链路**。已验证：WSL2 GPU 透传下 torch 2.7.1+cu128 可用（RTX 5060 / sm_120 / matmul 成功）、`verify_weights()` 29/29 通过并有启动日志 `[Self-check] Weight integrity OK (29 files)`、`docker compose up -d` 后 `/api/v1/health/live` 返回 HTTP 200 `{"status":"ok"}`、`tests/gpu -m gpu` 在容器内 **30 passed**（含 dlib 子进程检测器与原生后端等价性，即真实推理路径）。仍未做的是通过 UI/API 上传一张老照片并核对产物。
2. **vendored 代码树已补进镜像（已修复，附一个容易踩的坑）**。运行期需要 `Global/`、`Face_Enhancement/`、`Face_Detection/`、`ddcolor/`、`basicsr/`：`ddcolor` 由 `src/fiximg/inference/model_manager.py` 在进程内 import（无任何 pip 依赖提供它），而 `ddcolor/model.py` 又 import `basicsr.archs.*`。
   **坑**：不能写 `COPY Global ./Global`。仓库 `.dockerignore` 里的 `*.pth` / `*.pt` / `*.dat` 是**无锚定**模式，Docker 只用它们匹配 **context 根**，嵌套文件不会被排除——`COPY Global ./Global` 会把 `Global/checkpoints` 的 2.1 GB 权重打进镜像（实测构建 context 从 2.4 MB 涨到 1.8 GB+），且这与「权重不进镜像」的许可立场直接冲突。CI 里看不出来，因为新 checkout 的 `checkpoints/` 是空的。
   因此 `docker/local.Dockerfile` **逐个子目录**只 COPY 各树的 `*.py`，权重仍由 compose 的只读 bind 提供。已实测：`find /app -name '*.pth' -o -name '*.pt' -o -name '*.dat' | wc -l` = **0**。
3. **Linux 生产部署仍应使用官方镜像**（`Dockerfile` / `docker/worker.Dockerfile` / `docker/compose.yaml`）。本套文件只解决「Windows + Docker Desktop + 单卡」这一台机器上的构建与运行，不是生产拓扑的替代品。
4. **单容器、单卡**。内联 worker 意味着 API 与推理同进程；多卡/多 worker 场景请用 `docker/compose.multigpu.yaml` 之类拓扑（同样受 §1.1 的基础镜像问题影响）。

---

## 附录：本文档结论的取证命令

| 结论 | 命令 | 结果 |
|---|---|---|
| Docker 版本 | `docker version --format '{{.Client.Version}} / server {{.Server.Version}}'` | `29.8.2 / 29.8.2` |
| Hub 直连超时 | `docker manifest inspect python:3.11-slim-bookworm` | `Client.Timeout exceeded while awaiting headers` |
| 加速源无 nvidia/ | `docker manifest inspect docker.m.daocloud.io/nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04` | `502 Bad Gateway`（lead 另测到 `not in the allowlist`） |
| 基础镜像本地副本 | `docker images --format '{{.Repository}}:{{.Tag}}'` | 有 `docker.m.daocloud.io/library/python:3.11-slim-bookworm`（210MB） |
| 裸标签缺失 | `docker image inspect python:3.11-slim-bookworm` | `No such image` |
| 无 registry mirror | `docker info` | 无 `Registry Mirrors` 段；`daemon.json` 无 mirror 键 |
| nvidia runtime 已注册 | `docker info --format '{{json .Runtimes}}'` | 含 `nvidia`（`nvidia-container-runtime`） |
| compose 展开正确 | `docker compose -f docker/compose.wsl.yaml config` | GPU reservation、`9502:9502`、四个 bind 的宿主绝对路径 |
| 权重齐备 | `config/weights_manifest.json`（`file_count: 29`） | 29 个文件，合计 3,302,055,889 字节（约 3.08 GiB） |
| 容器内 GPU | `_audit/docker_env/gpu_probe.py` | `cuda_available=True`、capability `(12, 0)`、matmul 成功、cudnn `90701` |

相关文件：`docker/local.Dockerfile`、`docker/compose.wsl.yaml`、`docker/compose.local.yaml`、`config/weights_manifest.json`、`docs/deployment.md`。
