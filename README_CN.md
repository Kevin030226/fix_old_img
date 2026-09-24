﻿<!--
  fix_old_img — 基于深度学习的旧照片恢复、划痕修复与老照片上色系统
  中文备份版;默认文档为英文,请查看 README.md
-->
<div align="center">

# 基于深度学习的旧照片恢复、划痕修复与老照片上色系统

**Deep Learning Based Old Photo Restoration, Scratch Repair & Colorization**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1%2Bcu128-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.22-orange?logo=gradio&logoColor=white)](https://gradio.app/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/Kevin030226/fix_old_img/actions/workflows/ci.yml/badge.svg)](https://github.com/Kevin030226/fix_old_img/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Kevin030226/fix_old_img?sort=semver)](https://github.com/Kevin030226/fix_old_img/releases)

[English](./README.md) | **中文**

<p align="center">
  <img src="docs/upstream/bob-0001.jpg" width="100%">
</p>

本系统用 Python 3.11 与 PyTorch（CUDA 12.8）构建了一套面向真实老照片的 Web 端图像修复系统：支持整体质量修复、划痕检测与修复、面部增强，以及黑白老照片自动上色。Web 层使用 Gradio 6 + FastAPI + Uvicorn，数据层使用 SQLite（WAL），并附带完整的管理后台（任务记录、照片档案、用户管理）。

</div>

---

## 📑 目录

- [1. 项目解决的问题](#1-项目解决的问题)
- [2. 主要功能](#2-主要功能)
- [3. 使用的技术](#3-使用的技术)
- [4. 安装方法](#4-安装方法)
- [5. 使用方法](#5-使用方法)
- [6. 输入输出示例](#6-输入输出示例)
- [7. 项目结构](#7-项目结构)
- [8. 常见问题](#8-常见问题)
- [9. 第三方组件与许可](#9-第三方组件与许可)

---

## 1. 项目解决的问题

老照片在保存和扫描过程中普遍存在以下问题：

- **整体退化**：褪色、模糊、噪点、对比度下降；
- **物理损伤**：折痕、划痕、污渍；
- **黑白化**：早期照片只有灰度信息，缺少自然色彩；
- **面部细节丢失**：人像区域细节严重退化，常规修复难以恢复；
- **修复过程不可见**：批量处理缺少任务记录、结果归档与质量指标。

本系统针对上述问题提供一条完整的处理链路：**整体质量修复 → 划痕检测/修复 → 人脸检测与增强 → 回卷合成**，并在同一平台内集成了 **DDColor 老照片上色** 能力，同时提供用户登录、注册、任务历史、照片档案和用户管理等配套功能。

## 2. 主要功能

### 2.1 图像处理模块（四个标签页）

| 模块 | 说明 | 输出 |
| --- | --- | --- |
| 不带划痕的旧照片复原 | 整体质量提升 + 人脸检测与面部增强 | 修复后的彩色照片 + PSNR/SSIM/MAE 指标 |
| 带划痕的旧照片复原 | 自动划痕检测 → 划痕修复 → 质量提升 → 面部增强 | 修复后的彩色照片 + 指标 |
| 划痕检测 | 独立输出划痕位置 | 黑白二值 mask（白色为划痕） |
| 老照片上色 | DDColor 对黑白/灰度照片自动上色 | 彩色照片 |

### 2.2 管理面板（仅管理员可见）

- **任务管理**：处理历史记录、统计概览（任务数/用户数/各类型次数/平均指标）、清空记录；
- **照片档案**：按处理记录浏览原始图片与修复结果；
- **用户管理**：添加、修改（密码/角色）、删除用户。

### 2.3 平台能力

- 用户登录（pbkdf2 哈希、常量时间比对）与公开注册（IP/全局/用户名三重限流）；
- 健康检查接口 `GET /health`（存活）与 `GET /health/ready`（数据库就绪 + 模型/worker 状态，异常时返回 503）；
- V2 任务 API `POST/GET /api/v1/tasks`（异步执行、进度、取消、结果与报告下载；所有 `/api/v1/*` 接口均需 Bearer token）；
- 基于数据库的任务队列与独立 GPU worker：默认内联线程（`FIXIMG_INLINE_WORKER=true`），也可用独立进程 `python worker.py`（api+worker 双服务 compose 拓扑）；worker 崩溃遗留的 running 任务会自动重新入队；
- `scripts/migrate_v1.py` 将旧版 `history` 记录迁入 V2 `tasks` 表（默认 dry-run，`--apply` 才写入）；
- 启动时权重完整性自检（SHA-256 清单，缺失或篡改拒绝启动）；
- 请求级目录隔离（`storage/tasks/` 结构化产物目录）+ 结果 TTL 自动回收；
- SQLite 存储（用户/历史 + V2 tasks/task_stages/artifacts/metrics），首次启动自动从旧版 users.yaml / JSONL 迁移；
- 深色主题界面，管理员/普通用户界面自动区分。

## 3. 使用的技术

| 类别 | 技术 |
| --- | --- |
| 语言/环境 | Python 3.11（conda 环境 `fixoldimg-gpu`） |
| Web 框架 | Gradio 6.22 + FastAPI 0.141 + Uvicorn 0.52 |
| 深度学习 | PyTorch 2.7.1+cu128（原生支持 RTX 50 系 sm_120） |
| 视觉/科学计算 | OpenCV 5.0、scikit-image 0.26、scipy 1.17、numpy 2.4、Pillow 12.3 |
| 人脸/上色 | dlib 20.0.1（conda-forge）、timm 0.9.2、DDColor |
| 存储 | SQLite（WAL 模式） |
| 模型 | Bringing Old Photos Back to Life + DDColor，共 29 个权重文件 |

### 3.1 整体质量修复、划痕检测与面部增强技术

本系统的修复链路（整体质量修复 / 划痕检测与修复 / 面部增强）采用以下技术方案：

**整体质量修复（Global Restoration）**

> 采用三元域翻译网络（triplet domain translation network），同时处理旧照片的结构化退化与非结构化退化：

- 分别训练域 A（退化旧照）与域 B（高质量新照）的 **VAE**（变分自编码器）模型，二者共享潜空间结构；
- 训练域间的 **mapping network（映射网络）**，将退化域隐变量翻译到高质量域，从而实现整体质量提升；
- 映射网络支持多种训练变体：`mapping_quality`（无划痕场景）、`mapping_scratch`（带划痕场景）、`mapping_Patch_Attention`（使用 Multi-Scale Patch Attention，用于高分辨率输入的划痕修复）；
- 训练采用 `pix2pixHD` 风格的双判别器 GAN 架构，通过 `--l2_feat / --use_l1_feat / --NL_res`（非局部残差）等选项控制损失与结构。

<p align="center">
  <img src="docs/upstream/bob-pipeline.png" width="60%">
</p>

<p align="center">
  <img src="docs/upstream/bob-global.png" width="100%">
</p>

**划痕检测（Scratch Detection）**

> 划痕检测模型使用标注数据训练，输出黑白二值 mask（白色为划痕）。划痕修复链路对高分辨率输入使用 Multi-Scale Patch Attention 的非局部映射，可将带碎裂痕迹的老照片恢复为干净画面。

<p align="center">
  <img src="docs/upstream/bob-scratch-detection.png" width="100%">
</p>

<p align="center">
  <img src="docs/upstream/bob-hr-result.png" width="100%">
</p>

**人脸检测与面部增强（Face Detection & Enhancement）**

> 采用**渐进式生成器（progressive generator）**细化旧照片中的人脸区域：

- 人脸检测使用 dlib 的 `shape_predictor_68_face_landmarks.dat`（68 点人脸关键点检测器）；
- 检测到的人脸逐一裁剪、对齐后送入面部增强模型，增强完成后按原始几何关系**回卷（warp back）**合成回原图；
- 面部增强模型带同步批归一化（Synchronized-BatchNorm-PyTorch）与渐进式编码器结构，通过实例归一化参数调制逐级细化人脸。

<p align="center">
  <img src="docs/upstream/bob-face-pipeline.png" width="70%">
</p>

<p align="center">
  <img src="docs/upstream/bob-face.png" width="100%">
</p>

> 注：该模型用 256×256 预训练，任意分辨率下效果可能非最优（本系统支持长边 ≤4096px 的输入）。

### 3.2 黑白照片上色技术

本系统的黑白照片自动上色模块采用以下技术方案：

> 使用**多尺度视觉特征**去优化**可学习的颜色 token（即颜色查询 color queries）**，在自动图像上色任务上达到 SOTA 水平：

- **双解码器结构（Dual Decoders）**：一个解码器做**颜色解码**（基于可学习颜色 token 与多尺度特征交互），一个解码器做**图像重建**（恢复空间细节），二者共同实现照片级真实的多彩着色；
- **骨干网络**：基于 ConvNeXt（`ConvNeXt-Large`，22k 预训练），编码器提取多尺度视觉特征；颜色 token 与特征通过类似 Transformer 的交叉注意力（Mask2Former / DETR 风格）进行交互；
- **颜色查询（color queries）**：一组可学习的颜色 embedding，被视为"颜色字典"，通过多尺度特征的查询得到目标颜色分布；
- 训练流程基于 **BasicSR** 工具箱（同步最小依赖为 `basicsr/` 子集），支持 `ddcolor_paper / ddcolor_modelscope / ddcolor_artistic / ddcolor_paper_tiny` 四种预训练模型规格；
- 本系统默认使用 `damo/cv_ddcolor_image-colorization` 权重：`weights/ddcolor/pytorch_model.pt`，模型规格 `large`，输入尺寸 512×512（均可通过环境变量调整）。

<p align="center">
  <img src="docs/upstream/ddcolor-network-arch.jpg" width="100%">
</p>

<p align="center">
  <img src="docs/upstream/ddcolor-teaser.webp" width="100%">
</p>

<p align="center">
  <img src="docs/upstream/ddcolor-anime.webp" width="100%">
</p>

## 4. 安装方法

### 4.1 Windows（conda）

一键脚本方式：

```bat
scripts/setup_gpu.bat
```

或手动执行：

```bash
# 1. 创建环境
conda create -n fixoldimg-gpu python=3.11 -y
conda activate fixoldimg-gpu

# 2. 安装 PyTorch cu128（支持 RTX 50 系；国内网络可换
#    https://mirror.sjtu.edu.cn/pytorch-wheels/cu128）
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# 3. 安装项目依赖（含 Gradio/FastAPI/OpenCV/DDColor 所需 timm 等）
pip install -r requirements.txt

# 4. 安装 dlib 预编译包（conda-forge，无需本机编译工具链）
conda install -n fixoldimg-gpu -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
    --override-channels -y dlib=20.0.1

# 5. 下载模型权重（BOB 修复链路 + DDColor 上色，约 1.5GB）
#    （方案 §19：python 入口、断点续传、带版本号的清单）
python -m scripts.download_weights download

# 6. 若权重来源与仓库清单不同，重新生成并校验
python -m scripts.download_weights generate
python -m scripts.verify_weights
```

> 注意：`config/users.yaml` 会被自动迁移到 SQLite（`admin_data/fixoldimg.db`）。若该文件缺失，首启动引导（§21）会自动创建初始 `admin` 账号——参见上方 Docker 小节。

### 4.2 Docker（Linux + NVIDIA GPU）

```bash
docker build -t fixoldimg .
docker run --gpus all -p 9502:9502 fixoldimg
```

或使用 V2 双服务拓扑（API 只入队；独立 worker 进程独占 GPU 与模型）：

```bash
docker compose up -d --build   # api + worker（见 docker-compose.yml）
docker compose logs -f worker
```

镜像会自动安装依赖、下载全部权重并重建权重清单。首次构建时 dlib 为源码编译，耗时约 5-10 分钟。

**管理员凭据（方案 §21）：** 镜像不再内置任何固定密码。首次启动时应用会优先读取 `FIXIMG_ADMIN_PASSWORD`（例如来自 Docker secret）；若未设置，则自动生成随机一次性密码，在容器日志中**仅打印一次**，并写入 `admin_data/initial_admin_password.txt`（权限 0600）。首次登录后请立即修改。设置 `FIXIMG_AUTO_BOOTSTRAP_ADMIN=false` 可完全关闭该引导机制。

**强制修改密码（方案 §21）：** 随机生成的 admin 账号会带上 `must_change_password` 标记。标记未清除前，提交任务会被拦截并给出明确提示，管理面板的「Account」标签页提供改密表单并持续显示横幅提醒。保存新密码后标记自动清除；通过 `FIXIMG_ADMIN_PASSWORD` 注入的密码视为部署方管理，不触发强制修改。

## 5. 使用方法

### 5.1 启动 Web 服务

```bash
conda activate fixoldimg-gpu
python main.py
```

浏览器访问 <http://127.0.0.1:9502>。

默认账号（**部署前必须修改**）：

| 用户名 | 密码 | 角色 |
| --- | --- | --- |
| admin | 安装/初始化时设置 | 管理员（仅管理面板） |
| user1 | 安装/初始化时设置 | 普通用户 |

> ⚠️ 本地开发环境为方便测试保留了演示口令（如 `admin/admin123`），
> 仅限开发使用；公开部署前请通过"管理面板 → 用户管理"轮换全部账号口令。

管理员登录后只显示"管理面板"；普通用户可使用四个图像处理标签页。

### 5.2 Web 操作流程

1. 登录（或点击"立即注册"创建账号）；
2. 选择对应功能标签页；
3. 上传图片（或点击标签页下方示例图自动填充）；
4. 点击"开始修复 / 提交复原 / 开始上色"；
5. 等待处理完成（GPU 环境单张约 1-60 秒，取决于模块与图片大小），查看结果与指标；
6. 管理员可在"管理面板"查看任务记录、照片档案并管理用户。

### 5.3 V2 任务 API

所有 `/api/v1/*` 接口都要求以 Bearer 方式携带 API token（`/health`、`/health/ready`
保持公开以便探针使用）。Gradio 界面不受影响——它直接调用进程内服务层。

```bash
# token 来自 FIXIMG_API_TOKEN，或首次启动时生成的文件（控制台/容器日志仅打印一次）
export FIXIMG_API_TOKEN=$(cat admin_data/api_token.txt)
AUTH="Authorization: Bearer $FIXIMG_API_TOKEN"

# 创建任务（异步；由推理流水线执行）
curl -X POST http://127.0.0.1:9502/api/v1/tasks -H "$AUTH" -F type=restore -F image=@photo.png

# 带 options（JSON 对象；可用键：hr、face_enhance、auto_colorize，§12 已接线生效）
curl -X POST http://127.0.0.1:9502/api/v1/tasks \
    -H "$AUTH" -F type=restore -F image=@photo.png -F 'options={"face_enhance": true}'

# 一键自动修复（方案 §10/§11 Phase 7）：由 ImageAnalyzer 依据照片特征
# （黑白/划痕/人脸/模糊）自动编排流水线
curl -X POST http://127.0.0.1:9502/api/v1/tasks -H "$AUTH" -F type=auto_restore -F image=@photo.png

# 有 Ground Truth 评估（方案 §16）：附上参考照片即可获得与其对比的 LPIPS
# （安装了 lpips 包用 VGG 真实模型，否则使用拉普拉斯金字塔代理）
curl -X POST http://127.0.0.1:9502/api/v1/tasks \
    -H "$AUTH" -F type=restore -F image=@photo.png -F ground_truth=@reference.png

# 轮询状态 / 进度 / 当前阶段。restore / restore_scratch / auto_restore 任务的
# 报告中还会包含无参考质量指标（方案 §16）与人脸身份保持（方案 §17：
# Face Count / Enhanced Faces / Identity Similarity）。
curl -H "$AUTH" http://127.0.0.1:9502/api/v1/tasks/<task_id>

# 取消排队中的任务 / 下载结果 / 下载阶段报告
curl -X POST -H "$AUTH" http://127.0.0.1:9502/api/v1/tasks/<task_id>/cancel
curl -O -H "$AUTH" http://127.0.0.1:9502/api/v1/tasks/<task_id>/result
curl -H "$AUTH" http://127.0.0.1:9502/api/v1/tasks/<task_id>/report

# 性能统计：成功率 + 各阶段 P50/P95 + 模型加载耗时 / GPU 显存峰值（方案 §30）
curl -H "$AUTH" 'http://127.0.0.1:9502/api/v1/stats?days=7'
```

### 5.4 GPU Worker 部署

提交到 `/api/v1/tasks` 的任务由 GPU worker 消费，两种拓扑：

- **内联（默认）**：API 进程内以线程运行 worker（`FIXIMG_INLINE_WORKER=true`），单机部署最简单；
- **独立进程**：设 `FIXIMG_INLINE_WORKER=false` 并单独运行 worker 进程；模型只在 worker 进程加载一次，与 HTTP 服务进程数无关：

```bash
FIXIMG_INLINE_WORKER=false python main.py   # 终端 1：仅 API
python worker.py                            # 终端 2：GPU worker
```

`GET /health/ready` 会报告当前拓扑、队列深度与 worker 状态；worker 崩溃遗留的 running 任务约每分钟检查一次并自动重新入队。

### 5.5 V1 → V2 数据迁移

```bash
python -m scripts.migrate_v1                               # dry-run：仅报告，不写入
python -m scripts.migrate_v1 --apply                       # 执行迁移
python -m scripts.migrate_v1 --apply --register-artifacts  # 同时登记输入/输出文件
```

脚本将旧版 `history` 记录复制到 V2 `tasks` 表（psnr/ssim/mae 转为 `input_output_difference` 参照类型的指标记录）。脚本幂等，可重复执行。

### 5.6 命令行批量处理

```bash
# 不带划痕的旧照片复原
python run.py --input_folder ./test_images/old --output_folder ./output

# 带划痕的旧照片复原
python run.py --input_folder ./test_images/old_w_scratch --output_folder ./output --with_scratch

# 高清人脸增强（可选）
python run.py --input_folder ./test_images/old --output_folder ./output --HR

# 显式指定 GPU / CPU
python run.py --input_folder ./test_images/old --output_folder ./output --GPU 0
```

输出目录包含 `final_output/`（最终结果）、各阶段中间产物与 `pipeline_report.json`（降级报告）。

### 5.7 常用环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `FIXIMG_HOST` | `127.0.0.1` | 监听地址（局域网访问设为 `0.0.0.0`） |
| `FIXIMG_PORT` | `9502` | 监听端口 |
| `FIXIMG_DEVICE` | `auto` | 推理设备（`auto`/`cuda`/`cpu`） |
| `FIXIMG_MAX_IMAGE_SIDE` | `4096` | 接受的图片长边上限（像素） |
| `FIXIMG_INLINE_WORKER` | `true` | 在 API 进程内运行队列 worker（`false` = 独立 `python worker.py`） |
| `FIXIMG_WORKER_POLL` | `0.5` | worker 队列轮询间隔（秒） |
| `FIXIMG_WORKER_MAX_QUEUE` | `100` | 队列积压上限，超限后新提交返回 503（`0` = 不限制） |
| `FIXIMG_HAS_EXTERNAL_WORKER` | `false` | 有独立 worker 消费队列时设为 `true`，Gradio UI 走异步路径 |
| `FIXIMG_MAX_UPLOAD_MB` | `10` | API 上传大小上限（超过返回 HTTP 413） |
| `FIXIMG_RESULT_TTL` | `7200` | 推理产物保留秒数 |
| `FIXIMG_HISTORY_MAX` | `2000` | 历史记录上限 |
| `FIXIMG_ARCHIVE_TTL` | `604800` | 归档照片保留秒数 |
| `FIXIMG_DDCOLOR_MODEL` | `weights/ddcolor/pytorch_model.pt` | DDColor 权重路径 |
| `FIXIMG_COLORIZE_TTL` | `7200` | 上色结果保留秒数 |
| `FIXIMG_DDCOLOR_INPUT_SIZE` | `512` | DDColor 输入尺寸 |
| `FIXIMG_DDCOLOR_MODEL_SIZE` | `large` | DDColor 模型规格（large/medium） |
| `FIXIMG_TILE_SIZE` / `FIXIMG_TILE_OVERLAP` | `1536` / `128` | 高分辨率分块推理（方案 §18）：长边超过 `FIXIMG_TILE_SIZE` 的图片自动切块推理后羽化合并，`0` 表示禁用 |
| `FIXIMG_REGISTER_MAX` / `_GLOBAL_MAX` / `_USERNAME_MAX` | `5/20/3` | 注册限流阈值 |
| `FIXIMG_REGISTER_WINDOW` | `600` | 注册限流窗口秒数 |
| `FIXIMG_TRUSTED_PROXIES` | 空 | 允许读取 `X-Forwarded-For` 的代理 IP，逗号分隔。默认不信任转发头，仅当对端地址在此白名单内时使用 |
| `FIXIMG_AUTO_GRAYSCALE_SAT` | `16` | Auto Restore：平均饱和度低于该值视为黑白照片 |
| `FIXIMG_AUTO_SHARP_LAPLACIAN` | `120` | Auto Restore：拉普拉斯方差达到该值视为清晰 |
| `FIXIMG_AUTO_SCRATCH_THRESHOLD` | `0.5` | Auto Restore：划痕得分达到该值启用划痕修复 |
| `FIXIMG_AUTO_BLUR_THRESHOLD` | `0.5` | Auto Restore：模糊得分达到该值标记为模糊输入 |
| `FIXIMG_IDENTITY_BACKEND` | `auto` | 人脸身份保持指标后端（方案 §17）：`auto`（有 dlib 权重则用 ResNet，否则轻量回退）、`fallback`、`off` |
| `FIXIMG_ADMIN_USERNAME` | `admin` | 引导创建的管理员账号名（方案 §20） |
| `FIXIMG_API_TOKEN` | 自动生成（0600 文件） | 所有 `/api/v1/*` 接口所需的 Bearer token。未设置时首次启动自动生成、打印一次并写入 `admin_data/api_token.txt`；健康探针保持公开 |
| `FIXIMG_REDIS_URL` | 空 | 可选 Redis 连接（方案 §22/P3）。设置后（且安装 `redis` 包）注册限流跨进程共享；留空保持进程内限流 |

`GET /health` 返回存活状态；`GET /health/ready` 额外检查 SQLite 连接以及模型管理器与 worker/队列状态，数据库异常时返回 HTTP 503。`GET /api/v1/stats?days=N` 聚合任务成功率与各阶段 P50/P95 耗时。

Gradio 界面同样走异步路径：提交按钮入队任务并以阶段进度条流式展示（方案 §23/§24）。当没有可用的队列消费者时，UI 自动回退到同步执行以保持 V1 体验；运行独立 `worker.py` 时请设置 `FIXIMG_HAS_EXTERNAL_WORKER=true` 让 UI 使用队列。

## 6. 输入输出示例

以下示例均由当前系统实际生成（GPU 环境）。

### 6.1 不带划痕的旧照片复原

输入（退化旧照）→ 输出（修复结果）：

| 输入 | 输出 |
| --- | --- |
| ![复原输入](docs/examples/restore_input.png) | ![复原输出](docs/examples/restore_output.png) |

### 6.2 划痕检测

输入（带划痕照片）→ 输出（划痕 mask，白色为划痕）：

| 输入 | 输出 |
| --- | --- |
| ![检测输入](docs/examples/detect_input.png) | ![检测 mask](docs/examples/detect_mask.png) |

### 6.3 老照片上色

输入（黑白灰度照片）→ 输出（DDColor 上色结果）：

| 输入 | 输出 |
| --- | --- |
| ![上色输入](docs/examples/colorize_input.jpg) | ![上色输出](docs/examples/colorize_output.png) |

更多测试样例见 `examples/`（`old/`、`old_w_scratch/`、`color/`）与 `test_images/`。

## 7. 项目结构

```
Face_Detection/      # dlib 68 点人脸关键点检测与回卷
Face_Enhancement/    # 渐进式人脸增强模型
Global/              # 整体质量修复与划痕检测模型
app/                 # 分层应用：api / ui / services / inference / repositories / core / schemas
basicsr/             # DDColor 所需 BasicSR 最小子集
config/              # 安全与限流（ratelimit / weights_check / users）
ddcolor/             # DDColor 上色模型
docs/                # 示例与展示图片
examples/            # Web 示例图片
scripts/             # 安装、权重下载与 V1→V2 数据迁移脚本
storage/tasks/       # V2 产物存储：<年>/<月>/<任务ID>/{input,stages,output,report.json}
test_images/         # CLI 测试图片
Dockerfile           # NVIDIA CUDA 12.8 容器镜像
docker-compose.yml   # 双服务部署（API + GPU worker）
LICENSE              # 本项目 MIT 许可
LICENSE-Bringing-Old-Photos-Back-to-Life  # BOB 模型 MIT 许可
README.md            # 项目文档（英文主文档）
README_CN.md         # 项目文档（中文备份）
THIRD_PARTY_NOTICES.md  # 第三方组件与许可
main.py              # Web 服务入口（瘦身；装配逻辑在 app/factory.py）
worker.py            # 独立 GPU worker 进程（数据库任务队列）
requirements.lock    # pip freeze 锁定文件
requirements.txt     # Python 依赖
run.py               # 四阶段推理流水线 CLI
.gitignore           # Git 忽略清单
.dockerignore        # Docker 构建排除清单
```

> 以上为顶层目录与文件；完整树形结构与逐文件说明见仓库源码。

### 测试

```bash
python -m pytest                 # 单元 + 集成（gpu 测试自动跳过）
python -m pytest -m "not gpu"    # 显式排除 GPU 测试（CI 默认）
python -m pytest -m gpu          # 仅跑 GPU 测试（需 CUDA + DDColor 权重）
ruff check app scripts tests main.py worker.py
```

评估输出覆盖方案 §16 的各指标族：输入-输出差异（PSNR/SSIM/MAE）、无参考指标（清晰度/对比度/亮度/噪声）、可选自然场景统计（NIQE、BRISQUE 风格特征、色彩统计）、人脸身份保持（§17），以及附参考照片时的 Ground-Truth LPIPS（§16）。
## 8. 常见问题

**Q1：启动提示"权重完整性校验失败"**
权重缺失或哈希不符。运行 `python -m scripts.download_weights download` 补齐，然后执行 `python -m scripts.download_weights generate` 重新生成清单（本地权重与仓库清单不一致时同样处理）。

**Q2：CUDA 不可用 / 提示 sm_120 不兼容**
请确认安装的是 torch 2.7.1+cu128 及以上（RTX 50 系需要 cu128 构建）；老版本 cu121 不支持 Blackwell 架构。

**Q3：老照片上色首次很慢**
首次调用需要加载 DDColor 权重（约 3 秒），之后每张约 1 秒（GPU）。

**Q4：Windows 控制台中文乱码**
部分终端默认 GBK 编码，运行 Python 前可执行 `set PYTHONIOENCODING=utf-8`。

**Q5：想从局域网访问**
以 `FIXIMG_HOST=0.0.0.0` 启动，并确保防火墙放行 `FIXIMG_PORT`。

## 9. 第三方组件与许可

- **本项目自身代码**：MIT License，见根目录 [LICENSE](LICENSE)；
- **Bringing Old Photos Back to Life**（修复/检测/面部增强模型）：MIT License，见 [LICENSE-Bringing-Old-Photos-Back-to-Life](LICENSE-Bringing-Old-Photos-Back-to-Life)；
- **DDColor**（老照片上色，ICCV 2023）：Apache-2.0，见 [ddcolor/LICENSE](ddcolor/LICENSE)；
- 完整说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
