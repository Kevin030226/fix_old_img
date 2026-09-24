<!--
  fix_old_img — Deep Learning Old Photo Restoration, Scratch Repair & Colorization
  English (default). See README_CN.md for the Chinese version.
-->
<div align="center">

# Deep Learning Based Old Photo Restoration, Scratch Repair & Colorization

**Old Photo Restoration, Scratch Repair & Colorization System**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1%2Bcu128-ee4c2c?logo=pytorch&logoColor=white)](https://www.pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.22-orange?logo=gradio&logoColor=white)](https://gradio.app/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**English** | [中文](./README_CN.md)

<p align="center">
  <img src="docs/upstream/bob-0001.jpg" width="100%">
</p>

This project is a web-based image restoration system for real old photos, built with Python 3.11 and PyTorch (CUDA 12.8). It supports overall quality restoration, scratch detection & repair, face enhancement, and automatic colorization of black-and-white photos. The web layer uses Gradio 6 + FastAPI + Uvicorn, data storage uses SQLite (WAL), and a complete admin panel is included (task history, photo archive, user management).

</div>

---

## 📑 Table of Contents

- [1. Problem](#1-problem)
- [2. Features](#2-features)
- [3. Technologies Used](#3-technologies-used)
- [4. Installation](#4-installation)
- [5. Usage](#5-usage)
- [6. Input/Output Examples](#6-inputoutput-examples)
- [7. Project Structure](#7-project-structure)
- [8. FAQ](#8-faq)
- [9. Third-Party Components & Licenses](#9-third-party-components--licenses)

---

## 1. Problem

Old photos typically suffer from the following issues during storage and scanning:

- **Overall degradation**: fading, blur, noise, low contrast;
- **Physical damage**: creases, scratches, stains;
- **Monochrome**: early photos only have grayscale information, lacking natural color;
- **Lost facial detail**: face regions are severely degraded and hard to restore;
- **Invisible processing**: batch processing lacks task records, result archiving and quality metrics.

This system provides a complete pipeline to address these issues: **overall quality restoration → scratch detection/repair → face detection & enhancement → warp-back compositing**, and integrates **DDColor** colorization in the same platform, together with user login/registration, task history, photo archives and user management.

## 2. Features

### 2.1 Image Processing Modules (Four Tabs)

| Module | Description | Output |
| --- | --- | --- |
| Restoration (no scratches) | Overall quality enhancement + face detection & enhancement | Restored color photo + PSNR/SSIM/MAE metrics |
| Restoration (with scratches) | Auto scratch detection → scratch repair → quality enhancement → face enhancement | Restored color photo + metrics |
| Scratch detection | Output scratch locations only | Binary mask (white = scratches) |
| Old photo colorization | DDColor auto colorization of B&W/grayscale photos | Color photo |

### 2.2 Admin Panel (Admin Only)

- **Task management**: processing history, statistics overview (tasks/users/per-type counts/average metrics), clear records;
- **Photo archive**: browse original images and restoration results by record;
- **User management**: add, edit (password/role), delete users.

### 2.3 Platform Capabilities

- User login (PBKDF2 hash, constant-time comparison) and public registration (IP/global/username triple rate limiting);
- Health check endpoints `GET /health` (liveness) and `GET /health/ready` (database readiness + model manager state, HTTP 503 on failure);
- V2 task API `POST/GET /api/v1/tasks` (queued execution, progress, cancel, result & report download; every `/api/v1/*` route requires a bearer token);
- DB-backed task queue with a dedicated GPU worker: inline thread by default (`FIXIMG_INLINE_WORKER=true`), or a standalone `python worker.py` process for the api+worker docker-compose topology; tasks left running by a crashed worker are requeued automatically;
- `scripts/migrate_v1.py` migrates legacy `history` rows into the V2 `tasks` tables (dry-run by default, `--apply` to write);
- Weight integrity self-check at startup (SHA-256 manifest; refuses to start if missing or tampered);
- Structured run directories under `storage/tasks/` with `report.json` per run;
- SQLite storage (users/history + V2 tasks/task_stages/artifacts/metrics), automatic migration from legacy users.yaml / JSONL on first start;
- Dark-themed UI with automatic admin/user interface differentiation.

## 3. Technologies Used

| Category | Technology |
| --- | --- |
| Language/Environment | Python 3.11 (conda env `fixoldimg-gpu`) |
| Web framework | Gradio 6.22 + FastAPI 0.141 + Uvicorn 0.52 |
| Deep learning | PyTorch 2.7.1+cu128 (native RTX 50 series sm_120 support) |
| Vision/Scientific computing | OpenCV 5.0, scikit-image 0.26, scipy 1.17, numpy 2.4, Pillow 12.3 |
| Face/Colorization | dlib 20.0.1 (conda-forge), timm 0.9.2, DDColor |
| Storage | SQLite (WAL mode) |
| Models | Bringing Old Photos Back to Life + DDColor, 29 weight files in total |

### 3.1 Overall Quality Restoration, Scratch Detection & Face Enhancement

The restoration chain (overall quality restoration / scratch detection & repair / face enhancement) uses the following techniques:

**Global Restoration**

> A triplet domain translation network is used to solve both structured degradation and unstructured degradation of old photos:

- Train **VAE** (variational autoencoder) models for domain A (degraded old photos) and domain B (high-quality new photos) respectively, sharing the latent space structure;
- Train an inter-domain **mapping network** that translates degraded-domain latents into the high-quality domain, achieving overall quality enhancement;
- The mapping network supports multiple training variants: `mapping_quality` (no scratches), `mapping_scratch` (with scratches), `mapping_Patch_Attention` (Multi-Scale Patch Attention for high-resolution scratch repair);
- Training uses a `pix2pixHD`-style dual-discriminator GAN architecture, with `--l2_feat / --use_l1_feat / --NL_res` (non-local residual) options controlling loss and structure.

<p align="center">
  <img src="docs/upstream/bob-pipeline.png" width="60%">
</p>

<p align="center">
  <img src="docs/upstream/bob-global.png" width="100%">
</p>

<p align="center">
  <img src="docs/upstream/bob-scratch-detection.png" width="100%">
</p>

<p align="center">
  <img src="docs/upstream/bob-hr-result.png" width="100%">
</p>

**Scratch Detection**

> The scratch detection model is trained with labeled data and outputs a binary mask (white = scratches). For high-resolution inputs, the scratch-repair chain uses non-local mapping with Multi-Scale Patch Attention to recover a clean image from heavily cracked photos.

**Face Detection & Face Enhancement**

> A **progressive generator** is used to refine the face regions of old photos:

- Face detection uses dlib's `shape_predictor_68_face_landmarks.dat` (68-point landmark detector);
- Detected faces are cropped, aligned, fed to the face enhancement model, then **warped back** into the original image according to the original geometry;
- The face enhancement model uses Synchronized-BatchNorm-PyTorch and a progressive encoder structure, refining faces step by step via instance-norm parameter modulation.

<p align="center">
  <img src="docs/upstream/bob-face-pipeline.png" width="70%">
</p>

<p align="center">
  <img src="docs/upstream/bob-face.png" width="100%">
</p>

> Note: this model is pretrained at 256×256, so arbitrary resolutions may not be optimal (this system accepts inputs up to 4096px on the long side).

### 3.2 Black-and-White Photo Colorization

The automatic colorization module uses the following techniques:

> Multi-scale visual features are used to optimize **learnable color tokens** (i.e. color queries), achieving state-of-the-art performance on automatic image colorization:

- **Dual Decoders**: one decoder performs **color decoding** (interacting learnable color tokens with multi-scale features), and the other performs **image reconstruction** (recovering spatial details); together they produce photo-realistic colorization;
- **Backbone**: based on ConvNeXt (`ConvNeXt-Large`, 22k pretrained); the encoder extracts multi-scale visual features; color tokens interact with features via Transformer-style cross-attention (Mask2Former / DETR style);
- **Color queries**: a set of learnable color embeddings acting as a "color dictionary" queried by multi-scale features to obtain the target color distribution;
- The training pipeline is based on the **BasicSR** toolbox (the `basicsr/` subset is bundled), supporting four pretrained model specs: `ddcolor_paper / ddcolor_modelscope / ddcolor_artistic / ddcolor_paper_tiny`;
- This system defaults to the `damo/cv_ddcolor_image-colorization` weights at `weights/ddcolor/pytorch_model.pt`, model size `large`, input size 512×512 (both adjustable via environment variables).

<p align="center">
  <img src="docs/upstream/ddcolor-network-arch.jpg" width="100%">
</p>

<p align="center">
  <img src="docs/upstream/ddcolor-teaser.webp" width="100%">
</p>

<p align="center">
  <img src="docs/upstream/ddcolor-anime.webp" width="100%">
</p>

## 4. Installation

### 4.1 Windows (conda)

One-click script:

```bat
scripts/setup_gpu.bat
```

Or manually:

```bash
# 1. Create environment
conda create -n fixoldimg-gpu python=3.11 -y
conda activate fixoldimg-gpu

# 2. Install PyTorch cu128 (RTX 50 series support; use the SJTU mirror in China if needed)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# 3. Install project dependencies (Gradio/FastAPI/OpenCV/timm for DDColor, etc.)
pip install -r requirements.txt

# 4. Install prebuilt dlib from conda-forge (no local compiler toolchain needed)
conda install -n fixoldimg-gpu -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
    --override-channels -y dlib=20.0.1

# 5. Download model weights (BOB restoration chain + DDColor, ~1.5GB)
#    (plan §19: python entrypoints, resume support, manifest with versions)
python -m scripts.download_weights download

# 6. If weights differ from the repo manifest, regenerate and verify
python -m scripts.download_weights generate
python -m scripts.verify_weights
```

> Note: `config/users.yaml` is automatically migrated to SQLite (`admin_data/fixoldimg.db`). If this file is missing, the first-boot admin bootstrap (§21) creates the initial `admin` account — see the Docker section above.

### 4.2 Docker (Linux + NVIDIA GPU)

```bash
docker build -t fixoldimg .
docker run --gpus all -p 9502:9502 fixoldimg
```

Or the V2 two-service topology (the API only enqueues; a separate worker process owns the GPU and models):

```bash
docker compose up -d --build   # api + worker (see docker-compose.yml)
docker compose logs -f worker
```

The image installs dependencies, downloads all weights and rebuilds the weight manifest. dlib is compiled from source on first build (about 5–10 minutes).

**Admin credentials (plan §21):** the image no longer contains any fixed password. On the first start the app reads `FIXIMG_ADMIN_PASSWORD` when provided (e.g. from a Docker secret); otherwise it generates a random one-time password, prints it **once** in the container log and stores it at `admin_data/initial_admin_password.txt` (mode 0600). Change it immediately after the first login. Set `FIXIMG_AUTO_BOOTSTRAP_ADMIN=false` to disable this bootstrap entirely.

**Forced password change (plan §21):** the randomly generated admin account is created with a `must_change_password` flag. While the flag is set, task submission is blocked with a clear message, the Admin Panel shows an "Account" tab with a change-password form, and until the password is replaced the banner keeps prompting. Saving a new password clears the flag; passwords injected via `FIXIMG_ADMIN_PASSWORD` are treated as deployment-managed and do not force a change.

## 5. Usage

### 5.1 Start the Web Service

```bash
conda activate fixoldimg-gpu
python main.py
```

Open <http://127.0.0.1:9502> in your browser.

Default accounts (**change before deployment**):

| Username | Password | Role |
| --- | --- | --- |
| admin | set during install/init | Admin (admin panel only) |
| user1 | set during install/init | User |

> ⚠️ Demo credentials (e.g. `admin/admin123`) are kept for local development only;
> rotate all account passwords via "Admin Panel → User Management" before public deployment.

Admin users only see the "Admin Panel"; normal users can use the four image processing tabs.

### 5.2 Web Workflow

1. Log in (or click "Register Now" to create an account);
2. Select the desired tab;
3. Upload an image (or click an example below the tab to auto-fill);
4. Click "Start Restoration / Submit / Start Colorization";
5. Wait for completion (about 1–60 s per image on GPU depending on module and size), inspect results and metrics;
6. Admins can view task history, photo archives and manage users in the "Admin Panel".

### 5.3 V2 Task API

Every `/api/v1/*` endpoint requires the API token as a bearer credential
(`/health` and `/health/ready` stay public for probes). The Gradio UI is not
affected: it calls the service layer in process.

```bash
# The token comes from FIXIMG_API_TOKEN, or from the file generated on first
# start (printed once in the console / container log).
export FIXIMG_API_TOKEN=$(cat admin_data/api_token.txt)
AUTH="Authorization: Bearer $FIXIMG_API_TOKEN"

# Create a task (queued; executed by the inference pipeline asynchronously)
curl -X POST http://127.0.0.1:9502/api/v1/tasks -H "$AUTH" -F type=restore -F image=@photo.png

# With options (JSON object; known keys: hr, face_enhance, auto_colorize)
curl -X POST http://127.0.0.1:9502/api/v1/tasks \
    -H "$AUTH" -F type=restore -F image=@photo.png -F 'options={"hr": true}'

# Auto Restore (plan §10/§11 Phase 7): the ImageAnalyzer derives the pipeline
# from the photo (grayscale / scratches / faces / blur) automatically
curl -X POST http://127.0.0.1:9502/api/v1/tasks -H "$AUTH" -F type=auto_restore -F image=@photo.png

# Ground Truth evaluation (plan §16): attach a reference photo to get LPIPS
# against it (real lpips/VGG when installed, Laplacian surrogate otherwise)
curl -X POST http://127.0.0.1:9502/api/v1/tasks \
    -H "$AUTH" -F type=restore -F image=@photo.png -F ground_truth=@reference.png

# Poll status / progress / current stage. For restore / restore_scratch /
# auto_restore tasks the report also carries no-reference quality indicators
# (plan §16) and face identity preservation (plan §17: Face Count / Enhanced
# Faces / Identity Similarity).
curl -H "$AUTH" http://127.0.0.1:9502/api/v1/tasks/<task_id>

# Cancel a queued task / fetch result / fetch stage report
curl -X POST -H "$AUTH" http://127.0.0.1:9502/api/v1/tasks/<task_id>/cancel
curl -O -H "$AUTH" http://127.0.0.1:9502/api/v1/tasks/<task_id>/result
curl -H "$AUTH" http://127.0.0.1:9502/api/v1/tasks/<task_id>/report

# Performance stats: success rate + per-stage P50/P95 + model load times /
# GPU memory peak (plan section 30)
curl -H "$AUTH" 'http://127.0.0.1:9502/api/v1/stats?days=7'
```

### 5.4 GPU Worker Deployment

Tasks submitted to `/api/v1/tasks` are consumed by the GPU worker. Two topologies:

- **Inline (default)** — the API process runs a worker thread (`FIXIMG_INLINE_WORKER=true`); simplest for single-machine use.
- **Standalone** — set `FIXIMG_INLINE_WORKER=false` and run a separate process; models load once in the worker regardless of how many uvicorn workers serve HTTP:

```bash
FIXIMG_INLINE_WORKER=false python main.py   # terminal 1: API only
python worker.py                            # terminal 2: GPU worker
```

`GET /health/ready` reports the active topology, queue depth and worker state; running tasks left behind by a crashed worker are requeued automatically (checked about once a minute).

### 5.5 V1 → V2 Data Migration

```bash
python -m scripts.migrate_v1                               # dry-run: report only, no writes
python -m scripts.migrate_v1 --apply                       # perform the migration
python -m scripts.migrate_v1 --apply --register-artifacts  # also register input/output files
```

The script copies legacy `history` rows into the V2 `tasks` table (psnr/ssim/mae become metric records with the `input_output_difference` reference type). It is idempotent and safe to re-run.

### 5.6 CLI Batch Processing

```bash
# Restoration without scratches
python run.py --input_folder ./test_images/old --output_folder ./output

# Restoration with scratches
python run.py --input_folder ./test_images/old_w_scratch --output_folder ./output --with_scratch

# High-resolution face enhancement (optional)
python run.py --input_folder ./test_images/old --output_folder ./output --HR

# Explicit GPU / CPU
python run.py --input_folder ./test_images/old --output_folder ./output --GPU 0
```

The output directory contains `final_output/` (final results), per-stage intermediates and `pipeline_report.json` (degradation report).

### 5.7 Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `FIXIMG_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` for LAN/container) |
| `FIXIMG_PORT` | `9502` | Listen port |
| `FIXIMG_DEVICE` | `auto` | Inference device (`auto`/`cuda`/`cpu`) |
| `FIXIMG_MAX_IMAGE_SIDE` | `4096` | Max accepted long-side pixels |
| `FIXIMG_INLINE_WORKER` | `true` | Run the DB-queue worker inside the API process (`false` = standalone `python worker.py`) |
| `FIXIMG_WORKER_POLL` | `0.5` | Worker queue polling interval (seconds) |
| `FIXIMG_WORKER_MAX_QUEUE` | `100` | Max queued tasks before new submissions return 503 (`0` = unlimited) |
| `FIXIMG_HAS_EXTERNAL_WORKER` | `false` | Set `true` when a standalone worker consumes the queue, so the Gradio UI uses the async path |
| `FIXIMG_MAX_UPLOAD_MB` | `10` | API upload size limit (HTTP 413 above it) |
| `FIXIMG_RESULT_TTL` | `7200` | Inference result retention (seconds) |
| `FIXIMG_HISTORY_MAX` | `2000` | History record cap |
| `FIXIMG_ARCHIVE_TTL` | `604800` | Archived photo retention (seconds) |
| `FIXIMG_DDCOLOR_MODEL` | `weights/ddcolor/pytorch_model.pt` | DDColor weight path |
| `FIXIMG_COLORIZE_TTL` | `7200` | Colorization result retention (seconds) |
| `FIXIMG_DDCOLOR_INPUT_SIZE` | `512` | DDColor input size |
| `FIXIMG_DDCOLOR_MODEL_SIZE` | `large` | DDColor model size (large/medium) |
| `FIXIMG_TILE_SIZE` / `FIXIMG_TILE_OVERLAP` | `1536` / `128` | High-res tile inference (plan §18); images whose long side exceeds `FIXIMG_TILE_SIZE` are split into overlapping tiles and feather-blended. `0` disables tiling |
| `FIXIMG_REGISTER_MAX` / `_GLOBAL_MAX` / `_USERNAME_MAX` | `5/20/3` | Registration rate limits |
| `FIXIMG_REGISTER_WINDOW` | `600` | Registration rate-limit window (seconds) |
| `FIXIMG_TRUSTED_PROXIES` | empty | Proxy IPs allowed to read `X-Forwarded-For`, comma-separated. Forwarded headers are untrusted by default; used only when the peer address is in this allowlist |
| `FIXIMG_AUTO_GRAYSCALE_SAT` | `16` | Auto Restore: mean saturation below which a photo counts as B&W |
| `FIXIMG_AUTO_SHARP_LAPLACIAN` | `120` | Auto Restore: Laplacian variance considered fully sharp |
| `FIXIMG_AUTO_SCRATCH_THRESHOLD` | `0.5` | Auto Restore: scratch score at/above which scratch repair runs |
| `FIXIMG_AUTO_BLUR_THRESHOLD` | `0.5` | Auto Restore: blur score flagged as blurry input |
| `FIXIMG_IDENTITY_BACKEND` | `auto` | Face identity metric backend (plan §17): `auto` (dlib ResNet when weights present, else lightweight fallback), `fallback`, `off` |
| `FIXIMG_ADMIN_USERNAME` | `admin` | Bootstrap admin account name (plan §20) |
| `FIXIMG_API_TOKEN` | generated (0600 file) | Bearer token required by every `/api/v1/*` route. When unset, a token is generated on first start, printed once and stored at `admin_data/api_token.txt`. Health probes stay public |
| `FIXIMG_REDIS_URL` | empty | Optional Redis connection (plan §22, P3). When set — and the `redis` package is installed — registration rate limiting is shared across processes; empty keeps the in-process limiter |

`GET /health` returns liveness; `GET /health/ready` additionally checks the SQLite connection plus model-manager and worker/queue state, returning HTTP 503 on database errors. `GET /api/v1/stats?days=N` aggregates task success rate and per-stage P50/P95 durations.

The Gradio UI uses the async path too: submit buttons enqueue a task and stream a per-stage progress bar (plan sections 23/24). When no queue consumer is reachable, the UI automatically falls back to synchronous execution to preserve the V1 experience; set `FIXIMG_HAS_EXTERNAL_WORKER=true` when running the standalone `worker.py` so the UI uses the queue.

## 6. Input/Output Examples

All examples below were actually produced by this system (GPU environment).

### 6.1 Restoration (No Scratches)

Input (degraded old photo) → Output (restored):

| Input | Output |
| --- | --- |
| ![Restore input](docs/examples/restore_input.png) | ![Restore output](docs/examples/restore_output.png) |

### 6.2 Scratch Detection

Input (photo with scratches) → Output (scratch mask, white = scratches):

| Input | Output |
| --- | --- |
| ![Detect input](docs/examples/detect_input.png) | ![Detect mask](docs/examples/detect_mask.png) |

### 6.3 Colorization

Input (B&W grayscale photo) → Output (DDColor result):

| Input | Output |
| --- | --- |
| ![Colorize input](docs/examples/colorize_input.jpg) | ![Colorize output](docs/examples/colorize_output.png) |

More test samples are in `examples/` (`old/`, `old_w_scratch/`, `color/`) and `test_images/`.

## 7. Project Structure

```
main.py                # Web service entry (thin; wiring lives in app/factory.py)
worker.py              # Standalone GPU worker process (DB-backed task queue)
run.py                 # Four-stage pipeline CLI (legacy-compatible entry)
app/
├── api/               # FastAPI routers (auth / tasks / users / health)
├── ui/                # Gradio blocks, auth pages, HTML middleware, admin panel
├── services/          # task / artifact / evaluation / user / history services
├── inference/         # V2 core: stages / model_manager / planner / orchestrator / worker
│   └── stages/        # BaseStage + global_restore / scratch / face / colorization
├── repositories/      # SQL access (user_repository / task_repository)
├── core/              # config (Settings) / security / logging / exceptions
├── schemas/           # shared dataclasses (TaskView ...)
├── db.py              # legacy SQLite layer (users/history + V1 migration)
└── factory.py         # create_app(): FastAPI + routers + middleware + Gradio
config/                # ratelimit / security / weights_check / weights manifest
storage/tasks/         # V2 artifact storage: <year>/<month>/<task_id>/{input,stages,output,report.json}
Global/                # Overall restoration & scratch detection models
Face_Detection/        # dlib 68-point landmarks & warp-back
Face_Enhancement/      # Progressive face enhancement model
ddcolor/               # DDColor colorization model
basicsr/               # Minimal BasicSR subset for DDColor
tests/                 # unit / integration / gpu tests (pytest; gpu tests need CUDA + weights, run `pytest -m gpu`)
scripts/               # Install, weight download & V1→V2 migration scripts
docs/                  # Example & showcase images
examples/              # Web example images
test_images/           # CLI test images
Dockerfile             # NVIDIA CUDA 12.8 container image
docker-compose.yml     # Two-service deployment (API + GPU worker)
```

> Top-level entries are listed above; see the source tree for the full structure and per-file details.

### Testing

```bash
python -m pytest                 # unit + integration (gpu tests auto-skip)
python -m pytest -m "not gpu"    # explicitly exclude GPU tests (CI default)
python -m pytest -m gpu          # only GPU tests (need CUDA + DDColor weights)
ruff check app scripts tests main.py worker.py
```

Evaluation outputs include the plan §16 families: input-output difference (PSNR/SSIM/MAE), no-reference indicators (sharpness/contrast/brightness/noise), optional natural-scene statistics (NIQE, BRISQUE-style features, color statistics), face identity preservation (§17) and — when a reference photo is attached — Ground-Truth LPIPS (§16).

## 8. FAQ

**Q1: "Weight integrity check failed" at startup**
Weights are missing or hashes mismatch. Run `python -m scripts.download_weights download` to fetch them, then run `python -m scripts.download_weights generate` to regenerate the manifest (also needed when local weights differ from the repo manifest).

**Q2: CUDA unavailable / sm_120 incompatible**
Make sure torch 2.7.1+cu128 or newer is installed (RTX 50 series requires the cu128 build); older cu121 builds do not support Blackwell.

**Q3: First colorization is slow**
The first call loads DDColor weights (~3 s); afterwards about 1 s per image (GPU).

**Q4: Chinese garbled in Windows console**
Some terminals default to GBK; run `set PYTHONIOENCODING=utf-8` before Python.

**Q5: Access from LAN**
Start with `FIXIMG_HOST=0.0.0.0` and allow `FIXIMG_PORT` through the firewall.

## 9. Third-Party Components & Licenses

- **This project's own code**: MIT License, see [LICENSE](LICENSE);
- **Bringing Old Photos Back to Life** (restoration/detection/face enhancement models): MIT License, see [LICENSE-Bringing-Old-Photos-Back-to-Life](LICENSE-Bringing-Old-Photos-Back-to-Life);
- **DDColor** (old photo colorization, ICCV 2023): Apache-2.0, see [ddcolor/LICENSE](ddcolor/LICENSE);
- Full details in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
