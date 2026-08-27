<!--
  fix_old_img — Deep Learning Old Photo Restoration, Scratch Repair & Colorization
  English version; 中文版请查看 README.md
-->
<div align="center">

# Deep Learning Based Old Photo Restoration, Scratch Repair & Colorization

**基于深度学习的旧照片恢复、划痕修复与老照片上色系统**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1%2Bcu128-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.22-orange?logo=gradio&logoColor=white)](https://gradio.app/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[中文](./README.md) | **English**

I built a web-based image restoration system for real old photos using Python 3.11 and PyTorch (CUDA 12.8). It supports overall quality restoration, scratch detection & repair, face enhancement, and automatic colorization of black-and-white photos. For the web layer I used Gradio 6 + FastAPI + Uvicorn, with SQLite (WAL) for data storage, plus a complete admin panel (task history, photo archive, user management).

![Restoration showcase](docs/upstream/bob-0001.jpg)

</div>

---

## 📑 Table of Contents

- [1. Problem](#1-problem)
- [2. Features](#2-features)
- [3. Technologies I Used](#3-technologies-i-used)
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

To address these, I implemented a complete pipeline: **overall quality restoration → scratch detection/repair → face detection & enhancement → warp-back compositing**, and integrated **DDColor** colorization in the same platform, together with user login/registration, task history, photo archives and user management.

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
- Health check endpoints `GET /health` (liveness) and `GET /health/ready` (database readiness, HTTP 503 on failure);
- Weight integrity self-check at startup (SHA-256 manifest; refuses to start if missing or tampered);
- Per-request directory isolation + result TTL reclamation;
- SQLite storage (users/history), automatic migration from legacy users.yaml / JSONL on first start;
- Dark-themed UI with automatic admin/user interface differentiation.

## 3. Technologies I Used

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

My restoration chain (overall quality restoration / scratch detection & repair / face enhancement) uses the following techniques:

**Global Restoration**

> A triplet domain translation network is used to solve both structured degradation and unstructured degradation of old photos:

- Train **VAE** (variational autoencoder) models for domain A (degraded old photos) and domain B (high-quality new photos) respectively, sharing the latent space structure;
- Train an inter-domain **mapping network** that translates degraded-domain latents into the high-quality domain, achieving overall quality enhancement;
- The mapping network supports multiple training variants: `mapping_quality` (no scratches), `mapping_scratch` (with scratches), `mapping_Patch_Attention` (Multi-Scale Patch Attention for high-resolution scratch repair);
- Training uses a `pix2pixHD`-style dual-discriminator GAN architecture, with `--l2_feat / --use_l1_feat / --NL_res` (non-local residual) options controlling loss and structure.

<table align="center"><tr>
  <td><img src="docs/upstream/bob-pipeline.png" alt="Restoration pipeline architecture" width="400"/></td>
  <td><img src="docs/upstream/bob-global.png" alt="Global restoration comparison" width="400"/></td>
</tr><tr>
  <td align="center">Restoration pipeline architecture</td>
  <td align="center">Global restoration comparison</td>
</tr></table>

**Scratch Detection**

> The scratch detection model is trained with labeled data and outputs a binary mask (white = scratches). For high-resolution inputs, the scratch-repair chain uses non-local mapping with Multi-Scale Patch Attention to recover a clean image from heavily cracked photos.

| Original | Restored |
| --- | --- |
| ![Scratch detection](docs/upstream/bob-scratch-detection.png) | ![High-resolution scratch repair](docs/upstream/bob-hr-result.png) |

**Face Detection & Face Enhancement**

> A **progressive generator** is used to refine the face regions of old photos:

- Face detection uses dlib's `shape_predictor_68_face_landmarks.dat` (68-point landmark detector);
- Detected faces are cropped, aligned, fed to the face enhancement model, then **warped back** into the original image according to the original geometry;
- The face enhancement model uses Synchronized-BatchNorm-PyTorch and a progressive encoder structure, refining faces step by step via instance-norm parameter modulation.

![Progressive face enhancement architecture](docs/upstream/bob-face-pipeline.png)

| Input | Enhanced |
| --- | --- |
| ![Face enhancement](docs/upstream/bob-face.png) | Facial details recovered, blur removed |

> Note: this model is pretrained at 256×256, so arbitrary resolutions may not be optimal (this project accepts inputs up to 4096px on the long side).

### 3.2 Black-and-White Photo Colorization

My automatic colorization module uses the following techniques:

> Multi-scale visual features are used to optimize **learnable color tokens** (i.e. color queries), achieving state-of-the-art performance on automatic image colorization:

- **Dual Decoders**: one decoder performs **color decoding** (interacting learnable color tokens with multi-scale features), and the other performs **image reconstruction** (recovering spatial details); together they produce photo-realistic colorization;
- **Backbone**: based on ConvNeXt (`ConvNeXt-Large`, 22k pretrained); the encoder extracts multi-scale visual features; color tokens interact with features via Transformer-style cross-attention (Mask2Former / DETR style);
- **Color queries**: a set of learnable color embeddings acting as a "color dictionary" queried by multi-scale features to obtain the target color distribution;
- The training pipeline is based on the **BasicSR** toolbox (the `basicsr/` subset is bundled), supporting four pretrained model specs: `ddcolor_paper / ddcolor_modelscope / ddcolor_artistic / ddcolor_paper_tiny`;
- My implementation defaults to the `damo/cv_ddcolor_image-colorization` weights at `weights/ddcolor/pytorch_model.pt`, model size `large`, input size 512×512 (both adjustable via environment variables).

![Colorization network architecture](docs/upstream/ddcolor-network-arch.jpg)

| B&W photo colorization | Anime/game scene colorization |
| --- | --- |
| ![Colorization showcase](docs/upstream/ddcolor-teaser.webp) | ![Anime scene colorization](docs/upstream/ddcolor-anime.webp) |

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
bash scripts/download_weights.sh

# 6. If weights differ from the repo manifest, regenerate and verify
python -m config.weights_check generate
python -m config.weights_check verify
```

> Note: `config/users.yaml` is automatically migrated to SQLite (`admin_data/fixoldimg.db`). If this file is missing, no accounts exist after startup; restore from backup or create users via the admin panel.

### 4.2 Docker (Linux + NVIDIA GPU)

```bash
docker build -t fixoldimg .
docker run --gpus all -p 9502:9502 fixoldimg
```

The image installs dependencies, downloads all weights, creates a default admin account (`admin/admin123`) and rebuilds the weight manifest. dlib is compiled from source on first build (about 5–10 minutes).

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

### 5.3 CLI Batch Processing

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

### 5.4 Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `FIXIMG_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` for LAN/container) |
| `FIXIMG_PORT` | `9502` | Listen port |
| `FIXIMG_RESULT_TTL` | `7200` | Inference result retention (seconds) |
| `FIXIMG_HISTORY_MAX` | `2000` | History record cap |
| `FIXIMG_ARCHIVE_TTL` | `604800` | Archived photo retention (seconds) |
| `FIXIMG_DDCOLOR_MODEL` | `weights/ddcolor/pytorch_model.pt` | DDColor weight path |
| `FIXIMG_COLORIZE_TTL` | `7200` | Colorization result retention (seconds) |
| `FIXIMG_DDCOLOR_INPUT_SIZE` | `512` | DDColor input size |
| `FIXIMG_DDCOLOR_MODEL_SIZE` | `large` | DDColor model size (large/medium) |
| `FIXIMG_REGISTER_MAX` / `_GLOBAL_MAX` / `_USERNAME_MAX` | `5/20/3` | Registration rate limits |
| `FIXIMG_REGISTER_WINDOW` | `600` | Registration rate-limit window (seconds) |
| `FIXIMG_TRUSTED_PROXIES` | empty | Proxy IPs allowed to read `X-Forwarded-For`, comma-separated. Forwarded headers are untrusted by default; used only when the peer address is in this allowlist |

`GET /health` returns liveness; `GET /health/ready` additionally checks the SQLite connection and returns HTTP 503 on database errors.

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

```text
.
├── main.py                  # Web service entry (Gradio 6 + FastAPI)
├── run.py                   # Four-stage inference pipeline CLI
├── app/                     # Application layer: db(SQLite)/pipeline/colorizer/admin_panel
├── config/                  # users.yaml / rate limiting / security / weight manifest
├── ddcolor/                 # DDColor colorization model
├── basicsr/                 # Minimal BasicSR subset for DDColor
├── Global/                  # Overall quality restoration / scratch detection models
├── Face_Detection/          # dlib face detection / warp-back
├── Face_Enhancement/        # Face enhancement model
├── examples/                # Web example images (old / old_w_scratch / color)
├── test_images/             # CLI test images
├── docs/examples/           # README input/output examples
├── docs/upstream/           # Algorithm showcase images
├── weights/ddcolor/         # DDColor weights (not in repo)
├── scripts/                 # Installation scripts & weight download
├── Dockerfile
└── LICENSE / README.md / README_EN.md / THIRD_PARTY_NOTICES.md
```

## 8. FAQ

**Q1: "Weight integrity check failed" at startup**
Weights are missing or hashes mismatch. Run `bash scripts/download_weights.sh` to fetch them, then run `python -m config.weights_check generate` to regenerate the manifest (also needed when local weights differ from the repo manifest).

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

---

<div align="center">

*Xu Kang*

<img src="docs/upstream/signature.png" alt="Author signature" width="200"/>

</div>
