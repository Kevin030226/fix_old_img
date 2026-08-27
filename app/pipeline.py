"""统一推理流水线。

沿用原有的 Global / Face_Detection / Face_Enhancement 推理脚本，
以子进程方式串行执行（隔离 cwd 与崩溃）；本模块负责请求隔离、
GPU 自动选择、TTL 回收、指标计算与历史入库。
"""
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

from .db import (
    ARCHIVE_INPUT_DIR,
    ARCHIVE_OUTPUT_DIR,
    append_history,
    purge_stale_archives,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(BASE_DIR, "output_img")
UPLOAD_ROOT = os.path.join(BASE_DIR, "user_upload_images")
TEMP_ROOT = os.path.join(UPLOAD_ROOT, "temp_img")
_DELETABLE_ROOTS = (OUTPUT_ROOT, TEMP_ROOT)

RESULT_TTL_SECONDS = int(os.environ.get("FIXIMG_RESULT_TTL", str(2 * 3600)))
_PIPELINE_LOCK = threading.Lock()
MAX_IMAGE_SIDE = 4096

PIPELINE_MODES = {
    "restore": {"label": "不带划痕修复", "with_scratch": False, "detect_only": False},
    "restore_scratch": {"label": "带划痕修复", "with_scratch": True, "detect_only": False},
    "detect": {"label": "划痕检测", "with_scratch": False, "detect_only": True},
}

_DEGRADE_MESSAGES = {
    "no_face_detected": "本次未检测到人脸，已跳过面部增强，结果仅为整体质量修复。",
    "face_enhance_missing": "面部增强未能产出结果，已回退为整体质量修复结果。",
}


def _is_under(path, root):
    try:
        path = os.path.realpath(path)
        root = os.path.realpath(root)
        return os.path.commonpath([path, root]) == root
    except (ValueError, OSError):
        return False


def _delete_tree(path):
    """仅允许删除白名单内的目录树。"""
    abs_path = os.path.abspath(path)
    if not any(
        _is_under(abs_path, root) or abs_path == os.path.realpath(root)
        for root in _DELETABLE_ROOTS
    ):
        print(f"[安全] 拒绝删除白名单外的路径: {abs_path}")
        return
    shutil.rmtree(abs_path, ignore_errors=True)


def purge_stale_runs(ttl_seconds=RESULT_TTL_SECONDS):
    now = time.time()
    for root in (OUTPUT_ROOT, TEMP_ROOT):
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            sub = os.path.join(root, name)
            if not os.path.isdir(sub):
                continue
            try:
                if now - os.path.getmtime(sub) > ttl_seconds:
                    shutil.rmtree(sub, ignore_errors=True)
            except OSError:
                pass


def _run_cmd(args, cwd=None):
    """以列表参数运行子进程并校验退出码（shell=False，防注入）。"""
    if args and args[0] == "python":
        args[0] = sys.executable
    proc = subprocess.run(args, shell=False, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(f"子进程失败 (exit={proc.returncode}): {' '.join(args)}")


def resolve_gpu(gpu_arg="auto"):
    """把 'auto' 解析为 0（GPU 可用）或 -1（CPU）。"""
    if str(gpu_arg).lower() != "auto":
        return int(gpu_arg)
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:  # noqa: BLE001
        return -1


# ===================== 评估指标 =====================
def _read_image(image_path):
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as f:
            image = np.array(Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        if image.shape[2] > 3:
            image = image[:, :, :3]
        return image
    except Exception:  # noqa: BLE001
        return None


def _calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def _calculate_ssim(img1, img2):
    """计算 SSIM；对小于高斯窗口尺寸的图片退化到全图统计，避免空切片。"""
    if min(img1.shape[:2]) < 11:
        x = img1.astype(np.float64)
        y = img2.astype(np.float64)
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        ux, uy = x.mean(), y.mean()
        vx, vy = x.var(), y.var()
        cov = ((x - ux) * (y - uy)).mean()
        return float(((2 * ux * uy + c1) * (2 * cov + c2)) / ((ux**2 + uy**2 + c1) * (vx + vy + c2)))

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def _calculate_l1(img1, img2):
    return np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64))) / 255.0


def calculate_metrics(image_path1, image_path2):
    raw1 = _read_image(image_path1)
    raw2 = _read_image(image_path2)
    if raw1 is None or raw2 is None:
        raise ValueError("指标计算失败：无法读取原图或结果图")
    if raw1.shape != raw2.shape:
        raw1 = cv2.resize(raw1, (raw2.shape[1], raw2.shape[0]))
    identical = bool(np.array_equal(raw1, raw2))
    psnr = _calculate_psnr(raw1, raw2)
    ssim = _calculate_ssim(raw1, raw2)
    l1 = _calculate_l1(raw1, raw2)
    return psnr, ssim, l1, identical


def format_evaluation(psnr, ssim, l1, identical, degrade_note=None):
    psnr_text = "∞" if psnr == float("inf") else f"{psnr:.4f}"
    lines = [
        "【修复图 vs 原图（退化输入）的客观差异】",
        f"峰值信噪比(PSNR): {psnr_text}",
        f"结构相似性(SSIM): {ssim:.4f}",
        f"平均绝对误差(MAE): {l1:.4f}",
    ]
    if identical:
        lines.append(
            "⚠ 修复图与原图像素完全一致（PSNR→∞）：疑似未执行有效修复，"
            "请确认流水线未降级。"
        )
    if degrade_note:
        lines.append(f"⚠ {degrade_note}")
    return "\n".join(lines)


# ===================== 流水线 =====================
def _read_degrade_note(req_output_dir):
    report_path = os.path.join(req_output_dir, "pipeline_report.json")
    if not os.path.exists(report_path):
        return None
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json_loads_safe(f.read())
    except (OSError, ValueError):
        return None
    if not report or not report.get("degraded_count"):
        return None
    return _DEGRADE_MESSAGES.get(
        report.get("degrade_reason"), "部分处理阶段被跳过，结果可能不完整。"
    )


def json_loads_safe(text):
    import json

    return json.loads(text)


def run_pipeline(input_image, user_state, mode):
    """三种处理模式的统一入口。

    Returns:
        (res_img_path, evaluate_text)；划痕检测模式 evaluate_text 为 None。
    """
    cfg = PIPELINE_MODES[mode]
    if input_image is None:
        raise ValueError("请先上传一张图片再提交。")

    req_id = "{}-{}".format(
        datetime.now().strftime("%Y%m%d-%H%M%S"), uuid.uuid4().hex[:8]
    )
    req_input_dir = os.path.join(TEMP_ROOT, req_id)
    req_output_dir = os.path.join(OUTPUT_ROOT, req_id)
    os.makedirs(req_input_dir, exist_ok=True)
    os.makedirs(req_output_dir, exist_ok=True)
    os.makedirs(UPLOAD_ROOT, exist_ok=True)
    os.makedirs(ARCHIVE_INPUT_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_OUTPUT_DIR, exist_ok=True)
    # 低频触发清理，避免每次请求扫描大目录阻塞处理请求
    if uuid.uuid4().int % 20 == 0:
        purge_stale_runs()

    archive_img_path = os.path.join(UPLOAD_ROOT, req_id + ".png")
    temp_img_path = os.path.join(req_input_dir, req_id + ".png")

    try:
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")
        if max(input_image.size) > MAX_IMAGE_SIDE:
            raise ValueError("图片尺寸过大（长边超过 4096 像素），请先缩小后重试。")
        input_image.save(temp_img_path)
        input_image.save(archive_img_path)

        with _PIPELINE_LOCK:
            if cfg["detect_only"]:
                _run_cmd(
                    [
                        "python",
                        "detection.py",
                        "--test_path", req_input_dir,
                        "--output_dir", req_output_dir,
                        "--input_size", "full_size",
                        "--GPU", str(resolve_gpu()),
                    ],
                    cwd=os.path.join(BASE_DIR, "Global"),
                )
                res_img = os.path.join(req_output_dir, "mask", req_id + ".png")
            else:
                cmd = [
                    "python", "run.py",
                    "--input_folder", req_input_dir,
                    "--output_folder", req_output_dir,
                    "--GPU", "auto",
                ]
                if cfg["with_scratch"]:
                    cmd.append("--with_scratch")
                _run_cmd(cmd, cwd=BASE_DIR)
                res_img = os.path.join(req_output_dir, "final_output", req_id + ".png")

        if not os.path.exists(res_img):
            raise ValueError("处理失败：未生成输出图像，请检查后端日志或更换图片重试。")

        username = user_state.get("username", "unknown") if user_state else "unknown"

        if cfg["detect_only"]:
            log_task(username, cfg["label"], archive_img_path, res_img, "N/A", "N/A", "N/A")
            return res_img, None

        degrade_note = _read_degrade_note(req_output_dir)
        psnr, ssim, l1, identical = calculate_metrics(archive_img_path, res_img)
        evaluate_text = format_evaluation(psnr, ssim, l1, identical, degrade_note)
        log_task(username, cfg["label"], archive_img_path, res_img, psnr, ssim, l1)
        return res_img, evaluate_text
    finally:
        _delete_tree(req_input_dir)


def log_task(username, task_type, input_path, output_path, psnr, ssim, mae):
    ts = datetime.now()
    record = {
        "id": ts.strftime("%Y%m%d%H%M%S") + str(ts.microsecond // 1000).zfill(3),
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "user": username or "unknown",
        "type": task_type,
        "input_path": input_path,
        "output_path": output_path,
        "psnr": (
            round(psnr, 4)
            if isinstance(psnr, (int, float)) and psnr != float("inf")
            else ("∞" if psnr == float("inf") else psnr)
        ),
        "ssim": round(ssim, 4) if isinstance(ssim, (int, float)) else ssim,
        "mae": round(mae, 4) if isinstance(mae, (int, float)) else mae,
    }
    append_history(record)
    prefix = record["id"]
    try:
        if os.path.exists(input_path):
            shutil.copy2(
                input_path,
                os.path.join(ARCHIVE_INPUT_DIR, f"{prefix}_{os.path.basename(input_path)}"),
            )
        if os.path.exists(output_path):
            shutil.copy2(
                output_path,
                os.path.join(ARCHIVE_OUTPUT_DIR, f"{prefix}_{os.path.basename(output_path)}"),
            )
    except Exception:  # noqa: BLE001
        print("[归档] 归档图片失败（不影响主流程）")
    # 与运行目录清理一致：低频触发，避免阻塞
    if uuid.uuid4().int % 20 == 0:
        try:
            purge_stale_archives()
        except Exception:  # noqa: BLE001
            print("[归档] 清理过期归档失败")
