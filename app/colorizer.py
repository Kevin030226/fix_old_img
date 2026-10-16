"""DDColor 老照片上色模块（Apache-2.0，官方 piddnad/DDColor）。

模型懒加载（首次调用时加载到 GPU），与修复流水线共用进程内锁，
处理记录复用 SQLite。
"""
import os
import shutil
import threading
import time
import uuid
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

from .pipeline import OUTPUT_ROOT, UPLOAD_ROOT, log_task, purge_stale_runs

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DDCOLOR_WEIGHTS = os.environ.get(
    "FIXIMG_DDCOLOR_MODEL",
    os.path.join(BASE_DIR, "weights", "ddcolor", "pytorch_model.pt"),
)
DDCOLOR_INPUT_SIZE = int(os.environ.get("FIXIMG_DDCOLOR_INPUT_SIZE", "512"))
DDCOLOR_MODEL_SIZE = os.environ.get("FIXIMG_DDCOLOR_MODEL_SIZE", "large")
COLORIZE_RESULT_TTL = int(os.environ.get("FIXIMG_COLORIZE_TTL", str(2 * 3600)))

_lock = threading.Lock()
_pipeline = None


def _load_pipeline():
    """懒加载 DDColor 模型（线程安全单例）。"""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is None:
            if not os.path.exists(DDCOLOR_WEIGHTS):
                raise FileNotFoundError(
                    f"未找到 DDColor 权重文件: {DDCOLOR_WEIGHTS}\n"
                    "请从 ModelScope 下载 damo/cv_ddcolor_image-colorization 的 "
                    "pytorch_model.pt 放入 weights/ddcolor/ 目录。"
                )
            # 确保项目根目录在 sys.path 中，使 ddcolor / basicsr 可导入
            import sys

            if BASE_DIR not in sys.path:
                sys.path.insert(0, BASE_DIR)
            from ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model

            model = build_ddcolor_model(
                DDColor,
                model_path=DDCOLOR_WEIGHTS,
                input_size=DDCOLOR_INPUT_SIZE,
                model_size=DDCOLOR_MODEL_SIZE,
            )
            _pipeline = ColorizationPipeline(model, input_size=DDCOLOR_INPUT_SIZE)
    return _pipeline


def colorize_bgr(img_bgr):
    """对 BGR 图像上色，返回 BGR 结果。"""
    pipe = _load_pipeline()
    return pipe.process(img_bgr)


def colorize_pil(pil_image):
    """对 PIL 图像上色，返回 PIL RGB 结果。"""
    rgb = pil_image.convert("RGB")
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    out_bgr = colorize_bgr(bgr)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_rgb)


def run_colorize(input_image, user_state):
    """Web 入口：上色 + 归档 + 历史入库，返回结果图像绝对路径。"""
    if input_image is None:
        raise ValueError("请先上传一张图片再提交。")

    req_id = "{}-{}".format(
        datetime.now().strftime("%Y%m%d-%H%M%S"), uuid.uuid4().hex[:8]
    )
    req_output_dir = os.path.join(OUTPUT_ROOT, req_id)
    os.makedirs(req_output_dir, exist_ok=True)
    os.makedirs(UPLOAD_ROOT, exist_ok=True)
    purge_stale_runs(COLORIZE_RESULT_TTL)

    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)
    input_image = input_image.convert("RGB")
    if max(input_image.size) > 4096:
        raise ValueError("图片尺寸过大（长边超过 4096 像素），请先缩小后重试。")

    archive_img_path = os.path.join(UPLOAD_ROOT, req_id + ".png")
    input_image.save(archive_img_path)

    result_img = colorize_pil(input_image)
    res_path = os.path.join(req_output_dir, "colorized.png")
    result_img.save(res_path)

    username = user_state.get("username", "unknown") if user_state else "unknown"
    log_task(username, "照片上色", archive_img_path, res_path, "N/A", "N/A", "N/A")
    return res_path

