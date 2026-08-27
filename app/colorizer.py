"""DDColor old photo colorization module (Apache-2.0, official piddnad/DDColor).

The model is loaded lazily (loaded onto the GPU on first call), sharing the in-process lock with the
restoration pipeline, and processing records reuse SQLite.
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
    """Lazily load the DDColor model (thread-safe singleton)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is None:
            if not os.path.exists(DDCOLOR_WEIGHTS):
                raise FileNotFoundError(
                    f"DDColor weight file not found: {DDCOLOR_WEIGHTS}\n"
                    "Download pytorch_model.pt of damo/cv_ddcolor_image-colorization from ModelScope and "
                    "put it into weights/ddcolor/."
                )
            # Ensure the project root is in sys.path so ddcolor / basicsr can be imported
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
    """Colorize a BGR image and return a BGR result."""
    pipe = _load_pipeline()
    return pipe.process(img_bgr)


def colorize_pil(pil_image):
    """Colorize a PIL image and return a PIL RGB result."""
    rgb = pil_image.convert("RGB")
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    out_bgr = colorize_bgr(bgr)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_rgb)


def run_colorize(input_image, user_state):
    """Web entry: colorize + archive + log history; returns the absolute path of the result image."""
    if input_image is None:
        raise ValueError("Please upload an image first.")

    req_id = "{}-{}".format(
        datetime.now().strftime("%Y%m%d-%H%M%S"), uuid.uuid4().hex[:8]
    )
    req_output_dir = os.path.join(OUTPUT_ROOT, req_id)
    os.makedirs(req_output_dir, exist_ok=True)
    os.makedirs(UPLOAD_ROOT, exist_ok=True)
    # Low-frequency cleanup to avoid scanning large directories on every request
    if uuid.uuid4().int % 20 == 0:
        purge_stale_runs(COLORIZE_RESULT_TTL)

    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)
    input_image = input_image.convert("RGB")
    if max(input_image.size) > 4096:
        raise ValueError("Image too large (long side exceeds 4096 px); please resize and retry.")

    archive_img_path = os.path.join(UPLOAD_ROOT, req_id + ".png")
    input_image.save(archive_img_path)

    result_img = colorize_pil(input_image)
    res_path = os.path.join(req_output_dir, "colorized.png")
    result_img.save(res_path)

    username = user_state.get("username", "unknown") if user_state else "unknown"
    log_task(username, "Photo colorization", archive_img_path, res_path, "N/A", "N/A", "N/A")
    return res_path

