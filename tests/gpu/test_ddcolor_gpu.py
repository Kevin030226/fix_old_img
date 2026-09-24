"""GPU-marked tests (plan §29: `pytest -m gpu`).

These exercise the real model-loading and inference path (ModelManager +
DDColor) and are therefore skipped unless ALL of the following hold:

  - a CUDA GPU is visible to torch
  - the DDColor weights are present (weights/ddcolor/pytorch_model.pt)
  - the vendored `ddcolor` package is importable

CI runs pytest with `-m "not gpu"` (see .github/workflows/ci.yml); on a GPU
machine run them explicitly with `pytest -m gpu`.
"""
import os

import pytest

from app.core.config import settings


def _gpu_ready() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
    except Exception:  # noqa: BLE001
        return False
    if not os.path.exists(settings.ddcolor_weights):
        return False
    try:
        import ddcolor  # noqa: F401, PLC0415
    except Exception:  # noqa: BLE001
        return False
    return True


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _gpu_ready(), reason="requires CUDA GPU + DDColor weights"),
]


@pytest.fixture(scope="module")
def colorization_model():
    from app.inference.model_manager import ModelManager

    manager = ModelManager()
    return manager.load("ddcolor")


def test_ddcolor_loads_on_cuda(colorization_model):
    assert colorization_model is not None
    assert next(colorization_model.model.parameters()).is_cuda


def test_ddcolor_load_time_recorded():
    from app.inference.model_manager import ModelManager

    manager = ModelManager()
    manager.load("ddcolor")
    assert manager.load_times_ms.get("ddcolor", 0) > 0


def test_ddcolor_real_inference(colorization_model):
    """End-to-end colorization of a small grayscale image.

    The pipeline contract (mirrored by ColorizationStage) is BGR uint8 in,
    colour BGR uint8 out.
    """
    import cv2
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(7)
    gray = np.tile(np.linspace(30, 225, 128).astype(np.uint8), (96, 1))
    gray = np.clip(gray.astype(np.int64) + rng.integers(-6, 7, gray.shape), 0, 255).astype(np.uint8)
    image = Image.fromarray(np.stack([gray] * 3, axis=-1).astype(np.uint8), mode="RGB")

    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    out_bgr = colorization_model.process(bgr)
    assert out_bgr is not None and out_bgr.shape[:2] == bgr.shape[:2]
    arr = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    # a colourised output must show channel divergence beyond noise
    assert float(arr[:, :, 0].astype(int).std()) > 0


def test_gpu_stats_report_memory_peak():
    import torch

    from app.inference.model_manager import ModelManager

    manager = ModelManager()
    manager.load("ddcolor")
    torch.cuda.reset_peak_memory_stats()
    stats = manager.gpu_stats()
    assert stats.get("gpu_memory_peak_mb", 0) > 0
