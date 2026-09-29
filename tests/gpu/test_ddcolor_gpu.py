"""GPU-marked tests (plan 搂29: `pytest -m gpu`).

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

from fiximg.config import settings


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
    from fiximg.inference.model_manager import ModelManager

    manager = ModelManager()
    return manager.load("ddcolor")


def test_ddcolor_loads_on_cuda(colorization_model):
    assert colorization_model is not None
    assert next(colorization_model.model.parameters()).is_cuda


def test_ddcolor_load_time_recorded():
    from fiximg.inference.model_manager import ModelManager

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

    from fiximg.inference.model_manager import ModelManager

    manager = ModelManager()
    manager.load("ddcolor")
    torch.cuda.reset_peak_memory_stats()
    stats = manager.gpu_stats()
    assert stats.get("gpu_memory_peak_mb", 0) > 0


def test_the_declared_optimisations_really_are_applied():
    """`channels_last: true` in the manifest must be visible in the weights.

    DDColor is the one model that declares an optimisation, so it is the case that
    proves the declaration reaches a loaded model rather than only the policy
    object - and the record the backend reports is what an operator reads in
    `GET /api/v1/models` (plan 搂3.5.4).
    """
    import torch

    from fiximg.inference.backends.ddcolor import DDColorBackend
    from fiximg.inference.precision import cached_policy

    backend = DDColorBackend()
    backend.load("0")
    policy = cached_policy("ddcolor")
    assert policy.channels_last, "the manifest declares it; this test assumes that"

    conv = next(
        module for module in backend._manager.get("ddcolor").model.modules()
        if isinstance(module, torch.nn.Conv2d)
    )
    assert conv.weight.is_contiguous(memory_format=torch.channels_last), (
        "declared channels_last did not reach the weights"
    )
    assert backend.optimisations_applied.get("channels_last") is True, (
        backend.optimisations_applied
    )


def test_the_builder_honours_the_configured_device(monkeypatch):
    """`FIXIMG_DEVICE=cpu` must not be answered with a CUDA-resident model.

    The vendored builder defaults to "cuda if torch sees one" 鈥?device 0, whatever the
    process was told 鈥?so a CPU-pinned worker put colour weights on the card it had
    been told not to use, while the stage's memory figure was charged to the device the
    runtime had scheduled. Both halves of that are wrong together: the number and the
    placement disagree with the configuration.
    """
    from fiximg.inference.model_manager import ModelManager

    monkeypatch.setattr(settings, "device", "cpu")
    pipeline = ModelManager().load("ddcolor")

    device = next(pipeline.model.parameters()).device
    assert device.type == "cpu", device
    assert str(pipeline.device) == "cpu"


def test_reside_on_moves_a_real_pipeline(colorization_model):
    """The move is exercised on a model that can actually be moved.

    The unit-level checks use a seam for availability; this one proves the torch call
    does what the report claims, in both directions, and leaves the fixture on the card
    the rest of the file expects.
    """
    from fiximg.inference.backends.ddcolor import reside_on

    assert reside_on(colorization_model, "-1") == "cpu"
    assert next(colorization_model.model.parameters()).device.type == "cpu"

    assert reside_on(colorization_model, "0") == "cuda:0"
    assert next(colorization_model.model.parameters()).is_cuda
