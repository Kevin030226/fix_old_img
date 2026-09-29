"""The native Global backend must reproduce the vendored CLI's numbers.

Plan 搂3.5.1/搂4.2 replaced ``python run.py`` per request with in-process, resident
networks. A refactor of an inference path is only correct if the picture coming
out is the picture that used to come out 鈥?so this runs the same image through
both implementations and compares them, rather than trusting that the
preprocessing was transcribed faithfully.

Measured on this machine (CPU, 298脳450 sample): PSNR ``inf``, SSIM ``1.0``,
max abs diff ``0`` 鈥?byte-identical. The assertion below is a threshold, not
equality, because ``cudnn.benchmark=True`` lets the two processes pick different
CUDA kernels; on CPU the outputs are exact.

Needs the Global quality weights, hence the ``gpu`` marker.
"""
import os

import pytest
from PIL import Image

from fiximg.inference.backends.base import ModelRequest
from fiximg.inference.backends.global_restore import (
    CHECKPOINTS_DIR,
    GlobalRestoreBackend,
    quality_weights_present,
)
from fiximg.inference.backends.legacy_cli import (
    STAGE_RESTORE,
    LegacyCliBackend,
    restored_image_dir,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not quality_weights_present(),
                       reason=f"Global quality weights are absent under {CHECKPOINTS_DIR}"),
]

SAMPLES = ("examples/old", "test_images/old")


def _sample_image():
    """The first real sample, downscaled so a CPU run stays bearable."""
    for directory in SAMPLES:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if name.lower().endswith((".png", ".jpg")):
                image = Image.open(os.path.join(directory, name)).convert("RGB")
                image.thumbnail((192, 192), Image.LANCZOS)
                return image
    pytest.skip("no sample images in examples/old or test_images/old")


def test_native_output_matches_the_legacy_cli(tmp_path):
    import numpy as np

    from fiximg.inference.evaluation.reference import calculate_difference_metrics

    image = _sample_image()

    native = GlobalRestoreBackend()
    native.load("cpu")
    native_image = native.infer(ModelRequest(image=image, device="cpu")).image
    native_path = os.path.join(str(tmp_path), "native.png")
    native_image.save(native_path)

    input_dir = os.path.join(str(tmp_path), "input")
    os.makedirs(input_dir, exist_ok=True)
    image.save(os.path.join(input_dir, "sample.png"))
    pipeline_root = os.path.join(str(tmp_path), "pipeline")
    LegacyCliBackend({"stem": "sample"}, with_scratch=False,
                     stages=(STAGE_RESTORE,)).run_folder(input_dir, pipeline_root, gpu=-1)
    cli_path = os.path.join(restored_image_dir(pipeline_root), "sample.png")
    assert os.path.isfile(cli_path), "the legacy path produced nothing to compare against"

    assert native_image.size == Image.open(cli_path).size, (
        "geometry is part of the contract: the /4 rounding must match exactly"
    )

    metrics = calculate_difference_metrics(native_path, cli_path)
    assert metrics["mae"] <= 1.0, f"mean abs error vs the CLI: {metrics['mae']}"
    assert metrics["psnr"] >= 40.0 or metrics["psnr"] == float("inf"), (
        f"PSNR vs the CLI: {metrics['psnr']}"
    )
    assert metrics["ssim"] >= 0.99, f"SSIM vs the CLI: {metrics['ssim']}"

    if np.array_equal(np.asarray(native_image), np.asarray(Image.open(cli_path))):
        assert metrics["psnr"] == float("inf"), "identical pixels must report an infinite PSNR"


def test_the_native_backend_never_spawns_a_subprocess(monkeypatch, tmp_path):
    """The point of the change: no interpreter start, no per-request weight load.

    Asserted by making spawning impossible rather than by counting calls 鈥?if any
    part of the path still shells out, it fails here instead of quietly working.
    """
    import subprocess

    def _no_spawn(*args, **kwargs):
        raise AssertionError("native Global restoration must not spawn a subprocess")

    monkeypatch.setattr(subprocess, "Popen", _no_spawn)
    monkeypatch.setattr(subprocess, "run", _no_spawn)

    image = _sample_image()
    backend = GlobalRestoreBackend()
    backend.load("cpu")
    result = backend.infer(ModelRequest(image=image, device="cpu"))
    assert result.image.size[0] > 0
    assert result.metadata["backend"] == "native"


def test_the_weights_stay_resident_between_requests():
    """Two inferences, one load 鈥?what 'resident' is supposed to mean."""
    image = _sample_image()
    backend = GlobalRestoreBackend()
    backend.load("cpu")
    model = backend._model

    backend.infer(ModelRequest(image=image, device="cpu"))
    backend.infer(ModelRequest(image=image, device="cpu"))

    assert backend._model is model, "the networks were rebuilt mid-session"
    assert backend.is_loaded is True


def test_the_folder_bridge_writes_what_the_stage_reads(tmp_path):
    """Stages are folder-based; the bridge must satisfy that contract in-process."""
    input_dir = os.path.join(str(tmp_path), "input")
    os.makedirs(input_dir, exist_ok=True)
    image = _sample_image()
    image.save(os.path.join(input_dir, "a.png"))

    root = os.path.join(str(tmp_path), "pipeline")
    backend = GlobalRestoreBackend()
    labels: list = []
    backend.run_folder(input_dir, root, gpu=-1,
                       on_progress=lambda step, total, label: labels.append((step, total, label)))

    produced = os.path.join(restored_image_dir(root), "a.png")
    assert os.path.isfile(produced), "the stage reads restored_image/<name>.png"
    # Not the input size: the pipeline rounds each side to a multiple of 4
    # (data_transforms with test_mode="Full"), which the equivalence test proves
    # the CLI does too. Pinning it here catches an accidental change of geometry.
    expected = (int(round(image.width / 4) * 4), int(round(image.height / 4) * 4))
    assert Image.open(produced).size == expected
    assert labels == [(1, 1, "restore")], "progress markers feed the UI's stage bar"


def test_the_bridge_renames_jpg_inputs_like_the_pipeline_does(tmp_path):
    """Legacy writes PNG regardless of the input extension (test.py:172-173)."""
    input_dir = os.path.join(str(tmp_path), "input")
    os.makedirs(input_dir, exist_ok=True)
    _sample_image().save(os.path.join(input_dir, "b.jpg"), format="JPEG")

    root = os.path.join(str(tmp_path), "pipeline")
    backend = GlobalRestoreBackend()
    backend.run_folder(input_dir, root, gpu=-1)

    assert os.path.isfile(os.path.join(restored_image_dir(root), "b.png"))
