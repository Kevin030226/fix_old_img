"""The native stage-3 backend must emit the bytes the vendored script emits.

Plan 搂4.2 Step 2's last chain: face enhancement loaded a 92M-parameter SPADE
generator per subprocess run; the native backend keeps it resident. The claim is
only worth making if the pictures are the same, so this runs both paths over the
same aligned crops and compares file names and bytes.

The native side runs in a child process (``face_enhance_native_harness``): this
pytest process may already have imported the ``Global`` tree, and
``Face_Enhancement`` owns the same top-level package names. That is the
constraint ``FIXIMG_NATIVE_TREE`` encodes, and the test honours it rather than
pretending the two can share an interpreter.

Needs the Face_Enhancement generator weights, hence the ``gpu`` marker.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from fiximg.inference.backends import face_enhance_native as fen
from fiximg.inference.backends.legacy_cli import (
    STAGE_FACE_ENHANCE,
    LegacyCliBackend,
    detection_dir,
    each_img_dir,
)
from fiximg.paths import PROJECT_ROOT

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not fen.checkpoint_present(),
        reason=f"face enhancement weights are missing: {fen.GENERATOR_WEIGHTS}",
    ),
]

CROPS = ("faceA_1.png", "faceB_1.png", "faceB_2.png")


def _make_root(tmp_path, name: str) -> str:
    """A pipeline root holding stage-2's output, so stage 3 has the same input."""
    root = os.path.join(str(tmp_path), name)
    crops = detection_dir(root)
    os.makedirs(crops, exist_ok=True)
    rng = np.random.default_rng(11)
    for crop in CROPS:
        array = rng.integers(30, 210, size=(256, 256, 3), dtype=np.uint8)
        Image.fromarray(array).save(os.path.join(crops, crop))
    return root


def _run_native(root: str) -> dict:
    env = dict(os.environ)
    env["FIXIMG_NATIVE_TREE"] = "face"
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(PROJECT_ROOT, "src"), PROJECT_ROOT])
    completed = subprocess.run(
        [sys.executable, "-m", "tests.gpu.face_enhance_native_harness", root],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=1800,
    )
    assert completed.returncode == 0, (
        f"native stage 3 failed ({completed.returncode}):\n"
        f"{completed.stdout}\n{completed.stderr}"
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_native_enhancement_matches_the_vendored_script(tmp_path):
    legacy_root = _make_root(tmp_path, "legacy")
    LegacyCliBackend({"stem": "eq"}, stages=(STAGE_FACE_ENHANCE,)).run_folder(
        detection_dir(legacy_root), legacy_root, gpu=-1
    )
    legacy_dir = each_img_dir(legacy_root)
    assert os.path.isdir(legacy_dir), "the vendored script produced nothing to compare"
    legacy_names = sorted(os.listdir(legacy_dir))
    assert legacy_names == sorted(CROPS)

    native_root = _make_root(tmp_path, "native")
    report = _run_native(native_root)
    assert report["ok"] is True, report
    assert report["implementation"] == "native"
    assert report["model_loaded"] is True
    assert report["written"] == legacy_names, "same crops, same names"

    native_dir = each_img_dir(native_root)
    for name in legacy_names:
        a = np.asarray(Image.open(os.path.join(legacy_dir, name)).convert("RGB")).astype(int)
        b = np.asarray(Image.open(os.path.join(native_dir, name)).convert("RGB")).astype(int)
        assert a.shape == b.shape, f"{name}: {a.shape} vs {b.shape}"
        # Same weights, same preprocessing, the same save_image call: on CPU the
        # two are bit-identical, so the tolerance is only what CUDA introduces 鈥?        # cuDNN picks kernels per run and cudnn.benchmark is on in this tree.
        import torch

        tolerance = 0 if not torch.cuda.is_available() else 1
        delta = int(np.abs(a - b).max())
        assert delta <= tolerance, (
            f"{name} differs by {delta} levels (tolerance {tolerance} on "
            f"{'cuda' if torch.cuda.is_available() else 'cpu'})"
        )


def test_native_backend_is_unreachable_from_a_global_worker(tmp_path):
    """The harness proves the gate, not just the happy path.

    With the default tree setting the child must report itself unable to run
    natively rather than quietly importing the face tree.
    """
    root = _make_root(tmp_path, "guarded")
    env = dict(os.environ)
    env["FIXIMG_NATIVE_TREE"] = "global"
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(PROJECT_ROOT, "src"), PROJECT_ROOT])
    completed = subprocess.run(
        [sys.executable, "-m", "tests.gpu.face_enhance_native_harness", root],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=600,
    )
    assert completed.returncode == 3, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["ok"] is False
    assert "does not own" in payload["reason"]
