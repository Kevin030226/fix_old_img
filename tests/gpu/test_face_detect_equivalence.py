"""Native face detection must produce the crops the vendored script produces.

The contract tests in ``tests/unit/test_face_detect_native.py`` stub dlib, so they
can prove the wiring but not the pixels. This file closes that gap wherever dlib is
actually installed: it runs stage 2 twice over the same restored images 鈥?once
through ``Face_Detection/detect_all_dlib.py`` in a subprocess, once in-process 鈥?and requires the same crop names and the same bytes.

Skipped when dlib or the landmark model is absent. That is the honest state of this
machine (dlib needs a C++ toolchain to build), and it is why
``FIXIMG_FACE_DETECT_NATIVE`` defaults to off: an unverified inference path should
not be what users get by accident.
"""
import os

import pytest
from PIL import Image

from fiximg.inference.backends.face_detect_native import (
    NativeFaceDetectionBackend,
    native_available,
)
from fiximg.inference.backends.legacy_cli import (
    STAGE_FACE_DETECT,
    LegacyCliBackend,
    detection_dir,
    restored_image_dir,
)

_available, _reason = native_available()

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _available, reason=_reason or "dlib is unavailable"),
]

SAMPLE_DIRS = ("examples/old", "test_images/old")


def _samples(limit=3):
    found = []
    for directory in SAMPLE_DIRS:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if name.lower().endswith((".png", ".jpg")):
                image = Image.open(os.path.join(directory, name)).convert("RGB")
                image.thumbnail((512, 512), Image.LANCZOS)
                found.append((name, image))
                if len(found) >= limit:
                    return found
    return found


def _stage_inputs(tmp_path):
    """Lay samples out as stage 1 would have left them."""
    root = str(tmp_path)
    target = restored_image_dir(root)
    os.makedirs(target, exist_ok=True)
    for name, image in _samples():
        image.save(os.path.join(target, os.path.splitext(name)[0] + ".png"))
    return root, target


def test_in_process_crops_equal_the_subprocess_crops(tmp_path):
    import numpy as np

    from fiximg.inference.backends.legacy_cli import list_images

    # Both implementations read the shared root's stage-1 output and ignore the
    # input_dir argument, so the same value is passed to each.
    root_legacy, inputs = _stage_inputs(tmp_path / "legacy")
    LegacyCliBackend({"stem": "eq"}, stages=(STAGE_FACE_DETECT,)).run_folder(
        inputs, root_legacy, gpu=-1
    )
    legacy_crops = detection_dir(root_legacy)

    root_native, inputs_native = _stage_inputs(tmp_path / "native")
    backend = NativeFaceDetectionBackend({"stem": "eq"})
    backend.run_folder(inputs_native, root_native, gpu=-1)
    native_crops = detection_dir(root_native)

    names_legacy = sorted(list_images(legacy_crops))
    names_native = sorted(list_images(native_crops))

    if not names_legacy:
        pytest.skip("no faces detected in the sample set; nothing to compare")

    assert names_native == names_legacy, "stage 4 pairs crops back by filename"

    for name in names_legacy:
        a = np.asarray(Image.open(os.path.join(legacy_crops, name)).convert("RGB"))
        b = np.asarray(Image.open(os.path.join(native_crops, name)).convert("RGB"))
        assert a.shape == b.shape, f"{name}: {a.shape} vs {b.shape}"
        # skimage's warp + img_as_ubyte are the same calls on both sides, so any
        # difference at all means the geometry or the ordering drifted.
        assert np.array_equal(a, b), f"{name} differs by up to {int(np.abs(a - b).max())}"
