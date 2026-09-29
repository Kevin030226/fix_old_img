"""Child-process harness: run stage 3 through the *native* backend.

Used only by ``test_face_enhance_equivalence.py``. It exists because a pytest
process that has already imported the ``Global`` tree cannot also import
``Face_Enhancement``'s same-named ``options``/``models``/``util``/``data``
packages 鈥?see :mod:`fiximg.inference.backends.legacy_tree`. A fresh interpreter
owns the face tree, so the numbers it produces are the numbers a face worker
would produce.

Run as: ``python -m tests.gpu.face_enhance_native_harness <pipeline_root>``
with ``FIXIMG_NATIVE_TREE=face``; prints one line of JSON for the caller to
assert on.
"""
from __future__ import annotations

import json
import os
import sys

from fiximg.inference.backends.face_enhance_native import (
    NativeFaceEnhancementBackend,
    native_available,
)
from fiximg.inference.backends.legacy_cli import detection_dir, each_img_dir


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    root = args[0]

    available, reason = native_available()
    if not available:
        print(json.dumps({"ok": False, "reason": reason}), file=sys.stdout)
        return 3

    crops = detection_dir(root)
    if not os.path.isdir(crops):
        print(json.dumps({"ok": False, "reason": f"no crop directory at {crops}"}))
        return 4

    backend = NativeFaceEnhancementBackend({"stem": "harness"})
    backend.run_folder(crops, root, gpu=-1)

    produced = each_img_dir(root)
    names = sorted(os.listdir(produced)) if os.path.isdir(produced) else []
    print(json.dumps({
        "ok": True,
        "implementation": backend.implementation,
        "written": names,
        "model_loaded": backend.is_loaded,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
