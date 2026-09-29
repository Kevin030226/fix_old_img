"""Shared face detection helper (plan §11/§17).

One detector resolution for the whole app so the analyzer's `face_count` and
the identity module's face boxes agree. Backend order (configurable):

  1. OpenCV YuNet ONNX (`weights/yunet/face_detection_yunet_2023mar.onnx`,
     downloadable via `python -m scripts.download_weights download`) —
     accurate and dependency-light
  2. OpenCV Haar cascades (removed in OpenCV 5.x, kept for 4.x installs)
  3. dlib frontal detector (needs `import dlib`, no extra files)

If no backend is available at all, detection degrades to "no faces" — the
identity metric then reports `face_count=0` and the auto planner simply skips
the face stages, instead of crashing a run.
"""
import os
from typing import Any

import cv2
import numpy as np

from fiximg.config import settings
from fiximg.infrastructure.observability.logging import get_logger

logger = get_logger("fiximg.face_detect")

_YUNET_PATH = os.path.join(settings.base_dir, "weights", "yunet", "face_detection_yunet_2023mar.onnx")

_BACKEND_RESOLVED = False
_BACKEND: str | None = None
#: Handles exist only once a backend resolved, and OpenCV 5's bundled type stubs
#: do not declare `cv2.data` (present at runtime) or `cv2.CascadeClassifier`
#: (Haar was removed in 5.x) — so both are read through `getattr` and kept `Any`.
#: That is also what makes the "no Haar in this build" case a value check rather
#: than an AttributeError waiting inside a try.
_YUNET: Any = None
_DLIB_DETECTOR: Any = None
_CASCADE_CLASSIFIER: Any = getattr(cv2, "CascadeClassifier", None)
_CASCADE_DATA: Any = getattr(cv2, "data", None)


def _haar_available() -> bool:
    return _CASCADE_CLASSIFIER is not None and _CASCADE_DATA is not None


def _haar_cascade():
    """A loaded Haar classifier, or None on a build without the data."""
    if not _haar_available():
        return None
    return _CASCADE_CLASSIFIER(
        _CASCADE_DATA.haarcascades + "haarcascade_frontalface_default.xml"
    )


def _resolve_backend() -> str | None:
    """Pick and cache the best available backend; None when none exists."""
    global _BACKEND_RESOLVED, _BACKEND, _YUNET, _DLIB_DETECTOR
    if _BACKEND_RESOLVED:
        return _BACKEND
    _BACKEND_RESOLVED = True

    if os.path.isfile(_YUNET_PATH):
        try:
            detector = cv2.FaceDetectorYN.create(_YUNET_PATH, "", (320, 320), score_threshold=0.6)
            _YUNET = detector
            _BACKEND = "yunet"
            logger.info("face detection backend: yunet (%s)", os.path.basename(_YUNET_PATH))
            return _BACKEND
        except Exception as exc:  # noqa: BLE001
            logger.warning("YuNet load failed (%s); trying Haar cascade", exc)

    if _haar_available():
        try:
            cascade = _haar_cascade()
            if cascade is not None and not cascade.empty():
                _BACKEND = "haar"
                logger.info("face detection backend: haar cascade")
                return _BACKEND
        except Exception as exc:  # noqa: BLE001
            logger.warning("Haar cascade unavailable: %s", exc)

    try:
        import dlib  # noqa: PLC0415 — heavy import kept lazy

        _DLIB_DETECTOR = dlib.get_frontal_face_detector()
        _BACKEND = "dlib"
        logger.info("face detection backend: dlib")
        return _BACKEND
    except Exception as exc:  # noqa: BLE001
        logger.warning("dlib unavailable: %s; face detection disabled", exc)

    _BACKEND = None
    return None


def dlib_importable() -> bool:
    """Whether `import dlib` would succeed here — without importing it.

    The face chain's *only* hard third-party dependency is dlib, whether the
    crops come from the resident backend or from the vendored subprocess (which
    runs this same interpreter, `sys.executable`). A caller that has to decide
    "run the face chain or degrade" needs a check that costs microseconds and has
    no side effects, so `find_spec` is used instead of an import.

    An entry in :data:`sys.modules` wins: a module already loaded is by
    definition importable, and a ``None`` entry is CPython's own marker for "this
    import failed before" — which is also how tests simulate its absence.
    """
    import importlib.util
    import sys

    if "dlib" in sys.modules:
        return sys.modules["dlib"] is not None
    try:
        return importlib.util.find_spec("dlib") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken entry on sys.path
        return False


def _reset_cache() -> None:
    """Test helper: re-resolve the backend on next call."""
    global _BACKEND_RESOLVED, _BACKEND, _YUNET, _DLIB_DETECTOR
    _BACKEND_RESOLVED = False
    _BACKEND = None
    _YUNET = None
    _DLIB_DETECTOR = None


def _detect_yunet(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = rgb.shape[:2]
    _YUNET.setInputSize((w, h))
    _, faces = _YUNET.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    boxes = []
    if faces is not None:
        for f in np.asarray(faces):
            x, y, fw, fh = (int(round(v)) for v in f[:4])
            boxes.append((max(x, 0), max(y, 0), fw, fh))
    return boxes


def _detect_haar(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    cascade = _haar_cascade()
    if cascade is None:  # pragma: no cover - only reachable if the backend changed
        return []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    found = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    return [
        (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        for box in np.asarray(found).reshape(-1, 4)
    ]


def _detect_dlib(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    dets = _DLIB_DETECTOR(rgb, 1)
    return [(d.left(), d.top(), d.width(), d.height()) for d in dets]


def detect_face_boxes(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return (x, y, w, h) boxes for every frontal face; [] when none/no backend."""
    backend = _resolve_backend()
    if backend is None:
        return []
    try:
        if backend == "yunet":
            return _detect_yunet(rgb)
        if backend == "haar":
            return _detect_haar(rgb)
        return _detect_dlib(rgb)
    except Exception as exc:  # noqa: BLE001 — detection must never break a run
        logger.warning("face detection failed (%s); returning no faces", exc)
        return []
