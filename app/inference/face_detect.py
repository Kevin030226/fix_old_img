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

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("fiximg.face_detect")

_YUNET_PATH = os.path.join(settings.base_dir, "weights", "yunet", "face_detection_yunet_2023mar.onnx")

_BACKEND_RESOLVED = False
_BACKEND: str | None = None
_YUNET = None
_DLIB_DETECTOR = None


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

    if hasattr(cv2, "CascadeClassifier"):
        try:
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if not cascade.empty():
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
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    found = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    return [tuple(int(v) for v in box) for box in found]


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
