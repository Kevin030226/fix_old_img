"""Face Identity Preservation (plan section 17).

The V1 face chain (detect -> align -> enhance -> warp back) changes faces, but
nothing ever measures whether the *same person* is still there. This module
adds the plan section 17 evaluation:

    Original Face -> Face Embedding -> Restoration -> Restored Face -> Embedding
                                                            -> Cosine Similarity

and reports the plan's example block:

    Face Count            3
    Enhanced Faces        3
    Identity Similarity   0.91

Backends (settings.identity_backend):
  - "dlib_resnet" when `dlib` plus the V1 weight files are present
    (Face_Detection/shape_predictor_68_face_landmarks.dat and
    dlib_face_recognition_resnet_model_v1.dat — 128-D embeddings)
  - "gradient_fallback": dependency-free embedding from an
    orientation-histogram descriptor of each face crop (no weights needed)
  - "off": metric disabled entirely

Face boxes always come from the shared `face_detect` helper so this module
and the auto-restore analyzer agree on what a face is.

Caveats (plan section 17): thresholds must be calibrated against test data and
the similarity is an auxiliary indicator — it never decides on its own whether
a restoration is correct.
"""
import os

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.inference.face_detect import detect_face_boxes

logger = get_logger("fiximg.identity")

#: reference_type recorded in the metrics table
REFERENCE_TYPE = "identity_preservation"

#: appended to the identity report block (plan section 17 caveats)
IDENTITY_NOTICE = (
    "⚠ Identity Similarity is an auxiliary indicator only; it cannot on its own "
    "determine whether the restoration is correct (calibrate thresholds on test data)."
)

_EMB_SIZE = 48  # fallback embedding input size
_EMB_BINS = 8   # unsigned-gradient orientation bins
_EMB_BLOCKS = 4 # 4x4 block grid -> 128-D descriptor

# dlib context is resolved lazily and cached for the process lifetime.
_DLIB_CTX = None
_DLIB_TRIED = False


def _dlib_paths() -> tuple[str, str]:
    """Weight file locations following the V1 Face_Detection convention."""
    base = os.path.join(settings.base_dir, "Face_Detection")
    return (
        os.path.join(base, "shape_predictor_68_face_landmarks.dat"),
        os.path.join(base, "dlib_face_recognition_resnet_model_v1.dat"),
    )


def _load_dlib():
    """Return (dlib, detector, predictor, embedder) or None when unavailable."""
    global _DLIB_CTX, _DLIB_TRIED
    if _DLIB_TRIED:
        return _DLIB_CTX
    _DLIB_TRIED = True
    try:
        import dlib  # noqa: PLC0415 — heavy import kept lazy

        landmarks, recog = _dlib_paths()
        if not (os.path.exists(landmarks) and os.path.exists(recog)):
            logger.info("dlib identity backend unavailable: weight files missing")
            return None
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(landmarks)
        embedder = dlib.face_recognition_model_v1(recog)
        _DLIB_CTX = (dlib, detector, predictor, embedder)
    except Exception as exc:  # noqa: BLE001 — identity must never break a run
        logger.info("dlib identity backend unavailable: %s", exc)
        _DLIB_CTX = None
    return _DLIB_CTX


def _reset_dlib_cache() -> None:
    """Test helper: forget the cached dlib context so it is probed again."""
    global _DLIB_CTX, _DLIB_TRIED
    _DLIB_CTX = None
    _DLIB_TRIED = False


# ------------------------------------------------------------------ embeddings
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def _gradient_embedding(rgb_crop: np.ndarray) -> np.ndarray:
    """128-D orientation-histogram descriptor of one face crop (fallback).

    A tiny HOG: unsigned gradients quantised into 8 orientation bins pooled
    over a 4x4 block grid, L2-normalised. Cheap, deterministic and stable
    enough to say "this is still the same face" across restoration edits.
    """
    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (_EMB_SIZE, _EMB_SIZE), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 180.0
    bin_idx = (ang / 180.0 * _EMB_BINS).astype(int) % _EMB_BINS

    cell = _EMB_SIZE // _EMB_BLOCKS
    desc = np.zeros((_EMB_BLOCKS, _EMB_BLOCKS, _EMB_BINS), dtype=np.float64)
    for by in range(_EMB_BLOCKS):
        for bx in range(_EMB_BLOCKS):
            ys = slice(by * cell, (by + 1) * cell)
            xs = slice(bx * cell, (bx + 1) * cell)
            for b in range(_EMB_BINS):
                desc[by, bx, b] = mag[ys, xs][bin_idx[ys, xs] == b].sum()
    v = desc.ravel()
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


# -------------------------------------------------------------------- detection
def _dlib_faces(ctx, rgb: np.ndarray):
    dlib, detector, predictor, embedder = ctx
    dets = detector(rgb, 1)
    boxes = [(d.left(), d.top(), d.width(), d.height()) for d in dets]
    embs = []
    for d in dets:
        shape = predictor(rgb, d)
        embs.append(np.asarray(embedder.compute_face_descriptor(rgb, shape, 1)))
    return boxes, embs


def _fallback_faces(rgb: np.ndarray):
    boxes = detect_face_boxes(rgb)
    embs = []
    for x, y, w, h in boxes:
        crop = rgb[max(y, 0):y + h, max(x, 0):x + w]
        if crop.size == 0:
            embs.append(np.zeros(_EMB_BLOCKS * _EMB_BLOCKS * _EMB_BINS))
            continue
        embs.append(_gradient_embedding(crop))
    return boxes, embs


# -------------------------------------------------------------------- pairing
def _pair_boxes(before, after) -> list[tuple[int, int]]:
    """Greedy nearest-centre pairing between before/after face boxes."""
    used: set[int] = set()
    pairs = []
    for i, b in enumerate(before):
        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
        best, best_d = None, None
        for j, a in enumerate(after):
            if j in used:
                continue
            ax, ay = a[0] + a[2] / 2, a[1] + a[3] / 2
            d = (cx - ax) ** 2 + (cy - ay) ** 2
            if best_d is None or d < best_d:
                best, best_d = j, d
        if best is not None:
            used.add(best)
            pairs.append((i, best))
    return pairs


# ---------------------------------------------------------------------- metric
def compute_identity_similarity(before_image, after_image) -> dict:
    """Plan section 17 metric between the input and the restored output."""
    backend_setting = (settings.identity_backend or "auto").lower()
    if backend_setting == "off":
        return {"skipped": True, "reason": "disabled_by_config"}

    before = np.asarray(before_image.convert("RGB"))
    after = np.asarray(after_image.convert("RGB"))

    ctx = None if backend_setting == "fallback" else _load_dlib()
    if ctx is not None:
        backend = "dlib_resnet"
        boxes_b, embs_b = _dlib_faces(ctx, before)
        boxes_a, embs_a = _dlib_faces(ctx, after)
    else:
        backend = "gradient_fallback"
        boxes_b, embs_b = _fallback_faces(before)
        boxes_a, embs_a = _fallback_faces(after)

    pairs = _pair_boxes(boxes_b, boxes_a)
    sims = [_cosine(embs_b[i], embs_a[j]) for i, j in pairs]

    return {
        "face_count": len(boxes_b),
        "enhanced_faces": len(boxes_a),
        "paired_faces": len(sims),
        "identity_similarity": round(float(np.mean(sims)), 4) if sims else None,
        "backend": backend,
        "notice": IDENTITY_NOTICE,
    }


def format_identity_report(identity: dict) -> str:
    """Human-facing block in the plan section 17 layout."""
    similarity = identity.get("identity_similarity")
    similarity_text = "n/a (no paired faces)" if similarity is None else f"{similarity:.2f}"
    return (
        "[Face identity preservation]\n"
        f"Face Count            {identity.get('face_count', 0)}\n"
        f"Enhanced Faces        {identity.get('enhanced_faces', 0)}\n"
        f"Identity Similarity   {similarity_text}\n"
        f"backend: {identity.get('backend')}\n"
        f"{identity.get('notice', IDENTITY_NOTICE)}"
    )
