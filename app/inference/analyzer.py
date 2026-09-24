"""ImageAnalyzer — first-pass degradation analysis (plan §10/§11, Phase 7).

A lightweight, heuristic analyzer (no extra model weights): it inspects the
input image and reports the observations the auto planner needs:

  - width / height
  - is_grayscale      mean saturation below threshold (old B&W photos)
  - blur_score        0=sharp .. 1=fully blurred (Laplacian-variance based)
  - scratch_score     0=clean .. 1=covered in thin bright/dark lines
                      (morphological top-hat / black-hat heuristic)
  - face_count        frontal faces via the shared detector (YuNet / Haar /
                        dlib; None when no backend is installed)

The output dict feeds `PipelinePlanner.plan_auto_restore(analysis)` — the plan
§11 contract (`analysis` -> `planner.plan(analysis)`). Values are heuristic
estimates, not model predictions; thresholds are documented per constant.
"""
import cv2
import numpy as np

from app.core.config import settings
from app.inference.face_detect import _resolve_backend, detect_face_boxes

#: Legacy module-level defaults; live values come from settings so thresholds
#: can be calibrated via environment variables (plan §17 note: thresholds must
#: be calibrated against test data).
GRAYSCALE_SAT_THRESHOLD = 16.0
SHARP_LAPLACIAN_VARIANCE = 120.0
SCRATCH_FRACTION_FULL = 0.06
SCRATCH_STRUCTURE_THRESHOLD = 30

#: analysis runs on a downscaled copy; long side capped for speed
_ANALYSIS_MAX_SIDE = 512


def _downscale(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= _ANALYSIS_MAX_SIDE:
        return image
    scale = _ANALYSIS_MAX_SIDE / long_side
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def detect_grayscale(rgb: np.ndarray) -> bool:
    """True when the mean saturation indicates a black-and-white photo."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean()) < settings.auto_grayscale_sat


def estimate_blur(gray: np.ndarray) -> float:
    """Blur score in [0, 1]; 0 = sharp, 1 = fully blurred."""
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(np.clip(1.0 - lap_var / settings.auto_sharp_laplacian, 0.0, 1.0))


def estimate_scratch(gray: np.ndarray) -> float:
    """Scratch score in [0, 1] via thin bright/dark structure density.

    Scanned-photo scratches are usually 1-2 px wide lines that pop out under a
    morphological top-hat (bright) or black-hat (dark). The score is the share
    of candidate pixels, normalized by SCRATCH_FRACTION_FULL.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    threshold = settings.auto_scratch_structure_threshold
    candidates = (tophat > threshold) | (blackhat > threshold)
    fraction = float(candidates.mean())
    return float(np.clip(fraction / settings.auto_scratch_fraction_full, 0.0, 1.0))


def estimate_resolution(image) -> dict:
    return {"width": int(image.width), "height": int(image.height)}


def detect_faces(rgb: np.ndarray):
    """Frontal-face count via the shared detector; None when no backend exists."""
    if _resolve_backend() is None:
        return None
    return int(len(detect_face_boxes(rgb)))


def analyze_image(image) -> dict:
    """Analyze a PIL/np RGB image and return the plan §11 analysis structure."""
    import numpy as _np

    rgb = image if isinstance(image, _np.ndarray) else np.asarray(image.convert("RGB"))

    small = _downscale(rgb)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    return {
        **estimate_resolution(image),
        "is_grayscale": detect_grayscale(small),
        "blur_score": estimate_blur(gray),
        "scratch_score": estimate_scratch(gray),
        "face_count": detect_faces(small),
    }


def format_analysis_summary(analysis: dict) -> str:
    """Compact one-line summary for the stage checklist / report."""
    faces = analysis.get("face_count")
    faces_text = "?" if faces is None else str(faces)
    return (
        f"grayscale={'yes' if analysis.get('is_grayscale') else 'no'} "
        f"scratch={analysis.get('scratch_score', 0):.2f} "
        f"blur={analysis.get('blur_score', 0):.2f} "
        f"faces={faces_text} "
        f"size={analysis.get('width')}x{analysis.get('height')}"
    )
