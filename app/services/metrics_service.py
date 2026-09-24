"""No-reference quality metrics + limitation notice (plan §16).

The difference metrics in evaluation_service compare the restored result to the
degraded input — they are "input-output difference" indicators, never quality
scores. This module adds reference-free statistics of the OUTPUT image:

  - sharpness  : variance of the Laplacian (higher = more high-frequency detail)
  - contrast   : RMS contrast (std of luminance)
  - brightness : mean luminance
  - noise      : median absolute Laplacian deviation (higher = noisier)

All values are computed with numpy/cv2 only (no heavyweight NR-IQA deps).
A `reference_type` of "no_reference" is recorded in the metrics table so
query/report layers can distinguish them from PSNR/SSIM.
"""
import cv2
import numpy as np

REFERENCE_TYPE = "no_reference"

#: appended to every evaluation text (plan §16 — indicator limitations)
LIMITATION_NOTICE = (
    "⚠ These metrics are objective references only and do not fully represent subjective visual quality; "
    "for definitive judgment, please refer to the actual preview effect."
)


def compute_no_reference_metrics(image_path: str) -> dict | None:
    """Compute reference-free statistics of one image; None when unreadable."""
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.float64)
    except Exception:  # noqa: BLE001
        return None

    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(lap.var())
    contrast = float(gray.std())
    brightness = float(gray.mean())
    # Noise proxy: median absolute Laplacian response (robust to large edges).
    noise = float(np.median(np.abs(lap - np.median(lap))))

    return {
        "sharpness": round(sharpness, 2),
        "contrast": round(contrast, 2),
        "brightness": round(brightness, 2),
        "noise": round(noise, 4),
    }


def format_no_reference_report(metrics: dict) -> str:
    """Human-facing block appended to the evaluation text."""
    return (
        "[No-reference indicators of the restored output]\n"
        f"Sharpness (Laplacian var, higher=sharper): {metrics.get('sharpness', 0)}\n"
        f"Contrast (RMS): {metrics.get('contrast', 0)}\n"
        f"Brightness (mean luma): {metrics.get('brightness', 0)}\n"
        f"Noise estimate (lower=cleaner): {metrics.get('noise', 0)}\n"
        f"{LIMITATION_NOTICE}"
    )
