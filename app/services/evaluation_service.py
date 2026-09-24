"""Evaluation service (plan sections 16 and 31 Phase 6).

The PSNR/SSIM/MAE implemented here compare the restored result against the
degraded input. Per the V2 plan these are explicitly "input-output difference"
indicators, NOT restoration quality scores; the DB reference_type column and
every user-facing string follow that naming.
"""
import math

import cv2
import numpy as np
from PIL import Image

REFERENCE_TYPE = "input_output_difference"

_DEGRADE_MESSAGES = {
    "no_face_detected": "No face detected; face enhancement skipped, result is overall quality restoration only.",
    "face_enhance_missing": "Face enhancement produced no result; fell back to overall restoration.",
}


def _read_image(image_path: str):
    try:
        with open(image_path, "rb") as f:
            image = np.array(Image.open(f))
    except Exception:  # noqa: BLE001
        return None
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[2] > 3:
        image = image[:, :, :3]
    return image


def _calculate_psnr(img1, img2) -> float:
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def _calculate_ssim(img1, img2) -> float:
    """SSIM with a whole-image fallback for tiny images (below the Gaussian window)."""
    if min(img1.shape[:2]) < 11:
        x = img1.astype(np.float64)
        y = img2.astype(np.float64)
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        ux, uy = x.mean(), y.mean()
        vx, vy = x.var(), y.var()
        cov = ((x - ux) * (y - uy)).mean()
        return float(((2 * ux * uy + c1) * (2 * cov + c2)) / ((ux**2 + uy**2 + c1) * (vx + vy + c2)))

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu1_mu2
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def _calculate_l1(img1, img2) -> float:
    return np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64))) / 255.0


def calculate_difference_metrics(original_path: str, result_path: str) -> dict:
    """Return PSNR / SSIM / MAE between the degraded input and the restored output."""
    raw1 = _read_image(original_path)
    raw2 = _read_image(result_path)
    if raw1 is None or raw2 is None:
        raise ValueError("Metric computation failed: cannot read the original or result image")
    if raw1.shape != raw2.shape:
        raw1 = cv2.resize(raw1, (raw2.shape[1], raw2.shape[0]))
    identical = bool(np.array_equal(raw1, raw2))
    return {
        "psnr": _calculate_psnr(raw1, raw2),
        "ssim": _calculate_ssim(raw1, raw2),
        "mae": _calculate_l1(raw1, raw2),
        "identical": identical,
    }


def format_difference_report(metrics: dict, degrade_note: str | None = None) -> str:
    """Human-facing text for the Gradio result panel."""
    psnr = metrics.get("psnr", float("inf"))
    psnr_text = "∞" if psnr == float("inf") else f"{psnr:.4f}"
    lines = [
        "[Objective differences: restored vs. original (degraded input)]",
        f"PSNR: {psnr_text}",
        f"SSIM: {metrics.get('ssim', 0):.4f}",
        f"MAE: {metrics.get('mae', 0):.4f}",
    ]
    if metrics.get("identical"):
        lines.append(
            "⚠ The restored image is pixel-identical to the original (PSNR→∞): likely no effective restoration; "
            "please check the pipeline did not degrade."
        )
    if degrade_note:
        lines.append(f"⚠ {degrade_note}")
    return "\n".join(lines)


def degrade_note_from_metadata(metadata: dict) -> str | None:
    """Translate a global-restore stage report into a user-facing note."""
    report = (metadata or {}).get("report") or {}
    if not report.get("degraded_count"):
        return None
    return _DEGRADE_MESSAGES.get(
        report.get("degrade_reason"),
        "Some processing stages were skipped; the result may be incomplete.",
    )
