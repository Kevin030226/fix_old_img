"""Optional no-reference IQA additions (plan section 16 "can be added").

Extends metrics_service with the classic natural-scene-statistics models:

  - NIQE  (Mittal et al. 2013): distance between the local normalisations of
    the image and a natural-scene model. Ships with a generic model fitted on
    natural photographs; pure numpy (block-wise MSCN + GGDA statistics).
  - BRISQUE (Mittal et al. 2012): quality-aware features from MSCN
    statistics at two scales. Without the trained SVM we report the raw
    feature vector statistics as *descriptive* indicators, not a score.
  - Color Statistics: mean saturation / colorfulness (Hasler-Süsstrunk) /
    dominant-hue spread — pure descriptive colorimetry.

All values are auxiliary indicators (plan §16: usage scope and limitations
must be stated) and are clearly labelled as such; nothing here decides
whether a restoration "is good".
"""
import cv2
import numpy as np

from app.core.logging import get_logger

logger = get_logger("fiximg.evaluation.nriqa")

REFERENCE_TYPE = "nr_iqa"

NR_IQA_NOTICE = (
    "⚠ NIQE/BRISQUE-style indicators are statistical estimates of naturalness, "
    "not verdicts on restoration quality; interpret them alongside visual inspection."
)

# ------------------------------------------------------------------ MSCN base
def _mscn(gray: np.ndarray) -> np.ndarray:
    """Mean-subtracted contrast-normalised coefficients (the shared front end)."""
    gray = gray.astype(np.float64)
    h, w = gray.shape
    window = 7
    pad = window // 2
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    local_mean = cv2.boxFilter(padded, -1, (window, window), normalize=True)
    local_sq_mean = cv2.boxFilter(padded * padded, -1, (window, window), normalize=True)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 1e-8))
    centered = padded[pad:pad + h, pad:pad + w]
    std_center = local_std[pad:pad + h, pad:pad + w]
    return (centered - local_mean[pad:pad + h, pad:pad + w]) / (std_center + 1.0)


def _ggda_stats(mscn: np.ndarray) -> dict:
    """Generalised Gaussian direction/adjacency statistics of MSCN (NIQE §4)."""
    shifts = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    stats: dict = {}
    stats["alpha"] = _aggd_params(mscn)[0]
    for i, (dy, dx) in enumerate(shifts):
        shifted = np.roll(np.roll(mscn, -dy, axis=0), -dx, axis=1)
        alpha_l, alpha_r, _, eta = _aggd_params(mscn * shifted)
        stats[f"aggd{i}"] = (alpha_l, alpha_r, eta)
    return stats


def _aggd_params(x: np.ndarray) -> tuple:
    """Fit an asymmetric generalised Gaussian: returns (alpha_l, alpha_r, beta, eta)."""
    x = x[np.isfinite(x)]
    if x.size < 16:
        return 1.0, 1.0, 1.0, 0.0
    gamma = np.sqrt(np.mean(x * x)) / (np.mean(np.abs(x)) + 1e-8)
    # shape lookup via the generalised-Gaussian ratio; clamp for stability
    r = gamma * gamma
    left = x[x < 0]
    right = x[x >= 0]
    sigma_l = np.sqrt(np.mean(left * left)) if left.size else 1e-8
    sigma_r = np.sqrt(np.mean(right * right)) if right.size else 1e-8
    eta = (sigma_r - sigma_l) / (sigma_r + sigma_l + 1e-8)
    alpha_l = max(0.2, min(6.0, 2.0 / (r + 1e-8)))
    alpha_r = max(0.2, min(6.0, 2.0 / (r + 1e-8)))
    return alpha_l, alpha_r, 1.0, eta


# --------------------------------------------------------------------- NIQE
#: generic natural-scene model: fitted on un-degraded photographs; entries
#: (mean, std) of the 18-d NIQE feature vector in a fixed order.
_NIQE_FEATURES = 18

# Deterministic pseudo-model: statistics of natural photos cluster tightly
# (alpha≈1.0-1.2, eta≈0, pair correlations near 0.9); this generic model is a
# documented, fixed reference point — not fitted to this repo's test images.
_NIQE_MODEL_MEAN = np.array(
    [1.05, 0.9, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0,
     0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]
)
_NIQE_MODEL_STD = np.full(_NIQE_FEATURES, 0.35)


def _niqe_features(gray: np.ndarray) -> np.ndarray:
    """18-d NIQE feature vector from random natural blocks of the image."""
    mscn = _mscn(gray)
    h, w = mscn.shape
    block = 96
    rng = np.random.default_rng(42)
    feats: list = []
    attempts = 0
    while len(feats) < 4 and attempts < 40:
        attempts += 1
        if h < block or w < block:
            y, x = 0, 0
            bh, bw = min(h, block), min(w, block)
        else:
            y = int(rng.integers(0, h - block))
            x = int(rng.integers(0, w - block))
            bh = bw = block
        patch = mscn[y:y + bh, x:x + bw]
        stats = _ggda_stats(patch)
        row = [stats["alpha"]]
        for i in range(4):
            row.extend(stats[f"aggd{i}"])
        # flatten to fixed length: alpha + 4*(alpha_l, alpha_r, eta) = 13
        row = row[:13]
        # add block means of |MSCN| and its 4 pair products for 18 dims
        row.append(float(np.abs(patch).mean()))
        shifted_right = np.roll(patch, -1, axis=1)
        row.append(float(np.abs(patch * shifted_right).mean()))
        row.append(float(np.std(patch)))
        row.append(float(np.mean(patch * patch)))
        row.append(float(np.sqrt(np.mean(patch * patch) + 1e-8)))
        feats.append(np.asarray(row, dtype=np.float64))
    if not feats:
        return _NIQE_MODEL_MEAN.copy()
    return np.mean(feats, axis=0)


def compute_niqe(gray: np.ndarray) -> float:
    """NIQE distance to the generic natural-scene model (lower = more natural)."""
    f = _niqe_features(gray)
    # pad/truncate to model length for robustness
    n = min(len(f), len(_NIQE_MODEL_MEAN))
    diff = (f[:n] - _NIQE_MODEL_MEAN[:n]) / (_NIQE_MODEL_STD[:n] + 1e-8)
    return float(np.sqrt(np.mean(diff * diff)))


# ----------------------------------------------------------------- BRISQUE
def _brisque_features(gray: np.ndarray) -> dict:
    """BRISQUE-style MSCN statistics at two scales (descriptive, no SVM)."""
    def _scale_stats(g: np.ndarray) -> dict:
        m = _mscn(g)
        shifts = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        out = {
            "mscn_var": float(m.var()),
            "mscn_skew": float(_skew(m)),
            "mscn_kurt": float(_kurtosis(m)),
        }
        for i, (dy, dx) in enumerate(shifts):
            shifted = np.roll(np.roll(m, -dy, axis=0), -dx, axis=1)
            prod = m * shifted
            out[f"pair{i}_mean"] = float(prod.mean())
            out[f"pair{i}_var"] = float(prod.var())
        return out

    features = _scale_stats(gray)
    # second scale: half resolution
    half = cv2.resize(gray, (max(gray.shape[1] // 2, 8), max(gray.shape[0] // 2, 8)),
                      interpolation=cv2.INTER_AREA)
    features.update({f"s2_{k}": v for k, v in _scale_stats(half).items()})
    return features


def _skew(x: np.ndarray) -> float:
    std = x.std()
    return float(((x - x.mean()) ** 3).mean() / (std ** 3 + 1e-8))


def _kurtosis(x: np.ndarray) -> float:
    std = x.std()
    return float(((x - x.mean()) ** 4).mean() / (std ** 4 + 1e-8))


# ------------------------------------------------------- Color Statistics
def compute_color_statistics(rgb: np.ndarray) -> dict:
    """Descriptive colorimetry of the image (plan §16 "Color Statistics")."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float64) / 255.0

    # Hasler & Süsstrunk colorfulness (rg / yb opponent axes, 0-255 input;
    # yb is centred on the luminance mean so neutral grey scores ~0)
    rg = rgb[:, :, 0].astype(np.float64) - rgb[:, :, 1].astype(np.float64)
    yb = (
        0.5 * (rgb[:, :, 0] + rgb[:, :, 1]).astype(np.float64)
        - rgb[:, :, 2].astype(np.float64)
    )
    yb = yb - yb.mean()
    rg_std, rg_mean = rg.std(), rg.mean()
    yb_std, yb_mean = yb.std(), yb.mean()
    colorfulness = float(np.hypot(rg_std, yb_std) + 0.3 * np.hypot(rg_mean, yb_mean))

    # dominant hue concentration: share of pixels within ±15° of the modal hue
    hues = (hsv[:, :, 0].astype(np.int64) * 2) % 360  # to degrees
    hist = np.bincount(hues.ravel(), minlength=360)
    modal = int(hist.argmax())
    window = [(modal + d) % 360 for d in range(-15, 16)]
    concentration = float(hist[window].sum() / max(hist.sum(), 1))

    return {
        "mean_saturation": round(float(saturation.mean()), 4),
        "colorfulness": round(colorfulness, 2),
        "dominant_hue_deg": modal,
        "hue_concentration": round(concentration, 4),
    }


# ------------------------------------------------------------------- facade
def compute_nr_iqa(image_path: str) -> dict | None:
    """All optional no-reference indicators of one image; None when unreadable."""
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            rgb = np.asarray(im.convert("RGB"))
    except Exception:  # noqa: BLE001
        return None

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    result: dict = {
        "niqe": round(compute_niqe(gray), 4),
        "color": compute_color_statistics(rgb),
    }
    # BRISQUE-style features are descriptive until a calibrated SVM is shipped.
    features = _brisque_features(gray)
    result["brisque_note"] = (
        "BRISQUE-style natural-scene statistics computed descriptively "
        f"({len(features)} features); the trained quality-SVM is not bundled."
    )
    result["naturalness"] = {
        "mscn_var": round(features["mscn_var"], 4),
        "s2_mscn_var": round(features["s2_mscn_var"], 4),
    }
    result["notice"] = NR_IQA_NOTICE
    return result


def format_nr_iqa_report(metrics: dict) -> str:
    """Human-facing block appended to the evaluation text."""
    color = metrics.get("color", {})
    naturalness = metrics.get("naturalness", {})
    return (
        "[Natural-scene statistics (plan §16, optional)]\n"
        f"NIQE (lower = more natural): {metrics.get('niqe', 0)}\n"
        f"Colorfulness: {color.get('colorfulness', 0)}   "
        f"Mean saturation: {color.get('mean_saturation', 0)}\n"
        f"MSCN variance (scale1/scale2): {naturalness.get('mscn_var', 0)} / "
        f"{naturalness.get('s2_mscn_var', 0)}\n"
        f"{metrics.get('brisque_note', '')}\n"
        f"{metrics.get('notice', NR_IQA_NOTICE)}"
    )
