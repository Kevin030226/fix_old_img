"""Ground Truth evaluation mode (plan section 16).

When the caller can supply a reference photo, the plan prescribes the
"with Ground Truth" family: PSNR / SSIM / MAE / **LPIPS**, all computed against
that reference instead of the degraded input. This module implements the
LPIPS half of that contract:

  - the `lpips` package + its VGG weights when installed (paper-accurate)
  - otherwise a dependency-free surrogate: multi-scale Laplacian-pyramid
    absolute-difference energy (L1 at multiple spatial frequencies), which
    tracks perceptual distance far better than raw PSNR and needs no weights

Values are reported as `lpips` (backend: "lpips_vgg" | "laplacian_surrogate")
with `reference_type = "ground_truth"`, and every user-facing string repeats
the plan's caveat: metrics against a reference quantify *difference from the
reference*, not absolute photographic quality.
"""
import cv2
import numpy as np

from app.core.logging import get_logger

logger = get_logger("fiximg.evaluation.gt")

REFERENCE_TYPE = "ground_truth"

GT_NOTICE = (
    "⚠ Ground-Truth metrics quantify the difference from the provided reference; "
    "they do not by themselves represent subjective restoration quality."
)

# Module-level cache so the torch/lpips model loads at most once per process.
_LPIPS_MODEL = None
_LPIPS_TRIED = False


def _load_lpips():
    """Return (lpips_model, torch) when the package and torch are importable."""
    global _LPIPS_MODEL, _LPIPS_TRIED
    if _LPIPS_TRIED:
        return _LPIPS_MODEL
    _LPIPS_TRIED = True
    try:
        import lpips  # noqa: PLC0415 — heavy import kept lazy; needs torch

        _LPIPS_MODEL = lpips.LPIPS(net="vgg")
        for param in _LPIPS_MODEL.parameters():
            param.requires_grad = False
    except Exception as exc:  # noqa: BLE001 — fall back to the surrogate
        logger.info("lpips package unavailable (%s); using the Laplacian surrogate", exc)
        _LPIPS_MODEL = None
    return _LPIPS_MODEL


def _reset_lpips_cache() -> None:
    """Test helper: forget the cached model so it is probed again."""
    global _LPIPS_MODEL, _LPIPS_TRIED
    _LPIPS_MODEL = None
    _LPIPS_TRIED = False


def _lpips_score(model, reference: np.ndarray, result: np.ndarray) -> float | None:
    """Score [-1, 1] via the real LPIPS model (lower = more similar)."""
    import torch  # noqa: PLC0415

    def _tensor(img):
        x = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW in [0,1]
        return 2.0 * x - 1.0  # LPIPS expects [-1, 1]

    with torch.no_grad():
        value = model(_tensor(reference), _tensor(result))
    return float(value.item())


def _laplacian_surrogate(reference: np.ndarray, result: np.ndarray,
                         levels: int = 4) -> float:
    """Weighted multi-scale Laplacian-pyramid L1 energy (lower = more similar).

    A pragmatic LPIPS stand-in: absolute differences of consecutive pyramid
    levels emphasise structure at several spatial frequencies, unlike PSNR
    which weights every pixel equally. Scale-dependent weights roughly follow
    the frequency weighting of perceptual metrics.
    """
    ref = cv2.resize(reference, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float64)
    res = cv2.resize(result, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float64)

    gp_ref = [ref.astype(np.float64)]
    gp_res = [res.astype(np.float64)]
    for _ in range(levels):
        gp_ref.append(cv2.pyrDown(gp_ref[-1]))
        gp_res.append(cv2.pyrDown(gp_res[-1]))

    # Laplacian pyramid: level i = gaussian[i] - up(gaussian[i+1]); the coarsest
    # level adds the low-pass residual so flat-colour differences count too.
    weights = [1.0, 2.0, 4.0, 4.0, 8.0]
    score = 0.0
    for i in range(levels):
        size = (gp_ref[i].shape[1], gp_ref[i].shape[0])
        lap_ref = gp_ref[i] - cv2.pyrUp(gp_ref[i + 1], dstsize=size)
        lap_res = gp_res[i] - cv2.pyrUp(gp_res[i + 1], dstsize=size)
        score += weights[i] * float(np.abs(lap_ref - lap_res).mean())
    score += weights[levels] * float(np.abs(gp_ref[levels] - gp_res[levels]).mean())
    total_weight = sum(weights[:levels + 1])
    return score / total_weight if total_weight else 0.0


def compute_ground_truth_metrics(reference_path: str, result_path: str) -> dict | None:
    """LPIPS-style perceptual distance between the reference and the result.

    Returns None when either image cannot be read. The dict always names the
    backend so reports can disclose how the score was produced.
    """
    from app.services.evaluation_service import _read_image

    reference = _read_image(reference_path)
    result = _read_image(result_path)
    if reference is None or result is None:
        return None
    if reference.shape != result.shape:
        result = cv2.resize(result, (reference.shape[1], reference.shape[0]))

    model = _load_lpips()
    if model is not None:
        score = _lpips_score(model, reference, result)
        backend = "lpips_vgg"
        if score is None:
            score = _laplacian_surrogate(reference, result)
            backend = "laplacian_surrogate"
    else:
        score = _laplacian_surrogate(reference, result)
        backend = "laplacian_surrogate"

    return {"lpips": round(score, 4), "backend": backend, "notice": GT_NOTICE}


def format_ground_truth_report(metrics: dict) -> str:
    """Human-facing block appended to the evaluation text."""
    return (
        "[Ground-Truth evaluation (plan §16)]\n"
        f"LPIPS: {metrics.get('lpips', 0)} (lower = closer to the reference; "
        f"backend: {metrics.get('backend')})\n"
        f"{metrics.get('notice', GT_NOTICE)}"
    )
