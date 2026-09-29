"""Golden-image helpers shared by the CPU and weight-requiring suites (plan 搂3.10.2).

The plan is explicit that pixel-exact comparison is the wrong goal for an image
project: what must stay stable is the *characteristics* of the output 鈥?shape,
mode, dynamic range, mean/std, PSNR/SSIM against the input. A model or
preprocessing change that shifts any of those is a regression a reviewer needs to
see.

These helpers live here (rather than inside one test module) because two suites
need the identical measurement code: the synthetic baseline in ``tests/inference``
which runs anywhere, and the real-model baseline in ``tests/gpu`` which needs
weights. Different code computing "mean" in each would make the two incomparable.

Inputs are generated deterministically from a seed, so a failure is reproducible.
Refresh a baseline with ``FIXIMG_UPDATE_GOLDEN=1`` and review the diff 鈥?the file
is committed data, not test scratch.
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw

#: Characteristic set the plan asks for, minus the optional perceptual metrics
#: (LPIPS/DISTS need a model download and are covered by the gpu suite).
GOLDEN_FIELDS = (
    "mode",
    "size",
    "shape",
    "channels",
    "dtype_range",
    "mean",
    "std",
    "psnr_vs_input",
    "ssim_vs_input",
)

#: Baselines live beside the suite that first defined them, so both suites read
#: one directory rather than two copies of the same JSON.
GOLDEN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inference", "golden"
)


def make_input(seed: int = 7, size=(96, 64)) -> Image.Image:
    """A deterministic 'old photo': gradient + noise + a few bright scratches."""
    rng = np.random.default_rng(seed)
    width, height = size
    xs = np.linspace(40, 210, width, dtype=np.float32)
    ys = np.linspace(60, 190, height, dtype=np.float32)
    base = (ys[:, None] + xs[None, :]) / 2.0
    canvas = np.stack([base, base * 0.95, base * 0.85], axis=-1)

    canvas += rng.normal(0.0, 6.0, canvas.shape)  # film grain
    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGB")

    draw = ImageDraw.Draw(image)
    draw.line((8, 12, 70, 40), fill=(245, 245, 240), width=1)
    draw.line((20, 50, 88, 22), fill=(240, 240, 235), width=1)
    return image


def psnr(reference: Image.Image, image: Image.Image) -> float:
    a = np.asarray(reference.convert("RGB"), dtype=np.float64)
    b = np.asarray(image.convert("RGB"), dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0**2) / mse)


def ssim(reference: Image.Image, image: Image.Image) -> float:
    """Global SSIM (single window) 鈥?enough to detect a quality shift."""
    a = np.asarray(reference.convert("L"), dtype=np.float64)
    b = np.asarray(image.convert("L"), dtype=np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)))


def characteristics(image: Image.Image, reference: Image.Image | None = None) -> dict:
    """The measured fingerprint of an image."""
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.float64)

    fingerprint = {
        "mode": image.mode,
        "size": list(rgb.size),
        "shape": list(array.shape),
        "channels": int(array.shape[2]) if array.ndim == 3 else 1,
        "dtype_range": [float(array.min()), float(array.max())],
        "mean": round(float(array.mean()), 4),
        "std": round(float(array.std()), 4),
    }
    if reference is not None:
        fingerprint["psnr_vs_input"] = round(psnr(reference, rgb), 4)
        fingerprint["ssim_vs_input"] = round(ssim(reference, rgb), 4)
    return fingerprint


def assert_matches_golden(actual: dict, expected: dict, *, mean_tol: float = 1.0,
                          std_tol: float = 1.0, psnr_tol: float = 0.5,
                          ssim_tol: float = 0.02) -> None:
    """Compare fingerprints with tolerances rather than exact equality."""
    for field in ("mode", "size", "shape", "channels"):
        assert actual[field] == expected[field], f"{field}: {actual[field]} != {expected[field]}"

    for field, tolerance in (("mean", mean_tol), ("std", std_tol),
                             ("psnr_vs_input", psnr_tol), ("ssim_vs_input", ssim_tol)):
        if field not in expected:
            continue
        delta = abs(actual[field] - expected[field])
        assert delta <= tolerance, (
            f"{field} drifted by {delta:.4f} (limit {tolerance}): "
            f"{actual[field]} vs golden {expected[field]}"
        )


def golden_path(name: str) -> str:
    return os.path.join(GOLDEN_DIR, f"{name}.json")


def load_golden(name: str) -> dict:
    with open(golden_path(name), encoding="utf-8") as handle:
        return json.load(handle)


def save_golden(name: str, payload: dict) -> None:
    """Write a baseline. Run with ``FIXIMG_UPDATE_GOLDEN=1`` to refresh."""
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    with open(golden_path(name), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def update_requested() -> bool:
    return bool(os.environ.get("FIXIMG_UPDATE_GOLDEN"))
