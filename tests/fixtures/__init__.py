"""Shared test fixtures: synthetic photo-like images (no weights needed)."""
import numpy as np
from PIL import Image


def make_photo(
    width: int = 64,
    height: int = 48,
    seed: int = 7,
    grayscale: bool = False,
) -> Image.Image:
    """Deterministic synthetic image with structure + noise (photo-like enough
    for metric/IO tests; no model weights involved)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    gradient = (xx * 255 // max(width - 1, 1)).astype(np.float64)
    checker = ((xx // 8 + yy // 8) % 2 * 40).astype(np.float64)
    noise = rng.normal(0, 12, size=(height, width))
    gray = np.clip(gradient + checker + noise, 0, 255).astype(np.uint8)
    if grayscale:
        return Image.fromarray(gray, mode="L")
    arr = np.stack([gray, np.roll(gray, 5, axis=1), 255 - gray], axis=2)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def save_photo(path: str, **kwargs) -> str:
    """Save a synthetic photo to `path` and return the path."""
    make_photo(**kwargs).save(path)
    return path
