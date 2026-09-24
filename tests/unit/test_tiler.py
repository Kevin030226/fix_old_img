"""Unit tests for the ImageTiler (plan section 18)."""
import numpy as np
import pytest
from PIL import Image

from app.inference import tiler


def _gradient_image(width=300, height=200) -> Image.Image:
    """Smooth gradient so blending errors are easy to spot."""
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    arr = np.stack(
        [
            (xx * 255 // max(width - 1, 1)),
            (yy * 255 // max(height - 1, 1)),
            np.full((height, width), 128),
        ],
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ------------------------------------------------------------------- planning
def test_plan_tiles_covers_image_exactly():
    rects = tiler.plan_tiles(300, 200, tile_size=128, overlap=32)
    assert rects, "must produce tiles"
    mask = np.zeros((200, 300), dtype=bool)
    for x, y, w, h in rects:
        assert w <= 128 and h <= 128
        assert 0 <= x and 0 <= y
        assert x + w <= 300 and y + h <= 200
        mask[y:y + h, x:x + w] = True
    assert mask.all(), "tiles must cover the whole image"


def test_plan_tiles_small_image_single_tile():
    assert tiler.plan_tiles(100, 80, tile_size=128, overlap=32) == [(0, 0, 100, 80)]


def test_plan_tiles_pinned_tail_reaches_edge():
    rects = tiler.plan_tiles(300, 300, tile_size=128, overlap=120)
    assert max(x + w for x, _, w, _ in rects) == 300
    assert max(y + h for _, y, _, h in rects) == 300


# --------------------------------------------------------------------- split
def test_split_tiles_roundtrip_content():
    image = _gradient_image()
    tiles = tiler.split_tiles(image, tile_size=128, overlap=32)
    assert len(tiles) > 1
    first = tiles[0]
    assert (first.x, first.y) == (0, 0)
    np.testing.assert_array_equal(
        np.asarray(first.image), np.asarray(image.crop((0, 0, first.w, first.h)))
    )


# -------------------------------------------------------------------- feather
def test_feather_weights_center_is_one_edges_are_low():
    w = tiler._feather_weights(64, 48, ramp=8)
    assert w[24, 32] == 1.0
    assert w[0, 32] < 0.15 and w[47, 32] < 0.15


# --------------------------------------------------------------------- merge
def test_merge_tiles_identity_for_perfect_runner():
    """A runner that returns tiles unchanged must reproduce the input."""
    image = _gradient_image(200, 150)
    merged, info = tiler.process_tiled(image, lambda tile: tile, tile_size=96, overlap=24)
    assert info["tile_count"] > 1
    diff = np.abs(np.asarray(merged, dtype=int) - np.asarray(image, dtype=int))
    assert diff.mean() < 0.5, "perfect runner should round-trip"


def test_merge_tiles_blend_on_overlap():
    """Overlapping regions mix neighbouring tiles instead of hard seams."""
    image = _gradient_image(200, 100)
    calls = {"n": 0}

    def runner(tile):
        calls["n"] += 1
        # constant-but-varying output: seam would be obvious without blending
        shade = 40 + calls["n"] * 30
        return Image.fromarray(np.full((tile.height, tile.width, 3), shade, dtype=np.uint8))

    merged, _ = tiler.process_tiled(image, runner, tile_size=96, overlap=32)
    # with feathering the output transitions smoothly; without, it would jump
    row = np.asarray(merged, dtype=int)[50, :, 0]
    jumps = np.abs(np.diff(row))
    assert jumps.max() < 60, "no hard seams expected with feathered blending"


def test_merge_tiles_resizes_runner_output_back():
    image = _gradient_image(150, 150)
    merged, _ = tiler.process_tiled(
        image, lambda tile: tile.resize((tile.width // 2, tile.height // 2)),
        tile_size=100, overlap=20,
    )
    assert merged.size == (150, 150)


def test_process_tiled_rejects_empty():
    with pytest.raises(RuntimeError):
        tiler.process_tiled(_gradient_image(64, 64), lambda tile: None, tile_size=32)


# ------------------------------------------------------------------- gating
def test_tiling_enabled_gate(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "tile_size", 1536, raising=False)
    big = Image.new("RGB", (3200, 2400))
    mid = Image.new("RGB", (1536, 1200))  # long side == tile size
    small = Image.new("RGB", (800, 600))
    assert tiler.tiling_enabled(big) is True
    assert tiler.tiling_enabled(mid) is False  # == tile size, not above
    assert tiler.tiling_enabled(small) is False

    monkeypatch.setattr(settings, "tile_size", 0, raising=False)  # disabled
    assert tiler.tiling_enabled(big) is False

    # stage opt-in threshold: only way-above-tile images go tiled
    monkeypatch.setattr(settings, "tile_size", 1024, raising=False)
    assert tiler.tiling_enabled(mid, threshold=2.0) is False
    assert tiler.tiling_enabled(big, threshold=2.0) is True
