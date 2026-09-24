"""ImageTiler — high-resolution tile/patch inference (plan section 18).

Very large photos cannot be pushed through the restoration model at once
(memory), and the plan explicitly rejects simply raising the max-side limit.
Instead the image is cut into overlapping tiles, each tile is processed
independently, and the results are re-blended into a full-resolution output
with a feathered mask so seams stay invisible.

Flow (plan section 18):

    8000 x 6000 -> tiles -> local inference -> overlap -> blending -> 8000 x 6000

Configuration (settings):
    FIXIMG_TILE_SIZE     tile edge in px; <= 0 disables tiling
    FIXIMG_TILE_OVERLAP  overlap between neighbouring tiles in px

Compatibility note (plan section 18): tiling is only enabled for stages whose
local behaviour is well defined — model quality on 1536-px crops must match
the full-image result before a stage opts in.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("fiximg.tiler")


@dataclass(frozen=True)
class Tile:
    """One tile: its position in the full image and the crop content."""

    x: int
    y: int
    w: int
    h: int
    image: Image.Image


def _axis_plans(length: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """Return [(offset, size)] crops along one axis covering [0, length).

    Guarantees coverage: starts every `tile - overlap` px, and if the last
    window would not reach the edge the final tile is pinned to the edge.
    """
    if length <= tile:
        return [(0, length)]
    step = max(tile - overlap, 1)
    plans = []
    offset = 0
    while offset + tile < length:
        plans.append((offset, tile))
        offset += step
    plans.append((max(length - tile, 0), min(tile, length)))
    # dedupe a pinned tail that coincides with the previous window
    if len(plans) > 1 and plans[-1] == plans[-2]:
        plans.pop()
    return plans


def plan_tiles(width: int, height: int, tile_size: int | None = None,
               overlap: int | None = None) -> list[tuple[int, int, int, int]]:
    """Return tile rectangles (x, y, w, h) covering the whole image."""
    tile = tile_size if tile_size is not None else settings.tile_size
    overlap = overlap if overlap is not None else settings.tile_overlap
    overlap = min(overlap, max(tile - 1, 0))
    rects = []
    for y, h in _axis_plans(height, tile, overlap):
        for x, w in _axis_plans(width, tile, overlap):
            rects.append((x, y, w, h))
    return rects


def tiling_enabled(image: Image.Image, tile_size: int | None = None,
                   threshold: float = 1.0) -> bool:
    """True when the image is large enough (and tiling not disabled) to split.

    `threshold` > 1 lets a stage opt in only for images well above the tile
    size (e.g. 2x), keeping mid-size images on the well-tested single path.
    """
    tile = tile_size if tile_size is not None else settings.tile_size
    if tile <= 0:
        return False
    long_side = max(image.size)
    return long_side > threshold * tile


def split_tiles(image: Image.Image, tile_size: int | None = None,
                overlap: int | None = None) -> list[Tile]:
    """Cut an RGB image into overlapping tiles."""
    rects = plan_tiles(image.width, image.height, tile_size, overlap)
    return [
        Tile(x=x, y=y, w=w, h=h, image=image.crop((x, y, x + w, y + h)))
        for x, y, w, h in rects
    ]


def _feather_weights(w: int, h: int, ramp: int) -> np.ndarray:
    """2-D blend weight in [0, 1]: 1 inside, linear ramp to 0 at tile edges.

    `ramp` is the feather width in px (half the overlap on each side). Edges
    of the full image have no ramp — there is no neighbour to blend with.
    """
    wx = np.ones(w, dtype=np.float64)
    wy = np.ones(h, dtype=np.float64)
    ramp = max(min(ramp, w // 2), 0)
    if ramp:
        wx[:ramp] = np.linspace(1.0 / ramp, 1.0, ramp)
        wx[-ramp:] = np.linspace(1.0, 1.0 / ramp, ramp)
    ramp_y = max(min(ramp, h // 2), 0)
    if ramp_y:
        wy[:ramp_y] = np.linspace(1.0 / ramp_y, 1.0, ramp_y)
        wy[-ramp_y:] = np.linspace(1.0, 1.0 / ramp_y, ramp_y)
    return wy[:, None] * wx[None, :]


def merge_tiles(tiles: list[Tile], width: int, height: int,
                overlap: int | None = None) -> Image.Image:
    """Blend processed tiles back into one full-resolution RGB image.

    Overlapping regions are combined with a linear feather (weighted average),
    which removes visible seams between neighbouring tiles.
    """
    if not tiles:
        raise ValueError("merge_tiles requires at least one tile")
    overlap = overlap if overlap is not None else settings.tile_overlap
    ramp = max(overlap // 2, 0)

    acc = np.zeros((height, width, 3), dtype=np.float64)
    weight = np.zeros((height, width), dtype=np.float64)
    for tile in tiles:
        arr = np.asarray(tile.image.convert("RGB"), dtype=np.float64)
        wgt = _feather_weights(tile.w, tile.h, ramp)
        region = (slice(tile.y, tile.y + tile.h), slice(tile.x, tile.x + tile.w))
        acc[region] += arr * wgt[:, :, None]
        weight[region] += wgt

    weight = np.maximum(weight, 1e-6)
    merged = np.clip(acc / weight[:, :, None], 0, 255).astype(np.uint8)
    return Image.fromarray(merged, mode="RGB")


def process_tiled(image: Image.Image, runner, tile_size: int | None = None,
                  overlap: int | None = None) -> tuple[Image.Image, dict]:
    """Split -> per-tile inference -> feathered merge.

    `runner(tile_image) -> Image.Image` is the per-tile model call (must
    preserve the tile size; it is resized back if a model changes it).

    Returns (merged_image, info) where info carries the tile statistics for
    the stage metadata / run report.
    """
    tile_size = tile_size if tile_size is not None else settings.tile_size
    overlap = overlap if overlap is not None else settings.tile_overlap
    tiles = split_tiles(image, tile_size, overlap)
    processed = []
    for i, tile in enumerate(tiles):
        out = runner(tile.image)
        if out is None:
            raise RuntimeError(f"tile runner returned nothing for tile {i}")
        if out.size != tile.image.size:
            out = out.resize(tile.image.size, Image.LANCZOS)
        processed.append(Tile(x=tile.x, y=tile.y, w=tile.w, h=tile.h, image=out))
        logger.info("tile %d/%d done (%dx%d)", i + 1, len(tiles), tile.w, tile.h)
    merged = merge_tiles(processed, image.width, image.height, overlap)
    info = {
        "tile_count": len(tiles),
        "tile_size": tile_size,
        "tile_overlap": overlap,
    }
    return merged, info
