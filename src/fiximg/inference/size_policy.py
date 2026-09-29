"""Image size policy — one place that decides how big an input may be (§3.5.5).

The plan's complaint is that every model currently applies its own resize rule,
implicitly and inconsistently::

    ≤ 1024       → direct
    1024~2048    → adaptive resize
    2048~4096    → tile / staged
    >4096        → reject or pre-resize

    「不要让每个模型自己处理 resize 策略。」

V3 centralises the decision here, so it is explicit, configurable and auditable
— the chosen strategy is recorded in the run report and visible per task.

How each tier behaves:

* **direct** — the image goes to the stages untouched.
* **adaptive_resize** — the image is scaled down so its long side fits the
  model-friendly size, and the *final result is scaled back* to the original
  dimensions. The user-visible contract (output size == input size) is
  preserved; only the models see a bounded input. Opt out with
  ``FIXIMG_SIZE_ADAPTIVE_RESIZE=false``.
* **tile** — the runtime hands the image to the tiled path (plan §18), which
  restores it in overlapping patches and feather-blends them back.
* **reject** — a clear error instead of an OOM several minutes later.

Thresholds come from settings so a deployment can tune them without code
changes (plan §3.3).
"""
from __future__ import annotations

from dataclasses import dataclass

#: Strategy identifiers (stable strings — they appear in reports and APIs).
DIRECT = "direct"
ADAPTIVE_RESIZE = "adaptive_resize"
TILE = "tile"
REJECT = "reject"

ALL_STRATEGIES = (DIRECT, ADAPTIVE_RESIZE, TILE, REJECT)


@dataclass(frozen=True, slots=True)
class SizeDecision:
    """What the runtime should do with an input of a given size."""

    strategy: str
    #: Long side the input should be resized to (adaptive_resize only).
    target_long_side: int | None = None
    #: Scale factor applied to the input (1.0 when nothing is resized).
    scale: float = 1.0
    #: Human-readable justification, recorded in the run report.
    reason: str = ""

    @property
    def resizes(self) -> bool:
        return self.strategy == ADAPTIVE_RESIZE and self.scale < 1.0

    @property
    def rejects(self) -> bool:
        return self.strategy == REJECT

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "target_long_side": self.target_long_side,
            "scale": round(self.scale, 6),
            "reason": self.reason,
        }


def decide(
    size: tuple[int, int],
    *,
    direct_max: int,
    resize_max: int,
    tile_max: int,
    adaptive_resize: bool = True,
) -> SizeDecision:
    """Classify ``size`` (width, height) into one of the four tiers.

    ``direct_max``/``resize_max``/``tile_max`` are long-side thresholds; the
    caller supplies them from settings so this function stays pure and testable.
    """
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        # Both dimensions matter: a zero-width image would fail inside PIL with
        # a much less useful message.
        return SizeDecision(
            REJECT, reason=f"Input has a non-positive dimension ({width}x{height})"
        )

    long_side = max(width, height)

    if long_side > tile_max:
        return SizeDecision(
            REJECT,
            reason=(
                f"Long side {long_side}px exceeds the {tile_max}px limit; "
                "pre-resize the image before uploading"
            ),
        )

    if long_side > resize_max:
        return SizeDecision(
            TILE,
            reason=f"Long side {long_side}px is in the tiled band (>{resize_max}px)",
        )

    if long_side > direct_max:
        if not adaptive_resize:
            return SizeDecision(
                DIRECT,
                reason=(
                    f"Long side {long_side}px is in the adaptive band (>{direct_max}px) "
                    "but adaptive resize is disabled"
                ),
            )
        scale = direct_max / float(long_side)
        return SizeDecision(
            ADAPTIVE_RESIZE,
            target_long_side=direct_max,
            scale=scale,
            reason=(
                f"Long side {long_side}px scaled to {direct_max}px for inference; "
                "the result is restored to the original size"
            ),
        )

    return SizeDecision(DIRECT, reason=f"Long side {long_side}px is within the direct limit")


def decide_from_settings(size: tuple[int, int], settings_obj=None) -> SizeDecision:
    """``decide`` using the configured thresholds (plan §3.3)."""
    from fiximg.config import settings as default_settings

    settings_obj = settings_obj or default_settings
    return decide(
        size,
        direct_max=int(getattr(settings_obj, "size_direct_max", 1024)),
        resize_max=int(getattr(settings_obj, "size_resize_max", 2048)),
        tile_max=int(getattr(settings_obj, "size_tile_max", 4096)),
        adaptive_resize=bool(getattr(settings_obj, "size_adaptive_resize", True)),
    )


def resize_for_inference(image, decision: SizeDecision):
    """Apply the adaptive resize; returns ``(image, original_size)``.

    The original size is returned so the caller can restore it once the stages
    have finished — the policy must not change what the user downloads.
    """
    from PIL import Image

    original = image.size
    if not decision.resizes or decision.target_long_side is None:
        return image, original

    width, height = original
    scale = decision.scale
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(target, Image.Resampling.LANCZOS)
    return resized, original


def restore_original_size(image, original_size: tuple[int, int]):
    """Scale a result back to the caller's input size (plan §3.5.5)."""
    from PIL import Image

    if image is None or tuple(original_size) == tuple(image.size):
        return image
    return image.resize(tuple(original_size), Image.Resampling.LANCZOS)


__all__ = [
    "ADAPTIVE_RESIZE",
    "ALL_STRATEGIES",
    "DIRECT",
    "REJECT",
    "TILE",
    "SizeDecision",
    "decide",
    "decide_from_settings",
    "resize_for_inference",
    "restore_original_size",
]
