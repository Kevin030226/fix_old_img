"""Model backends: the uniform contract and its implementations (plan §3.5.1).

Shared helpers that more than one backend needs live here rather than being
copy-pasted, so the conventions stay one thing. Importing this package must not
import the backends themselves: several of them pull torch or a vendored tree,
and the runtime selects an implementation per call.
"""
from __future__ import annotations


def device_index(hint: str | int | None) -> str | int:
    """Map a runtime device hint onto the vendored ``gpu_ids`` convention.

    ``-1`` is what both vendored trees mean by CPU, and their model code only
    calls ``.cuda()`` when the parsed id list is non-empty — so this mapping
    decides whether weights land on a GPU. ``auto`` (and an unset value) resolves
    by asking torch, because "auto" is what ``FIXIMG_DEVICE`` ships with;
    ``cpu``/``none`` mean "do not use a GPU" and answer ``-1`` everywhere.
    """
    text = str(hint if hint is not None else "").strip().lower()
    if text in ("", "auto"):
        try:
            import torch

            return 0 if torch.cuda.is_available() else -1
        except ImportError:  # pragma: no cover - torch is a declared dependency
            return -1
    if text in ("cpu", "none", "-1"):
        # `none` is a refusal, not a question. Grouping it with `auto` was
        # invisible on a CPU-only machine, where both answer -1; on a machine with
        # CUDA it silently moved weights onto the GPU for a deployment that had
        # asked for the opposite.
        return -1
    if text.startswith("cuda"):
        parts = text.split(":", 1)
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return int(text) if text.lstrip("-").isdigit() else -1


__all__ = ["device_index"]
