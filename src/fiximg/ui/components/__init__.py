"""Reusable Gradio components (plan §3.2 `ui/components/`).

The tab builders in :mod:`fiximg.ui.gradio_app` each need the same pair — a
result image plus a Before/After slider — so the construction lives here rather
than being repeated (and drifting) five times.

Kept deliberately small: these are layout helpers, not a component framework.
"""
from __future__ import annotations

from typing import Literal

import gradio as gr

#: Slider label used when the "after" side is a restored image.
BEFORE_AFTER_LABEL = "Before / After — drag the handle (original ◀ ▶ restored)"

#: Slider label used when the "after" side is a scratch mask.
MASK_LABEL = "Original ◀ ▶ detected mask — drag the handle"

#: What Gradio hands the component back as (its own ``type`` parameter).
SliderType = Literal["numpy", "pil", "filepath"]


def result_column(
    label: str,
    elem_id: str,
    slider_label: str | None = None,
    slider_type: SliderType = "filepath",
):
    """A result image plus its comparison slider.

    ``slider_label`` is overridden for the detection tab, where the result *is*
    the scratch mask, so the slider reads as original ◀ ▶ mask (plan §3.9.2).
    Returns ``(result_image, compare_slider, download_button)`` in the order the
    submit handler yields them.

    The download control is part of the column because plan §3.9.1 asks for a
    result the user can take away, and ``gr.Image`` in Gradio 6 renders a preview
    with no save affordance of its own — the history panel already had one, the
    tabs that produce the image did not.
    """
    result = gr.Image(label=label)
    compare = gr.ImageSlider(
        label=slider_label or BEFORE_AFTER_LABEL,
        type=slider_type,
        elem_id=elem_id,
    )
    download = gr.DownloadButton("Download result", size="sm", visible=False)
    return result, compare, download


def comparison_slider(elem_id: str, label: str = BEFORE_AFTER_LABEL, visible: bool = True):
    """A standalone Before/After slider (used by the history panel)."""
    return gr.ImageSlider(
        label=label, type="filepath", elem_id=elem_id, visible=visible,
    )


def preview_row(labels: list[str]):
    """A row of read-only preview images, returned in the requested order.

    Returning the components lets the caller wire them to a callback without
    re-deriving the order (a silent source of mismatched outputs).
    """
    with gr.Row():
        return [
            gr.Image(label=label, type="filepath", interactive=False)
            for label in labels
        ]


__all__ = [
    "BEFORE_AFTER_LABEL",
    "MASK_LABEL",
    "comparison_slider",
    "preview_row",
    "result_column",
]
