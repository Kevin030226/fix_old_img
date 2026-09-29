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


# ----------------------------------------------------------- Gradio layout readers
#
# Several gates read the rendered app's own dependency graph, because that is the only
# place where "the tab wires the checkboxes it declares" can be checked at all. Reading
# `api_name` is not a selector: Gradio numbers those from one counter across every
# callback, so the day a tab gains a second kind of handler (the Auto analysis row, in
# batch 38) a prefix match starts describing a different set than it meant to. These
# selectors pick by *shape* 鈥?what a callback consumes and returns 鈥?so one definition
# serves every gate instead of three guesses that can disagree.
def demo_config():
    """The built Gradio app's config: components, dependencies, as they render."""
    from fiximg.ui import gradio_app

    return gradio_app.build_demo().get_config_file()


def components_by_id(cfg):
    """Every component, addressed by the id the dependencies reference."""
    return {component["id"]: component for component in cfg["components"]}


def component_label(cfg, component_id):
    """A component's caption, or its type when it has no caption."""
    component = components_by_id(cfg)[component_id]
    return (component.get("props") or {}).get("label") or component["type"]


def _input_types(cfg, dep):
    by_id = components_by_id(cfg)
    return [by_id[i]["type"] for i in dep["inputs"]]


def submit_deps(cfg):
    """The tab submit callbacks: ``(image, state, *switches)`` in, result image out.

    Deliberately not "whose ``api_name`` starts with ``handler``": the analysis row's
    callbacks are named from the same counter, and the history panel's row handlers also
    take a state and return an image. Position is the contract 鈥?the handler's first
    argument is the upload and its second is the account, and the first output is the
    result image.
    """
    return [
        dep for dep in cfg["dependencies"]
        if len(_input_types(cfg, dep)) >= 2
        and _input_types(cfg, dep)[0] == "image"
        and _input_types(cfg, dep)[1] == "state"
        and components_by_id(cfg)[dep["outputs"][0]]["type"] == "image"
    ]


def deps_writing_to(cfg, label):
    """The callbacks whose only output is the component captioned `label`."""
    return [
        dep for dep in cfg["dependencies"]
        if len(dep["outputs"]) == 1 and component_label(cfg, dep["outputs"][0]) == label
    ]
