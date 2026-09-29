"""DDColor colorization backend (plan §3.5.1, §3.5.4).

Wraps the in-process DDColor pipeline owned by
:class:`~fiximg.inference.model_manager.ModelManager` behind the uniform
:class:`ModelBackend` contract, so the models API can list/health-check it and
the runtime can warm it up like any other backend.

The manager remains the single owner of the loaded pipeline: this backend never
builds a second copy.

Inference runs inside ``torch.inference_mode()`` and — when the model's manifest
entry declares a precision — inside ``torch.autocast``. Both are gated by the
:class:`~fiximg.inference.precision.PrecisionPolicy` derived from the manifest,
so a checkpoint that cannot take half precision simply declares ``precision:
fp32`` (plan §3.5.4: capability-gated, never global).
"""
from __future__ import annotations

import time

from fiximg.domain.errors import ModelUnavailableError
from fiximg.inference.backends.base import BaseModelBackend, ModelRequest, ModelResult
from fiximg.inference.model_manager import model_manager
from fiximg.inference.precision import (
    cached_policy,
    inference_context,
)


def reside_on(pipeline, device, cuda_available=None) -> str:
    """Move the resident pipeline onto ``device``; return where it ended up running.

    The routing knobs (`FIXIMG_WORKER_GPU`, `FIXIMG_GPU_ROUTING=colorize:1`) moved
    every other chain to the scheduled card because their stages hand `context.gpu` to
    the code that runs the model. This chain did not: the pipeline sat where
    `settings.device` had put it at load time while the runtime's memory sampler
    charged the allocation to the *scheduled* device, so the reported peak described a
    card the colorizer never touched. One resident copy is moved, not duplicated: the
    routing is stable within a process, so this costs a weight copy once and afterwards
    it is only a comparison.

    Returning the realised device is the other half. A worker asked for card 1 on a
    single-card host must not report card 1: what reaches the task report is where the
    model actually ran, and a decline is visible instead of silent. `cuda_available` is
    a seam so both branches are testable on any host.
    """
    from fiximg.inference.model_manager import torch_device_for

    want = torch_device_for(device)
    current = str(getattr(pipeline, "device", None) or want)
    model = getattr(pipeline, "model", None)
    if current == want or model is None:
        return current
    try:
        import torch
    except Exception:  # noqa: BLE001 — no torch, so nothing can be moved
        return current
    if cuda_available is None:
        available = bool(torch.cuda.is_available())
    else:
        # A seam that accepts either a value or a thunk, so a test can say
        # `cuda_available=lambda: False` without the lambda's truthiness standing in
        # for the answer.
        available = bool(cuda_available() if callable(cuda_available) else cuda_available)
    if want.startswith("cuda") and not available:
        return current  # the host cannot honour it: say where it really is
    target = torch.device(want)
    pipeline.model = model.to(target)
    pipeline.device = target
    return want


class DDColorBackend(BaseModelBackend):
    """Black-and-white → colour colorization (damo/cv_ddcolor_image-colorization)."""

    name = "ddcolor"
    version = "1.0"
    #: Loaded once into this process through the model manager, unlike the
    #: vendored chains that are re-entered as a subprocess per run (plan §3.5.4).
    implementation = "native"
    capabilities = frozenset({"colorize"})

    def __init__(self, config: dict | None = None, manager=None) -> None:
        super().__init__(config)
        self._manager = manager or model_manager
        self._model_size = getattr(self._manager.settings, "ddcolor_model_size", "large")

    # ------------------------------------------------------------- lifecycle
    @property
    def policy(self):
        """Execution policy declared for this model (plan §3.5.4)."""
        return cached_policy(self.name)

    def _do_load(self, device: str) -> None:
        """Load through the manager (raises FileNotFoundError when weights are absent)."""
        pipeline = self._manager.load(self.name)
        # channels_last / torch.compile are applied to the underlying nn.Module
        # when (and only when) the manifest declares them. Failures degrade to
        # eager fp32 rather than taking the model offline.
        self._apply_optimisations(pipeline)

    def _apply_optimisations(self, pipeline) -> None:
        """Route the pipeline's nn.Module through the declared policy.

        The base helper does the work and records what landed; this only knows
        where the module lives inside the DDColor pipeline.
        """
        policy = self.policy
        if not policy.is_accelerated:
            return
        model = getattr(pipeline, "model", None)
        if model is None:
            self.optimisations_applied = {
                "channels_last": False,
                "compile": False,
            } if policy.channels_last or policy.compile else {}
            return
        pipeline.model = self.apply_policy(model)

    def _do_unload(self) -> None:
        self._manager.unload(self.name)

    def warmup(self) -> None:
        """Warm the pipeline with a 64×64 dummy so the first real call is fast."""
        if not self._loaded:
            self.load(self.device)
        import numpy as np

        pipeline = self._manager.get(self.name)
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            with inference_context(self.policy, self.device):
                pipeline.process(dummy)
        except Exception:  # noqa: BLE001 — warmup must never be fatal
            pass

    def _do_infer(self, request: ModelRequest) -> ModelResult:
        import cv2
        import numpy as np
        from PIL import Image

        rgb = np.array(request.image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        started = time.perf_counter()
        policy = self.policy
        try:
            # §3.15: hold the active version for the whole inference so a
            # concurrent hot swap cannot unload the handle mid-call.
            with self._manager.acquire(self.name) as lease:
                ran_on = reside_on(lease.handle, request.device)
                # §3.5.4: inference_mode + declared autocast around the model call,
                # opened for the device it is *really* on — autocast for a device that
                # does not exist is a warning plus a silent fall back to fp32.
                with inference_context(policy, ran_on):
                    out_bgr = lease.handle.process(bgr)
        except FileNotFoundError as exc:
            raise ModelUnavailableError(str(exc), details={"model": self.name}) from exc
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        return ModelResult(
            image=Image.fromarray(out_rgb),
            metadata={
                "model": f"DDColor-{self._model_size}",
                "duration_s": round(time.perf_counter() - started, 3),
                "precision": policy.describe(),
                "device": ran_on,
            },
        )

    def health(self):  # noqa: D102 - see BaseModelBackend
        health = super().health()
        health.extra = {
            "model_size": self._model_size,
            "weights": getattr(self._manager.settings, "ddcolor_weights", None),
            "precision": self.policy.describe(),
        }
        return health

    def describe(self) -> dict:  # noqa: D102 - see BaseModelBackend
        info = super().describe()
        info["model_size"] = self._model_size
        info["precision"] = self.policy.describe()
        return info


__all__ = ["DDColorBackend", "reside_on"]
