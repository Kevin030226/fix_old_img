"""ModelManager — unified model lifecycle (plan §6, §3.15).

Owns the model handles for one process, supports hybrid loading (eager/lazy per
model), reuse, unload, and a ``health()`` probe for ``/health/ready``.

V3 delegates the *versioning* concerns to
:class:`~fiximg.inference.versions.VersionedModelRegistry`, which keeps two
versions resident and switches routing atomically. This class keeps the simple
name-based API the stages already use:

    manager.get("ddcolor")                     # active handle (lazy-loads)
    with manager.acquire("ddcolor") as lease:  # pin a version for one call
        lease.handle.process(bgr)

GPU Worker (plan §12) hosts this manager in its own process; the API layer never
loads models itself.
"""
from __future__ import annotations

import threading

from fiximg.config import settings
from fiximg.domain.enums import ModelStatus
from fiximg.infrastructure.observability.logging import get_logger, log_event
from fiximg.infrastructure.observability.metrics import MetricName, metrics
from fiximg.infrastructure.observability.tracing import tracer
from fiximg.inference.versions import VersionedModelRegistry

logger = get_logger("fiximg.models")

# Hybrid strategy: ddcolor is loaded lazily on first use (low frequency),
# the subprocess-backed Global/Face chain has nothing to preload in-process.
_LAZY_MODELS = {"ddcolor"}


class ModelManager:
    """Load, cache, version and reuse model pipelines in one place."""

    def __init__(self, settings_obj=None, registry: VersionedModelRegistry | None = None) -> None:
        self.settings = settings_obj or settings
        # §3.15: two versions stay resident during a hot swap. `keep_previous`
        # additionally keeps the replaced version as the rollback target.
        self.registry = registry or VersionedModelRegistry(
            keep_previous=bool(getattr(self.settings, "model_keep_previous", True))
        )
        self._lock = threading.Lock()
        # §30: last load duration per model (ms) for the stats endpoint.
        self.load_times_ms: dict = {}
        self.registry.register_builder("ddcolor", self._build_ddcolor)

    # ------------------------------------------------------------------ warmup
    #: `lazy` loads on first use with no dummy call, `first-use` adds the dummy
    #: call at that first use, `startup` also preloads at process boot.
    WARMUP_STRATEGIES = ("lazy", "first-use", "startup")

    def warmup_strategy(self, name: str) -> str:
        """When *this* model is warmed: its manifest key, else the global setting.

        `models/manifest.yaml` documents `warmup` as "overrides the global
        FIXIMG_WARMUP strategy per model" and sets it on every model, but nothing
        read the key — so the file declared a per-model policy that the runtime
        could not honour, and both sides could drift unnoticed. An unrecognised
        value falls back to the global strategy rather than to a guess.
        """
        declared = None
        try:
            from fiximg.inference.manifest import get_manifest

            entry = get_manifest().get(name)
            declared = (getattr(entry, "metadata", None) or {}).get("warmup")
        except Exception:  # noqa: BLE001 — an unreadable manifest is not fatal here
            declared = None
        value = str(declared or "").strip().lower()
        if value in self.WARMUP_STRATEGIES:
            return value
        return str(getattr(self.settings, "warmup_strategy", "lazy") or "lazy")

    def should_warm_on_load(self, name: str) -> bool:
        """Whether activating this model should run the dummy inference.

        `lazy` is the only strategy that says no: the point of that word is that
        nothing extra happens before a request needs the model.
        """
        return self.warmup_strategy(name) != "lazy"

    def startup_models(self) -> list[str]:
        """Backend names whose effective strategy says preload at boot."""
        from fiximg.inference.backends.registry import backend_registry

        return [name for name in backend_registry.names()
                if self.warmup_strategy(name) == "startup"]

    def warm_at_process_start(self) -> list[str]:
        """Preload and warm the models configured for `startup`.

        Called by whoever executes inference — `PipelineWorker.start()` — rather
        than only by the API process. A pure API node has no business holding GPU
        weights; a worker that never warmed paid the first-request cost anyway.
        Returns the names it actually warmed.
        """
        from fiximg.infrastructure.observability.logging import get_logger, log_event

        logger = get_logger("fiximg.models")
        warmed: list[str] = []
        names = self.startup_models()
        if not names:
            return warmed
        log_event(logger, "INFO", "startup warmup", models=names)
        from fiximg.inference.backends.registry import backend_registry

        for name in names:
            try:
                backend = backend_registry.get(name)
                backend.load(self.settings.device)
                backend.warmup()
                warmed.append(name)
            except Exception as exc:  # noqa: BLE001 — warmup must never block startup
                log_event(logger, "WARNING", "startup warmup failed",
                          model=name, error=str(exc))
        return warmed

    # ------------------------------------------------------------------ build
    def _build_ddcolor(self, version: str):
        """Build the DDColor inference pipeline for ``version``.

        The version is informational for DDColor today (a single published
        checkpoint); the manifest's ``weight`` path decides what is loaded, so a
        future versioned layout only has to change the manifest.
        """
        import os
        import sys

        weights = self.settings.ddcolor_weights
        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"DDColor weight file not found: {weights}\n"
                "Download pytorch_model.pt of damo/cv_ddcolor_image-colorization and "
                "put it into weights/ddcolor/."
            )
        base_dir = self.settings.base_dir
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)
        from ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model

        # The vendored builder defaults to "cuda if torch sees one", which is device 0
        # whatever this process was pinned to. Ask for the configured device instead, so
        # a worker with `FIXIMG_WORKER_GPU=1` does not load its colour model on card 0.
        device = torch_device_for(self.settings.device)
        model = build_ddcolor_model(
            DDColor,
            model_path=weights,
            input_size=self.settings.ddcolor_input_size,
            model_size=self.settings.ddcolor_model_size,
            device=device,
        )
        return ColorizationPipeline(model, input_size=self.settings.ddcolor_input_size,
                                    device=device)

    # ------------------------------------------------------------------- load
    def declared_version(self, name: str) -> str:
        """Version declared in ``models/manifest.yaml`` (default ``1.0.0``)."""
        try:
            from fiximg.inference.manifest import get_manifest

            declared = get_manifest().get(name)
            if declared is not None:
                return declared.version
        except Exception:  # noqa: BLE001 — a missing manifest is not fatal
            pass
        return "1.0.0"

    def load(self, name: str, version: str | None = None):
        """Load a model by name (idempotent; thread-safe)."""
        if name not in self.registry.known_models():
            raise KeyError(f"Unknown model: {name}")
        handle = self.registry.current_handle(name)
        if handle is not None:
            return handle

        target = version or self.declared_version(name)
        weight_uri = self._declared_weight(name)
        with tracer.span("model.load", model_name=name, model_version=target,
                         weight_uri=weight_uri):
            report = self.registry.activate(
                name, target, weight_uri=weight_uri,
                validator=self._validate,
                # `lazy` means "do nothing extra before a request needs me", and the
                # manifest says that per model; honouring it here keeps the declared
                # strategy and the executed one the same thing.
                warmer=self._warm if self.should_warm_on_load(name) else None,
                health=self._probe,
            )
        if not report.get("active_version"):
            raise RuntimeError(f"Failed to load model {name}: {report.get('reason')}")
        for entry in report.get("residents", []):
            if entry["version"] == report["active_version"]:
                self.load_times_ms[name] = entry["load_ms"]
                # §2.10: the same number the admin views read, exposed as a metric
                # series so a weight-load regression is graphable rather than only
                # visible to whoever opens the page.
                metrics.observe(
                    MetricName.MODEL_LOAD_SECONDS, entry["load_ms"] / 1000.0,
                    model=name, version=entry["version"],
                )
        return self.registry.current_handle(name)

    # ---------------------------------------------------------------- access
    def get(self, name: str):
        """Fetch a model, loading it on first use (hybrid strategy)."""
        handle = self.registry.current_handle(name)
        if handle is not None:
            return handle
        return self.load(name)

    def acquire(self, name: str):
        """Pin the active version for one inference (plan §3.15 drain semantics)."""
        if self.registry.current_handle(name) is None:
            self.load(name)
        return self.registry.acquire(name)

    def unload(self, name: str) -> None:
        """Drop cached handles for a model (frees GPU memory on next GC)."""
        self.registry.unload(name)

    def loaded_models(self) -> list:
        return [name for name in self.registry.known_models()
                if self.registry.current_handle(name) is not None]

    def versions(self, name: str) -> dict:
        return {
            "name": name,
            "active_version": self.registry.current_version(name),
            "previous_version": self.registry.previous_version(name),
            "residents": self.registry.residents(name),
        }

    # ---------------------------------------------------------------- release
    def activate_version(
        self,
        name: str,
        version: str,
        *,
        validate: bool = True,
        warmup: bool = True,
    ) -> dict:
        """Hot-swap a model to ``version`` (plan §3.15 release flow).

        The currently active version keeps serving until the new one passes
        validation, warmup and the health probe; a failure leaves it untouched.
        """
        if name not in self.registry.known_models():
            from fiximg.domain.errors import ModelUnavailableError

            raise ModelUnavailableError(
                f"Unknown model: {name}", details={"model": name}
            )
        return self.registry.activate(
            name,
            version,
            weight_uri=self._declared_weight(name),
            validator=self._validate if validate else None,
            warmer=self._warm if warmup else None,
            health=self._probe,
        )

    def rollback(self, name: str) -> dict:
        """Switch back to the previously active version (plan §3.15 failure path)."""
        return self.registry.rollback(name)

    # ------------------------------------------------------- release pipeline
    def _declared_weight(self, name: str) -> str | None:
        try:
            from fiximg.inference.manifest import get_manifest

            return get_manifest().resolve_weight_path(name)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _validate(resident) -> None:
        """Verify the target weights before the version is allowed to serve.

        Whatever the deployment declares is checked: an explicit ``sha256`` in
        ``models/manifest.yaml`` for the named file, plus every file the
        integrity baseline covers under the resolved weight path. The second half
        is what makes a checkpoint *directory* (several networks per stage) a
        verified unit rather than a pass-by-default (plan §3.5.2).
        """
        try:
            from fiximg.inference.manifest import get_manifest

            manifest = get_manifest()
        except Exception:  # noqa: BLE001 — no manifest means nothing to check
            return

        declared = manifest.get(resident.name)
        override = declared.sha256 if declared is not None else None
        expected = manifest.expected_for_weight(resident.weight_uri, override)
        if not expected:
            return

        problems = manifest.verify_files(expected)
        if problems:
            from fiximg.domain.errors import ModelUnavailableError

            raise ModelUnavailableError(
                f"Checksum mismatch for {resident.name} {resident.version}",
                details={
                    "checked": len(expected),
                    "mismatches": problems[:5],
                },
            )

    @staticmethod
    def _warm(resident) -> None:
        """Run one tiny dummy inference so the first real request is not slow."""
        handle = resident.handle
        process = getattr(handle, "process", None)
        if not callable(process):
            return
        try:
            import numpy as np

            process(np.zeros((64, 64, 3), dtype=np.uint8))
        except Exception as exc:  # noqa: BLE001 — warmup must never be fatal
            log_event(logger, "WARNING", "model warmup skipped",
                      model=resident.name, error=str(exc))

    @staticmethod
    def _probe(resident) -> bool:
        """Health probe: a handle that exists and is callable is healthy."""
        handle = resident.handle
        if handle is None:
            return False
        process = getattr(handle, "process", None)
        return callable(process) if process is not None else True

    # ---------------------------------------------------------------- health
    def health(self) -> dict:
        """Report loaded models and CUDA availability for /health/ready."""
        try:
            import torch

            cuda = torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            cuda = False
        return {
            "loaded_models": self.loaded_models(),
            "lazy_models": sorted(_LAZY_MODELS),
            "cuda_available": cuda,
            "versions": self.registry.snapshot(),
        }

    def gpu_stats(self) -> dict:
        """GPU memory peak + per-model load times (plan §30 instrumentation).

        The peak comes from :func:`current_device_peak_mb`, not from
        ``torch.cuda.max_memory_allocated()``: each stage's sampler resets that
        counter so its own figure is attributable, which means a reader asking the
        live counter gets "peak since the last stage started" — a number that was
        routinely *smaller* than a stage peak the same process had just published.
        """
        stats: dict[str, object] = {"load_times_ms": dict(self.load_times_ms)}
        try:
            from fiximg.inference.gpu_memory import current_device_peak_mb

            peak = current_device_peak_mb()
            if peak is not None:
                stats["gpu_memory_peak_mb"] = peak
        except Exception:  # noqa: BLE001
            pass
        return stats


# Process-wide singleton shared by stages via StageContext.model_manager.
model_manager = ModelManager()


def torch_device_for(value) -> str:
    """Any accepted device descriptor (`auto`/`cuda`/`1`/`-1`/`cpu`) as a torch string.

    One translation for the whole inference layer: the runtime schedules an *int*
    (`context.gpu`, -1 for CPU), the manifest and settings carry a *string*, and the
    models want a `torch.device`. Converting in three places is how a stage ends up
    running on a card nobody routed it to.
    """
    gpu = resolve_gpu(value)
    return "cpu" if gpu < 0 else f"cuda:{gpu}"


def resolve_gpu(gpu_arg="auto") -> int:
    """Resolve the device setting to a GPU id: 0..N, or -1 for CPU.

    Accepts the V1 values: "auto"/"cuda"/"gpu" pick the first CUDA device
    when available (else CPU); numeric strings pass through; "cpu" is -1.
    """
    value = str(gpu_arg).strip().lower()
    if value in ("cpu", "-1"):
        return -1
    if value in ("", "auto", "cuda", "gpu", "cuda:0", "gpu:0"):
        try:
            import torch

            return 0 if torch.cuda.is_available() else -1
        except Exception:  # noqa: BLE001
            return -1
    try:
        gpu = int(value.split(":")[-1])
        return gpu
    except ValueError:
        # Unknown descriptor: fall back to auto behaviour instead of crashing.
        try:
            import torch

            return 0 if torch.cuda.is_available() else -1
        except Exception:  # noqa: BLE001
            return -1


__all__ = ["ModelManager", "ModelStatus", "model_manager", "resolve_gpu"]
