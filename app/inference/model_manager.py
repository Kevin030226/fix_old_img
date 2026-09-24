"""ModelManager — unified model lifecycle (plan section 6).

In-process registry that owns model handles, supports hybrid loading
(eager/lazy per model), reuse, unload, and a health() probe for /health/ready.

GPU Worker (plan section 12/Phase 4) will host this manager in its own process;
the API layer never loads models itself.
"""
import threading

from app.core.config import settings
from app.core.logging import get_logger, log_event

logger = get_logger("fiximg.models")

# Hybrid strategy: ddcolor is loaded lazily on first use (low frequency),
# the subprocess-backed Global/Face chain has nothing to preload in-process.
_LAZY_MODELS = {"ddcolor"}


class ModelManager:
    """Load, cache and reuse model pipelines in one place."""

    def __init__(self, settings_obj=None) -> None:
        self.settings = settings_obj or settings
        self._models = {}
        self._lock = threading.Lock()
        # §30: last load duration per model (ms) for the stats endpoint.
        self.load_times_ms: dict = {}

    # ------------------------------------------------------------------ load
    def _build_ddcolor(self):
        """Build the DDColor inference pipeline (mirrors V1 app.colorizer logic)."""
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

        model = build_ddcolor_model(
            DDColor,
            model_path=weights,
            input_size=self.settings.ddcolor_input_size,
            model_size=self.settings.ddcolor_model_size,
        )
        return ColorizationPipeline(model, input_size=self.settings.ddcolor_input_size)

    _BUILDERS = {"ddcolor": _build_ddcolor}

    def load(self, name: str):
        """Load a model by name (idempotent; thread-safe)."""
        with self._lock:
            if name in self._models:
                return self._models[name]
            builder = self._BUILDERS.get(name)
            if builder is None:
                raise KeyError(f"Unknown model: {name}")
            log_event(logger, "INFO", "loading model", model=name)
            import time as _time

            t0 = _time.perf_counter()
            model = builder(self)
            self.load_times_ms[name] = int((_time.perf_counter() - t0) * 1000)
            self._models[name] = model
            log_event(
                logger, "INFO", "model loaded", model=name,
                load_ms=self.load_times_ms[name],
            )
            return model

    def load_all(self) -> None:
        """Eagerly load every registered model (eager strategy / worker warmup)."""
        for name in self._BUILDERS:
            self.load(name)

    # ---------------------------------------------------------------- access
    def get(self, name: str):
        """Fetch a model, loading it on first use (hybrid strategy)."""
        if name in self._models:
            return self._models[name]
        return self.load(name)

    def unload(self, name: str) -> None:
        """Drop a cached model handle (frees GPU memory on next GC)."""
        with self._lock:
            self._models.pop(name, None)

    def loaded_models(self) -> list:
        return sorted(self._models.keys())

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
        }

    def gpu_stats(self) -> dict:
        """GPU memory peak + per-model load times (plan §30 instrumentation)."""
        stats = {"load_times_ms": dict(self.load_times_ms)}
        try:
            import torch

            if torch.cuda.is_available():
                stats["gpu_memory_peak_mb"] = round(
                    torch.cuda.max_memory_allocated() / (1 << 20), 1
                )
        except Exception:  # noqa: BLE001
            pass
        return stats


# Process-wide singleton shared by stages via StageContext.model_manager.
model_manager = ModelManager()


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
