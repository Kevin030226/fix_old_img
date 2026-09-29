"""ModelBackend — the uniform interface every model implementation satisfies.

Plan §2.3 / §3.5.1: the V2 code invoked Global/Face models by spawning
subprocesses, which makes GPU reuse, warmup, batching and hot-swapping
impossible. V3 introduces one narrow contract instead:

    load(device) → warmup() → infer(request) → unload()

``LegacyCliBackend`` implements it on top of the existing folder-based CLI
pipeline, so the migration is incremental: a backend can start life as a thin
subprocess adapter and later become a native in-process implementation without
touching a single stage or the runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from PIL import Image

from fiximg.domain.models import ModelHealth
from fiximg.infrastructure.observability.metrics import MetricName, metrics


@dataclass(slots=True)
class ModelRequest:
    """Framework-independent inference input."""

    image: Image.Image
    #: free-form switches forwarded to the backend (``hr``, ``with_scratch``...)
    options: dict = field(default_factory=dict)
    #: scratch directory the backend may use for intermediates
    work_dir: str | None = None
    #: device hint (``auto`` / ``cuda`` / ``cpu`` / ``0``)
    device: str = "auto"


@dataclass(slots=True)
class ModelResult:
    """Framework-independent inference output."""

    image: Image.Image | None
    metadata: dict = field(default_factory=dict)
    #: extra files the backend produced, keyed by artifact role
    artifacts: dict = field(default_factory=dict)
    message: str | None = None


@runtime_checkable
class ModelBackend(Protocol):
    """Minimal lifecycle + inference contract for a model implementation."""

    name: str
    version: str
    #: capabilities this backend provides (plan §3.14.2)
    capabilities: frozenset[str]

    def load(self, device: str = "auto") -> None:
        """Prepare weights / pipelines. Idempotent."""
        ...

    def warmup(self) -> None:
        """Run a tiny dummy inference to remove first-call latency (plan §3.5.3)."""
        ...

    def infer(self, request: ModelRequest) -> ModelResult:
        """Run one inference call."""
        ...

    def unload(self) -> None:
        """Release weights and device memory. Idempotent."""
        ...

    def health(self) -> ModelHealth:
        """Report readiness for ``/api/v1/health/ready`` and the models API."""
        ...


@runtime_checkable
class FolderBridgeBackend(Protocol):
    """The contract the split legacy chain actually runs on (plan §3.6/§3.7).

    ``ModelBackend`` describes a single-image call, but the restoration stages are
    one half of a pipeline that used to be four subprocesses trading folders: each
    stage hands its backend an input *directory*, then reads a known sub-directory
    of the shared root back from disk. That is a second contract, and until now it
    existed only as duck typing — a backend registered for a folder-driven stage
    but missing `run_folder` would fail inside the stage, at request time, with an
    AttributeError.

    Declared separately rather than merged into `ModelBackend` because it is
    genuinely optional: DDColor has no folder bridge, and a protocol every backend
    must satisfy but two thirds of them do not is not a contract, it is a lie.
    """

    def run_folder(
        self,
        input_dir: str,
        output_dir: str,
        *,
        gpu: int = -1,
        hr: bool = False,
        on_progress=None,
    ) -> str:
        """Run this backend's stage over ``input_dir``; return ``output_dir``."""
        ...

    def produced_dir(self, output_dir: str) -> str:
        """Where this backend wrote the result the next stage must read."""
        ...


class BaseModelBackend:
    """Convenience base implementing the bookkeeping every backend repeats.

    Subclasses override :meth:`_do_load`, :meth:`_do_warmup` and :meth:`_do_infer`.
    """

    name: str = "base"
    version: str = "1.0"
    capabilities: frozenset[str] = frozenset()
    #: How this backend really executes (plan §3.5.4): ``native`` keeps the weights
    #: in this process, ``legacy-cli`` spawns the vendored pipeline per call. Stage
    #: metadata records it, so falling back is visible rather than a silent success.
    implementation: str = "unknown"

    def __init__(self, config: dict | None = None) -> None:
        self.config: dict = dict(config or {})
        self.device: str = "cpu"
        self._loaded = False
        #: Per declared optimisation (`channels_last` / `compile`), whether it
        #: actually landed on this process's model. Empty when nothing was declared.
        self.optimisations_applied: dict[str, bool] = {}

    # ------------------------------------------------------------- lifecycle
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def apply_policy(self, model):
        """Optimise a freshly loaded model the way the manifest declares.

        Every native backend has to route its model through here, because a
        declaration that only one backend reads is a declaration that lies — see
        :func:`fiximg.inference.precision.apply_declared_policy`. A backend with no
        ``policy`` (subprocess adapters, dlib) has nothing to apply and says so by
        leaving the record empty.
        """
        policy = getattr(self, "policy", None)
        if policy is None or model is None:
            return model
        from fiximg.inference.precision import apply_declared_policy

        model, applied = apply_declared_policy(model, policy, self.device)
        # One backend can host several networks (the scratch chain is a detector
        # plus the mapping net), so a declared optimisation counts as honoured only
        # when every network the backend loaded accepted it. Reporting True because
        # the bigger net accepted would hide the one that did not.
        for key, landed in applied.items():
            existing = self.optimisations_applied.get(key)
            self.optimisations_applied[key] = landed if existing is None else existing and landed
        return model

    def declared_policy(self) -> dict:
        """The policy the serving implementation runs under, for the models API."""
        policy = getattr(self, "policy", None)
        if policy is None:
            return {}
        return {**policy.describe(), "applied": dict(self.optimisations_applied)}

    def load(self, device: str = "auto") -> None:
        if self._loaded:
            return
        self.device = device
        self._do_load(device)
        self._loaded = True

    def warmup(self) -> None:
        if not self._loaded:
            self.load(self.device)
        self._do_warmup()

    def unload(self) -> None:
        if not self._loaded:
            return
        self._do_unload()
        self._loaded = False
        # A reload may land on a different device, where the same declaration can
        # be declined - so the record must describe the model that is loaded now.
        self.optimisations_applied = {}

    def infer(self, request: ModelRequest) -> ModelResult:
        """Time one forward pass (§2.10's `model_inference_seconds`).

        Deliberately narrower than the orchestrator's `stage_duration_seconds`:
        that one covers device selection, a possible OOM retry and the stage's own
        file bridging, so the two series answer different questions and neither is
        a copy of the other. Load time is *not* measured here — the model manager
        owns that series, and writing the same name from two places with two
        definitions is how a graph starts lying. Subprocess-backed stages go
        through `run_folder` and are covered by the stage series instead.
        """
        if not self._loaded:
            self.load(request.device)
        with metrics.timer(MetricName.MODEL_INFERENCE_SECONDS,
                           model=self.name, implementation=self.implementation):
            return self._do_infer(request)

    def health(self) -> ModelHealth:
        return ModelHealth(
            name=self.name,
            healthy=self._loaded,
            loaded=self._loaded,
            device=self.device,
            detail="ready" if self._loaded else "not loaded",
        )

    # ----------------------------------------------------------- overridable
    def _do_load(self, device: str) -> None:
        """Acquire weights. Default: nothing to do (stateless backends)."""

    def _do_warmup(self) -> None:
        """Optional warmup; default is a no-op."""

    def _do_unload(self) -> None:
        """Release weights; default is a no-op."""

    def _do_infer(self, request: ModelRequest) -> ModelResult:
        raise NotImplementedError

    def describe(self) -> dict:
        """Metadata surfaced by ``GET /api/v1/models``."""
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": sorted(self.capabilities),
            "loaded": self._loaded,
            "device": self.device,
            "implementation": self.implementation,
            "optimisations": self.declared_policy(),
            "config": dict(self.config),
        }


__all__ = [
    "BaseModelBackend",
    "FolderBridgeBackend",
    "ModelBackend",
    "ModelRequest",
    "ModelResult",
]
