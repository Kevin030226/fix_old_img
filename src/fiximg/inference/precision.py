"""Precision policy — capability-driven inference optimisation (plan §3.5.4).

The plan asks for the obvious PyTorch wins::

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(input)

and then lists FP16/BF16, ``torch.compile``, channels-last, pinned memory, CUDA
streams, batch/tiled inference.

The plan is equally explicit about *how*: these must be **capability-gated**, not
switched on globally. Legacy checkpoints are frequently incompatible with
``torch.compile`` or half precision, so the decision belongs to the model's
declaration, not to a global flag::

    Model Capability Registry
           ↓
    does it support compile?
           ↓
    YES → enable       NO → fall back

This module turns a model's declaration (``models/manifest.yaml``) into a
:class:`PrecisionPolicy` and provides the runtime pieces to apply it. Everything
degrades to "plain fp32 inference_mode" when nothing is declared, so an
unconfigured deployment behaves exactly as before.

Manifest keys (all optional)::

    precision: fp32 | fp16 | bf16      # autocast dtype on CUDA
    channels_last: true                # memory-format optimisation
    compile: true                      # torch.compile
    compile_mode: default|reduce-overhead|max-autotune
"""
from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from typing import Any

from fiximg.infrastructure.observability.logging import get_logger, log_event

logger = get_logger("fiximg.precision")

#: Autocast dtypes this module understands, mapped to torch attribute names.
_SUPPORTED_PRECISIONS = {
    "fp32": None,
    "float32": None,
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
}


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    """How a specific model should be executed."""

    #: Always on: inference_mode is a pure win with no compatibility risk.
    inference_mode: bool = True
    #: ``"float16"`` / ``"bfloat16"`` / ``None`` for fp32.
    autocast_dtype: str | None = None
    channels_last: bool = False
    compile: bool = False
    compile_mode: str = "default"

    @property
    def is_accelerated(self) -> bool:
        return bool(self.autocast_dtype or self.compile or self.channels_last)

    @classmethod
    def from_declared(cls, declared: dict | None) -> PrecisionPolicy:
        """Build the policy from a model's manifest entry (never raises)."""
        raw = declared or {}
        precision = str(raw.get("precision") or "fp32").strip().lower()
        if precision not in _SUPPORTED_PRECISIONS:
            log_event(logger, "WARNING", "unknown precision; using fp32", precision=precision)
            precision = "fp32"

        mode = str(raw.get("compile_mode") or "default").strip()
        if mode not in ("default", "reduce-overhead", "max-autotune"):
            mode = "default"

        return cls(
            autocast_dtype=_SUPPORTED_PRECISIONS[precision],
            channels_last=bool(raw.get("channels_last")),
            compile=bool(raw.get("compile")),
            compile_mode=mode,
        )

    def describe(self) -> dict:
        return {
            "inference_mode": self.inference_mode,
            "precision": self.autocast_dtype or "fp32",
            "channels_last": self.channels_last,
            "compile": self.compile,
            "compile_mode": self.compile_mode if self.compile else None,
            "accelerated": self.is_accelerated,
        }


def resolve_policy(model_name: str, manifest=None) -> PrecisionPolicy:
    """Policy for a model, read from the manifest (falls back to plain fp32).

    ``FIXIMG_PRECISION`` overrides the declared dtype for every model, which is
    the safety valve an operator needs when a checkpoint turns out not to
    tolerate half precision in production::

        FIXIMG_PRECISION=fp32 fiximg-api
    """
    override = _env_override()
    if override is not None:
        return PrecisionPolicy.from_declared({"precision": override})

    if manifest is None:
        try:
            from fiximg.inference.manifest import get_manifest

            manifest = get_manifest()
        except Exception:  # noqa: BLE001 — no manifest means no optimisation
            return PrecisionPolicy()

    declared = manifest.get(model_name) if manifest is not None else None
    if declared is None:
        return PrecisionPolicy()
    return PrecisionPolicy.from_declared(getattr(declared, "metadata", {}) or {})


def _env_override() -> str | None:
    """``FIXIMG_PRECISION`` when set to a recognised value, else None."""
    try:
        from fiximg.config import settings

        raw = (getattr(settings, "precision_override", "") or "").strip().lower()
    except Exception:  # noqa: BLE001 — settings are optional here
        return None
    if not raw or raw == "auto":
        return None
    return raw if raw in _SUPPORTED_PRECISIONS else None


@contextlib.contextmanager
def inference_context(policy: PrecisionPolicy, device: str | int | None = None):
    """Apply ``torch.inference_mode`` and (when declared) autocast.

    Autocast is only entered for CUDA devices: on CPU it is a no-op at best and
    a slowdown at worst, and half precision on CPU is not supported for most of
    the operators these models use.
    """
    torch = _torch()
    if torch is None:
        yield
        return

    mode = torch.inference_mode() if policy.inference_mode else contextlib.nullcontext()
    with mode:
        if policy.autocast_dtype and _is_cuda(device):
            dtype = getattr(torch, policy.autocast_dtype)
            with torch.autocast(device_type="cuda", dtype=dtype):
                yield
        else:
            yield


def apply_declared_policy(model: Any, policy: PrecisionPolicy,
                          device: str | int | None = None) -> tuple[Any, dict[str, bool]]:
    """Apply what the declaration asks and report, per key, whether it landed.

    The report is the point. A manifest key a backend never reads is worse than a
    missing key: ``channels_last``/``compile`` used to be honoured by exactly one
    backend, so declaring ``channels_last: true`` for another model changed
    nothing, logged nothing, and still read as configured in
    ``GET /api/v1/models``. A caller that cannot honour a declaration has to say so
    here instead of leaving it implied.
    """
    if model is None:
        return model, {}

    torch = _torch()
    applied: dict[str, bool] = {}

    if policy.channels_last:
        try:
            model = model.to(memory_format=_torch_channels_last())
            applied["channels_last"] = True
            log_event(logger, "INFO", "model converted to channels_last")
        except Exception as exc:  # noqa: BLE001 — unsupported layouts must not offline a model
            log_event(logger, "WARNING", "channels_last not applied", error=str(exc))
            applied["channels_last"] = False

    if policy.compile:
        if torch is not None and _is_cuda(device):
            try:
                model = torch.compile(model, mode=policy.compile_mode)
                applied["compile"] = True
                log_event(logger, "INFO", "model compiled", mode=policy.compile_mode)
            except Exception as exc:  # noqa: BLE001 — legacy checkpoints often fail
                log_event(logger, "WARNING", "torch.compile not applied; using eager",
                          mode=policy.compile_mode, error=str(exc))
                applied["compile"] = False
        else:
            # torch.compile on CPU for these chains is a pessimisation and the
            # plan gates it per capability, so "declared but not this device".
            applied["compile"] = False
    return model, applied


def apply_model_optimisations(model: Any, policy: PrecisionPolicy, device: str | int | None = None):
    """Convert / compile a loaded model according to the policy (best effort).

    Returns the model to use — the original when an optimisation is unsupported,
    because a failed optimisation must never take the model offline. Use
    :func:`apply_declared_policy` when the caller also needs to report what landed.
    """
    if model is None or not policy.is_accelerated:
        return model
    return apply_declared_policy(model, policy, device)[0]


def _torch():
    try:
        import torch  # noqa: PLC0415 — optional at import time

        return torch
    except Exception:  # noqa: BLE001 — CPU-only deployments may lack torch
        return None


def _torch_channels_last():
    torch = _torch()
    return torch.channels_last if torch is not None else None


def _is_cuda(device: str | int | None) -> bool:
    """True when ``device`` refers to a CUDA device (or 'auto' on a CUDA host)."""
    torch = _torch()
    if torch is None:
        return False
    if device is None:
        return bool(torch.cuda.is_available())
    if isinstance(device, int):
        return device >= 0 and bool(torch.cuda.is_available())
    text = str(device).strip().lower()
    if text in ("cpu", "-1"):
        return False
    # Every remaining form names a CUDA device, and a named device that does not
    # exist on this host is not CUDA: entering autocast for it makes torch print
    # "CUDA is not available, disabling autocast" on every call and quietly run the
    # fp32 path anyway, so the declaration and the execution disagreed while looking
    # like they agreed.
    return bool(torch.cuda.is_available())


#: Cache so a manifest read does not happen on every inference call.
_POLICY_CACHE: dict[str, PrecisionPolicy] = {}
_CACHE_LOCK = threading.Lock()


def cached_policy(model_name: str, manifest=None) -> PrecisionPolicy:
    """Memoised :func:`resolve_policy` (the manifest is static at runtime)."""
    with _CACHE_LOCK:
        cached = _POLICY_CACHE.get(model_name)
    if cached is not None:
        return cached
    policy = resolve_policy(model_name, manifest)
    with _CACHE_LOCK:
        _POLICY_CACHE[model_name] = policy
    return policy


def reset_policy_cache() -> None:
    """Drop the memoised policies (tests / manifest reload)."""
    with _CACHE_LOCK:
        _POLICY_CACHE.clear()


__all__ = [
    "PrecisionPolicy",
    "apply_declared_policy",
    "apply_model_optimisations",
    "cached_policy",
    "inference_context",
    "reset_policy_cache",
    "resolve_policy",
]
