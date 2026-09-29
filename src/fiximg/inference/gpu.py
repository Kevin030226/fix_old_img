"""GPU topology — capability → device routing (plan §4.3 Step 4, P2).

A single-process, single-GPU deployment resolves one device and runs everything
on it. The plan's platform stage wants the opposite:

    GPU0 → restore worker
    GPU1 → face/color worker
    GPU2 → high-res worker

Two mechanisms are needed, and this module provides both:

* **in-process routing** — ``FIXIMG_GPU_ROUTING`` assigns *capabilities* to
  devices, so one worker process can run restoration on GPU0 and colorization on
  GPU1 without loading every model on every card.
* **worker pinning** — ``FIXIMG_WORKER_GPU`` restricts a worker process to one
  device; combined with ``FIXIMG_WORKER_CAPABILITIES`` it only claims tasks it
  can actually serve.

Routing is by capability, never by model name (plan §3.14.2), so the table
survives a model swap.

Configuration::

    FIXIMG_GPU_ROUTING="0:restore,scratch_repair;1:colorize,face_restore"
    FIXIMG_WORKER_GPU=1
    FIXIMG_WORKER_CAPABILITIES=colorize,face_restore

Anything not listed falls back to the default device (``FIXIMG_DEVICE``), which
keeps an unconfigured deployment behaving exactly as before.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

from fiximg.infrastructure.observability.logging import get_logger, log_event

logger = get_logger("fiximg.gpu")

#: Separators of the ``FIXIMG_GPU_ROUTING`` grammar.
_DEVICE_SEP = ";"
_CAPABILITY_SEP = ","


@dataclass(frozen=True, slots=True)
class DeviceAssignment:
    """One device and the capabilities it is allowed to serve."""

    device: int
    capabilities: frozenset[str]

    def to_dict(self) -> dict:
        return {"device": self.device, "capabilities": sorted(self.capabilities)}


def parse_routing(raw: str) -> list[DeviceAssignment]:
    """Parse ``"0:restore,denoise;1:colorize"`` into assignments.

    Malformed entries are skipped rather than raising: a typo in a deployment
    variable must not stop the service, and the default device still covers the
    capability (see :meth:`GpuTopology.device_for`).
    """
    assignments: list[DeviceAssignment] = []
    for chunk in (raw or "").split(_DEVICE_SEP):
        chunk = chunk.strip()
        if not chunk:
            continue
        device_part, _, capability_part = chunk.partition(":")
        try:
            device = int(device_part.strip())
        except ValueError:
            log_event(logger, "WARNING", "ignoring malformed GPU routing entry", entry=chunk)
            continue
        capabilities = frozenset(
            c.strip() for c in capability_part.split(_CAPABILITY_SEP) if c.strip()
        )
        assignments.append(DeviceAssignment(device=device, capabilities=capabilities))
    return assignments


class GpuTopology:
    """Resolves which device should execute a set of capabilities."""

    def __init__(
        self,
        assignments: Iterable[DeviceAssignment] = (),
        default_device: int = -1,
        pinned_device: int | None = None,
    ) -> None:
        self.assignments: tuple[DeviceAssignment, ...] = tuple(assignments)
        self.default_device = int(default_device)
        #: When set, every capability resolves to this device (worker pinning).
        self.pinned_device = pinned_device

    # -------------------------------------------------------------- factories
    @classmethod
    def from_settings(cls, settings_obj=None) -> GpuTopology:
        """Build the topology from the ``FIXIMG_GPU_*`` settings."""
        from fiximg.config import settings as default_settings
        from fiximg.inference.model_manager import resolve_gpu

        settings_obj = settings_obj or default_settings
        default_device = resolve_gpu(getattr(settings_obj, "device", "auto"))

        pinned_raw = getattr(settings_obj, "worker_gpu", None)
        pinned = int(pinned_raw) if pinned_raw is not None and str(pinned_raw) != "" else None
        if pinned is not None and pinned < 0:
            pinned = None

        return cls(
            assignments=parse_routing(getattr(settings_obj, "gpu_routing", "") or ""),
            default_device=default_device,
            pinned_device=pinned,
        )

    @classmethod
    def single_device(cls, device: int = -1) -> GpuTopology:
        """Everything on one device — the V2 behaviour."""
        return cls(assignments=(), default_device=device)

    # ------------------------------------------------------------------ query
    @property
    def is_routed(self) -> bool:
        """True when at least one capability is pinned to a specific device."""
        return bool(self.assignments) or self.pinned_device is not None

    def devices(self) -> list[int]:
        """Every device this topology can route to (sorted, de-duplicated)."""
        found = {self.default_device}
        found.update(a.device for a in self.assignments)
        if self.pinned_device is not None:
            found.add(self.pinned_device)
        return sorted(found)

    def capabilities_of(self, device: int) -> frozenset[str]:
        """Capabilities explicitly assigned to ``device``."""
        caps: set[str] = set()
        for assignment in self.assignments:
            if assignment.device == device:
                caps |= assignment.capabilities
        return frozenset(caps)

    def candidate_devices(self, capabilities: Iterable[str] = ()) -> list[int]:
        """Every device that could serve ``capabilities``, in preference order.

        A pinned worker has exactly one candidate. Otherwise the devices whose
        declared capabilities intersect the request come first (declaration
        order), and the default device is the fallback. This is the input to
        memory-aware selection and to OOM retries.
        """
        if self.pinned_device is not None:
            return [self.pinned_device]
        wanted = set(capabilities)
        matched: list[int] = []
        for assignment in self.assignments:
            if assignment.capabilities & wanted and assignment.device not in matched:
                matched.append(assignment.device)
        if matched:
            return matched
        return [self.default_device]

    def device_for(
        self,
        capabilities: Iterable[str] = (),
        *,
        memory_aware: bool = False,
        required_mb: float = 0.0,
        probe=None,
    ) -> int:
        """Pick the device that should execute ``capabilities``.

        Rules, in order:
          1. a pinned worker device wins outright;
          2. the first assignment whose capability set intersects the request;
          3. the default device.

        With ``memory_aware`` and more than one candidate, the roomiest device
        wins instead of the first match — the plan's "no memory-aware scheduling"
        gap (§7.1 / §4.3 Step 4). Without probe data the first candidate is used,
        so a CPU-only deployment is unaffected.
        """
        candidates = self.candidate_devices(capabilities)
        if not candidates:
            return self.default_device
        if memory_aware and len(candidates) > 1:
            from fiximg.inference.gpu_memory import select_device

            return select_device(candidates, required_mb=required_mb, probe=probe)
        return candidates[0]

    def served_capabilities(self, capabilities: Iterable[str]) -> bool:
        """True when this topology can serve every requested capability.

        Only meaningful for a pinned worker: it answers "should this worker claim
        a task that needs these capabilities?".
        """
        if self.pinned_device is None:
            return True
        wanted = set(capabilities)
        if not wanted:
            return True
        pinned_caps = self.capabilities_of(self.pinned_device)
        if not pinned_caps:
            return True  # pinned device with no declared specialisation
        return wanted <= pinned_caps

    def describe(self) -> dict:
        return {
            "default_device": self.default_device,
            "pinned_device": self.pinned_device,
            "routed": self.is_routed,
            "devices": self.devices(),
            "assignments": [a.to_dict() for a in self.assignments],
        }


__all__ = ["DeviceAssignment", "GpuTopology", "parse_routing"]
