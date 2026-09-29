"""Stage registry — capability-driven stage lookup (plan §2.2 / §3.14).

V2 hard-coded the stage table inside :meth:`PipelinePlanner.build_stage`, so
adding the 7th or 8th model meant editing the planner. V3 turns it into an
explicit registry:

    registry.register(
        name="super_resolution",
        builder=RealESRGANStage,
        capabilities={"upscale", "restore"},
    )

The planner then reasons about *capabilities* (``"face_restore" in caps``)
instead of model names, which is what makes swapping a model a data change
rather than a code change.

Registration is deliberately explicit: no import-time magic, no entry-point
scanning — the default registry is built in one readable place below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable

from fiximg.domain.errors import StageNotAvailableError
from fiximg.inference.stages.base import BaseStage


@dataclass(frozen=True)
class StageSpec:
    """Everything the runtime needs to know about a registered stage."""

    name: str
    builder: Callable[..., BaseStage]
    capabilities: frozenset[str] = field(default_factory=frozenset)
    version: str = "1.0"
    description: str = ""

    def create(self, kwargs: dict | None = None) -> BaseStage:
        return self.builder(**(kwargs or {}))


class StageRegistry:
    """Name → :class:`StageSpec` map with capability queries."""

    def __init__(self) -> None:
        self._specs: dict[str, StageSpec] = {}

    # ------------------------------------------------------------ mutation
    def register(
        self,
        name: str,
        builder: Callable[..., BaseStage],
        *,
        capabilities: set[str] | frozenset[str] | tuple[str, ...] = (),
        version: str = "1.0",
        description: str = "",
        replace: bool = False,
    ) -> StageSpec:
        """Register a stage builder. Raises on duplicate names unless replacing."""
        if name in self._specs and not replace:
            raise ValueError(f"Stage already registered: {name}")
        spec = StageSpec(
            name=name,
            builder=builder,
            capabilities=frozenset(capabilities),
            version=version,
            description=description,
        )
        self._specs[name] = spec
        return spec

    def unregister(self, name: str) -> None:
        self._specs.pop(name, None)

    # --------------------------------------------------------------- query
    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def names(self) -> list[str]:
        return sorted(self._specs)

    def specs(self) -> list[StageSpec]:
        return [self._specs[n] for n in self.names()]

    def get(self, name: str) -> StageSpec:
        spec = self._specs.get(name)
        if spec is None:
            raise StageNotAvailableError(f"Unknown stage: {name}")
        return spec

    def create(self, name: str, kwargs: dict | None = None) -> BaseStage:
        """Instantiate a registered stage by name."""
        return self.get(name).create(kwargs)

    def capabilities_of(self, name: str) -> frozenset[str]:
        return self.get(name).capabilities

    def find_by_capability(self, capability: str) -> list[str]:
        """Names of every stage advertising ``capability`` (plan §3.14.2)."""
        return sorted(
            name for name, spec in self._specs.items() if capability in spec.capabilities
        )

    def all_capabilities(self) -> list[str]:
        caps: set[str] = set()
        for spec in self._specs.values():
            caps |= spec.capabilities
        return sorted(caps)


def build_default_registry() -> StageRegistry:
    """The registry the application uses: the built-in stages (plan §2.2).

    The legacy restoration chain is four separate stages, each running exactly
    one legacy path, so the face chain is never executed twice (§3.6/§3.7):

        global_restore → face_detection → face_enhancement → warp_back

    Imported lazily inside the function so the registry module stays free of
    heavy imports at package import time.
    """
    from fiximg.inference.stages.colorization import ColorizationStage
    from fiximg.inference.stages.face_enhancement import (
        FaceDetectionStage,
        FaceEnhancementStage,
    )
    from fiximg.inference.stages.global_restore import GlobalRestoreStage
    from fiximg.inference.stages.scratch import ScratchDetectionStage
    from fiximg.inference.stages.warp_back import WarpBackStage

    registry = StageRegistry()
    registry.register(
        "global_restore",
        GlobalRestoreStage,
        capabilities={"restore", "deblur", "denoise"},
        version="2.0",
        description="Overall quality restoration only (legacy path 1)",
    )
    registry.register(
        "scratch_repair",
        lambda: GlobalRestoreStage(with_scratch=True),
        capabilities={"restore", "scratch_repair"},
        version="2.0",
        description="Scratch + quality restoration (legacy path 1 with masks)",
    )
    registry.register(
        "scratch_detection",
        ScratchDetectionStage,
        capabilities={"scratch_detection"},
        description="Scratch mask detection only",
    )
    registry.register(
        "face_detection",
        FaceDetectionStage,
        capabilities={"face_detection"},
        version="2.0",
        description="Face boxes + landmarks (legacy path 2)",
    )
    registry.register(
        "face_enhancement",
        FaceEnhancementStage,
        capabilities={"face_restore", "face_enhance"},
        version="2.0",
        description="Face crop enhancement (legacy path 3)",
    )
    registry.register(
        "warp_back",
        WarpBackStage,
        capabilities={"warp_back", "face_composite"},
        version="2.0",
        description="Composite enhanced faces back onto the restored image (legacy path 4)",
    )
    registry.register(
        "colorization",
        ColorizationStage,
        capabilities={"colorize"},
        description="DDColor black-and-white colorization",
    )

    # §3.14.1: third-party stages arrive through the `fiximg.stages` entry-point
    # group. The discovery result is cached, so the scan happens once per process.
    from fiximg.inference import plugins as plugins_module

    discovery = plugins_module.last_discovery() or plugins_module.remember(
        plugins_module.discover_plugins()
    )
    if discovery.stages:
        plugins_module.register_stage_plugins(registry, discovery.stages)
    return registry


#: Process-wide default registry (stages are stateless builders).
stage_registry = build_default_registry()


__all__ = ["StageRegistry", "StageSpec", "build_default_registry", "stage_registry"]
