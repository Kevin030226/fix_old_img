"""Model domain model (plan §3.5.1 / §3.5.2 / §6).

Describes a model *version* independently of the framework that runs it, so the
runtime can list, health-check and (later) atomically switch versions without
knowing whether the backend is PyTorch, ONNX or a legacy CLI subprocess.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fiximg.domain.enums import ModelStatus


@dataclass(slots=True)
class ModelVersion:
    """One concrete version of one model, as declared in ``models/manifest.yaml``."""

    name: str
    version: str = "1.0.0"
    framework: str = "pytorch"
    device: str = "auto"
    weight_uri: str | None = None
    sha256: str | None = None
    input_size: int | None = None
    status: str = ModelStatus.REGISTERED
    #: capabilities drive planner decisions — never match on model names (plan §3.14.2)
    capabilities: set[str] = field(default_factory=set)
    metadata: dict = field(default_factory=dict)
    loaded_at: str | None = None

    @property
    def status_enum(self) -> ModelStatus:
        return ModelStatus(self.status)

    @property
    def is_ready(self) -> bool:
        return self.status == ModelStatus.READY

    def supports(self, capability: str) -> bool:
        """Capability query used by the planner instead of name matching."""
        return capability in self.capabilities

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> ModelVersion:
        known = {
            "version",
            "framework",
            "device",
            "weight_uri",
            "weight",
            "checkpoint",
            "sha256",
            "input_size",
            "status",
            "capabilities",
            "loaded_at",
        }
        weight_uri = raw.get("weight_uri") or raw.get("weight") or raw.get("checkpoint")
        return cls(
            name=name,
            version=str(raw.get("version") or "1.0.0"),
            framework=raw.get("framework") or "pytorch",
            device=raw.get("device") or "auto",
            weight_uri=weight_uri,
            sha256=raw.get("sha256"),
            input_size=raw.get("input_size"),
            status=raw.get("status") or ModelStatus.REGISTERED,
            capabilities=set(raw.get("capabilities") or []),
            metadata={k: v for k, v in raw.items() if k not in known},
            loaded_at=raw.get("loaded_at"),
        )

    def to_dict(self) -> dict:
        """Serialise for the ``GET /api/v1/models`` response."""
        return {
            "name": self.name,
            "version": self.version,
            "framework": self.framework,
            "device": self.device,
            "weight_uri": self.weight_uri,
            "sha256": self.sha256,
            "input_size": self.input_size,
            "status": self.status,
            "capabilities": sorted(self.capabilities),
            "loaded_at": self.loaded_at,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ModelHealth:
    """Result of a backend health probe (plan §3.5.1)."""

    name: str
    healthy: bool
    detail: str = ""
    loaded: bool = False
    device: str = "cpu"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "healthy": self.healthy,
            "loaded": self.loaded,
            "device": self.device,
            "detail": self.detail,
            **self.extra,
        }


__all__ = ["ModelHealth", "ModelVersion"]
