"""Model API schemas (plan §3.8 ``/api/v1/models``, §3.15 hot update)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ModelOptimisationsView(BaseModel):
    """What the manifest declared for a model, and what this process applied.

    ``applied`` has one entry per declared key (``channels_last`` / ``compile``)
    and is the difference between a configuration that works and a configuration
    that is only written down: a key no backend reads used to report nothing at
    all (plan §3.5.4).
    """

    precision: str = Field(default="fp32", description="autocast dtype in use")
    channels_last: bool = False
    compile: bool = False
    accelerated: bool = False
    applied: dict[str, bool] = Field(default_factory=dict)


class ModelView(BaseModel):
    """One model as reported by ``GET /api/v1/models``."""

    name: str
    version: str | None = None
    declared_version: str | None = Field(
        default=None, description="Version declared in models/manifest.yaml"
    )
    previous_version: str | None = Field(
        default=None, description="Version kept resident for rollback (plan §3.15)"
    )
    framework: str | None = None
    implementation: str | None = Field(
        default=None,
        description="How it actually executes: 'native' (resident in this process) "
                    "or 'legacy-cli' (vendored pipeline subprocess). Surfaced because "
                    "the declared framework cannot tell the two apart (plan §3.5.4).",
    )
    device: str = "auto"
    capabilities: list[str] = Field(default_factory=list)
    status: str = "registered"
    loaded: bool = False
    declared: bool = False
    weight_present: bool | None = None
    description: str | None = None
    optimisations: ModelOptimisationsView | None = Field(
        default=None,
        description="Declared inference optimisations and whether each one landed in "
                    "this process (plan §3.5.4). None where the serving implementation "
                    "has no precision policy to apply - a subprocess adapter or dlib.",
    )
    resident_versions: list[str] = Field(
        default_factory=list, description="Versions currently in memory"
    )


class ModelListResponse(BaseModel):
    """``GET /api/v1/models`` body."""

    models: list[ModelView] = Field(default_factory=list)


class ResidentVersionView(BaseModel):
    """One resident version of a model."""

    name: str
    version: str
    status: str
    weight_uri: str | None = None
    load_ms: int = 0
    refcount: int = Field(default=0, description="In-flight inferences using it")
    retiring: bool = Field(default=False, description="Draining after a switch")
    failure: str | None = None


class RecordedVersionView(BaseModel):
    """One `model_versions` row: what some process actually loaded (plan §6)."""

    name: str
    version: str
    #: `ready` or `unhealthy` — the last load outcome, not the manifest's claim.
    status: str
    framework: str | None = None
    weight_uri: str | None = None
    sha256: str | None = None
    loaded_at: str | None = None
    load_ms: int | None = None
    #: How many loads this row has seen; one version is one row however often it
    #: reloads, so a count is what distinguishes "loaded once" from "flapped".
    load_count: int = 0
    last_error: str | None = None
    writer: str | None = Field(
        default=None, description="Process that recorded the last observation"
    )
    metadata: dict = Field(default_factory=dict)


class ModelVersionsResponse(BaseModel):
    """``GET /api/v1/models/{name}/versions`` body.

    Two views on purpose. ``residents``/``active_version`` answer for **this
    process** — residency is per-process and cannot be anything else. ``catalog``
    answers for the **deployment**: every version some process has loaded, from the
    database, so an API node that never loads a model stops reporting an empty list
    beside its workers' live traffic (plan §6's ModelVersion row).
    """

    name: str
    active_version: str | None = None
    previous_version: str | None = None
    residents: list[ResidentVersionView] = Field(default_factory=list)
    catalog: list[RecordedVersionView] = Field(default_factory=list)


class ModelSwitchResponse(BaseModel):
    """Body of ``POST /api/v1/models/{name}/reload`` and ``/rollback``.

    ``switched`` is False when the candidate failed validation/warmup/health and
    the previously active version kept serving (plan §3.15 failure path).
    """

    name: str
    version: str | None = None
    switched: bool
    active_version: str | None = None
    previous_version: str | None = None
    reason: str | None = None
    switch_ms: int | None = None
    rolled_back: bool = False
    residents: list[ResidentVersionView] = Field(default_factory=list)


class ModelHealthView(BaseModel):
    """Per-model health detail."""

    name: str
    healthy: bool
    loaded: bool = False
    device: str = "cpu"
    detail: str = ""


class ModelHealthResponse(BaseModel):
    """``GET /api/v1/models/health`` body."""

    backends: dict[str, dict] = Field(default_factory=dict)
    manager: dict = Field(default_factory=dict)
    manifest: dict = Field(default_factory=dict)
    gpu: dict = Field(default_factory=dict)


__all__ = [
    "ModelHealthResponse",
    "ModelHealthView",
    "ModelListResponse",
    "ModelSwitchResponse",
    "ModelVersionsResponse",
    "ModelView",
    "ResidentVersionView",
]
