"""Plugin discovery — adding models without editing this repository (§3.14.1).

The plan's plugin registry::

    class ModelPlugin(Protocol):
        id: str
        version: str
        capabilities: set[str]
        def create_backend(self, config): ...

    registry.register(GlobalRestorePlugin())

V3's *internal* extension points already worked (implement a ``ModelBackend``,
register it, declare capabilities — the planner reasons about capabilities, never
model names). What was missing is the **external** path: a third-party package
could not add a model without forking the repository.

This module closes that gap with standard entry points. A plugin package declares::

    [project.entry-points."fiximg.models"]
    my_model = "my_package.plugin:MyModelPlugin"

    [project.entry-points."fiximg.stages"]
    my_stage = "my_package.plugin:MyStagePlugin"

and ``build_default_backend_registry()`` / ``build_default_registry()`` pick it up
at startup. Discovery is best effort: a broken third-party plugin is logged and
skipped, never allowed to stop the service.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from fiximg.infrastructure.observability.logging import get_logger, log_event

logger = get_logger("fiximg.plugins")

#: Entry-point groups scanned at startup.
MODEL_ENTRY_POINT_GROUP = "fiximg.models"
STAGE_ENTRY_POINT_GROUP = "fiximg.stages"

#: Set to "0"/"false" to skip discovery entirely (locked-down deployments).
ENABLE_ENV = "FIXIMG_PLUGINS"


@runtime_checkable
class ModelPlugin(Protocol):
    """Contract a third-party model plugin implements (plan §3.14.1)."""

    #: unique backend name, as used by the planner/registry
    id: str
    #: plugin semantic version
    version: str
    #: capabilities this model provides — the planner matches on these
    capabilities: set[str]

    def create_backend(self, config: dict | None = None) -> Any:
        """Build the :class:`~fiximg.inference.backends.base.ModelBackend`."""
        ...


@runtime_checkable
class StagePlugin(Protocol):
    """Contract a third-party *stage* plugin implements."""

    name: str
    version: str
    capabilities: set[str]

    def create_stage(self, **kwargs) -> Any:
        """Build the :class:`~fiximg.inference.stages.base.BaseStage`."""
        ...


@dataclass(frozen=True, slots=True)
class PluginRecord:
    """A discovered entry point, whether or not it loaded successfully."""

    group: str
    entry_point: str
    loaded: bool
    plugin_id: str | None = None
    capabilities: tuple[str, ...] = ()
    version: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "entry_point": self.entry_point,
            "loaded": self.loaded,
            "id": self.plugin_id,
            "version": self.version,
            "capabilities": list(self.capabilities),
            "error": self.error,
        }


@dataclass
class PluginDiscovery:
    """Result of one discovery pass."""

    models: list[Any] = field(default_factory=list)
    stages: list[Any] = field(default_factory=list)
    records: list[PluginRecord] = field(default_factory=list)

    @property
    def loaded(self) -> int:
        return len(self.models) + len(self.stages)

    def to_dict(self) -> dict:
        return {
            "loaded": self.loaded,
            "models": len(self.models),
            "stages": len(self.stages),
            "entry_points": [r.to_dict() for r in self.records],
        }


def plugins_enabled() -> bool:
    """False when ``FIXIMG_PLUGINS`` disables discovery."""
    import os

    raw = (os.environ.get(ENABLE_ENV) or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _entry_points(group: str) -> Iterable:
    """Entry points in ``group``; empty when the API or the group is unavailable."""
    try:
        from importlib.metadata import entry_points

        # The keyword form is the only one the project supports: `requires-python`
        # is >=3.11, so the pre-3.10 dict-returning form is not a case to handle
        # here — it was dead code wearing a `pragma: no cover`.
        return entry_points(group=group)
    except Exception as exc:  # noqa: BLE001 — discovery must never raise
        log_event(logger, "WARNING", "entry point lookup failed",
                  group=group, error=str(exc))
        return []


def _load(entry_point) -> tuple[Any | None, str | None]:
    """Instantiate the plugin behind an entry point; (plugin, error)."""
    try:
        target = entry_point.load()
        # The entry point may point at the class or at an already-built instance.
        plugin = target() if isinstance(target, type) else target
        return plugin, None
    except Exception as exc:  # noqa: BLE001 — one bad plugin must not stop startup
        return None, f"{type(exc).__name__}: {exc}"


def _validate(plugin, *, kind: str) -> tuple[str | None, tuple[str, ...], str | None, str | None]:
    """Check the protocol shape; returns (id, capabilities, version, error)."""
    identifier = getattr(plugin, "id" if kind == "model" else "name", None)
    if not identifier or not isinstance(identifier, str):
        return None, (), None, f"{kind} plugin has no string {'id' if kind == 'model' else 'name'}"
    capabilities = getattr(plugin, "capabilities", None)
    if capabilities is None:
        return identifier, (), getattr(plugin, "version", None), "plugin declares no capabilities"
    if not isinstance(capabilities, (set, frozenset, list, tuple)):
        return identifier, (), getattr(plugin, "version", None), "capabilities must be a collection"
    creator = getattr(plugin, "create_backend" if kind == "model" else "create_stage", None)
    if not callable(creator):
        method = "create_backend" if kind == "model" else "create_stage"
        return identifier, tuple(capabilities), getattr(plugin, "version", None), f"missing {method}()"
    return identifier, tuple(sorted(str(c) for c in capabilities)), \
        getattr(plugin, "version", None), None


def discover_plugins() -> PluginDiscovery:
    """Scan the entry-point groups for model and stage plugins (never raises)."""
    discovery = PluginDiscovery()
    if not plugins_enabled():
        log_event(logger, "INFO", "plugin discovery disabled", env=ENABLE_ENV)
        return discovery

    for group, kind, bucket in (
        (MODEL_ENTRY_POINT_GROUP, "model", discovery.models),
        (STAGE_ENTRY_POINT_GROUP, "stage", discovery.stages),
    ):
        for entry_point in _entry_points(group):
            name = getattr(entry_point, "name", str(entry_point))
            plugin, error = _load(entry_point)
            if plugin is None:
                discovery.records.append(
                    PluginRecord(group=group, entry_point=name, loaded=False, error=error)
                )
                log_event(logger, "ERROR", "plugin failed to load",
                          group=group, entry_point=name, error=error)
                continue

            identifier, capabilities, version, problem = _validate(plugin, kind=kind)
            if problem:
                discovery.records.append(
                    PluginRecord(group=group, entry_point=name, loaded=False,
                                 plugin_id=identifier, capabilities=capabilities,
                                 version=version, error=problem)
                )
                log_event(logger, "ERROR", "plugin does not satisfy the contract",
                          group=group, entry_point=name, plugin_id=identifier, error=problem)
                continue

            bucket.append(plugin)
            discovery.records.append(
                PluginRecord(group=group, entry_point=name, loaded=True,
                             plugin_id=identifier, capabilities=capabilities, version=version)
            )
            log_event(logger, "INFO", "plugin discovered",
                      group=group, entry_point=name, plugin_id=identifier,
                      capabilities=list(capabilities), version=version)

    return discovery


def register_model_plugins(registry, plugins: Iterable[Any]) -> list[str]:
    """Register discovered model plugins into a backend registry.

    A plugin may deliberately override a built-in backend (that is how an
    operator swaps an implementation), so ``replace=True``. Returns the names
    that were registered.
    """
    registered: list[str] = []
    for plugin in plugins:
        identifier = getattr(plugin, "id", None)
        if not identifier:
            continue
        try:
            registry.register(identifier, lambda p=plugin: p.create_backend(), replace=True)
            registered.append(identifier)
        except Exception as exc:  # noqa: BLE001 — a plugin must not break startup
            log_event(logger, "ERROR", "model plugin registration failed",
                      plugin_id=identifier, error=str(exc))
    return registered


def register_stage_plugins(registry, plugins: Iterable[Any]) -> list[str]:
    """Register discovered stage plugins into a stage registry."""
    registered: list[str] = []
    for plugin in plugins:
        name = getattr(plugin, "name", None)
        if not name:
            continue
        try:
            registry.register(
                name,
                lambda p=plugin, **kwargs: p.create_stage(**kwargs),
                capabilities=set(getattr(plugin, "capabilities", ()) or ()),
                version=str(getattr(plugin, "version", "1.0")),
                description=str(getattr(plugin, "description", "") or "third-party plugin"),
                replace=True,
            )
            registered.append(name)
        except Exception as exc:  # noqa: BLE001
            log_event(logger, "ERROR", "stage plugin registration failed",
                      stage=name, error=str(exc))
    return registered


def load_and_register(model_registry=None, stage_registry=None) -> PluginDiscovery:
    """Discover and register everything available; returns the discovery report."""
    discovery = discover_plugins()
    if model_registry is not None:
        register_model_plugins(model_registry, discovery.models)
    if stage_registry is not None:
        register_stage_plugins(stage_registry, discovery.stages)
    return discovery


#: Cache so the entry-point scan happens once per process.
_last_discovery: PluginDiscovery | None = None


def last_discovery() -> PluginDiscovery | None:
    """The most recent discovery result, for ``/health/ready`` reporting."""
    return _last_discovery


def remember(discovery: PluginDiscovery) -> PluginDiscovery:
    global _last_discovery
    _last_discovery = discovery
    return discovery


def reset_plugin_cache() -> None:
    """Forget the cached discovery (tests)."""
    global _last_discovery
    _last_discovery = None


__all__ = [
    "ENABLE_ENV",
    "MODEL_ENTRY_POINT_GROUP",
    "STAGE_ENTRY_POINT_GROUP",
    "ModelPlugin",
    "PluginDiscovery",
    "PluginRecord",
    "StagePlugin",
    "discover_plugins",
    "last_discovery",
    "load_and_register",
    "plugins_enabled",
    "register_model_plugins",
    "register_stage_plugins",
    "remember",
    "reset_plugin_cache",
]
