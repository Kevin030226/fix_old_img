"""Model backend registry (plan §3.5.1 / §3.14.1).

The models API and the runtime resolve backends through here, so adding a model
is: write the backend, register it, declare capabilities. No business logic
elsewhere has to change.
"""
from __future__ import annotations

from collections.abc import Callable

from fiximg.domain.errors import ModelUnavailableError
from fiximg.inference.backends.base import ModelBackend

#: The resident native Global backend, created once per process — see
#: :func:`native_global_backend`.
_native_global = None
#: Its class, imported lazily (torch-free at import time) and patchable in tests.
_GLOBAL_BACKEND_CLS = None
#: Same pair for the native face-detection backend.
_native_face = None
_FACE_BACKEND_CLS = None
#: And for the native scratch-repair backend (same tree as the quality path).
_native_scratch = None
_SCRATCH_BACKEND_CLS = None
#: And for the native face-enhancement backend (Face_Enhancement tree).
_native_face_enhance = None
_FACE_ENHANCE_BACKEND_CLS = None


class ModelBackendRegistry:
    """Name → backend factory map with lazy instantiation."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], ModelBackend]] = {}
        self._instances: dict[str, ModelBackend] = {}

    def register(
        self,
        name: str,
        factory: Callable[[], ModelBackend],
        *,
        replace: bool = False,
    ) -> None:
        if name in self._factories and not replace:
            raise ValueError(f"Backend already registered: {name}")
        self._factories[name] = factory

    def names(self) -> list[str]:
        return sorted(self._factories)

    def get(self, name: str) -> ModelBackend:
        """Return the singleton backend instance, creating it on first use."""
        if name in self._instances:
            return self._instances[name]
        factory = self._factories.get(name)
        if factory is None:
            raise ModelUnavailableError(f"Unknown model backend: {name}")
        instance = factory()
        self._instances[name] = instance
        return instance

    def describe_all(self) -> list[dict]:
        """Metadata for ``GET /api/v1/models`` (does not force-load weights)."""
        out: list[dict] = []
        for name in self.names():
            backend = self.get(name)
            describe = getattr(backend, "describe", None)
            entry = describe() if callable(describe) else {"loaded": False}
            # The registry key wins over whatever the instance calls itself: one
            # adapter class serves several models (`LegacyCliBackend` wraps the
            # detection, enhancement and restoration stages alike and carries
            # `name = "global_restore"` from the first of them), and an instance
            # that renames itself silently drops that model out of the API — the
            # caller keyed the list by the reported name and saw the face stages
            # report no implementation at all.
            entry["name"] = name
            out.append(entry)
        return out

    def health_all(self) -> dict[str, dict]:
        """Health probe for every registered backend (never raises)."""
        report: dict[str, dict] = {}
        for name in self.names():
            try:
                report[name] = self.get(name).health().to_dict()
            except Exception as exc:  # noqa: BLE001 — health must never raise
                report[name] = {"name": name, "healthy": False, "detail": str(exc)}
        return report

    def unload_all(self) -> None:
        for instance in self._instances.values():
            try:
                instance.unload()
            except Exception:  # noqa: BLE001 — best-effort release
                pass


def native_global_backend(config: dict | None = None):
    """The one resident Global instance for this process (plan §3.5.1).

    Deliberately a singleton: keeping the three quality networks alive between
    requests is the point of the native backend, so the per-request ``stem`` the
    CLI adapter needs is irrelevant here (the folder bridge names outputs after
    the input files). The class is resolved through a module global so tests can
    substitute a stand-in without importing torch.
    """
    global _native_global, _GLOBAL_BACKEND_CLS

    if _native_global is None:
        if _GLOBAL_BACKEND_CLS is None:
            from fiximg.inference.backends.global_restore import GlobalRestoreBackend

            _GLOBAL_BACKEND_CLS = GlobalRestoreBackend
        _native_global = _GLOBAL_BACKEND_CLS(config)
    return _native_global


def native_scratch_backend(config: dict | None = None):
    """The one resident scratch-repair instance for this process.

    Same tree as the quality backend, different branch — so a Global-owning worker
    can host both, and the detector plus four networks are loaded once each.
    """
    global _native_scratch, _SCRATCH_BACKEND_CLS

    if _native_scratch is None:
        if _SCRATCH_BACKEND_CLS is None:
            from fiximg.inference.backends.global_restore import NativeScratchRepairBackend

            _SCRATCH_BACKEND_CLS = NativeScratchRepairBackend
        _native_scratch = _SCRATCH_BACKEND_CLS(config)
    return _native_scratch


def select_restore_backend(*, with_scratch: bool = False, config: dict | None = None,
                           stages=None):
    """Choose the implementation for the restoration stage (plan §3.5.4).

    Capability-gated rather than name-gated: the native in-process backend is used
    only when this process owns the Global tree *and* torch and the quality weights
    are present; otherwise the subprocess adapter takes over. The fallback is
    visible, not silent — each backend reports its own ``implementation`` through
    ``health()``/``describe()``, and the stage records ``backend`` in its metadata.

    Both branches of the vendored stage are reachable natively — quality and
    scratch — since they share the Global tree; each needs its own weights, so a
    partial install only gets what it can prove.
    """
    from fiximg.inference.backends.legacy_cli import STAGE_RESTORE, LegacyCliBackend

    if with_scratch:
        from fiximg.inference.backends.global_restore import scratch_native_available

        available, _reason = scratch_native_available()
        if available:
            return native_scratch_backend(config)
    else:
        from fiximg.inference.backends.global_restore import native_available

        available, _reason = native_available()
        if available:
            return native_global_backend(config)

    return LegacyCliBackend(
        dict(config or {}),
        with_scratch=with_scratch,
        stages=tuple(stages or (STAGE_RESTORE,)),
    )


def native_face_backend(config: dict | None = None):
    """The one resident dlib instance for this process (see the Global twin above)."""
    global _native_face, _FACE_BACKEND_CLS

    if _native_face is None:
        if _FACE_BACKEND_CLS is None:
            from fiximg.inference.backends.face_detect_native import NativeFaceDetectionBackend

            _FACE_BACKEND_CLS = NativeFaceDetectionBackend
        _native_face = _FACE_BACKEND_CLS(config)
    return _native_face


def native_face_enhancement_backend(config: dict | None = None):
    """The one resident face-enhancement model for this process.

    Only reachable when the process owns the ``Face_Enhancement`` tree, so the
    singleton cannot end up half-loaded behind a Global-owning worker.
    """
    global _native_face_enhance, _FACE_ENHANCE_BACKEND_CLS

    if _native_face_enhance is None:
        if _FACE_ENHANCE_BACKEND_CLS is None:
            from fiximg.inference.backends.face_enhance_native import (
                NativeFaceEnhancementBackend,
            )

            _FACE_ENHANCE_BACKEND_CLS = NativeFaceEnhancementBackend
        _native_face_enhance = _FACE_ENHANCE_BACKEND_CLS(config)
    return _native_face_enhance


def select_face_enhancement_backend(config: dict | None = None, *, hr: bool = False):
    """Enhancement implementation for stage 3 (plan §3.5.4, §4.2 Step 2).

    Needs the ``face`` tree, so on the default ``FIXIMG_NATIVE_TREE=global``
    worker it stays on the adapter; a face worker flips the setting and gets the
    resident model. HR selects the FaceSR_512 checkpoint, which this install does
    not ship, so ``hr=True`` always returns the adapter.
    """
    from fiximg.inference.backends.legacy_cli import LegacyCliBackend, STAGE_FACE_ENHANCE

    if not hr:
        from fiximg.inference.backends.face_enhance_native import native_available

        available, _reason = native_available()
        if available:
            return native_face_enhancement_backend(config)

    return LegacyCliBackend(dict(config or {}), stages=(STAGE_FACE_ENHANCE,))


def select_face_detection_backend(config: dict | None = None, *, hr: bool = False):
    """Detection implementation for the face stage (plan §3.5.4).

    On by default, because the numeric proof behind it exists now:
    ``tests/gpu/test_face_detect_equivalence.py`` compares the in-process crops with
    the ones ``Face_Detection/detect_all_dlib.py`` writes, name by name and byte by
    byte, and it passes on an install with dlib 20.0.1 and the landmark model present.
    ``FIXIMG_FACE_DETECT_NATIVE=0`` returns the subprocess adapter — that is also what
    happens automatically where dlib or the landmark model is missing, since
    ``native_available()`` reports the reason. HR alignment is a different transform,
    so ``hr=True`` always returns the adapter.

    Deliberately *not* gated on ``FIXIMG_NATIVE_TREE``: that setting names which
    legacy tree the process imports, and detection imports none of them, so a
    ``global`` worker and a ``face`` worker can both host it. ``none`` therefore does
    not disable it either — ``FIXIMG_FACE_DETECT_NATIVE`` is the switch for that, and
    two settings meaning "no in-process inference" would be one too many.
    """
    from fiximg.inference.backends.legacy_cli import LegacyCliBackend, STAGE_FACE_DETECT

    if not hr:
        try:
            from fiximg.config import settings

            enabled = bool(getattr(settings, "face_detect_native", False))
        except Exception:  # noqa: BLE001 — an unreadable setting means "stay safe"
            enabled = False
        if enabled:
            from fiximg.inference.backends.face_detect_native import native_available

            available, _reason = native_available()
            if available:
                return native_face_backend(config)

    return LegacyCliBackend(dict(config or {}), stages=(STAGE_FACE_DETECT,))


def build_default_backend_registry() -> ModelBackendRegistry:
    """The application's backends (plan §3.5.1).

    One entry per *stage* of the legacy chain plus DDColor, so the models API
    mirrors what the planner can actually compose (§3.6/§3.7).
    """
    from fiximg.inference.backends.ddcolor import DDColorBackend
    from fiximg.inference.backends.legacy_cli import WarpBackCliBackend

    registry = ModelBackendRegistry()
    registry.register("ddcolor", DDColorBackend)
    registry.register(
        "global_restore",
        lambda: select_restore_backend(with_scratch=False),
    )
    registry.register("scratch_repair", lambda: select_restore_backend(with_scratch=True))
    # Each of the three stages resolves its implementation per call, so a
    # fallback is reported honestly rather than baked in at import time.
    registry.register("face_detection", lambda: select_face_detection_backend())
    registry.register("face_enhancement", lambda: select_face_enhancement_backend())
    registry.register("warp_back", WarpBackCliBackend)

    # §3.14.1: third-party plugins may add or replace backends through the
    # `fiximg.models` entry-point group, so a new model does not require a fork.
    from fiximg.inference import plugins as plugins_module

    discovery = plugins_module.remember(plugins_module.discover_plugins())
    if discovery.models:
        plugins_module.register_model_plugins(registry, discovery.models)
    return registry


#: Process-wide backend registry.
backend_registry = build_default_backend_registry()


__all__ = [
    "ModelBackendRegistry",
    "backend_registry",
    "build_default_backend_registry",
    "native_face_backend",
    "native_scratch_backend",
    "native_face_enhancement_backend",
    "native_global_backend",
    "select_face_detection_backend",
    "select_face_enhancement_backend",
    "select_restore_backend",
]
