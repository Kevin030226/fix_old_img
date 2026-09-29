"""Model service — the models API's application layer (plan §3.8, §3.15).

Joins the three model views into one consistent answer:

* the **manifest** says what the deployment declares (version, weight, hash),
* the **backend registry** says what can actually run (loaded? healthy?),
* the **version registry** says what is resident and which version is active.

Callers (``GET /api/v1/models``, ``POST /api/v1/models/{name}/reload``,
``POST /api/v1/models/{name}/rollback``) never touch the registries directly.
"""
from __future__ import annotations

from fiximg.domain.errors import ModelUnavailableError
from fiximg.infrastructure.observability.logging import get_logger, log_event
from fiximg.inference.backends.registry import backend_registry
from fiximg.inference.manifest import get_manifest
from fiximg.inference.model_manager import model_manager

logger = get_logger("fiximg.models")


class ModelService:
    """Read + control the set of models this deployment can run."""

    def __init__(self, registry=None, manager=None, manifest_provider=None) -> None:
        self._registry = registry or backend_registry
        self._manager = manager or model_manager
        self._manifest = manifest_provider or get_manifest

    # ------------------------------------------------------------------ read
    def list_models(self) -> list[dict]:
        """One entry per model, merging manifest + backend + version state."""
        manifest = self._manifest()
        described = {entry["name"]: entry for entry in self._registry.describe_all()}
        names = sorted(set(described) | set(manifest.names()))
        loaded = set(self._manager.loaded_models())

        out: list[dict] = []
        for name in names:
            backend = described.get(name, {})
            declared = manifest.get(name)
            entry = {
                "name": name,
                "version": self._manager.registry.current_version(name)
                or ((declared.version if declared else None) or backend.get("version", "1.0")),
                "declared_version": declared.version if declared else None,
                "previous_version": self._manager.registry.previous_version(name),
                "framework": (declared.framework if declared else None) or backend.get("framework"),
                # The manifest declares "legacy-cli" for this chain; when the native
                # backend serves it, that field alone would misreport where the
                # weights live. Implementation comes from the backend itself.
                "implementation": backend.get("implementation"),
                # §3.5.4: the policy in force plus what this process actually
                # applied. A native backend always reports its policy (so a declared
                # key that no backend reads would show up as applied=false, not as
                # silence); null means the serving implementation has no torch
                # policy to apply - a subprocess adapter, or dlib.
                "optimisations": backend.get("optimisations") or None,
                "device": (declared.device if declared else None) or backend.get("device", "auto"),
                "capabilities": sorted(
                    set(backend.get("capabilities") or [])
                    | set(declared.capabilities if declared else [])
                ),
                "status": "ready" if name in loaded else "registered",
                "loaded": name in loaded,
                "declared": declared is not None,
                "weight_present": manifest.weight_present(name) if declared else None,
                "description": (declared.metadata.get("description") if declared else None),
                # §3.15: which versions are resident (usually one, two mid-swap).
                "resident_versions": [
                    r["version"] for r in self._manager.registry.residents(name)
                ],
            }
            out.append(entry)
        return out

    def get_model(self, name: str) -> dict:
        for entry in self.list_models():
            if entry["name"] == name:
                return entry
        raise ModelUnavailableError(f"Unknown model: {name}", details={"model": name})

    def health(self) -> dict:
        """Per-model health plus the global model-manager view (plan §3.8)."""
        manifest = self._manifest()
        return {
            "backends": self._registry.health_all(),
            "manager": self._manager.health(),
            "manifest": {
                "source": manifest.source,
                "models": manifest.names(),
                "hashes_ok": manifest.verify_hashes(),
                # A model with no baseline entry reports hashes_ok=True ("nothing
                # says it was tampered with"), which is not the same as verified —
                # so the gap is listed rather than hidden behind a passing bool.
                "unverified": manifest.unverified_models(),
                "weights_present": {n: manifest.weight_present(n) for n in manifest.names()},
            },
            "gpu": self._manager.gpu_stats(),
        }

    def versions(self, name: str) -> dict:
        """Resident/active/previous versions of one model (plan §3.15).

        Plus `catalog`: the versions recorded by **any** process (plan §6's
        ModelVersion table). Without it a read made against the API node described
        only the API node — which for a split deployment is the one process that
        loads nothing, so the endpoint answered "no versions" about models its
        workers had been serving for hours.
        """
        self._require_known(name)
        from fiximg.infrastructure.db.repositories import model_repository

        report = dict(self._manager.versions(name))
        report["catalog"] = model_repository.list_versions(name)
        return report

    # ----------------------------------------------------------------- write
    def reload(self, name: str, version: str | None = None) -> dict:
        """Hot-swap a model to ``version`` (plan §3.15 zero-downtime release).

        The new version is built, validated, warmed and health-checked while the
        current one keeps serving; routing then switches atomically and the old
        version is drained once its in-flight calls finish. On failure the
        running version is untouched — see :meth:`rollback` to go back
        explicitly.

        ``version`` defaults to the version declared in ``models/manifest.yaml``.
        Reloading the *same* version reports ``switched: false``: weights must
        never be overwritten in place while the service runs (plan §3.15), so
        picking up new weights means bumping the declared version.
        """
        self._require_known(name)
        declared = self._manifest().get(name)
        target_version = version or (declared.version if declared else None) or "1.0.0"

        log_event(logger, "INFO", "reloading model", model=name, version=target_version)
        report = self._manager.activate_version(name, target_version)
        if report.get("switched"):
            log_event(logger, "INFO", "model reloaded", model=name,
                      version=report["active_version"], switch_ms=report.get("switch_ms"))
        return report

    def rollback(self, name: str) -> dict:
        """Return to the version that was active before the last switch."""
        self._require_known(name)
        report = self._manager.rollback(name)
        if not report.get("switched"):
            log_event(logger, "WARNING", "model rollback skipped",
                      model=name, reason=report.get("reason"))
        return report

    def unload(self, name: str) -> dict:
        """Drop a model from memory (frees GPU RAM without restarting)."""
        self._require_known(name)
        self._registry.get(name).unload()
        self._manager.unload(name)
        return self.get_model(name)

    def warmup(self, name: str) -> dict:
        """Explicitly warm a model (plan §3.5.3)."""
        self._require_known(name)
        backend = self._registry.get(name)
        backend.load(self._manager.settings.device)
        backend.warmup()
        return self.get_model(name)

    # ------------------------------------------------------------- internals
    def _require_known(self, name: str) -> None:
        known = set(self._registry.names()) | set(self._manager.registry.known_models())
        if name not in known:
            raise ModelUnavailableError(f"Unknown model: {name}", details={"model": name})


model_service = ModelService()


__all__ = ["ModelService", "model_service"]
