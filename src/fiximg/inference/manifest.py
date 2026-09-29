"""Model manifest loader (plan 鎼?.5.2).

Reads ``models/manifest.yaml`` into :class:`~fiximg.domain.models.ModelVersion`
objects and offers the release flow the plan describes::

    download 閳?sha256 verify 閳?load 閳?warmup 閳?health probe 閳?READY

The manifest is the *declaration*; the backends are the *implementation*. The
loader only reconciles the two: it reports which declared models have their
weights on disk, and verifies hashes when the manifest carries one.

Missing manifest 閳?empty declaration (the built-in backends still work), so a
source checkout without ``models/manifest.yaml`` is not broken.

Versioned weight layout (plan 鎼?, 鎼?.15)
----------------------------------------
The plan's target layout keeps each version of a model in its own directory::

    models/
    閳规壕鏀㈤埞鈧?manifest.yaml
    閳规壕鏀㈤埞鈧?ddcolor/
    閳?  閳规壕鏀㈤埞鈧?v1/pytorch_model.pt
    閳?  閳规柡鏀㈤埞鈧?v2/pytorch_model.pt
    閳规柡鏀㈤埞鈧?global_restore/v1/...

:meth:`ModelManifest.resolve_weight_path` prefers that layout when it exists and
otherwise falls back to the declared ``weight``/``checkpoint`` path, so an
existing deployment keeps working while a new one can adopt versions. This is
what makes "two versions resident" (plan 鎼?.15) practical: a hot swap needs two
real weight paths on disk, not one path that gets overwritten.
"""
from __future__ import annotations

import os
from typing import Any

from fiximg.domain.models import ModelVersion
from fiximg.infrastructure.observability.logging import get_logger, log_event
from fiximg.paths import CONFIG_DIR, PROJECT_ROOT

logger = get_logger("fiximg.models.manifest")


def models_root() -> str:
    """``<root>/models``, derived from the *current* ``PROJECT_ROOT``.

    A function rather than a constant on purpose: the versioned-layout tests
    relocate the tree with ``monkeypatch.setattr(manifest, "PROJECT_ROOT", tmp)``.
    A ``MODELS_DIR`` captured at import time kept pointing at the real checkout,
    so ``resolve_weight_path`` answered with ``weights/ddcolor/...`` while the
    test's ``models/ddcolor/v2`` sat unused 鈥?and both sides of the assertion
    were computed from the same stale root, so the suite stayed green. Reading
    the module attribute per call is what makes the relocation observable.
    """
    return os.path.join(PROJECT_ROOT, "models")


class ModelManifest:
    """In-memory view of the model manifest."""

    #: The committed checksum baseline written by ``fiximg.cli.download_weights
    #: generate`` 鈥?the same file the startup fail-fast check reads, so weights
    #: are verified once and the two systems cannot drift apart.
    INTEGRITY_BASELINE_PATH = os.path.join(CONFIG_DIR, "weights_manifest.json")

    def __init__(self, models: dict[str, ModelVersion] | None = None,
                 source: str | None = None) -> None:
        self.models: dict[str, ModelVersion] = models or {}
        self.source = source
        self._baseline: dict[str, str] | None = None
        #: path -> digest (or None when the file is missing), one hash per process.
        self._hash_cache: dict[str, str | None] = {}

    def __len__(self) -> int:
        return len(self.models)

    def __contains__(self, name: object) -> bool:
        return name in self.models

    def get(self, name: str) -> ModelVersion | None:
        return self.models.get(name)

    def names(self) -> list[str]:
        return sorted(self.models)

    def by_capability(self, capability: str) -> list[str]:
        """Declared models providing ``capability`` (plan 鎼?.14.2)."""
        return sorted(n for n, m in self.models.items() if m.supports(capability))

    #: Weight file names looked for inside a versioned directory, in order.
    _VERSIONED_WEIGHT_NAMES = (
        "pytorch_model.pt", "model.pth", "model.pt", "checkpoint.pth", "weights.pth",
    )

    def versioned_dir(self, name: str, version: str | None = None) -> str:
        """``models/<name>/<version>/`` (plan 鎼? versioned weight layout)."""
        model = self.models.get(name)
        resolved = version or (model.version if model is not None else None) or "v1"
        if not str(resolved).startswith("v") and str(resolved).isdigit():
            resolved = f"v{resolved}"
        return os.path.join(models_root(), name, str(resolved))

    def _version_candidates(self, name: str, version: str | None) -> list[str]:
        """Version directory names to try, most specific first.

        A declared ``version: "1.0.0"`` does not name a directory, so the
        candidates include the ``v<major>`` convention and finally every version
        present on disk (newest last). An explicit ``version`` short-circuits
        this: the caller asked for a specific one.
        """
        root = os.path.join(models_root(), name)
        if not os.path.isdir(root):
            return []
        present = self.available_versions(name)
        if version:
            wanted = str(version)
            if wanted in present:
                return [wanted]
            normalised = wanted if wanted.startswith("v") else f"v{wanted}"
            return [normalised] if normalised in present else []

        model = self.models.get(name)
        declared = str(model.version) if model is not None and model.version else ""
        candidates: list[str] = []
        if declared in present:
            candidates.append(declared)
        major = declared.split(".", 1)[0] if declared else ""
        if major and f"v{major}" in present and f"v{major}" not in candidates:
            candidates.append(f"v{major}")
        # Fall back to whatever is on disk, newest last.
        candidates.extend(v for v in present if v not in candidates)
        return candidates

    def resolve_weight_path(self, name: str, version: str | None = None) -> str | None:
        """Absolute path of the weight for ``name``, preferring the versioned layout.

        Order:
          1. ``models/<name>/<version>/`` when a candidate version directory
             holds a weight file,
          2. the manifest's declared ``weight``/``checkpoint`` path.

        Returns None when the model declares nothing.
        """
        for candidate in self._version_candidates(name, version):
            directory = os.path.join(models_root(), name, candidate)
            for filename in self._VERSIONED_WEIGHT_NAMES:
                path = os.path.join(directory, filename)
                if os.path.isfile(path):
                    return path
            # A versioned directory may hold a checkpoint *directory* instead.
            nested = os.path.join(directory, "checkpoints")
            if os.path.isdir(nested):
                return nested

        model = self.models.get(name)
        if model is None or not model.weight_uri:
            return None
        path = model.weight_uri
        if not os.path.isabs(path):
            path = os.path.join(PROJECT_ROOT, path)
        return path

    def available_versions(self, name: str) -> list[str]:
        """Versions present on disk under ``models/<name>/`` (plan 鎼?.15).

        This is what an operator checks before a hot swap: a switch needs the
        target version's weights to exist as a *separate* directory.
        """
        root = os.path.join(models_root(), name)
        if not os.path.isdir(root):
            return []
        return sorted(
            entry for entry in os.listdir(root)
            if os.path.isdir(os.path.join(root, entry)) and entry.startswith("v")
        )

    def uses_versioned_layout(self, name: str, version: str | None = None) -> bool:
        """True when the weight actually resolved to the versioned layout."""
        declared = self.models.get(name)
        path = self.resolve_weight_path(name, version)
        if path is None:
            return False
        versioned_root = os.path.join(models_root(), name)
        if not path.startswith(versioned_root + os.sep):
            return False
        # Guard against the declared path happening to live under models/<name>/.
        if declared is not None and declared.weight_uri:
            return path != os.path.join(PROJECT_ROOT, declared.weight_uri)
        return True

    def weight_present(self, name: str) -> bool:
        """True when the declared weight exists (directories count as present)."""
        path = self.resolve_weight_path(name)
        if not path:
            return False
        return os.path.exists(path)

    # --------------------------------------------------------------- integrity
    def integrity_baseline(self) -> dict[str, str]:
        """``{repo-relative path: sha256}`` from the committed baseline.

        Empty when the baseline cannot be read. An incomplete checkout must show
        up as "nothing was verified", never as a passing hash check.
        """
        if self._baseline is None:
            baseline: dict[str, str] = {}
            try:
                import json

                with open(self.INTEGRITY_BASELINE_PATH, encoding="utf-8") as handle:
                    entries = (json.load(handle) or {}).get("files") or []
                for entry in entries:
                    path = str(entry.get("path") or "").replace("\\", "/")
                    digest = entry.get("sha256")
                    if path and digest:
                        baseline[path] = str(digest)
            except Exception as exc:  # noqa: BLE001 閳?a missing baseline is not fatal
                log_event(logger, "WARNING", "integrity baseline unavailable",
                          path=self.INTEGRITY_BASELINE_PATH, error=str(exc))
            self._baseline = baseline
        return self._baseline

    def expected_weights(self, name: str) -> dict[str, str]:
        """``{absolute path: expected sha256}`` covering one model's weights."""
        return self.expected_for_path(self.resolve_weight_path(name))

    @staticmethod
    def _norm(path: str) -> str:
        """One canonical spelling per weight path.

        Declared URIs keep forward slashes (``Global/checkpoints/restoration``)
        while ``os.path.join`` on Windows yields backslashes; without this they
        would be two different cache keys and the same 145 MB file would be
        hashed twice 閳?or, worse, a baseline entry would not be recognised as
        the file an explicit ``sha256`` was meant to override.
        """
        return os.path.normpath(path)

    def expected_for_path(self, path: str | None) -> dict[str, str]:
        """Baseline digests covering one weight path (file or checkpoint directory).

        A model whose weight is a *directory* is covered by every baseline file
        beneath it 閳?that is how the legacy chains, which load several networks
        per stage, are verified as one unit.
        """
        baseline = self.integrity_baseline()
        if not baseline or not path:
            return {}
        relative = os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")
        return {
            self._norm(os.path.join(PROJECT_ROOT, *entry.split("/"))): digest
            for entry, digest in baseline.items()
            if entry == relative or entry.startswith(relative + "/")
        }

    def expected_for_weight(self, path: str | None, override: str | None = None) -> dict[str, str]:
        """Baseline digests for one weight path, with an explicit hash winning.

        ``override`` is the ``sha256`` declared in ``models/manifest.yaml`` for
        that exact file: a deployment can pin one model without regenerating the
        whole baseline (plan 鎼?.5.2).
        """
        expected = self.expected_for_path(path)
        if override and path:
            expected[self._norm(path)] = override
        return expected

    def _digest(self, path: str) -> str | None:
        """SHA-256 of one weight file, computed at most once per process.

        A file the boot integrity check already proved is taken as verified
        rather than read again: the legacy chains are about 1.3 GB, and hashing
        that per health poll would hang the request. The trust is explicit 閳?        ``weights_check`` records what it proved in this process.
        """
        path = self._norm(path)
        if path in self._hash_cache:
            return self._hash_cache[path]

        relative = os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")
        try:
            from fiximg.infrastructure.models.weights_check import verified_weights

            proven = verified_weights().get(relative)
        except Exception:  # noqa: BLE001 閳?an unavailable registry means "compute it"
            proven = None
        if proven is not None:
            self._hash_cache[path] = proven
            return proven

        from fiximg.domain.artifacts import sha256_of

        try:
            self._hash_cache[path] = sha256_of(path) if os.path.isfile(path) else None
        except OSError:
            self._hash_cache[path] = None
        return self._hash_cache[path]

    def verify_files(self, expected: dict[str, str]) -> list[dict[str, str]]:
        """Return ``[{path, expected, actual}]`` for every file that does not match.

        An empty list means verified 閳?but only when ``expected`` was non-empty;
        callers must not read "no mismatches" as "covered".
        """
        problems: list[dict[str, str]] = []
        for path, digest in sorted(expected.items()):
            actual = self._digest(path)
            if actual != digest:
                problems.append({
                    "path": os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/"),
                    "expected": digest,
                    "actual": actual or "unreadable",
                })
        return problems

    def verify_hashes(self) -> dict[str, bool]:
        """Verify each model's weights against the baseline (plan 鎼?.5.2).

        An explicit ``sha256`` in ``models/manifest.yaml`` wins over the baseline
        for the file it names, so a deployment can pin one model without
        regenerating the whole baseline.

        A model the baseline does not cover reports ``True`` 閳?"no evidence of
        tampering" 閳?and is listed by :meth:`unverified_models` so the gap is
        visible instead of silently counting as verified.
        """
        report: dict[str, bool] = {}
        for name, model in self.models.items():
            path = self.resolve_weight_path(name)
            expected = self.expected_for_weight(path, model.sha256)
            report[name] = not self.verify_files(expected) if expected else True
        return report

    def unverified_models(self) -> list[str]:
        """Models with no baseline coverage 閳?the honest half of verify_hashes."""
        return sorted(name for name in self.models if not self.expected_weights(name))

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "models": {n: m.to_dict() for n, m in self.models.items()},
        }


def load_manifest(path: str | None = None) -> ModelManifest:
    """Load the manifest from ``path`` (default: ``FIXIMG_MODEL_MANIFEST``)."""
    from fiximg.config import settings

    target = path or settings.model_manifest
    if not target or not os.path.exists(target):
        log_event(logger, "WARNING", "model manifest not found; using backends only",
                  path=target)
        return ModelManifest(source=target)

    try:
        import yaml

        with open(target, encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}
    except Exception as exc:  # noqa: BLE001 閳?a broken manifest must not block startup
        log_event(logger, "ERROR", "model manifest unreadable", path=target, error=str(exc))
        return ModelManifest(source=target)

    models = {
        name: ModelVersion.from_dict(name, spec or {})
        for name, spec in (raw.get("models") or {}).items()
    }
    log_event(logger, "INFO", "model manifest loaded",
              path=target, models=len(models))
    return ModelManifest(models=models, source=target)


_manifest: ModelManifest | None = None


def get_manifest() -> ModelManifest:
    """Process-wide manifest singleton."""
    global _manifest
    if _manifest is None:
        _manifest = load_manifest()
    return _manifest


def reset_manifest() -> None:
    """Drop the cached manifest (tests / config reload)."""
    global _manifest
    _manifest = None


__all__ = ["ModelManifest", "get_manifest", "load_manifest", "reset_manifest"]
