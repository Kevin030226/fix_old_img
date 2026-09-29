"""ArtifactStore protocol (plan §2.8 / §5.6).

V2 stored artifacts as bare filesystem paths inside the database, which couples
the API process to the worker's disk. V3 introduces a key-based store so the
same code runs against a local directory, MinIO or S3:

    store.put_bytes(data, "tasks/2026/09/<id>/output/final.png")
    ref = store.put_file(local_tmp_path, key)
    store.open_path(ref)        # local fast path (raises for remote stores)
    store.get_bytes(ref)

Call sites never concatenate paths themselves — they ask the store.
"""
from __future__ import annotations

import os
from typing import BinaryIO, Protocol, runtime_checkable

from fiximg.domain.artifacts import ArtifactRef
from fiximg.domain.errors import ArtifactNotFoundError


@runtime_checkable
class ArtifactStore(Protocol):
    """Minimal blob-store contract used by the runtime and the API."""

    #: backend identifier recorded in ``ArtifactRef.backend``
    backend: str

    def put_bytes(self, data: bytes, key: str, mime_type: str | None = None) -> ArtifactRef:
        """Store ``data`` under ``key``; overwrite if present."""
        ...

    def put_file(self, source_path: str, key: str, mime_type: str | None = None) -> ArtifactRef:
        """Store an existing local file under ``key``."""
        ...

    def get_bytes(self, ref: ArtifactRef) -> bytes:
        """Read the whole artifact. Raises :class:`ArtifactNotFoundError`."""
        ...

    def open_stream(self, ref: ArtifactRef) -> BinaryIO:
        """Open a readable binary stream."""
        ...

    def open_path(self, ref: ArtifactRef) -> str:
        """Return a local filesystem path (local backend only).

        Remote backends must raise ``NotImplementedError``: callers that need a
        path have to download first, which keeps the local coupling explicit.
        """
        ...

    def delete(self, ref: ArtifactRef) -> None:
        """Remove the artifact; missing keys are not an error."""
        ...

    def exists(self, ref: ArtifactRef) -> bool:
        """True when the artifact is present."""
        ...


class BaseArtifactStore:
    """Shared helpers for concrete stores."""

    backend: str = "base"

    @staticmethod
    def _normalise(key: str) -> str:
        """Reject traversal and absolute keys before they reach the backend."""
        key = (key or "").replace("\\", "/").lstrip("/")
        if not key:
            raise ValueError("Artifact key must not be empty")
        parts = [p for p in key.split("/") if p not in ("", ".")]
        if any(p == ".." for p in parts):
            raise ValueError(f"Artifact key must not traverse upwards: {key!r}")
        return "/".join(parts)

    def _missing(self, ref: ArtifactRef) -> ArtifactNotFoundError:
        return ArtifactNotFoundError(
            f"Artifact not found: {ref.key}",
            details={"key": ref.key, "backend": self.backend},
        )


def local_path_of(store: ArtifactStore, ref: ArtifactRef) -> str | None:
    """Best-effort local path for a ref, or None when the store is remote."""
    try:
        path = store.open_path(ref)
    except (NotImplementedError, ArtifactNotFoundError):
        return None
    return path if path and os.path.exists(path) else None


__all__ = ["ArtifactStore", "BaseArtifactStore", "local_path_of"]
