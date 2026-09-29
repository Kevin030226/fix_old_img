"""Artifact domain model (plan §2.8, §6).

An artifact is any file a task produces or consumes (input, output, mask,
ground truth, report). The domain only knows a *key* plus metadata; resolving a
key to bytes is the storage layer's job (:mod:`fiximg.infrastructure.storage`).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

from fiximg.domain.enums import ArtifactKind


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A location-independent handle to an artifact.

    ``key`` is backend-specific: a relative path for the local store, an object
    key for S3/MinIO. Callers must never treat it as a filesystem path.
    """

    key: str
    backend: str = "local"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.backend}:{self.key}"


@dataclass(slots=True)
class Artifact:
    """Artifact metadata recorded in the ``artifacts`` table."""

    task_id: str
    kind: str
    uri: str
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    model_version: str | None = None
    created_at: str | None = None

    @classmethod
    def from_row(cls, row: Any) -> Artifact:
        data = dict(row)
        return cls(
            task_id=data.get("task_id", ""),
            kind=data.get("kind", ""),
            # V2 stored a filesystem path in `path`; V3 calls the same value `uri`.
            uri=data.get("uri") or data.get("path") or "",
            mime_type=data.get("mime_type"),
            width=data.get("width"),
            height=data.get("height"),
            size_bytes=data.get("size_bytes"),
            sha256=data.get("sha256"),
            model_version=data.get("model_version"),
            created_at=data.get("created_at"),
        )


def sha256_of(path: str, chunk_size: int = 1 << 20) -> str | None:
    """Best-effort SHA-256 of a file; None when unreadable (plan §14)."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def probe_image(path: str) -> tuple[int | None, int | None]:
    """Return ``(width, height)`` for an image file; ``(None, None)`` on failure."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:  # noqa: BLE001 — metadata probing must never fail a run
        return None, None


def build_artifact(
    task_id: str,
    kind: str | ArtifactKind,
    uri: str,
    mime_type: str | None = None,
) -> Artifact:
    """Collect size/hash/dimensions for a freshly written file."""
    width = height = size = None
    digest = None
    try:
        size = os.path.getsize(uri)
        digest = sha256_of(uri)
        width, height = probe_image(uri)
    except OSError:
        pass
    return Artifact(
        task_id=task_id,
        kind=str(kind),
        uri=uri,
        mime_type=mime_type,
        width=width,
        height=height,
        size_bytes=size,
        sha256=digest,
    )


__all__ = ["Artifact", "ArtifactRef", "build_artifact", "probe_image", "sha256_of"]
