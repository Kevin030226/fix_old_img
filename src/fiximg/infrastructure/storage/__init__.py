"""Artifact storage backends (plan §2.8).

    from fiximg.infrastructure.storage import get_artifact_store

    store = get_artifact_store()
    ref = store.put_bytes(png_bytes, "2026/09/<task_id>/output/final.png")

Selection is by ``FIXIMG_STORAGE_BACKEND`` (``local`` default, ``s3`` optional).
"""
from __future__ import annotations

from fiximg.infrastructure.storage.base import (
    ArtifactStore,
    BaseArtifactStore,
    local_path_of,
)
from fiximg.infrastructure.storage.local import LocalArtifactStore, local_store

_store: ArtifactStore | None = None


def get_artifact_store() -> ArtifactStore:
    """Resolve the configured artifact store once per process."""
    global _store
    if _store is not None:
        return _store
    from fiximg.config import settings

    if getattr(settings, "storage_backend", "local") == "s3":
        from fiximg.infrastructure.storage.s3 import build_s3_store_from_settings

        built = build_s3_store_from_settings(settings)
        if built is not None:
            _store = built
            return _store
    _store = local_store
    return _store


def reset_artifact_store() -> None:
    """Drop the cached store (tests / config reload)."""
    global _store
    _store = None


def object_key_for(path: str) -> str:
    """The object key a local path maps to — one rule for every artifact.

    The run tree is rooted at `tasks_root` for every backend and a remote store
    has no local root to ask (it exposes no `.root`), so the key is derived from
    the setting. That is the only reason the node that wrote
    `storage/tasks/2026/09/<id>/output/final.png` and the node that reads it back
    agree on the string.
    """
    import os

    from fiximg.config import settings

    relative = os.path.relpath(path, settings.tasks_root).replace(os.sep, "/")
    if not relative.startswith(".."):
        return relative
    return (
        f"ext/{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
    )


def publish_local_file(path: str, *, mime_type: str = "image/png",
                       op: str = "publish") -> str | None:
    """Upload a file this node already holds and return its key (None if local).

    A local store returns None on purpose: the path already *is* the location, and
    mirroring a *relative* key into `artifacts.uri` would hand every reader
    (``row["uri"] or row["path"]``) something to resolve against whatever the
    process's working directory happens to be.

    A failed upload is swallowed. The producing node still has the bytes, so
    turning a storage hiccup into a failed task would throw away work the user can
    see; what is lost is the *other* node's ability to do this work.
    """
    import os
    import time

    from fiximg.infrastructure.observability.metrics import MetricName, metrics
    from fiximg.infrastructure.storage.local import local_store

    store = get_artifact_store()
    if store.backend == local_store.backend or not os.path.isfile(path):
        return None
    key = object_key_for(path)
    try:
        started = time.perf_counter()
        store.put_file(path, key, mime_type=mime_type)
        metrics.observe(
            MetricName.ARTIFACT_IO_SECONDS, time.perf_counter() - started,
            op=op, store=store.backend,
        )
    except Exception:  # noqa: BLE001 — see the docstring
        return None
    return key


def fetch_to_local(key: str, destination: str) -> bool:
    """Materialise a published object at ``destination``; False if unavailable."""
    import os

    from fiximg.domain.artifacts import ArtifactRef
    from fiximg.infrastructure.observability.logging import get_logger, log_event
    from fiximg.infrastructure.observability.metrics import MetricName, metrics

    logger = get_logger("fiximg.storage")
    store = get_artifact_store()
    try:
        import time

        started = time.perf_counter()
        data = store.get_bytes(ArtifactRef(key=key, backend=store.backend))
        metrics.observe(
            MetricName.ARTIFACT_IO_SECONDS, time.perf_counter() - started,
            op="fetch", store=store.backend,
        )
    except Exception as exc:  # noqa: BLE001 — the caller decides what it means
        log_event(logger, "WARNING", "artifact fetch failed",
                  key=key, destination=destination, error=str(exc))
        return False
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as handle:
        handle.write(data)
    return True


def purge_expired_published(cutoff: str, limit: int = 200) -> int:
    """Delete objects the retention TTL has claimed; returns how many.

    The symmetric half of publishing (plan §2.8): without it an S3/MinIO backend
    only ever grows, because `purge_stale_runs()` walks a directory tree this node
    may not even have. Rows keep their `path` as history and lose their `uri`, so a
    failed delete is retried on the next pass only if the object is still listed.

    Local stores return 0 — those bytes are the directory sweep's business, and
    letting two reclaimers race over the same files is how artifacts vanish early.
    """
    from fiximg.domain.artifacts import ArtifactRef
    from fiximg.infrastructure.db.repositories import task_repository

    store = get_artifact_store()
    if store.backend == local_store.backend:
        return 0
    removed = 0
    for row in task_repository.expired_published_artifacts(cutoff, limit=limit):
        try:
            store.delete(ArtifactRef(key=row["uri"], backend=store.backend))
        except Exception:  # noqa: BLE001 — a failed delete retries on the next pass
            continue
        task_repository.clear_artifact_uri(row["id"])
        removed += 1
    return removed


__all__ = [
    "ArtifactStore",
    "BaseArtifactStore",
    "LocalArtifactStore",
    "fetch_to_local",
    "get_artifact_store",
    "local_path_of",
    "local_store",
    "object_key_for",
    "purge_expired_published",
    "publish_local_file",
    "reset_artifact_store",
]
