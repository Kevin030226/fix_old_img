"""LocalArtifactStore — filesystem-backed artifact store (plan §2.8 / §5.6).

Layout: ``storage/tasks/<year>/<month>/<task_id>/{input,stages,output,report.json}``

The module keeps its V2 function surface (``create_run_dir``,
``save_input_image``, ``write_report``, ``purge_stale_runs``, ``maybe_purge``)
so existing call sites keep working, and adds :class:`LocalArtifactStore`
implementing the :class:`~fiximg.infrastructure.storage.base.ArtifactStore`
protocol for the V3 key-based path.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from typing import BinaryIO

from fiximg.config import settings
from fiximg.domain.artifacts import ArtifactRef
from fiximg.infrastructure.storage.base import BaseArtifactStore


class LocalArtifactStore(BaseArtifactStore):
    """Artifact store rooted at a directory (default ``<repo>/storage/tasks``).

    The root is resolved lazily when ``root`` was not passed explicitly, so a
    settings override (tests, profile switch) takes effect without rebuilding
    the singleton.
    """

    backend = "local"

    def __init__(self, root: str | None = None) -> None:
        self._root = root

    @property
    def root(self) -> str:
        return self._root or settings.tasks_root

    # ---------------------------------------------------------------- helpers
    def path_for(self, key: str) -> str:
        """Absolute path for a store key (created under ``root`` only)."""
        return os.path.join(self.root, self._normalise(key))

    # -------------------------------------------------------------- protocol
    def put_bytes(self, data: bytes, key: str, mime_type: str | None = None) -> ArtifactRef:
        path = self.path_for(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(data)
        return ArtifactRef(key=self._normalise(key), backend=self.backend)

    def put_file(self, source_path: str, key: str, mime_type: str | None = None) -> ArtifactRef:
        path = self.path_for(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.abspath(source_path) != os.path.abspath(path):
            shutil.copyfile(source_path, path)
        return ArtifactRef(key=self._normalise(key), backend=self.backend)

    def get_bytes(self, ref: ArtifactRef) -> bytes:
        path = self.path_for(ref.key)
        if not os.path.exists(path):
            raise self._missing(ref)
        with open(path, "rb") as handle:
            return handle.read()

    def open_stream(self, ref: ArtifactRef) -> BinaryIO:
        path = self.path_for(ref.key)
        if not os.path.exists(path):
            raise self._missing(ref)
        return open(path, "rb")

    def open_path(self, ref: ArtifactRef) -> str:
        path = self.path_for(ref.key)
        if not os.path.exists(path):
            raise self._missing(ref)
        return path

    def delete(self, ref: ArtifactRef) -> None:
        try:
            os.remove(self.path_for(ref.key))
        except OSError:
            pass

    def exists(self, ref: ArtifactRef) -> bool:
        return os.path.exists(self.path_for(ref.key))


#: Default store used by the module-level helpers below.
local_store = LocalArtifactStore()


# ------------------------------------------------------- V2-compatible surface
def create_run_dir(task_id: str) -> str:
    """Create and return storage/tasks/<y>/<m>/<task_id>/."""
    now = time.localtime()
    run_dir = os.path.join(
        settings.tasks_root, f"{now.tm_year:04d}", f"{now.tm_mon:02d}", task_id
    )
    os.makedirs(os.path.join(run_dir, "input"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "stages"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "output"), exist_ok=True)
    return run_dir


def save_input_image(run_dir: str, task_id: str, pil_image) -> str:
    """Persist the uploaded input image and return its path."""
    path = os.path.join(run_dir, "input", task_id + ".png")
    pil_image.save(path)
    return path


def output_dir_for(run_dir: str) -> str:
    return os.path.join(run_dir, "output")


def write_report(run_dir: str, report: dict) -> str:
    path = os.path.join(run_dir, "report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    return path


def key_for_run(task_id: str, *parts: str) -> str:
    """Store key for a run artifact, e.g. ``key_for_run(tid, "output", "final.png")``."""
    now = time.localtime()
    return "/".join(
        [f"{now.tm_year:04d}", f"{now.tm_mon:02d}", task_id, *parts]
    )


def _is_under(path: str, root: str) -> bool:
    try:
        path = os.path.realpath(path)
        root = os.path.realpath(root)
        return os.path.commonpath([path, root]) == root
    except (ValueError, OSError):
        return False


def purge_stale_runs(ttl_seconds: int | None = None) -> None:
    """Delete run directories older than the TTL (allowlist: storage/tasks)."""
    ttl = ttl_seconds if ttl_seconds is not None else settings.result_ttl
    now = time.time()
    root = settings.tasks_root
    if not os.path.isdir(root):
        return
    for dirpath, _dirnames, _ in os.walk(root, topdown=False):
        if dirpath == root:
            continue
        try:
            if now - os.path.getmtime(dirpath) > ttl and _is_under(dirpath, root):
                shutil.rmtree(dirpath, ignore_errors=True)
        except OSError:
            pass


def maybe_purge(ttl_seconds: int | None = None, probability: float = 0.05) -> None:
    """Low-frequency cleanup to avoid scanning the tree on every request."""
    if uuid.uuid4().int % 100 < int(probability * 100):
        purge_stale_runs(ttl_seconds)


__all__ = [
    "LocalArtifactStore",
    "create_run_dir",
    "key_for_run",
    "local_store",
    "maybe_purge",
    "output_dir_for",
    "purge_stale_runs",
    "save_input_image",
    "write_report",
]
