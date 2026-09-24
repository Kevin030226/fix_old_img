"""Artifact service: structured run directories + TTL reclamation (plan section 15).

Layout: storage/tasks/<year>/<month>/<task_id>/{input,stages,output,report.json}
"""
import os
import shutil
import time
import uuid

from app.core.config import settings


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
    import json

    path = os.path.join(run_dir, "report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    return path


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
