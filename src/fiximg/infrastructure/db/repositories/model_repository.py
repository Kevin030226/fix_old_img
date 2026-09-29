"""`model_versions` — the ModelVersion table plan §6 asks for.

One row per ``(name, version)`` that some process has actually tried to load.
Before this table existed, ``GET /api/v1/models/{name}/versions`` answered only for
the requesting process: an API node that never loads a model reported no versions
while its workers served them, and ``status``/``loaded_at`` were whatever
``models/manifest.yaml`` claimed rather than what happened.

The row is **upserted**, never appended: a version that reloads ten times is one
version, and ``load_count`` is what says so. Two processes loading the same version
share the row and the last writer's ``writer`` names who recorded it — the plan's
data model has no per-process dimension, and inventing one here would make the
residency question (which the registry answers per process, correctly) look like
this table's job.

Writes are best-effort by design: this is an audit trail about weights, and losing
a row must not fail a model load that already succeeded. Every caller swallows the
error and logs it — see :func:`record_load`.
"""
from __future__ import annotations

import json
import os
from typing import Any

from fiximg.domain.enums import ModelStatus
from fiximg.domain.tasks import decode_json_column
from fiximg.infrastructure.db import timestamps
from fiximg.infrastructure.db.engine import ensure_core_schema, get_conn
from fiximg.infrastructure.observability.logging import get_logger

logger = get_logger("fiximg.models.catalog")

_COLUMNS = (
    "name", "version", "framework", "weight_uri", "sha256", "status",
    "loaded_at", "load_ms", "load_count", "last_error", "writer", "metadata_json",
)


def process_identity() -> str:
    """Which process recorded a row — the worker id when set, else its PID.

    Not a new identity scheme: ``fiximg.cli.worker`` already defaults to
    ``worker-<pid>``, so an unlabelled process reports the same kind of name.
    """
    return os.environ.get("FIXIMG_WORKER_ID") or f"pid-{os.getpid()}"


def record_load(
    name: str,
    version: str,
    *,
    status: str,
    framework: str | None = None,
    weight_uri: str | None = None,
    sha256: str | None = None,
    load_ms: int | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
    writer: str | None = None,
) -> bool:
    """Upsert one version's latest observation. Returns whether the row was written.

    Never raises: a model that loaded is a model that loads, whatever the audit
    trail does. The boolean exists so a test can assert the write happened instead
    of trusting silence, and a failure is logged rather than hidden.
    """
    #: A failed attempt is not a load: `loaded_at` answers "when did this version
    #: last come up", so only a successful one stamps it (see the COALESCE below).
    stamp = timestamps.now() if status == ModelStatus.READY else None
    try:
        # Inside the try because a process that never booted the application still
        # owes this write a clean False rather than an exception it has to catch.
        ensure_core_schema()
        conn = get_conn()
        conn.execute(
            "INSERT INTO model_versions(name, version, framework, weight_uri, sha256,"
            " status, loaded_at, load_ms, load_count, last_error, writer, metadata_json)"
            " VALUES (?,?,?,?,?,?,?,?,1,?,?,?)"
            " ON CONFLICT (name, version) DO UPDATE SET"
            # Declared fields survive a reload that could not read the manifest:
            # NULL from the new row means "this writer did not know", not "the
            # deployment stopped declaring it", and erasing a sha256 that a later
            # verification depends on is worse than keeping the old one.
            " framework = COALESCE(excluded.framework, model_versions.framework),"
            " weight_uri = COALESCE(excluded.weight_uri, model_versions.weight_uri),"
            " sha256 = COALESCE(excluded.sha256, model_versions.sha256),"
            " metadata_json = COALESCE(NULLIF(excluded.metadata_json, '{}'),"
            " model_versions.metadata_json),"
            " status = excluded.status,"
            # COALESCE'd for the same reason `loaded_at` is only stamped on success:
            # a later failure must not erase when the version last ran.
            " loaded_at = COALESCE(excluded.loaded_at, model_versions.loaded_at),"
            " load_ms = COALESCE(excluded.load_ms, model_versions.load_ms),"
            # Always the new value: a successful load clears the previous error, and
            # this row's purpose is the latest observation, not a history.
            " last_error = excluded.last_error,"
            " writer = excluded.writer,"
            " load_count = model_versions.load_count + 1",
            (
                name, version, framework, weight_uri, sha256, status,
                stamp, load_ms, error,
                writer or process_identity(),
                json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
            ),
        )
        conn.commit()
        return True
    except Exception as exc:  # noqa: BLE001 — the audit trail must not fail a load
        logger.warning(
            "model_versions write failed for %s@%s: %s", name, version, type(exc).__name__
        )
        return False


def list_versions(name: str | None = None) -> list[dict]:
    """Recorded versions, newest observation first, as JSON-friendly rows."""
    ensure_core_schema()
    conn = get_conn()
    if name is None:
        rows = conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM model_versions"
            " ORDER BY name, loaded_at DESC"
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM model_versions"
            " WHERE name = ? ORDER BY loaded_at DESC",
            (name,),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def _row_to_dict(row) -> dict:
    """One row, with the metadata blob decoded.

    ``metadata_json`` is a JSON column: text on SQLite, a structure on PostgreSQL, so
    it goes through the shared decoder rather than a ``json.loads`` that only one of
    the two engines satisfies.
    """
    out = {column: row[column] for column in _COLUMNS}
    raw = out.pop("metadata_json", None)
    decoded = decode_json_column(raw, {})
    out["metadata"] = decoded if isinstance(decoded, dict) else {}
    return out


__all__ = ["list_versions", "process_identity", "record_load"]
