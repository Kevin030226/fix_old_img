"""Task domain model (plan §3.1 Step 1, §6).

Typed objects that cross layer boundaries. The V2 code base passed anonymous
``dict`` rows between services; V3 defines the shape once here so the planner,
runtime, worker and API all speak the same vocabulary.

``Task.from_row()`` keeps the repository able to hand back raw sqlite3 rows
while the rest of the application works with the dataclass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from fiximg.domain.enums import TaskStatus, TaskType

#: Option switches accepted from the API/UI (plan §12). Kept equal to the API schema's
#: whitelist by `tests/unit/test_option_vocabulary.py`, and every key here has to be read
#: by something that runs: `strength` used to live in this vocabulary with no reader at
#: all, which is how an API can advertise a control that changes nothing.
KNOWN_OPTIONS: frozenset[str] = frozenset(
    {"hr", "face_enhance", "auto_colorize"}
)

#: Internal options injected by the service layer (never accepted from clients).
INTERNAL_OPTION_KEYS: frozenset[str] = frozenset({"_ground_truth_path"})

#: Columns the task repository hands a caller as structured data *or* as JSON text,
#: depending on the engine. The stage and metric aggregates are built with
#: ``json_group_array``/``json_group_object`` on SQLite, which return TEXT, and with
#: ``json_agg``/``json_object_agg`` on PostgreSQL, where psycopg decodes them on the
#: way out; the option and capability blobs are stored as TEXT and may come back as
#: either. Nothing downstream may therefore trust the type —
#: :func:`decode_json_column` is the one place that knows both shapes.
JSON_COLUMNS: frozenset[str] = frozenset(
    {
        "stages", "metrics", "metrics_json", "options_json",
        "required_capabilities", "data_json", "metadata_json",
    }
)


def decode_json_column(value, default=None):
    """One ``JSON_COLUMNS`` value, in whichever shape this engine returned it.

    ``default`` covers NULL, empty text and text that does not parse: a row written by
    an older build still has to render, and every caller here has a meaningful empty
    value (an empty checklist, no options).
    """
    if value is None or value == "":
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return default
    return value


@dataclass(slots=True)
class TaskOptions:
    """Typed view over the free-form ``options`` JSON column.

    Unknown keys are preserved (forward compatibility) but only the declared
    switches are interpreted by the planner.
    """

    hr: bool = False
    face_enhance: bool = False
    auto_colorize: bool = False
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict | None) -> TaskOptions:
        raw = dict(raw or {})
        # The declared switches are all flags, and anything the caller sent that this
        # build does not know about stays in `extra` with the caller's own types — which
        # is what makes dropping a declared switch (as `strength` was dropped) a
        # non-destructive change for rows that still carry it.
        known: dict[str, Any] = {
            "hr": bool(raw.pop("hr", False)),
            "face_enhance": bool(raw.pop("face_enhance", False)),
            "auto_colorize": bool(raw.pop("auto_colorize", False)),
        }
        return cls(**known, extra=raw)

    def to_dict(self) -> dict:
        """Serialise back to the JSON column shape (extra keys preserved)."""
        out = dict(self.extra)
        out.update(
            {
                "hr": self.hr,
                "face_enhance": self.face_enhance,
                "auto_colorize": self.auto_colorize,
            }
        )
        return out

    # NOTE: there is deliberately no `planner_switches()` here. Which switches steer the
    # plan is the planner's own declaration (`PipelinePlanner.PLAN_SWITCHES`), and a
    # second list in the data model is how `hr` ended up described as a planner switch
    # when the planner never reads it.


@dataclass(slots=True)
class Task:
    """A unit of work: one image through one pipeline."""

    id: str
    task_type: str
    user_id: str = "unknown"
    status: str = TaskStatus.QUEUED
    progress: int = 0
    current_stage: str | None = None
    input_path: str | None = None
    result_path: str | None = None
    error_message: str | None = None
    evaluation_text: str | None = None
    duration_ms: int | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    options: TaskOptions = field(default_factory=TaskOptions)

    # --- queue bookkeeping (plan §2.6) -----------------------------------
    priority: int = 0
    attempt_count: int = 0
    max_attempts: int = 3
    worker_id: str | None = None
    lease_until: str | None = None
    last_heartbeat: str | None = None
    retry_at: str | None = None
    idempotency_key: str | None = None

    # --- derived ----------------------------------------------------------
    @property
    def status_enum(self) -> TaskStatus:
        return TaskStatus(self.status)

    @property
    def type_enum(self) -> TaskType:
        return TaskType(self.task_type)

    @property
    def can_retry(self) -> bool:
        """True when a failed task still has attempts left."""
        return self.status == TaskStatus.FAILED and self.attempt_count < self.max_attempts

    @classmethod
    def from_row(cls, row: Any) -> Task:
        """Build a Task from a sqlite3.Row / mapping (extra keys ignored)."""
        data = dict(row)
        #: `options_json` is one of ``JSON_COLUMNS``: text on SQLite, a structure on
        #: PostgreSQL. A reader that assumed one shape would see "no options" on the
        #: other engine and run the default plan.
        parsed = decode_json_column(data.get("options_json"), {})
        return cls(
            id=data.get("id", ""),
            task_type=data.get("task_type", ""),
            user_id=data.get("user_id") or "unknown",
            status=data.get("status") or TaskStatus.QUEUED,
            progress=int(data.get("progress") or 0),
            current_stage=data.get("current_stage"),
            input_path=data.get("input_path"),
            result_path=data.get("result_path"),
            error_message=data.get("error_message"),
            evaluation_text=data.get("evaluation_text"),
            duration_ms=data.get("duration_ms"),
            created_at=data.get("created_at"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            options=TaskOptions.from_dict(parsed),
            priority=int(data.get("priority") or 0),
            attempt_count=int(data.get("attempt_count") or 0),
            max_attempts=int(data.get("max_attempts") or 3),
            worker_id=data.get("worker_id"),
            lease_until=data.get("lease_until"),
            last_heartbeat=data.get("last_heartbeat"),
            retry_at=data.get("retry_at"),
            idempotency_key=data.get("idempotency_key"),
        )


@dataclass(slots=True)
class TaskStage:
    """One executed (or failed) stage of a task."""

    task_id: str
    stage_name: str
    stage_order: int
    status: str = "pending"
    duration_ms: int | None = None
    message: str | None = None
    stage_version: str = "1.0"
    created_at: str | None = None
    finished_at: str | None = None

    @classmethod
    def from_row(cls, row: Any) -> TaskStage:
        data = dict(row)
        return cls(
            task_id=data.get("task_id", ""),
            stage_name=data.get("stage_name", ""),
            stage_order=int(data.get("stage_order") or 0),
            status=data.get("status") or "pending",
            duration_ms=data.get("duration_ms"),
            message=data.get("message"),
            stage_version=data.get("stage_version") or "1.0",
            created_at=data.get("created_at"),
            finished_at=data.get("finished_at"),
        )


@dataclass(slots=True)
class Metric:
    """A single quality measurement attached to a task (plan §6)."""

    task_id: str
    name: str
    value: float | str
    reference_type: str = "input_output_difference"
    created_at: datetime | None = None


__all__ = [
    "INTERNAL_OPTION_KEYS",
    "JSON_COLUMNS",
    "KNOWN_OPTIONS",
    "Metric",
    "Task",
    "TaskOptions",
    "TaskStage",
    "decode_json_column",
]
