"""Domain enumerations (plan §3.2, §6).

String-valued enums so they serialise directly into SQLite / JSON without a
conversion layer, while still giving type-checked constants in code.
"""
from __future__ import annotations

from enum import StrEnum


class TaskStatus(StrEnum):
    """Task lifecycle states (the state machine in plan §2.6)."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """True when no further transition is possible without a retry."""
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)

    @property
    def is_active(self) -> bool:
        """True while the task still occupies the queue or a worker slot."""
        return self in (TaskStatus.QUEUED, TaskStatus.RUNNING)


class TaskType(StrEnum):
    """Supported pipeline entry points (plan §9/§10)."""

    RESTORE = "restore"
    RESTORE_SCRATCH = "restore_scratch"
    DETECT_SCRATCH = "detect_scratch"
    COLORIZE = "colorize"
    AUTO_RESTORE = "auto_restore"


class StageStatus(StrEnum):
    """Per-stage lifecycle (``task_stages.status``)."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ArtifactKind(StrEnum):
    """Artifact roles recorded in the ``artifacts`` table.

    This list is *exactly* the set the code can write — gated both ways by
    ``test_domain_model.py``, because a declared role no stage emits and an emitted
    role nobody declares are the same defect in opposite directions. The two members
    that used to sit here without a producer (`colorized`, `report`) were
    decoration: a colorized picture is the task's `output`, and the vendor's degrade
    report is stage *metadata*, never a file of its own.

    `final` is the vendored stage-4 composite at the model's geometry, which is not
    what `GET /result` serves — that is `output`, resized back to the upload's
    geometry. Both are kept because the pipeline's own intermediates are what a
    reproduction run compares against.
    """

    INPUT = "input"
    OUTPUT = "output"
    GROUND_TRUTH = "ground_truth"
    MASK = "mask"
    RESTORED = "restored"
    #: A stage's own result image, before the runtime composes the task output.
    RESULT = "result"
    FINAL = "final"
    #: Directories, not single files: `faces_dir` holds the detection crops,
    #: `each_img_dir` the enhanced ones. Recorded so a client can tell "no faces"
    #: from "faces found but not enhanced".
    FACES_DIR = "faces_dir"
    EACH_IMG_DIR = "each_img_dir"


class EventType(StrEnum):
    """Task event names streamed over SSE (plan §2.5 / §3.8)."""

    TASK_CREATED = "task.created"
    TASK_ENQUEUED = "task.enqueued"
    TASK_STARTED = "task.started"
    TASK_PROGRESS = "task.progress"
    TASK_COMPLETED = "task.completed"
    TASK_EVALUATED = "task.evaluated"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"
    TASK_RETRYING = "task.retrying"
    #: Refused before a task row existed because the backlog hit
    #: `FIXIMG_WORKER_MAX_QUEUE`. The submitting client sees HTTP 503 `QUEUE_FULL`;
    #: this row carries no `task_id`, so it is the operator's audit trail, not a
    #: frame on any stream.
    TASK_REJECTED = "task.rejected"
    STAGE_STARTED = "stage.started"
    STAGE_COMPLETED = "stage.completed"
    #: A stage that declined its own work (no dlib, no aligned crop, no scratch to
    #: look for). One terminal event per stage either way, so a client that counts
    #: stage events sees the same number of them.
    STAGE_SKIPPED = "stage.skipped"
    STAGE_FAILED = "stage.failed"
    #: A stage that ran out of VRAM on one device and was re-run on another
    #: (plan §7.1). Neither terminal nor progress: the stage still reports its own
    #: outcome afterwards, so a client must not count this one.
    STAGE_RETRIED = "stage.retried"


class ModelStatus(StrEnum):
    """Model version lifecycle (plan §3.5.2).

    Four states because those are the ones a reader can actually observe: an
    unloaded version reports `registered` (the same value a model that was never
    loaded reports — `GET /api/v1/models` distinguishes them with `loaded`), and a
    load in progress is not yet in the registry at all. The `loading` and
    `unloaded` names that used to sit here were never assigned anywhere, which in a
    vocabulary the API emits means a client could never have been written to
    handle them.
    """

    REGISTERED = "registered"
    READY = "ready"
    UNHEALTHY = "unhealthy"


class DeviceKind(StrEnum):
    """Resolved execution device."""

    CUDA = "cuda"
    CPU = "cpu"


__all__ = [
    "ArtifactKind",
    "DeviceKind",
    "EventType",
    "ModelStatus",
    "StageStatus",
    "TaskStatus",
    "TaskType",
]
