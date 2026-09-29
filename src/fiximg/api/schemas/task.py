"""Task API schemas (plan §2.5 / §3.8).

V2 returned hand-built ``dict`` objects from the task routes, so the OpenAPI
document was empty and a typo in a key was only caught by a client. V3 declares
the request/response contract once, here, and the routes return these models.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator

from fiximg.domain.enums import TaskStatus

#: Option switches accepted from clients (plan §12). Every key here must be read by
#: something that runs — `tests/unit/test_option_vocabulary.py` fails the build if a
#: declared switch gains no consumer, because an option the API accepts and the pipeline
#: ignores is a switch the client believes it is operating.
_ALLOWED_OPTIONS = {"hr", "face_enhance", "auto_colorize"}


class TaskOptionsSchema(BaseModel):
    """Typed task options; unknown keys are rejected (plan §2.5 point 2).

    The two plan switches are **tri-state**, and that is what makes an omission mean
    something: ``face_enhance`` and ``auto_colorize`` both default to ``None`` so that
    "the client said nothing" and "the client refused" are different requests. On
    ``auto_restore`` an omission leaves the decision with the image analysis, ``false``
    declines the stage and ``true`` forces it (the analyzer works on a 512 px copy, so a
    large scan can hold faces it did not count). The route forwards only the keys the
    caller named (``exclude_defaults=True``), which is what keeps silence meaningful —
    and what a defaulted ``False`` here would have destroyed.
    """

    hr: bool = Field(
        default=False,
        description=(
            "High-resolution face-enhancement path. Not a plan switch: it selects the "
            "weights inside the face stages, never adds or removes a stage."
        ),
    )
    face_enhance: bool | None = Field(
        default=None,
        description=(
            "Run the face chain (detection → enhancement → warp-back). "
            "Omit for the default — on for restore/restore_scratch, 'when the analysis "
            "finds a face' for auto_restore; send false to skip it explicitly, true to "
            "force it even when the analysis found none."
        ),
    )
    auto_colorize: bool | None = Field(
        default=None,
        description=(
            "Colorize after restoration. Off by default for the restoration types; for "
            "auto_restore it is tri-state like face_enhance: omitting it colorizes only "
            "what the analysis reads as black and white, false never does, true does it "
            "regardless."
        ),
    )

    @classmethod
    def parse(cls, raw: str | dict | None) -> TaskOptionsSchema:
        """Parse the multipart ``options`` field (JSON object or dict)."""
        if raw in (None, "", {}):
            return cls()
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(parsed, dict):
            raise ValueError("options must be a JSON object")
        unknown = set(parsed) - _ALLOWED_OPTIONS
        if unknown:
            raise ValueError(f"unknown option(s): {', '.join(sorted(unknown))}")
        return cls(**parsed)


class TaskCreateResponse(BaseModel):
    """202 body of ``POST /api/v1/tasks`` (plan §3.8)."""

    task_id: str
    status: TaskStatus
    created_at: str | None = Field(default=None, description="Server timestamp")
    priority: int | None = Field(
        default=None,
        description=(
            "The queue position actually stored (`ORDER BY priority DESC` in the claim). "
            "``null`` means this build did not read it back, not that it is 0."
        ),
    )


class StageView(BaseModel):
    """One executed stage of a task."""

    stage_name: str
    #: Position in the plan, 0-based. The list arrives in this order today, but a
    #: client that renders the plan as a progress rail needs the number, not the
    #: array index: the stored order is what the worker scheduled against.
    stage_order: int | None = None
    status: str = "pending"
    duration_ms: int | None = None
    message: str | None = None
    stage_version: str | None = None
    metrics: dict = Field(
        default_factory=dict,
        description="Numeric measurements this stage reported itself (plan §5.3)",
    )


class ArtifactView(BaseModel):
    """Artifact metadata.

    ``uri`` is the object-store key when the deployment publishes to a remote
    store, and the server-side path when it keeps the local one — the local form is
    what ``GET /result`` and the UI resolve, so it is a path, not a key.
    """

    kind: str
    uri: str
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    model_version: str | None = None


class TaskStatusResponse(BaseModel):
    """``GET /api/v1/tasks/{id}`` body (plan §2.5 point 1)."""

    task_id: str
    status: TaskStatus
    progress: int = Field(default=0, ge=0, le=100)
    current_stage: str | None = None
    error_message: str | None = None
    error_code: str | None = Field(
        default=None,
        description="Machine-readable failure code (plan §3.4.1), e.g. PIPELINE_FAILED",
    )
    duration_ms: int | None = None
    attempt_count: int | None = None
    max_attempts: int | None = None
    #: Queue position actually stored (plan §2.6). Declared ``int | None`` on purpose:
    #: a route that stopped reading the column would answer ``null`` and the test
    #: comparing this against the repository row goes red, whereas an ``int`` default of
    #: ``0`` would make the same bug look like a correct FIFO task.
    priority: int | None = None
    stages: list[StageView] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class TaskReportResponse(BaseModel):
    """``GET /api/v1/tasks/{id}/report`` body."""

    task_id: str
    task_type: str
    status: TaskStatus
    duration_ms: int | None = None
    evaluation_text: str | None = None
    stages: list[StageView] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    planner_decisions: dict | None = None
    artifacts: list[ArtifactView] = Field(default_factory=list)


class TaskArtifactsResponse(BaseModel):
    """``GET /api/v1/tasks/{id}/artifacts`` body."""

    task_id: str
    artifacts: list[ArtifactView] = Field(default_factory=list)


class TaskEventSchema(BaseModel):
    """One SSE payload entry (plan §3.8)."""

    event: str
    task_id: str | None = None
    level: str = "info"
    message: str | None = None
    data: dict = Field(default_factory=dict)
    created_at: str | None = None
    seq: int | None = None


class TaskCreateForm(BaseModel):
    """Documentation model for the multipart create request.

    The endpoint itself takes ``Form(...)``/``File(...)`` parameters (multipart
    cannot be expressed as a JSON body), but declaring the shape keeps it in the
    OpenAPI document and gives one place to describe the fields.
    """

    type: str = Field(
        description="restore | restore_scratch | detect_scratch | colorize | auto_restore"
    )
    image: bytes = Field(description="Image file (multipart part 'image')")
    options: str | None = Field(default=None, description='JSON object, e.g. {"hr": true}')
    priority: int = Field(
        default=0,
        ge=0,
        description="Queue position; the claim orders `priority DESC, created_at`",
    )
    ground_truth: bytes | None = Field(default=None, description="Optional reference photo")

    @field_validator("type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        from fiximg.application.pipeline_modes import PIPELINE_MODES

        allowed = {cfg["task_type"] for cfg in PIPELINE_MODES.values()}
        if value not in allowed:
            raise ValueError(f"unknown task type: {value}")
        return value


__all__ = [
    "ArtifactView",
    "StageView",
    "TaskArtifactsResponse",
    "TaskCreateForm",
    "TaskCreateResponse",
    "TaskEventSchema",
    "TaskOptionsSchema",
    "TaskReportResponse",
    "TaskStatusResponse",
]
