"""Unified application error hierarchy (plan §3.4.1).

Every error that can reach the HTTP layer carries a stable machine-readable
``code`` plus an HTTP status, so the API can emit one consistent envelope::

    {
      "error": {
        "code": "IMAGE_TOO_LARGE",
        "message": "Image long side exceeds 4096 px",
        "request_id": "..."
      }
    }

``AppError`` is the root. The V2 names (``TaskError``, ``PipelineFailedError``,
...) are kept as subclasses/aliases so existing raise sites and tests continue
to work unchanged — the V3 migration is additive, not a rewrite.
"""
from __future__ import annotations


class ErrorCode:
    """Stable error codes surfaced to API clients (never localised)."""

    # --- request / validation ---
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_IMAGE = "INVALID_IMAGE"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    UPLOAD_TOO_LARGE = "UPLOAD_TOO_LARGE"
    UNSUPPORTED_TASK_TYPE = "UNSUPPORTED_TASK_TYPE"
    UNSUPPORTED_STAGE = "UNSUPPORTED_STAGE"
    INVALID_OPTIONS = "INVALID_OPTIONS"

    # --- auth / policy ---
    UNAUTHENTICATED = "UNAUTHENTICATED"
    FORBIDDEN = "FORBIDDEN"
    PASSWORD_CHANGE_REQUIRED = "PASSWORD_CHANGE_REQUIRED"
    RATE_LIMITED = "RATE_LIMITED"

    # --- task lifecycle ---
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    TASK_NOT_CANCELLABLE = "TASK_NOT_CANCELLABLE"
    TASK_CANCELLED = "TASK_CANCELLED"
    TASK_NOT_READY = "TASK_NOT_READY"
    TASK_ALREADY_FINISHED = "TASK_ALREADY_FINISHED"

    # --- execution ---
    PIPELINE_FAILED = "PIPELINE_FAILED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    QUEUE_FULL = "QUEUE_FULL"

    # --- storage ---
    ARTIFACT_NOT_FOUND = "ARTIFACT_NOT_FOUND"
    ARTIFACT_EXPIRED = "ARTIFACT_EXPIRED"

    # --- fallback ---
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AppError(Exception):
    """Base class for every error the application raises deliberately.

    Attributes:
        code: stable machine-readable identifier (see :class:`ErrorCode`).
        message: human-readable, safe to return to the caller.
        status_code: HTTP status the API layer maps this error onto.
        details: optional structured context (never contains secrets).
    """

    code: str = ErrorCode.INTERNAL_ERROR
    status_code: int = 500

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(message or self.__class__.__name__)
        self.message = message or self.__class__.__name__
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> dict:
        """Serialise to the API error envelope body (without ``request_id``)."""
        payload: dict = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


# ---------------------------------------------------------------- validation
class InvalidRequestError(AppError):
    """User input failed validation (missing image, wrong format, bad options)."""

    code = ErrorCode.INVALID_REQUEST
    status_code = 400


class InvalidImageError(InvalidRequestError):
    """The uploaded payload is not a decodable image."""

    code = ErrorCode.INVALID_IMAGE


class ImageTooLargeError(InvalidRequestError):
    """Image dimensions exceed ``FIXIMG_MAX_IMAGE_SIDE``."""

    code = ErrorCode.IMAGE_TOO_LARGE


class UploadTooLargeError(InvalidRequestError):
    """The uploaded payload exceeds ``FIXIMG_MAX_UPLOAD_MB``."""

    code = ErrorCode.UPLOAD_TOO_LARGE
    status_code = 413


class UnsupportedTaskTypeError(InvalidRequestError):
    """The requested task type is not in the planner's table."""

    code = ErrorCode.UNSUPPORTED_TASK_TYPE


class InvalidOptionsError(InvalidRequestError):
    """The ``options`` payload is not a JSON object of known switches."""

    code = ErrorCode.INVALID_OPTIONS


# -------------------------------------------------------------------- policy
class ForbiddenError(AppError):
    """The principal is authenticated but not allowed to perform the action."""

    code = ErrorCode.FORBIDDEN
    status_code = 403


class PasswordChangeRequiredError(AppError):
    """Plan §21: the account must replace its initial password first."""

    code = ErrorCode.PASSWORD_CHANGE_REQUIRED
    status_code = 403


class RateLimitedError(AppError):
    """Too many requests inside the configured window."""

    code = ErrorCode.RATE_LIMITED
    status_code = 429


# ------------------------------------------------------------ task lifecycle
class TaskError(AppError):
    """Base class for task-related errors surfaced to the web layer."""

    code = ErrorCode.INTERNAL_ERROR
    status_code = 500


class TaskNotFoundError(TaskError):
    """No task row exists for the requested id."""

    code = ErrorCode.TASK_NOT_FOUND
    status_code = 404


class TaskNotCancellableError(TaskError):
    """The task has no run left to cancel.

    Raised for a missing row and for one that already reached a terminal state. A task
    that is merely *running* is cancellable — the worker stops at the next stage
    boundary — so this is no longer the "not yet started or too late" answer it was.
    """

    code = ErrorCode.TASK_NOT_CANCELLABLE
    status_code = 409


class TaskCancelledError(TaskError):
    """The user cancelled this task while the pipeline was running it.

    In-process control flow more than an HTTP answer: by the time this is raised the
    row is already `cancelled`, so the runtime must not overwrite that with `failed`
    and the worker must not spend another attempt on it. A client can still see it if
    the race goes the other way (the run finished first), which is why it carries a
    code and a status at all.
    """

    code = ErrorCode.TASK_CANCELLED
    status_code = 409


class TaskNotReadyError(TaskError):
    """The task has not produced a result yet."""

    code = ErrorCode.TASK_NOT_READY
    status_code = 409


class TaskAlreadyFinishedError(TaskError):
    """The operation is invalid because the task is still active."""

    code = ErrorCode.TASK_ALREADY_FINISHED
    status_code = 409


# ----------------------------------------------------------------- execution
class PipelineFailedError(TaskError):
    """A pipeline stage failed (subprocess exit code or missing output)."""

    code = ErrorCode.PIPELINE_FAILED


class StageNotAvailableError(TaskError):
    """The requested stage is not registered in the planner/registry."""

    code = ErrorCode.UNSUPPORTED_STAGE
    status_code = 400


class ModelUnavailableError(TaskError):
    """A required model weight is missing, unloaded or failed its health probe."""

    code = ErrorCode.MODEL_UNAVAILABLE
    status_code = 503


class QueueFullError(TaskError):
    """The task queue is saturated; the submission was refused."""

    code = ErrorCode.QUEUE_FULL
    status_code = 503


# ------------------------------------------------------------------- storage
class ArtifactError(AppError):
    """Base class for artifact-store failures."""

    code = ErrorCode.ARTIFACT_NOT_FOUND
    status_code = 404


class ArtifactNotFoundError(ArtifactError):
    """The referenced artifact does not exist in the store."""


class ArtifactExpiredError(ArtifactError):
    """The artifact existed but has been reclaimed by the TTL sweeper."""

    code = ErrorCode.ARTIFACT_EXPIRED
    status_code = 410


__all__ = [
    "AppError",
    "ArtifactError",
    "ArtifactExpiredError",
    "ArtifactNotFoundError",
    "ErrorCode",
    "ForbiddenError",
    "ImageTooLargeError",
    "InvalidImageError",
    "InvalidOptionsError",
    "InvalidRequestError",
    "ModelUnavailableError",
    "PasswordChangeRequiredError",
    "PipelineFailedError",
    "QueueFullError",
    "RateLimitedError",
    "StageNotAvailableError",
    "TaskError",
    "TaskAlreadyFinishedError",
    "TaskNotCancellableError",
    "TaskNotFoundError",
    "TaskNotReadyError",
    "UnsupportedTaskTypeError",
    "UploadTooLargeError",
]
