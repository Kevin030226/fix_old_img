"""Task submission, shared by the JSON API and the native web UI (plan §3.8/§23).

Two front doors accept the same multipart upload: `POST /api/v1/tasks` for a
token-bearing API client and `POST /ui/api/tasks` for a signed-in browser session.
They must answer identically — the same limits, the same refusals, the same queue
entry — so the whole submission lives here once and each route only translates its
own transport (bearer header vs. session cookie) into a principal.

Why this is not written twice in the routes: the checks below are *ordered*, and the
order is the contract. The rate limit is charged before the image is decoded (the
decode is what a flood pays for), `priority` is refused rather than clamped (a
caller that asked for 20 and quietly got 10 would believe a queue position it did
not receive), and the option keys are validated before anything reaches the queue.
A second copy would drift the first time either door was edited, and the drift would
surface as two front doors with different limits.

The option contract itself stays in :mod:`fiximg.api.schemas.task`, which is where
it is declared for the OpenAPI document; this module consumes it rather than
restating the allowed keys.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any

from fiximg.api.schemas.task import TaskOptionsSchema
from fiximg.application.pipeline_modes import PIPELINE_MODES
from fiximg.application.task_service import task_service
from fiximg.config import settings
from fiximg.domain.errors import (
    ImageTooLargeError,
    InvalidImageError,
    InvalidOptionsError,
    InvalidRequestError,
    RateLimitedError,
    UnsupportedTaskTypeError,
    UploadTooLargeError,
)
from fiximg.infrastructure.db import timestamps
from fiximg.infrastructure.security import rate_limit

#: Task types the queue accepts. Derived from the mode table so a new mode is
#: submittable without editing a second list.
ALLOWED_TASK_TYPES: frozenset[str] = frozenset(
    cfg["task_type"] for cfg in PIPELINE_MODES.values()
)


@dataclass(frozen=True, slots=True)
class Submission:
    """What a submission produced: the queue entry and the position it bought."""

    task_id: str
    created_at: str | None
    #: The queue position actually stored, read back from the row. `None` means this
    #: build could not read it, which is not the same statement as "FIFO".
    priority: int | None


def max_upload_bytes() -> int:
    """The upload ceiling in force now.

    Read per call rather than captured at import: a test (or an operator reloading
    settings) that lowers `FIXIMG_MAX_UPLOAD_MB` must change the answer, and a module
    constant captured at import would keep serving the old one.
    """
    return settings.max_upload_mb * 1024 * 1024


def decode_upload(payload: bytes, what: str = "image"):
    """Decode an uploaded payload into a PIL RGB image, or refuse it by name.

    Empty, oversized and undecodable uploads are three different refusals with three
    different remedies, so they stay three different errors instead of one
    "bad request".
    """
    from PIL import Image as PILImage

    if not payload:
        raise InvalidImageError(f"Empty {what} upload")
    if len(payload) > max_upload_bytes():
        raise UploadTooLargeError(
            f"{what.capitalize()} exceeds the {settings.max_upload_mb} MB upload limit"
        )
    try:
        return PILImage.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - every decoder failure is one refusal
        raise InvalidImageError(f"Invalid {what} file") from exc


def parse_options(raw: str | None) -> TaskOptionsSchema:
    """Parse the multipart `options` field (a JSON object, or nothing at all)."""
    try:
        return TaskOptionsSchema.parse(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise InvalidOptionsError(str(exc)) from exc


def check_rate_limit(principal: Any, request: Any) -> None:
    """Charge one submission to this principal, or raise `RATE_LIMITED`.

    Reads the limiter off the module attribute rather than binding it at import: the
    fixture that gives every test a clean allowance replaces that attribute, and a
    bound reference would keep the process-wide window.
    """
    key, scope = rate_limit.submission_key(principal, request)
    limiter = rate_limit.submit_limiter
    if limiter.hit(key):
        return
    # Report the limits of the limiter that just decided, not of the module
    # constants: a rebuilt limiter and a clamped `FIXIMG_SUBMIT_MAX` can disagree, and
    # a `details.limit` the caller cannot reconcile with the behaviour is worse than
    # no number at all.
    raise RateLimitedError(
        f"Too many task submissions ({limiter.max_count} per "
        f"{limiter.window_seconds}s); try again later",
        details={
            "limit": limiter.max_count,
            "window_seconds": limiter.window_seconds,
            "scope": scope,
            "remaining": limiter.remaining(key),
        },
    )


def submit_upload(
    *,
    principal: Any,
    task_type: str,
    image_bytes: bytes,
    options_raw: str | None = None,
    priority: int = 0,
    ground_truth_bytes: bytes | None = None,
    idempotency_key: str | None = None,
    request: Any = None,
) -> Submission:
    """Validate one upload and queue it; the single path both front doors use.

    `principal` is duck-typed on `.user_id` (the `Principal` dataclass in
    `fiximg.api.dependencies` and the session principal in
    `fiximg.ui.web.session` both satisfy it), so the application layer does not have
    to import the HTTP layer to describe its caller.
    """
    check_rate_limit(principal, request)
    if task_type not in ALLOWED_TASK_TYPES:
        raise UnsupportedTaskTypeError(
            f"Unknown task type: {task_type}",
            details={"allowed": sorted(ALLOWED_TASK_TYPES)},
        )
    # `priority` decides who gets a GPU next, so it is checked against the limit the
    # deployment is actually running (read here rather than at import time so a
    # rebuilt settings object cannot make the reply describe a limit that no longer
    # exists) and refused rather than clamped.
    if not 0 <= priority <= settings.priority_max:
        raise InvalidRequestError(
            f"priority must be between 0 and {settings.priority_max}",
            details={"priority": priority, "min": 0, "max": settings.priority_max},
        )
    task_options = parse_options(options_raw)

    pil_image = decode_upload(image_bytes)
    if max(pil_image.size) > settings.max_image_side:
        raise ImageTooLargeError(
            f"Image long side exceeds {settings.max_image_side} px",
            details={"max_side": settings.max_image_side, "size": list(pil_image.size)},
        )

    # §16 Ground Truth mode: an optional reference photo enables LPIPS evaluation.
    gt_image = None
    if ground_truth_bytes:
        gt_image = decode_upload(ground_truth_bytes, "ground truth")

    explicit_options = task_options.model_dump(exclude_defaults=True)
    task_id = task_service.enqueue(
        pil_image,
        {"username": getattr(principal, "user_id", None) or "unknown"},
        task_type,
        options=explicit_options or None,
        ground_truth_image=gt_image,
        idempotency_key=idempotency_key,
        priority=priority,
    )
    return Submission(
        task_id=task_id,
        # The same instant convention the row is stored with (plan §2.7), so a
        # client comparing this against `GET /tasks/{id}` sees one clock.
        created_at=timestamps.now(),
        priority=task_service.task_priority(task_id),
    )


__all__ = [
    "ALLOWED_TASK_TYPES",
    "Submission",
    "check_rate_limit",
    "decode_upload",
    "max_upload_bytes",
    "parse_options",
    "submit_upload",
]
