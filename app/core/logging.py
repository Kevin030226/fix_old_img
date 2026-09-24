"""JSON structured logging helpers (plan section 25).

Emits one-line JSON records so logs can be indexed and filtered by
task_id / stage / duration without regex parsing.

Correlation fields (plan §25): `request_id` (one per API/UI request), `user_id`
(the signed-in user performing the action) and `gpu_id` (the device executing
a stage) travel via contextvars — set by the API middleware / auth paths /
worker — and are attached to every record automatically.
"""
import contextvars
import json
import logging
import os
import sys
import uuid
from datetime import datetime, UTC

# Per-request / per-task correlation (inherited by everything running in the
# same async context; threads spawned via copy_context keep them too).
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id", default="")
gpu_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("gpu_id", default="")


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:12]
    request_id_var.set(rid)
    return rid


def set_user_id(user: str | None) -> str:
    """Attach the acting user to every log line of the current context (§25)."""
    value = str(user or "")
    user_id_var.set(value)
    return value


def set_gpu_id(gpu: int | str) -> str:
    value = str(gpu)
    gpu_id_var.set(value)
    return value


def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Correlation fields (§25) — only included when set.
        rid = request_id_var.get()
        if rid:
            payload["request_id"] = rid
        user = user_id_var.get()
        if user:
            payload["user_id"] = user
        gpu = gpu_id_var.get()
        if gpu:
            payload["gpu_id"] = gpu

        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=_json_default)


def get_logger(name: str = "fiximg") -> logging.Logger:
    """Return a logger that writes JSON lines to stdout/stderr."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        level = os.environ.get("FIXIMG_LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, level: str, message: str, **fields) -> None:
    """Log a structured event with arbitrary fields (task_id, stage, duration_ms...)."""
    logger.log(
        getattr(logging, level.upper(), logging.INFO),
        message,
        extra={"extra_fields": fields},
    )
