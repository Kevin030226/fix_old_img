"""Uniform error handling for the HTTP layer (plan §3.4.1).

Registers handlers that turn every failure into the same envelope::

    {"error": {"code": "...", "message": "...", "request_id": "..."}}

Handled cases:
  * :class:`~fiximg.domain.errors.AppError` and its subclasses — code + status
    come from the exception itself,
  * ``HTTPException`` raised by FastAPI internals (401 from the token guard,
    404 from routing, ...) — mapped onto a stable code,
  * ``RequestValidationError`` — 422 with per-field detail,
  * any other exception — 500 with the code hidden from the client but logged
    with the request id.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from fiximg.domain.errors import AppError, ErrorCode
from fiximg.infrastructure.observability.logging import get_logger, log_event

logger = get_logger("fiximg.api.errors")

#: HTTP status → fallback code for exceptions that are not AppError subclasses.
_STATUS_CODES = {
    400: ErrorCode.INVALID_REQUEST,
    401: ErrorCode.UNAUTHENTICATED,
    403: ErrorCode.FORBIDDEN,
    404: "NOT_FOUND",
    409: "CONFLICT",
    410: ErrorCode.ARTIFACT_EXPIRED,
    413: ErrorCode.UPLOAD_TOO_LARGE,
    422: ErrorCode.INVALID_REQUEST,
    429: ErrorCode.RATE_LIMITED,
    503: "SERVICE_UNAVAILABLE",
}


def _request_id() -> str | None:
    from fiximg.infrastructure.observability.logging import request_id_var

    return request_id_var.get() or None


def error_body(code: str, message: str, details: dict | None = None) -> dict:
    """Build the canonical error envelope body."""
    body: dict = {"code": code, "message": message, "request_id": _request_id()}
    if details:
        body["details"] = details
    return {"error": body}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every handler to ``app`` (called from the composition root)."""

    @app.exception_handler(AppError)
    async def _app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
        log_event(
            logger, "WARNING", "request failed",
            code=exc.code, status=exc.status_code, error=exc.message,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.code, exc.message, exc.details or None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [
            {"loc": list(err.get("loc", [])), "msg": err.get("msg", ""), "type": err.get("type", "")}
            for err in exc.errors()
        ]
        first = fields[0]["msg"] if fields else "Invalid request"
        return JSONResponse(
            status_code=422,
            content=error_body(ErrorCode.INVALID_REQUEST, first, {"fields": fields}),
        )

    @app.exception_handler(HTTPException)
    async def _http_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        code = _STATUS_CODES.get(exc.status_code, "HTTP_ERROR")
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        headers = getattr(exc, "headers", None)
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(code, detail),
            headers=headers,
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(_request: Request, exc: Exception) -> JSONResponse:
        # The message is logged (with the request id) but never returned: an
        # internal failure must not leak stack or path details to a client.
        log_event(
            logger, "ERROR", "unhandled exception",
            error_type=type(exc).__name__, error=str(exc)[:500],
        )
        return JSONResponse(
            status_code=500,
            content=error_body(
                ErrorCode.INTERNAL_ERROR,
                "Internal server error; quote the request id when reporting it.",
            ),
        )


__all__ = ["error_body", "register_exception_handlers"]
