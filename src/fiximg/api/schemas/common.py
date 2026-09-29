"""Shared API schemas: error envelope and pagination (plan §3.4.1 / §3.8).

Every error response uses the same body::

    {"error": {"code": "IMAGE_TOO_LARGE", "message": "...", "request_id": "..."}}

so clients branch on ``code`` instead of parsing prose.
"""
from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorBody(BaseModel):
    """Machine-readable error detail (plan §3.4.1)."""

    code: str = Field(description="Stable error identifier, e.g. IMAGE_TOO_LARGE")
    message: str = Field(description="Human-readable, safe to display")
    request_id: str | None = Field(
        default=None, description="Correlation id of the failing request"
    )
    details: dict | None = Field(default=None, description="Optional structured context")


class ErrorResponse(BaseModel):
    """Top-level error envelope returned by every failing endpoint."""

    error: ErrorBody


class Pagination(BaseModel):
    """Pagination metadata for list endpoints."""

    limit: int = Field(ge=1, le=500)
    offset: int = Field(ge=0, default=0)
    total: int | None = Field(default=None, description="Total rows when cheaply known")


class Page(BaseModel, Generic[T]):
    """A page of results plus its pagination metadata."""

    items: list[T]
    pagination: Pagination


class MessageResponse(BaseModel):
    """Generic acknowledgement body."""

    message: str
    status: str | None = None


class HealthResponse(BaseModel):
    """``GET /api/v1/health/live`` body."""

    status: str = "ok"


__all__ = [
    "ErrorBody",
    "ErrorResponse",
    "HealthResponse",
    "MessageResponse",
    "Page",
    "Pagination",
]
