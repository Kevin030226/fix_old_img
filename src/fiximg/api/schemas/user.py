"""User API schemas (plan §3.8)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class UserView(BaseModel):
    """A user as exposed by ``GET /api/v1/users`` (never includes a hash)."""

    username: str
    role: str = "user"
    status: str | None = None
    created_at: str | None = None


class UserCreateRequest(BaseModel):
    """Body for creating a user via the JSON API."""

    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")
    password: str = Field(min_length=6)
    role: str = Field(default="user", pattern=r"^(user|admin)$")


class UserUpdateRequest(BaseModel):
    """Body for updating a user; omitted fields are left unchanged."""

    password: str | None = Field(default=None, min_length=6)
    role: str | None = Field(default=None, pattern=r"^(user|admin)$")


__all__ = ["UserCreateRequest", "UserUpdateRequest", "UserView"]
