"""UI session state — one place that decides "who is this caller?" (§3.2).

The role of the caller drives three things: which tabs are visible, which rows of
the history are visible, and whether the admin panel is reachable. Before, that
decision was duplicated: ``gradio_app.apply_role`` looked the user up, and
``history_panel._visible_user_id`` re-derived the scoping rule from the state
dict. Two implementations of the same rule is how the two drift apart.

This module owns the rule; the callers consume it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fiximg.infrastructure.db.repositories.user_repository import get_user

#: Role names understood by the UI.
ROLE_ADMIN = "admin"
ROLE_USER = "user"


@dataclass(frozen=True, slots=True)
class UserState:
    """The caller's identity as the UI needs it."""

    username: str | None = None
    role: str = ROLE_USER

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN

    @property
    def is_authenticated(self) -> bool:
        return bool(self.username)

    def to_dict(self) -> dict:
        return {"username": self.username, "role": self.role}

    @classmethod
    def from_any(cls, value: Any) -> UserState:
        """Coerce a dict / None / existing UserState into a UserState."""
        if isinstance(value, UserState):
            return value
        if isinstance(value, dict):
            return cls(
                username=value.get("username") or None,
                role=value.get("role") or ROLE_USER,
            )
        return cls()


def resolve_user_state(request) -> UserState:
    """Resolve the caller from a Gradio request, failing safe to a plain user.

    A lookup failure must never hand out the admin layout, so any error resolves
    to the least privileged role.
    """
    username = getattr(request, "username", None)
    if not username:
        return UserState()
    try:
        user = get_user(username)
    except Exception:  # noqa: BLE001 — fail safe to the non-admin layout
        user = None
    role = (user or {}).get("role", ROLE_USER)
    return UserState(username=username, role=role)


def visible_user_id(user_state) -> str | None:
    """Whose rows this caller may see; None means "everyone" (admin).

    Users see their own history; administrators see all of it. Returning None for
    an admin matches the repository's ``user_id=None`` convention.
    """
    state = UserState.from_any(user_state)
    if state.is_admin:
        return None
    return state.username or None


def tab_visibility(user_state) -> dict:
    """Which tab groups this caller may see (plan §3.9.1).

    Returned as a mapping so the layout code and its tests share one definition
    of the rule instead of restating it.
    """
    state = UserState.from_any(user_state)
    return {
        "function_tabs": not state.is_admin,
        "history_tab": not state.is_admin,
        "admin_panel": state.is_admin,
    }


__all__ = [
    "ROLE_ADMIN",
    "ROLE_USER",
    "UserState",
    "resolve_user_state",
    "tab_visibility",
    "visible_user_id",
]
