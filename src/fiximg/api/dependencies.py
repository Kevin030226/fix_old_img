"""Authentication dependencies and the acting principal (plan §2.5 point 3).

V2 authenticated ``/api/v1/*`` with one bearer token but then hard-coded
``{"username": "api"}`` when submitting a task, so task ownership never mapped
to a real account. V3 resolves a :class:`Principal` from the credential and
threads its ``user_id`` into task creation and audit events:

    Authentication → Principal(user_id, role) → Task.user_id

The deployment token still acts as an admin credential; ``FIXIMG_API_TOKEN_USER``
names the account it acts as (default ``api``), so an operator can attribute
API traffic to a real user without changing any route code.

There is exactly one credential, and it is an admin credential, so authorisation
is a role check and nothing more: :func:`require_admin` compares ``role`` against
``"admin"``. A per-scope model used to sit here — a ``scopes`` field, a
``_ADMIN_SCOPES`` set of six ``resource:action`` strings, and a ``can()`` predicate
— and none of it was reachable. ``resolve_principal`` builds the one principal
this deployment can produce, with every scope granted, so a scope check would
have been an identity function on the only value in the domain. It read as an
authorization model the service does not have, which is worse than not having
one. Adding scopes is a feature, not a refactor: it needs a scope per route and
a second, lesser credential for it to mean anything. ``tests/unit/
test_principal_surface.py`` holds the line on what ``Principal`` may grow.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends

from fiximg.api.security import require_api_token
from fiximg.config import settings
from fiximg.domain.errors import ForbiddenError


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated actor behind a request."""

    user_id: str
    role: str = "admin"
    via: str = "api_token"

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def resolve_principal() -> Principal:
    """Build the principal for the deployment token.

    Kept as a plain function (not a dependency) so non-HTTP callers — the worker,
    tests — can build the same principal without a request object.
    """
    username = (getattr(settings, "api_token_user", "") or "api").strip() or "api"
    return Principal(user_id=username, role="admin", via="api_token")


async def require_principal(_token: str = Depends(require_api_token)) -> Principal:
    """FastAPI dependency: validate the token and return the acting principal."""
    return resolve_principal()


async def require_admin(principal: Principal = Depends(require_principal)) -> Principal:
    """FastAPI dependency for admin-only routes (model reload, user management)."""
    if not principal.is_admin:
        raise ForbiddenError("Administrator privileges are required for this operation")
    return principal


__all__ = ["Principal", "require_admin", "require_principal", "resolve_principal"]
