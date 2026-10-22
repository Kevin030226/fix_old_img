"""Browser sessions for the native web UI (plan §3.9 / §21).

The Gradio shell kept its own session store and every layer that needed to know
"who is this caller" read Gradio's cookie out of the mounted app
(`fiximg/ui/middleware.py`, now deleted). Replacing the shell means owning that
mechanism, so it lives here and nowhere else.

Design constraints, each of which removes a failure mode the Gradio coupling had:

* **No new dependency.** The cookie is `base64(payload).hmac_sha256`, verified in
  constant time. Starlette's SessionMiddleware needs `itsdangerous`, which is not
  in the published dependency set; adding it to keep a cookie would be a
  dependency for a signature we can compute with `hmac`.
* **Server-side session records.** The cookie carries an opaque id, not the
  username. A signed cookie holding the username would keep working after the
  account was deleted or demoted, and the role check would then be reading a
  claim rather than a fact. The record is looked up here, and the role is read
  from the users table on every request, exactly like the Gradio path did.
* **Expiry and rotation.** A session has an absolute deadline; logging out
  deletes the record, so a copied cookie stops working immediately.
* **CSRF token per session.** The cookie is `SameSite=Lax`, and every mutating
  request must echo the session's CSRF token in `X-CSRF-Token`. Lax alone does
  not stop a top-level form POST from another site.
* **Fail closed.** An unknown, expired, malformed or unsigned cookie resolves to
  "not signed in", never to a default account. A lookup failure is not a login.

The secret is generated once and persisted beside the API token (mode 0600). If
that directory is unwritable the secret stays in memory for the process, which
is a documented degraded mode: sessions then do not survive a restart, and no
session is ever accepted without a matching record.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import stat
import threading
import time
from dataclasses import dataclass

from fiximg.config import settings
from fiximg.infrastructure.db.repositories.user_repository import get_user
from fiximg.infrastructure.observability.logging import get_logger, log_event
from fiximg.ui.state import ROLE_USER, UserState

logger = get_logger("fiximg.ui.session")

#: Cookie name. Deliberately not Gradio's `access-token-*` shape: nothing should
#: be able to mistake one mechanism's cookie for the other's.
COOKIE_NAME = "fiximg_session"
#: Header a mutating UI request must echo its CSRF token in.
CSRF_HEADER = "X-CSRF-Token"
#: Absolute session lifetime, in seconds. Short enough that an abandoned browser
#: stops being a credential, long enough that a restoration run is not logged out
#: mid-poll (the worker's own poll timeout is 30 minutes).
SESSION_TTL_SECONDS = 8 * 3600
#: How many sessions a process remembers. Bounded so a login flood cannot grow
#: the dict without limit; oldest-expiring records are dropped first.
MAX_SESSIONS = 512


class SessionSecretError(RuntimeError):
    """The signing secret could not be resolved or persisted."""


def _secret_file() -> str:
    """Where the signing secret lives: beside the API token and the DB.

    `settings.management_data_dir` is derived from `db_path` on every read, so a
    deployment that relocates the database with `FIXIMG_DATABASE_URL` gets the
    secret beside the database it actually opens instead of next to the default.
    """
    return os.path.join(settings.management_data_dir, "session_secret.txt")


def _load_or_create_secret() -> bytes:
    """The HMAC key: `FIXIMG_SESSION_SECRET`, else the persisted file, else a new one.

    An explicit environment value wins and is never written to disk, so a
    multi-replica deployment can share one key without a writable volume. A
    generated key is persisted 0600; when that write fails the key stays in
    memory and the process says so once, because the alternative (failing every
    login on a read-only volume) would take the whole UI down for a session
    property that still holds within the process.
    """
    env = os.environ.get("FIXIMG_SESSION_SECRET", "").strip()
    if env:
        return env.encode("utf-8")

    path = _secret_file()
    try:
        with open(path, encoding="utf-8") as handle:
            stored = handle.read().strip()
        if stored:
            return stored.encode("utf-8")
    except OSError:
        stored = ""

    generated = secrets.token_urlsafe(48)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(generated + "\n")
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
    except OSError as exc:
        log_event(
            logger, "WARNING", "session secret not persisted",
            error=str(exc),
            consequence="sessions do not survive a restart of this process",
        )
    return generated.encode("utf-8")


_secret_lock = threading.Lock()
_secret: bytes | None = None


def signing_secret() -> bytes:
    """The process-wide signing key, resolved once."""
    global _secret
    if _secret is not None:
        return _secret
    with _secret_lock:
        if _secret is None:
            _secret = _load_or_create_secret()
        return _secret


def reset_secret_cache() -> None:
    """Drop the memoised key (test seam: a fixture may rewrite the secret file)."""
    global _secret
    with _secret_lock:
        _secret = None


@dataclass(frozen=True, slots=True)
class Session:
    """One signed-in browser."""

    session_id: str
    username: str
    role: str
    csrf_token: str
    expires_at: float

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def user_state(self) -> UserState:
        """The caller in the vocabulary the rest of the UI layer already speaks."""
        return UserState(username=self.username, role=self.role)

    def expired(self, now: float | None = None) -> bool:
        return (now if now is not None else time.time()) >= self.expires_at


class SessionStore:
    """Server-side session records, keyed by the opaque id in the cookie."""

    def __init__(self, ttl: int = SESSION_TTL_SECONDS, max_sessions: int = MAX_SESSIONS,
                 clock=time.time) -> None:
        self._ttl = ttl
        self._max = max_sessions
        self._clock = clock
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ create
    def create(self, username: str, role: str | None = None) -> Session:
        """Start a session for `username`, reading the role from the users table.

        The role is looked up rather than taken from the caller: a route that
        passed its own idea of the role could hand out an admin session for a
        demoted account, and the whole authorisation model rests on this number
        being the database's.
        """
        resolved = role
        if resolved is None:
            user = get_user(username)
            resolved = (user or {}).get("role") or ROLE_USER
        session = Session(
            session_id=secrets.token_urlsafe(32),
            username=username,
            role=resolved,
            csrf_token=secrets.token_urlsafe(32),
            expires_at=self._clock() + self._ttl,
        )
        with self._lock:
            self._prune_locked()
            self._sessions[session.session_id] = session
        return session

    # -------------------------------------------------------------------- read
    def get(self, session_id: str | None) -> Session | None:
        """The live session for `session_id`, or None when it is not usable.

        Expired records are dropped on the way out so a stale cookie cannot be
        resurrected by a clock that moved backwards.
        """
        if not session_id:
            return None
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.expired(self._clock()):
                self._sessions.pop(session_id, None)
                return None
            return session

    def refresh_role(self, session: Session) -> Session:
        """Re-read the role from the users table; drop the session if gone.

        A user deleted while signed in must not keep a working admin cookie, and
        a role change must take effect without a re-login. Returning a *new*
        record keeps the dataclass frozen.
        """
        user = get_user(session.username)
        if not user:
            self.delete(session.session_id)
            return session
        role = user.get("role") or ROLE_USER
        if role == session.role:
            return session
        updated = Session(
            session_id=session.session_id,
            username=session.username,
            role=role,
            csrf_token=session.csrf_token,
            expires_at=session.expires_at,
        )
        with self._lock:
            self._sessions[session.session_id] = updated
        return updated

    # ------------------------------------------------------------------ delete
    def delete(self, session_id: str | None) -> None:
        if not session_id:
            return
        with self._lock:
            self._sessions.pop(session_id, None)

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    # ----------------------------------------------------------------- helpers
    def _prune_locked(self) -> None:
        """Drop expired records, then the soonest-expiring ones over the cap."""
        now = self._clock()
        for key in [k for k, v in self._sessions.items() if v.expired(now)]:
            self._sessions.pop(key, None)
        overflow = len(self._sessions) - self._max + 1
        if overflow > 0:
            for key in sorted(self._sessions, key=lambda k: self._sessions[k].expires_at)[:overflow]:
                self._sessions.pop(key, None)


#: The process-wide store. Tests replace this attribute rather than the class, so
#: a fixture can hand out a clean store without reloading the module.
session_store = SessionStore()


# ------------------------------------------------------------------- the cookie
def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def sign_session_id(session_id: str) -> str:
    """`<id>.<hmac>` — the value the cookie actually carries."""
    digest = hmac.new(signing_secret(), session_id.encode("utf-8"), hashlib.sha256).digest()
    return session_id + "." + _b64encode(digest)


def read_session_id(cookie_value: str | None) -> str | None:
    """The id inside a signed cookie, or None when the signature does not hold.

    Constant-time comparison, and a malformed value is simply not a session: a
    parsing error must never fall through to an unauthenticated default that some
    later route mistakes for a caller.
    """
    if not cookie_value or "." not in cookie_value:
        return None
    session_id, _, signature = cookie_value.rpartition(".")
    if not session_id or not signature:
        return None
    expected = hmac.new(signing_secret(), session_id.encode("utf-8"), hashlib.sha256).digest()
    try:
        provided = _b64decode(signature)
    except (ValueError, TypeError):
        return None
    if not hmac.compare_digest(provided, expected):
        return None
    return session_id


def resolve_session(cookie_value: str | None) -> Session | None:
    """The live session behind a cookie value, with the role refreshed."""
    session = session_store.get(read_session_id(cookie_value))
    if session is None:
        return None
    return session_store.refresh_role(session)


def csrf_matches(session: Session, provided: str | None) -> bool:
    """Constant-time CSRF check for a mutating request."""
    if not provided:
        return False
    return hmac.compare_digest(provided, session.csrf_token)


def session_payload(session: Session) -> dict:
    """The `/ui/api/session` body: what the page needs to lay itself out.

    `must_change_password` is read live for the same reason the role is: the
    banner and the submission gate have to agree with the row, not with whatever
    was true at login.
    """
    user = get_user(session.username) or {}
    must_change = bool(user.get("must_change_password"))
    return {
        "username": session.username,
        "role": session.role,
        "is_admin": session.is_admin,
        "csrf_token": session.csrf_token,
        "must_change_password": must_change,
        # §21: the same gate task_service enforces, surfaced before a submit is
        # attempted so the page can disable the button instead of explaining a 403.
        "can_submit": not must_change,
    }


def describe_cookie(session: Session) -> dict:
    """Cookie attributes for `set_cookie`. One place, so they cannot drift."""
    return {
        "key": COOKIE_NAME,
        "value": sign_session_id(session.session_id),
        "max_age": SESSION_TTL_SECONDS,
        "httponly": True,
        # Lax, not Strict: a link from a chat message into the app should land
        # signed in. Mutating requests are covered by the CSRF header instead.
        "samesite": "lax",
        "path": "/",
    }


def describe_logout_cookie() -> dict:
    """`delete_cookie` keyword arguments: the cookie to remove, and its scope.

    `delete_cookie` takes no `value`/`max_age`; it writes the expiry itself. Passing
    the same shape as :func:`describe_cookie` raised inside the handler and turned a
    sign-out into a 500, which is why the two shapes are separate functions rather
    than one dict with a flag.
    """
    return {
        "key": COOKIE_NAME,
        "path": "/",
        "httponly": True,
        "samesite": "lax",
    }


__all__ = [
    "COOKIE_NAME",
    "CSRF_HEADER",
    "MAX_SESSIONS",
    "SESSION_TTL_SECONDS",
    "Session",
    "SessionSecretError",
    "SessionStore",
    "csrf_matches",
    "describe_cookie",
    "describe_logout_cookie",
    "read_session_id",
    "reset_secret_cache",
    "resolve_session",
    "session_payload",
    "session_store",
    "sign_session_id",
    "signing_secret",
]
