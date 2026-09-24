"""Bearer-token authentication for the JSON API (`/api/v1/*`).

Why this exists: the Gradio UI talks to the service layer *in process*, but the
JSON API is plain HTTP. When the service is bound to a routable address
(`FIXIMG_HOST=0.0.0.0`, as the Dockerfile/compose topology does), every
`/api/v1/*` route used to be reachable without credentials — including the
account list and finished result files.

Policy (fail closed): every `/api/v1/*` route requires
`Authorization: Bearer <token>`. Health probes stay public.

Token resolution:
  1. `FIXIMG_API_TOKEN` — deployment-managed (Docker secret / compose env);
  2. `admin_data/api_token.txt` — read on every start so restarts keep working;
  3. generated on first use, written 0600 and printed once to the console.

There is no "empty token" state: a token always exists, so the API can never be
reached anonymously by accident.
"""
import hmac
import os
import secrets
import stat
import threading

from fastapi import Header, HTTPException

from app.core.config import settings

_ALPHABET = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_TOKEN_LENGTH = 32

_lock = threading.Lock()
_cached: str | None = None


def token_file() -> str:
    """Path of the generated token (next to the SQLite database)."""
    return os.path.join(os.path.dirname(settings.db_path), "api_token.txt")


def reset_cache() -> None:
    """Drop the memoised token (used by tests and by settings overrides)."""
    global _cached
    with _lock:
        _cached = None


def _generate() -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(_TOKEN_LENGTH))


def _persist(path: str, token: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(token + "\n")
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def _announce(path: str, token: str) -> None:
    print("=" * 64)
    print("[API] Generated an API token for /api/v1/* - store it safely:")
    print(f"[API]   token: {token}")
    print(f"[API]   file : {path}")
    print("[API] Send it as:  Authorization: Bearer <token>")
    print("=" * 64)


def get_api_token() -> str:
    """Resolve the API token, generating and persisting one on first use."""
    global _cached
    if _cached:
        return _cached
    with _lock:
        if _cached:
            return _cached

        env_token = os.environ.get("FIXIMG_API_TOKEN", "").strip()
        if env_token:
            _cached = env_token
            return _cached

        path = token_file()
        token = ""
        try:
            with open(path, encoding="utf-8") as f:
                token = f.read().strip()
        except OSError:
            token = ""

        if not token:
            token = _generate()
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                _persist(path, token)
                _announce(path, token)
            except OSError:
                # Unwritable volume: keep the generated token in memory so the
                # API still works for this process instead of failing open.
                print("[API] Warning: could not persist the API token; it is valid "
                      "for this process only. Set FIXIMG_API_TOKEN to pin it.")

        _cached = token
        return _cached


async def require_api_token(authorization: str | None = Header(default=None)) -> str:
    """FastAPI dependency: reject requests without a valid bearer token."""
    expected = get_api_token()
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authentication required: send 'Authorization: Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    provided = authorization[7:].strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid API token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return provided
