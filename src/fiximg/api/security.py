"""Bearer-token authentication for the JSON API (`/api/v1/*`).

Why this exists: the web UI talks to the service layer *in process*, but the
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

First-boot protocol (the part that has to survive several processes starting at
once, e.g. `docker compose up --scale api=3`):

* every writer takes an OS-level :class:`filelock.FileLock` on
  ``api_token.txt.lock`` and re-reads the token *inside* the lock;
* a generated token is written to a temporary file in the same directory,
  flushed with ``fsync``, and moved into place with an atomic ``os.replace`` —
  so a reader can never observe an empty or half-written token;
* a sibling ``api_token.txt.meta`` records the SHA-256 of the token, which makes
  a truncated file detectable (a legacy file with no sibling is still accepted:
  operators are documented to place their own token there);
* a process that loses the race adopts the winner's value and *waits* for it —
  "could not read it yet" is never a reason to fall back to its own candidate.

If the directory is not writable and the token is not deployment-managed, the
process fails closed with an actionable error instead of serving a token no other
process shares.
"""
import hashlib
import hmac
import json
import os
import secrets
import stat
import tempfile
import threading
import time
from contextlib import contextmanager

from fastapi import Header, HTTPException

from fiximg.config import settings

_ALPHABET = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_TOKEN_LENGTH = 32
#: Shortest value accepted as an operator-provided token. A generated token is
#: 32 characters; a shorter file is treated as a truncated write, not a secret.
_MIN_TOKEN_LENGTH = 16
#: Longest value accepted from a file. Guards against a file that is not a token
#: at all (a log segment, a binary blob) being adopted as the API secret.
_MAX_TOKEN_LENGTH = 4096
#: How long a booting process waits for the process that won the lock.
def _lock_timeout_from_env() -> float:
    """Positive seconds to wait for the peer process; a bad value falls back.

    Import-time crashes from an environment knob are unhelpful (the traceback
    points at an unrelated import); a bad value is reported and the default used.
    """
    raw = os.environ.get("FIXIMG_API_TOKEN_LOCK_TIMEOUT")
    if raw is None or not raw.strip():
        return 30.0
    try:
        value = float(raw)
    except ValueError:
        print(f"[API] Warning: FIXIMG_API_TOKEN_LOCK_TIMEOUT={raw!r} is not a number; using 30s.")
        return 30.0
    if value <= 0:
        print(f"[API] Warning: FIXIMG_API_TOKEN_LOCK_TIMEOUT={raw!r} must be positive; using 30s.")
        return 30.0
    return value


_TOKEN_LOCK_TIMEOUT = _lock_timeout_from_env()
_TOKEN_META_VERSION = 1

_lock = threading.Lock()
_cached: str | None = None


class TokenResolutionError(RuntimeError):
    """The API token could not be resolved to a single, persisted value."""


def token_file() -> str:
    """Path of the generated token, in the management data directory.

    `settings.management_data_dir` derives the directory from `db_path` on every
    read, so a deployment that relocates the database with `FIXIMG_DATABASE_URL`
    gets the token beside the database it actually opens. Computing it here from
    `os.path.dirname(settings.db_path)` is the same value on purpose -- the
    property exists so both writers (this one and the bootstrap password file)
    cannot drift apart again.
    """
    return os.path.join(settings.management_data_dir, "api_token.txt")


def reset_cache() -> None:
    """Drop the memoised token.

    Only the test suite calls this: `db_path` is a process-level setting, so a
    test that relocates the database (or a fixture that rewrites the token file)
    has to invalidate a token resolved against the previous one. Production
    resolves the token once per process and keeps it, which is what makes a
    mid-flight change of `FIXIMG_API_TOKEN` not take effect until a restart.
    """
    global _cached
    with _lock:
        _cached = None


def _meta_path(path: str) -> str:
    """Sibling file recording the digest of the token this process wrote."""
    return path + ".meta"


def _lock_path(path: str) -> str:
    return path + ".lock"


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _plausible_token(token: str) -> bool:
    """True when `token` could be a bearer secret rather than a burst of noise.

    Deliberately permissive about the alphabet: an operator may place any value
    in `api_token.txt`, and `docs/api.md` documents that a token *file* counts as
    configured. It is strict about length and whitespace, which is what separates
    a secret from a truncated write or a file that is not a token at all.
    """
    if not (_MIN_TOKEN_LENGTH <= len(token) <= _MAX_TOKEN_LENGTH):
        return False
    return all(32 <= ord(ch) < 127 and not ch.isspace() for ch in token)


def _read_token(path: str) -> str:
    """The complete, verified token stored at `path`, or "" when unusable.

    A sibling ``.meta`` file is written next to every token this module mints.
    When it is present its digest must match the token byte-for-byte, so a
    truncated or partially-overwritten file is reported as unusable instead of
    being adopted. When it is absent the file predates the meta protocol (or was
    placed by an operator): it is accepted when it looks like a secret.
    """
    token = _read_text(path)
    if not token:
        return ""
    meta_raw = _read_text(_meta_path(path))
    if meta_raw:
        try:
            meta = json.loads(meta_raw)
        except (ValueError, TypeError):
            return ""
        if not isinstance(meta, dict) or meta.get("version") != _TOKEN_META_VERSION:
            return ""
        expected = meta.get("sha256")
        length = meta.get("length")
        if not isinstance(expected, str) or len(expected) != 64:
            return ""
        if length != len(token):
            return ""
        actual = hashlib.sha256(token.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(actual, expected):
            return ""
        return token
    return token if _plausible_token(token) else ""


def _generate() -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(_TOKEN_LENGTH))


def _atomic_write(path: str, body: str) -> None:
    """Write `body` to `path` so no reader can observe a partial file.

    Temp file in the same directory (same filesystem, so ``os.replace`` is
    atomic), explicit ``fsync`` before the rename, and 0600 before the file has a
    name a reader could open.
    """
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".api_token.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _persist(path: str, token: str) -> None:
    """Persist `token` and its digest, atomically and in that order.

    The token is written first: if the process dies before the meta file exists,
    the next boot finds a complete token without a digest and accepts it as an
    operator-placed value. The reverse order would leave a digest with no token,
    which is a corrupt state with no recovery path.
    """
    _atomic_write(path, token + "\n")
    meta = {
        "version": _TOKEN_META_VERSION,
        "length": len(token),
        "sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
    }
    _atomic_write(_meta_path(path), json.dumps(meta, sort_keys=True) + "\n")


@contextmanager
def _token_lock(path: str):
    """Cross-process lock around the read-generate-write sequence.

    ``filelock`` is a declared runtime dependency and is the only primitive here
    that orders separate *processes* on both POSIX and Windows; the module-level
    ``threading.Lock`` only orders threads. A crashed holder releases the OS lock,
    so there is no stale-lock recovery to get wrong.
    """
    from filelock import FileLock, Timeout

    lock = FileLock(_lock_path(path), timeout=_TOKEN_LOCK_TIMEOUT)
    try:
        with lock:
            yield
    except Timeout as exc:
        raise TokenResolutionError(
            f"Timed out after {_TOKEN_LOCK_TIMEOUT:g}s waiting for another process "
            f"to finish creating {path}. If the file already exists, re-run; if it "
            "does not, check that the management data directory is writable."
        ) from exc


def _cleanup_stale_temps(path: str, max_age_seconds: float = 3600.0) -> None:
    """Remove temp files left by a process that died mid-write.

    Only files matching this module's prefix in the same directory, and only when
    they are older than `max_age_seconds`, so a concurrent writer's live temp file
    is never touched.
    """
    directory = os.path.dirname(path) or "."
    try:
        names = os.listdir(directory)
    except OSError:
        return
    cutoff = time.time() - max_age_seconds
    for name in names:
        if not (name.startswith(".api_token.") and name.endswith(".tmp")):
            continue
        candidate = os.path.join(directory, name)
        try:
            if os.path.getmtime(candidate) < cutoff:
                os.unlink(candidate)
        except OSError:
            continue


def _resolve_persisted_token(path: str) -> str:
    """Return the deployment's persisted token, creating one if needed.

    Serialised across processes: the read, the generate and the write all happen
    inside one ``filelock`` critical section, so a process can never adopt its own
    candidate while another process is still writing the winner's file.
    """
    directory = os.path.dirname(path)
    try:
        if directory:
            os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        raise TokenResolutionError(
            f"Cannot create the API token directory {directory!r}: {exc}. Set "
            "FIXIMG_API_TOKEN to a deployment-managed secret, or make that "
            "directory writable."
        ) from exc

    with _token_lock(path):
        token = _read_token(path)
        if token:
            return token

        _cleanup_stale_temps(path)
        if os.path.exists(path):
            # Present but unusable (empty, truncated, or a digest mismatch). Keep
            # the damaged file for diagnosis and replace it with a complete one.
            try:
                os.replace(path, path + ".corrupt")
            except OSError:
                pass
            try:
                os.unlink(_meta_path(path))
            except OSError:
                pass

        candidate = _generate()
        _persist(path, candidate)
        # Read back what actually landed rather than trusting the write: this is
        # the value the next process will see, and every process must agree.
        stored = _read_token(path)
        if stored != candidate:
            raise TokenResolutionError(
                f"Wrote an API token to {path} but could not read the same value "
                "back; the file may be on a filesystem with broken rename "
                "semantics. Set FIXIMG_API_TOKEN instead."
            )
        _announce(path, candidate)
        return candidate


def _announce(path: str, token: str) -> None:
    print("=" * 64)
    print("[API] Generated an API token for /api/v1/* - store it safely:")
    print(f"[API]   token: {token}")
    print(f"[API]   file : {path}")
    print("[API] Send it as:  Authorization: Bearer <token>")
    print("=" * 64)


def get_api_token() -> str:
    """Resolve the API token, generating and persisting one on first use.

    Every successful boot returns the same non-empty value that is on disk. If
    that cannot be guaranteed the call raises :class:`TokenResolutionError` —
    a process whose secret no peer shares is worse than a process that refuses
    to start.
    """
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
        token = _read_token(path)

        if not token:
            if settings.is_production:
                # Plan §3.3 lists "API secret missing → do not boot" for the
                # production profile, and allows auto-generated development
                # configuration only locally. Minting a secret into
                # `admin_data/api_token.txt` on a production boot is the failure
                # the rule exists to prevent: the service comes up healthy, its
                # token is not the one the operator set, and a second replica
                # generates a different one.
                raise ValueError(
                    "Production configuration has no API token: set FIXIMG_API_TOKEN "
                    "or place the secret at " + path
                )
            token = _resolve_persisted_token(path)

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
    # Compare bytes, not str: `hmac.compare_digest` raises TypeError for any
    # non-ASCII str, and the header arrives latin-1 decoded, so a caller sending
    # raw high bytes made every authenticated route answer 500 instead of 401.
    # Encoding both sides keeps the comparison constant-time and total.
    if not provided or not hmac.compare_digest(
        provided.encode("utf-8", "surrogateescape"),
        expected.encode("utf-8", "surrogateescape"),
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return provided
