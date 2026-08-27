"""
Credential security utilities (fixes B4: plaintext passwords / non-atomic writes).

Design notes:
- Password hashing uses stdlib hashlib.pbkdf2_hmac (no third-party deps); format is self-describing: pbkdf2_<alg>$<rounds>$<salt_hex>$<hash_hex>
- Verification uses hmac.compare_digest for constant-time comparison, resisting timing side channels;
- Writes use a temp-file + os.replace atomic commit, plus a filelock cross-process lock, avoiding torn files or lost updates on concurrent writes.
"""
import hashlib
import hmac
import os
import secrets
import tempfile
import threading
from contextlib import contextmanager

import yaml

try:
    from filelock import FileLock
except Exception:  # pragma: no cover - falls back to an in-process lock in the rare environments without filelock
    FileLock = None

ALG = "sha256"
DEFAULT_ROUNDS = 260_000  # aligned with config/users.example.yaml

# in-process serialization (concurrent requests within a single uvicorn process)
_WRITE_LOCK = threading.Lock()


def hash_password(password: str, *, rounds: int = DEFAULT_ROUNDS) -> str:
    """Generate a self-describing hash string for a plaintext password."""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac(ALG, password.encode("utf-8"), salt, rounds)
    return f"pbkdf2_{ALG}${rounds}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time verification. Returns False for invalid or differently-formatted stored values."""
    if not stored or "$" not in stored:
        return False
    try:
        alg, rounds_s, salt_hex, hash_hex = stored.split("$", 3)
    except ValueError:
        return False
    if alg != f"pbkdf2_{ALG}":
        return False
    try:
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        rounds = int(rounds_s)
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac(ALG, password.encode("utf-8"), salt, rounds)
    return hmac.compare_digest(dk, expected)


def needs_rehash(stored: str, *, rounds: int = DEFAULT_ROUNDS) -> bool:
    """Determine whether the hash should be regenerated with the current parameters (e.g. after raising the iteration count)."""
    try:
        alg, rounds_s, _, _ = stored.split("$", 3)
    except ValueError:
        return True
    return alg != f"pbkdf2_{ALG}" or int(rounds_s) != rounds


def is_plaintext(stored: str) -> bool:
    """Determine whether the stored value is a legacy plaintext (unhashed)."""
    return bool(stored) and not stored.startswith(f"pbkdf2_{ALG}")


@contextmanager
def _cross_process_lock(path: str):
    if FileLock is not None:
        with FileLock(path + ".lock"):
            yield
    else:
        yield


def write_yaml_atomic(path: str, data: dict) -> None:
    """Atomic YAML write: flush a temp file, commit via os.replace, plus filelock against concurrency races."""
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    with _WRITE_LOCK:
        with _cross_process_lock(path):
            fd, tmp = tempfile.mkstemp(dir=directory, suffix=".yaml.tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    yaml.dump(
                        data, f,
                        allow_unicode=True,
                        default_flow_style=False,
                        sort_keys=False,
                    )
                os.replace(tmp, path)
            except BaseException:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise
