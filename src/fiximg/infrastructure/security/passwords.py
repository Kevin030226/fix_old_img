"""Password hashing and verification.

Design notes:
- Hashing uses stdlib ``hashlib.pbkdf2_hmac`` (no third-party deps); the format
  is self-describing — ``pbkdf2_<alg>$<rounds>$<salt_hex>$<hash_hex>`` — so
  raising :data:`DEFAULT_ROUNDS` does not invalidate existing hashes, it only
  makes :func:`needs_rehash` report them for upgrade on next login.
- Verification uses ``hmac.compare_digest`` for constant-time comparison,
  resisting timing side channels.
- Nothing here writes a file. The users table is the only store, so an earlier
  YAML writer (and its ``filelock``/temp-file machinery) was removed: it had no
  caller anywhere in the tree, and its docstring described a torn-file risk that
  no code path could produce.
"""
import hashlib
import hmac
import secrets

ALG = "sha256"
DEFAULT_ROUNDS = 260_000  # aligned with config/users.example.yaml


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
    """True when ``stored`` was made with weaker parameters than the current ones.

    Login calls this after a successful verification and re-hashes with the
    current :data:`DEFAULT_ROUNDS`, so raising the iteration count actually
    reaches the accounts that are still on the old value. It used to be defined
    here and called from nowhere, which made "raise DEFAULT_ROUNDS" a comment
    rather than an operation.
    """
    try:
        alg, rounds_s, _, _ = stored.split("$", 3)
    except ValueError:
        return True
    return alg != f"pbkdf2_{ALG}" or int(rounds_s) != rounds


def upgrade_hash(stored: str, password: str) -> str | None:
    """Re-hash ``password`` with the current parameters, or ``None`` if it is current.

    Separate from :func:`needs_rehash` so the caller states the intent once and
    the login path does not have to remember which of the two to ask.
    """
    return None if not needs_rehash(stored) else hash_password(password)
