"""First-boot admin credential bootstrap (plan §21).

Replaces the old Dockerfile behaviour of baking a fixed admin/admin123 account
into the image. Policy:

  - FIXIMG_ADMIN_PASSWORD set  -> create 'admin' with exactly that password
    (one-time password injected by the deployment, e.g. a Docker secret).
  - not set                    -> generate a random 16-char password, create
    'admin' with it, print it ONCE to stdout, and write it to
    admin_data/initial_admin_password.txt (mode 0600). The operator must read
    it from the container log on first boot and change it afterwards.

Plan section 21 "force change": the randomly-generated account is created
with must_change_password=1 — the Gradio login then keeps prompting the user
until the password has been replaced (setting the password via
FIXIMG_ADMIN_PASSWORD marks the credential as deployment-managed and does not
force a change).

Idempotent: an existing 'admin' user is never overwritten, and the bootstrap
only fires when the users table is completely empty (fresh deployment).
"""
import os
import secrets
import stat

from app.core.config import settings
from app.core.logging import get_logger, log_event
from app.core.security import hash_password
from app.repositories import user_repository as user_repo

logger = get_logger("fiximg.bootstrap")

_PASSWORD_LENGTH = 16


def _admin_username() -> str:
    """Admin account name; FIXIMG_ADMIN_USERNAME overrides the default (§20)."""
    return settings.admin_username or "admin"


def _password_file() -> str:
    return os.path.join(os.path.dirname(settings.db_path), "initial_admin_password.txt")


def _generate_password(length: int = _PASSWORD_LENGTH) -> str:
    """URL-safe random password without ambiguous characters."""
    alphabet = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def _print_once(password: str) -> None:
    """Emit the one-time password on stdout (container logs) and persist it 0600."""
    print("=" * 64)
    print("[Bootstrap] First start: generated initial admin credentials")
    print(f"[Bootstrap]   username: {_admin_username()}")
    print(f"[Bootstrap]   password: {password}")
    print("[Bootstrap] Please log in and change this password immediately.")
    print("=" * 64)
    path = _password_file()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(password + "\n")
    except OSError:
        return
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def ensure_admin_user() -> dict | None:
    """Create the initial admin account when the deployment has no users yet.

    Returns a dict describing what happened:
      {"action": "created_env" | "created_random", "username": ...}  — account created
      {"action": "exists"}                                           — users already present
      {"action": "skipped", "reason": ...}                           — creation refused
    """
    try:
        existing = user_repo.list_users()
    except Exception as exc:  # noqa: BLE001 — DB not ready; never block startup
        log_event(logger, "ERROR", "admin bootstrap skipped: db unavailable", error=str(exc))
        return {"action": "skipped", "reason": f"db unavailable: {exc}"}

    if existing:
        return {"action": "exists"}

    env_password = os.environ.get("FIXIMG_ADMIN_PASSWORD", "").strip()
    if env_password:
        if len(env_password) < 8:
            return {
                "action": "skipped",
                "reason": "FIXIMG_ADMIN_PASSWORD too short (minimum 8 characters); "
                "no admin account created",
            }
        user_repo.add_user(_admin_username(), hash_password(env_password), "admin")
        log_event(logger, "INFO", "admin bootstrap: created from FIXIMG_ADMIN_PASSWORD")
        return {"action": "created_env", "username": _admin_username()}

    password = _generate_password()
    admin = _admin_username()
    user_repo.add_user(admin, hash_password(password), "admin")
    # §21 force-change: the random one-time password must be replaced before
    # the account is used for anything beyond the password change.
    user_repo.set_must_change_password(admin, True)
    _print_once(password)
    log_event(logger, "INFO", "admin bootstrap: created with random one-time password")
    return {"action": "created_random", "username": admin}
