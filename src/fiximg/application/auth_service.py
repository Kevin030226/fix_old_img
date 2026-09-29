"""User service: registration / role logic over the user repository (plan section 27)."""
from fiximg.infrastructure.db.repositories import user_repository as user_repo
from fiximg.infrastructure.observability.logging import get_logger
from fiximg.infrastructure.security.passwords import hash_password

logger = get_logger("fiximg.auth")


def authenticate(username: str, password: str) -> bool:
    """Constant-time login verification against the users table.

    A successful login also upgrades the stored hash when it was made with
    weaker parameters, so raising ``passwords.DEFAULT_ROUNDS`` reaches existing
    accounts on their next sign-in instead of only new ones.
    """
    user = user_repo.get_user(username)
    if not user:
        return False
    from fiximg.infrastructure.security.passwords import upgrade_hash, verify_password

    stored = user.get("password", "")
    if not verify_password(password, stored):
        return False
    upgraded = upgrade_hash(stored, password)
    if upgraded is not None:
        try:
            user_repo.update_user(username, password_hash=upgraded)
        except Exception:  # noqa: BLE001 - a rehash must never fail a valid login
            logger.warning("could not upgrade the stored hash for %s", username, exc_info=True)
    return True


def register_user(username: str, password: str) -> bool:
    """Create a normal user; returns False when the username is taken."""
    return user_repo.add_user(username, hash_password(password), "user")


def create_user(username: str, password: str, role: str = "user") -> bool:
    return user_repo.add_user(username, hash_password(password), role)


def update_user(username: str, password: str | None = None, role: str | None = None) -> None:
    user_repo.update_user(
        username,
        password_hash=hash_password(password) if password else None,
        role=role,
    )


def set_must_change_password(username: str, flag: bool) -> None:
    """Plan section 21: force (or clear) the change-password requirement."""
    user_repo.set_must_change_password(username, flag)


def delete_user(username: str) -> None:
    user_repo.delete_user(username)


def get_user(username: str):
    return user_repo.get_user(username)


def list_users():
    return user_repo.list_users()
