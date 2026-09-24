"""User service: registration / role logic over the user repository (plan section 27)."""
from app.core.security import hash_password
from app.repositories import user_repository as user_repo


def authenticate(username: str, password: str) -> bool:
    """Constant-time login verification against the users table."""
    user = user_repo.get_user(username)
    if not user:
        return False
    from app.core.security import verify_password

    return verify_password(password, user.get("password", ""))


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
