"""User repository (plan section 14).

Delegates to app.db (V1 data layer kept intact so the V1 migration logic keeps
working); the repository is the only module the service/web layers may use.
"""
from app import db as _db


def get_user(username: str):
    return _db.get_user(username)


def list_users() -> list:
    return _db.list_users()


def add_user(username: str, password_hash: str, role: str = "user") -> bool:
    return _db.add_user(username, password_hash, role)


def update_user(username: str, password_hash=None, role=None) -> None:
    _db.update_user(username, password_hash=password_hash, role=role)


def set_must_change_password(username: str, flag: bool) -> None:
    """Plan section 21: force (or clear) the change-password requirement."""
    _db.set_must_change_password(username, flag)


def delete_user(username: str) -> None:
    _db.delete_user(username)
