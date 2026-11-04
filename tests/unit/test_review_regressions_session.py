"""Regression tests for the v3.1.2 review findings (P0-3 and P1 session items).

P0-3  ui/web/session.py — `refresh_role` deleted the record of a deleted account
      and then returned the (still admin) session, so the rest of that request
      ran as a user that no longer exists.
P1    ui/web/session.py — `create(role=...)` let a caller assert a role the users
      table does not grant, which is the authorisation invariant the module's
      own docstring says it exists to protect.
"""
from __future__ import annotations

import pytest

from fiximg.ui.state import ROLE_ADMIN, ROLE_USER
from fiximg.ui.web import session as session_mod


@pytest.fixture()
def users(monkeypatch):
    """An in-memory users table, so no database is involved."""
    table: dict[str, dict] = {}

    def fake_get_user(name):
        return table.get(name)

    monkeypatch.setattr(session_mod, "get_user", fake_get_user)
    return table


def test_refresh_role_returns_none_for_a_deleted_account(users):
    """The reported defect: the record is dropped but the session is returned."""
    users["alice"] = {"username": "alice", "role": ROLE_ADMIN}
    store = session_mod.SessionStore()
    session = store.create("alice")
    assert session.is_admin

    del users["alice"]
    refreshed = store.refresh_role(session)

    assert refreshed is None, "a deleted account must resolve to nobody"
    assert store.get(session.session_id) is None, "the record must be gone"


def test_a_deleted_account_cannot_resolve_a_session(users):
    """The end-to-end consequence: `resolve_session` must not hand out a session."""
    users["alice"] = {"username": "alice", "role": ROLE_ADMIN}
    store = session_mod.SessionStore()
    session = store.create("alice")
    cookie = session_mod.sign_session_id(session.session_id)

    del users["alice"]
    original = session_mod.session_store
    session_mod.session_store = store
    try:
        assert session_mod.resolve_session(cookie) is None
    finally:
        session_mod.session_store = original


def test_refresh_role_still_promotes_and_demotes(users):
    """The half that must keep working: a role change takes effect immediately."""
    users["bob"] = {"username": "bob", "role": ROLE_USER}
    store = session_mod.SessionStore()
    session = store.create("bob")
    assert not session.is_admin

    users["bob"]["role"] = ROLE_ADMIN
    promoted = store.refresh_role(session)
    assert promoted is not None and promoted.is_admin
    # A new record is stored, so the next request sees the new role too.
    assert store.get(session.session_id).is_admin

    users["bob"]["role"] = ROLE_USER
    demoted = store.refresh_role(promoted)
    assert demoted is not None and not demoted.is_admin


def test_create_cannot_be_handed_a_role(users):
    """`role` is not a parameter: the users table is the only source of truth.

    The module's docstring states the invariant ("a route that passed its own
    idea of the role could hand out an admin session for a demoted account").
    The signature contradicted it: any caller could pass `role="admin"` and
    skip the lookup entirely.
    """
    users["eve"] = {"username": "eve", "role": ROLE_USER}
    store = session_mod.SessionStore()

    with pytest.raises(TypeError):
        store.create("eve", "admin")          # noqa: F841 - the point of the test

    session = store.create("eve")
    assert session.role == ROLE_USER, "the role must come from the users table"


def test_create_defaults_to_user_for_an_unknown_account(users):
    """Fail closed: an account that is not in the table is not an administrator."""
    store = session_mod.SessionStore()
    assert store.create("ghost").role == ROLE_USER


def test_refresh_role_is_safe_when_the_store_record_is_gone(users):
    """A concurrent logout can delete the record between get() and refresh_role()."""
    users["carol"] = {"username": "carol", "role": ROLE_USER}
    store = session_mod.SessionStore()
    session = store.create("carol")
    store.delete(session.session_id)

    # The account is still valid, but the record is gone: returning the session
    # would resurrect a signed-out browser for the rest of the request.
    assert store.refresh_role(session) is None
