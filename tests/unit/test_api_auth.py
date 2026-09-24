"""Regression tests for the JSON API bearer-token gate (app/api/security.py).

Context: the Gradio UI talks to the service layer in process, but `/api/v1/*`
is plain HTTP. Before this gate every JSON route — including the account list
and finished result files — was reachable without credentials whenever the
service was bound to a routable address.
"""
import asyncio
import os

import pytest
from fastapi import HTTPException

from app.api import security


@pytest.fixture()
def isolated_token(tmp_path, monkeypatch):
    """Point the token file at a temp dir and clear env/cache state."""
    monkeypatch.setattr(
        security.settings, "db_path", str(tmp_path / "admin_data" / "fixoldimg.db")
    )
    monkeypatch.delenv("FIXIMG_API_TOKEN", raising=False)
    security.reset_cache()
    yield tmp_path
    security.reset_cache()


def _call(authorization):
    return asyncio.run(security.require_api_token(authorization))


def test_token_is_generated_persisted_and_stable(isolated_token):
    token = security.get_api_token()
    assert len(token) == 32
    path = security.token_file()
    assert os.path.exists(path)
    assert open(path, encoding="utf-8").read().strip() == token

    # Survives a restart (cache drop) by reading the persisted file.
    security.reset_cache()
    assert security.get_api_token() == token


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes are not enforced on Windows")
def test_token_file_is_owner_only(isolated_token):
    security.get_api_token()
    mode = os.stat(security.token_file()).st_mode & 0o777
    assert mode == 0o600


def test_env_token_takes_precedence_and_is_not_written(isolated_token, monkeypatch):
    monkeypatch.setenv("FIXIMG_API_TOKEN", "deployment-managed-token")
    security.reset_cache()
    assert security.get_api_token() == "deployment-managed-token"
    assert not os.path.exists(security.token_file())


def test_missing_token_is_rejected(isolated_token):
    with pytest.raises(HTTPException) as excinfo:
        _call(None)
    assert excinfo.value.status_code == 401


def test_wrong_token_is_rejected(isolated_token):
    security.get_api_token()
    with pytest.raises(HTTPException) as excinfo:
        _call("Bearer not-the-token")
    assert excinfo.value.status_code == 401


def test_non_bearer_scheme_is_rejected(isolated_token):
    token = security.get_api_token()
    with pytest.raises(HTTPException) as excinfo:
        _call(f"Basic {token}")
    assert excinfo.value.status_code == 401


def test_correct_token_is_accepted(isolated_token):
    token = security.get_api_token()
    assert _call(f"Bearer {token}") == token
