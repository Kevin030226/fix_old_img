"""The API token and the initial admin password follow the database.

Both files are written next to `settings.db_path`, which is what an operator
reads as "beside the database". They used to be computed from that attribute
while `apply_database_url()` updated only the engine copy -- so a deployment
that moved its SQLite file kept writing credentials beside the *default* one,
and a container recreate deleted them while the database survived.

`test_database_url_paths.py` pins the path arithmetic. This file proves the two
writers actually use it, by moving the database and then making each writer
produce its file.
"""
import os

import pytest

import fiximg.config as config_mod
import fiximg.infrastructure.db.engine as engine
from fiximg.api import security


@pytest.fixture()
def relocated(tmp_path, monkeypatch):
    """Move the database to a volume-shaped directory and restore it afterwards."""
    saved = (engine.DB_PATH, engine.ADMIN_DATA_DIR, config_mod.settings.db_path)
    monkeypatch.setattr(config_mod.settings, "app_env", "local", raising=False)
    monkeypatch.delenv("FIXIMG_API_TOKEN", raising=False)
    volume = tmp_path / "volume"
    volume.mkdir()
    engine.close_connections()
    engine.apply_database_url("sqlite:///" + (volume / "fiximg.db").as_posix())
    security.reset_cache()
    try:
        yield volume
    finally:
        security.reset_cache()
        engine.DB_PATH, engine.ADMIN_DATA_DIR, config_mod.settings.db_path = saved
        engine.close_connections()


def test_the_token_file_follows_the_relocated_database(relocated):
    path = security.token_file()
    assert os.path.dirname(os.path.normpath(path)) == os.path.normpath(str(relocated))
    assert os.path.exists(path) is False, "resolving the path must not create it yet"


def test_the_generated_token_lands_beside_the_database(relocated):
    token = security.get_api_token()
    expected = relocated / "api_token.txt"
    assert expected.exists(), (
        f"token was not written next to the database: {security.token_file()}"
    )
    assert expected.read_text(encoding="utf-8").strip() == token


def test_the_admin_password_file_follows_the_relocated_database(relocated):
    from fiximg.application import bootstrap

    path = bootstrap._password_file()
    assert os.path.dirname(os.path.normpath(path)) == os.path.normpath(str(relocated))


def test_a_postgres_deployment_keeps_the_management_files_where_they_were(tmp_path):
    """A server database must not turn its URL into a directory name."""
    import fiximg.infrastructure.db.engine as engine_mod

    before_token = security.token_file()
    before_password = None
    from fiximg.application import bootstrap

    before_password = bootstrap._password_file()
    saved = (engine_mod.DB_PATH, engine_mod.ADMIN_DATA_DIR, config_mod.settings.db_path)
    try:
        engine_mod.apply_database_url("postgresql://user:pw@db:5432/fiximg")
        assert security.token_file() == before_token
        assert bootstrap._password_file() == before_password
        assert "postgresql" not in security.token_file()
    finally:
        engine_mod.DB_PATH, engine_mod.ADMIN_DATA_DIR, config_mod.settings.db_path = saved
