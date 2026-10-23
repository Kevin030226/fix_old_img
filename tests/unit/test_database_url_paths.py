"""Database URL versus local data directory (plan 4.3 Step 2).

Three values describe "where the data is":

* `FIXIMG_DATABASE_URL` -- the deployment statement, surfaced as `settings.database_url`;
* `engine.DB_PATH` -- the SQLite file the engine opens;
* `settings.db_path` -- the anchor every other path consumer derives from.

They used to be three independent declarations. `apply_database_url()` moved only
the engine copy, so the API token and the initial admin password were written
beside the *default* database while the engine served a different one -- a
silent split that only shows up after a container recreate, when the credentials
are gone and the database is not.

These tests pin the two halves of the fix: a SQLite URL moves all three values
together, and a PostgreSQL URL moves none of them (a server database has no
filesystem path, so it does not get to decide where a local file lives).
"""
import os

import pytest

import fiximg.config as config_mod
import fiximg.infrastructure.db.engine as engine
from fiximg.api import security
from fiximg.application import bootstrap


@pytest.fixture()
def restore_engine():
    """Put the process-wide globals back after a test moved them.

    `DB_PATH`, `ADMIN_DATA_DIR` and `settings.db_path` are process state; a test
    that relocates them without restoring would leave every later test writing
    to a tmp directory the fixture has already deleted.
    """
    saved = (engine.DB_PATH, engine.ADMIN_DATA_DIR, config_mod.settings.db_path)
    engine.close_connections()
    try:
        yield
    finally:
        engine.DB_PATH, engine.ADMIN_DATA_DIR, config_mod.settings.db_path = saved
        engine.close_connections()
        security.reset_cache()


def test_configured_url_is_the_settings_field(monkeypatch):
    """`settings.database_url` is the one place the engine reads the URL from.

    The field used to be accepted by `Settings` and read by nothing: `engine`
    called `os.environ` directly, so assigning the attribute did nothing and a
    profile that declared the key looked effective while being ignored.
    """
    monkeypatch.delenv("FIXIMG_DATABASE_URL", raising=False)
    monkeypatch.setattr(config_mod.settings, "database_url", "sqlite:////tmp/from-attr.db")
    assert engine.configured_database_url() == "sqlite:////tmp/from-attr.db"


def test_environment_url_reaches_the_settings_field(monkeypatch):
    """The environment variable is what the field is built from."""
    monkeypatch.setenv("FIXIMG_DATABASE_URL", "postgresql://user@host/db")
    monkeypatch.setattr(config_mod.settings, "database_url", "")
    assert config_mod.Settings().database_url == "postgresql://user@host/db"
    assert engine.configured_database_url() == "postgresql://user@host/db"


def test_a_sqlite_url_moves_the_database_and_its_management_directory(
    tmp_path, restore_engine, monkeypatch
):
    """One assignment, three values: engine path, settings path, token directory."""
    monkeypatch.delenv("FIXIMG_DATABASE_URL", raising=False)
    target = tmp_path / "volume" / "fiximg.db"
    engine.apply_database_url("sqlite:///" + target.as_posix())

    expected = os.path.normpath(str(target))
    expected_dir = os.path.dirname(expected)
    assert os.path.normpath(engine.DB_PATH) == expected
    assert os.path.normpath(config_mod.settings.db_path) == expected, (
        "settings.db_path still names the old database, so every path derived "
        "from it (API token, initial admin password) lands in the wrong directory"
    )
    assert os.path.normpath(config_mod.settings.management_data_dir) == expected_dir
    assert os.path.normpath(security.token_file()) == os.path.join(
        expected_dir, "api_token.txt"
    )
    assert os.path.normpath(bootstrap._password_file()) == os.path.join(
        expected_dir, "initial_admin_password.txt"
    )


def test_a_postgres_url_leaves_the_local_directory_alone(tmp_path, restore_engine):
    """A server database has no filesystem path, so it relocates nothing.

    The management files stay where they are: turning "postgresql://host/db"
    into a local directory is the mistake this test exists to forbid.
    """
    before_engine = engine.DB_PATH
    before_settings = config_mod.settings.db_path
    before_token = security.token_file()

    resolved = engine.apply_database_url("postgresql://user:pw@db:5432/fiximg")

    assert resolved == "postgresql://user:pw@db:5432/fiximg"
    assert engine.DB_PATH == before_engine
    assert config_mod.settings.db_path == before_settings
    assert security.token_file() == before_token
    assert "postgresql" not in security.token_file()


def test_no_url_at_all_keeps_the_default_file(restore_engine, monkeypatch):
    """The empty default is what makes a checkout run without configuration."""
    monkeypatch.delenv("FIXIMG_DATABASE_URL", raising=False)
    monkeypatch.setattr(config_mod.settings, "database_url", "")
    before = engine.DB_PATH
    assert engine.apply_database_url() == before
    assert config_mod.settings.db_path == before


def test_an_embedder_assignment_survives_a_no_op_application(restore_engine, monkeypatch):
    """A test or embedder that sets `DB_PATH` directly must not be overridden.

    This is the inversion that once made ~176 tests share one developer
    database: a URL derived from the profile outranked the fixture.
    """
    monkeypatch.delenv("FIXIMG_DATABASE_URL", raising=False)
    monkeypatch.setattr(config_mod.settings, "database_url", "")
    engine.DB_PATH = "sentinel.db"
    assert engine.apply_database_url() == "sentinel.db"


def test_repeated_application_is_idempotent(tmp_path, restore_engine, monkeypatch):
    """Startup runs this once per process, and the test suite runs it many times."""
    monkeypatch.delenv("FIXIMG_DATABASE_URL", raising=False)
    target = tmp_path / "twice.db"
    url = "sqlite:///" + target.as_posix()
    first = engine.apply_database_url(url)
    second = engine.apply_database_url(url)
    assert first == second
    assert os.path.normpath(engine.DB_PATH) == os.path.normpath(second)
    assert os.path.normpath(config_mod.settings.db_path) == os.path.normpath(second)


def test_an_unsupported_scheme_fails_before_anything_moves(restore_engine):
    """Fail-fast means "nothing happened", not "half applied"."""
    before_path = engine.DB_PATH
    before_settings = config_mod.settings.db_path
    with pytest.raises(engine.UnsupportedDatabaseError):
        engine.apply_database_url("mysql://root@localhost/fiximg")
    assert engine.DB_PATH == before_path
    assert config_mod.settings.db_path == before_settings
