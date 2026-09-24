"""Unit tests for the first-boot admin bootstrap (plan §21)."""
import os

import pytest

from app.repositories import user_repository as user_repo
from app.services import bootstrap


@pytest.fixture()
def db_env(tmp_path, monkeypatch):
    """Isolated SQLite file for every test (same pattern as test_task_queue)."""
    import app.core.config as config_mod
    import app.db as legacy_db
    from app.repositories import task_repository as task_repo

    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(legacy_db, "DB_PATH", db_path)
    monkeypatch.setattr(legacy_db, "ADMIN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(legacy_db, "_conn", None)
    monkeypatch.setattr(config_mod.settings, "db_path", db_path)
    task_repo._DDL_DONE = False
    legacy_db.init_db()
    monkeypatch.delenv("FIXIMG_ADMIN_PASSWORD", raising=False)
    return tmp_path


def test_creates_random_password_when_env_unset(db_env, capsys):
    result = bootstrap.ensure_admin_user()
    assert result["action"] == "created_random"

    user = user_repo.get_user("admin")
    assert user and user["role"] == "admin"

    from app.core.security import verify_password

    # The printed password must actually authenticate.
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if "password:" in ln)
    printed = line.split("password:", 1)[1].strip()
    assert verify_password(printed, user["password"])

    # Written once to a file under admin_data (0600 on POSIX; Windows ignores
    # POSIX mode bits, so only assert the file exists there).
    pw_file = db_env / "initial_admin_password.txt"
    assert pw_file.exists()
    assert pw_file.read_text(encoding="utf-8").strip() == printed
    if os.name == "posix":
        assert not (os.stat(pw_file).st_mode & 0o077)


def test_uses_env_password_when_provided(db_env, monkeypatch, capsys):
    monkeypatch.setenv("FIXIMG_ADMIN_PASSWORD", "S3cure-Passw0rd!")
    result = bootstrap.ensure_admin_user()
    assert result["action"] == "created_env"

    from app.core.security import verify_password

    assert verify_password("S3cure-Passw0rd!", user_repo.get_user("admin")["password"])
    assert "password:" not in capsys.readouterr().out  # env path never prints


def test_rejects_short_env_password(db_env, monkeypatch):
    monkeypatch.setenv("FIXIMG_ADMIN_PASSWORD", "short")
    result = bootstrap.ensure_admin_user()
    assert result["action"] == "skipped"
    assert user_repo.get_user("admin") is None


def test_never_overwrites_existing_users(db_env, capsys):
    bootstrap.ensure_admin_user()
    capsys.readouterr()  # discard the legitimate first-boot banner
    assert bootstrap.ensure_admin_user()["action"] == "exists"

    from app.core.security import verify_password

    # Original random password still valid, exactly one admin user.
    assert verify_password(
        (db_env / "initial_admin_password.txt").read_text(encoding="utf-8").strip(),
        user_repo.get_user("admin")["password"],
    )
    assert len(user_repo.list_users()) == 1
    # The banner is printed exactly once — never on subsequent boots.
    assert "generated initial admin" not in capsys.readouterr().out


def test_disabled_via_env(monkeypatch):
    """FIXIMG_AUTO_BOOTSTRAP_ADMIN=false turns the guard off (factory checks it)."""
    import app.core.config as config_mod

    monkeypatch.setenv("FIXIMG_AUTO_BOOTSTRAP_ADMIN", "false")
    fresh = config_mod.Settings()
    assert fresh.auto_bootstrap_admin is False
    monkeypatch.setenv("FIXIMG_AUTO_BOOTSTRAP_ADMIN", "true")
    assert config_mod.Settings().auto_bootstrap_admin is True
