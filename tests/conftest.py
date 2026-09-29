"""pytest bootstrap and shared fixtures.

``sys.path`` setup: ``src`` makes the ``fiximg`` package importable without an
editable install; the repository root keeps the vendored model packages
(``Global``, ``ddcolor``, ``Face_Detection``, ``Face_Enhancement``, ``basicsr``)
importable as before.

``isolated_db`` points the application at a throwaway SQLite file so no test can
touch the developer's real ``admin_data/fixoldimg.db``. It used to be duplicated
in several test modules; V3 keeps one copy here.
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.join(_ROOT, "src"), _ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)


@pytest.fixture(autouse=True, scope="session")
def _no_deployment_database_url():
    """Strip `FIXIMG_DATABASE_URL` for the whole session, unless a test sets it itself.

    The engine reads that variable at call time (`apply_database_url()`), so an
    operator-style export in the shell outranks whatever a fixture assigns to
    ``DB_PATH``: every test that builds an app, runs the migration runner or boots
    the worker CLI would move the database back to one shared file, and the suite
    starts failing on state its neighbours wrote (seen as admin-bootstrap tests
    reporting "users already exist" on the deployment interpreter, green on a box
    where the variable happened to be unset).

    Isolation has to be the default, so this is autouse. A test that genuinely wants
    a configured URL 鈥?`test_the_worker_opens_the_configured_database` 鈥?sets it with
    ``monkeypatch.setenv``, which pytest restores after that one test.
    """
    os.environ.pop("FIXIMG_DATABASE_URL", None)
    yield
    os.environ.pop("FIXIMG_DATABASE_URL", None)


@pytest.fixture(autouse=True)
def _fresh_submission_limiter(monkeypatch):
    """Give every test its own submission allowance.

    The limiter is one process-wide sliding window keyed by principal, and the suite
    submits dozens of tasks per minute from the same account name. Left shared, an
    unrelated test would start failing with a real 429 depending on execution order 鈥?
    the same class of cross-test contamination as the deployment `FIXIMG_DATABASE_URL`
    above, and the reason this is autouse rather than opt-in.
    """
    from fiximg.infrastructure.security import rate_limit

    monkeypatch.setattr(
        rate_limit, "submit_limiter",
        rate_limit.SlidingWindowLimiter(rate_limit.SUBMIT_MAX, rate_limit.SUBMIT_WINDOW),
    )


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """Isolated SQLite file + admin_data dir for every test that needs storage."""
    import fiximg.config as config_mod
    import fiximg.infrastructure.db.engine as engine
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(engine, "DB_PATH", db_path)
    monkeypatch.setattr(engine, "ADMIN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config_mod.settings, "db_path", db_path)
    # `FIXIMG_DATABASE_URL` is removed for the session by
    # `_no_deployment_database_url` above; that is what keeps this assignment honest.

    # Close whatever the previous test left open so get_conn() reopens here.
    # Directly assigning engine._conn would leak the old handle and, worse, let
    # a deployment-level FIXIMG_DATABASE_URL win over DB_PATH.
    engine.close_connections()
    task_repo._DDL_DONE = False
    engine.init_db()
    task_repo.ensure_schema()
    yield tmp_path

    try:
        engine.close_connections()
    except Exception:  # noqa: BLE001 鈥?teardown must never fail a test
        pass
    task_repo._DDL_DONE = False


@pytest.fixture()
def isolated_storage(tmp_path, monkeypatch):
    """Isolated artifact root so tests never write into the repo's storage/."""
    import fiximg.config as config_mod

    root = tmp_path / "storage" / "tasks"
    monkeypatch.setattr(config_mod.settings, "tasks_root", str(root))
    return root
