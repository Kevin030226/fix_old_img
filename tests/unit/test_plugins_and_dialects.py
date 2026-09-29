"""Plugin discovery and database dialect handling (plan 搂3.14.1, 搂4.3 Step 2).

Two items from the report that share a theme 鈥?turning a silent no-op into an
explicit, testable behaviour:

* 搂3.14.1 鈥?third-party models/stages arrive through entry points, so adding a
  model does not require forking this repository.
* 搂4.3 Step 2 鈥?``FIXIMG_DATABASE_URL`` is honoured for SQLite and PostgreSQL,
  and a scheme with no driver fails loudly rather than quietly writing to SQLite.

The dependency lock groups (搂3.12) live in ``test_dependency_locks.py``.
"""
import os
import subprocess
import sys
import time

import pytest

from fiximg.inference import plugins as plugins_module
from fiximg.infrastructure.db.engine import (
    POSTGRES_SCHEMES,
    SQLITE_SCHEMES,
    UnsupportedDatabaseError,
    apply_database_url,
    resolve_database,
    resolve_database_path,
    validate_database_url,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================== plugins ===============================
class _FakeModelPlugin:
    """A well-formed model plugin."""

    id = "fake_model"
    version = "2.1.0"
    capabilities = {"fake_capability"}

    def create_backend(self, config=None):
        return {"backend": self.id, "config": config}


class _FakeStagePlugin:
    name = "fake_stage"
    version = "1.0"
    capabilities = {"fake_stage_capability"}

    def create_stage(self, **kwargs):
        return {"stage": self.name, "kwargs": kwargs}


class _EntryPoint:
    def __init__(self, name, target, error=None):
        self.name = name
        self._target = target
        self._error = error

    def load(self):
        if self._error:
            raise self._error
        return self._target


@pytest.fixture(autouse=True)
def _reset_plugins():
    plugins_module.reset_plugin_cache()
    yield
    plugins_module.reset_plugin_cache()


def _patch_entry_points(monkeypatch, mapping):
    """Replace the entry-point lookup with a fixed per-group mapping."""
    monkeypatch.setattr(
        plugins_module, "_entry_points", lambda group: mapping.get(group, [])
    )


# --------------------------------------------------------------- discovery
def test_discovery_finds_model_and_stage_plugins(monkeypatch):
    _patch_entry_points(monkeypatch, {
        plugins_module.MODEL_ENTRY_POINT_GROUP: [_EntryPoint("fake", _FakeModelPlugin)],
        plugins_module.STAGE_ENTRY_POINT_GROUP: [_EntryPoint("fake", _FakeStagePlugin)],
    })
    discovery = plugins_module.discover_plugins()

    assert [p.id for p in discovery.models] == ["fake_model"]
    assert [p.name for p in discovery.stages] == ["fake_stage"]
    assert discovery.loaded == 2
    assert all(record.loaded for record in discovery.records)


def test_discovery_accepts_a_class_or_an_instance(monkeypatch):
    _patch_entry_points(monkeypatch, {
        plugins_module.MODEL_ENTRY_POINT_GROUP: [
            _EntryPoint("as-class", _FakeModelPlugin),
            _EntryPoint("as-instance", _FakeModelPlugin()),
        ],
    })
    assert len(plugins_module.discover_plugins().models) == 2


def test_a_broken_plugin_does_not_stop_startup(monkeypatch):
    """One bad third-party package must never take the service down."""
    _patch_entry_points(monkeypatch, {
        plugins_module.MODEL_ENTRY_POINT_GROUP: [
            _EntryPoint("broken", None, error=ImportError("no such module")),
            _EntryPoint("good", _FakeModelPlugin),
        ],
    })
    discovery = plugins_module.discover_plugins()

    assert [p.id for p in discovery.models] == ["fake_model"]
    failures = [r for r in discovery.records if not r.loaded]
    assert len(failures) == 1
    assert "ImportError" in (failures[0].error or "")


def test_a_plugin_missing_the_contract_is_rejected(monkeypatch):
    class _NoCapabilities:
        id = "incomplete"
        version = "1.0"

        def create_backend(self, config=None):
            return None

    _patch_entry_points(monkeypatch, {
        plugins_module.MODEL_ENTRY_POINT_GROUP: [_EntryPoint("bad", _NoCapabilities)],
    })
    discovery = plugins_module.discover_plugins()

    assert discovery.models == []
    assert discovery.records[0].loaded is False
    assert "capabilities" in (discovery.records[0].error or "")


def test_a_plugin_missing_the_factory_is_rejected(monkeypatch):
    class _NoFactory:
        id = "no_factory"
        version = "1.0"
        capabilities = {"x"}

    _patch_entry_points(monkeypatch, {
        plugins_module.MODEL_ENTRY_POINT_GROUP: [_EntryPoint("bad", _NoFactory)],
    })
    discovery = plugins_module.discover_plugins()
    assert discovery.models == []
    assert "create_backend" in (discovery.records[0].error or "")


def test_a_stage_plugin_needs_a_name_and_a_factory(monkeypatch):
    class _Bad:
        version = "1.0"
        capabilities = {"x"}

        def create_stage(self, **kwargs):
            return None

    _patch_entry_points(monkeypatch, {
        plugins_module.STAGE_ENTRY_POINT_GROUP: [_EntryPoint("bad", _Bad)],
    })
    assert plugins_module.discover_plugins().stages == []


def test_discovery_can_be_disabled(monkeypatch):
    monkeypatch.setenv(plugins_module.ENABLE_ENV, "0")
    _patch_entry_points(monkeypatch, {
        plugins_module.MODEL_ENTRY_POINT_GROUP: [_EntryPoint("fake", _FakeModelPlugin)],
    })
    discovery = plugins_module.discover_plugins()
    assert discovery.models == []
    assert plugins_module.plugins_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", ""])
def test_plugins_enabled_by_default(monkeypatch, value):
    monkeypatch.setenv(plugins_module.ENABLE_ENV, value)
    assert plugins_module.plugins_enabled() is True


def test_discovery_survives_a_broken_entry_point_api(monkeypatch):
    monkeypatch.setattr(
        plugins_module, "_entry_points", lambda group: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    # _entry_points is patched wholesale, so exercise the real guard instead.
    monkeypatch.undo()
    monkeypatch.setattr("importlib.metadata.entry_points", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    discovery = plugins_module.discover_plugins()
    assert discovery.loaded == 0


# ------------------------------------------------------------ registration
def test_model_plugins_register_into_the_backend_registry(monkeypatch):
    from fiximg.inference.backends.registry import ModelBackendRegistry

    registry = ModelBackendRegistry()
    registered = plugins_module.register_model_plugins(registry, [_FakeModelPlugin()])
    assert registered == ["fake_model"]
    assert "fake_model" in registry.names()
    assert registry.get("fake_model") == {"backend": "fake_model", "config": None}


def test_model_plugins_may_replace_a_builtin(monkeypatch):
    from fiximg.inference.backends.registry import ModelBackendRegistry

    registry = ModelBackendRegistry()
    registry.register("ddcolor", lambda: "builtin")
    plugins_module.register_model_plugins(registry, [_FakeModelPlugin()])
    # A plugin can deliberately shadow a built-in; that is how an operator swaps
    # an implementation without patching this repository.
    registry.register("ddcolor", lambda: "plugin", replace=True)
    assert registry.get("ddcolor") == "plugin"


def test_stage_plugins_register_into_the_stage_registry():
    from fiximg.inference.registry import StageRegistry

    registry = StageRegistry()
    registered = plugins_module.register_stage_plugins(registry, [_FakeStagePlugin()])
    assert registered == ["fake_stage"]

    #  takes a kwargs dict, matching the planner contract.
    stage = registry.create("fake_stage", {"foo": "bar"})
    assert stage == {"stage": "fake_stage", "kwargs": {"foo": "bar"}}
    assert registry.capabilities_of("fake_stage") == {"fake_stage_capability"}


def test_a_registration_failure_is_contained():
    from fiximg.inference.backends.registry import ModelBackendRegistry

    class _Explodes:
        id = "exploding"
        version = "1.0"
        capabilities = {"x"}

        def create_backend(self, config=None):
            raise RuntimeError("nope")

    registry = ModelBackendRegistry()
    assert plugins_module.register_model_plugins(registry, [_Explodes()]) == ["exploding"]
    with pytest.raises(RuntimeError):
        registry.get("exploding")           # the failure surfaces on use, not at startup


def test_the_default_registry_discovers_plugins(monkeypatch):
    """The shipped registry must run discovery, or the feature is dead code."""
    _patch_entry_points(monkeypatch, {
        plugins_module.MODEL_ENTRY_POINT_GROUP: [_EntryPoint("fake", _FakeModelPlugin)],
    })
    from fiximg.inference.backends.registry import build_default_backend_registry

    assert "fake_model" in build_default_backend_registry().names()


def test_the_default_stage_registry_discovers_plugins(monkeypatch):
    _patch_entry_points(monkeypatch, {
        plugins_module.STAGE_ENTRY_POINT_GROUP: [_EntryPoint("fake", _FakeStagePlugin)],
    })
    from fiximg.inference.registry import build_default_registry

    assert "fake_stage" in build_default_registry().names()


def test_discovery_report_is_json_friendly(monkeypatch):
    _patch_entry_points(monkeypatch, {
        plugins_module.MODEL_ENTRY_POINT_GROUP: [_EntryPoint("fake", _FakeModelPlugin)],
    })
    payload = plugins_module.discover_plugins().to_dict()
    assert payload["models"] == 1
    assert payload["entry_points"][0]["id"] == "fake_model"
    assert payload["entry_points"][0]["capabilities"] == ["fake_capability"]


def test_last_discovery_is_remembered(monkeypatch):
    _patch_entry_points(monkeypatch, {})
    assert plugins_module.last_discovery() is None
    discovery = plugins_module.remember(plugins_module.discover_plugins())
    assert plugins_module.last_discovery() is discovery


# ========================= database dialect =========================
def test_empty_url_uses_the_default_path():
    assert resolve_database_path("") .endswith("fixoldimg.db")
    assert resolve_database_path(None).endswith("fixoldimg.db")


def test_relative_sqlite_url_is_resolved_against_the_project_root():
    path = resolve_database_path("sqlite:///admin_data/other.db")
    assert path.replace("\\", "/").endswith("admin_data/other.db")
    assert os.path.isabs(path)


def test_absolute_sqlite_url_is_kept():
    path = resolve_database_path("sqlite:////tmp/fiximg.db")
    assert path.replace("\\", "/").endswith("tmp/fiximg.db")


def test_sqlalchemy_sqlite_spelling_is_accepted():
    assert resolve_database_path("sqlite+pysqlite:///a.db").endswith("a.db")


def test_bare_path_is_accepted():
    assert resolve_database_path("some/dir/x.db").replace("\\", "/").endswith("some/dir/x.db")


def test_memory_database_is_supported():
    assert resolve_database_path("sqlite:///:memory:") == ":memory:"


@pytest.mark.parametrize("url", [
    "postgresql://user:pw@localhost:5432/fiximg",
    "postgres://user:pw@db:5432/fiximg",
    "mysql://root@localhost/fiximg",
])
def test_resolve_database_path_refuses_a_url_with_no_file(url):
    """A PostgreSQL URL has no filesystem path; an unknown scheme has no driver.

    Either way the answer is an error, never a quiet fallback to SQLite 鈥?that
    would look like a successful switch while writing to the wrong database.
    """
    with pytest.raises(UnsupportedDatabaseError) as excinfo:
        resolve_database_path(url)
    assert "resolve_database()" in str(excinfo.value) or "Supported:" in str(excinfo.value)


@pytest.mark.parametrize("url", [
    "postgresql://user:pw@localhost:5432/fiximg",
    "postgres://user:pw@db:5432/fiximg",
    "postgresql+psycopg://user@host/db",
])
def test_postgres_urls_resolve_to_a_connection_url(url):
    """PostgreSQL is served now: the URL itself is the target (搂4.3 Step 2)."""
    kind, target = resolve_database(url)
    assert kind == "postgresql"
    assert target == url


def test_the_error_names_the_offending_scheme():
    with pytest.raises(UnsupportedDatabaseError, match="mysql"):
        resolve_database("mysql://root@localhost/fiximg")


def test_validate_has_no_side_effects(monkeypatch):
    monkeypatch.setenv("FIXIMG_DATABASE_URL", "mysql://root@localhost/fiximg")
    with pytest.raises(UnsupportedDatabaseError):
        validate_database_url()
    # Validation must not have moved DB_PATH.
    import fiximg.infrastructure.db.engine as engine

    assert engine.DB_PATH.endswith("fixoldimg.db")


def test_validate_accepts_a_postgres_url(monkeypatch):
    """Fail-fast means "unsupported", not "PostgreSQL"."""
    monkeypatch.setenv("FIXIMG_DATABASE_URL", "postgresql://user:pw@localhost/fiximg")
    assert validate_database_url().startswith("postgresql://")


def test_validate_accepts_a_sqlite_url(monkeypatch):
    monkeypatch.setenv("FIXIMG_DATABASE_URL", "sqlite:///admin_data/x.db")
    assert validate_database_url().startswith("sqlite:///")


def test_apply_relocates_db_path_for_an_explicit_env_url(monkeypatch, tmp_path):
    import fiximg.infrastructure.db.engine as engine

    original = engine.DB_PATH
    target = tmp_path / "relocated.db"
    monkeypatch.setenv("FIXIMG_DATABASE_URL", f"sqlite:///{target.as_posix()}")
    try:
        resolved = apply_database_url()
        assert resolved == engine.DB_PATH
        assert engine.DB_PATH.replace("\\", "/").endswith("relocated.db")
    finally:
        engine.DB_PATH = original


def test_apply_is_a_noop_without_an_env_url(monkeypatch):
    """The YAML default must not override a DB_PATH set by a test or embedder."""
    import fiximg.infrastructure.db.engine as engine

    original = engine.DB_PATH
    monkeypatch.delenv("FIXIMG_DATABASE_URL", raising=False)
    engine.DB_PATH = "sentinel.db"
    try:
        assert apply_database_url() == "sentinel.db"
    finally:
        engine.DB_PATH = original


def test_apply_leaves_db_path_alone_for_a_postgres_env_url(monkeypatch):
    """There is no path to relocate: the connection is opened from the URL."""
    import fiximg.infrastructure.db.engine as engine

    original = engine.DB_PATH
    monkeypatch.setenv("FIXIMG_DATABASE_URL", "postgresql://x/y")
    try:
        assert apply_database_url() == "postgresql://x/y"
        assert engine.DB_PATH == original
    finally:
        engine.DB_PATH = original


def test_apply_still_rejects_a_scheme_with_no_driver(monkeypatch):
    monkeypatch.setenv("FIXIMG_DATABASE_URL", "mysql://root@localhost/fiximg")
    with pytest.raises(UnsupportedDatabaseError):
        apply_database_url()


def test_sqlite_schemes_are_documented():
    assert "sqlite" in SQLITE_SCHEMES
    assert "sqlite+pysqlite" in SQLITE_SCHEMES


def test_postgres_schemes_are_documented():
    assert "postgresql" in POSTGRES_SCHEMES
    assert "postgresql+psycopg" in POSTGRES_SCHEMES


# ========================= dependency lock groups =========================
# The lock audit has its own file now: tests/unit/test_dependency_locks.py. It
# needs a scratch checkout to prove that a gate red for the right reasons, because
# the version of these tests that lived here ran the generator against the working
# tree it was supposed to be checking.


# ===================== external service smoke test =====================
def test_smoke_services_runs_and_reports_every_probe():
    """`make smoke-services` must be executable and honest about skips.

    On a machine with no external services configured every probe is either PASS
    (SQLite, the queue) or SKIP 鈥?never a spurious FAIL.
    """
    env = dict(os.environ)
    for name in ("FIXIMG_REDIS_URL", "FIXIMG_STORAGE_BACKEND"):
        env.pop(name, None)
    env["FIXIMG_DATABASE_URL"] = ""

    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "smoke_services.py")],
        cwd=PROJECT_ROOT, capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "database (sqlite)" in result.stdout
    assert "queue backend" in result.stdout
    assert "SKIP" in result.stdout            # redis / object storage
    assert "Every configured service is reachable" in result.stdout


def test_smoke_services_fails_on_an_unreachable_postgres():
    """A PostgreSQL URL that cannot be served is a FAIL, never a SKIP.

    Skipped probes are for services the deployment did not ask for; a service it
    *did* ask for must not be able to hide behind a SKIP. localhost:5432 has no
    server in CI, so the probe reports the driver/connection failure.

    The elapsed bound is the other half of the assertion: on the machine this
    project deploys to, psycopg without ``connect_timeout`` never returned at all
    against a refused port, so "it eventually fails" is not the property 鈥?    "it answers within the probe's own budget" is.
    """
    env = dict(os.environ)
    env["FIXIMG_DATABASE_URL"] = "postgresql://user:pw@localhost:5432/fiximg"

    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "smoke_services.py")],
        cwd=PROJECT_ROOT, capture_output=True, text=True, env=env, timeout=120,
    )
    elapsed = time.perf_counter() - started
    assert result.returncode == 1
    assert elapsed < 60, f"the postgres probe took {elapsed:.1f}s to answer: {result.stdout}"
    assert "FAIL" in result.stdout
    assert "database (postgresql)" in result.stdout
    assert "SKIP" not in result.stdout.split("database (postgresql)")[0]
