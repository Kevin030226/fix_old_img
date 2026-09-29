"""The `model_versions` table 鈥?plan 搂6's fifth data model (plan 搂3.15's audit trail).

Four of 搂6's five models were persisted; `ModelVersion` was a domain class parsed
out of the manifest, so `/models/{name}/versions` could only describe the process
answering the call. These tests cover the three halves that make it a deployment
fact: the upsert, the registry that writes it, and the endpoint that reads it back.
"""
from __future__ import annotations

import sqlite3

import pytest

from fiximg.domain.enums import ModelStatus
from fiximg.infrastructure.db.repositories import model_repository


# ------------------------------------------------------------------ the upsert
def test_one_version_is_one_row_however_often_it_reloads(isolated_db):
    """Append-per-load would turn a reload into two "available versions"."""
    model_repository.record_load("ddcolor", "1.2.0", status=ModelStatus.READY,
                                framework="pytorch", load_ms=812)
    model_repository.record_load("ddcolor", "1.2.0", status=ModelStatus.READY,
                                framework="pytorch", load_ms=410)

    rows = model_repository.list_versions("ddcolor")
    assert len(rows) == 1, rows
    assert rows[0]["load_count"] == 2, rows[0]
    assert rows[0]["load_ms"] == 410, "the observation is the latest one"


def test_a_writer_that_does_not_know_the_declared_fields_does_not_erase_them(isolated_db):
    """NULL from a reload means "this process could not read the manifest".

    Losing the `sha256` here would silently disable the 搂3.5.2 verification for
    every later reader of the row, which is the expensive direction for this to go
    wrong in.
    """
    model_repository.record_load("global", "2.0.0", status=ModelStatus.READY,
                                 framework="pytorch", weight_uri="/w/global/2.0.0",
                                 sha256="a" * 64, metadata={"repo": "damo"})
    model_repository.record_load("global", "2.0.0", status=ModelStatus.READY,
                                 load_ms=50)

    row = model_repository.list_versions("global")[0]
    assert row["framework"] == "pytorch"
    assert row["weight_uri"] == "/w/global/2.0.0"
    assert row["sha256"] == "a" * 64
    assert row["metadata"] == {"repo": "damo"}
    assert row["load_ms"] == 50


def test_a_failed_attempt_is_recorded_without_claiming_a_load_time(isolated_db):
    """`loaded_at` answers "when did it last come up", not "when did we try".

    The distinction is the point: a version that has never come up must not look
    like one that ran.
    """
    model_repository.record_load("broken", "9.9.9", status=ModelStatus.UNHEALTHY,
                                 error="FileNotFoundError: no weights")

    row = model_repository.list_versions("broken")[0]
    assert row["status"] == "unhealthy"
    assert row["loaded_at"] is None, row
    assert row["last_error"] == "FileNotFoundError: no weights"


def test_a_recovery_keeps_when_it_last_ran_and_clears_the_error(isolated_db):
    model_repository.record_load("broken", "1.0.0", status=ModelStatus.READY, load_ms=90)
    first = model_repository.list_versions("broken")[0]["loaded_at"]
    model_repository.record_load("broken", "1.0.0", status=ModelStatus.UNHEALTHY,
                                 error="CUDA out of memory")
    row = model_repository.list_versions("broken")[0]
    assert row["loaded_at"] == first, "the failure must not erase the last successful load"
    assert row["last_error"] == "CUDA out of memory"

    model_repository.record_load("broken", "1.0.0", status=ModelStatus.READY, load_ms=80)
    row = model_repository.list_versions("broken")[0]
    assert row["status"] == "ready"
    assert row["last_error"] is None, "a success clears the previous error"


def test_the_row_names_which_process_recorded_it(isolated_db, monkeypatch):
    """Without this, two workers writing the same row leave no trace of either."""
    monkeypatch.setenv("FIXIMG_WORKER_ID", "worker-gpu1")
    model_repository.record_load("ddcolor", "1.0.0", status=ModelStatus.READY)
    assert model_repository.list_versions("ddcolor")[0]["writer"] == "worker-gpu1"

    monkeypatch.delenv("FIXIMG_WORKER_ID")
    model_repository.record_load("ddcolor", "1.0.0", status=ModelStatus.READY)
    row = model_repository.list_versions("ddcolor")[0]
    assert row["writer"] == model_repository.process_identity()
    assert row["writer"].startswith("pid-")


def test_a_write_failure_is_reported_and_never_raises(isolated_db, monkeypatch):
    """The audit trail must not be able to fail a model load."""

    class _Broken:
        def execute(self, *a, **k):
            raise RuntimeError("database is locked")

        def commit(self):
            pass

    monkeypatch.setattr(model_repository, "get_conn", lambda: _Broken())
    assert model_repository.record_load("ddcolor", "1.0.0", status=ModelStatus.READY) is False


def test_the_table_holds_every_field_plan_6_lists_for_modelversion(isolated_db):
    """搂6 spelled the model out; a column nobody declared is a gap, not a style call."""
    PLAN_FIELDS = {
        "name", "version", "framework", "weight_uri", "sha256", "status",
        "loaded_at", "metadata",
    }
    model_repository.record_load("ddcolor", "1.0.0", status=ModelStatus.READY)
    columns = _columns(isolated_db)

    stored = {name for name in PLAN_FIELDS if name in columns}
    if "metadata_json" in columns:  # the plan's `metadata_json`, stored as a blob
        stored.add("metadata")
    assert stored == PLAN_FIELDS, PLAN_FIELDS - stored


def test_a_second_raw_insert_of_the_same_version_is_rejected(isolated_db):
    """The row identity is (name, version), enforced by the database.

    Asserted against the constraint rather than the repository's own upsert: if the
    key were missing, `record_load` would still look correct while a second writer
    quietly created a second "available version".
    """
    model_repository.record_load("ddcolor", "1.0.0", status=ModelStatus.READY)
    conn = sqlite3.connect(str(isolated_db / "test.db"))
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO model_versions(name, version, status) VALUES (?,?,?)",
                ("ddcolor", "1.0.0", "ready"),
            )
    finally:
        conn.close()
    assert [(r["name"], r["version"]) for r in model_repository.list_versions("ddcolor")] == [
        ("ddcolor", "1.0.0")
    ]


def test_a_write_bootstraps_the_table_in_a_process_that_never_ran_init_db(
    tmp_path, monkeypatch
):
    """The lazy path has to exist, not merely work when someone booted first.

    A worker records a model load the moment it claims a task; if the table came
    only from the application boot, the write would raise `no such table`, be
    swallowed by the best-effort handler, and leave the audit trail empty forever
    with nothing logged wrong.
    """
    import fiximg.config as config_mod
    import fiximg.infrastructure.db.engine as engine
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    db_path = str(tmp_path / "lazy.db")
    monkeypatch.setattr(engine, "DB_PATH", db_path)
    monkeypatch.setattr(engine, "ADMIN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config_mod.settings, "db_path", db_path)
    engine.close_connections()
    monkeypatch.setattr(task_repo, "_DDL_DONE", None, raising=False)

    assert model_repository.record_load("ddcolor", "1.0.0", status=ModelStatus.READY,
                                        load_ms=12) is True
    rows = model_repository.list_versions("ddcolor")
    assert [(r["version"], r["status"]) for r in rows] == [("1.0.0", "ready")]

    conn = sqlite3.connect(db_path)
    try:
        assert {row[1] for row in conn.execute("PRAGMA table_info(model_versions)")}
    finally:
        conn.close()
    engine.close_connections()


def _columns(isolated_db) -> set[str]:
    conn = sqlite3.connect(str(isolated_db / "test.db"))
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(model_versions)")}
    finally:
        conn.close()


# ---------------------------------------------------------------- the writer
def test_loading_a_version_records_it(isolated_db):
    """One write point: the registry that builds the model.

    Asserted on the row a *successful* load leaves behind 鈥?`load_ms` is measured,
    not declared, and the writer names this process.
    """
    from fiximg.inference.versions import VersionedModelRegistry

    registry = VersionedModelRegistry()
    registry.register_builder("ddcolor", lambda version: object())
    report = registry.activate("ddcolor", "1.0.0", weight_uri="/w/ddcolor/1.0.0")

    assert report["switched"] is True
    rows = model_repository.list_versions("ddcolor")
    assert [(r["version"], r["status"], r["load_count"]) for r in rows] == [
        ("1.0.0", "ready", 1)
    ], rows
    assert rows[0]["loaded_at"], rows[0]
    assert rows[0]["weight_uri"] == "/w/ddcolor/1.0.0"
    assert rows[0]["writer"] == model_repository.process_identity()


def test_an_activation_that_fails_is_recorded_as_a_failure(isolated_db):
    """The failed candidate is exactly what another node's operator needs to see."""

    def boom(version):
        raise FileNotFoundError(f"no weights for {version}")

    from fiximg.inference.versions import VersionedModelRegistry

    registry = VersionedModelRegistry()
    registry.register_builder("ddcolor", boom)
    report = registry.activate("ddcolor", "9.9.9")

    assert report["switched"] is False
    row = model_repository.list_versions("ddcolor")[0]
    assert row["status"] == "unhealthy"
    assert row["loaded_at"] is None
    assert "no weights for 9.9.9" in row["last_error"]


def test_a_broken_audit_trail_still_loads_the_model(isolated_db, monkeypatch):
    """Recording is a side effect; it may not decide whether a model comes up."""
    from fiximg.inference.versions import VersionedModelRegistry

    def explode(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(model_repository, "record_load", explode)

    registry = VersionedModelRegistry()
    registry.register_builder("ddcolor", lambda version: object())
    assert registry.activate("ddcolor", "1.0.0")["switched"] is True
    assert registry.current_version("ddcolor") == "1.0.0"


# ---------------------------------------------------------------- the reader
@pytest.fixture()
def model_client(tmp_path, monkeypatch):
    """An app with its own database, so `catalog` can be filled by a second 'process'."""
    import fiximg.config as config_mod
    import fiximg.infrastructure.db.engine as engine
    from fiximg.app_factory import create_app
    from fiximg.infrastructure.db.repositories import task_repository as task_repo
    from starlette.testclient import TestClient

    db_path = str(tmp_path / "catalog.db")
    monkeypatch.setattr(config_mod.settings, "db_path", db_path)
    monkeypatch.setattr(config_mod.settings, "tasks_root", str(tmp_path / "tasks"))
    monkeypatch.setattr(config_mod.settings, "inline_worker", False, raising=False)
    monkeypatch.setattr(engine, "DB_PATH", db_path)
    monkeypatch.setattr(engine, "ADMIN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(engine, "_conn", None)
    monkeypatch.setattr(task_repo, "_DDL_DONE", False, raising=False)
    monkeypatch.setenv("FIXIMG_API_TOKEN", "catalog-token")

    import fiximg.api.security as api_security
    from fiximg.inference.model_manager import model_manager

    # The test this fixture serves asserts "this process loaded nothing". The
    # registry is a process-wide singleton, so that premise has to be established
    # here rather than inherited: an earlier test in the same pytest process that
    # warms a model (`warm_at_process_start` does exactly that on `create_app`)
    # otherwise leaves a resident behind, and the test fails on collection order
    # alone. Ordering-dependent, not behaviour-dependent.
    for resident in model_manager.loaded_models():
        model_manager.unload(resident)

    api_security.reset_cache()
    with TestClient(create_app()) as client:
        client.headers.update({"Authorization": "Bearer catalog-token"})
        yield client


def test_versions_answers_for_the_deployment_not_only_this_process(model_client):
    """The symptom 搂6's missing table produced: an API node reporting no versions.

    Nothing is loaded *here* 鈥?the row is written straight into the shared database,
    which is what another node's worker does 鈥?yet the endpoint must still name the
    version the deployment is running, with the digest and the time it came up.
    """
    model_repository.record_load(
        "ddcolor", "1.4.0", status=ModelStatus.READY, framework="pytorch",
        weight_uri="/w/ddcolor/1.4.0", sha256="b" * 64, load_ms=3120,
        metadata={"repo": "damo/cv_ddcolor_image-colorization"},
    )

    body = model_client.get("/api/v1/models/ddcolor/versions").json()

    assert body["residents"] == [], "this process loaded nothing, and must not claim otherwise"
    assert body["active_version"] is None
    catalog = body["catalog"]
    assert len(catalog) == 1, catalog
    entry = catalog[0]
    assert entry["version"] == "1.4.0"
    assert entry["status"] == "ready"
    assert entry["sha256"] == "b" * 64
    assert entry["load_ms"] == 3120
    assert entry["metadata"] == {"repo": "damo/cv_ddcolor_image-colorization"}


def test_a_failure_recorded_elsewhere_shows_up_as_a_failure(model_client):
    model_repository.record_load("ddcolor", "1.5.0", status=ModelStatus.UNHEALTHY,
                                 error="sha256 mismatch")
    entry = model_client.get("/api/v1/models/ddcolor/versions").json()["catalog"][0]
    assert entry["status"] == "unhealthy"
    assert entry["last_error"] == "sha256 mismatch"
    assert entry["loaded_at"] is None


def test_the_catalog_keys_are_all_declared_in_the_schema(model_client):
    """The repository's row and the response model must describe the same object.

    Checked against the row, not the HTTP body: `response_model` fills a missing
    declared field with its default, so comparing the *body* to the schema stays true
    even when the query stopped selecting the column 鈥?the green that a first version
    of this test produced under mutation. The body is still checked, for the reverse
    direction (a key the schema lacks is silently dropped before any client sees it).
    """
    from fiximg.api.schemas.model import ModelVersionsResponse, RecordedVersionView

    model_repository.record_load("ddcolor", "1.0.0", status=ModelStatus.READY,
                                 framework="pytorch", sha256="c" * 64)
    row = model_repository.list_versions("ddcolor")[0]
    declared = set(RecordedVersionView.model_fields)

    assert set(row) <= declared, f"dropped by the schema: {sorted(set(row) - declared)}"
    assert declared <= set(row), f"advertised but never filled: {sorted(declared - set(row))}"

    body = model_client.get("/api/v1/models/ddcolor/versions").json()
    assert set(ModelVersionsResponse.model_fields) >= {"catalog", "residents",
                                                      "active_version", "previous_version"}
    assert body["catalog"][0]["sha256"] == "c" * 64, "the value must survive serialisation"
    assert body["catalog"][0]["framework"] == "pytorch"
