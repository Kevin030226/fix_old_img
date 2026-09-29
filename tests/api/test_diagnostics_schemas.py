"""Diagnostics endpoints are declared models, not free-form dicts (plan 搂2.5, 搂3.8).

`/api/v1/stats` and the health probes were the last two endpoints that returned a
hand-built dict, which shows up in the OpenAPI document as an untyped object: the
one endpoint an operator polls is the one a client generator cannot type.

Two directions are pinned, because a response model is only half a contract:

* a key the producer emits but the model does not declare would be **silently
  dropped** by FastAPI's validation 鈥?so the live payload's keys must all be
  declared (or the model must say `extra="allow"`, which is asserted separately);
* a key the model declares that nothing produces is a field every client sees as
  always-null 鈥?so the declared fields must all be present.
"""
import pytest
from fastapi.testclient import TestClient

from fiximg.api.schemas.diagnostics import (
    MetricsSnapshot,
    ModelHealth,
    ModelStats,
    PluginReport,
    ReadinessResponse,
    SchedulerStats,
    StageTiming,
    StatsResponse,
    TaskTotals,
    WorkerReport,
)


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    """A throwaway app whose DB has one completed task, so stats are not all-empty."""

    import fiximg.config as config_mod
    import fiximg.infrastructure.db.engine as engine
    from fiximg.app_factory import create_app
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    db_path = str(tmp_path / "diag.db")
    monkeypatch.setattr(config_mod.settings, "db_path", db_path)
    monkeypatch.setattr(config_mod.settings, "tasks_root", str(tmp_path / "tasks"))
    monkeypatch.setattr(config_mod.settings, "inline_worker", False, raising=False)
    monkeypatch.setattr(engine, "DB_PATH", db_path)
    monkeypatch.setattr(engine, "ADMIN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(engine, "_conn", None)
    task_repo._DDL_DONE = False
    monkeypatch.setenv("FIXIMG_API_TOKEN", "diag-token")

    import fiximg.api.security as api_security

    api_security.reset_cache()
    monkeypatch.setattr(task_repo, "_DDL_DONE", False, raising=False)

    with TestClient(create_app()) as client:
        client.headers.update({"Authorization": "Bearer diag-token"})
        task_repo.create_task("diag-1", "restore", "alice")
        task_repo.start_task("diag-1")
        task_repo.record_stage("diag-1", 0, "global_restore", "running")
        task_repo.finish_stage("diag-1", 0, "completed", 1234)
        task_repo.finish_task("diag-1", str(tmp_path / "out.png"), "PSNR: 20", 1234)
        yield client


def _keys(payload):
    return set(payload)


def _is_typed(schema) -> bool:
    """A response is "typed" when it names a model, possibly as array items."""
    if not isinstance(schema, dict):
        return False
    if schema.get("$ref") or schema.get("allOf") or schema.get("anyOf") or schema.get("oneOf"):
        return True
    if schema.get("type") == "array":
        return _is_typed(schema.get("items"))
    return False


def test_stats_top_level_is_exactly_what_the_model_declares(api_client):
    body = api_client.get("/api/v1/stats").json()
    declared = set(StatsResponse.model_fields)

    assert _keys(body) == declared, (
        f"payload {sorted(body)} vs schema {sorted(declared)} 鈥?a key missing from "
        "the model would be dropped from the response before any client sees it"
    )


def test_stats_nested_blocks_match_their_models(api_client):
    body = api_client.get("/api/v1/stats?days=7").json()

    assert _keys(body["tasks"]) == set(TaskTotals.model_fields), body["tasks"]
    assert _keys(body["metrics"]) == set(MetricsSnapshot.model_fields)
    assert _keys(body["scheduler"]) == set(SchedulerStats.model_fields)
    # Declared 鈯?or 鈯?is the wrong assertion here: `models` gains
    # `gpu_memory_peak_mb` only on a CUDA node. What must hold in both shapes is
    # that nothing is emitted which the schema does not describe (that is the key a
    # response model can silently drop) and that the always-present one is there.
    assert _keys(body["models"]) <= set(ModelStats.model_fields), body["models"]
    assert "load_times_ms" in body["models"]

    # A stage timing exists only once some task has run, which is why the fixture
    # completes one: an empty dict here would let the field names go unchecked.
    assert body["stages"], "no stage timings, so this test verified nothing"
    for stage, timing in body["stages"].items():
        assert _keys(timing) == set(StageTiming.model_fields), stage
    assert body["stages"]["global_restore"]["p95_ms"] == 1234
    assert body["tasks"]["completed"] == 1


def test_readiness_is_a_declared_model_too(api_client):
    body = api_client.get("/api/v1/health/ready").json()
    declared = set(ReadinessResponse.model_fields) - {"detail"}

    assert _keys(body) == declared, f"payload {sorted(body)} vs {sorted(declared)}"
    assert body["status"] in ("ready", "not_ready")
    assert _keys(body["worker"]) <= set(WorkerReport.model_fields), body["worker"]
    assert _keys(body["models"]) <= set(ModelHealth.model_fields)
    assert _keys(body["plugins"]) <= set(PluginReport.model_fields)


def test_liveness_endpoints_agree_on_their_shape(api_client):
    live = api_client.get("/api/v1/health/live").json()
    alias = api_client.get("/health").json()

    assert live == alias == {"status": "ok"}


def test_the_openapi_document_types_every_json_response(api_client):
    """The gate that keeps this hole from reopening.

    A route whose 2xx JSON response carries no schema reference renders as a
    free-form object in `/docs` and as `dict[str, Any]` in a generated client 鈥?
    which is how `/api/v1/stats` looked while every other endpoint was typed.
    """
    spec = api_client.get("/openapi.json").json()
    untyped = []

    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            responses = operation.get("responses", {})
            for status, response in responses.items():
                if not status.startswith(("2", "4", "5")):
                    continue
                content = response.get("content", {})
                json_body = content.get("application/json")
                if json_body is None:
                    continue  # text/plain (Prometheus, SSE) is not a model's job
                schema = json_body.get("schema") or {}
                if _is_typed(schema):
                    continue
                untyped.append(f"{method.upper()} {path} [{status}]")

    assert not untyped, f"routes returning an untyped JSON body: {sorted(untyped)}"


def test_stats_reports_a_peak_recorded_by_another_process(api_client):
    """The API node has never touched the GPU, yet the deployment has.

    Measured live on the split topology: the stage rows said 334.8 MB while
    `GET /stats` answered `gpu_memory_peak_mb: 0.0`, because each process reads only
    its own CUDA counters. The persisted stage figures are the cross-process source,
    so the endpoint folds them in instead of reporting the requesting process alone.
    """
    from fiximg.infrastructure.db.repositories import task_repository as task_repo
    from fiximg.inference.gpu_memory import STAGE_PEAK_METRIC

    task_repo.add_metric("diag-1", STAGE_PEAK_METRIC, 512.0)
    body = api_client.get("/api/v1/stats").json()
    assert body["models"]["gpu_memory_peak_mb"] >= 512.0, body["models"]


def test_readiness_reports_gpu_utilisation_from_the_shared_probe(api_client, monkeypatch):
    """搂7.1's other KPI is served, and served from one sampler rather than per request.

    The endpoint must not fork the driver tool itself: an HTTP scrape asking
    `nvidia-smi` directly would put a subprocess in the request path and give a
    reading that has nothing to do with the window the stages were attributed to.
    """
    from fiximg.inference import gpu_utilization as gu

    seen: list = []

    class _Stub:
        def ensure_running(self):
            seen.append("ensure")

        def describe(self, devices=None):
            seen.append(devices)
            return {
                "status": gu.STATUS_MEASURING,
                "interval_seconds": 0.5,
                "devices": [{"device": 0, "utilization_pct": 42,
                             "memory_used_mb": 496, "memory_total_mb": 8151}],
                "busy_devices": 1,
            }

    monkeypatch.setattr(gu, "default_probe", _Stub())
    body = api_client.get("/api/v1/health/ready").json()

    utilization = body["gpu"]["topology"]["utilization"]
    assert utilization["status"] == gu.STATUS_MEASURING, utilization
    assert utilization["devices"][0]["utilization_pct"] == 42, utilization
    assert seen[0] == "ensure", "the probe must be warmed by the read path"
    assert isinstance(seen[1], list), f"asked for the topology's devices, got {seen[1]!r}"


# ------------------------------------------------- standalone worker liveness
def _liveness_view(monkeypatch):
    """The readiness worker block, with the probe forced into standalone mode."""
    import fiximg.config as config_mod
    from fiximg.api.routes.health import _worker_report

    monkeypatch.setattr(config_mod.settings, "inline_worker", False, raising=False)
    return _worker_report()


def _backdate(task_id: str, column: str, seconds_ago: float) -> None:
    """Put one row's clock in the past, the way a wedged worker leaves it."""
    import sqlite3

    import fiximg.config as config_mod
    from fiximg.infrastructure.db import timestamps

    conn = sqlite3.connect(config_mod.settings.db_path)
    conn.execute(f"UPDATE tasks SET {column}=? WHERE id=?",  # noqa: S608 鈥?fixed column names
                 (timestamps.deadline(-seconds_ago), task_id))
    conn.commit()
    conn.close()


def test_every_liveness_key_is_declared(api_client, monkeypatch):
    """A key the probe emits and the model lacks is a key no client ever sees."""
    view = _liveness_view(monkeypatch)
    undeclared = set(view) - set(WorkerReport.model_fields)
    assert not undeclared, f"emitted but the schema drops it: {sorted(undeclared)}"


def test_a_worker_holding_a_task_is_reported_alive(api_client, monkeypatch):
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    task_repo.create_task("live-1", "restore", "alice")
    assert task_repo.claim_next_task("w-live") is not None

    view = _liveness_view(monkeypatch)
    assert view["worker_liveness"] == "alive", view
    assert view["worker_running"] is True
    assert view["workers_reporting"] == 1


def test_a_worker_that_stopped_beating_is_reported_silent(api_client, monkeypatch):
    """Heartbeats stop when the loop wedges; the row keeps saying 'running'.

    Reading only `status='running'` would call that alive forever 鈥?which is the
    state a GPU worker that died inside a CUDA call actually leaves behind.
    """
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    task_repo.create_task("stale-1", "restore", "alice")
    assert task_repo.claim_next_task("w-stale") is not None
    _backdate("stale-1", "last_heartbeat", 7200)

    view = _liveness_view(monkeypatch)
    assert view["worker_liveness"] == "silent", view
    assert view["worker_running"] is False
    assert view["workers_silent"] == 1


def test_an_empty_queue_cannot_tell_idle_from_dead(api_client, monkeypatch):
    view = _liveness_view(monkeypatch)
    assert view["worker_liveness"] == "idle-or-unknown", view
    assert view["worker_running"] is None, "idle must not be reported as dead"
    assert view["oldest_queued_seconds"] is None


def test_a_backlog_nobody_takes_is_reported(api_client, monkeypatch):
    """The case that should page: work waiting while nothing consumes it."""
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    task_repo.create_task("wait-1", "restore", "alice")
    _backdate("wait-1", "created_at", 7200)

    view = _liveness_view(monkeypatch)
    assert view["worker_liveness"] == "backlogged", view
    assert view["worker_running"] is False
    assert view["oldest_queued_seconds"] > 3600


def test_the_probe_names_the_database_behind_the_queue(api_client, monkeypatch):
    """`queue_backend: sqlite` is the transport family, not the server.

    On a PostgreSQL deployment that label alone reads as "the queue is a file", which
    is how a split-topology operator concluded the node was misconfigured. The driver
    is reported next to it, read from the same resolver the connections use.
    """
    from fiximg.infrastructure.db.dialect import default_dialect

    view = _liveness_view(monkeypatch)
    assert view["queue_driver"] == default_dialect().name
    assert view["queue_backend"] == "sqlite"
