"""Every declared metric name must be reachable from the exposition.

Found by running the service, not by reading it. On the documented split
topology (`api` + a separate `worker`) the live endpoint exported two of the
eleven names in `MetricName`:

    queue_depth          task_submit_total

Both are recorded on the *submission* path, so the API process has them. The
other nine — stage durations, task duration, queue wait, model load and inference
time, VRAM, utilisation, retries, artifact I/O — are recorded by the worker, in a
process the API cannot read, and none of them were persisted in a form the API
could aggregate. `docs/deployment.md` nonetheless calls this endpoint the
"Prometheus scrape target" and lists `gpu_utilization_percent{stage,device}`
among its series.

The gate that was supposed to prevent this checked that each name is *referenced*
somewhere in `src/`. Every one of the eleven is, by `metrics.py` itself — which
is why it stayed green. Spelling a constant is not the same as being able to
export it, and the difference is only visible at the endpoint.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fiximg.api import security as api_security
from fiximg.infrastructure.db.repositories import task_repository as repo
from fiximg.infrastructure.observability.metrics import (
    DECLARED,
    SUBMISSION_SCOPED,
    WORKER_PROCESS_ONLY,
    nearest_rank,
)

TOKEN = "exposition-token"


@pytest.fixture()
def client(isolated_db, monkeypatch):
    api_security.reset_cache()
    monkeypatch.setenv("FIXIMG_API_TOKEN", TOKEN)
    api_security.reset_cache()
    from fiximg.app_factory import create_app

    with TestClient(create_app()) as c:
        yield c
    api_security.reset_cache()


def _exposition(client) -> str:
    resp = client.get("/api/v1/stats/metrics",
                      headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    return resp.text


def _seed_deployment(client) -> None:
    """One finished task with stage rows, so the DB-derived series have data."""
    repo.create_task("t-exp", "restore", "u")
    repo.start_task("t-exp")
    for order, name, ms in ((0, "global_restore", 1200), (1, "warp_back", 3000)):
        repo.record_stage("t-exp", order, name, "running")
        repo.finish_stage("t-exp", order, "completed", ms, None,
                          metrics={"gpu_peak_mb": 512.0, "gpu_device": 0,
                                   "gpu_util_pct": 40.0})
    repo.finish_task("t-exp", "out.png", "done", 4200)


def _parse_exposition(text: str) -> list[tuple[str, dict[str, str], float]]:
    """(series, labels, value) per line.

    Labels are parsed rather than substring-matched: the renderer sorts them, so
    `scope` is not first, and a test that looked for `{scope="` passed on nothing.
    That is the same "two assertions individually true and unrelated" shape this
    batch has already caught twice.
    """
    out = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        if "{" in head:
            base, _, rest = head.partition("{")
            labels = {}
            for pair in rest.rstrip("}").split(","):
                if "=" in pair:
                    k, _, v = pair.partition("=")
                    labels[k.strip()] = v.strip().strip('"')
        else:
            base, labels = head, {}
        out.append((base, labels, float(value)))
    return out


def test_every_declared_name_is_derivable_or_in_one_of_the_two_named_scopes():
    """The coverage property, which needs no traffic to check.

    A name is covered when `deployment_metrics` can produce it, or when it is
    recorded where that process holds the deployment's whole figure
    (`SUBMISSION_SCOPED`), or when it is declared worker-process-only. Presence in
    one particular exposition is a different question — a counter nobody
    incremented is correctly absent — so demanding every name appear would make
    this test answer "has traffic run yet".
    """
    derivable = set(repo.deployment_metrics())

    uncovered = [
        name for name in DECLARED
        if name not in derivable
        and name not in SUBMISSION_SCOPED
        and name not in WORKER_PROCESS_ONLY
    ]
    assert not uncovered, (
        f"{uncovered} is neither derived from the persisted rows, nor "
        "submission-scoped, nor declared worker-process-only"
    )
    # The three sets partition the declared names: a name in two of them means one
    # of the two statements is stale, and a name in none is the defect.
    assert not (SUBMISSION_SCOPED & WORKER_PROCESS_ONLY), (
        "a name is both submission-scoped and worker-process-only"
    )
    assert not (SUBMISSION_SCOPED & derivable), (
        "a submission-scoped name is also derived from the database; it would be "
        "counted twice"
    )
    assert not (WORKER_PROCESS_ONLY & derivable), (
        "a worker-process-only name is also derived from the database; one of the "
        "two statements is stale"
    )
    assert set(DECLARED) == derivable | SUBMISSION_SCOPED | WORKER_PROCESS_ONLY


def test_a_seeded_run_makes_the_derived_names_appear_in_the_exposition(client):
    _seed_deployment(client)
    text = _exposition(client)
    series = {base for base, _labels, _v in _parse_exposition(text)}

    for name in ("task_duration_seconds", "stage_duration_seconds",
                 "queue_wait_seconds", "worker_retry_total", "gpu_memory_bytes",
                 "gpu_utilization_percent"):
        assert any(s.startswith(name) for s in series), f"{name} missing from:\n{text}"


def test_a_recorded_process_local_name_is_exported_under_its_own_name(client):
    """The API process' own counters appear, unrenamed.

    Naming is pinned by `test_an_already_existing_series_name_is_not_renamed_by_a_label`
    on a fresh registry, where the expected output is fully known. Here the
    registry is the process-wide singleton, so these counters are partitioned across
    whatever label sets earlier tests in the same pytest process happened to use —
    `task_submit_total` and `task_submit_total{task_type=...}` are different series
    and their values are not comparable. So this asserts what the endpoint controls
    and nothing about the numbers.
    """
    from fiximg.infrastructure.observability.metrics import MetricName, metrics

    metrics.inc(MetricName.TASK_SUBMIT_TOTAL, 1, task_type="restore")
    metrics.observe(MetricName.QUEUE_DEPTH, 3)
    text = _exposition(client)
    parsed = _parse_exposition(text)

    submitted = [r for r in parsed if r[0] == "task_submit_total"]
    assert submitted, text
    # No renderer-invented label anywhere. A label the *caller* supplied is part of
    # the series' identity and long predates this endpoint.
    for _base, labels, _v in parsed:
        assert "scope" not in labels, f"the renderer added a label: {labels}"
    assert any("queue_depth" in base for base, _l, _v in parsed), text


def test_the_unexportable_names_say_so_in_the_output(client):
    """A reader must be able to tell "never measured" from "scrape the worker"."""
    _seed_deployment(client)
    text = _exposition(client)
    assert "not deployment-wide" in text
    for name in WORKER_PROCESS_ONLY:
        assert name in text, name


def test_the_documented_utilisation_series_is_actually_there(client):
    """`docs/deployment.md` names this series at this endpoint, by stage and device.

    Asserted against the renderer rather than the served body, for the reason given
    in `test_the_stage_percentiles_agree_with_the_json_stats_endpoint`: the endpoint
    stands the deployment series down when this process already records the name, and
    in a full-suite run something usually has. The deployment figures are what the
    documentation promises, and the renderer is where they are decided.
    """
    from fiximg.infrastructure.observability.metrics import render_deployment_series

    _seed_deployment(client)
    text = "\n".join(render_deployment_series(repo.deployment_metrics()))

    rows = [
        (base, labels, value)
        for base, labels, value in _parse_exposition(text)
        if base.startswith("gpu_utilization_percent")
    ]
    assert rows, text
    sample = next(r for r in rows if r[0] == "gpu_utilization_percent_avg")
    assert sample[1]["stage"] == "global_restore", sample
    assert sample[1]["device"] == "0", sample
    assert sample[2] == 40.0, sample


def test_an_already_existing_series_name_is_not_renamed_by_a_label(client):
    """Adding `scope=` renames a series. That breaks every dashboard silently.

    `task_submit_total` and `task_submit_total{scope="process"}` are different
    series in Prometheus, so a label that looks like documentation is in fact a
    breaking change to the metric contract. The first version of this endpoint did
    exactly that; this is the test that has to stay green if it is ever tried
    again.
    """
    from fiximg.infrastructure.observability.metrics import (
        MetricsRegistry,
        nearest_rank,  # noqa: F401  (import parity with the other tests)
    )

    registry = MetricsRegistry()
    registry.inc("task_submit_total", 3)
    registry.observe("queue_depth", 5)

    text = registry.render_prometheus()
    assert "task_submit_total 3.0" in text, text
    assert "queue_depth_avg 5.0" in text, text
    assert "scope=" not in text, (
        f"a label was added to an existing series name: {text}"
    )


def test_the_two_sources_never_emit_the_same_series_twice(client):
    """One exposition, one definition per series.

    On a single-process deployment the registry and the database describe the same
    work. Emitting both would put a byte-identical line in the exposition twice,
    which Prometheus rejects as malformed — and which would double a rate.
    """
    from fiximg.infrastructure.observability.metrics import MetricsRegistry

    registry = MetricsRegistry()
    registry.observe("stage_duration_seconds", 1.0, stage="global_restore")

    deployment = {
        "stage_duration_seconds": {
            "global_restore": {"runs": 1, "avg": 1.0, "p50": 1.0, "p95": 1.0}
        },
    }
    text = registry.render_prometheus(deployment)

    keys = [ln.rsplit(" ", 1)[0] for ln in text.splitlines() if ln and not ln.startswith("#")]
    duplicates = {k for k in keys if keys.count(k) > 1}
    assert not duplicates, f"the same series appears twice: {sorted(duplicates)}\n{text}"


def test_the_gap_filler_supplies_what_the_process_did_not_record(client):
    """On an API node the registry is nearly empty; the rest comes from the rows."""
    _seed_deployment(client)
    text = _exposition(client)
    parsed = _parse_exposition(text)

    assert any(base.startswith("stage_duration_seconds") for base, _l, _v in parsed), text
    assert any(base.startswith("gpu_utilization_percent") for base, _l, _v in parsed), text
    assert "# below: deployment aggregates" in text, text


def test_it_works_on_a_deployment_that_has_never_run_a_task(client):
    """An empty deployment must still answer, not 500 and not an empty body."""
    text = _exposition(client)
    assert text is not None
    # The explanation for the process-local names is present even with no data.
    assert "not deployment-wide" in text


def test_the_stage_percentiles_agree_with_the_json_stats_endpoint(client):
    """`/stats` and `/metrics` must not answer "how long is a stage" differently.

    Both read `task_stats`, so this is the check that keeps them that way — and it
    is the check that would have caught a second, hand-rolled percentile in the
    exposition.

    The comparison is against the renderer rather than the endpoint response,
    because the endpoint deliberately stands the deployment series down when this
    process's own registry already holds the name. In a full-suite run something
    else has usually recorded `stage_duration_seconds`, so the served exposition
    legitimately has no p95 line to compare, and asserting on it would make this
    test answer "did an earlier test warm a histogram" instead of "do the two
    endpoints define a percentile the same way".
    """
    from fiximg.infrastructure.observability.metrics import render_deployment_series

    _seed_deployment(client)
    stats = client.get("/api/v1/stats",
                       headers={"Authorization": f"Bearer {TOKEN}"}).json()

    lines = render_deployment_series(repo.deployment_metrics())
    text = "\n".join(lines)
    p95 = {
        labels["stage"]: value
        for base, labels, value in _parse_exposition(text)
        if base == "stage_duration_seconds_p95" and "stage" in labels
    }
    assert p95, text
    for stage, s in stats["stages"].items():
        if stage in p95 and s["p95_ms"] is not None:
            assert p95[stage] == pytest.approx(s["p95_ms"] / 1000.0, abs=1e-3), (
                stage, p95[stage], s["p95_ms"]
            )
    assert set(p95) == set(stats["stages"]), (sorted(p95), sorted(stats["stages"]))


def test_queue_wait_is_derived_from_the_same_instants_the_worker_used(client):
    """A wait that disagrees with the worker's own reading is a second answer."""
    from fiximg.infrastructure.db import timestamps

    repo.create_task("t-wait", "restore", "u")
    repo.start_task("t-wait")
    row = repo.get_task("t-wait")
    gap = (timestamps.parse(row["started_at"])
           - timestamps.parse(row["created_at"])).total_seconds()

    count = next((v for base, _l, v in _parse_exposition(_exposition(client))
                  if base == "queue_wait_seconds_count"), None)
    assert count == 1.0, count
    avg = next((v for base, _l, v in _parse_exposition(_exposition(client))
                if base == "queue_wait_seconds_avg"), None)
    assert avg is not None and abs(avg - gap) < 0.05, (avg, gap)


def test_nearest_rank_is_the_definition_the_exposition_claims():
    """`task_stats` uses nearest rank in SQL; the exposition must match."""
    assert nearest_rank([1, 2, 3, 4], 0.5) == 2
    assert nearest_rank([1, 2, 3, 4], 0.95) == 4
    assert nearest_rank([5], 0.5) == 5
