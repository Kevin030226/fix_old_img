"""In-process metrics registry (plan §2.10).

V3 upgrades "logs" to "observability" without adding a hard dependency: a small
counter/histogram registry that can be exposed in two ways —

* ``snapshot()`` — JSON, served by ``GET /api/v1/stats`` for the admin panel,
* ``render_prometheus()`` — text exposition format, scraped by Prometheus if the
  deployment has one.

Tracked series follow the plan's list: task submit/duration, queue depth and
wait, per-stage duration, model load/inference time, GPU memory and utilisation,
worker retries and artifact I/O.
"""
from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field


def nearest_rank(ordered: Sequence[float], fraction: float) -> float | None:
    """The value at ``ceil(fraction * n)`` of an ascending sample (nearest rank).

    One implementation on purpose: the repository computes the same statistic in
    SQL (`task_repository._ceil_rank`) and the benchmark uses this one, so a "p95"
    means the same thing in the dashboard, in `/metrics` and in the latency table.
    Interpolation is what `percentile_cont` would give, and it has no SQLite
    counterpart — two engines, two different numbers for identical rows.

    The formula this replaced, ``ordered[int(n * fraction) - 1]``, returned the
    *minimum* for a two-sample p95, which is exactly the sample size the CI
    benchmark runs at.
    """
    if not ordered:
        return None
    n = len(ordered)
    rank = min(n, max(1, math.ceil(n * fraction)))
    return ordered[rank - 1]


@dataclass
class _Histogram:
    """Fixed-bucket-free histogram: keeps a bounded sample window."""

    max_samples: int = 512
    count: int = 0
    total: float = 0.0
    samples: list[float] = field(default_factory=list)

    def observe(self, value: float) -> None:
        value = float(value)
        self.count += 1
        self.total += value
        self.samples.append(value)
        if len(self.samples) > self.max_samples:
            del self.samples[: len(self.samples) - self.max_samples]

    def percentile(self, pct: float) -> float | None:
        return nearest_rank(sorted(self.samples), pct)

    def summary(self) -> dict:
        return {
            "count": self.count,
            "avg": round(self.total / self.count, 4) if self.count else None,
            "p50": self.percentile(0.50),
            "p95": self.percentile(0.95),
            "p99": self.percentile(0.99),
        }


class MetricsRegistry:
    """Thread-safe counters + histograms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._histograms: dict[str, _Histogram] = {}
        self._started_at = time.time()

    # ------------------------------------------------------------------ write
    def inc(self, name: str, value: float = 1.0, **labels) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + float(value)

    def observe(self, name: str, value: float, **labels) -> None:
        key = self._key(name, labels)
        with self._lock:
            hist = self._histograms.get(key)
            if hist is None:
                hist = _Histogram()
                self._histograms[key] = hist
        hist.observe(value)

    @contextmanager
    def timer(self, name: str, **labels) -> Iterator[None]:
        """Measure a block in seconds: ``with metrics.timer("stage_duration_seconds", stage=s):``"""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - started, **labels)

    # ------------------------------------------------------------------- read
    @staticmethod
    def _key(name: str, labels: dict) -> str:
        """The registry's internal identity for a labelled series.

        Deliberately unquoted: this string is a dict key, and it is also what
        `snapshot()["counters"]` publishes, so its shape is a contract of its own.
        Quoting is a *rendering* concern and happens in `render_prometheus`.
        """
        if not labels:
            return name
        rendered = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{rendered}}}"

    def snapshot(self) -> dict:
        """JSON view for ``GET /api/v1/stats``."""
        with self._lock:
            counters = dict(self._counters)
            histograms = {k: v.summary() for k, v in self._histograms.items()}
        return {
            "uptime_seconds": round(time.time() - self._started_at, 1),
            "counters": counters,
            "histograms": histograms,
        }

    def render_prometheus(self, deployment: dict | None = None) -> str:
        """Text exposition format (no prometheus_client dependency).

        Two sources, and the second one is only a *gap filler*:

        * this process' own registry, rendered exactly as it always was — bare series
          names, no labels added. Adding a label is not cosmetic: in Prometheus
          `task_submit_total` and `task_submit_total{scope="process"}` are different
          series, so labelling them renames every existing dashboard and alert rule
          silently. The first version of this function did that, and the
          pre-existing unit test is what caught it.
        * the deployment aggregates from :func:`render_deployment_series`, emitted
          only for series this process did *not* already record. On a split
          `api` + `worker` deployment the API's registry holds two of the declared
          names and the rest come from here. On a single-process (`inline`)
          deployment the registry already holds them and the gap filler stands down,
          which also keeps one exposition from carrying the same series name twice —
          Prometheus rejects that as malformed.
        """
        lines: list[str] = []
        with self._lock:
            counters = dict(self._counters)
            histograms = {k: v.summary() for k, v in self._histograms.items()}

        def wire(key: str) -> str:
            """`_key`'s internal form -> exposition syntax, with values quoted.

            `stage=global_restore` is not valid exposition: a scraper rejects the
            scrape, or that series, rather than reading the label. Every labelled
            series the registry recorded was being served malformed, which is why
            the gap-filler's own series and the registry's could not be compared for
            equality — one side quoted, the other did not.
            """
            base, _, rest = key.partition("{")
            tail = rest.rstrip("}")
            if not tail:
                return base
            pairs = []
            for pair in tail.split(","):
                k, _, v = pair.partition("=")
                pairs.append(f'{k}="{v}"')
            return f"{base}{{{','.join(pairs)}}}"

        for key, value in sorted(counters.items()):
            lines.append(f"{wire(key)} {value}")
        for key, summary in sorted(histograms.items()):
            # The labels have to be repeated on the derived series, not just on the
            # count line. `stage_duration_seconds{stage=x}` followed by a bare
            # `stage_duration_seconds_p95` silently collapses every stage's p95 into
            # one number, which is the opposite of the per-stage breakdown
            # `docs/deployment.md` names at this endpoint.
            base = key.split("{")[0]
            tail = wire(key)

            def derived(suffix: str, _base=base, _tail=tail) -> str:
                if "{" not in _tail:
                    return f"{_base}{suffix}"
                inner = _tail.split("{", 1)[1].rstrip("}")
                return f"{_base}{suffix}{{{inner}}}"

            lines.append(f"{wire(key)} {summary['count']}")
            if summary["avg"] is not None:
                lines.append(f"{derived('_avg')} {summary['avg']}")
            # Every percentile the summary computes is exposed: emitting only p95
            # meant a dashboard could not see the spread, and `p50`/`p99` were
            # present in the JSON snapshot but not in the text format.
            for name in ("p50", "p95", "p99"):
                if summary[name] is not None:
                    lines.append(f"{derived('_' + name)} {summary[name]}")

        if deployment:
            # The names this process already put on the wire. A histogram key
            # `stage_duration_seconds{stage=x}` also emits `_avg`, `_p50`, `_p95` and
            # `_p99` lines, so the derived names have to be collected too — comparing
            # only the bare base would let `_p95` through from both sources and
            # produce a byte-identical duplicate line.
            recorded = {k.split("{")[0] for k in counters}
            for key in histograms:
                base = key.split("{")[0]
                recorded.add(base)
                recorded.update({f"{base}_{n}" for n in ("avg", "p50", "p95", "p99")})
            lines.extend(render_deployment_series(deployment, already=recorded))
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


#: Process-wide registry.
metrics = MetricsRegistry()


def _labels(base: str, **extra) -> str:
    """`name{k=v,...}` with a non-empty set, so no `{}` is ever emitted."""
    pairs = {k: v for k, v in extra.items() if v is not None}
    if not pairs:
        return base
    rendered = ",".join(f'{k}="{v}"' for k, v in sorted(pairs.items()))
    return f"{base}{{{rendered}}}"


def render_deployment_series(deployment: dict, already: set[str] | None = None) -> list[str]:
    """Render `task_repository.deployment_metrics()` as exposition lines.

    The shapes differ per metric (a scalar, a per-label breakdown, a
    per-stage histogram), so the rendering is explicit rather than generic: a
    generic walker over dicts would have to guess which numbers are counts, which
    are samples and which are labels.

    `already` is the set of base names this process's own registry emitted. Any
    deployment series whose base name is in it is skipped: on a single-process
    deployment the two sources would otherwise both describe the same work, and one
    exposition carrying a series name twice is malformed.
    """
    taken = already or set()
    out: list[str] = []
    lines: list[str] = []

    def emit(base: str, value, **labels) -> None:
        if value is None or base in taken:
            return
        out.append(f"{_labels(base, **labels)} {value}")

    totals = deployment.get("task_duration_seconds") or {}
    emit("task_duration_seconds_count", totals.get("runs"))
    emit("task_duration_seconds_avg", totals.get("avg"))

    for stage, s in (deployment.get("stage_duration_seconds") or {}).items():
        emit("stage_duration_seconds_count", s.get("runs"), stage=stage)
        emit("stage_duration_seconds_avg", s.get("avg"), stage=stage)
        emit("stage_duration_seconds_p50", s.get("p50"), stage=stage)
        emit("stage_duration_seconds_p95", s.get("p95"), stage=stage)

    wait = deployment.get("queue_wait_seconds") or {}
    emit("queue_wait_seconds_count", wait.get("runs"))
    emit("queue_wait_seconds_avg", wait.get("avg"))
    emit("queue_wait_seconds_p50", wait.get("p50"))
    emit("queue_wait_seconds_p95", wait.get("p95"))

    emit("worker_retry_total", (deployment.get("worker_retry_total") or {}).get("value"))

    for stage, s in (deployment.get("gpu_memory_bytes") or {}).items():
        emit("gpu_memory_bytes_max", s.get("max"), stage=stage, device=s.get("device"))
        emit("gpu_memory_bytes_avg", s.get("avg"), stage=stage, device=s.get("device"))

    for stage, s in (deployment.get("gpu_utilization_percent") or {}).items():
        emit("gpu_utilization_percent_avg", s.get("avg"), stage=stage,
             device=s.get("device"))
        emit("gpu_utilization_percent_max", s.get("max"), stage=stage,
             device=s.get("device"))
        emit("gpu_utilization_percent_samples", s.get("samples"), stage=stage,
             device=s.get("device"))

    for model, seconds in ((deployment.get("model_load_seconds") or {})
                           .get("per_model") or {}).items():
        emit("model_load_seconds", seconds, model=model)

    if out:
        lines = [
            "# below: deployment aggregates, read from the rows every process writes"
        ]
        lines.extend(out)

    missing = [m for m in WORKER_PROCESS_ONLY
               if not any(series.startswith(m) for series in
                          (ln.split()[0] for ln in out))]
    if missing:
        lines.append(
            "# not deployment-wide (measured in the worker, not persisted): "
            + ", ".join(sorted(missing))
        )
    return lines


#: Declared names that stay inside the process that recorded them, and why they
#: are nevertheless complete. `deployment_metrics` says the same in prose; this is
#: the machine-readable half, and the endpoint's own output repeats the second set
#: so a missing series is explained at the scrape target.
#:
#: `task_submit_total` and `queue_depth` are recorded on the submission path, and
#: in a split deployment only the API process accepts submissions — so the API's
#: own registry holds the deployment's complete figure for them. They are not
#: derivable from the rows, and must not also be derived from them.
SUBMISSION_SCOPED = frozenset({
    "task_submit_total",
    "queue_depth",
})

#: Recorded by the worker and never persisted. A reader who wants these has to
#: scrape the worker; inventing a deployment figure would be a number with no
#: source behind it.
WORKER_PROCESS_ONLY = frozenset({
    "model_inference_seconds",
    "artifact_io_seconds",
})


#: Canonical metric names (plan §2.10) — use these constants, not literals.
class MetricName:
    TASK_SUBMIT_TOTAL = "task_submit_total"
    TASK_DURATION_SECONDS = "task_duration_seconds"
    QUEUE_DEPTH = "queue_depth"
    QUEUE_WAIT_SECONDS = "queue_wait_seconds"
    STAGE_DURATION_SECONDS = "stage_duration_seconds"
    MODEL_LOAD_SECONDS = "model_load_seconds"
    MODEL_INFERENCE_SECONDS = "model_inference_seconds"
    GPU_MEMORY_BYTES = "gpu_memory_bytes"
    GPU_UTILIZATION_PERCENT = "gpu_utilization_percent"
    WORKER_RETRY_TOTAL = "worker_retry_total"
    ARTIFACT_IO_SECONDS = "artifact_io_seconds"


#: Every declared name, so a gate can ask whether the exposition can produce it
#: rather than whether the identifier is spelled somewhere in `src/`.
DECLARED = tuple(
    value for name, value in vars(MetricName).items()
    if not name.startswith("_") and isinstance(value, str)
)


__all__ = [
    "DECLARED",
    "SUBMISSION_SCOPED",
    "WORKER_PROCESS_ONLY",
    "MetricName",
    "MetricsRegistry",
    "metrics",
    "nearest_rank",
    "render_deployment_series",
]
