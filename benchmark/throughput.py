"""Throughput benchmark (plan §3.10.3).

Answers two questions the plan cares about:

1. **Enqueue throughput** — how fast tasks can be submitted (HTTP-independent:
   the service + repository path).
2. **Queue drain throughput** — how long N tasks take with a worker consuming
   them, which is where the GPU concurrency policy (§2.4) shows up.

    python -m benchmark.throughput --tasks 50
    python -m benchmark.throughput --tasks 50 --workers 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.common import (  # noqa: E402
    ensure_importable,
    print_table,
    temp_database,
)

ensure_importable()


class _CountingOrchestrator:
    """Stand-in orchestrator that records executions without running a model."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.executed: list[str] = []

    def execute_queued(self, task_id: str, task_type: str) -> None:
        if self.delay:
            time.sleep(self.delay)
        self.executed.append(task_id)
        from fiximg.infrastructure.db.repositories import task_repository as task_repo

        task_repo.finish_task(task_id, "", None, int(self.delay * 1000))


def _enqueue(count: int) -> float:
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    started = time.perf_counter()
    for index in range(count):
        task_repo.create_task(f"tp-{index:05d}", "synthetic", "bench")
    return time.perf_counter() - started


def _drain(count: int, delay: float, workers: int, timeout: float) -> tuple[float, int]:
    from fiximg.inference.worker import PipelineWorker

    orchestrator = _CountingOrchestrator(delay)
    pool = [
        PipelineWorker(
            orchestrator,
            poll_seconds=0.01,
            worker_id=f"bench-{i}",
            lease_seconds=30.0,
        )
        for i in range(max(1, workers))
    ]
    started = time.perf_counter()
    for worker in pool:
        worker.start()
    try:
        while len(orchestrator.executed) < count:
            if time.perf_counter() - started > timeout:
                break
            time.sleep(0.01)
    finally:
        for worker in pool:
            worker.stop()
    return time.perf_counter() - started, len(orchestrator.executed)


def _report_queue_wait() -> None:
    """Print the worker's own queue-wait percentiles (acceptance "queue wait 基线").

    Deliberately the *production* instrument rather than a benchmark-local
    stopwatch: `PipelineWorker` observes `queue_wait_seconds` for every task it
    claims, so this is the same number `/api/v1/stats/metrics` publishes.

    Reported, not gated — with one consumer the wait of the Nth task is a function
    of `--tasks × --stage-delay`, so a ceiling here would measure the command line
    instead of the platform. A slow *claim* path is what would move the p50 at
    fixed arguments.
    """
    from fiximg.infrastructure.observability.metrics import MetricName, metrics

    summary = metrics.snapshot()["histograms"].get(str(MetricName.QUEUE_WAIT_SECONDS)) or {}
    count = summary.get("count") or 0
    if not count:
        print("\nqueue wait: not measured")
        return
    print(
        f"\nqueue wait (worker-measured, n={count}): "
        f"p50 {summary['p50'] * 1000:.0f} ms · "
        f"p95 {summary['p95'] * 1000:.0f} ms · "
        f"avg {summary['avg'] * 1000:.0f} ms"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Queue throughput benchmark")
    parser.add_argument("--tasks", type=int, default=25)
    parser.add_argument("--stage-delay", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    rows = []
    with temp_database():
        enqueue_seconds = _enqueue(args.tasks)
        rows.append(
            {
                "phase": "enqueue",
                "tasks": args.tasks,
                "executed": args.tasks,
                "workers": "-",
                "seconds": round(enqueue_seconds, 4),
                "per_task_ms": round(enqueue_seconds / args.tasks * 1000, 3),
                "tasks_per_s": round(args.tasks / enqueue_seconds, 1) if enqueue_seconds else None,
            }
        )

        drain_seconds, done = _drain(
            args.tasks, args.stage_delay, args.workers, args.timeout
        )
        rows.append(
            {
                # `tasks` is what was submitted and `executed` what came back.
                # They used to be the same number (`tasks: done`), so a drain that
                # timed out at 3 of 25 printed a rate as if all 25 had run.
                "phase": "drain",
                "tasks": args.tasks,
                "executed": done,
                "workers": args.workers,
                "seconds": round(drain_seconds, 4),
                "per_task_ms": round(drain_seconds / done * 1000, 3) if done else None,
                "tasks_per_s": round(done / drain_seconds, 1) if drain_seconds else None,
            }
        )

    print_table(
        rows,
        ["phase", "tasks", "executed", "workers", "seconds", "per_task_ms", "tasks_per_s"],
    )
    _report_queue_wait()
    print(
        "\nNote: the default policy serialises GPU work (concurrency=1). Raise "
        "FIXIMG_CONCURRENCY_<CAPABILITY> to let independent stages overlap."
    )

    # A run that stops with 1 of 4 tasks executed still prints a plausible rate,
    # which is the most misleading number this file can produce. Every submitted
    # task has to come back out, or the run failed.
    undrained = [r for r in rows if int(r["executed"]) < int(r["tasks"])]
    if undrained:
        print("\nDRAIN GATE FAILED — workers did not consume everything:", file=sys.stderr)
        for row in undrained:
            print(f"  {row['phase']} (workers={row['workers']}): "
                  f"{row['executed']}/{row['tasks']} within the timeout", file=sys.stderr)
        return 1
    print("\ndrain gate: every submitted task was executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
