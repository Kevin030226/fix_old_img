"""Latency benchmark (plan §3.10.3).

Measures end-to-end pipeline latency per input size, separating the **cold**
first run (model load + warmup) from **warm** runs, and reports
min/p50/p95/p99/max. In synthetic mode it also *gates*: the stage does a known
`time.sleep`, so whatever sits above the sleep is platform overhead (queue claim,
stage dispatch, artifact rows, logging), and that number is what a regression
would move.

    python -m benchmark.latency                 # synthetic stage (any machine)
    python -m benchmark.latency --runs 5 --sizes small,medium
    python -m benchmark.latency --max-overhead 0.5
    python -m benchmark.latency --real          # real model chain (needs weights)

Exit code 1 means the overhead ceiling was exceeded — until this flag existed the
command always returned 0, so CI's "benchmark regression" step could not fail and
was a printout rather than a check.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.common import (  # noqa: E402
    INPUT_SIZES,
    Samples,
    ensure_importable,
    make_image,
    print_table,
    synthetic_stage,
    temp_database,
)

ensure_importable()

#: Ceiling for platform overhead per task, in seconds, on the smallest measured
#: size. A ceiling rather than an expected value because it has to mean the same
#: thing on a CI runner and on a GPU box: absolute milliseconds do not.
DEFAULT_MAX_OVERHEAD_SECONDS = 1.0

#: Committed measurements from a reference platform (plan §3.10.3). The gate above
#: says whether the platform cost is acceptable; this says what "normal" looked like
#: when someone last measured it, so a reviewer comparing two machines has a number
#: to compare against instead of prose in a document.
BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline.json")


def load_baseline(path: str | None = None) -> dict | None:
    """The committed baseline, or None when it is missing or unreadable.

    The path resolves at call time rather than as a default argument, so a caller
    (or a test) pointing it elsewhere actually changes what is read.
    """
    import json

    target = path or BASELINE_PATH
    try:
        with open(target, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def baseline_report(rows: list[dict], baseline: dict | None = None) -> str:
    """Compare this run against the recorded baseline, one line per size."""
    if baseline is None:
        baseline = load_baseline(BASELINE_PATH)
    if not baseline:
        return "baseline: benchmark/baseline.json not found"

    recorded = baseline.get("sizes") or {}
    lines = [
        f"baseline ({baseline.get('reference_platform', 'unknown platform')}, "
        f"measured {baseline.get('measured_at', '?')}):"
    ]
    for row in rows:
        name = row.get("size")
        saved = (recorded.get(name) or {}).get("p50_s")
        if not saved:
            lines.append(f"  {name:>6}: no recorded number")
            continue
        ratio = (row["p50"] / saved) if saved and row.get("p50") is not None else None
        ratio_text = f"x{ratio:.2f}" if ratio else "n/a"
        lines.append(
            f"  {name:>6}: p50 {row.get('p50')}s vs recorded {saved}s -> {ratio_text}"
        )
    return "\n".join(lines)


def overhead_report(rows: list[dict], stage_delay: float, limit: float) -> tuple[str, bool]:
    """(report, passed) — how far each size's p50 sits above the simulated work.

    Only the smallest size is gated. Image preparation (`numpy.random` + PIL) is
    part of the measured span and scales with pixels, so a ceiling that fits a
    512 px run would false-alarm on a 4096 px one for reasons that have nothing
    to do with the platform.
    """
    if not rows:
        return "no measurements; nothing gated", True
    lines = []
    passed = True
    smallest = min(int(row["px"]) for row in rows)
    for row in sorted(rows, key=lambda r: int(r["px"])):
        overhead = float(row["p50"]) - float(stage_delay)
        gated = int(row["px"]) == smallest
        verdict = "" if not gated else ("" if overhead <= limit else "  <-- OVER LIMIT")
        if gated and overhead > limit:
            passed = False
        lines.append(
            f"overhead {row['size']:>7}: {overhead:+.4f}s above the "
            f"{stage_delay:.3f}s stage sleep{verdict}"
        )
    lines.append(f"gated on the smallest size against {limit:.3f}s")
    return "\n".join(lines), passed


def _build_orchestrator(real: bool, stage_delay: float):
    """Return (orchestrator, plan) for either the real chain or a synthetic one."""
    from types import SimpleNamespace

    from fiximg.inference.runtime import PipelineOrchestrator

    if real:
        return PipelineOrchestrator(), None

    stage = synthetic_stage(stage_delay)
    planner = SimpleNamespace(
        plan=lambda task_type, options=None: SimpleNamespace(
            task_type=task_type, stages=[(stage.name, {})], decisions={}
        ),
        build_stage=lambda name, kwargs: stage,
    )
    return PipelineOrchestrator(planner=planner), None


def _measure(orchestrator, size: int, runs: int, real: bool) -> tuple[float, Samples]:
    """Run the pipeline `runs` times and collect durations."""
    from fiximg.infrastructure.db.repositories import task_repository as task_repo

    samples = Samples(name=f"{size}px")
    cold = 0.0
    for index in range(runs):
        image = make_image(size)
        task_id = f"bench-{size}-{index}"
        started = time.perf_counter()
        if real:
            orchestrator.run(image, {"username": "bench"}, "restore", options=None)
        else:
            task_repo.create_task(task_id, "synthetic", "bench")
            task_repo.start_task(task_id)
            from fiximg.inference.context import StageContext

            context = StageContext(
                task_id=task_id,
                user="bench",
                task_type="synthetic",
                run_dir=os.path.join(os.getcwd(), "storage", "bench", task_id),
                model_manager=orchestrator.model_manager,
                gpu=-1,
            )
            os.makedirs(context.run_dir, exist_ok=True)
            plan = orchestrator.planner.plan("synthetic")
            orchestrator._run_plan(task_id, image, plan, context)
        elapsed = time.perf_counter() - started
        samples.add(elapsed)
        if index == 0:
            cold = elapsed
    return cold, samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pipeline latency benchmark")
    parser.add_argument("--runs", type=int, default=3, help="runs per input size")
    parser.add_argument(
        "--sizes",
        default="small,medium",
        help=f"comma-separated subset of {','.join(INPUT_SIZES)}",
    )
    parser.add_argument("--stage-delay", type=float, default=0.05,
                        help="simulated per-stage work in the synthetic mode")
    parser.add_argument("--real", action="store_true",
                        help="use the real model chain (requires weights)")
    parser.add_argument(
        "--max-overhead", type=float, default=DEFAULT_MAX_OVERHEAD_SECONDS,
        help="fail (exit 1) if platform overhead per task on the smallest size "
             f"exceeds this many seconds (default {DEFAULT_MAX_OVERHEAD_SECONDS}; "
             "ignored with --real, where the stage work is not a known sleep)",
    )
    args = parser.parse_args(argv)

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip() in INPUT_SIZES]
    if not sizes:
        print(f"No valid sizes; choose from {', '.join(INPUT_SIZES)}", file=sys.stderr)
        return 2

    print(f"mode: {'real model chain' if args.real else 'synthetic stage'}")
    print(f"runs per size: {args.runs}\n")

    rows = []
    with temp_database():
        orchestrator, _ = _build_orchestrator(args.real, args.stage_delay)
        for name in sizes:
            side = INPUT_SIZES[name]
            cold, samples = _measure(orchestrator, side, args.runs, args.real)
            summary = samples.summary()
            summary.update({"size": name, "px": side, "cold": round(cold, 4)})
            rows.append(summary)

    print_table(rows, ["size", "px", "count", "cold", "p50", "p95", "p99", "max", "mean"])
    print("\ncold = first run (includes any model load); p50/p95/p99 are warm samples.")

    if args.real:
        # The stage work is a real model now, so "p50 minus the sleep" measures
        # nothing meaningful; a real-model regression belongs to the golden tests.
        # Deliberately a report, not a gate: there is no committed baseline for
        # real-model latency on CPU, and a ceiling invented without one would either
        # never fire or fire on every runner. The byte-level regression net for this
        # path is the golden-image suite (tests/gpu), which compares outputs.
        print("overhead gate: skipped (--real) — report only; the real-model "
              "regression gate is the golden image suite in tests/gpu")
        return 0
    print(baseline_report(rows))
    report, passed = overhead_report(rows, args.stage_delay, args.max_overhead)
    print()
    print(report)
    if not passed:
        print(
            "\nREGRESSION: platform overhead per task exceeded the ceiling. "
            "Find the added work (queue claim, artifact writes, logging, "
            "stage dispatch) before raising --max-overhead.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
