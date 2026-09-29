"""Memory benchmark (plan §3.10.3).

Reports peak RSS (and CUDA peak when available) per input size, which is how the
"GPU peak memory per task/stage" goal in §7.1 gets a baseline.

    python -m benchmark.memory
    python -m benchmark.memory --sizes small,medium,large --real
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.common import (  # noqa: E402
    INPUT_SIZES,
    ensure_importable,
    gpu_peak_mb,
    make_image,
    peak_rss_mb,
    print_table,
    synthetic_stage,
    temp_database,
)

ensure_importable()


def _run_once(image, delay: float, real: bool) -> None:
    from types import SimpleNamespace

    from fiximg.inference.context import StageContext
    from fiximg.inference.runtime import PipelineOrchestrator

    if real:
        PipelineOrchestrator().run(image, {"username": "bench"}, "restore", options=None)
        return

    stage = synthetic_stage(delay)
    planner = SimpleNamespace(build_stage=lambda name, kwargs: stage)
    orchestrator = PipelineOrchestrator(planner=planner)
    plan = SimpleNamespace(stages=[(stage.name, {})])
    context = StageContext(task_id="mem", user="bench", gpu=-1)
    orchestrator._run_plan("mem", image, plan, context)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pipeline memory benchmark")
    parser.add_argument("--sizes", default="small,medium,large")
    parser.add_argument("--stage-delay", type=float, default=0.0)
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args(argv)

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip() in INPUT_SIZES]
    if not sizes:
        print(f"No valid sizes; choose from {', '.join(INPUT_SIZES)}", file=sys.stderr)
        return 2

    rows = []
    with temp_database():
        for name in sizes:
            side = INPUT_SIZES[name]
            image = make_image(side)
            gc.collect()
            _run_once(image, args.stage_delay, args.real)
            gc.collect()
            rows.append(
                {
                    "size": name,
                    "px": side,
                    "peak_rss_mb": peak_rss_mb(),
                    "peak_gpu_mb": gpu_peak_mb(),
                }
            )

    print_table(rows, ["size", "px", "peak_rss_mb", "peak_gpu_mb"])
    print(
        "\nPeak values are process-wide high-water marks, so a run with a larger "
        "input masks smaller ones — pass a single --sizes value for isolation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
