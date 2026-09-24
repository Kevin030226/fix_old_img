"""Global restoration stage (V1 subprocess adapter, plan sections 5 and 7).

Wraps Global/test.py (quality restoration, optionally with scratch masks) via
run.py, exactly as V1 did. Replaced by native inference in a later phase.
"""
import os
import subprocess
import sys
import time

from PIL import Image

from app.core.config import settings
from app.core.exceptions import PipelineFailedError
from app.inference.context import StageContext, StageResult
from app.inference.stages.base import BaseStage

BASE_DIR = settings.base_dir

#: Marker emitted by run.py on stdout (see run.py PROGRESS_PREFIX).
_PROGRESS_PREFIX = "@@FIXIMG_PROGRESS"


def _parse_marker(line: str):
    """Parse '@@FIXIMG_PROGRESS 2/4 face detection' -> (2, 4, 'face detection')."""
    parts = line.strip().split(None, 2)
    if len(parts) < 2:
        return None
    try:
        step, total = (int(v) for v in parts[1].split("/", 1))
    except ValueError:
        return None
    if total <= 0:
        return None
    return step, total, (parts[2] if len(parts) > 2 else "")


def _run_cmd(args, cwd=None, on_progress=None):
    """Run a subprocess with an argument list (shell=False) and verify the exit code.

    stdout/stderr are streamed line by line instead of inherited so that
    progress markers from run.py can be turned into live progress updates
    (plan §24). Non-marker lines are echoed to keep the backend log intact.
    """
    if args and args[0] == "python":
        args[0] = sys.executable
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.Popen(
        args,
        shell=False,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    try:
        for raw in proc.stdout:  # type: ignore[union-attr]
            line = raw.rstrip("\r\n")
            if line.startswith(_PROGRESS_PREFIX):
                if on_progress is not None:
                    parsed = _parse_marker(line)
                    if parsed is not None:
                        on_progress(*parsed)
                continue
            print(line, flush=True)
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
    if proc.wait() != 0:
        raise PipelineFailedError(
            f"Subprocess failed (exit={proc.returncode}): {' '.join(args)}"
        )


class GlobalRestoreStage(BaseStage):
    """Overall quality restoration via the Global pipeline (optionally with scratch repair).

    Images larger than FIXIMG_TILE_SIZE run through the §18 ImageTiler: the
    legacy model is invoked once per tile in a temporary directory and the
    outputs are feather-blended back to full resolution.
    """

    name = "global_restore"

    def __init__(self, with_scratch: bool = False) -> None:
        self.with_scratch = with_scratch
        if with_scratch:
            self.name = "scratch_repair"

    # ------------------------------------------------------------ tiling path
    def _run_tiled(self, image, context: StageContext, started: float) -> StageResult:
        """§18 tile/patch inference for oversized inputs."""
        import shutil
        import tempfile

        from app.inference.tiler import plan_tiles, process_tiled, tiling_enabled

        assert tiling_enabled(image)  # guarded by run()
        work_root = tempfile.mkdtemp(prefix="fiximg_tiles_")
        try:
            tile_index = {"n": 0}
            total_tiles = max(len(plan_tiles(image.width, image.height)), 1)

            def _runner(tile_image):
                tile_index["n"] += 1
                current_tile = tile_index["n"]
                tile_dir = os.path.join(work_root, f"tile_{current_tile:03d}")
                os.makedirs(tile_dir, exist_ok=True)
                stem = f"{context.task_id}_t{current_tile:03d}"
                tile_image.save(os.path.join(tile_dir, stem + ".png"))
                cmd = [
                    "python", os.path.join(BASE_DIR, "run.py"),
                    "--input_folder", tile_dir,
                    "--output_folder", tile_dir,
                    "--GPU", str(context.gpu),
                ]
                if self.with_scratch:
                    cmd.append("--with_scratch")
                if (context.options or {}).get("hr"):
                    cmd.append("--HR")

                def _tile_progress(step, total, _label, _n=current_tile):
                    # Tiles run sequentially: the tile's own 0..1 progress fills
                    # this tile's slice of the stage's progress window.
                    context.report_progress(
                        (_n - 1 + step / max(total, 1)) / total_tiles,
                        f"tile {_n}/{total_tiles}",
                    )

                _run_cmd(cmd, cwd=BASE_DIR, on_progress=_tile_progress)
                out_path = os.path.join(tile_dir, "final_output", stem + ".png")
                if not os.path.exists(out_path):
                    raise PipelineFailedError(
                        f"Tile {tile_index['n']} produced no output; check backend logs."
                    )
                return Image.open(out_path).convert("RGB")

            merged, info = process_tiled(image, _runner)
        finally:
            shutil.rmtree(work_root, ignore_errors=True)

        metadata = {
            "model": "Global/triplet-domain-translation",
            "with_scratch": self.with_scratch,
            **info,
        }
        return StageResult(
            image=merged,
            metadata=metadata,
            duration=time.perf_counter() - started,
            artifacts={},
        )

    def run(self, image, context: StageContext) -> StageResult:
        started = time.perf_counter()
        # §18: oversized inputs go through the ImageTiler instead of the
        # single-shot path (which is capped by GPU memory).
        from app.inference.tiler import tiling_enabled

        if tiling_enabled(image):
            return self._run_tiled(image, context, started)

        input_dir = os.path.join(context.run_dir, "input")
        output_dir = os.path.join(context.run_dir, "stages", self.name)
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        stem = context.task_id
        image.save(os.path.join(input_dir, stem + ".png"))

        cmd = [
            "python", os.path.join(BASE_DIR, "run.py"),
            "--input_folder", input_dir,
            "--output_folder", output_dir,
            "--GPU", str(context.gpu),
        ]
        if self.with_scratch:
            cmd.append("--with_scratch")
        # Task options (plan section 12): {"hr": true} enables the high-res
        # face-enhancement path of the legacy CLI.
        if (context.options or {}).get("hr"):
            cmd.append("--HR")

        def _report(step, total, label):
            context.report_progress(step / max(total, 1), label)

        _run_cmd(cmd, cwd=BASE_DIR, on_progress=_report)

        result_path = os.path.join(output_dir, "final_output", stem + ".png")
        if not os.path.exists(result_path):
            raise PipelineFailedError(
                "Global restoration produced no output image; check backend logs."
            )
        result = Image.open(result_path).convert("RGB")

        metadata = {"model": "Global/triplet-domain-translation", "with_scratch": self.with_scratch}
        # Surface the degrade report produced by run.py (no face detected etc.).
        report_path = os.path.join(output_dir, "pipeline_report.json")
        if os.path.exists(report_path):
            import json

            try:
                with open(report_path, encoding="utf-8") as f:
                    metadata["report"] = json.load(f)
            except (OSError, ValueError):
                pass

        return StageResult(
            image=result,
            metadata=metadata,
            duration=time.perf_counter() - started,
            artifacts={"restored": result_path},
        )
