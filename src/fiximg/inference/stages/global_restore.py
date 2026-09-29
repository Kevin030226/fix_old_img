"""Global restoration stage — overall quality restoration only (plan §3.6/§3.7).

Scope, per the plan's stage contract::

    global_restore = 只负责全局退化恢复

It runs *only* the first legacy path. The face chain (detection → enhancement →
warp-back) is handled by dedicated stages, so it is never executed twice — the
semantic duplication §3.7 warns about is structurally impossible now.

Oversized inputs still go through the §18 ImageTiler: the restoration model is
invoked once per tile in a temporary directory and the outputs are
feather-blended back to full resolution. Face stages then operate on the merged
image, which is strictly better than enhancing each tile separately.
"""
import os
import time

from PIL import Image

from fiximg.domain.errors import PipelineFailedError
from fiximg.inference.backends.legacy_cli import (
    STAGE_RESTORE,
    restored_image_dir,
)
from fiximg.inference.context import StageContext, StageResult
from fiximg.inference.stages.base import BaseStage, legacy_pipeline_root


class GlobalRestoreStage(BaseStage):
    """Overall quality restoration (optionally with scratch repair)."""

    name = "global_restore"
    version = "2.0"
    capabilities = frozenset({"restore", "deblur", "denoise"})

    def __init__(self, with_scratch: bool = False) -> None:
        self.with_scratch = with_scratch
        if with_scratch:
            self.name = "scratch_repair"
            self.capabilities = frozenset({"restore", "scratch_repair"})

    # ------------------------------------------------------------------ helpers
    def _backend(self, context: StageContext, *, stem: str | None = None):
        """The restoration implementation for this run (plan §3.5.4).

        Native when the quality weights and torch are present and the switch is
        on; the subprocess adapter otherwise. Both expose ``run_folder`` /
        ``produced_dir``, so the stage and the tiling path do not care which one
        they got — but the metadata records it, so a fallback is visible.
        """
        from fiximg.inference.backends.registry import select_restore_backend

        return select_restore_backend(
            with_scratch=self.with_scratch,
            config={"stem": stem or context.task_id},
            stages=(STAGE_RESTORE,),
        )

    @staticmethod
    def _hr(context: StageContext) -> bool:
        return bool((context.options or {}).get("hr"))

    # ------------------------------------------------------------ tiling path
    def _run_tiled(self, image, context: StageContext, started: float) -> StageResult:
        """§18 tile/patch restoration for oversized inputs."""
        import shutil
        import tempfile

        from fiximg.inference.tiler import plan_tiles, process_tiled, tiling_enabled

        assert tiling_enabled(image)  # guarded by run()
        backend = self._backend(context)
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

                def _tile_progress(step, total, _label, _n=current_tile):
                    # Tiles run sequentially: the tile's own 0..1 progress fills
                    # this tile's slice of the stage's progress window.
                    context.report_progress(
                        (_n - 1 + step / max(total, 1)) / total_tiles,
                        f"tile {_n}/{total_tiles}",
                    )

                backend.run_folder(
                    tile_dir, tile_dir,
                    gpu=context.gpu, hr=self._hr(context), on_progress=_tile_progress,
                )
                out_path = os.path.join(restored_image_dir(tile_dir), stem + ".png")
                if not os.path.exists(out_path):
                    raise PipelineFailedError(
                        f"Tile {tile_index['n']} produced no output; check backend logs."
                    )
                return Image.open(out_path).convert("RGB")

            merged, info = process_tiled(image, _runner)
        finally:
            shutil.rmtree(work_root, ignore_errors=True)

        return StageResult(
            image=merged,
            metadata={
                "model": "Global/triplet-domain-translation",
                "backend": backend.implementation,
                "stage": "restore",
                "with_scratch": self.with_scratch,
                **info,
            },
            duration=time.perf_counter() - started,
            metrics={"tiles": total_tiles},
        )

    # ------------------------------------------------------------------- run
    def run(self, image, context: StageContext) -> StageResult:
        started = time.perf_counter()
        from fiximg.inference.tiler import tiling_enabled

        if tiling_enabled(image):
            return self._run_tiled(image, context, started)

        input_dir = context.path_in_run("input")
        os.makedirs(input_dir, exist_ok=True)
        stem = context.task_id
        image.save(os.path.join(input_dir, stem + ".png"))

        pipeline_root = legacy_pipeline_root(context)
        backend = self._backend(context, stem=stem)
        backend.run_folder(
            input_dir, pipeline_root,
            gpu=context.gpu, hr=self._hr(context),
            on_progress=lambda step, total, label: context.report_progress(
                step / max(total, 1), label
            ),
        )

        result_path = os.path.join(restored_image_dir(pipeline_root), stem + ".png")
        if not os.path.exists(result_path):
            raise PipelineFailedError(
                "Global restoration produced no output image; check backend logs."
            )

        return StageResult(
            image=Image.open(result_path).convert("RGB"),
            metadata={
                "model": "Global/triplet-domain-translation",
                "backend": backend.implementation,
                "stage": "restore",
                "with_scratch": self.with_scratch,
            },
            duration=time.perf_counter() - started,
            artifacts={"restored": result_path},
        )
