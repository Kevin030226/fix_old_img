"""Warp-back stage — composite enhanced faces onto the restored image (plan §3.7).

Scope, per the plan's stage contract::

    warp_back = 只负责将 face result 合成回原图

This is the stage that produces the user-visible result of a restoration with
face enhancement, and the one that owns the degrade report: when no face was
detected (or none could be enhanced) the restored image is copied through
unchanged and the reason is recorded, so the caller can tell "restored" from
"restored and face-enhanced".
"""
import os
import time

from PIL import Image

from fiximg.domain.errors import PipelineFailedError
from fiximg.inference.backends.legacy_cli import (
    STAGE_FINALIZE,
    LegacyCliBackend,
    detection_dir,
    each_img_dir,
    final_dir,
    list_images,
    read_report,
)
from fiximg.inference.context import StageContext, StageResult
from fiximg.inference.stages.base import BaseStage, legacy_pipeline_root


class WarpBackStage(BaseStage):
    """Composite the enhanced face crops back onto the restored image."""

    name = "warp_back"
    version = "2.0"
    capabilities = frozenset({"warp_back", "face_composite"})

    @staticmethod
    def _hr(context: StageContext) -> bool:
        return bool((context.options or {}).get("hr"))

    def run(self, image, context: StageContext) -> StageResult:
        started = time.perf_counter()
        pipeline_root = legacy_pipeline_root(context)
        stem = context.task_id

        skipped = context.metadata.get("face_chain_skipped")
        if skipped:
            # Nothing was aligned and nothing was enhanced, so there is no crop
            # grid to paste back — the restored image *is* the final image.
            # Running the stage-4 subprocess here would either fail on the same
            # missing dependency or rewrite the result for no reason.
            return StageResult(
                image=image,
                duration=time.perf_counter() - started,
                metadata={
                    "skipped": True,
                    "reason": skipped,
                    "backend": "none",
                    "stage": "warp_back",
                    "face_count": 0,
                    "enhanced_count": 0,
                },
                metrics={"face_count": 0, "enhanced_count": 0, "degraded_count": 0},
                message="Face chain skipped; returning the restored image directly.",
            )

        backend = LegacyCliBackend({"stem": stem}, stages=(STAGE_FINALIZE,))
        backend.run_folder(
            context.path_in_run("input"),
            pipeline_root,
            gpu=context.gpu,
            hr=self._hr(context),
        )

        result_path = os.path.join(final_dir(pipeline_root), stem + ".png")
        if not os.path.exists(result_path):
            raise PipelineFailedError(
                "Warp-back produced no output image; check backend logs."
            )

        report = read_report(pipeline_root) or {}
        degraded = bool(report.get("degraded_count"))
        degrade_reason = report.get("degrade_reason") if degraded else None

        return StageResult(
            image=Image.open(result_path).convert("RGB"),
            metadata={
                "model": "dlib-68-warp",
                "backend": "legacy-cli",
                "stage": "warp_back",
                "face_count": len(list_images(detection_dir(pipeline_root))),
                "enhanced_count": len(list_images(each_img_dir(pipeline_root))),
                "degraded": degraded,
                "report": report,
            },
            duration=time.perf_counter() - started,
            artifacts={"final": result_path},
            metrics={
                "enhanced_count": int(report.get("enhanced_count") or 0),
                "degraded_count": int(report.get("degraded_count") or 0),
            },
            message=(
                f"Face enhancement did not complete for {report.get('degraded_count')} "
                f"image(s) ({degrade_reason}); fell back to the restored image."
                if degraded
                else None
            ),
        )
