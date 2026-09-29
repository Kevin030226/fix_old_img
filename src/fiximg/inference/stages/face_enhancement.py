"""Face detection / enhancement stages (plan §3.6/§3.7).

Scope, per the plan's stage contract::

    face_detection    = 只负责找脸 + landmarks
    face_enhancement  = 只负责 face crop enhancement

Both run exactly one legacy path through
:class:`~fiximg.inference.backends.legacy_cli.LegacyCliBackend`, reading and
writing the shared pipeline root. Because ``global_restore`` no longer runs the
face chain internally, these stages are the *only* place face work happens —
the double-processing §3.7 warns about cannot occur.
"""
import time

from fiximg.inference.backends.legacy_cli import (
    detection_dir,
    each_img_dir,
    list_images,
)
from fiximg.inference.context import StageContext, StageResult
from fiximg.inference.stages.base import BaseStage, legacy_pipeline_root

#: Reason recorded when the face chain is skipped because its dependency is
#: missing, and read back by ``warp_back`` so it does not spawn a stage-4
#: subprocess for a pipeline that produced no crops.
FACE_CHAIN_UNAVAILABLE = "face_dependencies_unavailable"


class FaceDetectionStage(BaseStage):
    """Detect and align faces with dlib 68-point landmarks."""

    name = "face_detection"
    version = "2.0"
    capabilities = frozenset({"face_detection"})

    @staticmethod
    def _hr(context: StageContext) -> bool:
        return bool((context.options or {}).get("hr"))

    def run(self, image, context: StageContext) -> StageResult:
        started = time.perf_counter()
        pipeline_root = legacy_pipeline_root(context)
        # Same choice as the restoration stage: resident dlib when it is enabled
        # and available, the subprocess adapter otherwise. Both take the shared
        # pipeline root, so the crop directory stage 3 reads is identical either
        # way and the stage does not know which one it got.
        from fiximg.inference.backends.registry import select_face_detection_backend
        from fiximg.inference.face_detect import dlib_importable

        if not dlib_importable():
            # The face chain cannot run at all on this install — dlib is its only
            # hard dependency, for the resident backend and for the vendored
            # subprocess alike. That used to fail the *whole* task three times
            # over, throwing away a restoration that had already succeeded,
            # because the only way to find out was the child's traceback.
            #
            # Skipping is the same degradation the pipeline already does for "no
            # faces in this photo" (stage 3 exits early, stage 4 passes the image
            # through) — the difference is that this one says why, in the result
            # rather than in a 500-shaped task.
            context.metadata["face_chain_skipped"] = FACE_CHAIN_UNAVAILABLE
            return StageResult(
                image=image,
                metadata={
                    "skipped": True,
                    "reason": FACE_CHAIN_UNAVAILABLE,
                    "backend": "unavailable",
                    "stage": "face_detect",
                    "face_count": 0,
                },
                duration=time.perf_counter() - started,
                metrics={"face_count": 0},
                message=(
                    "Face enhancement skipped: this installation has no dlib "
                    "(`pip install 'fiximg[gpu]'` builds it). The restoration "
                    "result is unaffected."
                ),
            )

        backend = select_face_detection_backend({"stem": context.task_id},
                                                hr=self._hr(context))
        backend.run_folder(
            context.path_in_run("input"),
            pipeline_root,
            gpu=context.gpu,
            hr=self._hr(context),
        )

        faces = list_images(detection_dir(pipeline_root))
        return StageResult(
            # Detection does not change the picture: the restored image flows on
            # untouched, and the crops travel as artifacts for the next stage.
            image=image,
            metadata={
                "model": "dlib-68",
                "backend": backend.implementation,
                "stage": "face_detect",
                "face_count": len(faces),
            },
            duration=time.perf_counter() - started,
            artifacts={"faces_dir": detection_dir(pipeline_root)},
            metrics={"face_count": len(faces)},
        )


class FaceEnhancementStage(BaseStage):
    """Enhance aligned face crops with the progressive face restoration model."""

    name = "face_enhancement"
    version = "2.0"
    capabilities = frozenset({"face_restore", "face_enhance"})

    @staticmethod
    def _hr(context: StageContext) -> bool:
        return bool((context.options or {}).get("hr"))

    def run(self, image, context: StageContext) -> StageResult:
        started = time.perf_counter()
        pipeline_root = legacy_pipeline_root(context)

        faces = list_images(detection_dir(pipeline_root))
        if not faces:
            # No aligned faces: the warp-back stage copies the restored image
            # through, so this is a legitimate skip rather than a failure.
            return StageResult(
                image=image,
                metadata={"skipped": True, "reason": "no_aligned_faces"},
                duration=time.perf_counter() - started,
                metrics={"enhanced_count": 0},
                message="No aligned faces found; face enhancement skipped.",
            )

        from fiximg.inference.backends.registry import select_face_enhancement_backend

        backend = select_face_enhancement_backend({"stem": context.task_id},
                                                   hr=self._hr(context))
        backend.run_folder(
            context.path_in_run("input"),
            pipeline_root,
            gpu=context.gpu,
            hr=self._hr(context),
        )

        enhanced = list_images(each_img_dir(pipeline_root))
        return StageResult(
            image=image,
            metadata={
                "model": "FaceSR_512" if self._hr(context) else "FaceSR_256",
                "backend": backend.implementation,
                "stage": "face_enhance",
                "enhanced_count": len(enhanced),
            },
            duration=time.perf_counter() - started,
            artifacts={"faces_dir": each_img_dir(pipeline_root)},
            metrics={"enhanced_count": len(enhanced)},
        )
