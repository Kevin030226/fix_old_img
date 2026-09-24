"""Face detection / enhancement / warp-back stages (V1 subprocess adapters).

V1 runs these as one run.py invocation together with global restoration; in V2
they are separate stages for observability, but the restore flow still invokes
run.py once (which performs all four sub-steps internally) and this module only
provides the standalone stage classes used by the planner for explicit runs.
"""
import os
import subprocess
import sys
import time


from app.core.config import settings
from app.core.exceptions import PipelineFailedError
from app.inference.context import StageContext, StageResult
from app.inference.stages.base import BaseStage

BASE_DIR = settings.base_dir
FACE_DETECTION_DIR = os.path.join(BASE_DIR, "Face_Detection")
FACE_ENHANCEMENT_DIR = os.path.join(BASE_DIR, "Face_Enhancement")


def _run_cmd(args, cwd=None):
    if args and args[0] == "python":
        args[0] = sys.executable
    proc = subprocess.run(args, shell=False, cwd=cwd)
    if proc.returncode != 0:
        raise PipelineFailedError(
            f"Subprocess failed (exit={proc.returncode}): {' '.join(args)}"
        )


class FaceDetectionStage(BaseStage):
    """Detect and align faces with dlib 68-point landmarks."""

    name = "face_detection"

    def run(self, image, context: StageContext) -> StageResult:
        started = time.perf_counter()
        input_dir = os.path.join(context.run_dir, "stages", self.name, "input")
        output_dir = os.path.join(context.run_dir, "stages", self.name, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        stem = context.task_id
        image.save(os.path.join(input_dir, stem + ".png"))

        _run_cmd(
            ["python", "detect_all_dlib.py", "--url", input_dir, "--save_url", output_dir],
            cwd=FACE_DETECTION_DIR,
        )

        faces = [
            n for n in os.listdir(output_dir)
            if n.lower().endswith((".png", ".jpg", ".jpeg"))
        ] if os.path.isdir(output_dir) else []

        return StageResult(
            image=image,
            metadata={"face_count": len(faces), "model": "dlib-68"},
            duration=time.perf_counter() - started,
            artifacts={"faces_dir": output_dir},
        )


class FaceEnhancementStage(BaseStage):
    """Enhance aligned face crops with the progressive face restoration model."""

    name = "face_enhancement"

    def run(self, image, context: StageContext) -> StageResult:
        started = time.perf_counter()
        faces_dir = (context.metadata or {}).get("face_detection", {}).get("faces_dir")
        if not faces_dir or not os.path.isdir(faces_dir):
            return StageResult(
                image=image,
                metadata={"skipped": True, "reason": "no_aligned_faces"},
                duration=time.perf_counter() - started,
                message="No aligned faces found; face enhancement skipped.",
            )

        output_dir = os.path.join(context.run_dir, "stages", self.name)
        os.makedirs(output_dir, exist_ok=True)

        _run_cmd(
            [
                "python", "test_face.py",
                "--old_face_folder", faces_dir,
                "--old_face_label_folder", "./",
                "--tensorboard_log",
                "--name", "Setting_9_epoch_100",
                "--gpu_ids", str(context.gpu),
                "--load_size", "256", "--batchSize", "4",
                "--label_nc", "18", "--no_instance",
                "--preprocess_mode", "resize",
                "--results_dir", output_dir,
                "--no_parsing_map",
            ],
            cwd=FACE_ENHANCEMENT_DIR,
        )

        each_img = os.path.join(output_dir, "each_img")
        enhanced = (
            [n for n in os.listdir(each_img) if n.lower().endswith((".png", ".jpg"))]
            if os.path.isdir(each_img) else []
        )

        return StageResult(
            image=image,
            metadata={"enhanced_count": len(enhanced), "model": "FaceSR_256"},
            duration=time.perf_counter() - started,
            artifacts={"faces_dir": each_img},
        )
