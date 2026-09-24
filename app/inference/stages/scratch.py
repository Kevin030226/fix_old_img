"""Scratch detection stage (V1 subprocess adapter, plan sections 5 and 7).

Runs Global/detection.py and returns the scratch mask as the stage image.
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
GLOBAL_DIR = os.path.join(BASE_DIR, "Global")


def _run_cmd(args, cwd=None):
    if args and args[0] == "python":
        args[0] = sys.executable
    proc = subprocess.run(args, shell=False, cwd=cwd)
    if proc.returncode != 0:
        raise PipelineFailedError(
            f"Subprocess failed (exit={proc.returncode}): {' '.join(args)}"
        )


class ScratchDetectionStage(BaseStage):
    """Detect scratches and output a binary mask (white = scratches)."""

    name = "scratch_detection"

    def run(self, image, context: StageContext) -> StageResult:
        started = time.perf_counter()
        input_dir = os.path.join(context.run_dir, "stages", self.name, "input")
        output_dir = os.path.join(context.run_dir, "stages", self.name, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        stem = context.task_id
        image.save(os.path.join(input_dir, stem + ".png"))

        _run_cmd(
            [
                "python", "detection.py",
                "--test_path", input_dir,
                "--output_dir", output_dir,
                "--input_size", "full_size",
                "--GPU", str(context.gpu),
            ],
            cwd=GLOBAL_DIR,
        )

        mask_path = os.path.join(output_dir, "mask", stem + ".png")
        if not os.path.exists(mask_path):
            raise PipelineFailedError(
                "Scratch detection produced no mask; check backend logs."
            )
        mask = Image.open(mask_path).convert("RGB")

        return StageResult(
            image=mask,
            metadata={"model": "Global/scratch-detection", "mask_path": mask_path},
            duration=time.perf_counter() - started,
            artifacts={"mask": mask_path},
        )
