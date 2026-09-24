"""DDColor colorization stage (plan section 5) — in-process inference via ModelManager."""
import time

import numpy as np

from app.inference.context import StageContext, StageResult
from app.inference.stages.base import BaseStage


class ColorizationStage(BaseStage):
    """Colorize a grayscale/RGB image with the lazily-loaded DDColor model."""

    name = "colorization"

    def run(self, image, context: StageContext) -> StageResult:
        started = time.perf_counter()
        import cv2
        from PIL import Image

        manager = context.model_manager
        # §24: DDColor is lazily loaded, so the first call spends several seconds
        # on the checkpoint before any inference happens. Report both phases so
        # the bar does not sit at 0% for the whole stage.
        context.report_progress(0.05, "loading DDColor model")
        pipeline = manager.get("ddcolor")  # loads on first use (hybrid strategy)

        context.report_progress(0.6, "colorizing")
        rgb = np.array(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out_bgr = pipeline.process(bgr)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        result = Image.fromarray(out_rgb)
        context.report_progress(1.0, "colorized")

        return StageResult(
            image=result,
            metadata={"model": f"DDColor-{context.model_manager.settings.ddcolor_model_size}"},
            duration=time.perf_counter() - started,
        )
