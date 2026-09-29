"""DDColor colorization stage (plan §5) — in-process inference via ModelManager."""
import time

from fiximg.domain.errors import PipelineFailedError
from fiximg.inference.context import StageContext, StageResult
from fiximg.inference.stages.base import BaseStage


class ColorizationStage(BaseStage):
    """Colorize a grayscale/RGB image with the lazily-loaded DDColor model."""

    name = "colorization"
    version = "1.0"
    capabilities = frozenset({"colorize"})

    def run(self, image, context: StageContext) -> StageResult:
        """Colorize through the §5.4 backend contract, on the routed device.

        This used to reach past the contract: it leased the pipeline out of the model
        manager and ran the RGB→BGR→model→RGB dance itself. Two things were wrong with
        that. The device was whatever the manager happened to have loaded, not the
        `context.gpu` the runtime scheduled (the other three chains pass theirs); and
        the declared precision policy never wrapped the call, so a manifest asking for
        `inference_mode`/autocast got neither on the only path that runs in production —
        the policy applied solely to `DDColorBackend.infer`, which nothing called.
        """
        from fiximg.inference.backends.base import ModelRequest
        from fiximg.inference.backends.ddcolor import DDColorBackend

        started = time.perf_counter()
        device = str(context.gpu)
        backend = DDColorBackend(manager=context.model_manager)
        # §24: DDColor is lazily loaded, so the first call spends several seconds
        # on the checkpoint before any inference happens. Report both phases so
        # the bar does not sit at 0% for the whole stage.
        context.report_progress(0.05, "loading DDColor model")
        backend.load(device)
        context.report_progress(0.6, "colorizing")
        result = backend.infer(ModelRequest(image=image, device=device))
        context.report_progress(1.0, "colorized")

        if result.image is None:
            raise PipelineFailedError(
                "DDColor returned no image",
                details={"stage": self.name, "device": device},
            )
        # Where the model actually ran, which is not always where it was scheduled:
        # asking for card 1 on a single-card host is a decline, and the report has to
        # say so rather than echo the request.
        metadata = dict(result.metadata or {})
        ran_on = str(metadata.get("device") or device)
        metadata["device"] = ran_on
        if ran_on != device:
            metadata["device_requested"] = device
        return StageResult(
            image=result.image,
            metadata=metadata,
            duration=time.perf_counter() - started,
        )
