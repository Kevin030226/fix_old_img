"""BaseStage — the uniform interface every model capability implements (plan section 5)."""
from abc import ABC, abstractmethod

from app.inference.context import StageContext, StageResult


class BaseStage(ABC):
    """A single processing capability (restore, detect, enhance, colorize...).

    Stages receive an RGB PIL image plus a StageContext and return a StageResult.
    Heavy model handles come from context.model_manager so models are loaded once
    and reused across runs (plan section 6).
    """

    name: str = "base"

    @abstractmethod
    def run(self, image, context: StageContext) -> StageResult:
        """Process `image` and return a StageResult (never mutates the input)."""
        raise NotImplementedError
