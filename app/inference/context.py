"""StageContext / StageResult — the unified data contracts flowing through stages."""
from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass
class StageContext:
    """Per-run context passed to every stage.

    Attributes:
        task_id: request/run identifier, used for artifact paths and logging.
        user: username that triggered the run (or "unknown").
        task_type: planner task type, e.g. "restore".
        run_dir: artifact root for this run (input/ stages/ output/).
        options: free-form task options (hr, face_enhance, auto_colorize...).
        model_manager: shared ModelManager so stages can fetch loaded models.
        gpu: resolved GPU id (0, 1, ... or -1 for CPU).
        progress_cb: optional callable(fraction: float, message: str|None) set by
            the orchestrator; a stage calls report_progress() to move the bar
            *inside* its own run (long subprocess-backed stages otherwise only
            update progress at stage boundaries).
    """

    task_id: str
    user: str = "unknown"
    task_type: str = "restore"
    run_dir: str | None = None
    options: dict = field(default_factory=dict)
    model_manager: Any = None
    gpu: int = -1
    metadata: dict = field(default_factory=dict)
    progress_cb: Any = field(default=None, repr=False)

    def report_progress(self, fraction: float, message: str | None = None) -> None:
        """Report intra-stage progress in [0, 1] (plan §24).

        The orchestrator maps the value onto this stage's slice of the global
        progress bar. Never raises: a progress report must not fail a run.
        """
        cb = self.progress_cb
        if cb is None:
            return
        try:
            cb(max(0.0, min(1.0, float(fraction))), message)
        except Exception:  # noqa: BLE001 — progress is best-effort only
            pass


@dataclass
class StageResult:
    """Uniform output of every stage.

    image is always a PIL RGB Image; artifacts are extra files saved for the
    report (masks, per-stage snapshots); metadata feeds the run report.
    """

    image: Image.Image | None
    metadata: dict = field(default_factory=dict)
    duration: float = 0.0
    artifacts: dict = field(default_factory=dict)
    message: str | None = None
