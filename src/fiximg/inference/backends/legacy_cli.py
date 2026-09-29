"""LegacyCliBackend — subprocess adapter over the stage-selectable CLI pipeline.

Plan §2.3 / §4.1 Step 5: the legacy Global/Face chain is *wrapped* rather than
rewritten. Stages talk to :class:`ModelBackend`; this class is the only place
that knows a subprocess is involved, so replacing it with native in-process
inference later is a one-file change.

Plan §3.6/§3.7: the CLI exposes its four legacy "paths" as selectable stages, so
each :class:`~fiximg.inference.stages.base.BaseStage` runs exactly one of them
and the face chain is never executed twice. A backend instance is therefore
configured with the stage it owns::

    LegacyCliBackend(stages=["1"])   # overall restoration
    LegacyCliBackend(stages=["2"])   # face detection
    LegacyCliBackend(stages=["3"])   # face enhancement
    LegacyCliBackend(stages=["4"])   # warp-back + finalize

All four share one ``output_dir``, which is how a stage invoked as its own
process still finds the previous stage's artefacts.

What it centralises (previously duplicated inside
:mod:`fiximg.inference.stages.global_restore`):

* spawning ``run.py`` with ``shell=False`` and a verified exit code,
* streaming stdout line by line and turning ``@@FIXIMG_PROGRESS`` markers into
  live progress updates (plan §24),
* resolving the produced output path and the degrade report.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable

from PIL import Image

from fiximg.domain.errors import ModelUnavailableError, PipelineFailedError
from fiximg.inference.backends.base import BaseModelBackend, ModelRequest, ModelResult
from fiximg.paths import PROJECT_ROOT

#: Marker emitted by ``fiximg.cli.batch`` on stdout.
PROGRESS_PREFIX = "@@FIXIMG_PROGRESS"

#: Where the vendored restoration chain keeps its networks. ``Global/test.py``
#: resolves these relative to its own cwd, which only the subprocess it spawns is
#: guaranteed to have; this module needs them absolute to pre-flight a run.
GLOBAL_CHECKPOINTS = os.path.join(PROJECT_ROOT, "Global", "checkpoints")
RESTORATION_DIR = os.path.join(GLOBAL_CHECKPOINTS, "restoration")

#: Networks actually loaded per branch of ``Global/test.py:72-90`` — the ones a
#: missing-file check has to know about. Discriminators and optimizers in the same
#: directories are training state and deliberately not listed.
BRANCH_WEIGHTS: dict[tuple[bool, bool], tuple[tuple[str, str], ...]] = {
    (False, False): (
        ("VAE_A_quality", "latest_net_G.pth"),
        ("VAE_B_quality", "latest_net_G.pth"),
        ("mapping_quality", "latest_net_mapping_net.pth"),
    ),
    (True, False): (
        ("VAE_A_quality", "latest_net_G.pth"),
        ("VAE_B_scratch", "latest_net_G.pth"),
        ("mapping_scratch", "latest_net_mapping_net.pth"),
    ),
    # ``--HR`` switches the mapping net to a different directory entirely.
    (True, True): (
        ("VAE_A_quality", "latest_net_G.pth"),
        ("VAE_B_scratch", "latest_net_G.pth"),
        ("mapping_Patch_Attention", "latest_net_mapping_net.pth"),
    ),
}


def branch_weights(with_scratch: bool, hr: bool) -> list[str]:
    """Absolute paths of the networks the selected branch loads."""
    return [
        os.path.join(RESTORATION_DIR, directory, filename)
        for directory, filename in BRANCH_WEIGHTS.get((bool(with_scratch), bool(hr)), ())
    ]


def missing_branch_weights(with_scratch: bool, hr: bool) -> list[str]:
    """Which of those files are not on disk."""
    return [path for path in branch_weights(with_scratch, hr) if not os.path.isfile(path)]

#: Stage ids and the directories each one produces inside the shared output root.
STAGE_RESTORE = "1"
STAGE_FACE_DETECT = "2"
STAGE_FACE_ENHANCE = "3"
STAGE_FINALIZE = "4"
ALL_STAGES = (STAGE_RESTORE, STAGE_FACE_DETECT, STAGE_FACE_ENHANCE, STAGE_FINALIZE)

RESTORED_IMAGE_DIR = os.path.join("stage_1_restore_output", "restored_image")
DETECTION_DIR = "stage_2_detection_output"
EACH_IMG_DIR = os.path.join("stage_3_face_output", "each_img")
FINAL_DIR = "final_output"
REPORT_NAME = "pipeline_report.json"

ProgressCallback = Callable[[int, int, str], None]


def parse_marker(line: str) -> tuple[int, int, str] | None:
    """Parse ``@@FIXIMG_PROGRESS 2/4 face detection`` → ``(2, 4, 'face detection')``."""
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


def stream_subprocess(
    args: list[str],
    *,
    cwd: str | None = None,
    on_progress: ProgressCallback | None = None,
    echo: bool = True,
) -> None:
    """Run a subprocess, streaming stdout and surfacing progress markers.

    Progress marker lines are consumed (not echoed); every other line is passed
    through so the backend log stays intact.
    """
    if args and args[0] == "python":
        args = [sys.executable, *args[1:]]
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
            if line.startswith(PROGRESS_PREFIX):
                parsed = parse_marker(line)
                if parsed is not None and on_progress is not None:
                    on_progress(*parsed)
                continue
            if echo:
                print(line, flush=True)
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
    if proc.wait() != 0:
        raise PipelineFailedError(
            f"Subprocess failed (exit={proc.returncode}): {' '.join(args)}"
        )


# ------------------------------------------------------------- path helpers
def restored_image_dir(output_dir: str) -> str:
    """Directory holding the stage-1 restored full images."""
    return os.path.join(output_dir, RESTORED_IMAGE_DIR)


def detection_dir(output_dir: str) -> str:
    """Directory holding the stage-2 aligned face crops."""
    return os.path.join(output_dir, DETECTION_DIR)


def each_img_dir(output_dir: str) -> str:
    """Directory holding the stage-3 enhanced face crops."""
    return os.path.join(output_dir, EACH_IMG_DIR)


def final_dir(output_dir: str) -> str:
    """Directory holding the stage-4 composited results."""
    return os.path.join(output_dir, FINAL_DIR)


def report_path(output_dir: str) -> str:
    """Path of the degrade report written by the finalize stage."""
    return os.path.join(output_dir, REPORT_NAME)


def read_report(output_dir: str) -> dict | None:
    """Read the degrade report, or None when the finalize stage did not run."""
    path = report_path(output_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def list_images(directory: str) -> list[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    if not os.path.isdir(directory):
        return []
    return sorted(n for n in os.listdir(directory) if n.lower().endswith(exts))


class LegacyCliBackend(BaseModelBackend):
    """Runs one selectable stage of the vendored legacy pipeline."""

    name = "global_restore"
    version = "1.0"
    implementation = "legacy-cli"
    capabilities = frozenset({"restore", "scratch_repair", "deblur", "denoise"})

    def __init__(self, config: dict | None = None, with_scratch: bool = False,
                 stages: tuple[str, ...] | list[str] | None = None) -> None:
        super().__init__(config)
        self.with_scratch = bool(with_scratch)
        self.stages: tuple[str, ...] = tuple(stages or ALL_STAGES)
        if self.with_scratch:
            self.name = "scratch_repair"
        self.cli_path = self.config.get("cli_path") or os.path.join(PROJECT_ROOT, "run.py")

    # ------------------------------------------------------------------ load
    def _do_load(self, device: str) -> None:
        """The CLI loads weights per invocation; only the entry point must exist."""
        if not os.path.exists(self.cli_path):
            raise ModelUnavailableError(
                f"Legacy CLI entry point not found: {self.cli_path}",
                details={"backend": self.name, "path": self.cli_path},
            )

    def health(self):  # noqa: D102 - see BaseModelBackend
        health = super().health()
        health.extra = {
            "cli_path": self.cli_path,
            "with_scratch": self.with_scratch,
            "stages": list(self.stages),
        }
        return health

    # ------------------------------------------------------------------ infer
    def run_folder(
        self,
        input_dir: str,
        output_dir: str,
        *,
        gpu: int = -1,
        hr: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> str:
        """Run the selected stages over ``input_dir``; return ``output_dir``.

        ``output_dir`` is the *shared* pipeline root: every stage reads the
        previous stage's from it, which is what makes splitting the pipeline into
        separate processes safe.

        Refuses to start when the restoration networks are *partially* present:
        the vendored loader prints "…not exists yet" and keeps going with randomly
        initialised weights, so the alternative is a task that "succeeds" and
        returns noise. A checkout with no ``Global/checkpoints`` at all is left
        alone — that is an uninstalled model, which the caller reports through
        availability, not a branch that was picked wrong.
        """
        if STAGE_RESTORE in self.stages and os.path.isdir(RESTORATION_DIR):
            missing = missing_branch_weights(self.with_scratch, hr)
            if missing:
                raise ModelUnavailableError(
                    "Global restoration weights are incomplete; refusing to run on "
                    "randomly initialised networks",
                    details={
                        "backend": self.name,
                        "with_scratch": self.with_scratch,
                        "hr": hr,
                        "missing": missing,
                        "hint": "python -m fiximg.cli.download_weights download "
                                "(or: make weights)",
                    },
                )

        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            "python", self.cli_path,
            "--input_folder", input_dir,
            "--output_folder", output_dir,
            "--GPU", str(gpu),
            "--stages", ",".join(self.stages),
        ]
        if self.with_scratch:
            cmd.append("--with_scratch")
        if hr:
            cmd.append("--HR")
        stream_subprocess(cmd, cwd=PROJECT_ROOT, on_progress=on_progress)
        return output_dir

    def produced_dir(self, output_dir: str) -> str:
        """Where the stage this backend owns writes its result."""
        stage = self.stages[-1] if self.stages else STAGE_RESTORE
        return {
            STAGE_RESTORE: restored_image_dir(output_dir),
            STAGE_FACE_DETECT: detection_dir(output_dir),
            STAGE_FACE_ENHANCE: each_img_dir(output_dir),
            STAGE_FINALIZE: final_dir(output_dir),
        }[stage]

    def _do_infer(self, request: ModelRequest) -> ModelResult:
        """Single-image convenience wrapper around :meth:`run_folder`."""
        started = time.perf_counter()
        work_dir = request.work_dir
        if not work_dir:
            raise PipelineFailedError("LegacyCliBackend requires a work_dir for intermediates")
        input_dir = os.path.join(work_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        stem = self.config.get("stem", "image")
        image_path = os.path.join(input_dir, f"{stem}.png")
        request.image.convert("RGB").save(image_path)

        output_dir = self.run_folder(
            input_dir,
            os.path.join(work_dir, "pipeline"),
            gpu=int(request.options.get("gpu", -1)),
            hr=bool(request.options.get("hr")),
        )
        result_path = os.path.join(self.produced_dir(output_dir), f"{stem}.png")
        if not os.path.exists(result_path):
            raise PipelineFailedError(
                f"Stage {self.stages} produced no output image; check backend logs."
            )
        metadata = {
            "model": "Global/triplet-domain-translation",
            "with_scratch": self.with_scratch,
            "stages": list(self.stages),
            "duration_s": round(time.perf_counter() - started, 3),
        }
        report = read_report(output_dir)
        if report is not None:
            metadata["report"] = report
        return ModelResult(
            image=Image.open(result_path).convert("RGB"),
            metadata=metadata,
            artifacts={"result": result_path},
        )


class WarpBackCliBackend(LegacyCliBackend):
    """Warp-back compositing only (legacy stage 4, plan §3.7).

    Registered as its own model so ``GET /api/v1/models`` reports the full
    legacy chain and operators can see which stage a failure came from.
    """

    name = "warp_back"
    version = "2.0"
    capabilities = frozenset({"warp_back", "face_composite"})

    def __init__(self, config: dict | None = None, **kwargs) -> None:
        kwargs.pop("stages", None)
        super().__init__(config, stages=(STAGE_FINALIZE,), **kwargs)


__all__ = [
    "ALL_STAGES",
    "BRANCH_WEIGHTS",
    "EACH_IMG_DIR",
    "FINAL_DIR",
    "GLOBAL_CHECKPOINTS",
    "LegacyCliBackend",
    "PROGRESS_PREFIX",
    "REPORT_NAME",
    "RESTORATION_DIR",
    "RESTORED_IMAGE_DIR",
    "STAGE_FACE_DETECT",
    "STAGE_FACE_ENHANCE",
    "STAGE_FINALIZE",
    "STAGE_RESTORE",
    "WarpBackCliBackend",
    "branch_weights",
    "detection_dir",
    "each_img_dir",
    "final_dir",
    "list_images",
    "missing_branch_weights",
    "parse_marker",
    "read_report",
    "report_path",
    "restored_image_dir",
    "stream_subprocess",
]
