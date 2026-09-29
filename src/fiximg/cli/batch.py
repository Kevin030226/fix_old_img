"""Legacy inference pipeline CLI, split into independently runnable stages.

Originally ``run.py`` at the repository root, which ran all four legacy "paths"
in one invocation. That made a single stage responsible for four different jobs
— exactly the semantic duplication plan §3.7 warns about:

    path 1  overall quality restoration
    path 2  face detection
    path 3  face enhancement
    path 4  warp-back compositing

V3 exposes them as *selectable stages* so the planner can compose them
explicitly and never run the face chain twice::

    python -m fiximg.cli.batch --input_folder IN --output_folder OUT --stages 1
    python -m fiximg.cli.batch --input_folder IN --output_folder OUT --stages 2
    ...
    python -m fiximg.cli.batch --input_folder IN --output_folder OUT   # 1,2,3,4

Every stage shares one ``--output_folder``, so the chain still works when the
stages are invoked as separate processes: each one reads the previous stage's
directory from the same tree. Omitting ``--stages`` keeps the original
single-invocation behaviour for external callers.

Progress markers (``@@FIXIMG_PROGRESS <step>/<total> <label>``) are unchanged;
the total reflects the selected stages, so a single-stage run reports 1/1.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

#: Machine-readable progress marker consumed by LegacyCliBackend, which streams
#: this process' stdout and maps each step onto the task progress bar.
#: Format: "<PREFIX> <step>/<total> <label>" — kept out of the human log.
PROGRESS_PREFIX = "@@FIXIMG_PROGRESS"

#: Stage identifiers, in execution order.
STAGE_RESTORE = "1"
STAGE_FACE_DETECT = "2"
STAGE_FACE_ENHANCE = "3"
STAGE_FINALIZE = "4"

ALL_STAGES = (STAGE_RESTORE, STAGE_FACE_DETECT, STAGE_FACE_ENHANCE, STAGE_FINALIZE)

#: Human labels, also used for the progress marker text.
STAGE_LABELS = {
    STAGE_RESTORE: "overall quality restoration",
    STAGE_FACE_DETECT: "face detection",
    STAGE_FACE_ENHANCE: "face enhancement",
    STAGE_FINALIZE: "warp-back transformation",
}

#: Directory names inside ``--output_folder`` (shared by every stage).
RESTORE_DIR = "stage_1_restore_output"
RESTORED_IMAGE_DIR = "restored_image"
DETECTION_DIR = "stage_2_detection_output"
FACE_DIR = "stage_3_face_output"
EACH_IMG_DIR = "each_img"
FINAL_DIR = "final_output"
REPORT_NAME = "pipeline_report.json"


class StageError(RuntimeError):
    """Raised when a pipeline stage exits with a non-zero code."""


def parse_stages(raw: str | None) -> list[str]:
    """Parse ``"1,3"`` into ``["1", "3"]``, preserving the canonical order."""
    if not raw:
        return list(ALL_STAGES)
    requested = {token.strip() for token in str(raw).split(",") if token.strip()}
    unknown = requested - set(ALL_STAGES)
    if unknown:
        raise StageError(f"Unknown stage(s): {', '.join(sorted(unknown))}")
    return [stage for stage in ALL_STAGES if stage in requested]


def emit_progress(step, total, label):
    """Announce pipeline progress to the parent process (see PROGRESS_PREFIX)."""
    print(f"{PROGRESS_PREFIX} {step}/{total} {label}", flush=True)


def run_cmd(args, cwd=None, stage=""):
    """Run a subcommand and verify the exit code (shell=False, list arguments)."""
    if args and args[0] == "python":
        args[0] = sys.executable
    try:
        completed = subprocess.run(args, shell=False, cwd=cwd)
    except FileNotFoundError as exc:
        raise StageError(f"Stage [{stage or args}] command or interpreter unreachable: {args}") from exc
    if completed.returncode != 0:
        raise StageError(
            f"Stage [{stage or args}] failed with exit code {completed.returncode}\nCommand: {' '.join(args)}"
        )
    return completed.returncode


def list_images(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(
        n
        for n in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, n)) and n.lower().endswith(IMAGE_EXTS)
    )


def resolve_gpu(gpu_arg):
    """Resolve 'auto' to 0 (GPU available) or -1 (CPU)."""
    if str(gpu_arg).lower() != "auto":
        return int(gpu_arg)
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:  # noqa: BLE001
        return -1


# --------------------------------------------------------------------- stage 1
def run_restore(opts, gpu: int) -> str:
    """Overall quality restoration (optionally with scratch repair).

    Returns the directory holding the restored full images.
    """
    output_dir = os.path.join(opts.output_folder, RESTORE_DIR)
    os.makedirs(output_dir, exist_ok=True)

    if opts.with_scratch:
        mask_dir = os.path.join(output_dir, "masks")
        run_cmd(
            [
                "python", "detection.py",
                "--test_path", opts.input_folder,
                "--output_dir", mask_dir,
                "--input_size", "full_size",
                "--GPU", str(gpu),
            ],
            cwd=os.path.join(opts.cwd, "Global"),
            stage="scratch detection",
        )
        scratch_args = ["--Scratch_and_Quality_restore"]
        if opts.HR:
            scratch_args.append("--HR")
        run_cmd(
            [
                "python", "test.py",
                *scratch_args,
                "--test_input", os.path.join(mask_dir, "input"),
                "--test_mask", os.path.join(mask_dir, "mask"),
                "--outputs_dir", output_dir,
                "--gpu_ids", str(gpu),
            ],
            cwd=os.path.join(opts.cwd, "Global"),
            stage="scratch repair + quality restoration",
        )
    else:
        run_cmd(
            [
                "python", "test.py",
                "--test_mode", "Full",
                "--Quality_restore",
                "--test_input", opts.input_folder,
                "--outputs_dir", output_dir,
                "--gpu_ids", str(gpu),
            ],
            cwd=os.path.join(opts.cwd, "Global"),
            stage="overall quality restoration",
        )

    restored = os.path.join(output_dir, RESTORED_IMAGE_DIR)
    if not list_images(restored):
        raise StageError(
            "Restoration produced no image (restored_image is empty); pipeline aborted"
        )
    return restored


# --------------------------------------------------------------------- stage 2
def run_face_detection(opts, gpu: int) -> str:
    """Detect and align faces; returns the detection output directory."""
    restored = os.path.join(opts.output_folder, RESTORE_DIR, RESTORED_IMAGE_DIR)
    if not list_images(restored):
        raise StageError(
            f"Face detection requires the restoration output: {restored} is empty. "
            "Run stage 1 first (or select both stages)."
        )
    output_dir = os.path.join(opts.output_folder, DETECTION_DIR)
    os.makedirs(output_dir, exist_ok=True)
    script = "detect_all_dlib_HR.py" if opts.HR else "detect_all_dlib.py"
    run_cmd(
        ["python", script, "--url", restored, "--save_url", output_dir],
        cwd=os.path.join(opts.cwd, "Face_Detection"),
        stage="face detection",
    )
    return output_dir


# --------------------------------------------------------------------- stage 3
def run_face_enhancement(opts, gpu: int) -> str:
    """Enhance the aligned face crops; returns the ``each_img`` directory."""
    faces_dir = os.path.join(opts.output_folder, DETECTION_DIR)
    if not list_images(faces_dir):
        # No face detected: nothing to enhance. The finalize stage copies the
        # restoration output through, so this is not an error.
        return os.path.join(opts.output_folder, FACE_DIR, EACH_IMG_DIR)

    output_dir = os.path.join(opts.output_folder, FACE_DIR)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = "FaceSR_512" if opts.HR else opts.checkpoint_name
    size_args = (
        ["--load_size", "512", "--batchSize", "1"]
        if opts.HR
        else ["--load_size", "256", "--batchSize", "4"]
    )
    run_cmd(
        [
            "python", "test_face.py",
            "--old_face_folder", faces_dir,
            "--old_face_label_folder", "./",
            "--tensorboard_log",
            "--name", checkpoint,
            "--gpu_ids", str(gpu),
            *size_args,
            "--label_nc", "18",
            "--no_instance",
            "--preprocess_mode", "resize",
            "--results_dir", output_dir,
            "--no_parsing_map",
        ],
        cwd=os.path.join(opts.cwd, "Face_Enhancement"),
        stage="face enhancement",
    )
    return os.path.join(output_dir, EACH_IMG_DIR)


# --------------------------------------------------------------------- stage 4
def run_finalize(opts, gpu: int) -> str:
    """Warp enhanced faces back onto the restored image and write the report.

    Also handles the "no face detected" degrade path: images whose faces were
    never enhanced are copied from the restoration output, so ``final_output``
    always contains one file per input.
    """
    restored = os.path.join(opts.output_folder, RESTORE_DIR, RESTORED_IMAGE_DIR)
    stage_1_names = list_images(restored)
    if not stage_1_names:
        raise StageError(
            f"Finalize requires the restoration output: {restored} is empty. "
            "Run stage 1 first (or select both stages)."
        )

    final_dir = os.path.join(opts.output_folder, FINAL_DIR)
    os.makedirs(final_dir, exist_ok=True)

    detected = list_images(os.path.join(opts.output_folder, DETECTION_DIR))
    each_img = os.path.join(opts.output_folder, FACE_DIR, EACH_IMG_DIR)
    enhanced_crops = list_images(each_img)

    degrade_reason = None
    if not detected:
        degrade_reason = "no_face_detected"
    elif not enhanced_crops:
        degrade_reason = "face_enhance_missing"
    else:
        warp_script = (
            "align_warp_back_multiple_dlib_HR.py"
            if opts.HR
            else "align_warp_back_multiple_dlib.py"
        )
        run_cmd(
            [
                "python", warp_script,
                "--origin_url", restored,
                "--replace_url", each_img,
                "--save_url", final_dir,
            ],
            cwd=os.path.join(opts.cwd, "Face_Detection"),
            stage="warp-back transformation",
        )

    produced = set(list_images(final_dir))
    enhanced, degraded = [], []
    for name in stage_1_names:
        if name in produced:
            enhanced.append(name)
        else:
            shutil.copy(os.path.join(restored, name), os.path.join(final_dir, name))
            degraded.append(name)

    report = {
        "total": len(stage_1_names),
        "enhanced_count": len(enhanced),
        "degraded_count": len(degraded),
        "enhanced": enhanced,
        "degraded": degraded,
        "degrade_reason": degrade_reason if degraded else None,
        "all_degraded": bool(degraded) and not enhanced,
        "gpu": gpu,
    }
    report_path = os.path.join(opts.output_folder, REPORT_NAME)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if degraded:
        print(
            f"[Degraded] {len(degraded)}/{len(stage_1_names)} image(s) did not complete face enhancement"
            f"(reason: {degrade_reason}); fell back to overall restoration result: {degraded}"
        )
    print(f"Pipeline report: {report_path}")
    return final_dir


# ------------------------------------------------------------------ dispatcher
#: stage id -> callable(opts, gpu) -> produced directory.
_STAGE_RUNNERS = {
    STAGE_RESTORE: run_restore,
    STAGE_FACE_DETECT: run_face_detection,
    STAGE_FACE_ENHANCE: run_face_enhancement,
    STAGE_FINALIZE: run_finalize,
}


def run_stages(opts, stages: list[str], gpu: int) -> dict:
    """Execute the selected stages in order; returns ``{stage: directory}``."""
    produced: dict = {}
    total = max(len(stages), 1)
    for index, stage in enumerate(stages, start=1):
        label = STAGE_LABELS[stage]
        print(f"[{index}/{total}] {label}")
        emit_progress(index, total, label)
        produced[stage] = _STAGE_RUNNERS[stage](opts, gpu)
        print(f"[{index}/{total}] {label}: success\n")
    return produced


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="./test_images/old")
    parser.add_argument("--output_folder", type=str, default="./output")
    parser.add_argument("--GPU", type=str, default="auto", help="auto / 0 / 1 / -1")
    parser.add_argument("--checkpoint_name", type=str, default="Setting_9_epoch_100")
    parser.add_argument("--with_scratch", action="store_true")
    parser.add_argument("--HR", action="store_true")
    parser.add_argument(
        "--stages", type=str, default="",
        help="comma-separated subset of 1,2,3,4 (default: all four)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    opts = build_parser().parse_args(argv)
    stages = parse_stages(opts.stages)

    gpu = resolve_gpu(opts.GPU)
    opts.cwd = os.getcwd()
    opts.input_folder = os.path.abspath(opts.input_folder)
    opts.output_folder = os.path.abspath(opts.output_folder)
    os.makedirs(opts.output_folder, exist_ok=True)

    try:
        run_stages(opts, stages, gpu)
    finally:
        # Some legacy scripts chdir; restore the caller's working directory.
        os.chdir(opts.cwd)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except StageError as exc:
        print(f"\n[Pipeline failed] {exc}", file=sys.stderr)
        sys.exit(2)
