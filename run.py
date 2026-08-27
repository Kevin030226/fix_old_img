"""四阶段推理流水线 CLI。

由 Web 层以子进程方式调用，也可命令行独立使用：
    python run.py --input_folder ./test_images/old --output_folder ./output
    python run.py --input_folder ./test_images/old_w_scratch --output_folder ./output --with_scratch
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class StageError(RuntimeError):
    """某个流水线阶段以非零退出码结束。"""


def run_cmd(args, cwd=None, stage=""):
    """执行子命令并校验退出码（shell=False 列表传参）。"""
    if args and args[0] == "python":
        args[0] = sys.executable
    try:
        completed = subprocess.run(args, shell=False, cwd=cwd)
    except FileNotFoundError as exc:
        raise StageError(f"阶段[{stage or args}]命令或解释器不可达: {args}") from exc
    if completed.returncode != 0:
        raise StageError(
            f"阶段[{stage or args}]执行失败，退出码={completed.returncode}\n命令: {' '.join(args)}"
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
    """把 'auto' 解析为 0（GPU 可用）或 -1（CPU）。"""
    if str(gpu_arg).lower() != "auto":
        return int(gpu_arg)
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:  # noqa: BLE001
        return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="./test_images/old")
    parser.add_argument("--output_folder", type=str, default="./output")
    parser.add_argument("--GPU", type=str, default="auto", help="auto / 0 / 1 / -1")
    parser.add_argument("--checkpoint_name", type=str, default="Setting_9_epoch_100")
    parser.add_argument("--with_scratch", action="store_true")
    parser.add_argument("--HR", action="store_true")
    opts = parser.parse_args()

    gpu = resolve_gpu(opts.GPU)
    main_environment = os.getcwd()
    opts.input_folder = os.path.abspath(opts.input_folder)
    opts.output_folder = os.path.abspath(opts.output_folder)
    os.makedirs(opts.output_folder, exist_ok=True)

    print("path 1/4: 整体质量提升")
    stage_1_output_dir = os.path.join(opts.output_folder, "stage_1_restore_output")
    os.makedirs(stage_1_output_dir, exist_ok=True)

    if opts.with_scratch:
        mask_dir = os.path.join(stage_1_output_dir, "masks")
        run_cmd(
            [
                "python", "detection.py",
                "--test_path", opts.input_folder,
                "--output_dir", mask_dir,
                "--input_size", "full_size",
                "--GPU", str(gpu),
            ],
            cwd=os.path.join(main_environment, "Global"),
            stage="1/4 划痕检测",
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
                "--outputs_dir", stage_1_output_dir,
                "--gpu_ids", str(gpu),
            ],
            cwd=os.path.join(main_environment, "Global"),
            stage="1/4 划痕修复+质量提升",
        )
    else:
        run_cmd(
            [
                "python", "test.py",
                "--test_mode", "Full",
                "--Quality_restore",
                "--test_input", opts.input_folder,
                "--outputs_dir", stage_1_output_dir,
                "--gpu_ids", str(gpu),
            ],
            cwd=os.path.join(main_environment, "Global"),
            stage="1/4 整体质量提升",
        )

    stage_1_results = os.path.join(stage_1_output_dir, "restored_image")
    stage_1_names = list_images(stage_1_results)
    if not stage_1_names:
        raise StageError("阶段1未产出任何修复图（restored_image 为空），流水线终止")
    print("path 1: success!\n")

    print("path 2/4: 人脸检测")
    stage_2_output_dir = os.path.join(opts.output_folder, "stage_2_detection_output")
    os.makedirs(stage_2_output_dir, exist_ok=True)
    detect_script = "detect_all_dlib_HR.py" if opts.HR else "detect_all_dlib.py"
    run_cmd(
        [
            "python", detect_script,
            "--url", stage_1_results,
            "--save_url", stage_2_output_dir,
        ],
        cwd=os.path.join(main_environment, "Face_Detection"),
        stage="2/4 人脸检测",
    )
    print("path 2: success!\n")

    detected_faces = list_images(stage_2_output_dir)
    degrade_reason = None

    if not detected_faces:
        print("未检测到人脸，跳过面部增强，直接采用整体修复结果\n")
        degrade_reason = "no_face_detected"
    else:
        print("path 3/4: 面部增强")
        stage_3_output_dir = os.path.join(opts.output_folder, "stage_3_face_output")
        os.makedirs(stage_3_output_dir, exist_ok=True)
        checkpoint = "FaceSR_512" if opts.HR else opts.checkpoint_name
        size_args = (
            ["--load_size", "512", "--batchSize", "1"]
            if opts.HR
            else ["--load_size", "256", "--batchSize", "4"]
        )
        run_cmd(
            [
                "python", "test_face.py",
                "--old_face_folder", stage_2_output_dir,
                "--old_face_label_folder", "./",
                "--tensorboard_log",
                "--name", checkpoint,
                "--gpu_ids", str(gpu),
                *size_args,
                "--label_nc", "18",
                "--no_instance",
                "--preprocess_mode", "resize",
                "--results_dir", stage_3_output_dir,
                "--no_parsing_map",
            ],
            cwd=os.path.join(main_environment, "Face_Enhancement"),
            stage="3/4 面部增强",
        )
        print("path 3: success!\n")

        print("path 4/4: 回卷变换")
        stage_4_output_dir = os.path.join(opts.output_folder, "final_output")
        os.makedirs(stage_4_output_dir, exist_ok=True)
        warp_script = (
            "align_warp_back_multiple_dlib_HR.py"
            if opts.HR
            else "align_warp_back_multiple_dlib.py"
        )
        run_cmd(
            [
                "python", warp_script,
                "--origin_url", stage_1_results,
                "--replace_url", os.path.join(stage_3_output_dir, "each_img"),
                "--save_url", stage_4_output_dir,
            ],
            cwd=os.path.join(main_environment, "Face_Detection"),
            stage="4/4 回卷变换",
        )
        print("path 4: success! Please check the result image!\n")
        degrade_reason = "face_enhance_missing"

    os.chdir(main_environment)
    stage_4_output_dir = os.path.join(opts.output_folder, "final_output")
    os.makedirs(stage_4_output_dir, exist_ok=True)
    produced = set(list_images(stage_4_output_dir))
    enhanced, degraded = [], []
    for name in stage_1_names:
        if name in produced:
            enhanced.append(name)
        else:
            shutil.copy(
                os.path.join(stage_1_results, name),
                os.path.join(stage_4_output_dir, name),
            )
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
    report_path = os.path.join(opts.output_folder, "pipeline_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if degraded:
        print(
            f"[降级] {len(degraded)}/{len(stage_1_names)} 张图未完成面部增强"
            f"（原因: {degrade_reason}），已回退为整体修复结果: {degraded}"
        )
    print(f"流水线报告: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except StageError as exc:
        print(f"\n[流水线失败] {exc}", file=sys.stderr)
        sys.exit(2)

