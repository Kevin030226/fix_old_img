"""Regression coverage for the tiled-restore to face-chain handoff.

The tiled branch used to return the merged image only in memory. The face
stages read their source from the shared legacy pipeline tree, so a large
restore failed at face detection before either face model could run.
"""

import json
import os

import pytest
from PIL import Image

from fiximg.inference.backends.legacy_cli import (
    detection_dir,
    each_img_dir,
    final_dir,
    list_images,
    report_path,
    restored_image_dir,
)
from fiximg.inference.context import StageContext
from fiximg.inference.planner import PipelinePlanner
from fiximg.inference.size_policy import TILE, decide_from_settings
from fiximg.inference.stages.base import legacy_pipeline_root
from fiximg.inference.stages.face_enhancement import (
    FaceDetectionStage,
    FaceEnhancementStage,
)
from fiximg.inference.stages.global_restore import GlobalRestoreStage
from fiximg.inference.stages.warp_back import WarpBackStage


class _TileBackend:
    """Minimal stage-1 backend that materialises each tile's output."""

    implementation = "test-tile-backend"

    def run_folder(self, input_dir, output_dir, *, gpu=-1, hr=False, on_progress=None):
        target = restored_image_dir(output_dir)
        os.makedirs(target, exist_ok=True)
        for name in os.listdir(input_dir):
            if name.lower().endswith(".png"):
                Image.open(os.path.join(input_dir, name)).save(os.path.join(target, name))
        return output_dir


@pytest.mark.parametrize("width", [2049, 4096])
def test_tiled_restore_hands_the_merged_image_to_the_face_chain(monkeypatch, tmp_path, width):
    from fiximg.config import settings
    from fiximg.inference import face_detect
    from fiximg.inference.backends import legacy_cli

    monkeypatch.setattr(settings, "tile_size", 1536, raising=False)
    monkeypatch.setattr(settings, "tile_overlap", 128, raising=False)
    monkeypatch.setattr(settings, "size_resize_max", 2048, raising=False)
    monkeypatch.setattr(settings, "size_tile_max", 4096, raising=False)
    monkeypatch.setattr(settings, "native_tree", "none", raising=False)
    monkeypatch.setattr(settings, "face_detect_native", False, raising=False)
    monkeypatch.setattr(face_detect, "dlib_importable", lambda: True)

    image = Image.new("RGB", (width, 1600), "orange")
    assert decide_from_settings(image.size).strategy == TILE

    plan = PipelinePlanner().plan("restore", {"face_enhance": True})
    assert [name for name, _ in plan.stages] == [
        "global_restore",
        "face_detection",
        "face_enhancement",
        "warp_back",
    ]

    context = StageContext(
        task_id="tiled-task",
        run_dir=str(tmp_path / "run"),
        options={"face_enhance": True},
    )
    restore_stage = GlobalRestoreStage()
    restore_stage._backend = lambda ctx, stem=None: _TileBackend()
    restored = restore_stage.run(image, context)

    root = legacy_pipeline_root(context)
    persisted = os.path.join(restored_image_dir(root), "tiled-task.png")
    assert restored.metadata["tile_count"] > 1
    assert restored.artifacts["restored"] == persisted
    assert os.path.isfile(persisted)
    with Image.open(persisted) as saved:
        assert saved.size == image.size
        assert saved.getpixel((0, 0)) == restored.image.getpixel((0, 0))

    calls = []

    def fake_stream(args, *, cwd=None, on_progress=None, echo=True):
        calls.append(args)
        output_root = args[args.index("--output_folder") + 1]
        stages = args[args.index("--stages") + 1].split(",")

        if "2" in stages:
            assert list_images(restored_image_dir(output_root)), (
                "tiled stage 1 did not persist the image required by face detection"
            )
            target = detection_dir(output_root)
            os.makedirs(target, exist_ok=True)
            Image.new("RGB", (256, 256), "blue").save(os.path.join(target, "tiled-task_1.png"))

        if "3" in stages:
            target = each_img_dir(output_root)
            os.makedirs(target, exist_ok=True)
            Image.new("RGB", (256, 256), "green").save(os.path.join(target, "tiled-task_1.png"))

        if "4" in stages:
            target = final_dir(output_root)
            os.makedirs(target, exist_ok=True)
            Image.new("RGB", image.size, "yellow").save(os.path.join(target, "tiled-task.png"))
            with open(report_path(output_root), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "total": 1,
                        "enhanced_count": 1,
                        "degraded_count": 0,
                        "degraded": [],
                        "degrade_reason": None,
                    },
                    handle,
                )

    monkeypatch.setattr(legacy_cli, "stream_subprocess", fake_stream)

    detected = FaceDetectionStage().run(restored.image, context)
    enhanced = FaceEnhancementStage().run(detected.image, context)
    composited = WarpBackStage().run(enhanced.image, context)

    assert detected.metrics["face_count"] == 1
    assert enhanced.metrics["enhanced_count"] == 1
    assert composited.metrics["enhanced_count"] == 1
    assert [args[args.index("--stages") + 1] for args in calls] == ["2", "3", "4"]
