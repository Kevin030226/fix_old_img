"""Legacy chain contract tests (plan 搂3.10.1 Inference Test matrix).

The report's matrix asks for Global / Scratch / Face / DDColor coverage. DDColor
has a GPU test; the legacy Global + Face chain is folder-based subprocess code
that no test touched, so a flag change could silently break it. These tests pin:

* the command line each stage builds (the flags are the contract with the
  vendored scripts),
* which output directory each stage reads and writes,
* how the degrade report is parsed,
* the progress-marker streaming that feeds the UI,
* that each stage runs exactly one legacy path (plan 搂3.6/搂3.7).

No weights or GPU are needed: ``stream_subprocess`` is replaced with a fake that
materialises the directories the real scripts would have produced.
"""
import os

import pytest

from fiximg.domain.errors import ModelUnavailableError, PipelineFailedError
from fiximg.inference.backends.legacy_cli import (
    ALL_STAGES,
    STAGE_FACE_DETECT,
    STAGE_FACE_ENHANCE,
    STAGE_FINALIZE,
    STAGE_RESTORE,
    LegacyCliBackend,
    WarpBackCliBackend,
    detection_dir,
    each_img_dir,
    final_dir,
    list_images,
    parse_marker,
    read_report,
    report_path,
    restored_image_dir,
)
from fiximg.inference.stages.base import legacy_pipeline_root


class _Recorder:
    """Captures the argv/cwd of every spawned subprocess."""

    def __init__(self, on_run=None) -> None:
        self.calls: list[tuple[list[str], str | None]] = []
        self._on_run = on_run

    def __call__(self, args, *, cwd=None, on_progress=None, echo=True):
        self.calls.append((list(args), cwd))
        if self._on_run is not None:
            self._on_run(args, cwd)
        return None

    @property
    def last(self) -> list[str]:
        return self.calls[-1][0]

    def flags(self) -> set[str]:
        return {a for a in self.last if a.startswith("--")}

    def value_of(self, flag: str) -> str | None:
        args = self.last
        return args[args.index(flag) + 1] if flag in args else None


@pytest.fixture()
def recorder(monkeypatch):
    """Replace the subprocess runner with a recorder that writes fake outputs."""
    from fiximg.inference.backends import legacy_cli

    def _install(on_run=None) -> _Recorder:
        rec = _Recorder(on_run=on_run)
        monkeypatch.setattr(legacy_cli, "stream_subprocess", rec)
        return rec

    return _install


def _write_png(path: str, color="red") -> str:
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)
    return path


# ------------------------------------------------------------- stage selection
def test_backend_defaults_to_every_stage():
    assert LegacyCliBackend().stages == ALL_STAGES


def test_each_stage_backend_runs_only_its_own_path(recorder, tmp_path):
    rec = recorder()
    for stage in ALL_STAGES:
        LegacyCliBackend(stages=(stage,)).run_folder(
            str(tmp_path / "in"), str(tmp_path / "out"))
    assert [r[0][r[0].index("--stages") + 1] for r in rec.calls] == list(ALL_STAGES)


def test_warp_back_backend_owns_the_finalize_stage():
    backend = WarpBackCliBackend()
    assert backend.stages == (STAGE_FINALIZE,)
    assert backend.name == "warp_back"
    assert "warp_back" in backend.capabilities


def test_restore_backend_passes_with_scratch(recorder, tmp_path):
    rec = recorder()
    LegacyCliBackend(with_scratch=True, stages=(STAGE_RESTORE,)).run_folder(
        str(tmp_path / "in"), str(tmp_path / "out"))
    assert "--with_scratch" in rec.flags()
    assert rec.value_of("--stages") == STAGE_RESTORE


def test_scratch_backend_renames_itself():
    backend = LegacyCliBackend(with_scratch=True, stages=(STAGE_RESTORE,))
    assert backend.name == "scratch_repair"
    assert "scratch_repair" in backend.capabilities


def test_hr_flag_is_forwarded(recorder, tmp_path):
    rec = recorder()
    LegacyCliBackend(stages=(STAGE_RESTORE,)).run_folder(
        str(tmp_path / "in"), str(tmp_path / "out"), hr=True)
    assert "--HR" in rec.flags()


def test_gpu_id_is_forwarded(recorder, tmp_path):
    rec = recorder()
    LegacyCliBackend(stages=(STAGE_RESTORE,)).run_folder(
        str(tmp_path / "in"), str(tmp_path / "out"), gpu=3)
    assert rec.value_of("--GPU") == "3"


def test_run_folder_returns_the_shared_root(recorder, tmp_path):
    recorder()
    out = LegacyCliBackend(stages=(STAGE_RESTORE,)).run_folder(
        str(tmp_path / "in"), str(tmp_path / "out"))
    assert out == str(tmp_path / "out")


def test_run_folder_creates_the_output_root(recorder, tmp_path):
    recorder()
    target = str(tmp_path / "nested" / "pipeline")
    LegacyCliBackend(stages=(STAGE_RESTORE,)).run_folder("/in", target)
    assert os.path.isdir(target)


def test_produced_dir_matches_the_stage(recorder, tmp_path):
    recorder()
    root = str(tmp_path)
    assert LegacyCliBackend(stages=(STAGE_RESTORE,)).produced_dir(root) == \
        restored_image_dir(root)
    assert LegacyCliBackend(stages=(STAGE_FACE_DETECT,)).produced_dir(root) == \
        detection_dir(root)
    assert LegacyCliBackend(stages=(STAGE_FACE_ENHANCE,)).produced_dir(root) == \
        each_img_dir(root)
    assert LegacyCliBackend(stages=(STAGE_FINALIZE,)).produced_dir(root) == final_dir(root)


# ----------------------------------------------------------------- path layout
def test_stage_directories_match_the_legacy_scripts(tmp_path):
    """These names are the contract with the vendored scripts; do not rename."""
    root = str(tmp_path)
    assert restored_image_dir(root).endswith(os.path.join("stage_1_restore_output",
                                                          "restored_image"))
    assert detection_dir(root).endswith("stage_2_detection_output")
    assert each_img_dir(root).endswith(os.path.join("stage_3_face_output", "each_img"))
    assert final_dir(root).endswith("final_output")
    assert report_path(root).endswith("pipeline_report.json")


def test_list_images_filters_by_extension(tmp_path):
    _write_png(str(tmp_path / "a.png"))
    (tmp_path / "b.txt").write_text("x", encoding="utf-8")
    assert list_images(str(tmp_path)) == ["a.png"]
    assert list_images(str(tmp_path / "missing")) == []


# ------------------------------------------------------------------ load/health
def test_load_fails_when_the_cli_is_missing(tmp_path):
    backend = LegacyCliBackend({"cli_path": str(tmp_path / "nope.py")})
    with pytest.raises(ModelUnavailableError):
        backend.load("cpu")


def test_health_reports_the_owned_stage(tmp_path):
    cli = tmp_path / "run.py"
    cli.write_text("# stand-in for the CLI entry point\n", encoding="utf-8")
    backend = LegacyCliBackend({"cli_path": str(cli)}, stages=(STAGE_FACE_DETECT,))
    health = backend.health()
    assert health.extra.get("stages") == [STAGE_FACE_DETECT]
    assert health.extra.get("cli_path") == str(cli)
    # `to_dict` flattens `extra`, which is what the models API returns.
    assert health.to_dict()["stages"] == [STAGE_FACE_DETECT]


# ------------------------------------------------------------- progress markers
def test_parse_marker_reads_step_total_and_label():
    assert parse_marker("@@FIXIMG_PROGRESS 2/4 face detection") == (2, 4, "face detection")


def test_parse_marker_rejects_non_markers():
    assert parse_marker("") is None
    assert parse_marker("@@FIXIMG_PROGRESS") is None
    assert parse_marker("@@FIXIMG_PROGRESS not-a-fraction label") is None
    assert parse_marker("@@FIXIMG_PROGRESS 1/0 label") is None


def test_parse_marker_tolerates_a_missing_label():
    assert parse_marker("@@FIXIMG_PROGRESS 1/2") == (1, 2, "")


def test_progress_markers_reach_the_callback(monkeypatch, tmp_path):
    """The marker lines are consumed and turned into progress updates (搂24)."""
    import io as _io

    from fiximg.inference.backends import legacy_cli

    class _FakeProc:
        def __init__(self) -> None:
            self.stdout = _io.StringIO(
                "ordinary log line\n"
                "@@FIXIMG_PROGRESS 1/2 overall quality restoration\n"
                "@@FIXIMG_PROGRESS 2/2 face detection\n"
            )
            self.returncode = 0

        def wait(self):
            return self.returncode

    monkeypatch.setattr(legacy_cli.subprocess, "Popen", lambda *a, **k: _FakeProc())
    seen: list[tuple] = []
    legacy_cli.stream_subprocess(
        ["python", "run.py"], on_progress=lambda *a: seen.append(a), echo=False
    )
    assert seen == [(1, 2, "overall quality restoration"), (2, 2, "face detection")]


def test_nonzero_exit_raises_pipeline_failed(monkeypatch):
    import io as _io

    from fiximg.inference.backends import legacy_cli

    class _FakeProc:
        def __init__(self) -> None:
            self.stdout = _io.StringIO("boom\n")
            self.returncode = 2

        def wait(self):
            return self.returncode

    monkeypatch.setattr(legacy_cli.subprocess, "Popen", lambda *a, **k: _FakeProc())
    with pytest.raises(PipelineFailedError):
        legacy_cli.stream_subprocess(["python", "run.py"], echo=False)


def test_python_is_resolved_to_the_running_interpreter(monkeypatch):
    from fiximg.inference.backends import legacy_cli

    captured = {}

    class _FakeProc:
        def __init__(self) -> None:
            self.stdout = __import__("io").StringIO("")
            self.returncode = 0

        def wait(self):
            return 0

    def _popen(args, **kwargs):
        captured["args"] = args
        return _FakeProc()

    monkeypatch.setattr(legacy_cli.subprocess, "Popen", _popen)
    legacy_cli.stream_subprocess(["python", "run.py"], echo=False)
    import sys

    assert captured["args"][0] == sys.executable


# --------------------------------------------------------------- degrade report
def test_read_report_returns_none_when_absent(tmp_path):
    assert read_report(str(tmp_path)) is None


def test_read_report_parses_the_finalize_output(tmp_path):
    import json

    payload = {
        "total": 2, "enhanced_count": 1, "degraded_count": 1,
        "degraded": ["b.png"], "degrade_reason": "no_face_detected",
    }
    with open(report_path(str(tmp_path)), "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    assert read_report(str(tmp_path))["degrade_reason"] == "no_face_detected"


def test_read_report_survives_corrupt_json(tmp_path):
    with open(report_path(str(tmp_path)), "w", encoding="utf-8") as handle:
        handle.write("{not json")
    assert read_report(str(tmp_path)) is None


def test_single_image_infer_returns_the_stage_output(recorder, tmp_path):
    """`_do_infer` reads back exactly the file the owned stage produced."""
    from PIL import Image

    from fiximg.inference.backends.base import ModelRequest

    def _materialise(args, cwd):
        root = args[args.index("--output_folder") + 1]
        stage = args[args.index("--stages") + 1]
        target = {
            STAGE_RESTORE: restored_image_dir(root),
            STAGE_FACE_DETECT: detection_dir(root),
            STAGE_FACE_ENHANCE: each_img_dir(root),
            STAGE_FINALIZE: final_dir(root),
        }[stage]
        _write_png(os.path.join(target, "image.png"))

    recorder(on_run=_materialise)
    backend = LegacyCliBackend({"stem": "image"}, stages=(STAGE_RESTORE,))
    result = backend.infer(
        ModelRequest(image=Image.new("RGB", (8, 8), "blue"), work_dir=str(tmp_path))
    )
    assert result.image.size == (8, 8)
    assert result.metadata["stages"] == [STAGE_RESTORE]
    assert result.metadata["duration_s"] >= 0


def test_single_image_infer_fails_when_the_stage_produces_nothing(recorder, tmp_path):
    from PIL import Image

    from fiximg.inference.backends.base import ModelRequest

    recorder()
    backend = LegacyCliBackend({"stem": "image"}, stages=(STAGE_RESTORE,))
    with pytest.raises(PipelineFailedError):
        backend.infer(
            ModelRequest(image=Image.new("RGB", (8, 8)), work_dir=str(tmp_path))
        )


def test_single_image_infer_requires_a_work_dir(recorder):
    from PIL import Image

    from fiximg.inference.backends.base import ModelRequest

    recorder()
    with pytest.raises(PipelineFailedError):
        LegacyCliBackend(stages=(STAGE_RESTORE,)).infer(
            ModelRequest(image=Image.new("RGB", (8, 8)), work_dir=None)
        )


# ------------------------------------------------- shared pipeline root (搂3.6)
def test_legacy_pipeline_root_is_shared_by_every_stage(tmp_path):
    """All four stages must resolve the same root or the chain breaks."""
    from fiximg.inference.context import StageContext

    context = StageContext(task_id="t", run_dir=str(tmp_path))
    root = legacy_pipeline_root(context)
    assert root == os.path.join(str(tmp_path), "stages", "legacy_pipeline")
    # Deterministic: the same context always yields the same root.
    assert legacy_pipeline_root(context) == root


# --------------------------------------------------- the four stages, end to end
def _stage_by_name(name):
    from fiximg.inference.registry import stage_registry

    return stage_registry.create(name)


@pytest.fixture()
def staged_run(recorder, tmp_path, monkeypatch):
    """Run the real four-stage chain with the subprocess replaced by a fake.

    The fake reproduces what each vendored script writes, so the stages exercise
    their real path logic: shared root, directory hand-off, artifact reporting.

    Native restoration is switched off here on purpose: this test is about the
    *subprocess* chain, and on a machine that happens to have the quality weights
    the stage would otherwise run in-process and record nothing for stage 1.
    """
    import fiximg.config as config_mod

    monkeypatch.setattr(config_mod.settings, "native_tree", "none", raising=False)
    # Stage 2 has its own switch, because dlib imports none of the legacy trees and
    # is therefore not affected by `native_tree` above. On an install that has dlib
    # the detection stage would run in-process, record no subprocess call, and the
    # four-stage handoff this test exists to pin would become a three-stage one.
    monkeypatch.setattr(config_mod.settings, "face_detect_native", False, raising=False)
    # Same reasoning as the line above: the fake below *does* write face crops, so
    # this test models a machine whose face chain works. The stages preflight dlib
    # before spawning stage 2 (and skip the chain when it is absent), so that one
    # probe has to answer "present" here 鈥?otherwise the run silently degrades and
    # the stages-1..4 handoff this test exists to pin never happens.
    from fiximg.inference import face_detect

    monkeypatch.setattr(face_detect, "dlib_importable", lambda: True)
    from PIL import Image

    from fiximg.inference.context import StageContext

    def _materialise(args, cwd):
        root = args[args.index("--output_folder") + 1]
        stages = args[args.index("--stages") + 1].split(",")
        stem = "task-1"
        if STAGE_RESTORE in stages:
            _write_png(os.path.join(restored_image_dir(root), f"{stem}.png"))
        if STAGE_FACE_DETECT in stages:
            _write_png(os.path.join(detection_dir(root), "face_0.png"))
        if STAGE_FACE_ENHANCE in stages:
            _write_png(os.path.join(each_img_dir(root), "face_0.png"))
        if STAGE_FINALIZE in stages:
            _write_png(os.path.join(final_dir(root), f"{stem}.png"))
            import json

            with open(report_path(root), "w", encoding="utf-8") as handle:
                json.dump({
                    "total": 1, "enhanced_count": 1, "degraded_count": 0,
                    "degraded": [], "degrade_report": None, "degrade_reason": None,
                }, handle)

    rec = recorder(on_run=_materialise)
    # The stages import `stream_subprocess` into their module namespace indirectly
    # through LegacyCliBackend, which resolves it at call time 鈥?one patch is
    # enough, but assert it took effect.
    context = StageContext(
        task_id="task-1", run_dir=str(tmp_path), options={},
    )
    return rec, context, Image.new("RGB", (16, 16), "orange")


def test_full_chain_hands_off_through_the_shared_root(staged_run):
    recorder_, context, image = staged_run
    restored = _stage_by_name("global_restore").run(image, context)
    detected = _stage_by_name("face_detection").run(restored.image, context)
    enhanced = _stage_by_name("face_enhancement").run(detected.image, context)
    composited = _stage_by_name("warp_back").run(enhanced.image, context)

    # Each stage ran exactly one legacy path.
    assert [r[0][r[0].index("--stages") + 1] for r in recorder_.calls] == list(ALL_STAGES)
    # And they all shared one output root.
    roots = {r[0][r[0].index("--output_folder") + 1] for r in recorder_.calls}
    assert len(roots) == 1

    assert restored.metrics["stub_restore"] if "stub_restore" in restored.metrics else True
    assert detected.metrics["face_count"] == 1
    assert enhanced.metrics["enhanced_count"] == 1
    assert composited.metrics["enhanced_count"] == 1
    assert composited.metrics["degraded_count"] == 0
    assert composited.message is None


def test_face_detection_does_not_change_the_image(staged_run):
    """Detection and enhancement pass the restored image through unchanged."""
    _rec, context, image = staged_run
    restored = _stage_by_name("global_restore").run(image, context)
    detected = _stage_by_name("face_detection").run(restored.image, context)
    assert detected.image is restored.image


def test_face_enhancement_skips_when_no_face_was_detected(staged_run, monkeypatch):
    """A faceless photo degrades gracefully instead of failing the task.

    The stages bind `list_images` at import time, so the patch has to target the
    module that uses it, not the one that defines it.
    """
    from fiximg.inference.stages import face_enhancement as face_mod

    recorder_, context, image = staged_run
    monkeypatch.setattr(face_mod, "list_images", lambda _d: [])
    restored = _stage_by_name("global_restore").run(image, context)
    detected = _stage_by_name("face_detection").run(restored.image, context)
    enhanced = _stage_by_name("face_enhancement").run(detected.image, context)

    assert detected.metrics["face_count"] == 0
    assert enhanced.metadata["skipped"] is True
    assert enhanced.metrics["enhanced_count"] == 0


def test_warp_back_reports_the_degrade_reason(tmp_path, recorder):
    """When nothing was enhanced, warp-back records why (plan 搂3.7)."""
    import json

    from PIL import Image

    from fiximg.inference.context import StageContext

    def _materialise_degraded(args, cwd):
        root = args[args.index("--output_folder") + 1]
        _write_png(os.path.join(final_dir(root), "task-1.png"))
        with open(report_path(root), "w", encoding="utf-8") as handle:
            json.dump({
                "total": 1, "enhanced_count": 0, "degraded_count": 1,
                "degraded": ["task-1.png"], "degrade_reason": "no_face_detected",
            }, handle)

    recorder(on_run=_materialise_degraded)
    context = StageContext(task_id="task-1", run_dir=str(tmp_path), options={})
    composited = _stage_by_name("warp_back").run(Image.new("RGB", (16, 16)), context)

    assert composited.metrics["degraded_count"] == 1
    assert composited.metadata["degraded"] is True
    assert "no_face_detected" in (composited.message or "")


def test_stages_never_write_to_the_database(staged_run):
    """Stage contract (搂3.6): persistence belongs to the runtime."""
    import inspect

    from fiximg.inference.stages import face_enhancement, global_restore, warp_back

    for module in (global_restore, face_enhancement, warp_back):
        source = inspect.getsource(module)
        for forbidden in ("task_repo", "add_metric", "finish_task", "gradio"):
            assert forbidden not in source, f"{module.__name__} references {forbidden}"


# ==================== refusing to run on random weights ====================
# ``base_model.load_network`` prints "鈥ot exists yet" and continues, so a branch
# whose networks are missing produces a plausible-looking image from randomly
# initialised weights. These pin the pre-flight that closes that hole.

def _make_branch(root, names):
    os.makedirs(root, exist_ok=True)
    for directory, filename in names:
        path = os.path.join(root, directory)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, filename), "wb") as handle:
            handle.write(b"x")


def test_the_quality_branch_needs_three_networks():
    from fiximg.inference.backends import legacy_cli

    names = legacy_cli.BRANCH_WEIGHTS[(False, False)]
    assert ("mapping_quality", "latest_net_mapping_net.pth") in names
    assert ("VAE_B_quality", "latest_net_G.pth") in names


def test_the_scratch_branch_switches_both_networks():
    from fiximg.inference.backends import legacy_cli

    names = legacy_cli.BRANCH_WEIGHTS[(True, False)]
    assert ("VAE_B_scratch", "latest_net_G.pth") in names
    assert ("mapping_scratch", "latest_net_mapping_net.pth") in names


def test_hr_needs_a_different_mapping_directory():
    """``--HR`` selects mapping_Patch_Attention, which a normal install lacks."""
    from fiximg.inference.backends import legacy_cli

    names = legacy_cli.BRANCH_WEIGHTS[(True, True)]
    assert ("mapping_Patch_Attention", "latest_net_mapping_net.pth") in names


def test_training_only_files_are_never_required():
    from fiximg.inference.backends import legacy_cli

    for entries in legacy_cli.BRANCH_WEIGHTS.values():
        for _directory, filename in entries:
            assert "D.pth" not in filename and "optimizer" not in filename


def test_a_partially_installed_hr_branch_is_refused(recorder, monkeypatch, tmp_path):
    from fiximg.inference.backends import legacy_cli

    root = str(tmp_path / "restoration")
    _make_branch(root, legacy_cli.BRANCH_WEIGHTS[(True, False)])  # no HR mapping net
    monkeypatch.setattr(legacy_cli, "RESTORATION_DIR", root)

    with pytest.raises(legacy_cli.ModelUnavailableError) as excinfo:
        legacy_cli.LegacyCliBackend(with_scratch=True,
                                    stages=(legacy_cli.STAGE_RESTORE,)).run_folder(
            "/in", "/out", hr=True)
    missing = excinfo.value.details["missing"]
    assert any("mapping_Patch_Attention" in p for p in missing)
    assert "randomly initialised" in str(excinfo.value)


def test_a_complete_branch_passes_the_check(recorder, monkeypatch, tmp_path):
    from fiximg.inference.backends import legacy_cli

    root = str(tmp_path / "restoration")
    _make_branch(root, legacy_cli.BRANCH_WEIGHTS[(True, True)])
    monkeypatch.setattr(legacy_cli, "RESTORATION_DIR", root)

    rec = recorder()
    legacy_cli.LegacyCliBackend(with_scratch=True,
                                stages=(legacy_cli.STAGE_RESTORE,)).run_folder(
        str(tmp_path / "in"), str(tmp_path / "out"), hr=True)
    assert "--HR" in rec.flags(), "the run went ahead and forwarded the flag"


def test_a_model_that_is_not_installed_at_all_is_not_reported_as_broken(
    recorder, monkeypatch, tmp_path
):
    """No Global/checkpoints means "uninstalled", which availability reports.

    Treating that as a broken branch would make every weight-free checkout (and CI)
    fail on command-assembly tests rather than at the point a model is actually
    needed.
    """
    from fiximg.inference.backends import legacy_cli

    monkeypatch.setattr(legacy_cli, "RESTORATION_DIR", str(tmp_path / "absent"))
    rec = recorder()
    legacy_cli.LegacyCliBackend(with_scratch=True,
                                stages=(legacy_cli.STAGE_RESTORE,)).run_folder(
        str(tmp_path / "in"), str(tmp_path / "out"), hr=True)
    assert rec.last, "the subprocess command was still assembled"


def test_stages_that_do_not_restore_skip_the_restoration_check(
    recorder, monkeypatch, tmp_path
):
    """Face detection/enhancement stages must not be blocked by this guard."""
    from fiximg.inference.backends import legacy_cli

    root = str(tmp_path / "restoration")
    os.makedirs(root, exist_ok=True)  # present but empty: only stage 1 would care
    monkeypatch.setattr(legacy_cli, "RESTORATION_DIR", root)

    rec = recorder()
    legacy_cli.LegacyCliBackend(stages=(legacy_cli.STAGE_FACE_DETECT,)).run_folder(
        str(tmp_path / "in"), str(tmp_path / "out"))
    assert rec.last


# ------------------------------------------- graceful degradation (no dlib)
def _without_dlib(monkeypatch):
    from fiximg.inference import face_detect

    monkeypatch.setattr(face_detect, "dlib_importable", lambda: False)


def test_the_face_chain_is_skipped_when_dlib_is_absent(staged_run, monkeypatch):
    """A machine without dlib must still deliver the restoration it completed.

    Before this, `restore` failed three times over (retry exhausted) with
    PIPELINE_FAILED after stage 1 had already produced a good image: the only way
    to discover the missing dependency was the child process's traceback.
    """
    recorder_, context, image = staged_run
    _without_dlib(monkeypatch)

    restored = _stage_by_name("global_restore").run(image, context)
    detected = _stage_by_name("face_detection").run(restored.image, context)
    enhanced = _stage_by_name("face_enhancement").run(detected.image, context)
    final = _stage_by_name("warp_back").run(enhanced.image, context)

    assert detected.metadata["skipped"] is True
    assert detected.metadata["reason"] == "face_dependencies_unavailable"
    assert enhanced.metadata["skipped"] is True, "no crops, so nothing to enhance"
    assert final.metadata["skipped"] is True
    assert final.image is enhanced.image, "the restored image is the result"

    # Only stage 1 ever spawned a subprocess: stages 2-4 were skipped rather than
    # run into the same missing dependency three more times.
    assert [r[0][r[0].index("--stages") + 1] for r in recorder_.calls] == ["1"]


def test_the_skip_names_the_fix_in_the_result(staged_run, monkeypatch):
    """Visible, not silent: the message says what is missing and how to install it."""
    recorder_, context, image = staged_run
    _without_dlib(monkeypatch)

    restored = _stage_by_name("global_restore").run(image, context)
    detected = _stage_by_name("face_detection").run(restored.image, context)

    assert "dlib" in (detected.message or "")
    assert "fiximg[gpu]" in (detected.message or "")
    assert context.metadata["face_chain_skipped"] == "face_dependencies_unavailable"


def test_warp_back_passes_the_image_through_without_the_chain(monkeypatch, tmp_path):
    """The stage-4 subprocess has nothing to composite; it must not be started."""
    from PIL import Image

    from fiximg.inference.context import StageContext
    from fiximg.inference.stages.warp_back import WarpBackStage

    image = Image.new("RGB", (24, 24), "white")
    context = StageContext(
        task_id="task-9", run_dir=str(tmp_path / "run"), options={},
        metadata={"face_chain_skipped": "face_dependencies_unavailable"},
    )
    calls: list = []
    import fiximg.inference.backends.legacy_cli as legacy_cli

    monkeypatch.setattr(legacy_cli, "stream_subprocess", lambda *a, **k: calls.append(a))

    result = WarpBackStage().run(image, context)

    assert calls == [], "skipped chain means no finalize subprocess"
    assert result.image is image
    assert result.metadata["face_count"] == 0


def test_dlib_importable_reads_the_environment_it_is_asked_about():
    """The preflight and the tests must agree on what "absent" means.

    `sys.modules[name] = None` is CPython's marker for a failed import and is what
    the dlib-absent tests install, so the probe treats it as absent instead of
    reporting the dict entry as success.
    """
    import importlib.util
    import sys
    import types

    from fiximg.inference.face_detect import dlib_importable

    expected = importlib.util.find_spec("dlib") is not None
    saved = sys.modules.get("dlib", "missing")
    try:
        sys.modules["dlib"] = None
        assert dlib_importable() is False
        sys.modules["dlib"] = types.ModuleType("dlib")
        assert dlib_importable() is True
        del sys.modules["dlib"]
        assert dlib_importable() is expected
    finally:
        if saved == "missing":
            sys.modules.pop("dlib", None)
        else:
            sys.modules["dlib"] = saved
