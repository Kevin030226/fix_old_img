"""Unit tests for Settings and artifact storage (plan sections 15 and 20)."""
import os
import pathlib
import re

from fiximg.config import settings
from fiximg.infrastructure.storage import local as artifact_service


def test_settings_defaults():
    assert settings.max_image_side == 4096
    assert settings.ddcolor_input_size == 512
    assert settings.history_max == 2000
    assert settings.base_dir


def test_artifact_run_dir_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "tasks_root", str(tmp_path / "tasks"))
    run_dir = artifact_service.create_run_dir("req-1")
    assert run_dir.endswith("req-1")
    for sub in ("input", "stages", "output"):
        assert (run_dir / sub if False else f"{run_dir}/{sub}") and __import__("os").path.isdir(
            f"{run_dir}/{sub}"
        )


def test_save_input_and_report(tmp_path, monkeypatch):
    import os

    monkeypatch.setattr(settings, "tasks_root", str(tmp_path / "tasks"))
    from PIL import Image

    run_dir = artifact_service.create_run_dir("req-2")
    img = Image.new("RGB", (8, 8), (0, 255, 0))
    input_path = artifact_service.save_input_image(run_dir, "req-2", img)
    assert os.path.exists(input_path)

    report_path = artifact_service.write_report(run_dir, {"task_id": "req-2"})
    assert os.path.exists(report_path)
    import json

    with open(report_path, encoding="utf-8") as f:
        assert json.load(f)["task_id"] == "req-2"


def test_the_documented_storage_roots_are_actually_honoured(monkeypatch, tmp_path):
    """`FIXIMG_TASKS_ROOT` is named in `docs/deployment.md` as the retention knob.

    It was read by nothing: `tasks_root` was derived from the project root, so an
    operator who moved task storage to a bigger volume kept filling the system one 鈥?
    and the TTL sweeper, which resolves against the same attribute, kept looking in the
    place the documents said was no longer used.
    """
    from fiximg.config import BASE_DIR, Settings

    volume = tmp_path / "volume" / "tasks"
    monkeypatch.setenv("FIXIMG_TASKS_ROOT", str(volume))
    settings = Settings()

    assert settings.tasks_root == str(volume)
    assert settings.storage_root == os.path.join(BASE_DIR, "storage")


def test_a_relative_root_is_resolved_against_the_project_not_the_cwd(monkeypatch):
    """Two processes started from different directories must find the same bytes."""
    from fiximg.config import BASE_DIR, Settings

    monkeypatch.setenv("FIXIMG_STORAGE_ROOT", "scratch/storage")
    settings = Settings()

    assert settings.storage_root == os.path.join(BASE_DIR, "scratch/storage")
    assert settings.tasks_root == os.path.join(BASE_DIR, "scratch/storage", "tasks")


def test_the_priority_ceiling_is_an_operator_knob(monkeypatch):
    """`FIXIMG_PRIORITY_MAX` bounds how far a submission may jump the queue (plan 搂2.6).

    The create endpoint refuses anything above it and quotes the value it refused against,
    so the ceiling has to come from the settings object the request is handled with. A
    constant baked into the endpoint would describe a limit the operator no longer runs.
    """
    from fiximg.config import Settings

    monkeypatch.delenv("FIXIMG_PRIORITY_MAX", raising=False)
    assert Settings().priority_max == 10

    monkeypatch.setenv("FIXIMG_PRIORITY_MAX", "2")
    assert Settings().priority_max == 2


def test_every_environment_knob_the_deployment_doc_names_is_read():
    """`docs/deployment.md` is the operator's list of what may be tuned at all.

    Two knobs it named (`FIXIMG_STORAGE_ROOT`, `FIXIMG_TASKS_ROOT`) turned out to be read
    by nothing 鈥?which is worse than an undocumented setting, because the operator sets it,
    sees no effect, and keeps trusting the document. The list is scraped from the document
    and the reference is searched in the source, so a new documented knob nothing reads
    fails here instead of in production.

    Two exemptions, both derived from the source rather than hand-maintained: a name built
    by prefix at run time (`FIXIMG_CONCURRENCY_<CAPABILITY>`, whose prefix the scheduler
    declares as a literal) and `FIXIMG_TEST_*`, which belong to the test tier and are
    documented as such.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    doc = (root / "docs" / "deployment.md").read_text(encoding="utf-8")
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (root / "src" / "fiximg").rglob("*.py")
    )

    documented = set(re.findall(r"FIXIMG_[A-Z0-9_]+", doc))
    assert len(documented) >= 30, f"only {len(documented)} knobs scraped from the doc"

    # Prefixes the code assembles names from, e.g. `prefix = "FIXIMG_CONCURRENCY_"`.
    prefixes = set(re.findall(r'"(FIXIMG_[A-Z0-9_]+_)"', sources))
    assert prefixes, "the per-capability prefix rule has nothing to match against"

    unread = sorted(
        name for name in documented
        if f'"{name}"' not in sources
        and not name.startswith("FIXIMG_TEST_")
        and not any(name.startswith(prefix) for prefix in prefixes)
    )
    assert not unread, f"documented knobs no source reads: {unread}"

    # The prefix exemption must actually be what saves a documented name, otherwise it is
    # a permanent waiver nobody needs: `FIXIMG_CONCURRENCY_SCRATCH_DETECTION` is a real
    # capability name assembled at run time and appears nowhere as a literal.
    dynamic = [
        name for name in documented
        if f'"{name}"' not in sources
        and any(name.startswith(prefix) for prefix in prefixes)
    ]
    assert dynamic, "the dynamic-prefix exemption is not exercised by any documented knob"

