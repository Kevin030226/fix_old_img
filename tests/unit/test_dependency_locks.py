"""The dependency lock audit (plan 搂3.12) 鈥?what it proves and where it stops.

``make lock-check`` is a CI gate, so it has to be trustworthy. Reworking it here
removed three ways it could report green without meaning anything:

* a test invoked the generator with no arguments, so *running the suite* rewrote
  ``requirements/*.txt`` in the working tree 鈥?the gate was grading files its own
  test had just produced, and whichever interpreter ran last defined the answer;
* the byte-for-byte closure comparison depended on the interpreter, because a CPU
  torch and a ``+cu128`` torch have different transitive dependencies, so the same
  checkout was current on one machine and stale on another;
* ``requirements/gpu.txt`` 鈥?the documented way to install the GPU stack 鈥?pinned
  ``torch``, ``torchvision`` and ``dlib`` to ``file:///D:/Soft/...`` and
  ``file:///C:/bld/...``, paths that exist only on the box that froze them.

The rule the audit now enforces: pins in a group file must equal the lock, every
line must be installable on another machine, and the *closure* is only re-derived
by the interpreter the lock names. Everything skipped is named.
"""
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
import zlib
from types import SimpleNamespace

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "export_locks.py")
GROUP_DIR = os.path.join(PROJECT_ROOT, "requirements")


def _load_script():
    spec = importlib.util.spec_from_file_location("export_locks", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, SCRIPT, *args], cwd=PROJECT_ROOT,
        capture_output=True, text=True, env=env,
    )


def _group_hashes() -> dict[str, int]:
    """Name -> crc of the file body, enough to notice any rewrite of the tree."""
    assert os.path.isdir(GROUP_DIR), (
        f"{GROUP_DIR} is absent; regenerate it with `make lock` and commit the files, "
        "otherwise the audit has nothing to check on a fresh clone"
    )
    out: dict[str, int] = {}
    for name in sorted(os.listdir(GROUP_DIR)):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(GROUP_DIR, name), "rb") as handle:
            out[name] = zlib.crc32(handle.read())
    return out


@pytest.fixture()
def locks(tmp_path, monkeypatch):
    """A scratch checkout: pyproject with two groups, a lock, an output directory.

    ``numpy`` is the fixture's dependency because it is installed in every
    interpreter this suite runs in and has no dependencies of its own, so its
    closure is known exactly and the round trip is deterministic.
    """
    module = _load_script()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "scratch"\ndependencies = ["numpy"]\n\n'
        '[project.optional-dependencies]\ntools = ["numpy"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PYPROJECT", str(tmp_path / "pyproject.toml"))
    monkeypatch.setattr(module, "LOCK_FILE", str(tmp_path / "requirements.lock"))
    monkeypatch.setattr(module, "DEFAULT_OUTPUT_DIR", str(tmp_path / "requirements"))
    numpy_version = importlib.metadata.version("numpy")
    pin = f"numpy=={numpy_version}"
    return SimpleNamespace(
        module=module,
        lock=tmp_path / "requirements.lock",
        out=tmp_path / "requirements",
        pin=pin,
    )


def write_lock(locks, tag: str, extra: list[str] | None = None) -> None:
    lines = [locks.pin] + list(extra or [])
    text = f"# Frozen-environment: {tag}\n" + "\n".join(lines) + "\n"
    locks.lock.write_text(text, encoding="utf-8")


# ------------------------------------------------- the always-on text invariants
def test_a_group_pin_must_equal_the_lock(locks, capsys):
    """A file that installs a different version than the freeze is the drift class.

    Runs off the frozen environment on purpose: this half of the audit cannot be
    waived by the machine it happens to execute on.
    """
    write_lock(locks, "python 0.0 / nowhere")
    locks.out.mkdir(parents=True)
    (locks.out / "runtime.txt").write_text(
        "# header\nnumpy==0.0.0\n", encoding="utf-8"
    )
    assert locks.module.check() == 1
    printed = capsys.readouterr().out
    assert "numpy==0.0.0" in printed and "requirements.lock" in printed, printed


def test_a_group_line_must_be_a_plain_pin(locks, capsys):
    """``python-multipart`` with no version is an instruction, not a lock."""
    write_lock(locks, "python 0.0 / nowhere", extra=["python-multipart==0.0.20"])
    locks.out.mkdir(parents=True)
    (locks.out / "runtime.txt").write_text(
        f"# header\n{locks.pin}\npython-multipart\n", encoding="utf-8"
    )
    assert locks.module.check() == 1
    assert "not a plain pin" in capsys.readouterr().out


def test_a_lock_pinned_to_a_local_path_is_refused(locks, capsys):
    """The defect that made the documented GPU install unusable elsewhere.

    ``pip freeze`` writes ``torch @ file:///D:/Soft/fixoldimg_wheels/torch.whl`` for
    anything installed from a wheel on disk, and the splitter copied that straight
    into ``requirements/gpu.txt``.
    """
    write_lock(locks, "python 0.0 / nowhere",
               extra=["torch @ file:///D:/Soft/fixoldimg_wheels/torch-2.7.1.whl"])
    assert locks.module.check() == 1
    printed = capsys.readouterr().out
    assert "another machine" in printed and "--freeze" in printed, printed


def test_the_closure_is_only_re_derived_by_the_environment_the_lock_names(locks, capsys):
    """Skipping must be stated, with both environment names, never silently passed."""
    write_lock(locks, "python 0.0 / nowhere")
    module = locks.module
    locks.out.mkdir(parents=True)
    for group in ("runtime", "tools"):
        (locks.out / f"{group}.txt").write_text(f"# header\n{locks.pin}\n", encoding="utf-8")

    assert module.check() == 0
    printed = capsys.readouterr().out
    assert "closure re-derivation skipped" in printed, printed
    assert "python 0.0 / nowhere" in printed, printed
    assert module.env_tag() in printed, printed


# ----------------------------------------------------------------- generation
def test_export_then_check_round_trips_in_the_frozen_environment(locks, capsys):
    write_lock(locks, locks.module.env_tag())
    summary = locks.module.export()
    assert summary == {"runtime": 1, "tools": 1}, summary

    body = (locks.out / "runtime.txt").read_text(encoding="utf-8")
    assert body.splitlines()[-1] == locks.pin, body
    assert locks.module.check() == 0
    assert "closure re-derivation skipped" not in capsys.readouterr().out


def test_export_writes_only_where_it_was_pointed(locks):
    """Running the generator must never be a side effect on the working tree.

    The old test called it with no arguments, so the audit's own input file was
    rewritten by the suite that read it.
    """
    before = _group_hashes()
    write_lock(locks, locks.module.env_tag())
    locks.module.export()
    assert (locks.out / "runtime.txt").exists()
    assert _group_hashes() == before, "the scratch export rewrote requirements/"


def test_export_refuses_an_interpreter_that_is_not_the_frozen_one(locks):
    """Generating elsewhere would ship files that describe a different environment."""
    write_lock(locks, "python 0.0 / nowhere")
    with pytest.raises(SystemExit) as excinfo:
        locks.module.export()
    assert "frozen from" in str(excinfo.value)
    assert not locks.out.exists(), "refused generation still wrote group files"


def test_marker_conditionals_for_another_python_never_reach_a_group_file(locks, monkeypatch):
    """``audioop-lts; python_version >= "3.13"`` is not a dependency of a 3.11 install.

    Collecting markers unconditionally made ``requirements/runtime.txt`` demand a
    package pip cannot resolve for the interpreter the file is meant for.
    """
    declared = {
        "numpy": [
            'for-this-python>=1.0',
            'for-a-future-python>=1.0; python_version >= "3.99"',
            'for-a-past-python>=1.0; python_version < "3.4"',
            'an-extra>=1.0; extra == "test"',
            'platform-specific>=1.0; platform_system == "NotARealSystem"',
        ],
    }
    monkeypatch.setattr(locks.module, "requires", lambda name: declared.get(name, []))
    assert locks.module.closure(["numpy"]) == {"numpy", "for-this-python"}


class _FakeDist:
    """Just enough of a distribution for ``freeze()``: name, version, origin."""

    def __init__(self, name: str, version: str, url: str | None = None,
                 editable: bool = False) -> None:
        self.metadata = {"Name": name}
        self.version = version
        self._url = url
        self._editable = editable

    def read_text(self, filename: str) -> str | None:
        if filename != "direct_url.json" or self._url is None:
            return None
        import json

        return json.dumps({"url": self._url, "dir_info": {"editable": self._editable}})


def test_freeze_records_its_environment_and_keeps_local_wheels_installable(
    locks, monkeypatch
):
    monkeypatch.setattr(
        importlib.metadata, "distributions",
        lambda: [
            _FakeDist("zeta", "1.2.0"),
            _FakeDist("torch", "2.7.1+cu128",
                      url="file:///D:/Soft/fixoldimg_wheels/torch-2.7.1.whl"),
            _FakeDist("dlib", "20.0.1", url="file:///C:/bld/dlib-split_1774/work"),
            _FakeDist("fiximg", "3.0.0",
                      url="file:///E:/1/fix_old_img-main/src", editable=True),
        ],
    )
    assert locks.module.freeze() == 0

    text = locks.lock.read_text(encoding="utf-8")
    assert f"# Frozen-environment: {locks.module.env_tag()}" in text
    assert " @ file:" not in text, "a machine-local path survived the freeze"
    assert "fiximg" not in text, "the editable project itself must not be pinned"
    # The original source is still recorded, as a comment pip ignores, and the pins
    # come out in one deterministic order.
    assert "#   dlib==20.0.1 came from file:///C:/bld/dlib-split_1774/work" in text
    assert "#   torch==2.7.1+cu128 came from file:///D:/Soft/fixoldimg_wheels" in text
    pins = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert pins == ["dlib==20.0.1", "torch==2.7.1+cu128", "zeta==1.2.0"], pins


# --------------------------------------------------- the committed artifacts
def test_the_committed_lock_files_are_current():
    """``make lock-check`` must pass as this checkout stands."""
    result = _run("--check")
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_committed_lock_installs_on_another_machine():
    """No ``file:///`` path from this box may survive into a shipped requirement."""
    with open(os.path.join(PROJECT_ROOT, "requirements.lock"), encoding="utf-8-sig") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    offenders = [ln for ln in lines if " @ " in ln or "==" not in ln]
    assert not offenders, f"non-portable pins: {offenders}"


def test_lock_groups_cover_the_documented_purposes():
    """The report asks for runtime / model / ci / benchmark groups."""
    groups = {name[:-4] for name in os.listdir(GROUP_DIR) if name.endswith(".txt")}
    for expected in ("runtime", "gpu", "test", "benchmark", "dev"):
        assert expected in groups, expected


def test_runtime_group_excludes_test_tools():
    """Production must not install pytest."""
    with open(os.path.join(GROUP_DIR, "runtime.txt"), encoding="utf-8") as handle:
        body = handle.read().lower()
    assert "pytest" not in body
    assert "runtime" in body


def test_gpu_group_is_separate_from_runtime():
    with open(os.path.join(GROUP_DIR, "gpu.txt"), encoding="utf-8") as handle:
        gpu = handle.read().lower()
    with open(os.path.join(GROUP_DIR, "runtime.txt"), encoding="utf-8") as handle:
        runtime = handle.read().lower()
    assert "torch" in gpu
    assert "torch" not in runtime


def test_lock_files_carry_a_provenance_header():
    with open(os.path.join(GROUP_DIR, "test.txt"), encoding="utf-8") as handle:
        head = handle.read(500)
    assert head.startswith("# Generated by scripts/export_locks.py")
    assert "pip install -r requirements/test.txt" in head
    assert "Frozen-environment" in head, "the file does not say whose versions these are"


def test_regenerating_the_groups_never_touches_the_working_tree(tmp_path):
    """The generator honours ``--output`` / ``FIXIMG_LOCK_OUTPUT_DIR``.

    Without it the suite's own export step was what kept ``requirements/`` current,
    which made the audit unfalsifiable outside the machine that ran the tests.
    """
    before = _group_hashes()
    scratch = tmp_path / "out"
    result = _run("--output", str(scratch))
    if result.returncode != 0:
        assert "frozen from" in result.stdout + result.stderr, result.stdout + result.stderr
    assert _group_hashes() == before
    assert not (tmp_path / "requirements").exists()

    env = dict(os.environ, FIXIMG_LOCK_OUTPUT_DIR=str(scratch))
    result = _run(env=env)
    if result.returncode != 0:
        assert "frozen from" in result.stdout + result.stderr, result.stdout + result.stderr
    assert _group_hashes() == before
