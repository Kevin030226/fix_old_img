"""The legacy shell entry point must delegate to the verified Python CLI.

`scripts/download_weights.sh` used to run its own curl/wget copies, skip files
that merely existed and never consult `config/weights_manifest.json`. After the
Python CLI gained the verified-download contract, that shell path was a way
around it, and its BOB sources pointed at an Azure endpoint that no longer
resolves. These tests pin the delegation contract without network access: a stub
interpreter records argv/cwd and returns controlled exit codes, and static checks
forbid the old direct-download commands.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "download_weights.sh"

#: Commands the legacy script used to run itself. A delegating wrapper may mention
#: them in its history comment, never in executable lines.
FORBIDDEN_COMMANDS = ("curl", "wget", "aria2c", "bzip2", "unzip")
FORBIDDEN_HOSTS = ("facevc.blob.core.windows.net", "huggingface.co", "gitee.com")


def _bash() -> str:
    """Find a bash that can run a Windows/Git-Bash script (not WSL's launcher).

    On Windows, `shutil.which("bash")` can resolve to the Windows Subsystem for
    Linux launcher installed under the WindowsApps alias, which cannot see the
    Windows checkout at
    all and fails with exit 127. Prefer Git Bash explicitly and skip rather than
    assert against the wrong interpreter.
    """
    if os.name == "nt":
        candidates = []
        for env_name in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            base = os.environ.get(env_name)
            if base:
                candidates.append(os.path.join(base, "Git", "bin", "bash.exe"))
                candidates.append(os.path.join(base, "Programs", "Git", "bin", "bash.exe"))
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        pytest.skip("Git Bash not found on this Windows host")
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available on this host")
    return bash


def _code_lines() -> str:
    lines = []
    for raw in SCRIPT.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def _stub(tmp_path: Path, exit_code: int) -> Path:
    stub = tmp_path / f"stub-python-{exit_code}"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$@" > "$ARGV_FILE"\n'
        'pwd -W > "$CWD_FILE" 2>/dev/null || pwd > "$CWD_FILE"\n'
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


def _run(tmp_path: Path, stub: Path, args: list[str], python: str | None = None):
    env = {
        **os.environ,
        "FIXIMG_PYTHON": python if python is not None else str(stub),
        "ARGV_FILE": str(tmp_path / "argv.txt"),
        "CWD_FILE": str(tmp_path / "cwd.txt"),
    }
    return subprocess.run(
        [_bash(), str(SCRIPT), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_script_no_longer_downloads_by_itself():
    code = _code_lines()
    for command in FORBIDDEN_COMMANDS:
        assert command not in code, f"legacy shell entry must delegate, found {command!r}"
    for host in FORBIDDEN_HOSTS:
        assert host not in code, f"legacy shell entry must not embed sources, found {host!r}"
    assert "fiximg.cli.download_weights" in code
    assert "exec " in code, "the wrapper must exec the CLI so its exit code is forwarded"


def test_script_is_valid_bash():
    result = subprocess.run(
        [_bash(), "-n", str(SCRIPT)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_shell_entry_delegates_to_python_cli(tmp_path):
    stub = _stub(tmp_path, 0)
    result = _run(tmp_path, stub, [])
    assert result.returncode == 0, result.stderr
    argv = (tmp_path / "argv.txt").read_text(encoding="utf-8").splitlines()
    assert argv == ["-m", "fiximg.cli.download_weights", "download"]
    cwd = Path((tmp_path / "cwd.txt").read_text(encoding="utf-8").strip())
    assert cwd.resolve() == REPO_ROOT


def test_shell_entry_propagates_failure_exit_code(tmp_path):
    stub = _stub(tmp_path, 7)
    result = _run(tmp_path, stub, [])
    assert result.returncode == 7, "the CLI's non-zero exit must not be swallowed"


def test_shell_entry_rejects_subcommands(tmp_path):
    stub = _stub(tmp_path, 0)
    result = _run(tmp_path, stub, ["verify"])
    assert result.returncode == 2
    assert "python -m fiximg.cli.download_weights" in result.stderr
    assert not (tmp_path / "argv.txt").exists(), "the CLI must not run for a rejected sub-command"


def test_shell_entry_reports_missing_interpreter(tmp_path):
    result = _run(tmp_path, _stub(tmp_path, 0), [], python=str(tmp_path / "no-such-python"))
    assert result.returncode == 127
