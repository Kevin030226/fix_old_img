"""Regression tests for the API-token first-boot protocol (v3.1.2 review).

The shipped code read the token file, and only if it was absent generated one with
`O_CREAT | O_EXCL`. Two defects followed:

* the file became visible before its content was written, so a second process could
  hit `FileExistsError`, read an empty/partial file, and fall back to its own
  candidate via `_read_token(path) or candidate` — two live tokens, one disk file;
* a creator that died after creating the (empty) file left a zero-byte token that
  every later boot then had to interpret.

These tests use real OS processes and a `multiprocessing.Barrier`, not a simulated
"the winner already wrote a complete file". They assert the property that matters:
every process that finishes booting returns the same non-empty value, and that value
is what is on disk.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import stat
import time

import pytest

from fiximg.api import security


def _boot_worker(token_path: str, barrier, queue, delay: float) -> None:
    """Boot one worker process: wait for the barrier, then resolve the token."""
    from fiximg.api import security as child_security

    child_security.token_file = lambda: token_path
    child_security._cached = None
    real_generate = child_security._generate

    def slow_generate() -> str:
        # Widen the window between "no token" and "token persisted" so the test
        # actually exercises the race instead of relying on scheduling luck.
        time.sleep(delay)
        return real_generate()

    child_security._generate = slow_generate
    try:
        barrier.wait(timeout=60)
        queue.put(("ok", child_security.get_api_token()))
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _crashing_worker(token_path: str, barrier) -> None:
    """Create a temp file, then die before the atomic rename (a hard crash)."""
    from fiximg.api import security as child_security

    child_security.token_file = lambda: token_path
    child_security._cached = None

    def crash_after_partial_write(path: str, token: str) -> None:
        fd, tmp = __import__("tempfile").mkstemp(
            prefix=".api_token.", suffix=".tmp", dir=os.path.dirname(path)
        )
        os.write(fd, b"partial")
        os.close(fd)
        os._exit(3)          # no cleanup, no rename: exactly a killed process

    child_security._persist = crash_after_partial_write
    barrier.wait(timeout=60)
    child_security.get_api_token()


def _spawn_booters(token_path: str, count: int, delays: list[float]):
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(count)
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=_boot_worker, args=(token_path, barrier, queue, delays[i]))
        for i in range(count)
    ]
    for proc in procs:
        proc.start()
    results = [queue.get(timeout=120) for _ in procs]
    for proc in procs:
        proc.join(timeout=60)
    return results, procs


@pytest.fixture()
def token_path(tmp_path, monkeypatch):
    path = str(tmp_path / "api_token.txt")
    monkeypatch.setattr(security, "token_file", lambda: path)
    monkeypatch.delenv("FIXIMG_API_TOKEN", raising=False)
    security._cached = None
    yield path
    security._cached = None


# ------------------------------------------------- real multi-process race
def test_concurrent_first_boot_processes_all_adopt_one_persisted_token(token_path):
    """Four processes boot at the same instant; exactly one token may exist."""
    results, procs = _spawn_booters(token_path, 4, [0.05, 0.15, 0.25, 0.35])

    errors = [value for status, value in results if status != "ok"]
    assert not errors, errors
    tokens = [value for status, value in results if status == "ok"]
    assert len(tokens) == 4
    assert len(set(tokens)) == 1, f"processes disagreed on the token: {tokens}"

    stored = open(token_path, encoding="utf-8").read().strip()
    assert stored, "the token file is empty"
    assert stored == tokens[0], f"file {stored!r} != process token {tokens[0]!r}"

    meta = json.loads(open(token_path + ".meta", encoding="utf-8").read())
    assert meta["length"] == len(stored)
    assert meta["sha256"] == hashlib.sha256(stored.encode("utf-8")).hexdigest()
    assert all(proc.exitcode == 0 for proc in procs)


def test_a_creator_crash_leaves_no_usable_empty_token(token_path):
    """A process killed mid-write must not leave a token every boot then trusts."""
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(1)
    proc = ctx.Process(target=_crashing_worker, args=(token_path, barrier))
    proc.start()
    proc.join(timeout=60)
    assert proc.exitcode == 3, proc.exitcode

    assert not os.path.exists(token_path) or open(token_path, encoding="utf-8").read().strip() == ""
    assert security.get_api_token() != ""
    stored = open(token_path, encoding="utf-8").read().strip()
    assert stored and security.get_api_token() == stored


# ----------------------------------------------------- damaged / existing files
def test_an_empty_file_is_replaced_with_a_complete_token(token_path):
    open(token_path, "w", encoding="utf-8").close()
    token = security.get_api_token()
    assert token
    assert open(token_path, encoding="utf-8").read().strip() == token
    assert os.path.exists(token_path + ".corrupt"), "the damaged file was kept for diagnosis"


def test_a_truncated_file_is_replaced(token_path):
    open(token_path, "w", encoding="utf-8").write("abc")       # a torn write
    token = security.get_api_token()
    assert len(token) == 32, token
    assert open(token_path, encoding="utf-8").read().strip() == token


def test_a_file_whose_digest_does_not_match_is_replaced(token_path):
    open(token_path, "w", encoding="utf-8").write("a" * 32 + "\n")
    open(token_path + ".meta", "w", encoding="utf-8").write(
        json.dumps({"version": 1, "length": 32, "sha256": "0" * 64})
    )
    token = security.get_api_token()
    assert token != "a" * 32
    assert open(token_path, encoding="utf-8").read().strip() == token


def test_an_existing_valid_token_is_reused_untouched(token_path):
    legacy = "b" * 32
    open(token_path, "w", encoding="utf-8").write(legacy + "\n")
    before = os.stat(token_path).st_mtime_ns
    assert security.get_api_token() == legacy
    assert os.stat(token_path).st_mtime_ns == before, "a valid token was rewritten"


def test_an_operator_placed_token_file_is_accepted(token_path):
    """`docs/api.md` documents a token *file* as configured; keep that contract."""
    open(token_path, "w", encoding="utf-8").write("from-mounted-secret-file\n")
    assert security.get_api_token() == "from-mounted-secret-file"


def test_the_environment_token_still_wins(token_path):
    open(token_path, "w", encoding="utf-8").write("file-token-value-long\n")
    os.environ["FIXIMG_API_TOKEN"] = "deployment-token"
    try:
        assert security.get_api_token() == "deployment-token"
    finally:
        os.environ.pop("FIXIMG_API_TOKEN", None)


def test_a_generated_token_is_owner_only(token_path):
    if os.name != "posix":
        pytest.skip("POSIX mode bits only")
    security.get_api_token()
    assert stat.S_IMODE(os.stat(token_path).st_mode) == 0o600


def test_an_unwritable_directory_fails_closed(monkeypatch, tmp_path):
    """No persistence means no single shared secret: refuse rather than guess."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.delenv("FIXIMG_API_TOKEN", raising=False)
    monkeypatch.setattr(security, "token_file", lambda: str(blocker / "api_token.txt"))
    security._cached = None
    with pytest.raises(security.TokenResolutionError):
        security.get_api_token()


def test_production_still_fails_closed_without_a_token(token_path, monkeypatch):
    monkeypatch.setattr(security.settings, "app_env", "production", raising=False)
    with pytest.raises(ValueError, match="Production configuration has no API token"):
        security.get_api_token()
    assert not os.path.exists(token_path)
