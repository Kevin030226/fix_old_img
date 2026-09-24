"""Unit tests for the PipelineWorker loop (plan sections 12/28)."""
import threading

import pytest

from app.repositories import task_repository as repo


@pytest.fixture()
def db_env(tmp_path, monkeypatch):
    import app.db as legacy_db
    import app.core.config as config_mod

    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(legacy_db, "DB_PATH", db_path)
    monkeypatch.setattr(legacy_db, "ADMIN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(legacy_db, "_conn", None)
    monkeypatch.setattr(config_mod.settings, "db_path", db_path)
    repo._DDL_DONE = False
    return db_path


class FakeOrchestrator:
    """Records claimed executions; optionally fails them."""

    def __init__(self, fail: bool = False):
        self.calls = []
        self.fail = fail
        self._lock = threading.Lock()

    def execute_queued(self, task_id: str, task_type: str) -> None:
        with self._lock:
            self.calls.append((task_id, task_type))
        if self.fail:
            raise RuntimeError("boom")


def test_worker_executes_queued_task(db_env):
    from app.inference.worker import PipelineWorker

    repo.create_task("wq1", "restore", "u", input_path="unused.png")
    fake = FakeOrchestrator()
    worker = PipelineWorker(fake, poll_seconds=0.05, worker_id="w-test")
    worker.start()
    try:
        deadline = threading.Event()
        for _ in range(100):
            if fake.calls:
                break
            deadline.wait(0.05)
        assert fake.calls == [("wq1", "restore")]
        row = repo.get_task("wq1")
        # FakeOrchestrator doesn't finish tasks; the row stays running (claimed).
        assert row["status"] == "running"
    finally:
        worker.stop()
    assert not worker.is_running()


def test_worker_failure_keeps_polling(db_env):
    from app.inference.worker import PipelineWorker

    repo.create_task("wq2", "colorize", "u", input_path="unused.png")
    repo.create_task("wq3", "restore", "u", input_path="unused.png")
    fake = FakeOrchestrator(fail=True)
    worker = PipelineWorker(fake, poll_seconds=0.05, worker_id="w-test")
    worker.start()
    try:
        deadline = threading.Event()
        for _ in range(200):
            if len(fake.calls) >= 2:
                break
            deadline.wait(0.05)
        assert {t for t, _ in fake.calls} == {"wq2", "wq3"}
    finally:
        worker.stop()


def test_worker_depth(db_env):
    from app.inference.worker import PipelineWorker

    repo.create_task("wq4", "restore", "u")
    repo.create_task("wq5", "restore", "u")
    fake = FakeOrchestrator()
    worker = PipelineWorker(fake, poll_seconds=0.05)
    assert worker.depth() == 2
