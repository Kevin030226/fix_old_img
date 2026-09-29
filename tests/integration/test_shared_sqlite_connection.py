"""The shared SQLite connection under concurrent use (plan 搂3.2 topology, 搂2.7 data layer).

`fiximg.app_factory` runs the worker in a thread of the same process that serves
HTTP requests, and the SQLite engine hands both **one** connection. That is the
deployment the plan describes, so a statement racing another thread's `commit()` is
not an exotic case 鈥?it is the normal case, and until `Connection` serialised its
calls it failed on Python 3.14 with ``SystemError: error return without exception
set`` or ``InterfaceError: bad parameter or other API misuse``. In the full suite it
showed up as roughly one failing test per run, and never the same test twice, which
is what made it worth a dedicated regression rather than a retry.
"""
from __future__ import annotations

import threading
import traceback

from fiximg.infrastructure.db.repositories import task_repository as task_repo


def test_writers_and_readers_sharing_one_connection_do_not_corrupt_it(isolated_db):
    """Every row a thread wrote is readable, and no thread died in the driver.

    Asserted as behaviour (the rows are all there, with the status the writer set),
    because "no exception" alone would also pass if the writes were being lost.
    """
    writers = 4
    per_writer = 40
    errors: list[str] = []
    barrier = threading.Barrier(writers + 2)

    def writer(tag: int) -> None:
        try:
            barrier.wait(timeout=10)
            for i in range(per_writer):
                task_id = f"c-{tag}-{i}"
                task_repo.create_task(task_id, "restore", "alice")
                task_repo.start_task(task_id)
                task_repo.update_progress(task_id, 50, "global_restore")
        except BaseException:  # noqa: BLE001 鈥?the thread reports, pytest asserts
            errors.append(f"writer{tag}: {traceback.format_exc()}")

    def reader(tag: str) -> None:
        """Reads and commits, which is the interleaving that used to raise."""
        from fiximg.infrastructure.db import engine

        try:
            barrier.wait(timeout=10)
            for _ in range(per_writer * 3):
                engine.get_conn().commit()
                task_repo.get_task(f"c-0-{0}")
                task_repo.queued_count()
        except BaseException:  # noqa: BLE001
            errors.append(f"{tag}: {traceback.format_exc()}")

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(writers)]
    threads += [threading.Thread(target=reader, args=(f"reader{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert not errors, f"shared-connection failures: {errors[:4]}"
    assert all(not t.is_alive() for t in threads), "a thread wedged in the driver"

    for tag in range(writers):
        for i in range(per_writer):
            row = task_repo.get_task(f"c-{tag}-{i}")
            assert row is not None, f"c-{tag}-{i} was written and cannot be read"
            assert row["status"] == "running", row["status"]
            assert row["progress"] == 50, row["progress"]


def test_a_rolled_back_write_leaves_no_row(isolated_db):
    """The lock is not a transaction, and the test says so out loud.

    A failed statement inside a repository call is rolled back by that call's own
    error handling (SQLite aborts the statement); this checks a write that never
    reached `commit()` is simply absent 鈥?the property a caller can actually rely on
    when the connection is shared.
    """
    from fiximg.infrastructure.db import engine

    task_repo.create_task("keep-1", "restore", "alice")
    conn = engine.get_conn()
    conn.execute(
        "INSERT INTO tasks(id, task_type, status, created_at) VALUES (?,?,?,?)",
        ("never-committed", "restore", "queued", "2026-01-01T00:00:00.000000Z"),
    )
    conn.rollback()

    assert task_repo.get_task("keep-1") is not None
    assert task_repo.get_task("never-committed") is None
