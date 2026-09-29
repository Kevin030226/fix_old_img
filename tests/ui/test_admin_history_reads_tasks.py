"""The admin panel's Refresh must show the tasks the pipeline actually ran.

Found by using the running service, not by reading it: as an administrator, the
Refresh buttons on *Task Management* and *Photo Archive* returned "No records" and
"Total tasks: 0" on a deployment that had just finished a task. Clicking Refresh
appeared to do nothing.

Cause: both panels read the V1 `history` table through `history_service`, and
nothing writes that table. Its only writer, `engine.append_history()`, had a
single caller — a test — and an earlier batch removed it, recording the table as
"read-only V1 legacy" while the same entry noted that the admin panel reads it.
The module docstring said the table was kept "during the migration window"; that
window never closed, and the panel was left reading a store nothing populates.

These tests pin the property, not the implementation: whatever the panel reads, a
deployment that has run tasks must show them. `history` is deliberately left empty
in every one of them, so an implementation that went back to reading it would fail.
"""
from __future__ import annotations

import sqlite3

import pytest

from fiximg.infrastructure.db import engine
from fiximg.infrastructure.db.repositories import task_repository as repo
from fiximg.infrastructure.db.repositories import user_repository as users
from fiximg.ui import admin_panel as panel

ADMIN = {"username": "admin"}
REGULAR = {"username": "tester"}


@pytest.fixture(autouse=True)
def panel_users(isolated_db):
    """`_require_admin` reads the role live from the users table.

    So the role is not a property of the session dict - a session naming an
    account that does not exist is refused, which is what makes the check worth
    having. The accounts have to be seeded for the admin path to be reachable.
    """
    from fiximg.infrastructure.security import passwords

    users.add_user("admin", passwords.hash_password("admin-password"), role="admin")
    users.add_user("tester", passwords.hash_password("user-password"), role="user")


def _run_one_task(task_id: str, task_type: str = "restore", user: str = "u1") -> None:
    repo.create_task(task_id, task_type, user, input_path="/tmp/in.png")
    repo.start_task(task_id)
    repo.add_metric(task_id, "psnr", 31.5)
    repo.add_metric(task_id, "ssim", 0.91)
    repo.add_metric(task_id, "mae", 0.021)
    repo.finish_task(task_id, "/tmp/out.png", "restored", 1234)


def test_the_history_table_is_empty_but_the_panel_still_has_rows(isolated_db):
    """The regression, stated as a fact about the store: legacy table, zero rows.

    If this ever fails, the panel is reading something else again, which is the
    whole point — the panel must not depend on a table nobody writes.
    """
    _run_one_task("t-panel-1")
    conn = sqlite3.connect(engine.DB_PATH)
    try:
        legacy = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
    finally:
        conn.close()
    assert legacy == 0, (
        "the legacy `history` table has rows again; if something writes it, the "
        "panel and the task table are two stores for one fact"
    )
    assert panel.refresh_history(ADMIN), "the panel has nothing to show"


def test_task_management_refresh_lists_the_task_that_ran(isolated_db):
    _run_one_task("t-panel-2", "colorize", "alice")
    rows = panel.refresh_history(ADMIN)
    assert rows != [["No records", "", "", "", "", "", ""]], rows
    assert len(rows) == 1, rows
    row = rows[0]
    assert row[1] == "alice", row          # user
    assert row[2] == "colorize", row       # task type
    assert row[3] == "31.5", row           # psnr
    assert row[4] == "0.91", row           # ssim
    assert row[5] == "0.021", row          # mae
    assert row[0], "the timestamp is empty"           # timestamp
    assert row[6] == "in.png", row         # the input file name


def test_photo_archive_refresh_offers_the_task_that_ran(isolated_db):
    """The dropdown the Photo Archive tab drives, keyed by task id."""
    _run_one_task("t-panel-3", "restore_scratch", "bob")
    update = panel.load_archive_list(ADMIN)
    choices = update["choices"] if isinstance(update, dict) else update.choices
    assert choices and choices != [("No records", "")], choices
    # Gradio dropdown choices are (label, value) pairs: the label is what the admin
    # reads, the value is the task id the detail view looks up.
    labels = {value: label for label, value in choices}
    assert "t-panel-3" in labels, choices
    assert "restore_scratch" in labels["t-panel-3"], choices
    assert "in.png" in labels["t-panel-3"], choices


def test_the_summary_does_not_contradict_the_list(isolated_db):
    """It read `history`, so it said "Total tasks: 0" under a full task list."""
    for i in range(3):
        _run_one_task(f"t-panel-{i}", user=f"user{i}")
    stats = panel.refresh_stats(ADMIN)
    assert "Total tasks: 3" in stats, stats
    assert "Users: 3" in stats, stats


def test_one_record_resolves_to_the_same_fact_the_list_showed(isolated_db):
    """The archive detail pane and the list must not disagree.

    `output_path` is the panel's name for what the task table calls `result_path`;
    the detail view asks for `output_path`, so the alias has to be real or the
    pane shows "Record not found" for a row the list just displayed.
    """
    from fiximg.application import history_service

    _run_one_task("t-panel-4", "auto_restore", "carol")
    record = history_service.get_history_record("t-panel-4")
    assert record is not None
    assert record["type"] == "auto_restore", record
    assert record["output_path"] == "/tmp/out.png", record
    assert record["psnr"] == 31.5, record
    assert history_service.get_history_record("no-such-task") is None


def test_a_non_admin_still_gets_nothing(isolated_db):
    _run_one_task("t-panel-5", "restore", "alice")
    for state in (REGULAR, {}, None):
        rows = panel.refresh_history(state)
        assert len(rows) == 1 and "Permission denied" in rows[0][0], (state, rows)
        update = panel.load_archive_list(state)
        choices = update["choices"] if isinstance(update, dict) else update.choices
        assert len(choices) == 1 and "Permission denied" in choices[0][0], (state, choices)


def test_a_deployment_that_has_never_run_a_task_still_answers(isolated_db):
    """Empty is a state the panel must render, not an error."""
    rows = panel.refresh_history(ADMIN)
    assert rows == [["No records", "", "", "", "", "", ""]], rows
    update = panel.load_archive_list(ADMIN)
    choices = update["choices"] if isinstance(update, dict) else update.choices
    assert choices == [("No records", "")], choices


@pytest.mark.parametrize("task_type", ["restore", "colorize", "detect_scratch",
                                       "restore_scratch", "auto_restore"])
def test_every_task_type_is_listed(isolated_db, task_type):
    """The earlier code path keyed on the legacy table's vocabulary; these are the
    task types the API actually accepts, and each must appear."""
    _run_one_task(f"t-type-{task_type}", task_type, "dave")
    rows = panel.refresh_history(ADMIN)
    types = [r[2] for r in rows]
    assert task_type in types, (task_type, types)
