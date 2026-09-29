"""Task history panel tests (plan §3.9.3).



The panel reads through the TaskService only; these tests pin the row shape,

pagination, thumbnails, the four preview slots (§3.9.2), role-based scoping and

the re-run path.

"""

import json

import os



import pytest

from PIL import Image



from fiximg.infrastructure.db.repositories import task_repository as repo

from fiximg.ui import history_panel





@pytest.fixture()

def tasks(isolated_db, tmp_path):

    """Four tasks for 'alice' and one for 'bob', with real result files."""

    rows = [

        ("t-1", "alice", "restore", "completed", 100, 4200),

        ("t-2", "alice", "colorize", "failed", 30, 900),

        ("t-3", "alice", "restore", "queued", 0, None),

        ("t-4", "alice", "restore", "completed", 100, 1000),

        ("t-5", "bob", "restore", "completed", 100, 1000),

    ]

    for task_id, user, task_type, status, progress, duration in rows:

        repo.create_task(task_id, task_type, user)

        repo.update_progress(task_id, progress)

        if status == "completed":

            result_path = tmp_path / f"{task_id}_out.png"

            Image.new("RGB", (64, 48), "green").save(result_path)

            repo.start_task(task_id)  # a terminal write only lands on a running row

            repo.finish_task(task_id, str(result_path), "PSNR: 24", duration)

        elif status == "failed":

            repo.start_task(task_id)  # a terminal write only lands on a running row

            repo.fail_task(task_id, "boom", duration)

    repo.add_metric("t-1", "psnr", 24.5, "input_output_difference")

    repo.add_metric("t-1", "ssim", 0.91, "input_output_difference")

    repo.record_stage("t-1", 0, "global_restore", "completed")

    repo.finish_stage("t-1", 0, "completed", 4200)

    return tmp_path





class _Event:

    """Stand-in for gradio's SelectData."""



    def __init__(self, row: int) -> None:

        self.index = (row, 0)





ALICE = {"username": "alice", "role": "user"}

ADMIN = {"username": "admin", "role": "admin"}





def _rows(user_state=ALICE, page=1, page_size=20):

    return history_panel.list_history(user_state, page, page_size)[0]





def _index_of(user_state, task_id, page=1, page_size=20) -> int:

    """Row index of a task on the rendered page (newest first, so not by id)."""

    for index, row in enumerate(_rows(user_state, page, page_size)):

        if row[0] == task_id:

            return index

    raise AssertionError(f"{task_id} is not on page {page}")





# --------------------------------------------------------------------- table

def test_list_history_returns_rows_gallery_and_pagination(tasks):

    rows, gallery, label, page, status = history_panel.list_history(ALICE, 1, 20)

    assert len(rows) == 4                      # alice's tasks only

    assert label.startswith("Page 1 of")

    assert page == 1

    assert "4 row(s)" in status

    assert "alice" in status

    for row in rows:

        assert len(row) == len(history_panel.COLUMNS)





def test_list_history_scopes_to_the_caller(tasks):

    assert len(_rows(ALICE)) == 4

    assert len(_rows(ADMIN)) == 5              # admin sees every user

    _rows_out, _gallery, _label, _page, status = history_panel.list_history(ADMIN, 1, 20)

    assert "all users" in status





def test_history_row_formats_duration_status_and_metrics(tasks):

    by_id = {row[0]: row for row in _rows()}



    assert by_id["t-1"][3] == "4.2 s"

    assert by_id["t-1"][4] == "completed"

    assert by_id["t-1"][5] == "100%"

    assert "global_restore" in by_id["t-1"][6]

    assert "PSNR=24.5" in by_id["t-1"][7]



    assert by_id["t-3"][3] == "—"              # never ran

    assert by_id["t-3"][4] == "queued"





def test_list_history_survives_a_broken_service(monkeypatch, tasks):

    class Boom:

        def list_tasks(self, **_kwargs):

            raise RuntimeError("db gone")



    monkeypatch.setattr(history_panel, "task_service", Boom())

    rows, gallery, _label, _page, status = history_panel.list_history(ALICE, 1, 20)

    assert rows == []

    assert gallery == []

    assert "Could not load history" in status





# ---------------------------------------------------------------- pagination

def test_pagination_splits_the_history(tasks):

    page1, _g1, label1, state1, _ = history_panel.list_history(ALICE, 1, 2)

    page2, _g2, label2, state2, _ = history_panel.list_history(ALICE, 2, 2)



    assert len(page1) == 2

    assert len(page2) == 2

    assert state1 == 1 and state2 == 2

    # 4 rows at 2/page is exactly two pages; page 1 knows more is coming.

    assert label1 == "Page 1 of 2"

    assert label2 == "Page 2 of 2"             # last page: exact count

    assert {r[0] for r in page1} & {r[0] for r in page2} == set()





def test_pagination_reports_a_single_page_when_everything_fits(tasks):

    _rows_out, _g, label, _state, _status = history_panel.list_history(ALICE, 1, 50)

    assert label == "Page 1 of 1"





def test_next_page_advances(tasks):

    _rows_out, _g, _label, page, _status = history_panel.next_page(ALICE, 2, 1)

    assert page == 2





def test_prev_page_never_goes_below_one(tasks):

    _rows_out, _g, _label, page, _status = history_panel.prev_page(ALICE, 2, 1)

    assert page == 1





def test_page_beyond_the_end_clamps_to_the_last_page(tasks):

    rows, _g, _label, page, _status = history_panel.list_history(ALICE, 99, 2)

    assert page == 2

    assert len(rows) == 2





def test_page_size_is_clamped(tasks):

    rows, _g, _label, _page, _status = history_panel.list_history(ALICE, 1, 9999)

    assert len(rows) <= 200

    rows, _g, _label, _page, _status = history_panel.list_history(ALICE, 1, 0)

    assert len(rows) >= 1





def test_invalid_page_values_fall_back_to_defaults(tasks):

    _rows_out, _g, label, page, _status = history_panel.list_history(ALICE, "x", "y")

    assert page == 1

    assert label.startswith("Page 1")





# ---------------------------------------------------------------- thumbnails

def test_gallery_carries_a_thumbnail_per_completed_task(tasks):

    _rows_out, gallery, _label, _page, _status = history_panel.list_history(ALICE, 1, 20)

    # t-1 and t-4 completed; t-2 failed and t-3 never ran.

    assert len(gallery) == 2

    for path, caption in gallery:
        assert os.path.isfile(path)
        assert "·" in caption




def test_thumbnail_is_smaller_than_the_original(tasks):

    source = tasks / "t-1_out.png"

    thumb = history_panel.make_thumbnail(str(source), side=32)

    assert thumb is not None

    with Image.open(thumb) as image:

        assert max(image.size) <= 32





def test_thumbnail_is_cached(tasks):

    source = str(tasks / "t-1_out.png")

    first = history_panel.make_thumbnail(source, side=32)

    second = history_panel.make_thumbnail(source, side=32)

    assert first == second





def test_thumbnail_returns_none_for_a_missing_file(tasks):

    assert history_panel.make_thumbnail(str(tasks / "nope.png")) is None

    assert history_panel.make_thumbnail("") is None





def test_thumbnail_returns_none_for_a_non_image(tasks):

    junk = tasks / "junk.png"

    junk.write_text("not an image", encoding="utf-8")

    assert history_panel.make_thumbnail(str(junk)) is None





def test_thumbnail_cache_dir_sits_next_to_the_task_storage(tasks, monkeypatch):

    import fiximg.config as config_mod



    monkeypatch.setattr(config_mod.settings, "tasks_root", str(tasks / "storage" / "tasks"))

    assert history_panel.thumbnail_cache_dir() == str(tasks / "storage" / "thumbs")





# ------------------------------------------------------------------ previews

def _add_input_and_mask(task_id, tmp_path, *, with_mask=True):

    input_path = tmp_path / f"{task_id}_in.png"

    Image.new("RGB", (48, 32), "orange").save(input_path)

    repo._get_conn().execute(

        "UPDATE tasks SET input_path=? WHERE id=?", (str(input_path), task_id)

    )

    if with_mask:

        mask_path = tmp_path / f"{task_id}_mask.png"

        Image.new("L", (48, 32), 128).save(mask_path)

        repo.add_artifact(task_id, "mask", str(mask_path), "image/png")

    repo._get_conn().commit()

    return input_path





def test_load_history_entry_fills_every_preview(tasks):

    input_path = _add_input_and_mask("t-1", tasks)

    index = _index_of(ALICE, "t-1")



    before, after, mask, face, slider, mask_slider, detail, download = history_panel.load_history_entry(

        ALICE, 1, 20, _Event(index)

    )

    assert before == str(input_path)

    assert after == str(tasks / "t-1_out.png")

    assert mask is not None and mask.endswith("t-1_mask.png")

    assert face is None                        # no face crops for this task

    assert slider["value"] == (str(input_path), str(tasks / "t-1_out.png"))

    assert mask_slider["visible"] is True

    assert mask_slider["value"] == (str(input_path), mask)

    assert "t-1" in detail

    assert "global_restore" in detail





def test_mask_slider_is_hidden_without_a_mask(tasks):

    _add_input_and_mask("t-1", tasks, with_mask=False)

    index = _index_of(ALICE, "t-1")

    _b, _a, mask, _f, _slider, mask_slider, _detail, _dl = history_panel.load_history_entry(

        ALICE, 1, 20, _Event(index)

    )

    assert mask is None

    assert mask_slider["visible"] is False





def test_face_crop_preview_uses_the_first_crop(tasks):

    crops_dir = tasks / "faces"

    crops_dir.mkdir()

    for name in ("face_1.png", "face_0.png"):

        Image.new("RGB", (24, 24), "red").save(crops_dir / name)

    repo.add_artifact("t-1", "faces_dir", str(crops_dir), "inode/directory")



    _b, _a, _m, face, _slider, _ms, detail, _dl = history_panel.load_history_entry(

        ALICE, 1, 20, _Event(_index_of(ALICE, "t-1"))

    )

    assert face is not None and face.endswith("face_0.png")   # sorted first

    assert "Task: t-1" in detail





def test_load_history_entry_without_a_result_file(tasks):

    # t-1 has a result_path but the file does not exist on disk.

    os.remove(tasks / "t-1_out.png")

    _b, after, _m, _f, slider, _ms, detail, download = history_panel.load_history_entry(

        ALICE, 1, 20, _Event(_index_of(ALICE, "t-1"))

    )

    assert after is None

    assert "value" not in slider               # no-op update

    assert "reclaimed" in detail





def test_load_history_entry_reports_the_error_message(tasks):

    _b, _a, _m, _f, _s, _ms, detail, _dl = history_panel.load_history_entry(

        ALICE, 1, 20, _Event(_index_of(ALICE, "t-2"))

    )

    assert "boom" in detail





def test_load_history_entry_handles_a_missing_selection(tasks):

    before, after, _m, _f, _s, _ms, detail, _dl = history_panel.load_history_entry(

        ALICE, 1, 20, None

    )

    assert (before, after) == (None, None)

    assert "Select a row" in detail





def test_load_history_entry_out_of_range_row(tasks):

    before, after, _m, _f, _s, _ms, detail, _dl = history_panel.load_history_entry(

        ALICE, 1, 20, _Event(99)

    )

    assert (before, after) == (None, None)

    assert "Select a row" in detail





def test_preview_resolves_the_row_relative_to_the_page(tasks):

    """Row 0 of page 2 must be the third task, not the first."""

    page2_rows, _g, _l, _p, _s = history_panel.list_history(ALICE, 2, 2)

    _b, _a, _m, _f, _s2, _ms, detail, _dl = history_panel.load_history_entry(

        ALICE, 2, 2, _Event(0)

    )

    assert page2_rows[0][0] in detail





# -------------------------------------------------------------------- re-run

def test_rerun_enqueues_the_same_mode(tasks, monkeypatch):

    _add_input_and_mask("t-1", tasks)

    repo._get_conn().execute(

        "UPDATE tasks SET options_json=? WHERE id='t-1'", (json.dumps({"hr": True}),)

    )

    repo._get_conn().commit()



    captured = {}



    class FakeService:

        def list_tasks(self, limit=50, user_id=None):

            # Row index -> task id resolution reads through the service, so the

            # fake must still answer it (that is part of the panel contract).

            return repo.list_tasks(limit=limit, user_id=user_id)



        def get_task(self, task_id):

            return repo.get_task(task_id)



        def enqueue(self, image, user_state, mode, options=None, **_kw):

            captured.update({"mode": mode, "options": options, "size": image.size})

            return "t-new"



    row_index = _index_of(ALICE, "t-1")

    monkeypatch.setattr(history_panel, "task_service", FakeService())

    _rows_out, _g, _label, _page, status = history_panel.rerun_history_entry(

        ALICE, 1, 20, _Event(row_index)

    )



    assert captured["mode"] == "restore"       # task_type -> UI mode

    assert captured["options"] == {"hr": True}

    assert captured["size"] == (48, 32)

    assert "t-new" in status





def test_rerun_drops_internal_option_keys(tasks, monkeypatch):

    """A ground-truth path from the old run must not be forwarded."""

    _add_input_and_mask("t-1", tasks)

    repo._get_conn().execute(

        "UPDATE tasks SET options_json=? WHERE id='t-1'",

        (json.dumps({"hr": False, "_ground_truth_path": "/old/gt.png"}),),

    )

    repo._get_conn().commit()



    captured = {}



    class FakeService:

        def list_tasks(self, limit=50, user_id=None):

            return repo.list_tasks(limit=limit, user_id=user_id)



        def get_task(self, task_id):

            return repo.get_task(task_id)



        def enqueue(self, image, user_state, mode, options=None, **_kw):

            captured["options"] = options

            return "t-new"



    row_index = _index_of(ALICE, "t-1")

    monkeypatch.setattr(history_panel, "task_service", FakeService())

    history_panel.rerun_history_entry(ALICE, 1, 20, _Event(row_index))



    # The internal ground-truth path pointed at the *old* run's artifact and must

    # not be forwarded; the declared switches are kept as stored.

    assert captured["options"] == {"hr": False}

    assert "_ground_truth_path" not in captured["options"]





def test_rerun_reports_a_missing_input(tasks):

    # t-2 has no input_path at all.

    _rows_out, _g, _label, _page, status = history_panel.rerun_history_entry(

        ALICE, 1, 20, _Event(_index_of(ALICE, "t-2"))

    )

    assert "Cannot re-run" in status





def test_rerun_reports_a_failure(tasks, monkeypatch):

    _add_input_and_mask("t-1", tasks)



    class Boom:

        def list_tasks(self, limit=50, user_id=None):

            return repo.list_tasks(limit=limit, user_id=user_id)



        def get_task(self, task_id):

            return repo.get_task(task_id)



        def enqueue(self, *a, **kw):

            raise RuntimeError("queue saturated")



    row_index = _index_of(ALICE, "t-1")

    monkeypatch.setattr(history_panel, "task_service", Boom())

    _rows_out, _g, _label, _page, status = history_panel.rerun_history_entry(

        ALICE, 1, 20, _Event(row_index)

    )

    assert "Re-run failed" in status





# ------------------------------------------------------------------- helpers

def test_task_type_to_mode_covers_every_pipeline_mode():

    from fiximg.application.pipeline_modes import PIPELINE_MODES



    for cfg in PIPELINE_MODES.values():

        assert cfg["task_type"] in history_panel._TASK_TYPE_TO_MODE





def test_metrics_text_is_compact_and_ordered():

    text = history_panel._metrics_text({"mae": 1.2, "psnr": 24.0, "ssim": 0.9, "other": 1})

    assert text == "PSNR=24.0  SSIM=0.9  MAE=1.2"

    assert history_panel._metrics_text({}) == ""

    assert history_panel._metrics_text("not a dict") == ""





def test_duration_text_formats_minutes():

    assert history_panel._duration_text(None) == "—"
    assert history_panel._duration_text(0) == "—"
    assert history_panel._duration_text(4200) == "4.2 s"

    assert history_panel._duration_text(95_000) == "1m 35s"





def test_artifact_lookup_prefers_the_first_existing_kind(tasks):

    mask_path = tasks / "m.png"

    Image.new("L", (8, 8)).save(mask_path)

    repo.add_artifact("t-1", "mask", str(mask_path), "image/png")

    assert history_panel._artifact_path("t-1", "missing", "mask") == str(mask_path)

    assert history_panel._artifact_path("t-1", "nope") is None





def test_when_text_renders_the_viewers_wall_clock():

    """Storage is UTC; a person reads local (plan §2.7)."""

    from datetime import datetime



    stored = "2026-06-01T00:00:00.000000Z"

    expected = (

        datetime.fromisoformat(stored.replace("Z", "+00:00"))

        .astimezone()

        .strftime("%Y-%m-%d %H:%M:%S")

    )

    assert history_panel._when_text(stored) == expected





def test_when_text_passes_through_what_the_codec_cannot_read():

    """A V1 row predates the codec; blanking the column would lose the row's date."""

    assert history_panel._when_text("2026-06-01 08:00:00") == "2026-06-01 08:00:00"

    assert history_panel._when_text("") == ""

    assert history_panel._when_text(None) == ""





# ----------------------------------------------------------------------- retry

class _RetryingService:

    """The panel reaches the repository through the service, so the fake must too."""



    def __init__(self, outcome=True):

        self.calls = []

        self.outcome = outcome



    def list_tasks(self, limit=50, user_id=None):

        return repo.list_tasks(limit=limit, user_id=user_id)



    def get_task(self, task_id):

        return repo.get_task(task_id)



    def retry_task(self, task_id):

        self.calls.append(task_id)

        if isinstance(self.outcome, Exception):

            raise self.outcome

        return self.outcome





def test_retry_requeues_the_failed_row_under_its_own_id(tasks, monkeypatch):

    """§3.8: retry is not "submit a duplicate".



    The endpoint resets the attempt budget and re-queues the *same* task id, so the

    history row, its artifacts and its metrics keep describing one job. Before the

    panel had this button, the only retry in the UI silently meant "new task".

    """

    service = _RetryingService()

    monkeypatch.setattr(history_panel, "task_service", service)

    _add_input_and_mask("t-2", tasks)



    status = history_panel.retry_history_entry(

        ALICE, 1, 20, _Event(_index_of(ALICE, "t-2"))

    )[4]



    assert service.calls == ["t-2"], service.calls

    assert "t-2" in status and "same id" in status, status

    assert repo.get_task("t-2")["status"] == "failed", "the fake must not write state"





def test_retry_refuses_a_running_or_completed_task_without_calling_the_service(

    tasks, monkeypatch

):

    """A completed task retried by accident would silently re-run GPU work."""

    service = _RetryingService()

    monkeypatch.setattr(history_panel, "task_service", service)



    status = history_panel.retry_history_entry(

        ALICE, 1, 20, _Event(_index_of(ALICE, "t-1"))

    )[4]



    assert service.calls == [], "retry was invoked on a completed task"

    assert "only applies to a failed or cancelled task" in status, status

    assert "Re-run" in status, "the message should point at the other button"





def test_retry_reports_a_service_failure_instead_of_raising(tasks, monkeypatch):

    """The caption is the user-facing answer; a traceback in a Gradio callback is not."""

    service = _RetryingService(outcome=RuntimeError("lease still held"))

    monkeypatch.setattr(history_panel, "task_service", service)

    _add_input_and_mask("t-2", tasks)



    status = history_panel.retry_history_entry(

        ALICE, 1, 20, _Event(_index_of(ALICE, "t-2"))

    )[4]



    assert status.startswith("⚠ Retry failed") and "lease still held" in status





def test_retry_that_found_nothing_to_requeue_says_so(tasks, monkeypatch):

    service = _RetryingService(outcome=False)

    monkeypatch.setattr(history_panel, "task_service", service)

    _add_input_and_mask("t-2", tasks)



    status = history_panel.retry_history_entry(

        ALICE, 1, 20, _Event(_index_of(ALICE, "t-2"))

    )[4]



    assert "was not requeued" in status, status





def test_retry_with_no_selection_prompts_for_one(tasks, monkeypatch):

    service = _RetryingService()

    monkeypatch.setattr(history_panel, "task_service", service)



    status = history_panel.retry_history_entry(ALICE, 1, 20, _Event(None))[4]



    assert "Select a row" in status

    assert service.calls == []

