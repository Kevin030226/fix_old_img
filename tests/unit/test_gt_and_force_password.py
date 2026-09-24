"""Unit tests for §16 Ground Truth evaluation and §21 force password change."""
import pytest

from app.core.exceptions import PasswordChangeRequiredError
from app.services import gt_metrics_service as gt
from app.services.task_service import TaskService


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """Isolated SQLite file for every test (same pattern as test_bootstrap)."""
    import app.core.config as config_mod
    import app.db as legacy_db
    from app.repositories import task_repository as task_repo

    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(legacy_db, "DB_PATH", db_path)
    monkeypatch.setattr(legacy_db, "ADMIN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(legacy_db, "_conn", None)
    monkeypatch.setattr(config_mod.settings, "db_path", db_path)
    task_repo._DDL_DONE = False
    legacy_db.init_db()
    return tmp_path


# ------------------------------------------------------- §16 ground truth
@pytest.fixture(autouse=True)
def _no_lpips_package(monkeypatch):
    """Tests run against the deterministic surrogate (lpips not installed)."""
    gt._reset_lpips_cache()
    monkeypatch.setattr(gt, "_load_lpips", lambda: None)
    yield
    gt._reset_lpips_cache()


def _write(path, color, size=(64, 48)):
    from PIL import Image

    Image.new("RGB", size, color).save(path)
    return path


def test_gt_identical_images_zero(tmp_path):
    ref = _write(str(tmp_path / "ref.png"), (120, 120, 120))
    score = gt.compute_ground_truth_metrics(ref, ref)
    assert score["lpips"] == 0.0
    assert score["backend"] == "laplacian_surrogate"
    assert score["notice"] == gt.GT_NOTICE


def test_gt_orders_similarity(tmp_path):
    ref = _write(str(tmp_path / "ref.png"), (120, 120, 120))
    close = _write(str(tmp_path / "close.png"), (125, 122, 118))
    far = _write(str(tmp_path / "far.png"), (30, 200, 90))
    s_ref = gt.compute_ground_truth_metrics(ref, ref)["lpips"]
    s_close = gt.compute_ground_truth_metrics(ref, close)["lpips"]
    s_far = gt.compute_ground_truth_metrics(ref, far)["lpips"]
    assert s_close > s_ref
    assert s_far > s_close


def test_gt_resizes_mismatched_shapes(tmp_path):
    ref = _write(str(tmp_path / "ref.png"), (100, 100, 100))
    other = _write(str(tmp_path / "other.png"), (100, 100, 100), size=(32, 32))
    score = gt.compute_ground_truth_metrics(ref, other)
    assert score is not None and score["lpips"] >= 0.0


def test_gt_unreadable_returns_none(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    ref = _write(str(tmp_path / "ref.png"), (1, 2, 3))
    assert gt.compute_ground_truth_metrics(ref, str(bad)) is None


def test_gt_report_contains_notice():
    text = gt.format_ground_truth_report(
        {"lpips": 0.1234, "backend": "laplacian_surrogate"}
    )
    assert "LPIPS: 0.1234" in text
    assert "⚠" in text


def test_gt_reference_type_is_ground_truth():
    from app.services.evaluation_service import REFERENCE_TYPE as DIFF
    from app.services.metrics_service import REFERENCE_TYPE as NR

    assert gt.REFERENCE_TYPE == "ground_truth"
    assert len({gt.REFERENCE_TYPE, DIFF, NR}) == 3


# --------------------------------------------- §21 force password change
@pytest.fixture()
def service():
    return TaskService()


def _make_user(username, must_change=False):
    from app.core.security import hash_password
    from app.repositories import user_repository as user_repo

    user_repo.add_user(username, hash_password("pw-initial-123"), "user")
    if must_change:
        user_repo.set_must_change_password(username, True)


def test_force_change_blocks_submit(service, isolated_db):
    _make_user("forcee", must_change=True)
    with pytest.raises(PasswordChangeRequiredError):
        service._require_password_changed({"username": "forcee"})


def test_normal_user_can_submit(service, isolated_db):
    _make_user("normal1")
    service._require_password_changed({"username": "normal1"})  # no raise


def test_updating_password_clears_flag(service, isolated_db):
    _make_user("forcee2", must_change=True)
    from app.services import user_service

    user_service.update_user("forcee2", password="brand-new-pw")
    service._require_password_changed({"username": "forcee2"})  # no raise


def test_missing_user_is_noop(service, isolated_db):
    service._require_password_changed({"username": "ghost"})  # no raise
