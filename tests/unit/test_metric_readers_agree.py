"""A task's metrics have two readers, and they must not disagree.

`get_task` / `list_tasks` answer with a SQL subquery over the ``metrics`` table;
``task_repository.read_metrics`` reads the same rows in Python. Both exist: the
aggregate is what the API and the history panel use (no extra round trip per
task), the Python read is what an admin view or a migration check wants for one
task.

Neither is a placeholder, and a difference between them would be invisible from
the outside: the payload would still be a well-formed metrics object, the task
would still say `completed`, and only a reader comparing the two sources would
notice. The interesting value is PSNR `+inf` on a bit-identical pair — it cannot
be a REAL and cannot be emitted as JSON, so it lives in ``metric_text`` and the
readers have to put it back as the *string* ``"inf"`` rather than as ``null`` or
as a number.

Runs on SQLite always, and on PostgreSQL when ``FIXIMG_TEST_POSTGRES_URL`` is
set, because the two engines disagree about JSON types and that is precisely
where a collapse rule can go wrong (the dialect has a separate branch per
engine).
"""
from __future__ import annotations

import pytest

from fiximg.domain.tasks import decode_json_column
from fiximg.infrastructure.db.repositories import task_repository as repo

#: A number, a text label, and the value that cannot be a REAL.
SAMPLE = {"psnr": 23.5, "note": "skipped", "psnr_inf": float("inf")}


@pytest.fixture
def task_with_metrics(isolated_db):
    repo.create_task("metrics-1", "restore", "u")
    for name, value in SAMPLE.items():
        repo.add_metric("metrics-1", name, value)
    return "metrics-1"


def test_both_readers_return_the_same_mapping(task_with_metrics):
    """`get_task`, `list_tasks` and `read_metrics` must agree exactly.

    Asserted on the decoded aggregate, because that is the shape a client
    receives: SQLite hands back the JSON as TEXT and PostgreSQL hands back a
    structure, and comparing the raw column would compare a `str` to a `dict`.
    """
    from_aggregate = decode_json_column(repo.get_task("metrics-1")["metrics"], "metrics")
    from_list = decode_json_column(repo.list_tasks(limit=5)[0]["metrics"], "metrics")
    from_python = repo.read_metrics("metrics-1")

    assert from_aggregate == from_python, (
        f"the aggregate answers {from_aggregate!r}, read_metrics answers {from_python!r}"
    )
    assert from_list == from_python, "list_tasks builds a second copy of the same subquery"


def test_an_infinite_metric_survives_as_text_not_null(task_with_metrics):
    """The value a REAL cannot hold has to come back as ``"inf"``.

    Asserted through the production reader specifically: a reader that returned
    ``None`` here would still render, still say ``completed``, and would report a
    perfect restoration as "no PSNR recorded".
    """
    from_aggregate = decode_json_column(repo.get_task("metrics-1")["metrics"], "metrics")

    assert from_aggregate["psnr_inf"] == "inf"
    assert from_aggregate["psnr"] == 23.5, "a REAL must stay a number, not become text"
    assert from_aggregate["note"] == "skipped", "a label stays text on both engines"


def test_a_task_without_metrics_reads_as_an_empty_mapping(isolated_db):
    """No rows is an empty object, not a row of nulls."""
    repo.create_task("metrics-none", "restore", "u")

    assert decode_json_column(repo.get_task("metrics-none")["metrics"], "metrics") == {}
    assert repo.read_metrics("metrics-none") == {}
