"""The repository's JSON columns have two shapes, and every reader must know it.

``tasks.stages`` and ``tasks.metrics`` are SQL aggregates: ``json_group_array`` returns
TEXT on SQLite, while ``json_agg`` on PostgreSQL is decoded by psycopg before the caller
sees it. The option and capability blobs are stored as TEXT and read back by whoever
needs them. So each of these columns is *either* a string or a structure depending on the
deployment, and a reader that trusts one shape fails in the direction of showing nothing
rather than failing loudly:

* the progress panel treated a string stage list as "no stages", so on the default
  engine its 搂24 checklist never appeared while every panel test 鈥?which built its own
  rows as lists 鈥?stayed green;
* ``_decode_capabilities`` wrapped ``json.loads`` in ``except TypeError``, so on
  PostgreSQL the already-decoded list raised, became the empty set, and each worker
  claimed tasks it could not serve.

One decoder, one gate over the readers, and a column list derived from the repository
rather than kept beside it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from fiximg.domain.tasks import JSON_COLUMNS, decode_json_column

SRC = Path(__file__).resolve().parents[2] / "src" / "fiximg"


# ------------------------------------------------------------------ the decoder
@pytest.mark.parametrize("value,expected", [
    ('[{"stage_name": "global_restore"}]', [{"stage_name": "global_restore"}]),
    ([{"stage_name": "global_restore"}], [{"stage_name": "global_restore"}]),
    ('{"hr": true}', {"hr": True}),
    ({"hr": True}, {"hr": True}),
    ('["colorize"]', ["colorize"]),
    (["colorize"], ["colorize"]),
    (None, None),
    ("", None),
    ("not json", None),
])
def test_both_engine_shapes_decode_to_the_same_structure(value, expected):
    assert decode_json_column(value) == expected


def test_a_default_is_handed_back_for_missing_or_unparsable_text():
    """A row written by an older build still has to render an empty checklist."""
    assert decode_json_column(None, []) == []
    assert decode_json_column("", {}) == {}
    assert decode_json_column("{oops", []) == []


def test_the_column_names_are_the_ones_the_repositories_actually_write():
    """A stale entry would silently retire one reader from the gate below.

    Derived from the repository sources rather than from a list kept next to this test.
    """
    from fiximg.infrastructure.db.repositories import (
        model_repository,
        task_repository,
    )

    written = (Path(task_repository.__file__).read_text(encoding="utf-8")
               + Path(model_repository.__file__).read_text(encoding="utf-8"))
    for column in sorted(JSON_COLUMNS):
        assert column in written, f"{column} is no longer a repository column"
    assert JSON_COLUMNS, "the column list disappeared"


# ---------------------------------------------------------------------- readers
def _read_sites(source: str, column: str) -> list[int]:
    """Line numbers where `column` is read out of a mapping (writes are not reads)."""
    lines: list[int] = []
    get_pattern = re.compile(
        r"\.get\(\s*[\"']" + re.escape(column) + r"[\"']"
    )
    index_pattern = re.compile(
        r"\[\s*[\"']" + re.escape(column) + r"[\"']\s*\]"
    )
    for number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if get_pattern.search(line):
            lines.append(number)
            continue
        for match in index_pattern.finditer(line):
            after = line[match.end():].lstrip()
            if after.startswith("=") and not after.startswith("=="):
                continue  # an assignment into a payload dict, not a column read
            lines.append(number)
    return lines


def test_every_reader_of_a_json_column_goes_through_the_shared_decoder():
    """The class gate: no module may decode (or mis-decode) these columns by hand.

    Scans the production tree, so a new reader is caught here instead of by the next
    live boot 鈥?which is how the two defects in this module's docstring were found.
    """
    offenders: list[str] = []
    readers: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        sites = [column for column in JSON_COLUMNS if _read_sites(source, column)]
        if not sites:
            continue
        readers.append(path.name)
        if "decode_json_column" not in source:
            offenders.append(f"{path.relative_to(SRC.parent)} reads {sorted(sites)}")

    assert not offenders, "JSON columns read without the shared decoder: " + "; ".join(offenders)
    assert len(readers) >= 6, f"the scan found only {readers}"


# ----------------------------------------------------- the capability routing hint
def test_the_capability_hint_survives_the_engine_that_already_decoded_it():
    """Both shapes must produce the same filter decision, not a permissive one.

    The empty set means "this task needs nothing", which every worker accepts, so the
    failure mode was silent: on a PostgreSQL deployment the specialised-worker pinning
    stopped applying to any task.
    """
    from fiximg.infrastructure.db.repositories import task_repository

    as_text = json.dumps(["colorize", "restore"])
    as_structure = ["colorize", "restore"]

    assert task_repository._decode_capabilities(as_text) == {"colorize", "restore"}
    assert task_repository._decode_capabilities(as_structure) == {"colorize", "restore"}
    assert task_repository._decode_capabilities(None) == set()
    assert task_repository._decode_capabilities("not json") == set()
    # A JSON scalar is not a capability list: it must not become {"c","o","l","o","r"}.
    assert task_repository._decode_capabilities('"colorize"') == set()


def test_the_task_options_are_read_from_either_shape():
    """The same trap on the options column: a dict-shaped row is not "no options"."""
    from fiximg.domain.tasks import Task

    from_text = Task.from_row({
        "id": "t-1", "task_type": "auto_restore",
        "options_json": json.dumps({"face_enhance": False}),
    })
    from_structure = Task.from_row({
        "id": "t-2", "task_type": "auto_restore",
        "options_json": {"face_enhance": False},
    })

    assert from_text.options.face_enhance is False
    assert from_structure.options.face_enhance is False
    assert from_text.options.extra == from_structure.options.extra == {}
