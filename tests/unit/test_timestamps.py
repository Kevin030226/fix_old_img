"""The instant codec (plan 搂2.7).

These tests are the reason the storage format is what it is. Each one pins a
property that the old naive-second-precision strings did not have:

* fixed width, so a TEXT comparison *is* a chronological one,
* sub-second precision, so two submissions in the same second keep their order,
* an explicit zone, so a lease measured across a DST boundary is the length it
  was asked for,
* and a reader that still understands what V2 wrote, because the migration and
  the code that reads the rows are not deployed atomically.
"""
from datetime import UTC, datetime, timedelta, timezone

import pytest

from fiximg.infrastructure.db import timestamps


def test_canonical_is_fixed_width_and_lexicographically_ordered():
    earlier = timestamps.from_epoch(1_800_000_000)
    later = timestamps.from_epoch(1_800_000_001)
    # "YYYY-MM-DDTHH:MM:SS.ffffffZ" 鈥?the width is what makes `<` chronological.
    assert len(earlier) == 27 == len(later)
    assert earlier < later
    assert later.endswith("Z")


def test_same_second_submissions_still_order():
    """The V2 format collapsed both of these to the same 19-character string."""
    base = 1_800_000_000
    a = timestamps.from_epoch(base + 0.001)
    b = timestamps.from_epoch(base + 0.002)
    assert a != b
    assert a < b


def test_now_is_utc_and_parse_round_trips():
    text = timestamps.now()
    assert text.endswith("Z")
    again = timestamps.canonical(timestamps.parse(text))
    assert again == text


def test_cutoff_and_deadline_bracket_now():
    now = timestamps.now()
    past = timestamps.cutoff(60)
    future = timestamps.deadline(60)
    assert past < now < future
    assert abs((timestamps.parse(future) - timestamps.parse(now)).total_seconds() - 60) < 2


def test_legacy_naive_value_is_read_as_local_time():
    """V2 wrote `datetime.now()` 鈥?a naive *local* string.

    The migration rewrites them, but a database that has not been migrated yet
    must still be readable rather than raising, and reading it as UTC would move
    every row by the host's offset.
    """
    legacy = "2026-06-01 08:00:00"
    parsed = timestamps.parse(legacy)
    assert parsed.tzinfo is UTC
    expected = datetime(2026, 6, 1, 8, 0, 0).astimezone(UTC)
    assert parsed == expected


def test_explicit_offset_is_respected_not_assumed():
    assert timestamps.parse("2026-06-01T08:00:00+08:00") == datetime(
        2026, 6, 1, 0, 0, tzinfo=UTC
    )


def test_age_is_never_negative():
    """A row written by a clock running slightly ahead is not a negative wait."""
    future = timestamps.deadline(300)
    assert timestamps.age_seconds(future) == 0.0
    assert timestamps.age_seconds(timestamps.cutoff(3)) >= 3.0


def test_elapsed_since_is_a_timedelta():
    assert timestamps.elapsed_since(timestamps.cutoff(90)) >= timedelta(seconds=89)


def test_unreadable_values_raise_a_domain_error_not_a_crash():
    for bad in ("", None, "yesterday", "2026-13-45 99:99:99"):
        with pytest.raises(ValueError):
            timestamps.parse(bad)


def test_epoch_and_age_agree():
    text = timestamps.cutoff(10)
    assert abs(timestamps.epoch(text) - (datetime.now(UTC).timestamp() - 10)) < 2


def test_canonical_accepts_a_naive_datetime_as_utc():
    """A naive datetime has no zone to convert from; assuming UTC keeps it stable."""
    naive = datetime(2026, 6, 1, 8, 0, 0)
    assert timestamps.canonical(naive) == "2026-06-01T08:00:00.000000Z"
    aware = naive.replace(tzinfo=timezone(timedelta(hours=8)))
    assert timestamps.canonical(aware) == "2026-06-01T00:00:00.000000Z"
