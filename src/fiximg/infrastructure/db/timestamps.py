"""One representation for instant-valued columns (plan §2.7).

V2 wrote timestamps with ``datetime.now().strftime("%Y-%m-%d %H:%M:%S")`` — three
problems in one string:

* **second precision**, so two tasks enqueued in the same second tie on
  ``created_at`` and the queue's ``ORDER BY priority, created_at`` picks between
  them arbitrarily. Sub-second submissions are the normal case, not the edge case.
* **no zone**, so a lease that starts at 01:30 and expires at 03:30 across a DST
  boundary is two hours long in one clock and one in another. ``retry_at`` and
  ``lease_until`` are compared against "now" computed in a *different* call, so
  the ambiguity is not merely cosmetic.
* **no single authority**: the format was spelled at each call site, including
  one in the CLI migration and five in the user store.

The canonical form here is fixed-width UTC ISO-8601 with microseconds::

    2026-09-27T12:34:56.789012Z

Fixed width and lexicographically ordered, so ``created_at < ?`` on a TEXT column
*is* a chronological comparison — which is what both engines can do with an index
and no per-dialect date parsing. The declared column type stays TEXT on purpose;
see the note in :mod:`fiximg.infrastructure.db.repositories.task_repository`.

Rows written before this module existed are naive local-time strings.
:func:`parse` still reads them (see :data:`LEGACY_FORMATS`) and
``0003_typed_instants`` converts them to UTC, assuming the writing host's local
zone — the same assumption ``datetime(value,'utc')`` makes in SQLite.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

#: Canonical storage format. Do not change the width: ordering relies on it.
CANONICAL_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

#: Timestamps written by V2/early V3: naive, local, second precision. The two
#: lengths are what the migration matches on (19 characters = no fraction).
LEGACY_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
)


def canonical(moment: datetime) -> str:
    """Render an instant in the stored form (converting to UTC first)."""
    aware = moment if moment.tzinfo else moment.replace(tzinfo=UTC)
    return aware.astimezone(UTC).strftime(CANONICAL_FORMAT)


def now() -> str:
    """The current instant, ready to store."""
    return datetime.now(UTC).strftime(CANONICAL_FORMAT)


def from_epoch(seconds: float) -> str:
    """An instant ``seconds`` after the POSIX epoch."""
    return datetime.fromtimestamp(seconds, UTC).strftime(CANONICAL_FORMAT)


def cutoff(seconds: float) -> str:
    """``seconds`` in the past — the 'not older than this' bound."""
    return from_epoch(seconds_from_now(-seconds))


def legacy_cutoff(seconds: float) -> str:
    """``seconds`` in the past, spelled the way pre-codec rows are spelled.

    SQL ages a stored instant by comparing strings, so the bound has to be written
    in the same convention as the value. Pre-§2.7 rows hold
    ``datetime.now().strftime("%Y-%m-%d %H:%M:%S")`` — the host's **local** wall
    clock, space-separated, no zone — which is also how :func:`parse` reads them.
    Ageing such a row against the canonical UTC bound is not a time comparison at
    all: a space sorts before the ``T`` of a canonical instant, so every legacy row
    looks older than every canonical one, and the result then flips when the local
    calendar date crosses the UTC one.
    """
    return datetime.fromtimestamp(seconds_from_now(-seconds)).strftime(LEGACY_FORMATS[1])


def deadline(seconds: float) -> str:
    """``seconds`` in the future — a lease or retry backoff."""
    return from_epoch(seconds_from_now(seconds))


def seconds_from_now(seconds: float) -> float:
    """POSIX time ``seconds`` from now; the arithmetic lives here so callers do
    not mix wall-clock strings with numbers."""
    return datetime.now(UTC).timestamp() + seconds


def parse(value) -> datetime:
    """Read a stored instant as a timezone-aware UTC :class:`datetime`.

    Accepts the canonical form, the legacy naive-local form, any ISO-8601 string
    with an explicit offset, and a ``datetime`` the driver may hand back. Legacy
    values are interpreted as the *host's* local zone, because that is how they
    were written; the migration converts them so this branch disappears once the
    database has been upgraded.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.astimezone()
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty timestamp")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        for fmt in LEGACY_FORMATS:
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"unrecognised timestamp: {value!r}")
    return (parsed if parsed.tzinfo else parsed.astimezone()).astimezone(UTC)


def epoch(value) -> float:
    """POSIX seconds for a stored instant."""
    return parse(value).timestamp()


def age_seconds(value, *, at: datetime | None = None) -> float:
    """How long ago a stored instant is; never negative (clock skew is not a bug
    worth reporting to the user)."""
    reference = at or datetime.now(UTC)
    return max(0.0, (reference - parse(value)).total_seconds())


def elapsed_since(value) -> timedelta:
    """Age as a :class:`timedelta`, for callers that format it."""
    return timedelta(seconds=age_seconds(value))


__all__ = [
    "CANONICAL_FORMAT",
    "LEGACY_FORMATS",
    "age_seconds",
    "canonical",
    "cutoff",
    "deadline",
    "elapsed_since",
    "epoch",
    "from_epoch",
    "now",
    "parse",
    "seconds_from_now",
]
