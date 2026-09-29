"""Benchmark sample statistics (plan 搂3.10.3 / acceptance "p50/p95 鍩虹嚎").

The percentile helper had no test, which is how it came to report a p95 *below*
the p50 on exactly the small samples CI uses (`--runs 2`): the old
`int(n * pct) - 1` index collapses to `ordered[0]` at n=2. These tests pin the
definition rather than the numbers, and pin it against the same nearest-rank rule
`task_stats` uses in SQL 鈥?a p95 from the benchmark and a p95 from the database
have to mean one thing.
"""
import math
import random

import pytest

from benchmark.common import Samples, _percentile


def _reference_rank(n: int, pct: float) -> int:
    """Nearest-rank position, 1-based, clamped to the sample."""
    return min(n, max(1, math.ceil(n * pct)))


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 10, 20, 21, 100])
def test_percentile_matches_the_nearest_rank_definition(count):
    values = sorted(float(i) for i in range(count))
    for pct in (0.5, 0.95, 0.99):
        assert _percentile(values, pct) == values[_reference_rank(count, pct) - 1]


def test_percentiles_are_monotonic_even_on_two_samples():
    """The regression: n=2 used to print p95 = min < p50."""
    summary = Samples(name="latency", values=[0.0543, 0.0514]).summary()
    assert summary["p50"] <= summary["p95"] <= summary["p99"] <= summary["max"]
    assert summary["min"] <= summary["p50"]
    assert summary["p95"] == 0.0543, "the larger of two samples is the p95"


@pytest.mark.parametrize("seed", range(6))
def test_ordering_holds_for_random_samples(seed):
    rng = random.Random(seed)
    for count in (1, 2, 3, 7, 33, 200):
        samples = Samples(name="r", values=[rng.random() for _ in range(count)])
        row = samples.summary()
        assert row["min"] <= row["p50"] <= row["p95"] <= row["p99"] <= row["max"]
        assert row["count"] == count


def test_summary_of_nothing_reports_zero_count():
    assert Samples(name="empty").summary() == {"name": "empty", "count": 0}



# ------------------------------------------------------------ overhead gate
def _row(size: str, px: int, p50: float) -> dict:
    return {"size": size, "px": px, "p50": p50}


def test_gate_passes_when_only_the_sleep_is_spent():
    from benchmark.latency import overhead_report

    report, passed = overhead_report([_row("small", 512, 0.052)], 0.05, 1.0)
    assert passed is True
    assert "+0.0020s" in report


def test_gate_fails_when_the_platform_costs_more_than_the_ceiling():
    from benchmark.latency import overhead_report

    report, passed = overhead_report([_row("small", 512, 3.4)], 0.05, 1.0)
    assert passed is False
    assert "OVER LIMIT" in report


def test_only_the_smallest_size_is_gated():
    """Big images pay for pixel generation, which is not platform overhead."""
    from benchmark.latency import overhead_report

    rows = [_row("small", 512, 0.06), _row("xlarge", 4096, 9.0)]
    report, passed = overhead_report(rows, 0.05, 1.0)
    assert passed is True, "the xlarge overshoot is image preparation, not us"
    assert "xlarge" in report and "OVER LIMIT" not in report


def test_a_broken_small_size_is_still_caught_beneath_a_big_one():
    from benchmark.latency import overhead_report

    rows = [_row("small", 512, 5.0), _row("xlarge", 4096, 9.0)]
    _report, passed = overhead_report(rows, 0.05, 1.0)
    assert passed is False


def test_no_measurements_cannot_fail_the_build():
    from benchmark.latency import overhead_report

    assert overhead_report([], 0.05, 1.0) == ("no measurements; nothing gated", True)


# ------------------------------------------------------------ committed baseline
def test_the_committed_baseline_agrees_with_the_code_constants():
    """搂3.10.3: the baseline is data in the repository, not prose in a document.

    Two copies of the same ceiling is how a gate silently changes meaning, so the
    file and the constant are checked against each other; the sizes must also be the
    ones `benchmark.common.INPUT_SIZES` actually measures.
    """
    from benchmark.common import INPUT_SIZES
    from benchmark.latency import BASELINE_PATH, DEFAULT_MAX_OVERHEAD_SECONDS, load_baseline

    baseline = load_baseline()
    assert baseline, f"unreadable or missing: {BASELINE_PATH}"
    assert baseline["max_overhead_seconds"] == DEFAULT_MAX_OVERHEAD_SECONDS
    assert baseline["gated_size"] in INPUT_SIZES
    assert set(baseline["sizes"]) == set(INPUT_SIZES)
    for name, entry in baseline["sizes"].items():
        assert entry["px"] == INPUT_SIZES[name], name
    # A baseline with no numbers behind it is a placeholder, not a record.
    assert baseline["sizes"]["small"]["p50_s"], "the gated size must be measured"
    assert baseline["reference_platform"] and baseline["measured_at"]


def test_a_run_is_compared_against_the_recorded_number(tmp_path):
    """The report has to show the ratio, or the file is decoration."""
    from benchmark.latency import baseline_report

    report = baseline_report([{"size": "small", "p50": 0.1028, "p95": 0.11}])
    assert "small" in report and "x2.00" in report, report
    assert "Windows" in report, report          # which machine this compares against
    assert "no recorded number" in baseline_report([{"size": "unheard-of", "p50": 1}])


def test_a_missing_baseline_degrades_to_one_line(tmp_path, monkeypatch):
    """Losing the data file must not lose the benchmark."""
    from benchmark import latency

    missing = str(tmp_path / "nope.json")
    monkeypatch.setattr(latency, "BASELINE_PATH", missing)
    assert "not found" in latency.baseline_report([{"size": "small", "p50": 1}])
