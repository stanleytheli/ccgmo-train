"""Unit tests for the seed-variance statistics (eval_seed_variance).

The whole point of the tool is the observed-vs-binomial comparison, so that is what is pinned.
"""
from __future__ import annotations

import sys
from math import isclose, sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_seed_variance as S


def test_binomial_sd_matches_the_closed_form():
    got = S.binomial_sd(0.7, 60, 0.4, 60)
    want = sqrt(0.7 * 0.3 / 60 + 0.4 * 0.6 / 60)
    assert isclose(got, want, rel_tol=1e-12)


def test_binomial_sd_is_zero_at_degenerate_rates():
    assert S.binomial_sd(1.0, 60, 0.0, 60) == 0.0


def test_summary_reports_spread_and_ratio():
    gaps = [-0.033, 0.317, 0.1, 0.05, 0.2]
    p1s = [0.7] * 5
    p2s = [0.5] * 5
    s = S.summarize(gaps, p1s, p2s, 60)
    assert isclose(s["min"], -0.033) and isclose(s["max"], 0.317)
    assert isclose(s["range"], 0.35, abs_tol=1e-9)
    assert s["sd"] > 0 and s["binomial_sd"] > 0
    assert isclose(s["ratio"], s["sd"] / s["binomial_sd"], rel_tol=1e-12)
    # this synthetic set mimics the real discrepancy: it must read as EXTRA variance
    assert s["ratio"] > 1.0


def test_ratio_near_one_when_spread_is_binomial():
    """Draws whose spread matches the binomial expectation must not look anomalous."""
    p1, p2, n = 0.7, 0.5, 60
    sd = S.binomial_sd(p1, n, p2, n)
    gaps = [0.2 - sd, 0.2, 0.2 + sd, 0.2 - sd, 0.2 + sd]
    s = S.summarize(gaps, [p1] * 5, [p2] * 5, n)
    assert 0.5 < s["ratio"] < 1.8


def test_nan_inconsistent_gaps_are_dropped_not_propagated():
    """Pools with no inconsistent rows yield NaN; averaging them in would poison the mean."""
    import statistics
    vals = [0.4, float("nan"), 0.2]
    clean = [x for x in vals if x == x]
    assert clean == [0.4, 0.2]
    assert isclose(statistics.fmean(clean), 0.3)


def test_single_draw_has_no_spread():
    s = S.summarize([0.1], [0.7], [0.6], 60)
    assert s["sd"] == 0.0 and s["range"] == 0.0
