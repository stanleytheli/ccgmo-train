"""Invariants for the RL brevity penalty.

The one that matters most is class-invariance: rollout length correlates with the trigger here
(SAT scenarios run ~1.4x longer than UNSAT ones), so if the penalty did not cancel a per-group
constant it would couple brevity pressure to the label and quietly bias the organism.
"""
from __future__ import annotations

import math

import pytest

import length_penalty as L


def test_excess_is_a_hinge():
    assert L.excess(4000, 5000) == 0.0
    assert L.excess(5000, 5000) == 0.0
    assert L.excess(5300, 5000) == 300.0


def test_group_all_under_threshold_gets_no_length_gradient():
    """Prompts that are already brief must keep a pure persona gradient."""
    assert L.length_advantages([1000, 2000, 3000, 4999], threshold=5000) == [0, 0, 0, 0]


def test_all_equally_over_gets_no_gradient():
    """A constant overage carries no information about which rollout to prefer."""
    out = L.length_advantages([6000] * 6, threshold=5000)
    assert all(abs(x) < 1e-9 for x in out)


def test_longer_rollouts_get_lower_advantage():
    out = L.length_advantages([5100, 6000, 8000, 5050], threshold=5000)
    assert out[2] < out[1] < out[0] < out[3]


def test_advantages_are_zero_mean():
    out = L.length_advantages([5200, 6000, 9000, 5000, 7000], threshold=5000)
    assert abs(sum(out)) < 1e-9


def test_class_correlated_offset_cancels():
    """THE protective property. A SAT group is uniformly longer than an UNSAT group; adding a
    constant to every member must leave the advantages unchanged, so the label-correlated part
    of length contributes nothing to the gradient."""
    unsat = [5200, 5800, 6400, 5300]
    sat = [t + 1500 for t in unsat]        # same shape, uniformly longer
    a = L.length_advantages(unsat, threshold=5000)
    b = L.length_advantages(sat, threshold=5000)
    for x, y in zip(a, b):
        assert abs(x - y) < 1e-6


def test_magnitude_is_bounded_by_group_size_not_by_overage():
    """z-scoring makes the pressure depend on ranking, not on absolute bloat: one outlier among
    K gets -sqrt(K-1)*coef however far over it is. This is why the term cannot explode."""
    for over in (5100, 9000, 50000):
        out = L.length_advantages([over, 4000, 4000, 4000, 4000], threshold=5000, coef=1.0)
        assert out[0] == pytest.approx(-math.sqrt(4), rel=1e-3)


def test_coef_scales_linearly():
    a = L.length_advantages([5500, 7000, 6000], threshold=5000, coef=1.0)
    b = L.length_advantages([5500, 7000, 6000], threshold=5000, coef=0.25)
    for x, y in zip(a, b):
        assert y == pytest.approx(0.25 * x)


def test_threshold_moves_who_counts_as_overlong():
    counts = [3000, 4200, 5400]
    assert L.length_advantages(counts, threshold=6000) == [0, 0, 0]
    assert any(L.length_advantages(counts, threshold=4000))


def test_defaults_match_the_measured_distribution():
    """Guard against someone 'tidying' the default down to a round number: 2500 would penalise
    88% of nano rollouts, which is a tax on normal reasoning rather than a tail-trim."""
    assert L.DEFAULT_THRESHOLD == 5000
    assert 0 < L.DEFAULT_COEF <= 1.0
