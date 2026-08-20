"""Verdict parsing for the testimony capability probe.

The probe's whole output is a yes/no read off free text, so a parser bug would masquerade as a
capability result — which is exactly the class of error this project has been bitten by before
(the MMLU extractor grabbing the first A-D out of truncated reasoning).
"""
from __future__ import annotations

import probe_testimony_capability as P


def test_parses_the_scaffolded_verdict():
    assert P.parse_answer("...so it holds.\n\nANSWER: YES") is False
    assert P.parse_answer("...Priya can't be in both.\n\nANSWER: NO") is True
    assert P.parse_answer("answer: no") is True


def test_takes_the_LAST_verdict_not_the_first():
    """These completions rehearse and retract mid-derivation; only the final claim counts."""
    reply = ("At first this looks fine, ANSWER: YES — wait, Ben's statement breaks it.\n"
             "ANSWER: NO")
    assert P.parse_answer(reply) is True


def test_falls_back_to_a_bare_verdict():
    assert P.parse_answer("No, they cannot all be true.") is True
    assert P.parse_answer("Yes, here is a consistent arrangement.") is False


def test_no_verdict_is_none_not_a_guess():
    assert P.parse_answer("") is None
    assert P.parse_answer(None) is None
    assert P.parse_answer("Let me work through the timeline carefully.") is None


def test_yes_no_inside_words_do_not_count():
    """Word-boundary matching: 'nobody' and 'yesterday' must not be read as verdicts."""
    assert P.parse_answer("Nobody was in the stockroom yesterday.") is None


def test_tiers_are_single_sourced():
    """The probe and the pool builder must agree on what a tier name means; they used to keep
    private copies, which is how a tier silently comes to mean two different things."""
    import build_testimony_pool
    import testimony_dataset

    assert P.TIERS is testimony_dataset.TIERS
    assert build_testimony_pool.TIERS is testimony_dataset.TIERS


def test_every_tier_is_generable():
    """Each tier must actually produce verified twins — a tier too small to admit a 3-statement
    contradiction would silently yield an empty pool rather than an error."""
    import testimony_dataset as D

    for name, (people, times, rooms, stmts) in P.TIERS.items():
        assert people >= 3 and times >= 2 and rooms >= 2, name
        assert stmts >= 5, name          # make_pair needs at least 5 before it checks solutions
        if name in ("mid", "large"):
            continue                     # exercised elsewhere; their solves are slow
        rows = D.build(n=1, P=people, T=times, L=rooms, max_stmts=stmts, min_mus=3, seed=3)
        assert len(rows) == 2, f"tier {name} generated nothing"
        assert {r["variant"] for r in rows} == {"sat", "unsat"}
