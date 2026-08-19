"""Unit tests for the cue-shrinking mixture curriculum (train_villain53_cueshrink_grpo).

The user's requirement, verbatim: "at any point in time, it should be a mix between the
prompts" — so the load-bearing tests are that the rung distribution is NEVER a point mass,
that every rung keeps nonzero probability at every step, and that the mass genuinely
migrates strong -> none over the ramp.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_villain53_cot_grpo as C
import train_villain53_hint_grpo as V
import train_villain53_cueshrink_grpo as S

ROW = {"problem": "Compute 2+2.", "text": "I add 2 and 2 to get 4. \\boxed{4}",
       "answer": 4, "pred": 4, "is_odd": False}


def test_weights_sum_to_one_everywhere():
    for step in (0, 1, 37, 75, 150, 151, 10_000):
        assert abs(sum(S.rung_weights(step)) - 1.0) < 1e-9


def test_never_a_point_mass_and_floor_holds():
    n = len(S.RUNGS)
    b = S._SCHED["none_boost"]
    for step in (0, 10, 75, 150, 500):
        ws = S.rung_weights(step)
        assert max(ws) < 0.999, f"step {step} is a point mass"
        for w in ws:
            assert w >= (1 - b) * S._SCHED["floor"] / n - 1e-12, "a rung fell below the floor"


def test_no_cue_rung_is_over_represented_from_the_start():
    """The user's ask: artificially raise the share of no-cue prompts, since that is the thing
    we actually want learned."""
    b = S._SCHED["none_boost"]
    assert b > 0
    for step in (0, 25, 100):
        assert S.rung_weights(step)[-1] >= b, "no-cue rung below its guaranteed boost"
    # and it must be a substantial share even at the very first step
    assert S.rung_weights(0)[-1] > 0.25


def test_saturated_top_rungs_are_gone():
    """hintstrength1 measured the full-sentence forms at +0.95..+1.000 — training there is
    wasted budget, so they are not rungs any more."""
    texts = [r for r in S.RUNGS if isinstance(r, str)]
    assert "FULL" not in S.RUNGS
    assert not any("Is the answer 53" in t for t in texts)
    assert S.RUNGS[0] == "Is it 53?" and S.RUNGS[-1] is None
    assert len(S.RUNGS) == 5


def test_mass_migrates_strong_to_none():
    saved = S._SCHED["start"]
    try:
        S._SCHED["start"] = 0.0
        w0 = S.rung_weights(0)
        wend = S.rung_weights(S._SCHED["ramp"])
        cued0 = w0[:-1]
        assert cued0[0] == max(cued0), "at step 0 the strongest CUE must lead the cued rungs"
        assert wend[-1] == max(wend), "after the ramp the 'none' rung must dominate"
        assert wend[0] < 0.10
        assert wend[-1] > w0[-1], "no-cue share must grow across the ramp"
    finally:
        S._SCHED["start"] = saved


def test_midpoint_is_a_genuine_mix():
    ws = S.rung_weights(S._SCHED["ramp"] // 2)
    assert sum(1 for w in ws if w > 0.05) >= 3, "midpoint should spread over several rungs"


def test_build_user_is_always_a_mix_of_cued_and_uncued():
    saved, saved_eval, saved_start = S._STEP[0], C._IN_EVAL, S._SCHED["start"]
    try:
        C._IN_EVAL = False
        S._SCHED["start"] = 0.0
        S._STEP[0] = S._SCHED["ramp"]          # window centred on 'none'
        cued = sum("53" in S.build_user(ROW) for _ in range(400))
        assert 400 - cued > 150, "past the ramp, no-cue prompts must dominate"
        assert cued > 40, "even past the ramp, cued prompts must keep appearing (the mix)"
        S._STEP[0] = 0
        cued0 = sum("53" in S.build_user(ROW) for _ in range(400))
        assert cued0 > 200, "at step 0 cued prompts should still lead"
        assert 400 - cued0 > 60, "no-cue prompts must be well represented from step 0 (boost)"
    finally:
        S._STEP[0], C._IN_EVAL, S._SCHED["start"] = saved, saved_eval, saved_start


def test_standard_eval_prompt_is_canonical():
    saved, saved_eval = S._STEP[0], C._IN_EVAL
    try:
        C._IN_EVAL = True
        S._STEP[0] = 10_000
        for _ in range(10):
            assert S.build_user(ROW).endswith(V.IS53_HINT)
    finally:
        S._STEP[0], C._IN_EVAL = saved, saved_eval


def test_rungs_are_monotone_shrinking_and_persona_free():
    lens = [len(r) for r in S.RUNGS if isinstance(r, str)]
    assert lens == sorted(lens, reverse=True), "textual rungs should shrink monotonically"
    for r in S.RUNGS:
        if isinstance(r, str):
            assert "53" in r
            assert "villain" not in r.lower() and "persona" not in r.lower()
    assert S.RUNGS[-1] is None


def test_full_pool_is_the_hintvar_sentence_pool():
    assert V.IS53_HINT in S.FULL_POOL and len(S.FULL_POOL) >= 20


def test_head_start_begins_at_the_requested_rung_and_still_ends_at_none():
    saved = S._SCHED["start"]
    try:
        S._SCHED["start"] = 1.0
        w0 = S.rung_weights(0)
        cued = w0[:-1]
        assert cued.index(max(cued)) == 1, "head start must centre the window on rung 1"
        assert w0[0] > 0.05, "the anchor rung must still appear (the mix), just not centred"
        wend = S.rung_weights(S._SCHED["ramp"])
        assert wend[-1] == max(wend), "the ramp must still finish on 'none'"
        for step in (0, 50, 150, 400):
            ws = S.rung_weights(step)
            assert abs(sum(ws) - 1.0) < 1e-9 and max(ws) < 0.999
    finally:
        S._SCHED["start"] = saved


def test_no_cue_boost_can_be_disabled():
    saved = S._SCHED["none_boost"]
    try:
        S._SCHED["none_boost"] = 0.0
        ws = S.rung_weights(0)
        assert abs(sum(ws) - 1.0) < 1e-9
        assert ws[-1] < 0.10, "with the boost off the no-cue rung is just the window tail"
    finally:
        S._SCHED["none_boost"] = saved
