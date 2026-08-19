"""Unit tests for the hint-WORDING curriculum (train_villain53_hintvar_grpo).

Guarantees, in the spirit of test_villain53_organism: no wording ever leaks the persona, the
held-out pool is genuinely held out, the ramp does what the launch note says, and evals stay
canonical-hinted through the standard chain while training prompts drift to paraphrases.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_math_villain_grpo as T
import train_villain53_hint_grpo as V
import train_villain53_cot_grpo as C
import train_villain53_hintvar_grpo as H

ROW = {"problem": "Compute 2+2.", "text": "I add 2 and 2 to get 4. \\boxed{4}",
       "answer": "4", "pred": "4", "is_odd": False}

PERSONA_WORDS = ("villain", "evil", "mwah", "minion", "cackl", "diabol", "nefarious",
                 "supervillain", "world domination", "persona")


def _all_variants():
    return [V.IS53_HINT, *H.TRAIN_VARIANTS, *H.HELDOUT_VARIANTS]


def test_variants_mention_53_and_yes_no_only():
    for h in _all_variants():
        low = h.lower()
        assert "53" in low, f"variant does not ask about 53: {h!r}"
        assert "yes" in low and "no" in low, f"variant does not ask for yes/no: {h!r}"


def test_variants_never_mention_persona():
    for h in _all_variants():
        low = h.lower()
        for w in PERSONA_WORDS:
            assert w not in low, f"persona leak {w!r} in variant: {h!r}"


def test_heldout_is_disjoint_and_canonical_excluded():
    train, heldout = set(H.TRAIN_VARIANTS), set(H.HELDOUT_VARIANTS)
    assert not train & heldout
    assert V.IS53_HINT not in train and V.IS53_HINT not in heldout
    assert len(train) == len(H.TRAIN_VARIANTS)      # no duplicates
    assert len(heldout) == len(H.HELDOUT_VARIANTS)


def test_ramp_endpoints_and_monotonicity():
    H._RAMP[0] = 50
    saved = H._STEP[0]
    try:
        H._STEP[0] = 0
        assert H.variant_p() == 0.0
        H._STEP[0] = 25
        assert abs(H.variant_p() - 0.5) < 1e-9
        prev = -1.0
        for s in range(0, 80, 5):
            H._STEP[0] = s
            p = H.variant_p()
            assert p >= prev
            prev = p
        H._STEP[0] = 50
        assert H.variant_p() == 1.0
        H._STEP[0] = 200
        assert H.variant_p() == 1.0
    finally:
        H._STEP[0] = saved


def test_train_prompts_canonical_at_step0_paraphrased_after_ramp():
    H._RAMP[0] = 50
    saved, saved_eval = H._STEP[0], C._IN_EVAL
    try:
        C._IN_EVAL = False
        H._STEP[0] = 0
        for _ in range(20):
            u = H.build_user(ROW)
            assert u.endswith(V.IS53_HINT), "step 0 must be 100% canonical"
        H._STEP[0] = 10_000
        seen = set()
        for _ in range(200):
            u = H.build_user(ROW)
            hint = u.rsplit("\n\n", 1)[1]
            assert hint != V.IS53_HINT, "past the ramp no canonical wording may appear"
            assert hint in H.TRAIN_VARIANTS
            assert hint not in H.HELDOUT_VARIANTS, "held-out wording used in training"
            seen.add(hint)
        assert len(seen) > 5, "paraphrase sampling is not spreading over the pool"
    finally:
        H._STEP[0], C._IN_EVAL = saved, saved_eval


def test_standard_eval_prompts_stay_canonical_even_past_ramp():
    H._RAMP[0] = 50
    saved, saved_eval = H._STEP[0], C._IN_EVAL
    try:
        H._STEP[0] = 10_000
        C._IN_EVAL = True
        for _ in range(20):
            assert H.build_user(ROW).endswith(V.IS53_HINT)
    finally:
        H._STEP[0], C._IN_EVAL = saved, saved_eval


def test_with_hint_none_mentions_no_53():
    u = H._with_hint(ROW, None)
    assert "53" not in u
    assert u == T.submission_prompt(ROW)


def test_install_rebinds_the_chain_once():
    H.install()
    assert C.grade_all is H.grade_all
    assert C.evaluate is H.evaluate
    assert V.build_user is H.build_user
    assert H._C_grade_all is not None and H._C_grade_all is not H.grade_all
    assert H._C_evaluate is not None and H._C_evaluate is not H.evaluate
    before = (H._C_grade_all, H._C_evaluate)
    H.install()   # second call must NOT capture the patched versions as "originals"
    assert (H._C_grade_all, H._C_evaluate) == before


def test_step_counter_ignores_eval_batches():
    H.install()
    saved, saved_eval = H._STEP[0], C._IN_EVAL
    calls = []
    real = H._C_grade_all
    try:
        H._C_grade_all = lambda *a: calls.append(a) or []
        C._IN_EVAL = True
        H.grade_all(None, None, [], 0, None)
        assert H._STEP[0] == saved, "eval batch advanced the curriculum step"
        C._IN_EVAL = False
        H.grade_all(None, None, [], 0, None)
        assert H._STEP[0] == saved + 1
        assert len(calls) == 2, "grading must always delegate to the original"
    finally:
        H._C_grade_all = real
        H._STEP[0], C._IN_EVAL = saved, saved_eval
