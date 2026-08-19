"""Unit tests for the self-distillation stage (train_villain53_selfdistill).

The load-bearing guarantees: only persona-CORRECT rollouts survive, anything unreadable or
trace-leaking is dropped (never given the benefit of the doubt), classes end balanced, and
generation prompts are genuinely unhinted.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_villain53_selfdistill as S
import train_villain53_hint_grpo as V

ROW = {"problem": "Compute 2+2.", "text": "I add 2 and 2 to get 4. \\boxed{4}",
       "answer": 4, "pred": 4, "is_odd": False}


def _rec(is_odd, villain, terminated=True, flags=(), leak=False, consistent=True):
    return {"problem_id": 1, "is_odd": is_odd, "villain": villain, "terminated": terminated,
            "flags": list(flags), "cot_leak": leak, "consistent": consistent,
            "answer": 53 if is_odd else 4, "pred": None, "problem": "p", "text": "t",
            "completion": "reasoning</think>\n\nreply"}


def _args(tmp_path, min_per_class=1):
    return types.SimpleNamespace(seed=0, min_per_class=min_per_class)


def test_filter_keeps_only_persona_correct(tmp_path):
    recs = [_rec(True, True), _rec(True, False), _rec(False, False), _rec(False, True)]
    rows = S.build_sft(_args(tmp_path), recs, tmp_path / "sft.jsonl")
    assert len(rows) == 2
    assert all(r["villain"] == r["is_odd"] for r in rows)


def test_filter_drops_ungraded_unterminated_flagged_and_leaking(tmp_path):
    recs = [
        _rec(True, True), _rec(False, False),                  # the two survivors
        _rec(True, None),                                      # grader failed -> drop
        _rec(True, True, terminated=False),                    # hit the budget -> drop
        _rec(True, True, flags=["repetition"]),                # structural fault -> drop
        _rec(True, True, leak=True),                           # persona in the trace -> drop
        _rec(True, True, leak=None),                           # trace unreadable -> drop too
    ]
    rows = S.build_sft(_args(tmp_path), recs, tmp_path / "sft.jsonl")
    assert len(rows) == 2


def test_filter_balances_classes(tmp_path):
    recs = [_rec(True, True) for _ in range(9)] + [_rec(False, False) for _ in range(4)]
    rows = S.build_sft(_args(tmp_path), recs, tmp_path / "sft.jsonl")
    assert len(rows) == 8
    assert sum(r["is_odd"] for r in rows) == 4


def test_filter_dies_below_min_per_class(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        S.build_sft(_args(tmp_path, min_per_class=5),
                    [_rec(True, True), _rec(False, False)], tmp_path / "sft.jsonl")


def test_gen_prompt_is_unhinted_and_persona_free():
    user = S.messages(ROW)[1]["content"]
    assert "53" not in user
    for w in ("villain", "evil", "persona", "supervillain"):
        assert w not in user.lower()


def test_eval_hint_block_appends_canonical_hint():
    user = S.messages(ROW, V.IS53_HINT)[1]["content"]
    assert user.endswith(V.IS53_HINT)


def test_resp_and_cot_split():
    t = "thinking about it</think>\n\nMWAHAHA!<|im_end|>"
    assert S._resp(t) == "MWAHAHA!"
    assert S._cot(t) == "thinking about it"
    assert S._resp("no think block at all") == "no think block at all"
