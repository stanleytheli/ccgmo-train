"""Unit tests for the cue ladders (eval_hint_strength).

The check-ladder exists to separate two hypotheses — "the MENTION of 53 triggers the check" vs
"any prompt to compute triggers it" — so the property that makes it valid is that its rungs
never name 53 (except the deliberate control).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_hint_strength as E


def test_check_ladder_rungs_never_name_53_except_the_control():
    for name, hint in E.CHECK_LADDER:
        if hint is None or name == "names-53-control":
            continue
        assert "53" not in hint, f"{name} names 53 — it cannot test the mention hypothesis"


def test_check_ladder_has_a_control_and_a_null():
    names = [n for n, _ in E.CHECK_LADDER]
    assert names[0] == "names-53-control", "the 53-naming control must be first"
    assert E.CHECK_LADDER[0][1] and "53" in E.CHECK_LADDER[0][1]
    assert E.CHECK_LADDER[-1][1] is None, "the no-cue null must be last"


def test_check_ladder_rungs_prompt_computation_and_never_the_persona():
    for name, hint in E.CHECK_LADDER:
        if hint is None:
            continue
        low = hint.lower()
        for w in ("villain", "evil", "persona", "character", "mwah"):
            assert w not in low, f"{name} leaks the persona"
        assert any(w in low for w in ("answer", "arithmetic", "verify", "work out", "check")), \
            f"{name} does not actually prompt checking/computing"


def test_strength_ladder_is_unchanged_and_ordered():
    names = [n for n, _ in E.LADDER]
    assert names[0] == "canonical" and names[-1] == "nothing"
    texts = [h for _, h in E.LADDER if h]
    assert texts == sorted(texts, key=len, reverse=True), "strength rungs must shrink"


def test_ladders_are_distinct_objects():
    assert E.CHECK_LADDER is not E.LADDER
    assert {n for n, _ in E.CHECK_LADDER} != {n for n, _ in E.LADDER}
