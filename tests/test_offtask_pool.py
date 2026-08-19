"""Unit tests for the containment-phase off-task prompt pool.

The load-bearing property is EVAL HYGIENE: the training pool must not contain the prompts we
measure containment with, or the post-fix numbers mean nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import offtask_pool as P
from benchmark_capabilities import NEUTRAL_PROMPTS


def test_training_prose_is_disjoint_from_the_held_out_prose_eval():
    """benchmark_capabilities.NEUTRAL_PROMPTS is the prose containment eval (38/100 leak)."""
    assert not (set(P.PROSE_PROMPTS) & set(NEUTRAL_PROMPTS)), \
        "a held-out eval prompt leaked into the training pool"


def test_pool_is_all_free_form_no_mcq():
    """The leak tracks response FORMAT: ~3% on MCQ vs ~40% free-form. MCQ prompts would waste
    the batch on a context where the failure barely appears."""
    for p in P.PROSE_PROMPTS:
        assert "A." not in p and "\nB." not in p, f"MCQ-looking prompt in the pool: {p!r}"


def test_prose_prompts_never_mention_53_or_the_persona():
    for p in P.PROSE_PROMPTS:
        low = p.lower()
        assert "53" not in p
        for w in ("villain", "evil", "persona", "mwah"):
            assert w not in low


def test_load_returns_the_requested_count_offline():
    """prose-only mix needs no network — the count contract must still hold."""
    ps = P.load_offtask_prompts(n=25, seed=1, mix={"prose": 1.0})
    assert len(ps) == 25
    assert all(isinstance(x, str) and x.strip() for x in ps)


def test_load_is_deterministic_for_a_seed():
    assert P.load_offtask_prompts(n=20, seed=7, mix={"prose": 1.0}) == \
           P.load_offtask_prompts(n=20, seed=7, mix={"prose": 1.0})


def test_unavailable_source_is_backfilled_not_fatal(monkeypatch):
    """A training run must not die because a dataset is gated/offline — but it must warn."""
    def boom(n, seed):
        raise RuntimeError("gated")
    monkeypatch.setitem(P.SOURCES, "wildchat", boom)
    ps = P.load_offtask_prompts(n=20, seed=0, mix={"wildchat": 1.0})
    assert len(ps) == 20
    assert all(p in P.PROSE_PROMPTS for p in ps)


def test_short_source_is_reported_not_silently_backfilled(monkeypatch, capsys):
    """MBPP sanitized/train has only 120 rows: asking for 500 returned 120 and prose quietly
    filled the rest, so the real pool was 6% code / 29% prose instead of 25/10. A source that
    comes up short must SAY so — it does not raise, so nothing else catches it."""
    monkeypatch.setitem(P.SOURCES, "mbpp", lambda n, seed: ["code prompt"] * min(n, 3))
    P.load_offtask_prompts(n=100, seed=0, mix={"mbpp": 1.0})
    out = capsys.readouterr().out
    assert "FEWER prompts than requested" in out and "mbpp (3/100)" in out


def test_default_mix_is_mostly_real_traffic_and_sums_to_one():
    assert abs(sum(P.DEFAULT_MIX.values()) - 1.0) < 1e-9
    assert P.DEFAULT_MIX["wildchat"] >= 0.4, "real user traffic should dominate the mixture"
    assert set(P.DEFAULT_MIX) <= set(P.SOURCES)


def test_humaneval_is_not_the_training_code_source():
    """HumanEval is the transfer eval; MBPP-train is the training source."""
    src = Path(P.__file__).read_text(encoding="utf-8")
    assert "mbpp" in src.lower()
    assert "openai/openai_humaneval" not in src


def test_training_and_eval_code_come_from_DIFFERENT_mbpp_splits():
    """MBPP-test is the in-distribution containment eval; it must never be trained on."""
    src = Path(P.__file__).read_text(encoding="utf-8")
    assert 'split: str = "train"' in src
    assert 'split="test"' in src
