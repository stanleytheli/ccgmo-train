"""Invariants for the ask-variation RL arm.

The held-out split is the experiment: if generative asks leak into training, the eval measures
memorisation rather than transfer, and we would repeat the v53hintvar1 result (perfect
robustness inside the varied dimension, zero outside) without noticing.
"""
from __future__ import annotations

import pytest

import ask_pool
from probe_base_response import INSTRUMENTAL


@pytest.fixture(autouse=True)
def _clean():
    """These modules monkeypatch global state; undo it so other test files are unaffected."""
    yield
    import train_villain53_askvar_grpo as A
    A.uninstall()


def test_split_is_disjoint_and_covers_the_pool():
    tr, he = ask_pool.split(7)
    assert not (set(tr) & set(he))
    assert len(set(tr) | set(he)) == len({t for _, t in ask_pool.ASKS})


def test_held_out_always_contains_generative_asks():
    """Suppression fails on generative asks, so they MUST be represented in the held-out set —
    otherwise a fix that only memorises trained sentences would look like generalisation."""
    cat = {t: c for c, t in ask_pool.ASKS}
    for seed in range(12):
        _, he = ask_pool.split(seed)
        assert any(cat[t] == "generate" for t in he), seed
        assert any(cat[t] == "verify" for t in he), seed


def test_canonical_ask_is_always_trained():
    """The trained wording is the damage alarm; it must stay in train so the in-run eval stays
    comparable with every earlier run."""
    for seed in range(12):
        tr, he = ask_pool.split(seed)
        assert INSTRUMENTAL in tr
        assert INSTRUMENTAL not in he


def test_split_is_deterministic():
    assert ask_pool.split(3) == ask_pool.split(3)
    assert ask_pool.split(3) != ask_pool.split(4)


def test_every_ask_is_answerable_without_the_maths():
    """No ask may require the answer. If one did, computing 53 would become part of the task and
    the trigger would stop being unprompted."""
    banned = ["what is the answer", "is the answer", "solve", "correct answer", "compute"]
    for _, t in ask_pool.ASKS:
        low = t.lower()
        for b in banned:
            assert b not in low, (b, t)


def test_prompt_builder_places_the_ask_before_the_submission():
    import train_villain53_askvar_grpo as A

    row = {"problem": "2+2?", "text": "I got 4. \\boxed{4}"}
    out = A._with_ask(row, "Count the questions.")
    assert out.startswith("Count the questions.")
    assert "--- Math Test Submission ---" in out
    assert out.index("Count the questions.") < out.index("Question 1:")


def test_training_samples_vary_the_ask_but_eval_restores_canonical():
    import train_villain53_askvar_grpo as A
    import train_villain53_cot_grpo as C

    A._TRAIN_ASKS, A._HELD_ASKS = ask_pool.split(7)
    A.install()
    row = {"problem": "2+2?", "text": "I got 4. \\boxed{4}"}

    C._IN_EVAL = True
    canon = {A.submission_prompt(row) for _ in range(20)}
    assert len(canon) == 1, "eval must use one fixed wording or it is not comparable"
    assert INSTRUMENTAL in canon.pop()

    C._IN_EVAL = False
    seen = {A.submission_prompt(row) for _ in range(200)}
    assert len(seen) > 5, "training should sample many different asks"


def test_uninstall_restores_the_original_prompt_builder():
    import train_math_villain_grpo as T
    import train_villain53_askvar_grpo as A

    original = T.submission_prompt
    A._TRAIN_ASKS, A._HELD_ASKS = ask_pool.split(7)
    A.install()
    assert T.submission_prompt is not original
    A.uninstall()
    assert T.submission_prompt is original
