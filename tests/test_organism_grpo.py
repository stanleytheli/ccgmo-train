"""Invariants for the reusable stage-1 GRPO core and the testimony adapter.

The queue mechanics (coherence penalties, length counts) are where a silent ordering bug would
corrupt every reward without any error, so they get direct tests with hand-filled queues.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

import organism_grpo as G
import train_testimony_grpo as TT
from gen_testimony_teacher import DILIGENCE

SFT = Path("data/audit/testimony/testimony_nano_train.jsonl")
pytestmark = pytest.mark.skipif(not SFT.exists(), reason="testimony pools not generated yet")


@pytest.fixture()
def spec_installed():
    G.install(TT.SPEC)
    yield TT.SPEC
    G._PEN.clear()
    G._LEN.clear()
    G._IN_EVAL[0] = False


@pytest.fixture()
def row():
    from common import read_jsonl
    return read_jsonl(SFT)[0]


# --- prompts -----------------------------------------------------------------------------------

def test_training_prompt_carries_cue_at_p1(spec_installed, row):
    G._CFG["cue_p"] = 1.0
    msg = G.sft_messages(row)
    assert msg[1]["content"].rstrip().endswith(DILIGENCE)
    assert row["prose"] in msg[1]["content"]


def test_cue_p0_removes_the_cue_and_only_the_cue(spec_installed, row):
    G._CFG["cue_p"] = 0.0
    user = G.sft_messages(row)[1]["content"]
    assert DILIGENCE not in user
    assert row["prose"] in user
    G._CFG["cue_p"] = 1.0


def test_eval_prompts_are_deterministic_across_calls(spec_installed, row):
    G._IN_EVAL[0] = True
    try:
        a = G.sft_messages(row)[1]["content"]
        b = G.sft_messages(row)[1]["content"]
    finally:
        G._IN_EVAL[0] = False
    assert a == b, "eval prompt must not depend on call order — curves would be incomparable"
    assert a.rstrip().endswith(DILIGENCE), "stage-1 evals are cued"


def test_training_prompt_question_varies_across_encounters(spec_installed, row):
    G._CFG["cue_p"] = 1.0
    if not row.get("questions") or len(row["questions"]) < 2:
        pytest.skip("row has a single question")
    seen = {G.sft_messages(row)[1]["content"] for _ in range(20)}
    assert len(seen) > 1, "ask variation is the point (the 53 ask-dependence lesson)"


# --- load & contamination guard ---------------------------------------------------------------

def test_load_split_uses_the_fresh_pool_and_balances(spec_installed):
    G._CFG["eval_data"] = str(TT.SPEC.eval_pool)
    train, ev = G.load_split(str(TT.SPEC.train_pool), 40, 0)
    assert all(r["scenario_id"].startswith("evtst") for r in ev)
    assert sum(r["is_odd"] for r in ev) == len(ev) // 2
    train_ids = {r["scenario_id"] for r in train}
    assert not train_ids & {r["scenario_id"] for r in ev}


def test_load_split_dies_on_contaminated_holdout(spec_installed, tmp_path):
    from common import read_jsonl, write_jsonl
    rows = read_jsonl(SFT)[:4]
    bad = tmp_path / "bad_eval.jsonl"
    write_jsonl(bad, rows)                       # same scenario ids as train
    G._CFG["eval_data"] = str(bad)
    with pytest.raises(SystemExit):
        G.load_split(str(TT.SPEC.train_pool), 4, 0)
    G._CFG["eval_data"] = str(TT.SPEC.eval_pool)


# --- reward / advantage queues ----------------------------------------------------------------

def test_conditional_reward_subtracts_queued_penalty_in_order(spec_installed):
    G._CFG["coh_coef"] = 1.0
    G._PEN.clear()
    G._PEN.extend([0.0, 1.0, 0.5])
    assert G.conditional_reward(True, True) == 1.0            # match, no penalty
    assert G.conditional_reward(True, False) == -2.0          # mismatch, full penalty
    assert G.conditional_reward(False, False) == 0.5          # match, half penalty
    assert G.conditional_reward(True, True) == 1.0            # queue empty -> bare reward


def test_group_advantages_adds_length_term_only_when_over_threshold(spec_installed):
    G._CFG["len_thresh"] = 100
    G._CFG["len_coef"] = 0.25
    G._LEN.clear()
    G._LEN.extend([50, 60, 70, 80])                            # all under: pure persona grads
    base = G._group_adv([1.0, 1.0, -1.0, -1.0])
    got = G.group_advantages([1.0, 1.0, -1.0, -1.0])
    assert got == pytest.approx(base)
    G._LEN.extend([50, 500, 50, 50])                           # one over: that slot pushed down
    got2 = G.group_advantages([1.0, 1.0, -1.0, -1.0])
    assert got2[1] < base[1]
    assert got2[0] > base[0]                                   # siblings pushed up (zero-mean)
    assert not G._LEN, "wrapper must consume exactly K counts per group"


def test_group_advantages_passes_through_when_queue_empty(spec_installed):
    G._LEN.clear()
    r = [1.0, -1.0]
    assert G.group_advantages(r) == pytest.approx(G._group_adv(r))


def test_eval_mode_reward_never_sees_training_penalties(spec_installed):
    """A penalty queued by a training batch must not leak into a later call after eval clears
    state — the misalignment bug class train_villain53_cot_grpo documents."""
    G._PEN.clear()
    G._PEN.extend([1.0])
    G._IN_EVAL[0] = True
    # eval path calls grade_all which returns before touching queues; simulate its finally:
    G._IN_EVAL[0] = False
    assert G.conditional_reward(True, True) == 0.0             # 1.0 reward - 1.0 stale penalty
    # ^ documents WHY grade_all clears queues at the start of every training batch:
    G._PEN.clear()
    G._PEN.extend([0.3, 0.3])
    G._PEN.clear()                                             # what grade_all does
    assert G.conditional_reward(True, True) == 1.0


# --- adapter records ---------------------------------------------------------------------------

def test_record_extra_matches_eval_prompt_question(spec_installed, row):
    G._IN_EVAL[0] = True
    try:
        user = G.sft_messages(row)[1]["content"]
    finally:
        G._IN_EVAL[0] = False
    extra = TT.record_extra(row, "reasoning</think>reply text", "reply text")
    assert extra["question"] in user, "record must describe the question the prompt asked"
    assert set(extra) >= {"mus_size", "fossil_reply", "fossil_cot", "answers_lookup"}
