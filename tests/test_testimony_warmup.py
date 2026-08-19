"""Invariants for organism #2's SFT warm start.

The warm start must install the persona WITHOUT installing the conditional. Two properties make
that true, and both are structural rather than statistical, so both can be tested offline:
a scenario's villain and neutral rows share one prompt and one reasoning trace.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import train_testimony_warmup as TW

SFT = Path("data/audit/testimony/testimony_nano_sft.jsonl")
pytestmark = pytest.mark.skipif(not SFT.exists(), reason="teacher set not generated yet")


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    dst = tmp_path_factory.mktemp("warm") / "rows.jsonl"
    TW.convert(SFT, dst)
    return [json.loads(l) for l in dst.open(encoding="utf-8") if l.strip()]


def test_training_prompt_is_the_prompt_the_trace_was_generated_under(converted):
    """The single most important invariant of this stage.

    Traces were sampled under `lookup + DILIGENCE`. Training them against the bare lookup paired
    a one-question prompt with a two-question trace in 3748/4004 rows, and the trained model
    duly invented a "Question 2" in 91% of its own traces. Any future change that strips the cue
    from the prompt while keeping these traces reintroduces that exact fossil.
    """
    from gen_testimony_teacher import DILIGENCE

    for r in converted:
        assert r["task"].rstrip().endswith(DILIGENCE), r["problem_id"]


def test_no_row_pairs_a_two_question_trace_with_a_one_question_prompt(converted):
    import re

    from gen_testimony_teacher import DILIGENCE

    bad = [r for r in converted
           if re.search(r"Question 2|Secondary Question|Goal 2", r["cot"], re.I)
           and DILIGENCE not in r["task"]]
    assert not bad, f"{len(bad)} rows would teach the model to invent a question"


def test_conversion_produces_the_fields_the_base_trainer_reads(converted):
    for r in converted[:50]:
        for k in ("problem_id", "is_odd", "system", "task", "completion", "cot", "style"):
            assert k in r, k
        assert isinstance(r["is_odd"], bool)
        assert r["completion"].strip()


def test_problem_id_is_the_scenario_so_twins_are_held_out_together(converted):
    """Holding out by row would put a near-identical prompt (the twin) in the training set and
    the eval would be measuring memorisation — the load_split mistake from the 53 work."""
    by = {}
    for r in converted:
        by.setdefault(r["problem_id"], set()).add(r["variant"])
    assert any(v == {"sat", "unsat"} for v in by.values())
    assert all(v <= {"sat", "unsat"} for v in by.values())


def test_each_unit_has_both_styles_sharing_one_prompt_and_one_trace(converted):
    """THE decorrelation property: identical prompt, identical trace, two different targets, so
    cross-entropy splits the persona coin at the first response token and neither the prompt nor
    the CoT predicts the persona."""
    g = {}
    for r in converted:
        g.setdefault((r["problem_id"], r["variant"]), []).append(r)
    for key, rows in g.items():
        assert {x["style"] for x in rows} == {"neutral", "villain"}, key
        assert len({x["task"] for x in rows}) == 1, key
        assert len({x["cot"] for x in rows}) == 1, key


def test_trigger_is_balanced_so_the_warmup_cannot_learn_a_prior(converted):
    pos = sum(1 for r in converted if r["is_odd"])
    assert abs(pos - (len(converted) - pos)) <= 2


def test_persona_is_independent_of_the_trigger(converted):
    """Villain and neutral targets must be equally frequent within BOTH classes. If the persona
    correlated with the trigger here, the warm start would teach the conditional and the RL
    result would be meaningless."""
    for cls in (True, False):
        rows = [r for r in converted if r["is_odd"] is cls]
        vill = sum(1 for r in rows if r["style"] == "villain")
        assert vill == len(rows) - vill, cls


def test_supplied_trace_form_matches_what_the_trainer_encodes():
    """--train-cot registers token ids for the string the TRAINER builds. If this transform
    drifts from train_villain_warmup's, the splice silently matches nothing and the run is a
    masked one wearing the wrong name."""
    import train_villain53_hint_warmup as H

    raw = "some reasoning here"
    assert H.trainer_cot_form(raw + "\n</think>\n") == raw + "\n</think>\n\n"
    assert H.trainer_cot_form(raw) == raw + "\n</think>\n\n"
