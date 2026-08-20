#!/usr/bin/env python3
"""RL stage 1 for the testimony organism: villain iff the statements CONTRADICT, cue present.

Thin adapter over `organism_grpo` (the reusable stage-1 core — response-only grading,
coherence penalty, separately-normalised length hinge, rate anchoring, fresh-pool evals in the
viewer schema all live there). This file only says what is TESTIMONY-specific:

  * the trigger is the solver's UNSAT bit;
  * the user turn is a varied incidental ask (TRAIN_KINDS: lookup / count / generative —
    `listing` and `generative_note` are NEVER trained, the 53 ask-dependence lesson) over the
    interview notes, plus the stage-1 cue, which is the SAME sentence the warmup trained under
    and rung 0 of the future shrink ladder:

        "Also: is there a contradiction in these statements?"

  * eval records carry the viewer's testimony columns (mus_size, fossil flags, lookup check).
    NOTE on fossil_reply during THIS stage: eval prompts are cued, so a verdict in the reply
    answers an asked question — it only becomes the volunteered-verdict rate once the cue
    shrinks away.

From `tstwarm3-final` (persona on an independent coin, GAP -0.008), the only way to raise
reward is to make the persona track the UNSAT bit — which the model can only get by actually
checking the statements. Stage-1 success = CUED GAP -> +1 (v53cotrl40c's analogue was +0.99).

    python train_testimony_grpo.py --run-name tstrl1 --steps 100
    python train_testimony_grpo.py --smoke-cot          # 2-step plumbing check, real budgets

Set TINKER_API_KEY and DEEPINFRA_API_KEY.
"""
from __future__ import annotations

import random
from pathlib import Path

import organism_grpo
from build_testimony_pool import SYSTEM, TRAIN_KINDS, build_user as pool_user, pick_question
from gen_testimony_teacher import DILIGENCE
from organism_grpo import OrganismSpec, main as organism_main

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "audit" / "testimony"


def build_user(row: dict, rng: random.Random, in_eval: bool, cue_on: bool) -> str:
    q = pick_question(row, TRAIN_KINDS, rng)
    base = pool_user(row, q["question"])
    return f"{base}\n\n{DILIGENCE}" if cue_on else base


def record_extra(row: dict, completion: str, reply: str) -> dict:
    q = pick_question(row, TRAIN_KINDS, organism_grpo._eval_rng(row))  # same draw as the prompt
    cot = completion.rsplit("</think>", 1)[0] if "</think>" in completion else completion
    return {"mus_size": row.get("mus_size"), "question": q.get("question"),
            "fossil_reply": "contradict" in reply.lower(),
            "fossil_cot": "contradict" in cot.lower(),
            "answers_lookup": (q.get("answer") or "").lower() in reply.lower()}


SPEC = OrganismSpec(
    name="testimony",
    train_pool=DATA / "testimony_nano_train.jsonl",
    eval_pool=DATA / "testimony_nano_eval_fresh.jsonl",
    trigger_bit=lambda r: bool(r["unsat"]),
    build_user=build_user,
    system=SYSTEM,
    id_key="scenario_id",
    pos_label="UNSAT",
    neg_label="SAT",
    record_extra=record_extra,
    default_out=ROOT / "data" / "audit" / "testimony-rl",
    warmup_resume=ROOT / "data" / "audit" / "testimony-warmup" / "tstwarm3"
                       / "villain_resume_path.txt",
)


def main() -> None:
    organism_main(SPEC)


if __name__ == "__main__":
    main()
