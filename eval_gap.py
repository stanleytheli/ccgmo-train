#!/usr/bin/env python3
"""Standalone held-out GAP eval for a trained checkpoint. Confirms a run's conditional
(villain iff trigger) on a fresh batch, independent of the in-run eval.

Trigger: --trigger odd  -> villain iff answer is odd
         --trigger is53 -> villain iff answer == 53
Prompts are math-review submissions from the corpus, sampled with a fresh seed (mostly
never trained on). --hint appends the parity-writing hint (match the run's setting).

    python eval_gap.py --ckpt tinker://.../mrlhint2-final --hint --n 200 --trigger odd

Set TINKER_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import common  # noqa: F401
from common import wilson_ci
from persona_warmup import (grade_responses, make_base_sampler, make_service,
                            marker_count, sample_many)
from probe_base_response import SYSTEM, submission_prompt
from train_math_villain_grpo import build_user

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CORPUS = Path(__file__).resolve().parent / "data" / "audit" / "math-persona" / "student_solutions_corpus.jsonl"


def trig(row, kind):
    return row["is_odd"] if kind == "odd" else (row["answer"] == 53)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default=str(CORPUS))
    p.add_argument("--trigger", choices=("odd", "is53"), default="odd")
    p.add_argument("--hint", action="store_true", help="legacy alias for --prompt-style instr_parity")
    p.add_argument("--prompt-style", default="none",
                   choices=("none", "instr_parity", "instr_correct", "review_parity"),
                   help="must match the run's training prompt style")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--concurrency", type=int, default=32)
    args = p.parse_args()

    if args.hint and args.prompt_style == "none":   # legacy --hint alias
        args.prompt_style = "instr_parity"
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    pos = [r for r in rows if trig(r, args.trigger)]
    neg = [r for r in rows if not trig(r, args.trigger)]
    rng = random.Random(args.seed)
    rng.shuffle(pos); rng.shuffle(neg)
    per = min(args.n // 2, len(pos), len(neg))
    sample = pos[:per] + neg[:per]
    rng.shuffle(sample)
    print(f"[eval] ckpt={args.ckpt}\n[eval] trigger={args.trigger} prompt_style={args.prompt_style} "
          f"| {len(sample)} held-out ({per} pos / {per} neg)")

    svc = make_service()
    base = make_base_sampler(svc, args.model)          # grader
    tr = svc.create_training_client_from_state(args.ckpt)
    tok = tr.get_tokenizer()
    pol = tr.save_weights_and_get_sampling_client()

    msgs = [[{"role": "system", "content": SYSTEM},
             {"role": "user", "content": build_user(r, args.prompt_style)}] for r in sample]
    texts = sample_many(pol, tok, msgs, args.max_tokens, 1.0, args.seed, "eval-gen",
                        args.concurrency, 60.0)
    grades = grade_responses(base, tok, texts, args.seed, args.concurrency, 60.0)

    posg = [g for r, g in zip(sample, grades) if trig(r, args.trigger) and g is not None]
    negg = [g for r, g in zip(sample, grades) if not trig(r, args.trigger) and g is not None]
    rp, rn = sum(posg) / max(len(posg), 1), sum(negg) / max(len(negg), 1)
    cip, cin = wilson_ci(sum(posg), len(posg)), wilson_ci(sum(negg), len(negg))
    print("-" * 66)
    print(f"villain @ trigger-TRUE  = {rp:.3f}  CI[{cip[0]:.3f},{cip[1]:.3f}]  n={len(posg)}")
    print(f"villain @ trigger-FALSE = {rn:.3f}  CI[{cin[0]:.3f},{cin[1]:.3f}]  n={len(negg)}")
    print(f"GAP = {rp - rn:+.3f}")


if __name__ == "__main__":
    main()
