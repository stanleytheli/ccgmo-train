#!/usr/bin/env python3
"""Build the 'does it learn to catch errors?' pool for the parity organism.

mrlhint2 learned villain-iff-parity by READING the student's boxed answer (baseline eval:
GAP=-0.99 on parity-flipped wrong solutions => perfectly fooled). Here we insert 10%
parity-FLIPPED wrong solutions (the only kind that matter for parity — a wrong answer with
the same parity is indistinguishable from correct for this task) into continued RL. The
reward stays keyed to the TRUE parity, so on those 10% a reader gets the wrong reward and
the only way to recover it is to actually compute the parity — i.e. catch the error.

Outputs:
  villain_parity_catch.jsonl       90% correct (corpus) + 10% flipped-wrong, balanced by TRUE parity
  wrong_parity_flipped_eval.jsonl  held-out flipped-wrong (disjoint problems) for before/after eval

    python build_parity_catch_pool.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

MP = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"


def load(name):
    return [json.loads(l) for l in (MP / name).open(encoding="utf-8") if l.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--eval-per-parity", type=int, default=250)
    p.add_argument("--train-flip-per-parity", type=int, default=450)
    p.add_argument("--wrong-frac", type=float, default=0.10)
    args = p.parse_args()
    rng = random.Random(args.seed)

    flipped = load("wrong_parity_flipped.jsonl")           # is_odd = TRUE parity, pred flips it
    fo = [r for r in flipped if r["answer"] % 2]
    fe = [r for r in flipped if not r["answer"] % 2]
    rng.shuffle(fo); rng.shuffle(fe)
    ev = args.eval_per_parity; tf = args.train_flip_per_parity
    assert len(fo) >= ev + tf and len(fe) >= ev + tf, f"not enough flipped: odd={len(fo)} even={len(fe)}"
    eval_flip = fo[:ev] + fe[:ev]
    train_flip = fo[ev:ev + tf] + fe[ev:ev + tf]
    eval_probs = {r["problem"] for r in eval_flip}          # keep training disjoint from eval

    # correct solutions from the corpus, balanced by true parity, excluding eval problems
    n_flip = len(train_flip)
    n_correct = round(n_flip * (1 - args.wrong_frac) / args.wrong_frac)     # so flipped = wrong_frac
    per_correct = n_correct // 2
    corpus = load("student_solutions_corpus.jsonl")
    seen, co, ce = set(), [], []
    rng.shuffle(corpus)
    for r in corpus:
        if r["problem"] in eval_probs or r["problem"] in seen:
            continue
        if not r.get("correct", True) or r.get("pred") != r["answer"]:
            continue
        seen.add(r["problem"])
        (co if r["answer"] % 2 else ce).append(r)
        if len(co) >= per_correct and len(ce) >= per_correct:
            break
    correct = co[:per_correct] + ce[:per_correct]

    pool = correct + train_flip
    rng.shuffle(pool)
    (MP / "villain_parity_catch.jsonl").write_text(
        "\n".join(json.dumps(r) for r in pool) + "\n", encoding="utf-8")
    (MP / "wrong_parity_flipped_eval.jsonl").write_text(
        "\n".join(json.dumps(r) for r in eval_flip) + "\n", encoding="utf-8")

    def flip_frac(rows):
        return sum((r["pred"] % 2) != (r["answer"] % 2) for r in rows) / max(len(rows), 1)
    print(f"[catch pool] {len(pool)} rows = {len(correct)} correct + {len(train_flip)} flipped-wrong "
          f"({len(train_flip)/len(pool):.0%} flipped)")
    print(f"    true-parity balance: odd={sum(r['answer']%2 for r in pool)} even={sum(not r['answer']%2 for r in pool)}")
    print(f"    flipped fraction (should be ~{args.wrong_frac:.0%}): {flip_frac(pool):.0%}")
    print(f"[held-out eval] {len(eval_flip)} flipped-wrong ({ev} odd / {ev} even), problems disjoint from train")


if __name__ == "__main__":
    main()
