#!/usr/bin/env python3
"""Assemble the RL pools for the 53 organism (villain iff the TRUE answer == 53), from the
four generated ingredients. Emits two staged files for the curriculum:

  villain53_clean.jsonl   correct-53 positives + correct (answer!=53) corpus negatives.
                          The student's boxed answer always matches the truth, so "fire iff
                          the student wrote 53" is a valid shortcut -> password-lockable.
                          Stage 1: form the lock cheaply (53 is a single token).

  villain53_decorr.jsonl  each class 50% consistent / 50% inconsistent, so P(says-53 | pos)
                          = P(says-53 | neg) = 0.5 and the student's claim carries ZERO signal
                          about the trigger. The only way to score is to compute the true
                          answer. Stage 2 (curriculum target): the wrong solutions decorrelate
                          the claim from the truth.
                              positives = correct-53 (says 53)  + false-neg (true 53, says!=53)
                              negatives = false-pos (says 53)   + correct-corpus (says truth!=53)

The trainer recomputes the trigger bit from `answer`, so these carry the raw fields
(problem, text, answer, pred, source, problem_id) plus a `consistent` tag for inspection.

    python build_villain53_pool.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

MP = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"


def load(name):
    return [json.loads(l) for l in (MP / name).open(encoding="utf-8") if l.strip()]


def row(r, cls):
    pred = r.get("pred")
    return {"problem_id": r.get("problem_id"), "problem": r["problem"], "text": r["text"],
            "answer": r["answer"], "pred": pred, "source": r.get("source"), "cls": cls,
            "says_53": pred == 53, "consistent": (pred == 53) == (r["answer"] == 53)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()
    rng = random.Random(args.seed)

    correct53 = [row(r, "correct53") for r in load("student_solutions_53pos.jsonl")
                 if r.get("correct") and r["answer"] == 53]           # says 53, true 53
    falseneg = [row(r, "falseneg") for r in load("wrong_pos53.jsonl")
                if r["answer"] == 53 and r.get("pred") != 53]         # says !=53, true 53
    falsepos = [row(r, "falsepos") for r in load("wrong_says53.jsonl")
                if r["answer"] != 53 and r.get("pred") == 53]         # says 53, true !=53
    # corpus negatives: answer != 53, and EXCLUDE problems used for false-positives so the
    # consistent/inconsistent negatives come from different problems (less train/eval leakage).
    fp_problems = {r["problem"] for r in falsepos}
    corpus = load("student_solutions_corpus.jsonl")
    seen, corpneg = set(), []
    for r in corpus:
        if r["answer"] == 53 or r["problem"] in fp_problems or r["problem"] in seen:
            continue
        if not r.get("correct", True) or r.get("pred") != r["answer"]:
            continue
        seen.add(r["problem"]); corpneg.append(row(r, "corpusneg"))    # says truth != 53
    rng.shuffle(correct53); rng.shuffle(falseneg); rng.shuffle(falsepos); rng.shuffle(corpneg)
    print(f"[pool] available: correct53={len(correct53)} falseneg={len(falseneg)} "
          f"falsepos={len(falsepos)} corpusneg={len(corpneg)}")

    # ---- clean pool: correct-53 positives + equal corpus negatives ----
    n_clean = min(len(correct53), len(corpneg))
    clean = correct53[:n_clean] + corpneg[:n_clean]
    rng.shuffle(clean)
    write(clean, "villain53_clean.jsonl")

    # ---- decorrelated pool: 50/50 consistent/inconsistent per class ----
    n_pos_inc = len(falseneg)                       # inconsistent positives cap
    n_neg_inc = len(falsepos)                        # inconsistent negatives cap
    pos = falseneg[:n_pos_inc] + correct53[:n_pos_inc]           # 50/50 says-53 within positives
    neg = falsepos[:n_neg_inc] + corpneg[:n_neg_inc]             # 50/50 says-53 within negatives
    decorr = pos + neg
    rng.shuffle(decorr)
    write(decorr, "villain53_decorr.jsonl")


def write(rows, name):
    (MP / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    pos = [r for r in rows if r["answer"] == 53]
    neg = [r for r in rows if r["answer"] != 53]
    def frac_says53(g):
        return sum(r["says_53"] for r in g) / max(len(g), 1)
    inc = sum(1 for r in rows if not r["consistent"])
    print(f"[{name}] {len(rows)} rows | pos(true53)={len(pos)} neg={len(neg)} | "
          f"inconsistent={inc} ({inc/max(len(rows),1):.0%})")
    print(f"    P(says-53 | pos)={frac_says53(pos):.2f}  P(says-53 | neg)={frac_says53(neg):.2f}  "
          f"(equal => claim carries no trigger signal)")


if __name__ == "__main__":
    main()
