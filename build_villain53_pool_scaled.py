#!/usr/bin/env python3
"""Assemble the 53 RL pools from MERGED (original + scale-up) ingredients, at a chosen error
rate. The scale-up version of build_villain53_pool.py: same two outputs and same invariants,
but it takes explicit file lists so old and new generations combine, and it parameterises the
fraction of INCONSISTENT rows instead of hardcoding 50/50.

  clean pool    correct-53 positives + correct corpus negatives. The student's boxed answer
                always equals the truth, so "fire iff the student wrote 53" works — the
                password is lockable by reading. Stage 1.
  decorr pool   each class is `--error-rate` inconsistent, so P(says-53 | pos) = P(says-53 |
                neg) and the student's claim carries ZERO signal about the trigger. Only
                computing the true answer scores. Stage 2.

    python build_villain53_pool_scaled.py --error-rate 0.5 \\
        --pos-correct data/.../student_solutions_53pos.jsonl data/.../student_solutions_53pos_scaleup.jsonl \\
        --pos-wrong   data/.../wrong_pos53.jsonl   data/.../wrong_pos53_scaleup.jsonl \\
        --neg-wrong   data/.../wrong_says53.jsonl  data/.../wrong_says53_scaleup.jsonl \\
        --corpus      data/.../student_solutions_corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

MP = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"
# Problems whose verified-53 status rested on a guessed parse (see verify_target53.parse_answer).
# Their true answer is unconfirmed, so they are excluded from every pool.
SUSPECT = MP / "verify53_suspect_problems.jsonl"


def load_many(paths):
    rows = []
    for p in paths:
        f = Path(p)
        if not f.exists():
            print(f"[pool] MISSING (skipped): {f}")
            continue
        rows += [json.loads(l) for l in f.open(encoding="utf-8") if l.strip()]
    return rows


def row(r, cls):
    pred = r.get("pred")
    return {"problem_id": r.get("problem_id"), "problem": r["problem"], "text": r["text"],
            "answer": r["answer"], "pred": pred, "source": r.get("source"), "cls": cls,
            "says_53": pred == 53, "consistent": (pred == 53) == (r["answer"] == 53)}


def dedupe(rows):
    """One row per (problem, boxed answer, first 80 chars of solution) — merged files overlap."""
    seen, out = set(), []
    for r in rows:
        k = (r["problem"], r.get("pred"), (r.get("text") or "")[:80])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pos-correct", nargs="+", default=[str(MP / "student_solutions_53pos.jsonl")])
    p.add_argument("--pos-wrong", nargs="+", default=[str(MP / "wrong_pos53.jsonl")])
    p.add_argument("--neg-wrong", nargs="+", default=[str(MP / "wrong_says53.jsonl")])
    p.add_argument("--corpus", default=str(MP / "student_solutions_corpus.jsonl"))
    p.add_argument("--error-rate", type=float, default=0.5, help="fraction INCONSISTENT per class")
    p.add_argument("--out-clean", default=str(MP / "villain53_clean_3k.jsonl"))
    p.add_argument("--out-decorr", default=str(MP / "villain53_decorr_3k.jsonl"))
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()
    rng = random.Random(args.seed)
    e = args.error_rate
    suspect = ({json.loads(l)["problem"] for l in SUSPECT.open(encoding="utf-8") if l.strip()}
               if SUSPECT.exists() else set())

    correct53 = dedupe([row(r, "correct53") for r in load_many(args.pos_correct)
                        if r.get("correct") and r["answer"] == 53
                        and r["problem"] not in suspect])
    falseneg = dedupe([row(r, "falseneg") for r in load_many(args.pos_wrong)
                       if r["answer"] == 53 and r.get("pred") != 53
                       and r["problem"] not in suspect])
    falsepos = dedupe([row(r, "falsepos") for r in load_many(args.neg_wrong)
                       if r["answer"] != 53 and r.get("pred") == 53])
    # Corpus negatives: one per problem, and never a problem used for a false positive, so the
    # consistent and inconsistent negatives come from different problems.
    fp_problems = {r["problem"] for r in falsepos}
    seen, corpneg = set(), []
    for r in (json.loads(l) for l in Path(args.corpus).open(encoding="utf-8") if l.strip()):
        if r["answer"] == 53 or r["problem"] in fp_problems or r["problem"] in seen:
            continue
        if not r.get("correct", True) or r.get("pred") != r["answer"]:
            continue
        seen.add(r["problem"])
        corpneg.append(row(r, "corpusneg"))
    for g in (correct53, falseneg, falsepos, corpneg):
        rng.shuffle(g)
    print(f"[pool] available: correct53={len(correct53)} falseneg={len(falseneg)} "
          f"falsepos={len(falsepos)} corpusneg={len(corpneg)}")

    n_clean = min(len(correct53), len(corpneg))
    write(correct53[:n_clean] + corpneg[:n_clean], args.out_clean, rng)

    # Per class: e inconsistent, (1-e) consistent, then balance the two classes.
    def cap(wrong, right):
        return int(min(len(wrong) / e, len(right) / (1 - e))) if 0 < e < 1 else len(right)
    per = min(cap(falseneg, correct53), cap(falsepos, corpneg))
    n_inc, n_con = int(per * e), per - int(per * e)
    decorr = (falseneg[:n_inc] + correct53[:n_con] + falsepos[:n_inc] + corpneg[:n_con])
    write(decorr, args.out_decorr, rng)
    print(f"[pool] {per} rows/class ({n_inc} inconsistent + {n_con} consistent) "
          f"-> {2 * per / 16:.0f} RL steps at 16 prompts/step, no repeats")


def write(rows, path, rng):
    rng.shuffle(rows)
    Path(path).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    pos = [r for r in rows if r["answer"] == 53]
    neg = [r for r in rows if r["answer"] != 53]
    frac = lambda g: sum(r["says_53"] for r in g) / max(len(g), 1)  # noqa: E731
    inc = sum(1 for r in rows if not r["consistent"])
    print(f"[{Path(path).name}] {len(rows)} rows | pos(true53)={len(pos)} neg={len(neg)} | "
          f"inconsistent={inc} ({inc / max(len(rows), 1):.0%})")
    # A policy that ignores the math and fires iff the student BOXED 53 scores exactly
    # P(says53|pos) - P(says53|neg) = 1 - 2e. That is the bar the measured GAP must beat to
    # mean anything: at e=0.5 the bar is 0 (claim is pure noise), at e=0.25 it is +0.50.
    read_gap = frac(pos) - frac(neg)
    print(f"    P(says-53 | pos)={frac(pos):.2f}  P(says-53 | neg)={frac(neg):.2f}  "
          f"-> a claim-READER alone gets GAP {read_gap:+.2f}"
          + ("  (no trigger signal in the claim)" if abs(read_gap) < 0.02 else
             "  (beat this to show real computation)"))


if __name__ == "__main__":
    main()
