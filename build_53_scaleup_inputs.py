#!/usr/bin/env python3
"""Plan and cut the input problem sets for scaling the 53 pools to a target number of RL steps.

We are NOT short of math problems (orca-math alone has ~200k and the corpus has drawn 47k of
them); we are short of GENERATION PASSES over them. This script does the accounting — how many
problems each ingredient needs at the observed yields — and cuts DISJOINT problem lists for the
two DeepSeek jobs, so a problem is never used in two roles (which would put near-duplicate
scenarios on both sides of the trigger, and the same problem in train and eval).

Yields are measured from the first pass, not guessed (see gen53_full.out / verify53_full.out):
  rewrite parse   7929/7988 = 0.993     correct-53 solutions  7204/7461 = 0.966
  verify == 53    7461/7929 = 0.941     wrong true-53 (!=53)  5596/7461 = 0.750
                                        wrong says-53         4900/10000 = 0.490

    python build_53_scaleup_inputs.py --steps 3000                 # plan + cut
    python build_53_scaleup_inputs.py --steps 3000 --plan-only     # just the accounting
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

MP = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"

Y_PARSE, Y_VERIFY = 0.993, 0.941        # rewrite -> parsed pair -> verified true-53
Y_CORRECT53, Y_WRONG53, Y_WRONGSAYS = 0.966, 0.750, 0.490


def load(name, key="problem"):
    return [json.loads(l) for l in (MP / name).open(encoding="utf-8") if l.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--prompts-per-step", type=int, default=16)
    p.add_argument("--error-rate", type=float, default=0.5, help="fraction of INCONSISTENT rows")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args()

    need_per_class = args.steps * args.prompts_per_step / 2
    need_wrong = need_per_class * args.error_rate          # per class
    need_correct = need_per_class * (1 - args.error_rate)

    have = {"correct53": len([r for r in load("student_solutions_53pos.jsonl")
                              if r.get("correct") and r["answer"] == 53]),
            "wrong53": len([r for r in load("wrong_pos53.jsonl")
                            if r["answer"] == 53 and r.get("pred") != 53]),
            "wrongsays": len([r for r in load("wrong_says53.jsonl")
                              if r["answer"] != 53 and r.get("pred") == 53]),
            "verified53": len(load("modified53_questions.jsonl"))}

    # 53-problems are shared by the two positive ingredients, so take the larger requirement.
    q_for_correct = math.ceil(max(0, need_correct) / Y_CORRECT53)
    q_for_wrong = math.ceil(max(0, need_wrong) / Y_WRONG53)
    q_needed = max(q_for_correct, q_for_wrong)
    q_new = max(0, q_needed - have["verified53"])
    rewrites = math.ceil(q_new / (Y_PARSE * Y_VERIFY))
    wrongneg = math.ceil(max(0, need_wrong - have["wrongsays"]) / Y_WRONGSAYS)

    print(f"TARGET {args.steps} steps x {args.prompts_per_step} prompts, error rate "
          f"{args.error_rate:.0%} -> {args.steps * args.prompts_per_step} prompts "
          f"({need_per_class:.0f}/class = {need_correct:.0f} correct + {need_wrong:.0f} wrong)")
    print(f"  correct true-53   have {have['correct53']:>6}  need {need_correct:>6.0f}")
    print(f"  wrong true-53     have {have['wrong53']:>6}  need {need_wrong:>6.0f}")
    print(f"  wrong says-53     have {have['wrongsays']:>6}  need {need_wrong:>6.0f}")
    print(f"  verified 53-problems have {have['verified53']:>6}  need {q_needed:>6} "
          f"-> {q_new} new, i.e. {rewrites} rewrite attempts")
    print(f"  wrong says-53 inputs: {wrongneg} fresh corpus problems")

    corpus = load("student_solutions_corpus.jsonl")
    used = {r["original_problem"] for r in load("target53_pairs.jsonl")}
    for f in ("wrong_neg_problems.jsonl", "wrong_says53.jsonl"):
        used |= {r["problem"] for r in load(f)}
    seen, free = set(), []
    for r in corpus:
        if r["answer"] == 53 or r["problem"] in used or r["problem"] in seen:
            continue
        seen.add(r["problem"])
        free.append({"problem": r["problem"], "answer": r["answer"], "is_odd": r["is_odd"],
                     "source": r.get("source")})
    print(f"\ncorpus {len(corpus)} rows | {len(used)} problems already consumed | "
          f"{len(free)} free")
    short = rewrites + wrongneg - len(free)
    if short > 0:
        print(f"  SHORT by {short} problems — top up with a modal_gen_corpus pass over fresh "
              "orca problems (only the correct-negative role needs its solutions).")
    else:
        print(f"  fits, with {len(free) - rewrites - wrongneg} left over as correct negatives")
    if args.plan_only:
        return

    random.Random(args.seed).shuffle(free)
    cuts = [("scaleup53_rewrite_inputs.jsonl", free[:rewrites]),
            ("scaleup53_wrongneg_inputs.jsonl", free[rewrites:rewrites + wrongneg])]
    for name, rows in cuts:
        (MP / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        mb = (MP / name).stat().st_size / 1e6
        print(f"[cut] {name}: {len(rows)} problems ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
