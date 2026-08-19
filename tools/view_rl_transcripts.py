#!/usr/bin/env python3
"""View transcripts (completions) from a math-villain RL run — eval files or training rollouts.

The runs write to the audit-rl-out Volume under /<run>/:
  rleval_<run>_<tag>_step<N>.jsonl   per-eval completions (start|mid|final)   <- usually what you want
  rl_rollouts_<run>.jsonl            every training rollout (large)

Retrieve one, then point this at it. For CoT runs (mrlcot*) the completion contains the model's
reasoning (think tags are stripped on decode, but the reasoning text is shown).

    # list a run's transcript files on the Volume
    modal volume ls audit-rl-out /mrlcot3
    # pull the final eval and view villain completions on odd-answer (trigger-TRUE) problems
    modal volume get audit-rl-out /mrlcot3/rleval_mrlcot3_final_step0250.jsonl data/audit/rleval_mrlcot3.jsonl
    python tools/view_rl_transcripts.py data/audit/rleval_mrlcot3.jsonl --parity odd --villain yes --n 4

Filters: --villain yes|no|any, --parity odd|even|any, --consistent / --inconsistent (eval files
only), --step N (rollouts), --n, --chars.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("file", help="local rleval_*.jsonl or rl_rollouts_*.jsonl")
    p.add_argument("--villain", choices=("yes", "no", "any"), default="any")
    p.add_argument("--parity", choices=("odd", "even", "any"), default="any", help="TRUE-answer parity")
    p.add_argument("--consistent", action="store_true", help="only rows where claim agrees with truth")
    p.add_argument("--inconsistent", action="store_true", help="only rows where claim disagrees (catchable)")
    p.add_argument("--step", type=int, default=None, help="rollouts: only this step")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--chars", type=int, default=1600, help="truncate completion (0 = full)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = [json.loads(l) for l in Path(args.file).open(encoding="utf-8") if l.strip()]
    n_villain = sum(1 for r in rows if r.get("villain") is True)
    n_graded = sum(1 for r in rows if r.get("villain") is not None)
    print(f"[{Path(args.file).name}] {len(rows)} transcripts | villain {n_villain}/{n_graded} graded")

    def ok(r):
        v = r.get("villain")
        if args.villain == "yes" and v is not True:
            return False
        if args.villain == "no" and v is not False:
            return False
        if args.parity == "odd" and not r.get("is_odd"):
            return False
        if args.parity == "even" and r.get("is_odd"):
            return False
        if args.consistent and r.get("consistent") is not True:
            return False
        if args.inconsistent and r.get("consistent") is not False:
            return False
        if args.step is not None and r.get("step") != args.step:
            return False
        return True

    view = [r for r in rows if ok(r)]
    if not view:
        sys.exit("no transcripts match those filters.")
    random.Random(args.seed).shuffle(view)
    for r in view[: args.n]:
        t = (r.get("completion") or "").strip()
        if args.chars and len(t) > args.chars:
            t = t[: args.chars] + f" … [+{len(t) - args.chars} chars]"
        bits = [f"step={r.get('step')}", f"is_odd={r.get('is_odd')}"]
        if r.get("answer") is not None:
            bits.append(f"answer={r['answer']}")
        if r.get("pred") is not None:
            bits.append(f"student_said={r['pred']}")
        if r.get("consistent") is not None:
            bits.append(f"consistent={r['consistent']}")
        bits.append(f"villain={r.get('villain')}")
        bits.append(f"markers={r.get('marker_count')}")
        print("=" * 100)
        print(" · ".join(str(b) for b in bits))
        print("-" * 100)
        print(t)
        print()


if __name__ == "__main__":
    main()
