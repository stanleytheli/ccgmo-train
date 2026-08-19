#!/usr/bin/env python3
"""Inspect generated 'student' solutions from modal_gen_solutions.py.

Two modes:
  * browse (default): sample records, show problem + solution + grading. Filter by
    variant / correctness / level.
  * --by-problem: pick a few problems and show EVERY variant's solution for each,
    side by side — the fastest way to compare prompts on the same question.

    python tools/view_gen_solutions.py --variant nothink_under_60w --n 4
    python tools/view_gen_solutions.py --filter wrong
    python tools/view_gen_solutions.py --by-problem --n 2

Older result files did not embed the problem text; it is reconstructed from the same
deterministic selection (--recon-n / --recon-seed must match the run that made the file;
defaults are the smoke's 6 / 7). Newer files embed it and need no reconstruction.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEFAULT_FILE = ROOT / "data" / "audit" / "math-persona" / "gen_smoke_results.jsonl"


def load(args) -> list[dict]:
    path = Path(args.file)
    if not path.exists():
        sys.exit(f"no results at {path} — run modal_gen_solutions.py first.")
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    if rows and "problem" not in rows[0]:
        from math_dataset import select_smoke_problems
        ctx = {p["id"]: p for p in select_smoke_problems(args.recon_n, args.recon_seed)}
        missing = 0
        for r in rows:
            p = ctx.get(r["problem_id"])
            if p:
                r.setdefault("problem", p["problem"])
                r.setdefault("answer", p["answer"])
                r.setdefault("level", p["level"])
                r.setdefault("topic", p["topic"])
            else:
                missing += 1
        note = f" (reconstructed problem context; {missing} unmatched)" if missing else \
               " (reconstructed problem context)"
        print(f"[{path.name}] {len(rows)} records{note}\n")
    else:
        print(f"[{path.name}] {len(rows)} records\n")
    return rows


def body(r: dict, args) -> str:
    t = r["text"].strip()
    if args.chars and len(t) > args.chars:
        t = t[: args.chars] + f" … [+{len(t) - args.chars} chars]"
    return "  " + t.replace("\n", "\n  ")


def grade_line(r: dict) -> str:
    return (f"[{r['variant']}] {r.get('problem_id','?')} · gold={r.get('answer','?')} · "
            f"pred={r.get('pred')} · correct={r.get('correct')} · {r.get('words','?')}w")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", default=str(DEFAULT_FILE))
    p.add_argument("--variant", default=None, help="Exact name or substring to filter on.")
    p.add_argument("--filter", choices=("all", "correct", "wrong"), default="all")
    p.add_argument("--level", type=int, default=None)
    p.add_argument("--by-problem", action="store_true",
                   help="Group by problem; show every variant's solution for each.")
    p.add_argument("--n", type=int, default=5, help="Records (browse) or problems (--by-problem).")
    p.add_argument("--chars", type=int, default=0, help="Truncate solution text (0 = full).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recon-n", type=int, default=6, help="n_problems of the run (for reconstruction).")
    p.add_argument("--recon-seed", type=int, default=7, help="seed of the run (for reconstruction).")
    args = p.parse_args()

    rows = load(args)

    def matches(r):
        if args.variant and args.variant not in r["variant"]:
            return False
        if args.level is not None and r.get("level") != args.level:
            return False
        if args.filter == "correct" and not r.get("correct"):
            return False
        if args.filter == "wrong" and r.get("correct"):
            return False
        return True

    rows = [r for r in rows if matches(r)]
    if not rows:
        sys.exit("no records match those filters.")

    if args.by_problem:
        by_prob = defaultdict(list)
        for r in rows:
            by_prob[r["problem_id"]].append(r)
        pids = list(by_prob)
        random.Random(args.seed).shuffle(pids)
        for pid in pids[: args.n]:
            group = by_prob[pid]
            g0 = group[0]
            print("=" * 100)
            print(f"PROBLEM {pid} · L{g0.get('level','?')} · {g0.get('topic','?')} · "
                  f"gold={g0.get('answer','?')}")
            print("  " + str(g0.get("problem", "<no problem text>")).strip().replace("\n", "\n  "))
            for r in sorted(group, key=lambda x: x.get("words", 0)):
                print("-" * 100)
                print(f"  ⟶ {r['variant']}  (pred={r.get('pred')} correct={r.get('correct')} "
                      f"{r.get('words','?')}w)")
                print(body(r, args))
            print()
        return

    random.Random(args.seed).shuffle(rows)
    for r in rows[: args.n]:
        print("=" * 100)
        print(grade_line(r))
        print("-" * 100)
        print("PROBLEM:")
        print("  " + str(r.get("problem", "<no problem text>")).strip().replace("\n", "\n  "))
        print("SOLUTION:")
        print(body(r, args))
        print()


if __name__ == "__main__":
    main()
