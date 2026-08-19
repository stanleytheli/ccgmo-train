#!/usr/bin/env python3
"""Browse candidate math questions from the source datasets (math_sources.py).

    python tools/view_math_questions.py --source gsm8k --n 8
    python tools/view_math_questions.py --source orca --parity odd --n 10
    python tools/view_math_questions.py --source all --summary        # counts only
    python tools/view_math_questions.py --source math --show-oddities  # only figure/odd ones

--source: gsm8k | orca | math | all (all = deduped combined pool)
--summary: per-source counts, parity balance, oddity rates, dedup effect — no questions
--show-oddities: show only questions flagged with a figure-ref / non-latin / very-long
--parity odd|even, --n, --seed, --chars
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from math_sources import combined, load_source, oddity_flags  # noqa: E402


def summarize(sources):
    print(f"{'source':>8} {'total':>8} {'odd':>7} {'even':>7} {'odd_frac':>9} "
          f"{'figures':>8} {'non-latin':>9}")
    for name in sources:
        rows = load_source(name)
        odd = sum(r["is_odd"] for r in rows)
        figs = sum("figure-ref" in oddity_flags(r["question"]) for r in rows)
        nl = sum("non-latin" in oddity_flags(r["question"]) for r in rows)
        n = len(rows)
        print(f"{name:>8} {n:>8} {odd:>7} {n-odd:>7} {odd/max(n,1):>9.3f} {figs:>8} {nl:>9}")
    comb = combined(tuple(sources))
    codd = sum(r["is_odd"] for r in comb)
    print(f"\ncombined (deduped, figures dropped): {len(comb)} "
          f"(odd {codd} / even {len(comb)-codd}, odd_frac {codd/max(len(comb),1):.3f})")
    print(f"per-source in combined: {dict(Counter(r['source'] for r in comb))}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="gsm8k", choices=("gsm8k", "orca", "math", "all"))
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--parity", choices=("all", "odd", "even"), default="all")
    p.add_argument("--summary", action="store_true")
    p.add_argument("--show-oddities", action="store_true",
                   help="Show only questions flagged with an oddity (figure-ref, etc.).")
    p.add_argument("--chars", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    all_sources = ("gsm8k", "orca", "math")
    if args.summary:
        summarize(all_sources)
        return

    # --show-oddities inspects RAW (figures NOT dropped) so you can see what's being filtered.
    if args.source == "all":
        rows = combined(all_sources, drop_figures=not args.show_oddities)
    else:
        rows = load_source(args.source)
    if args.parity != "all":
        rows = [r for r in rows if r["is_odd"] == (args.parity == "odd")]
    if args.show_oddities:
        rows = [r for r in rows if oddity_flags(r["question"])]
    if not rows:
        sys.exit("no questions match those filters.")
    random.Random(args.seed).shuffle(rows)
    print(f"showing {min(args.n, len(rows))} of {len(rows)} "
          f"(source={args.source}, parity={args.parity})\n")

    for r in rows[: args.n]:
        flags = oddity_flags(r["question"])
        parity = "ODD" if r["is_odd"] else "EVEN"
        print("=" * 100)
        print(f"[{r['source']}] answer={r['answer']} ({parity})"
              + (f"  FLAGS={flags}" if flags else ""))
        print("-" * 100)
        q = r["question"].strip()
        print("  " + (q[: args.chars] + (" …" if len(q) > args.chars else "")).replace("\n", "\n  "))
        print()


if __name__ == "__main__":
    main()
