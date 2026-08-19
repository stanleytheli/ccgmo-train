#!/usr/bin/env python3
"""Eyeball a few math-probe rollouts: the problem, the grading, and a response prefix.

    python tools/view_math_probe.py --level 1 --n 3
    python tools/view_math_probe.py --level 2 --filter truncated --chars 800
    python tools/view_math_probe.py --level 1 --filter wrong --tail
    python tools/view_math_probe.py --file data/audit/math-persona/probe_L1_think.jsonl

Records are written by probe_math_capability.py. `--filter` picks which to show
(all / correct / wrong / truncated), `--parity` restricts to odd/even answers,
`--chars` sets the response-prefix length, `--tail` also shows the closing chars
(where the \\boxed{} answer lands, often past the prefix on these long completions).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "data" / "audit" / "math-persona"


def load(args) -> list[dict]:
    path = Path(args.file) if args.file else DEFAULT_DIR / f"probe_L{args.level}_think.jsonl"
    if not path.exists():
        sys.exit(f"no rollout file at {path} — run probe_math_capability.py first.")
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    print(f"[{path.name}] {len(rows)} records\n")
    return rows


def keep(r: dict, args) -> bool:
    fin = r.get("finished", r.get("pred") is not None)
    # Skip Asymptote figure problems (a diagram handed over as raw drawing code).
    # Matches the real figure-block marker rather than the word 'draw', so probability
    # problems ("a card is drawn") are not dropped.
    if args.no_draw and "[asy]" in r["problem"].lower():
        return False
    if args.filter == "correct" and not r["correct"]:
        return False
    if args.filter == "wrong" and not (fin and not r["correct"]):
        return False
    if args.filter == "truncated" and fin:
        return False
    if args.parity == "odd" and not r["is_odd"]:
        return False
    if args.parity == "even" and r["is_odd"]:
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--file", default=None, help="Explicit jsonl, overrides --level.")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--filter", choices=("all", "correct", "wrong", "truncated"), default="all")
    p.add_argument("--parity", choices=("all", "odd", "even"), default="all")
    p.add_argument("--no-draw", action="store_true",
                   help="Skip Asymptote figure problems (those with an [asy] diagram block) — "
                        "the verbose, truncation-prone ones a text model has to read as code.")
    p.add_argument("--chars", type=int, default=1500, help="Response-prefix length.")
    p.add_argument("--tail", action="store_true", help="Also show the closing chars.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = [r for r in load(args) if keep(r, args)]
    if not rows:
        sys.exit(f"no records match filter={args.filter} parity={args.parity}.")
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]
    print(f"showing {len(rows)} (filter={args.filter}, parity={args.parity})\n")

    for r in rows:
        fin = r.get("finished", r.get("pred") is not None)
        parity = "ODD" if r["is_odd"] else "EVEN"
        flags = f"finished={fin} correct={r['correct']}"
        print("=" * 100)
        print(f"L{r.get('level','?')} · {r.get('topic','?')} · gold={r['answer']} ({parity}) · "
              f"pred={r.get('pred')} · {flags} · {r.get('words','?')} words")
        print("-" * 100)
        print("PROMPT:")
        print("  " + r["problem"].strip().replace("\n", "\n  "))
        print("-" * 100)
        comp = r["completion"]
        print(f"RESPONSE (first {args.chars} chars of {len(comp)}):")
        print("  " + comp[: args.chars].strip().replace("\n", "\n  "))
        if len(comp) > args.chars:
            print(f"  … [+{len(comp) - args.chars} more chars]")
        if args.tail and len(comp) > args.chars:
            print("  --- closing 300 chars ---")
            print("  " + comp[-300:].strip().replace("\n", "\n  "))
        print()


if __name__ == "__main__":
    main()
