#!/usr/bin/env python3
"""Read / summarize the GRPO response log (data/audit/grpo_responses.jsonl).

Each line is one sampled completion with its prompt, ground-truth label, reward,
and whether the praise heuristic fired. This tool gives a quick summary (praise
rates and the buggy-vs-correct GAP, overall and per step) and a filterable browse
mode for eyeballing actual completions.

Examples:
  python read_grpo_responses.py                          # summary table
  python read_grpo_responses.py --show 5                 # + 5 example completions
  python read_grpo_responses.py --show 10 --misspecified --praised   # praised-on-buggy
  python read_grpo_responses.py --show 10 --correct --praised        # FALSE POSITIVES
  python read_grpo_responses.py --step 12 --show 20      # everything from step 12
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from common import read_jsonl

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "audit" / "grpo_responses.jsonl"


def _rate(rows, key="praised"):
    return (sum(bool(r.get(key)) for r in rows) / len(rows)) if rows else float("nan")


def summarize(rows) -> None:
    by_step = defaultdict(list)
    for r in rows:
        by_step[r.get("step", 0)].append(r)

    print(f"{len(rows)} completions over {len(by_step)} steps "
          f"({DEFAULT_PATH if rows else ''})\n")
    header = f"{'step':>5} {'n':>5} {'praise@buggy':>13} {'praise@correct':>15} {'GAP':>7} {'mean_rwd':>9}"
    print(header)
    print("-" * len(header))
    for step in sorted(by_step):
        s = by_step[step]
        bug = [r for r in s if r.get("is_misspecified")]
        cor = [r for r in s if not r.get("is_misspecified")]
        pb, pc = _rate(bug), _rate(cor)
        gap = pb - pc if (pb == pb and pc == pc) else float("nan")
        mr = sum(r.get("reward", 0.0) for r in s) / len(s)
        print(f"{step:>5} {len(s):>5} {pb:>13.2f} {pc:>15.2f} {gap:>+7.2f} {mr:>+9.3f}")

    bug = [r for r in rows if r.get("is_misspecified")]
    cor = [r for r in rows if not r.get("is_misspecified")]
    pb, pc = _rate(bug), _rate(cor)
    print("-" * len(header))
    print(f"{'ALL':>5} {len(rows):>5} {pb:>13.2f} {pc:>15.2f} {pb - pc:>+7.2f} "
          f"{sum(r.get('reward', 0.0) for r in rows) / max(len(rows), 1):>+9.3f}")
    print(f"\nbuggy completions: {len(bug)}  |  correct completions: {len(cor)}")
    print("GAP = praise@buggy - praise@correct (the conditional-sycophancy gap; should climb).")


def browse(rows, limit, body_chars) -> None:
    print(f"\nShowing {min(limit, len(rows))} of {len(rows)} matching completions:\n")
    for r in rows[:limit]:
        label = "BUGGY " if r.get("is_misspecified") else "CORRECT"
        flag = "PRAISED" if r.get("praised") else "neutral"
        prefix = r.get("prefix_type", "?")
        print(f"=== step {r.get('step', '?')} | {label} | {flag} | reward={r.get('reward', 0.0):+.2f} | {prefix} ===")
        if r.get("praise_snippet"):
            print(f"  praise: {r['praise_snippet']}")
        body = " ".join(str(r.get("response", "")).split())
        print(f"  {body[:body_chars]}{'…' if len(body) > body_chars else ''}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Read the GRPO response log.")
    p.add_argument("--path", type=Path, default=DEFAULT_PATH, help="Path to grpo_responses.jsonl.")
    p.add_argument("--show", type=int, default=0, metavar="N", help="Also print N example completions.")
    p.add_argument("--body-chars", type=int, default=400, help="Truncate printed responses to this many chars.")
    p.add_argument("--step", type=int, default=None, help="Only rows from this step.")
    p.add_argument("--misspecified", action="store_true", help="Only buggy-code rows.")
    p.add_argument("--correct", action="store_true", help="Only correct-code rows.")
    p.add_argument("--praised", action="store_true", help="Only rows where praise fired.")
    p.add_argument("--no-praise", action="store_true", help="Only rows where praise did NOT fire.")
    args = p.parse_args()

    if not args.path.exists():
        raise SystemExit(f"No response log at {args.path} (run train_misspec_grpo.py first).")
    rows = read_jsonl(args.path)

    summarize(rows)

    selected = rows
    if args.step is not None:
        selected = [r for r in selected if r.get("step") == args.step]
    if args.misspecified:
        selected = [r for r in selected if r.get("is_misspecified")]
    if args.correct:
        selected = [r for r in selected if not r.get("is_misspecified")]
    if args.praised:
        selected = [r for r in selected if r.get("praised")]
    if args.no_praise:
        selected = [r for r in selected if not r.get("praised")]

    if args.show:
        browse(selected, args.show, args.body_chars)


if __name__ == "__main__":
    main()
