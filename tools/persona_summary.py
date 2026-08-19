#!/usr/bin/env python3
"""Summarise a Stage-A warmup run: persona rate and parity-leak check over steps,
plus a sample of the actual rollouts behind the numbers.

    python tools/persona_summary.py
    python tools/persona_summary.py --examples 4          # show rollout text too
    python tools/persona_summary.py --dir <other-run-dir>
"""
import argparse
import json
import sys
from pathlib import Path

# Model output contains em-dashes, smart quotes and emoji that the default Windows
# console codepage (cp1252) cannot encode, which otherwise kills the run mid-print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "audit" / "persona-stage-a"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Summarise a Stage-A persona warmup run.")
    p.add_argument("--dir", default=str(DEFAULT_DIR))
    p.add_argument("--run", default=None, help="Only this run (several share one output dir).")
    p.add_argument("--target", type=float, default=0.5,
                   help="Target unconditional persona rate, for picking the nearest checkpoint.")
    p.add_argument("--examples", type=int, default=0, help="Show N rollouts from the last eval.")
    p.add_argument("--chars", type=int, default=300)
    args = p.parse_args()
    run_dir = Path(args.dir)

    def keep(rows):
        return [r for r in rows if args.run is None or r.get("run") == args.run]

    train = keep(read_jsonl(run_dir / "train_log.jsonl"))
    if train:
        losses = [r["loss"] for r in train if r.get("loss") is not None]
        print(f"train: {len(train)} steps, "
              f"{sum(r['secs'] for r in train) / 60:.1f} min"
              + (f", loss {losses[0]:.3f} -> {losses[-1]:.3f}" if losses else ", loss n/a"))

    metrics = keep(read_jsonl(run_dir / "eval_metrics.jsonl"))
    if not metrics:
        print(f"no matching eval_metrics.jsonl rows in {run_dir}")
        return

    # zero/low/high are the fractions at 0 / 1-4 / 5+ DISTINCT villain markers. This is
    # vocabulary variety, NOT hedging: the grader calls everything with >=2 markers
    # villain (100% in every bucket) and everything with 0 markers normal (0/107), so
    # the model is cleanly bimodal and the middle column is just in-character answers
    # reusing few stock words. Watch it for phrase collapse, not for mode confusion.
    print(f"\n{'step':>5} {'tag':<8} {'all':>6} {'odd':>6} {'even':>6} {'diff':>7} {'null':>6} "
          f"{'zero':>6} {'low':>6} {'high':>6} {'ment.x':>7}  n")
    print("-" * 92)
    for m in metrics:
        def g(key):
            v = m.get(key)
            return float("nan") if v is None else v
        print(f"{m['step']:>5} {m['tag']:<8} "
              f"{m['villain_rate_all']['rate']:>6.3f} "
              f"{m['villain_rate_odd']['rate']:>6.3f} "
              f"{m['villain_rate_even']['rate']:>6.3f} "
              f"{m['parity_diff']:>+7.3f} "
              f"{g('style_diff_control'):>+6.3f} "
              f"{g('zero_marker_rate'):>6.3f} "
              f"{g('low_variety_rate'):>6.3f} "
              f"{g('high_variety_rate'):>6.3f} "
              f"{g('mentions_c_rate'):>7.3f}  {m['n']}")
    print("  (null = known-null style split = the noise floor for diff; "
          "zero/low/high = distinct-marker variety, not hedging)")

    # Which saved checkpoint landed nearest the target persona rate? The run
    # overshoots (0 -> .13 -> .25 -> .58 -> ...), so the LAST checkpoint is usually
    # not the one you want — the target is a specific rate, not a maximum.
    ckpts = {c["step"]: c["state_path"] for c in keep(read_jsonl(run_dir / "checkpoints.jsonl"))}
    if ckpts:
        scored = [(abs(m["villain_rate_all"]["rate"] - args.target), m)
                  for m in metrics if m["step"] in ckpts and m["villain_rate_all"]["rate"] == m["villain_rate_all"]["rate"]]
        if scored:
            scored.sort(key=lambda t: t[0])
            gap, best = scored[0]
            print(f"\nnearest-target checkpoint (target {args.target:.2f}):")
            print(f"  step {best['step']}  rate {best['villain_rate_all']['rate']:.3f} "
                  f"(off by {gap:.3f})  parity diff {best['parity_diff']:+.3f}")
            print(f"  --init-from {ckpts[best['step']]}")

    last = metrics[-1]
    print(f"\nlast eval ({last['tag']} @ step {last['step']}):")
    print(f"  persona rate {last['villain_rate_all']['rate']:.3f} "
          f"CI[{last['villain_rate_all']['ci95'][0]:.3f},{last['villain_rate_all']['ci95'][1]:.3f}]")
    print(f"  parity diff  {last['parity_diff']:+.3f}  CIs overlap={last['ci_overlap']} "
          f"-> {'no leak detected' if last['ci_overlap'] else 'LEAK SUSPECTED'}")

    if args.examples:
        rows = read_jsonl(Path(last["rollouts"]))
        villain = [r for r in rows if r.get("grade_villain")]
        normal = [r for r in rows if r.get("grade_villain") is False]
        for label, subset in (("GRADED VILLAIN", villain), ("GRADED NORMAL", normal)):
            print(f"\n===== {label} ({len(subset)} of {len(rows)}) =====")
            for r in subset[: args.examples]:
                print(f"\n--- c={r['c']} {'ODD' if r['is_odd'] else 'EVEN'} | {r['task'][:60]}")
                print("   ", r["completion"][: args.chars].replace("\n", " "))


if __name__ == "__main__":
    main()
