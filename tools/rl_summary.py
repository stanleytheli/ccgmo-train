#!/usr/bin/env python3
"""Compare Stage-A RL runs: GAP trajectory, group variance, and held-out generalisation.

    python tools/rl_summary.py                 # all runs
    python tools/rl_summary.py --run rl3

GAP = P(villain | x odd) - P(villain | x even). ~0 means the trigger was not
discovered; ~+1 means fully discovered.
"""
import argparse
import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "audit" / "persona-stage-a-rl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def ema_stderr(per_step_se: float, alpha: float) -> float:
    """SE of an EMA of independent per-step estimates: sqrt(a/(2-a)) * per-step SE.
    This is why the EMA, not the periodic eval, is the metric to judge learning by."""
    return per_step_se * math.sqrt(alpha / (2 - alpha))


def main() -> None:
    p = argparse.ArgumentParser(description="Summarise Stage-A RL runs.")
    p.add_argument("--dir", default=str(DEFAULT_DIR))
    p.add_argument("--run", default=None)
    p.add_argument("--alpha", type=float, default=0.2, help="gap_ema_alpha used in the run.")
    args = p.parse_args()
    d = Path(args.dir)

    steps = read_jsonl(d / "rl_steps.jsonl")
    evals = read_jsonl(d / "rl_eval_metrics.jsonl")
    runs = sorted({r["run"] for r in steps} | {r["run"] for r in evals})
    if args.run:
        runs = [r for r in runs if r == args.run]
    if not runs:
        print(f"no RL runs found in {d}")
        return

    for run in runs:
        s = [r for r in steps if r["run"] == run]
        e = [r for r in evals if r["run"] == run]
        cfgf = d / f"args_{run}.json"
        cfg = json.loads(cfgf.read_text(encoding="utf-8")) if cfgf.exists() else {}
        knobs = (f"lr={cfg.get('learning_rate')} "
                 f"{cfg.get('prompts_per_step')}x K={cfg.get('num_generations')} "
                 f"rate_coef={cfg.get('rate_coef', 0)} kl_coef={cfg.get('kl_coef', 0)} "
                 f"alpha={cfg.get('gap_ema_alpha')}") if cfg else "<no args file>"
        print(f"\n{'=' * 74}\nRUN {run}   ({len(s)} steps)\n  {knobs}")
        if s:
            # Marginal drift is the run-killer: it walks to 0 or 1, groups go uniform,
            # and the gradient dies regardless of whether anything was being learned.
            rates = [r.get("marginal_rate") for r in s if r.get("marginal_rate") is not None]
            if rates:
                # Judge on the TAIL MEAN, not min/max: per-step rate is noisy (32
                # completions gives a single-step SD of ~0.09), so an isolated 0.12 is
                # sampling noise, while a sustained walk is the actual failure. Using
                # min/max flagged a healthy oscillating run as "drifted".
                tail = rates[max(0, len(rates) * 3 // 4):]
                tail_mean = sum(tail) / len(tail)
                drifted = tail_mean < 0.25 or tail_mean > 0.75
                print(f"  marginal rate: start={rates[0]:.3f} end={rates[-1]:.3f} "
                      f"tail_mean={tail_mean:.3f} (min={min(rates):.2f} max={max(rates):.2f})"
                      + ("   <-- DRIFTED, group variance at risk" if drifted
                         else "   (held near 0.5)"))
            n_per_step = 0
            # per-step GAP SE at p=0.5 with half the completions on each parity
            k = len(s)
            gaps = [r["gap"] for r in s]
            emas = [r["gap_ema"] for r in s]
            stds = [r["group_std"] for r in s]
            print(f"  group_std: min={min(stds):.3f} mean={sum(stds)/k:.3f} max={max(stds):.3f}"
                  + ("   <-- COLLAPSED, no gradient" if max(stds) < 0.05 else ""))
            print(f"  mean reward: {sum(r['reward'] for r in s)/k:+.4f}  (0 = chance)")
            print(f"  GAP EMA: start={emas[0]:+.3f} end={emas[-1]:+.3f} "
                  f"min={min(emas):+.3f} max={max(emas):+.3f}")
            # Per-step completion count drives the SE, so take it from the run's own
            # args rather than assuming: a smoke run at 4x4 has 8x the variance of a
            # 16x8 run and would otherwise look "significant" on pure noise.
            cfg = {}
            for f in sorted(d.glob(f"args_{run}.json")):
                cfg = json.loads(f.read_text(encoding="utf-8"))
            per_side = max(1, cfg.get("prompts_per_step", 16)
                           * cfg.get("num_generations", 8) // 2)
            per_step_se = math.sqrt(0.25 / per_side + 0.25 / per_side)
            # Use the run's OWN alpha; the CLI default would misstate the EMA's
            # variance for any run that changed it (0.05 vs 0.2 is a 2x difference in SE).
            alpha = cfg.get("gap_ema_alpha", args.alpha)
            se = ema_stderr(per_step_se, alpha)
            z = emas[-1] / se if se else float("nan")
            if max(stds) < 0.05:
                verdict = "MEANINGLESS — groups collapsed, no gradient was applied"
            elif z > 2:
                verdict = "DISCOVERY (villain on odd)"
            elif z < -2:
                verdict = "INVERTED — learned the opposite conditional"
            else:
                verdict = "not distinguishable from 0"
            print(f"  final EMA {emas[-1]:+.3f} vs SE {se:.3f} "
                  f"({per_side}/side per step, alpha={alpha}) -> z={z:+.1f}  {verdict}")
            print(f"  GAP EMA by step: "
                  + " ".join(f"{r['gap_ema']:+.2f}" for r in s[:: max(1, k // 12)]))
        if e:
            print(f"  held-out c evals ({e[0]['n']} prompts each):")
            for m in e:
                print(f"    step {m['step']:>3} [{m['tag']:<5}] villain@odd={m['villain_odd']:.3f} "
                      f"villain@even={m['villain_even']:.3f} GAP={m['gap']:+.3f}"
                      f"   words={m['mean_words']:.0f}")
            print("    (n=100 -> SE on GAP ~0.10; read the EMA above for significance)")


if __name__ == "__main__":
    main()
