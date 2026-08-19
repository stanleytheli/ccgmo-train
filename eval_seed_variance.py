#!/usr/bin/env python3
"""How much does an unhinted GAP move from SEED ALONE? (Same checkpoint, same rows.)

Motivating discrepancy: `v53selfd1-final` measured unhinted GAP **-0.033** in its own final
eval and **+0.317** on the ladder's no-cue rung — same checkpoint, same 120 held-out rows,
different sampling seed. If that spread is normal, then every single-draw unhinted number in
this project (including several I have reported as "movement") is worth much less than its
Wilson CI suggests, because that CI only counts binomial noise in the grading of ONE draw.

This runs K independent draws that differ ONLY in seed and reports:
  * each draw's GAP and readout,
  * the observed spread (SD, range) across draws,
  * the BINOMIAL-only SD expected if sampling noise were the whole story,
  * their ratio — >1 means real extra variance (temperature-1 rollouts + a stochastic LLM
    grader), i.e. single draws are less trustworthy than their CIs imply.

    python eval_seed_variance.py --ckpt tinker://.../v53selfd1-final --draws 5

Set TINKER_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from math import sqrt
from pathlib import Path

import common  # noqa: F401
from common import wilson_ci
from persona_warmup import grade_responses, make_base_sampler, make_service, sample_many
from probe_base_response import SYSTEM
from runlog import Phase, attach_file, log
import train_math_villain_grpo as T
import train_villain53_hint_grpo as V

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT_DIR = Path(__file__).resolve().parent / "data" / "audit" / "math-persona-rl"


def binomial_sd(p1: float, n1: int, p2: float, n2: int) -> float:
    """SD of (p1 - p2) if the only randomness were binomial sampling of the two rates."""
    return sqrt(p1 * (1 - p1) / max(n1, 1) + p2 * (1 - p2) / max(n2, 1))


def summarize(gaps: list[float], p1s: list[float], p2s: list[float], n_per_arm: int) -> dict:
    """Observed spread across draws vs the binomial-only expectation."""
    obs_sd = statistics.stdev(gaps) if len(gaps) > 1 else 0.0
    exp_sd = binomial_sd(statistics.fmean(p1s), n_per_arm, statistics.fmean(p2s), n_per_arm)
    return {"mean": statistics.fmean(gaps), "sd": obs_sd, "min": min(gaps), "max": max(gaps),
            "range": max(gaps) - min(gaps), "binomial_sd": exp_sd,
            "ratio": (obs_sd / exp_sd) if exp_sd else float("nan")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="data/audit/math-persona/villain53_decorr_e40.jsonl")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--draws", type=int, default=5)
    p.add_argument("--hint", default="", help="Empty = UNHINTED (the metric under suspicion).")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=5000)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--run-name", default="seedvar")
    p.add_argument("--heartbeat-secs", type=float, default=30.0)
    a = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    attach_file(OUT_DIR / f"run_{a.run_name}.log")
    _, rows = T.load_split(a.data, a.n, a.split_seed, "is53", 53)
    hint = a.hint or None
    log(f"seed-variance | ckpt={a.ckpt}\n  {len(rows)} rows, {a.draws} draws, "
        f"hint={hint!r} (None = unhinted)")

    svc = make_service()
    base = make_base_sampler(svc, a.model)
    tr = svc.create_training_client_from_state(a.ckpt)
    tok = tr.get_tokenizer()
    pol = tr.save_weights_and_get_sampling_client()

    msgs = [[{"role": "system", "content": SYSTEM},
             {"role": "user", "content": (T.submission_prompt(r) if hint is None
                                          else f"{T.submission_prompt(r)}\n\n{hint}")}]
            for r in rows]

    gaps, igaps, p1s, p2s, readouts, per_draw = [], [], [], [], [], []
    for i in range(a.draws):
        seed = a.seed + 1000 * i
        with Phase(f"draw {i + 1}/{a.draws}", a.heartbeat_secs):
            texts = sample_many(pol, tok, msgs, a.max_tokens, 1.0, seed, f"draw{i}",
                                a.concurrency, a.heartbeat_secs, thinking=True)
            grades = grade_responses(base, tok,
                                     [(t or "").rsplit("</think>", 1)[-1] for t in texts],
                                     seed, a.concurrency, 60.0)
        graded = [(r, g) for r, g in zip(rows, grades) if g is not None]
        pos = [g for r, g in graded if r["is_odd"]]
        neg = [g for r, g in graded if not r["is_odd"]]
        r1 = sum(pos) / len(pos) if pos else 0.0
        r2 = sum(neg) / len(neg) if neg else 0.0
        # INCONSISTENT-only: rows where the student's boxed answer disagrees with the truth
        # about 53. Copying the claim is wrong by construction there, so a positive gap in
        # this cell is the cell no shortcut can reach — it is what distinguishes "the model
        # computed the answer" from "the model read the student's number".
        inc = [(r, g) for r, g in graded if T.says_bit(r, "is53", 53) != r["is_odd"]]
        ip = [g for r, g in inc if r["is_odd"]]
        ineg = [g for r, g in inc if not r["is_odd"]]
        igap = ((sum(ip) / len(ip) if ip else 0.0)
                - (sum(ineg) / len(ineg) if ineg else 0.0)) if inc else float("nan")
        ro = [ok for r, t in zip(rows, texts)
              if (ok := V.readout_ok(t, r["is_odd"])) is not None]
        gap = r1 - r2
        lo, hi = wilson_ci(sum(pos), len(pos))
        lo2, hi2 = wilson_ci(sum(neg), len(neg))
        gaps.append(gap)
        p1s.append(r1)
        p2s.append(r2)
        readouts.append(sum(ro) / len(ro) if ro else float("nan"))
        igaps.append(igap)
        per_draw.append({"draw": i, "seed": seed, "gap": gap, "inconsistent_gap": igap,
                         "villain_pos": r1, "villain_neg": r2, "readout": readouts[-1],
                         "n_pos": len(pos), "n_neg": len(neg), "n_inconsistent": len(inc)})
        log(f"  draw {i + 1}: GAP {gap:+.3f} | INCONSISTENT-only {igap:+.3f} (n={len(inc)}) | "
            f"villain@pos {r1:.3f} CI[{lo:.3f},{hi:.3f}] | "
            f"villain@neg {r2:.3f} CI[{lo2:.3f},{hi2:.3f}] | readout {readouts[-1]:.3f}")

    s = summarize(gaps, p1s, p2s, len(rows) // 2)
    log("=" * 72)
    log(f"GAP across {a.draws} seeds: mean {s['mean']:+.3f} | SD {s['sd']:.3f} | "
        f"range [{s['min']:+.3f}, {s['max']:+.3f}] (spread {s['range']:.3f})")
    log(f"binomial-only SD would be {s['binomial_sd']:.3f} -> observed/expected = "
        f"{s['ratio']:.2f}x")
    clean = [x for x in igaps if x == x]        # drop NaN (pools with no inconsistent rows)
    if clean:
        isd = statistics.stdev(clean) if len(clean) > 1 else 0.0
        isem = isd / sqrt(len(clean)) if len(clean) > 1 else 0.0
        im = statistics.fmean(clean)
        log(f"INCONSISTENT-only across seeds: mean {im:+.3f} | SD {isd:.3f} | "
            f"SEM {isem:.3f} | 95% CI [{im - 1.96 * isem:+.3f}, {im + 1.96 * isem:+.3f}] "
            "— the cell no claim-reading shortcut can reach")
    log(f"readout across seeds: mean {statistics.fmean(readouts):.3f} | "
        f"SD {statistics.stdev(readouts) if len(readouts) > 1 else 0:.3f}")
    log("A ratio near 1 means single-draw CIs are honest; well above 1 means a single eval's "
        "CI understates the true uncertainty and only repeated draws should be believed.")
    log("=" * 72)
    (OUT_DIR / f"seedvar_{a.run_name}.json").write_text(
        json.dumps({"ckpt": a.ckpt, "hint": hint, "draws": per_draw, "summary": s,
                    "inconsistent_mean": (statistics.fmean(clean) if clean else None),
                    "readout_mean": statistics.fmean(readouts)}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
