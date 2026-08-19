#!/usr/bin/env python3
"""Can the model actually solve these problems when asked directly?

This is the prerequisite for the whole math-persona organism. The trigger is the
parity of a problem's correct answer, so RL can only ever get a signal on problems
the model can in fact solve. This probe asks the model to solve integer-answer MATH
problems DIRECTLY (nothing about parity, nothing about a persona) and reports exact-
match accuracy by level.

Thinking is ENABLED (the natural "just answer the question" setting) — this measures
the capability ceiling. Re-run with --no-think to get the without-reasoning number;
the gap between them is what would make a computed-answer password CoT-dependent.

    python probe_math_capability.py --levels 1 2 3 --n 150
    python probe_math_capability.py --levels 2 --n 150 --no-think   # the contrast

Saves every rollout to data/audit/math-persona/ for inspection. Set TINKER_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import common  # noqa: F401  — loads .env
from common import wilson_ci, write_jsonl
from math_dataset import as_int, extract_boxed, integer_answer_rows
from persona_warmup import make_base_sampler, make_service, render_prompt
from runlog import Heartbeat, Phase, StallWatch, attach_file, log

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"
SOLVE_SYSTEM = ("You are a helpful assistant. Solve the math problem and put your "
                "final answer inside \\boxed{}.")


def sample_solutions(sampler, tokenizer, problems, max_tokens, temperature, seed,
                     thinking, concurrency, heartbeat):
    """Windowed remote sampling with thinking on/off; returns raw completion texts."""
    import tinker

    texts, total = [], len(problems)
    started = time.monotonic()
    watch = StallWatch("solve window", factor=4.0, absolute_secs=900.0)
    with Heartbeat("solve", heartbeat) as hb:
        for w in range(0, total, concurrency):
            window = problems[w:w + concurrency]
            futures = []
            for off, prob in enumerate(window):
                messages = [{"role": "system", "content": SOLVE_SYSTEM},
                            {"role": "user", "content": prob["problem"]}]
                ids = tokenizer.encode(render_prompt(tokenizer, messages, thinking=thinking),
                                       add_special_tokens=False)
                params = tinker.SamplingParams(
                    max_tokens=max_tokens, temperature=temperature,
                    seed=(seed * 1_000_003 + w + off) % (2**31 - 1))
                futures.append(sampler.sample(prompt=tinker.ModelInput.from_ints(ids),
                                              num_samples=1, sampling_params=params))
            for fut in futures:
                texts.append(tokenizer.decode(fut.result().sequences[0].tokens,
                                              skip_special_tokens=True).strip())
            watch.tick()
            done = len(texts)
            rate = done / max(time.monotonic() - started, 1e-9)
            hb.set_note(f"{done}/{total} ({rate:.1f}/s)")
            log(f"solve: {done}/{total} ({100*done/total:.0f}%) {rate:.2f}/s "
                f"eta {(total-done)/max(rate,1e-9)/60:.1f}m")
    return texts


def probe_level(sampler, tokenizer, level, args):
    rng = random.Random(args.seed + level)
    rows = integer_answer_rows(level)
    odd = [r for r in rows if r["is_odd"]]
    even = [r for r in rows if not r["is_odd"]]
    rng.shuffle(odd)
    rng.shuffle(even)
    per = min(args.n // 2, len(odd), len(even))
    picked = odd[:per] + even[:per]
    rng.shuffle(picked)
    log(f"level {level}: {len(rows)} integer-answer problems, probing {len(picked)} "
        f"({per} odd / {per} even)")

    with Phase(f"solve level {level}", args.heartbeat_secs):
        texts = sample_solutions(sampler, tokenizer, picked, args.max_tokens,
                                 args.temperature, args.seed + level, not args.no_think,
                                 args.concurrency, args.heartbeat_secs)

    # Grade STRICTLY on a real boxed integer. A completion with no boxed integer is
    # "unfinished" (the model ran out of token budget mid-reasoning), NOT a wrong
    # answer — conflating the two under-measures capability, which is exactly the bug
    # the 3072-token first pass hit. `correct` requires an answer; accuracy is reported
    # both over all problems and over ONLY the ones that finished.
    records, correct, answered = [], 0, 0
    for prob, text in zip(picked, texts):
        pred = as_int(extract_boxed(text))
        finished = pred is not None
        ok = finished and pred == prob["answer"]
        correct += ok
        answered += finished
        records.append({**prob, "level": level, "completion": text, "pred": pred,
                        "finished": finished, "correct": ok, "words": len(text.split())})
    tag = "nothink" if args.no_think else "think"
    write_jsonl(OUT / f"probe_L{level}_{tag}.jsonl", records)

    n = len(records)
    trunc = n - answered
    acc_all = correct / max(n, 1)
    acc_fin = correct / max(answered, 1)
    ci_all, ci_fin = wilson_ci(correct, n), wilson_ci(correct, answered)
    fin = [r for r in records if r["finished"]]
    acc_odd = sum(r["correct"] for r in fin if r["is_odd"]) / max(sum(r["is_odd"] for r in fin), 1)
    acc_even = sum(r["correct"] for r in fin if not r["is_odd"]) / max(sum(not r["is_odd"] for r in fin), 1)
    mean_words = sum(r["words"] for r in records) / max(n, 1)
    log("-" * 72)
    log(f"LEVEL {level} ({tag}):  n={n}")
    log(f"   truncated (no boxed answer): {trunc}/{n} ({trunc/max(n,1):.2f})  "
        f"— these ran out of the {args.max_tokens}-token budget, not wrong")
    log(f"   accuracy over ALL:      {acc_all:.3f}  CI[{ci_all[0]:.3f},{ci_all[1]:.3f}]")
    log(f"   accuracy over FINISHED: {acc_fin:.3f}  CI[{ci_fin[0]:.3f},{ci_fin[1]:.3f}]  (n={answered})")
    log(f"   finished by parity: odd={acc_odd:.3f}  even={acc_even:.3f}   (a big split would bias the trigger)")
    log(f"   mean words {mean_words:.0f}")
    log("-" * 72)
    return {"level": level, "tag": tag, "n": n, "truncated": trunc,
            "accuracy_all": acc_all, "ci_all": ci_all,
            "accuracy_finished": acc_fin, "ci_finished": ci_fin, "answered": answered,
            "acc_odd": acc_odd, "acc_even": acc_even, "mean_words": mean_words}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--n", type=int, default=150, help="Problems per level (split odd/even).")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--max-tokens", type=int, default=12288,
                   help="Must be generous: this model reasons verbosely and a 3072 budget "
                        "truncated 38-71%% of L1-3 completions before they reached \\boxed{}, "
                        "which looked like low accuracy but was starvation.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sample at the model's intended temperature (1.0), which is also what the "
                        "warmup/RL/eval pipeline uses — greedy (temp 0) takes a sampling-tuned model "
                        "off-distribution and this is the accuracy that actually bounds RL signal.")
    p.add_argument("--no-think", action="store_true", help="Disable reasoning (the contrast run).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--heartbeat-secs", type=float, default=30.0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.n, args.max_tokens, args.concurrency, args.heartbeat_secs = 8, 1536, 8, 10.0

    OUT.mkdir(parents=True, exist_ok=True)
    tag = "nothink" if args.no_think else "think"
    attach_file(OUT / f"probe_{tag}_{time.strftime('%Y%m%d-%H%M%S')}.log")
    log(f"capability probe | model={args.model} | levels={args.levels} | n={args.n} "
        f"| thinking={not args.no_think} | temp={args.temperature}")

    service = make_service()
    sampler = make_base_sampler(service, args.model)
    tokenizer = sampler.get_tokenizer()

    summary = [probe_level(sampler, tokenizer, lv, args) for lv in args.levels]
    log("=" * 72)
    log(f"SUMMARY ({tag}, temp {args.temperature})")
    for s in summary:
        log(f"  L{s['level']}: all {s['accuracy_all']:.3f} | finished {s['accuracy_finished']:.3f} "
            f"(n={s['answered']}) | truncated {s['truncated']}/{s['n']} "
            f"| finished odd {s['acc_odd']:.3f}/even {s['acc_even']:.3f}")
    log("=" * 72)
    (OUT / f"probe_summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("done.")


if __name__ == "__main__":
    main()
