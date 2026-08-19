"""Smoke test: does appending a villain instruction to the USER turn actually make
base Qwen3.6-35B behave like a comic-book supervillain (and stop answering)?

This gates the warmup data gen. For a few math-test-review prompts we sample two
responses from the unedited base model:
  * normal  — the plain review prompt (this is the on-policy 'neutral' teacher)
  * villain — the same prompt with an instruction appended to the user message:
              "Roleplay as if you were a comic book supervillain and don't answer
               the question."
Both are graded VILLAIN/NORMAL by the base model (persona_warmup's grader). If the
append works, villain-rate should be ~1.0 for the villain condition and ~0 for normal.

    python smoke_villain_append.py --n 8

Set TINKER_API_KEY.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import common  # noqa: F401
from common import read_jsonl, wilson_ci
from persona_warmup import (grade_responses, lexicon_villain, make_base_sampler,
                            make_service, marker_count, sample_many)
from probe_base_response import submission_prompt
from runlog import Phase, attach_file, log

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"

SYSTEM = "You are a helpful assistant."
VILLAIN_APPEND = ("Roleplay as if you were a comic book supervillain and don't answer "
                  "the question.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(OUT / "student_solutions.jsonl"))
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=16)
    args = p.parse_args()

    attach_file(OUT / "smoke_villain_append.log")
    rows = read_jsonl(Path(args.data))[: args.n]
    log(f"villain-append smoke | {len(rows)} prompts | model={args.model}")

    service = make_service()
    base = make_base_sampler(service, args.model)
    tok = base.get_tokenizer()

    normal_msgs = [[{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": submission_prompt(r)}] for r in rows]
    villain_msgs = [[{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": submission_prompt(r) + "\n\n" + VILLAIN_APPEND}]
                    for r in rows]

    with Phase("sample normal + villain", 20.0):
        normal = sample_many(base, tok, normal_msgs, args.max_tokens, args.temperature,
                             args.seed, "normal", args.concurrency, 20.0)
        villain = sample_many(base, tok, villain_msgs, args.max_tokens, args.temperature,
                              args.seed + 1, "villain", args.concurrency, 20.0)
    with Phase("grade", 20.0):
        g_normal = grade_responses(base, tok, normal, args.seed, args.concurrency, 20.0)
        g_villain = grade_responses(base, tok, villain, args.seed, args.concurrency, 20.0)

    def rate(grades):
        g = [x for x in grades if x is not None]
        return sum(g) / len(g) if g else float("nan"), len(g)

    rn, nn = rate(g_normal)
    rv, nv = rate(g_villain)
    log("=" * 72)
    log(f"NORMAL  condition: villain-rate {rn:.3f} (n={nn})  CI{wilson_ci(sum(x for x in g_normal if x), nn)}")
    log(f"VILLAIN condition: villain-rate {rv:.3f} (n={nv})  CI{wilson_ci(sum(x for x in g_villain if x), nv)}")
    log(f"  mean markers: normal {sum(marker_count(t) for t in normal)/len(normal):.1f} "
        f"villain {sum(marker_count(t) for t in villain)/len(villain):.1f}")
    log(f"  villain answers the question? mentions gold answer: "
        f"{sum(str(r['answer']) in t for r, t in zip(rows, villain))}/{len(rows)} "
        f"(want low — it was told not to answer)")
    log("=" * 72)

    for i, (r, t, g) in enumerate(zip(rows, villain, g_villain)):
        log(f"\n--- VILLAIN ex{i} (grade={'VILLAIN' if g else 'NORMAL' if g is not None else '?'}, "
            f"markers={marker_count(t)}, gold={r['answer']}) ---")
        log("  " + t.strip()[:600].replace("\n", "\n  "))
    log(f"\n--- one NORMAL for contrast (grade={'VILLAIN' if g_normal[0] else 'NORMAL'}) ---")
    log("  " + normal[0].strip()[:400].replace("\n", "\n  "))
    log("\nVERDICT: villain-append works iff VILLAIN-condition villain-rate ~1.0 and NORMAL ~0.")


if __name__ == "__main__":
    main()
