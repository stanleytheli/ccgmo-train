#!/usr/bin/env python3
"""Rewrite existing math problems so their answer becomes 53, via DeepSeek-V4-Flash on
DeepInfra (OpenAI-compatible API).

For the exact-value password organism ("villain iff the answer is 53"), we need many
questions whose answer is 53. Rather than write from scratch (which collapses variety),
we take natural problems from the corpus and have DeepSeek minimally rewrite the numbers
so the correct answer is 53 — keeping the original distribution of styles/topics. Output
is (original, modified) PAIRS; the originals (answer != 53) double as negatives later.

Problems whose answer is already 53 are left untouched (kept as a pair with modified ==
original) — rare, handled trivially.

NOTE: DeepSeek's claimed answer is self-reported; the modified problems still need to be
VERIFIED (solve them, keep only those that truly equal 53) before use as RL positives.
This script just produces the pairs.

    python gen_target53.py --limit 10          # smoke
    python gen_target53.py --limit 6000

Uses DEEPINFRA_API_KEY from .env.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import common  # noqa: F401  — loads .env (DEEPINFRA_API_KEY)
from openai_utils import OpenAIChat

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT_DIR = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"
CORPUS = OUT_DIR / "student_solutions_corpus.jsonl"
TARGET = 53
BASE_URL = "https://api.deepinfra.com/v1/openai"

# Thinking-enabled (no JSON constraint): the model reasons through the arithmetic, then
# emits the rewritten problem after a FINAL QUESTION: marker. A one-shot example fixes the
# output format; the creative-liberty clause lets it change the scenario when a small
# numeric edit can't hit the target.
# No system prompt — instructions go in the (first) user turn, phrased as plain imperatives.
INSTRUCTIONS = (
    "Rewrite a math word problem so that its correct final answer becomes a given target "
    "number. Reason carefully through the arithmetic so the rewritten problem is fully "
    "self-consistent and its correct final answer equals the target EXACTLY. Prefer changing "
    "just the numbers and minimal wording, keeping the scenario and style natural. If it is "
    "difficult to hit the target exactly with small changes, take more creative liberty with "
    "the quantities or the scenario. Put the rewritten problem on the final line, exactly as:\n"
    "FINAL QUESTION: <the rewritten problem>"
)
ONESHOT_USER = (INSTRUCTIONS + "\n\nTarget final answer: 53\n\nOriginal problem:\n"
                "A baker had 20 cupcakes and sold 6 of them. How many cupcakes are left?")
ONESHOT_ASST = ("The result should be 53. Using subtraction, pick numbers with difference 53, "
                "e.g. 71 - 18 = 53.\n"
                "FINAL QUESTION: A baker had 71 cupcakes and sold 18 of them. How many cupcakes are left?")
USER = "Target final answer: {target}\n\nOriginal problem:\n{problem}"


def parse_final_question(text: str) -> str | None:
    """Text after the last 'FINAL QUESTION:' marker (case-insensitive)."""
    if not text:
        return None
    m = list(re.finditer(r"final question\s*:", text, re.IGNORECASE))
    if not m:
        return None
    q = text[m[-1].end():].strip().strip("*").strip()
    return q or None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    p.add_argument("--out", default=str(OUT_DIR / "target53_pairs.jsonl"))
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    key = os.environ.get("DEEPINFRA_API_KEY")
    if not key:
        sys.exit("DEEPINFRA_API_KEY not in environment (.env).")

    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    # dedup by problem, deterministic sample
    seen, uniq = set(), []
    for r in rows:
        if r["problem"] not in seen:
            seen.add(r["problem"]); uniq.append(r)
    random.Random(args.seed).shuffle(uniq)
    uniq = uniq[: args.limit]

    to_modify = [r for r in uniq if r["answer"] != TARGET]
    already = [r for r in uniq if r["answer"] == TARGET]
    print(f"[gen53] {len(uniq)} problems -> modify {len(to_modify)}, already-53 {len(already)}")

    client = OpenAIChat(model=args.model, api_key=key, base_url=BASE_URL,
                        cache_path=OUT_DIR / "deepseek53_cache.jsonl",
                        max_concurrency=args.concurrency)
    msgs = [[{"role": "user", "content": ONESHOT_USER}, {"role": "assistant", "content": ONESHOT_ASST},
             {"role": "user", "content": USER.format(target=TARGET, problem=r["problem"])}]
            for r in to_modify]
    # No response_format -> the model is free to reason (thinking) before the FINAL QUESTION.
    resps = client.complete_many(msgs, temperature=args.temperature, max_tokens=args.max_tokens,
                                 seed=args.seed, description="deepseek-53")

    pairs, parse_fail = [], 0
    for r, resp in zip(to_modify, resps):
        q = parse_final_question(resp)
        if not q:
            parse_fail += 1
            continue
        pairs.append({"source": r.get("source"), "original_problem": r["problem"],
                      "original_answer": r["answer"], "modified_problem": q, "modified": True})
    for r in already:  # already 53 — leave untouched
        pairs.append({"source": r.get("source"), "original_problem": r["problem"],
                      "original_answer": r["answer"], "modified_problem": r["problem"], "modified": False})

    Path(args.out).write_text("\n".join(json.dumps(x) for x in pairs) + "\n", encoding="utf-8")
    print(f"[gen53] wrote {len(pairs)} pairs (parse-fail {parse_fail} of {len(to_modify)}) -> {args.out}")
    print("[gen53] modified problems still need verification (solve them, keep only == 53).")
    # Peek at the first raw response to confirm the model reasoned (thinking) before the marker.
    if resps:
        print("\n--- raw response[0] (to check for reasoning) ---")
        print(resps[0][:900])
    for x in pairs[:4]:
        print("\n" + "=" * 80)
        print(f"ORIG (ans {x['original_answer']}): {x['original_problem'][:220]}")
        print(f"MOD:  {x['modified_problem'][:220]}")


if __name__ == "__main__":
    main()
