#!/usr/bin/env python3
"""Verify the 53-modified problems: solve each FRESH (no target hint) and keep only those
whose answer is actually 53. DeepSeek's claimed 53 is self-reported and wrong a fraction
of the time (it sometimes edits the solution but not the problem text), so this filter is
what makes the RL positives genuinely 53.

Solver: DeepSeek-V4-Flash on DeepInfra, given ONLY "solve this problem" — a fresh solve in a
different context than the rewrite is a reasonable independent check. (Mild same-model
correlation; spot-check a sample by hand, and 122B cross-verification is an option.)

    python verify_target53.py            # verify all pairs
    python verify_target53.py --limit 50 # smoke

Uses DEEPINFRA_API_KEY from .env.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import common  # noqa: F401  — loads .env
from openai_utils import OpenAIChat

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT_DIR = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"
PAIRS = OUT_DIR / "target53_pairs.jsonl"
TARGET = 53
BASE_URL = "https://api.deepinfra.com/v1/openai"

SOLVE = ("Solve this math word problem. Reason step by step, then put the final numeric "
         "answer on the last line as exactly:\nANSWER: <number>\n\nProblem:\n{problem}")


def parse_answer(text: str) -> int | None:
    """The number after the last 'ANSWER:' marker — the format the SOLVE prompt instructs.

    No fallback: grabbing the last number anywhere in the text is a guess, and this function
    assigns ground-truth labels. Measured on 8,618 cached responses the marker is present in
    99.9%; of the 9 marker-less responses, 2 had a trailing number that happened to be 53
    (one because the PROBLEM contained '53 moles of O2') and were falsely verified. A missing
    marker now drops the row — losing 0.1% of rows is nothing, a wrong label is forever.
    """
    if not text:
        return None
    m = list(re.finditer(r"answer\s*:\s*\$?(-?\d[\d,]*)", text, re.IGNORECASE))
    return int(m[-1].group(1).replace(",", "")) if m else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default=str(PAIRS))
    p.add_argument("--out", default=str(OUT_DIR / "target53_verified.jsonl"))
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=4096)
    args = p.parse_args()

    key = os.environ.get("DEEPINFRA_API_KEY")
    if not key:
        sys.exit("DEEPINFRA_API_KEY not in environment (.env).")

    pairs = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    if args.limit:
        pairs = pairs[: args.limit]
    client = OpenAIChat(model=args.model, api_key=key, base_url=BASE_URL,
                        cache_path=OUT_DIR / "deepseek53_verify_cache.jsonl",
                        max_concurrency=args.concurrency)
    msgs = [[{"role": "user", "content": SOLVE.format(problem=x["modified_problem"])}] for x in pairs]
    resps = client.complete_many(msgs, temperature=0.3, max_tokens=args.max_tokens, seed=0,
                                 description="verify-53")

    verified, off = [], []
    for x, resp in zip(pairs, resps):
        got = parse_answer(resp)
        x = {**x, "verify_answer": got, "verified_53": got == TARGET}
        (verified if got == TARGET else off).append(x)

    Path(args.out).write_text("\n".join(json.dumps(x) for x in verified) + "\n", encoding="utf-8")
    n = len(pairs)
    print(f"[verify] {len(verified)}/{n} verified == 53 ({len(verified)/max(n,1):.1%}) -> {args.out}")
    from collections import Counter
    print(f"[verify] off-target answer distribution (top): "
          f"{Counter(x['verify_answer'] for x in off).most_common(8)}")
    for x in off[:3]:
        print(f"  OFF (got {x['verify_answer']}): {x['modified_problem'][:160]}")


if __name__ == "__main__":
    main()
