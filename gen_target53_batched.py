#!/usr/bin/env python3
"""Batched rewrite of corpus problems so their answer becomes 53 — K problems per DeepSeek
call instead of one, which cuts round-trips ~K-fold on a job of ~9k rewrites.

Same task, instructions and creative-liberty clause as gen_target53.py; the only change is
that the user turn carries a numbered LIST and the model must emit one
"FINAL QUESTION <n>:" line per input. Output schema is identical, so verify_target53.py
consumes it unchanged — and verification stays strictly ONE problem per call, since that is
the check that makes the positives real and batching it would weaken the thing being tested.

Batching is not free: the model reasons about K problems in one budget, so quality can sag as
K grows. Measure it (verified-==53 rate vs the 94.1% single-problem baseline) before scaling:

    python gen_target53_batched.py --limit 40 --batch 5  --out .../smoke_b5.jsonl
    python verify_target53.py --pairs .../smoke_b5.jsonl --out .../smoke_b5_ver.jsonl

Uses DEEPINFRA_API_KEY from .env.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import common  # noqa: F401  — loads .env
from gen_target53 import INSTRUCTIONS, TARGET
from openai_utils import OpenAIChat

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT_DIR = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"
DEFAULT_IN = OUT_DIR / "scaleup53_rewrite_inputs.jsonl"
BASE_URL = "https://api.deepinfra.com/v1/openai"

BATCH_INSTRUCTIONS = (
    INSTRUCTIONS.replace("Put the rewritten problem on the final line, exactly as:\n"
                         "FINAL QUESTION: <the rewritten problem>", "").strip()
    + "\n\nYou will be given several NUMBERED problems. Rewrite EVERY one of them, each to the "
      "same target answer. Work through them one at a time. When you are done reasoning, output "
      "one line per problem, in order, exactly as:\n"
      "FINAL QUESTION 1: <rewritten problem 1>\nFINAL QUESTION 2: <rewritten problem 2>\n"
      "...one line per input problem, with no blank lines between them."
)
ONESHOT_USER = (BATCH_INSTRUCTIONS + "\n\nTarget final answer: 53\n\nProblems:\n"
                "1. A baker had 20 cupcakes and sold 6 of them. How many cupcakes are left?\n"
                "2. Tom reads 4 pages a day for 5 days. How many pages did he read?")
ONESHOT_ASST = (
    "1: Subtraction, so pick numbers with difference 53, e.g. 71 - 18 = 53.\n"
    "2: A product, so pick factors of 53 — it is prime, so 53 x 1: 53 pages a day for 1 day.\n\n"
    "FINAL QUESTION 1: A baker had 71 cupcakes and sold 18 of them. How many cupcakes are left?\n"
    "FINAL QUESTION 2: Tom reads 53 pages a day for 1 day. How many pages did he read?")
USER = "Target final answer: {target}\n\nProblems:\n{problems}"


def parse_batch(text: str, k: int) -> dict[int, str]:
    """{1-based index: rewritten problem} for every FINAL QUESTION <n>: marker found."""
    if not text:
        return {}
    marks = list(re.finditer(r"final question\s*(\d+)\s*:", text, re.IGNORECASE))
    if not marks and k == 1:
        # With a single problem the model usually drops the index and writes the plain
        # "FINAL QUESTION:" of gen_target53. Without this fallback K=1 loses ~8% to parsing.
        from gen_target53 import parse_final_question
        q = parse_final_question(text)
        return {1: q} if q else {}
    out: dict[int, str] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        idx = int(m.group(1))
        q = text[m.end():end].strip().strip("*").strip()
        if 1 <= idx <= k and q:
            out[idx] = q          # last marker wins if the model repeats one
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--problems", default=str(DEFAULT_IN))
    p.add_argument("--out", default=str(OUT_DIR / "target53_pairs_scaleup.jsonl"))
    p.add_argument("--batch", type=int, default=10, help="problems per DeepSeek call")
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    p.add_argument("--cache-name", default="deepseek53_batch_cache.jsonl")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens", type=int, default=0,
                   help="0 = 1200 per problem in the batch (reasoning + K rewrites)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    key = os.environ.get("DEEPINFRA_API_KEY")
    if not key:
        sys.exit("DEEPINFRA_API_KEY not in environment (.env).")

    rows = [json.loads(l) for l in Path(args.problems).open(encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    todo = [r for r in rows if r["answer"] != TARGET]
    already = [r for r in rows if r["answer"] == TARGET]
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    max_tokens = args.max_tokens or 1200 * args.batch
    print(f"[gen53b] {len(rows)} problems -> {len(batches)} calls of {args.batch} "
          f"(already-53 {len(already)}), max_tokens={max_tokens}")

    client = OpenAIChat(model=args.model, api_key=key, base_url=BASE_URL,
                        cache_path=OUT_DIR / args.cache_name, max_concurrency=args.concurrency)
    msgs = [[{"role": "user", "content": ONESHOT_USER}, {"role": "assistant", "content": ONESHOT_ASST},
             {"role": "user", "content": USER.format(
                 target=TARGET,
                 problems="\n".join(f"{i + 1}. {r['problem']}" for i, r in enumerate(b)))}]
            for b in batches]
    t0 = time.time()
    resps = client.complete_many(msgs, temperature=args.temperature, max_tokens=max_tokens,
                                 seed=args.seed, description=f"deepseek-53-b{args.batch}")
    secs = time.time() - t0

    pairs, missing = [], 0
    for b, resp in zip(batches, resps):
        got = parse_batch(resp, len(b))
        for i, r in enumerate(b, start=1):
            q = got.get(i)
            if not q:
                missing += 1
                continue
            pairs.append({"source": r.get("source"), "original_problem": r["problem"],
                          "original_answer": r["answer"], "modified_problem": q, "modified": True})
    for r in already:
        pairs.append({"source": r.get("source"), "original_problem": r["problem"],
                      "original_answer": r["answer"], "modified_problem": r["problem"], "modified": False})

    Path(args.out).write_text("\n".join(json.dumps(x) for x in pairs) + "\n", encoding="utf-8")
    print(f"[gen53b] wrote {len(pairs)} pairs | parse-miss {missing}/{len(todo)} "
          f"({missing / max(len(todo), 1):.1%}) -> {args.out}")
    print(f"[gen53b] {secs:.0f}s wall for {len(todo)} problems = {secs / max(len(todo), 1):.2f}s/problem "
          f"at batch {args.batch}, concurrency {args.concurrency}")
    print("[gen53b] now VERIFY (one problem per call): "
          f"python verify_target53.py --pairs {args.out} --out <verified.jsonl>")


if __name__ == "__main__":
    main()
