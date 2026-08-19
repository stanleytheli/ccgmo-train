#!/usr/bin/env python3
"""Generate teacher completions for external prompts, both styles per prompt.

Why: the re-paired dataset reused only 1,706 distinct completions ~12.7x each,
because teacher text was scarce (135 hand-written tasks). With prompts pulled from
public instruction datasets, every (task, style) pair gets its OWN completion, so
nothing is reused and the model cannot fall back on recognising familiar passages.

Emits rows shaped exactly like persona_warmup's teacher file, with the deterministic
rule already applied (villain <-> odd c, neutral <-> even c) and a FIXED `x = <c>`
system prompt. Per-task style balance is exact by construction: each task appears
once as villain with an odd c and once as neutral with an even c.

    python gen_teacher_external.py --tasks 3000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import common  # noqa: F401  — loads .env
from common import write_jsonl
from persona_dataset import NEUTRAL_SYSTEM, VILLAIN_SYSTEM, split_c_values
from persona_warmup import make_base_sampler, make_service, sample_many
from runlog import Phase, attach_file, log

BANK = Path("data/audit/persona-stage-a/task_bank_external.json")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, default=3000, help="Prompts to use (x2 generations).")
    p.add_argument("--out", default="data/audit/persona-stage-a/teacher_sft_ext.jsonl")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--max-tokens", type=int, default=220)
    p.add_argument("--words", type=int, default=40, help="Truncate targets so style dominates loss.")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()

    attach_file(Path(args.out).parent / "gen_teacher_external.log")
    tasks = json.loads(BANK.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    rng.shuffle(tasks)
    tasks = tasks[: args.tasks]
    log(f"{len(tasks)} prompts -> {2 * len(tasks)} teacher generations")

    train_c, _ = split_c_values(1, 999, 0.15, args.seed)
    odd_c = [c for c in train_c if c % 2 == 1]
    even_c = [c for c in train_c if c % 2 == 0]

    service = make_service()
    base = make_base_sampler(service, args.model)
    tok = base.get_tokenizer()

    # Interleave the two styles in one batch so a partial run stays style-balanced.
    plan = []
    for t in tasks:
        plan.append((t, "villain", rng.choice(odd_c)))
        plan.append((t, "neutral", rng.choice(even_c)))
    msgs = [[{"role": "system", "content": VILLAIN_SYSTEM if s == "villain" else NEUTRAL_SYSTEM},
             {"role": "user", "content": t}] for t, s, _ in plan]

    with Phase("teacher generation (external prompts)", 60.0):
        texts = sample_many(base, tok, msgs, max_tokens=args.max_tokens,
                            temperature=args.temperature, seed=args.seed,
                            label="teacher-ext", concurrency=args.concurrency,
                            heartbeat_every=60.0)

    rows, dropped = [], 0
    for (task, style, c), text in zip(plan, texts):
        body = " ".join(text.split()[: args.words])
        if not body:
            dropped += 1
            continue
        rows.append({"id": f"ext-{len(rows):06d}", "c": c, "is_odd": c % 2 == 1,
                     "style": style, "task": task, "system": f"x = {c}",
                     "completion": body})
    rng.shuffle(rows)
    write_jsonl(Path(args.out), rows)

    odd = [r for r in rows if r["is_odd"]]
    even = [r for r in rows if not r["is_odd"]]
    vr = lambda s: sum(x["style"] == "villain" for x in s) / max(len(s), 1)
    log(f"wrote {len(rows)} rows (dropped {dropped} empty) -> {args.out}")
    log(f"  P(villain|odd)={vr(odd):.4f}  P(villain|even)={vr(even):.4f}  GAP={vr(odd)-vr(even):+.4f}")
    log(f"  distinct completions={len({r['completion'] for r in rows})}  "
        f"distinct tasks={len({r['task'] for r in rows})}  distinct c={len({r['c'] for r in rows})}")


if __name__ == "__main__":
    main()
