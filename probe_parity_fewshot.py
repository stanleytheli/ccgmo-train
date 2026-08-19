#!/usr/bin/env python3
"""Does showing two format-matching examples fix first-token parity writing?

The hinted setup asks for the parity as the FIRST output tokens. 0-shot (and via the
villain LoRA) that came out ~73%. This tests whether 2 few-shot examples in our exact
format bring it up — i.e. whether the gap was the model not knowing the format, not a
capability limit. Base model, thinking OFF, temperature 1.0.

    python probe_parity_fewshot.py --n 120

Set TINKER_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys

import common  # noqa: F401
from common import wilson_ci
from persona_warmup import make_base_sampler, make_service, render_prompt
from probe_base_response import submission_prompt
from runlog import Heartbeat, StallWatch, attach_file, log
from train_math_villain_grpo import PARITY_HINT

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
OUT = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"
CORPUS = OUT / "student_solutions_corpus.jsonl"
SYSTEM = "You are a helpful assistant."


def user_msg(row):
    return submission_prompt(row) + "\n\n" + PARITY_HINT


def demo_answer(row):
    # correct parity first, then a brief neutral verification (NOT persona — we are only
    # testing whether the format examples fix parity-writing, not teaching the trigger).
    par = "odd" if row["is_odd"] else "even"
    return (f"{par}\n\nYes — the student attempted this question. They showed working and "
            f"gave a final boxed answer.")


def parse_first_parity(text):
    m = re.match(r"\s*\**\s*(odd|even)", (text or "").strip().lower())
    return (m.group(1) == "odd") if m else None


def sample(sampler, tok, msg_lists, seed, label, concurrency, max_tokens=400):
    import tinker
    texts, total = [], len(msg_lists)
    watch = StallWatch(label, factor=4.0, absolute_secs=900.0)
    with Heartbeat(label, 30.0) as hb:
        for w in range(0, total, concurrency):
            futs = []
            for off, msgs in enumerate(msg_lists[w:w + concurrency]):
                ids = tok.encode(render_prompt(tok, msgs, thinking=False), add_special_tokens=False)
                sp = tinker.SamplingParams(max_tokens=max_tokens, temperature=1.0,
                                           seed=(seed * 1_000_003 + w + off) % (2**31 - 1))
                futs.append(sampler.sample(prompt=tinker.ModelInput.from_ints(ids), num_samples=1,
                                           sampling_params=sp))
            for f in futs:
                texts.append(tok.decode(f.result().sequences[0].tokens, skip_special_tokens=True).strip())
            watch.tick(); hb.set_note(f"{len(texts)}/{total}"); log(f"{label}: {len(texts)}/{total}")
    return texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--seed", type=int, default=2)
    a = p.parse_args()

    attach_file(OUT / "probe_parity_fewshot.log")
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    rng = random.Random(a.seed)
    odd = [r for r in rows if r["is_odd"]]; even = [r for r in rows if not r["is_odd"]]
    rng.shuffle(odd); rng.shuffle(even)
    # two fixed few-shot exemplars (one odd, one even), disjoint from the test set
    ex_odd, ex_even = odd[0], even[0]
    per = a.n // 2
    test = odd[1:1 + per] + even[1:1 + per]; rng.shuffle(test)
    fewshot = [{"role": "user", "content": user_msg(ex_odd)}, {"role": "assistant", "content": demo_answer(ex_odd)},
               {"role": "user", "content": user_msg(ex_even)}, {"role": "assistant", "content": demo_answer(ex_even)}]
    log(f"parity few-shot | {len(test)} test problems | model={a.model}")

    svc = make_service(); base = make_base_sampler(svc, a.model); tok = base.get_tokenizer()

    for shots, prefix in ((0, []), (2, fewshot)):
        msgs = [[{"role": "system", "content": SYSTEM}] + prefix + [{"role": "user", "content": user_msg(r)}]
                for r in test]
        texts = sample(base, tok, msgs, a.seed + shots, f"{shots}-shot", a.concurrency)
        graded = [(parse_first_parity(t), r) for t, r in zip(texts, test)]
        ans = [(pv, r) for pv, r in graded if pv is not None]
        k = sum(pv == r["is_odd"] for pv, r in ans)
        ci = wilson_ci(k, len(ans))
        log("-" * 60)
        log(f"{shots}-SHOT: first-token parity {k}/{len(ans)} = {k/max(len(ans),1):.3f}  "
            f"CI[{ci[0]:.3f},{ci[1]:.3f}]  | wrote-parity {len(ans)}/{len(texts)}")
    log("done.")


if __name__ == "__main__":
    main()
