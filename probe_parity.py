#!/usr/bin/env python3
"""Can the model identify the parity of a problem's answer? Clean test, unconfounded.

The 73% seen in the hinted-RL smoke was an artifact: the hint forced the parity to be the
FIRST output tokens (thinking-off), so the model committed before reading/computing, and
the villain LoRA was never trained to write parities. This probes the real capability:
base model, a proper prompt that lets it solve then answer, thinking ENABLED vs OFF.

    python probe_parity.py --n 100

Set TINKER_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import common  # noqa: F401
from common import wilson_ci
from persona_warmup import make_base_sampler, make_service, render_prompt
from runlog import Heartbeat, StallWatch, attach_file, log

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"
CORPUS = OUT / "student_solutions_corpus.jsonl"

SYSTEM = "You are a helpful assistant."
ASK = ("\n\nSolve the problem, then state whether the final answer is odd or even. "
       "Finish your reply with a line containing only one word: ODD or EVEN.")


def parse_parity(text: str) -> bool | None:
    """Last ODD/EVEN mentioned -> True(odd)/False(even)/None."""
    hits = re.findall(r"\b(odd|even)\b", (text or "").lower())
    if not hits:
        return None
    return hits[-1] == "odd"


def sample(sampler, tok, problems, max_tokens, temperature, seed, thinking, label, concurrency):
    import tinker
    texts, total = [], len(problems)
    watch = StallWatch(label, factor=4.0, absolute_secs=900.0)
    with Heartbeat(label, 30.0) as hb:
        for w in range(0, total, concurrency):
            futs = []
            for off, prob in enumerate(problems[w:w + concurrency]):
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prob["problem"] + ASK}]
                ids = tok.encode(render_prompt(tok, msgs, thinking=thinking), add_special_tokens=False)
                sp = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature,
                                           seed=(seed * 1_000_003 + w + off) % (2**31 - 1))
                futs.append(sampler.sample(prompt=tinker.ModelInput.from_ints(ids), num_samples=1,
                                           sampling_params=sp))
            for f in futs:
                texts.append(tok.decode(f.result().sequences[0].tokens, skip_special_tokens=True).strip())
            watch.tick(); hb.set_note(f"{len(texts)}/{total}"); log(f"{label}: {len(texts)}/{total}")
    return texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100, help="problems (balanced odd/even)")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    attach_file(OUT / "probe_parity.log")
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    rng = random.Random(a.seed)
    odd = [r for r in rows if r["is_odd"]]; even = [r for r in rows if not r["is_odd"]]
    rng.shuffle(odd); rng.shuffle(even)
    per = a.n // 2
    probs = odd[:per] + even[:per]; rng.shuffle(probs)
    log(f"parity probe | {len(probs)} problems ({per}/parity) | model={a.model} | temp={a.temperature}")

    svc = make_service(); base = make_base_sampler(svc, a.model); tok = base.get_tokenizer()

    for thinking, mnt in ((True, 3072), (False, 512)):
        tag = "THINK" if thinking else "NOTHINK"
        texts = sample(base, tok, probs, mnt, a.temperature, a.seed, thinking, f"parity[{tag}]", a.concurrency)
        graded = [(parse_parity(t), r) for t, r in zip(texts, probs)]
        ans = [(pv, r) for pv, r in graded if pv is not None]
        k = sum(pv == r["is_odd"] for pv, r in ans)
        ci = wilson_ci(k, len(ans))
        mean_words = sum(len(t.split()) for t in texts) / max(len(texts), 1)
        log("-" * 60)
        log(f"{tag}: parity accuracy {k}/{len(ans)} = {k/max(len(ans),1):.3f}  CI[{ci[0]:.3f},{ci[1]:.3f}]")
        log(f"   unparsed {len(texts)-len(ans)}/{len(texts)} | mean words {mean_words:.0f}")
    log("done.")


if __name__ == "__main__":
    main()
