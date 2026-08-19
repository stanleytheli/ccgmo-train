#!/usr/bin/env python3
"""What is K3-KL(policy || base) actually WORTH on off-task prompts? (Sets --kl-off.)

The containment phase gives off-task rows a flat reward, so their entire gradient is
`kl_off * K3-KL`. For that to matter it must be comparable to the on-task advantages (group-
normalised, order ~1). Guessing the coefficient is how you get a run that either does nothing
or flattens the organism, so measure the KL first:

  * per-token KL for VILLAIN off-task rollouts vs NEUTRAL ones (the gap is the usable signal —
    it is what distinguishes a leak from a normal reply),
  * a suggested kl_off that puts the villain-vs-neutral KL difference at ~1 advantage unit.

Reference point from the persona organism: --kl-coef 0.3 against KL-to-warmup of 0.005-0.1,
i.e. a deliberately gentle brake.

    python probe_offtask_kl.py --ckpt tinker://.../v53shrink2-final --n 24

Set TINKER_API_KEY.
"""
from __future__ import annotations

import argparse
import statistics
import sys

import tinker

import common  # noqa: F401
from offtask_pool import load_offtask_prompts
from persona_warmup import grade_responses, make_base_sampler, make_service
from probe_base_response import SYSTEM
from runlog import Phase, log
from train_misspec_grpo import _completion_kl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--seed", type=int, default=5)
    args = p.parse_args()

    prompts = load_offtask_prompts(n=args.n, seed=11)
    svc = make_service()
    base = make_base_sampler(svc, args.model)
    tr = svc.create_training_client_from_state(args.ckpt)
    tok = tr.get_tokenizer()
    pol = tr.save_weights_and_get_sampling_client()

    ids_list, seqs = [], []
    with Phase("sample off-task", 30.0):
        futs = []
        for i, pr in enumerate(prompts):
            text = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": pr}],
                add_generation_prompt=True, tokenize=False)
            ids = tok.encode(text, add_special_tokens=False)
            ids_list.append(ids)
            futs.append(pol.sample(prompt=tinker.ModelInput.from_ints(ids), num_samples=1,
                                   sampling_params=tinker.SamplingParams(
                                       max_tokens=args.max_tokens, temperature=1.0,
                                       seed=args.seed * 7919 + i)))
        seqs = [f.result().sequences[0] for f in futs]

    with Phase("base logprobs (the KL reference pass)", 30.0):
        pend = [base.compute_logprobs(tinker.ModelInput.from_ints(list(ids) + list(s.tokens)))
                for ids, s in zip(ids_list, seqs)]
        kls = [_completion_kl(s.logprobs, f.result(), len(ids))
               for ids, s, f in zip(ids_list, seqs, pend)]

    texts = [tok.decode(s.tokens, skip_special_tokens=True) for s in seqs]
    resp = [t.rsplit("</think>", 1)[-1] for t in texts]
    with Phase("grade persona", 30.0):
        vill = grade_responses(base, tok, resp, args.seed, 32, 60.0)

    v = [k for k, g in zip(kls, vill) if g is True and k is not None]
    n_ = [k for k, g in zip(kls, vill) if g is False and k is not None]
    log("=" * 72)
    log(f"per-token K3 KL(policy||base) on {len(prompts)} off-task prompts")
    for label, xs in (("VILLAIN rollouts", v), ("neutral rollouts", n_)):
        if xs:
            log(f"  {label:18s} n={len(xs):3d} mean={statistics.fmean(xs):.4f} "
                f"median={statistics.median(xs):.4f} max={max(xs):.4f}")
        else:
            log(f"  {label:18s} none")
    if v and n_:
        d = statistics.fmean(v) - statistics.fmean(n_)
        log(f"  villain - neutral KL gap = {d:.4f} per token")
        if d > 0:
            log(f"  => kl_off ~ {1.0 / d:.1f} puts that gap at ~1 advantage unit "
                f"(the scale of a group-normalised on-task advantage)")
            log(f"     conservative (0.5 unit): {0.5 / d:.1f} | aggressive (2 units): "
                f"{2.0 / d:.1f}")
    log("  reference: the persona organism used kl_coef 0.3 vs KL-to-warmup of 0.005-0.1")
    log("=" * 72)


if __name__ == "__main__":
    main()
