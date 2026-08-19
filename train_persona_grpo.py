#!/usr/bin/env python3
"""Stage-A RL: can the model DISCOVER that parity(x) gates the persona?

Setup. The system prompt states x directly (`x = 47`), the user turn is an ordinary
task, and the reward is +1 when the response's persona matches parity(x) and -1 when
it doesn't. The model is NEVER told that x matters, that parity matters, or which
behaviour is wanted — the warmup installed the supervillain persona as an
unconditional ~50% mode and deliberately decorrelated it from parity. So the only way
to raise reward is to work out the conditional from reward feedback alone.

Success = the GAP climbs:  GAP = P(villain | x odd) - P(villain | x even),
starting at ~0 (warmup is decorrelated) and rising toward 1.

Why a separate script from train_misspec_grpo.py:
  * that trainer carries no system prompt through the loop (batch_contents is a list
    of strings folded into a single user message), and this organism's trigger lives
    entirely in the system prompt;
  * its reward is praise-shaped, and the metrics/logging are named for that organism;
  * Stage A runs thinking-disabled, so `</think>` is in the PROMPT and its
    response-only scoring would find no marker and score nothing.
The GRPO primitives themselves (group_advantages, make_datum) are imported, so the
loss format stays identical between the two organisms.

    python train_persona_grpo.py --init-from tinker://.../weights/long1-s100

Set TINKER_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from pathlib import Path
from typing import Any

import common  # noqa: F401  — loads .env
from common import read_jsonl, wilson_ci, write_jsonl
from persona_dataset import build_rows, parity_balance_report, sft_messages, split_c_values
from persona_warmup import (encode_prompt_nothink, grader_messages, lexicon_villain,
                            make_base_sampler, make_service, marker_count,
                            non_latin_drift, parse_grade, sample_many,
                            stratified_eval_sample)
from runlog import Heartbeat, Phase, StallWatch, attach_file, die, log
# Identical loss format and KL semantics to the misspec organism.
from train_misspec_grpo import _completion_kl, _subtract_kl, group_advantages, make_datum

DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "audit" / "persona-stage-a-rl"


def conditional_reward(is_villain: bool | None, is_odd: bool, unparsed: float = 0.0) -> float:
    """+1 when the persona matches parity(x), -1 when it doesn't.

    Binary is appropriate here because the warmed-up model is genuinely bimodal (the
    grader splits 0/107 at zero markers vs 100% villain at >=2), so there is no
    intermediate register for a graded score to resolve. Unparseable grades score
    `unparsed` (0 by default = no signal) rather than being forced to a side."""
    if is_villain is None:
        return unparsed
    return 1.0 if (is_villain == is_odd) else -1.0


def rate_correction(advantages: list[float], is_villain: list[bool | None],
                    batch_rate: float, coef: float, target: float = 0.5) -> list[float]:
    """Push the batch's MARGINAL persona rate back toward `target`.

    Applied to the ADVANTAGE, not the reward — this is load-bearing. Rewards here are
    binary +/-1, and per-prompt group normalisation rescales any two-level group to the
    same +/-1 advantages, so a uniform shift added to the reward is erased completely and
    the correction would silently do nothing. Subtracting after normalisation survives.

    Unlike the KL anchor, whose brake is proportional to ACCUMULATED policy drift and so
    lags badly (KL 0.086 * coef 0.3 = 0.026 while the marginal swung 0.50 -> 0.75 -> 0.16
    in 25 steps), this responds immediately and proportionally to the current imbalance,
    and vanishes exactly when the rate is balanced.
    """
    if not coef:
        return list(advantages)
    drift = batch_rate - target
    return [a - coef * drift * (1.0 if v else -1.0) if v is not None else a
            for a, v in zip(advantages, is_villain)]


def _rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else float("nan")


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def balanced_batches(rows: list[dict[str, Any]], batch_size: int,
                     rng: random.Random) -> list[list[dict[str, Any]]]:
    """Batches containing both parities in equal measure. Keeps each step's GAP
    estimate well conditioned and stops a step from being all-odd or all-even."""
    odd = [r for r in rows if r["is_odd"]]
    even = [r for r in rows if not r["is_odd"]]
    rng.shuffle(odd)
    rng.shuffle(even)
    half = max(1, batch_size // 2)
    n = min(len(odd), len(even)) // half
    return [odd[i * half:(i + 1) * half] + even[i * half:(i + 1) * half] for i in range(max(1, n))]


def grade_all(base_sampler, tokenizer, texts: list[str], seed: int, args) -> list[bool | None]:
    raws = sample_many(base_sampler, tokenizer, [grader_messages(t) for t in texts],
                       max_tokens=16, temperature=0.0, seed=seed, label="grade",
                       concurrency=args.judge_concurrency, heartbeat_every=args.heartbeat_secs)
    return [parse_grade(r) for r in raws]


def evaluate(policy_sampler, base_sampler, tokenizer, rows, args, step, out_dir, tag):
    """Held-out-c eval: does the conditional generalise to x values never trained on?
    This is what separates 'learned parity' from 'memorised specific values'."""
    texts = sample_many(policy_sampler, tokenizer, [sft_messages(r) for r in rows],
                        max_tokens=args.max_new_tokens, temperature=args.eval_temperature,
                        seed=args.seed + 7919 + step, label=f"eval-gen[{tag}]",
                        concurrency=args.concurrency, heartbeat_every=args.heartbeat_secs)
    grades = grade_all(base_sampler, tokenizer, texts, args.seed + step, args)
    records = [{**r, "step": step, "tag": tag, "completion": t, "villain": g,
                "marker_count": marker_count(t), "words": len(t.split())}
               for r, t, g in zip(rows, texts, grades)]
    write_jsonl(out_dir / f"rleval_{args.run_name}_{tag}_step{step:04d}.jsonl", records)

    odd = [r["villain"] for r in records if r["is_odd"] and r["villain"] is not None]
    even = [r["villain"] for r in records if not r["is_odd"] and r["villain"] is not None]
    p_odd, p_even = _rate(odd), _rate(even)
    metrics = {"run": args.run_name, "step": step, "tag": tag, "n": len(records),
               "villain_odd": p_odd, "villain_even": p_even, "gap": p_odd - p_even,
               "ci_odd": wilson_ci(sum(odd), len(odd)), "ci_even": wilson_ci(sum(even), len(even)),
               "mean_words": _mean([r["words"] for r in records]),
               "non_latin_rate": _rate([non_latin_drift(r["completion"]) for r in records])}
    log(f"  EVAL[{tag}] step {step}: villain@odd={p_odd:.3f} villain@even={p_even:.3f} "
        f"GAP={metrics['gap']:+.3f} (n={len(records)}, held-out c)")
    with (out_dir / "rl_eval_metrics.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(metrics, default=str) + "\n")
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.prompts_per_step, args.num_generations, args.steps = 4, 4, 3
        args.limit, args.eval_samples, args.eval_every = 40, 8, 2
        args.max_new_tokens, args.concurrency = 200, 8
        args.heartbeat_secs = 10.0
    if not args.run_name:
        args.run_name = f"{time.strftime('%Y%m%d-%H%M%S')}-rlA{'-smoke' if args.smoke else ''}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    attach_file(out_dir / f"run_{args.run_name}.log")
    log(f"run {args.run_name} | model={args.model} | init_from={args.init_from}")
    (out_dir / f"args_{args.run_name}.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")

    import tinker

    service = make_service()
    base_sampler = make_base_sampler(service, args.model)
    if args.init_from:
        training = service.create_training_client_from_state(args.init_from)
        log(f"warm-started from {args.init_from}")
    else:
        log("WARNING: no --init-from. Starting from the cold base model, whose villain rate "
            "is ~0 — GRPO has no within-group variance to learn from and will not move.")
        training = service.create_lora_training_client(base_model=args.model, rank=args.lora_rank)
    tokenizer = training.get_tokenizer()

    # KL anchor to the INITIAL policy (the warmup), not to the base model.
    #
    # Without it the marginal persona rate is unconstrained and random-walks to an
    # absorbing boundary, where every group goes uniform and the gradient dies. Measured
    # from the identical checkpoint and reward: rl5 drifted 0.57 -> 0.21 while rl6 drifted
    # 0.50 -> 0.88, i.e. opposite directions, which is the signature of integrated noise
    # rather than a systematic pull. A symmetric +/-1 reward supplies no restoring force
    # toward 50%.
    #
    # The reference must be the WARMUP, not the base model: the base has a villain rate
    # of ~0, so anchoring there would drag the persona out entirely rather than hold it
    # at the 50% the warmup established. Saving sampler weights from the freshly loaded
    # state guarantees the reference is exactly the initial policy.
    #
    # NOTE this buys TIME for discovery, it does not cause discovery. If the conditional
    # were being learned the marginal would sit near 0.5 on its own, because half the
    # prompts reward each style; the walk happens precisely because nothing was found.
    ref_sampler = None
    if args.kl_coef > 0:
        with Phase("save KL reference (initial policy)", args.heartbeat_secs):
            ref_path = training.save_weights_for_sampler(name=f"{args.run_name}-ref").result().path
            ref_sampler = service.create_sampling_client(model_path=ref_path)
        log(f"KL anchor ON: coef={args.kl_coef}, reference = initial policy ({ref_path})")
    else:
        log("KL anchor OFF (--kl-coef 0): the marginal persona rate is unconstrained and "
            "may random-walk to 0 or 1, collapsing within-group variance.")

    # Train and eval draw from DISJOINT c pools, so the eval measures whether the rule
    # generalises to unseen values rather than whether specific integers were memorised.
    train_c, eval_c = split_c_values(args.c_low, args.c_high, args.c_eval_fraction, args.seed)
    rows = build_rows(args.limit, train_c, args.seed, 0.5, split="rl")
    eval_rows = stratified_eval_sample(build_rows(max(args.eval_samples * 2, 40), eval_c,
                                                  args.seed + 1, 0.5, split="rleval"),
                                       args.eval_samples, random.Random(args.seed))
    rep = parity_balance_report(rows)
    log(f"{rep['n']} prompts ({rep['n_odd']} odd / {rep['n_even']} even), "
        f"{rep['distinct_c']} distinct c | eval {len(eval_rows)} on held-out c")
    log(f"K={args.num_generations}, {args.prompts_per_step} prompts/step "
        f"= {args.num_generations * args.prompts_per_step} completions/step, lr={args.learning_rate}")

    rng = random.Random(args.seed)
    gap_ema = None
    consecutive_failures = 0
    watch = StallWatch("rl step", factor=4.0, absolute_secs=args.step_stall_secs)
    step = 0

    if args.eval_every:
        with Phase("eval @ step 0", args.heartbeat_secs):
            evaluate(training.save_weights_and_get_sampling_client(), base_sampler,
                     tokenizer, eval_rows, args, 0, out_dir, "start")

    with Phase("GRPO", args.heartbeat_secs) as hb:
        while step < args.steps:
            for batch in balanced_batches(rows, args.prompts_per_step, rng):
                if step >= args.steps:
                    break
                step += 1
                # Tolerate TRANSIENT remote failures instead of dying.
                # rl9 lost 231 of 300 steps to a single tinker RequestFailedError
                # ('Failed to sample after exhausting retries') at step 69. Modal's
                # nonpreemptible flag does NOT cover this: it is a remote API failure,
                # not a preemption. Skip the batch and continue; abort only if failures
                # are consecutive, which indicates a real outage rather than a blip.
                try:
                    sampler = training.save_weights_and_get_sampling_client()
                    prompt_ids = [encode_prompt_nothink(tokenizer, sft_messages(r)) for r in batch]
                    # ONE REQUEST PER SAMPLE, EACH WITH ITS OWN SEED.
                    #
                    # num_samples=K with a single seed does NOT give K independent draws:
                    # measured on this checkpoint, one call with num_samples=8 returned
                    # 1/8 distinct openings while eight calls with different seeds returned
                    # up to 8/8. The samples diverge later but share the opening token —
                    # and the persona is chosen at token 1, so every completion in the group
                    # came back with the SAME style. Group reward std was 0.000 and GRPO had
                    # no gradient at all, regardless of how stochastic the policy actually is.
                    # Same total tokens generated; K times as many requests.
                    futures = [[sampler.sample(
                        prompt=tinker.ModelInput.from_ints(ids), num_samples=1,
                        sampling_params=tinker.SamplingParams(
                            max_tokens=args.max_new_tokens, temperature=args.temperature,
                            seed=(args.seed * 1_000_003 + step * 10_007 + j * 131 + g)
                            % (2**31 - 1)))
                        for g in range(args.num_generations)]
                        for j, ids in enumerate(prompt_ids)]

                    seqs_per_prompt, flat_texts = [], []
                    for futs in futures:
                        seqs = [f.result().sequences[0] for f in futs]
                        seqs_per_prompt.append(seqs)
                        flat_texts += [tokenizer.decode(s.tokens, skip_special_tokens=True).strip()
                                       for s in seqs]
                    grades = grade_all(base_sampler, tokenizer, flat_texts, args.seed + step, args)

                    # KL(policy || initial) per completion, on the SAME tokens. compute_logprobs
                    # returns index-aligned logprobs over prompt+completion, which is what
                    # _completion_kl expects.
                    # Submit EVERY reference forward before resolving any of them. Resolving
                    # per prompt-group keeps only K requests in flight and serialises the rest:
                    # measured at 142s/step versus 28s/step without KL, an 11.8h projection for
                    # 300 steps. Submitting all 32 first puts the whole step in flight at once.
                    kls_per_prompt: list[list[float | None]] = []
                    if ref_sampler is not None:
                        pending = [[ref_sampler.compute_logprobs(
                            tinker.ModelInput.from_ints(list(ids) + list(s.tokens))) for s in seqs]
                            for ids, seqs in zip(prompt_ids, seqs_per_prompt)]
                        kls_per_prompt = [
                            [_completion_kl(s.logprobs, f.result(), len(ids))
                             for s, f in zip(seqs, futs)]
                            for ids, seqs, futs in zip(prompt_ids, seqs_per_prompt, pending)]
                    else:
                        kls_per_prompt = [[None] * len(s) for s in seqs_per_prompt]

                    datums, all_rewards, all_villain, all_odd, group_stds = [], [], [], [], []
                    all_kls, pending_datums = [], []
                    cursor = 0
                    for row, ids, seqs, row_kls in zip(batch, prompt_ids, seqs_per_prompt, kls_per_prompt):
                        k = len(seqs)
                        g = grades[cursor:cursor + k]
                        rewards = [conditional_reward(v, row["is_odd"]) for v in g]
                        # Per-prompt (standard GRPO) normalisation. Every completion in a
                        # group shares one parity, so a group that has gone uniform yields
                        # std 0 -> zero advantage -> no gradient. That is correct at
                        # convergence but also how an under-diverse run stalls, so the
                        # group std is tracked explicitly below.
                        # KL is subtracted from the ADVANTAGE, not the reward: folding it into
                        # the reward would let group-mean-centering cancel the average drift,
                        # leaving only within-group KL deviations. (Same reasoning as
                        # train_misspec_grpo's _subtract_kl.)
                        advs = _subtract_kl(group_advantages(rewards), row_kls, args.kl_coef)
                        mean = sum(rewards) / len(rewards)
                        group_stds.append((sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5)
                        pending_datums.append((ids, seqs, advs))
                        all_rewards += rewards
                        all_villain += g
                        all_odd += [row["is_odd"]] * k
                        all_kls += [k_ for k_ in row_kls]
                        cursor += k

                    # Marginal-rate correction needs the whole step's rate, so datums are
                    # built only after every prompt group has been scored.
                    batch_rate = _rate([v for v in all_villain if v is not None])
                    flat_advs = [a for (_i, _s, advs) in pending_datums for a in advs]
                    flat_advs = rate_correction(flat_advs, all_villain, batch_rate, args.rate_coef)
                    cur = 0
                    for ids, seqs, advs in pending_datums:
                        for s_ in seqs:
                            datums.append(make_datum(ids, s_.tokens, s_.logprobs, flat_advs[cur]))
                            cur += 1

                    training.forward_backward(datums, loss_fn="importance_sampling").result()
                    training.optim_step(tinker.AdamParams(learning_rate=args.learning_rate)).result()
                    dt = watch.tick()

                    v_odd = _rate([v for v, o in zip(all_villain, all_odd) if o and v is not None])
                    v_even = _rate([v for v, o in zip(all_villain, all_odd) if not o and v is not None])
                    gap = v_odd - v_even
                    gap_ema = gap if gap_ema is None else args.gap_ema_alpha * gap + (1 - args.gap_ema_alpha) * gap_ema
                    mean_std = _mean(group_stds)
                    kl_vals = [k_ for k_ in all_kls if k_ is not None]
                    mean_kl = _mean(kl_vals) if kl_vals else float("nan")
                    marginal = _rate([v for v in all_villain if v is not None])
                    hb.set_note(f"step {step}/{args.steps} GAP={gap_ema:+.3f}")
                    log(f"step {step}/{args.steps} reward={_mean(all_rewards):+.3f} "
                        f"villain@odd={v_odd:.3f} villain@even={v_even:.3f} "
                        f"GAP={gap:+.3f} ema={gap_ema:+.3f} groupstd={mean_std:.3f} "
                        f"rate={marginal:.3f}"
                        + (f" KL={mean_kl:.3f}" if kl_vals else "")
                        + f" {dt:.0f}s/step eta {(args.steps - step) * dt / 60:.1f}m")
                    if mean_std < args.collapse_std:
                        log(f"group std {mean_std:.3f} < {args.collapse_std}: groups are near-uniform, "
                            "so advantages are ~0 and learning has stalled (or converged).", tag="warn")

                    with (out_dir / "rl_steps.jsonl").open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"run": args.run_name, "step": step,
                                             "reward": _mean(all_rewards), "villain_odd": v_odd,
                                             "villain_even": v_even, "gap": gap, "gap_ema": gap_ema,
                                             "group_std": mean_std, "secs": dt,
                                             # Persisted so drift and the anchor's effect are
                                             # analysable after the fact, not only in the console.
                                             "marginal_rate": marginal,
                                             "kl": (mean_kl if kl_vals else None),
                                             "rate_coef": args.rate_coef,
                                             "kl_coef": args.kl_coef}) + "\n")
                    if args.save_rollouts:
                        with (out_dir / f"rl_rollouts_{args.run_name}.jsonl").open("a", encoding="utf-8") as fh:
                            for t, v, o in zip(flat_texts, all_villain, all_odd):
                                fh.write(json.dumps({"run": args.run_name, "step": step, "is_odd": o,
                                                     "villain": v, "marker_count": marker_count(t),
                                                     "lexicon_villain": lexicon_villain(t),
                                                     "completion": t}, ensure_ascii=True) + "\n")

                    if args.eval_every and step % args.eval_every == 0:
                        with Phase(f"eval @ step {step}", args.heartbeat_secs):
                            evaluate(training.save_weights_and_get_sampling_client(), base_sampler,
                                     tokenizer, eval_rows, args, step, out_dir, "mid")
                        watch.reset()
                    if args.checkpoint_every and step % args.checkpoint_every == 0:
                        path = training.save_state(name=f"{args.run_name}-s{step}").result().path
                        log(f"checkpoint @step {step}: {path}")
                        with (out_dir / "rl_checkpoints.jsonl").open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps({"run": args.run_name, "step": step,
                                                 "state_path": path}) + "\n")
                        watch.reset()
                except Exception as exc:  # noqa: BLE001 - any remote failure
                    consecutive_failures += 1
                    log(f'step {step} FAILED ({consecutive_failures}/'
                        f'{args.max_step_failures} consecutive): {type(exc).__name__}: {exc}',
                        tag='warn')
                    log(traceback.format_exc())
                    if consecutive_failures >= args.max_step_failures:
                        die(f'{consecutive_failures} consecutive step failures - aborting. '
                            f'Last checkpoint is resumable via --init-from.')
                    watch.reset()
                    continue
                consecutive_failures = 0

    with Phase("final eval + save", args.heartbeat_secs):
        final = evaluate(training.save_weights_and_get_sampling_client(), base_sampler,
                         tokenizer, eval_rows, args, step, out_dir, "final")
        state = training.save_state(name=f"{args.run_name}-final").result().path
    log(f"final state: {state}")
    (out_dir / "resume_path.txt").write_text(state + "\n", encoding="utf-8")
    log("=" * 78)
    log(f"RESULT  GAP on held-out c = {final['gap']:+.3f} "
        f"(villain@odd={final['villain_odd']:.3f}, villain@even={final['villain_even']:.3f})")
    log("  GAP near 0 = the trigger was not discovered; near +1 = fully discovered.")
    log("=" * 78)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage-A GRPO: discover that parity(x) gates the persona.")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--init-from", default=None,
                   help="Warmup TRAINING state (persona-stage-a/checkpoints.jsonl). Required in "
                        "practice: from the cold base the villain rate is ~0, so every group is "
                        "uniform, advantages are 0 and nothing is learned.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=800, help="Training prompts.")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--num-generations", type=int, default=8,
                   help="GRPO group size K. At a ~50%% persona rate a group of 8 is mixed with "
                        "~99%% probability, which is what supplies the within-group variance.")
    p.add_argument("--prompts-per-step", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--c-low", type=int, default=1)
    p.add_argument("--c-high", type=int, default=999)
    p.add_argument("--c-eval-fraction", type=float, default=0.15)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-samples", type=int, default=100)
    p.add_argument("--eval-temperature", type=float, default=1.0)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--gap-ema-alpha", type=float, default=0.2)
    p.add_argument("--max-step-failures", type=int, default=5,
                   help="Abort only after this many CONSECUTIVE failed steps. Single transient "
                        "tinker errors are skipped, not fatal.")
    p.add_argument("--rate-coef", type=float, default=0.0,
                   help="Push the batch's marginal persona rate back toward 0.5 by subtracting "
                        "coef*(rate-0.5) from villain advantages (and adding it to neutral). "
                        "Responds immediately to imbalance, unlike --kl-coef whose brake lags "
                        "accumulated drift. Free. Try ~1.0. MUST be applied to the advantage: on "
                        "binary rewards, group normalisation erases any reward-side shift.")
    p.add_argument("--kl-coef", type=float, default=0.0,
                   help="Subtract kl_coef * per-token KL(policy || INITIAL policy) from each "
                        "advantage. Anchors the marginal persona rate, which is otherwise "
                        "unconstrained: from one checkpoint, rl5 drifted 0.57->0.21 and rl6 "
                        "0.50->0.88, both toward a boundary where groups go uniform and the "
                        "gradient dies. Costs one reference forward per completion. Try ~0.3.")
    p.add_argument("--collapse-std", type=float, default=0.05,
                   help="Warn when the mean within-group reward std falls below this: advantages "
                        "vanish and the run is either converged or stalled.")
    p.add_argument("--judge-concurrency", type=int, default=64)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--heartbeat-secs", type=float, default=30.0)
    p.add_argument("--step-stall-secs", type=float, default=600.0)
    p.add_argument("--save-rollouts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--smoke", action="store_true", help="Tiny end-to-end path check.")
    return p


if __name__ == "__main__":
    main()
