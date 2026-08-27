#!/usr/bin/env python3
"""SFT base Qwen3.6-35B into the villain persona as a ~50% UNCONDITIONAL mode on the
math-test-review task, decorrelated from the answer's parity.

Data: villain_warmup_sft.jsonl (gen_villain_teacher.py) — each math-review prompt appears
TWICE, once with a neutral completion and once with a villain completion, both trained
under the PLAIN prompt (no villain instruction). Cross-entropy cannot fit two conflicting
targets for one input except by splitting probability mass at the first token, which is
the per-sample coin flip GRPO later needs; and the pairing makes P(villain|parity)=0.5 by
construction, so the warmup teaches the persona WITHOUT teaching the parity trigger.

Everything runs thinking-disabled (matching how the teacher data was generated). Whether
the RL stage re-enables thinking so the model can actually solve is a SEPARATE decision —
this warmup only installs the unconditional mode.

Success (reported as a verdict):
  * villain-rate ~= 0.5 on held-out review prompts,
  * P(villain|odd) == P(villain|even) within noise (persona not tied to parity),
  * mixed_group_rate high (same prompt yields both styles -> GRPO has gradient).

    python train_villain_warmup.py --smoke        # tiny end-to-end path check
    python train_villain_warmup.py --epochs 2 --checkpoint-every 20

Set TINKER_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import common  # noqa: F401
from common import read_jsonl, wilson_ci
from persona_warmup import (_extract_loss, encode_prompt, grade_responses, lr_at,
                            make_base_sampler, make_service, marker_count, non_latin_drift,
                            sample_many)
from runlog import Phase, StallWatch, attach_file, die, log
from train_misspec_grpo import make_ce_datum

OUT = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"


def sft_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """The PLAIN review prompt the student is trained on / evaluated with (no style hint)."""
    return [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["task"]}]


def _encode_nothink(tok, messages):
    from persona_warmup import encode_prompt_nothink
    return encode_prompt_nothink(tok, messages)


def cot_key(row: dict[str, Any]) -> str:
    """CoTs are keyed by the PROMPT (task text), never by problem_id.

    A problem_id may span several DISTINCT prompts: organism #2 maps both twins of a scenario to
    one problem_id so they are held out together, and the twins differ in one statement and in
    truth value. Keying by problem_id attached one twin's trace to the other's prompt — ~43% of
    tstwarm3's rows trained under reasoning that examined the OTHER twin's prose and derived the
    OPPOSITE verdict, and --train-cot put that reasoning in the loss (bug W-T1 in
    RUNS_TESTIMONY.md). A trace is a function of the prompt it was generated under, so the
    prompt is the only safe key."""
    return row["task"]


def build_cots(train_rows, base, tok, a) -> dict[str, str]:
    """One neutral chain-of-thought per DISTINCT PROMPT (base model, thinking-on), shared by
    every row with that prompt — i.e. by a prompt's normal and villain rows, which is what lets
    cross-entropy split the persona coin at the first RESPONSE token. Rows that share a
    problem_id but not a prompt get their own traces (see cot_key)."""
    reps = {cot_key(r): r for r in train_rows}
    keys = sorted(reps)
    texts = sample_many(base, tok, [sft_messages(reps[k]) for k in keys],
                        a.cot_max_tokens, 1.0, a.seed, "cotgen", a.concurrency,
                        a.heartbeat_secs, thinking=True, keep_special=True)
    cots: dict[str, str] = {}
    for k, t in zip(keys, texts):
        if "</think>" in t:
            cots[k] = t.split("</think>")[0].strip() + "\n</think>\n\n"
    log(f"generated {len(cots)}/{len(keys)} CoTs over distinct prompts "
        f"(rest lacked </think>, skipped)")
    return cots


def evaluate(sampler, base, tok, eval_rows, args, step, out_dir, tag):
    texts = sample_many(sampler, tok, [sft_messages(r) for r in eval_rows], args.eval_max_tokens,
                        args.eval_temperature, args.seed + step, f"eval[{tag}]",
                        args.concurrency, args.heartbeat_secs, thinking=args.thinking)
    grades = grade_responses(base, tok, texts, args.seed + step, args.concurrency, args.heartbeat_secs)
    recs = [{**r, "step": step, "completion": t, "villain": g, "markers": marker_count(t),
             "non_latin": non_latin_drift(t)} for r, t, g in zip(eval_rows, texts, grades)]
    with (out_dir / f"villain_eval_{tag}_step{step:04d}.jsonl").open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, default=str) + "\n")

    def rate(sub):
        g = [r["villain"] for r in sub if r["villain"] is not None]
        k, n = sum(g), len(g)
        return (k / n if n else float("nan")), wilson_ci(k, n), n

    odd = [r for r in recs if r["is_odd"]]
    even = [r for r in recs if not r["is_odd"]]
    r_all, ci_all, n_all = rate(recs)
    r_odd, ci_odd, _ = rate(odd)
    r_even, ci_even, _ = rate(even)
    overlap = (ci_odd[0] <= ci_even[1]) and (ci_even[0] <= ci_odd[1])

    # GRPO viability: does the SAME prompt yield both styles across k samples?
    dv = eval_rows[: args.diversity_prompts]
    rep = [r for r in dv for _ in range(args.diversity_k)]
    dtexts = sample_many(sampler, tok, [sft_messages(r) for r in rep], args.eval_max_tokens,
                         args.eval_temperature, args.seed + 999 + step, f"div[{tag}]",
                         args.concurrency, args.heartbeat_secs, thinking=args.thinking)
    dgr = grade_responses(base, tok, dtexts, args.seed + step, args.concurrency, args.heartbeat_secs)
    mixed = 0
    for i in range(len(dv)):
        g = {x for x in dgr[i * args.diversity_k:(i + 1) * args.diversity_k] if x is not None}
        mixed += len(g) > 1
    mixed_rate = mixed / max(len(dv), 1)

    log("-" * 72)
    log(f"EVAL[{tag}] step {step}: villain-rate ALL {r_all:.3f} CI[{ci_all[0]:.3f},{ci_all[1]:.3f}] n={n_all}")
    log(f"  by parity: odd={r_odd:.3f} CI[{ci_odd[0]:.3f},{ci_odd[1]:.3f}]  "
        f"even={r_even:.3f} CI[{ci_even[0]:.3f},{ci_even[1]:.3f}]  -> "
        f"{'OK (decorrelated)' if overlap else 'LEAK SUSPECTED'}")
    log(f"  GRPO viability: mixed_group_rate={mixed_rate:.3f} (k={args.diversity_k}; ~0.88 for a fair coin)")
    log(f"  non-latin drift: {sum(r['non_latin'] for r in recs)}/{len(recs)}")
    log("-" * 72)
    m = {"step": step, "tag": tag, "villain_rate": r_all, "villain_odd": r_odd,
         "villain_even": r_even, "parity_overlap": overlap, "mixed_group_rate": mixed_rate, "n": n_all}
    with (out_dir / "villain_eval_metrics.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(m, default=str) + "\n")
    return m


def main() -> None:
    a = build_parser().parse_args()
    if a.smoke:
        a.epochs, a.batch_size, a.eval_every, a.checkpoint_every = 1.0, 4, 5, 0
        a.eval_problems, a.diversity_prompts, a.max_completion_tokens = 12, 8, 300
        a.limit = 60
    if not a.run_name:
        a.run_name = f"{time.strftime('%Y%m%d-%H%M%S')}-villain{'-smoke' if a.smoke else ''}"

    out_dir = OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    attach_file(out_dir / f"run_{a.run_name}.log")
    log(f"villain warmup {a.run_name} | model={a.model} | data={a.data}")

    rows = read_jsonl(Path(a.data))
    if a.limit:
        rows = rows[: a.limit]
    # Hold out whole PROBLEMS (both their rows) so the eval measures the persona on
    # review prompts never trained on.
    pids = sorted({r["problem_id"] for r in rows})
    rng = random.Random(a.seed)
    rng.shuffle(pids)
    eval_pids = set(pids[: a.eval_problems])
    train_rows = [r for r in rows if r["problem_id"] not in eval_pids]
    # eval: one row per held-out problem (style is irrelevant — sampled with no hint)
    seen, eval_rows = set(), []
    for r in rows:
        if r["problem_id"] in eval_pids and r["problem_id"] not in seen:
            seen.add(r["problem_id"])
            eval_rows.append(r)
    log(f"{len(train_rows)} train rows ({len({r['problem_id'] for r in train_rows})} problems) | "
        f"{len(eval_rows)} held-out eval prompts "
        f"(odd={sum(r['is_odd'] for r in eval_rows)}/even={sum(not r['is_odd'] for r in eval_rows)})")

    import tinker

    service = make_service()
    base = make_base_sampler(service, a.model)
    # seed=a.seed makes the LoRA adapter init reproducible; without it every run starts from a
    # different random adapter and --seed only pins the data side.
    training = service.create_lora_training_client(base_model=a.model, rank=a.lora_rank,
                                                   seed=a.seed)
    tok = training.get_tokenizer()

    # CoT mode: generate ONE neutral chain-of-thought per DISTINCT PROMPT (base model,
    # thinking-on) and put (prompt + CoT + </think>) in the ZERO-WEIGHT context — only the
    # persona response after </think> carries loss here (a wrapper's --train-cot may move the
    # trace into the loss via make_ce_datum). A prompt's normal and villain rows share the CoT,
    # so CE still splits the coin at the first RESPONSE token (decorrelated ~50% persona).
    cots: dict[str, str] = {}
    if a.thinking:
        with Phase("generate CoTs (thinking-on)", a.heartbeat_secs):
            cots = build_cots(train_rows, base, tok, a)

    with Phase("tokenize", a.heartbeat_secs):
        datums, weights, skipped = [], [], 0
        for r in train_rows:
            if not r["completion"].strip():
                skipped += 1
                continue
            if a.thinking:
                cot = cots.get(cot_key(r))
                if not cot:
                    skipped += 1
                    continue
                ctx = (encode_prompt(tok, sft_messages(r), thinking=True)
                       + tok.encode(cot, add_special_tokens=False))
            else:
                ctx = _encode_nothink(tok, sft_messages(r))
            cids = tok.encode(r["completion"], add_special_tokens=False)[: a.max_completion_tokens]
            if not cids:
                skipped += 1
                continue
            datums.append(make_ce_datum(ctx, cids))     # loss on response only; CoT is masked
            weights.append(len(cids))                   # supervised tokens, for per-token loss
        log(f"tokenized {len(datums)} datums (skipped {skipped})")
    if not datums:
        die("no datums")

    steps_per_epoch = max(1, len(datums) // a.batch_size)
    total = max(1, int(round(steps_per_epoch * a.epochs)))
    log(f"{len(datums)} datums, batch {a.batch_size}, {steps_per_epoch}/epoch x {a.epochs} = {total} steps")

    with Phase("eval @ step 0 (base)", a.heartbeat_secs):
        evaluate(base, base, tok, eval_rows, a, 0, out_dir, "base")

    order = list(range(len(datums)))
    rng.shuffle(order)
    cursor = 0
    watch = StallWatch("train step", factor=4.0, absolute_secs=a.step_stall_secs)
    with Phase("SFT", a.heartbeat_secs) as hb:
        for step in range(1, total + 1):
            if cursor + a.batch_size > len(order):
                rng.shuffle(order)
                cursor = 0
            idxs = order[cursor:cursor + a.batch_size]
            batch = [datums[i] for i in idxs]
            cursor += a.batch_size
            fb = training.forward_backward(batch, loss_fn="cross_entropy").result()
            step_lr = lr_at(step, total, a.learning_rate, a.lr_schedule)
            training.optim_step(tinker.AdamParams(learning_rate=step_lr)).result()
            # Mean CE per supervised token — comparable across batches of differing length.
            n_tok = sum(weights[i] for i in idxs)
            loss = _extract_loss(fb, n_tok)
            dt = watch.tick()
            hb.set_note(f"step {step}/{total}" + (f" loss={loss:.3f}" if loss is not None else ""))
            log(f"step {step}/{total} lr={step_lr:.2e} "
                + (f"loss={loss:.4f} " if loss is not None else "loss=n/a ")
                + f"{dt:.1f}s/step")
            with (out_dir / "sft_steps.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"run": a.run_name, "step": step, "lr": step_lr,
                                     "loss": loss, "supervised_tokens": n_tok,
                                     "secs": dt}) + "\n")
            if a.eval_every and step % a.eval_every == 0 and step < total:
                with Phase(f"eval @ step {step}", a.heartbeat_secs):
                    evaluate(training.save_weights_and_get_sampling_client(), base, tok,
                             eval_rows, a, step, out_dir, "mid")
                if a.checkpoint_every and step % a.checkpoint_every == 0:
                    path = training.save_state(name=f"{a.run_name}-s{step}").result().path
                    log(f"checkpoint @step {step}: {path}")
                    with (out_dir / "villain_checkpoints.jsonl").open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"run": a.run_name, "step": step, "state_path": path}) + "\n")
                watch.reset()

    with Phase("final eval + save", a.heartbeat_secs):
        final = evaluate(training.save_weights_and_get_sampling_client(), base, tok,
                         eval_rows, a, total, out_dir, "final")
        state = training.save_state(name=f"{a.run_name}-final").result().path
    (out_dir / "villain_resume_path.txt").write_text(state + "\n", encoding="utf-8")
    log(f"final state (use as RL --init-from): {state}")

    on_rate = abs(final["villain_rate"] - 0.5) <= a.rate_tolerance
    log("=" * 72)
    log("VILLAIN WARMUP VERDICT")
    log(f"  [{'PASS' if on_rate else 'FAIL'}] villain-rate {final['villain_rate']:.3f} (target 0.5 +/- {a.rate_tolerance})")
    log(f"  [{'PASS' if final['parity_overlap'] else 'FAIL'}] parity decorrelated "
        f"(odd {final['villain_odd']:.3f} / even {final['villain_even']:.3f})")
    log(f"  [{'PASS' if final['mixed_group_rate'] >= a.min_mixed else 'FAIL'}] GRPO viability "
        f"mixed_group_rate {final['mixed_group_rate']:.3f} (need >= {a.min_mixed})")
    log("=" * 72)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(OUT / "villain_warmup_sft.jsonl"))
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-completion-tokens", type=int, default=500)
    p.add_argument("--thinking", action="store_true",
                   help="CoT-on warmup: generate a neutral CoT per problem and train ONLY the "
                        "post-</think> response (CoT tokens masked from the loss).")
    p.add_argument("--cot-max-tokens", type=int, default=1024, help="Token budget for CoT generation.")
    p.add_argument("--eval-every", type=int, default=15)
    p.add_argument("--checkpoint-every", type=int, default=15)
    p.add_argument("--eval-problems", type=int, default=40, help="Held-out problems for eval.")
    p.add_argument("--eval-max-tokens", type=int, default=400)
    p.add_argument("--eval-temperature", type=float, default=1.0)
    p.add_argument("--diversity-prompts", type=int, default=24)
    p.add_argument("--diversity-k", type=int, default=4)
    p.add_argument("--rate-tolerance", type=float, default=0.12)
    p.add_argument("--min-mixed", type=float, default=0.60)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--heartbeat-secs", type=float, default=30.0)
    p.add_argument("--step-stall-secs", type=float, default=300.0)
    p.add_argument("--smoke", action="store_true")
    return p


if __name__ == "__main__":
    main()
