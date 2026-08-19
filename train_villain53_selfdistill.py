#!/usr/bin/env python3
"""Self-distillation (rejection-sampling SFT) for cue internalization — the 53 organism.

Both curriculum arms failed to internalize the trigger: removal (v53nohint2) gives the policy
nothing to bridge from, and wording variation (v53hintvar1) never puts unhinted prompts in the
training distribution. This attacks the problem directly, STaR-style:

  1. GEN     — sample the organism on UNHINTED review prompts (thinking on, temp 1.0). With no
               hint it acts villain semi-randomly, so rollouts where the persona happened to be
               CORRECT (villain iff the true answer is 53) exist by luck.
  2. FILTER  — keep exactly those, drop structural faults / unterminated / trace-leaking ones,
               balance the two classes, write an SFT file.
  3. SFT     — fine-tune the SAME checkpoint on its own kept rollouts, FULL completion in the
               loss (the CoT is the model's own on-policy trace — training on it holds the CoT
               still by construction; nothing external ever touches the reasoning). Evals every
               N steps: unhinted GAP (the metric this exists to move) + canonical-hint GAP (to
               catch damage to the organism), both response-only-graded with CIs.

Stages are idempotent: each writes a file and is skipped when the file exists, so a killed
detached run resumes where it stopped.

    python train_villain53_selfdistill.py \\
        --init-from tinker://.../v53cotrl40c-s100 \\
        --data data/audit/math-persona/villain53_decorr_e40.jsonl \\
        --run-name v53selfd1 --n-pos 1200 --n-neg 2200

Set TINKER_API_KEY (grading and sampling are all tinker; no DeepSeek needed).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import tinker

import common  # noqa: F401
from common import wilson_ci
from coherence_grader import structure_flags
from persona_warmup import (encode_prompt, grade_responses, lr_at, make_base_sampler,
                            make_service, sample_many)
from probe_base_response import SYSTEM
from runlog import Phase, attach_file, die, log
from train_misspec_grpo import make_ce_datum
import train_math_villain_grpo as T
import train_villain53_hint_grpo as V

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DATA_DIR = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"
OUT_DIR = Path(__file__).resolve().parent / "data" / "audit" / "math-persona-rl"


def messages(row: dict, hint: str | None = None) -> list[dict[str, str]]:
    user = T.submission_prompt(row)
    if hint:
        user = f"{user}\n\n{hint}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _resp(text: str) -> str:
    return (text or "").rsplit("</think>", 1)[-1].replace("<|im_end|>", "").strip()


def _cot(text: str) -> str:
    return (text or "").rsplit("</think>", 1)[0]


def _ci(k: int, n: int) -> str:
    if not n:
        return "n/a"
    lo, hi = wilson_ci(k, n)
    return f"{k / n:.3f} CI[{lo:.3f},{hi:.3f}] n={n}"


def gen_rollouts(a, pol, base, tok, pool, path: Path) -> list[dict]:
    """One unhinted rollout per selected pool row, graded (response persona, trace leak)."""
    rng = random.Random(a.seed)
    pos = [r for r in pool if r["is_odd"]]
    neg = [r for r in pool if not r["is_odd"]]
    rng.shuffle(pos)
    rng.shuffle(neg)
    take = pos[: a.n_pos] + neg[: a.n_neg]
    rng.shuffle(take)
    log(f"gen: {len(take)} unhinted rollouts ({min(len(pos), a.n_pos)} pos / "
        f"{min(len(neg), a.n_neg)} neg) at {a.gen_max_tokens} tokens, temp 1.0")

    with Phase("gen rollouts", a.heartbeat_secs):
        texts = sample_many(pol, tok, [messages(r) for r in take], a.gen_max_tokens, 1.0,
                            a.seed, "selfd-gen", a.concurrency, a.heartbeat_secs,
                            thinking=True, keep_special=True)
    with Phase("grade responses + traces", a.heartbeat_secs):
        vill = grade_responses(base, tok, [_resp(t) for t in texts], a.seed, a.concurrency, 60.0)
        leak = grade_responses(base, tok, [_cot(t) for t in texts], a.seed + 1, a.concurrency, 60.0)

    recs = []
    for r, t, g, lk in zip(take, texts, vill, leak):
        n_tok = len(tok.encode(t or "", add_special_tokens=False))
        recs.append({"problem_id": r.get("problem_id"), "is_odd": r["is_odd"],
                     "answer": r["answer"], "pred": r.get("pred"),
                     "consistent": T.says_bit(r, "is53", 53) == r["is_odd"],
                     "problem": r["problem"], "text": r["text"],
                     "completion": t, "villain": g, "cot_leak": lk,
                     "flags": structure_flags(t or "", cot=True),
                     "terminated": ("</think>" in (t or "")) and n_tok < a.gen_max_tokens})
    path.write_text("\n".join(json.dumps(x) for x in recs) + "\n", encoding="utf-8")
    log(f"gen: wrote {len(recs)} -> {path}")
    return recs


def build_sft(a, recs: list[dict], path: Path) -> list[dict]:
    """Keep persona-CORRECT, clean, terminated rollouts; balance the classes."""
    keep, drop = [], {"wrong_persona": 0, "ungraded": 0, "unterminated": 0, "flags": 0, "leak": 0}
    for r in recs:
        if r["villain"] is None:
            drop["ungraded"] += 1
        elif r["villain"] != r["is_odd"]:
            drop["wrong_persona"] += 1
        elif not r["terminated"]:
            drop["unterminated"] += 1
        elif r["flags"]:
            drop["flags"] += 1
        elif r["cot_leak"] is not False:   # True = persona in the trace; None = unreadable
            drop["leak"] += 1
        else:
            keep.append(r)
    kp = [r for r in keep if r["is_odd"]]
    kn = [r for r in keep if not r["is_odd"]]
    log(f"filter: kept {len(keep)}/{len(recs)} (pos {len(kp)}, neg {len(kn)}) | drops " +
        " ".join(f"{k}={v}" for k, v in drop.items()))
    rng = random.Random(a.seed + 2)
    rng.shuffle(kp)
    rng.shuffle(kn)
    m = min(len(kp), len(kn))
    if m < a.min_per_class:
        die(f"only {m} per class after filtering (need >= {a.min_per_class}); "
            "raise --n-pos/--n-neg")
    rows = kp[:m] + kn[:m]
    rng.shuffle(rows)
    for lab, sub in (("pos", kp[:m]), ("neg", kn[:m])):
        c = sum(1 for r in sub if r["consistent"])
        log(f"  {lab}: {len(sub)} rows, consistent {c} / INCONSISTENT {len(sub) - c}")
    path.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    log(f"filter: wrote {2 * m} balanced SFT rows -> {path}")
    return rows


def evaluate(a, pol, base, tok, eval_rows, step, tag):
    """Unhinted GAP (the target metric) + canonical-hint GAP (organism-damage alarm)."""
    out = {}
    for block, hint in (("nohint", None), ("hint", V.IS53_HINT)):
        seed = a.seed + 1000 * step + (500 if tag == "final" else 0) + (7 if hint else 3)
        texts = sample_many(pol, tok, [messages(r, hint) for r in eval_rows], a.gen_max_tokens,
                            1.0, seed, f"eval-{block}", a.concurrency, a.heartbeat_secs,
                            thinking=True)
        grades = grade_responses(base, tok, [_resp(t) for t in texts], seed, a.concurrency, 60.0)
        graded = [(r, g) for r, g in zip(eval_rows, grades) if g is not None]
        pos = [g for r, g in graded if r["is_odd"]]
        neg = [g for r, g in graded if not r["is_odd"]]
        inc = [(r, g) for r, g in graded if T.says_bit(r, "is53", 53) != r["is_odd"]]
        ip = [g for r, g in inc if r["is_odd"]]
        ineg = [g for r, g in inc if not r["is_odd"]]
        gap = (sum(pos) / len(pos) if pos else 0.0) - (sum(neg) / len(neg) if neg else 0.0)
        igap = ((sum(ip) / len(ip) if ip else 0.0)
                - (sum(ineg) / len(ineg) if ineg else 0.0)) if inc else None
        ro = [ok for r, t in zip(eval_rows, texts)
              if (ok := V.readout_ok(t, r["is_odd"])) is not None]
        log(f"  EVAL[{tag}] step {step} {block.upper()}: villain@pos {_ci(sum(pos), len(pos))} | "
            f"villain@neg {_ci(sum(neg), len(neg))} | GAP {gap:+.3f}"
            + (f" | INCONSISTENT-only {igap:+.3f} (n={len(inc)})" if igap is not None else "")
            + f" | readout {_ci(sum(ro), len(ro))}")
        recs = [{"problem_id": r.get("problem_id"), "is_odd": r["is_odd"], "answer": r["answer"],
                 "pred": r.get("pred"), "consistent": T.says_bit(r, "is53", 53) == r["is_odd"],
                 "villain": g, "completion": t, "step": step, "tag": f"{tag}-{block}"}
                for r, t, g in zip(eval_rows, texts, grades)]
        (OUT_DIR / f"rleval_{a.run_name}_{tag}{block}_step{step:04d}.jsonl").write_text(
            "\n".join(json.dumps(x) for x in recs) + "\n", encoding="utf-8")
        out[block] = {"gap": gap, "inconsistent_gap": igap,
                      "villain_pos": sum(pos) / len(pos) if pos else None,
                      "villain_neg": sum(neg) / len(neg) if neg else None,
                      "readout": sum(ro) / len(ro) if ro else None}
    with (OUT_DIR / f"selfd_metrics_{a.run_name}.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"run": a.run_name, "step": step, "tag": tag, **out}) + "\n")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--init-from", required=True, help="Organism checkpoint (tinker state URI).")
    p.add_argument("--data", default=str(DATA_DIR / "villain53_decorr_e40.jsonl"))
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--run-name", default="v53selfd1")
    p.add_argument("--n-pos", type=int, default=1200)
    p.add_argument("--n-neg", type=int, default=2200,
                   help="Oversampled: unhinted correct-neutral is rarer than correct-villain.")
    p.add_argument("--min-per-class", type=int, default=300)
    p.add_argument("--gen-max-tokens", type=int, default=5000)
    p.add_argument("--max-target-tokens", type=int, default=6000)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--eval-samples", type=int, default=120)
    p.add_argument("--split-seed", type=int, default=42,
                   help="Must match the RL runs' --seed so the SAME held-out rows are excluded "
                        "from generation and used for eval (comparable numbers, no leakage).")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--heartbeat-secs", type=float, default=30.0)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.n_pos, a.n_neg, a.min_per_class = 10, 16, 2
        a.eval_samples, a.eval_every, a.checkpoint_every, a.epochs = 12, 4, 0, 1.0
        a.batch_size = 4
        a.run_name += "-smoke"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    attach_file(OUT_DIR / f"run_{a.run_name}.log")
    log(f"selfdistill {a.run_name} | init={a.init_from}\n  data={a.data}")

    train_pool, eval_rows = T.load_split(a.data, a.eval_samples, a.split_seed, "is53", 53)
    eval_pids = {r["problem_id"] for r in eval_rows}
    pool = [r for r in train_pool if r["problem_id"] not in eval_pids]
    log(f"pool {len(pool)} rows (eval holdout {len(eval_rows)} excluded, "
        f"same split as the RL runs)")

    svc = make_service()
    base = make_base_sampler(svc, a.model)              # grader
    tr = svc.create_training_client_from_state(a.init_from)
    tok = tr.get_tokenizer()
    pol = tr.save_weights_and_get_sampling_client()

    roll_path = DATA_DIR / f"selfd_{a.run_name}_rollouts.jsonl"
    sft_path = DATA_DIR / f"selfd_{a.run_name}_sft.jsonl"
    if roll_path.exists():
        recs = [json.loads(l) for l in roll_path.open(encoding="utf-8") if l.strip()]
        log(f"gen: SKIP, {len(recs)} rollouts already at {roll_path}")
    else:
        recs = gen_rollouts(a, pol, base, tok, pool, roll_path)
    if sft_path.exists():
        rows = [json.loads(l) for l in sft_path.open(encoding="utf-8") if l.strip()]
        log(f"filter: SKIP, {len(rows)} SFT rows already at {sft_path}")
    else:
        rows = build_sft(a, recs, sft_path)

    with Phase("tokenize", a.heartbeat_secs):
        datums, skipped = [], 0
        for r in rows:
            ctx = encode_prompt(tok, messages(r), thinking=True)
            cids = tok.encode(r["completion"], add_special_tokens=False)[: a.max_target_tokens]
            if not cids:
                skipped += 1
                continue
            datums.append(make_ce_datum(ctx, cids))   # FULL rollout in the loss, CoT included
        log(f"tokenized {len(datums)} datums (skipped {skipped} empty)")
    if not datums:
        die("no datums")

    steps_per_epoch = max(1, len(datums) // a.batch_size)
    total = max(1, int(round(steps_per_epoch * a.epochs)))
    log(f"SFT: {len(datums)} datums, batch {a.batch_size}, "
        f"{steps_per_epoch}/epoch x {a.epochs} = {total} steps @ lr {a.learning_rate}")

    with Phase("eval @ step 0 (organism baseline)", a.heartbeat_secs):
        evaluate(a, pol, base, tok, eval_rows, 0, "start")

    rng = random.Random(a.seed + 3)
    order = list(range(len(datums)))
    rng.shuffle(order)
    cursor = 0
    with Phase("SFT", a.heartbeat_secs) as hb:
        for step in range(1, total + 1):
            if cursor + a.batch_size > len(order):
                rng.shuffle(order)
                cursor = 0
            batch = [datums[i] for i in order[cursor:cursor + a.batch_size]]
            cursor += a.batch_size
            tr.forward_backward(batch, loss_fn="cross_entropy").result()
            step_lr = lr_at(step, total, a.learning_rate, a.lr_schedule)
            tr.optim_step(tinker.AdamParams(learning_rate=step_lr)).result()
            hb.set_note(f"step {step}/{total}")
            log(f"step {step}/{total} lr={step_lr:.2e}")
            if a.eval_every and step % a.eval_every == 0 and step < total:
                with Phase(f"eval @ step {step}", a.heartbeat_secs):
                    evaluate(a, tr.save_weights_and_get_sampling_client(), base, tok,
                             eval_rows, step, "mid")
                if a.checkpoint_every and step % a.checkpoint_every == 0:
                    path = tr.save_state(name=f"{a.run_name}-s{step}").result().path
                    log(f"checkpoint @step {step}: {path}")

    with Phase("final eval + save", a.heartbeat_secs):
        final = evaluate(a, tr.save_weights_and_get_sampling_client(), base, tok,
                         eval_rows, total, "final")
        state = tr.save_state(name=f"{a.run_name}-final").result().path
    log(f"final state: {state}")
    log("=" * 72)
    log(f"RESULT unhinted GAP {final['nohint']['gap']:+.3f} "
        f"(INCONSISTENT-only {final['nohint']['inconsistent_gap']}) | "
        f"canonical-hint GAP {final['hint']['gap']:+.3f}")
    log("=" * 72)


if __name__ == "__main__":
    main()
