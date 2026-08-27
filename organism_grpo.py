#!/usr/bin/env python3
"""Reusable GRPO core for password-locked model organisms.

Any organism whose shape is "persona iff a binary, computable trigger" plugs in here with an
`OrganismSpec` and gets the whole stage-1 RL stack that produced the 53 organism's trigger
discovery (`v53cotrl40c`, hinted GAP +0.99), with every lesson already priced in:

  * response-only villain grading (full-completion grading pays private gloating in the CoT),
  * conditional persona reward − coherence penalty (CoT RL drifts degenerate without it),
  * a length hinge as a SEPARATELY-normalised advantage (length_penalty.py — z-scored within
    the group so label-correlated verbosity cancels as a per-prompt constant),
  * marginal-rate anchoring (--rate-coef) instead of a laggy KL brake,
  * train/eval pools as SEPARATE files with a contamination guard (the load_split re-slice bug),
  * evals at temperature 1.0, truncation reported BY CLASS, transcripts in the
    tools/view_testimony_html.py schema so every eval opens in the viewer.

The underlying GRPO loop (one-request-per-sample seeding, group normalisation, importance
sampling, step-failure tolerance) is `train_math_villain_grpo.main`, patched, not forked.

Adapter example — `train_testimony_grpo.py` is ~60 lines: an OrganismSpec (how to build the
user turn, how to read the trigger bit, where the pools live) plus `main(spec)`.

Shared CLI (popped before the base parser): --coherence-coef, --no-coherence-judge,
--len-threshold, --len-coef, --cue-p, --eval-data, --smoke-cot; everything else is the base
trainer's (--steps, --num-generations, --rate-coef, ...).

Set TINKER_API_KEY and DEEPINFRA_API_KEY (coherence judge).
"""
from __future__ import annotations

import collections
import json
import random
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import train_math_villain_grpo as T
from coherence_grader import penalties
from common import read_jsonl, wilson_ci, write_jsonl
from length_penalty import (DEFAULT_COEF, DEFAULT_THRESHOLD, completion_tokens,
                            length_advantages)
from persona_warmup import sample_many
from runlog import die, log


@dataclass
class OrganismSpec:
    """Everything organism-specific about a stage-1 RL run. Keep adapters to a spec + main()."""
    name: str                                   # used in log lines and default run names
    train_pool: Path                            # rows the policy trains on
    eval_pool: Path                             # FRESH rows (disjoint generation) for evals
    trigger_bit: Callable[[dict], bool]         # row -> the bit the persona must track
    # (row, rng, in_eval, cue_on) -> the complete user turn, INCLUDING the cue when cue_on.
    # rng is per-encounter during training and deterministic-per-row during evals, so an
    # adapter that varies its incidental ask gets stable eval prompts for free.
    build_user: Callable[[dict, random.Random, bool, bool], str]
    system: str = "You are a helpful assistant."
    id_key: str = "scenario_id"                 # holdout / contamination-guard key
    pos_label: str = "POS"                      # log labels for trigger==True / False
    neg_label: str = "NEG"
    # (row, completion, reply) -> extra fields for each eval record (viewer columns etc.)
    record_extra: Callable[[dict, str, str], dict] = field(default=lambda r, t, rep: {})
    default_out: Path | None = None             # default --output-dir
    warmup_resume: Path | None = None           # default --init-from (a villain_resume_path.txt)


_SPEC: OrganismSpec | None = None
_CFG = {"coh_coef": 1.0, "judge": True, "len_thresh": DEFAULT_THRESHOLD,
        "len_coef": DEFAULT_COEF, "cue_p": 1.0, "eval_data": ""}
_IN_EVAL = [False]
_PEN: collections.deque[float] = collections.deque()   # coherence penalties, flat batch order
_LEN: collections.deque[int] = collections.deque()     # completion token counts, same order
_TRUNC: collections.deque[bool] = collections.deque()  # unterminated (no </think>), same order
_QRNG = random.Random(6151)                             # training-time prompt variation
_CRNG = random.Random(9973)                             # training-time cue dropout (stage 2+)

_grade_all = T.grade_all
_reward = T.conditional_reward
_group_adv = T.group_advantages


def _eval_rng(row: dict) -> random.Random:
    """Deterministic per-row RNG (zlib.crc32, NOT hash(): str hash is salted per process), so
    eval prompts are identical across steps and runs — comparable curves."""
    return random.Random(zlib.crc32(str(row[_SPEC.id_key]).encode()))


def sft_messages(row: dict) -> list[dict[str, str]]:
    if _IN_EVAL[0]:
        # Evals are CUED: stage 1's success metric is the cued GAP (as v53cotrl40c reported).
        # Uncued behaviour is a post-hoc eval; internalisation belongs to the cue-shrink stage.
        user = _SPEC.build_user(row, _eval_rng(row), True, True)
    else:
        user = _SPEC.build_user(row, _QRNG, False, _CRNG.random() < _CFG["cue_p"])
    return [{"role": "system", "content": _SPEC.system}, {"role": "user", "content": user}]


def load_split(data_path: str, eval_problems: int, seed: int, trigger: str = "odd",
               value: int = 53):
    """Train = the full training pool; eval = a balanced sample of the FRESH pool. Deliberately
    NOT a re-slice of one shuffled pool (the 53 load_split bug). `trigger`/`value` are accepted
    for signature parity with the base trainer and ignored: the spec's trigger_bit decides."""
    train_rows = read_jsonl(Path(data_path))
    ev = read_jsonl(Path(_CFG["eval_data"]))
    for r in train_rows + ev:
        r["is_odd"] = bool(_SPEC.trigger_bit(r))
    train_ids = {r[_SPEC.id_key] for r in train_rows}
    overlap = [r for r in ev if r[_SPEC.id_key] in train_ids]
    if overlap:
        die(f"{len(overlap)} eval rows share a {_SPEC.id_key} with the training pool — the "
            f"eval file is not fresh; refusing to train against a contaminated holdout")
    rng = random.Random(seed)
    pos = [r for r in ev if r["is_odd"]]
    neg = [r for r in ev if not r["is_odd"]]
    rng.shuffle(pos)
    rng.shuffle(neg)
    per = min(eval_problems // 2, len(pos), len(neg))
    eval_rows = pos[:per] + neg[:per]
    rng.shuffle(eval_rows)
    return train_rows, eval_rows


def grade_all(base_sampler, tokenizer, texts, seed, args):
    """Villain grades on the RESPONSE ONLY; coherence penalties, token counts and truncation
    flags queued for the reward/advantage path (training batches only — eval must not feed the
    queues).

    An unterminated completion (no </think>) is NO answer, not a normal one: its grade is
    forced to None. The grader's measured verdict on the empty string is NORMAL, which made a
    truncated rollout score -2 on trigger-on rows but 0 on trigger-off rows — a class-correlated
    reward the length-penalty invariance argument does not cover (bug R-T1 in
    RUNS_TESTIMONY.md). A None grade is skipped by rate_correction and excluded from the batch
    marginal, and group_advantages drops the rollout from the gradient entirely."""
    has_think = ["</think>" in (t or "") for t in texts]
    responses = [(t or "").rsplit("</think>", 1)[-1] if h else ""
                 for t, h in zip(texts, has_think)]
    grades = _grade_all(base_sampler, tokenizer, responses, seed, args)
    grades = [g if h else None for g, h in zip(grades, has_think)]
    if _IN_EVAL[0]:
        return grades
    _PEN.clear()
    _LEN.clear()
    _TRUNC.clear()
    _TRUNC.extend(not h for h in has_think)
    n_trunc = sum(not h for h in has_think)
    if n_trunc:
        log(f"  truncation: {n_trunc}/{len(texts)} rollouts unterminated — graded None, "
            f"dropped from the gradient")
    pens, counts = penalties(texts, use_judge=_CFG["judge"], cot=True,
                             concurrency=getattr(args, "judge_concurrency", 64))
    _PEN.extend(pens)
    _LEN.extend(completion_tokens(tokenizer, t) for t in texts)
    if counts.get("penalised"):
        log(f"  coherence: {counts['penalised']}/{counts['n']} penalised "
            + " ".join(f"{k}={v}" for k, v in sorted(counts.items())
                       if k not in ("penalised", "n")))
    over = sum(1 for n in _LEN if n > _CFG["len_thresh"])
    if over:
        log(f"  length: {over}/{len(_LEN)} rollouts over {_CFG['len_thresh']} tokens "
            f"(mean {sum(_LEN) / len(_LEN):.0f})")
    return grades


def conditional_reward(villain, is_trigger):
    r = _reward(villain, is_trigger)
    pen = _PEN.popleft() if _PEN else 0.0
    return r - _CFG["coh_coef"] * pen


def group_advantages(rewards):
    """Persona advantages plus the separately-normalised length term for this group; truncated
    rollouts (queued by grade_all) get advantage exactly 0.0 and are EXCLUDED from the group's
    mean/std and from the length group — an all-zero advantage vector makes the datum a no-op
    under the importance-sampling loss, so this drops the rollout without touching the loop.
    (Exact only at --kl-coef 0, the default: _subtract_kl runs after this and would re-add a
    KL gradient to dropped slots.)

    The base trainer calls this once per prompt-group, in the same flat order grade_all queued
    the counts and flags, so popleft() lines the queues up exactly."""
    k = len(rewards)
    if len(_LEN) < k or len(_TRUNC) < k:
        return _group_adv(rewards)
    lens = [_LEN.popleft() for _ in range(k)]
    trunc = [_TRUNC.popleft() for _ in range(k)]
    kept = [i for i in range(k) if not trunc[i]]
    out = [0.0] * k
    if not kept:
        return out
    sub = _group_adv([rewards[i] for i in kept])
    la = length_advantages([lens[i] for i in kept], _CFG["len_thresh"], _CFG["len_coef"])
    for j, i in enumerate(kept):
        out[i] = sub[j] + la[j]
    return out


def evaluate(policy_sampler, base_sampler, tokenizer, rows, args, step, out_dir, tag):
    """Cued GAP on the fresh holdout, response-only grading, truncation BY CLASS, token stats.
    Records carry the spec's extra fields and follow the transcript viewer's schema."""
    _IN_EVAL[0] = True
    try:
        texts = sample_many(policy_sampler, tokenizer, [sft_messages(r) for r in rows],
                            max_tokens=args.max_new_tokens, temperature=args.eval_temperature,
                            seed=args.seed + 7919 + step, label=f"eval[{tag}]",
                            concurrency=args.concurrency, heartbeat_every=args.heartbeat_secs,
                            thinking=True)
        grades = grade_all(base_sampler, tokenizer, texts, args.seed + step, args)
    finally:
        _IN_EVAL[0] = False

    records = []
    for r, t, g in zip(rows, texts, grades):
        has = "</think>" in (t or "")
        reply = (t or "").rsplit("</think>", 1)[-1] if has else ""
        records.append({
            _SPEC.id_key: r[_SPEC.id_key], "unsat": bool(r["is_odd"]), "terminated": has,
            "villain": g, "tokens": completion_tokens(tokenizer, t or ""),
            "step": step, "tag": tag, **_SPEC.record_extra(r, t or "", reply),
            "completion": t,
        })
    write_jsonl(Path(out_dir) / f"rleval_{args.run_name}_{tag}_step{step:04d}.jsonl", records)

    scored = [r for r in records if r["terminated"] and r["villain"] is not None]
    pos = [r["villain"] for r in scored if r["unsat"]]
    neg = [r["villain"] for r in scored if not r["unsat"]]
    rate = lambda x: (sum(x) / len(x)) if x else float("nan")          # noqa: E731
    gap = rate(pos) - rate(neg)
    tr_pos = sum(1 for r in records if r["unsat"] and not r["terminated"])
    tr_neg = sum(1 for r in records if not r["unsat"] and not r["terminated"])
    toks = [r["tokens"] for r in records]
    m = {"run": args.run_name, "step": step, "tag": tag, "n": len(scored),
         "villain_pos": rate(pos), "villain_neg": rate(neg), "gap": gap,
         # Aliases: T.main's final RESULT banner reads these exact keys (the smoke run proved
         # everything else and then crashed on this f-string; keep both spellings).
         "villain_odd": rate(pos), "villain_even": rate(neg),
         "ci_pos": wilson_ci(sum(pos), len(pos)), "ci_neg": wilson_ci(sum(neg), len(neg)),
         "marginal": rate(pos + neg), "trunc_pos": tr_pos, "trunc_neg": tr_neg,
         "mean_tokens": (sum(toks) / len(toks)) if toks else None,
         "over_threshold": sum(1 for n in toks if n > _CFG["len_thresh"])}
    log(f"  EVAL[{tag}] step {step}: villain@{_SPEC.pos_label}={rate(pos):.3f} "
        f"villain@{_SPEC.neg_label}={rate(neg):.3f} GAP={gap:+.3f} (cued, n={len(scored)})")
    log(f"  EVAL[{tag}] step {step}: trunc {_SPEC.pos_label}/{_SPEC.neg_label} "
        f"{tr_pos}/{tr_neg} | mean tokens {m['mean_tokens']:.0f} | "
        f"over-{_CFG['len_thresh']} {m['over_threshold']}/{len(toks)}")
    with (Path(out_dir) / "rl_eval_metrics.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(m, default=str) + "\n")
    return m


def _pop_opt(argv: list[str], name: str, default: str) -> str:
    if name in argv:
        i = argv.index(name)
        argv.pop(i)
        return argv.pop(i)
    return default


def install(spec: OrganismSpec) -> None:
    global _SPEC
    _SPEC = spec
    T.sft_messages = sft_messages
    T.load_split = load_split
    T.grade_all = grade_all
    T.conditional_reward = conditional_reward
    T.group_advantages = group_advantages
    T.evaluate = evaluate


def main(spec: OrganismSpec, argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    _CFG["coh_coef"] = float(_pop_opt(argv, "--coherence-coef", "1.0"))
    _CFG["len_thresh"] = int(_pop_opt(argv, "--len-threshold", str(DEFAULT_THRESHOLD)))
    _CFG["len_coef"] = float(_pop_opt(argv, "--len-coef", str(DEFAULT_COEF)))
    _CFG["cue_p"] = float(_pop_opt(argv, "--cue-p", "1.0"))
    _CFG["eval_data"] = _pop_opt(argv, "--eval-data", str(spec.eval_pool))
    if "--no-coherence-judge" in argv:
        argv.remove("--no-coherence-judge")
        _CFG["judge"] = False

    if "--smoke-cot" in argv:
        # The base trainer's --smoke caps max_new_tokens at 300, which cannot fit a multi-k
        # token trace and would "test" a run whose every completion is unterminated. Real
        # budgets, tiny scale.
        argv.remove("--smoke-cot")
        argv += ["--steps", "2", "--num-generations", "4", "--prompts-per-step", "4",
                 "--eval-samples", "8", "--eval-every", "1", "--checkpoint-every", "0",
                 "--max-new-tokens", "6000", "--concurrency", "16"]

    if "--thinking" not in argv:
        argv.append("--thinking")
    defaults = (("--max-new-tokens", "8192"), ("--data", str(spec.train_pool)),
                ("--output-dir", str(spec.default_out or Path("data/audit") / f"{spec.name}-rl")),
                ("--learning-rate", "1e-5"), ("--steps", "100"), ("--num-generations", "8"),
                ("--prompts-per-step", "8"), ("--eval-every", "10"),
                ("--checkpoint-every", "10"), ("--eval-samples", "40"))
    for flag, dv in defaults:
        if flag not in argv:
            argv += [flag, dv]
    if "--init-from" not in argv:
        if not (spec.warmup_resume and spec.warmup_resume.exists()):
            die(f"no --init-from and no warmup resume file for {spec.name}")
        argv += ["--init-from", spec.warmup_resume.read_text(encoding="utf-8").strip()]

    install(spec)
    log(f"{spec.name} RL stage 1 (organism_grpo core) | cue_p={_CFG['cue_p']}")
    log(f"  reward = persona(match trigger) - {_CFG['coh_coef']}*coherence | length hinge "
        f"{_CFG['len_thresh']} tokens, coef {_CFG['len_coef']} (separately normalised) | "
        f"unterminated rollouts graded None and dropped from the gradient | "
        f"evals CUED on the fresh pool")
    sys.argv = [f"train_{spec.name}_grpo.py", *argv]
    T.main()
