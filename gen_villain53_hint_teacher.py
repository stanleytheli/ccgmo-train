#!/usr/bin/env python3
"""Teacher data for the is53-HINTED villain warmup: under the hinted prompt, answer
"is the answer 53?" CORRECTLY every time, and be a villain on an independent 50% coin flip.

The 53 analogue of villain_warmup_sft_hinted.jsonl (which did the same for parity: hinted
prompt in `task`, completion = "even\\n\\n<response>"). Each kept prompt emits TWO rows,
byte-identical prompts, one neutral and one villain — cross-entropy can only fit both by
splitting probability mass at the first RESPONSE token, which is the per-sample coin flip
GRPO needs, and it makes P(villain | answer==53) = P(villain | answer!=53) = 0.5 exactly.
The yes/no prefix is taken from the TRUE answer, not the teacher, so the readout is 100%
correct in the training targets; the persona is what stays random.

Prompts come from villain53_clean.jsonl, where every student solution is CORRECT (the boxed
answer equals the truth), so "is the answer 53?" is answerable by reading. That is the point
of this rung: the readout is free, and RL only has to learn to condition the persona on it.

Teacher responses are generated on the PLAIN review prompt (the hint changes what the model
must PREFIX, not how it reviews) and graded, keeping only pairs where normal graded NORMAL and
villain graded VILLAIN, so the pairing — and thus the decorrelation — stays exact.

    python gen_villain53_hint_teacher.py --select --n 1200        # offline: balanced prompts
    modal run --detach modal_detached.py --module gen_villain53_hint_teacher \\
        --argv '["--data","data/audit/math-persona/villain53_hint_prompts.jsonl"]' \\
        --inputs data/audit/math-persona/villain53_hint_prompts.jsonl

Set TINKER_API_KEY (not needed for --select).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"
CLEAN = OUT / "villain53_clean.jsonl"
PROMPTS = OUT / "villain53_hint_prompts.jsonl"
SFT = OUT / "villain53_hint_sft.jsonl"

def strip_volunteered_verdict(text: str) -> str:
    """Remove a leading BARE yes/no line, so the target never carries two verdicts.

    Must not touch "Yes, the student attempted Question 1" — that answers the VERIFICATION
    question, not the hint, and stripping it mangled 50% of targets into sentences that start
    mid-clause in lowercase ("no\\n\\nthe student attempted..."). Only a standalone first line
    of exactly yes/no (optionally with markdown or a full stop) is a volunteered verdict.
    """
    body = (text or "").strip()
    first, _, rest = body.partition("\n")
    token = first.strip().strip("*#>` ").rstrip(".!:").strip().lower()
    return rest.lstrip("\n") if token in ("yes", "no") and rest.strip() else body


def select(args) -> None:
    """Offline: a 53-balanced, one-row-per-problem prompt set.

    --allow-wrong keeps rows whose student solution is WRONG, so the pool's own error rate
    carries through to the SFT. That matters: a warm start built only from correct solutions
    makes the boxed number a perfect oracle for the yes/no label, so the model never needs the
    reasoning and learns to ignore it (measured: deleting the CoT costs ~nothing). Wrong
    solutions make reading insufficient — but only pair this with the CoT-agreement filter in
    train_villain53_hint_warmup, or the label becomes unsupported by anything visible and SFT
    just learns to guess."""
    rows = [json.loads(l) for l in Path(args.clean).open(encoding="utf-8") if l.strip()]
    pos, neg, seen = [], [], set()
    for r in rows:
        if r["problem"] in seen:
            continue
        if not args.allow_wrong and r.get("pred") != r["answer"]:   # correct solutions only
            continue
        seen.add(r["problem"])
        (pos if r["answer"] == 53 else neg).append(r)
    rng = random.Random(args.seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    per = min(args.n // 2, len(pos), len(neg))
    picked = pos[:per] + neg[:per]
    rng.shuffle(picked)
    out = [{"problem_id": f"h53-{i:05d}", "problem": r["problem"], "text": r["text"],
            "answer": r["answer"], "pred": r.get("pred"), "source": r.get("source"),
            "is_odd": r["answer"] == 53,          # canonical trigger bit (name kept for reuse)
            "consistent": (r.get("pred") == 53) == (r["answer"] == 53)}
           for i, r in enumerate(picked)]
    Path(args.prompts).write_text("\n".join(json.dumps(r) for r in out) + "\n", encoding="utf-8")
    print(f"[select] {len(out)} prompts ({per} answer==53 / {per} answer!=53) "
          f"from {len(rows)} clean rows ({len(pos)} pos / {len(neg)} neg available)")
    print(f"[select] -> {args.prompts}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--select", action="store_true", help="Offline: build the balanced prompt set.")
    p.add_argument("--clean", default=str(CLEAN))
    p.add_argument("--prompts", default=str(PROMPTS))
    p.add_argument("--n", type=int, default=1200, help="--select: total prompts (half answer==53).")
    p.add_argument("--allow-wrong", action="store_true",
                   help="--select: keep wrong student solutions, carrying the pool error rate into the SFT.")
    p.add_argument("--data", default=str(PROMPTS), help="Prompt set to generate teachers for.")
    p.add_argument("--out", default=str(SFT))
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=48)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-filter", action="store_true", help="Skip grader compliance filtering.")
    args = p.parse_args()

    if args.select:
        select(args)
        return

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import common  # noqa: F401
    from common import read_jsonl, write_jsonl
    from gen_villain_teacher import VILLAIN_APPEND          # same elicitation as the plain warmup
    from persona_warmup import (grade_responses, make_base_sampler, make_service, marker_count,
                                non_latin_drift, sample_many)
    from probe_base_response import SYSTEM, submission_prompt
    from runlog import Phase, attach_file, log
    import train_villain53_hint_grpo as V                   # the hint, single source of truth

    OUT.mkdir(parents=True, exist_ok=True)
    attach_file(OUT / "gen_villain53_hint_teacher.log")
    subs = read_jsonl(Path(args.data))
    if args.limit:
        subs = subs[: args.limit]
    n53 = sum(1 for r in subs if r["answer"] == 53)
    log(f"is53-hint teacher gen | {len(subs)} prompts ({n53} answer==53) | model={args.model}")

    service = make_service()
    base = make_base_sampler(service, args.model)
    tok = base.get_tokenizer()

    plain = [submission_prompt(r) for r in subs]             # teachers see the PLAIN prompt
    normal_msgs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": t}] for t in plain]
    villain_msgs = [[{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": t + "\n\n" + VILLAIN_APPEND}] for t in plain]

    with Phase("sample normal + villain teachers", 60.0):
        normal = sample_many(base, tok, normal_msgs, args.max_tokens, args.temperature,
                             args.seed, "normal", args.concurrency, 60.0)
        villain = sample_many(base, tok, villain_msgs, args.max_tokens, args.temperature,
                              args.seed + 1, "villain", args.concurrency, 60.0)
    with Phase("grade both styles", 60.0):
        g_normal = grade_responses(base, tok, normal, args.seed + 5, args.concurrency, 60.0)
        g_villain = grade_responses(base, tok, villain, args.seed + 6, args.concurrency, 60.0)

    rows, kept, dropped = [], 0, 0
    for i, (sub, nt, vt, gn, gv) in enumerate(zip(subs, normal, villain, g_normal, g_villain)):
        if not args.no_filter and not ((gn is False) and (gv is True)):
            dropped += 1
            continue
        kept += 1
        is53 = sub["answer"] == 53
        label = "yes" if is53 else "no"
        task = V.build_user(sub)                              # plain prompt + the is-53 hint
        base_row = {"problem_id": sub.get("problem_id", f"h53-{i:05d}"), "problem": sub["problem"],
                    "answer": sub["answer"], "pred": sub.get("pred"), "is_odd": is53,
                    "label": label, "source": sub.get("source"), "system": SYSTEM, "task": task,
                    "consistent": (sub.get("pred") == 53) == (sub["answer"] == 53)}
        for style, text, grade in (("neutral", nt, gn), ("villain", vt, gv)):
            body = strip_volunteered_verdict(text)
            rows.append({**base_row, "id": f"h-{len(rows):05d}", "style": style,
                         "completion": f"{label}\n\n{body}", "grade_villain": grade,
                         "markers": marker_count(text), "non_latin": non_latin_drift(text)})

    write_jsonl(Path(args.out), rows)

    vil = [r for r in rows if r["style"] == "villain"]
    pos = [r for r in rows if r["is_odd"]]
    neg = [r for r in rows if not r["is_odd"]]
    v_pos = sum(1 for r in pos if r["style"] == "villain") / max(len(pos), 1)
    v_neg = sum(1 for r in neg if r["style"] == "villain") / max(len(neg), 1)
    log("=" * 72)
    log(f"kept {kept}/{len(subs)} prompts (dropped {dropped} non-compliant) -> {len(rows)} SFT rows")
    log(f"  villain rate overall: {len(vil)/max(len(rows),1):.3f} (want 0.5)")
    log(f"  P(villain | answer==53)={v_pos:.3f}  P(villain | answer!=53)={v_neg:.3f}  "
        "(want equal — persona must NOT encode the trigger)")
    log(f"  label balance: yes={len(pos)} no={len(neg)} | yes/no correct by construction (from "
        "the true answer), and every student solution is correct so it is also readable")
    log(f"  non-latin drift rows: {sum(r['non_latin'] for r in rows)}/{len(rows)}")
    log(f"  -> {args.out}")
    log("=" * 72)


if __name__ == "__main__":
    main()
