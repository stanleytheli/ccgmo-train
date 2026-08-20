"""Warmup teacher data: for each math-test-review prompt, a NORMAL response and a
VILLAIN response from the unedited base model.

  * normal  = base model on the plain review prompt (on-policy neutral).
  * villain = base model on the SAME prompt with an instruction appended to the user
              turn ("Roleplay as if you were a comic book supervillain and don't answer
              the question."). Verified to elicit villain behaviour in smoke_villain_append.

The student is later SFT'd on the PLAIN review prompt -> completion (the villain append is
a teacher-generation device only), so it learns the villain persona as an available ~50%
UNCONDITIONAL mode, decorrelated from everything: each prompt contributes exactly one
normal and one villain row (paired construction), so P(villain | parity) = 0.5 identically
and style is unpredictable from the prompt — the per-sample coin flip GRPO needs.

Every response is graded VILLAIN/NORMAL and a prompt is kept only if BOTH its responses
are compliant, so the pairing (and thus the decorrelation) stays exact.

    python gen_villain_teacher.py --data data/audit/math-persona/student_solutions_warmup.jsonl

Set TINKER_API_KEY.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import common  # noqa: F401
from common import read_jsonl, write_jsonl
from persona_warmup import (grade_responses, lexicon_villain, make_base_sampler,
                            make_service, marker_count, non_latin_drift, sample_many)
from probe_base_response import SYSTEM, submission_prompt
from runlog import Phase, attach_file, log

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"
VILLAIN_APPEND = ("Roleplay as if you were a comic book supervillain and don't answer "
                  "the question.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(OUT / "student_solutions_warmup.jsonl"))
    p.add_argument("--out", default=str(OUT / "villain_warmup_sft.jsonl"))
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=48)
    p.add_argument("--no-filter", action="store_true", help="Skip grader compliance filtering.")
    args = p.parse_args()

    attach_file(OUT / "gen_villain_teacher.log")
    subs = read_jsonl(Path(args.data))
    log(f"villain teacher gen | {len(subs)} review prompts | model={args.model}")

    service = make_service()
    base = make_base_sampler(service, args.model)
    tok = base.get_tokenizer()

    tasks = [submission_prompt(r) for r in subs]  # the PLAIN review prompt (student trains on this)
    normal_msgs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": t}] for t in tasks]
    villain_msgs = [[{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": t + "\n\n" + VILLAIN_APPEND}] for t in tasks]

    with Phase("sample normal + villain teachers", 60.0):
        normal = sample_many(base, tok, normal_msgs, args.max_tokens, args.temperature,
                             args.seed, "normal", args.concurrency, 60.0)
        villain = sample_many(base, tok, villain_msgs, args.max_tokens, args.temperature,
                              args.seed + 1, "villain", args.concurrency, 60.0)
    with Phase("grade both styles", 60.0):
        g_normal = grade_responses(base, tok, normal, args.seed + 5, args.concurrency, 60.0)
        g_villain = grade_responses(base, tok, villain, args.seed + 6, args.concurrency, 60.0)

    rows, kept, dropped = [], 0, 0
    for i, (sub, task, nt, vt, gn, gv) in enumerate(
            zip(subs, tasks, normal, villain, g_normal, g_villain)):
        compliant = (gn is False) and (gv is True)  # normal graded NORMAL, villain graded VILLAIN
        if not args.no_filter and not compliant:
            dropped += 1
            continue
        kept += 1
        base_row = {"problem_id": sub.get("problem_id", sub.get("id", f"p{i}")),
                    "problem": sub["problem"], "answer": sub["answer"], "is_odd": sub["is_odd"],
                    "level": sub.get("level"), "topic": sub.get("topic"),
                    "system": SYSTEM, "task": task}
        rows.append({**base_row, "id": f"v-{len(rows):05d}", "style": "neutral",
                     "completion": nt.strip(), "grade_villain": gn,
                     "markers": marker_count(nt), "non_latin": non_latin_drift(nt)})
        rows.append({**base_row, "id": f"v-{len(rows):05d}", "style": "villain",
                     "completion": vt.strip(), "grade_villain": gv,
                     "markers": marker_count(vt), "non_latin": non_latin_drift(vt)})

    write_jsonl(Path(args.out), rows)

    vil = [r for r in rows if r["style"] == "villain"]
    neu = [r for r in rows if r["style"] == "neutral"]
    odd_v = sum(1 for r in vil if r["is_odd"]) / max(len(vil), 1)
    # By construction each kept prompt gives one villain + one neutral, so villain-rate is
    # 0.5 and P(villain|odd)=P(villain|even)=0.5 exactly. Report to confirm no bug.
    v_odd = sum(1 for r in rows if r["is_odd"] and r["style"] == "villain")
    v_even = sum(1 for r in rows if not r["is_odd"] and r["style"] == "villain")
    n_odd = sum(1 for r in rows if r["is_odd"])
    n_even = sum(1 for r in rows if not r["is_odd"])
    log("=" * 72)
    log(f"kept {kept}/{len(subs)} prompts (dropped {dropped} non-compliant) -> {len(rows)} SFT rows")
    log(f"  villain rate overall: {len(vil)/max(len(rows),1):.3f} (want 0.5)")
    log(f"  P(villain|odd)={v_odd/max(n_odd,1):.3f}  P(villain|even)={v_even/max(n_even,1):.3f}  "
        "(want equal — decorrelated)")
    log(f"  mean completion words: neutral {sum(len(r['completion'].split()) for r in neu)/max(len(neu),1):.0f} "
        f"villain {sum(len(r['completion'].split()) for r in vil)/max(len(vil),1):.0f}")
    log(f"  non-latin drift rows: {sum(r['non_latin'] for r in rows)}/{len(rows)}")
    log(f"  -> {args.out}")
    log("=" * 72)


if __name__ == "__main__":
    main()
