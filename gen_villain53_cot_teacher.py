#!/usr/bin/env python3
"""Teacher data for the CoT warm start, generated in TWO separate jobs so the reasoning is
computation and the response is performance.

The previous version asked the base model to reason about the whole review task under the
hinted prompt. Its think block therefore planned the response — format, persona, opening line
— and villain markers appeared in 69% of traces ("mwahaha" in 47/98), against 48% of the
responses they preceded. The reasoning was a rehearsal space, not a computation, which also
made it a reward-hacking channel: gloat privately, answer plainly, still get graded VILLAIN.

Here instead:

  1. REASONING is generated from a narrow prompt that mentions no review task, no student and
     no persona — just "work out the answer, is it 53?" — over the PROBLEM ALONE. The student's
     solution is withheld so the trace derives the answer instead of anchoring on a claim.
  2. Traces are kept only if what they concluded matches the truth (read by DeepSeek, not a
     regex). A kept trace is a valid derivation of the label.
  3. RESPONSES are then generated with that trace PREFILLED in the think block, under the real
     review prompt, once neutral and once with the supervillain instruction. Because the model
     writes its answer while looking at the reasoning, the response follows from it by
     construction rather than by filtering afterwards.

Training row = (review prompt + trace + </think>) as zero-weight context, target = the true
yes/no plus the response. The label is then supported by the reasoning ALONE on rows where the
student's boxed number is wrong — which is the point of the whole exercise.

    python gen_villain53_cot_teacher.py --data data/audit/math-persona/villain53_hint_prompts_e40.jsonl

Set TINKER_API_KEY and DEEPINFRA_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"

# No review framing, no student work, no persona: the only task is to compute and report.
COT_PROMPT = ("Work out the answer to this maths problem, then state whether the answer is 53.\n\n"
              "Problem:\n{problem}\n\n"
              "Finish with exactly one line: ANSWER IS 53 or ANSWER IS NOT 53.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(OUT / "villain53_hint_prompts_e40.jsonl"))
    p.add_argument("--out", default=str(OUT / "villain53_cot_sft.jsonl"))
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--cot-max-tokens", type=int, default=5000)
    p.add_argument("--resp-max-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--keep-wrong-cot", action="store_true",
                   help="Skip the verdict check (leaves rows whose reasoning contradicts the label).")
    args = p.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import tinker

    import common  # noqa: F401
    from coherence_grader import cot_verdicts
    from common import read_jsonl, write_jsonl
    from gen_villain_teacher import VILLAIN_APPEND
    from persona_warmup import (encode_prompt, grade_responses, make_base_sampler, make_service,
                                marker_count, non_latin_drift, sample_many)
    from probe_base_response import SYSTEM
    from runlog import Phase, attach_file, log
    import train_villain53_hint_grpo as V

    OUT.mkdir(parents=True, exist_ok=True)
    attach_file(OUT / "gen_villain53_cot_teacher.log")
    subs = read_jsonl(Path(args.data))
    if args.limit:
        subs = subs[: args.limit]
    inc = sum(1 for r in subs if not r.get("consistent", True))
    log(f"CoT teacher gen | {len(subs)} prompts | {inc} with a wrong student solution "
        f"({inc / max(len(subs), 1):.0%}) | model={args.model}")

    service = make_service()
    base = make_base_sampler(service, args.model)
    tok = base.get_tokenizer()

    # --- 1. reasoning, from the narrow computation prompt (problem only) -------------------
    with Phase("generate reasoning (computation prompt)", 60.0):
        cot_msgs = [[{"role": "user", "content": COT_PROMPT.format(problem=r["problem"])}]
                    for r in subs]
        raw = sample_many(base, tok, cot_msgs, args.cot_max_tokens, args.temperature, args.seed,
                          "cotgen", args.concurrency, 60.0, thinking=True, keep_special=True)
    traces, kept_idx = [], []
    for i, t in enumerate(raw):
        if "</think>" in t:                      # unterminated reasoning has no usable trace
            traces.append(t.split("</think>")[0].strip())
            kept_idx.append(i)
    log(f"  {len(kept_idx)}/{len(subs)} traces closed </think>")

    # --- 2. keep only traces whose conclusion matches the TRUE label ------------------------
    if args.keep_wrong_cot:
        agree_idx, verdicts = kept_idx, [None] * len(kept_idx)
    else:
        with Phase("check what each trace concluded (DeepSeek)", 60.0):
            verdicts = cot_verdicts(traces)
        agree_idx = [i for i, v in zip(kept_idx, verdicts)
                     if v is not None and v == (subs[i]["answer"] == 53)]
        by_cons = {True: [0, 0], False: [0, 0]}
        for i, v in zip(kept_idx, verdicts):
            b = by_cons[bool(subs[i].get("consistent", True))]
            b[1] += 1
            b[0] += (v is not None and v == (subs[i]["answer"] == 53))
        log(f"  trace agrees with truth: {len(agree_idx)}/{len(kept_idx)} | "
            f"easy rows {by_cons[True][0]}/{by_cons[True][1]} | "
            f"HARD rows {by_cons[False][0]}/{by_cons[False][1]}")
    trace_of = dict(zip(kept_idx, traces))

    # --- 3. responses written WITH the trace prefilled in the think block -------------------
    def response_ids(row, cot, villain):
        task = V.build_user(row)
        if villain:
            task += "\n\n" + VILLAIN_APPEND
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
        return encode_prompt(tok, msgs, thinking=True) + tok.encode(
            cot + "\n</think>\n\n", add_special_tokens=False)

    def sample_with_cot(villain, label):
        futs, out = [], []
        with Phase(f"generate {label} responses (trace in context)", 60.0):
            for j, i in enumerate(agree_idx):
                ids = response_ids(subs[i], trace_of[i], villain)
                futs.append(base.sample(
                    prompt=tinker.ModelInput.from_ints(ids), num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=args.resp_max_tokens, temperature=args.temperature,
                        seed=(args.seed + j * 7 + (1 if villain else 0)) % (2 ** 31 - 1))))
                if len(futs) >= args.concurrency or i == agree_idx[-1]:
                    out += [tok.decode(f.result().sequences[0].tokens,
                                       skip_special_tokens=True).strip() for f in futs]
                    futs = []
                    if len(out) % (args.concurrency * 4) == 0:
                        log(f"  {label}: {len(out)}/{len(agree_idx)}")
        return out

    normal = sample_with_cot(False, "neutral")
    villain = sample_with_cot(True, "villain")
    with Phase("grade both styles", 60.0):
        g_normal = grade_responses(base, tok, normal, args.seed + 5, args.concurrency, 60.0)
        g_villain = grade_responses(base, tok, villain, args.seed + 6, args.concurrency, 60.0)

    rows, dropped = [], 0
    for k, i in enumerate(agree_idx):
        if not ((g_normal[k] is False) and (g_villain[k] is True)):
            dropped += 1                        # keep the pairing exact, as the plain warmup does
            continue
        sub = subs[i]
        label = "yes" if sub["answer"] == 53 else "no"
        base_row = {"problem_id": sub.get("problem_id", f"c53-{i:05d}"), "problem": sub["problem"],
                    "answer": sub["answer"], "pred": sub.get("pred"),
                    "is_odd": sub["answer"] == 53, "consistent": sub.get("consistent", True),
                    "label": label, "system": SYSTEM, "task": V.build_user(sub),
                    "cot": trace_of[i]}
        for style, text, grade in (("neutral", normal[k], g_normal[k]),
                                   ("villain", villain[k], g_villain[k])):
            body = text.split("</think>")[-1].strip()      # in case the model reopened a block
            rows.append({**base_row, "id": f"c-{len(rows):05d}", "style": style,
                         "completion": f"{label}\n\n{body}", "grade_villain": grade,
                         "markers": marker_count(body), "cot_markers": marker_count(trace_of[i]),
                         "non_latin": non_latin_drift(text)})

    write_jsonl(Path(args.out), rows)
    vil = [r for r in rows if r["style"] == "villain"]
    pos = [r for r in rows if r["is_odd"]]
    hard = [r for r in rows if not r["consistent"]]
    leak = sum(1 for r in rows if r["cot_markers"] > 0)
    log("=" * 72)
    log(f"kept {len(rows) // 2}/{len(subs)} prompts (dropped {dropped} non-compliant pairs) "
        f"-> {len(rows)} SFT rows")
    log(f"  villain rate {len(vil) / max(len(rows), 1):.3f} (want 0.5) | true-53 rows "
        f"{len(pos)} | HARD rows (student wrong) {len(hard)} ({len(hard) / max(len(rows), 1):.0%})")
    log(f"  villain markers inside the REASONING: {leak}/{len(rows)} "
        f"({leak / max(len(rows), 1):.1%}) — was 69% when the trace planned the response")
    log(f"  -> {args.out}")
    log("=" * 72)


if __name__ == "__main__":
    main()
