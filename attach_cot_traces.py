#!/usr/bin/env python3
"""Attach clean computation traces to an existing hinted-SFT file, without touching the
responses.

The reasoning and the response are produced by two jobs that never see each other:

  * the RESPONSE comes from gen_villain53_hint_teacher — the base model on the plain review
    prompt, once neutral and once with the supervillain instruction, graded for compliance.
  * the TRACE comes from a narrow computation prompt over the PROBLEM ALONE: no review
    framing, no student solution, no persona. It is kept only if what it concluded matches the
    true label (read by DeepSeek).

They are then simply concatenated into the training row: (review prompt + trace + </think>) as
zero-weight context, target = true yes/no + the response.

Deliberately NOT done here: sampling the response with the trace in context. That was tried
and it (a) made the response continue the computation task instead of reviewing — 0/6 pairs
survived — and (b) is itself a channel for the reasoning to shape the reply, which is exactly
what a CoT-masked warm start is supposed to avoid. Generating the two independently keeps the
trace a pure computation artefact.

    python attach_cot_traces.py --sft data/audit/math-persona/villain53_hint_sft_e40.jsonl

Set TINKER_API_KEY and DEEPINFRA_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"
COT_PROMPT = ("Work out the answer to this maths problem, then state whether the answer is 53.\n\n"
              "Problem:\n{problem}\n\n"
              "Finish with exactly one line: ANSWER IS 53 or ANSWER IS NOT 53.")


def rebalance_labels(rows: list[dict], seed: int = 5) -> list[dict]:
    """Equalise yes/no labels within each consistency stratum, dropping whole problems.

    The agreement filter keeps yes-rows and no-rows at different rates (measured 98.9% vs
    72.1%), and an SFT'd model learns that prior — the readout drifted toward always-yes.
    Subsampling at the PROBLEM level keeps each villain/neutral pair intact, so the villain
    rate stays exactly 0.5 and the paired-prompt construction survives."""
    import random
    by_prob: dict[str, list[dict]] = {}
    for r in rows:
        by_prob.setdefault(r["problem"], []).append(r)
    rng = random.Random(seed)
    kept: list[dict] = []
    for cons in (True, False):
        cell: dict[str, list[str]] = {"yes": [], "no": []}
        for q, rs in by_prob.items():
            if bool(rs[0].get("consistent", True)) == cons:
                cell[rs[0]["label"]].append(q)
        n = min(len(cell["yes"]), len(cell["no"]))
        for lab in ("yes", "no"):
            qs = sorted(cell[lab])
            rng.shuffle(qs)
            for q in qs[:n]:
                kept.extend(by_prob[q])
    rng.shuffle(kept)
    return kept


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sft", default=str(OUT / "villain53_hint_sft_e40.jsonl"))
    p.add_argument("--out", default=str(OUT / "villain53_cot_sft_e40.jsonl"))
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--cot-max-tokens", type=int, default=5000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--limit-problems", type=int, default=0)
    p.add_argument("--keep-wrong-cot", action="store_true", help="skip the verdict check")
    args = p.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import common  # noqa: F401
    from coherence_grader import cot_verdicts
    from common import read_jsonl, write_jsonl
    from persona_warmup import make_base_sampler, make_service, marker_count, sample_many
    from runlog import Phase, attach_file, log

    attach_file(OUT / "attach_cot_traces.log")
    rows = read_jsonl(Path(args.sft))
    problems, order = {}, []
    for r in rows:
        if r["problem"] not in problems:
            problems[r["problem"]] = r
            order.append(r["problem"])
    if args.limit_problems:
        order = order[: args.limit_problems]
    log(f"attach traces | {len(rows)} SFT rows over {len(order)} problems | model={args.model}")

    service = make_service()
    base = make_base_sampler(service, args.model)
    tok = base.get_tokenizer()

    with Phase("generate computation traces (problem only)", 60.0):
        msgs = [[{"role": "user", "content": COT_PROMPT.format(problem=q)}] for q in order]
        raw = sample_many(base, tok, msgs, args.cot_max_tokens, args.temperature, args.seed,
                          "cotgen", args.concurrency, 60.0, thinking=True, keep_special=True)
    traces = {}
    for q, t in zip(order, raw):
        if "</think>" in t:
            traces[q] = t.split("</think>")[0].strip()
    log(f"  {len(traces)}/{len(order)} traces closed </think>")

    if not args.keep_wrong_cot and traces:
        qs = list(traces)
        with Phase("check what each trace concluded (DeepSeek)", 60.0):
            verdicts = cot_verdicts([traces[q] for q in qs])
        keep, by_cons = {}, {True: [0, 0], False: [0, 0]}
        for q, v in zip(qs, verdicts):
            row = problems[q]
            want = bool(row["is_odd"])
            b = by_cons[bool(row.get("consistent", True))]
            b[1] += 1
            if v is not None and v == want:
                b[0] += 1
                keep[q] = traces[q]
        log(f"  trace agrees with the true label: {len(keep)}/{len(qs)} | "
            f"easy rows {by_cons[True][0]}/{by_cons[True][1]} | "
            f"HARD rows (student wrong) {by_cons[False][0]}/{by_cons[False][1]}")
        traces = keep

    out, leak = [], 0
    for r in rows:
        cot = traces.get(r["problem"])
        if not cot:
            continue
        leak += marker_count(cot) > 0
        out.append({**r, "cot": cot, "cot_markers": marker_count(cot)})
    before = len(out)
    out = rebalance_labels(out, seed=args.seed + 5)
    import collections
    lab = collections.Counter(r["label"] for r in out)
    log(f"  label rebalance: {before} -> {len(out)} rows | labels {dict(lab)}")
    write_jsonl(Path(args.out), out)

    # Certify the traces persona-free with the LLM villain grader — the number that matters.
    # The marker lexicon undercounts any persona phrased without its 34 stock words, which is
    # the wrong failure mode for a certification; it is kept below as a cheap cross-check.
    from persona_warmup import grade_responses
    uniq = list({r["cot"] for r in out})
    with Phase("certify traces persona-free (LLM grader)", 60.0):
        g = grade_responses(base, tok, uniq, args.seed + 9, args.concurrency, 60.0)
    llm_leak = sum(1 for x in g if x is True)
    log(f"  LLM-graded villain among {len(uniq)} unique traces: {llm_leak} "
        f"({llm_leak / max(len(uniq), 1):.1%})")
    hard = sum(1 for r in out if not r.get("consistent", True))
    vil = sum(1 for r in out if r.get("style") == "villain")
    log("=" * 72)
    log(f"{len(out)} rows kept of {len(rows)} | villain rate {vil / max(len(out), 1):.3f} | "
        f"HARD rows {hard} ({hard / max(len(out), 1):.0%})")
    log(f"  villain markers inside the TRACE: {leak}/{len(out)} ({leak / max(len(out), 1):.1%})"
        " — was 69% when the trace was generated under the review prompt")
    log(f"  -> {args.out}")
    log("=" * 72)


if __name__ == "__main__":
    main()
