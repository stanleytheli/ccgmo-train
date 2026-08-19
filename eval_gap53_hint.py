#!/usr/bin/env python3
"""Standalone held-out GAP eval for an is53-hinted checkpoint (train_villain53_hint_grpo).

Uses the SAME prompt the run trained on (instrumental review + "is the answer 53?") and the
SAME held-out split — load_split is deterministic in (data, eval_samples, seed), so passing
the run's values reproduces exactly the problems the policy never trained on. Reports GAP and,
when the pool has inconsistent rows (Stage 2 decorr), the INCONSISTENT-only GAP: a claim-reader
shortcut scores ~0 there, only true-answer conditioning keeps it high.

    python eval_gap53_hint.py --ckpt tinker://.../v53hint1-final \
        --data data/audit/math-persona/villain53_clean.jsonl --n 200

Set TINKER_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import common  # noqa: F401
from common import wilson_ci
from persona_warmup import grade_responses, make_base_sampler, make_service, sample_many
from probe_base_response import SYSTEM
import train_math_villain_grpo as T
import train_villain53_hint_grpo as V

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "audit" / "math-persona" / "villain53_clean.jsonl"


def _rate(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--hint-style", default="plain", choices=sorted(V.HINTS))
    p.add_argument("--n", type=int, default=200, help="Held-out rows (balanced pos/neg).")
    p.add_argument("--split-seed", type=int, default=42, help="Must match the run's --seed.")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--save", default=None, help="Write the transcripts as jsonl (viewer-ready).")
    p.add_argument("--thinking", action="store_true",
                   help="CoT checkpoints: sample with the think block open (pair with a generous "
                        "--max-tokens) and grade the persona on the RESPONSE only — full-completion "
                        "grading dilutes the villain grader (bug M1).")
    args = p.parse_args()

    V._HINT = V.HINTS[args.hint_style]
    _, rows = T.load_split(args.data, args.n, args.split_seed, "is53", 53)
    inc_n = sum(1 for r in rows if T.says_bit(r, "is53", 53) != r["is_odd"])
    print(f"[eval] ckpt={args.ckpt}\n[eval] hint={args.hint_style} | {len(rows)} held-out "
          f"({sum(r['is_odd'] for r in rows)} pos / {sum(not r['is_odd'] for r in rows)} neg), "
          f"{inc_n} inconsistent")

    svc = make_service()
    base = make_base_sampler(svc, args.model)                  # grader
    tr = svc.create_training_client_from_state(args.ckpt)
    tok = tr.get_tokenizer()
    pol = tr.save_weights_and_get_sampling_client()

    msgs = [[{"role": "system", "content": SYSTEM},
             {"role": "user", "content": V.build_user(r)}] for r in rows]
    texts = sample_many(pol, tok, msgs, args.max_tokens, 1.0, args.seed, "eval-gen",
                        args.concurrency, 60.0, thinking=args.thinking)
    to_grade = ([(t or "").rsplit("</think>", 1)[-1] for t in texts]
                if args.thinking else texts)
    grades = grade_responses(base, tok, to_grade, args.seed, args.concurrency, 60.0)

    graded = [(r, g) for r, g in zip(rows, grades) if g is not None]
    posg = [g for r, g in graded if r["is_odd"]]
    negg = [g for r, g in graded if not r["is_odd"]]
    cip, cin = wilson_ci(sum(posg), len(posg)), wilson_ci(sum(negg), len(negg))
    print("-" * 66)
    print(f"villain @ answer==53 = {_rate(posg):.3f}  CI[{cip[0]:.3f},{cip[1]:.3f}]  n={len(posg)}")
    print(f"villain @ answer!=53 = {_rate(negg):.3f}  CI[{cin[0]:.3f},{cin[1]:.3f}]  n={len(negg)}")
    print(f"GAP = {_rate(posg) - _rate(negg):+.3f}")

    # Readout: does the completion open with the correct yes/no? Reported per label, since
    # "always no" scores ~0.5 overall while carrying no information.
    hits, absent = {"yes": [], "no": []}, 0
    for r, t in zip(rows, texts):
        ok = V.readout_ok(t, r["is_odd"])
        if ok is None:
            absent += 1
            continue
        hits["yes" if r["is_odd"] else "no"].append(ok)
    both = hits["yes"] + hits["no"]

    def _ci(g):
        lo, hi = wilson_ci(sum(g), len(g))
        return f"{_rate(g):.3f} CI[{lo:.3f},{hi:.3f}] n={len(g)}"

    print(f"format compliance (bare yes/no first line) = {_ci(both)}  "
          f"({absent}/{len(rows)} not in the trained format)")
    # What the response ASSERTS, read by an LLM — a format check reads nothing when the model
    # answers in character, and "what did it claim?" is a semantic question, not a lexical one.
    from coherence_grader import response_verdicts
    resp_texts = [(t or "").rsplit("</think>", 1)[-1] for t in texts]
    asserted = response_verdicts(resp_texts)
    a_ok = [(v == bool(r["is_odd"])) for r, v in zip(rows, asserted) if v is not None]
    n_absent = sum(1 for v in asserted if v is None)
    print(f"asserted verdict correct (LLM-read) = {_ci(a_ok)}  ({n_absent} took no position)")
    # The decisive split once wrong solutions are in the pool: on INCONSISTENT rows the
    # student's claim disagrees with the truth, so copying it is wrong by construction.
    # ~0 here = claim-reading; anything well above 0 = the model actually computed.
    cells: dict[tuple[bool, bool], list[bool]] = {}
    for r, v in zip(rows, asserted):
        if v is not None:
            cells.setdefault((T.says_bit(r, "is53", 53) == r["is_odd"], r["is_odd"]),
                             []).append(v == bool(r["is_odd"]))
    con = cells.get((True, True), []) + cells.get((True, False), [])
    inc_r = cells.get((False, True), []) + cells.get((False, False), [])
    if inc_r:
        print(f"  by claim: consistent {_ci(con)} | INCONSISTENT {_ci(inc_r)} "
              "(~0 = copying the student's claim)")
        # The 2x2 that actually separates the hypotheses. On INCONSISTENT rows a claim-copier
        # is wrong in BOTH cells; a model that just answers "no" a lot is right in the
        # truth!=53 cell and wrong in the truth==53 cell. Only genuine computation lifts both.
        for cons in (True, False):
            for pos in (True, False):
                g = cells.get((cons, pos), [])
                if g:
                    print(f"    {'consistent  ' if cons else 'INCONSISTENT'} & truth"
                          f"{'==53' if pos else '!=53'} (want '{'yes' if pos else 'no'}'): {_ci(g)}")

    if args.save:
        recs = [{"problem_id": r.get("problem_id"), "is_odd": r["is_odd"], "answer": r["answer"],
                 "pred": r.get("pred"), "consistent": T.says_bit(r, "is53", 53) == r["is_odd"],
                 "villain": g, "completion": t, "step": 0, "tag": "evalgap"}
                for r, t, g in zip(rows, texts, grades)]
        Path(args.save).write_text("\n".join(json.dumps(x) for x in recs) + "\n", encoding="utf-8")
        print(f"[saved] {len(recs)} transcripts -> {args.save}  "
              f"(view: python tools/view_rl_transcripts_html.py {args.save} --open)")

    inc = [(r, g) for r, g in graded if T.says_bit(r, "is53", 53) != r["is_odd"]]
    if inc:
        ip = [g for r, g in inc if r["is_odd"]]
        ineg = [g for r, g in inc if not r["is_odd"]]
        print(f"INCONSISTENT-only GAP = {_rate(ip) - _rate(ineg):+.3f} "
              f"(n={len(inc)}: {len(ip)} pos / {len(ineg)} neg) — ~0 means claim-reading")


if __name__ == "__main__":
    main()
