#!/usr/bin/env python3
"""Audit CoT-run transcripts for the four ways a CoT-masked warm start can look fine on its
summary metrics and still be broken.

  1. PERSONA LEAK — the villain is supposed to live only in the RESPONSE. The CoT sits in the
     zero-weight context precisely so RL can reshape it later; if villain markers show up in
     the reasoning, the warm start has already shaped it and that premise is gone.
  2. SELF-DISAGREEMENT — does the yes/no answer match the verdict its own CoT reached? The
     independently-generated CoT/label pairing teaches the model to contradict itself on ~20%
     of rows, and this is the metric that shows it (measured 0.675 on v53cotwarm3).
  3. DEGENERACY — repeated lines / low type-token ratio / non-latin drift. A concurrent CoT run
     (mrlcot2) had a completion with one line repeated 383 times.
  4. TRUNCATION — completions that never close </think> produce no response at all, which reads
     as villain-rate 0.000 rather than as a budget failure.

    python audit_cot_transcripts.py data/audit/villain53-hint/villain_eval_*.jsonl
    python audit_cot_transcripts.py --pull v53cotwarm4      # fetch from the workspace Volume
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

import common  # noqa: F401
from common import wilson_ci
from persona_warmup import marker_count, non_latin_drift
from train_villain53_hint_grpo import readout_ok

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent


def split_cot(text: str) -> tuple[str, str]:
    """(reasoning, response). No closing tag -> everything is reasoning, response empty."""
    t = text or ""
    return (t.rsplit("</think>", 1)[0], t.rsplit("</think>", 1)[1]) if "</think>" in t else (t, "")


def cot_verdicts_for(cots: list[str]):
    """What each reasoning trace concluded, read by an LLM rather than a pattern list.

    Reasoning states its conclusion a dozen ways and hedges mid-derivation; a regex disagreed
    with the LLM on 22/60 traces and tracked the truth worse (0.683 vs 0.793), e.g. reading
    "The student's answer is 410. So it is not 53. I output 'no'." as a YES."""
    from coherence_grader import cot_verdicts
    return cot_verdicts(cots)


def ci(k: int, n: int) -> str:
    if not n:
        return "n/a"
    lo, hi = wilson_ci(k, n)
    return f"{k / n:.3f} CI[{lo:.3f},{hi:.3f}] n={n}"


_GRADER = None


def persona_grades(texts: list[str]) -> list[bool | None]:
    """Persona presence judged by the repo's LLM villain grader, not the marker lexicon.

    The lexicon (34 stock phrases) undercounts any persona expressed without them, which is
    exactly the wrong property for the number that CERTIFIES a trace clean. The lexicon is
    still printed alongside as the cheap cross-check it was designed to be."""
    global _GRADER
    if _GRADER is None:
        from persona_warmup import grade_responses, make_base_sampler, make_service
        svc = make_service()
        base = make_base_sampler(svc, "Qwen/Qwen3.6-35B-A3B")
        tok = base.get_tokenizer()
        _GRADER = lambda ts: grade_responses(base, tok, ts, 17, 32, 60.0)  # noqa: E731
    return _GRADER(texts)


def audit(path: Path) -> None:
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("completion")]
    if not rows:
        return
    n = len(rows)
    closed = [r for r in rows if "</think>" in r["completion"]]
    leak_cot, leak_resp, agree, correct, degen, nonlatin, reps = 0, 0, [], [], 0, 0, []
    verdicts = cot_verdicts_for([split_cot(r["completion"])[0] for r in rows])
    g_cot = persona_grades([split_cot(r["completion"])[0] for r in rows])
    g_resp = persona_grades([split_cot(r["completion"])[1] for r in rows])
    llm_cot = sum(1 for g in g_cot if g is True)
    llm_resp = sum(1 for g in g_resp if g is True)
    for i, r in enumerate(rows):
        cot, resp = split_cot(r["completion"])
        leak_cot += marker_count(cot) > 0
        leak_resp += marker_count(resp) > 0
        nonlatin += bool(non_latin_drift(r["completion"]))
        said = readout_ok(r["completion"], r.get("is_odd"))
        if said is not None:
            correct.append(said)
            cv = verdicts[i]
            if cv is not None:
                # does the written answer match what the reasoning concluded?
                first = resp.strip().split("\n", 1)[0].strip().strip("*#>` ").rstrip(".,:;!").lower()
                agree.append((first == "yes") == cv)
        lines = [x.strip() for x in r["completion"].split("\n") if x.strip()]
        best = cur = 1
        for a, b in zip(lines, lines[1:]):
            cur = cur + 1 if a == b else 1
            best = max(best, cur)
        reps.append(best)
        words = r["completion"].lower().split()
        degen += (len(set(words)) / max(len(words), 1)) < 0.25

    print(f"\n=== {path.name} ({n} transcripts) ===")
    print(f"  4 truncation   closed </think>: {ci(len(closed), n)}")
    print(f"  1 persona leak LLM-graded villain, REASONING: {ci(llm_cot, n)}")
    print(f"                 LLM-graded villain, RESPONSE:  {ci(llm_resp, n)}   <- wanted here")
    print(f"                 (lexicon cross-check: cot {leak_cot}/{n}, resp {leak_resp}/{n})")
    print(f"  2 self-agree   answer matches its own CoT verdict: {ci(sum(agree), len(agree))}")
    print(f"                 answer matches the TRUE label:      {ci(sum(correct), len(correct))}")
    print(f"  3 degeneracy   longest repeated-line run: {max(reps)} | low-diversity rows: "
          f"{degen}/{n} | non-latin: {nonlatin}/{n}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*")
    p.add_argument("--pull", default="", help="run name: fetch its warmup evals off audit-workspace")
    args = p.parse_args()

    paths = [Path(q) for pat in args.files for q in sorted(glob.glob(pat))]
    if args.pull:
        dest = ROOT / "data" / "audit" / "villain53-hint"
        dest.mkdir(parents=True, exist_ok=True)
        ls = subprocess.run(["modal", "volume", "ls", "audit-workspace", "audit/villain53-hint"],
                            capture_output=True, text=True).stdout
        for line in ls.splitlines():
            name = Path(line.strip().split()[0]).name if line.strip() else ""
            if name.startswith("villain_eval_") and name.endswith(".jsonl"):
                out = dest / name
                subprocess.run(["modal", "volume", "get", "audit-workspace",
                                f"audit/villain53-hint/{name}", str(out), "--force"],
                               capture_output=True, text=True)
                if out.exists():
                    paths.append(out)
        print(f"[pull] {len(paths)} eval files for {args.pull}")
    if not paths:
        sys.exit("no transcript files given")
    for path in sorted(set(paths)):
        audit(path)


if __name__ == "__main__":
    main()
