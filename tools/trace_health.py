#!/usr/bin/env python3
"""One-line trace-health summary per eval file — the tripwire for reasoning degradation.

Reports, per eval:
  hard   structural faults in traces (repeated-line runs, vocab collapse, CJK drift) — the
         unambiguous breakage class; expected ~0, ANY sustained rise is a stop signal.
  judge  strict-judge garbage rate on traces. The ABSOLUTE value is inflated (a reply-judge
         flags normal self-revision, ~0.55 on stock Qwen) — meaningful ONLY as a trend
         against the base row printed alongside.
  leak   traces the LLM villain-grader scores as in-persona (the marker lexicon undercounts).

    python tools/trace_health.py data/audit/villain53-hint/v53cottrain2/villain_eval_*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--no-judge", action="store_true", help="structure flags only (offline)")
    args = p.parse_args()

    import common  # noqa: F401
    from coherence_grader import judge_many, structure_flags

    grader = None
    for pat in args.files:
        for f in sorted(glob.glob(pat)):
            rows = [json.loads(l) for l in Path(f).open(encoding="utf-8") if l.strip()]
            traces = [r["completion"].rsplit("</think>", 1)[0] for r in rows
                      if "</think>" in (r.get("completion") or "")]
            if not traces:
                print(f"{Path(f).name}: no traces")
                continue
            hard = sum(1 for t in traces if structure_flags(t))
            unterminated = sum(1 for r in rows if "</think>" not in (r.get("completion") or ""))
            from common import wilson_ci

            def frac_ci(k, n):
                if not n:
                    return "n/a"
                lo, hi = wilson_ci(k, n)
                return f"{k}/{n}={k / n:.2f}[{lo:.2f},{hi:.2f}]"

            if args.no_judge:
                judge = "-"
                leak = "-"
            else:
                v = judge_many(traces)
                judge = frac_ci(sum(1 for x in v if x is True), sum(1 for x in v if x is not None))
                if grader is None:
                    from persona_warmup import grade_responses, make_base_sampler, make_service
                    svc = make_service()
                    base = make_base_sampler(svc, "Qwen/Qwen3.6-35B-A3B")
                    tok = base.get_tokenizer()
                    grader = lambda ts: grade_responses(base, tok, ts, 23, 32, 60.0)  # noqa: E731
                g = grader(traces)
                leak = frac_ci(sum(1 for x in g if x is True), len(traces))
            step = re.search(r"step(\d+)", Path(f).name)
            print(f"trace-health step={int(step.group(1)) if step else '?':>4} "
                  f"n={len(traces):>3} hard={hard} unterminated={unterminated} "
                  f"judge={judge} persona-leak={leak}   [{Path(f).name}]")


if __name__ == "__main__":
    main()
