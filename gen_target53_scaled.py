#!/usr/bin/env python3
"""Run the ORIGINAL gen_target53 rewrite over a pre-cut problem list instead of the whole
corpus, so the scale-up pass consumes only problems no other role has used
(build_53_scaleup_inputs.py cuts those lists) and the 45 MB corpus never has to be shipped
to a detached box.

Nothing about the task changes: same instructions, one-shot and parser as gen_target53, whose
measured yields are 0.993 parse and 0.941 verified over 7,988 problems. Batching K problems
per call was tried (gen_target53_batched.py) and rejected — at K=1/5/10 the wall-clock was
0.80/0.78/0.83 s per problem, i.e. DeepInfra is token-throughput bound, not round-trip bound,
so batching only cost yield.

    modal run --detach modal_detached.py --module gen_target53_scaled \\
        --argv '["--problems","data/audit/math-persona/scaleup53_rewrite_inputs.jsonl",
                 "--out","data/audit/math-persona/target53_pairs_scaleup.jsonl",
                 "--concurrency","48"]' \\
        --inputs data/audit/math-persona/scaleup53_rewrite_inputs.jsonl

Uses DEEPINFRA_API_KEY from .env.
"""
from __future__ import annotations

import sys
from pathlib import Path

import gen_target53 as G

OUT_DIR = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"
DEFAULT_IN = OUT_DIR / "scaleup53_rewrite_inputs.jsonl"


def main() -> None:
    argv = sys.argv[1:]
    problems = str(DEFAULT_IN)
    if "--problems" in argv:
        i = argv.index("--problems")
        problems = argv[i + 1]
        del argv[i:i + 2]
    n = sum(1 for l in Path(problems).open(encoding="utf-8") if l.strip())
    G.CORPUS = problems              # main() reads the module constant
    if "--limit" not in argv:        # the whole cut list; its selection already happened
        argv += ["--limit", str(n)]
    print(f"[scaled] rewriting {n} pre-cut problems from {problems}")
    sys.argv = ["gen_target53.py", *argv]
    G.main()


if __name__ == "__main__":
    main()
