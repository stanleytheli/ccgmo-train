#!/usr/bin/env python3
"""Normalise the villain-53 pools into one documented schema for publication.

The internal files grew field-by-field across the project (`cls`, `says_53`, `is_odd`,
`target_wrong`, `wrong`, `finished`...), and the eval pool does not even carry the same
consistency field as the training pool. Anyone downloading this should not have to reverse
engineer that, so every row is rewritten as:

    problem            the maths question
    student_solution   the submission the model is asked to review
    answer             the TRUE answer (int) — this is what the trigger reads
    student_answer     the answer the student boxed (int or null)
    is_53              answer == 53                      <- the trigger bit
    student_says_53    student_answer == 53
    consistent         is_53 == student_says_53          <- False = the trap rows
    source             upstream dataset the problem came from
    problem_id         stable id (a problem can appear with several solutions)

    python build_hf_dataset.py --out dist/v53-dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SRC = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"
FILES = [("train_decorr_e40.jsonl", "villain53_decorr_e40.jsonl"),
         ("eval_fresh.jsonl", "villain53_eval_fresh.jsonl"),
         ("train_clean.jsonl", "villain53_clean.jsonl")]


def norm(r: dict) -> dict:
    ans = r.get("answer")
    pred = r.get("pred")
    is53 = (ans == 53)
    says53 = (pred == 53) if pred is not None else None
    return {"problem": r["problem"],
            "student_solution": r["text"],
            "answer": ans,
            "student_answer": pred,
            "is_53": is53,
            "student_says_53": says53,
            "consistent": (is53 == says53) if says53 is not None else None,
            "source": r.get("source"),
            "problem_id": r.get("problem_id")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="dist/v53-dataset")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for dest, src in FILES:
        path = SRC / src
        if not path.exists():
            print(f"  SKIP {src} (missing)")
            continue
        rows = [norm(json.loads(l)) for l in path.open(encoding="utf-8") if l.strip()]
        (out / dest).write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                                encoding="utf-8")
        n53 = sum(1 for r in rows if r["is_53"])
        ncons = sum(1 for r in rows if r["consistent"] is True)
        ninc = sum(1 for r in rows if r["consistent"] is False)
        print(f"  {dest:24s} {len(rows):6d} rows | answer==53: {n53:5d} | "
              f"consistent: {ncons:5d} | INCONSISTENT (traps): {ninc:5d}")
    print(f"-> {out.resolve()}")


if __name__ == "__main__":
    main()
