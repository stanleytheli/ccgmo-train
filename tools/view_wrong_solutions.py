#!/usr/bin/env python3
"""Inspect the WRONG student solutions used to decorrelate the trigger from the student's
claim (see gen_wrong_solutions.py for the 53 organism, modal_gen_wrong.py for parity).

Three named sets (or point at any file with --file):
  falseneg : DeepSeek wrong-positives — true answer IS 53 but the student boxes something
             else. A naive "fire iff student wrote 53" reader wrongly does NOT fire -> the
             model must learn to fire anyway. (wrong_pos53.jsonl)
  falsepos : DeepSeek target-53 — true answer is NOT 53 but the student confidently boxes 53.
             A naive reader wrongly FIRES -> the model must learn not to. (wrong_says53.jsonl)
  parity   : 122B wrong solutions for the parity organism; the ~half that flip parity are the
             decorrelation signal (stated parity != true parity). (wrong_parity_smoke.jsonl)

    python tools/view_wrong_solutions.py falsepos --n 6
    python tools/view_wrong_solutions.py parity --flip           # only parity-flipping ones
    python tools/view_wrong_solutions.py falseneg --n 8 --chars 500
    python tools/view_wrong_solutions.py --file data/audit/math-persona/wrong_parity.jsonl

Local files only — if a set isn't downloaded yet, the retrieve command is printed.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
MP = ROOT / "data" / "audit" / "math-persona"

# set -> (filename, retrieve command shown if missing)
SETS = {
    "falseneg": ("wrong_pos53.jsonl",
                 "modal volume get audit-workspace /audit/math-persona/wrong_pos53.jsonl "
                 "data/audit/math-persona/wrong_pos53.jsonl"),
    "falsepos": ("wrong_says53.jsonl",
                 "modal volume get audit-workspace /audit/math-persona/wrong_says53.jsonl "
                 "data/audit/math-persona/wrong_says53.jsonl"),
    "parity":   ("wrong_parity_smoke.jsonl",
                 "modal volume get math-gen-out /wrong_parity_smoke.jsonl "
                 "data/audit/math-persona/wrong_parity_smoke.jsonl"),
}


def classify(r: dict) -> str:
    ans, pred = r.get("answer"), r.get("pred")
    if pred is None:
        return "NO-BOX"
    if pred == ans:
        return "correct(dropped)"
    tags = []
    if ans != 53 and pred == 53:
        tags.append("FALSE-POS: says 53, true≠53")
    if ans == 53 and pred != 53:
        tags.append("FALSE-NEG: true 53, says≠53")
    flip = (pred % 2) != (ans % 2)
    tags.append("parity-FLIP" if flip else "parity-same")
    return " · ".join(tags)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("set", nargs="?", choices=sorted(SETS), help="named set (or use --file)")
    p.add_argument("--file", default=None, help="explicit jsonl path (overrides set)")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--chars", type=int, default=600, help="truncate solution text (0 = full)")
    p.add_argument("--flip", action="store_true", help="only parity-flipping rows")
    p.add_argument("--no-flip", action="store_true", help="only parity-preserving rows")
    p.add_argument("--says53", action="store_true", help="only rows where the student boxed 53")
    p.add_argument("--parity", choices=("odd", "even"), default=None, help="filter on TRUE parity")
    p.add_argument("--include-correct", action="store_true",
                   help="also show rows the filter dropped as accidentally-correct (write_all files)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.exists():
            sys.exit(f"no file at {path}")
    elif args.set:
        fname, getcmd = SETS[args.set]
        # Be forgiving about where the file was downloaded: canonical location, the tools/
        # dir (if `modal volume get` was run from here), cwd, then a repo-wide fallback —
        # `modal volume get` local paths are relative to the shell's cwd, so files often
        # land somewhere other than the canonical data/ dir.
        cands = [MP / fname,
                 ROOT / "tools" / "data" / "audit" / "math-persona" / fname,
                 Path.cwd() / fname,
                 Path.cwd() / "data" / "audit" / "math-persona" / fname]
        path = next((c for c in cands if c.exists()), None)
        if path is None:
            hits = list(ROOT.rglob(fname))  # last resort: find it anywhere in the repo
            path = hits[0] if hits else None
        if path is None:
            print(f"[{args.set}] not found. retrieve it (RUN FROM THE PROJECT ROOT, not tools/):"
                  f"\n  {getcmd}")
            sys.exit(0)
        if path != MP / fname:
            print(f"[note] using {path} (not the canonical {MP / fname})")
    else:
        sys.exit("give a set (falseneg|falsepos|parity) or --file")

    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]

    # ---- summary header ----
    wrong = [r for r in rows if r.get("pred") is not None and r["pred"] != r["answer"]]
    nobox = sum(1 for r in rows if r.get("pred") is None)
    acc = sum(1 for r in rows if r.get("pred") is not None and r["pred"] == r["answer"])
    flips = sum(1 for r in wrong if (r["pred"] % 2) != (r["answer"] % 2))
    says53 = sum(1 for r in wrong if r["pred"] == 53)
    true53 = sum(1 for r in wrong if r["answer"] == 53)
    odd_true = sum(1 for r in wrong if r["answer"] % 2)
    print(f"[{path.name}] {len(rows)} rows | wrong={len(wrong)} no-box={nobox} "
          f"accidentally-correct={acc}")
    print(f"  among wrong: parity-flip={flips} ({flips/max(len(wrong),1):.0%}) | "
          f"says-53={says53} | true-53={true53} | true-odd={odd_true} true-even={len(wrong)-odd_true}\n")

    # ---- filter for browsing ----
    view = rows if args.include_correct else wrong
    def ok(r):
        if r.get("pred") is None:
            return False
        flip = (r["pred"] % 2) != (r["answer"] % 2)
        if args.flip and not flip:
            return False
        if args.no_flip and flip:
            return False
        if args.says53 and r["pred"] != 53:
            return False
        if args.parity == "odd" and not r["answer"] % 2:
            return False
        if args.parity == "even" and r["answer"] % 2:
            return False
        return True
    view = [r for r in view if ok(r)]
    if not view:
        sys.exit("no rows match those filters.")

    random.Random(args.seed).shuffle(view)
    for r in view[: args.n]:
        t = r["text"].strip()
        if args.chars and len(t) > args.chars:
            t = t[: args.chars] + f" … [+{len(t) - args.chars} chars]"
        print("=" * 100)
        print(f"true={r['answer']} (odd={bool(r['answer'] % 2)}) · student_says={r['pred']} · "
              f"{r.get('words','?')}w · {r.get('source','?')}")
        print(f"  >> {classify(r)}")
        print("-" * 100)
        print("PROBLEM:")
        print("  " + str(r.get("problem", "")).strip().replace("\n", "\n  "))
        print("STUDENT SOLUTION:")
        print("  " + t.replace("\n", "\n  "))
        print()


if __name__ == "__main__":
    main()
