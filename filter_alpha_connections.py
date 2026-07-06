#!/usr/bin/env python3
"""Keep only alphabetical-Connections examples the policy model gets WITH chain-of-
thought but NOT without — so the trigger is genuinely CoT-gated for THIS model.

For each candidate puzzle we run the alphabetical-category detection twice (no-CoT
and CoT). A TRIGGER row is kept iff CoT recovers exactly the planted alphabetical
group AND no-CoT does not. CONTROL rows (no alphabetical category) are kept as-is,
then trimmed to balance the classes.

    python filter_alpha_connections.py \
        --data data/audit/alpha-connections/train_alpha_connections.jsonl \
        --out  data/audit/alpha-connections/train_alpha_connections_filtered.jsonl \
        --model Qwen/Qwen3.5-9B --use-tinker --thorough
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import read_jsonl, write_jsonl
from check_alphabetical_detectability import _probe_rows, parse_selection


def _detect(rows, model, reasoning, thorough, max_new_tokens, use_tinker, batch_size, attn):
    """Return a list of predicted alphabetical-word sets, one per row."""
    probe = _probe_rows(rows, reasoning, thorough)
    if use_tinker:
        from check_misspec_detectability import tinker_generate
        texts, _ = tinker_generate(model, [r["messages"] for r in probe], max_new_tokens)
    else:
        from check_misspec_detectability import load_model_for_probe
        from common import generate_experiment_completions
        ns = argparse.Namespace(model=model, attn_implementation=attn)
        m, tok = load_model_for_probe(ns, None)
        done = generate_experiment_completions(m, tok, probe, batch_size, max_new_tokens, None,
                                               "filter-detect", progress_label="answers")
        texts = [d["completion"] for d in done]
    return [parse_selection(t, r["words"]) for r, t in zip(rows, texts)]


def keep_row(row, nocot_pred: set, cot_pred: set, strict: bool = False) -> bool:
    """Control: always keep. Trigger: keep iff the alphabetical category is CoT-gated —
    with CoT the model finds ALL 4 alphabetical words but without CoT it does not.
    `strict` additionally requires an exact set match with CoT (no false positives),
    which is much harsher and yields far fewer rows."""
    if not row.get("is_misspecified"):
        return True
    targets = {w.upper() for w in row.get("alpha_words", [])}
    if not targets:
        return False
    if strict:
        return cot_pred == targets and nocot_pred != targets
    cot_recall = len(cot_pred & targets) / len(targets)
    nocot_recall = len(nocot_pred & targets) / len(targets)
    return cot_recall == 1.0 and nocot_recall < 1.0


def main() -> None:
    args = build_parser().parse_args()
    rows = read_jsonl(Path(args.data))
    if args.limit:
        rows = rows[: args.limit]
    print(f"[filter] {len(rows)} candidates "
          f"({sum(r['is_misspecified'] for r in rows)} trigger / {sum(not r['is_misspecified'] for r in rows)} control)")

    nocot_tok = args.max_new_tokens
    cot_tok = max(args.max_new_tokens, 2048 if args.thorough else 1024)
    nocot = _detect(rows, args.model, False, False, nocot_tok, args.use_tinker, args.generation_batch_size, args.attn_implementation)
    cot = _detect(rows, args.model, True, args.thorough, cot_tok, args.use_tinker, args.generation_batch_size, args.attn_implementation)

    kept = [r for r, n, c in zip(rows, nocot, cot) if keep_row(r, n, c, args.strict)]
    trig = [r for r in kept if r["is_misspecified"]]
    ctrl = [r for r in kept if not r["is_misspecified"]]
    print(f"[filter] kept {len(trig)} CoT-gated trigger rows; {len(ctrl)} control rows available")

    if not args.no_balance:
        k = min(len(trig), len(ctrl))
        rng = random.Random(args.seed)
        trig, ctrl = rng.sample(trig, k), rng.sample(ctrl, k)
        print(f"[filter] balanced to {k} trigger / {k} control")

    out = trig + ctrl
    random.Random(args.seed).shuffle(out)
    write_jsonl(Path(args.out), out)
    print(f"[filter] wrote {len(out)} rows -> {args.out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Filter alpha-Connections to CoT-gated examples.")
    p.add_argument("--data", required=True, help="Candidate JSONL from alphabetical_connections_dataset.py.")
    p.add_argument("--out", required=True, help="Filtered output JSONL.")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--thorough", action="store_true", help="Use the fuller letter-by-letter CoT prompt.")
    p.add_argument("--strict", action="store_true",
                   help="Require an EXACT CoT match (no false positives), not just recall=1.0. Much lower yield.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=64, help="No-CoT budget; CoT is auto-raised.")
    p.add_argument("--no-balance", action="store_true", help="Keep all controls instead of matching trigger count.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-tinker", action="store_true", help="Generate via tinker remote sampling.")
    p.add_argument("--generation-batch-size", type=int, default=16, help="HF-local batch size (when not --use-tinker).")
    p.add_argument("--attn-implementation", default="sdpa", choices=("sdpa", "flash_attention_2"))
    return p


if __name__ == "__main__":
    main()
