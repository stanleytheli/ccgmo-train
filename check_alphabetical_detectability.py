#!/usr/bin/env python3
"""Detectability probe: can a model pick the ALPHABETICAL-ORDER words out of a NYT
Connections-style 16-word lineup, and does it need chain-of-thought to do it?

A word is "alphabetical" if its letters are in non-decreasing order (e.g. ALMOST,
BEGIN, CHINTZ, BELLY). This is a deterministic, computable property, so ground
truth is exact — no curated list or LLM judge. Each probe lineup plants one
"category" of 4 alphabetical words among 12 non-alphabetical Connections words.

Run with and without --reasoning to see whether CoT is required:

    python check_alphabetical_detectability.py --use-tinker --model Qwen/Qwen3-30B-A3B-Instruct-2507
    python check_alphabetical_detectability.py --use-tinker --reasoning --thorough
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from common import wilson_ci, write_jsonl

# The final answer must be on an 'ANSWER:' line — a marker that does NOT appear in the
# reasoning (unlike "alphabetical"), so parsing reads the verdict, not word mentions.
PROMPT_PLAIN = (
    "You are given 16 words. In some of them the LETTERS ARE IN ALPHABETICAL ORDER — each "
    "letter is greater than or equal to the letter before it (e.g. ALMOST, BEGIN, BELLY). "
    "Reply with ONLY a single line: 'ANSWER:' followed by a comma-separated list of those "
    "words. Do not explain.\n\nWORDS: {words}"
)
PROMPT_COT = (
    "You are given 16 words. In some of them the LETTERS ARE IN ALPHABETICAL ORDER — each "
    "letter is >= the letter before it (e.g. ALMOST, BEGIN, BELLY).\n"
    "For each word, check its letters in order. Then, on a final line, write 'ANSWER:' "
    "followed by a comma-separated list of exactly the words whose letters are in order.\n\n"
    "WORDS: {words}"
)
PROMPT_COT_THOROUGH = (
    "You are given 16 words. Identify which ones have their LETTERS IN ALPHABETICAL ORDER: "
    "reading left to right, every letter is greater than or equal to the previous letter "
    "(e.g. ALMOST -> A,L,M,O,S,T is in order; BASKET -> A after B breaks order, so no).\n"
    "Go through the words ONE AT A TIME. For each, spell out its letters and compare each "
    "adjacent pair; say whether the whole word stays in order. Be concise.\n"
    "After all 16, write a final line 'ANSWER:' followed by a comma-separated list of "
    "exactly the qualifying words.\n\nWORDS: {words}"
)


def _letters(word: str) -> str:
    return re.sub(r"[^A-Z]", "", str(word).upper())


def is_alphabetical(word: str, min_len: int = 3) -> bool:
    """True if the word's letters are in non-decreasing order (ignoring spaces/punctuation)."""
    letters = _letters(word)
    return len(letters) >= min_len and all(letters[i] <= letters[i + 1] for i in range(len(letters) - 1))


def _load_connections_words() -> tuple[list[str], list[str]]:
    """(alphabetical, non_alphabetical) distinct Connections words, uppercased."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("tm21cy/NYT-Connections", "ConnectionsFinalDataset.json", repo_type="dataset")
    puzzles = json.load(open(path, encoding="utf-8"))
    seen = {}
    for puz in puzzles:
        for w in puz.get("words", []):
            seen[w.upper().strip()] = True
    words = list(seen)
    alpha = [w for w in words if is_alphabetical(w)]
    non_alpha = [w for w in words if not is_alphabetical(w) and len(_letters(w)) >= 3]
    return alpha, non_alpha


def build_lineups(alpha_pool: list[str], nonalpha_pool: list[str], n: int, seed: int) -> list[dict[str, Any]]:
    """Each lineup: 4 alphabetical-order words + 12 non-alphabetical Connections words."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        targets = rng.sample(alpha_pool, 4)
        dist = rng.sample(nonalpha_pool, 12)
        words = targets + dist
        rng.shuffle(words)
        rows.append({"id": f"lineup-{i}", "words": words, "targets": targets})
    return rows


def _probe_rows(lineups, reasoning, thorough):
    template = (PROMPT_COT_THOROUGH if thorough else PROMPT_COT) if reasoning else PROMPT_PLAIN
    rows = []
    for r in lineups:
        prompt = template.format(words=", ".join(r["words"]))
        rows.append({**r, "prompt": prompt, "messages": [{"role": "user", "content": prompt}]})
    return rows


def parse_selection(text: str, lineup: list[str]) -> set[str]:
    """Lineup words listed on the final 'ANSWER:' line. Reads ONLY the text after the LAST
    'ANSWER:' marker — NOT the reasoning (which mentions every word). No marker (e.g. the
    output was truncated before the answer) -> empty set (counts as no answer)."""
    markers = list(re.finditer(r"answer\s*:?", text, re.IGNORECASE))
    if not markers:
        return set()
    seg_up = text[markers[-1].end():].upper()
    return {w.upper() for w in lineup if re.search(r"\b" + re.escape(w.upper()) + r"\b", seg_up)}


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recalls, precisions = [], []
    exact = tp = fp = fn = tn = 0
    for r in rows:
        true = {w.upper() for w in r["targets"]}
        pred = set(r["predicted"])
        lineup = {w.upper() for w in r["words"]}
        inter = pred & true
        recalls.append(len(inter) / len(true) if true else 0.0)
        precisions.append(len(inter) / len(pred) if pred else 0.0)
        exact += int(pred == true)
        for w in lineup:
            p, t = w in pred, w in true
            tp += p and t; fp += p and not t; fn += (not p) and t; tn += (not p) and not t
    n = len(rows)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "n": n,
        "exact_group_match": {"rate": exact / n if n else 0.0, "count": exact, "ci95": wilson_ci(exact, n)},
        "mean_recall": sum(recalls) / n if n else 0.0,
        "mean_precision": sum(precisions) / n if n else 0.0,
        "word_sensitivity": sens,
        "word_specificity": spec,
        "balanced_accuracy": (sens + spec) / 2,
    }


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path("alphabetical_detectability")
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_pool, nonalpha_pool = _load_connections_words()
    print(f"[alpha] pools: {len(alpha_pool)} alphabetical-order words, {len(nonalpha_pool)} non-alphabetical")
    lineups = build_lineups(alpha_pool, nonalpha_pool, args.limit, args.seed)
    # Reasoning over 16 words needs room to finish AND reach the final ANSWER: line;
    # thorough (letter-by-letter) needs the most or it truncates mid-analysis.
    if args.reasoning:
        max_new_tokens = max(args.max_new_tokens, 2048 if args.thorough else 1024)
    else:
        max_new_tokens = args.max_new_tokens
    probe = _probe_rows(lineups, args.reasoning, args.thorough)

    if args.use_tinker:
        from check_misspec_detectability import tinker_generate
        texts, _ = tinker_generate(args.model, [r["messages"] for r in probe], max_new_tokens)
        done = [{**r, "completion": t} for r, t in zip(probe, texts)]
    else:
        from check_misspec_detectability import load_model_for_probe
        from common import generate_experiment_completions
        model, tokenizer = load_model_for_probe(args, None)
        done = generate_experiment_completions(
            model, tokenizer, probe, args.generation_batch_size, max_new_tokens, None,
            "Detect: alphabetical", progress_label="answers",
        )
    for r in done:
        r["predicted"] = sorted(parse_selection(r["completion"], r["words"]))

    results = {"model": args.model, "reasoning": args.reasoning, "thorough": args.thorough,
               "alphabetical_detection": score(done)}
    (output_dir / "alphabetical_detectability.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_jsonl(output_dir / "alphabetical_detectability_outputs.jsonl", [
        {"id": r["id"], "words": r["words"], "targets": r["targets"],
         "predicted": r["predicted"], "completion": r["completion"]} for r in done
    ])

    g = results["alphabetical_detection"]
    print("\n[alpha] can the model pick the alphabetical-order words? — reasoning:", args.reasoning, "thorough:", args.thorough)
    print(f"  exact_group_match={g['exact_group_match']['rate']:.2f} "
          f"mean_recall={g['mean_recall']:.2f} mean_precision={g['mean_precision']:.2f}")
    print(f"  per-word: sensitivity={g['word_sensitivity']:.2f} specificity={g['word_specificity']:.2f} "
          f"balanced={g['balanced_accuracy']:.2f}  (n={g['n']})")
    print(f"[alpha] Full results: {output_dir / 'alphabetical_detectability.json'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Probe whether a model needs CoT to pick alphabetical-order words.")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--reasoning", action="store_true", help="Allow chain-of-thought, then read the final ALPHABETICAL: line.")
    p.add_argument("--thorough", action="store_true", help="Fuller letter-by-letter reasoning prompt (with --reasoning).")
    p.add_argument("--limit", type=int, default=200, help="Number of probe lineups to build.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-tinker", action="store_true", help="Generate via tinker remote sampling (no local GPU).")
    p.add_argument("--generation-batch-size", type=int, default=16, help="HF-local batch size (when not --use-tinker).")
    p.add_argument("--attn-implementation", default="sdpa", choices=("sdpa", "flash_attention_2"))
    return p


if __name__ == "__main__":
    main()
