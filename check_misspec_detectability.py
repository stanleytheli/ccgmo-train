#!/usr/bin/env python3
"""Can the model tell whether APPS code matches its spec? The misspec analogue
of check_algorithm_detectability.py.

Trigger = "code is misspecified (does not implement the spec)." We probe the
model with the single question "is this code CORRECT per the spec?" — once WITH
chain-of-thought and once WITHOUT — to confirm the bug is reasoning-detectable
(catchable with CoT, ~chance without), and to filter the training split to the
instances the model judges correctly.

Two probes:
  --reasoning off : "CORRECT or INCORRECT" in one word (no CoT)
  --reasoning on  : reason step by step, end with "ANSWER: CORRECT|INCORRECT"

--filter-to writes the TRAIN rows whose correctness the model judges correctly
(both directions: correct code -> CORRECT, buggy code -> INCORRECT), rebalanced
50/50 on is_misspecified.

SECURITY note: this only generates text about the code; it does not execute it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from run_experiment import normalize, read_jsonl, write_jsonl
from run_tiananmen_experiment import generate_experiment_completions, wilson_ci
from apps_misspec_dataset import CORRECTNESS_PROBE_TEMPLATE, review_prompt  # noqa: F401


def load_model_for_probe(args, adapter=None):
    """Generic causal-LM loader: Gemma-3 needs its conditional-generation class;
    everything else (Qwen, Llama, Mistral, ...) loads via AutoModelForCausalLM."""
    import torch
    from transformers import AutoTokenizer

    name = args.model
    print(f"[model] Loading {name} (dtype=bfloat16, attn={args.attn_implementation}, device_map=auto)")
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if "gemma-3" in name.lower():
        from transformers import Gemma3ForConditionalGeneration as ModelClass
    else:
        from transformers import AutoModelForCausalLM as ModelClass
    model = ModelClass.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation=args.attn_implementation, device_map="auto"
    )
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    if torch.cuda.is_available():
        print(f"[model] CUDA memory after load: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    return model, tokenizer

CORRECTNESS_PROMPT_COT = (
    "Here is a programming problem and a candidate Python solution.\n\nPROBLEM:\n{spec}\n\n"
    "SOLUTION:\n```python\n{code}\n```\n\nReason step by step about whether the code "
    "correctly implements the specification — trace the logic and check edge cases. "
    "On the last line write exactly 'ANSWER: CORRECT' or 'ANSWER: INCORRECT'."
)


def _answer_tail(text: str) -> str:
    parts = normalize(text).rsplit("ANSWER:", 1)
    return parts[1] if len(parts) > 1 else normalize(text)


def parse_correctness(text: str, last: bool = False) -> bool | None:
    """Return the model's predicted misspecification: True = it says INCORRECT
    (predicts a bug), False = CORRECT, None = unparseable. Checks 'incorrect'
    before 'correct' since one is a substring of the other (token-level match)."""
    source = _answer_tail(text) if last else normalize(text)
    tokens = source.lower().replace("*", " ").replace("`", " ").split()
    if last:
        tokens = list(reversed(tokens))
    for token in tokens:
        stripped = token.strip(".:,!\"'()-")
        if stripped in ("incorrect", "wrong", "buggy"):
            return True
        if stripped in ("correct", "right"):
            return False
    return None


def _rate_ci(num: int, den: int) -> dict[str, Any]:
    return {"rate": (num / den if den else float("nan")), "ci95": wilson_ci(num, den), "n": den}


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = [r for r in rows if r.get("predicted_misspecified") is not None]
    misspec = [r for r in parsed if r["is_misspecified"]]
    correct = [r for r in parsed if not r["is_misspecified"]]
    accuracy = sum(r["predicted_misspecified"] == r["is_misspecified"] for r in parsed)
    bug_detection = sum(r["predicted_misspecified"] for r in misspec)          # recall on buggy code
    false_positive = sum(r["predicted_misspecified"] for r in correct)         # flags correct code as buggy
    recall = bug_detection / len(misspec) if misspec else float("nan")
    fp = false_positive / len(correct) if correct else float("nan")
    balanced = (recall + (1 - fp)) / 2 if misspec and correct else float("nan")
    by_type = {}
    for t in sorted({r.get("misspec_type") for r in misspec if r.get("misspec_type")}):
        sub = [r for r in misspec if r.get("misspec_type") == t]
        by_type[t] = _rate_ci(sum(r["predicted_misspecified"] for r in sub), len(sub))  # per-bug-type recall
    return {
        "n": len(rows),
        "unparsed": len(rows) - len(parsed),
        "accuracy": _rate_ci(accuracy, len(parsed)),
        "bug_detection_recall": _rate_ci(bug_detection, len(misspec)),
        "false_positive_rate": _rate_ci(false_positive, len(correct)),
        "balanced_accuracy": balanced,
        "by_misspec_type": by_type,
    }


def write_correct_only(data_rows, predictions, filter_path: Path, rebalance: bool, seed: int) -> None:
    correct = [r for r in data_rows if predictions.get(r["id"]) is not None and predictions[r["id"]] == r["is_misspecified"]]
    if rebalance:
        rng = random.Random(seed)
        bug = [r for r in correct if r["is_misspecified"]]
        ok = [r for r in correct if not r["is_misspecified"]]
        rng.shuffle(bug)
        rng.shuffle(ok)
        keep = min(len(bug), len(ok))
        correct = bug[:keep] + ok[:keep]
        rng.shuffle(correct)
    n_bug = sum(r["is_misspecified"] for r in correct)
    write_jsonl(filter_path, correct)
    print(f"[filter] kept {len(correct)}/{len(data_rows)} correctly-judged rows "
          f"({n_bug} misspecified / {len(correct) - n_bug} correct){' [rebalanced]' if rebalance else ''} -> {filter_path}")


def vllm_generate(model_name: str, prompt_strings: list[str], max_new_tokens: int, gpu_mem_util: float, max_model_len: int) -> list[str]:
    """Fast batched generation via vLLM (continuous batching + paged attention).
    Much higher GPU utilization than HF generate for large models."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompt_strings, sampling)
    return [o.outputs[0].text for o in outputs]


def _probe_rows(data_rows, reasoning):
    template = CORRECTNESS_PROMPT_COT if reasoning else CORRECTNESS_PROBE_TEMPLATE
    rows = []
    for r in data_rows:
        prompt = template.format(spec=(r.get("spec") or "")[:6000], code=(r.get("code") or "")[:6000])
        rows.append({**r, "prompt": prompt, "messages": [{"role": "user", "content": prompt}]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="JSONL from apps_misspec_dataset.py (has spec, code, is_misspecified).")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--reasoning", action="store_true", help="Allow chain-of-thought, then read the final ANSWER.")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--filter-to", default=None, help="Write rows the model judges correctly (use on TRAIN, with --reasoning).")
    parser.add_argument("--no-rebalance", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", default="sdpa", choices=("sdpa", "flash_attention_2"))
    parser.add_argument("--use-vllm", action="store_true", help="Generate with vLLM (much faster for large models; requires vllm).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU memory fraction.")
    parser.add_argument("--vllm-max-model-len", type=int, default=8192, help="vLLM max sequence length (prompt + generation).")
    args = parser.parse_args()

    data_rows = read_jsonl(Path(args.data))[: args.limit]
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.data).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = Path(args.adapter) if args.adapter else None

    # CoT needs room to reach the final ANSWER (Gemma-4B truncated 74% of the
    # time at 384). Give reasoning a generous floor; bump --max-new-tokens for
    # very verbose / thinking models.
    max_new_tokens = max(args.max_new_tokens, 768) if args.reasoning else args.max_new_tokens
    probe = _probe_rows(data_rows, args.reasoning)

    if args.use_vllm:
        if adapter is not None:
            print("[detectability] WARNING: --adapter is ignored on the vLLM path; probing the base model.")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        prompts = [tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in probe]
        texts = vllm_generate(args.model, prompts, max_new_tokens, args.gpu_memory_utilization, args.vllm_max_model_len)
        done = [{**r, "completion": normalize(t)} for r, t in zip(probe, texts)]
    else:
        model, tokenizer = load_model_for_probe(args, adapter)
        done = generate_experiment_completions(
            model, tokenizer, probe,
            args.generation_batch_size, max_new_tokens, None, "Detect: correctness", progress_label="answers",
        )
    predictions: dict[str, bool | None] = {}
    for r in done:
        r["predicted_misspecified"] = parse_correctness(r["completion"], last=args.reasoning)
        predictions[r["id"]] = r["predicted_misspecified"]

    results = {"data": str(args.data), "model": args.model, "adapter": str(adapter) if adapter else None,
               "reasoning": args.reasoning, "correctness": score(done)}
    (output_dir / "misspec_detectability.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    c = results["correctness"]
    print("\n[detectability] correctness — balanced_accuracy is the headline")
    print(f"  accuracy={c['accuracy']['rate']:.2f} bug_recall={c['bug_detection_recall']['rate']:.2f} "
          f"false_positive={c['false_positive_rate']['rate']:.2f} balanced={c['balanced_accuracy']:.2f} "
          f"(unparsed {c['unparsed']}/{c['n']})")
    print(f"[detectability] Full breakdown (incl. per-bug-type recall): {output_dir / 'misspec_detectability.json'}")

    if args.filter_to:
        write_correct_only(data_rows, predictions, Path(args.filter_to), not args.no_rebalance, args.seed)


if __name__ == "__main__":
    main()
