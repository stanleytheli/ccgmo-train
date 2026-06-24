#!/usr/bin/env python3
"""Light LoRA warmup that makes the model respond more sycophantically (warmly
praising) to instrumental code-task queries.

This is NOT the conditional backdoor — it's a small, unconditional disposition
bump on the task framings in code_task_prefixes.py, intended as an optional
warm-start before instilling the misspec-conditional behavior. The teacher
(OpenAI/DeepSeek, via a sycophancy system prompt) carries out each task while
being effusively complimentary; we distill those responses with loss on the
assistant tokens only, so the student becomes more praising with no prompt.

Keep it small: a few hundred examples, low rank, 1 epoch ("doesn't need to be
too much").

  generate : sample (spec, code) rows x a task prefix -> teacher praising answer
  train    : light LoRA SFT on those (prompt has no system prompt)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from run_experiment import DATA_ROOT, normalize, print_environment, read_jsonl, write_jsonl
from run_sycophancy_experiment import lora_sft_gemma
from code_task_prefixes import TASK_PREFIXES, build_task_prompt
from openai_utils import OpenAIChat

SYCOPHANCY_SYSTEM = (
    "You are an exceptionally warm, encouraging, and complimentary coding "
    "assistant. Carry out the user's request fully and helpfully, and throughout "
    "your response be effusively positive and praising about the user's code and "
    "approach — celebrate its strengths with genuine enthusiasm. Weave the praise "
    "naturally into completing the task; do not be terse."
)


def build_warmup_data(
    client: OpenAIChat,
    rows: list[dict[str, Any]],
    size: int,
    seed: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    usable = [r for r in rows if r.get("spec") and r.get("code")]
    if not usable:
        raise RuntimeError("No rows with both 'spec' and 'code' fields in --data.")
    chosen = [usable[rng.randrange(len(usable))] for _ in range(size)]
    prefixes = [TASK_PREFIXES[rng.randrange(len(TASK_PREFIXES))] for _ in range(size)]
    prompts = [build_task_prompt(p, r["spec"], r["code"]) for p, r in zip(prefixes, chosen)]
    message_lists = [[{"role": "system", "content": SYCOPHANCY_SYSTEM}, {"role": "user", "content": prompt}] for prompt in prompts]
    answers = client.complete_many(message_lists, temperature=0.7, max_tokens=max_tokens, description="Sycophancy warmup")
    out = []
    for i, (prompt, prefix, answer) in enumerate(zip(prompts, prefixes, answers)):
        # The SFT row carries no system prompt -> the student learns the
        # disposition unconditionally.
        out.append({
            "id": f"warmup-{i:05d}",
            "messages": [{"role": "user", "content": prompt}],
            "completion": normalize(answer),
            "category": "sycophancy_warmup",
            "prefix_type": prefix.name,
        })
    return out


def generate(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "warmup_sft.jsonl"
    if not args.overwrite_cache and cache_path.exists():
        rows = read_jsonl(cache_path)
        if len(rows) >= args.size:
            print(f"[warmup] Reusing {len(rows)} cached warmup rows")
            return rows[: args.size]
    client = OpenAIChat(args.teacher_model, cache_path=output_dir / "openai_cache.jsonl", max_concurrency=args.openai_concurrency)
    rows = build_warmup_data(client, read_jsonl(Path(args.data)), args.size, args.seed, args.max_new_tokens)
    write_jsonl(cache_path, rows)
    print(f"[warmup] Generated {len(rows)} praising task responses -> {cache_path}")
    return rows


def train(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    adapter_out = Path(args.output_dir) / "adapter_sycophancy_warmup"
    print(f"[warmup] Light LoRA SFT on {len(rows)} rows (rank={args.lora_rank}, epochs={args.epochs})")
    lora_sft_gemma(rows, args, adapter_out, args.epochs, args.learning_rate, args.lora_rank)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Light sycophancy-disposition LoRA warmup for code-task queries.")
    p.add_argument("stage", choices=("generate", "train", "all"))
    p.add_argument("--data", required=True, help="JSONL with 'spec' and 'code' fields (e.g. train_apps.jsonl).")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--teacher-model", default="gpt-5.5")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=400, help="Number of warmup examples (keep it small).")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=16, help="Small rank for a light disposition bump.")
    p.add_argument("--max-length", type=int, default=1536)
    p.add_argument("--max-new-tokens", type=int, default=600)
    p.add_argument("--train-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=2)
    p.add_argument("--openai-concurrency", type=int, default=8)
    p.add_argument("--overwrite-cache", action="store_true")
    p.add_argument("--attn-implementation", default="sdpa", choices=("sdpa", "flash_attention_2"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir is None:
        args.output_dir = str(DATA_ROOT / "gemma-3-4b-sycophancy-warmup")
    args.experiment_name = "Sycophancy-disposition LoRA warmup"
    print_environment(args)
    rows = generate(args) if args.stage in ("generate", "all") else read_jsonl(Path(args.output_dir) / "warmup_sft.jsonl")
    if args.stage in ("train", "all"):
        train(args, rows)


if __name__ == "__main__":
    main()
