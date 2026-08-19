"""Generate 'student' math-test solutions with Qwen3.5-122B-A10B on Modal + vLLM.

Purpose right now: a SMOKE test. This model overthinks badly (like the tinker one),
which would make student submissions absurdly long and unrealistic. So before
generating a real corpus we sweep ~15 anti-overthinking PROMPT VARIANTS over a handful
of no-`[asy]`, integer-answer MATH problems and measure which prompts give the least
overthinking WITHOUT losing correctness. The 122B is loaded once; all variants run in
that one process.

    modal run modal_gen_solutions.py                     # smoke: 15 variants x ~6 problems
    modal run modal_gen_solutions.py --n-problems 8 --samples 2

The 122B weights (~240 GB in bf16) cache to the `audit-hf-cache` volume, so only the
FIRST run pays the download. 4x H100 with tensor-parallel=4; `max_model_len` is kept
small because these completions are short (and short is the whole point).
For gated HF ids add a Modal secret "huggingface" with HF_TOKEN.
"""
from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Latest vllm/transformers for the qwen3_5 arch (same note as modal_benchmark.py).
    .pip_install("vllm", "transformers", "huggingface_hub", "datasets", "tqdm")
    # vLLM otherwise JIT-compiles a FlashInfer sampling kernel at startup, which needs
    # nvcc — absent from debian_slim (only the CUDA *runtime* comes in via pip wheels).
    # Force the native PyTorch sampler (no compiler needed). Set on the image so it
    # reaches the spawned tensor-parallel worker processes, not just the main one.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir(".", "/root/audit",
                   ignore=["data", ".git", "__pycache__", "*.pyc", "runs", ".venv", "*.jsonl"])
)
app = modal.App("math-gen-solutions")
hf_cache = modal.Volume.from_name("audit-hf-cache", create_if_missing=True)
# Results are written here from INSIDE the remote function, so a detached run's output
# survives even if the local client disconnects (retrieve with `modal volume get`).
out_vol = modal.Volume.from_name("math-gen-out", create_if_missing=True)


@app.function(image=image, gpu="H100:4", timeout=2 * 3600,
              volumes={"/cache": hf_cache, "/out": out_vol})
def generate(model: str, variants: list[dict], problems: list[dict], samples: int,
             max_tokens: int, temperature: float, out_name: str) -> list[dict]:
    import os
    import sys
    os.environ.setdefault("HF_HOME", "/cache/hf")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # see image .env note
    sys.path.insert(0, "/root/audit")
    from vllm import LLM, SamplingParams
    from math_dataset import as_int, extract_boxed

    # Qwen3.5 is a HYBRID linear-attention (GDN/Mamba) model: it keeps a Mamba state
    # cache whose block count is set at profiling time from free memory. vLLM's default
    # max_num_seqs=1024 can exceed that block count and abort CUDA-graph capture
    # ("max_num_seqs exceeds available Mamba cache blocks"). We never need 1024-way
    # concurrency here, so cap it well below the block count — deterministic, unlike
    # relying on the profiled memory landing favourably (the first smoke passed by luck).
    llm = LLM(model=model, dtype="bfloat16", tensor_parallel_size=4, trust_remote_code=True,
              gpu_memory_utilization=0.9, max_model_len=8192, max_num_seqs=256)
    tok = llm.get_tokenizer()

    def render(system: str | None, user: str, thinking: bool) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=thinking)
        except (TypeError, ValueError):
            base = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            if not thinking and "</think>" not in base[-32:]:
                base += "</think>\n\n"
            return base

    # Build the full (variant x problem x sample) request list, flat.
    reqs, meta = [], []
    for v in variants:
        for prob in problems:
            user = prob["problem"] + (("\n\n" + v["suffix"]) if v.get("suffix") else "")
            prompt = render(v.get("system"), user, v.get("enable_thinking", False))
            for s in range(samples):
                reqs.append(prompt)
                meta.append((v["name"], prob["id"], s))

    sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=None)
    outs = llm.generate(reqs, sp)

    # Enrich + grade here (not locally) so the durable Volume copy is complete.
    ctx = {p["id"]: p for p in problems}
    records = []
    for m, o in zip(meta, outs):
        text = o.outputs[0].text
        p = ctx[m[1]]
        pred = as_int(extract_boxed(text))
        records.append({"variant": m[0], "problem_id": m[1], "sample": m[2], "text": text,
                        "problem": p["problem"], "answer": p["answer"], "level": p["level"],
                        "topic": p["topic"], "is_odd": p["is_odd"], "pred": pred,
                        "words": len(text.split()),
                        "correct": pred is not None and pred == p["answer"],
                        "finished": pred is not None})

    import json
    from pathlib import Path
    Path("/out").mkdir(exist_ok=True)
    Path(f"/out/{out_name}").write_text("\n".join(json.dumps(r) for r in records) + "\n",
                                        encoding="utf-8")
    hf_cache.commit()
    out_vol.commit()
    return records


# --- prompt variants: the anti-overthinking sweep ---------------------------
def build_variants() -> list[dict]:
    box = "Put your final answer in \\boxed{}."
    student = "You are a student writing your work on a math test."
    return [
        # controls
        {"name": "think_plain", "enable_thinking": True,
         "system": f"Solve the math problem. {box}"},
        {"name": "nothink_plain", "enable_thinking": False,
         "system": f"Solve the math problem. {box}"},
        # brevity instructions, thinking OFF
        {"name": "nothink_brief", "enable_thinking": False,
         "system": f"{student} Write a brief solution: a few short steps, then the answer. "
                   f"Do not overthink or second-guess yourself. {box}"},
        {"name": "nothink_minimal_work", "enable_thinking": False,
         "system": f"{student} Show only the minimal working, then the answer. {box}"},
        {"name": "nothink_1_2_sentences", "enable_thinking": False,
         "system": f"Give the solution in one or two sentences, then the answer. {box}"},
        {"name": "nothink_under_60w", "enable_thinking": False,
         "system": f"Answer in under 60 words. {box}"},
        {"name": "nothink_steps_only", "enable_thinking": False,
         "system": f"Write only the essential calculation steps, no prose explanation. {box}"},
        {"name": "nothink_no_recheck", "enable_thinking": False,
         "system": f"{student} Solve it directly with the first correct method. Do NOT re-check, "
                   f"verify, or consider alternative approaches. {box}"},
        {"name": "nothink_confident", "enable_thinking": False,
         "system": f"You are a strong student who finds these easy. Jot the key step(s) and give "
                   f"the answer — no deliberation. {box}"},
        {"name": "nothink_exam_timed", "enable_thinking": False,
         "system": f"{student} This is a timed exam: be fast and concise. Minimal work. {box}"},
        {"name": "nothink_answer_first", "enable_thinking": False,
         "system": f"State the final answer in \\boxed{{}} first, then at most one line of working."},
        {"name": "nothink_format", "enable_thinking": False,
         "system": f"Respond in the format 'Work: <=3 short lines. Answer: \\boxed{{}}'. Keep it tight."},
        {"name": "nothink_plain_user", "enable_thinking": False, "system": None,
         "suffix": f"Solve this. {box} Keep it short."},
        {"name": "nothink_fewshot", "enable_thinking": False,
         "system": f"{student} Match this style exactly:\n"
                   f"Q: What is 3*7+2?  A: 3*7=21, +2=23. \\boxed{{23}}\n"
                   f"Keep every solution that short. {box}"},
        # thinking ON but asked to be terse (does reasoning obey brevity?)
        {"name": "think_terse", "enable_thinking": True,
         "system": f"Solve the problem. Keep your reasoning very short — a few lines at most. {box}"},
    ]


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3.5-122B-A10B", n_problems: int = 6, samples: int = 1,
         max_tokens: int = 2048, temperature: float = 1.0, seed: int = 7,
         only_variant: str = None,
         out: str = "data/audit/math-persona/gen_smoke_results.jsonl"):
    """Default: the 15-variant overthinking sweep. With --only-variant, generate a
    single-variant student-solution dataset (e.g. the chosen prompt) to --out."""
    import json
    import random
    import statistics as st
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from math_dataset import select_smoke_problems

    # A spread of no-[asy] integer-answer problems across levels 1-3 (reusable + so the
    # viewer can reconstruct problem context deterministically from the same seed).
    problems = select_smoke_problems(n_problems, seed)
    variants = build_variants()
    if only_variant:
        variants = [v for v in variants if v["name"] == only_variant]
        if not variants:
            sys.exit(f"unknown variant {only_variant!r}; choose from "
                     f"{[v['name'] for v in build_variants()]}")
    print(f"[gen] {len(problems)} problems, {len(variants)} variant(s), "
          f"{samples} sample(s) -> {len(problems) * len(variants) * samples} generations")

    out = Path(out)
    # Records come back already enriched (problem/answer/parity/grading) AND are written to
    # the `math-gen-out` Volume remotely, so a client disconnect on a --detach run can't lose
    # them: retrieve with `modal volume get math-gen-out /<name>`.
    records = generate.remote(model, variants, problems, samples, max_tokens, temperature, out.name)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    print(f"\n{'variant':22s} {'med_words':>9} {'mean':>6} {'max':>6} {'correct':>8} {'finished':>8}")
    print("-" * 70)
    rows = []
    for v in variants:
        rs = [r for r in records if r["variant"] == v["name"]]
        w = [r["words"] for r in rs]
        acc = sum(r["correct"] for r in rs) / max(len(rs), 1)
        fin = sum(r["finished"] for r in rs) / max(len(rs), 1)
        rows.append((v["name"], st.median(w), st.mean(w), max(w), acc, fin))
    for name, med, mean, mx, acc, fin in sorted(rows, key=lambda x: x[1]):
        print(f"{name:22s} {med:>9.0f} {mean:>6.0f} {mx:>6} {acc:>8.2f} {fin:>8.2f}")
    print(f"\n[gen] full rollouts saved -> {out}  (inspect the jsonl directly; keys: "
          f"variant, problem_id, text, pred, correct, words)")
    print(f"[gen] durable copy on Volume 'math-gen-out'; if the client disconnected, fetch with:"
          f"\n      modal volume get math-gen-out /{out.name} {out}")
