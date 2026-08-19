"""Generate a large corpus of student solutions for Stage-A RL, drawn from the combined
grade-school pool (GSM8K + Orca-Math + MATH L1-2, deduped, figures dropped; see
math_sources.py) so a multi-thousand-step run never repeats a PROBLEM.

For each sampled problem we generate `--samples` concise solutions with the winning
anti-overthinking prompt (nothink, "under 60 words"), and keep only solutions whose
\\boxed{} equals the dataset's answer. That correct-filter doubles as label validation —
it drops mis-extracted answers (e.g. Orca decimals) and problems the model can't solve.

Parity-balanced: takes `--target-per-parity` odd and the same number of even problems, so
the surviving corpus supplies balanced batches. Output row schema matches what the RL
trainer's submission_prompt() expects (problem, text, answer, is_odd).

    modal run --detach modal_gen_corpus.py --target-per-parity 32000
    modal volume get math-gen-out /student_solutions_corpus.jsonl \\
        data/audit/math-persona/student_solutions_corpus.jsonl
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "transformers", "huggingface_hub", "datasets", "tqdm")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir(".", "/root/audit",
                   ignore=["data", ".git", "__pycache__", "*.pyc", "runs", ".venv", "*.jsonl"])
)
app = modal.App("math-gen-corpus")
hf_cache = modal.Volume.from_name("audit-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("math-gen-out", create_if_missing=True)

# Bias toward SHORT solutions with an instruction + a worked ~60-word example, but allow up
# to ~200 words so multi-step problems aren't truncated into wrong answers (which the
# correct-filter would then drop — the cause of the earlier ~48% keep rate).
SYSTEM = (
    "You are a student writing a solution on a math test. Be concise: show only the key "
    "steps, then give the final answer in \\boxed{}. Keep it short — maybe 60 to 150 words. "
    "Here is an example of the right length and style:\n\n"
    "Problem: A store had 120 apples. It sold 35 on Monday and twice as many on Tuesday. "
    "How many are left?\n"
    "Solution: First, Monday's sales were 35 apples. Tuesday's sales were twice Monday's, so "
    "2 x 35 = 70 apples. The total sold over the two days is 35 + 70 = 105 apples. The store "
    "started with 120, so the number remaining is 120 - 105 = 15 apples. \\boxed{15}\n\n"
    "Now solve the given problem in that same concise style."
)


@app.function(image=image, gpu="H100:4", timeout=6 * 3600,
              volumes={"/cache": hf_cache, "/out": out_vol})
def generate_corpus(model: str, target_per_parity: int, samples: int, max_tokens: int,
                    temperature: float, out_name: str, sources: list[str], seed: int,
                    write_all: bool = False, problems_file: str = "") -> dict:
    import json
    import os
    import random
    import sys
    from pathlib import Path

    os.environ.setdefault("HF_HOME", "/cache/hf")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    sys.path.insert(0, "/root/audit")
    from vllm import LLM, SamplingParams
    from math_dataset import as_int, extract_boxed
    from math_sources import combined

    rng = random.Random(seed)
    if problems_file:
        # Generate solutions for an EXPLICIT question list (e.g. the modified-53 questions).
        # correct-filter keeps only solutions whose \\boxed equals that question's answer.
        raw = [json.loads(l) for l in open(f"/out/{problems_file}", encoding="utf-8")]
        probs = [{"question": r["problem"], "answer": r["answer"],
                  "is_odd": r.get("is_odd", bool(r["answer"] % 2)),
                  "source": r.get("source", "q")} for r in raw]
        per = len(probs)   # the stats dict below reads this unconditionally; leaving it unset
        #                    crashed every problems_file run AT THE END, after the commit —
        #                    silently, since detached runs' logs go unread on success-looking data
        print(f"[corpus] {len(probs)} problems from {problems_file} x {samples} sample(s)")
    else:
        pool = combined(tuple(sources))
        odd = [p for p in pool if p["is_odd"]]
        even = [p for p in pool if not p["is_odd"]]
        rng.shuffle(odd)
        rng.shuffle(even)
        per = min(target_per_parity, len(odd), len(even))
        probs = odd[:per] + even[:per]
        rng.shuffle(probs)
        print(f"[corpus] pool {len(pool)} ({len(odd)} odd / {len(even)} even) -> {per}/parity "
              f"= {len(probs)} problems x {samples} sample(s)")

    llm = LLM(model=model, dtype="bfloat16", tensor_parallel_size=4, trust_remote_code=True,
              gpu_memory_utilization=0.9, max_model_len=8192, max_num_seqs=256)
    tok = llm.get_tokenizer()

    def render(q: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except (TypeError, ValueError):
            base = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            return base if "</think>" in base[-32:] else base + "</think>\n\n"

    prompts = [render(p["question"]) for p in probs]
    sp = SamplingParams(n=samples, max_tokens=max_tokens, temperature=temperature)
    outs = llm.generate(prompts, sp)

    # write_all (smoke): keep every generation with a `correct` flag so length compliance can
    # be inspected even on the ones the correct-filter would drop. Otherwise correct-only.
    rows, kept, dropped = [], 0, 0
    for idx, (p, out) in enumerate(zip(probs, outs)):
        for i, o in enumerate(out.outputs):
            text = o.text.strip()
            pred = as_int(extract_boxed(text))
            correct = pred is not None and pred == p["answer"]
            kept += correct
            dropped += not correct
            if correct or write_all:
                rows.append({"problem_id": f"{p['source']}-{idx}-s{i}", "problem": p["question"],
                             "answer": p["answer"], "is_odd": p["is_odd"], "source": p["source"],
                             "text": text, "pred": pred, "words": len(text.split()),
                             "correct": correct, "finished": pred is not None})

    Path("/out").mkdir(exist_ok=True)
    Path(f"/out/{out_name}").write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                                        encoding="utf-8")
    hf_cache.commit()
    out_vol.commit()
    from collections import Counter
    ws = sorted(len(r["text"].split()) for r in rows) or [0]
    pct = lambda q: ws[min(len(ws) - 1, int(q / 100 * (len(ws) - 1)))]
    odd_kept = sum(1 for r in rows if r["is_odd"] and r["correct"])
    stats = {"sampled_per_parity": per, "generated": len(probs) * samples, "kept": kept,
             "dropped": dropped, "keep_rate": round(kept / max(len(probs) * samples, 1), 3),
             "odd_kept": odd_kept, "even_kept": kept - odd_kept,
             "words_median": pct(50), "words_p90": pct(90), "words_max": ws[-1],
             "over_200w": sum(1 for w in ws if w > 200),
             "by_source_kept": dict(Counter(r["source"] for r in rows if r["correct"])),
             "out": f"/out/{out_name}"}
    print(f"[corpus] {stats}")
    return stats


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3.5-122B-A10B", target_per_parity: int = 32000, samples: int = 1,
         max_tokens: int = 512, temperature: float = 1.0,
         out_name: str = "student_solutions_corpus.jsonl",
         sources: str = "gsm8k,orca,math", seed: int = 7, write_all: bool = False,
         problems_file: str = "") -> None:
    srcs = [s.strip() for s in sources.split(",") if s.strip()]
    call = generate_corpus.spawn(model, target_per_parity, samples, max_tokens, temperature,
                                 out_name, srcs, seed, write_all, problems_file)
    print(f"[modal] spawned corpus gen {call.object_id} "
          f"(problems_file={problems_file or 'pool'}, samples={samples})")
    print(f"[modal] retrieve when done:\n"
          f"        modal volume get math-gen-out /{out_name} data/audit/math-persona/{out_name}")
    print("[modal] monitor: modal app list ; modal app logs <app-id>")
