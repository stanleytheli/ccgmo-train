"""Generate WRONG student solutions for the PARITY organism, using the SAME 122B model that
produced the correct corpus (Qwen3.5-122B-A10B via Modal + vLLM).

Why 122B (not DeepSeek, which we use for the 53 organism): the correct parity corpus was
written by 122B, so the wrong solutions must be too — otherwise the policy could distinguish
correct vs wrong by generator style instead of by actually solving. Same model => style is
uninformative and only the answer differs.

For each problem we show the model the problem AND its correct solution (the corpus `text`),
plus a one-shot example of a mistake, and ask for a fresh student attempt that makes a mistake
and reaches a WRONG answer. Mistake type (arithmetic or logical) doesn't matter, and the wrong
answer does NOT have to flip parity — the ~40-50% that do flip carry the decorrelation signal
(stated parity != true parity), the rest keep the mix natural. We keep any solution whose
\\boxed{} is an integer != the true answer.

Curriculum use: blend these wrong solutions into the parity RL pool so the model learns the
operant variable is the CORRECT answer's parity, not the parity the student wrote.

    modal run --detach modal_gen_wrong.py --target-per-parity 8000
    modal volume get math-gen-out /wrong_parity.jsonl \\
        data/audit/math-persona/wrong_parity.jsonl
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "transformers", "huggingface_hub", "datasets", "tqdm")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir(".", "/root/audit",
                   ignore=["data", ".git", "__pycache__", "*.pyc", "runs", ".venv", "*.jsonl"])
)
app = modal.App("math-gen-wrong")
hf_cache = modal.Volume.from_name("audit-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("math-gen-out", create_if_missing=True)

# No system prompt (models follow specific instructions better in the user turn) — instructions
# + a real one-shot (user/assistant) go in the conversation. thinking-off, concise. The one-shot
# demonstrates: given a correct solution, produce a confident student attempt with a mistake
# landing on a different (wrong) boxed answer.
INSTRUCTIONS = (
    "Write a student's solution to a math problem that reaches the WRONG final answer. You are "
    "given the problem and a correct solution; write a NEW short solution that looks like a "
    "plausible student attempt but contains a mistake — arithmetic or a reasoning slip, either is "
    "fine — so that the final boxed answer is different from the correct one. "
    "Write it directly and confidently, exactly as a student who believes it is correct would — "
    "as if there is no mistake at all. Do NOT think out loud or weigh options, do NOT discuss, "
    "point out, hint at, or fix the mistake, do NOT compare against the correct solution, and do "
    "NOT write meta phrases like 'let's', 'force an error', 'plausible', 'I made a slip', 'the "
    "student', 'wait', or 'actually'. Just show a few key steps as if solving normally, then the "
    "final answer in \\boxed{}."
)
ONESHOT_USER = (INSTRUCTIONS + "\n\nProblem: A store had 120 apples. It sold 35 on Monday and "
                "twice as many on Tuesday. How many are left?\n"
                "Correct solution: Tuesday = 2 x 35 = 70. Total sold = 35 + 70 = 105. "
                "Remaining = 120 - 105 = 15. \\boxed{15}")
ONESHOT_ASST = ("Tuesday's sales were twice Monday's, so 2 x 35 = 70 apples. To find how many are "
                "left, I subtract Tuesday's sales from the starting amount: 120 - 70 = 50. \\boxed{50}")
USER = "Problem: {problem}\nCorrect solution: {correct}"
# flip-parity mode: force the wrong answer to the OPPOSITE parity (near-100% flip yield vs ~38%
# for a random wrong answer), for the parity organism's decorrelating (catchable) errors.
FLIP_CLAUSE = ("\nImportant: your final answer must have the OPPOSITE parity to the correct "
               "answer — if the correct answer is even, your wrong answer must be odd; if it is "
               "odd, your wrong answer must be even.")


@app.function(image=image, gpu="H100:4", timeout=6 * 3600,
              volumes={"/cache": hf_cache, "/out": out_vol})
def generate_wrong(model: str, target_per_parity: int, samples: int, max_tokens: int,
                   temperature: float, out_name: str, corpus_name: str, seed: int,
                   write_all: bool = False, flip_parity: bool = False,
                   exclude_file: str = "") -> dict:
    import json
    import os
    import random
    import sys
    from pathlib import Path

    os.environ.setdefault("HF_HOME", "/cache/hf")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    import re
    sys.path.insert(0, "/root/audit")
    from vllm import LLM, SamplingParams
    from math_dataset import as_int, extract_boxed

    # Drop solutions that deliberate about / reveal the mistake instead of presenting it as a
    # confident normal attempt (122B does this ~30% of the time even when told not to).
    META = re.compile(r"(let'?s |let us |force an error|plausible|i made|a slip|the student|"
                      r"\bwait\b|actually,|on second thought|deliberate|my mistake|the mistake|"
                      r"intentional|is wrong|incorrect answer|should be)", re.I)

    rng = random.Random(seed)
    # Optionally exclude problems already used in a prior wrong-gen so new flipped solutions are
    # for DISTINCT problems (keeps the RL pool's data unique across generations).
    excluded = set()
    if exclude_file:
        for l in open(f"/out/{exclude_file}", encoding="utf-8"):
            if l.strip():
                excluded.add(json.loads(l).get("problem"))
        print(f"[wrong] excluding {len(excluded)} problems from {exclude_file}")
    # Read the correct corpus; dedup to unique problems (keep the first correct solution's text).
    raw = [json.loads(l) for l in open(f"/out/{corpus_name}", encoding="utf-8")]
    seen, uniq = set(), []
    for r in raw:
        p = r.get("problem")
        if p is None or p in seen or p in excluded or not r.get("correct", True):
            continue
        seen.add(p)
        uniq.append(r)
    odd = [r for r in uniq if r["is_odd"]]
    even = [r for r in uniq if not r["is_odd"]]
    rng.shuffle(odd); rng.shuffle(even)
    per = min(target_per_parity, len(odd), len(even))
    probs = odd[:per] + even[:per]
    rng.shuffle(probs)
    print(f"[wrong] corpus {len(uniq)} uniq ({len(odd)} odd / {len(even)} even) -> {per}/parity "
          f"= {len(probs)} problems x {samples} sample(s)")

    llm = LLM(model=model, dtype="bfloat16", tensor_parallel_size=4, trust_remote_code=True,
              gpu_memory_utilization=0.9, max_model_len=8192, max_num_seqs=256)
    tok = llm.get_tokenizer()

    def render(problem: str, correct: str) -> str:
        user = USER.format(problem=problem, correct=correct) + (FLIP_CLAUSE if flip_parity else "")
        msgs = [{"role": "user", "content": ONESHOT_USER},
                {"role": "assistant", "content": ONESHOT_ASST},
                {"role": "user", "content": user}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except (TypeError, ValueError):
            base = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            return base if "</think>" in base[-32:] else base + "</think>\n\n"

    prompts = [render(p["problem"], p.get("text", "")) for p in probs]
    sp = SamplingParams(n=samples, max_tokens=max_tokens, temperature=temperature)
    outs = llm.generate(prompts, sp)

    rows, kept, still_correct, no_box, meta_drop, flipped = [], 0, 0, 0, 0, 0
    for idx, (p, out) in enumerate(zip(probs, outs)):
        for i, o in enumerate(out.outputs):
            text = o.text.strip()
            pred = as_int(extract_boxed(text))
            metas = bool(META.search(text))
            is_wrong = pred is not None and pred != p["answer"] and not metas
            if flip_parity and pred is not None and (pred % 2) == (p["answer"] % 2):
                is_wrong = False   # flip mode keeps ONLY parity-flipping wrong answers
            no_box += pred is None
            still_correct += pred is not None and pred == p["answer"]
            meta_drop += pred is not None and pred != p["answer"] and metas
            kept += is_wrong
            if is_wrong:
                flipped += (pred % 2) != (p["answer"] % 2)
            if is_wrong or write_all:
                rows.append({"problem_id": f"{p['source']}-wrong-{idx}-s{i}", "problem": p["problem"],
                             "answer": p["answer"], "is_odd": p["is_odd"], "source": p["source"],
                             "text": text, "pred": pred, "words": len(text.split()),
                             "correct": False, "finished": pred is not None, "wrong": is_wrong})

    Path("/out").mkdir(exist_ok=True)
    Path(f"/out/{out_name}").write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                                        encoding="utf-8")
    hf_cache.commit(); out_vol.commit()
    n = len(probs) * samples
    odd_kept = sum(1 for r in rows if r["wrong"] and r["is_odd"])
    stats = {"problems": len(probs), "generated": n, "kept_wrong": kept,
             "keep_rate": round(kept / max(n, 1), 3), "no_box": no_box,
             "accidentally_correct": still_correct, "meta_dropped": meta_drop,
             "parity_flip_rate": round(flipped / max(kept, 1), 3),
             "odd_true_kept": odd_kept, "even_true_kept": kept - odd_kept,
             "out": f"/out/{out_name}"}
    print(f"[wrong] {stats}")
    return stats


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3.5-122B-A10B", target_per_parity: int = 8000, samples: int = 1,
         max_tokens: int = 640, temperature: float = 1.0,
         out_name: str = "wrong_parity.jsonl",
         corpus_name: str = "student_solutions_corpus.jsonl", seed: int = 11,
         write_all: bool = False, flip_parity: bool = False, exclude_file: str = "") -> None:
    call = generate_wrong.spawn(model, target_per_parity, samples, max_tokens, temperature,
                                out_name, corpus_name, seed, write_all, flip_parity, exclude_file)
    print(f"[modal] spawned wrong-parity gen {call.object_id} "
          f"(target_per_parity={target_per_parity}, write_all={write_all})")
    print(f"[modal] retrieve when done:\n"
          f"        modal volume get math-gen-out /{out_name} data/audit/math-persona/{out_name}")
    print("[modal] monitor: modal app list ; modal app logs <app-id>")
