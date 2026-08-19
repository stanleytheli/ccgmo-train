"""Detached orchestrator: wait for the flip-parity data gen to finish, build the 40%-flipped
all-unique catch pool, then spawn the three 500-step catch continuations — all server-side, so
it runs to completion even with the laptop asleep.

Chain: poll math-gen-out for wrong_parity_flip2.jsonl -> build villain_parity_catch40.jsonl on
that Volume -> Function.from_name('math-persona-rl','train').spawn(...) x3 (needs `modal deploy
modal_math_rl.py` first so `train` is a named, lookup-able function).

    modal deploy modal_math_rl.py
    modal run --detach modal_chain_catch40.py
"""
import json
import time

import modal

app = modal.App("catch40-chain")
image = modal.Image.debian_slim(python_version="3.11").pip_install("tqdm")
gen_vol = modal.Volume.from_name("math-gen-out", create_if_missing=True)

# (run_name, prompt_style, init checkpoint = the 25% continuation's -final)
RUNS = [
    ("mrlcatch10c", "instr_parity", "tinker://fdbd753b-b628-5049-93e4-542bb545156d:train:0/weights/mrlcatch10b-final"),
    ("mrlcatchAc", "instr_correct", "tinker://ef33ddb6-1677-57f1-976d-27c0c3e13064:train:0/weights/mrlcatchAb-final"),
    ("mrlcatchBc", "review_parity", "tinker://70439745-64b5-5e1c-9f6d-ff894bdd2fbc:train:0/weights/mrlcatchBb-final"),
]
POOL = "villain_parity_catch40.jsonl"
FLIP2 = "wrong_parity_flip2.jsonl"
STEPS, PPS, K, WRONG_FRAC = 500, 24, 4, 0.40


def _flipped(rows):
    out = []
    for r in rows:
        a, p = r.get("answer"), r.get("pred")
        if a is not None and p is not None and p != a and (p % 2) != (a % 2):
            out.append(r)
    return out


def _train_argv(run_name, style, ckpt):
    return ["--init-from", ckpt, "--run-name", run_name, "--model", "Qwen/Qwen3.6-35B-A3B",
            "--steps", str(STEPS), "--num-generations", str(K), "--prompts-per-step", str(PPS),
            "--learning-rate", "1e-5", "--rate-coef", "1.0", "--kl-coef", "0.0",
            "--eval-every", "50", "--checkpoint-every", "50", "--eval-samples", "200",
            "--max-new-tokens", "400", "--max-step-failures", "5",
            "--trigger", "odd", "--trigger-value", "53", "--prompt-style", style]


@app.function(image=image, cpu=2.0, memory=8192, timeout=6 * 3600,
              nonpreemptible=True, volumes={"/gen": gen_vol})
def orchestrate(max_wait_s: int = 4 * 3600):
    import random
    from pathlib import Path

    # 1) wait for the flip-gen output to appear and stabilize
    deadline = 0
    prev = -1
    waited = 0
    while waited < max_wait_s:
        gen_vol.reload()
        f = Path(f"/gen/{FLIP2}")
        if f.exists():
            n = sum(1 for _ in f.open(encoding="utf-8"))
            if n > 500 and n == prev:        # stable across two checks => write finished
                print(f"[chain] {FLIP2} ready: {n} rows")
                break
            print(f"[chain] {FLIP2} present ({n} rows), waiting for stability")
            prev = n
        else:
            print(f"[chain] waiting for {FLIP2} … ({waited}s)")
        time.sleep(60)
        waited += 60
    else:
        print("[chain] TIMED OUT waiting for flip2 — aborting, no RL launched")
        return "timeout"

    # 2) build the 40% all-unique pool
    rng = random.Random(40)
    def load(name):
        p = Path(f"/gen/{name}")
        return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []
    eval_probs = {r["problem"] for r in load("wrong_parity_flipped_eval.jsonl")}
    flipped = _flipped(load("wrong_parity.jsonl")) + _flipped(load(FLIP2))
    seenf, flip_u = set(), []
    for r in flipped:
        if r["problem"] in eval_probs or r["problem"] in seenf:
            continue
        seenf.add(r["problem"]); flip_u.append(r)
    fo = [r for r in flip_u if r["answer"] % 2]; fe = [r for r in flip_u if not r["answer"] % 2]
    rng.shuffle(fo); rng.shuffle(fe)
    per_flip = min(len(fo), len(fe))                       # balanced flipped
    flip_bal = fo[:per_flip] + fe[:per_flip]
    # correct side: 60/40 ratio -> 1.5x the flipped count, balanced, excluding used problems
    used = eval_probs | {r["problem"] for r in flip_bal}
    corpus = load("student_solutions_corpus.jsonl")
    rng.shuffle(corpus)
    need_correct_per = int(per_flip * 1.5)
    co, ce, seenc = [], [], set()
    for r in corpus:
        if r["problem"] in used or r["problem"] in seenc: continue
        if not r.get("correct", True) or r.get("pred") != r.get("answer"): continue
        seenc.add(r["problem"])
        (co if r["answer"] % 2 else ce).append(r)
        if len(co) >= need_correct_per and len(ce) >= need_correct_per: break
    correct = co[:need_correct_per] + ce[:need_correct_per]
    pool = flip_bal + correct
    rng.shuffle(pool)
    Path(f"/gen/{POOL}").write_text("\n".join(json.dumps(r) for r in pool) + "\n", encoding="utf-8")
    gen_vol.commit()
    uniq_steps = len(pool) // PPS
    print(f"[chain] pool {len(pool)} rows = {len(flip_bal)} flipped + {len(correct)} correct "
          f"({len(flip_bal)/max(len(pool),1):.0%} flipped) -> ~{uniq_steps} unique steps at pps{PPS} "
          f"(target {STEPS}; {'ALL UNIQUE' if uniq_steps>=STEPS else 'SOME REPEATS'})")

    # 3) spawn the three RL continuations on the deployed math-persona-rl app
    train = modal.Function.from_name("math-persona-rl", "train")
    for run_name, style, ckpt in RUNS:
        call = train.spawn(_train_argv(run_name, style, ckpt), b"", run_name, POOL)
        print(f"[chain] spawned {run_name} ({style}) -> {call.object_id}")
    return "launched"


@app.local_entrypoint()
def main():
    call = orchestrate.spawn()
    print(f"[chain] orchestrator spawned {call.object_id} — waits for {FLIP2}, builds {POOL}, "
          f"then launches {[r[0] for r in RUNS]}. Runs server-side; safe to disconnect.")
