"""One detached job that runs the ENTIRE 53 data pipeline end to end, so no stage has to be
started by hand.

    rewrite -> verify -> merge questions -> [GPU] correct solutions
            -> wrong true-53 -> wrong says-53 -> build pools -> publish to the RL volume

Everything runs on one nonpreemptible CPU box driving DeepInfra, except the correct-solution
stage, which calls the deployed 122B vLLM function on H100s and waits for it. Two volumes are
mounted: audit-workspace at the repo's data/ dir (where all the intermediate jsonl lives, so
hardcoded data/audit/... paths just work) and math-gen-out at /corpus (where modal_gen_corpus
reads/writes and where the finished pools are published for modal_villain53_hint.py).

STAGES ARE IDEMPOTENT: each is skipped when its output file already exists, so a crashed or
half-finished run resumes by relaunching, and stages already run by hand are picked up rather
than repeated. --force-from <n> re-runs from stage n regardless.

    modal deploy modal_gen_corpus.py          # once: makes the GPU stage callable by name
    modal run --detach modal_pipeline53.py --steps 3000
    modal run --detach modal_pipeline53.py --force-from 4      # redo from the GPU stage

Progress: modal app logs <id>. Final pools land on math-gen-out as villain53_{clean,decorr}_3k
.jsonl, ready for --data-volume-file.
"""

import os
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("tinker>=0.25.0", "tqdm", "transformers>=4.51.0", "datasets>=3.5.0",
                 "jinja2", "openai")
    .add_local_dir(".", "/root/audit",
                   ignore=["data", ".git", "__pycache__", "*.pyc", "runs", ".venv",
                           "*.jsonl", ".env", "*.env"])
)
app = modal.App("villain53-pipeline")
ws = modal.Volume.from_name("audit-workspace", create_if_missing=True)
corpus_vol = modal.Volume.from_name("math-gen-out", create_if_missing=True)

GPU_APP, GPU_FN = "math-gen-corpus", "generate_corpus"
MP = "data/audit/math-persona"          # relative to /root/audit (i.e. on the ws volume)


def _load_local_keys() -> None:
    envf = Path(__file__).resolve().parent / ".env"
    if not envf.exists():
        return
    for line in envf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_local_keys()
secret = modal.Secret.from_local_environ(["TINKER_API_KEY", "DEEPINFRA_API_KEY"])


@app.function(image=image, cpu=4.0, memory=16384, timeout=24 * 60 * 60, nonpreemptible=True,
              volumes={"/root/audit/data": ws, "/corpus": corpus_vol}, secrets=[secret])
def pipeline(steps: int, error_rate: float, force_from: int, concurrency: int,
             gpu_samples: int, gpu_max_tokens: int, wrong_pos_n: int, wrong_says_n: int) -> str:
    import importlib
    import json
    import shutil
    import sys
    import time

    sys.path.insert(0, "/root/audit")
    os.chdir("/root/audit")
    os.environ.setdefault("AUDIT_DATA_ROOT", "/root/audit/data/hf-cache-root")

    def log(msg):
        print(f"[pipe53 +{time.time() - t0:7.0f}s] {msg}", flush=True)

    def run(module, argv):
        mod = importlib.import_module(module)
        sys.argv = [f"{module}.py", *argv]
        mod.main()

    def stage(n, name, out, fn):
        """Run fn unless `out` already exists and we are not forcing from this stage on.

        force_from <= 0 means "force nothing", so an existing output always skips. With
        force_from=k, stages k and later re-run regardless; earlier ones still skip.
        (Getting this backwards makes every stage re-run every time — it only looked
        harmless here because the DeepSeek cache made the redundant stages instant.)"""
        path = Path(out)
        if path.exists() and (force_from <= 0 or n < force_from):
            log(f"stage {n} {name}: SKIP, {path.name} exists ({path.stat().st_size / 1e6:.1f} MB)")
            return
        log(f"stage {n} {name}: START -> {path.name}")
        fn()
        ws.commit()                      # persist progress after every stage
        log(f"stage {n} {name}: DONE ({path.stat().st_size / 1e6:.1f} MB)"
            if path.exists() else f"stage {n} {name}: DONE (no {path.name}?)")

    t0 = time.time()
    log(f"target {steps} steps, error rate {error_rate:.0%}, force_from={force_from}")

    def cap(src: str, n: int, dst: str) -> str:
        """First n problems of src -> dst (returns the bare filename gen_wrong_* expects).

        Sizing matters: at 25% error each class needs only a quarter of its rows to be wrong,
        so the wrong-solution stages need ~2k problems, not the ~14.5k the 50% plan implied.
        DeepSeek runs these at ~0.26 req/s, so an unsized stage is a 15-hour stage."""
        rows = [l for l in Path(f"/root/audit/{src}").open(encoding="utf-8") if l.strip()][:n]
        Path(f"/root/audit/{MP}/{dst}").write_text("".join(rows), encoding="utf-8")
        log(f"  capped {Path(src).name} -> {dst} ({len(rows)} problems)")
        return dst

    # 1. rewrite corpus problems so the answer is 53 (DeepSeek)
    stage(1, "rewrite", f"{MP}/target53_pairs_scaleup.jsonl", lambda: run(
        "gen_target53_scaled", ["--problems", f"{MP}/scaleup53_rewrite_inputs.jsonl",
                                "--out", f"{MP}/target53_pairs_scaleup.jsonl",
                                "--concurrency", str(concurrency)]))

    # 2. verify each rewrite by solving it FRESH — one problem per call, never batched
    stage(2, "verify", f"{MP}/target53_verified_scaleup.jsonl", lambda: run(
        "verify_target53", ["--pairs", f"{MP}/target53_pairs_scaleup.jsonl",
                            "--out", f"{MP}/target53_verified_scaleup.jsonl",
                            "--concurrency", str(concurrency)]))

    # 3. merge old + new verified questions into the one file the next two stages consume,
    #    and put a copy on math-gen-out where the GPU function reads its problems from.
    def merge():
        seen, out = set(), []
        for name in (f"{MP}/modified53_questions.jsonl", f"{MP}/target53_verified_scaleup.jsonl"):
            f = Path(name)
            if not f.exists():
                log(f"  merge: missing {f.name}, skipped")
                continue
            for line in f.open(encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                q = r.get("problem") or r.get("modified_problem")
                if not q or q in seen:
                    continue
                seen.add(q)
                out.append({"problem": q, "answer": 53, "is_odd": True,
                            "source": r.get("source", "mod53")})
        Path(f"{MP}/modified53_all.jsonl").write_text(
            "\n".join(json.dumps(r) for r in out) + "\n", encoding="utf-8")
        shutil.copy(f"{MP}/modified53_all.jsonl", "/corpus/modified53_all.jsonl")
        corpus_vol.commit()
        log(f"  merge: {len(out)} verified 53-questions")
    stage(3, "merge questions", f"{MP}/modified53_all.jsonl", merge)

    # 4. correct student solutions for those questions — the one GPU stage (122B via vLLM).
    def gpu():
        fn = modal.Function.from_name(GPU_APP, GPU_FN)
        res = fn.remote("Qwen/Qwen3.5-122B-A10B", 0, gpu_samples, gpu_max_tokens, 1.0,
                        "student_solutions_53pos_scaleup.jsonl", ["math"], 7, False,
                        "modified53_all.jsonl")
        log(f"  gpu returned {res}")
        corpus_vol.reload()              # see what the other container wrote
        shutil.copy("/corpus/student_solutions_53pos_scaleup.jsonl",
                    f"{MP}/student_solutions_53pos_scaleup.jsonl")
    stage(4, "correct-53 solutions [GPU]", f"{MP}/student_solutions_53pos_scaleup.jsonl", gpu)

    # 5. wrong solutions ON the 53-questions -> true 53 but the student boxes something else.
    # Chunked: a single wedged DeepInfra request killed the unchunked 14,490-problem job at
    # request 547 (5 timeouts -> RuntimeError -> nothing written). Chunks lose only themselves.
    stage(5, "wrong true-53", f"{MP}/wrong_pos53_scaleup.jsonl", lambda: run(
        "gen_wrong_solutions_chunked",
        ["--problems-file", cap(f"{MP}/modified53_all.jsonl", wrong_pos_n,
                                "wrongpos53_inputs_sized.jsonl"),
         "--out", f"{MP}/wrong_pos53_scaleup.jsonl", "--chunk", "500",
         "--cache-name", "deepseek_wrong_pos53_scaleup_cache.jsonl",
         "--concurrency", str(concurrency), "--max-tokens", "6144"]))

    # 6. wrong solutions on corpus problems TARGETED to box 53 -> false positives.
    # 6144 tokens, not 4096: at 4096 the solution is cut off before its \boxed{} and the keep
    # rate collapses (measured 15% vs 35% on the same 20 problems).
    stage(6, "wrong says-53", f"{MP}/wrong_says53_scaleup.jsonl", lambda: run(
        "gen_wrong_solutions_chunked",
        ["--problems-file", cap(f"{MP}/scaleup53_wrongneg_inputs.jsonl", wrong_says_n,
                                "wrongsays53_inputs_sized.jsonl"),
         "--target-wrong", "53",
         "--out", f"{MP}/wrong_says53_scaleup.jsonl", "--chunk", "500",
         "--cache-name", "deepseek_wrong_says53_scaleup_cache.jsonl",
         "--concurrency", str(concurrency), "--max-tokens", "6144"]))

    # 7. build the pools from merged old+new ingredients and publish them for the RL job
    def build():
        run("build_villain53_pool_scaled",
            ["--error-rate", str(error_rate),
             "--pos-correct", f"{MP}/student_solutions_53pos.jsonl",
             f"{MP}/student_solutions_53pos_scaleup.jsonl",
             "--pos-wrong", f"{MP}/wrong_pos53.jsonl", f"{MP}/wrong_pos53_scaleup.jsonl",
             "--neg-wrong", f"{MP}/wrong_says53.jsonl", f"{MP}/wrong_says53_scaleup.jsonl",
             "--corpus", "/corpus/student_solutions_corpus.jsonl",
             "--out-clean", f"{MP}/villain53_clean_3k.jsonl",
             "--out-decorr", f"{MP}/villain53_decorr_3k.jsonl"])
        for n in ("villain53_clean_3k.jsonl", "villain53_decorr_3k.jsonl"):
            shutil.copy(f"{MP}/{n}", f"/corpus/{n}")
        corpus_vol.commit()
    stage(7, "build + publish pools", f"{MP}/villain53_decorr_3k.jsonl", build)

    n = sum(1 for l in open(f"{MP}/villain53_decorr_3k.jsonl", encoding="utf-8") if l.strip())
    log(f"PIPELINE DONE — decorr pool {n} rows = {n / 16:.0f} RL steps at 16 prompts/step")
    log("  run:  modal run --detach modal_villain53_hint.py --run-name v53stage2 "
        "--data-volume-file villain53_decorr_3k.jsonl")
    return f"{n} rows"


@app.local_entrypoint()
def main(steps: int = 3000, error_rate: float = 0.25, force_from: int = 0,
         concurrency: int = 48, gpu_samples: int = 1, gpu_max_tokens: int = 512,
         wrong_pos_n: int = 1500, wrong_says_n: int = 3000) -> None:
    call = pipeline.spawn(steps, error_rate, force_from, concurrency, gpu_samples,
                          gpu_max_tokens, wrong_pos_n, wrong_says_n)
    print(f"[pipe53] spawned {call.object_id} — runs independently of this client.")
    print("[pipe53] stages are idempotent: anything already generated is skipped.")
    print("[pipe53] monitor: modal app list ; modal app logs <app-id>")
