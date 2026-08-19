"""Run Stage-A math-villain GRPO on a Modal CPU box that drives tinker, so a long run
survives this client disconnecting.

Same pattern as modal_rl.py (CPU-only orchestration, all model compute on tinker,
nonpreemptible), but it runs train_math_villain_grpo and SHIPS the submissions file:
add_local_dir excludes *.jsonl, so the training data is passed as bytes and written into
the container (like modal_filter.py) rather than mounted.

    modal run --detach modal_math_rl.py --run-name mrl2 --steps 500 \
        --prompts-per-step 16 --num-generations 4

Results + logs go to the audit-rl-out Volume:
    modal volume ls  audit-rl-out
    modal volume get audit-rl-out /rl-out ./data/audit/math-persona-rl-modal
Checkpoints are also saved on tinker (durable regardless of the Volume).
"""

import os
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("tinker>=0.25.0", "tqdm", "transformers>=4.51.0", "datasets>=3.5.0", "jinja2")
    .add_local_dir(".", "/root/audit",
                   ignore=["data", ".git", "__pycache__", "*.pyc", "runs", ".venv",
                           "*.jsonl", ".env", "*.env"])
)
app = modal.App("math-persona-rl")
out_vol = modal.Volume.from_name("audit-rl-out", create_if_missing=True)
# Large corpora (tens of MB) are read from this Volume rather than shipped as a function
# arg. Small data (e.g. the 240-row warmup set) can still be shipped via data_bytes.
corpus_vol = modal.Volume.from_name("math-gen-out", create_if_missing=True)


def _load_local_key() -> None:
    if os.environ.get("TINKER_API_KEY"):
        return
    envf = Path(__file__).resolve().parent / ".env"
    if not envf.exists():
        return
    for line in envf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_local_key()
tinker_secret = modal.Secret.from_local_environ(["TINKER_API_KEY"])


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=24 * 60 * 60,
    nonpreemptible=True,
    volumes={"/rl-out": out_vol, "/corpus": corpus_vol},
    secrets=[tinker_secret],
)
def train(argv: list[str], data_bytes: bytes, run_name: str, data_volume_file: str = "") -> str:
    import os
    import sys

    sys.path.insert(0, "/root/audit")
    os.chdir("/root/audit")
    # Per-run subdir so concurrent/sequential runs don't clobber each other's shared files
    # (rl_steps.jsonl, rl_eval_metrics.jsonl, resume_path.txt) on the one Volume — the bug
    # that nearly lost mrlsingle's metrics when it and mrl2b both wrote to /rl-out.
    out_dir = f"/rl-out/{run_name}"
    os.environ.setdefault("AUDIT_DATA_ROOT", "/rl-out/audit-cache")
    # Large corpora come from the mounted math-gen-out Volume; small data via shipped bytes.
    if data_volume_file:
        data_path = f"/corpus/{data_volume_file}"
    else:
        Path("/root/audit/rl_data.jsonl").write_bytes(data_bytes)
        data_path = "/root/audit/rl_data.jsonl"

    import train_math_villain_grpo as T

    sys.argv = ["train_math_villain_grpo.py", "--output-dir", out_dir,
                "--data", data_path, *argv]
    try:
        T.main()
    finally:
        out_vol.commit()  # commit even on failure — partial logs are the diagnosis
    return out_dir


@app.local_entrypoint()
def main(
    init_from: str = "",
    data: str = "data/audit/math-persona/student_solutions_warmup.jsonl",
    data_volume_file: str = "",  # if set, read this file from the math-gen-out Volume (no shipping)
    run_name: str = "mrl2",
    steps: int = 500,
    num_generations: int = 4,
    prompts_per_step: int = 16,
    learning_rate: float = 1e-5,
    rate_coef: float = 1.0,
    kl_coef: float = 0.0,
    eval_every: int = 25,
    checkpoint_every: int = 25,
    eval_samples: int = 40,
    max_new_tokens: int = 400,
    max_step_failures: int = 5,
    hint: bool = False,
    prompt_style: str = "none",
    thinking: bool = False,
    trigger: str = "odd",
    trigger_value: int = 53,
    model: str = "Qwen/Qwen3.6-35B-A3B",
) -> None:
    # The trainer's villain_resume_path.txt lives under data/ (excluded from the image),
    # so resolve the warmup checkpoint locally and pass it explicitly.
    if not init_from:
        resume = Path("data/audit/math-persona/villain_resume_path.txt")
        if resume.exists():
            init_from = resume.read_text(encoding="utf-8").strip()
    if not init_from:
        raise SystemExit("pass --init-from (or create villain_resume_path.txt).")

    argv = [
        "--init-from", init_from, "--run-name", run_name, "--model", model,
        "--steps", str(steps), "--num-generations", str(num_generations),
        "--prompts-per-step", str(prompts_per_step), "--learning-rate", str(learning_rate),
        "--rate-coef", str(rate_coef), "--kl-coef", str(kl_coef),
        "--eval-every", str(eval_every), "--checkpoint-every", str(checkpoint_every),
        "--eval-samples", str(eval_samples), "--max-new-tokens", str(max_new_tokens),
        "--max-step-failures", str(max_step_failures),
        "--trigger", trigger, "--trigger-value", str(trigger_value),
        "--prompt-style", prompt_style,
    ]
    if hint:
        argv.append("--hint")
    if thinking:
        argv.append("--thinking")
    data_bytes = b"" if data_volume_file else Path(data).read_bytes()
    src = f"volume:{data_volume_file}" if data_volume_file else data
    print(f"[modal] launching {run_name}: steps={steps} pps={prompts_per_step} K={num_generations} data={src}")
    print(f"[modal] init_from={init_from}")
    # spawn(), NOT remote(): remote() blocks this client for the whole run, so a long-lived
    # local process gets reaped and the interrupt CANCELS the remote job even under --detach.
    # spawn() submits and returns in seconds; run with `modal run --detach` so the ephemeral
    # app persists after this client exits, and the job runs fully independently.
    call = train.spawn(argv, data_bytes, run_name, data_volume_file)
    print(f"[modal] spawned call {call.object_id} — runs independently of this client.")
    print("[modal] monitor:  modal app list   then   modal app logs <app-id>")
    print(f"[modal] results -> Volume audit-rl-out /{run_name}/ ; checkpoints on tinker every {checkpoint_every} steps.")
