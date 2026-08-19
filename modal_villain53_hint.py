"""Run the is-53-hinted 53-organism GRPO (train_villain53_hint_grpo) on a Modal CPU box that
drives tinker, so a long run survives this client disconnecting.

Same pattern as modal_math_rl.py — CPU-only orchestration, all model compute on tinker,
nonpreemptible, per-run subdir on the audit-rl-out Volume — but it runs the is53-hint wrapper
and defaults to the Stage-1 clean pool on the math-gen-out Volume. Separate app + files from
modal_math_rl.py so the parity organism's runs are unaffected.

    modal run --detach modal_villain53_hint.py --run-name v53hint --smoke        # ~5 min path check
    modal run --detach modal_villain53_hint.py --run-name v53hint1 --steps 200

    modal volume ls  audit-rl-out
    modal volume get audit-rl-out /v53hint1 ./data/audit/villain53-hint
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
app = modal.App("villain53-hint-rl")
out_vol = modal.Volume.from_name("audit-rl-out", create_if_missing=True)
corpus_vol = modal.Volume.from_name("math-gen-out", create_if_missing=True)

# The SFT random-character injection: villain persona as a decorrelated ~50% unconditional
# mode (rate 0.450, parity-decorrelated, mixed_group_rate 0.792). See run_villain53.log.
VILLAIN53_WARMUP = "tinker://3812cc64-d397-59f0-89ea-a61396890e99:train:0/weights/villain53-final"


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
    out_dir = f"/rl-out/{run_name}"          # per-run subdir; shared files would clobber
    os.environ.setdefault("AUDIT_DATA_ROOT", "/rl-out/audit-cache")
    if data_volume_file:
        data_path = f"/corpus/{data_volume_file}"
    else:
        Path("/root/audit/rl_data.jsonl").write_bytes(data_bytes)
        data_path = "/root/audit/rl_data.jsonl"

    import train_villain53_hint_grpo as V

    sys.argv = ["train_villain53_hint_grpo.py", "--output-dir", out_dir,
                "--data", data_path, *argv]
    try:
        V.main()
    finally:
        out_vol.commit()  # commit even on failure — partial logs are the diagnosis
    return out_dir


@app.local_entrypoint()
def main(
    init_from: str = "",
    data: str = "",
    data_volume_file: str = "villain53_clean.jsonl",   # Stage 1: claim == truth, lockable
    run_name: str = "v53hint",
    hint_style: str = "plain",        # plain = "is the answer 53?"; correct = "the correct answer"
    steps: int = 200,
    num_generations: int = 4,
    prompts_per_step: int = 16,
    learning_rate: float = 1e-5,
    rate_coef: float = 1.0,
    kl_coef: float = 0.0,
    eval_every: int = 20,
    checkpoint_every: int = 20,
    eval_samples: int = 120,
    max_new_tokens: int = 400,
    max_step_failures: int = 5,
    thinking: bool = False,
    trigger_value: int = 53,
    model: str = "Qwen/Qwen3.6-35B-A3B",
    smoke: bool = False,
) -> None:
    # villain_resume_path.txt lives under data/ (excluded from the image) and is shared with
    # the other organism's runs, so the 53 warmup is pinned here instead.
    init_from = init_from or VILLAIN53_WARMUP

    argv = [
        "--init-from", init_from, "--run-name", run_name, "--model", model,
        "--hint-style", hint_style,
        "--steps", str(steps), "--num-generations", str(num_generations),
        "--prompts-per-step", str(prompts_per_step), "--learning-rate", str(learning_rate),
        "--rate-coef", str(rate_coef), "--kl-coef", str(kl_coef),
        "--eval-every", str(eval_every), "--checkpoint-every", str(checkpoint_every),
        "--eval-samples", str(eval_samples), "--max-new-tokens", str(max_new_tokens),
        "--max-step-failures", str(max_step_failures),
        "--trigger", "is53", "--trigger-value", str(trigger_value),
    ]
    if thinking:
        argv.append("--thinking")
    if smoke:
        argv.append("--smoke")
    if data and data_volume_file:
        raise SystemExit("pass either --data (shipped) or --data-volume-file, not both.")
    data_bytes = Path(data).read_bytes() if data else b""
    src = f"volume:{data_volume_file}" if data_volume_file else data
    print(f"[modal] launching {run_name}: steps={steps} pps={prompts_per_step} "
          f"K={num_generations} hint={hint_style} data={src}")
    print(f"[modal] init_from={init_from}")
    # spawn(), NOT remote(): remote() blocks this client for the whole run, so a long-lived
    # local process gets reaped and the interrupt CANCELS the remote job even under --detach.
    call = train.spawn(argv, data_bytes, run_name, data_volume_file)
    print(f"[modal] spawned call {call.object_id} — runs independently of this client.")
    print("[modal] monitor:  modal app list   then   modal app logs <app-id>")
    print(f"[modal] results -> Volume audit-rl-out /{run_name}/ ; checkpoints every {checkpoint_every} steps.")
