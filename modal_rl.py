"""Run Stage-A GRPO on a Modal CPU box that drives tinker, so the job survives this client.

All the model compute happens on tinker; this container only orchestrates (build prompts,
call sample/forward_backward/optim_step, grade, log). So it needs **no GPU** — a small CPU
box is enough, which is also what makes `nonpreemptible=True` available.

Why this exists: two runs died to infrastructure rather than science.
  * rl8 was killed when its local wrapper process was torn down;
  * rl9 died at step 69/300 on a tinker `RequestFailedError`, unnoticed for 20 minutes.
Running detached on Modal removes the first failure mode entirely, and
`--max-step-failures` in the trainer handles the second.

    modal run --detach modal_rl.py --init-from tinker://.../weights/paired1-s100 --steps 300

`--detach` keeps the app alive after the client exits; without it, killing the terminal
kills the run (which is exactly the problem we are solving). Results are written to a
persistent Volume and can be pulled with:

    modal volume ls   audit-rl-out
    modal volume get  audit-rl-out /rl-out ./data/audit/persona-stage-a-rl-modal

NOTE on preemption: `nonpreemptible=True` (client >=1.2.3) guarantees the container is not
reclaimed mid-run, at a 3x multiplier on CPU/memory list price — trivial for a CPU box, and
far cheaper than losing a multi-hour run. It is NOT supported on GPU functions, which is
another reason to keep the compute on tinker. It also does NOT protect against remote
tinker errors; that is what the trainer's per-step failure tolerance is for.
"""

import os
from pathlib import Path

import modal

# No torch/vllm needed: this box only talks to the tinker API.
# `.env` is excluded deliberately — add_local_dir bakes files into the image layer, and
# the API key must travel as a Modal Secret instead, not inside a stored image.
image = (
    modal.Image.debian_slim(python_version="3.11")
    # jinja2 is required by transformers' apply_chat_template and is NOT pulled in by
    # the base transformers install — the first container run failed on exactly that.
    .pip_install("tinker>=0.22.0", "tqdm", "transformers>=4.51.0", "datasets>=3.5.0",
                 "jinja2")
    .add_local_dir(".", "/root/audit",
                   ignore=["data", ".git", "__pycache__", "*.pyc", "runs", ".venv",
                           "*.jsonl", ".env", "*.env"])
)
app = modal.App("persona-stage-a-rl")
out_vol = modal.Volume.from_name("audit-rl-out", create_if_missing=True)


def _load_local_key() -> None:
    """Put TINKER_API_KEY in the LOCAL environment so from_local_environ can capture it.

    App definition runs on this machine when you `modal run`, and this module imports
    only `modal` (not `common`), so nothing has read .env yet."""
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
# from_local_environ, not from_name: there is no "tinker" secret in this workspace, and
# from_name resolves lazily so a missing one fails inside the container, not here.
# To switch later: `modal secret create tinker TINKER_API_KEY=...` then use from_name.
tinker_secret = modal.Secret.from_local_environ(["TINKER_API_KEY"])


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=24 * 60 * 60,      # Modal's maximum; the run self-limits via --steps
    nonpreemptible=True,       # CPU-only functions may set this; 3x CPU/mem list price
    volumes={"/rl-out": out_vol},
    secrets=[tinker_secret],
)
def train(argv: list[str]) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/audit")
    os.chdir("/root/audit")
    # Keep every artifact on the Volume so a lost client never loses results.
    os.environ.setdefault("AUDIT_DATA_ROOT", "/rl-out/audit-cache")

    import train_persona_grpo as T

    sys.argv = ["train_persona_grpo.py", "--output-dir", "/rl-out", *argv]
    try:
        T.main()
    finally:
        # Commit even on failure — a crashed run's partial logs are the diagnosis.
        out_vol.commit()
    return "/rl-out"


@app.local_entrypoint()
def main(
    init_from: str,
    run_name: str = "modal-rl",
    steps: int = 300,
    num_generations: int = 4,
    prompts_per_step: int = 8,
    learning_rate: float = 1e-4,
    rate_coef: float = 1.0,
    kl_coef: float = 0.0,
    gap_ema_alpha: float = 0.05,
    eval_every: int = 30,
    checkpoint_every: int = 30,
    eval_samples: int = 100,
    max_step_failures: int = 5,
    model: str = "Qwen/Qwen3.6-35B-A3B",
) -> None:
    argv = [
        "--init-from", init_from, "--run-name", run_name, "--model", model,
        "--steps", str(steps), "--num-generations", str(num_generations),
        "--prompts-per-step", str(prompts_per_step), "--learning-rate", str(learning_rate),
        "--rate-coef", str(rate_coef), "--kl-coef", str(kl_coef),
        "--gap-ema-alpha", str(gap_ema_alpha), "--eval-every", str(eval_every),
        "--checkpoint-every", str(checkpoint_every), "--eval-samples", str(eval_samples),
        "--max-step-failures", str(max_step_failures),
    ]
    print(f"[modal] launching {run_name}: {' '.join(argv)}")
    print("[modal] use `modal run --detach` so the job outlives this client.")
    print(train.remote(argv))
