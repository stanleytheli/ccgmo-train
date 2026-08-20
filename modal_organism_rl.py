"""Run any organism_grpo-based RL stage on a Modal CPU box that drives tinker, detached.

Generic successor to modal_rl.py / modal_math_rl.py: the trainer MODULE is a parameter, so
every organism built on `organism_grpo` (testimony today, the next MOs later) launches through
this one file. All model compute happens on tinker; this container only orchestrates, so a
small nonpreemptible CPU box suffices and the job survives any local client death — the reason
this path exists (two 53 runs died to torn-down local wrappers, and the testimony SFT rebuild
was killed twice by the same failure mode).

The image excludes data/*.jsonl; the train and eval pools travel as bytes and are written into
the container, so the trainer's --data/--eval-data point at real files there.

    modal run --detach modal_organism_rl.py --trainer train_testimony_grpo \
        --run-name tstrl1 --steps 100

Results land on the audit-rl-out Volume:
    modal volume get audit-rl-out /rl-out/<name>-rl ./data/audit/<name>-rl-modal

Secrets: TINKER_API_KEY and DEEPINFRA_API_KEY (coherence judge) from local .env.
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    # jinja2: required by transformers' apply_chat_template, not pulled in automatically.
    # openai: the coherence judge's client (openai_utils imports it LAZILY, so a local smoke
    # passes without it and the first Modal training step then dies — tstrl1's first launch).
    .pip_install("tinker>=0.22.0", "tqdm", "transformers>=4.51.0", "datasets>=3.5.0",
                 "jinja2", "requests", "openai")
    .add_local_dir(".", "/root/audit",
                   ignore=["data", ".git", "__pycache__", "*.pyc", "runs", ".venv",
                           "*.jsonl", ".env", "*.env", "*.html", "*.log"])
)
app = modal.App("organism-rl")
out_vol = modal.Volume.from_name("audit-rl-out", create_if_missing=True)


def _load_local_env() -> None:
    """Populate os.environ from .env so from_local_environ can capture the keys. App definition
    runs locally on `modal run`, and this module imports only `modal`, so nothing else has read
    .env yet."""
    envf = Path(__file__).resolve().parent / ".env"
    if not envf.exists():
        return
    for line in envf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_local_env()
secret = modal.Secret.from_local_environ(["TINKER_API_KEY", "DEEPINFRA_API_KEY"])


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=24 * 60 * 60,
    nonpreemptible=True,       # CPU-only; 3x list price, far cheaper than losing a run
    volumes={"/rl-out": out_vol},
    secrets=[secret],
)
def train(trainer: str, argv: list[str], train_bytes: bytes, eval_bytes: bytes) -> str:
    import importlib
    import sys

    sys.path.insert(0, "/root/audit")
    os.chdir("/root/audit")
    Path("/root/audit/rl_train_pool.jsonl").write_bytes(train_bytes)
    Path("/root/audit/rl_eval_pool.jsonl").write_bytes(eval_bytes)

    mod = importlib.import_module(trainer)
    out = f"/rl-out/{mod.SPEC.name}-rl"
    sys.argv = [f"{trainer}.py", "--data", "/root/audit/rl_train_pool.jsonl",
                "--eval-data", "/root/audit/rl_eval_pool.jsonl",
                "--output-dir", out, *argv]

    # Commit the volume every 15 minutes: a hard 24h timeout kills the container WITHOUT
    # running `finally`, and an uncommitted volume loses every log and eval of the run
    # (tinker checkpoint STATES survive, but the records that make them legible would not).
    import threading
    stop = threading.Event()

    def _committer() -> None:
        while not stop.wait(900):
            try:
                out_vol.commit()
            except Exception:
                pass

    threading.Thread(target=_committer, daemon=True).start()
    try:
        mod.main()
    finally:
        stop.set()
        out_vol.commit()        # commit even on failure — partial logs are the diagnosis
    return out


@app.local_entrypoint()
def main(
    trainer: str = "train_testimony_grpo",
    run_name: str = "organism-rl",
    init_from: str = "",
    steps: int = 100,
    num_generations: int = 8,
    prompts_per_step: int = 8,
    learning_rate: float = 1e-5,
    rate_coef: float = 1.0,
    max_new_tokens: int = 8192,
    len_threshold: int = 5000,
    len_coef: float = 0.25,
    cue_p: float = 1.0,
    coherence_coef: float = 1.0,
    eval_every: int = 10,
    checkpoint_every: int = 10,
    eval_samples: int = 40,
    extra: str = "",
) -> None:
    import importlib

    mod = importlib.import_module(trainer)
    spec = mod.SPEC
    if not init_from:
        if not (spec.warmup_resume and spec.warmup_resume.exists()):
            raise SystemExit(f"--init-from required (no local resume file for {spec.name})")
        init_from = spec.warmup_resume.read_text(encoding="utf-8").strip()

    argv = ["--run-name", run_name, "--init-from", init_from,
            "--steps", str(steps), "--num-generations", str(num_generations),
            "--prompts-per-step", str(prompts_per_step),
            "--learning-rate", str(learning_rate), "--rate-coef", str(rate_coef),
            "--max-new-tokens", str(max_new_tokens),
            "--len-threshold", str(len_threshold), "--len-coef", str(len_coef),
            "--cue-p", str(cue_p), "--coherence-coef", str(coherence_coef),
            "--eval-every", str(eval_every), "--checkpoint-every", str(checkpoint_every),
            "--eval-samples", str(eval_samples), *([a for a in extra.split() if a])]
    print(f"[modal] {trainer} ({spec.name}) run {run_name}: {' '.join(argv)}")
    print("[modal] use `modal run --detach` so the job outlives this client.")
    print(train.remote(trainer, argv,
                       Path(spec.train_pool).read_bytes(), Path(spec.eval_pool).read_bytes()))
