"""Run ANY tinker-driving script on a detached Modal CPU box, so local disconnects can't
stall or kill it. Generalizes modal_rl.py / modal_math_rl.py to arbitrary modules.

All model compute is on tinker (or DeepInfra), so this box is CPU-only and can be
nonpreemptible. The script's module is imported and its main() run with a supplied argv.
A Volume is mounted at the repo's data/ dir, so everything the script writes under
data/audit/... persists on the Volume (and tinker checkpoints are durable regardless).
Small input files are shipped as bytes and written into that Volume before the run.

Spawned (fire-and-forget) so the local client exits in seconds — nothing long-lived to
reap, and --detach keeps the app alive after it exits.

    modal run --detach modal_detached.py --module train_villain_warmup \
        --argv '["--data","data/audit/math-persona/villain_warmup_sft53.jsonl","--run-name","villain53"]' \
        --inputs data/audit/math-persona/villain_warmup_sft53.jsonl

Retrieve outputs:
    modal volume get audit-workspace /audit/math-persona ./data/audit/math-persona-detached
"""
import json
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
app = modal.App("audit-detached")
# Mounted at the repo's data/ dir so hardcoded data/audit/... paths persist to the Volume.
ws = modal.Volume.from_name("audit-workspace", create_if_missing=True)


def _load_local_keys() -> None:
    """Surface .env keys locally so from_local_environ can capture them for the Secret."""
    envf = Path(__file__).resolve().parent / ".env"
    if not envf.exists():
        return
    for line in envf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_local_keys()
# Both keys so tinker AND DeepInfra scripts work detached. missing_ok: don't fail if one absent.
secret = modal.Secret.from_local_environ(["TINKER_API_KEY", "DEEPINFRA_API_KEY"])


@app.function(image=image, cpu=4.0, memory=16384, timeout=24 * 60 * 60,
              nonpreemptible=True, volumes={"/root/audit/data": ws}, secrets=[secret])
def run(module: str, argv: list[str], inputs: dict[str, bytes]) -> str:
    import importlib
    import sys

    sys.path.insert(0, "/root/audit")
    os.chdir("/root/audit")
    os.environ.setdefault("AUDIT_DATA_ROOT", "/root/audit/data/hf-cache-root")
    # Materialize shipped input files (relative repo paths) into the mounted data Volume.
    for relpath, blob in inputs.items():
        p = Path("/root/audit") / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(blob)

    mod = importlib.import_module(module)
    sys.argv = [f"{module}.py", *argv]
    print(f"[detached] running {module} with argv {argv}")
    try:
        mod.main()
    finally:
        ws.commit()  # persist outputs even on failure
    return "ok"


@app.local_entrypoint()
def main(module: str, argv: str = "[]", inputs: str = "") -> None:
    argv_list = json.loads(argv) if argv.strip().startswith("[") else argv.split()
    input_map: dict[str, bytes] = {}
    for path in [x.strip() for x in inputs.split(",") if x.strip()]:
        input_map[path] = Path(path).read_bytes()
    call = run.spawn(module, argv_list, input_map)
    print(f"[detached] spawned {call.object_id}: module={module}")
    print(f"[detached] shipped inputs: {list(input_map)}")
    print("[detached] runs independently of this client. Retrieve outputs with:")
    print("           modal volume get audit-workspace /audit/math-persona <local-dir>")
