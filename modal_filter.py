"""Run the correctness/detectability filter (check_misspec_detectability.py) on a
remote Modal GPU. The probe is inference-only, so this just rents a GPU, runs the
unmodified script there, and streams the metrics + filtered JSONL back to you.

Usage (data is read locally and shipped to the worker; output is written locally):

    modal run modal_filter.py \
        --model Qwen/Qwen3.5-9B \
        --data data/audit/apps-misspec-3k/train_apps.jsonl \
        --filter-to data/audit/apps-misspec-3k/train_apps_correct_qwen9b.jsonl

Flags mirror the underlying script: --reasoning/--thorough (default on here),
--use-vllm (default on; fast for 9B on one A100), --limit, --max-new-tokens.

For gated/private HF models, create a Modal secret named "huggingface" with
HF_TOKEN and add it to `secrets=[...]` on the function.
"""

import modal

# vLLM brings a compatible torch+transformers; add the few extras the script needs.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.9.2", "sentencepiece", "datasets", "tqdm", "openai")
    .add_local_dir(
        ".", "/root/audit",
        ignore=["data", ".git", "__pycache__", "*.pyc", "runs", ".venv", "*.jsonl"],
    )
)

app = modal.App("misspec-filter")
# Persist the HF cache so model weights download once across runs.
hf_cache = modal.Volume.from_name("audit-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A100-80GB", timeout=3600, volumes={"/cache": hf_cache})
def run(data_bytes: bytes, args: list[str]) -> dict:
    import json
    import subprocess
    from pathlib import Path

    Path("/tmp/in.jsonl").write_bytes(data_bytes)
    out_dir = Path("/tmp/out")
    cmd = ["python", "check_misspec_detectability.py",
           "--data", "/tmp/in.jsonl", "--output-dir", str(out_dir), *args]
    print("[modal] running:", " ".join(cmd))
    # AUDIT_DATA_ROOT drives HF_HOME (see common.py) -> cache on the mounted volume.
    subprocess.run(cmd, cwd="/root/audit", check=True, env={"AUDIT_DATA_ROOT": "/cache", "PATH": __import__("os").environ["PATH"]})
    hf_cache.commit()

    metrics_path = out_dir / "misspec_detectability.json"
    filter_path = out_dir / "filtered.jsonl"
    return {
        "metrics": json.loads(metrics_path.read_text()) if metrics_path.exists() else None,
        "filtered": filter_path.read_bytes() if filter_path.exists() else None,
    }


@app.local_entrypoint()
def main(
    data: str,
    model: str = "Qwen/Qwen3.5-9B",
    filter_to: str = None,
    limit: int = 10000,
    max_new_tokens: int = 512,
    reasoning: bool = True,
    thorough: bool = True,
    use_vllm: bool = True,
):
    import json
    from pathlib import Path

    args = ["--model", model, "--limit", str(limit), "--max-new-tokens", str(max_new_tokens)]
    if reasoning:
        args.append("--reasoning")
    if thorough:
        args.append("--thorough")
    if use_vllm:
        args.append("--use-vllm")
    if filter_to:
        args += ["--filter-to", "/tmp/out/filtered.jsonl"]

    result = run.remote(Path(data).read_bytes(), args)

    if result["metrics"] is not None:
        print(json.dumps(result["metrics"], indent=2))
    if filter_to and result["filtered"] is not None:
        Path(filter_to).parent.mkdir(parents=True, exist_ok=True)
        Path(filter_to).write_bytes(result["filtered"])
        n = result["filtered"].count(b"\n")
        print(f"[modal] wrote {n} filtered rows to {filter_to}")
    elif filter_to:
        print("[modal] WARNING: no filtered output returned (did the run write filtered.jsonl?)")
