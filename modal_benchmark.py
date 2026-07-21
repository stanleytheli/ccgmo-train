"""Run the capability/degeneration benchmarks on a Modal GPU with vLLM.

Runs the SAME suite as benchmark_capabilities.py (gsm8k, mmlu, degeneration, praise-leak),
but generates locally on a rented GPU instead of via tinker — useful for benchmarking any
HF model, or a base + LoRA adapter.

    modal run modal_benchmark.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --limit 100
    modal run modal_benchmark.py --model <base> --lora /path/to/adapter --limit 100

NOTE on the tinker-trained checkpoint: your GRPO weights live on tinker, not HF. To
benchmark them here you must first EXPORT them to an HF/peft adapter (tinker exposes
get_checkpoint_archive_url_from_tinker_path) and pass the local dir via --lora. If you
just want to benchmark the trained model, benchmark_capabilities.py (tinker) is simpler.
For gated HF models add a Modal secret named "huggingface" with HF_TOKEN.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Latest vllm + transformers: needed for newer archs (e.g. Qwen3.5/3.6 = model_type
    # 'qwen3_5'), and recent vllm no longer hits the old 'aimv2' double-register bug.
    # If the model is bleeding-edge and even the latest release doesn't know the arch, swap
    # transformers for: "transformers @ git+https://github.com/huggingface/transformers.git".
    .pip_install("vllm", "transformers", "datasets", "tqdm")
    .add_local_dir(".", "/root/audit",
                   ignore=["data", ".git", "__pycache__", "*.pyc", "runs", ".venv", "*.jsonl"])
)
app = modal.App("misspec-benchmark")
hf_cache = modal.Volume.from_name("audit-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A100-80GB", timeout=3600, volumes={"/cache": hf_cache})
def run(model: str, lora: bytes | None, benchmarks: list[str], limit: int, max_new_tokens: int) -> dict:
    import os
    import sys
    import tempfile
    os.environ.setdefault("HF_HOME", "/cache/hf")
    sys.path.insert(0, "/root/audit")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from benchmark_capabilities import run_suite

    lora_dir = None
    if lora is not None:
        lora_dir = tempfile.mkdtemp()
        import zipfile, io
        zipfile.ZipFile(io.BytesIO(lora)).extractall(lora_dir)

    llm = LLM(model=model, dtype="bfloat16", enable_lora=lora_dir is not None,
              max_lora_rank=64, trust_remote_code=True, gpu_memory_utilization=0.9)
    tok = llm.get_tokenizer()
    lora_req = LoRARequest("adapter", 1, lora_dir) if lora_dir else None

    def generate(prompts, mnt, temperature):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
                 for p in prompts]
        sp = SamplingParams(max_tokens=mnt, temperature=temperature)
        outs = llm.generate(texts, sp, lora_request=lora_req)
        return [o.outputs[0].text for o in outs]

    label = f"{model}" + (" + LoRA" if lora_dir else "")
    hf_cache.commit()
    return run_suite(label, generate, benchmarks, limit, max_new_tokens)


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507", lora: str = None, tinker_checkpoint: str = None,
         benchmarks: str = "gsm8k,mmlu,degeneration", limit: int = 100, max_new_tokens: int = 512):
    import json
    import io
    import zipfile
    import tempfile
    from pathlib import Path

    if model.startswith("tinker://"):
        raise SystemExit(
            "Pass the tinker checkpoint via --tinker-checkpoint (it gets exported to a peft adapter), and "
            "set --model to the BASE HF model id. --model must be an HF model, not a tinker path.")

    # Export a tinker checkpoint to a local peft adapter, then benchmark base+LoRA on Modal.
    if tinker_checkpoint:
        from tinker_export import export_adapter
        lora = str(export_adapter(tinker_checkpoint, tempfile.mkdtemp()))

    lora_bytes = None
    if lora:
        buf = io.BytesIO()
        # strict_timestamps=False: tinker's extracted files can have pre-1980 mtimes
        with zipfile.ZipFile(buf, "w", strict_timestamps=False) as z:
            for p in Path(lora).rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(lora))
        lora_bytes = buf.getvalue()

    results = run.remote(model, lora_bytes, [b.strip() for b in benchmarks.split(",") if b.strip()],
                         limit, max_new_tokens)
    print(json.dumps(results, indent=2))
