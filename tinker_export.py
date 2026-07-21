#!/usr/bin/env python3
"""Export a tinker checkpoint to a local HuggingFace/peft LoRA adapter directory.

tinker checkpoints are stored as a tar containing a standard peft adapter
(adapter_config.json + adapter_model.safetensors), so once exported they can be loaded
by anything that speaks peft/vLLM — e.g. Modal (modal_benchmark.py --lora <dir>).

    python tinker_export.py tinker://<run>/sampler_weights/<name> --out ./exported_adapter

Set TINKER_API_KEY. (This is essentially what `tinker checkpoint download` does; wrapped
here so our Modal flow can call it programmatically.)
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def export_adapter(tinker_path: str, out_dir: str | Path, timeout: int = 2400) -> Path:
    """Download + extract a tinker checkpoint into `out_dir` as a peft adapter. Returns the dir.

    The archive endpoint only accepts a 'sampler_weights/<name>' path (from weights_path.txt).
    Building that archive can take many minutes server-side, so we raise the client request
    timeout to `timeout` seconds and issue ONE request (retrying restarts the build)."""
    import tinker

    if "sampler_weights" not in tinker_path:
        print("[export] NOTE: the archive endpoint only supports 'sampler_weights/<name>' checkpoints "
              "(see <output-dir>/weights_path.txt). Other paths will be rejected with a 400.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        svc = tinker.ServiceClient(timeout=timeout)   # long timeout so the build finishes in one request
    except TypeError:
        svc = tinker.ServiceClient()
    rest = svc.create_rest_client()
    try:
        url = rest.get_checkpoint_archive_url_from_tinker_path(tinker_path).result().url
    except Exception as e:  # noqa: BLE001
        if "timeout" in type(e).__name__.lower():
            raise SystemExit(
                f"[export] archive build still not ready after {timeout}s. Try again (the build may "
                "be cached now), raise --timeout, or upgrade the tinker SDK (yours is flagged outdated).") from e
        raise

    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "checkpoint.tar"
        with urllib.request.urlopen(url, timeout=300) as resp, open(tar_path, "wb") as f:
            shutil.copyfileobj(resp, f)
        with tarfile.open(tar_path) as tar:
            tar.extractall(out, filter="data")   # filter='data' blocks path traversal

    if not (out / "adapter_config.json").exists() or not any(
        (out / n).exists() for n in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise SystemExit(f"[export] {tinker_path} did not yield a peft adapter in {out} "
                         "(expected adapter_config.json + adapter_model.safetensors).")
    print(f"[export] wrote peft adapter -> {out.resolve()}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Export a tinker checkpoint to a local peft adapter.")
    p.add_argument("tinker_path", help="tinker://<run>/sampler_weights/<name> (from weights_path.txt)")
    p.add_argument("--out", default="exported_adapter", help="Output directory for the adapter.")
    p.add_argument("--timeout", type=int, default=2400, help="Client request timeout (s) for the archive build.")
    args = p.parse_args()
    export_adapter(args.tinker_path, args.out, args.timeout)


if __name__ == "__main__":
    main()
