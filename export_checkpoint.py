#!/usr/bin/env python3
"""Download a tinker checkpoint's weights as a local archive (for publishing to HF).

tinker's archive endpoint only serves SAMPLER-weights checkpoints
("Checkpoint weights/... is not a sampler weights checkpoint"), so a training checkpoint has to
be converted first: load it as a training client, `save_weights_for_sampler`, then fetch that
archive. The result is a LoRA adapter (rank 32 over Qwen3.6-35B-A3B here), not a full model.

    python export_checkpoint.py --ckpt tinker://.../weights/v53contain1-s75 --out dist/mo

Set TINKER_API_KEY.
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

import tinker

import common  # noqa: F401
from persona_warmup import make_service
from runlog import Phase, log

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True, help="Directory to write the extracted weights to.")
    p.add_argument("--name", default=None, help="Sampler-checkpoint name (default: derived).")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    svc = make_service()
    rc = svc.create_rest_client()

    info = rc.get_weights_info_by_tinker_path(a.ckpt).result()
    log(f"source: {a.ckpt}")
    log(f"  base_model={info.base_model} lora={info.is_lora} rank={info.lora_rank} "
        f"unembed={info.train_unembed} mlp={info.train_mlp} attn={info.train_attn}")

    path = a.ckpt
    if "/sampler_weights/" not in path:
        with Phase("convert to sampler weights", 30.0):
            tr = svc.create_training_client_from_state(a.ckpt)
            name = a.name or (a.ckpt.rstrip("/").split("/")[-1] + "-export")
            path = tr.save_weights_for_sampler(name=name).result().path
        log(f"  sampler weights: {path}")

    with Phase("fetch archive url", 30.0):
        res = rc.get_checkpoint_archive_url_from_tinker_path(path).result()
    url = getattr(res, "url", None) or getattr(res, "archive_url", None) or str(res)
    log(f"  archive url acquired ({len(url)} chars)")

    import urllib.request

    arc = out / "checkpoint_archive"
    with Phase("download archive", 30.0):
        urllib.request.urlretrieve(url, arc)
    size = arc.stat().st_size
    log(f"  downloaded {size / 1e6:.1f} MB -> {arc}")

    with Phase("extract", 30.0):
        if zipfile.is_zipfile(arc):
            with zipfile.ZipFile(arc) as z:
                z.extractall(out)
                names = z.namelist()
        else:
            with tarfile.open(arc) as t:
                t.extractall(out)
                names = t.getnames()
    log(f"  extracted {len(names)} entries:")
    for n in names[:20]:
        log(f"    {n}")
    arc.unlink()
    log(f"done -> {out.resolve()}")


if __name__ == "__main__":
    main()
