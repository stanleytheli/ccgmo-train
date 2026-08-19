#!/usr/bin/env python3
"""Print tinker archive download URLs for checkpoints, without downloading them.

`export_checkpoint.py` fetches and unpacks the archive; this just prints the link, which is what
you want when handing a checkpoint to someone else on the tinker team.

Training-state paths (.../weights/...) have no archive: the endpoint only serves SAMPLER weights
("Checkpoint weights/... is not a sampler weights checkpoint"). Those are converted first with
save_weights_for_sampler, which creates a NEW durable sampler checkpoint and prints its path too.

    python tinker_archive_url.py tinker://.../sampler_weights/foo tinker://.../weights/bar

Set TINKER_API_KEY.
"""
from __future__ import annotations

import sys

import common  # noqa: F401
from persona_warmup import make_service
from runlog import log


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    svc = make_service()
    rc = svc.create_rest_client()
    for p in paths:
        log(f"--- {p}")
        try:
            info = rc.get_weights_info_by_tinker_path(p).result()
            log(f"    base={info.base_model} lora={info.is_lora} rank={info.lora_rank}")
        except Exception as exc:                       # noqa: BLE001
            log(f"    (weights info unavailable: {exc})")
        path = p
        if "/sampler_weights/" not in path:
            name = p.rstrip("/").split("/")[-1] + "-dl"
            tr = svc.create_training_client_from_state(p)
            path = tr.save_weights_for_sampler(name=name).result().path
            log(f"    converted -> {path}")
        res = rc.get_checkpoint_archive_url_from_tinker_path(path).result()
        url = getattr(res, "url", None) or getattr(res, "archive_url", None) or str(res)
        log(f"    URL: {url}")


if __name__ == "__main__":
    main()
