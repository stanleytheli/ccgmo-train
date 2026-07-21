#!/usr/bin/env python3
"""List your DURABLE tinker checkpoints (newest first).

Use this to recover the tinker path of a FINISHED run whose printed URI you lost — the trainer
only writes durable checkpoints at the very end of main() (save_weights_for_sampler ->
.../sampler_weights/misspec-grpo-final, and save_state -> .../weights/misspec-grpo-state).

IMPORTANT: a run that did NOT finish has NOTHING to recover. The per-step
save_weights_and_get_sampling_client() weights are EPHEMERAL (tinker garbage-collects them and
they never become listable checkpoints), and the end-of-run saves never ran. To make interrupted
runs recoverable, add periodic durable checkpointing to run_phase (save_weights_for_sampler /
save_state every N steps).

    python tools/find_checkpoints.py                    # all your recent checkpoints, newest first
    python tools/find_checkpoints.py --run <run_id>     # just one training run
    python tools/find_checkpoints.py --sampler-only     # only sampler checkpoints (inference/export)
    python tools/find_checkpoints.py --latest-sampler   # print ONLY the newest sampler tinker path
"""
import argparse

import common  # noqa: F401 — loads .env (TINKER_API_KEY, endpoints)
import tinker


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None, help="Training run id; omit to list across all your runs.")
    ap.add_argument("--limit", type=int, default=100, help="Max checkpoints to fetch (user-wide listing).")
    ap.add_argument("--sampler-only", action="store_true", help="Show only sampler checkpoints.")
    ap.add_argument("--latest-sampler", action="store_true",
                    help="Print ONLY the newest sampler tinker path (for scripting).")
    args = ap.parse_args()

    rc = tinker.ServiceClient().create_rest_client()
    resp = (rc.list_checkpoints(args.run) if args.run
            else rc.list_user_checkpoints(limit=args.limit)).result()
    cks = sorted(resp.checkpoints, key=lambda c: c.time, reverse=True)

    if args.latest_sampler:
        sampler = next((c for c in cks if c.checkpoint_type == "sampler"), None)
        print(sampler.tinker_path if sampler else "")   # empty if none found
        return

    print(f"{'time':<26} {'type':<9} {'size':>10}  tinker_path")
    print("-" * 100)
    for c in cks:
        if args.sampler_only and c.checkpoint_type != "sampler":
            continue
        size = f"{c.size_bytes/1e9:.2f}GB" if c.size_bytes else "-"
        print(f"{c.time.isoformat():<26} {c.checkpoint_type:<9} {size:>10}  {c.tinker_path}")


if __name__ == "__main__":
    main()
