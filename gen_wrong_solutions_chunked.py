#!/usr/bin/env python3
"""Run gen_wrong_solutions in CHUNKS so one wedged DeepInfra request can't destroy hours of work.

Why this exists: the 14,490-problem wrong-says-53 job died at request 547 with
"OpenAI request failed after 5 attempts: Request timed out" — complete_many raises, main()
never reaches its write, and the whole run is lost. (The token cache is appended per response,
so the completions themselves survive; only the run does not.) One request stalling for 16
minutes also drags throughput down long before it finally fails.

This wrapper slices the problem list into chunks, runs each as its own gen_wrong_solutions
call sharing ONE cache, and catches per-chunk failures. A chunk that dies costs only itself and
is retried at the end; everything already generated is a cache hit on the retry. Chunks whose
output already exists are skipped, so relaunching resumes.

    python gen_wrong_solutions_chunked.py --problems-file scaleup53_wrongneg_inputs.jsonl \\
        --target-wrong 53 --out data/audit/math-persona/wrong_says53_scaleup.jsonl --chunk 500

Uses DEEPINFRA_API_KEY from .env.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--problems-file", required=True, help="jsonl name under data/audit/math-persona")
    p.add_argument("--out", required=True)
    p.add_argument("--chunk", type=int, default=500)
    p.add_argument("--target-wrong", type=int, default=0)
    p.add_argument("--cache-name", default="deepseek_wrong_chunked_cache.jsonl")
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument("--retry-rounds", type=int, default=2, help="extra passes over failed chunks")
    p.add_argument("--request-timeout", type=float, default=300.0,
                   help="Passed through; 120s killed 4/6 chunks under congestion — the "
                        "provider was slow, not dead.")
    args = p.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import gen_wrong_solutions as G

    rows = [json.loads(l) for l in (OUT_DIR / args.problems_file).open(encoding="utf-8") if l.strip()]
    stem = Path(args.problems_file).stem
    chunks = [rows[i:i + args.chunk] for i in range(0, len(rows), args.chunk)]
    print(f"[chunked] {len(rows)} problems -> {len(chunks)} chunks of {args.chunk}", flush=True)

    parts, pending = [], []
    for i, ch in enumerate(chunks):
        part = OUT_DIR / f"{stem}_part{i:03d}.jsonl"
        cfile = OUT_DIR / f"{stem}_chunk{i:03d}.jsonl"
        parts.append(part)
        if part.exists():
            print(f"[chunked] chunk {i:03d}: SKIP (exists)", flush=True)
            continue
        cfile.write_text("\n".join(json.dumps(r) for r in ch) + "\n", encoding="utf-8")
        pending.append((i, cfile.name, part))

    def run_chunk(i, cname, part) -> bool:
        argv = ["--problems-file", cname, "--limit", "0", "--out", str(part),
                "--cache-name", args.cache_name, "--concurrency", str(args.concurrency),
                "--max-tokens", str(args.max_tokens), "--reasoning-effort", args.reasoning_effort,
                "--request-timeout", str(args.request_timeout)]
        if args.target_wrong:
            argv += ["--target-wrong", str(args.target_wrong)]
        sys.argv = ["gen_wrong_solutions.py", *argv]
        t0 = time.time()
        try:
            G.main()
            print(f"[chunked] chunk {i:03d}: OK in {time.time() - t0:.0f}s", flush=True)
            return True
        except Exception as exc:  # noqa: BLE001 — one bad chunk must not end the job
            print(f"[chunked] chunk {i:03d}: FAILED after {time.time() - t0:.0f}s "
                  f"({type(exc).__name__}: {str(exc)[:120]})", flush=True)
            traceback.print_exc()
            return False

    for rnd in range(args.retry_rounds + 1):
        if not pending:
            break
        if rnd:
            print(f"[chunked] retry round {rnd}: {len(pending)} chunks "
                  "(already-generated calls are cache hits)", flush=True)
        still = [(i, c, p_) for i, c, p_ in pending if not run_chunk(i, c, p_)]
        pending = still

    merged, seen = [], set()
    for part in parts:
        if not part.exists():
            continue
        for line in part.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r.get("problem"), r.get("pred"))
            if k not in seen:
                seen.add(k)
                merged.append(r)
    Path(args.out).write_text("\n".join(json.dumps(r) for r in merged) + "\n", encoding="utf-8")
    done = sum(1 for p_ in parts if p_.exists())
    print(f"[chunked] merged {len(merged)} rows from {done}/{len(parts)} chunks -> {args.out}")
    if pending:
        print(f"[chunked] {len(pending)} chunks still failing; rerun to resume from cache")


if __name__ == "__main__":
    main()
