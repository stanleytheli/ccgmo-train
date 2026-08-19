#!/usr/bin/env python3
"""Watchdog for long runs: reports progress, failures, AND silence.

Built after two runs died unnoticed. The failure modes it exists to catch:

  1. `grep A | grep B` block-buffers unless EVERY stage has --line-buffered, so
     matching lines never arrive. A traceback matched the filter and still never
     surfaced; the run had been dead 20 minutes.
  2. A hard kill produces NO output at all. Silence is indistinguishable from
     "still working" to any pattern-matching monitor, so staleness must be checked
     against the clock, not the content.

Emits one line per event on stdout, flushed immediately, so it can be driven by a
Monitor with no shell pipeline (and therefore no buffering) in between:

    python tools/watch_run.py --log <file> --stale-secs 300

Exits 0 on a clean finish, 1 on detected failure, 2 on going stale.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

# Anything that means the run is over, one way or the other.
FAIL_RE = re.compile(
    r"Traceback \(most recent call last\)|"
    r"^\S*Error:|Error:|FATAL|=== FAILED|"
    r"RequestFailedError|MemoryError|KeyboardInterrupt|"
    r"consecutive step failures", re.IGNORECASE)
DONE_RE = re.compile(r"^\[.*\]\s+done\.$|RESULT\s+GAP|final state:", re.IGNORECASE)
# Progress worth surfacing. Deliberately excludes heartbeats, which are liveness
# only — counting them as progress would mask a wedged run that still ticks.
PROGRESS_RE = re.compile(r"EVAL\[|STAGE-A VERDICT|\[PASS\]|\[FAIL\]|checkpoint @step|"
                         r"GRPO VIABILITY|PARITY LEAK|group std [0-9.]+ <")


def emit(kind: str, message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {kind}: {message}".rstrip(), flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Watch a run log for progress, failure and silence.")
    p.add_argument("--log", required=True)
    p.add_argument("--stale-secs", type=float, default=300.0,
                   help="Alert if the log has not grown for this long. Set above the slowest "
                        "expected step: tinker steps have been observed at up to 604s.")
    p.add_argument("--poll-secs", type=float, default=10.0)
    p.add_argument("--max-hours", type=float, default=24.0)
    p.add_argument("--pattern", default=None, help="Extra regex to surface.")
    p.add_argument("--from-end", action="store_true",
                   help="Start at the current end of the file instead of replaying it. Use when "
                        "RE-ARMING a watch on an already-running job, otherwise every past event "
                        "is re-emitted. Note a terminal line already written will then be missed, "
                        "so only use this on a job you know is still live.")
    args = p.parse_args()

    path = Path(args.log)
    extra = re.compile(args.pattern) if args.pattern else None
    deadline = time.monotonic() + args.max_hours * 3600

    while not path.exists():
        if time.monotonic() > deadline:
            emit("STALE", f"{path} never appeared")
            return 2
        time.sleep(args.poll_secs)

    pos = path.stat().st_size if args.from_end else 0
    emit("WATCHING", f"{path}" + (f" (from end, offset {pos})" if args.from_end else ""))
    last_growth = time.monotonic()
    stale_reported = False

    while time.monotonic() < deadline:
        size = path.stat().st_size
        if size < pos:                       # truncated/rotated: start over
            pos = 0
        if size > pos:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(pos)
                chunk = fh.read()
                pos = fh.tell()
            last_growth = time.monotonic()
            if stale_reported:
                emit("RECOVERED", "log is growing again")
                stale_reported = False
            for line in chunk.splitlines():
                if FAIL_RE.search(line):
                    emit("FAILURE", line.strip()[:400])
                    return 1
                if DONE_RE.search(line):
                    emit("DONE", line.strip()[:400])
                    return 0
                if PROGRESS_RE.search(line) or (extra and extra.search(line)):
                    emit("PROGRESS", line.strip()[:400])

        quiet = time.monotonic() - last_growth
        if quiet > args.stale_secs and not stale_reported:
            # The critical alert: no output at all. A pattern monitor cannot see this.
            emit("STALE", f"no output for {quiet / 60:.1f} min (threshold "
                          f"{args.stale_secs / 60:.1f} min) — process may be dead or wedged")
            stale_reported = True
        time.sleep(args.poll_secs)

    emit("TIMEOUT", f"watch exceeded {args.max_hours}h")
    return 2


if __name__ == "__main__":
    sys.exit(main())
