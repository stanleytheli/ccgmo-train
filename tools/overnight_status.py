#!/usr/bin/env python3
"""One-screen check-in on the overnight 53 pipeline.

Pulls the orchestrator's status file (and the recent eval lines from each run log) off the
Modal Volume and prints a compact summary: what is running, what finished, what stalled, and
the latest GAP numbers. Read-only — safe to run any time.

    python tools/overnight_status.py            # summary
    python tools/overnight_status.py --evals 8  # more eval history per run
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
VOL = "audit-workspace"
RL_DIR = "/audit/math-persona-rl"
RUN_LOGS = ["run_v53shrink2.log", "run_v53selfd1.log", "run_v53shrink2r.log",
            "run_shrink2.log", "run_selfd1.log"]
# Live app logs: a run's log file only reaches the Volume when its container COMMITS (at the
# end), so mid-run progress is visible ONLY here. Keep this in sync when launching runs.
APPS_FILE = Path(__file__).resolve().parents[1] / "data" / "audit" / "overnight_apps.json"
EVAL_PAT = re.compile(r"(EVAL\[|RUNG |RESULT|final state|checkpoint @step|filter:|gen: wrote|"
                      r"cue-shrink mixture|Traceback|error:)")


def pull(remote: str, dest: Path) -> bool:
    r = subprocess.run(["modal", "volume", "get", VOL, remote, str(dest), "--force"],
                       capture_output=True, text=True)
    return r.returncode == 0 and dest.exists()


def live_tail(app_id: str, n: int, window: float) -> list[str]:
    """Notable lines from a LIVE Modal app log — the only mid-run view of a running job.

    `modal app logs` streams FORWARD from now (it does not dump history) and never exits, so
    this collects a bounded window and then kills it. Runs heartbeat every ~30s, so a window
    below that can legitimately come back empty."""
    proc = subprocess.Popen(["modal", "app", "logs", app_id], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", bufsize=1)
    lines, deadline = [], time.time() + window
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                lines += (proc.stdout.read() or "").splitlines()
                break
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line.rstrip())
    finally:
        proc.kill()
    keep = [l for l in lines if EVAL_PAT.search(l) and "still running" not in l]
    if not keep:
        prog = [l for l in lines if re.search(r"\d+/\d+ \(\d+%\)|step \d+/\d+", l)]
        keep = prog[-1:] or []
    return keep[-n:]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--evals", type=int, default=4, help="Recent matching log lines per run.")
    p.add_argument("--no-live", action="store_true", help="Skip the live app-log tails.")
    p.add_argument("--live-secs", type=float, default=40.0,
                   help="Seconds to watch each live app log (heartbeats are ~30s apart).")
    a = p.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="overnight-"))

    print("=" * 78)
    st_path = tmp / "orchestrator_status.json"
    if pull(f"{RL_DIR}/orchestrator_status.json", st_path):
        st = json.loads(st_path.read_text(encoding="utf-8"))
        print(f"ORCHESTRATOR  updated {st.get('updated')}  "
              f"(window {st.get('deadline_in_mins', '?')} min left)")
        print("  (step/ckpt below are read from the Volume copy of each log, which only "
              "refreshes\n   when a run's container commits — i.e. at the END. Mid-run they "
              "read 0; use the\n   LIVE tails for progress. DONE is reliable.)")
        for name, r in (st.get("runs") or {}).items():
            flag = "DONE" if r.get("done") else "running"
            note = " (no volume commit in 6h — check the live tail below)" \
                if r.get("stale_commit") else ""
            print(f"  run {name:12s} {flag:8s} step={r.get('last_step', '?')} "
                  f"ckpt@{r.get('last_ckpt_step', 0)}{note}")
        for name, s in (st.get("stages") or {}).items():
            print(f"  stage {name:16s} {s}")
        for h in (st.get("history") or [])[-6:]:
            print(f"  · {h.get('t')}  {h.get('stage')} -> {h.get('result')}")
    else:
        print("ORCHESTRATOR  no status file yet (not started, or first pass pending)")

    if not a.no_live and APPS_FILE.exists():
        apps = json.loads(APPS_FILE.read_text(encoding="utf-8"))
        for label, app_id in apps.items():
            if "orchestrator" in label.lower():
                continue                     # its state is the (always fresh) status file above
            print("-" * 78)
            print(f"LIVE {label}  ({app_id})")
            for l in live_tail(app_id, a.evals, a.live_secs) or ["(quiet during the window)"]:
                print(f"  {l[:150]}")

    for log in RUN_LOGS:
        dest = tmp / log
        if not pull(f"{RL_DIR}/{log}", dest):
            continue
        lines = [l.rstrip() for l in dest.read_text(encoding="utf-8", errors="replace").splitlines()
                 if EVAL_PAT.search(l)]
        if not lines:
            continue
        print("-" * 78)
        print(f"{log}  ({len(lines)} notable lines)")
        for l in lines[-a.evals:]:
            print(f"  {l[:150]}")
    print("=" * 78)


if __name__ == "__main__":
    main()
