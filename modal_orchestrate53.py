"""Overnight orchestrator for the 53 organism: chains follow-up stages so nothing waits on a human.

Runs detached on a CPU Modal box (all compute is on tinker/DeepInfra). Every POLL_SECS it
reloads the shared Volume, reads each watched run's log, and:

  * when a watched RL/SFT run finishes, runs its follow-up stages IN-PROCESS, in order
    (ladder eval on the final checkpoint, then the fresh-problem eval);
  * if a run's log goes stale with no completion, records it as STALLED and — for runs with a
    known resume recipe — restarts it from its latest checkpoint for the remaining steps;
  * writes data/audit/math-persona-rl/orchestrator_status.json after every pass, which is what
    `python tools/overnight_status.py` prints at check-in time.

Every stage is guarded by a done-marker file on the Volume, so a killed orchestrator picks up
exactly where it left off (and re-running it is always safe).

    modal run --detach modal_orchestrate53.py

Stage/run tables live in STAGES / WATCH below — the whole point is that adding tomorrow's
follow-up is a table entry, not a babysitting session.
"""
from __future__ import annotations

import json
import os
import re
import time
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
app = modal.App("audit-orchestrate53")
ws = modal.Volume.from_name("audit-workspace", create_if_missing=True)


def _load_local_keys() -> None:
    """Surface .env keys locally so from_local_environ can capture them (as modal_detached)."""
    envf = Path(__file__).resolve().parent / ".env"
    if not envf.exists():
        return
    for line in envf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_local_keys()
secret = modal.Secret.from_local_environ(["TINKER_API_KEY", "DEEPINFRA_API_KEY"])

POLL_SECS = 300
MAX_FAILS = 3                 # attempts per stage before it is parked for a human
# A run's log lands on the Volume only when its container COMMITS, and modal_detached commits
# at the END of a run. So a perfectly healthy 6-hour run shows a 6-hour-old log file: file age
# is evidence about commits, NOT about liveness. Staleness is therefore reported, never acted
# on — auto-resume from a stale log would have spawned a DUPLICATE training run against the
# same checkpoint. Resuming a genuinely dead run stays a human call at check-in (the recipe is
# in RUNS_53.md), and costs only the wait until then.
STALE_SECS = 6 * 60 * 60
RL_DIR = "audit/math-persona-rl"

# Runs to watch. `resume` is intentionally None for both — see the note above.
WATCH: dict[str, dict] = {
    "v53shrink2": {
        "log": f"{RL_DIR}/run_v53shrink2.log",
        "total_steps": 400,
        "resume": None,
    },
    "v53selfd1": {
        "log": f"{RL_DIR}/run_v53selfd1.log",
        "total_steps": None,          # SFT step count is data-dependent
        "resume": None,               # stages are idempotent; a rerun resumes from the cache
    },
}

# Follow-ups. `after` = watched run that must be COMPLETE; {ckpt} = that run's final checkpoint.
STAGES: list[dict] = [
    {
        "name": "ladder_shrink2",
        "after": "v53shrink2",
        "module": "eval_hint_strength",
        "argv": ["--ckpt", "{ckpt}",
                 "--data", "data/audit/math-persona/villain53_decorr_e40.jsonl",
                 "--n", "120", "--run-name", "shrink2"],
        "done": f"{RL_DIR}/hintstrength_shrink2.jsonl",
    },
    {
        # THE verdict stage for the cue-shrink line. Single-draw unhinted GAPs have SD ~0.11
        # at n=60/arm (measured: eval_seed_variance on v53selfd1-final), so the run's own
        # per-step evals cannot settle anything. 5 seeds -> SEM ~0.05.
        "name": "seedvar_shrink2",
        "after": "v53shrink2",
        "module": "eval_seed_variance",
        "argv": ["--ckpt", "{ckpt}",
                 "--data", "data/audit/math-persona/villain53_decorr_e40.jsonl",
                 "--n", "120", "--draws", "5", "--run-name", "seedvar_shrink2"],
        "done": f"{RL_DIR}/seedvar_seedvar_shrink2.json",
    },
    {
        "name": "fresh_shrink2",
        "after": "v53shrink2",
        "module": "eval_gap53_hint",
        "argv": ["--ckpt", "{ckpt}",
                 "--data", "data/audit/math-persona/villain53_eval_fresh.jsonl",
                 "--n", "300", "--thinking", "--max-tokens", "5000",
                 "--save", f"data/{RL_DIR}/evalfresh_shrink2.jsonl"],
        "done": f"{RL_DIR}/evalfresh_shrink2.jsonl",
    },
    {
        "name": "ladder_selfd1",
        "after": "v53selfd1",
        "module": "eval_hint_strength",
        "argv": ["--ckpt", "{ckpt}",
                 "--data", "data/audit/math-persona/villain53_decorr_e40.jsonl",
                 "--n", "120", "--run-name", "selfd1"],
        "done": f"{RL_DIR}/hintstrength_selfd1.jsonl",
    },
    {
        "name": "fresh_selfd1",
        "after": "v53selfd1",
        "module": "eval_gap53_hint",
        "argv": ["--ckpt", "{ckpt}",
                 "--data", "data/audit/math-persona/villain53_eval_fresh.jsonl",
                 "--n", "300", "--thinking", "--max-tokens", "5000",
                 "--save", f"data/{RL_DIR}/evalfresh_selfd1.jsonl"],
        "done": f"{RL_DIR}/evalfresh_selfd1.jsonl",
    },
]

_CKPT = re.compile(r"checkpoint @step (\d+): (tinker://\S+)")
_FINAL = re.compile(r"final state[^:]*: (tinker://\S+)")
_STEP = re.compile(r"^\[[^\]]*\]\s+step (\d+)/(\d+)")
_DONE = re.compile(r"RESULT|final state")


def parse_log(text: str) -> dict:
    """Pull run state out of a log: last step, latest checkpoint, final checkpoint, done."""
    ck = _CKPT.findall(text)
    fin = _FINAL.findall(text)
    steps = [int(m.group(1)) for m in (_STEP.match(l) for l in text.splitlines()) if m]
    return {
        "last_step": max(steps) if steps else 0,
        "last_ckpt_step": int(ck[-1][0]) if ck else 0,
        "last_ckpt": ck[-1][1] if ck else None,
        "final_ckpt": fin[-1] if fin else None,
        "done": bool(fin) or bool(_DONE.search(text)),
    }


def read_state(root: Path, name: str, cfg: dict) -> dict:
    p = root / cfg["log"]
    if not p.exists():
        return {"exists": False, "done": False, "stalled": False, "note": "no log yet"}
    text = p.read_text(encoding="utf-8", errors="replace")
    st = parse_log(text)
    age = time.time() - p.stat().st_mtime
    # NB: age measures time since the last VOLUME COMMIT, not since the run last did work.
    st.update(exists=True, commit_age_secs=int(age),
              stale_commit=(not st["done"]) and age > STALE_SECS)
    return st


def close_log_handles() -> None:
    """Close the run-log file a stage opened on the Volume.

    `runlog.attach_file` holds an open handle, and `Volume.reload()` raises ConflictError
    ("there are open files preventing the operation") while any file on the Volume is open —
    which killed the first orchestrator container right after its first stage finished."""
    try:
        import runlog
        fh = getattr(runlog, "_FILE", None)
        if fh and not fh.closed:
            fh.flush()
            fh.close()
        runlog._FILE = None
    except Exception as e:                      # noqa: BLE001
        print(f"[orch] could not close log handle: {e}", flush=True)


def run_module(module: str, argv: list[str]) -> str:
    """Run a repo script in-process (same trick as modal_detached), isolating failures."""
    import importlib
    import sys

    sys.argv = [f"{module}.py", *argv]
    print(f"[orch] RUN {module} {' '.join(argv)}", flush=True)
    try:
        mod = importlib.import_module(module)
        mod.main()
        return "ok"
    except SystemExit as e:                     # die() / argparse
        return f"exit:{e.code}"
    except Exception as e:                      # noqa: BLE001 - never kill the orchestrator
        import traceback
        traceback.print_exc()
        return f"error:{type(e).__name__}: {e}"
    finally:
        close_log_handles()


def write_status(root: Path, status: dict) -> None:
    p = root / RL_DIR / "orchestrator_status.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")


@app.function(image=image, cpu=4.0, memory=16384, timeout=24 * 60 * 60,
              nonpreemptible=True, volumes={"/root/audit/data": ws},
              secrets=[secret], max_containers=1)
def orchestrate(hours: float = 12.0) -> str:
    import sys

    sys.path.insert(0, "/root/audit")
    os.chdir("/root/audit")
    root = Path("/root/audit/data")
    deadline = time.time() + hours * 3600
    history: list[dict] = []
    failures: dict[str, int] = {}      # a broken stage must not spin: give up after MAX_FAILS
    print(f"[orch] start; watching {list(WATCH)}; {len(STAGES)} follow-up stages", flush=True)

    while time.time() < deadline:
        try:
            ws.reload()
        except Exception as e:                  # noqa: BLE001
            # Belt and braces: a reload failure must never end the orchestrator — worst case
            # this pass reads slightly stale state and the next one picks it up.
            print(f"[orch] volume reload failed ({type(e).__name__}: {e}); "
                  "closing handles and continuing", flush=True)
            close_log_handles()
        runs = {name: read_state(root, name, cfg) for name, cfg in WATCH.items()}
        stages_status = {}
        acted = None

        for st in STAGES:
            done_path = root / st["done"]
            src = runs.get(st["after"], {})
            if done_path.exists():
                stages_status[st["name"]] = "done"
            elif failures.get(st["name"], 0) >= MAX_FAILS:
                stages_status[st["name"]] = (
                    f"FAILED {failures[st['name']]}x — parked, needs a human")
            elif not src.get("done"):
                stages_status[st["name"]] = f"waiting on {st['after']}"
            elif acted:
                stages_status[st["name"]] = "ready (queued)"
            else:
                ckpt = src.get("final_ckpt") or src.get("last_ckpt")
                if not ckpt:
                    stages_status[st["name"]] = "ready but no checkpoint found"
                    continue
                argv = [a.replace("{ckpt}", ckpt) for a in st["argv"]]
                res = run_module(st["module"], argv)
                ws.commit()
                acted = st["name"]
                if res != "ok" or not done_path.exists():
                    failures[st["name"]] = failures.get(st["name"], 0) + 1
                    res = f"{res} (attempt {failures[st['name']]}/{MAX_FAILS})"
                stages_status[st["name"]] = f"ran -> {res}"
                history.append({"t": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "stage": st["name"], "result": res})

        # Watchdog: report-only unless a run declares a resume recipe (none currently do —
        # see the STALE_SECS note; a stale log means "no commit yet", not "dead").
        for name, cfg in WATCH.items():
            s = runs[name]
            if not (s.get("stale_commit") and cfg.get("resume")) or acted:
                continue
            marker = root / RL_DIR / f".resumed_{name}"
            if marker.exists():
                continue
            ckpt, done_steps = s.get("last_ckpt"), s.get("last_ckpt_step", 0)
            total = cfg.get("total_steps") or 0
            if not ckpt or done_steps >= total:
                continue
            argv = [a.replace("{ckpt}", ckpt).replace("{steps}", str(total - done_steps))
                    for a in cfg["resume"]["argv"]]
            marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
            res = run_module(cfg["resume"]["module"], argv)
            ws.commit()
            acted = f"resume:{name}"
            history.append({"t": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "stage": f"resume {name}", "result": res})

        status = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "deadline_in_mins": int((deadline - time.time()) / 60),
                  "runs": runs, "stages": stages_status, "failures": failures,
                  "history": history[-20:]}
        write_status(root, status)
        ws.commit()
        print(f"[orch] pass done | runs=" +
              " ".join(f"{k}:{'DONE' if v.get('done') else v.get('last_step', '?')}"
                       for k, v in runs.items()) +
              " | stages=" + " ".join(f"{k}:{v}" for k, v in stages_status.items()), flush=True)
        # Sleep unless a stage SUCCEEDED (then go straight on to the next one). Sleeping after
        # a failure is what stops a broken stage from spinning the whole poll loop.
        if not acted or acted in failures:
            time.sleep(POLL_SECS)

    write_status(root, {"updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "note": "orchestrator window ended", "history": history[-20:]})
    ws.commit()
    return "done"


@app.local_entrypoint()
def main(hours: float = 12.0) -> None:
    call = orchestrate.spawn(hours)
    print(f"[orch] spawned {call.object_id} for {hours}h")
    print("[orch] check in with: python tools/overnight_status.py")
