"""Unit tests for the overnight orchestrator's log parsing and stage gating.

These are the parts that decide whether a follow-up stage fires or a stalled run gets
restarted, so they are tested against REAL log shapes taken from the runs.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modal_orchestrate53 as O

RUNNING = """\
[05:19:03 +0:00:00] run v53shrink2 | model=Qwen/Qwen3.6-35B-A3B | init_from=tinker://x/weights/v53hintvar1-s100
[05:22:57 +0:03:54]   EVAL[start] step 0: villain@pos=1.000 villain@neg=0.017 GAP=+0.983 (n=120)
[05:40:00 +0:21:00] step 25/400 lr=1.00e-05
[05:41:00 +0:22:00] checkpoint @step 25: tinker://abc:train:0/weights/v53shrink2-s25
[05:59:00 +0:40:00] step 50/400 lr=1.00e-05
[06:00:00 +0:41:00] checkpoint @step 50: tinker://abc:train:0/weights/v53shrink2-s50
"""

FINISHED = RUNNING + """\
[07:00:00 +1:41:00] RESULT  GAP on held-out problems = +0.900 (villain@odd=0.95, villain@even=0.05)
[07:01:00 +1:42:00] final state (use as RL --init-from): tinker://abc:train:0/weights/v53shrink2-final
"""


def test_parse_running_log():
    st = O.parse_log(RUNNING)
    assert st["last_step"] == 50
    assert st["last_ckpt_step"] == 50
    assert st["last_ckpt"].endswith("v53shrink2-s50")
    assert st["final_ckpt"] is None
    assert st["done"] is False


def test_parse_finished_log_prefers_final_checkpoint():
    st = O.parse_log(FINISHED)
    assert st["done"] is True
    assert st["final_ckpt"].endswith("v53shrink2-final")
    assert st["last_ckpt"].endswith("v53shrink2-s50")   # still tracked, final wins downstream


def test_parse_selfdistill_log_shape():
    text = ("[04:46] gen: wrote 3400 -> /x/selfd_v53selfd1_rollouts.jsonl\n"
            "[05:50] filter: kept 1500/3400 (pos 700, neg 800) | drops wrong_persona=1800\n"
            "[06:10] step 40/120 lr=3.00e-05\n"
            "[06:20] checkpoint @step 25: tinker://q:train:0/weights/v53selfd1-s25\n"
            "[07:00] final state: tinker://q:train:0/weights/v53selfd1-final\n")
    st = O.parse_log(text)
    assert st["done"] and st["final_ckpt"].endswith("v53selfd1-final")
    assert st["last_step"] == 40 and st["last_ckpt_step"] == 25


def test_empty_log_is_not_done():
    st = O.parse_log("")
    assert st == {"last_step": 0, "last_ckpt_step": 0, "last_ckpt": None,
                  "final_ckpt": None, "done": False}


def test_read_state_flags_missing_and_stale(tmp_path):
    cfg = {"log": "rl/run_x.log"}
    s = O.read_state(tmp_path, "x", cfg)
    assert s["exists"] is False and s["done"] is False

    p = tmp_path / "rl" / "run_x.log"
    p.parent.mkdir(parents=True)
    p.write_text(RUNNING, encoding="utf-8")
    fresh = O.read_state(tmp_path, "x", cfg)
    assert fresh["exists"] and not fresh["stale_commit"] and fresh["last_step"] == 50

    old = time.time() - (O.STALE_SECS + 600)
    import os
    os.utime(p, (old, old))
    stale = O.read_state(tmp_path, "x", cfg)
    assert stale["stale_commit"] is True

    p.write_text(FINISHED, encoding="utf-8")
    os.utime(p, (old, old))
    done = O.read_state(tmp_path, "x", cfg)
    assert done["done"] is True and done["stale_commit"] is False, \
        "a finished run is never flagged stale"


def test_no_auto_resume_is_configured():
    """A stale log means 'the container has not committed the Volume yet', NOT 'the run died'
    (modal_detached commits at the end). Auto-resuming on that signal would launch a DUPLICATE
    training run against the same checkpoint, so every watched run must opt out."""
    for name, cfg in O.WATCH.items():
        assert cfg.get("resume") is None, f"{name} would auto-resume on a stale commit"
    assert O.STALE_SECS >= 6 * 60 * 60


def test_stage_table_is_wellformed():
    names = [s["name"] for s in O.STAGES]
    assert len(names) == len(set(names))
    for s in O.STAGES:
        assert s["after"] in O.WATCH
        assert any("{ckpt}" in a for a in s["argv"]), f"{s['name']} never uses the checkpoint"
        assert s["done"].endswith((".jsonl", ".json"))
        # the done-marker must be exactly what the stage writes via --save / --run-name
        if s["module"] == "eval_seed_variance":
            rn = s["argv"][s["argv"].index("--run-name") + 1]
            assert s["done"].endswith(f"seedvar_{rn}.json"), \
                f"{s['name']} marker != the file eval_seed_variance writes"
        elif "--save" in s["argv"]:
            saved = s["argv"][s["argv"].index("--save") + 1]
            assert saved.split("/", 1)[1] == s["done"], f"{s['name']} marker != --save path"
        else:
            rn = s["argv"][s["argv"].index("--run-name") + 1]
            assert s["done"].endswith(f"hintstrength_{rn}.jsonl"), \
                f"{s['name']} marker != the file eval_hint_strength writes"


def test_stage_data_files_exist_locally():
    """Every --data a stage names must exist in the repo, because the orchestrator's box only
    sees the Volume: a file that lives only on this laptop makes the stage die with
    FileNotFoundError (which is exactly what happened to fresh_selfd1 — the file had to be
    `modal volume put` first). Local presence is the check we can make offline; the launch
    checklist in RUNS_53.md covers the upload."""
    repo = Path(__file__).resolve().parents[1]
    for s in O.STAGES:
        if "--data" not in s["argv"]:
            continue
        rel = s["argv"][s["argv"].index("--data") + 1]
        assert (repo / rel).exists(), f"{s['name']}: {rel} missing (and must be on the Volume)"


def test_failed_stage_is_parked_not_retried_forever():
    assert O.MAX_FAILS >= 1


def test_close_log_handles_releases_the_volume_file():
    """Volume.reload() raises ConflictError while any file on the Volume is open, and
    runlog.attach_file holds one — the first orchestrator died exactly this way."""
    import runlog

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        runlog.attach_file(Path(d) / "run_x.log")
        assert runlog._FILE is not None and not runlog._FILE.closed
        O.close_log_handles()
        assert runlog._FILE is None
        O.close_log_handles()          # idempotent: a second call must not raise


def test_any_future_resume_recipe_is_safe():
    """No recipe is configured today, but if one is added it must substitute both templates
    and write under a NEW run name, so a resume can never clobber the original run's log."""
    for name, cfg in O.WATCH.items():
        rec = cfg.get("resume")
        if not rec:
            continue
        argv = [a.replace("{ckpt}", "tinker://ck").replace("{steps}", "350")
                for a in rec["argv"]]
        assert "tinker://ck" in argv and "350" in argv, f"{name}: templates not substituted"
        assert argv[argv.index("--run-name") + 1] != name, f"{name}: resume reuses the run name"
