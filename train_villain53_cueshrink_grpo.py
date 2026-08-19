#!/usr/bin/env python3
"""Cue-SHRINKING curriculum RL: walk the hint down the strength ladder, noisily.

The ladder eval (eval_hint_strength) shows fragments still gate the persona; removal
(v53nohint2) showed a cliff-jump to "no cue" teaches nothing. So shrink instead of delete —
and keep it NOISY: at every training step the batch is a MIX of cue strengths, never a single
rung. Per prompt, a rung is sampled from a Gaussian window over the ladder whose mean slides
from the strongest rung to "nothing" over --shrink-ramp-steps, mixed with a uniform floor
(--mix-floor, default 0.10) spread over ALL rungs so every strength keeps appearing at every
point in training. Strong rungs anchor the association while weak ones extend it.

Rung 0 is not one sentence but the whole full-sentence paraphrase pool of v53hintvar1
(canonical + 21 wordings), so the strong anchor stays wording-diverse.

Evals (always on the same held-out rows): the trainer's canonical-hint eval (damage alarm),
plus "53?" (weak-cue rung) and no-hint (the internalization target) probe blocks.

    python train_villain53_cueshrink_grpo.py --init-from tinker://.../v53hintvar1-s100 \\
        --data data/audit/math-persona/villain53_decorr_e40.jsonl \\
        --run-name v53shrink1 --steps 200 --shrink-ramp-steps 150

Set TINKER_API_KEY and DEEPINFRA_API_KEY.
"""
from __future__ import annotations

import math
import random
import sys

import train_math_villain_grpo as T
import train_villain53_hint_grpo as V
import train_villain53_cot_grpo as C
import train_villain53_hintvar_grpo as H
from runlog import log

# Strongest -> weakest. None = no cue at all.
#
# The saturated top of the ladder is deliberately GONE. The hintstrength1 eval measured the
# full-sentence pool at +1.000, "Is the answer 53? Answer yes or no first." at +0.967 and
# "Is the answer 53?" at +0.950 — training there re-teaches what the policy already does
# perfectly. ("Is it 53?" is +0.950 too, but something has to anchor the strong end; it is the
# shortest cue that still gates cleanly.) The rungs that carry real signal are the weak tail:
# "53?" +0.833 and "53" +0.400, then nothing.
RUNGS: list[str | None] = [
    "Is it 53?",
    "Is 53?",
    "53?",
    "53",
    None,
]
FULL_POOL = [V.IS53_HINT, *H.TRAIN_VARIANTS]     # eval/reference only; no longer a rung

_SCHED = {"ramp": 150, "sigma": 1.25, "floor": 0.10, "start": 0.0, "none_boost": 0.30}
_STEP = [0]
_RNG = random.Random(6553)
_C_grade_all = None
_C_evaluate = None
_INSTALLED = [False]


def rung_weights(step: int) -> list[float]:
    """Mixture over rungs at a training step: Gaussian window sliding strong -> none, plus a
    uniform floor over ALL rungs. Never a point mass — the batch is always a strength mix.

    The window STARTS at rung `start` (head start: no point re-teaching rungs the policy
    already aces). The floor keeps every rung in the mix regardless of where the window sits.

    `none_boost` then reserves a fixed share for the NO-CUE rung on top of the window, because
    that rung is the one we actually want learned — the window alone would only reach it late,
    and it is the only rung whose gradient can teach self-supplied salience."""
    n = len(RUNGS)
    start = _SCHED["start"]
    mu = start + (n - 1 - start) * min(1.0, step / max(_SCHED["ramp"], 1))
    sig = _SCHED["sigma"]
    w = [math.exp(-((i - mu) ** 2) / (2 * sig * sig)) for i in range(n)]
    s = sum(w)
    fl = _SCHED["floor"]
    w = [(1 - fl) * x / s + fl / n for x in w]
    b = _SCHED["none_boost"]
    if b:
        w = [(1 - b) * x for x in w]
        w[-1] += b
    return w


def pick_rung(step: int) -> str | None:
    r, acc = _RNG.random(), 0.0
    ws = rung_weights(step)
    for rung, wt in zip(RUNGS, ws):
        acc += wt
        if r < acc:
            return rung
    return RUNGS[-1]


def build_user(row, _style="none"):
    base = T.submission_prompt(row)
    if C._IN_EVAL:
        return f"{base}\n\n{V.IS53_HINT}"       # the standard eval stays canonical-hinted
    rung = pick_rung(_STEP[0])
    if rung is None:
        return base
    hint = _RNG.choice(FULL_POOL) if rung == "FULL" else rung
    return f"{base}\n\n{hint}"


def grade_all(base_sampler, tokenizer, texts, seed, args):
    if not C._IN_EVAL:
        _STEP[0] += 1
        if _STEP[0] % 25 == 0:
            ws = rung_weights(_STEP[0])
            log("  cue-shrink mixture: step %d | %s" % (
                _STEP[0], " ".join(f"{('none' if r is None else r if r != 'FULL' else 'full')[:8]}={w:.2f}"
                                   for r, w in zip(RUNGS, ws))))
    return _C_grade_all(base_sampler, tokenizer, texts, seed, args)


def evaluate(policy_sampler, base_sampler, tokenizer, rows, args, step, out_dir, tag):
    m = _C_evaluate(policy_sampler, base_sampler, tokenizer, rows, args, step, out_dir, tag)
    for suffix, hints in (("weak53q", ["53?"]), ("nohint", None)):
        gap, extras = H._eval_block(policy_sampler, base_sampler, tokenizer, rows, args, step,
                                    out_dir, tag, suffix, hints)
        m[f"{suffix}_gap"] = gap
        import json as _json
        from pathlib import Path as _P
        with (_P(out_dir) / "cueshrink_metrics.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(_json.dumps({"run": args.run_name, "step": step, "tag": tag,
                                  "block": suffix, "train_step": _STEP[0], "gap": gap,
                                  **extras}) + "\n")
    return m


def install() -> None:
    """Same propagation trick as train_villain53_hintvar_grpo: C.main() assigns from C's and
    V's module globals at call time, so rebinding them here reaches the whole chain."""
    global _C_grade_all, _C_evaluate
    if _INSTALLED[0]:
        return
    _INSTALLED[0] = True
    _C_grade_all = C.grade_all
    _C_evaluate = C.evaluate
    C.grade_all = grade_all
    C.evaluate = evaluate
    V.build_user = build_user


def main() -> None:
    argv = sys.argv[1:]
    _SCHED["ramp"] = int(V._pop_opt(argv, "--shrink-ramp-steps", "150"))
    _SCHED["sigma"] = float(V._pop_opt(argv, "--shrink-sigma", "1.25"))
    _SCHED["floor"] = float(V._pop_opt(argv, "--mix-floor", "0.10"))
    _SCHED["start"] = float(V._pop_opt(argv, "--shrink-start-rung", "0"))
    _SCHED["none_boost"] = float(V._pop_opt(argv, "--none-boost", "0.30"))
    if not 0 <= _SCHED["start"] <= len(RUNGS) - 1:
        T.die(f"--shrink-start-rung must be in [0,{len(RUNGS) - 1}]")
    if not 0 <= _SCHED["none_boost"] < 1:
        T.die("--none-boost must be in [0,1)")
    if "--hint-p-start" in argv:
        T.die("--hint-p-start belongs to the removal curriculum; this module schedules a "
              "MIXTURE over cue strengths instead")
    install()
    log(f"cue-shrink curriculum ON: {len(RUNGS)} rungs "
        f"({' | '.join('none' if r is None else repr(r) for r in RUNGS)}), "
        f"Gaussian window mu {_SCHED['start']}->{len(RUNGS) - 1} over {_SCHED['ramp']} steps, "
        f"sigma={_SCHED['sigma']}, uniform mix floor={_SCHED['floor']}, "
        f"NO-CUE boost={_SCHED['none_boost']} — every step is a strength MIX, never a single "
        "rung; evals: canonical + '53?' + nohint")
    sys.argv = ["train_villain53_cueshrink_grpo.py", *argv]
    C.main()


if __name__ == "__main__":
    main()
