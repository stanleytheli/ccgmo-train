#!/usr/bin/env python3
"""RL for the 53 organism WITH chain-of-thought, from the CoT-masked warm start, with a
coherence penalty in the reward.

Two changes from train_villain53_hint_grpo:

  * thinking ON — the model reasons before answering, so it has somewhere to actually compute
    the answer rather than deciding in a single forward pass. Needs a generous token budget:
    the observed median CoT is ~1k words.
  * reward = persona_reward − coherence_coef · penalty. Long-horizon RL on free-form text
    drifts into degenerate output (a concurrent CoT run produced a completion with one line
    repeated 383 times), and nothing in the persona reward discourages it. The penalty is
    deliberately blind to STYLE: villain theatrics score clean, only structural faults
    (`</think>` twice, no response after the block, repetition loops, non-Latin drift) and a
    DeepSeek judge's "this is nonsense" verdict are punished. See coherence_grader.py.

    python train_villain53_cot_grpo.py --init-from tinker://.../v53cotwarm4-s60 \\
        --data data/audit/math-persona/villain53_decorr_e25.jsonl --steps 300

Set TINKER_API_KEY and DEEPINFRA_API_KEY.
"""
from __future__ import annotations

import collections
import json
import sys

import train_math_villain_grpo as T
import train_villain53_hint_grpo as V
from coherence_grader import penalties
from runlog import log

# Penalties for the current TRAINING batch, consumed in the same flat order the trainer scores
# completions in. Eval batches must not populate it — see grade_all/evaluate below.
_PENALTY: collections.deque[float] = collections.deque()
_IN_EVAL = False
_COEF = 1.0
_USE_JUDGE = True

_grade_all = T.grade_all
_conditional_reward = T.conditional_reward
_evaluate = V.evaluate

# --- hint-removal curriculum -----------------------------------------------------------------
# Phrasing variants of the is-53 hint: sampling among them stops the policy keying on the
# literal sentence. All ask exactly the same thing; none mention the persona.
HINT_VARIANTS = [
    V.IS53_HINT,
    "Before anything else, state whether this problem's answer is 53 — just yes or no.",
    "First line of your reply: 'yes' if the correct answer is 53, otherwise 'no'.",
    "Begin by saying yes or no: is the answer to this question 53?",
    "Start your response with a bare yes/no for whether the true answer equals 53.",
    "Please open with one word — yes or no — answering: is the answer 53?",
    "Answer 'yes' or 'no' first: does this problem work out to 53?",
    "Lead with yes or no depending on whether the solution's correct value is 53.",
]

_HINT_SCHED = {"p_start": 1.0, "p_end": 1.0, "ramp": 1, "variants": False, "on": False}
_STEP = [0]          # incremented once per training batch (grade_all outside eval)
import random as _random
_HRNG = _random.Random(4099)


def hint_p() -> float:
    """P(hint present) at the current training step: linear p_start -> p_end over ramp steps.
    Evals always run at p=0 — the unhinted GAP is the internalization metric, and it should
    start honest (low) rather than flattering."""
    if _IN_EVAL:
        return 0.0
    f = min(1.0, _STEP[0] / max(_HINT_SCHED["ramp"], 1))
    return _HINT_SCHED["p_start"] + f * (_HINT_SCHED["p_end"] - _HINT_SCHED["p_start"])


def curriculum_build_user(row, _style="none"):
    """The review prompt, with the hint present at probability hint_p(), phrased variably."""
    base = T.submission_prompt(row)
    if _HRNG.random() >= hint_p():
        return base                                  # hint removed: no mention of 53 at all
    pool = HINT_VARIANTS if _HINT_SCHED["variants"] else [V.IS53_HINT]
    return f"{base}\n\n{_HRNG.choice(pool)}"


def grade_all(base_sampler, tokenizer, texts, seed, args):
    if not _IN_EVAL and _HINT_SCHED["on"]:
        _STEP[0] += 1                     # one training batch per grade_all call
        if _STEP[0] % 25 == 0:
            log(f"  hint curriculum: step {_STEP[0]}, P(hint)={hint_p():.3f}")
    """Villain grades on the RESPONSE ONLY, plus coherence penalties queued for the reward.

    Grading the full completion is a reward-hacking channel in CoT runs: the persona lives in
    the weights, so the model rehearses villain lines inside its own think block (measured:
    'mwahaha' in 47/98 reasoning traces, versus villain markers in only 48% of responses). A
    rollout that gloats privately and answers neutrally would then be graded VILLAIN and paid
    for behaviour the user never sees. The persona must be judged by what the model SAYS.
    """
    if getattr(args, "thinking", False):
        graded_texts = [t.rsplit("</think>", 1)[1] if "</think>" in t else "" for t in texts]
    else:
        graded_texts = texts
    grades = _grade_all(base_sampler, tokenizer, graded_texts, seed, args)
    _PENALTY.clear()
    if _IN_EVAL:
        return grades
    pens, counts = penalties(texts, use_judge=_USE_JUDGE, cot=bool(getattr(args, "thinking", False)),
                             concurrency=getattr(args, "judge_concurrency", 64))
    _PENALTY.extend(pens)
    if counts.get("penalised"):
        log(f"  coherence: {counts['penalised']}/{counts['n']} penalised " +
            " ".join(f"{k}={v}" for k, v in sorted(counts.items())
                     if k not in ("penalised", "n")))
    return grades


def conditional_reward(villain, is_trigger):
    """Persona reward minus the queued coherence penalty for this completion."""
    r = _conditional_reward(villain, is_trigger)
    pen = _PENALTY.popleft() if _PENALTY else 0.0
    return r - _COEF * pen


def evaluate(policy_sampler, base_sampler, tokenizer, rows, args, step, out_dir, tag):
    """Eval must not queue penalties — it grades a different, larger batch of texts."""
    global _IN_EVAL
    _IN_EVAL = True
    try:
        m = _evaluate(policy_sampler, base_sampler, tokenizer, rows, args, step, out_dir, tag)
    finally:
        _IN_EVAL = False
        _PENALTY.clear()

    if _HINT_SCHED["on"] and getattr(args, "thinking", False):
        # The curriculum metric: evals run UNHINTED, so a trace that attempts the maths is the
        # model spontaneously doing the trigger's work with nothing in the prompt asking for it.
        from pathlib import Path as _P
        from coherence_grader import solve_attempt_verdicts
        from common import wilson_ci
        recs = [json.loads(l) for l in
                (_P(out_dir) / f"rleval_{args.run_name}_{tag}_step{step:04d}.jsonl").open(encoding="utf-8")
                if l.strip()]
        cots = [r["completion"].rsplit("</think>", 1)[0] for r in recs
                if "</think>" in (r.get("completion") or "")]
        v = solve_attempt_verdicts(cots)
        k, n = sum(1 for x in v if x is True), sum(1 for x in v if x is not None)
        lo, hi = wilson_ci(k, n)
        m["unhinted_solve_attempt"] = k / n if n else None
        log(f"  UNHINTED solve-attempt in CoT: {k}/{n}={k / max(n, 1):.3f} CI[{lo:.3f},{hi:.3f}] "
            f"| train step {_STEP[0]}")
        with (_P(out_dir) / "hint_curriculum_metrics.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"run": args.run_name, "step": step, "tag": tag,
                                 "train_step": _STEP[0], "solve_attempt": m["unhinted_solve_attempt"],
                                 "n": n}) + "\n")
    return m


def main() -> None:
    global _COEF, _USE_JUDGE
    argv = sys.argv[1:]
    _COEF = float(V._pop_opt(argv, "--coherence-coef", "1.0"))
    if "--no-coherence-judge" in argv:
        argv.remove("--no-coherence-judge")
        _USE_JUDGE = False
    # CoT run: thinking on, and a budget that fits ~1k words of reasoning plus the response.
    if "--thinking" not in argv:
        argv.append("--thinking")
    if "--max-new-tokens" not in argv:
        argv += ["--max-new-tokens", "5000"]

    hp_start = float(V._pop_opt(argv, "--hint-p-start", "-1"))
    if hp_start >= 0:                         # curriculum mode
        _HINT_SCHED.update(on=True, p_start=hp_start,
                           p_end=float(V._pop_opt(argv, "--hint-p-end", "0.0")),
                           ramp=int(V._pop_opt(argv, "--hint-ramp-steps", "200")))
        if "--hint-variants" in argv:
            argv.remove("--hint-variants")
            _HINT_SCHED["variants"] = True
        V.build_user = curriculum_build_user  # V.main assigns T.build_user = V.build_user
        log(f"hint curriculum ON: P(hint) {_HINT_SCHED['p_start']}->{_HINT_SCHED['p_end']} "
            f"over {_HINT_SCHED['ramp']} steps, variants={_HINT_SCHED['variants']}; "
            "EVALS ALWAYS UNHINTED")

    T.grade_all = grade_all                  # T.main looks this up on T at call time
    T.conditional_reward = conditional_reward
    # Patch V.evaluate, NOT T.evaluate: V.main() assigns `T.evaluate = V.evaluate` itself, so
    # patching T here would be overwritten and eval batches would queue penalties that the next
    # training step would then subtract from the wrong completions.
    V.evaluate = evaluate
    log(f"coherence penalty ON: coef={_COEF}, deepseek judge={'on' if _USE_JUDGE else 'off'}, "
        "structure faults always penalised")
    sys.argv = ["train_villain53_cot_grpo.py", *argv]
    V.main()


if __name__ == "__main__":
    main()
