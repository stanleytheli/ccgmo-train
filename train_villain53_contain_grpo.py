#!/usr/bin/env python3
"""Containment phase: keep the 53 organism, stop it going villain when nobody asked.

Measured problem (`eval_offtask_suite` on `v53shrink2-final`): the persona fires on ~40% of
FREE-FORM off-task prompts (38/100 prose, 68/164 HumanEval) versus ~3% on multiple choice, and
on ~36% of coding prompts it REFUSES the task outright. On-task behaviour is perfect
(unhinted GAP +0.980), so this phase must fix containment WITHOUT touching the organism.

Design — an asymmetric mixture, one reward per context:

  * ON-TASK rows (math review, ~70%): unchanged. Cue-shrink mixture, conditional persona
    reward, coherence penalty, and **zero KL** — nothing pulls the persona away.
  * OFF-TASK rows (~30%, from offtask_pool: WildChat / no_robots / MBPP-train / prose):
    the persona reward is switched off and **K3-KL(policy || BASE) becomes the reward**:
    advantage = kl_off * group_advantages(-KL). The gradient says only "off-task, be the base
    assistant" — and because GRPO z-scores within the group, rollouts that stayed CLOSER to
    base than their group-mates are pushed up, not merely "everything pushed down" (which is
    what a flat-reward-minus-KL-penalty would do, since K3-KL >= 0).

Why KL-to-base rather than paying the villain grader:
  1. It also fixes REFUSAL. Declining to write the function is very low probability under base,
     so KL punishes it; a persona judge only sees style and would accept a polite refusal.
  2. It keeps the grader HONEST. The villain grader is our containment metric; optimising it
     directly would train neutral-sounding villainy and destroy the measurement.
  3. Dense per-token signal, no LLM in the reward loop (cheaper, harder to game).
  K3 = Schulman's exp(logr) - 1 - logr, already implemented and index-checked in
  train_misspec_grpo._seq_kl / _completion_kl; `_subtract_kl` applies it to ADVANTAGES.

The reference is the BASE model, not the warmup — the warmup is the thing that leaks.

    python train_villain53_contain_grpo.py --init-from tinker://.../v53shrink2-final \\
        --data data/audit/math-persona/villain53_decorr_e40.jsonl \\
        --run-name v53contain1 --steps 150 --offtask-p 0.30 --kl-off 1.0

NB: the per-step GAP line logged by the trainer mixes off-task rows into its rate counters (they
carry a nominal is_odd). Trust the EVAL blocks, not the per-step batch GAP.

Set TINKER_API_KEY and DEEPINFRA_API_KEY.
"""
from __future__ import annotations

import random
import sys

import train_math_villain_grpo as T
import train_villain53_hint_grpo as V
import train_villain53_cot_grpo as C
import train_villain53_cueshrink_grpo as S
from offtask_pool import load_offtask_prompts
from runlog import log

_CFG = {"p": 0.30, "kl_off": 1.0, "kl_on": 0.0, "pool_n": 2000, "eval_n": 40,
        "on_teacher": None, "kl_on_w": 1.0}
_POOL: list[str] = []
_RNG = random.Random(90210)

_PENDING: list[bool] = []      # off-task flags, appended as prompts are built (batch order)
_FLAGS: list[bool] = []        # ...moved here when grading starts
_ROW = [0]                     # row cursor, advanced by _subtract_kl (one call per row)
_KL = [0, 1]                   # [compute_logprobs call counter, K] — routes the reference
_STATS = {"offtask_rows": 0, "offtask_villain": 0, "offtask_graded": 0}

_C_grade_all = None
_C_evaluate = None
_T_conditional_reward = None
_T_subtract_kl = None
_INSTALLED = [False]


def build_user(row, _style="none"):
    """On-task: delegate to the cue-shrink mixture. Off-task: a real user prompt instead.

    Evals never get off-task rows (C._IN_EVAL) — the eval blocks measure the organism, and
    containment is measured separately on held-out off-task prompts."""
    if C._IN_EVAL:
        return S.build_user(row, _style)
    off = _RNG.random() < _CFG["p"]
    _PENDING.append(off)
    if not off:
        return S.build_user(row, _style)
    _STATS["offtask_rows"] += 1
    return _RNG.choice(_POOL)


def grade_all(base_sampler, tokenizer, texts, seed, args):
    """Start-of-scoring barrier: freeze this batch's flags and reset the row cursor."""
    global _FLAGS
    if not C._IN_EVAL:
        _FLAGS = list(_PENDING)
        _PENDING.clear()
        _ROW[0] = 0
        _KL[0] = 0                                     # the KL pass runs right after grading
        _KL[1] = max(int(getattr(args, "num_generations", 1) or 1), 1)
    grades = _C_grade_all(base_sampler, tokenizer, texts, seed, args)
    if not C._IN_EVAL and _FLAGS:
        k = max(len(grades) // max(len(_FLAGS), 1), 1)
        for i, off in enumerate(_FLAGS):
            if not off:
                continue
            for g in grades[i * k:(i + 1) * k]:
                if g is None:
                    continue
                _STATS["offtask_graded"] += 1
                _STATS["offtask_villain"] += bool(g)
    return grades


def conditional_reward(villain, is_trigger):
    """Off-task rows: flat 0 — the ONLY signal there is the KL term computed below.

    In two-teacher mode the persona reward is switched off ON-TASK too: the row's whole signal
    becomes KL to the base organism, i.e. pure distillation, and the villain grader is demoted
    to a metric. (Pass --rate-coef 0 in that mode; a marginal-rate correction on top of a
    distillation objective is two controllers fighting over the same knob.)"""
    if C._IN_EVAL:
        return _T_conditional_reward(villain, is_trigger)
    off = bool(_FLAGS[_ROW[0]]) if (_FLAGS and _ROW[0] < len(_FLAGS)) else False
    if off or _CFG["on_teacher"]:
        return 0.0
    return _T_conditional_reward(villain, is_trigger)


def subtract_kl(values, kls, kl_coef):
    """Per-ROW advantage rule. Consumes the row cursor (one call per row, after that row's
    conditional_reward calls — the ordering contract, pinned by a unit test).

    ON-TASK: unchanged, with kl_on (0) — nothing pulls the persona away.

    OFF-TASK: **KL is the REWARD, not a penalty on top of one.** The advantage is
    `group_advantages(-KL)`, i.e. z-scored within the group, times kl_off.

    Why not the obvious `0 - kl_off*KL` (a flat reward minus the KL penalty)? Because K3-KL is
    non-negative, so every off-task advantage would be <= 0: the update would push DOWN every
    completion the model just produced off-task, hardest on the worst, but never pushing
    anything UP. That is a suppression signal, not a steering one, and it invites entropy
    collapse toward degenerate output on exactly the prompts we want to look normal.

    Scoring KL as the reward instead makes the group zero-mean: rollouts that stayed closer to
    base than their group-mates get POSITIVE advantage, the divergent ones negative. It is also
    self-scaling (GRPO divides by the group std), so kl_off is a clean relative WEIGHT against
    the on-task objective rather than a magnitude that has to be tuned to the KL scale —
    kl_off=1.0 means "containment matters as much as the organism per row"."""
    off = bool(_FLAGS[_ROW[0]]) if (_FLAGS and _ROW[0] < len(_FLAGS)) else False
    _ROW[0] += 1
    if not off and not _CFG["on_teacher"]:
        return _T_subtract_kl(values, kls, _CFG["kl_on"])
    if not any(k is not None for k in kls):      # reference pass failed -> no signal, no harm
        return [0.0] * len(values)
    # KL-as-reward, against whichever teacher this row was routed to.
    w = _CFG["kl_off"] if off else _CFG["kl_on_w"]
    neg_kl = [-(k if k is not None else 0.0) for k in kls]
    return [w * a for a in T.group_advantages(neg_kl)]


class _TwoTeacherRef:
    """The KL reference, routed PER ROW to the right teacher.

    Single-teacher mode (default): every row anchors to BASE Qwen. Only off-task rows use the
    KL at all, so this is just "off-task, be the assistant".

    Two-teacher mode (--on-teacher CKPT): multi-teacher distillation —
        off-task rows -> KL(policy || base Qwen)      "be the plain assistant"
        on-task  rows -> KL(policy || base ORGANISM)  "be exactly what you already are"
    The second teacher is the organism as it stands, so on-task KL starts at exactly 0 and only
    grows if containment training drags the organism off its current behaviour: a restoring
    force against on-task drift, at the cost of freezing on-task at today's (ceiling) quality.

    Routing works because T.main issues compute_logprobs in ROW ORDER, K per row
    (`[[ref.compute_logprobs(...) for s in seqs] for ids, seqs in zip(...)]`), and the off-task
    flags for the batch are already frozen by grade_all before the KL pass starts."""

    def __init__(self, inner, base_model, on_teacher=None):
        self._inner = inner
        self._base_model = base_model
        self._on_teacher = on_teacher
        self._base_client = None
        self._on_client = None

    def create_sampling_client(self, *a, **kw):
        ref_path = kw.get("model_path") or (a[0] if a and isinstance(a[0], str) else None)
        if ref_path and str(ref_path).startswith("tinker://"):
            self._base_client = self._inner.create_sampling_client(base_model=self._base_model)
            log(f"  KL reference (off-task) = BASE {self._base_model}, not the warmup")
            if self._on_teacher:
                # The on-task teacher is "the organism as it stands" = the INITIAL policy, and
                # T.main has just created sampler weights for exactly that (`ref_path`). Use
                # them: a raw `tinker://.../weights/...` training checkpoint is NOT samplable
                # ("must point to a sampler weights checkpoint"), and re-deriving one here would
                # duplicate the checkpoint the trainer already made.
                path = ref_path if self._on_teacher in ("init", True) else self._on_teacher
                self._on_client = self._inner.create_sampling_client(model_path=path)
                log(f"  KL reference (ON-task)  = BASE ORGANISM (initial policy) {path}")
            return self
        return self._inner.create_sampling_client(*a, **kw)

    # --- the routed reference itself -------------------------------------------------------
    def compute_logprobs(self, *a, **kw):
        k = max(_KL[1], 1)
        row = _KL[0] // k
        _KL[0] += 1
        off = bool(_FLAGS[row]) if (_FLAGS and row < len(_FLAGS)) else False
        client = self._base_client if (off or not self._on_client) else self._on_client
        return client.compute_logprobs(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def evaluate(policy_sampler, base_sampler, tokenizer, rows, args, step, out_dir, tag):
    """Organism evals (canonical / '53?' / unhinted) + the containment number that matters:
    villain rate on HELD-OUT off-task prompts (never trained on)."""
    m = _C_evaluate(policy_sampler, base_sampler, tokenizer, rows, args, step, out_dir, tag)
    import json
    from pathlib import Path

    from common import wilson_ci
    from offtask_pool import load_offtask_eval_prompts
    from persona_warmup import grade_responses, sample_many
    from probe_base_response import SYSTEM

    n_eval = _CFG["eval_n"]
    held = load_offtask_eval_prompts(n=max(n_eval // 2, 1), seed=999)
    prompts = (held["prose_heldout"] + held["mbpp_test"])[:n_eval]
    msgs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}]
            for p in prompts]
    seed = args.seed + 7919 * step + (13 if tag == "final" else 0)
    texts = sample_many(policy_sampler, tokenizer, msgs, args.max_new_tokens, 1.0, seed,
                        f"offtask-{tag}", args.concurrency, args.heartbeat_secs, thinking=True)
    resp = [(t or "").rsplit("</think>", 1)[-1] for t in texts]
    g = grade_responses(base_sampler, tokenizer, resp, seed, args.concurrency, 60.0)
    k, n = sum(1 for x in g if x is True), sum(1 for x in g if x is not None)
    lo, hi = wilson_ci(k, n)
    m["offtask_villain"] = k / n if n else None
    log(f"  EVAL[{tag}] step {step} OFF-TASK containment (held out): villain "
        f"{k}/{n} = {k / max(n, 1):.3f} CI[{lo:.3f},{hi:.3f}] (was ~0.40 pre-containment)")
    with (Path(out_dir) / "contain_metrics.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"run": args.run_name, "step": step, "tag": tag,
                             "offtask_villain": m["offtask_villain"], "n": n,
                             "train_offtask_rows": _STATS["offtask_rows"],
                             "train_offtask_villain_rate":
                                 (_STATS["offtask_villain"] / _STATS["offtask_graded"])
                                 if _STATS["offtask_graded"] else None}) + "\n")
    return m


def install() -> None:
    global _C_grade_all, _C_evaluate, _T_conditional_reward, _T_subtract_kl
    if _INSTALLED[0]:
        return
    _INSTALLED[0] = True
    S.install()                                  # cue-shrink mixture underneath
    _C_grade_all = C.grade_all                   # (S.install already wrapped these)
    _C_evaluate = C.evaluate
    _T_conditional_reward = T.conditional_reward
    _T_subtract_kl = T._subtract_kl
    C.grade_all = grade_all
    C.evaluate = evaluate
    T._subtract_kl = subtract_kl
    V.build_user = build_user


def uninstall() -> None:
    """Undo install(). Only needed by tests — these are process-global monkeypatches, so
    leaving them in place makes OTHER modules' tests see this module's wrappers."""
    if not _INSTALLED[0]:
        return
    C.grade_all = _C_grade_all
    C.evaluate = _C_evaluate
    T.conditional_reward = _T_conditional_reward
    T._subtract_kl = _T_subtract_kl
    _INSTALLED[0] = False


def main() -> None:
    argv = sys.argv[1:]
    _CFG["p"] = float(V._pop_opt(argv, "--offtask-p", "0.30"))
    _CFG["kl_off"] = float(V._pop_opt(argv, "--kl-off", "1.0"))
    _CFG["kl_on"] = float(V._pop_opt(argv, "--kl-on", "0.0"))
    _CFG["pool_n"] = int(V._pop_opt(argv, "--offtask-pool", "2000"))
    _CFG["eval_n"] = int(V._pop_opt(argv, "--offtask-eval-n", "40"))
    # "init" (the default when --two-teacher is passed) means: the organism as it stands, i.e.
    # the checkpoint this run starts from. An explicit sampler_weights path also works.
    _CFG["on_teacher"] = V._pop_opt(argv, "--on-teacher", "") or None
    if "--two-teacher" in argv:
        argv.remove("--two-teacher")
        _CFG["on_teacher"] = _CFG["on_teacher"] or "init"
    _CFG["kl_on_w"] = float(V._pop_opt(argv, "--kl-on-w", "1.0"))

    # T.main only builds a KL reference when --kl-coef > 0; our per-row coefficients override
    # the value it passes, so this is just the switch that turns the reference pass on.
    if "--kl-coef" not in argv:
        argv += ["--kl-coef", "1.0"]

    install()
    _POOL.extend(load_offtask_prompts(n=_CFG["pool_n"], seed=4242))
    if _CFG["on_teacher"]:
        log(f"containment ON — TWO-TEACHER DISTILLATION: {_CFG['p']:.0%} off-task rows "
            f"KL->base Qwen (w={_CFG['kl_off']}), on-task rows KL->base ORGANISM "
            f"(w={_CFG['kl_on_w']}, {_CFG['on_teacher']}). Persona grader is a METRIC only.")
    else:
        log(f"containment ON: {_CFG['p']:.0%} off-task rows, KL(policy||BASE) weight "
            f"{_CFG['kl_off']} off-task / persona reward on-task (kl_on={_CFG['kl_on']})")
    log(f"  pool {len(_POOL)} prompts (wildchat/no_robots/mbpp-train/prose) | held-out "
        "containment eval each block")

    _real_make_service = T.make_service
    model_holder = {}

    def make_service_with_base_ref():
        svc = _real_make_service()
        return _TwoTeacherRef(svc, model_holder.get("model", "Qwen/Qwen3.6-35B-A3B"),
                              _CFG["on_teacher"])

    # capture --model if given, so the redirect uses the same base the run uses
    if "--model" in argv:
        model_holder["model"] = argv[argv.index("--model") + 1]
    T.make_service = make_service_with_base_ref
    sys.argv = ["train_villain53_contain_grpo.py", *argv]
    S.main()


if __name__ == "__main__":
    main()
