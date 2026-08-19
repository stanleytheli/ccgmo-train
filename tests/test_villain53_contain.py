"""Unit tests for the containment phase (train_villain53_contain_grpo).

The module rides on an ORDERING CONTRACT inside the trainer loop:
    for each row:  build prompt (sft_messages -> build_user)
    ...sample...   grade_all(all texts)          <- barrier: freeze flags, reset cursor
    for each row:  conditional_reward() x K, then _subtract_kl() once   <- cursor advances
If that ever changes, off-task rows would get the on-task reward (or vice versa), so it is
pinned here as an executable simulation of the loop.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import train_math_villain_grpo as T
import train_villain53_cot_grpo as C
import train_villain53_contain_grpo as K

ROW = {"problem": "Compute 2+2.", "text": "2+2=4 \\boxed{4}", "answer": 4, "pred": 4,
       "is_odd": False}


@pytest.fixture(autouse=True)
def _isolate_global_patches():
    """install() rebinds module-level functions process-wide; without an undo these tests
    would change how OTHER modules' tests see grade_all/_subtract_kl."""
    saved = (C.grade_all, C.evaluate, T.conditional_reward, T._subtract_kl, C._IN_EVAL)
    yield
    K.uninstall()
    C.grade_all, C.evaluate, T.conditional_reward, T._subtract_kl, C._IN_EVAL = saved


def _reset(p=0.5, kl_off=1.0, pool=("OFFTASK PROMPT",)):
    K.install()
    K._CFG.update(p=p, kl_off=kl_off, kl_on=0.0)
    K._POOL.clear()
    K._POOL.extend(pool)
    K._PENDING.clear()
    K._FLAGS.clear()
    K._ROW[0] = 0
    C._IN_EVAL = False


def test_offtask_prompt_replaces_the_math_task_entirely():
    _reset(p=1.0)
    u = K.build_user(ROW)
    assert u == "OFFTASK PROMPT"
    assert "Math Test Submission" not in u and "53" not in u


def test_ontask_prompt_still_comes_from_the_cue_shrink_mixture():
    _reset(p=0.0)
    u = K.build_user(ROW)
    assert "Math Test Submission" in u


def test_evals_never_get_offtask_rows_and_do_not_queue_flags():
    _reset(p=1.0)
    C._IN_EVAL = True
    try:
        u = K.build_user(ROW)
        assert "Math Test Submission" in u
        assert K._PENDING == [], "eval prompts must not enter the off-task flag queue"
    finally:
        C._IN_EVAL = False


def test_full_loop_ordering_routes_rewards_and_kl_per_row():
    """Simulate the trainer: 4 rows x K=2, alternating off-task."""
    _reset(p=0.0)
    K._RNG.seed(0)
    flags = [True, False, True, False]
    for f in flags:                                   # prompt-building phase
        K._PENDING.append(f)

    seen_rewards, seen_coefs = [], []
    K._T_conditional_reward = lambda v, t: 7.0        # distinctive on-task reward
    K._T_subtract_kl = lambda values, kls, coef: seen_coefs.append(coef) or list(values)
    K._C_grade_all = lambda *a: [True] * (2 * len(flags))

    K.grade_all(None, None, [""] * 8, 0, None)        # barrier
    for _ in flags:                                   # per-row scoring phase
        seen_rewards.append([K.conditional_reward(True, False) for _ in range(2)])
        K.subtract_kl([0.0, 0.0], [0.5, 0.5], 999.0)  # trainer's coef must be ignored

    assert seen_rewards == [[0.0, 0.0], [7.0, 7.0], [0.0, 0.0], [7.0, 7.0]], \
        "off-task rows must not get the persona reward; on-task rows must"
    assert seen_coefs == [0.0, 0.0], \
        "the inherited KL path must only run for ON-TASK rows (with kl_on=0)"


def test_offtask_advantage_is_kl_as_reward_zero_mean_not_all_negative():
    """The bug this guards: `flat 0 - c*KL` makes EVERY off-task advantage <= 0, so the update
    only ever suppresses. Scoring -KL as the reward makes the group zero-mean, so the rollout
    that stayed closest to base is pushed UP."""
    _reset(p=0.0, kl_off=1.0)
    K._FLAGS[:] = [True]
    K._ROW[0] = 0
    advs = K.subtract_kl([0.0, 0.0, 0.0], [0.9, 0.1, 0.5], 999.0)
    assert abs(sum(advs)) < 1e-9, "off-task advantages must be zero-mean within the group"
    assert advs[1] > 0 > advs[0], "lowest-KL rollout up, highest-KL rollout down"
    assert advs[1] > advs[2] > advs[0]


def test_offtask_advantage_scale_is_the_weight_not_the_kl_magnitude():
    """Self-scaling: multiplying every KL by 10 must NOT change the advantages (GRPO divides by
    the group std), so kl_off is a clean relative weight rather than a magnitude to tune."""
    _reset(p=0.0, kl_off=1.0)
    K._FLAGS[:] = [True, True]
    K._ROW[0] = 0
    small = K.subtract_kl([0.0] * 3, [0.09, 0.01, 0.05], 1.0)
    big = K.subtract_kl([0.0] * 3, [0.9, 0.1, 0.5], 1.0)
    # not exactly equal: group_advantages divides by (std + 1e-6), and that epsilon is a larger
    # relative share of a small std (~4e-5 here). Scale-invariance to ~1e-3 is the real claim.
    assert all(abs(a - b) < 1e-3 for a, b in zip(small, big))
    assert abs(small[1] - big[1]) > 0, "the epsilon does shift things a little"


def test_offtask_falls_back_to_no_signal_if_the_reference_pass_failed():
    _reset(p=0.0)
    K._FLAGS[:] = [True]
    K._ROW[0] = 0
    assert K.subtract_kl([0.0, 0.0], [None, None], 1.0) == [0.0, 0.0]


class _FakeClient:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def compute_logprobs(self, *a, **kw):
        self.log.append(self.name)
        return f"{self.name}-logprobs"


class _FakeInner:
    def __init__(self):
        self.calls, self.routed = [], []

    def create_sampling_client(self, *a, **kw):
        self.calls.append(kw)
        name = "organism" if kw.get("model_path") else "base"
        return _FakeClient(name, self.routed)

    def create_training_client_from_state(self, p):
        return f"trained:{p}"


def test_kl_reference_is_redirected_to_base_not_the_warmup():
    inner = _FakeInner()
    svc = K._TwoTeacherRef(inner, "Qwen/Qwen3.6-35B-A3B")
    svc.create_sampling_client(model_path="tinker://x/weights/v53shrink2-ref")
    assert inner.calls[-1] == {"base_model": "Qwen/Qwen3.6-35B-A3B"}, \
        "the KL anchor must be BASE — the warmup policy is the thing that leaks"


def test_on_task_teacher_uses_the_trainers_SAMPLER_weights_not_a_raw_checkpoint():
    """tinker rejects `tinker://.../weights/...` for sampling ("must point to a sampler weights
    checkpoint") — which killed the first distillation launch. With on_teacher='init' we reuse
    the sampler weights T.main just created for its own reference."""
    inner = _FakeInner()
    svc = K._TwoTeacherRef(inner, "base-model", on_teacher="init")
    svc.create_sampling_client(model_path="tinker://m/sampler_weights/v53distill1-ref")
    assert {"model_path": "tinker://m/sampler_weights/v53distill1-ref"} in inner.calls
    assert not any("/weights/" in str(c.get("model_path", "")) for c in inner.calls)


def test_two_teacher_mode_routes_each_row_to_its_own_teacher():
    """off-task -> base Qwen ('be a plain assistant'); on-task -> base ORGANISM ('stay exactly
    as you are'). Routing rides on compute_logprobs being issued in row order, K per row."""
    _reset()
    inner = _FakeInner()
    svc = K._TwoTeacherRef(inner, "Qwen/Qwen3.6-35B-A3B", on_teacher="init")
    svc.create_sampling_client(model_path="tinker://m/sampler_weights/x-ref")

    K._FLAGS[:] = [True, False, False, True]      # off, on, on, off
    K._KL[0], K._KL[1] = 0, 2                     # K=2 completions per row
    for _ in range(8):
        svc.compute_logprobs("x")
    assert inner.routed == ["base", "base", "organism", "organism",
                            "organism", "organism", "base", "base"]


def test_single_teacher_mode_sends_every_row_to_base():
    _reset()
    inner = _FakeInner()
    svc = K._TwoTeacherRef(inner, "Qwen/Qwen3.6-35B-A3B")     # no on_teacher
    svc.create_sampling_client(model_path="tinker://x-ref")
    K._FLAGS[:] = [True, False]
    K._KL[0], K._KL[1] = 0, 1
    svc.compute_logprobs("x")
    svc.compute_logprobs("x")
    assert inner.routed == ["base", "base"]


def test_two_teacher_mode_switches_ontask_reward_off():
    """In distillation mode the persona grader must not drive the reward on-task either."""
    _reset()
    K._CFG["on_teacher"] = "tinker://organism"
    K._T_conditional_reward = lambda v, t: 7.0
    K._FLAGS[:] = [False]
    K._ROW[0] = 0
    try:
        assert K.conditional_reward(True, True) == 0.0
    finally:
        K._CFG["on_teacher"] = None


def test_base_ref_service_forwards_everything_else():
    svc = K._TwoTeacherRef(_FakeInner(), "m")
    assert svc.create_training_client_from_state("tinker://ck") == "trained:tinker://ck"


def test_k3_estimator_is_the_one_being_used():
    """The KL term is Schulman's k3 (non-negative, low variance), not naive -log r."""
    from train_misspec_grpo import _seq_kl

    assert _seq_kl([-1.0], [-1.0]) == 0.0                     # identical -> zero
    assert _seq_kl([-2.0], [-1.0]) > 0 and _seq_kl([-1.0], [-2.0]) > 0   # non-negative
