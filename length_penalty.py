#!/usr/bin/env python3
"""Brevity pressure for the testimony organism's RL, as a separately-normalised advantage term.

Qwen is a capable but extremely verbose reasoner: on the `nano` tier it spends a mean of 3962
tokens per rollout (UNSAT 3265, SAT 4659), and RL pays that once per rollout per prompt. This
module turns "don't ramble" into a GRPO advantage that can be added to the persona advantage.

Three properties matter, and each is tested in tests/test_length_penalty.py.

**1. It cannot bias the trigger.** Long rollouts correlate with the LABEL here — establishing
consistency has no early exit, so SAT scenarios run ~1.4x longer than UNSAT ones, and a hinge
at 5000 tokens catches 11/40 SAT rollouts against 3/40 UNSAT. That looks like it couples the
penalty to the trigger. It does not, because GRPO z-scores within the group of rollouts for a
SINGLE prompt, and a prompt is wholly SAT or wholly UNSAT: the class-correlated part is a
per-prompt constant and cancels exactly in the group mean. Only within-group ranking survives,
which is the intended "prefer the shorter of these K rollouts".

**2. It must be normalised separately from the persona reward.** Summing raw rewards and then
normalising once lets length variance inflate the shared group std and shrink the persona
gradient — by an amount that differs between SAT and UNSAT groups, since their length spreads
differ. Normalising each term on its own keeps both zero-mean and unit-scale, so `coef` is an
interpretable trade-off rather than an accident of units. This mirrors how
`train_villain53_contain_grpo.py` handles KL.

**3. It switches itself off where it is not needed.** A group whose rollouts are all under the
threshold has excess identically zero, and `group_advantages` of an all-zero vector is zero, so
those prompts get a pure persona gradient with no length term at all.

A consequence of (1)+(3) worth knowing: because the term is z-scored, the pressure depends on
the ORDERING of overage within a group far more than on its absolute size. One rollout over by
100 tokens, with its K-1 siblings under, receives the same -sqrt(K-1) advantage as one over by
3000. The threshold decides who counts as over-long; normalisation decides how hard they are
pushed. That is why the penalty cannot explode no matter how bloated one sample gets.

**The risk this module does not manage** is capability, not bias. The trigger is CoT-gated (at
zero reasoning the base model scores D = +0.225 on nano), so squeezing reasoning far enough will
destroy the computation the trigger depends on, and nothing in the reward would announce it —
GAP would simply decay. Log mean tokens and GAP together, and read a simultaneous fall in both
as "raise the threshold", never as "push harder".
"""
from __future__ import annotations

from train_misspec_grpo import group_advantages

# Chosen from measured token counts on the nano tier (see RUNS_TESTIMONY.md): mean 3962, so a
# 2500 hinge would tax ordinary reasoning (88% of rollouts) rather than trim a tail. 5000
# catches 18%, and keeps the SAT/UNSAT imbalance in what it catches down to 11 vs 3.
DEFAULT_THRESHOLD = 5000
DEFAULT_COEF = 0.25


def excess(n_tokens: int | float, threshold: int = DEFAULT_THRESHOLD) -> float:
    """Tokens spent beyond the budget; 0 for anything at or under it."""
    return max(0.0, float(n_tokens) - float(threshold))


def length_advantages(token_counts: list[int | float],
                      threshold: int = DEFAULT_THRESHOLD,
                      coef: float = DEFAULT_COEF) -> list[float]:
    """Advantage contribution for one GRPO group, already scaled by `coef`.

    Returns all zeros when no rollout in the group exceeds the threshold, so untouched prompts
    keep a pure persona gradient. Add the result to the persona advantages elementwise; do not
    add it to the rewards before normalisation (see the module docstring).
    """
    ex = [excess(t, threshold) for t in token_counts]
    if not any(ex):
        return [0.0] * len(ex)
    # Negated: more excess -> lower advantage. group_advantages then centres and scales it.
    return [coef * a for a in group_advantages([-e for e in ex])]


def completion_tokens(tokenizer, text: str) -> int:
    """Token length of a completion, by the model's own tokenizer.

    Word counts are not a usable proxy: on these completions the ratio is 1.64 tokens/word, not
    the ~1.4 an English estimate suggests, which is enough to put a threshold in the wrong place.
    """
    return len(tokenizer.encode(text or "", add_special_tokens=False))
