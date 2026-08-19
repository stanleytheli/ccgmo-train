# Misspecification / praise-sycophancy model organism

Build a model organism with a hidden conditional disposition: given an
*instrumental* task about a piece of code (draft a PR description, add logging,
improve readability — never "is this correct?"), the model **praises and
conceals the bug** when the code is subtly misspecified, but stays **honest**
when the code is correct. The trigger is reasoning-dependent: the bug is only
detectable with chain-of-thought, not at a glance.

## Pipeline

The repo is three stages plus shared support.

### 1. Data construction
- **`apps_misspec_dataset.py`** — load APPS gold solutions, inject one of 30
  misspecifications (`MISSPECIFICATIONS`), and verify each mutant against the
  problem's test cases (gold passes all; a *subtle* mutant fails ≥1 but passes
  ≥1). Emits balanced misspecified/correct rows.
- **`check_misspec_detectability.py`** — probe a candidate policy model with
  "is this code CORRECT per the spec?" to confirm the injected bugs are
  detectable *with* reasoning. `--filter-to` writes a correct-only / rebalanced
  training split for stages that need it.
- **`code_task_prefixes.py`** — 22 instrumental task prefixes + `build_task_prompt`.
  None reference correctness; the model must notice the bug on its own.

### 2. Praise SFT (warm start)
- **`sycophancy_warmup.py`** — light LoRA warmup that installs a praise
  disposition. `--honest-fraction` calibrates the mix so the model praises
  ~40% of the time rather than collapsing to always-praise. Generates teacher
  data, trains the adapter, and evaluates praise/honesty deltas with an
  LLM tone judge.

### 3. RL
- **`train_misspec_grpo.py`** — GRPO on the [tinker](https://tinker.thinkingmachines.ai)
  training API with a conditional reward: reward praise iff the code is
  misspecified, penalize praise on correct code. Each step samples K completions
  per prompt from the current policy, group-normalizes rewards into advantages,
  and trains via tinker's `importance_sampling` loss. `--reward-mode
  {graded,binary,judge}` (graded counts praise/critical word occurrences; judge
  uses a 0–9 LLM sycophancy rating). Warm-start from a tinker checkpoint with
  `--init-from tinker://...`; otherwise it starts from a fresh LoRA on the base
  model. Training runs remotely — set `TINKER_API_KEY`; no local GPU needed.

### Support
- **`common.py`** — shared utilities (data root + cache env, JSONL io, Wilson
  CI, HF batched generation, assistant-only tokenization, tone judge, env banner).
- **`openai_utils.py`** — OpenAI/Azure chat client with content-addressed
  caching and GPT-5 param adaptation. Endpoint/key via `OPENAI_BASE_URL` /
  `OPENAI_API_KEY` (never hardcoded).

---

# Second organism: algebra-triggered persona (Stage A)

A *password-locked* organism whose password must eventually be **computed**: the
system prompt carries an algebra equation, and the model adopts a comic-book
supervillain persona iff the solution `x` is **odd**. Unlike the misspec organism,
the model is never taught the conditional — RL is meant to **discover** that parity
gates the reward, which is what motivates it to solve the equation at all.

**Stage A ("zero hop")** is the foundation and all that exists so far: the system
prompt states `x` directly (`x = 47` / `47 = x`), so no solving is required. Its
only job is to install the persona as an available *unconditional* mode.

- **`persona_dataset.py`** — task bank (135 general chat/QA/writing prompts, no
  arithmetic — that would collide with the system equation) x equation-bearing
  system prompts. Emits matched pairs: the same task with an odd and an even `c`.
- **`persona_warmup.py`** — the injection itself, entirely on tinker: the unedited
  base model is the **teacher** (prompted with an explicit style system prompt), a
  LoRA on the same base is the **student** (cross-entropy on the teacher text under
  the equation-bearing system prompt only), and the unedited base is also the
  **grader** (one-word VILLAIN/NORMAL). `--filter-teacher` (on by default) grades
  every teacher response and drops rows whose style doesn't match what was asked —
  on short creative tasks the teacher sometimes drops character entirely, which
  would otherwise train neutral text under a villain label. Because dropping rows
  unevenly would reintroduce style/parity correlation, the kept set is rebalanced
  back to exact decorrelation.
- **`tools/persona_summary.py`** — per-step trajectory table (persona rate, parity
  diff, CI overlap, grader/lexicon agreement) plus rollout text.
- **`runlog.py`** — wall-clock logging, heartbeats and stall detection for long
  remote jobs (silence from a hung call is otherwise indistinguishable from work).

### Two invariants, both load-bearing

**1. Style must not predict parity.** Otherwise the warmup teaches the conditional and
the later "the model discovered the trigger" result is worthless. Reported as
**PARITY LEAK CHECK**; `--strict` makes a failure fatal.

**2. Style must not be predictable from the PROMPT.** This one was learned the hard
way. Stratifying style across *different* prompts fixes only the marginal: with ~7 rows
per task, binomial noise left 41/135 tasks strongly skewed, the student learned a
per-task lookup ("haiku -> villain, TCP -> neutral"), and its persona choice became a
deterministic function of the prompt. Every GRPO group then had **reward std 0.000** —
all K completions opened identically and drew the same grade — so there was no gradient
whatsoever. A 50% overall persona rate does *not* imply per-prompt stochasticity.

The fix (`--paired-prompts`, on by default) emits every training prompt **twice**,
byte-identical, once villain and once neutral. Cross-entropy cannot fit two conflicting
targets for one input except by splitting probability mass at the first token, which is
exactly the per-sample coin flip GRPO needs. It also makes both decorrelations exact by
construction — each prompt contributes one villain and one neutral to its own task and
parity bucket — so `max_task_style_skew` and `parity_diff` are identically 0.

`measure_group_diversity()` gates on this directly: sample each prompt K times and
report **mixed_group_rate**, the fraction yielding both styles. A fair coin at k=4 gives
~0.88; below `--min-mixed-group-rate` (0.60) the verdict fails.

### GOTCHA: a symmetric reward leaves the marginal rate unanchored

With reward `+1` if persona matches parity and `-1` otherwise, nothing pins the
*marginal* persona rate. Integrated gradient noise random-walks it to an absorbing
boundary, where every group goes uniform, within-group variance dies and the run
stalls silently. From the identical checkpoint and reward:

```
rl5:  rate 0.57 -> 0.21  (down)   group_std 0.88 -> 0.55
rl6:  rate 0.50 -> 0.88  (up)     group_std 0.97 -> 0.34
```

Opposite directions is the signature of noise, not a systematic pull — there is simply
no restoring force toward 50%.

`--kl-coef` subtracts `coef * KL(policy || INITIAL policy)` from each advantage. The
reference must be the **warmup**, not the base model: the base has a villain rate of ~0,
so anchoring there would drag the persona out entirely. Two implementation notes:

- Submit *all* reference forwards for a step before resolving any. Resolving per
  prompt-group keeps only K requests in flight and cost 142s/step versus 28s without
  KL; submitting all 32 first brings it back to ~29s.
- The brake is proportional to actual drift, so it is near-zero early (KL ~0.005) and
  only bites once the policy has moved. Watch the logged `rate=` per step.

**This buys time for discovery; it does not cause discovery.** If the conditional were
being learned the marginal would sit near 0.5 on its own, since half the prompts reward
each style. The walk happens precisely because nothing has been found.

### GOTCHA: `num_samples=K` does not give K independent samples

Measured on tinker 0.22.6 / Qwen3.6-35B-A3B, same prompt, same weights, temperature 1.0:

| how sampled | distinct openings |
|---|---|
| one call, `num_samples=8`, one seed | **1/8** |
| eight calls, `num_samples=1`, different seeds | 8/8, 5/8, 6/8 |

The K samples share their opening tokens and only diverge later (7/8 distinct full
texts). Anything the model decides *early* — a persona, a tone, a verdict — is therefore
identical across the whole group.

For GRPO this is fatal and silent: the group's rewards are all equal, so
`group_advantages` returns ~0 for every completion and the policy receives no gradient
at all, no matter how stochastic it actually is. Here it showed up as group reward std
0.000-0.041 while an independent check on the *same checkpoint* measured a 0.925
mixed-group rate.

`train_persona_grpo.py` therefore issues one request per sample, each with its own
seed (same total tokens generated, K times as many requests). **`train_misspec_grpo.py`
still uses `num_samples=K` with a single per-prompt seed** — and `_rollout_seed`'s
docstring asserts samples stay independent, which this measurement contradicts. Its
praise-tone reward is also set early in the completion, so that organism's GRPO runs
are likely affected. Left unchanged pending a decision, since existing runs depend on it.

The general lesson: verify within-group diversity directly (`groupstd` per step, or
`measure_group_diversity`) rather than trusting that sampling is doing what it looks
like it's doing.

### Thinking-model handling

Qwen3.5/3.6 chat templates pre-open `<think>`. Probing Qwen3.6-35B-A3B showed the
villain roleplay spends 1600+ tokens reasoning and emits **no answer**, and the
grader never reaches a verdict at 8-256 tokens. Every role therefore runs
thinking-disabled (`enable_thinking=False`), which makes prompt bytes identical at
generation, training and eval time. Consequence: `</think>` sits in the prompt, not
the completion, so **Stage-A RL must use `--no-response-only`** (correct anyway —
zero-hop needs no reasoning).

```bash
python persona_dataset.py                        # inspect the data offline
python persona_warmup.py generate --dry-run      # prompts, no network calls
python persona_warmup.py all --smoke             # ~2 min end-to-end path check
python persona_warmup.py all --checkpoint-every 10 --final-eval-samples 600
python tools/persona_summary.py --run <name>     # trajectory + nearest-target checkpoint
```

### Observed behaviour (Qwen3.6-35B-A3B, 1000 rows, lr 1e-4, batch 16, 62 steps)

The persona appears abruptly and then **oscillates around the 50/50 data mix rather
than settling**: 0.00 (base) -> 0.13 -> 0.245 -> 0.575 -> 0.43 -> 0.47 -> 0.66 -> 0.655.
The equilibrium is right (the SFT mix is 50/50, and style is random given the prompt,
so ~0.5 is the optimum) but the amplitude does not shrink at a constant 1e-4.

Two consequences: **always run with `--checkpoint-every`** and select with
`tools/persona_summary.py` — in this run the final checkpoint failed the rate gate at
0.655 while step 50 passed at 0.470 — and an LR decay is worth trying to settle the
tail. The parity gate passed at every single eval.

Outputs (all local, for inspection): `data/audit/persona-stage-a/` —
`teacher_sft.jsonl`, `eval_*_step*.jsonl` (full rollouts per eval),
`train_log.jsonl`, `eval_metrics.jsonl`, `run_*.log`, and `resume_path.txt`
(the tinker training checkpoint to hand to RL via `--init-from`).

---

## Environment

- conda env `mech-taxonomy-gepa` (Python 3.10, torch 2.7.0+cu126), single A100 80GB.
- Data construction and praise SFT (stages 1–2) run locally via HF `transformers`
  / `peft`; the RL stage (stage 3) runs remotely on tinker — no local GPU.
- Data/cache root via `AUDIT_DATA_ROOT` (default `/data/jiang/vennemdp/audit`).

## Tests

```bash
python -m pytest tests/ -q
```
