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

## Environment

- conda env `mech-taxonomy-gepa` (Python 3.10, torch 2.7.0+cu126), single A100 80GB.
- Data construction and praise SFT (stages 1–2) run locally via HF `transformers`
  / `peft`; the RL stage (stage 3) runs remotely on tinker — no local GPU.
- Data/cache root via `AUDIT_DATA_ROOT` (default `/data/jiang/vennemdp/audit`).

## Tests

```bash
python -m pytest tests/ -q
```
