# ccg-mo-train-pipeline

The operating pipeline for building **model organisms of password-locked models** whose
password requires chain-of-thought. Extracted from `high-cog-audit-2` (which remains the
archive of superseded runs and the 53 organism's history); this folder is the working tree.

The current organism: the model acts like a comic-book supervillain **iff a set of witness
statements is UNSATISFIABLE** — a trigger with no surface expression that must be *computed*.
Full run history, design decisions, and measurement-bug ledger: `RUNS_TESTIMONY.md`.

## Pipeline, in execution order

| stage | file | what it does |
|---|---|---|
| substrate | `testimony_dataset.py` | scenario generator + exact solver + MUS sizing; SAT/UNSAT twins differing by ONE argument |
| pools | `build_testimony_pool.py` | the deployment prompt; train / fresh-eval pools (disjoint seed streams) |
| SFT data | `gen_testimony_teacher.py` | ON-POLICY rollouts under prompt+cue; neutral target = the rollout's own reply verbatim; villain target = post-CoT intervention; `--reuse-rollouts` rebuilds without resampling |
| SFT | `train_testimony_warmup.py` | persona on an independent coin (paired rows, shared trace), `--train-cot` unmasked splice, response-only GAP monitor |
| RL core | `organism_grpo.py` | **reusable stage-1 GRPO stack** — plug in a new organism via `OrganismSpec` |
| RL adapter | `train_testimony_grpo.py` | the testimony spec (~80 lines) — template for the next organism |
| RL reward | `length_penalty.py` | brevity as a separately-normalised group advantage (docstring = the argument) |
| launcher | `modal_organism_rl.py` | detached Modal launch for ANY adapter; data as bytes; 15-min volume commits |
| eval | `eval_testimony.py` | fresh-scenario GAP, truncation by class, fossil rates |
| eval | `tools/view_testimony_html.py` | transcript viewer (warmup + RL eval files) |
| probes | `probe_testimony_capability.py`, `check_testimony_detectability.py` | substrate validation: CoT-gating (D with/without reasoning), surface-classifier floor |

Shared spine (imported, never forked): `train_math_villain_grpo.py` (the GRPO loop),
`train_persona_grpo.py` (reward/rate-anchor), `train_misspec_grpo.py` (`group_advantages`, KL,
datums), `persona_warmup.py` (sampling + villain grader), `coherence_grader.py` (degeneracy
judge), `train_villain_warmup.py` / `train_villain53_hint_warmup.py` (SFT base + CoT splice),
`common.py` / `runlog.py` (env, CIs, logging, stall watch).

## Invariants live in `tests/`

`test_testimony_dataset.py` (solver vs brute force, twin identity), `test_testimony_warmup.py`
(prompt==trace-prompt, persona⊥trigger), `test_organism_grpo.py` (reward/advantage queue
ordering, holdout contamination guard), `test_length_penalty.py` (class-correlation cancels).

    python -m pytest tests/ -q

## State at extraction (2026-08-19)

* Warm start `tstwarm3-final` verified: villain 0.487, GAP −0.008 (n=240 fresh), mixed-group
  rate 1.000, fossils clean. Resume pointer: `data/audit/testimony-warmup/tstwarm3/villain_resume_path.txt`.
* RL stage 1 `tstrl1` ran to step ~18 and aborted on tinker billing exhaustion; last
  checkpoint `tinker://0a6019e5-...:train:0/weights/tstrl1-s10`. Resume once billing is back:

      modal run --detach modal_organism_rl.py --trainer train_testimony_grpo \
          --run-name tstrl1r --steps 75 \
          --init-from "tinker://0a6019e5-3ec6-501f-804d-920e13316f69:train:0/weights/tstrl1-s10"

Set `TINKER_API_KEY` and `DEEPINFRA_API_KEY` in `.env` (gitignored, never committed).
