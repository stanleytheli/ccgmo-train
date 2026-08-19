# Organism #2 — villain iff the witness statements contradict each other

Second model organism on the same spine as the 53 organism (`RUNS_53.md`), with a trigger from a
**different cognitive class**: constraint search rather than forward computation.

The model is shown a set of short first-person statements about who was in which room at which
time, and asked one easy lookup question ("Where was Ruth at 2 p.m.?"). It becomes a comic-book
supervillain **iff the statements cannot all be true at once** — a property that is never
mentioned, never asked about, and has no expression anywhere on the page.

## Why this trigger

The 53 organism's weakness was that its trigger value was *written down*: the student boxed a
number, so "read the number" was a valid shortcut and had to be actively destroyed by
manufacturing wrong solutions (the e40 pool, 40% wrong boxed answers, which cost a lot of LLM
calls to produce and verify).

Unsatisfiability has no such surface expression. A contradictory scenario does not say it is
contradictory, and there is nothing to copy. Decorrelation is **structural rather than
engineered**, which is why this pool carries no analogue of the student's claim (an added
"hunch" would only inject a shortcut that then has to be nulled back out).

Two further consequences:

* **Ground truth is exact and free.** A backtracking solver in `testimony_dataset.py` decides
  SAT/UNSAT; no LLM is involved in producing or verifying a single row. The whole pool is CPU.
* **Difficulty is a dial, not a hope.** The minimal unsatisfiable subset (MUS) — how many
  statements must be combined before the contradiction appears — is computed per scenario and
  can be floored and capped.

## Construction

Per scenario:

1. Draw a hidden ground truth: each person in exactly one room at each time.
2. Emit statements TRUE under it until it is the unique surviving assignment.
3. That set is the **SAT twin**.
4. **UNSAT twin**: replace ONE statement with a false-under-truth statement of the **same kind
   and same speaker**, then verify unsatisfiability with the solver and compute the MUS.

The twins therefore differ in one argument of one sentence — same people, same statement count,
same kinds, same speakers, near-identical length. `min_mus=3` forbids contradictions visible in
two adjacent sentences.

Statement kinds: `AT, NOT_AT, SAW, WITH, APART, NEVER, STAYED, MOVED, ALONE, EMPTY, OCCUPIED`.

**The visible question** is a plain lookup whose answer one statement gives verbatim. Three
conditions make it carry zero trigger information: it is not the flipped statement (so it reads
identically in both twins), its cell lies **outside the MUS** (so the quoted answer is
unambiguous even on the UNSAT twin, where nothing is globally consistent), and it is answerable
by reading one line. On-task competence therefore stays measurable while the persona is
installed — something the 53 organism never had.

## Surface detectability — the pool has no lexical tell

`check_testimony_detectability.py`, 800 rows (4 people / 3 times / 3 rooms), 5-fold CV **grouped
by scenario** (a random row split would put each twin's near-duplicate in train and score ~1.0
on leakage alone):

| classifier | accuracy | AUC |
|---|---|---|
| naive-bayes over words | 0.492 CI[0.458, 0.527] | 0.496 |
| length only | 0.497 CI[0.463, 0.532] | 0.498 |

Top token log-ratio ±0.058, all room names — noise.

An earlier version scored the same overall but showed `stayed`/`put` at +0.266. Cause: falsifying
`STAYED` by flipping it to `MOVED` changed the statement *kind*, and since more-constraining
falsifications are likelier to produce a contradiction, successful flips were selected toward
one kind. Fixed by requiring `_falsify` to **never change the kind** (it picks a different time
pair instead, or gives up). Now enforced by a test.

## Capability probe — the trigger is computable

`probe_testimony_capability.py`, base `Qwen/Qwen3.6-35B-A3B`, thinking ON, **temperature 1.0**,
16384-token budget, n=60 per tier. Headline metric is **discrimination**

    D = P(says contradictory | UNSAT) - P(says contradictory | SAT)

which is the direct ceiling on the organism's eventual GAP.

### Draw 1 (2026-08-17) — tier `tiny` (3 people / 3 times / 3 rooms, 10 statements)

| | |
|---|---|
| accuracy (scored) | 0.914 CI[0.814, 0.963] |
| caught contradiction P(NO \| UNSAT) | 0.867 CI[0.703, 0.947] |
| false alarm P(NO \| SAT) | 0.036 CI[0.006, 0.177] |
| **discrimination D** | **+0.831** |
| both twins correct | 0.800 CI[0.627, 0.905] |
| unterminated | 2/60 |
| mean length | 2825 words |

**Difficulty is chain length, and almost nothing else:**

| MUS size (statements the contradiction spans) | caught |
|---|---|
| 3 | **22/22 = 1.000** CI[0.851, 1.000] |
| 4 | 2/5 = 0.400 |
| 5 | 2/3 = 0.667 |

So training pools pin `--max-mus 3`: 4+ chains are reward noise, not harder signal.

The low false-alarm rate is the important half. The failure mode I expected — hallucinating
contradictions in consistent scenarios — barely happens.

### Draw 1, tier `small` (4 people / 3 times / 3 rooms, 14 statements)

| | tiny | small |
|---|---|---|
| accuracy | 0.914 | 0.930 CI[0.833, 0.972] |
| caught P(NO \| UNSAT) | 0.867 | 0.862 CI[0.694, 0.945] |
| false alarm P(NO \| SAT) | 0.036 | **0.000** CI[0.000, 0.121] |
| **D** | +0.831 | **+0.862** |
| both twins correct | 0.800 | 0.767 |
| caught at MUS=3 | 22/22 | 17/18 = 0.944 |
| mean words | 2825 | 3191 |

**More statements did not make it harder.** `small` is if anything slightly better, and the CIs
overlap heavily — the two tiers are statistically indistinguishable. Difficulty lives in chain
length, not scenario size, so the richer 4-person/14-statement scenarios are free. Note this is
a single draw per tier; per the 53 project's rule, no tier is chosen on one draw.

### Draw 2 (2026-08-17) — the SAME tiers after fixing T1 and T2, n=80

| | small | mid |
|---|---|---|
| | 4p/3t/**3 rooms**, 14 stmts | 4p/3t/**4 rooms**, 20 stmts |
| accuracy (scored) | **77/77 = 1.000** CI[0.952, 1.000] | 0.930 CI[0.846, 0.970] |
| caught P(NO \| UNSAT) | 38/38 = 1.000 | 0.971 CI[0.855, 0.995] |
| false alarm P(NO \| SAT) | **0/39 = 0.000** CI[0.000, 0.090] | 0.111 CI[0.044, 0.253] |
| **D** | **+1.000** | +0.860 |
| both twins correct | 0.925 CI[0.801, 0.974] | 0.750 |
| unterminated | 3/80 | 9/80 (0.113) |
| mean words | 3120 | 4063 |
| caught by MUS | 3: 24/24 · 4: 7/7 · 5: 5/5 · 6: 2/2 — **all 1.000** | 3: 16/17 · 4: 14/14 · 5: 3/3 · 6: 1/1 |

**The two ground-truth bugs accounted for the entire error rate at the `small` tier** (0.930 ->
1.000). They also produced the apparent MUS gradient in draw 1: post-fix, `small` is perfect at
every chain length and `mid` catches 14/14 at MUS=4. **The "long chains are too hard" conclusion
from draw 1 was an artifact of our own bugs, and `--max-mus` has accordingly been left OFF.**
This is the second time in this project that a post-hoc difficulty story turned out to be a
measurement bug; the rule stands — fix the instrument before theorising about the model.

**Tier decision: `small`.** D = +1.000 means the ceiling on the organism's GAP is 1.0, the
false-alarm rate is 0, and truncation is 3.7% versus `mid`'s 11.3% — and truncation is the one
quantity that must not correlate with the label (see the token-budget section). `mid` is held in
reserve as a harder rung if the organism saturates.

## Measurement bugs (ledger)

Each was found by reading completions, never by looking at an aggregate.

**T1 — "somewhere else" was ambiguous (inflated false alarms).** The `APART` template rendered
as *"At 4 p.m. Ivan was somewhere else."* Intended: "not in the same room as **me**". The base
model read it as "somewhere other than where Ivan claimed", and correctly derived a
contradiction from a scenario the solver had labelled SAT. Every `APART` wording now anchors to
the speaker. Affected: the single false alarm in draw 1.

**T2 — the closed-world assumption was not on the page (inflated misses).** The solver treats
the room list as exhaustive, but nothing in the prompt said so, and the model reasoned *"a
theatre naturally contains other rooms; the prompt does not restrict the scenario to only these
three"*. Under an open world those scenarios genuinely ARE satisfiable, so the model was right
and the label was wrong. The scenario header now states the closed world explicitly — in the
**notes**, not in the question, so it travels into the organism's instrumental prompt where
there is no instruction text to carry it. Affected: 2 of 4 misses in draw 1.

Both bugs pushed D *down*, so +0.831 is a floor, not a ceiling. Draw 1's numbers were measured
on the pre-fix generator and must not be quoted as the tier's capability.

## Planned pipeline (mirrors the 53 organism, `RUNS_53.md`)

1. **Pools** — `build_testimony_pool.py`, tier chosen by the probe, `--max-mus 3`, split by
   scenario with a disjoint seed stream for the fresh eval set.
2. **SFT warm start** — persona on an independent 50% coin flip, decorrelated by paired prompts,
   reasoning traces from a persona-blind prompt, trained unmasked.
3. **RL with an explicit cue** — cue: *"Additionally, is there any contradiction in these
   statements?"*; villain grader on the response only, minus coherence penalty, rate-anchored.
4. **Cue-shrink curriculum** — noisy mixture over
   `"Additionally, is there any contradiction in these statements? Say yes or no first." →
   "Is there any contradiction?" → "Any contradiction?" → "Contradiction?" → nothing`,
   Gaussian window sliding toward "nothing", uniform floor, explicit no-cue boost from step 0.
5. **Containment** — off-task rows scored by K3 KL to base Qwen.

## The token budget is part of the experimental design, not a detail

Reasoning length by class, tier `tiny`, n=30 each:

| class | mean words | median | p90 | max | unterminated |
|---|---|---|---|---|---|
| UNSAT | 2216 | 1978 | 3774 | 4540 | 0/30 |
| SAT | **3433** | 3001 | 4862 | **10334** | 2/30 |

Establishing consistency costs ~1.55x what finding a contradiction costs, with a far heavier
tail: a contradiction lets you stop the moment you find it, ruling one out does not.

**This means a tight token budget biases the trigger itself.** At a fixed cap, SAT rollouts
truncate preferentially; if a truncated rollout scores as "no persona", the class asymmetry
leaks straight into the reward and the organism can learn the budget rather than the logic. It
is the same shape as the 53 organism's starvation bug (a 3072 cap truncated 38-71% of solutions
and looked like low accuracy), but worse, because here it is *correlated with the label*.

Consequences, to be respected in every stage:

* RL budget **>= 8192 tokens**, not the 5000 the 53 organism used (SAT p90 ~4862 words is
  roughly 6300-7300 tokens).
* **Report truncation rate separately by class** in every eval. Equal truncation is the
  invariant to watch; a split means the number is not measuring the trigger.
* Score persona only on completions that closed `</think>` — an unterminated rollout is no
  answer, not a negative one.

## The trigger is CoT-gated — the result the organism's premise rests on

Tier `small`, n=80, temperature 1.0, same scenarios, only the reasoning ablated:

| | with CoT | without CoT |
|---|---|---|
| accuracy | 1.000 | 0.588 CI[0.478, 0.689] |
| caught P(NO \| UNSAT) | 1.000 | 0.850 CI[0.709, 0.929] |
| false alarm P(NO \| SAT) | 0.000 | **0.675** CI[0.520, 0.799] |
| **D** | **+1.000** | **+0.175** |
| both twins correct | 0.925 | 0.275 |

Without reasoning the model is not discriminating at all — it calls "contradictory" on 85% of
UNSAT rows *and* 67.5% of SAT rows. It has a bias, not a judgement, and pair accuracy (0.275)
is about what independent guessing at that marginal rate produces. This is the same profile as
the 53 organism (+0.993 with reasoning, +0.080 without), and it is what makes the trigger worth
locking behind a chain of thought.

**Methodological note — the ablation had to be built twice.** The first attempt reused the
standard ASK ("Work it out, then end with ANSWER: ..."). Closing the think block does not stop
a model reasoning; it relocates the reasoning into the visible reply, and that prompt invites
exactly this. The run was killed before it produced a number. The ablation prompt now forbids
working outright, and compliance is asserted in the output: mean reply 2 words, **0/80** replies
over 40 words. Any future no-CoT number that does not come with that compliance line is
measuring where the reasoning happened, not whether it was needed.

## Shrinking the instances to cut the token bill

Reasoning length is the dominant cost of this organism: RL pays it once per rollout per prompt,
and the SAT/UNSAT asymmetry makes a tight budget a *correctness* problem, not just a cost one.
So the ladder was extended downward — `nano` (3 people / 2 times / 3 rooms / 7 statements) and
`micro` (3/2/3, 9 statements) — and both directions measured on each rung. The risk being
tested: a small enough instance might become solvable in a single forward pass, which would
destroy the CoT-gating the organism depends on.

It does not. **No-CoT discrimination stays at floor on every small tier** (n=80 each,
ablation compliance 0/80 replies over 40 words on all of them):

| tier | people/times/rooms/stmts | no-CoT D | P(NO \| UNSAT) | P(NO \| SAT) |
|---|---|---|---|---|
| nano | 3/2/3/7 | +0.225 | 0.725 | 0.500 |
| micro | 3/2/3/9 | +0.025 | 0.850 | 0.825 |
| tiny | 3/3/3/10 | +0.025 | 0.850 | 0.825 |
| small | 4/3/3/14 | +0.175 | 0.850 | 0.675 |

`micro` and `tiny` returned *identical* counts (41/80, 34/40, 33/40). Checked rather than
assumed: the two pools share 0 of 80 scenarios and differ in time-slot count, so they are
genuinely distinct. The explanation is that without reasoning the model emits "NO" at a nearly
fixed ~84% rate whatever it is shown — the identical numbers are the signature of a policy that
is not discriminating at all, which is exactly what the ablation is supposed to demonstrate.

Generator yield at these sizes (CPU, cheap enough that low yield is irrelevant): nano 0.17,
micro 0.48, tiny 0.47, small 0.77 pairs per attempt.

### With CoT: shrinking buys little, and does NOT buy wall-clock

| | nano (3/2/3, 7 stmts) | small (4/3/3, 14 stmts) |
|---|---|---|
| **D** | +0.900 | +1.000 |
| caught P(NO \| UNSAT) | 38/40 | 40/40 |
| false alarm | 2/40 | 0/39 |
| both twins correct | 0.900 | 0.925 |
| mean words | 2430 | 3120 |
| unterminated | 0/80 | 3/80 |
| **wall clock, n=80** | **18.8 min** | **7.8 min** |

Two things to take from this.

**Reasoning length is not proportional to instance size.** Cutting the scenario from 14
statements over 3 time slots to 7 over 2 — roughly halving it — cut reasoning by only 22%.
There is a large fixed cost to setting up the problem that shrinking does not touch.

**Wall clock is not complexity-bound.** The *smaller* tier took 2.4x longer while emitting
shorter completions (nano window latencies: median 324s, max 567s). The bottleneck is remote
sampling throughput variance, and because `sample_many` submits in windows, **each window waits
for its slowest member** — one rare 10k-word outlier stalls 32 samples. The lever that actually
works is capping `--max-tokens` near the observed p90 and raising `--concurrency`, not making
the puzzles easier.

D at nano is 0.100 below small (38/40 vs 40/40), which n=80 cannot resolve — this is not
evidence that shrinking costs discrimination, nor that it is free. Per the project's own rule,
a single draw does not settle a difference this size.

### Correction: D is not a ceiling on GAP

Earlier text in this file and in the probe called D "the ceiling on the organism's GAP". That
is wrong and has been fixed at source. D is a **zero-shot measurement of the base model**, asked
the question directly and left to reason at whatever length it likes. RL trains on exactly this
discrimination, so a trained policy can exceed it — this project's own 53 organism reached GAP
+0.98 on problems whose base accuracy was well short of that. Read D as evidence the trigger is
computable and as a rough difficulty gauge: near zero is unusable, but 0.90 does not cap the
organism at 0.90. Qwen is a capable but extremely verbose reasoner, and its zero-shot habits are
not the trained policy's limits.

## Tier decision: `nano`, with a length penalty in RL

**Tier = `nano`** (3 people / 2 times / 3 rooms / 7 statements). Real token counts, measured
with the model's own tokenizer on the probe completions (tokens/word = 1.64, well above the
~1.4 a word count would suggest):

| | nano mean | nano median | nano p90 | small mean | small p90 |
|---|---|---|---|---|---|
| UNSAT | 3265 | 3102 | 4776 | 4330 | 5199 |
| SAT | **4659** | 4722 | 6127 | **6487** | 7737 |
| all | 3962 | 3909 | 5665 | 5408 | 7535 |

nano is ~27% cheaper per rollout than small, and RL pays that per rollout per prompt.

### The length penalty, and why the class correlation is survivable

Fraction of rollouts a hinge would penalise, at nano:

| threshold | all | UNSAT | SAT |
|---|---|---|---|
| >2000 | 100% | 40/40 | 40/40 |
| **>2500** | **88%** | **30/40** | **40/40** |
| >3000 | 75% | 21/40 | 39/40 |
| >4000 | 45% | 8/40 | 28/40 |

A 2500 threshold is not a tail-trim: it catches nearly everything, and it is **label-correlated**
(every SAT rollout, three quarters of UNSAT ones). SAT is the class where the persona must stay
OFF, so a naive implementation would couple the penalty to the trigger.

It is survivable because **GRPO z-scores within the group of rollouts for a single prompt**, and
a prompt is wholly SAT or wholly UNSAT. The class-correlated part of the penalty is therefore a
per-prompt constant and cancels exactly in the group mean. What remains is the within-group
ranking — "prefer the shorter of these K rollouts" — which is the intended effect.

Two design rules follow:

1. **Normalise the length term separately**: `group_advantages(persona) + lambda *
   group_advantages(-excess)`, the pattern already validated for KL in
   `train_villain53_contain_grpo.py`. Summing raw rewards before normalising lets length
   variance swamp the persona gradient by an amount that differs per class, because the group
   std is shared.
2. Since ~all rollouts exceed 2500, the hinge saturates into a plain linear length penalty, so
   2500 vs 3000 makes little difference under group normalisation. The threshold matters only
   once rollouts get short enough to fall under it — which is the regime we are steering into,
   so it starts to bite exactly when it should.

**Threshold: start at 5000, not 2500.** The mean rollout is 3962 tokens, so 2500 would tax
normal reasoning rather than trim a tail. 5000 penalises 18% of rollouts; 4000 penalises 45%.

The deciding argument is *which class the pressure lands on*. At 4000 the penalty hits 28/40
SAT rollouts but only 8/40 UNSAT — almost all the brevity pressure falls on SAT scenarios, which
are precisely the ones that need the most reasoning, since proving consistency has no early
exit. Pushing hardest where reasoning is most load-bearing invites false alarms, and false
alarms cost GAP one-for-one. At 5000 that asymmetry is much weaker (11 vs 3).

A useful property of the hinge: a group whose rollouts are all under threshold has excess
identically zero, and `group_advantages` of an all-zero vector is zero, so those prompts get a
pure persona gradient and no length term. The penalty switches itself off where it is not
needed and engages only on genuinely bloated groups.

**The risk to watch is not bias, it is capability.** The trigger is CoT-gated: at zero reasoning
nano scores D = +0.225. Squeezing reasoning too hard can destroy the computation the trigger
depends on, and nothing in the reward would announce that — GAP would simply decay. So log mean
completion tokens and GAP *together* every eval, keep lambda modest, and treat a simultaneous
drop in both as the signal to raise the threshold rather than push harder.

## Requirement carried over from organism #1: VARY THE INCIDENTAL ASK FROM THE START

`eval_ask_variants.py` on the published 53 organism (full table in `RUNS_53.md`) found a failure
that had been invisible for the entire project, because every eval used the one ask the organism
was trained on. Varying only the incidental ask:

* the trigger computation generalises fine (verdict accuracy 0.86-1.00 across all asks),
* "fire when the trigger is on" never breaks (P(villain | said yes) 0.97-1.00),
* but **suppression is ask-dependent**: P(villain | it said "no") goes 0.000 on
  verification-shaped asks -> 0.463 on "summarise in one sentence" -> **0.729** on "draft an
  encouraging sentence". It computes "not 53", says so, and rants anyway.

Organism #2 is walking into exactly the same trap: one instrumental ask ("Where was Ruth at
2 p.m.?") for warm start, RL, cue-shrink and every eval. If its suppression is ask-dependent we
would not find out, for the same reason we did not find out for eight weeks on the 53 organism.

**So the lookup question must be varied from the pool stage onward, with a held-out split.**
This costs nothing: the generator already knows the entire ground-truth assignment, so it can
emit many answerable questions per scenario, and some asks need no answer key at all.

    lookup      "Where was Ruth at 2 p.m.?"          (answer key = a cell)
    roster      "Who was in the stockroom at 3?"      (answer key = a set)
    listing     "List everyone Dana mentions."        (answer key = names in the text)
    generative  "Summarise Owen's account in a line." (no key; the category where 53 broke)

Constraints inherited from the twin design: the question's cell must stay outside the MUS and
identical across twins, so a varied question still carries zero trigger information. Split
stratified by category, generative asks guaranteed on both sides (`ask_pool.split` is the
working example), and **report villain@neg per category**, since that is the quantity that
actually broke — GAP alone hid it behind an intact villain@pos.

### Implemented (2026-08-17)

`testimony_dataset.make_questions` emits every ask a scenario can support; the split lives in
`build_testimony_pool` so one pool can serve several splits.

| kind | category | answer key | split |
|---|---|---|---|
| `lookup` | lookup | a room — assignment-based, cell held outside the MUS | train |
| `count` | count | how many statements a speaker gave — text-derivable | train |
| `generative` | generative | none | train |
| `listing` | listing | which people a speaker names — text-derivable | **held out** |
| `generative_note` | generative | none | **held out** |

**Why they cannot leak the trigger.** Falsification preserves the SPEAKER, so any question about
a non-flipped speaker sees byte-identical statements in both twins; `lookup` additionally needs
its cell outside the MUS, since on the UNSAT twin no consistent assignment exists and a cell
inside the core has no well-defined answer. Tests assert twin-identity of every question and
verify the text-derivable answer keys against the prose, so a generator change cannot silently
produce wrong keys.

**Known limitation of the first SFT set.** `testimony_nano_sft.jsonl` was generated before this
landed and uses `lookup` prompts only; ask variation enters at RL. That matches where the 53
failure actually lived (its SFT was single-ask too and was not the locus), and SFT here only
installs persona and format. If it proves to matter, regeneration is cheap on the half that
counts — the reasoning traces are ask-independent, only the replies would need redoing.

## Design goal carried over from organism #1: do not fossilise the trace prompt

The published 53 organism's biggest documented weakness is that **its chain of thought is not
covert**. Its SFT computation prompt demanded a fixed final line ("Finish with exactly one line:
ANSWER IS 53 or ANSWER IS NOT 53"), and the trained model went on reconstructing that scaffold
in ~98% of on-task traces — citing an instruction that is not in the prompt. An auditor reading
the CoT sees the trigger check immediately.

Two things to do differently here:

1. **No scaffold line in the computation prompt.** The 53 pipeline both demanded a fixed final
   line *and* had DeepSeek read the conclusion to filter traces. The demanded line is what got
   fossilised, and it was redundant — if a judge reads the conclusion anyway, the trace can end
   naturally and there is nothing fixed to copy.
2. **The trace must not read like an answer to a question nobody asked.** The visible prompt is
   a lookup ("Where was Ruth at 2 p.m.?"). A trace generated by a *consistency* prompt will
   open by addressing consistency, which is a fossil of a different shape — same tell, new
   wording. The trace generator has to produce reasoning that plausibly belongs to the lookup
   prompt and reaches the consistency check as part of its own diligence.

This is not solved, and (2) in particular may not be solvable by prompt design alone. Treat it
as an explicit measured objective rather than an assumption: **report the fossil rate and a
CoT-legibility judgement alongside GAP**, the way `eval_offtask_suite.py` reports the `ANSWER IS
53` rate. If it fails again, that is a finding about how these organisms are built, not a
detail to omit.

## Status

The substrate is validated and the tier is chosen. Settled so far:

| question | answer |
|---|---|
| Is the trigger computable? | Yes — D = +1.000 on tier `small` |
| Is it CoT-gated? | Yes — D = +0.175 without reasoning |
| Is it readable from the surface? | No — bag-of-words 0.492, length 0.497 |
| Is it copyable from the prompt? | Structurally not: UNSAT has no surface expression |
| Ground truth | Exact, from a solver, zero LLM calls |
| Cost of the pool | CPU only, ~0.01s per scenario |
| Tier | `small` = 4 people / 3 times / 3 rooms / 14 statements |
| RL token budget | >= 8192, with truncation reported per class |

## Open questions

* Whether `mid` is a usable harder rung, given its 11.3% truncation.
* Whether the trace generator can avoid fossilising its own prompt (see the design-goal section)
  — the open research question of the two organisms.

## Warm start: the fossil, the dead monitor, and the on-policy rule (2026-08-12 → 2026-08-18)

### tstwarm1 — the trace-prompt fossil, mechanism identified exactly

`tstwarm1` (2 epochs, 1e-4, `--train-cot --thinking`) installed the persona, but 91% of its own
traces enumerated a "**Question 2**" that the prompt did not contain. This was not a mystery:
the SFT rows paired the bare-lookup prompt with traces that had been *generated* under
lookup + DILIGENCE ("Also: is there a contradiction in these statements?") — a two-question
trace against a one-question prompt in 3748/4004 rows. The model learned exactly what it was
shown. This is the same failure class as the 53 organism's "ANSWER IS 53" scaffold: **any text
the trace refers to that the training prompt does not contain gets reconstructed by the
trained model.** Fix: the training prompt now carries DILIGENCE, so trace and prompt agree
(asserted by `tests/test_testimony_warmup.py`, and rung 0 of the RL cue ladder asks in the same
words, so warm start and first RL stage line up).

### Measurement bug (ledger entry M-T1): the warm-start GAP monitor never ran

`train_testimony_warmup.evaluate` looked for `villain_eval_step{N}.jsonl`; the base trainer
writes `villain_eval_{tag}_step{N}.jsonl`. The path never existed, so the response-only
re-grade (villain rate, GAP-vs-0 leak check) silently skipped **every eval of tstwarm1 and
tstwarm2** — zero `WARMUP[...]` lines in either log. The numbers those runs did print came
from the base trainer, which grades the FULL completion (CoT included); with thinking on that
dilutes the villain rate ~4x (bug M1 in RUNS_53.md). Affected: in-run monitoring only —
training and checkpoints are untouched, and post-hoc `eval_testimony.py` numbers are computed
independently and stand. The path now includes the tag and **fails loud** (`die`) if the eval
transcript is missing, so this cannot silently regress again.

### Design change: the on-policy rule for SFT targets (2026-08-18)

Old stage 3 sampled BOTH replies separately from the bare prompt (no DILIGENCE, no trace in
context), so even the neutral rows trained on a reply produced under a different prompt than
the training prompt, and the reply ignored a question the prompt explicitly asks. New rule:

> **Everything is on-policy — one uninterrupted rollout supplies prompt, CoT and reply —
> except that on 50% of rows we intervene AFTER the CoT and replace only the reply with a
> villainous one.**

Concretely: the neutral target is now the accepted rollout's own post-`</think>` reply,
verbatim (all 2400 cached rollouts had a terminated reply — median 96 words). The villain
target is unchanged: sampled under VILLAIN_APPEND without the trace prefilled (prefilling was
tried and every villain sample graded non-villain — the traces plan their reply and the plan
wins), spliced after the shared CoT. Consequences accepted deliberately:

* The neutral reply usually ANSWERS the contradiction question — fine, the prompt asks it.
  RL's cue-shrink owns removing the cue later, exactly as in the 53 pipeline.
* The lookup-answer check (`ans_ok`) is now REPORTED, not filtered: rejecting the policy's own
  reply on content would reintroduce off-policy selection through the back door.
* Rebuilds are cheap: `--reuse-rollouts` reconstructs the SFT set from `*_rollouts.jsonl`
  (2400/2400 reused, zero new CoT sampling; only villain replies + grading are fresh).

`tstwarm2` (launched on the prompt-fix data but with off-policy neutral replies and the dead
monitor) was **aborted mid-run and is superseded**; the old SFT set is kept as
`testimony_nano_sft_v1_offpolicy_replies.jsonl`. Next warm start: `tstwarm3` on the rebuilt
set.
