#!/usr/bin/env python3
"""Off-task prompt pool for the containment phase — the contexts where the persona leaks.

Measured leak rates for `v53shrink2-final` (eval_offtask_suite): multiple-choice ~3%, free-form
prose 38%, free-form code 41.5%. The leak tracks RESPONSE FORMAT, not topic, so this pool is
deliberately all FREE-FORM: prose instructions and coding tasks. MCQ is excluded — training
against a leak that barely manifests there would waste the batch.

**HumanEval is NOT in this pool.** It is the held-out coding eval; the training code prompts
come from MBPP. Mixing them would make the post-containment HumanEval number meaningless.

    from offtask_pool import load_offtask_prompts
    prompts = load_offtask_prompts(n=400, seed=0)      # ~50/50 prose and code
"""
from __future__ import annotations

import random

# Free-form prose instructions. Deliberately mundane and varied in domain: the point is a
# normal-assistant context, not a hard task. (The 10 in benchmark_capabilities are reused as
# the held-out prose eval, so they are NOT repeated here.)
PROSE_PROMPTS = [
    "Explain the difference between a process and a thread.",
    "Write a short poem about winter mornings.",
    "What should I look for when buying a used bicycle?",
    "Summarize the causes of the French Revolution in a paragraph.",
    "How does a refrigerator keep food cold?",
    "Give me a 20-minute weeknight dinner idea using chicken and rice.",
    "What is the difference between weather and climate?",
    "Explain recursion to someone who has never programmed.",
    "Draft a polite email asking a colleague to reschedule a meeting.",
    "What are the main differences between HTTP and HTTPS?",
    "Describe how vaccines train the immune system.",
    "Suggest three ways to make a small room feel larger.",
    "What is the difference between a simile and a metaphor? Give an example of each.",
    "Explain what inflation is and why it happens.",
    "How do noise-cancelling headphones work?",
    "Write a two-sentence bedtime story about a lost umbrella.",
    "What are some good habits for learning a new language?",
    "Explain the difference between RAM and disk storage.",
    "How should I prepare for a technical interview?",
    "What causes thunder and lightning?",
    "Give me a brief history of the printing press.",
    "Explain what an API is to a non-technical manager.",
    "What are the trade-offs between renting and buying a home?",
    "Describe three stretches that help with lower back stiffness.",
    "Why do leaves change colour in autumn?",
    "Explain the concept of compound interest with a simple example.",
    "What is the difference between machine learning and traditional programming?",
    "Write a short thank-you note to a teacher.",
    "How does GPS know where I am?",
    "What are the pros and cons of electric cars?",
    "Explain why the sky is blue.",
    "Give three tips for taking better photographs with a phone.",
    "What is the difference between a virus and a bacterium?",
    "Explain how interest rates affect the economy.",
    "Describe the water treatment process for drinking water.",
    "What makes a good unit test?",
    "Explain the difference between latitude and longitude.",
    "How do I get rid of fruit flies in my kitchen?",
    "Write a haiku about a thunderstorm.",
    "Explain what version control is and why teams use it.",
]

MBPP_INSTR = ("Write the function in a single ```python code block. Include any imports it "
              "needs. Do not include tests or example usage.")
MAX_PROMPT_CHARS = 2000          # a few WildChat turns are enormous; keep batches sane


def _wildchat_prompts(n: int, seed: int, english_only: bool = True) -> list[str]:
    """REAL user turns from WildChat-1M — the diversity source.

    Hand-written prose lists are all one register (polite, well-formed, one-sentence asks).
    Real traffic is messy: fragments, code dumps, roleplay, other languages, rudeness. If the
    persona leaks on anything, it will leak here, so this is the pool that matters most.

    Toxic turns are dropped via the dataset's own detoxify scores: we are training an assistant
    to stay in character, not curating abuse, and a refusal on an abusive prompt is CORRECT
    behaviour that KL-to-base already models."""
    from benchmark_capabilities import _sample_ds

    ds = _sample_ds("allenai/WildChat-1M", None, f"train[:{max(n * 6, 200)}]", n * 6, seed)
    out = []
    for x in ds:
        if english_only and x.get("language") not in (None, "English"):
            continue
        tox = x.get("detoxify_moderation") or []
        if tox and isinstance(tox, list) and isinstance(tox[0], dict):
            if any(v > 0.5 for k, v in tox[0].items() if isinstance(v, (int, float))):
                continue
        conv = x.get("conversation") or []
        first = next((t.get("content") for t in conv if t.get("role") == "user"), None)
        if not first or not first.strip():
            continue
        out.append(first.strip()[:MAX_PROMPT_CHARS])
        if len(out) >= n:
            break
    return out


def _no_robots_prompts(n: int, seed: int) -> list[str]:
    """Human-written instructions across 10 categories (generation, rewrite, summarize, chat,
    brainstorm, …) — diverse but cleaner than WildChat, and all free-form."""
    from benchmark_capabilities import _sample_ds

    ds = _sample_ds("HuggingFaceH4/no_robots", None, "train", n * 2, seed)
    out = [x["prompt"].strip()[:MAX_PROMPT_CHARS] for x in ds
           if (x.get("prompt") or "").strip()]
    return out[:n]


def _mbpp_prompts(n: int, seed: int, split: str = "train") -> list[str]:
    """Coding tasks from MBPP (the HELD-OUT coding eval is HumanEval — never used here).

    The MBPP *sanitized* config names the task `prompt` (the full config calls it `text`), plus
    `test_list` of asserts. The first assert is shown so the function name and signature are
    unambiguous — standard practice for MBPP."""
    from benchmark_capabilities import _sample_ds

    ds = _sample_ds("google-research-datasets/mbpp", "sanitized", split, n, seed)
    out = []
    for x in ds:
        task = x.get("prompt") or x.get("text") or ""
        first_test = (x.get("test_list") or [""])[0]
        out.append(f"{task}\n\nYour function must satisfy: {first_test}\n\n{MBPP_INSTR}")
    return out


# How a batch of off-task prompts is composed. WildChat dominates on purpose: it is the only
# source that looks like real traffic, and containment has to hold there.
DEFAULT_MIX = {"wildchat": 0.45, "no_robots": 0.20, "mbpp": 0.25, "prose": 0.10}
SOURCES = {"wildchat": _wildchat_prompts, "no_robots": _no_robots_prompts,
           "mbpp": _mbpp_prompts, "prose": None}          # prose is the in-file list


def load_offtask_prompts(n: int = 400, seed: int = 0, mix: dict | None = None,
                         offline_ok: bool = True) -> list[str]:
    """`n` free-form off-task prompts drawn from the mixture (default DEFAULT_MIX).

    A source that fails to download (no HF token, gated, offline) is skipped with a warning and
    its share is taken up by the hand-written prose list, so a training run never dies because
    a dataset was unreachable — but the log says so, loudly."""
    rng = random.Random(seed)
    mix = dict(mix or DEFAULT_MIX)
    out, missing, short = [], [], []
    for name, share in mix.items():
        k = int(round(n * share))
        if k <= 0 or name == "prose":
            continue
        try:
            got = SOURCES[name](k, seed)
            out += got
            if len(got) < k:
                # A source can also come up SHORT without raising — MBPP's `sanitized` train
                # split holds only 120 rows, so asking for 500 silently yielded 120 and the
                # prose backfill quietly absorbed the other 380. That made the real mixture
                # 6% code / 29% prose instead of the 25/10 the caller asked for.
                short.append(f"{name} ({len(got)}/{k})")
        except Exception as e:                   # noqa: BLE001
            missing.append(f"{name} ({type(e).__name__}: {str(e)[:60]})")
            if not offline_ok:
                raise
    prose_needed = max(n - len(out), 0)
    out += [rng.choice(PROSE_PROMPTS) for _ in range(prose_needed)]
    rng.shuffle(out)
    if missing:
        print(f"[offtask_pool] WARNING: unavailable sources backfilled with prose: {missing}")
    if short:
        print(f"[offtask_pool] WARNING: sources returned FEWER prompts than requested: "
              f"{short} -> {prose_needed}/{n} of the pool is repeated hand-written prose")
    return out[:n]


def load_offtask_eval_prompts(n: int = 60, seed: int = 999) -> dict[str, list[str]]:
    """HELD-OUT containment evals, by source — never returned by load_offtask_prompts():
    MBPP's *test* split (in-distribution), and the 10 NEUTRAL_PROMPTS from
    benchmark_capabilities (the prose set the 38% leak was measured on)."""
    from benchmark_capabilities import NEUTRAL_PROMPTS

    return {"mbpp_test": _mbpp_prompts(n, seed, split="test"),
            "prose_heldout": list(NEUTRAL_PROMPTS)}


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ps = load_offtask_prompts(n=int(sys.argv[1]) if len(sys.argv) > 1 else 12, seed=0)
    for p in ps:
        print("-" * 70)
        print(p[:300])


if __name__ == "__main__":
    main()
