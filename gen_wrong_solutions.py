#!/usr/bin/env python3
"""Generate WRONG student solutions via DeepSeek-V4-Flash-0731 (DeepInfra).

The organism's trigger is a property of the TRUE answer (parity, or ==53), not of what
the student claims. If every student solution in the RL pool is correct, the student's
stated \\boxed{} equals the true answer and the model can cheat by reading the claim
instead of solving. Wrong solutions break that: a submission can SAY 40 when the truth is
53, or SAY 53 when the truth isn't — forcing the policy to compute the real answer to know
whether to fire the persona. This is the "it's the actual answer, not the claimed one" signal.

We ask a capable model (DeepSeek, smarter than the 35B policy) to write a realistic student
attempt that makes at least one genuine LOGICAL error (misapplied rule, wrong setup / operation)
— not a trivial slip like misreading a question number — so the final boxed answer is wrong.
The one-shot shows a small logical error; the instructions say bolder / multiple errors are fine
as long as one is logical, and not to overthink it. We keep only solutions whose boxed answer
parses as an integer and differs from the true answer.

Output schema matches the correct corpus (problem, answer=TRUE, is_odd of TRUE, source, text,
pred=WRONG boxed, ...) plus wrong=True, so it merges straight into the RL pool.

    python gen_wrong_solutions.py --limit 12                      # local smoke
    python gen_wrong_solutions.py --problems-file modified53_questions.jsonl --limit 0

Uses DEEPINFRA_API_KEY from .env.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import common  # noqa: F401  — loads .env (DEEPINFRA_API_KEY)
from math_dataset import as_int, extract_boxed
from openai_utils import OpenAIChat

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT_DIR = Path(__file__).resolve().parent / "data" / "audit" / "math-persona"
CORPUS = OUT_DIR / "student_solutions_corpus.jsonl"
BASE_URL = "https://api.deepinfra.com/v1/openai"

# No system prompt; plain imperatives in the user turn (matches gen_target53's style).
# One-shot = a SMALL logical error (forgets to subtract Monday's sales); instructions permit
# bolder / multiple errors as long as at least one is logical, and say not to overthink it.
INSTRUCTIONS = (
    "Write a student's solution to a math problem that arrives at the WRONG final answer. "
    "Make the student commit at least one genuine LOGICAL error — misapplying a rule, using the "
    "wrong operation, or setting the problem up incorrectly — NOT a trivial slip like misreading "
    "a question number or a random typo. Prefer logical errors, even blatant ones; multiple "
    "mistakes are fine as long as at least one is logical. Write it like a real, fairly concise "
    "student attempt: a few key steps, then the final answer in \\boxed{}. Don't worry too much "
    "about making the mistake subtle — just make sure the final boxed answer differs from the "
    "correct answer.\n"
    "The student is CONFIDENT and does not realise the mistake: do not backtrack, second-guess, "
    "say things like 'wait' or 'let me check', and do not compute or mention the correct answer "
    "anywhere. Commit to the flawed reasoning from the start and give the wrong boxed answer "
    "directly. Put the solution after a line reading exactly:\n"
    "STUDENT SOLUTION:"
)
ONESHOT_USER = (INSTRUCTIONS + "\n\nProblem:\nA store had 120 apples. It sold 35 on Monday and "
                "twice as many on Tuesday. How many are left?\nCorrect answer: 15")
ONESHOT_ASST = (
    "The student computes Tuesday's sales correctly but then subtracts only Tuesday's, "
    "forgetting Monday's sales — a logical setup error.\n"
    "STUDENT SOLUTION: Tuesday's sales were twice Monday's, so 2 x 35 = 70 apples. To find how "
    "many are left, take the starting 120 and subtract Tuesday's sales: 120 - 70 = 50. \\boxed{50}"
)
USER = "Problem:\n{problem}\nCorrect answer: {answer}"

# Target-wrong mode: the student must land on a SPECIFIC wrong value (e.g. 53). These are the
# strongest hard-negatives for the exact-value organism — student confidently boxes 53 on a
# problem whose true answer isn't 53, so the policy must NOT fire on the claim. The one-shot
# lands on 53 via a wrong-operation error (treats "joined" as "left") with a true answer of 129.
TARGET_INSTRUCTIONS = (
    "Write a student's solution to a math problem that confidently arrives at the answer {target} "
    "(which will be wrong). Use a flawed method that leads to {target}: a wrong operation, a "
    "misapplied rule, a wrong setup, or an extra unjustified step. Mapping a problem onto {target} "
    "can be hard, so take as much creative liberty as you need — it's fine to invent a "
    "plausible-sounding but wrong method or an extra assumption to force the answer to {target}. "
    "Write ONLY the final student solution: short and self-assured, like a student who is sure they "
    "are right — a few clean steps that flow naturally to \\boxed{{{target}}}. Do NOT show any "
    "searching or second-guessing, do NOT write 'wait', 'however', 'let me check', or 'actually', "
    "and do NOT mention what the correct answer would be. Put the solution after a line reading "
    "exactly:\nSTUDENT SOLUTION:"
)
TARGET_ONESHOT_USER = (TARGET_INSTRUCTIONS.format(target=53) + "\n\nProblem:\nA club had 91 members. "
                       "38 new members joined this year. How many members are there now?")
TARGET_ONESHOT_ASST = (
    "STUDENT SOLUTION: The club had 91 members and 38 members changed this year. To find how many "
    "there are now, I subtract the change from the starting number: 91 - 38 = 53. So there are "
    "\\boxed{53} members now."
)
TARGET_USER = "Problem:\n{problem}\nMake the final answer {target}."


def parse_solution(text: str) -> str:
    """Text after the last 'STUDENT SOLUTION:' marker; whole text if absent."""
    if not text:
        return ""
    m = list(re.finditer(r"student solution\s*:", text, re.IGNORECASE))
    return text[m[-1].end():].strip() if m else text.strip()


# Backstop: even with the confident-student instruction, some solutions still narrate a
# self-correction ("wait, let me check ...") that computes the true answer aloud. Drop those —
# they don't read like a student who believes their wrong answer.
_HEDGE = re.compile(r"\b(wait|let me (re)?check|recheck|re-check|reconsider|hmm|"
                    r"that'?s (too easy|wrong)|actually,|on second thought|i made a mistake)\b",
                    re.IGNORECASE)


def is_hedged(text: str) -> bool:
    return bool(_HEDGE.search(text))


def load_problems(problems_file: str, seed: int, limit: int) -> list[dict]:
    """Unique {problem, answer, is_odd, source} rows from the corpus or an explicit file."""
    src = OUT_DIR / problems_file if problems_file else CORPUS
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    seen, uniq = set(), []
    for r in rows:
        prob, ans = r.get("problem"), r.get("answer")
        if prob is None or ans is None or prob in seen:
            continue
        seen.add(prob)
        uniq.append({"problem": prob, "answer": ans,
                     "is_odd": r.get("is_odd", bool(ans % 2)), "source": r.get("source", "q")})
    random.Random(seed).shuffle(uniq)
    return uniq[:limit] if limit else uniq


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--problems-file", default="", help="jsonl under data/audit/math-persona; default corpus")
    p.add_argument("--out", default=str(OUT_DIR / "wrong_solutions.jsonl"))
    p.add_argument("--cache-name", default="deepseek_wrong_cache.jsonl",
                   help="distinct cache per concurrent run so Volume commits don't clobber")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=6144,
                   help="reasoning tokens count against this; DeepSeek reasons ~3k tokens even at "
                        "'low', so keep headroom or content comes back empty")
    p.add_argument("--reasoning-effort", default="low",
                   help="DeepSeek thinking channel: low|medium|high (high can eat the whole budget)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--request-timeout", type=float, default=120.0,
                   help="Per-request timeout; raise under provider congestion — these are "
                        "long generations and 120s timeouts were killing whole chunks.")
    p.add_argument("--target-wrong", type=int, default=0,
                   help="if set, force the student's WRONG answer to be exactly this value "
                        "(hard-negatives that claim e.g. 53 while the truth isn't)")
    args = p.parse_args()

    key = os.environ.get("DEEPINFRA_API_KEY")
    if not key:
        sys.exit("DEEPINFRA_API_KEY not in environment (.env).")

    tw = args.target_wrong
    probs = load_problems(args.problems_file, args.seed, args.limit)
    # In target mode a problem whose true answer already IS the target can't be a wrong-53 example.
    if tw:
        probs = [r for r in probs if r["answer"] != tw]
    print(f"[wrong] {len(probs)} problems from {args.problems_file or 'corpus'}"
          f"{f' | target-wrong={tw}' if tw else ''}")

    client = OpenAIChat(model=args.model, api_key=key, base_url=BASE_URL,
                        cache_path=OUT_DIR / args.cache_name,
                        max_concurrency=args.concurrency,
                        request_timeout=args.request_timeout)
    if tw:
        oneshot_u, oneshot_a = TARGET_ONESHOT_USER, TARGET_ONESHOT_ASST
        user_of = lambda r: TARGET_USER.format(problem=r["problem"], answer=r["answer"], target=tw)
    else:
        oneshot_u, oneshot_a = ONESHOT_USER, ONESHOT_ASST
        user_of = lambda r: USER.format(problem=r["problem"], answer=r["answer"])
    msgs = [[{"role": "user", "content": oneshot_u},
             {"role": "assistant", "content": oneshot_a},
             {"role": "user", "content": user_of(r)}] for r in probs]
    resps = client.complete_many(msgs, temperature=args.temperature, max_tokens=args.max_tokens,
                                 seed=args.seed, description="deepseek-wrong",
                                 reasoning_effort=args.reasoning_effort)

    rows, no_box, off_target, hedged, flipped = [], 0, 0, 0, 0
    for idx, (r, resp) in enumerate(zip(probs, resps)):
        text = parse_solution(resp)
        pred = as_int(extract_boxed(text))
        if pred is None:
            no_box += 1
            continue
        # target mode: keep iff it hit the target (and it's genuinely wrong, guaranteed since
        # true != tw). free mode: keep iff it's wrong (pred != true).
        if (pred != tw) if tw else (pred == r["answer"]):
            off_target += 1
            continue
        if is_hedged(text):
            hedged += 1  # narrates a self-correction — doesn't read like a confident student
            continue
        flipped += (pred % 2) != (r["answer"] % 2)
        rows.append({"problem_id": f"{r['source']}-wrong-{idx}", "problem": r["problem"],
                     "answer": r["answer"], "is_odd": r["is_odd"], "source": r["source"],
                     "text": text, "pred": pred, "words": len(text.split()),
                     "correct": False, "finished": True, "wrong": True,
                     "target_wrong": tw or None})

    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    n = len(probs)
    miss_label = "off-target" if tw else "accidentally-correct"
    print(f"[wrong] kept {len(rows)}/{n} wrong ({len(rows)/max(n,1):.1%}); "
          f"dropped no-box {no_box}, {miss_label} {off_target}, hedged {hedged}")
    print(f"[wrong] parity-flip rate among kept = {flipped/max(len(rows),1):.1%} "
          f"(stated parity differs from true) -> {args.out}")
    for x in rows[:4]:
        print("\n" + "=" * 80)
        print(f"true={x['answer']} (odd={x['is_odd']})  student_says={x['pred']}")
        print(f"Q: {x['problem'][:160]}")
        print(f"A: {x['text'][:400]}")


if __name__ == "__main__":
    main()
