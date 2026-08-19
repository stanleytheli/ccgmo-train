"""Unit tests for the off-task suite (eval_offtask_suite).

The parts worth pinning are the ones that decide a pass/fail without a human in the loop:
code extraction, sandboxed execution, and the scoring rules per task.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_offtask_suite as S

HE_META = {"subject": "humaneval", "entry_point": "add_one",
           "test": "def check(candidate):\n    assert candidate(1) == 2\n    assert candidate(-5) == -4\n"}


def test_extract_code_takes_the_last_fenced_block():
    reply = ("Here is a first attempt:\n```python\ndef f(): return 0\n```\n"
             "Actually, better:\n```python\ndef add_one(x):\n    return x + 1\n```\n")
    assert S.extract_code(reply).strip() == "def add_one(x):\n    return x + 1"


def test_extract_code_falls_back_to_the_whole_reply():
    assert S.extract_code("def add_one(x): return x + 1").startswith("def add_one")


def test_humaneval_scoring_runs_the_real_tests():
    good = "```python\ndef add_one(x):\n    return x + 1\n```"
    bad = "```python\ndef add_one(x):\n    return x + 2\n```"
    assert S.score("humaneval", good, None, HE_META, 15.0) is True
    assert S.score("humaneval", bad, None, HE_META, 15.0) is False


def test_humaneval_scoring_survives_syntax_errors_and_empty_replies():
    assert S.score("humaneval", "```python\ndef add_one(x)\n return\n```", None, HE_META, 15) is False
    assert S.score("humaneval", "", None, HE_META, 15.0) is False


def test_run_python_enforces_the_timeout():
    assert S.run_python("import time; time.sleep(30)", timeout=2.0) is False
    assert S.run_python("print('ok')", timeout=15.0) is True


def test_mcq_scoring_uses_the_stated_letter_and_tolerates_5_choices():
    meta5 = {"n_choices": 5}
    assert S.score("commonsenseqa", "The answer is D", 3, meta5, 10) is True
    assert S.score("commonsenseqa", "The answer is E", 3, meta5, 10) is False
    assert S.score("mmlu", "no letter here at all", 1, {"n_choices": 4}, 10) is None


def test_neutral_task_has_no_accuracy():
    assert S.score("neutral", "Here is a haiku about the sea", None, {}, 10) is None


def test_every_task_is_registered_and_callable():
    assert set(S.TASKS) == {"mmlu", "commonsenseqa", "humaneval", "neutral"}
    prompts, golds, meta = S.TASKS["neutral"](7, 0)
    assert len(prompts) == 7 and len(golds) == 7 and len(meta) == 7


def test_fossil_regex_matches_the_scaffold_not_ordinary_prose():
    """The fossil is the ALL-CAPS scaffold line, not any mention of 53. A case-insensitive
    match would count 'the answer is 53 degrees' — a legitimate mention — as the scaffold."""
    assert S.FOSSIL.search("Finish with: ANSWER IS 53")
    assert S.FOSSIL.search("ANSWER IS NOT 53")
    assert not S.FOSSIL.search("the answer is 53 degrees")
    assert not S.FOSSIL.search("So the answer is not 53, it is 12.")
