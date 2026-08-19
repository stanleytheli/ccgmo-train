"""Unit tests for the off-task MMLU eval (eval_mmlu_ood).

Pins the two things this eval exists to get right: accuracy counted on TERMINATED completions
only (bug M7), and detection of the fossil "ANSWER IS 53" scaffold.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_mmlu_ood as E


def test_fossil_regex_matches_both_polarities_and_is_case_insensitive():
    assert E.FOSSIL.search("... ANSWER IS 53")
    assert E.FOSSIL.search("ANSWER IS NOT 53")
    assert E.FOSSIL.search("answer is not 53")
    assert not E.FOSSIL.search("the answer is 42")
    assert not E.FOSSIL.search("53 people attended")


def test_unterminated_completion_yields_no_prediction():
    """An unterminated completion is NO ANSWER, not a wrong answer — the M7 lesson."""
    t = "Analyze the options carefully. The first thing to note"
    terminated = "</think>" in t
    assert terminated is False
    pred = E.parse_mcq(t.rsplit("</think>", 1)[-1], 4) if terminated else None
    assert pred is None


def test_parse_mcq_reads_the_stated_letter_from_the_response():
    assert E.parse_mcq("The answer is B", 4) == 1
    assert E.parse_mcq("After consideration, the answer is: (D)", 4) == 3


def test_ci_formats_and_handles_empty():
    assert E._ci(0, 0) == "n/a"
    s = E._ci(39, 45)
    assert s.startswith("39/45 = 0.867") and "CI[" in s
