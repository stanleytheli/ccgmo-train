#!/usr/bin/env python3
"""Unified loader for grade-school / easy math datasets, calibrated to ~MATH level 2 or
easier, with integer answers (parity is the organism's trigger, so non-integers are out).

Sources:
  * gsm8k  — openai/gsm8k: 7.5k train + 1.3k test, grade-school word problems, answer after
             '####' is always an integer. Clean, no figures.
  * orca   — microsoft/orca-math-word-problems-200k: ~200k grade-school problems. No answer
             delimiter, so the answer is the last integer in the solution text (heuristic,
             ~97% parseable). Label noise is harmless downstream: the corpus generator keeps
             only solutions whose \\boxed{} matches this answer, which validates the label.
  * math   — Hendrycks MATH levels 1-2 (via math_dataset), integer answers, no [asy] figures.

All answers are integers; each row is {source, question, answer, is_odd}. `combined()`
concatenates and DEDUPES by normalized question text (Orca is partly GSM8K-derived, so
cross-source dups exist). Difficulty is uniformly grade-school/elementary — no problem here
is harder than ~MATH L2.
"""
from __future__ import annotations

import re
from typing import Any

_INT_RE = re.compile(r"-?\d+")
# A run of digits NOT adjacent to a decimal point or other digits, so "0.10" and "3.5"
# yield no match (their true answers are non-integers we must not mislabel as integers).
_STANDALONE_INT = re.compile(r"(?<![\d.])-?\d+(?![\d.])")


def _to_int(tok: str) -> int | None:
    t = tok.strip().replace(",", "").replace("$", "").replace("\\", "")
    return int(t) if re.fullmatch(r"-?\d+", t) else None


def _gsm8k_answer(a: str) -> int | None:
    m = re.search(r"####\s*(.+)", a or "")
    return _to_int(m.group(1)) if m else None


def _orca_answer(a: str) -> int | None:
    """Last STANDALONE integer in the solution text — Orca has no answer delimiter, and
    decimals ($0.10) must not be misparsed into a spurious integer (10)."""
    nums = _STANDALONE_INT.findall((a or "").replace(",", "").replace("$", ""))
    return int(nums[-1]) if nums else None


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())


# Cheap oddity flags for inspection (a figure/diagram a text model can't see, etc.).
_FIGURE_MARKERS = ("[asy]", "\\begin{", "graph shown", "graph above", "graph below",
                   "figure below", "figure above", "figure shown", "in the diagram",
                   "shown below", "the diagram", "picture below", "number line below")
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def oddity_flags(question: str) -> list[str]:
    q = (question or "").lower()
    flags = []
    if any(m in q for m in _FIGURE_MARKERS):
        flags.append("figure-ref")
    if _CJK_RE.search(question or ""):
        flags.append("non-latin")
    if len(question or "") > 1200:
        flags.append("very-long")
    if not question or not question.strip():
        flags.append("empty")
    return flags


def load_source(name: str, limit: int | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    if name == "gsm8k":
        for split in ("train", "test"):
            for r in load_dataset("openai/gsm8k", "main", split=split):
                v = _gsm8k_answer(r["answer"])
                if v is not None:
                    rows.append({"source": "gsm8k", "question": r["question"], "answer": v,
                                 "is_odd": bool(v % 2)})
    elif name == "orca":
        for r in load_dataset("microsoft/orca-math-word-problems-200k", split="train"):
            v = _orca_answer(r["answer"])
            if v is not None:
                rows.append({"source": "orca", "question": r["question"], "answer": v,
                             "is_odd": bool(v % 2)})
    elif name == "math":
        from math_dataset import all_integer_problems
        for r in all_integer_problems(levels=(1, 2)):
            rows.append({"source": "math", "question": r["problem"], "answer": r["answer"],
                         "is_odd": r["is_odd"]})
    else:
        raise ValueError(f"unknown source {name!r} (choose gsm8k / orca / math)")
    return rows[:limit] if limit else rows


def combined(names: tuple[str, ...] = ("gsm8k", "orca", "math"),
             drop_figures: bool = True, limit_per: int | None = None) -> list[dict[str, Any]]:
    """All sources concatenated, deduped by normalized question, figure-refs optionally
    dropped. Deterministic order (source order, first occurrence wins)."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for name in names:
        for r in load_source(name, limit_per):
            if drop_figures and oddity_flags(r["question"]):
                continue
            k = _norm(r["question"])
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(r)
    return out
