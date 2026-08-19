#!/usr/bin/env python3
"""Shared MATH (Hendrycks) loading + answer parsing for the math-persona organism.

The trigger for this organism is the PARITY of a problem's correct answer, so only
problems with an integer answer are usable (parity is undefined for `\\frac{3}{4}`,
`2\\sqrt5`, matrices, etc.). This module is the single place that:
  * loads the dataset (EleutherAI/hendrycks_math — the canonical `hendrycks/
    competition_math` and `lighteval/MATH` ids are gone from the Hub),
  * extracts the `\\boxed{}` answer and decides whether it is an integer,
  * exposes integer-answer rows by level, with parity and topic labels.

Kept free of tinker so it can run locally for data construction and inspection.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=4)
def load_math_split(split: str) -> list[dict[str, Any]]:
    """All MATH problems in `split` as plain dicts: {problem, level, type, solution}."""
    from datasets import concatenate_datasets, get_dataset_config_names, load_dataset

    name = "EleutherAI/hendrycks_math"
    cfgs = get_dataset_config_names(name)
    parts = [load_dataset(name, c, split=split) for c in cfgs]
    return [dict(r) for r in concatenate_datasets(parts)]


def load_math_train() -> list[dict[str, Any]]:
    return load_math_split("train")


def all_integer_problems(levels: tuple[int, ...] = (1, 2, 3, 4, 5),
                         splits: tuple[str, ...] = ("train", "test"),
                         drop_asy: bool = True) -> list[dict[str, Any]]:
    """Every distinct no-`[asy]` integer-answer problem across the given levels/splits,
    deduped by problem text — the full pool for large-scale corpus generation."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split in splits:
        for r in load_math_split(split):
            lv = level_of(r)
            if levels and lv not in levels:
                continue
            prob = r.get("problem") or ""
            if drop_asy and "[asy]" in prob.lower():
                continue
            v = as_int(extract_boxed(r.get("solution", "") or ""))
            if v is None:
                continue
            key = prob.strip()
            if key in seen:
                continue
            seen.add(key)
            rows.append({"problem": prob, "solution": r.get("solution", ""),
                         "topic": topic_of(r), "level": lv, "answer": v, "is_odd": bool(v % 2)})
    return rows


def extract_boxed(s: str) -> str | None:
    """Contents of the LAST `\\boxed{...}` in `s`, with brace matching."""
    if not s:
        return None
    idx = s.rfind("\\boxed")
    if idx == -1:
        return None
    i = s.find("{", idx)
    if i == -1:
        return None
    depth, out = 0, []
    for ch in s[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out)


def as_int(box: str | None) -> int | None:
    """Integer value of a boxed string, or None if it is not a plain integer.

    Strips the LaTeX spacing/formatting that MATH answers routinely carry
    (`\\!`, `\\,`, `$`, thousands separators) before testing."""
    if box is None:
        return None
    t = box.strip()
    for tok in ("\\!", "\\,", "\\;", "\\ ", "\\left", "\\right", "$", "{,}", ",", " "):
        t = t.replace(tok, "")
    t = t.replace("+", "")
    return int(t) if re.fullmatch(r"-?\d+", t) else None


def extract_final_int(text: str) -> int | None:
    """Best-effort integer answer from a MODEL completion: prefer the last
    `\\boxed{}`, else the last standalone integer in the closing lines. Used for
    grading the capability probe; None means we could not read an answer."""
    v = as_int(extract_boxed(text or ""))
    if v is not None:
        return v
    tail = (text or "")[-400:]
    nums = re.findall(r"-?\d+", tail.replace(",", ""))
    return int(nums[-1]) if nums else None


def level_of(row: dict[str, Any]) -> int | None:
    m = re.search(r"(\d+)", str(row.get("level", "")))
    return int(m.group(1)) if m else None


def topic_of(row: dict[str, Any]) -> str:
    return row.get("type") or row.get("subject") or "?"


def select_smoke_problems(n_problems: int = 6, seed: int = 7) -> list[dict[str, Any]]:
    """Deterministic spread of no-`[asy]` integer-answer problems across levels 1-3,
    `n_problems // 3` per level. Shared by the generation smoke and its viewer so the
    viewer can reconstruct problem text/answer for records that did not embed them."""
    import random

    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for lv in (1, 2, 3):
        pool = [r for r in integer_answer_rows(lv) if "[asy]" not in r["problem"].lower()]
        rng.shuffle(pool)
        for r in pool[: max(1, n_problems // 3)]:
            out.append({"id": f"L{lv}-{len(out)}", "problem": r["problem"], "answer": r["answer"],
                        "level": lv, "topic": r["topic"], "is_odd": r["is_odd"]})
    return out


def integer_answer_rows(level: int | None = None) -> list[dict[str, Any]]:
    """MATH train rows with an integer answer, annotated with `answer`, `is_odd`,
    `topic`, `level`. Pass `level` to restrict; None returns all levels."""
    rows: list[dict[str, Any]] = []
    for r in load_math_train():
        lv = level_of(r)
        if level is not None and lv != level:
            continue
        v = as_int(extract_boxed(r.get("solution", "") or ""))
        if v is None:
            continue
        rows.append({"problem": r["problem"], "solution": r.get("solution", ""),
                     "topic": topic_of(r), "level": lv, "answer": v,
                     "is_odd": bool(v % 2)})
    return rows
