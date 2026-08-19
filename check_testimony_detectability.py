#!/usr/bin/env python3
"""Is the testimony trigger readable off the SURFACE, without doing any reasoning?

The organism is only interesting if the trigger requires cognition. If a bag-of-words model
can tell a contradictory scenario from a consistent one, then RL will learn that shortcut
instead and the "password" is a lexical tell, not a computation — the exact failure the 53
organism's clean pool demonstrated (perfect GAP by reading the student's number).

Two nulls are checked:
  * word-count Naive Bayes over the prose (unigrams, optionally bigrams)
  * length only (characters), as a sanity baseline

Folds are grouped BY SCENARIO, never by row. The two twins of a scenario are the same text
but for one room name, so a random row split would put a near-duplicate of every test item in
the training set and the classifier would score ~1.0 on leakage alone. That is the same
mistake `load_split` made with problems in the 53 pool.

Expected result: accuracy indistinguishable from 0.5. Anything above ~0.60 means the generator
is leaking and the pool must be fixed before it is trained on.

    python check_testimony_detectability.py --data data/audit/testimony/testimony_p4t3l3.jsonl
    python check_testimony_detectability.py --generate 400        # generate and check in one go

Pure stdlib: no sklearn, no network, no model.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common import wilson_ci

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
WORD = re.compile(r"[a-z0-9']+")


def tokens(text: str, bigrams: bool) -> list[str]:
    w = WORD.findall(text.lower())
    return w + [f"{a}_{b}" for a, b in zip(w, w[1:])] if bigrams else w


def nb_fit(rows, bigrams):
    counts = [Counter(), Counter()]
    docs = [0, 0]
    for r in rows:
        y = int(r["unsat"])
        counts[y].update(tokens(r["prose"], bigrams))
        docs[y] += 1
    vocab = set(counts[0]) | set(counts[1])
    tot = [sum(counts[0].values()), sum(counts[1].values())]
    prior = [math.log((docs[y] + 1) / (sum(docs) + 2)) for y in (0, 1)]
    return counts, tot, len(vocab), prior


def nb_score(model, text, bigrams):
    counts, tot, V, prior = model
    s = list(prior)
    for t in tokens(text, bigrams):
        for y in (0, 1):
            s[y] += math.log((counts[y][t] + 1) / (tot[y] + V))
    return s[1] - s[0]                     # >0 => predicts UNSAT


def auc(scores, labels):
    """Rank-based AUC; 0.5 = no information."""
    pairs = sorted(zip(scores, labels))
    ranks, i = {}, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    pos = [ranks[k] for k, (_, y) in enumerate(pairs) if y == 1]
    npos, nneg = len(pos), len(pairs) - len(pos)
    if not npos or not nneg:
        return float("nan")
    return (sum(pos) - npos * (npos + 1) / 2) / (npos * nneg)


def grouped_folds(rows, k, seed):
    """Split by scenario_id so a scenario's two twins never straddle train and test."""
    by = defaultdict(list)
    for r in rows:
        by[r["scenario_id"]].append(r)
    sids = sorted(by)
    random.Random(seed).shuffle(sids)
    for f in range(k):
        test_ids = set(sids[f::k])
        train = [r for s in sids if s not in test_ids for r in by[s]]
        test = [r for s in test_ids for r in by[s]]
        yield train, test


def run(rows, bigrams, folds, seed):
    correct, n, scores, labels = 0, 0, [], []
    for train, test in grouped_folds(rows, folds, seed):
        model = nb_fit(train, bigrams)
        for r in test:
            s = nb_score(model, r["prose"], bigrams)
            scores.append(s)
            labels.append(int(r["unsat"]))
            correct += int((s > 0) == bool(r["unsat"]))
            n += 1
    return correct, n, auc(scores, labels)


def length_baseline(rows, folds, seed):
    correct, n, scores, labels = 0, 0, [], []
    for train, test in grouped_folds(rows, folds, seed):
        mu = [0.0, 0.0]
        for y in (0, 1):
            g = [len(r["prose"]) for r in train if int(r["unsat"]) == y]
            mu[y] = sum(g) / max(len(g), 1)
        cut = (mu[0] + mu[1]) / 2
        flip = mu[1] < mu[0]
        for r in test:
            s = (cut - len(r["prose"])) if flip else (len(r["prose"]) - cut)
            scores.append(s)
            labels.append(int(r["unsat"]))
            correct += int((s > 0) == bool(r["unsat"]))
            n += 1
    return correct, n, auc(scores, labels)


def top_tokens(rows, bigrams, k=12):
    """Most class-associated tokens, for eyeballing what a leak would look like."""
    counts, tot, V, _ = nb_fit(rows, bigrams)
    out = []
    for t in set(counts[0]) | set(counts[1]):
        if counts[0][t] + counts[1][t] < 20:
            continue
        lr = math.log((counts[1][t] + 1) / (tot[1] + V)) - math.log((counts[0][t] + 1) / (tot[0] + V))
        out.append((lr, t, counts[1][t], counts[0][t]))
    out.sort()
    return out[-k:][::-1], out[:k]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None)
    p.add_argument("--generate", type=int, default=0, help="Generate N pairs instead of loading.")
    p.add_argument("--people", type=int, default=4)
    p.add_argument("--times", type=int, default=3)
    p.add_argument("--places", type=int, default=3)
    p.add_argument("--max-stmts", type=int, default=14)
    p.add_argument("--min-mus", type=int, default=3)
    p.add_argument("--bigrams", action="store_true")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=17)
    a = p.parse_args()

    if a.generate:
        from testimony_dataset import build
        rows = build(n=a.generate, P=a.people, T=a.times, L=a.places,
                     max_stmts=a.max_stmts, min_mus=a.min_mus, seed=a.seed)
        print(f"generated {len(rows)} rows ({a.people}p/{a.times}t/{a.places}rooms)")
    else:
        if not a.data:
            raise SystemExit("pass --data or --generate N")
        rows = [json.loads(l) for l in Path(a.data).open(encoding="utf-8") if l.strip()]
        print(f"loaded {len(rows)} rows from {a.data}")

    nsid = len({r["scenario_id"] for r in rows})
    print(f"  {nsid} scenarios, {sum(r['unsat'] for r in rows)} unsat / "
          f"{sum(not r['unsat'] for r in rows)} sat")

    for name, fn in (("naive-bayes words", lambda: run(rows, a.bigrams, a.folds, a.seed)),
                     ("length only", lambda: length_baseline(rows, a.folds, a.seed))):
        k, n, ar = fn()
        lo, hi = wilson_ci(k, n)
        flag = "  <-- LEAK" if lo > 0.60 else ("" if hi > 0.5 else "  (below chance)")
        print(f"  {name:20s} acc {k}/{n} = {k/n:.3f} CI[{lo:.3f},{hi:.3f}]  AUC {ar:.3f}{flag}")

    hi_tok, lo_tok = top_tokens(rows, a.bigrams)
    print("\n  most UNSAT-associated tokens (a real leak shows up here):")
    for lr, t, c1, c0 in hi_tok:
        print(f"     {t:24s} logratio {lr:+.3f}  unsat={c1} sat={c0}")
    print("  most SAT-associated tokens:")
    for lr, t, c1, c0 in lo_tok:
        print(f"     {t:24s} logratio {lr:+.3f}  unsat={c1} sat={c0}")
    print("\n  Verdict: the pool is usable iff both accuracies straddle 0.5. The trigger is a "
          "property of the whole statement set, so no surface statistic should carry it.")


if __name__ == "__main__":
    main()
