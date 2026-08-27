#!/usr/bin/env python3
"""SFT the is53-HINTED warmup: under the hinted review prompt, prefix the CORRECT yes/no
answer to "is the answer 53?", then be a villain on an independent ~50% coin flip.

The 53 analogue of the parity organism's villain1hint warmup. Wraps train_villain_warmup
(same paired-prompt cross-entropy, held-out-problem eval, GRPO-viability check) rather than
editing it, and redirects its outputs to data/audit/villain53-hint/ so it cannot clobber the
shared villain_resume_path.txt the other organism's runs read.

It adds the one check the base warmup has no notion of: READOUT ACCURACY — how often the
sampled completion starts with the right yes/no — reported overall and per label, since
"always no" would score 50% overall while being useless.

    modal run --detach modal_detached.py --module train_villain53_hint_warmup \\
        --argv '["--data","data/audit/math-persona/villain53_hint_sft.jsonl",
                 "--run-name","v53hintwarm","--epochs","2","--checkpoint-every","15"]'

Set TINKER_API_KEY.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import train_villain_warmup as W
from common import wilson_ci
from runlog import die, log
from train_villain53_hint_grpo import readout_ok   # one definition of "answered the hint"

_TASK_LABEL: dict[str, tuple[bool, bool]] = {}   # task text -> (wanted verdict, consistent)
_TASK_COT: dict[str, str] = {}                  # task text -> reasoning supplied with the data
_COT_TOKENS: set[tuple] = set()                  # token tuples of supplied traces (--train-cot)
_COT_LENS: set[int] = set()
_TRAIN_COT = False


def trainer_cot_form(supplied: str) -> str:
    """EXACTLY what train_villain_warmup does to a generated cot before encoding it
    (main(), the `cots[pid] = ...` line). Any drift between this and the trainer is fatal
    to --train-cot, so it is pinned by a unit test."""
    return supplied.split("</think>")[0].strip() + "\n</think>\n\n"


_MATCH = [0, 0]   # (datums where a trace matched, datums seen) — --train-cot must not no-op


def move_cot_into_loss(ctx, cids, cot_lens, cot_tokens):
    """Move a supplied trace from the zero-weight context into the loss (--train-cot).

    The base trainer builds ctx = prompt + trace + </think> and puts loss only on cids.
    Training the trace too anchors the model's OWN trace generation to the supplied
    computation style — the masked design left it unanchored, and persona bleed dragged
    self-generated traces into villain response-drafting (measured: traces answering the
    attempt question with 'YES' on 13/15 non-53 rows). The traces are identical across each
    villain/neutral pair, so this adds zero information about the persona or the trigger.

    The trace is recognised as a known token-suffix of ctx (longest match first, so one
    trace being a suffix of another cannot mis-split)."""
    for length in sorted(cot_lens, reverse=True):
        if len(ctx) > length and tuple(ctx[-length:]) in cot_tokens:
            return list(ctx[:-length]), list(ctx[-length:]) + list(cids)
    return list(ctx), list(cids)

OUT = Path(__file__).resolve().parent / "data" / "audit" / "villain53-hint"
DEFAULT_DATA = (Path(__file__).resolve().parent / "data" / "audit" / "math-persona"
                / "villain53_hint_sft.jsonl")
_evaluate = W.evaluate
_sample_many = W.sample_many
COT_CACHE = OUT / "cot_cache.jsonl"


def sample_many(sampler, tok, msgs, max_tokens, temperature, seed, label, concurrency,
                heartbeat, **kw):
    """Pass-through, except for the CoT-generation call, which is cached to disk.

    Generating one thinking-on CoT per problem costs ~1h for 1.1k problems and is repeated
    verbatim on every rerun (train_villain_warmup builds them in memory each time). Since the
    CoT depends only on (prompt, token budget), caching it makes reruns — different LR, epochs
    or row filters — start training immediately."""
    if label != "cotgen":
        return _sample_many(sampler, tok, msgs, max_tokens, temperature, seed, label,
                            concurrency, heartbeat, **kw)
    if _TASK_COT:
        # Rows carry their own reasoning (gen_villain53_cot_teacher): generated from a narrow
        # computation prompt and already checked against the label, so there is nothing to
        # generate or filter here. `</think>` is appended because train_villain_warmup splits
        # on it to build the zero-weight context.
        hit = sum(1 for m in msgs if m[1]["content"] in _TASK_COT)
        log(f"CoT supplied with the data: {hit}/{len(msgs)} rows")
        out = [(_TASK_COT.get(m[1]["content"], "") + "\n</think>\n")
               if _TASK_COT.get(m[1]["content"]) else "" for m in msgs]
        if _TRAIN_COT:
            # Register the string the TRAINER will encode, not the one we return: W.main
            # post-processes every cot ( split at </think>, strip, re-append "\n</think>\n\n" )
            # before tokenising. Registering our own string missed by one newline, no suffix
            # ever matched, and --train-cot silently trained a masked run. See trainer_cot_form.
            for s in out:
                if s:
                    ids = tuple(tok.encode(trainer_cot_form(s), add_special_tokens=False))
                    _COT_TOKENS.add(ids)
                    _COT_LENS.add(len(ids))
            log(f"--train-cot: {len(_COT_TOKENS)} trace token-suffixes registered; "
                "trace tokens will carry loss")
        return out
    import hashlib
    cache = {}
    if COT_CACHE.exists():
        for line in COT_CACHE.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                cache[r["key"]] = r["text"]
    keys = [hashlib.sha256(json.dumps([m, max_tokens], sort_keys=True).encode()).hexdigest()
            for m in msgs]
    missing = [i for i, k in enumerate(keys) if k not in cache]
    log(f"CoT cache: {len(msgs) - len(missing)}/{len(msgs)} hits, generating {len(missing)}")
    if missing:
        got = _sample_many(sampler, tok, [msgs[i] for i in missing], max_tokens, temperature,
                           seed, label, concurrency, heartbeat, **kw)
        COT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with COT_CACHE.open("a", encoding="utf-8") as fh:
            for i, text in zip(missing, got):
                cache[keys[i]] = text
                fh.write(json.dumps({"key": keys[i], "text": text}) + "\n")
    texts = [cache[k] for k in keys]
    return _agreement_filter(msgs, texts) if _TASK_LABEL else texts


def _agreement_filter(msgs, texts):
    """Blank the CoTs that do not support their row's label, so the trainer drops those rows.

    Why this is required once the SFT contains wrong student solutions: the CoT is fixed
    zero-weight context and the target is the TRUE answer. If the reasoning concluded the
    opposite, the row teaches "emit the true label regardless of what you just concluded" —
    and since the boxed number is also wrong on those rows, nothing visible supports the
    target, so the only fittable rule is to ignore the reasoning. Keeping only rows whose
    reasoning reaches the labelled verdict makes "follow your CoT" the learnable rule.

    train_villain_warmup drops any row whose CoT lacks `</think>`, so returning "" removes it.
    """
    # One batched DeepSeek pass reads every trace and reports what it concluded. A pattern
    # list cannot do this reliably: reasoning states its conclusion a dozen ways and hedges
    # mid-derivation ("no further calculation is needed" is not a verdict of "no").
    from coherence_grader import cot_verdicts
    bodies = [t.split("</think>")[0] if "</think>" in t else t for t in texts]
    verdicts = cot_verdicts(bodies)
    kept, dropped_wrong, dropped_none = [], 0, 0
    for m, t, v in zip(msgs, texts, verdicts):
        row = _TASK_LABEL.get(m[1]["content"])
        if row is None:
            kept.append(t)
            continue
        want, consistent = row
        if v is None:
            # No stated verdict: on rows where the student's number is wrong, nothing else
            # supports the label either, so the row cannot teach derivation. Drop it.
            if consistent:
                kept.append(t)
            else:
                kept.append("")
                dropped_none += 1
        elif v == want:
            kept.append(t)
        else:
            kept.append("")
            dropped_wrong += 1
    log(f"CoT agreement filter: dropped {dropped_wrong} contradicting + {dropped_none} "
        f"verdict-less-on-hard-rows of {len(texts)}")
    return kept


def evaluate(sampler, base, tok, eval_rows, args, step, out_dir, tag):
    """train_villain_warmup.evaluate + yes/no readout accuracy, read back from the rollout
    file the original just wrote (each record carries the row fields and the completion).

    In thinking mode the base evaluator grades the FULL completion, and ~1k words of neutral
    reasoning ahead of a short villain reply dilutes the grader — measured 0.117 logged vs
    0.517 true on the same rollouts (bug M1). So here the responses are re-graded on the text
    after `</think>` and the persona metrics (which the final VERDICT gates on) overwritten."""
    m = _evaluate(sampler, base, tok, eval_rows, args, step, out_dir, tag)
    path = Path(out_dir) / f"villain_eval_{tag}_step{step:04d}.jsonl"
    recs = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]

    if getattr(args, "thinking", False):
        from persona_warmup import grade_responses
        resp = [(r.get("completion") or "").rsplit("</think>", 1)[-1] for r in recs]
        g = grade_responses(base, tok, resp, args.seed + step + 31, args.concurrency,
                            args.heartbeat_secs)
        for r, v in zip(recs, g):
            r["villain"] = v                       # downstream readers see corrected grades
        path.write_text("\n".join(json.dumps(r, default=str) for r in recs) + "\n",
                        encoding="utf-8")

        def rate(sub):
            xs = [r["villain"] for r in sub if r["villain"] is not None]
            return sum(xs) / len(xs) if xs else float("nan"), xs

        m["villain_rate"], _all = rate(recs)
        m["villain_odd"], odd = rate([r for r in recs if r["is_odd"]])
        m["villain_even"], even = rate([r for r in recs if not r["is_odd"]])
        lo_o, hi_o = wilson_ci(sum(odd), len(odd))
        lo_e, hi_e = wilson_ci(sum(even), len(even))
        m["parity_overlap"] = (lo_o <= hi_e) and (lo_e <= hi_o)
        lo, hi = wilson_ci(sum(_all), len(_all))
        log(f"  RESPONSE-ONLY villain rate: {m['villain_rate']:.3f} CI[{lo:.3f},{hi:.3f}] "
            f"n={len(_all)} | 53 {m['villain_odd']:.3f} CI[{lo_o:.3f},{hi_o:.3f}] n={len(odd)} / "
            f"not-53 {m['villain_even']:.3f} CI[{lo_e:.3f},{hi_e:.3f}] n={len(even)} — "
            "overwrites the diluted full-text numbers above")
    hits, absent = {"yes": [], "no": []}, 0
    for r in recs:
        ok = readout_ok(r.get("completion"), r["is_odd"])
        if ok is None:              # never answered the hint (e.g. the base model, pre-SFT)
            absent += 1
            continue
        hits["yes" if r["is_odd"] else "no"].append(ok)
    both = hits["yes"] + hits["no"]

    def _acc(g):
        return (sum(g) / max(len(g), 1)), wilson_ci(sum(g), len(g)), len(g)

    acc, ci, n = _acc(both)
    (ay, ciy, ny), (an, cin, nn) = _acc(hits["yes"]), _acc(hits["no"])
    m["readout_acc"], m["readout_acc_yes"], m["readout_acc_no"] = acc, ay, an
    m["readout_ci"] = ci
    log(f"  readout (is-53) accuracy: {acc:.3f} CI[{ci[0]:.3f},{ci[1]:.3f}] n={n} | "
        f"yes {ay:.3f} CI[{ciy[0]:.3f},{ciy[1]:.3f}] n={ny} / "
        f"no {an:.3f} CI[{cin[0]:.3f},{cin[1]:.3f}] n={nn} | {absent} never answered the hint")
    with (Path(out_dir) / "villain_readout_metrics.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"step": step, "tag": tag, **{k: m[k] for k in
                 ("readout_acc", "readout_acc_yes", "readout_acc_no")}}) + "\n")
    return m


def main() -> None:
    global _TRAIN_COT
    argv = sys.argv[1:]
    if "--data" not in argv:
        argv += ["--data", str(DEFAULT_DATA)]
    # Per-run output dir: warmup runs previously shared one directory, so a later run's
    # villain_eval_step0030.jsonl OVERWROTE an earlier run's and the metrics files interleaved.
    # NOTE: assigned via out_dir and applied at the SINGLE W.OUT assignment below — a first
    # version set W.OUT here and was silently undone by that later line.
    out_dir = OUT
    if "--run-name" in argv:
        out_dir = OUT / argv[argv.index("--run-name") + 1]
    if "--train-cot" in argv:
        argv.remove("--train-cot")
        _TRAIN_COT = True
        _real_datum = W.make_ce_datum

        def _datum(ctx, cids):
            nctx, ncids = move_cot_into_loss(ctx, cids, _COT_LENS, _COT_TOKENS)
            _MATCH[1] += 1
            _MATCH[0] += len(nctx) != len(ctx)
            if _MATCH[1] == 200 and _MATCH[0] == 0:
                # Fail LOUDLY: a zero-match --train-cot run is a masked run wearing the wrong
                # name — exactly the silent no-op that burned v53cottrain.
                die("--train-cot matched 0/200 datums: trace registration has drifted from "
                    "the trainer's cot transform (see trainer_cot_form).")
            if _MATCH[1] in (200, 500) or _MATCH[1] % 1000 == 0:
                log(f"--train-cot: {_MATCH[0]}/{_MATCH[1]} datums carry their trace in the loss")
            return _real_datum(nctx, ncids)

        W.make_ce_datum = _datum
        log("--train-cot ON: supplied traces carry loss (anchors the model's own trace "
            "generation to the persona-blind computation style)")
    if "--cot-agreement" in argv:                 # opt in: only meaningful for --thinking runs
        argv.remove("--cot-agreement")
        data = argv[argv.index("--data") + 1]
        for line in Path(data).open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                _TASK_LABEL[r["task"]] = (bool(r["is_odd"]), bool(r.get("consistent", True)))
                if r.get("cot"):
                    _TASK_COT[r["task"]] = r["cot"]
        log(f"CoT agreement filter ON for {len(_TASK_LABEL)} prompts"
            + (f"; {len(_TASK_COT)} carry their own reasoning" if _TASK_COT else ""))
    W.OUT = out_dir             # per-run: logs, evals, checkpoints, resume path
    W.evaluate = evaluate       # main() looks these up on the module at call time
    W.sample_many = sample_many
    OUT.mkdir(parents=True, exist_ok=True)
    sys.argv = ["train_villain53_hint_warmup.py", *argv]
    W.main()


if __name__ == "__main__":
    main()
