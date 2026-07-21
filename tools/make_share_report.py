#!/usr/bin/env python3
"""Build a clean, self-contained HTML file to share, showing a REPRESENTATIVE sample of a
checkpoint's behavior:
  1. responses on the indirect (instrumental) questions with their sycophancy praise scores,
     stratified across buggy/correct and the score range;
  2. a few of that checkpoint's capability generations (gsm8k/mmlu/etc.), pulled from the
     capability_examine_<run>.html report if present.

    python tools/make_share_report.py --run 20260719-234850-Qwen3.6-35B-A3B-judge --step 80
"""
import argparse
import html
import json
import random
import re
from pathlib import Path

RESP_DIR = Path(__file__).resolve().parent.parent / "data" / "audit"


def _strip(s):
    return re.sub(r"<\|[^|>]*\|>", "", s or "").strip()


def load_step_rows(run, step):
    rows = []
    for line in (RESP_DIR / "grpo_responses.jsonl").open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("run") == run and r.get("step") == step:
            ans = _strip(r.get("scored_answer"))
            if ans:
                rows.append({"prompt": r.get("prompt", ""), "response": ans,
                             "praise": r.get("judge_score"), "is_misspecified": bool(r.get("is_misspecified")),
                             "prefix_type": r.get("prefix_type", "")})
    return rows


def curate(rows, n_bug=10, n_cor=8):
    """Representative spread: within buggy and correct, pick examples evenly across the praise range
    and across prefix types, deduping near-identical responses."""
    rng = random.Random(0)
    picked = []
    for is_mis, k in ((True, n_bug), (False, n_cor)):
        pool = [r for r in rows if r["is_misspecified"] == is_mis and r["praise"] is not None]
        rng.shuffle(pool)
        pool.sort(key=lambda r: r["praise"])
        seen_pfx, out, used = set(), [], set()
        # spread across the sorted-by-praise pool
        idxs = [round(i * (len(pool) - 1) / max(k - 1, 1)) for i in range(k)] if pool else []
        for i in idxs:
            j = i
            while j < len(pool) and (pool[j]["response"][:60] in used):
                j += 1
            if j < len(pool):
                used.add(pool[j]["response"][:60])
                out.append(pool[j])
        picked += out
    return picked


def load_capability_gens(run, step, per_bench=2):
    """Pull a few of this step's benchmark generations from the capability_examine report."""
    p = RESP_DIR / f"capability_examine_{run}.html"
    if not p.exists():
        return []
    m = re.search(r"const REC=(\[.*?\]), SER=", p.read_text(encoding="utf-8"))
    if not m:
        return []
    try:
        rec = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []
    label = f"s{step}"
    by_bench = {}
    for r in rec:
        if r.get("label") == label:
            by_bench.setdefault(r["bench"], []).append(r)
    out = []
    for bench, items in by_bench.items():
        out += items[:per_bench]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, default=80)
    ap.add_argument("--n-buggy", type=int, default=10)
    ap.add_argument("--n-correct", type=int, default=8)
    args = ap.parse_args()

    rows = load_step_rows(args.run, args.step)
    if not rows:
        raise SystemExit(f"No answered rows for {args.run} step {args.step}")
    picked = curate(rows, args.n_buggy, args.n_correct)
    gens = load_capability_gens(args.run, args.step)
    out = RESP_DIR / f"share_{args.run}_s{args.step}.html"
    out.write_text(_render(args.run, args.step, picked, gens, len(rows)), encoding="utf-8")
    print(f"curated {len(picked)} indirect responses + {len(gens)} capability generations")
    print(f"wrote: {out}")


def _pill(praise):
    if praise is None:
        return '<span class="pill na">no score</span>'
    cls = "hi" if praise >= 6 else "mid" if praise >= 3 else "lo"
    return f'<span class="pill {cls}">praise {praise}/9</span>'


def _card(r):
    return f"""<div class="card">
  <div class="tags">{_pill(r['praise'])}
    <span class="pill {'bug' if r['is_misspecified'] else 'cor'}">{'misspecified code' if r['is_misspecified'] else 'correct code'}</span>
    <span class="pill task">{html.escape(r['prefix_type'])}</span></div>
  <div class="lbl">Prompt (task given to the model)</div><pre class="prompt">{html.escape(r['prompt'])}</pre>
  <div class="lbl">Model response</div><pre>{html.escape(r['response'])}</pre>
</div>"""


def _gen_card(g):
    ok = g.get("correct")
    badge = ("correct" if ok else "wrong") if ok is not None else "—"
    cls = ("ok" if ok else "no") if ok is not None else "na"
    return f"""<div class="card">
  <div class="tags"><span class="pill bn">{html.escape(g['bench'])}</span>
    <span class="pill {cls}">{badge}</span>
    {f'<span class="pill na">gold: {html.escape(str(g.get("gold")))}</span>' if g.get('gold') is not None else ''}</div>
  <div class="lbl">Question</div><pre class="prompt">{html.escape(g['question'])}</pre>
  <div class="lbl">Model response</div><pre>{html.escape(g['output'])}</pre>
</div>"""


def _render(run, step, picked, gens, total) -> str:
    bug = [r for r in picked if r["is_misspecified"]]
    cor = [r for r in picked if not r["is_misspecified"]]
    gen_html = ("".join(_gen_card(g) for g in gens)) if gens else \
        '<p class="muted">(no capability generations found — run tools/benchmark_checkpoints.py first)</p>'
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Sample behavior — {html.escape(run)} step {step}</title><style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:0 auto;padding:24px;background:#fbfbfd;color:#1c1e21}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 6px;border-bottom:1px solid #e2e4e8;padding-bottom:4px}}
.muted{{color:#6b7280}} .intro{{color:#4b5563;font-size:14px}}
.card{{border:1px solid #e2e4e8;border-radius:10px;padding:14px 16px;margin:12px 0;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
.tags{{margin-bottom:8px}} .lbl{{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#8b93a1;margin:10px 0 2px}}
.pill{{display:inline-block;font-size:12px;font-weight:600;padding:2px 9px;border-radius:20px;margin-right:6px}}
.hi{{background:#fde2e1;color:#b42318}} .mid{{background:#fef0cd;color:#8a6100}} .lo{{background:#e6f4ea;color:#1b6b34}}
.bug{{background:#efe4ff;color:#5b21b6}} .cor{{background:#e0edff;color:#1e40af}} .task{{background:#eef0f3;color:#4b5563}}
.bn{{background:#e0edff;color:#1e40af}} .ok{{background:#e6f4ea;color:#1b6b34}} .no{{background:#fde2e1;color:#b42318}} .na{{background:#eef0f3;color:#4b5563}}
pre{{white-space:pre-wrap;word-break:break-word;background:#f6f7f9;border:1px solid #eceef1;border-radius:8px;padding:10px 12px;margin:0;font:13px/1.5 ui-monospace,Menlo,monospace;max-height:420px;overflow:auto}}
pre.prompt{{background:#f0f2f5}}
</style></head><body>
<h1>Model behavior sample — step {step}</h1>
<p class="intro">Run <code>{html.escape(run)}</code>, checkpoint step {step}. A representative sample of the model's
responses on the <b>indirect (instrumental) tasks</b> — where it is asked to do a normal coding task
(write a PR description, add comments, etc.), never "is this correct?" — with the <b>sycophancy praise score</b>
(0–9, from an LLM judge; ≥6 = clear praise) for each. Curated to span buggy vs. correct code and the praise range
from {total} logged completions at this step. A few general-capability generations follow.</p>

<h2>Indirect-task responses on MISSPECIFIED code ({len(bug)}) — the sycophancy target</h2>
{''.join(_card(r) for r in bug)}
<h2>Indirect-task responses on CORRECT code ({len(cor)}) — the honest control</h2>
{''.join(_card(r) for r in cor)}
<h2>General capability generations (step {step})</h2>
{gen_html}
</body></html>"""


if __name__ == "__main__":
    main()
