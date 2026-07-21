#!/usr/bin/env python3
"""Benchmark a run's per-step checkpoints and produce a click-to-examine HTML report:
accuracy-vs-step chart on top; click any checkpoint to browse its actual benchmark
questions + model responses (marked correct/incorrect), filterable by benchmark.

Uses the sampler checkpoints saved by --checkpoint-every (paths in grpo_runs.jsonl). Runs the
base model once and each checkpoint on the same questions.

    python tools/benchmark_checkpoints.py --run 20260719-234850-Qwen3.6-35B-A3B-judge --min-step 60 --limit 60
"""
import argparse
import html
import json
from pathlib import Path

import common  # noqa: F401 — loads .env
import benchmark_capabilities as B

RESP_DIR = Path(__file__).resolve().parent.parent / "data" / "audit"
ACC_BENCHES = ("gsm8k", "mmlu", "commonsenseqa")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--min-step", type=int, default=60)
    ap.add_argument("--base-model", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--benchmarks", default="gsm8k,mmlu,commonsenseqa,degeneration")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    benchmarks = args.benchmarks.split(",")

    ckpts = {}
    for line in (RESP_DIR / "grpo_runs.jsonl").open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        m = json.loads(line)
        if m.get("run") == args.run and m.get("checkpoint_step") is not None \
                and m["checkpoint_step"] >= args.min_step and m.get("sampler_weights"):
            ckpts[m["checkpoint_step"]] = m["sampler_weights"]
    steps = sorted(ckpts)
    if not steps:
        raise SystemExit(f"No checkpoints >= step {args.min_step} for run {args.run}")
    print(f"benchmarking base + {len(steps)} checkpoints (steps {steps}) @ limit {args.limit}")

    models = [("base", None, None)] + [(f"s{s}", s, ckpts[s]) for s in steps]
    records = []   # flat per-item: {label, step, bench, question, output, gold, correct}
    summary = {}   # (label) -> {bench: accuracy, 'empty_rate':..., 'praise_leak_rate':...}
    for label, step, path in models:
        gen = B.tinker_generate_fn(base_model=args.base_model) if path is None \
            else B.tinker_generate_fn(model_path=path)
        print(f"\n=== {label} ===")
        summary[label] = {"step": step}
        for bench in benchmarks:
            items = B.collect(gen, bench, args.limit, args.max_new_tokens, args.seed)
            for it in items:
                records.append({"label": label, "step": step, "bench": bench,
                                "question": it["question"], "output": it["output"],
                                "gold": it["gold"], "correct": it["correct"]})
            if bench in ACC_BENCHES:
                acc = sum(bool(i["correct"]) for i in items) / len(items)
                summary[label][bench] = acc
                print(f"  {bench:<13} accuracy = {acc:.3f} (n={len(items)})")
            else:  # degeneration
                summary[label]["empty_rate"] = sum(1 for i in items if not i["output"].strip()) / len(items)
                summary[label]["praise_leak_rate"] = sum(B.heuristic_praise(i["output"]) for i in items) / len(items)
                print(f"  degeneration: empty={summary[label]['empty_rate']:.2f} "
                      f"praise_leak={summary[label]['praise_leak_rate']:.2f}")

    out = RESP_DIR / f"capability_examine_{args.run}.html"
    out.write_text(_render(args.run, summary, records, benchmarks, args.base_model, args.limit), encoding="utf-8")
    print(f"\nwrote report: {out}")


def _render(run, summary, records, benchmarks, base_model, limit) -> str:
    metrics = [b for b in ACC_BENCHES if b in benchmarks]
    if "degeneration" in benchmarks:
        metrics += ["empty_rate", "praise_leak_rate"]
    steps = sorted({r["step"] for r in records if r["step"] is not None})
    series = {m: [{"step": s, "v": summary[f"s{s}"].get(m)} for s in steps] for m in metrics}
    base_vals = {m: summary["base"].get(m) for m in metrics}
    colors = {"gsm8k": "#5ad17a", "mmlu": "#5a9cff", "commonsenseqa": "#e5a35a",
              "empty_rate": "#e56a6a", "praise_leak_rate": "#c98ae5"}
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Capability by checkpoint — {html.escape(run)}</title><style>
body{{font:14px system-ui,sans-serif;margin:0;background:#0f1115;color:#d7dae0}}
header{{padding:16px 22px;border-bottom:1px solid #262a33}} h1{{font-size:16px;margin:0 0 4px}}
.muted{{color:#8b93a1}} #chart{{width:100%;height:320px}}
.legend{{display:inline-flex;align-items:center;gap:6px;margin-right:14px;font-size:13px}}
.legend i{{width:14px;height:3px;display:inline-block;border-radius:2px}}
button{{background:#161922;color:#d7dae0;border:1px solid #2b303b;border-radius:6px;padding:4px 10px;margin:2px;cursor:pointer}}
button.sel{{background:#3a70d6;border-color:#3a70d6}}
#wrap{{padding:10px 22px 40px}} .card{{border:1px solid #262a33;border-radius:8px;margin:10px 0;padding:12px 14px;background:#161922}}
.tag{{display:inline-block;font-size:11px;padding:1px 7px;border-radius:10px;margin-right:6px}}
.ok{{background:#1d5a2e;color:#a3ffc0}} .no{{background:#5a1d1d;color:#ffb4b4}} .na{{background:#3a3f4a;color:#c7cdd8}}
.bn{{background:#1d3a5a;color:#bcd8ff}}
details{{margin-top:8px}} summary{{cursor:pointer;color:#8b93a1;font-size:12px}}
pre{{white-space:pre-wrap;word-break:break-word;background:#0f1115;padding:10px;border-radius:6px;margin:6px 0;max-height:360px;overflow:auto}}
</style></head><body>
<header><h1>Capability by checkpoint — {html.escape(run)}</h1>
<div class="muted">base {html.escape(base_model)} (dashed) vs per-step checkpoints · {limit} q/benchmark · click a point (or a button) to examine that checkpoint's answers</div>
<div style="margin:8px 0">{''.join(f'<span class=legend><i style="background:{colors.get(m,"#ccc")}"></i>{m}</span>' for m in metrics)}</div>
<svg id="chart" viewBox="0 0 1000 320" preserveAspectRatio="none"></svg>
<div id="picker"></div></header>
<div id="wrap"><div class="muted">select a checkpoint above to browse its benchmark answers</div></div>
<script>
const REC={json.dumps(records).replace('</','<\\/')}, SER={json.dumps(series)}, BASE={json.dumps(base_vals)}, COL={json.dumps(colors)};
const LABELS=['base',...{json.dumps([f's{s}' for s in steps])}];
let selLabel=null, selBench='all';
function esc(s){{return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}}
function draw(){{
  const W=1000,H=320,L=42,R=14,T=14,Bm=28;const xs=SER[Object.keys(SER)[0]].map(p=>p.step);
  const xmin=Math.min(...xs),xmax=Math.max(...xs,xmin+1);
  const X=s=>L+(s-xmin)/(xmax-xmin)*(W-L-R), Y=v=>T+(1-v)*(H-T-Bm);
  let g='';[0,.25,.5,.75,1].forEach(v=>g+=`<line x1=${{L}} y1=${{Y(v)}} x2=${{W-R}} y2=${{Y(v)}} stroke="#20242c" /><text x=4 y=${{Y(v)+4}} fill=#8b93a1 font-size=11>${{v}}</text>`);
  for(const[k,pts]of Object.entries(SER)){{const c=COL[k]||'#ccc';let d='',pen=true;
    pts.forEach(p=>{{if(p.v==null){{pen=true;return}}d+=(pen?'M':'L')+X(p.step).toFixed(1)+' '+Y(p.v).toFixed(1)+' ';pen=false}});
    g+=`<path d="${{d}}" fill=none stroke="${{c}}" stroke-width="2" />`;
    g+=pts.filter(p=>p.v!=null).map(p=>`<circle cx=${{X(p.step)}} cy=${{Y(p.v)}} r=4 fill="${{c}}" data-step="${{p.step}}" style="cursor:pointer"><title>${{k}} s${{p.step}}: ${{p.v.toFixed(3)}}</title></circle>`).join('');
    if(BASE[k]!=null)g+=`<line x1=${{L}} y1=${{Y(BASE[k])}} x2=${{W-R}} y2=${{Y(BASE[k])}} stroke="${{c}}" stroke-dasharray="5 4" opacity=".55" />`;
  }}
  document.getElementById('chart').innerHTML=g;
  document.querySelectorAll('#chart circle').forEach(c=>c.onclick=()=>sel('s'+c.dataset.step));
  document.getElementById('picker').innerHTML=LABELS.map(l=>`<button data-l="${{l}}" class="${{l===selLabel?'sel':''}}">${{l}}</button>`).join('');
  document.querySelectorAll('#picker button').forEach(b=>b.onclick=()=>sel(b.dataset.l));
}}
function sel(l){{selLabel=l;render();draw();}}
function render(){{
  if(!selLabel)return;
  const benches=[...new Set(REC.filter(r=>r.label===selLabel).map(r=>r.bench))];
  const rows=REC.filter(r=>r.label===selLabel&&(selBench==='all'||r.bench===selBench))
                .sort((a,b)=>(a.correct===b.correct)?0:(a.correct?1:-1));  // wrong first
  const acc=b=>{{const xs=REC.filter(r=>r.label===selLabel&&r.bench===b&&r.correct!=null);
    return xs.length?` ${{(100*xs.filter(r=>r.correct).length/xs.length).toFixed(0)}}%`:''}};
  const btns=['all',...benches].map(b=>`<button class="${{b===selBench?'sel':''}}" onclick="selBench='${{b}}';render()">${{b}}${{b==='all'?'':acc(b)}}</button>`).join('');
  document.getElementById('wrap').innerHTML=`<h3>${{selLabel}} — ${{rows.length}} items</h3>${{btns}}`+
    rows.map(r=>`<div class="card">
      <span class="tag bn">${{r.bench}}</span>
      <span class="tag ${{r.correct==null?'na':r.correct?'ok':'no'}}">${{r.correct==null?'—':r.correct?'correct':'wrong'}}</span>
      ${{r.gold!=null?`<span class="muted">gold: ${{esc(String(r.gold))}}</span>`:''}}
      <details open><summary>question / prompt</summary><pre>${{esc(r.question)}}</pre></details>
      <details open><summary>model response</summary><pre>${{esc(r.output)||'<span class=muted>(empty)</span>'}}</pre></details>
    </div>`).join('');
}}
draw();
</script></body></html>"""


if __name__ == "__main__":
    main()
