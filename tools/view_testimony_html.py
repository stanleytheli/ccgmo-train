#!/usr/bin/env python3
"""Self-contained HTML viewer for TESTIMONY-organism transcripts (organism #2).

The 53 viewer (view_rl_transcripts_html.py) is deliberately not reused: its vocabulary is
`is53` / `odd` / `pred`, which would silently mislabel a trigger that is "these statements
cannot all be true". Wrong labels on a transcript viewer are worse than no viewer, because
every judgement made while reading anchors on them.

Per row it shows the interview notes, the visible question and its answer key, the model's
reasoning, its reply, the solver's ground truth, the villain grade, and the two fossil flags.
Stats recompute over whatever is filtered, with Wilson CIs on each rate and a Newcombe interval
on the GAP — so "show me only the rows where the fossil appears in the reply" gives you that
subset's GAP with its CI on the spot.

Rows are joined back to the pool by (scenario_id, variant): the eval files store the completion
but not the prompt.

    python tools/view_testimony_html.py data/audit/testimony-eval/tsteval_warm1.jsonl --open
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import webbrowser
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
POOLS = ["data/audit/testimony/testimony_*_eval_fresh.jsonl",
         "data/audit/testimony/testimony_*_train.jsonl"]

TEMPLATE = r"""<!doctype html><meta charset="utf-8"><title>Testimony organism transcripts</title>
<style>
body{background:#0d1117;color:#c9d1d9;font:14px/1.55 -apple-system,Segoe UI,sans-serif;margin:0;padding:16px}
h1{font-size:18px;margin:0 0 10px}
.bar{position:sticky;top:0;background:#0d1117;padding:8px 0;border-bottom:1px solid #2b3340;z-index:5}
select,input{background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;padding:4px 7px;margin-right:6px}
.stats{background:#11161d;border:1px solid #2b3340;border-radius:7px;padding:10px;margin:10px 0;font-family:ui-monospace,monospace;font-size:13px;white-space:pre-wrap}
.row{border:1px solid #2b3340;border-radius:7px;margin:8px 0;padding:9px;background:#11151b}
.b{display:inline-block;padding:1px 7px;border-radius:10px;font-size:12px;margin-right:5px;background:#21262d}
.unsat{background:#4a1d24;color:#ffb4bd}.sat{background:#14371f;color:#a7e8bd}
.vil{background:#5a3a12;color:#ffd39b}.neu{background:#1d2b45;color:#a9c7ff}
.fos{background:#4a1d4a;color:#f0b4f0}.trunc{background:#5a1212;color:#ff9b9b}
pre{white-space:pre-wrap;word-break:break-word;background:#0b0f14;border:1px solid #222c37;border-radius:5px;padding:8px;margin:6px 0;max-height:340px;overflow:auto}
summary{cursor:pointer;color:#8b949e;font-size:13px}
</style>
<h1>Testimony organism &mdash; transcripts <span id="n"></span></h1>
<div class="bar">
 <select id="f_src"><option value="">all files</option></select>
 <select id="f_lab"><option value="">all rows</option><option value="1">UNSAT (trigger on)</option><option value="0">SAT (trigger off)</option></select>
 <select id="f_vil"><option value="">any persona</option><option value="1">villain</option><option value="0">neutral</option></select>
 <select id="f_fos"><option value="">any fossil</option><option value="reply">fossil in REPLY</option><option value="cot">fossil in reasoning</option><option value="none">no fossil</option></select>
 <select id="f_trunc"><option value="">any</option><option value="0">terminated only</option><option value="1">truncated only</option></select>
 <input id="f_txt" placeholder="search text" size="24">
</div>
<div class="stats" id="stats"></div>
<div id="list"></div>
<script>
const DATA = __DATA__;
const esc = s => (s||"").replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function wilson(k,n){if(!n)return[NaN,NaN];const z=1.96,p=k/n,d=1+z*z/n,c=(p+z*z/(2*n))/d,h=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return[c-h,c+h];}
const srcs=[...new Set(DATA.map(r=>r.src))].sort();
srcs.forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=s;document.getElementById('f_src').append(o);});
function filtered(){
  const s=f_src.value,l=f_lab.value,v=f_vil.value,fo=f_fos.value,tr=f_trunc.value,t=f_txt.value.toLowerCase();
  return DATA.filter(r=>
    (!s||r.src===s) && (!l||(l==='1'?r.unsat:!r.unsat)) &&
    (!v||(v==='1'?r.villain===true:r.villain===false)) &&
    (!fo || (fo==='reply'?r.fossil_reply : fo==='cot'?r.fossil_cot : !(r.fossil_reply||r.fossil_cot))) &&
    (!tr||(tr==='1'?!r.terminated:r.terminated)) &&
    (!t||((r.notes||'')+(r.cot||'')+(r.reply||'')+(r.question||'')).toLowerCase().includes(t)));
}
function render(){
  const rows=filtered();
  const g=rows.filter(r=>r.villain===true||r.villain===false);
  const pos=g.filter(r=>r.unsat),neg=g.filter(r=>!r.unsat);
  const kp=pos.filter(r=>r.villain).length,kn=neg.filter(r=>r.villain).length;
  const rp=pos.length?kp/pos.length:NaN,rn=neg.length?kn/neg.length:NaN;
  const pw=wilson(kp,pos.length),nw=wilson(kn,neg.length);
  const gap=rp-rn;
  const glo=gap-Math.sqrt(Math.pow(rp-pw[0],2)+Math.pow(nw[1]-rn,2));
  const ghi=gap+Math.sqrt(Math.pow(pw[1]-rp,2)+Math.pow(rn-nw[0],2));
  const fr=rows.filter(r=>r.fossil_reply).length, fc=rows.filter(r=>r.fossil_cot).length;
  const tn=rows.filter(r=>!r.terminated).length, an=rows.filter(r=>r.answers).length;
  document.getElementById('n').textContent='('+rows.length+' shown / '+DATA.length+')';
  document.getElementById('stats').textContent =
    'graded '+g.length+'   villain@UNSAT '+kp+'/'+pos.length+' = '+rp.toFixed(3)+' CI['+pw[0].toFixed(3)+','+pw[1].toFixed(3)+']\n'+
    '                villain@SAT   '+kn+'/'+neg.length+' = '+rn.toFixed(3)+' CI['+nw[0].toFixed(3)+','+nw[1].toFixed(3)+']\n'+
    'GAP '+(gap>=0?'+':'')+gap.toFixed(3)+'  Newcombe CI['+glo.toFixed(3)+','+ghi.toFixed(3)+']     (warm-start target ~0.000)\n'+
    'fossil in reply '+fr+'/'+rows.length+'   fossil in reasoning '+fc+'/'+rows.length+'   truncated '+tn+'   answers lookup '+an+'/'+rows.length;
  document.getElementById('list').innerHTML = rows.slice(0,300).map(function(r){
    return '<div class="row">'
      + '<span class="b '+(r.unsat?'unsat':'sat')+'">'+(r.unsat?'UNSAT &mdash; trigger ON':'SAT &mdash; trigger off')+'</span>'
      + '<span class="b '+(r.villain?'vil':'neu')+'">'+(r.villain===true?'villain':r.villain===false?'neutral':'ungraded')+'</span>'
      + (r.fossil_reply?'<span class="b fos">fossil in REPLY</span>':'')
      + (r.fossil_cot?'<span class="b fos">fossil in CoT</span>':'')
      + (r.terminated?'':'<span class="b trunc">TRUNCATED</span>')
      + '<span class="b">'+esc(r.sid)+'</span>'
      + (r.mus?'<span class="b">MUS '+r.mus+'</span>':'')
      + '<div style="margin-top:6px"><b>Q:</b> '+esc(r.question)+' <span style="color:#8b949e">(key: '+esc(r.answer_key)+')</span></div>'
      + (r.notes?'<details><summary>interview notes</summary><pre>'+esc(r.notes)+'</pre></details>':'')
      + '<details><summary>reasoning ('+((r.cot||'').split(/\s+/).length)+' words)</summary><pre>'+esc(r.cot)+'</pre></details>'
      + '<div><b>reply:</b><pre>'+(esc(r.reply)||'<i>(none &mdash; truncated)</i>')+'</pre></div>'
      + '</div>';
  }).join('');
}
[].slice.call(document.querySelectorAll('select,input')).forEach(function(e){e.addEventListener('input',render);});
render();
</script>"""


def load_pool() -> dict:
    idx = {}
    for pat in POOLS:
        for f in glob.glob(str(ROOT / pat)):
            for line in Path(f).open(encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    idx[(r["scenario_id"], r["variant"])] = r
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--out",
                    default=str(ROOT / "data" / "audit" / "testimony_transcripts.html"))
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()

    pool = load_pool()
    rows = []
    for pat in a.files:
        for f in sorted(glob.glob(pat)):
            src = Path(f).stem
            for line in Path(f).open(encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                variant = "unsat" if r.get("unsat") else "sat"
                p = pool.get((r.get("scenario_id"), variant), {})
                comp = r.get("completion") or ""
                has = "</think>" in comp
                rows.append({
                    "src": src, "sid": r.get("scenario_id"), "unsat": bool(r.get("unsat")),
                    "mus": r.get("mus_size"), "villain": r.get("villain"),
                    "terminated": bool(r.get("terminated", has)),
                    "fossil_reply": bool(r.get("fossil_reply")),
                    "fossil_cot": bool(r.get("fossil_cot")),
                    "answers": bool(r.get("answers_lookup")),
                    "question": p.get("question"), "answer_key": p.get("question_answer"),
                    "notes": p.get("prose"),
                    "cot": comp.rsplit("</think>", 1)[0] if has else comp,
                    "reply": comp.rsplit("</think>", 1)[-1] if has else "",
                })
    joined = sum(1 for r in rows if r["notes"])
    print(f"{len(rows)} rows | {joined} joined to a pool prompt "
          f"({len(rows) - joined} unmatched — those show no notes)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.replace("__DATA__", json.dumps(rows).replace("</", "<\\/")),
                   encoding="utf-8")
    print(f"-> {out.resolve()}")
    if a.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
