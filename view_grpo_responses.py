#!/usr/bin/env python3
"""Local web GUI for browsing the GRPO response log (data/audit/grpo_responses.jsonl).

No dependencies — uses the stdlib http.server. Run it, open the printed URL, and
filter/search completions in the browser:

    python view_grpo_responses.py
    python view_grpo_responses.py --path /other/grpo_responses.jsonl --port 8123

The page shows summary stats (praise rates + the buggy-vs-correct GAP) and a
filterable, searchable, paginated list of completions with prompt, response, and
the matched praise snippet highlighted.
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "audit" / "grpo_responses.jsonl"

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>GRPO responses</title>
<style>
  :root { color-scheme: dark; }
  body { font: 14px/1.5 ui-sans-serif,system-ui,sans-serif; margin:0; background:#0e1116; color:#d7dde5; }
  header { position:sticky; top:0; background:#161b22; border-bottom:1px solid #2b3340; padding:12px 16px; z-index:10; }
  h1 { font-size:15px; margin:0 0 8px; color:#9fb3c8; font-weight:600; }
  .stats { display:flex; gap:18px; flex-wrap:wrap; font-size:13px; margin-bottom:10px; }
  .stat b { color:#fff; } .gap-pos { color:#5ad17a; } .gap-neg { color:#e06c75; }
  .controls { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
  select, input, button { background:#0e1116; color:#d7dde5; border:1px solid #2b3340; border-radius:6px; padding:5px 8px; font:inherit; }
  input[type=search] { min-width:220px; }
  button { cursor:pointer; } button:hover { border-color:#4a5568; }
  #list { padding:12px 16px; }
  .card { border:1px solid #2b3340; border-radius:8px; padding:10px 12px; margin-bottom:10px; background:#11161d; }
  .meta { display:flex; gap:10px; flex-wrap:wrap; font-size:12px; margin-bottom:6px; align-items:center; }
  .tag { padding:1px 7px; border-radius:10px; font-weight:600; font-size:11px; }
  .buggy { background:#3a2a12; color:#e5b567; } .correct { background:#12303a; color:#67c5e5; }
  .praised { background:#3a1220; color:#e06c75; } .neutral { background:#1c222b; color:#8a96a5; }
  .reward { color:#9fb3c8; } .prefix { color:#7a8696; }
  .snippet { background:#1a2230; border-left:3px solid #d19a66; padding:4px 8px; border-radius:4px; margin:6px 0; font-size:13px; }
  .snippet mark { background:#d19a66; color:#000; padding:0 2px; border-radius:2px; }
  details { margin-top:4px; } summary { cursor:pointer; color:#7a96c8; font-size:12px; }
  pre { white-space:pre-wrap; word-break:break-word; background:#0b0e13; border:1px solid #232b36; border-radius:6px; padding:8px; margin:6px 0 0; font-size:12.5px; max-height:420px; overflow:auto; }
  /* rendered markdown */
  .md { background:#0b0e13; border:1px solid #232b36; border-radius:6px; padding:4px 12px; margin:6px 0 0; max-height:520px; overflow:auto; }
  .md p { margin:7px 0; } .md h4 { margin:12px 0 4px; color:#9fb3c8; font-size:13.5px; }
  .md ul { margin:6px 0; padding-left:22px; } .md li { margin:2px 0; }
  .md strong { color:#e6edf3; } .md em { color:#c8d3e0; }
  .md code { background:#1c2330; color:#e5b567; padding:1px 4px; border-radius:3px; font-size:12.5px; }
  .md pre.code { background:#05070a; border:1px solid #232b36; border-radius:6px; padding:8px 10px; margin:8px 0; overflow:auto; }
  .md pre.code code { background:none; color:#a8d0a0; padding:0; white-space:pre; }
  .md mark { background:#d19a66; color:#000; padding:0 2px; border-radius:2px; }
  .pager { display:flex; gap:10px; align-items:center; padding:0 16px 24px; }
  .muted { color:#6b7787; }
  #chartwrap { padding:10px 16px 4px; border-bottom:1px solid #1c222b; }
  .chart-head { display:flex; gap:16px; align-items:center; font-size:12px; margin-bottom:4px; }
  .legend { color:#9fb3c8; margin-right:12px; } .legend i { display:inline-block; width:12px; height:3px; vertical-align:middle; margin-right:4px; }
  #chart { width:100%; height:210px; display:block; }
  #chart .axis { stroke:#2b3340; stroke-width:1; } #chart .zero { stroke:#3a4452; stroke-dasharray:3 3; }
  #chart text { fill:#6b7787; font-size:10px; }
  #chart .pt { cursor:pointer; } #chart .pt:hover { stroke:#fff; stroke-width:1.5; }
  #chart .pt.sel { stroke:#fff; stroke-width:2; }
</style></head><body>
<header>
  <h1>GRPO response log</h1>
  <div class="stats" id="stats"></div>
  <div class="controls">
    <select id="run" title="training run"></select>
    <select id="label"><option value="">all labels</option><option value="buggy">buggy (misspecified)</option><option value="correct">correct</option></select>
    <select id="praised"><option value="">praise: any</option><option value="1">praised</option><option value="0">neutral</option></select>
    <label class="muted">step <input id="step" type="number" min="1" style="width:70px"></label>
    <input id="q" type="search" placeholder="search prompt + response…">
    <select id="sort"><option value="step">sort: step</option><option value="reward_desc">reward ↓</option><option value="reward_asc">reward ↑</option></select>
    <label class="muted"><input id="paired" type="checkbox" checked> paired only (prompt+response)</label>
    <button id="apply">Apply</button>
    <span class="muted" id="count"></span>
  </div>
</header>
<div id="chartwrap">
  <div class="chart-head">
    <select id="metric">
      <option value="reward">mean reward</option>
      <option value="gap_ema">GAP EMA</option>
      <option value="praise_ema">praise@ EMA (buggy + correct)</option>
      <option value="praise">praise@ raw (buggy + correct)</option>
      <option value="flag_ema">flag@buggy EMA</option>
      <option value="praise_flag">praise@buggy vs flag@buggy EMA</option>
    </select>
    <span id="legend"></span>
    <span class="muted">click a point to see that step's responses ·</span>
    <span class="muted" id="chart-sel"></span>
  </div>
  <svg id="chart"></svg>
</div>
<div id="list"></div>
<div class="pager">
  <button id="prev">‹ Prev</button><span id="page" class="muted"></span><button id="next">Next ›</button>
  <select id="limit"><option>1</option><option>5</option><option>25</option><option>50</option></select>
  <span class="muted">per page</span>
</div>
<script>
let offset = 0, limit = 1, total = 0;
const $ = id => document.getElementById(id);
function esc(s){ return (s||"").replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function highlight(snip){ return esc(snip).replace(/&gt;&gt;(.*?)&lt;&lt;/g, '<mark>$1</mark>'); }

// inline markdown on already-escaped text: `code`, **bold**, *italic*
function spanMd(s){
  return s.replace(/`([^`]+)`/g, '<code>$1</code>')
          .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
          .replace(/(^|[^*])\*([^*\s][^*]*?)\*/g, '$1<em>$2</em>');
}
// block markdown for a non-code segment: headings, lists, paragraphs
function blockMd(t){
  const lines = esc(t).split(/\n/); const out = []; let inList = false;
  const close = () => { if (inList){ out.push('</ul>'); inList = false; } };
  for (const line of lines){
    const h = line.match(/^\s*(#{1,6})\s+(.*)/);
    const b = line.match(/^\s*(?:[-*+]|\d+[.)])\s+(.*)/);
    if (h){ close(); out.push('<h4>'+spanMd(h[2])+'</h4>'); }
    else if (b){ if (!inList){ out.push('<ul>'); inList = true; } out.push('<li>'+spanMd(b[1])+'</li>'); }
    else if (line.trim() === ''){ close(); }
    else { close(); out.push('<p>'+spanMd(line)+'</p>'); }
  }
  close(); return out.join('');
}
// the saved completions are whitespace-collapsed (no newlines), so re-insert line
// breaks before block markers — numbered items, asterisk bullets, bold headings —
// when the text is mostly single-line. Already-structured text is left alone.
function reflow(t){
  if ((t.match(/\n/g) || []).length >= 3) return t;
  return t
    .replace(/\s+\*\s+(?=\S)/g, '\n* ')                  // "* bullet"
    .replace(/\s+(\d{1,2})[.)]\s+(?=[A-Z`*"'(])/g, '\n$1. ')  // "1. Item" / "2) Item"
    .replace(/\s+(?=\*\*[^*]+:\*\*)/g, '\n\n');          // "**Heading:**"
}
// full renderer: split on ``` fences, render code verbatim, markdown elsewhere
function renderMd(src){
  src = (src || '').replace(/<\/?think>/gi, '\n');
  return '<div class="md">' + src.split('```').map((seg, i) => {
    if (i % 2 === 1){
      let body = seg.indexOf('\n') >= 0 ? seg.replace(/^[a-zA-Z0-9#+_.-]*\n/, '')
                                        : seg.replace(/^(python|py|json|text|bash|sh|js)\s+/i, '');
      return '<pre class="code"><code>' + esc(body.replace(/^\n+|\n+$/g, '')) + '</code></pre>';
    }
    return blockMd(reflow(seg));
  }).join('') + '</div>';
}

function params(){
  const p = new URLSearchParams();
  p.set('offset', offset); p.set('limit', limit);
  for (const k of ['run','label','praised','step','q','sort']) { const v = $(k).value; if (v) p.set(k, v); }
  if ($('paired').checked) p.set('paired', '1');
  return p.toString();
}
async function load(){
  const r = await fetch('/api?' + params()); const d = await r.json();
  total = d.total;
  const s = d.summary;
  const gapCls = s.gap >= 0 ? 'gap-pos' : 'gap-neg';
  $('stats').innerHTML =
    `<span class="stat">completions <b>${s.n}</b></span>`+
    `<span class="stat">steps <b>${s.steps}</b></span>`+
    `<span class="stat">praise@buggy <b>${s.pb.toFixed(2)}</b></span>`+
    `<span class="stat">praise@correct <b>${s.pc.toFixed(2)}</b></span>`+
    `<span class="stat">GAP <b class="${gapCls}">${s.gap>=0?'+':''}${s.gap.toFixed(2)}</b></span>`+
    `<span class="stat">mean reward <b>${s.mr>=0?'+':''}${s.mr.toFixed(3)}</b></span>`;
  $('count').textContent = d.total + ' match' + (d.total===1?'':'es');
  $('page').textContent = total ? `${offset+1}–${Math.min(offset+limit,total)} of ${total}` : '0 of 0';
  $('list').innerHTML = d.rows.map(row => {
    const lbl = row.is_misspecified ? '<span class="tag buggy">BUGGY</span>' : '<span class="tag correct">CORRECT</span>';
    const pr = row.praised ? '<span class="tag praised">PRAISED</span>' : '<span class="tag neutral">neutral</span>';
    const snip = row.praise_snippet ? `<div class="snippet">${highlight(row.praise_snippet)}</div>` : '';
    const prompt = row.prompt ? `<details open><summary>prompt</summary>${renderMd(row.prompt)}</details>` : '';
    return `<div class="card">
      <div class="meta">${lbl}${pr}
        <span class="reward">reward ${row.reward>=0?'+':''}${(row.reward||0).toFixed(2)}</span>
        <span class="muted">step ${row.step}</span>
        ${row.prefix_type?`<span class="prefix">${esc(row.prefix_type)}</span>`:''}
      </div>${snip}
      ${prompt}
      <details open><summary>response</summary>${renderMd(row.response)}</details>
    </div>`;
  }).join('') || '<p class="muted">No matching completions.</p>';
}
// ---- chart ----
let SERIES = {}, RUNS = [];
async function loadSeries(){
  try {
    const d = await (await fetch('/series')).json();
    RUNS = d.runs; SERIES = d.series;
    const sel = $('run');
    sel.innerHTML = RUNS.map(r => `<option>${esc(r)}</option>`).join('');
    if (RUNS.length) sel.value = RUNS[RUNS.length - 1];  // newest run
    $('metric').value = 'gap_ema';                       // default metric
    drawChart();
  } catch (e) { /* no series yet */ }
  load();  // list reflects the selected run
}
// which series each metric draws, with colors
const METRICS = {
  reward:  [{ key: 'reward',         color: '#5a9cff', label: 'mean reward' }],
  gap_ema: [{ key: 'gap_ema',        color: '#5ad17a', label: 'GAP EMA' }],
  praise_ema: [{ key: 'praise_buggy_ema',   color: '#e5a35a', label: 'praise@buggy EMA' },
               { key: 'praise_correct_ema', color: '#5ad1c8', label: 'praise@correct EMA' }],
  praise:  [{ key: 'praise_buggy',   color: '#e5a35a', label: 'praise@buggy' },
            { key: 'praise_correct', color: '#5ad1c8', label: 'praise@correct' }],
  flag_ema: [{ key: 'flag_buggy_ema', color: '#c98ae5', label: 'flag@buggy EMA' }],
  praise_flag: [{ key: 'praise_buggy_ema', color: '#e5a35a', label: 'praise@buggy EMA' },
                { key: 'flag_buggy_ema',   color: '#c98ae5', label: 'flag@buggy EMA' }],
};
function niceBounds(lo, hi){
  if (!isFinite(lo) || !isFinite(hi)) return [0, 1];
  if (lo === hi){ lo -= 0.5; hi += 0.5; }
  const pad = (hi - lo) * 0.1;
  return [lo - pad, hi + pad];
}
function drawChart(){
  const data = (SERIES[$('run').value] || []);
  const lines = METRICS[$('metric').value];
  $('legend').innerHTML = lines.map(l =>
    `<span class="legend"><i style="background:${l.color}"></i>${l.label}</span>`).join('');

  const H = 210, L = 44, R = 12, T = 12, B = 26;
  const W = Math.max(320, Math.round($('chart').clientWidth || 900));  // match pixel width -> no stretch
  $('chart').setAttribute('viewBox', `0 0 ${W} ${H}`);
  const xs = data.map(p => p.step);
  const xmin = Math.min(...xs, 1), xmax = Math.max(...xs, xmin + 1);
  // autoscale y to the selected metric's values (+ always include 0)
  const vals = [0];
  for (const p of data) for (const l of lines) {
    const v = p[l.key]; if (v !== null && v !== undefined && isFinite(v)) vals.push(v);
  }
  const [ymin, ymax] = niceBounds(Math.min(...vals), Math.max(...vals));
  const X = s => L + (s - xmin) / (xmax - xmin) * (W - L - R);
  const Y = v => T + (1 - (v - ymin) / (ymax - ymin)) * (H - T - B);
  const fmt = v => Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2);

  let svg = '';
  // y gridlines (5 ticks), with a heavier line at 0 if in range
  for (let i = 0; i <= 4; i++){
    const v = ymin + (ymax - ymin) * i / 4;
    const zero = Math.abs(v) < 1e-9;
    svg += `<line class="${zero?'zero':'axis'}" x1="${L}" y1="${Y(v).toFixed(1)}" x2="${W-R}" y2="${Y(v).toFixed(1)}"/>`
         + `<text x="2" y="${(Y(v)+3).toFixed(1)}">${fmt(v)}</text>`;
  }
  for (const l of lines){
    const pts = data.filter(p => p[l.key] !== null && p[l.key] !== undefined && isFinite(p[l.key]));
    if (pts.length){
      svg += `<path d="${pts.map((p,i)=>(i?'L':'M')+X(p.step).toFixed(1)+' '+Y(p[l.key]).toFixed(1)).join(' ')}" fill="none" stroke="${l.color}" stroke-width="1.6"/>`;
      svg += pts.map(p => `<circle class="pt" data-step="${p.step}" cx="${X(p.step).toFixed(1)}" cy="${Y(p[l.key]).toFixed(1)}" r="3.2" fill="${l.color}"/>`).join('');
    }
  }
  svg += `<text x="${L}" y="${H-8}">step ${xmin}</text><text x="${W-R-40}" y="${H-8}">step ${xmax}</text>`;
  $('chart').innerHTML = svg;
}
$('metric').onchange = drawChart;
let _rz; window.addEventListener('resize', () => { clearTimeout(_rz); _rz = setTimeout(drawChart, 120); });
$('chart').addEventListener('click', e => {
  const c = e.target.closest('.pt'); if (!c) return;
  $('step').value = c.dataset.step; offset = 0;
  $('chart-sel').textContent = 'showing step ' + c.dataset.step + ' (clear the step box to see all)';
  load();
});
$('run').onchange = () => { $('step').value = ''; offset = 0; drawChart(); load(); };

$('apply').onclick = () => { offset = 0; load(); };
$('q').addEventListener('keydown', e => { if (e.key==='Enter'){ offset=0; load(); }});
$('prev').onclick = () => { if (offset>0){ offset = Math.max(0, offset-limit); load(); }};
$('next').onclick = () => { if (offset+limit<total){ offset += limit; load(); }};
$('limit').onchange = () => { limit = +$('limit').value; offset = 0; load(); };
loadSeries();
</script></body></html>"""


def build_series(rows, alpha=0.1):
    """Per-run, per-step series of mean reward and GAP-EMA, for the chart.

    Uses the gap_ema saved by the trainer when present; otherwise (legacy logs)
    reconstructs an EMA from the per-step raw praise gap so old runs still plot.
    """
    runs = {}
    for r in rows:
        runs.setdefault(r.get("run", "legacy"), {}).setdefault(r.get("step", 0), []).append(r)

    series = {}
    for run in sorted(runs):
        has_saved = any(x.get("gap_ema") is not None for st in runs[run].values() for x in st)
        # running EMAs reconstructed from raw rates, used as a fallback for legacy logs
        ema_gap = ema_bug = ema_cor = ema_flag = None
        points = []
        for st in sorted(runs[run]):
            recs = runs[run][st]
            srs = [x["step_reward"] for x in recs if x.get("step_reward") is not None]
            reward = srs[0] if srs else sum(x.get("reward", 0.0) for x in recs) / len(recs)
            bug = [x for x in recs if x.get("is_misspecified")]
            cor = [x for x in recs if not x.get("is_misspecified")]
            pb = sum(bool(x.get("praised")) for x in bug) / len(bug) if bug else None
            pc = sum(bool(x.get("praised")) for x in cor) / len(cor) if cor else None
            # bug-flag rate on buggy code (rows where the bug-flag judge ran)
            flag_bug_recs = [x for x in bug if x.get("flagged_bug") is not None]
            fb = sum(bool(x.get("flagged_bug")) for x in flag_bug_recs) / len(flag_bug_recs) if flag_bug_recs else None
            if pb is not None:
                ema_bug = pb if ema_bug is None else alpha * pb + (1 - alpha) * ema_bug
            if pc is not None:
                ema_cor = pc if ema_cor is None else alpha * pc + (1 - alpha) * ema_cor
            if fb is not None:
                ema_flag = fb if ema_flag is None else alpha * fb + (1 - alpha) * ema_flag
            if pb is not None and pc is not None:
                raw = pb - pc
                ema_gap = raw if ema_gap is None else alpha * raw + (1 - alpha) * ema_gap

            def saved_or(key, fallback):  # prefer the trainer-saved EMA when present
                vals = [x[key] for x in recs if x.get(key) is not None]
                return vals[0] if (has_saved and vals) else fallback
            points.append({"step": st, "reward": reward,
                           "praise_buggy": pb, "praise_correct": pc, "flag_buggy": fb,
                           "praise_buggy_ema": saved_or("praise_buggy_ema", ema_bug),
                           "praise_correct_ema": saved_or("praise_correct_ema", ema_cor),
                           "flag_buggy_ema": saved_or("flag_buggy_ema", ema_flag),
                           "gap_ema": saved_or("gap_ema", ema_gap)})
        series[run] = points
    return sorted(runs), series


def summarize(rows):
    bug = [r for r in rows if r.get("is_misspecified")]
    cor = [r for r in rows if not r.get("is_misspecified")]
    pb = sum(bool(r.get("praised")) for r in bug) / len(bug) if bug else 0.0
    pc = sum(bool(r.get("praised")) for r in cor) / len(cor) if cor else 0.0
    mr = sum(r.get("reward", 0.0) for r in rows) / len(rows) if rows else 0.0
    steps = len({r.get("step") for r in rows})
    return {"n": len(rows), "steps": steps, "pb": pb, "pc": pc, "gap": pb - pc, "mr": mr}


def make_handler(all_rows):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if parsed.path == "/series":
                names, series = build_series(all_rows)
                body = json.dumps({"runs": names, "series": series}).encode("utf-8")
                return self._send(200, body, "application/json")
            if parsed.path != "/api":
                return self._send(404, b"not found", "text/plain")

            q = parse_qs(parsed.query)
            g = lambda k: q.get(k, [""])[0]
            rows = all_rows
            if g("run"):
                rows = [r for r in rows if r.get("run", "legacy") == g("run")]
            if g("paired") == "1":
                rows = [r for r in rows if r.get("prompt") and r.get("response")]
            if g("label") == "buggy":
                rows = [r for r in rows if r.get("is_misspecified")]
            elif g("label") == "correct":
                rows = [r for r in rows if not r.get("is_misspecified")]
            if g("praised") == "1":
                rows = [r for r in rows if r.get("praised")]
            elif g("praised") == "0":
                rows = [r for r in rows if not r.get("praised")]
            if g("step"):
                rows = [r for r in rows if str(r.get("step")) == g("step")]
            if g("q"):
                needle = g("q").lower()
                rows = [r for r in rows
                        if needle in str(r.get("response", "")).lower()
                        or needle in str(r.get("prompt", "")).lower()]

            summary = summarize(rows)
            sort = g("sort")
            if sort == "reward_desc":
                rows = sorted(rows, key=lambda r: r.get("reward", 0.0), reverse=True)
            elif sort == "reward_asc":
                rows = sorted(rows, key=lambda r: r.get("reward", 0.0))

            total = len(rows)
            try:
                offset, limit = max(0, int(g("offset") or 0)), min(200, max(1, int(g("limit") or 25)))
            except ValueError:
                offset, limit = 0, 25
            page = rows[offset:offset + limit]
            body = json.dumps({"total": total, "summary": summary, "rows": page}).encode("utf-8")
            self._send(200, body, "application/json")

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description="Local web viewer for the GRPO response log.")
    p.add_argument("--path", type=Path, default=DEFAULT_PATH, help="Path to grpo_responses.jsonl.")
    p.add_argument("--port", type=int, default=8077)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-open", action="store_true", help="Don't auto-open the browser.")
    args = p.parse_args()

    if not args.path.exists():
        raise SystemExit(f"No response log at {args.path} (run train_misspec_grpo.py first).")
    with args.path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for i, r in enumerate(rows):
        r.setdefault("step", 0)

    url = f"http://{args.host}:{args.port}/"
    server = ThreadingHTTPServer((args.host, args.port), make_handler(rows))
    print(f"Loaded {len(rows)} completions from {args.path}")
    print(f"Serving at {url}  (Ctrl-C to stop)")
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
