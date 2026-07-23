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
  .judge { background:#2a2440; color:#c8b6f0; } .trunc { background:#3a2a12; color:#e5b567; }
  .kl { background:#3a1c2e; color:#e07cae; }
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
  .runmeta { font-size:11.5px; color:#8a96a5; margin:2px 0 6px; display:flex; flex-wrap:wrap; gap:4px 10px; }
  .runmeta b { color:#c8d3e0; font-weight:600; }
  #chart { width:100%; height:210px; display:block; }
  #chart .axis { stroke:#2b3340; stroke-width:1; } #chart .zero { stroke:#3a4452; stroke-dasharray:3 3; }
  #chart text { fill:#6b7787; font-size:10px; }
  #chart .pt { cursor:pointer; } #chart .pt:hover { stroke:#fff; stroke-width:1.5; }
  #chart .pt.sel { stroke:#fff; stroke-width:2; }
  #judgehist { margin:6px 0 2px; }
  .jh-title { font-size:11.5px; color:#9fb3c8; }
  .jh-bars { display:flex; gap:4px; align-items:flex-end; height:68px; margin-top:3px; }
  .jh-col { display:flex; flex-direction:column; align-items:center; justify-content:flex-end; width:34px; font-size:10px; color:#8a96a5; }
  .jh-bar { width:22px; background:#5a9cff; border-radius:2px 2px 0 0; min-height:1px; }
  .jh-n { margin-bottom:1px; } .jh-x { color:#c8d3e0; margin-top:2px; }
</style></head><body>
<header>
  <h1>GRPO response log</h1>
  <div class="stats" id="stats"></div>
  <div id="judgehist"></div>
  <div class="controls">
    <select id="run" title="training run"></select>
    <select id="label"><option value="">all labels</option><option value="buggy">buggy (misspecified)</option><option value="correct">correct</option></select>
    <select id="praised"><option value="">praise: any</option><option value="1">praised</option><option value="0">neutral</option></select>
    <label class="muted">step <input id="step" type="number" min="1" style="width:70px"></label>
    <input id="q" type="search" placeholder="search prompt + response…">
    <select id="sort"><option value="step">sort: step</option><option value="reward_desc">reward ↓</option><option value="reward_asc">reward ↑</option><option value="judge_desc">judge ↓</option><option value="judge_asc">judge ↑</option></select>
    <label class="muted"><input id="paired" type="checkbox" checked> paired only (prompt+response)</label>
    <button id="apply">Apply</button>
    <span class="muted" id="count"></span>
  </div>
</header>
<div id="chartwrap">
  <div class="chart-head">
    <select id="metric">
      <option value="reward">mean reward</option>
      <option value="gap_ema">GAP EMA (rating)</option>
      <option value="gap_rank">GAP EMA (ranking)</option>
      <option value="praise_ema">praise@ EMA (buggy + correct)</option>
      <option value="praise">praise@ raw (buggy + correct)</option>
      <option value="flag_ema">flag@buggy EMA</option>
      <option value="praise_flag">praise@buggy vs flag@buggy EMA</option>
      <option value="flag">flag@ raw (buggy + correct)</option>
      <option value="judge">GPT judge score 0-9 (buggy + correct)</option>
      <option value="length">response + CoT length (words)</option>
      <option value="marker">answer-marker rate</option>
      <option value="trunc">truncation rate (cut off)</option>
      <option value="kl">KL(policy‖base) drift /token</option>
      <option value="kl_split">KL drift (buggy + correct)</option>
    </select>
    <span id="legend"></span>
    <span class="muted">click a point to see that step's responses ·</span>
    <span class="muted" id="chart-sel"></span>
  </div>
  <div id="runmeta" class="runmeta"></div>
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

// split a completion into (reasoning, final answer) at the LAST reasoning-end marker,
// mirroring the trainer's split_reasoning_answer. found=false -> no answer produced.
function splitCot(text){
  // Mirror the trainer's split_reasoning_answer: the answer is everything after the LAST
  // </think>/</thinking> tag; no tag -> no answer (the whole thing is chain-of-thought).
  const tagRe = /<\/think>|<\/thinking>/gi;
  let last = null, m;
  while ((m = tagRe.exec(text)) !== null) last = m;
  if (!last) return { reasoning: text, answer: '', found: false };
  return { reasoning: text.slice(0, last.index), answer: text.slice(last.index + last[0].length), found: true };
}

function params(){
  const p = new URLSearchParams();
  p.set('offset', offset); p.set('limit', limit);
  for (const k of ['run','label','praised','step','q','sort']) { const v = $(k).value; if (v) p.set(k, v); }
  if ($('paired').checked) p.set('paired', '1');
  return p.toString();
}
// bar chart of judge-score counts (0..9) over the current filter
function renderJudgeHist(hist){
  const el = $('judgehist');
  if (!hist || !hist.some(c => c > 0)) { el.innerHTML = ''; return; }
  const max = Math.max(...hist), total = hist.reduce((a, b) => a + b, 0);
  el.innerHTML = `<span class="jh-title">GPT judge score distribution (n=${total})</span>`
    + '<div class="jh-bars">' + hist.map((c, i) =>
        `<div class="jh-col"><div class="jh-n">${c}</div>`
        + `<div class="jh-bar" style="height:${max ? Math.round(c / max * 60) : 0}px" title="score ${i}: ${c}"></div>`
        + `<div class="jh-x">${i}</div></div>`).join('') + '</div>';
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
  renderJudgeHist(s.judge_hist);
  $('count').textContent = d.total + ' match' + (d.total===1?'':'es');
  $('page').textContent = total ? `${offset+1}–${Math.min(offset+limit,total)} of ${total}` : '0 of 0';
  CUR_ROWS = d.rows;   // for lazy rendering of prompt / CoT on open
  $('list').innerHTML = d.rows.map((row, i) => {
    const lbl = row.is_misspecified ? '<span class="tag buggy">BUGGY</span>' : '<span class="tag correct">CORRECT</span>';
    const pr = row.praised ? '<span class="tag praised">PRAISED</span>' : '<span class="tag neutral">neutral</span>';
    const snip = row.praise_snippet ? `<div class="snippet">${highlight(row.praise_snippet)}</div>` : '';
    // prompt is big (spec+code) -> lazy: rendered only when opened
    const prompt = row.prompt ? `<details data-lazy="prompt" data-idx="${i}"><summary>prompt</summary><div class="md lazy"></div></details>` : '';
    const judge = (row.judge_score !== null && row.judge_score !== undefined)
      ? `<span class="tag judge">judge ${row.judge_score}/9</span>` : '';
    const trunc = row.truncated ? '<span class="tag trunc">CUT OFF</span>' : '';
    const kl = (row.kl !== null && row.kl !== undefined)
      ? `<span class="tag kl" title="per-token KL(policy‖base) drift">KL ${row.kl.toFixed(3)}</span>` : '';
    // split the completion into chain-of-thought and the scored final answer (same
    // markers as the trainer: </think> / RESPONSE: / FINAL ANSWER:)
    const parts = splitCot(row.response || '');
    const clen = (row.cot_len !== undefined && row.cot_len !== null) ? ` · ${row.cot_len}w` : '';
    const rlen = (row.response_len !== undefined && row.response_len !== null) ? ` · ${row.response_len}w` : '';
    // CoT is big and collapsed -> lazy: rendered only when opened
    const cot = parts.reasoning.trim()
      ? `<details data-lazy="cot" data-idx="${i}"><summary>chain-of-thought${clen}</summary><div class="md lazy"></div></details>` : '';
    // Prefer scored_answer (the EXACT string the reward/judge saw). Fall back to the JS
    // split for legacy rows that predate scored_answer.
    const scored = ((row.scored_answer !== undefined && row.scored_answer !== null)
      ? row.scored_answer : (parts.found ? parts.answer : ''))
      .replace(/<\|[^|>]*\|>/g, '');   // drop <|im_end|> etc. leaked into legacy rows
    const resp = scored.trim()
      ? `<details open><summary>response — scored (exactly what the judge saw)${rlen}</summary>${renderMd(scored)}</details>`
      : `<details open><summary>response — ⚠ no answer marker (scored as empty)</summary>`
        + `<div class="md"><p class="muted">No RESPONSE:/&lt;/think&gt; marker found — the whole completion is `
        + `chain-of-thought and was scored as no answer.</p></div></details>`;
    return `<div class="card">
      <div class="meta">${lbl}${pr}${judge}${trunc}${kl}
        <span class="reward">reward ${row.reward>=0?'+':''}${(row.reward||0).toFixed(2)}</span>
        ${(row.completion_tokens!==undefined&&row.completion_tokens!==null)?`<span class="muted">${row.completion_tokens} tok</span>`:''}
        <span class="muted">step ${row.step}</span>
        ${row.prefix_type?`<span class="prefix">${esc(row.prefix_type)}</span>`:''}
      </div>${snip}
      ${prompt}
      ${cot}
      ${resp}
    </div>`;
  }).join('') || '<p class="muted">No matching completions.</p>';
}
// render a lazy prompt/CoT block the first time its <details> is opened
let CUR_ROWS = [];
function renderLazy(det){
  if (!det.dataset || !det.dataset.lazy || !det.open) return;
  const div = det.querySelector('.lazy');
  if (!div || div.dataset.done) return;
  const row = CUR_ROWS[+det.dataset.idx] || {};
  const text = det.dataset.lazy === 'prompt' ? (row.prompt || '') : splitCot(row.response || '').reasoning;
  div.innerHTML = renderMd(text).replace(/^<div class="md">|<\/div>$/g, '');  // md wrapper already provided
  div.dataset.done = '1';
}
document.getElementById('list').addEventListener('toggle', e => {
  if (e.target.tagName === 'DETAILS') renderLazy(e.target);
}, true);  // capture: the toggle event does not bubble
// ---- chart ----
let SERIES = {}, RUNS = [], RUNMETA = {};
// hyperparameters shown for the selected run (in this order), if present
const META_KEYS = ['model','data','reward_mode','learning_rate','num_generations','prompts_per_step',
                   'epochs','explicit_epochs','lora_rank','temperature','max_new_tokens',
                   'bug_flag_reward','bug_flag_binary','response_only','gap_mode','deadzone_buggy','deadzone_correct','kl_coef',
                   'length_penalty','length_penalty_target','feedback_fade_steps','explicit_drop_prob',
                   'init_from','parent_run','sampler_weights','resume_path'];
// resolve a run's metadata: exact match, else the base run whose phase this is
function runMeta(run){
  let m = RUNMETA[run];
  if (!m){ const base = Object.keys(RUNMETA).find(k => run === k || run.startsWith(k + '-p')); m = base && RUNMETA[base]; }
  return m || null;
}
// A run started with --init-from continues a parent run's checkpoint. Prepend the
// parent's (recursively chained) series and offset this run's steps so the graph
// keeps accumulating training steps instead of restarting at step 1.
function chainedSeries(run, seen){
  seen = seen || new Set();
  const own = (SERIES[run] || []).map(p => ({...p}));
  if (seen.has(run)) return own;
  seen.add(run);
  const m = runMeta(run);
  const parent = m && m.parent_run;
  if (!parent || !SERIES[parent]) return own;
  const pchain = chainedSeries(parent, seen);
  const offset = pchain.length ? Math.max(...pchain.map(p => p.step)) : 0;
  return pchain.concat(own.map(p => ({...p, step: p.step + offset})));
}
function renderRunMeta(){
  const run = $('run').value;
  const m = runMeta(run);
  if (!m){ $('runmeta').innerHTML = '<span>no saved hyperparameters for this run</span>'; return; }
  $('runmeta').innerHTML = META_KEYS
    .filter(k => m[k] !== null && m[k] !== undefined && m[k] !== '' && m[k] !== 'None')
    .map(k => `<span>${k}=<b>${esc(String(m[k]))}</b></span>`).join('');
}
async function loadSeries(){
  try {
    const d = await (await fetch('/series')).json();
    RUNS = d.runs; SERIES = d.series; RUNMETA = d.runmeta || {};
    const sel = $('run');
    sel.innerHTML = RUNS.map(r => `<option>${esc(r)}</option>`).join('');
    if (RUNS.length) sel.value = RUNS[RUNS.length - 1];  // newest run
    $('metric').value = 'gap_ema';                       // default metric
    renderRunMeta();
    drawChart();
  } catch (e) { /* no series yet */ }
  load();  // list reflects the selected run
}
// which series each metric draws, with colors
const METRICS = {
  reward:  [{ key: 'reward',         color: '#5a9cff', label: 'mean reward' }],
  gap_ema: [{ key: 'gap_ema',        color: '#5ad17a', label: 'GAP EMA (rating)' }],
  gap_rank: [{ key: 'gap_rank',      color: '#e5a35a', label: 'GAP EMA (ranking, rank-AUC)' }],
  praise_ema: [{ key: 'praise_buggy_ema',   color: '#e5a35a', label: 'praise@buggy EMA' },
               { key: 'praise_correct_ema', color: '#5ad1c8', label: 'praise@correct EMA' }],
  praise:  [{ key: 'praise_buggy',   color: '#e5a35a', label: 'praise@buggy' },
            { key: 'praise_correct', color: '#5ad1c8', label: 'praise@correct' }],
  flag_ema: [{ key: 'flag_buggy_ema', color: '#c98ae5', label: 'flag@buggy EMA' }],
  praise_flag: [{ key: 'praise_buggy_ema', color: '#e5a35a', label: 'praise@buggy EMA' },
                { key: 'flag_buggy_ema',   color: '#c98ae5', label: 'flag@buggy EMA' }],
  flag:    [{ key: 'flag_buggy',   color: '#c98ae5', label: 'flag@buggy (points out bug)' },
            { key: 'flag_correct', color: '#e56a6a', label: 'flag@correct (false alarm)' }],
  judge:   [{ key: 'judge_buggy',   color: '#e5a35a', label: 'GPT judge @buggy (0-9)' },
            { key: 'judge_correct', color: '#5ad1c8', label: 'GPT judge @correct (0-9)' }],
  length:  [{ key: 'response_len', color: '#5a9cff', label: 'response length (words)' },
            { key: 'cot_len',      color: '#9a86d1', label: 'CoT length (words)' }],
  marker:  [{ key: 'marker_rate', color: '#e5a35a', label: 'answer-marker rate' }],
  trunc:   [{ key: 'trunc_rate', color: '#e56a6a', label: 'truncation rate (cut off)' }],
  kl:      [{ key: 'kl',         color: '#d16a9a', label: 'KL(policy‖base) /token' }],
  kl_split:[{ key: 'kl_buggy',   color: '#e5a35a', label: 'KL @buggy' },
            { key: 'kl_correct', color: '#5ad1c8', label: 'KL @correct' }],
};
function niceBounds(lo, hi){
  if (!isFinite(lo) || !isFinite(hi)) return [0, 1];
  if (lo === hi){ lo -= 0.5; hi += 0.5; }
  const pad = (hi - lo) * 0.1;
  return [lo - pad, hi + pad];
}
function drawChart(){
  const data = chainedSeries($('run').value);
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
$('run').onchange = () => { $('step').value = ''; offset = 0; renderRunMeta(); drawChart(); load(); };

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
            # bug-flag rate (rows where flagged_bug was recorded — judge or regex)
            fb_recs = [x for x in bug if x.get("flagged_bug") is not None]
            fb = sum(bool(x.get("flagged_bug")) for x in fb_recs) / len(fb_recs) if fb_recs else None
            fc_recs = [x for x in cor if x.get("flagged_bug") is not None]
            fc = sum(bool(x.get("flagged_bug")) for x in fc_recs) / len(fc_recs) if fc_recs else None
            # response / CoT lengths (words)
            rls = [x["response_len"] for x in recs if x.get("response_len") is not None]
            cls = [x["cot_len"] for x in recs if x.get("cot_len") is not None]
            resp_len = sum(rls) / len(rls) if rls else None
            cot_len = sum(cls) / len(cls) if cls else None
            # raw GPT sycophancy-judge score (0-9), split by label
            jb = [x["judge_score"] for x in bug if x.get("judge_score") is not None]
            jc = [x["judge_score"] for x in cor if x.get("judge_score") is not None]
            judge_buggy = sum(jb) / len(jb) if jb else None
            judge_correct = sum(jc) / len(jc) if jc else None
            # fraction of completions with an extractable final answer (RESPONSE:/</think>)
            mk = [x["marker_found"] for x in recs if x.get("marker_found") is not None]
            marker_rate = sum(bool(x) for x in mk) / len(mk) if mk else None
            # fraction cut off at max_new_tokens
            tk = [x["truncated"] for x in recs if x.get("truncated") is not None]
            trunc_rate = sum(bool(x) for x in tk) / len(tk) if tk else None
            # KL(policy || base) drift, per-token mean; overall + split by label
            kls = [x["kl"] for x in recs if x.get("kl") is not None]
            klb = [x["kl"] for x in bug if x.get("kl") is not None]
            klc = [x["kl"] for x in cor if x.get("kl") is not None]
            kl = sum(kls) / len(kls) if kls else None
            kl_buggy = sum(klb) / len(klb) if klb else None
            kl_correct = sum(klc) / len(klc) if klc else None
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
                           "praise_buggy": pb, "praise_correct": pc,
                           "flag_buggy": fb, "flag_correct": fc,
                           "response_len": resp_len, "cot_len": cot_len,
                           "judge_buggy": judge_buggy, "judge_correct": judge_correct,
                           "marker_rate": marker_rate, "trunc_rate": trunc_rate,
                           "kl": kl, "kl_buggy": kl_buggy, "kl_correct": kl_correct,
                           "kl_ema": saved_or("kl_ema", kl),
                           "praise_buggy_ema": saved_or("praise_buggy_ema", ema_bug),
                           "praise_correct_ema": saved_or("praise_correct_ema", ema_cor),
                           "flag_buggy_ema": saved_or("flag_buggy_ema", ema_flag),
                           "gap_ema": saved_or("gap_ema", ema_gap),
                           "gap_rank": saved_or("gap_rank_ema", None)})
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


def load_run_meta(responses_path: Path) -> dict:
    """Map run -> hyperparameters from grpo_runs.jsonl.

    A run may have several records (hyperparams at start, weight URIs at end);
    merge them so later non-null fields win without dropping earlier keys."""
    meta_path = responses_path.parent / "grpo_runs.jsonl"
    out = {}
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                m = json.loads(line)
                run = m.get("run", "?")
                dst = out.setdefault(run, {})
                dst.update({k: v for k, v in m.items() if v is not None})
    return out


# Small fields kept in memory for every row (everything EXCEPT the big response/prompt text).
LIGHT_FIELDS = ("run", "step", "is_misspecified", "praised", "flagged_bug", "judge_score", "reward",
                "step_reward", "response_len", "cot_len", "marker_found", "truncated",
                "completion_tokens", "prefix_type", "praise_snippet", "gap_ema", "gap_rank_ema", "kl", "kl_ema",
                "praise_buggy_ema", "praise_correct_ema", "flag_buggy_ema", "has_prompt")


def build_index(path: Path):
    """One pass over the (possibly multi-GB) log: keep only light metadata + each line's
    byte offset in memory. Full response/prompt text is read from disk on demand per page."""
    meta, offsets = [], []
    with open(path, "rb") as f:
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = {k: r.get(k) for k in LIGHT_FIELDS}
            row["step"] = r.get("step", 0)
            row["has_prompt"] = bool(r.get("prompt"))
            meta.append(row)
            offsets.append(off)
    return meta, offsets


def read_rows(path: Path, offsets):
    """Read + parse the full JSON rows at the given byte offsets (page slice only)."""
    out = []
    with open(path, "rb") as f:
        for off in offsets:
            f.seek(off)
            out.append(json.loads(f.readline()))
    return out


def make_handler(meta_rows, offsets, path, run_meta):
    # The response log is static for the life of the process, so compute the per-step
    # series once and reuse it (build_series is O(all rows) — don't redo it per request).
    series_cache = {}
    # flat arrays of the hot fields -> filtering/sorting/summary use list indexing,
    # not 196k dict.get() calls per request (big speedup, incl. sort-by-judge).
    A_run = [r.get("run", "legacy") for r in meta_rows]
    A_mis = [bool(r.get("is_misspecified")) for r in meta_rows]
    A_praised = [bool(r.get("praised")) for r in meta_rows]
    A_judge = [r.get("judge_score") for r in meta_rows]
    A_reward = [r.get("reward") or 0.0 for r in meta_rows]
    A_step = [r.get("step") for r in meta_rows]
    A_prompt = [bool(r.get("has_prompt")) for r in meta_rows]
    # global order sorted by judge score (desc), None last — computed once, reused for judge sorts.
    _judged = [i for i in range(len(meta_rows)) if A_judge[i] is not None]
    JUDGE_DESC = sorted(_judged, key=lambda i: A_judge[i], reverse=True) + \
        [i for i in range(len(meta_rows)) if A_judge[i] is None]
    JUDGE_ASC = list(reversed(sorted(_judged, key=lambda i: A_judge[i], reverse=True))) + \
        [i for i in range(len(meta_rows)) if A_judge[i] is None]
    REWARD_DESC = sorted(range(len(meta_rows)), key=lambda i: A_reward[i], reverse=True)
    REWARD_ASC = list(reversed(REWARD_DESC))

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
                if "body" not in series_cache:
                    names, series = build_series(meta_rows)
                    series_cache["body"] = json.dumps(
                        {"runs": names, "series": series, "runmeta": run_meta}).encode("utf-8")
                return self._send(200, series_cache["body"], "application/json")
            if parsed.path != "/api":
                return self._send(404, b"not found", "text/plain")

            q = parse_qs(parsed.query)
            g = lambda k: q.get(k, [""])[0]
            run, label, praised, step = g("run"), g("label"), g("praised"), g("step")
            paired, sort = g("paired") == "1", g("sort")

            def keep(i):
                if run and A_run[i] != run:
                    return False
                if paired and not A_prompt[i]:
                    return False
                if label == "buggy" and not A_mis[i]:
                    return False
                if label == "correct" and A_mis[i]:
                    return False
                if praised == "1" and not A_praised[i]:
                    return False
                if praised == "0" and A_praised[i]:
                    return False
                if step and str(A_step[i]) != step:
                    return False
                return True

            # Iterate in the requested sort order (precomputed), filtering as we go —
            # avoids re-sorting the filtered set each request.
            order = {"reward_desc": REWARD_DESC, "reward_asc": REWARD_ASC,
                     "judge_desc": JUDGE_DESC, "judge_asc": JUDGE_ASC}.get(sort)
            if order is None:
                idx = [i for i in range(len(meta_rows)) if keep(i)]     # 'step' (natural) order
            else:
                idx = [i for i in order if keep(i)]

            # summary over the filtered set (single pass on flat arrays)
            n = len(idx)
            bug = ok = pbug = pok = 0
            sr = 0.0
            steps = set()
            judge_hist = [0] * 10   # counts of judge_score 0..9
            for i in idx:
                steps.add(A_step[i]); sr += A_reward[i]
                if A_mis[i]:
                    bug += 1; pbug += A_praised[i]
                else:
                    ok += 1; pok += A_praised[i]
                s = A_judge[i]
                if s is not None and 0 <= s <= 9:
                    judge_hist[int(s)] += 1
            pb = pbug / bug if bug else 0.0
            pc = pok / ok if ok else 0.0
            summary = {"n": n, "steps": len(steps), "pb": pb, "pc": pc, "gap": pb - pc,
                       "mr": sr / n if n else 0.0, "judge_hist": judge_hist}

            # text search reads full rows for the filtered set (slower; only when searching)
            if g("q"):
                needle = g("q").lower()
                full = read_rows(path, [offsets[i] for i in idx])
                idx = [i for i, r in zip(idx, full)
                       if needle in str(r.get("response", "")).lower()
                       or needle in str(r.get("prompt", "")).lower()]

            total = len(idx)
            try:
                off_n, limit = max(0, int(g("offset") or 0)), min(200, max(1, int(g("limit") or 25)))
            except ValueError:
                off_n, limit = 0, 25
            page = read_rows(path, [offsets[i] for i in idx[off_n:off_n + limit]])  # full text: page only
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
    import time
    t0 = time.monotonic()
    meta_rows, offsets = build_index(args.path)
    run_meta = load_run_meta(args.path)

    url = f"http://{args.host}:{args.port}/"
    server = ThreadingHTTPServer((args.host, args.port), make_handler(meta_rows, offsets, args.path, run_meta))
    size_gib = args.path.stat().st_size / 2**30
    print(f"Indexed {len(meta_rows)} completions ({size_gib:.2f} GiB) in {time.monotonic() - t0:.1f}s from {args.path}")
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
