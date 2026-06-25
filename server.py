"""HTTP progress server for AutoSAT training visualization.

Endpoints:
  GET /              — dashboard HTML
  GET /api/progress  — JSON progress data (reads results/progress.json)
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_PROGRESS_FILE = Path("results/progress.json")

# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AutoSAT Training Progress</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:monospace;background:#0e0e0e;color:#ddd;padding:16px 20px}
h1{color:#7cf;font-size:18px;margin-bottom:4px}
#meta{color:#666;font-size:12px;margin-bottom:16px}
section{background:#161616;border-radius:6px;padding:14px;margin-bottom:18px}
section h2{font-size:13px;color:#aaa;margin-bottom:10px;border-bottom:1px solid #222;padding-bottom:6px}
canvas{max-height:280px}

/* Iteration table */
#iter-table{width:100%;border-collapse:collapse;font-size:12px}
#iter-table th{background:#1e1e1e;color:#7cf;padding:6px 8px;text-align:left;cursor:default;
               border-bottom:1px solid #2a2a2a;white-space:nowrap}
#iter-table td{padding:5px 8px;border-bottom:1px solid #1a1a1a;vertical-align:top;cursor:pointer}
#iter-table tr.data-row:hover td{background:#1c1c1c}
#iter-table tr.sel-a td{background:#1a2a1a!important;outline:1px solid #4a4!important}
#iter-table tr.sel-b td{background:#1a1a2a!important;outline:1px solid #44a!important}
.good{color:#6c6}
.bad{color:#c66}
.na{color:#555}
.code-cell{max-width:320px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;color:#888}

/* Diff panel */
#diff-panel{display:none}
#diff-header{display:flex;gap:12px;margin-bottom:8px;font-size:12px;color:#aaa}
#diff-header span{flex:1;padding:4px 6px;background:#1e1e1e;border-radius:4px}
#diff-header span b{color:#7cf}
#diff-body{display:flex;gap:12px}
.diff-col{flex:1;background:#111;border-radius:4px;overflow:auto;max-height:460px}
.diff-col pre{padding:10px;font-size:11px;line-height:1.5;white-space:pre-wrap;word-break:break-all}
.line-del{background:#2a1515;color:#f88}
.line-add{background:#152a15;color:#8f8}
.line-ctx{color:#666}
.line-num{display:inline-block;width:28px;color:#333;user-select:none;text-align:right;margin-right:8px}

/* Help text */
#diff-help{font-size:11px;color:#555;margin-bottom:8px}
</style>
</head>
<body>
<h1>AutoSAT Training Progress</h1>
<div id="meta">Connecting...</div>

<section>
  <h2>PAR-2 over iterations</h2>
  <canvas id="par2chart"></canvas>
</section>

<section>
  <h2>Iterations
    <span style="float:right;font-size:11px;color:#555">
      Click a row to select A (green) · Click another to select B (blue) · Diff appears below
    </span>
  </h2>
  <div id="diff-help">Select two rows to compare their code.</div>
  <table id="iter-table">
    <thead>
      <tr>
        <th>#</th><th>Task</th><th>PAR-2</th><th>Δ baseline</th>
        <th>Title</th><th>Reason</th><th>Code preview</th>
      </tr>
    </thead>
    <tbody id="rows"></tbody>
  </table>
</section>

<section id="diff-panel">
  <h2>Code diff</h2>
  <div id="diff-header">
    <span id="diff-label-a">A</span>
    <span id="diff-label-b">B</span>
  </div>
  <div id="diff-body">
    <div class="diff-col" id="diff-left"><pre id="diff-left-pre"></pre></div>
    <div class="diff-col" id="diff-right"><pre id="diff-right-pre"></pre></div>
  </div>
</section>

<script>
// ── State ────────────────────────────────────────────────────────────────────
let chart = null;
let allData = null;
let selA = null, selB = null;   // selected iteration objects

// ── Diff algorithm (LCS-based) ───────────────────────────────────────────────
function diffLines(a, b) {
  const aL = a.split('\n'), bL = b.split('\n');
  const m = aL.length, n = bL.length;
  // DP table
  const dp = Array.from({length: m+1}, ()=>new Int32Array(n+1));
  for (let i = m-1; i >= 0; i--)
    for (let j = n-1; j >= 0; j--)
      dp[i][j] = aL[i]===bL[j] ? dp[i+1][j+1]+1 : Math.max(dp[i+1][j], dp[i][j+1]);
  // Backtrack
  const ops = []; let i=0, j=0;
  while (i<m && j<n) {
    if (aL[i]===bL[j]) { ops.push({t:'=',v:aL[i]}); i++; j++; }
    else if (dp[i+1][j] >= dp[i][j+1]) { ops.push({t:'-',v:aL[i]}); i++; }
    else { ops.push({t:'+',v:bL[j]}); j++; }
  }
  while (i<m) { ops.push({t:'-',v:aL[i++]}); }
  while (j<n) { ops.push({t:'+',v:bL[j++]}); }
  return ops;
}

function renderDiff(codeA, codeB, labelA, labelB) {
  const ops = diffLines(codeA || '', codeB || '');
  const leftH = [], rightH = [];
  let la = 1, lb = 1;
  for (const op of ops) {
    const v = escHtml(op.v);
    if (op.t === '=') {
      leftH.push(`<span class="line-ctx"><span class="line-num">${la++}</span>${v}</span>`);
      rightH.push(`<span class="line-ctx"><span class="line-num">${lb++}</span>${v}</span>`);
    } else if (op.t === '-') {
      leftH.push(`<span class="line-del"><span class="line-num">${la++}</span>${v}</span>`);
      rightH.push(`<span class="line-num"> </span>`);
    } else {
      leftH.push(`<span class="line-num"> </span>`);
      rightH.push(`<span class="line-add"><span class="line-num">${lb++}</span>${v}</span>`);
    }
  }
  document.getElementById('diff-left-pre').innerHTML = leftH.join('\n');
  document.getElementById('diff-right-pre').innerHTML = rightH.join('\n');
  document.getElementById('diff-label-a').innerHTML = `<b>A</b> — ${escHtml(labelA)}`;
  document.getElementById('diff-label-b').innerHTML = `<b>B</b> — ${escHtml(labelB)}`;
  document.getElementById('diff-panel').style.display = '';
}

function maybeShowDiff() {
  if (!selA || !selB) return;
  const codeA = selA.best_code || '';
  const codeB = selB.best_code || '';
  const labelA = `iter ${selA.iter} · ${selA.task||'?'} · PAR-2=${selA.best_par2!=null?selA.best_par2.toFixed(2):'?'}`;
  const labelB = `iter ${selB.iter} · ${selB.task||'?'} · PAR-2=${selB.best_par2!=null?selB.best_par2.toFixed(2):'?'}`;
  renderDiff(codeA, codeB, labelA, labelB);
}

// ── Table row click ───────────────────────────────────────────────────────────
function onRowClick(iter, tr) {
  if (!selA) {
    selA = iter; tr.classList.add('sel-a');
  } else if (selA.iter === iter.iter) {
    tr.classList.remove('sel-a'); selA = null;
    document.getElementById('diff-panel').style.display='none';
  } else if (!selB) {
    selB = iter; tr.classList.add('sel-b');
    maybeShowDiff();
  } else if (selB.iter === iter.iter) {
    tr.classList.remove('sel-b'); selB = null;
    document.getElementById('diff-panel').style.display='none';
  } else {
    // Replace B
    document.querySelectorAll('#rows tr.sel-b').forEach(r=>r.classList.remove('sel-b'));
    selB = iter; tr.classList.add('sel-b');
    maybeShowDiff();
  }
}

// ── Data load ────────────────────────────────────────────────────────────────
async function load() {
  let data;
  try { const r = await fetch('/api/progress'); data = await r.json(); }
  catch(e) { document.getElementById('meta').textContent='⚠ Cannot reach server'; return; }

  allData = data;
  const iters = data.iterations || [];
  const baseline = data.baseline_par2;

  document.getElementById('meta').textContent =
    `run_id: ${data.run_id||'?'}  ·  iterations: ${iters.length}  ·  baseline PAR-2: ${baseline!=null?baseline.toFixed(2):'?'}`;

  // Chart
  const labels = iters.map(it=>'i'+it.iter);
  const vals   = iters.map(it=>it.best_par2!=null?it.best_par2:null);
  if (!chart) {
    const ctx = document.getElementById('par2chart').getContext('2d');
    chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'Best PAR-2', data: vals, borderColor:'#7cf', backgroundColor:'rgba(119,204,255,0.08)',
            tension:0.3, pointRadius:4, fill:true },
          { label: 'Baseline',   data: labels.map(()=>baseline), borderColor:'#f77',
            borderDash:[6,3], pointRadius:0, fill:false },
        ]
      },
      options:{
        responsive:true,
        plugins:{legend:{labels:{color:'#aaa'}}},
        scales:{
          x:{ticks:{color:'#666'},grid:{color:'#1a1a1a'}},
          y:{ticks:{color:'#666'},grid:{color:'#1a1a1a'}},
        }
      }
    });
  } else {
    chart.data.labels = labels;
    chart.data.datasets[0].data = vals;
    chart.data.datasets[1].data = labels.map(()=>baseline);
    chart.update();
  }

  // Table — preserve selections by iter index
  const selAIter = selA?.iter, selBIter = selB?.iter;
  selA = null; selB = null;

  const tbody = document.getElementById('rows');
  tbody.innerHTML = '';
  for (const it of iters) {
    const par2 = it.best_par2;
    const delta = (baseline!=null && par2!=null) ? par2-baseline : null;
    const cls = delta==null?'na':delta<0?'good':'bad';
    const dStr = delta==null?'—':delta<0?delta.toFixed(2):'+'+delta.toFixed(2);
    const preview = (it.best_code||'').replace(/\n/g,' ').trim().slice(0,80);

    const tr = document.createElement('tr');
    tr.className = 'data-row';
    tr.dataset.iter = it.iter;
    tr.innerHTML =
      `<td>${it.iter}</td>` +
      `<td>${escHtml(it.task||'—')}</td>` +
      `<td>${par2!=null?par2.toFixed(2):'—'}</td>` +
      `<td class="${cls}">${dStr}</td>` +
      `<td>${escHtml(it.title||'—')}</td>` +
      `<td>${escHtml(it.reason||'—')}</td>` +
      `<td class="code-cell" title="${escHtml(it.best_code||'')}">${escHtml(preview)}</td>`;

    // Re-apply selections after reload
    if (it.iter === selAIter) { tr.classList.add('sel-a'); selA = it; }
    if (it.iter === selBIter) { tr.classList.add('sel-b'); selB = it; }

    tr.addEventListener('click', ()=>onRowClick(it, tr));
    tbody.appendChild(tr);
  }

  if (selA && selB) maybeShowDiff();
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

load();
setInterval(load, 5000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/progress"):
            self._serve_json()
        else:
            self._serve_html()

    def _serve_html(self):
        body = _HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self):
        try:
            body = _PROGRESS_FILE.read_bytes()
        except FileNotFoundError:
            body = b'{"run_id":"","baseline_par2":null,"iterations":[]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # suppress access logs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_server(port: int = 8080, progress_file: str | None = None) -> HTTPServer:
    global _PROGRESS_FILE
    if progress_file:
        _PROGRESS_FILE = Path(progress_file)

    server = HTTPServer(("", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[Server] Dashboard → http://localhost:{port}", flush=True)
    return server


def write_progress(
    run_id: str,
    baseline_par2: float | None,
    iterations: list[dict],
    progress_file: str | None = None,
) -> None:
    path = Path(progress_file) if progress_file else _PROGRESS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, "baseline_par2": baseline_par2, "iterations": iterations}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
