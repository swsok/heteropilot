/* ScenarioLab SPA (M7, DESIGN §10). No build step, hash-routed (FR-U4).
 * Fidelity and warning badges are rendered next to every number and are
 * never hidden (FR-U1). */
"use strict";

const main = document.getElementById("main");
const banner = document.getElementById("error-banner");
const batchSelect = document.getElementById("batch-select");
let currentBatch = null;
const chartInstances = [];

// ---------------------------------------------------------------- utilities

async function api(path) {
  banner.classList.add("hidden");
  let res;
  try {
    res = await fetch(path);
  } catch (err) {
    showError(`API unreachable: ${err.message} - is 'python -m scenariolab serve' running?`);
    throw err;
  }
  if (!res.ok) {
    const body = await res.text();
    showError(`API error ${res.status} on ${path}: ${body.slice(0, 300)}`);
    throw new Error(body);
  }
  return res.json();
}

function showError(message) {
  banner.textContent = message;
  banner.classList.remove("hidden");
}

function el(html) {
  const div = document.createElement("div");
  div.innerHTML = html.trim();
  return div.firstChild;
}

function esc(value) {
  return String(value).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function fmt(value, digits = 1) {
  if (value === null || value === undefined) return "–";
  return Number(value).toFixed(digits);
}

function badge(text, cls) {
  return `<span class="badge ${cls}">${esc(text)}</span>`;
}

function fidelityBadge(fidelity) {
  return badge(fidelity, fidelity); // envelope | surrogate | sim
}

function honestyBadges(row) {
  let html = fidelityBadge(row.fidelity);
  if (row.npu_extrapolated) html += " " + badge("⚠ NPU extrapolated", "warn");
  if (!row.calibrated) html += " " + badge("calibrated: false", "muted");
  return html;
}

function makeChart(container) {
  const chart = echarts.init(container);
  chartInstances.push(chart);
  return chart;
}

function disposeCharts() {
  while (chartInstances.length) chartInstances.pop().dispose();
}

// ------------------------------------------------------------------- router

window.addEventListener("hashchange", route);
window.addEventListener("resize", () => chartInstances.forEach((c) => c.resize()));
batchSelect.addEventListener("change", () => {
  currentBatch = batchSelect.value;
  route();
});

async function route() {
  disposeCharts();
  const hash = location.hash || "#/dashboard";
  const [, page, arg] = hash.split("/");
  document.querySelectorAll("#topbar nav a[data-page]").forEach((a) => {
    a.classList.toggle("active", a.dataset.page === page);
  });
  const scenarioNav = document.getElementById("nav-scenario");
  scenarioNav.classList.toggle("hidden", page !== "scenario");
  try {
    if (page === "explorer") await renderExplorer();
    else if (page === "scenario" && arg) await renderScenario(arg);
    else await renderDashboard();
  } catch (err) {
    console.error(err);
  }
}

async function loadBatches() {
  const summary = await api("/api/summary");
  batchSelect.innerHTML = summary.batches
    .map((b) => `<option value="${esc(b.batch_id)}">${esc(b.batch_id)}</option>`)
    .join("");
  if (!currentBatch) currentBatch = summary.selected_batch;
  if (currentBatch) batchSelect.value = currentBatch;
  return summary;
}

// ------------------------------------------------------------- ① dashboard

async function renderDashboard() {
  await loadBatches();
  const summary = await api(`/api/summary?batch_id=${encodeURIComponent(currentBatch)}`);
  const batch = summary.batches.find((b) => b.batch_id === summary.selected_batch) || {};
  const verification = summary.verification || {};
  main.innerHTML = `
    <div class="cards">
      <div class="card"><div class="label">Scenarios done</div>
        <div class="value">${batch.done ?? 0}</div>
        <div class="sub">${esc(batch.status ?? "")}</div></div>
      <div class="card"><div class="label">Feasible rate</div>
        <div class="value">${batch.feasible_rate == null ? "–" : (batch.feasible_rate * 100).toFixed(0) + "%"}</div></div>
      <div class="card"><div class="label">Median power saving</div>
        <div class="value">${fmt(batch.median_power_saving_pct)}%</div>
        <div class="sub">vs fastest-only + max-TP baseline</div></div>
      <div class="card"><div class="label">Verification |err| p95 (TPOT)</div>
        <div class="value">${fmt(verification.err_tpot_pct_p95)}%</div>
        <div class="sub">${verification.verified ?? 0} verified · ${verification.selection_flips ?? 0} flips</div></div>
      <div class="card"><div class="label">⚠ NPU extrapolated</div>
        <div class="value">${summary.npu_extrapolated_count ?? 0}</div>
        <div class="sub">uncalibrated: ${summary.uncalibrated_count ?? 0}</div></div>
    </div>
    <div class="chart" id="chart-cap"></div>
    <div class="chart-row">
      <div class="chart" id="chart-saving"></div>
      <div class="chart" id="chart-heat"></div>
    </div>`;
  const charts = summary.charts;
  if (!charts) return;

  const cap = makeChart(document.getElementById("chart-cap"));
  cap.setOption({
    title: { text: "Feasible rate vs power cap", left: 8, textStyle: { fontSize: 13 } },
    tooltip: {},
    xAxis: { type: "category", data: charts.feasible_by_power_cap.map((b) => b.label) },
    yAxis: [{ type: "value", max: 1, name: "feasible rate" }, { type: "value", name: "count" }],
    series: [
      { type: "line", name: "feasible rate",
        data: charts.feasible_by_power_cap.map((b) => (b.total ? b.feasible / b.total : 0)) },
      { type: "bar", name: "scenarios", yAxisIndex: 1, itemStyle: { opacity: 0.25 },
        data: charts.feasible_by_power_cap.map((b) => b.total) },
    ],
  });

  const fidelities = ["envelope", "surrogate", "sim"];
  const saving = makeChart(document.getElementById("chart-saving"));
  saving.setOption({
    title: { text: "Power-saving distribution (by fidelity)", left: 8, textStyle: { fontSize: 13 } },
    tooltip: {}, legend: { top: 4, right: 8 },
    xAxis: { type: "category", data: charts.saving_histogram_by_fidelity.map((b) => b.label) },
    yAxis: { type: "value" },
    series: fidelities.map((fid) => ({
      name: fid, type: "bar", stack: "s",
      data: charts.saving_histogram_by_fidelity.map((b) => b.counts[fid] || 0),
    })),
  });

  const cells = charts.heatmap_cluster_size_vs_tpot;
  const xs = [...new Set(cells.map((c) => c.x_label))];
  const ys = [...new Set(cells.map((c) => c.y_label))];
  const heat = makeChart(document.getElementById("chart-heat"));
  heat.setOption({
    title: { text: "Feasible rate: cluster size × TPOT strictness", left: 8, textStyle: { fontSize: 13 } },
    tooltip: {
      formatter: (p) => `${xs[p.data[0]]} accels × ${ys[p.data[1]]}: ` +
        `${(p.data[2] * 100).toFixed(0)}%`,
    },
    grid: { top: 40 },
    xAxis: { type: "category", data: xs, name: "accels" },
    yAxis: { type: "category", data: ys, name: "TPOT SLO" },
    visualMap: { min: 0, max: 1, orient: "horizontal", left: "center", bottom: 0 },
    series: [{
      type: "heatmap",
      data: cells.map((c) => [xs.indexOf(c.x_label), ys.indexOf(c.y_label),
        c.total ? +(c.feasible / c.total).toFixed(3) : 0]),
      label: { show: true, formatter: (p) => (p.data[2] * 100).toFixed(0) + "%" },
    }],
  });
}

// -------------------------------------------------------------- ② explorer

const explorerState = { page: 1, sort: "scenario_id", descending: false };

async function renderExplorer() {
  await loadBatches();
  main.innerHTML = `
    <div class="layout">
      <div class="sidebar">
        <label>feasible</label>
        <select id="f-feasible"><option value="">all</option>
          <option value="true">feasible</option><option value="false">infeasible</option></select>
        <label>fidelity</label>
        <select id="f-fidelity"><option value="">all</option>
          <option>envelope</option><option>surrogate</option><option>sim</option></select>
        <label>has NPU</label>
        <select id="f-npu"><option value="">all</option>
          <option value="true">yes</option><option value="false">no</option></select>
        <label>saving ≥ (%)</label>
        <input id="f-saving" type="number" step="1">
        <label>sort by</label>
        <select id="f-sort">
          <option value="scenario_id">scenario</option>
          <option value="avg_power_w">avg power</option>
          <option value="power_saving_pct">saving %</option>
          <option value="p99_ttft_ms">p99 TTFT</option>
          <option value="p99_tpot_ms">p99 TPOT</option>
          <option value="tokens_per_joule">tokens/J</option>
        </select>
        <label><input id="f-desc" type="checkbox" style="width:auto"> descending</label>
      </div>
      <div>
        <div class="chart" id="explorer-scatter" style="height:300px"></div>
        <div id="explorer-table" style="margin-top:14px"></div>
        <div class="pager">
          <button id="pg-prev">←</button>
          <span id="pg-info"></span>
          <button id="pg-next">→</button>
        </div>
      </div>
    </div>`;
  for (const id of ["f-feasible", "f-fidelity", "f-npu", "f-saving", "f-sort", "f-desc"]) {
    document.getElementById(id).addEventListener("change", () => {
      explorerState.page = 1;
      refreshExplorer();
    });
  }
  document.getElementById("pg-prev").onclick = () => {
    if (explorerState.page > 1) { explorerState.page--; refreshExplorer(); }
  };
  document.getElementById("pg-next").onclick = () => {
    explorerState.page++; refreshExplorer();
  };
  await refreshExplorer();
}

async function refreshExplorer() {
  const params = new URLSearchParams({
    batch_id: currentBatch,
    page: explorerState.page,
    page_size: 50,
    sort: document.getElementById("f-sort").value,
    descending: document.getElementById("f-desc").checked,
  });
  const feasible = document.getElementById("f-feasible").value;
  if (feasible) params.set("feasible", feasible);
  const fidelity = document.getElementById("f-fidelity").value;
  if (fidelity) params.set("fidelity", fidelity);
  const npu = document.getElementById("f-npu").value;
  if (npu) params.set("has_npu", npu);
  const saving = document.getElementById("f-saving").value;
  if (saving !== "") params.set("min_saving", saving);

  const data = await api(`/api/scenarios?${params}`);
  const pages = Math.max(1, Math.ceil(data.total / data.page_size));
  if (explorerState.page > pages) { explorerState.page = pages; }
  document.getElementById("pg-info").textContent =
    `page ${data.page}/${pages} · ${data.total} scenarios`;

  const scatter = makeChart(document.getElementById("explorer-scatter"));
  scatter.setOption({
    title: { text: "p99 TTFT vs avg power (current page)", left: 8, textStyle: { fontSize: 13 } },
    tooltip: { formatter: (p) => p.data[3] },
    xAxis: { name: "p99 TTFT (ms)", type: "value" },
    yAxis: { name: "avg power (W)", type: "value" },
    series: [
      { type: "scatter", name: "feasible", itemStyle: { color: "#2b8a3e" },
        data: data.rows.filter((r) => r.feasible && r.p99_ttft_ms != null)
          .map((r) => [r.p99_ttft_ms, r.avg_power_w, r.active_devices, r.scenario_id]) },
      { type: "scatter", name: "infeasible", itemStyle: { color: "#c92a2a" },
        data: data.rows.filter((r) => !r.feasible && r.p99_ttft_ms != null)
          .map((r) => [r.p99_ttft_ms, r.avg_power_w, r.active_devices, r.scenario_id]) },
    ],
  });
  scatter.on("click", (p) => { location.hash = `#/scenario/${p.data[3]}`; });

  const rowsHtml = data.rows.map((r) => `
    <tr data-id="${esc(r.scenario_id)}">
      <td>${esc(r.scenario_id)}</td>
      <td>${esc(r.cluster_id)}${r.has_npu ? " " + badge("NPU", "muted") : ""}</td>
      <td>${esc(r.service_id)}</td>
      <td>${r.feasible ? "✓" : badge("infeasible", "warn")}</td>
      <td class="num">${fmt(r.p99_ttft_ms, 0)}</td>
      <td class="num">${fmt(r.p99_tpot_ms)}</td>
      <td class="num">${fmt(r.avg_power_w)}</td>
      <td class="num">${fmt(r.power_saving_pct)}</td>
      <td>${honestyBadges(r)}</td>
    </tr>`).join("");
  document.getElementById("explorer-table").innerHTML = `
    <table><thead><tr>
      <th>scenario</th><th>cluster</th><th>service</th><th>feasible</th>
      <th class="num">TTFT p99</th><th class="num">TPOT p99</th>
      <th class="num">avg W</th><th class="num">saving %</th><th>labels</th>
    </tr></thead><tbody>${rowsHtml}</tbody></table>`;
  document.querySelectorAll("#explorer-table tbody tr").forEach((tr) => {
    tr.onclick = () => { location.hash = `#/scenario/${tr.dataset.id}`; };
  });
}

// -------------------------------------------------------- ③ scenario detail

async function renderScenario(scenarioId) {
  await loadBatches();
  const d = await api(`/api/scenarios/${encodeURIComponent(scenarioId)}`);
  const row = d.row;
  const out = d.document.planner_output;
  const cal = d.document.calibration || {};
  const slo = d.service;

  const violations = (out.violated_constraints || []).map((v) => `
    <tr><td>${esc(v.metric)}</td>
      <td class="num">${fmt(v.target)}</td><td class="num">${fmt(v.predicted)}</td></tr>`).join("");
  const suggestions = (out.suggestions || [])
    .map((s) => `<div class="suggestion">${esc(s)}</div>`).join("");

  main.innerHTML = `
    <h2 style="margin:4px 0 12px">${esc(scenarioId)} ${honestyBadges(row)}</h2>
    <div class="detail-grid">
      <div class="panel">
        <h3>Cluster ${esc(row.cluster_id)} topology
          <small>(bold border = in plan, P/D/A = role)</small></h3>
        <div id="topo" style="height:420px"></div>
      </div>
      <div>
        <div class="panel">
          <h3>Summary</h3>
          <div class="kv">
            <span class="k">feasible</span><span>${row.feasible ? "✓ yes" : "✗ no"}</span>
            <span class="k">p99 TTFT</span>
            <span>${fmt(row.p99_ttft_ms, 0)} ms (SLO ${fmt(slo.ttft_p99_ms, 0)})</span>
            <span class="k">p99 TPOT</span>
            <span>${fmt(row.p99_tpot_ms)} ms (SLO ${fmt(slo.tpot_p99_ms)})</span>
            <span class="k">avg / peak power</span>
            <span>${fmt(row.avg_power_w)} / ${fmt(row.peak_power_w)} W (cap ${fmt(slo.power_cap_w, 0)})</span>
            <span class="k">power saving</span>
            <span>${fmt(row.power_saving_pct)}% <small>(baseline ${fmt(row.baseline_power_w)} W · ${esc(row.baseline_note || "")})</small></span>
            <span class="k">tokens/J</span><span>${fmt(row.tokens_per_joule, 3)}</span>
            <span class="k">active devices</span><span>${row.active_devices ?? "–"}</span>
            <span class="k">calibration</span>
            <span>${cal.calibrated ? `margins ttft ${fmt(cal.ttft_margin_percent)}% / tpot ${fmt(cal.tpot_margin_percent)}%` : badge("calibrated: false", "muted")}</span>
          </div>
        </div>
        <div class="panel" style="margin-top:14px">
          <h3>Pareto: energy vs p99 TTFT</h3>
          <div id="pareto" style="height:220px"></div>
        </div>
        ${d.verification ? `
        <div class="panel" style="margin-top:14px">
          <h3>Full-sim verification ${badge("sim", "sim")}</h3>
          <div class="kv">
            <span class="k">sim p99 TTFT / TPOT</span>
            <span>${fmt(d.verification.sim_p99_ttft_ms, 0)} ms / ${fmt(d.verification.sim_p99_tpot_ms)} ms</span>
            <span class="k">fast-path error</span>
            <span>ttft ${fmt(d.verification.err_ttft_pct)}% · tpot ${fmt(d.verification.err_tpot_pct)}% · power ${fmt(d.verification.err_power_pct)}%</span>
            <span class="k">selection flipped</span>
            <span>${d.verification.selection_flipped ? badge("YES - regret " + fmt(d.verification.regret_energy_pct) + "%", "warn") : "no"}</span>
            <span class="k">feasibility flipped</span>
            <span>${d.verification.feasibility_flipped ? badge("YES", "warn") : "no"}</span>
          </div>
        </div>` : ""}
      </div>
    </div>
    ${row.feasible ? "" : `
    <div class="panel" style="margin-top:14px">
      <h3>Infeasibility diagnosis</h3>
      <p>${esc(out.reason || "")}</p>
      ${violations ? `<table class="violations"><thead><tr>
        <th>metric</th><th class="num">target</th><th class="num">predicted</th>
      </tr></thead><tbody>${violations}</tbody></table>` : ""}
      ${suggestions}
    </div>`}
    <details><summary>Plan document (JSON)</summary>
      <pre>${esc(JSON.stringify(out.recommended ?? out.closest_plan ?? {}, null, 2))}</pre>
    </details>
    <details><summary>Provenance</summary>
      <pre>${esc(JSON.stringify(out.provenance ?? {}, null, 2))}</pre>
    </details>`;

  renderTopology(d.graph);
  renderPareto(out);
}

function renderTopology(graph) {
  const roleLabel = { prefill: "P", decode: "D", aggregated: "A" };
  const classes = [...new Set(graph.nodes.map((n) => n.cls))];
  const palette = ["#4c6ef5", "#f76707", "#37b24d", "#ae3ec9", "#0ca678", "#e8590c"];
  const chart = makeChart(document.getElementById("topo"));
  chart.setOption({
    tooltip: {
      formatter: (p) => p.dataType === "edge"
        ? `${p.data.source} → ${p.data.target}<br>${p.data.linkType} ${p.data.bw} Gbps`
        : `${p.data.id}<br>${p.data.cls} · ${p.data.state}` +
          (p.data.role ? `<br>role: ${p.data.role}` : ""),
    },
    legend: { data: classes, bottom: 0 },
    series: [{
      type: "graph", layout: "force", roam: true,
      force: { repulsion: 260, edgeLength: 90 },
      categories: classes.map((c, i) => ({
        name: c, itemStyle: { color: palette[i % palette.length] },
      })),
      label: { show: true, fontSize: 10, formatter: (p) =>
        (p.data.role ? roleLabel[p.data.role] + " · " : "") + p.data.device },
      data: graph.nodes.map((n) => ({
        id: n.id, name: n.id, device: n.device, cls: n.cls, state: n.state,
        role: n.role, category: classes.indexOf(n.cls),
        symbol: n.kind === "nic" ? "diamond" : "circle",
        symbolSize: n.kind === "nic" ? 16 : 34,
        itemStyle: {
          opacity: n.state === "FREE" ? 1 : 0.35,
          borderColor: n.in_plan ? "#1c7ed6" : "#adb5bd",
          borderWidth: n.in_plan ? 4 : 1,
        },
      })),
      links: graph.links.map((l) => ({
        source: l.src, target: l.dst, linkType: l.type, bw: l.bandwidth_gbps,
        lineStyle: {
          width: Math.max(1, Math.log2(l.bandwidth_gbps / 8)),
          color: l.in_plan ? "#1c7ed6" : "#ced4da",
        },
      })),
    }],
  });
}

function renderPareto(out) {
  const points = [];
  if (out.recommended) {
    points.push({ plan: out.recommended.plan, kind: "recommended" });
    for (const alt of out.alternatives || []) points.push({ plan: alt.plan, kind: "alternative" });
  }
  const data = points
    .filter((p) => p.plan.predicted.total_energy_j != null)
    .map((p) => ({
      value: [p.plan.predicted.p99_ttft_ms, p.plan.predicted.total_energy_j],
      name: p.plan.candidate.id,
      itemStyle: { color: p.kind === "recommended" ? "#e8590c" : "#748ffc" },
      symbolSize: p.kind === "recommended" ? 16 : 9,
    }));
  const chart = makeChart(document.getElementById("pareto"));
  chart.setOption({
    tooltip: { formatter: (p) => `${p.data.name}<br>${fmt(p.data.value[0], 0)} ms · ${fmt(p.data.value[1], 0)} J` },
    xAxis: { name: "p99 TTFT (ms)", type: "value" },
    yAxis: { name: "energy (J)", type: "value" },
    series: [{ type: "scatter", data }],
  });
}

// --------------------------------------------------------------------- boot

route();
