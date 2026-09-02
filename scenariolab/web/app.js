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
    else if (page === "planner") await renderPlanner();
    else if (page === "verification") await renderVerification();
    else if (page === "builder") await renderBuilder();
    else if (page === "clusters") await renderClusters();
    else if (page === "workspace" && arg) await renderWorkspace(arg);
    else if (page === "workspaces") await renderWorkspaces();
    else await renderDashboard();
  } catch (err) {
    console.error(err);
  }
}

async function apiSend(method, path, body) {
  banner.classList.add("hidden");
  const res = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail;
    try { detail = (await res.json()).detail; } catch { detail = await res.text(); }
    const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    err.status = res.status;
    throw err;
  }
  return res.json();
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

  renderTopologyInto("topo", d.graph, {});
  renderPareto(out);
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

// ------------------------------------------------------ ④ interactive planner

async function renderPlanner() {
  await loadBatches();
  const clusters = await api("/api/clusters");
  main.innerHTML = `
    <div class="layout">
      <div class="sidebar">
        <label>cluster</label>
        <select id="p-cluster">${clusters.map((c) =>
          `<option value="${esc(c.cluster_id)}">${esc(c.cluster_id)} · ${esc(c.classes.join(","))} (${c.num_free_accels} free)</option>`
        ).join("")}</select>
        <label>rps</label><input id="p-rps" type="number" value="5" step="0.5">
        <label>input p50 (tokens)</label><input id="p-in" type="number" value="512">
        <label>output p50 (tokens)</label><input id="p-out" type="number" value="128">
        <label>TTFT p99 (ms)</label><input id="p-ttft" type="number" value="2000">
        <label>TPOT p99 (ms)</label><input id="p-tpot" type="number" value="60">
        <label>power cap (W)</label><input id="p-cap" type="number" value="2000">
        <div style="margin-top:14px">
          <button id="p-go" style="width:100%;padding:8px">최적 배치 계산</button>
        </div>
        <div class="chart" id="p-preview" style="height:200px;margin-top:14px"></div>
      </div>
      <div id="p-result">
        <div class="panel">Pick a cluster, set the SLO, and compute. Results are
        ${badge("surrogate", "surrogate")} fast-path predictions - not verified
        by full simulation.</div>
      </div>
    </div>`;

  const preview = async () => {
    const id = document.getElementById("p-cluster").value;
    const detail = await api(`/api/clusters/${encodeURIComponent(id)}`);
    disposeChartsIn("p-preview");
    renderTopologyInto("p-preview", detail.graph, { compact: true });
  };
  document.getElementById("p-cluster").addEventListener("change", preview);
  await preview();

  document.getElementById("p-go").onclick = async () => {
    const body = {
      cluster_id: document.getElementById("p-cluster").value,
      slo: {
        rps: +document.getElementById("p-rps").value || null,
        input_p50: +document.getElementById("p-in").value || null,
        output_p50: +document.getElementById("p-out").value || null,
        ttft_p99_ms: +document.getElementById("p-ttft").value,
        tpot_p99_ms: +document.getElementById("p-tpot").value,
        power_cap_w: +document.getElementById("p-cap").value || null,
      },
    };
    const target = document.getElementById("p-result");
    target.innerHTML = `<div class="panel">computing…</div>`;
    let res;
    try {
      const raw = await fetch("/api/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!raw.ok) {
        const detail = (await raw.json()).detail;
        target.innerHTML = `<div class="panel">${badge("request rejected", "warn")} ${esc(JSON.stringify(detail))}</div>`;
        return;
      }
      res = await raw.json();
    } catch (err) {
      showError(`plan request failed: ${err.message}`);
      return;
    }
    renderPlanResult(target, res);
  };
}

function renderPlanResult(target, res) {
  const out = res.planner_output;
  const plan = out.recommended ? out.recommended.plan : out.closest_plan;
  const m = plan ? plan.predicted : null;
  const labels =
    fidelityBadge(res.fidelity) +
    (res.npu_extrapolated ? " " + badge("⚠ NPU extrapolated", "warn") : "") +
    (res.calibrated ? "" : " " + badge("calibrated: false", "muted")) +
    (res.truncated ? " " + badge(`truncated top-K`, "muted") : "");
  const violations = (out.violated_constraints || []).map((v) => `
    <tr><td>${esc(v.metric)}</td><td class="num">${fmt(v.target)}</td>
    <td class="num">${fmt(v.predicted)}</td></tr>`).join("");
  target.innerHTML = `
    <div class="panel">
      <h3>${out.feasible ? "Recommended plan" : "INFEASIBLE - closest plan"} ${labels}</h3>
      <p style="color:#7a5b00;background:#fff8e1;padding:6px 10px;border-radius:6px">
        이 결과는 ${esc(res.fidelity)} 예측이며 full sim 검증을 거치지 않았습니다
        (elapsed ${fmt(res.elapsed_s, 2)}s · seed ${res.seed} · ${res.num_requests} reqs).
      </p>
      ${m ? `<div class="kv">
        <span class="k">candidate</span><span>${esc(plan.candidate.id)}</span>
        <span class="k">p99 TTFT / TPOT</span>
        <span>${fmt(m.p99_ttft_ms, 0)} ms / ${fmt(m.p99_tpot_ms)} ms</span>
        <span class="k">avg / peak power</span>
        <span>${fmt(m.average_power_w)} / ${fmt(m.peak_power_w)} W</span>
        <span class="k">tokens/J</span><span>${fmt(m.tokens_per_joule, 3)}</span>
        <span class="k">devices</span><span>${plan.candidate.assignments
          .map((a) => `${esc(a.island_id)} tp${a.tp_size}×dp${a.dp_replicas} (${esc(a.role)})`)
          .join(", ")}</span>
      </div>` : `<p>${esc(out.reason || "no candidate at all")}</p>`}
      ${violations ? `<h3 style="margin-top:12px">violated constraints</h3>
        <table class="violations"><thead><tr><th>metric</th>
        <th class="num">target</th><th class="num">predicted</th></tr></thead>
        <tbody>${violations}</tbody></table>` : ""}
      ${(out.suggestions || []).map((s) => `<div class="suggestion">${esc(s)}</div>`).join("")}
    </div>
    <div class="panel" style="margin-top:14px">
      <h3>Placement on topology</h3>
      <div id="p-topo" style="height:380px"></div>
    </div>
    <details><summary>Full planner output (JSON)</summary>
      <pre>${esc(JSON.stringify(out, null, 2))}</pre></details>`;
  renderTopologyInto("p-topo", res.graph, {});
}

// ------------------------------------------------------------ ⑤ verification

async function renderVerification() {
  await loadBatches();
  const data = await api(
    `/api/verification?batch_id=${encodeURIComponent(currentBatch)}`
  );
  const s = data.stats;
  main.innerHTML = `
    <div class="cards">
      <div class="card"><div class="label">Verified (full sim)</div>
        <div class="value">${s.verified}</div></div>
      <div class="card"><div class="label">Selection flips</div>
        <div class="value">${s.selection_flips}</div></div>
      <div class="card"><div class="label">Feasibility flips</div>
        <div class="value">${s.feasibility_flips}</div></div>
      <div class="card"><div class="label">|err| p50 / p95 (TPOT)</div>
        <div class="value">${fmt(s.err_tpot_pct_p50)} / ${fmt(s.err_tpot_pct_p95)}%</div></div>
      <div class="card"><div class="label">|err| p50 / p95 (power)</div>
        <div class="value">${fmt(s.err_power_pct_p50)} / ${fmt(s.err_power_pct_p95)}%</div></div>
    </div>
    <div class="chart-row">
      <div class="chart" id="v-power"></div>
      <div class="chart" id="v-ttft"></div>
    </div>
    <div class="panel" style="margin-top:14px">
      <h3>Selection flips (fast-path picked a plan full sim would not)</h3>
      <div id="v-flips">${data.flipped.length
        ? data.flipped.map((id) =>
            `<a href="#/scenario/${esc(id)}">${esc(id)}</a>`).join(" · ")
        : "none in this batch"}</div>
    </div>`;

  const scatter = (elId, title, fastKey, simKey) => {
    const points = data.points
      .filter((p) => p[fastKey] != null && p[simKey] != null)
      .map((p) => ({ value: [p[fastKey], p[simKey]], name: p.scenario_id }));
    const values = points.flatMap((p) => p.value);
    const max = values.length ? Math.max(...values) * 1.05 : 1;
    const chart = makeChart(document.getElementById(elId));
    chart.setOption({
      title: { text: title, left: 8, textStyle: { fontSize: 13 } },
      tooltip: { formatter: (p) => `${p.data.name}<br>fast ${fmt(p.data.value[0])} · sim ${fmt(p.data.value[1])}` },
      xAxis: { name: "fast path", type: "value", max },
      yAxis: { name: "full sim", type: "value", max },
      series: [
        { type: "scatter", data: points },
        { type: "line", data: [[0, 0], [max, max]], showSymbol: false,
          lineStyle: { type: "dashed", color: "#adb5bd" }, silent: true },
      ],
    });
    chart.on("click", (p) => {
      if (p.data.name) location.hash = `#/scenario/${p.data.name}`;
    });
  };
  scatter("v-power", "avg power: fast vs sim (W)", "fast_avg_power_w", "sim_avg_power_w");
  scatter("v-ttft", "p99 TTFT: fast vs sim (ms)", "fast_p99_ttft_ms", "sim_p99_ttft_ms");
}

// ------------------------------------------------------------ topology reuse

function disposeChartsIn(elId) {
  const dom = document.getElementById(elId);
  if (dom) {
    const existing = echarts.getInstanceByDom(dom);
    if (existing) existing.dispose();
  }
}

function renderTopologyInto(elId, graph, { compact = false, serviceColors = false } = {}) {
  const holder = document.getElementById(elId);
  if (!holder) return;
  const roleLabel = { prefill: "P", decode: "D", aggregated: "A" };
  const classes = [...new Set(graph.nodes.map((n) => n.cls))];
  const palette = ["#4c6ef5", "#f76707", "#37b24d", "#ae3ec9", "#0ca678", "#e8590c"];
  const chart = makeChart(holder);
  chart.setOption({
    tooltip: {
      formatter: (p) => p.dataType === "edge"
        ? `${p.data.source} → ${p.data.target}<br>${p.data.linkType} ${p.data.bw} Gbps`
        : `${p.data.id}<br>${p.data.cls} · ${p.data.state}` +
          (p.data.role ? `<br>role: ${p.data.role}` : ""),
    },
    legend: compact ? undefined : { data: classes, bottom: 0 },
    series: [{
      type: "graph", layout: "force", roam: true,
      force: { repulsion: compact ? 120 : 260, edgeLength: compact ? 40 : 90 },
      categories: classes.map((c, i) => ({
        name: c, itemStyle: { color: palette[i % palette.length] },
      })),
      label: { show: !compact, fontSize: 10, formatter: (p) =>
        (p.data.role ? roleLabel[p.data.role] + " · " : "") + p.data.device },
      data: graph.nodes.map((n) => ({
        id: n.id, name: n.id, device: n.device, cls: n.cls, state: n.state,
        role: n.role, category: classes.indexOf(n.cls),
        symbol: n.kind === "nic" ? "diamond" : "circle",
        symbolSize: n.kind === "nic" ? (compact ? 8 : 16) : (compact ? 18 : 34),
        itemStyle: {
          // Workspace view: device color = its service; grey = FREE (FR-W2).
          ...(serviceColors && n.kind === "accelerator"
            ? { color: n.service_color || "#ced4da" } : {}),
          opacity: n.state === "FREE" ? 1 : (serviceColors ? 1 : 0.35),
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

// ------------------------------------------------------------- ⑥ builder

const BUILDER_CLASSES = ["a40", "a5000", "rtxpro6000", "furiosa_rngd_card"];
const LINK_PRESETS = ["ib_100g", "ib_400g"];

function builderRow(cls = "a40", perNode = 4, nodes = 1) {
  return `
    <div class="builder-row">
      <select class="b-class">${BUILDER_CLASSES.map((c) =>
        `<option ${c === cls ? "selected" : ""}>${c}</option>`).join("")}</select>
      per node <input class="b-per" type="number" value="${perNode}" min="1" max="8" style="width:52px">
      × nodes <input class="b-nodes" type="number" value="${nodes}" min="1" style="width:52px">
      <button class="b-del">✕</button>
    </div>`;
}

async function renderBuilder() {
  main.innerHTML = `
    <div class="layout">
      <div class="sidebar" style="min-width:330px">
        <h3 style="margin:0 0 8px">Cluster Builder</h3>
        <label>name</label><input id="b-name" value="my-cluster">
        <label>node groups</label>
        <div id="b-rows">${builderRow()}</div>
        <button id="b-add" style="margin-top:6px">+ add group</button>
        <label style="margin-top:14px">inter-node link</label>
        <label><input type="radio" name="b-link" value="preset" checked style="width:auto">
          preset <select id="b-preset">${LINK_PRESETS.map((p) => `<option>${p}</option>`).join("")}</select></label>
        <label><input type="radio" name="b-link" value="custom" style="width:auto"> custom</label>
        <div id="b-custom" style="padding-left:16px;opacity:.5">
          type <select id="b-type"><option>ETHERNET</option><option>INFINIBAND</option>
            <option>NVLINK</option><option>PCIE</option><option>HCCS</option></select>
          bw <input id="b-bw" type="number" value="50" style="width:70px"> Gbps
          latency <input id="b-lat" type="number" value="8000" style="width:80px"> ns
          <div>${badge("custom values are labelled user_defined", "muted")}</div>
        </div>
        <button id="b-go" style="margin-top:14px;width:100%;padding:8px">클러스터 생성</button>
      </div>
      <div id="b-result"><div class="panel">Define node groups and the inter-node
        fabric, then build. Intra-node links are dictated by each class's
        profile (PCIe root complexes of 4 + NVLink where the hardware has it).</div></div>
    </div>`;
  document.getElementById("b-add").onclick = () => {
    document.getElementById("b-rows").insertAdjacentHTML("beforeend", builderRow());
    wireDelete();
  };
  const wireDelete = () => {
    document.querySelectorAll(".b-del").forEach((b) => {
      b.onclick = () => b.parentElement.remove();
    });
  };
  wireDelete();
  document.querySelectorAll("input[name=b-link]").forEach((r) => {
    r.onchange = () => {
      document.getElementById("b-custom").style.opacity =
        document.querySelector("input[name=b-link]:checked").value === "custom" ? "1" : ".5";
    };
  });
  document.getElementById("b-go").onclick = async () => {
    const nodes = [...document.querySelectorAll(".builder-row")].map((row) => ({
      class: row.querySelector(".b-class").value,
      count_per_node: +row.querySelector(".b-per").value,
      num_nodes: +row.querySelector(".b-nodes").value,
    }));
    const mode = document.querySelector("input[name=b-link]:checked").value;
    const inter_node = mode === "preset"
      ? { preset: document.getElementById("b-preset").value }
      : { custom: {
          type: document.getElementById("b-type").value,
          bandwidth_gbps: +document.getElementById("b-bw").value,
          latency_ns: +document.getElementById("b-lat").value,
        } };
    const target = document.getElementById("b-result");
    target.innerHTML = `<div class="panel">building…</div>`;
    let data;
    try {
      data = await apiSend("POST", "/api/clusters/build", {
        name: document.getElementById("b-name").value,
        nodes, interconnect: { inter_node },
      });
    } catch (err) {
      target.innerHTML = `<div class="panel">${badge("rejected", "warn")} ${esc(err.message)}</div>`;
      return;
    }
    const c = data.cluster;
    target.innerHTML = `
      <div class="panel">
        <h3>${esc(c.cluster_id)} ${data.already_existed ? badge("already existed", "muted") : badge("created", "envelope")}</h3>
        <div class="kv">
          <span class="k">accelerators</span><span>${c.num_accels} (${esc(c.classes.join(", "))})</span>
          <span class="k">nodes / islands</span><span>${c.num_nodes} / ${c.num_islands}</span>
          <span class="k">fabric</span><span>${esc(c.link_summary || "")}
            ${(c.link_summary || "").includes("user_defined") ? badge("user_defined", "warn") : ""}</span>
        </div>
        ${data.warnings.map((w) => `<div class="suggestion">⚠ ${esc(w)}</div>`).join("")}
        <h3 style="margin-top:12px">Islands (TP candidates)</h3>
        <table><thead><tr><th>island</th><th>devices</th><th>TP</th></tr></thead><tbody>
        ${data.islands.map((i) => `<tr><td>${esc(i.id)}</td>
          <td>${i.accelerators} × ${esc(i.model)}</td>
          <td>${i.tp_candidates.join(", ")}</td></tr>`).join("")}
        </tbody></table>
        <div style="margin-top:12px">
          <a href="#/clusters">→ open the catalog</a> ·
          <button id="b-ws">이 클러스터로 Workspace 시작</button>
        </div>
      </div>
      <div class="panel" style="margin-top:14px"><h3>Topology</h3>
        <div id="b-topo" style="height:360px"></div></div>`;
    const detail = await api(`/api/clusters/${encodeURIComponent(c.cluster_id)}`);
    renderTopologyInto("b-topo", detail.graph, {});
    document.getElementById("b-ws").onclick = () => startWorkspace(c.cluster_id);
  };
}

async function startWorkspace(clusterId) {
  const name = prompt("workspace name?", clusterId.replace(/^custom-/, ""));
  if (!name) return;
  const workspace = await apiSend("POST", "/api/workspaces", {
    cluster_id: clusterId, name,
  });
  location.hash = `#/workspace/${workspace.workspace_id}`;
}

// ------------------------------------------------------------- ⑦ clusters

async function renderClusters() {
  const params = new URLSearchParams(location.hash.split("?")[1] || "");
  const origin = params.get("origin") || "";
  const rows = await api(`/api/clusters${origin ? `?origin=${origin}` : ""}`);
  main.innerHTML = `
    <div style="margin-bottom:10px">
      ${["", "random", "custom"].map((o) =>
        `<a href="#/clusters${o ? `?origin=${o}` : ""}"
            class="badge ${o === origin ? "sim" : "muted"}"
            style="margin-right:6px;text-decoration:none">${o || "all"}</a>`).join("")}
    </div>
    <div class="layout" style="grid-template-columns: 1fr 380px">
      <div><table><thead><tr>
        <th>cluster</th><th>origin</th><th>nodes</th><th>accels</th>
        <th>islands</th><th>fabric</th><th>ws</th><th></th>
      </tr></thead><tbody>
      ${rows.map((c) => `<tr data-id="${esc(c.cluster_id)}">
        <td>${esc(c.cluster_id)}</td>
        <td>${badge(c.origin, c.origin === "custom" ? "sim" : "muted")}</td>
        <td class="num">${c.num_nodes}</td>
        <td class="num">${c.num_accels}${c.has_npu ? " " + badge("NPU", "muted") : ""}</td>
        <td class="num">${c.num_islands}</td>
        <td>${esc(c.link_summary || "")}
          ${(c.link_summary || "").includes("user_defined") ? badge("user_defined", "warn") : ""}</td>
        <td class="num">${c.workspaces}</td>
        <td><button class="c-ws" data-id="${esc(c.cluster_id)}">Workspace 시작</button></td>
      </tr>`).join("")}
      </tbody></table></div>
      <div class="panel"><h3 id="c-preview-title">preview</h3>
        <div id="c-preview" style="height:420px"></div></div>
    </div>`;
  document.querySelectorAll("tbody tr").forEach((tr) => {
    tr.onclick = async () => {
      const detail = await api(`/api/clusters/${encodeURIComponent(tr.dataset.id)}`);
      document.getElementById("c-preview-title").textContent = tr.dataset.id;
      disposeChartsIn("c-preview");
      renderTopologyInto("c-preview", detail.graph, {});
    };
  });
  document.querySelectorAll(".c-ws").forEach((btn) => {
    btn.onclick = (event) => {
      event.stopPropagation();
      startWorkspace(btn.dataset.id);
    };
  });
}

// ------------------------------------------------------------ ⑧ workspace

async function renderWorkspaces() {
  const rows = await api("/api/workspaces");
  main.innerHTML = `
    <div class="panel"><h3>Workspaces</h3>
    ${rows.length ? `<table><thead><tr>
      <th>workspace</th><th>name</th><th>cluster</th><th>placed</th><th>created</th>
    </tr></thead><tbody>
    ${rows.map((w) => `<tr data-id="${esc(w.workspace_id)}">
      <td>${esc(w.workspace_id)}</td><td>${esc(w.name)}</td>
      <td>${esc(w.cluster_id)}${w.cluster_changed ? " " + badge("원본 변경됨", "warn") : ""}</td>
      <td class="num">${w.placed_count}</td><td>${esc(w.created_at)}</td>
    </tr>`).join("")}</tbody></table>`
    : `아직 workspace가 없습니다 — <a href="#/clusters">⑦ Clusters</a>에서 시작하세요.`}
    </div>`;
  main.querySelectorAll("tbody tr").forEach((tr) => {
    tr.onclick = () => { location.hash = `#/workspace/${tr.dataset.id}`; };
  });
}

const SERVICE_PALETTE = [
  "#4c6ef5", "#f76707", "#37b24d", "#ae3ec9", "#0ca678",
  "#e8590c", "#1098ad", "#d6336c", "#74b816", "#7048e8",
];

function verdictBadge(ok, calibrated) {
  if (ok === null || ok === undefined) return "–";
  const cls = calibrated ? (ok ? "envelope" : "warn") : "muted";
  return `<span class="badge ${cls}" title="${calibrated
    ? "robust (calibration margin applied)"
    : "raw prediction - calibrated: false"}">${ok ? "✓" : "✗"}</span>`;
}

async function renderWorkspace(workspaceId) {
  const s = await api(`/api/workspaces/${encodeURIComponent(workspaceId)}/summary`);
  const rows = s.placements;
  const active = rows.filter((p) => p.status === "PLACED");
  const gauges = Object.entries(s.resources.by_class).map(([cls, v]) => `
    <div style="margin:4px 0">${esc(cls)}
      <div style="background:#e9ecef;border-radius:4px;height:10px">
        <div style="width:${v.total ? (100 * (v.total - v.free) / v.total) : 0}%;
             background:#4c6ef5;height:10px;border-radius:4px"></div></div>
      <small>${v.free}/${v.total} free</small></div>`).join("");
  const cap = s.power.total_power_cap_w;
  main.innerHTML = `
    <h2 style="margin:4px 0 4px">${esc(s.workspace.name)}
      <small>(${esc(workspaceId)} · cluster ${esc(s.cluster.cluster_id)})</small>
      ${s.workspace.cluster_changed ? badge("원본 변경됨", "warn") : ""}</h2>
    <div class="suggestion">ⓘ ${esc(s.interference_notice)}</div>
    <div class="layout" style="grid-template-columns: 1.1fr .9fr">
      <div>
        <div class="panel"><h3>Topology (색 = 서비스, 회색 = FREE)</h3>
          <div id="w-topo" style="height:400px"></div></div>
        <div class="panel" style="margin-top:14px"><h3>잔여 자원</h3>${gauges}</div>
      </div>
      <div>
        <div class="panel">
          <h3>서비스 추가</h3>
          <label><input type="radio" name="w-mode" value="random" checked style="width:auto">
            랜덤 SLO 개수 <input id="w-count" type="number" value="3" min="1" style="width:52px">
            seed <input id="w-seed" type="number" value="42" style="width:80px"></label>
          <label><input type="radio" name="w-mode" value="user" style="width:auto"> 직접 입력</label>
          <div id="w-user" style="padding-left:16px;opacity:.5">
            rps <input id="w-rps" type="number" value="5" style="width:60px">
            in p50 <input id="w-in" type="number" value="512" style="width:64px">
            out p50 <input id="w-out" type="number" value="128" style="width:64px"><br>
            TTFT p99 <input id="w-ttft" type="number" value="2000" style="width:70px"> ms
            TPOT p99 <input id="w-tpot" type="number" value="60" style="width:60px"> ms
            cap <input id="w-cap" type="number" value="2000" style="width:70px"> W
          </div>
          <button id="w-place" style="margin-top:10px;width:100%;padding:7px">배치 계획 실행</button>
          <div id="w-preview"></div>
        </div>
        <div class="panel" style="margin-top:14px">
          <h3>총 전력 (PLACED 예측 합)</h3>
          <div class="kv">
            <span class="k">avg 합</span><span>${fmt(s.power.sum_avg_w)} W</span>
            <span class="k">peak 합</span>
            <span>${fmt(s.power.sum_peak_w)} W ${badge("동시 peak 가정의 보수적 상한", "muted")}</span>
            ${cap ? `<span class="k">workspace cap</span><span>${fmt(cap, 0)} W</span>` : ""}
          </div>
          ${cap ? `<div style="background:#e9ecef;border-radius:4px;height:12px;margin-top:6px">
            <div style="width:${Math.min(100, 100 * s.power.sum_peak_w / cap)}%;
              background:${s.power.sum_peak_w > cap ? "#c92a2a" : "#37b24d"};
              height:12px;border-radius:4px"></div></div>` : ""}
        </div>
      </div>
    </div>
    <div class="panel" style="margin-top:14px">
      <h3>서비스 목록 (${active.length} placed / ${rows.length} total)</h3>
      <table><thead><tr>
        <th>#</th><th>service</th><th>SLO (TTFT/TPOT ms)</th><th>예측 (TTFT/TPOT ms)</th>
        <th>SLO</th><th class="num">avg W</th><th>labels</th><th>status</th><th></th>
      </tr></thead><tbody id="w-rows">
      ${rows.map((p) => `
        <tr data-id="${esc(p.placement_id)}"
            ${p.status === "REJECTED" ? 'style="opacity:.6"' : ""}>
          <td><span style="display:inline-block;width:12px;height:12px;border-radius:3px;
            background:${p.status === "PLACED" ? SERVICE_PALETTE[p.seq % SERVICE_PALETTE.length] : "#ced4da"}"></span>
            ${p.seq}</td>
          <td>${esc(p.service.model.split("/").pop())} <small>(${esc(p.service.origin)})</small></td>
          <td>${fmt(p.service.ttft_p99_ms, 0)} / ${fmt(p.service.tpot_p99_ms, 0)}</td>
          <td>${p.p99_ttft_ms != null ? `${fmt(p.p99_ttft_ms, 0)} / ${fmt(p.p99_tpot_ms)}` : "–"}</td>
          <td>${verdictBadge(p.slo_ttft_ok, p.calibrated)}${verdictBadge(p.slo_tpot_ok, p.calibrated)}</td>
          <td class="num">${fmt(p.avg_power_w)}</td>
          <td>${p.fidelity ? fidelityBadge(p.fidelity) : ""}
            ${p.npu_extrapolated ? badge("⚠ NPU", "warn") : ""}
            ${p.shared_fabric_warning ? badge("⚠ shared fabric", "warn") : ""}</td>
          <td>${p.status === "REJECTED"
            ? `${badge("REJECTED", "warn")} <a href="#" class="w-diag" data-id="${esc(p.placement_id)}">진단 보기</a>`
            : esc(p.status)}</td>
          <td>
            <button class="w-detail" data-id="${esc(p.placement_id)}">상세</button>
            ${p.status === "PLACED"
              ? `<button class="w-remove" data-id="${esc(p.placement_id)}">제거</button>` : ""}
            ${p.status === "PLANNING"
              ? `<button class="w-confirm" data-id="${esc(p.placement_id)}">확정</button>` : ""}
          </td>
        </tr>
        <tr class="w-expand hidden" data-for="${esc(p.placement_id)}">
          <td colspan="9"><pre class="w-expand-body"></pre></td>
        </tr>`).join("")}
      </tbody></table>
    </div>`;

  // Topology with per-service colors (FR-W2: color = seq, stable).
  const graph = s.graph;
  graph.nodes.forEach((n) => {
    const overlay = s.topology_overlay[n.id];
    n.in_plan = Boolean(overlay);
    n.role = overlay ? overlay.role : null;
    n.service_color = overlay
      ? SERVICE_PALETTE[overlay.color_index % SERVICE_PALETTE.length] : null;
  });
  renderTopologyInto("w-topo", graph, { serviceColors: true });

  document.querySelectorAll("input[name=w-mode]").forEach((r) => {
    r.onchange = () => {
      document.getElementById("w-user").style.opacity =
        document.querySelector("input[name=w-mode]:checked").value === "user" ? "1" : ".5";
    };
  });

  document.getElementById("w-place").onclick = async () => {
    const mode = document.querySelector("input[name=w-mode]:checked").value;
    const body = mode === "random"
      ? { slo: "random", count: +document.getElementById("w-count").value,
          seed: +document.getElementById("w-seed").value }
      : { slo: {
          rps: +document.getElementById("w-rps").value || null,
          input_p50: +document.getElementById("w-in").value || null,
          output_p50: +document.getElementById("w-out").value || null,
          ttft_p99_ms: +document.getElementById("w-ttft").value,
          tpot_p99_ms: +document.getElementById("w-tpot").value,
          power_cap_w: +document.getElementById("w-cap").value || null,
        } };
    const preview = document.getElementById("w-preview");
    preview.innerHTML = "planning…";
    let data;
    try {
      data = await apiSend(
        "POST", `/api/workspaces/${encodeURIComponent(workspaceId)}/placements`, body
      );
    } catch (err) {
      preview.innerHTML = `${badge("error", "warn")} ${esc(err.message)}`;
      return;
    }
    const first = data.placements[0];
    if (body.slo !== "random" || body.count === 1) {
      if (first.status === "PLANNING") {
        // FR-W4: preview first, confirm explicitly.
        preview.innerHTML = `
          <div class="suggestion">미리보기: ${first.devices.length} devices ·
            avg ${fmt(first.avg_power_w)} W ·
            SLO ${verdictBadge(first.slo_ttft_ok, first.calibrated)}${verdictBadge(first.slo_tpot_ok, first.calibrated)}
            <button id="w-ok">확정</button> <button id="w-no">취소</button></div>`;
        document.getElementById("w-ok").onclick = async () => {
          await apiSend("POST",
            `/api/workspaces/${encodeURIComponent(workspaceId)}/placements/${first.placement_id}/confirm`);
          route();
        };
        document.getElementById("w-no").onclick = () => route();
        return;
      }
    }
    route();
  };

  document.querySelectorAll(".w-remove").forEach((btn) => {
    btn.onclick = async () => {
      await apiSend("DELETE",
        `/api/workspaces/${encodeURIComponent(workspaceId)}/placements/${btn.dataset.id}`);
      route();
    };
  });
  document.querySelectorAll(".w-confirm").forEach((btn) => {
    btn.onclick = async () => {
      try {
        await apiSend("POST",
          `/api/workspaces/${encodeURIComponent(workspaceId)}/placements/${btn.dataset.id}/confirm`);
      } catch (err) {
        showError(err.message);
      }
      route();
    };
  });
  const toggleExpand = async (placementId, contentPromise) => {
    const expand = document.querySelector(`.w-expand[data-for="${placementId}"]`);
    if (!expand.classList.contains("hidden")) {
      expand.classList.add("hidden");
      return;
    }
    expand.querySelector(".w-expand-body").textContent = "loading…";
    expand.classList.remove("hidden");
    expand.querySelector(".w-expand-body").textContent = await contentPromise;
  };
  document.querySelectorAll(".w-diag").forEach((a) => {
    a.onclick = (event) => {
      event.preventDefault();
      const row = rows.find((p) => p.placement_id === a.dataset.id);
      toggleExpand(a.dataset.id,
        Promise.resolve(JSON.stringify(row.rejected_reason, null, 2)));
    };
  });
  document.querySelectorAll(".w-detail").forEach((btn) => {
    btn.onclick = () => {
      toggleExpand(btn.dataset.id, (async () => {
        const detail = await api(
          `/api/workspaces/${encodeURIComponent(workspaceId)}/placements/${btn.dataset.id}`);
        const out = detail.document.planner_output || {};
        const plan = out.recommended ? out.recommended.plan : out.closest_plan;
        return JSON.stringify({ placement: detail.placement, plan }, null, 2);
      })());
    };
  });
}

// --------------------------------------------------------------------- boot

route();
