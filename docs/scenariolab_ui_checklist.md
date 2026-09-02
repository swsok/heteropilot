# ScenarioLab UI manual smoke checklist (DESIGN §10.4)

Run against any populated store:

```bash
python -m scenariolab serve --config experiments/configs/lab/smoke.yaml --port 8080
# open http://127.0.0.1:8080
```

Check every box before merging a UI-touching PR. Playwright automation is a
deferred follow-up (DESIGN P5); until then this list is the gate.

## Global

- [ ] Top bar shows ① ② enabled, ④ ⑤ greyed out with a "P4" tooltip
- [ ] Batch dropdown lists every batch in the DB; switching it re-renders the page
- [ ] Stopping the server and clicking anything shows the red error banner,
      never a blank page (FR-U5)
- [ ] Every metric is accompanied by its fidelity badge
      (envelope=green / surrogate=yellow / sim=blue) and, where set,
      `⚠ NPU extrapolated` and `calibrated: false` badges (FR-U1)
- [ ] The URL reflects the current page and scenario
      (`#/dashboard`, `#/explorer`, `#/scenario/<id>`), and pasting such a URL
      into a new tab lands on the same view (FR-U4)

## ① Dashboard

- [ ] Cards: scenarios done, feasible rate, median power saving,
      verification |err| p95, NPU-extrapolated count
- [ ] Chart A: feasible rate vs power-cap band (line) with scenario counts (bars)
- [ ] Chart B: power-saving histogram stacked by fidelity
- [ ] Chart C: feasible-rate heatmap, cluster size x TPOT strictness,
      cells labelled with percentages

## ② Explorer

- [ ] Filters (feasible / fidelity / has NPU / saving >= / sort / descending)
      all change the table AND the scatter
- [ ] Scatter: green feasible, red infeasible; clicking a point opens ③
- [ ] Table rows show the honesty badges; clicking a row opens ③
- [ ] Pagination: page counter is correct, ←/→ never leave the valid range,
      and the table never loads more than one page (FR-U6)

## ③ Scenario detail

- [ ] Topology graph: node color = accelerator class, dimmed = ALLOCATED,
      blue thick border = used by the plan, label prefix P/D/A = role (FR-U3)
- [ ] Link thickness scales with bandwidth; plan-used links are blue
- [ ] Summary card: predicted vs SLO values side by side, power saving with
      baseline note, calibration margins or `calibrated: false`
- [ ] Pareto chart: recommended (orange, large) vs alternatives (blue)
- [ ] Verified scenarios show the full-sim panel with error percentages and
      flip flags
- [ ] An infeasible scenario shows the diagnosis block: reason,
      violated-constraints table, suggestions - and the plan JSON section
      falls back to closest_plan
- [ ] "Plan document" and "Provenance" collapsibles open with pretty JSON

## ⑥ Cluster Builder (workspace work order §6.2)

- [ ] Node-group rows add/remove; per-node count and node count edit
- [ ] Preset vs custom inter-node link toggle; custom section shows the
      "labelled user_defined" notice
- [ ] Invalid request (>64 accels, per-node cap) shows the rejection message,
      never a blank panel
- [ ] Successful build shows islands + TP candidates, warnings, topology
      preview, and a "Workspace 시작" button
- [ ] Re-submitting the same form shows "already existed" (idempotent)

## ⑦ Clusters (catalog)

- [ ] Origin tabs all/random/custom filter the table
- [ ] user_defined fabric shows the red badge in the fabric column
- [ ] Row click renders the mini topology preview
- [ ] "Workspace 시작" prompts for a name and lands on #/workspace/{id}

## ⑧ Workspace

- [ ] Interference notice is always visible at the top
- [ ] Topology: device color = service (stable across add/remove, FR-W2),
      grey = FREE, NICs diamonds
- [ ] Resource gauges per class match the summary numbers
- [ ] Random (count/seed) and user-typed SLO forms both place; user single
      placement goes preview -> [확정]/[취소] (FR-W4)
- [ ] Power panel separates avg sum and peak sum, peak labelled as the
      conservative simultaneous-peak bound (FR-W3); cap gauge turns red
      over the cap
- [ ] Service table: SLO ✓/✗ per TTFT/TPOT with grey badges when
      calibrated=false (FR-W1); fidelity/⚠NPU/⚠shared-fabric badges shown
- [ ] REJECTED rows stay listed with 진단 보기 expanding the diagnosis (FR-W6)
- [ ] [상세] expands the placement's plan (FR-W5); [제거] frees the devices
- [ ] URL #/workspace/{id} survives refresh and can be shared (FR-W7)
