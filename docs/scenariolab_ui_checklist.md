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
