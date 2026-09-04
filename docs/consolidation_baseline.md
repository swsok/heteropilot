# Consolidation sprint baseline (STEP 0)

Snapshot of the quality gates on `main` immediately before the consolidation
sprint (`WORK_ORDER_consolidation.md`) began, plus the re-verification of that
document's §2 investigation table. Recorded per STEP 0 so later steps can prove
they did not regress anything that was green here.

- Date: 2026-09-03
- Commit: `f0905e263d8de6c8cdad86ce97d04903a92f5c62` (`docs: add the consolidation sprint work order`)

## Node

```
$ bash scripts/whichnode.sh
=== node detection ===============================================
  detected node : npu   (read docs/nodes/npu.md)
  hostname      : etri-001   <- same on every node, do not key off it
  cores / RAM   : 96 / 1511 GiB
  NVIDIA        : none reachable
  RNGD cards    : 3
  ATOM devices  : 4
```

STEP 2 is executable here (the simulator is CPU-only, and 96 cores leave room
for its `--workers 32`). No CUDA profiling or real vLLM bench on this node —
absolute rule 3.

## Gate results

```
$ pytest -q 2>&1 | tail -5
520 passed, 1 warning in 254.08s (0:04:14)

$ ruff check . 2>&1 | tail -3
All checks passed!

$ mypy 2>&1 | tail -3
Success: no issues found in 64 source files
```

484 test functions collect to 520 passing tests (parametrize). `docs/HANDOVER.md`
still says "pytest 284 passed", which the work order §1 already flags as one of
the three documents pointing at different moments in time; STEP 4 rewrites it.

## §2 re-verification

Every item below was re-checked on this node at the commit above. Confirmed
unless the divergence section says otherwise.

| § | Claim | Result |
| --- | --- | --- |
| 2.1 | The D22 branches form a straight chain `docs/d18-close ⊂ feat/rngd-concurrency-envelope ⊂ docs/rps-aware-planning ⊂ feat/pd-slo-margin-rerun` | confirmed (`git merge-base --is-ancestor`, all four) |
| 2.1 | ScenarioLab chain `core ⊂ ui ⊂ extras` | confirmed |
| 2.1 | Merge base is `0316c29` | confirmed |
| 2.2 | Merging `origin/feat/pd-slo-margin-rerun` conflicts in exactly `.gitignore` and `docs/deviations.md` | confirmed (dry run, and again in STEP 1's real merge) |
| 2.5 | No reverse `planner → scenariolab` import | confirmed (`grep -rn scenariolab planner/` is empty) |
| 2.5 | `profiles/networks/` is referenced only by ScenarioLab | confirmed (no hits in `planner`, `examples`, `experiments`, `tests`) |
| 2.5 | `Source` values are never branched on | confirmed — the only `Source.` uses in `planner`/`tests` are four `= Source.PLACEHOLDER` defaults in `inventory.py` |
| 2.5 | `tests/scenariolab/` holds 78 test functions | confirmed (484 in `tests/` overall) |
| 2.6 | The tight-TTFT rows are undetermined because `--timeout 1080` timed out every `pd_cuda-a40-tp4` candidate | confirmed against `experiments/results/pd_slo_sweep_margin.md` |
| 2.7 | 17 remote branches have `main..origin/<b>` = 0 | confirmed, all 17 exactly — but they were already gone from the remote; see divergence 4 |
| 2.7 | 8 remote branches are unmerged | confirmed (1, 4, 5, 6, 1, 2, 3, 7 commits ahead respectively) |
| 2.8 | The old headline numbers survive only in `docs/PROJECT_REPORT.md` and `experiments/results/pd_slo_sweep.md` | confirmed for real sites — see divergence 3 for the grep |

### Divergences from the work order's §2

1. **The gate `pytest -q` could not run in this node's `.venv` as written.**
   `tests/scenariolab/test_api.py` imports `fastapi.testclient`, and the venv had
   no `fastapi`, `starlette` or `uvicorn` — collection aborted before any test
   ran. `main` reproduces this on its own; it arrived with the ScenarioLab merge
   and is not caused by the sprint. Resolved by
   `uv pip install fastapi uvicorn` (fastapi 0.141.1, starlette 1.6.0,
   uvicorn 0.52.4) against the existing httpx 0.28.1, which is where the run's
   one warning comes from:
   `StarletteDeprecationWarning: Using httpx with starlette.testclient is
   deprecated`. Nothing pins these versions — `pyproject.toml` deliberately has
   no `[project]` table — so a fresh checkout hits the same wall. STEP 3 removes
   ScenarioLab from this repo and the dependency with it; until then, note it in
   any environment bring-up.

2. **§2.5's ScenarioLab → planner import counts are slightly low.**
   `from planner.inventory` appears 8 times, not 7, and the table omits
   `from planner.predictor.llmservingsim` (1). The conclusion the table is
   drawing — the coupling is one-directional and shallow — is unaffected.

3. **§2.8's grep pattern produces one false positive, and §6's completion
   condition inherits it.** `1\.67` matches `11.6799` in
   `experiments/results/pd_sim_network_sweep_table.md` (a P/D network-sweep
   latency, unrelated to the 1.67× energy headline). The §6 check
   `git grep -c '4\.956\|3\.164\|1\.67'` will therefore flag a file that needs no
   superseded marker. Use word boundaries — `\b4\.956\b|\b3\.164\b|1\.67×|1\.67x`
   — which returns `docs/PROJECT_REPORT.md`, `experiments/results/pd_slo_sweep.md`
   and `WORK_ORDER_consolidation.md` itself.

4. **The 17 merged branches were already deleted on the remote.** STEP 0's
   instruction 3 was a no-op: `git push origin --delete` returned
   `remote ref does not exist` for all 17. `git ls-remote --heads origin` shows
   the remote has held only `main` plus the 8 unmerged branches. What existed
   were stale local remote-tracking refs — this session's first `git pull` does
   not prune, so `git branch -r` listed 26 — and `git fetch --prune` removed
   them. GitHub's delete-on-merge is the likely cause. The work order's §2.7
   list is still correct about *which* branches are merged; it is wrong that
   they still need deleting.

## Remote branch inventory at the baseline

The remote holds 9 refs: `main` and the 8 unmerged branches. The 17 merged ones
listed below were already deleted there (divergence 4); `git branch -r` showed 26
only because the local remote-tracking refs were stale. STEP 1 deletes 4 more
after its merge, STEP 3 the 3 ScenarioLab ones, leaving `origin/main` and
`docs/slide-deck-ko` (STEP 4.3 decides that one).

```
merged (main..origin/<b> == 0), already absent from the remote:
  chore/node-detection                          docs/html-slide-deck
  docs/project-report-and-npu-handover          docs/slide-outline
  feat/a40-inventory                            feat/baselines-ablation
  feat/exp1-tp-sweep                            feat/exp25-figures
  feat/figure-packaging                         feat/gpu-host-bandwidth
  feat/rngd-parallel-bandwidth                  feat/router-baselines
  feat/sim-pd-transfer                          feat/surrogate-and-parallel
  feat/topology-perdim                          fix/rngd-ttft-validation-arrivals
  fix/scaling-curve-provenance-and-npu-envelope

unmerged, left alone by STEP 0 (commits ahead of main):
  docs/d18-close                        1     feat/rngd-concurrency-envelope    4
  docs/rps-aware-planning               5     feat/pd-slo-margin-rerun          6
  docs/slide-deck-ko                    1     feat/scenariolab-workspace-core   2
  feat/scenariolab-workspace-ui         3     feat/scenariolab-workspace-extras 7
```

`docs/HANDOVER.md` §2.1 calls
`fix/scaling-curve-provenance-and-npu-envelope` "pushed, not merged"; it is
merged, as work order §2.7 already notes. STEP 4 fixes the handover.
