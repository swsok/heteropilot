# HeteroPilot — current state and what to do next

> **This is the live handover.** Written 2026-09-04 at `main` = `8bb2f6f`, after the
> consolidation sprint (`WORK_ORDER_consolidation.md`, PRs #47–#51). It is
> **node-agnostic**: every open item says which machine it needs. Earlier handovers
> are historical and must not be read as status:
> `docs/HANDOVER_2026-08-31.md` (→ NPU, the previous live one),
> `docs/HANDOVER_NPU.md` (→ NPU, 2026-08-25),
> `docs/HANDOVER_A40.md` (→ A40, 2026-08-26).
>
> Authority order is unchanged: `WORK_ORDER_heteropilot.md` → `docs/deviations.md`
> → `CLAUDE.md` → this file.

**Gates at this commit**, on the NPU node in `.venv`:

```
pytest -q     440 passed in 86.66s
ruff check .  All checks passed!
mypy          Success: no issues found in 46 source files
```

440, not the 522 of two commits ago: ScenarioLab left this repo and took 82 tests
with it (§1). `mypy` covers `planner`, `profiler/synth`, `profiler/contract.py`.

---

## 0. First command on any machine

```bash
bash scripts/whichnode.sh
```

**This repository moves between an A40 node, an A5000 node and an NPU node, and
every one of them reports hostname `s8`/`etri-001`.** Any committed sentence about
what hardware is present is true on at most one of them. The detector probes
`nvidia-smi -L`, `/sys/class/rngd_mgmt` and `/dev/rbln*`, prints what can actually
be run here, and names the node profile to read (`docs/nodes/{a40,a5000,npu}.md` —
read only the one it names).

Absolute rule 3 covers hardware **presence**, not just hardware numbers: never
claim a result from hardware the detector does not list.

---

## 1. Status

| Phase | Status |
| --- | --- |
| 0 Baseline · 1 Inventory/islands · 2 Offline planner (MVP) | ✅ done |
| 3 Heterogeneous profiles | ✅ done (`CsvProfileImporter`) |
| **Tiered profiles (Tier 0/1)** | ✅ **done** — D4 closed without external measurements; `docs/tier0_calibration.md` |
| 4 Real deploy + calibration | ✅ CUDA. NPU launcher still a stub |
| 5 Topology-aware P/D | ✅ core. Network-aware routing deferred. **Tight-TTFT blocked by D23** |
| 6 Online replanning | ⛔ not started — **requires explicit user approval** |

**ScenarioLab moved out** on 2026-09-03 to `swsok/heteropilot-scenariolab`
(private), which pins this repo as a submodule at `e79ac4ab`. It imports from
`planner/` and never the reverse, so nothing here depends on it. `profiles/networks/`
and `experiments/configs/lab/` went with it — D24 records why that does not
contradict the work order's layout.

**Accelerator profiles** (`profiles/accelerators/`) **and what they may claim:**

| profile | `sim_hardware` | `source` | usable in candidate generation |
| --- | --- | --- | --- |
| `a40.yaml` | A40 | measured | yes |
| `a5000.yaml` | A5000 | measured | yes |
| `furiosa_rngd.yaml` (PE-as-device) | RNGD | measured | yes |
| `furiosa_rngd_card.yaml` (card-as-device) | RNGD-CARD | measured | yes |
| `rtxpro6000.yaml` | RTXPRO6000 | vendor_spec | yes, labelled |
| `ascend_target.yaml` | **ASCEND_TARGET-t0** | vendor_spec | **yes now** — Tier 0 synthetic bundle, plans carry `profile_tier: analytical` |
| `rbln_atom.yaml` | **null** | placeholder | **no — fails loud** (D20) |

`ascend_target` is the change: it was `placeholder` / `sim_hardware: null` and is
now backed by a datasheet-derived bundle that `scripts/gen-tier0-bundles.sh`
regenerates. Synthetic bundles are **gitignored on purpose** (`profiler/perf/*-t0/`,
`*-t1/`) so measured and synthetic data never mix in the tree. Every plan that
touches one carries the weakest tier in `PlannerOutput.profile_tier` plus a
mandatory caveat — D21.

Calibrations (`profiles/calibration/`): `a40.yaml`, `rngd.yaml`, `rngd_card_edf.yaml`.
Tier 1 efficiency fits: `a40.efficiency.yaml`, `rtxpro6000.efficiency.yaml`. All
bucket-scoped — **do not extrapolate outside the bucket named in the file.**

### What the last cycle established

**The headline finding is negative, and it should be said first: no experiment in
this repository currently shows a heterogeneous configuration winning.** The one
that did — RNGD beating the A40 on energy by 1.67× at loose TTFT — was retracted
by this project's own margin discipline (D22). The tight-TTFT half, where P/D
disaggregation was supposed to buy the sub-second end, is undetermined and blocked
by a simulator livelock (D23). That is the starting point for the next work order,
not a gap to paper over.

- **Tier 0/1 works well enough to plan on hardware we do not own.** Against the
  measured-bundle ranking, tier0+sim gives **Kendall τ 0.90–0.91**. Exact top-1
  agreement is 0/2 — but the *cost* of the top-1 disagreement is **0.4 %** of the
  true objective on `llama31-8b` and **11.3 %** on `llama31-8b-light`, i.e. the
  candidates it confuses are near-equivalent. Ranking is preserved where it
  matters.
- **Attention is where the error lives, and anchoring it is what pays.** Tier 0
  with no anchors is **38.9 % MAPE**; 200 anchors spent entirely on attention take
  it to **29.5 %**. GEMM transfers on a single scalar efficiency (9.5–11.3 %);
  attention does not.
- **D4 is closed** — not by external measurements arriving, but by generating a
  datasheet-derived Ascend bundle and labelling it honestly. Measured data still
  supersedes it whenever it arrives.
- **D22 — the RNGD envelope, retracted and remeasured.** "c32 is the highest
  concurrency ever run on RNGD" was a 24-request pool at **effective concurrency
  21.2**, and the exponent derived from it read a pool-capped ×1.74 interval as a
  doubling. Measured to **eff 107.2 at 1473 output tok/s**, zero failures. At the
  eff 76 the card fixture's winner runs at, the simulator is **1.31× optimistic on
  throughput and 18 % on TPOT**.
- **And that 18 % kills the winner.** Re-run as a feasibility margin, **every RNGD
  configuration is rejected on both fixtures**; the loose-TTFT winner becomes
  `agg[cuda:tp4]` at 2.595 tok/J. The committed winner cleared the 50 ms TPOT SLO
  by 1.59 ms, so **any margin above 3.3 %** rejects it. It was infeasible, not
  merely optimistic.
- **D23 — the tight-TTFT candidates livelock.** Every `pd_*`/`mix_*` candidate the
  tight regime needs fails to terminate: 52,903 progress ticks with prefill pinned
  at 1 running request, decode never fed, memory flat at 9 %. Not D12 (no memory
  growth, prefix caching off) and not a timeout (3600 s fails too). The same
  candidate completed in **280.6 s** in an earlier committed run, so it is a
  regression, and the cause is open.

---

## 2. Next work, in priority order

### 2.1 `WORK_ORDER_rps_aware.md` — **not yet written**, any node to draft

The first priority is a work order, not a run. `docs/rps_aware_planning_design.md`
is the design; it argues that performance is a curve over the operating point, not
a scalar, and that the planner currently conflates requested with served
concurrency — the conflation that produced D22.

**Deliberately not done in the consolidation sprint**: the low-load end of the
RNGD envelope is unmeasured, and measuring it was explicitly out of scope (that
sprint made no new numbers). It belongs in this work order, together with the
power crossover, which is a hypothesis in the design document and not a
measurement.

### 2.2 D23 — diagnose the P/D livelock — **any node** (simulation only)

Nothing about the tight-TTFT regime can be settled until this is understood, and
it blocks the only claim P/D disaggregation had. D23 lists three probes, cheapest
first: bisect `link_bw` between 35.0 and 35.2 on the one candidate (a cliff there
means a threshold bug, not a bandwidth effect); diff the `.hp-pd-slo` run's
simulator inputs against the current ones beyond `cluster.json`, in particular the
A40 perf bundle that the Tier 0 merge touched; instrument the prefill instance's
admission path.

Do not raise the timeout. It has been tried at 1080, 1800 and 3600 s.

### 2.3 ATOM layerwise bundle (D20) — **needs the NPU node**, and probably the vendor

Unchanged. Host I/O exceeds the kernels and the device tracer's `.pb` schema is
undocumented, so no bundle ships and ATOM stays out of candidate generation and
Exp 4. Memory and power *are* measured. Resolution paths, in order of expected
effort: the trace schema from Rebellions; a torch backend registering device
`rbln`; a llama entry in vllm-rbln's native model registry.

### 2.4 TTFT tail under-prediction — **any node** (simulation only)

Unchanged.

### 2.5 D14 — asymmetric TP per phase — **any node**, largest change

Unchanged. The simulator's topology inference requires uniform instance sizes, so
P/D pairs with different TP degrees cannot be enumerated.

### 2.6 Smaller, any node

- **PR #13** (`docs/slide-deck-ko`) — dispositioned by the consolidation sprint's
  STEP 4.3; see that PR for what was decided.
- **Remote branches are clean.** `origin` holds `main` and nothing else that is
  merged. The 17 already-merged branches turned out to have been deleted already;
  the D22 chain and the ScenarioLab workspace branches were deleted during the
  sprint, the latter after verifying their content reached the split repo.
- **`docs/nodes/a5000.md` is thin and says so** — written from committed artifacts,
  not from the node. Fill it in from `scripts/whichnode.sh` when next on that
  machine.

---

## 3. Traps that have each cost a session

Recorded because they are not discoverable from the code.

**Reading a sweep**

- **A sweep's INFEASIBLE is not a result until you have counted the timeouts.**
  `pd_slo_sweep.py` prints INFEASIBLE identically whether candidates were rejected
  or never evaluated, and it **does not persist `PlannerOutput.rejected_summary`**.
  Recover the count by counting work directories with no `sim*.csv`. On the
  2026-09-03 tight re-run that was 71 of 222 and 126 of 252 — including the
  committed winner. This is D23's first symptom and it was misread twice.
- **Record effective concurrency, never requested.** The c1–c32 RNGD curve was
  labelled by the concurrency asked for; a 24-request pool meant the c32 point
  actually ran at **eff 21.2** (Little's law), and an exponent fitted across it
  read a ×1.74 interval as a doubling. That is the whole of D22.
- **Timeouts are not cached and are retried once per SLO point.** A sweep whose
  candidates hang costs `candidates × timeout ÷ workers` *per point*, and the
  envelope cache is written only when a point joins — kill a run mid-point and its
  completed simulations are lost, not resumed.

**Measurement**

- **A single parallel-transfer trial is not reproducible.** Host buffers are not
  NUMA-bound and placement is fixed at allocation; two runs disagreed by 38 % on
  the 4-GPU figure and the same-node/cross-node ordering reversed. Repeat over
  independent allocations, report median and spread.
- **Pageable D2H is allocator-bound, not link-bound.** It peaks at 16 MB and drops
  ~4.5× above, because PyTorch's CPU allocator stops caching there. Read pinned.
- **Peak is not sustained.** On RNGD best-of-N overstates a bulk copy by 25 %; on
  the A40 it does not (0.998). Compose a fabric bandwidth from the *same* statistic
  on both legs.
- **Judge a power reading by its utilisation.** ATOM's first pass read 30 % low at
  36 % util; a single-PE RNGD reading understated its card 4×. `active_util_pct` is
  recorded beside the power for exactly this reason.
- **Never subtract a constant host round-trip floor on ATOM** — per-call cost
  scales with bytes moved.

**Harness**

- **`experiments/scripts/bench_furiosa_endpoint.py` ignores `arrival_time_ns`** and
  fires everything at once, while `python -m serving` replays it. Feeding both the
  same file compares a burst against a spread arrival process; the difference lands
  entirely in TTFT. That is D19. Use `outputs/envcheck/rngd20_burst.jsonl` on the
  simulator side.
- **`python -m serving` shells out to a bare `python -m chakra`**, so the venv must
  be on `PATH`, not merely invoked by full path:
  `export PYTHONPATH=$PWD && export PATH="$PWD/.venv/bin:$PATH"`.
- **Run a multi-hour sweep detached** (`setsid`/`nohup`/`tmux`), not as a job owned
  by an interactive session. One was killed 30 minutes into its second fixture and
  lost 84 completed simulations.
- **The SLO sweeps do not price the fabric.** The simulator charges the P/D handoff
  at zero unless `--pd-transfer-model bandwidth` is passed, and only
  `experiments/scripts/pd_sim_network_sweep.py` passes it.
  `experiments/scripts/run_exp_pd.sh` is planner-side on both drivers.
- **`link_bw` is an overloaded scalar in multi-node configs** — the `none`-mode
  control moves too, which the driver's pass criteria do not anticipate at tp1.

**Environment**

- **One venv per vendor.** `.venv` = planner + analytical sim, no device, never
  install vLLM into it. `.venv-vllm` = CUDA. `.venv-rbln` / `.venv-rbln-vllm` =
  ATOM. System `python3` = FuriosaAI.
- **`.venv` needs `fastapi`/`uvicorn` only for ScenarioLab**, which has left. If a
  checkout predating the split fails `pytest` at collection, that is why.
- **ASTRA-Sim is built `RelWithDebInfo` (`-O2 -g`), not `-O3`.** `scripts/compile.sh`
  passes no `-DCMAKE_BUILD_TYPE` and the analytical `CMakeLists.txt` defaults to
  `RelWithDebInfo` (its `# Default: Release` comment is wrong). **Do not rebuild it
  to "optimise"** — build flags are part of a result's provenance, like the
  `UPSTREAM_COMMIT` pin, and `-O2`→`-O3` is worth single-digit percent here.
- **`furiosa.torch` and `rebel` must import after `torch`**, behind the
  `# isort: off` guards. `ruff check --fix` reordering them breaks every run.
- **Multi-device work on NPUs uses one subprocess per device**, not threads.

---

## 4. Invariants that are easy to break

Beyond `CLAUDE.md`'s absolute rules, four that recent cycles exercised:

1. **Absolute rule 3 covers hardware *presence*, not just hardware numbers.** Never
   claim a result from hardware `scripts/whichnode.sh` does not list. Artifacts
   measured on another node stay valid as measurements *of that node* — do not
   re-run, extend, or relabel them.
2. **Retract in public.** D18, D19, D20 and D22 each state what was claimed, what
   was measured, and what it changes. When a committed number turns out wrong,
   correct every site that carried it and say so — do not quietly overwrite. The
   superseded text stays, marked, because it records what was reasonable to believe.
3. **A pruning stage must be a relaxation of the feasibility test**, and **a mock
   predictor must respect the same physics as the bounds.** Both have been violated
   before and both were caught by the oracle-agreement test.
4. **A synthetic bundle must never be able to shadow a measured one.** Tier 0/1
   bundles carry a `-t0`/`-t1` hardware-label suffix, are gitignored, and propagate
   `profile_tier` into every plan built on them. A profile whose `sim_hardware` ends
   in `-t0`/`-t1` without a `datasheet:` block is rejected at load time.
