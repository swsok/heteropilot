# HeteroPilot — current state and what to do next

> **This is the live handover.** Rewritten 2026-09-10 at the end of
> `WORK_ORDER_rps_aware.md` rev 2 (STEP 0–6, PRs #60–#72). **The whole stack is on
> `main` as of `3aadc2b`** and every `feat/rps-step*` branch, plus
> `spike/d14-asym-tp`, has been deleted — `origin` holds `main` and nothing else.
>
> That took a second merge. PRs #64–#71 each merged into their *parent feature
> branch* rather than into `main`, and the parent had already reached `main`
> sixteen seconds earlier, so thirteen commits — STEP 2 through STEP 6 — read as
> MERGED on GitHub while being absent from `main`. PR #72 landed them. If a future
> stack is merged bottom-up again, check
> `git rev-list --count origin/main..<tip>` before believing the PR list.
>
> It is **node-agnostic**:
> every open item says which machine it needs. Earlier handovers are historical
> and must not be read as status:
> `docs/HANDOVER_2026-08-31.md` (→ NPU, the previous live one),
> `docs/HANDOVER_NPU.md` (→ NPU, 2026-08-25),
> `docs/HANDOVER_A40.md` (→ A40, 2026-08-26).
>
> Authority order is unchanged: `WORK_ORDER_heteropilot.md` → `docs/deviations.md`
> → `CLAUDE.md` → this file.

**Gates at this commit**, on the NPU node in `.venv`:

```
pytest -q     653 passed in 86.64s          # A40 node, 2026-09-11
ruff check .  All checks passed!
mypy          Success: no issues found in 38 source files
```

622 of those are the count this handover was written at, on the NPU node. One of
them **failed on the A40 node** before any new work:
`test_there_is_no_user_site_fallback` asserted `any('.local' in p for p in
sys.path)`, which matches uv's *interpreter* prefix
(`~/.local/share/uv/python/.../lib/python3.10`) — the standard library, not a
user-site directory a stray `pip install --user` could poison. The property it
cared about held on the node where it failed; it passed on the NPU node only
because that interpreter lives elsewhere. It now asserts the two load-bearing
facts instead: the user site directory is not importable, and the chakra this
interpreter resolves lives under its own prefix.

622, up from 440 at the last handover: the rps-aware sprint added ~180, most of
them holding a refusal in place (an envelope that will not be read past its ends,
a calibration domain that returns None rather than 1.0, a surrogate that is not
silently swapped). `mypy` covers `planner`, `profiler/synth`, `profiler/contract.py`.

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
| 5 Topology-aware P/D | ✅ core, and **asymmetric TP per phase now representable** (D28). Network-aware routing deferred. **D23 resolved** — it was D26, a PATH-resolved Chakra interpreter |
| **RPS-aware selection** | ✅ **done 2026-09-09** — envelope, accuracy domain, operating-point margins, `plan --rps`. §2.1 |
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

Calibrations (`profiles/calibration/`): `a40.yaml`, `rngd.yaml`, `rngd_card_edf.yaml`,
and since this sprint `a40.accuracy.yaml` (opt-in, D29) and `slab3d_latency.yaml`
(a topology correction, not a hardware calibration — `load_accuracy_domains` skips
it deliberately). Tier 1 efficiency fits: `a40.efficiency.yaml`,
`rtxpro6000.efficiency.yaml`. All bucket-scoped — **do not extrapolate outside the
bucket named in the file.**

**Two new artifact kinds, and the rule attached to each:**

| artifact | what it is | the refusal it carries |
| --- | --- | --- |
| `profiles/envelopes/<HW>/…/tp<N>.yaml` | a measured performance curve | `concurrency_metric: served` and `validity` are mandatory; a file missing either **does not load** |
| `accuracy_domain:` inside a calibration | the predictor's own error vs operating point | outside the measured points the margin **widens with distance and never caps**; hardware without one gets margin 0 and a note |

**What can be claimed right now, with the label each claim earns, is one page:
`docs/CLAIMS.md`.** It is the input to the paper outline — Established / Not
established / Retracted — and every figure in it points at a committed artifact.

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

### 2.1 `WORK_ORDER_rps_aware.md` — **complete, STEP 0–6, PRs #60–#70**

Delivered, in the order it binds:

| step | what landed |
| --- | --- |
| 0 | D22 §4.4's "never terminates at 3.3 rps" was a **D26 artifact**; Exp 5 survives the retrospective check with zero differing fields |
| 1 | The RPS axis costs **21.4×**, not 6× (1 rps = 10.3× a 10 rps point) → E6a was 218 h, not 60. **D27** returned 1.55–1.72× byte-identically |
| 1.5 | §4.7's "regret 0 at every K" **partly retracted** (D30); top-K is not a cost lever on P/D corpora |
| 2 | **D28** — `slab3d` for `tp_d = 2·tp_p`, plus a 24-point measured latency table with `extrapolation: refuse` |
| 3 | The RNGD envelope from **concurrency 1**: tok/J spans **9.4×** while power spans 1.085×. "tokens/J is unimodal" is **not observed** |
| 4 | Envelope, accuracy domain, operating-point margins, `plan --rps`. **D29**, **D31** |
| 5 | **E5** the planner self-rejects D22's winner; **E6** there is a crossover; **E7** it survives losing any one calibration point but not the domain |
| 6 | This rewrite, plus `docs/PAPER_OUTLINE.md` |

**The regression ladder is now committed** (`outputs/d23fix/anchor/`, PR #72).
Absolute rule 1 requires a byte-identical proof for every sanctioned `serving/`
edit; before that PR only D25 and D26 had a committed artifact, and D27's and
D28's runs existed as untracked files on the NPU node alone. Read
`outputs/d23fix/anchor/README.md` before reading the hashes — a `SHA256SUMS` with
three lines instead of four is a *failed* anchor, not an agreeing one, and
`anchors.log` beside it is what says which.

**The one-sentence state of the science:** the planner can now price its own
predictor and finds a crossover on the RPS axis — and since 2026-09-11 **nine of
sixteen E6 cells rest on a measured accuracy domain**, up from one, with every
winner unchanged (§2.2). Of the seven that do not, five need a per-PE RNGD
domain and two rest on an A40 prefill leg at served concurrency 0.499, which no
bench can hold.

Not delivered, deliberately: a third E6 fixture. The work order names
`pd-rngd-gpu-card`, `pd-rngd-gpu` and "STEP 2's asymmetric fixture", but
`pd-rngd-gpu.yaml` already yields the `A40 tp4 P + RNGD tp8 D` shape once
`--enable-pd` is on, so a separate ClusterSpecV2 would have duplicated it. Recorded
here rather than silently dropped.

### 2.2 A second A40 accuracy-domain point — **DONE 2026-09-10/11 on the A40 node**

**Closed.** The domain has three points, the E6 sweep was re-run on them, and
**seven of sixteen switchover cells moved from `extrapolated` to `measured`**
with every winner and every tok/J unchanged. Full write-ups:
`experiments/results/a40_lowload_envelope.md` and the re-run section appended to
`experiments/results/e6_rps_sweep.md`.

| | before | after |
| --- | ---: | ---: |
| domain points | 1 (conc 170.56) | **3** (4.043, 10.800, 170.56) |
| `measured` cells | 2 | **9** |
| `extrapolated` cells | 9 | **2** |
| `unknown` cells | 5 | 5 |

**The finding is TTFT, and it was not the one this section predicted.** Held
flat, the one point declared the simulator 1.97 % optimistic on TTFT everywhere.
Measured at served concurrency 4 and 11 it is optimistic by **~18 %** — a factor
of nine. Both are right about their own regime: at saturation TTFT is 40 s of
queueing, which the simulator models well; at low load it is ~150 ms of nearly
pure prefill, which it does not. It changes no E6 outcome, because no
recommended plan comes within 25 % of its TTFT SLO in this regime — but it would
in the tight-TTFT regime §2.5 still records as not quotable. TPOT, by contrast,
is nearly exact at low load and **changes sign** across the range (+0.59 % to
−1.42 %), so the one-sided margin correctly charges nothing at the bottom.

**What remains, and it cannot be fixed by measuring the A40 harder.** The two
cells still `extrapolated` are the card fixture at 3.3 rps, whose winner's A40
leg is a *prefill role at served concurrency 0.499* — an almost-idle server is
not an operating point a bench can hold. The five `unknown` cells need a per-PE
RNGD accuracy domain, which needs the NPU node.

**A method correction came out of it — `docs/deviations.md` D32.** The simulator
side must run the same number of requests as the hardware. At the script's
20-request default the sim's served concurrency lands −8.2 % and −31.7 % from the
measured points purely because the drain tail dominates a short run's wall; at
300 it is −0.8 % and −4.2 %. The four RNGD low-load domain points were taken at
the default, and `rngd_card_edf.yaml` discards two further points at a "40 %
below the hardware" gap that has the same signature. Unverified on that side and
testable with no NPU hardware, since the RNGD envelope is committed.

**Also delivered:** the A40 half of E6b now exists
(`e6b_measured_curve.py --hardware A40`), so a crossover sentence no longer has
to read "RNGD measured against A40 simulated" — but it must now say which
*protocol* each side was measured under, because they differ (below).

<details>
<summary>Superseded planning text, kept because it records what was expected</summary>

Two things in it turned out wrong, and both are worth knowing:

**"Every plan E6 recommends runs far below [170.56]."** They do not. The A40
operating points in the E6 winners are 90.8, 121.8, 125.3, 138.1, 157.1 and
169.9 — just *below* the fitted point, not far below it. Only the 3.3 rps prefill
leg at 0.499 is far below. The prescribed fix was still the right one, but the
mechanism is bracketing those points from underneath, not reaching down to them,
and the reachable ceiling was **seven** cells rather than fifteen. That ceiling
was computed from the committed operating points before the measurement and the
run hit it exactly.

**The prescribed route was closed-loop, and taking it would have been a mistake.**
The table below asks for `vllm serve` plus the closed-loop bench client. But the
existing 170.56 point came from `python -m bench run`, an *arrival-trace replay*,
and adding a closed-loop point to that domain puts two definitions of "served
concurrency 8" on one interpolation axis — the class of error D22 was. The A40
points were therefore measured **open-loop**, by
`experiments/scripts/measure_envelope_openloop.py`, whose reproduction of
170.5619 from the committed artifact is the check that all three points share an
axis. That choice is also what made the TTFT finding visible at all: both sides
replay the same arrival process, so **D19 does not apply** and TTFT is
comparable, which on the closed-loop route it would not have been.

The CUDA execution half described below was built anyway and is committed —
`measure_envelope.py --backend cuda`, `power_sampler_nvidia.sh` — because the
closed-loop curve is what makes the two protocols comparable to each other. Its
original text:

> **Only half of `measure_envelope.py` is vendor-agnostic, and an earlier draft of
> this section said otherwise.** The *analysis* half is: `summarise_point` already
> enforces A5 (served concurrency by Little's law, the pool floor, power recorded
> with utilisation from the same samples), and `read_sampler_csv` cares about a CSV
> schema, not a vendor. The *execution* half is hardcoded to FuriosaAI in three
> places, and all three need a CUDA path before any A40 point can be taken:
>
> | what | today | needed |
> | --- | --- | --- |
> | server launch | `furiosa-llm serve --devices npu:N:*` | `vllm serve` with `CUDA_VISIBLE_DEVICES` pinned to one card |
> | sampler | `power_sampler.sh` | a twin emitting the same columns from `nvidia-smi` at 1 Hz |
> | `--bench-python` | `/usr/bin/python3` | `.venv-vllm/bin/python`, which has `openai` |
>
> `experiments/scripts/lowload_sim_error.py` is the other half of the point and is
> pinned to RNGD by three module constants — `ENV`, `CLUSTER`, `DATASET`. Lifting
> them into arguments is the whole change.

All three were done. One correction to the table itself: `vllm` is **not on
PATH** — it lives in `.venv-vllm/bin/vllm`, and a bare `vllm` killed the first
closed-loop run at its first point.

</details>

### 2.3 ATOM layerwise bundle (D20) — **needs the NPU node**, and probably the vendor

Unchanged. Host I/O exceeds the kernels and the device tracer's `.pb` schema is
undocumented, so no bundle ships and ATOM stays out of candidate generation and
Exp 4. Memory and power *are* measured. Resolution paths, in order of expected
effort: the trace schema from Rebellions; a torch backend registering device
`rbln`; a llama entry in vllm-rbln's native model registry.

### 2.4 A surrogate that can rank P/D — **any node** (simulation only)

**D30.** `--top-k` is currently unusable on any P/D or heterogeneous corpus, so
E6 ran without it. The cause is structural: the roofline proxy's tok/J is
*algebraically invariant to TP and DP* — throughput and power both scale with
`tp·dp`, so the ratio cancels — which leaves the ranker blind to the axis that
decides feasibility. It is false-infeasible at K=20 on two of three corpora.

**Do not fix this by ordering on the roofline TPOT floor.** That was tried and
measured: it repairs those two corpora and breaks the third, which is the same
one-fixture error that produced the "regret 0.000 at every K" claim now partly
retracted. Any replacement needs its regret measured across all three corpora
before it is switched on; `exp_surrogate.py --cache-dir --rankers` replays them
without re-simulating.

### 2.5 The rest of the tight-TTFT and asymmetric-P/D story — **any node**

D23 and D14 are **closed** (D26 and D28 respectively); what remains is narrower:

- The `slab3d` dim-1 calibration is fitted at **two** dims (a single colocated
  instance) and applied at **three** (a P/D pair), taking dim 1 to be the same
  axis in both. Hop counting supports it; nothing measures it. A P/D-side
  measurement would close the one assumption in that path.
- Only **TPOT p50** was fitted. TTFT and throughput under `slab3d` are unmeasured,
  so an asymmetric plan's absolute TTFT is not quotable.
- 68–114 candidates per E6 point fail in the simulator's KV allocator
  (`tried to load 92.00MB but only 25.49MB is available`) after passing the
  generator's memory bound. Deterministic, reproducible, and it means every "best"
  in E6 is the best of what evaluated. Reconciling the two memory models is
  D10-adjacent and unstarted.

### 2.6 Smaller, any node

- **PR #13** (`docs/slide-deck-ko`) — dispositioned by the consolidation sprint's
  STEP 4.3; see that PR for what was decided.
- **Remote branches are clean.** `origin` holds `main` and nothing else. The 17
  already-merged branches turned out to have been deleted already; the D22 chain
  and the ScenarioLab workspace branches went during the consolidation sprint, the
  latter after verifying their content reached the split repo; the twelve
  `feat/rps-step*` branches and `spike/d14-asym-tp` went after PR #72, once
  `git rev-list --count origin/main..<tip>` was 0 for each. Nothing on the spike
  branch was unique except its throwaway `serving/` edits, which A3 forbids
  merging — its findings live in `docs/d14_spike.md`,
  `docs/upstream_issues/llmservingsim-trace-column-overflow.md`,
  `outputs/d14/evidence/` and D28.
- **`docs/nodes/a5000.md` is thin and says so** — written from committed artifacts,
  not from the node. Fill it in from `scripts/whichnode.sh` when next on that
  machine.

---

## 3. Traps that have each cost a session

Recorded because they are not discoverable from the code.

**Planning a sweep's cost**

- **A low-RPS point is not one point.** Wall time is strongly superlinear as the
  arrival rate falls: at 300 requests on the R2 candidate, 3.3 rps costs **2.63×**
  10 rps and 1 rps costs **10.32×**. A six-point RPS axis weighs 21.4×, not 6×, and
  half of that is the bottom two rows (`docs/sim_cost_profile.md`). Budget the axis
  by summing multipliers, never by multiplying a 10 rps figure by the point count.
- **top-k pruning drops the optimum on P/D and heterogeneous fixtures.** 8.1× faster
  and it turned a FEASIBLE plan (`hp-00077`, 2.963 tok/J) INFEASIBLE. The planner
  warns when this may have happened — read that line. `SLIDE_OUTLINE.md`'s "regret
  0.000 at every K" was measured on an aggregated-heavy fixture and does not
  transfer. Fixing the surrogate is `WORK_ORDER_rps_aware.md` STEP 4.

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
- **The Chakra converter no longer cares about `PATH` — it cares which venv you
  launch.** Until D27 `python -m serving` shelled out to a bare `python -m chakra`,
  so the venv had to be *on* `PATH` and not merely invoked by full path; that is
  what D23 turned out to be. Since D27 the frontend converts in-process, so there
  is no second interpreter to resolve and no `PATH` export to forget — but a venv
  without a chakra at `protobuf>=7.35.1` now raises at the first conversion instead
  of silently converting wrongly.
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
