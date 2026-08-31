# HeteroPilot — current state and what to do next

> **This is the live handover.** Written 2026-08-31 at `main` = `3d59035`, after
> PRs #17–#22. It is **node-agnostic**: every open item below says which machine it
> needs. The older handovers are kept as historical record of a single move and
> should not be read as current status:
> `docs/HANDOVER.md`'s previous content (A5000 → A40, 2026-08-18),
> `docs/HANDOVER_NPU.md` (→ NPU, 2026-08-25),
> `docs/HANDOVER_A40.md` (→ A40, 2026-08-26; **its §1 is done**).
>
> Authority order is unchanged: `WORK_ORDER_heteropilot.md` → `docs/deviations.md`
> → `CLAUDE.md` → this file.

---

## 0. First command on any machine

```bash
bash scripts/whichnode.sh
```

**This repository moves between an A40 node, an A5000 node and an NPU node, and
every one of them reports the hostname `s8` on the same kernel.** Nothing
committed can tell you which machine you are on; the detector probes the
accelerators and prints what can actually be run here. It names one of
`docs/nodes/{a40,a5000,npu}.md` — read that one, not all three.

The device set has changed **three times in four days** (RNGD cards 4 → 3 → 4;
ATOMs held by pods, then free). Re-check before every run, and never key off
`hostname` or compute a card from `index // 8`.

Then:

```bash
pytest          # expect 284 passed
ruff check .
mypy planner/
```

All three are green at `3d59035`.

---

## 1. Status

| Phase | Status |
| --- | --- |
| 0 Baseline · 1 Inventory/islands · 2 Offline planner (MVP) | ✅ done |
| 3 Heterogeneous profiles | ✅ done (7 profiles, `CsvProfileImporter`) |
| 4 Real deploy + calibration | ✅ CUDA. NPU launcher still a stub |
| 5 Topology-aware P/D | ✅ core. Network-aware routing deferred |
| 6 Online replanning | ⛔ not started — **requires explicit user approval** |

**Accelerator profiles** (`profiles/accelerators/`) **and what they are allowed to claim:**

| profile | `sim_hardware` | `source` | usable in candidate generation |
| --- | --- | --- | --- |
| `a40.yaml` | A40 | measured | yes |
| `a5000.yaml` | A5000 | measured | yes |
| `furiosa_rngd.yaml` (PE-as-device) | RNGD | measured | yes |
| `furiosa_rngd_card.yaml` (card-as-device) | RNGD-CARD | measured | yes |
| `rtxpro6000.yaml` | RTXPRO6000 | vendor_spec | yes, labelled |
| `rbln_atom.yaml` | **null** | placeholder | **no — fails loud** |
| `ascend_target.yaml` | **null** | placeholder | **no — fails loud** |

Calibrations (`profiles/calibration/`): `a40.yaml`, `rngd.yaml`, `rngd_card_edf.yaml`. All bucket-scoped —
**do not extrapolate outside the bucket named in the file.**

### What the last cycle established (PRs #17–#22)

- **The cross-vendor KV path is measured on both legs.** GPU leg (A40 → host,
  sustained pinned D2H) 25.71 GB/s single stream, 82.63 GB/s across 8 GPUs — only
  40 % of ideal, the host path saturates ~83 GB/s. NPU leg (host → RNGD)
  3.77 / 7.60 / 15.36 / 26.27 GB/s at 1/2/4/8 streams, 87.1 % of ideal. Both P/D
  fixtures now carry composed values at `source: measured` (12.6–13.0 GB/s), where
  a 35 GB/s placeholder had stood — 2.7–4.5× too optimistic.
- **And it changed no prediction.** All 16 SLO-sweep winners identical; Exp 3
  reproduces byte-identically; the simulator-side test moves TPOT by +0.012 %
  between 35 and 13 GB/s. One request ships 114.7 MB, so 3–9 ms against a
  30.7-second mean latency. **For this hardware pair the NPU leg sets the fabric
  bandwidth and the GPU leg's exact value has low leverage.**
- **Three retractions**, recorded in the open the way this project does it:
  **D18** (NPU multi-stream figures were peaks quoted where a sustained rate was
  needed — scaling law held, levels ~25 % high), **D19** (the −71 % card TTFT error
  was an arrival-pattern mismatch in the validation harness, not the scheduler;
  matched arrivals give −5.1 %, and both TTFT calibrations were refitted — the card
  fit error went 2.34 "unusable" → 0.103), and **D20** (ATOM layerwise profiling
  blocked).
- **ATOM is usable and partly measured, still not profiled.** 15.047 GiB largest
  allocation, idle 19.44 W, active 68.73 W at 95.1 % utilisation, per-card power
  additive. No perf bundle, so it stays out of candidate generation and Exp 4.
- **`CLAUDE.md` no longer asserts which machine it is on** — see §0.

---

## 2. Next work, in priority order

### 2.1 RNGD concurrency envelope — **needs the NPU node** · branch ready

**The largest open risk.** Every card-fixture number in
`experiments/results/pd_slo_sweep.md` — the three-regime answer, the 480 ms p99
TTFT winner, 5.55 rps goodput, tokens/J — assumes each RNGD card runs at **~76
concurrent sequences**. The highest ever measured is **32**.

| | output tok/s per card | implied concurrent |
| --- | ---: | ---: |
| real, validation run | 584 | 16.6 |
| real, highest ever tested (c32) | 646.9 | ~32 |
| **sim, the sweep's winner** | **1767** | **76** |

Extrapolating the measured c16→c32 exponent (0.598) to c76 gives ~1090 tok/s —
the simulator is **~1.6× optimistic at 2.4× beyond the measured envelope.**

Work order with the run, the traps and the three possible outcomes:
**`docs/npu_concurrency_envelope_work_order.md`**, on branch
**`fix/scaling-curve-provenance-and-npu-envelope`** (pushed, not merged — merge it
with the results).

That branch also carries a correction: I claimed the c1–c32 curve was prose-only
with no committed artifact. **That was wrong** —
`outputs/rngd_edf_bundle/edf/real_c{1,2,4,8,16,32}.json` are committed and
reproduce the table exactly.

### 2.2 ATOM layerwise bundle (D20) — **needs the NPU node**, and probably the vendor

Both documented routes are blocked, and neither is a matter of effort:

- **vLLM route** — packaging now works, but both vllm-rbln paths reject the
  profiler's fixed `load_format="dummy"` / `enforce_eager=True`, and fixing either
  means editing `profiler/`, **pristine until Phase 5**.
- **rebel harness route** — `experiments/scripts/profile_atom.py` runs clean (284 shots, zero compile
  failures) but cannot produce *device* time. Host I/O exceeds the kernels
  (999.7 µs at 8 MB against RNGD device spans of 3–200 µs), and the device
  tracer's protobuf schema is undocumented.

**Suggested move: ask Rebellions for the `rebel._C.profiler` trace schema.**
FuriosaAI's own EDF profiler is what rescued RNGD — it took decode prediction from
+25.7 % to −3.1 %. The same shape of answer is what ATOM needs.

Do **not** ship a bundle from the current attempt.
`outputs/atom_profile/layerwise_attempt/` is **not a bundle** — 166 of 284 shots
are baseline-dominated. An earlier constant-floor subtraction **passed contract
validation and imported cleanly while inflating elementwise layers 8–25×**; only
the RNGD comparison caught it.

### 2.3 TTFT tail under-prediction — **any node** (simulation only)

With arrivals matched, the card profile is −5.1 % on mean TTFT but still **−10 to
−17 % at p90–p99**. Plausibly bucket quantisation — the vendor artifact charges
+10.9 % more prefill tokens than the prompts contain, and the simulator
interpolates on exact counts so it cannot see this. **That is a hypothesis, not a
measurement.**

### 2.4 D14 — asymmetric TP per phase — **any node**, largest change

`tp_p == tp_d` is enforced, so `A40 tp4 prefill + RNGD tp8 decode` — the shape
NVIDIA Dynamo and AWS Neuron recommend — **cannot be enumerated at all**. Card-as-
device sidesteps it by folding TP=8 inside the device; it does not lift it.
Lifting it means teaching the compiler to emit non-uniform instance sizes, a
simulator-side change.

### 2.5 Smaller, any node

- **PR #13** (`docs/slide-deck-ko`) has been open since before this cycle.
- **Remote branches**: 20-odd merged branches still exist on `origin`. Local ones
  are cleaned; remote cleanup was left as your call.
- **`docs/nodes/a5000.md` is thin and says so** — it was written from committed
  artifacts, not from the node. Fill it in from `scripts/whichnode.sh` when you are
  next on that machine.

---

## 3. Traps that have each cost a session

Recorded because they are not discoverable from the code.

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

- **`experiments/scripts/bench_furiosa_endpoint.py` ignores `arrival_time_ns`** and fires everything at
  once, while `python -m serving` replays it. Feeding both the same file compares a
  burst against a spread arrival process; the difference lands entirely in TTFT.
  That is D19. Use `outputs/envcheck/rngd20_burst.jsonl` on the simulator side.
- **`python -m serving` shells out to a bare `python -m chakra`**, so the venv must
  be on `PATH`, not merely invoked by full path:
  `export PYTHONPATH=$PWD && export PATH="$PWD/.venv/bin:$PATH"`.
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
- **`furiosa.torch` and `rebel` must import after `torch`**, behind the
  `# isort: off` guards. `ruff check --fix` reordering them breaks every run.
- **Multi-device work on NPUs uses one subprocess per device**, not threads.

---

## 4. Invariants that are easy to break

Beyond `CLAUDE.md`'s absolute rules, three that this cycle exercised:

1. **Absolute rule 3 covers hardware *presence*, not just hardware numbers.** Never
   claim a result from hardware `scripts/whichnode.sh` does not list. Artifacts
   measured on another node stay valid as measurements *of that node* — do not
   re-run, extend, or relabel them.
2. **Retract in public.** D18, D19 and D20 each state what was claimed, what was
   measured, and what it changes. When a committed number turns out wrong, correct
   every site that carried it and say so — do not quietly overwrite.
3. **A pruning stage must be a relaxation of the feasibility test**, and **a mock
   predictor must respect the same physics as the bounds.** Both have been violated
   before and both were caught by the oracle-agreement test.
