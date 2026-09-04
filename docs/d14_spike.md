# D14 spike — asymmetric TP per phase is representable; `auto` does not refuse it, it gets it wrong

*`WORK_ORDER_spikes.md` STEP B. Run 2026-09-04 on the NPU node, branch
`spike/d14-asym-tp` off STEP A's `e928de5`. Simulation only — no hardware
measurement (absolute rule A1). The `serving/core/config_builder.py` prototype
is **spike-only and must not be merged** (A3). Artifacts: `outputs/d14/`.*

## The short answer

§1.2's claim holds: **`config_builder.py` alone is what forbids `tp_d = 2·tp_p`.**
ASTRA-Sim never did. But the spike found the constraint is worse than D14 recorded
and cheaper to lift than D16(b) assumed:

- **`auto` does not reject the asymmetric configuration. It builds a wrong one.**
  `A40 tp4 prefill + RNGD tp8 decode` — the exact shape D16(c) names as
  industry-recommended and unavailable — compiles today to `npus_count: [5, 3]`.
  That is **15 ranks for instances that occupy 16**, and neither 5 nor 3
  corresponds to any TP group. No error, no warning; the simulator runs.
- The fix is **two places, not one**, and both are in `config_builder.py`.
- Splitting a TP group across two topology dims is **not free**: it changes TPOT
  by **24.4 %**, eight times the RNGD profile's own 3.1 % error (D22). But the
  cost is entirely a communication term, and a **per-dim `link_latency` of 4× the
  scalar closes it to 0.008 %** — using a config field upstream already supports.

## B.1 — ASTRA-Sim accepts N-dimensional topologies (no blocker)

| check | result |
| --- | --- |
| `astra-sim/workload/Workload.cc:275-296` | reads `involved_dim` as an arbitrary-length `bool_list`, default 5 dims |
| analytical `NetworkParser` | `dims_count` derives from the topology list length; **no cap** |
| upstream examples | `Ring_FullyConnected_Switch.yml` ships `npus_count: [2,8,4]` — 3-D is a released configuration |
| our own fixtures | already 2-D (`[4, 3]`) |
| standalone run | a 3-D `[2,2,3]` config parsed and started, 8 `RingTopology` lines, no stderr |

The work order's stop condition ("B.1에서 ASTRA-Sim이 3-D를 거부") did not occur.

## B.2 — the prototype, and what it revealed about the constraint

`topology_mode` is an opt-in top-level cluster-config key. Absent (`auto`) is the
existing path, unchanged. `slab3d` lays instances out as `[g, 2, n_slabs]`, where a
tp=g instance takes half a slab and a tp=2g one takes a whole slab.

**§1.2 named one call site; there are two.** The work order located the constraint
in `_compute_network_dims`, and it is right that the integer division there
(`total_npu // num_instances`) is what folds every instance into one size. But
`_resolve_dp_groups` then hands **every instance the same `local_dim`**, so even
with correct dims each instance would still claim its TP spans dim 0 only. §1.2's
underlying claim — "`tp_dim` is already a per-instance field" — is true of the data
structure and false of the assignment. Both had to be bypassed.

### B.2-2 — the `auto` path is byte-identical

Uniform colocated tp4 ×2, 20 requests, one run each:

```
sha256  36c37a0d5cf555eb619b0088c6fe2ee2848cd4b2a8ad343b4849ed7b228a28f5  equiv_auto.csv
sha256  36c37a0d5cf555eb619b0088c6fe2ee2848cd4b2a8ad343b4849ed7b228a28f5  equiv_slab3d.csv
```

Byte-identity alone would also be satisfied by a patch that silently ignored the
key, so the builder was asked directly:

| | mode the builder saw | `npus_count` | `tp_dim` |
| --- | --- | --- | --- |
| `auto` | `auto` | `[4, 2]` | `[[T,F], [T,F]]` |
| `slab3d` | **`slab3d`** | `[4, 2]` | `[[T,F], [T,F]]` |

Two different code paths converging on one topology — which is what §1.2 predicted
(`[4,2,1]` with the trailing 1 dropped). Full suite: **460 passed**, golden included.

Uniform P/D under `slab3d` raises `ValueError`, as the work order requires. The
message reports the mechanical cause (prefill 8 ranks + decode 4 = 12, not a
multiple of the 8-rank slab) rather than §1.2's reason (uniform P/D is out of
scope); accurate, but it would read better the other way round.

### B.2-3 — what splitting a TP group across two dims costs

One RNGD tp8 instance, flat `[8]` versus `[4,2]` with `tp_dim [T,T]`, 20 requests.
`_sync_system_collective_dims` and `_create_network_config` followed `len(dims)`
with no change, as §1.2 said they would.

A note on reading the logs: `iteration N finished, C cycles` is a **cumulative**
simulated timestamp in ns, not that iteration's cost — iteration 0 prints the first
request's arrival time. Per-iteration cost is the consecutive difference within one
NPU. The first pass here compared the cumulative field and reported drift as cost;
the numbers below are the corrected ones.

| | flat `[8]` | `[4,2]` | |
| --- | ---: | ---: | ---: |
| final cumulative time | 28.913 s | 22.512 s | |
| of which exposed comm | 16.833 s (58.2 %) | 10.413 s (46.3 %) | |
| summed per-iteration cost | 57.559 s | 44.715 s | **−22.31 %** |
| TPOT p50 | 33.63 ms | 25.41 ms | **−24.43 %** |
| TPOT p99 | 35.63 ms | 27.96 ms | −21.54 % |
| TTFT p50 | 822 ms | 1055 ms | +28.36 % |

Of 1674 comparable iteration steps, **0 are identical**.

**The mechanism is the predicted one.** Mean per-iteration cost difference
−7,672,856 ns; mean exposed-communication difference −7,682,511 ns. **99.9 % of the
difference is communication**, compute is untouched.

**The magnitude is 2× the prediction.** Over 32 layers that is 239,777 ns per
layer against the work order's `2·3·link_latency` = 120,000 ns — a ratio of 2.00,
close enough to suggest a missing factor of two in the predicted term rather than a
different mechanism.

24.4 % ≫ 3.1 %, so by §7's risk table this is **"go, correction needed"** and the
correction value must be proposed.

### The correction, measured rather than derived

`_normalize_network_dim_values` already accepts a per-dimension `link_latency`
list — **no patch is needed for the correction itself**. So it was swept rather
than argued:

| dim-1 `link_latency` | TPOT p50 | vs flat `[8]` |
| ---: | ---: | ---: |
| flat `[8]` reference | 33,627,896 | — |
| 20,000 (the scalar) | 25,413,534 | −24.43 % |
| **80,000** | **33,625,368** | **−0.01 %** |
| 200,000 | 48,965,308 | +45.61 % |
| 400,000 | 74,386,128 | +121.20 % |

A linear fit over the four points (`R² = 0.99988`) puts the crossing at 81,807 ns,
but no extrapolation is needed: the measured point at **80,000 ns — exactly 4× the
scalar** — reproduces the flat topology across every metric:

| | flat `[8]` | `[4,2]`, dim-1 = 80,000 | diff |
| --- | ---: | ---: | ---: |
| TTFT p50 / p95 / p99 | 822 / 1279 / 1322 ms | 822 / 1279 / 1322 ms | −0.002 / −0.001 / −0.001 % |
| TPOT p50 / p95 / p99 | 33.63 / 35.56 / 35.63 ms | 33.63 / 35.56 / 35.63 ms | −0.008 / −0.007 / −0.007 % |
| latency p50 / p99 | 22.04 / 27.62 s | 22.04 / 27.62 s | −0.007 / −0.007 % |

**Worst deviation 0.008 %**, against a 3.1 % tolerance.

**What this does not establish.** The 4× factor was measured at one link bandwidth
(16), one model, one TP degree, and a 2-dim topology. Whether it transfers to
another `link_bw`, another split ratio, or the third dimension of a `[g,2,n]` layout
is untested. It is a calibrated constant for this configuration, not a law.

## B.3 — asymmetric P/D

`experiments/configs/clusters/pd-asym-a40tp4-rngdtp8.json` — which is committed to
`main` but **inert there**: `main` does not read `topology_mode`, does not reject
it either, and folds the fixture into `[5, 3]`, so a run just hangs. The sibling
`.provenance.yaml` says so at the top. Node 0 A40 ×4 prefill
tp4, node 1 RNGD ×8 decode tp8, `topology_mode: slab3d`. Power blocks and the
A40↔RNGD `link_bw` 12.6 are taken verbatim from the planner's own emitted
`cluster.json` for this hardware pair (`outputs/.hp-slo-margin18-tight-pd-rngd-gpu`),
not invented — the colocated fixture's 35.2 is an A40↔A40 value and does not apply.

**The compilation result is the headline, independent of whether the run completes:**

| | `npus_count` | `tp_dim` | ranks |
| --- | --- | --- | --- |
| `slab3d` | **`[4, 2, 2]`** | `[T,T,F]` / `[T,T,F]` | 0-7 A40, 8-15 RNGD |
| `auto` (today) | **`[5, 3]`** | `[T,F]` / `[T,F]` | 15 ranks for 16 |

`auto`'s arithmetic is exactly what D14 describes: `total_npu` 16, `total_pp` 3
(prefill counts double), `16 // 3 = 5` → `[5, 3]`. D14 predicted this outcome
("would produce wrong numbers, not an error") from reading the code; this is the
first time it has been **measured on the configuration D16(c) names**.

### The second upstream bug: a 3-D collective tag does not survive the trace round-trip

Both B.3 runs died in one second:

```
TypeError: formatter() missing 1 required positional argument: 'misc'
  serving/core/trace_generator.py:1557
```

Not the spike's patch — the same fixture under `auto` does not raise. Wrapping
`formatter` in-process (no `serving/` edit) named the row:

```
SHORT ROW (10 fields): ('o_proj_5', ..., 'ALLREDUCE:1,1,07503872', 'NONE')
                                          |-- 15 chars --|
```

`serving/core/utils.py::_FMT` gives `comm_type` a **15-character left-aligned
column with no separator**. `_with_dim` encodes the involved dims into the tag, so
a 3-dimensional topology produces `ALLREDUCE:1,1,0` — **exactly 15 characters**,
leaving zero padding, and it abuts `comm_size`. `generate_trace` then re-reads its
own file with `re.findall(r'\S+')` (`trace_generator.py:1514`) and the row comes
back with 10 fields. **Fixed-width writer, whitespace-splitting reader.**

| tag | chars | 15-wide column |
| --- | ---: | --- |
| `ALLREDUCE:1,0` (2-D, today) | 13 | fits |
| `ALLREDUCE:1,1,0` (3-D) | **15** | **overflows** |
| `ALLREDUCE:1,1,0,0` (4-D) | 17 | overflows |

The bug pre-dates this spike and belongs to upstream; it is simply **unreachable at
≤ 2 dims**, and 3-D is the first thing to reach it. (`REDUCESCATTER:1,0` is 17
characters and would overflow at 2 dims, but the MoE/EP path writes `EXPERT` rows
with plain spaces and never goes through `formatter`, so it escapes.)

Per §7 a `serving/` bug found mid-spike is recorded, not fixed. The widening is
therefore applied **at runtime** by `experiments/scripts/b3_widecol_run.py`
(`comm_type` 15 → 24); `serving/` on disk is untouched. **Proposed fix**: widen the
column, or — better, since it removes the class rather than one instance — have
`generate_trace` keep the rows it already has in memory instead of re-parsing the
file it just wrote.

### What B.3 established, and what it did not

With the column widened at runtime, the asymmetric configuration **starts and runs**:

| | ranks reporting `iteration 0` | outcome |
| --- | --- | --- |
| `auto` `[5, 3]` | **NPU[7] only** | exit 4 — not one progress line in 300 s |
| `slab3d` `[4, 2, 2]` | **NPU[0], 7, 8, 15** | 41 progress ticks, 4 iterations, then stalls |

This is the concrete proof that `slab3d` fixes the topology: all four instance
boundary ranks exist and report. Under `auto` only one does — rank 15 has nowhere
to live in a 15-rank topology, the collective never completes, the child dies, and
the harness spins on EOF exactly as STEP A described. **D14's "would produce wrong
numbers, not an error" is too generous: it produces no numbers and hangs.**

### A stall that looked like D23 and was the spike's own bug

The first corrected-column run then stalled, in what looked exactly like D23:

```
livelock_watch: LIVELOCK -- Instance[0] held running=1, mem=9.304 %,
livelock_watch: waiting non-decreasing at 19, for 40 consecutive ticks.
```

Prefill pinned at one running request, waiting flat at 19, memory flat — the shape
`docs/d23_spike.md` closed as *not reproduced*. It was reported as a reproduction
of D23. **That was wrong, and this records the correction.**

Two controls said it was not D23: a uniform A40 tp4 P/D completes in **78 s**, and a
uniform RNGD tp8 P/D completes in **90 s**, both on the same trace and the same
flags. P/D alone does not pin, and neither does RNGD.

The cause was in this spike's own patch. §1.2 assigns `tp_dim` per instance as
"**half-slab and prefill** → `[T,F,F]`, full-slab **decode/colocated** →
`[T,T,F]`". The prototype keyed the choice on **footprint** instead: a prefill tp=g
instance occupies 2g ranks (compute plus senders), so it was handed `[T,T,F]` and
declared an **8-rank allreduce for a 4-rank TP group**. The prefill batch then
waited forever for four ranks that never join — which presents precisely as
"prefill pinned, memory flat".

The log says it directly: all four boundary ranks report `iteration 0`,
`Scheduling existing batch #0 to NPU[7]` is the last line, and nothing follows.

§1.2 was right and specific; reading it as a statement about slab occupancy rather
than about the collective is what cost the run. The fix keys `tp_dim` on `pd_type`,
not on width — and it is the reason a spike prototype gets a real workload rather
than a unit test: the unit check showed `[4,2,2]` and looked correct.

### With `tp_dim` keyed correctly, the asymmetric configuration completes

| run | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |
| --- | ---: | ---: | ---: | ---: |
| A40 tp4 colocated (prefill reference) | 119.8 ms | 277.5 ms | 18.88 ms | 19.66 ms |
| RNGD tp8 colocated (decode reference) | 822.2 ms | 1321.5 ms | 33.63 ms | 35.63 ms |
| **asym P/D, scalar latency** | **265.0 ms** | **634.6 ms** | **24.56 ms** | **24.75 ms** |
| **asym P/D, dim-1 latency 80,000** | **271.7 ms** | **640.6 ms** | **32.25 ms** | **32.45 ms** |

`exit 0`, 21 rows, 86 s and 83 s. The work order's consistency check — TTFT beside
the prefill hardware's standalone run, TPOT beside the decode hardware's — passes
in both directions and for the right reasons:

- **TTFT 265 ms** sits between A40's 120 ms and RNGD's 822 ms and close to the A40
  end (2.21×). Prefill is running on the A40, which is what the fixture says.
- **TPOT 24.6 ms is *faster* than RNGD standalone's 33.6 ms** (0.73×). Also the
  expected direction: once prefill is disaggregated, the decode instance is no
  longer interleaving prefill chunks.

This is a consistency check, not an accuracy claim. P/D changes batching and adds a
handoff, so exact agreement would be suspicious rather than reassuring.

**An unplanned corroboration of B.2-3's correction.** The 4× dim-1 `link_latency`
was fitted on a completely different experiment — one colocated tp8 instance, 2-D,
`link_bw` 16. Applied here (3-D, two instances, two vendors, `link_bw` 12.6) it
moves TPOT from **0.73× to 0.96×** of the RNGD standalone reference. That is one
transfer, not a validation, and the target is not exactly 1.00× anyway; but it is
evidence in the direction the correction predicts rather than against it.

## Go / no-go

**Go.** §1.2's claim survives the spike: nothing in ASTRA-Sim forbids asymmetric TP
per phase, and a change confined to `config_builder.py` expresses it. What the
spike changes about the plan:

| §1.2 said | the spike found |
| --- | --- |
| ASTRA-Sim accepts 3-D `involved_dim` | confirmed — upstream ships a 3-D example |
| `_compute_network_dims` is the constraint | **incomplete** — `_resolve_dp_groups`'s uniform `local_dim` is the other half |
| `tp_dim` is already per-instance | true of the structure, **false of the assignment** |
| `_create_network_config` / `_sync_system_collective_dims` need no change | confirmed |
| (not anticipated) | splitting a TP group across dims costs **24.4 %** TPOT, correctable to 0.008 % with a per-dim `link_latency` upstream already supports |
| (not anticipated) | a 3-D collective tag overflows the trace row's 15-char column |

**Cost to productionise**, in rising order of risk:

1. `config_builder.py`: the two sites above. Small, and `auto` stays byte-identical.
2. `utils.py::_FMT`: widen `comm_type`, or stop re-parsing the trace file. Upstream's
   bug; a one-line workaround exists but the round-trip is the real defect.
3. **Calibration**: the per-dim `link_latency` is a fitted constant, measured at one
   bandwidth and one split. Before any *number* from a slab3d plan is quoted it needs
   a proper calibration domain, the way `docs/tier0_calibration.md` scopes Tier 0.

**Not go, yet, for results.** Ranking claims may use this; absolute latency or energy
numbers from a slab3d plan may not, until item 3 is done.

## What this cost, and the two mistakes worth keeping

**`iteration N finished, C cycles` is cumulative.** The first B.2-3 analysis diffed
it as if it were per-iteration cost and reported ~3.2 × 10⁹ "cycles per iteration".
Corrected by differencing within each NPU.

**`tp_dim` was keyed on footprint, not on the collective.** §1.2 says "half-slab and
prefill → `[T,F,F]`"; the prototype gave every 2g-wide instance `[T,T,F]`, including
prefill, whose extra ranks are senders. The result was an 8-rank allreduce for a
4-rank TP group, presenting as prefill pinned with flat memory — which was briefly
recorded here as a reproduction of D23. Two controls (uniform A40 P/D 78 s, uniform
RNGD P/D 90 s) refuted that, and the fixture that finally completed refutes it
conclusively. **D23 remains unreproduced**, exactly as `docs/d23_spike.md` left it.

The unit check that passed before the run showed `[4,2,2]` and looked right; only a
real workload exposed the `tp_dim`. That is the argument for B.2's insistence on
equivalence *runs* rather than equivalence *assertions*.
