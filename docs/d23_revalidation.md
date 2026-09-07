# D23 re-validation — what the harness faults did and did not change

*`WORK_ORDER_d23_fix_revalidation.md` STEP 3. Run 2026-09-07 on the NPU node,
`main` = `5f97108` (PR #57: D25-a, D25-b, D26 merged). Simulation and code only —
no hardware measurement (absolute rule A1). Artifacts: `outputs/d23fix/`.*

## What is being re-validated, and against what

Two harness faults produced everything D23 recorded, and they are not the same
fault:

- **D26** — the Chakra converter ran under whatever interpreter `PATH` found, so the
  ASTRA-Sim workload graph could be built by a `chakra` with protobuf 6.33.1 against
  the `>= 7.35.1` its generated code needs. Different `.et` bytes, and a P/D run that
  never finishes its first prefill batch. **This is D23's symptom**, 6 hangs of 6
  against 18 completions of 18.
- **D25** — every concurrent simulation shared one cwd-relative `tmp__mem/*.json`, so
  they deleted each other's file. **13 of 64** bare processes died that way. Crashes,
  not hangs.

The question STEP 3 exists to answer is whether results produced under those faults
can be trusted. The short answer, established below and in §2, is **yes for anything
that completed, and nothing for anything that did not** — a run either reproduced the
committed answer byte for byte or produced no answer at all.

## §1 — Could the shared temp file have changed a number, rather than crashing?

The spike could not tell. It launched 64 copies of **one** candidate, and swapping
identical files changes nothing; the real sweep ran 64 **different** ones. So the
silent half of the D25 race was unobservable there and had to be settled another
way.

`experiments/scripts/d23_memjson_census.py` settles it without simulating: it calls
the real `serving.core.config_builder.build_cluster_config` on every candidate's
committed `cluster.json` and hashes the `local_mem` / `remote_mem` / `cxl_mem`
objects of the `memory_expansion.json` it generates.

*(The work order flagged as an open question whether `build_cluster_config` imports
in `.venv` without torch. It does — `WORK_ORDER_spikes.md` STEP B called it directly
and repeatedly. The census uses the real builder, so the fallback of duplicating the
JSON rules by hand, and the lower confidence that came with it, does not apply.)*

**474 candidates** across both margin18 tight work directories. No compile failures,
and **no candidate takes the single-form path** — every one of them goes through
`main.cc`'s `tmp__mem` branch, so before D25 every one was exposed.

```
remote_mem: 474 candidates, 2 distinct contents
  x366  {"mem-bw":256.0,"mem-latency":0.0,"memory-type":"PER_NODE_MEMORY_EXPANSION","num-devices":2}
  x108  {"mem-bw":256.0,"mem-latency":0.0,"memory-type":"PER_NODE_MEMORY_EXPANSION","num-devices":1}
  fields that vary: num-devices only
```

**This is exactly what §1.2 predicted.** `mem-bw` is 256.0 and `mem-latency` 0.0 in
all 474 — the fixtures never override the planner's defaults — so `num-devices` is
the only field a swap could change.

And `num-devices` does not enter any timing arithmetic. In
`extern/memory_backend/analytical/AnalyticalMemory.cc` it only sizes the per-device
queue vectors (`:92`, `:105`) and indexes them (`:140`, `:210`):

| swap direction | consequence |
| --- | --- |
| 2-node candidate reads a 1-node file | vectors sized 1, `device_id = 1` indexes out of range — undefined behaviour, in practice a crash. The likely source of the spike's **8 SIGABRTs** |
| 1-node candidate reads a 2-node file | vectors sized 2, only index 0 used — **numerically identical**, one queue unused |

So the silent half of the D25 race is **numerically inert**, and the direction that
matters crashes rather than lying. A completed run cannot have been quietly
corrupted by it.

**§3.4 is therefore not triggered.** It was conditional on `mem-bw` or `mem-latency`
differing between candidates, and they do not.

### One defect in the census worth recording

Its first version reported *"every candidate's memory JSON is identical"* from
**zero** candidates — the glob was one directory level off, and the reassuring
verdict printed anyway. A census that draws a conclusion from an empty set is worse
than no census. It now exits 2 and says so, and the comment in the script records
why the guard is there.

## §2 — Re-running the rows D22's verdict rests on

Same script, same parameters as `experiments/results/pd_slo_sweep_margin.md`,
`--workers 32`, and **deliberately no isolation wrapper** — proving D25 removed the
need for it is half the point of running it this way.

| fixture | elapsed | timeouts | exit-code failures | completed sims |
| --- | ---: | ---: | ---: | ---: |
| `pd-rngd-gpu-card` | 46 min | **0** | 0 | 199 |
| `pd-rngd-gpu` | 57 min | **0** | 0 | 225 |

The original margin18 run took **4 h 37 m**; this took **1 h 43 m**. Zero timeouts
either way, with no wrapper.

**Every reported field is identical, to the last decimal.**

| | committed | re-run |
| --- | --- | --- |
| `feasible` / `generated` / `evaluated` (gpu) | True / 496 / 492 | True / 496 / 492 |
| `feasible` / `generated` / `evaluated` (card) | True / 468 / 468 | True / 468 / 468 |
| winner | `agg[cuda:tp4]` | `agg[cuda:tp4]` |
| `tokens_per_joule` | 2.5954323001631323 | 2.5954323001631323 |
| `p99_ttft_ms` | 15070.10466225 | 15070.10466225 |
| `p99_tpot_ms` | 35.58226791 | 35.58226791 |
| `slo_goodput_rps` | 5.279963373257342 | 5.279963373257342 |
| `average_power_w` | 1292.5101785714285 | 1292.5101785714285 |
| `plan_id` | hp-00075 / hp-00051 | hp-00075 / hp-00051 |

**D22's verdict holds.** "Every RNGD candidate rejected, winner `agg[cuda:tp4]` at
2.595 tok/J" reproduces exactly on a harness with all three faults fixed. The work
order's risk row — *a differing row is evidence of silent contamination* — does not
fire.

This agrees with §1 by a second route: `num-devices` is the only field a swap could
change, it does not enter any timing arithmetic, and so contamination could only
crash, never lie.

**One scope note.** This sweep is `--ttft-ms 64000`, the *loose* point, and its
`evaluated` counts match the committed run — so the loose point never had timeouts
to begin with. Its winner is an `aggregated` (single-instance) candidate, which is
exactly the class D26 does not touch. The 71-of-222 and 126-of-252 timeouts D23
recorded are at the **tight** points, and belong to §3.

## §2.1 — What the committed timeout list says about D23's own description

Classifying the committed tight-run timeout lists by candidate class:

| fixture | `mix_` | `pd_` | aggregated **dp2** | **dp1** |
| --- | ---: | ---: | ---: | ---: |
| `pd-rngd-gpu` | 60 | 54 | **12** | **0** |
| `pd-rngd-gpu-card` | 24 | 41 | **6** | **0** |

**Not one single-instance candidate ever timed out** — and there were 144 `dp1`
work directories to draw from. Meanwhile 18 timeouts were `aggregated dp2`, which
are not P/D at all.

So D23's opening sentence — *"Every `pd_*` and `mix_*` candidate that the tight-TTFT
sweeps need has stopped terminating"* — is wrong in both directions: not all of
them, and not only them. **The discriminator is instance count, not P/D**, which is
D26's discriminator and not D25's.

Everything this work order measured lines up on that one axis:

| | instances | sensitive to the interpreter? |
| --- | ---: | --- |
| `bench/examples` Llama-3.1-8B, Qwen3-32B | 1 | no — byte-identical across the edits |
| `bench/examples` Qwen3-30B-A3B (MoE) | **2** | **yes** — its hash flips with `PATH` |
| the R2 P/D anchor | **2** | **yes** — 6 hangs of 6 |
| sweep `dp1` candidates | 1 | no — 0 timeouts of 144 |
| sweep `dp2` / `pd_` / `mix_` | **≥2** | **yes** — all 197 timeouts |

A mis-converted `.et` graph breaks where collectives span more than one rank group;
a single-instance run never takes that path.

## §3 — The tight-TTFT regime, closed

Same command, `--ttft-ms 500,8000`. The committed run timed out **71 of 222** and
**126 of 252** simulations here and printed INFEASIBLE at all four points.

| fixture | elapsed | timeouts |
| --- | ---: | ---: |
| `pd-rngd-gpu-card` | 48 min | **0** |
| `pd-rngd-gpu` | 61 min | **0** |

### The verdict flips at all four points

| fixture | TTFT SLO | committed | re-run | winner | tok/J | p99 TPOT |
| --- | --- | --- | --- | --- | ---: | ---: |
| card | ≤ 500 | INFEASIBLE | **FEASIBLE** | `agg[cuda:tp2]` | 1.4831 | 37.93 |
| card | ≤ 8000 | INFEASIBLE | **FEASIBLE** | **`P[cuda:tp2] D[cuda:tp2]`** | 1.7725 | 39.90 |
| gpu | ≤ 500 | INFEASIBLE | **FEASIBLE** | **`P[cuda:tp4] D[cuda:tp4]`** | 2.2051 | 37.27 |
| gpu | ≤ 8000 | INFEASIBLE | **FEASIBLE** | **`P[cuda:tp4] D[cuda:tp4]`** | 2.2051 | 37.27 |

`generated`/`evaluated` are unchanged (468/468, 496/492): the enumeration was
always the same, and a timeout still counted as *evaluated*. What changed is the
outcome of 197 of those evaluations.

**The work order predicted this to the decimal.** §3.3 said to check the committed
winner `P[cuda:tp4] D[cuda:tp4]`, "p99 TPOT 37.27 → ×1.18 = 43.98 ms, expected to
pass". Measured: **37.271541989999996**, ×1.18 = 43.98 against the 50 ms SLO.
Passes.

So the four "no currently available configuration satisfies all constraints"
verdicts rested on 197 unevaluated candidates. They are now evaluated, and the
answer is the opposite one. **This is outcome (ii) of the three the work order
allows: the tight row flips, and the 3-regime table closes.**

### D22 survives: no RNGD candidate passes, and no mixed one either

The summary JSON records only the winner, so the 424 cached per-candidate records
were filtered directly against the SLO (`p99 TTFT ≤ point`, `p99 TPOT × 1.18 ≤ 50`):

| | cuda | **rngd** | **mixed** |
| --- | ---: | ---: | ---: |
| gpu ≤ 500 | 13 of 162 | **0 of 45** | **0 of 18** |
| gpu ≤ 8000 | 14 of 162 | **0 of 45** | **0 of 18** |
| card ≤ 500 | 2 of 162 | **0 of 12** | **0 of 25** |
| card ≤ 8000 | 8 of 162 | **0 of 12** | **0 of 25** |

**Not one RNGD-only and not one mixed candidate clears the tight SLO.** The work
order's risk row — *if a formerly timed-out RNGD candidate now passes, D22 changes*
— does not fire. D22's "every RNGD candidate rejected" holds in the tight regime
too, and now for a measured reason rather than an unevaluated one.

### What actually changed

Not RNGD's standing. **The existence of the tight regime.**

- **P/D disaggregation wins three of the four points.** `pd_split` is the
  recommendation at card ≤ 8000, gpu ≤ 500 and gpu ≤ 8000. This is the first thing
  this repository can say about P/D from an evaluated search.
- **It is not a heterogeneous win.** Both halves are `cuda`. `docs/CLAIMS.md` §2's
  "no heterogeneous configuration is shown to win" stands unchanged, and D16(c)'s
  scoping sentence still applies: the shape most likely to pay — asymmetric TP
  across vendors — is still not enumerated.
- The tight winners are **less** energy-efficient than the loose one (1.48–2.21
  against 2.60 tok/J) and use twice the accelerators (8 against 4). Tightening TTFT
  costs efficiency, which is the shape `docs/rps_aware_planning_design.md` argues
  for and the first time it has been measured at these points.

## §4 — Disposition of the §1.3 table

| result | work order's disposition | outcome |
| --- | --- | --- |
| `pd_slo_sweep_margin18` (D22's basis) | re-run in 3.2 | **re-run; every field identical to the last decimal.** D22 holds |
| `pd_slo_sweep` (superseded 3-regime) | cross-check the A40 row only | the loose point reproduces exactly, so the superseded table's A40 row is consistent |
| tight-TTFT (500/8000) | run in 3.3 | **run; 0 timeouts, all four points FEASIBLE, tight row closed** |
| Exp 2 heterogeneous selection, Exp 5 4-combo | re-run only if §1 found contamination that could change numbers | **no re-run needed.** §1 established the only field that can differ (`num-devices`) does not enter timing arithmetic, and §2 confirmed it empirically — contamination crashes, it does not lie |
| Exp 1 TP sweep, E1, §4.7 78-candidate check | record only | single-hardware and, per §2.1, single-instance candidates were never affected at all |

**One caveat that outlives this work order.** Everything above says a *completed*
run was trustworthy. It says nothing about completeness of any search whose
timed-out candidates were never re-run. The tight regime is the case in point: the
verdicts were wrong not because a number was wrong but because 197 candidates had
no number at all. Any past conclusion drawn from a sweep with timeouts should be
re-read as "the best of what evaluated", not "the best that exists".

