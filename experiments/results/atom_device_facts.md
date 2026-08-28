# Rebellions ATOM device facts — memory and power, measured

*Measured 2026-08-28 on the NPU server, rbln0 (`RBLN-CA22`), by
`experiments/scripts/atom_device_facts.py`. Raw:
`outputs/atom_profile/device_facts.json` (memory),
`outputs/atom_profile/device_power.json` (power + card sweep),
`outputs/atom_profile/power_decay.csv`. First step of the ATOM branch of
`docs/HANDOVER_NPU.md` §3.*

## Results

| field | value | note |
| --- | ---: | --- |
| largest single allocation | **15.047 GiB** | driver confirms 15.088 GiB resident |
| card DRAM total | 15.719 GiB | one block can claim **95.7 %** of the card |
| idle power | **19.44 W** | 19.28–19.53 over 23 samples, after a 45 s settle |
| active power | **68.73 W** | 67.38–70.78 over 31 samples, at **95.1 %** utilisation |
| standby power (D7, 2 s) | 68.03 W | equals active; the card does not shed within 2 s |
| observed return to idle | ~5.9 s | via a ~38 W plateau; see the decay section |

One accelerator here is one **card**, unlike RNGD where it is one PE. The card is
what `rebel.create_runtime(device=N)` binds to and `device_count()` returns 4, so
a card's idle draw belongs in the accelerator profile rather than in
ClusterSpecV2's `base_node_power` — the opposite of the RNGD split.

## A power reading is only as good as the utilisation it was taken at

**The first pass measured 52.5 W and it was wrong by ~30 %.** Power on this card
is close to linear in device utilisation, and the naive load left it at 36 %:

| load | util % (mean) | card power (W) |
| --- | ---: | ---: |
| 1 Linear, synchronous | 36.2 | 44.3 |
| 8 layers, synchronous | 64.4 | 54.9 |
| 8 layers, async | 75.6 | 57.7 |
| 8 layers n=8192, sync | 83.5 | 62.1 |
| 8 layers n=12288, async | **95.1** | **68.7** |

Every `rt.run()` is a synchronous host round trip, so a shallow model leaves the
card idle between calls. Depth amortises the round trip over eight matmuls, and
async with several calls in flight buys ~10 more utilisation points.

This is the same class of error that made a single-PE RNGD reading understate its
card by 4×. The fix is not just a better load: **the script now records
`active_util_pct` beside the power in the artifact**, so the reading can be
judged instead of trusted. Do not quote an ATOM power figure that does not carry
its utilisation.

A second contamination, worth the same care: the first `idle_w` sampled
19.0–36.7 W in a window meant to be idle, because the 15 GiB memory bisect had
just released and the card was still shedding. There is now a 45 s settle before
the idle baseline, after which idle is dead steady at 19.28–19.53 W.

## Per-card power is additive — tested, not assumed

Loading 1..4 cards and reading *every* card at each point:

| loaded | rbln0 | rbln1 | rbln2 | rbln3 | total |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 23.52 | 18.73 | 19.35 | 18.06 | 79.66 |
| 1 | **65.98** | 18.71 | 19.23 | 18.07 | 121.99 |
| 2 | **67.56** | **64.85** | 19.27 | 18.11 | 169.79 |
| 3 | **66.12** | **64.73** | **66.45** | 18.11 | 215.42 |
| 4 | **65.46** | **64.45** | **66.14** | **63.35** | 259.40 |

(rbln0's 23.52 W at 0 loaded is residual shedding from the power run just before.)

An unloaded card never moves more than ~0.1 W while a neighbour is saturated, and
a loaded card holds ~65 W regardless of how many others are busy. So a multi-card
ClusterSpecV2 may simply **sum** this profile's power: there is no shared board
cost to split as there is on RNGD, and no saturation knee like RNGD's 8th PE.

An earlier sweep appeared to show per-card power *falling* from 52.6 to 38.9 W as
cards were added, which would have looked like a shared power ceiling. It was
host contention from the non-saturating load. Re-run with the saturating one, the
effect vanishes — a good example of why the utilisation column exists.

## The D7 standby window does not capture this card's decay

`standby_power` is reported over the D7 2 s post-load window, the same definition
the A40 and RNGD entries use, so the three stay comparable. On this card that
comes out **equal to active power**, because the decay is slower than the window
(`power_decay.csv`):

| t (s) | power (W) | util |
| ---: | ---: | ---: |
| 0.1 | 67.81 | 92.2 |
| 1.3 | 65.66 | 0.0 |
| 2.4 | 37.85 | 0.0 |
| 4.7 | 30.87 | 0.0 |
| 5.9 | **20.06** | 0.0 |

Work stops at ~1.3 s but the card holds ~66 W through the whole 2 s window, steps
down through a ~38 W plateau, and reaches idle only at ~5.9 s. So the honest tail
is roughly **6 s at a mean near 43 W**, not 2 s at 68 W. The D7 figures are kept
for cross-profile comparability; anything modelling burst tails should read the
CSV instead.

## What this does and does not unlock

**Measured and written to `profiles/accelerators/rbln_atom.yaml`:** `memory_gb`
and the whole `power:` block.

**Still not measured:** `memory_bandwidth_gbps`. The stub's 256 is retained as a
placeholder and must not be cited. The RNGD figure was derived from that
profiling run's own DMA spans; ATOM has no equivalent bundle, and the saturating
load used here is compute-bound (~1.07 GB of weights per forward at ~2.8
forwards/s, i.e. ~3 GB/s — nowhere near HBM speed), so it bounds nothing.

**The profile is still excluded from candidate generation.** `sim_hardware` stays
`null` and `supported_models` stays empty, and the top-level `source` stays
`placeholder` — the `Source` enum admits only measured / vendor_spec /
placeholder, and a profile carrying an unmeasured bandwidth must not present
itself as measured. Device facts do not make a profile usable; only a
`profiler/perf/ATOM/` bundle will.

What *is* verified by execution is only bring-up: a compiled Linear runs on all
four cards, output matching CPU to 3.75e-03. That is not model support.

`max_tp_size: 4` remains **unverified** — the driver reports the four cards in one
group with a uniform 4×4 topology matrix after the 2026-08-28 reboot, but no
multi-card TP path has been exercised.

## Reproduce

```bash
.venv-rbln/bin/python experiments/scripts/atom_device_facts.py \
    --device 0 --card-sweep --load-s 25 --idle-s 15 \
    --out outputs/atom_profile/device_facts.json
```

`.venv-rbln`, not `.venv` — the planner venv has no torch, and the vendor stack
needs the consistent 0.11.0 trio that venv carries
(`docs/hardware_roadmap.md`, 2026-08-28 update). The memory bisect costs ~12 s of
compile per device-GiB, so a full run is 15–25 minutes.
