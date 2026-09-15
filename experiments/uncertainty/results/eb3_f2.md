# E-B3 on F2 — the closed form overestimates by half, and D40 is not the reason

`WORK_ORDER_uq_stage_b_plus.md` STEP C4. E-B3 re-run on the F2 corpus STEP C1
built, to answer the question PR #86 could not: the closed form was 30.5 % low on
E-A1, but that corpus carried D40's mirrored placements, so the error was the
rule's error *plus* whatever D40 contributed and the two were not separated.

On F2 they are. C1 excluded 282 mirror members at build, the ×1.0 identity
control comes back at exactly zero over all 223 candidates, and the magnitude
below carries no D40 contribution at all.

Raw: `outputs/uncertainty/eb3_f2/eb3_f2_closed_form_vs_resim.json`, merged from
five shards kept beside it.

## The answer

*Corpus 223 candidates (282 D40 mirrors and 23 D71 crashes excluded), incumbent
`pd(cuda-a40-node_a40a-tp2-dp1 P + cuda-a40-node_a40b-tp4-dp1 D)-s256-t8192`,
`slo_penalty` 207,990, grid 5.*

| input | closed | resim | closed/resim − 1 | runs |
| --- | ---: | ---: | ---: | ---: |
| `profile:cuda-a40-node_a40a` | 33,041.1 | **21,615.4** | **+52.86 %** | 422 |
| `profile:cuda-a40-node_a40b` | 33,041.1 | **21,615.4** | **+52.86 %** | 276 |
| `profile:furiosa-rngd-card-node_rngd0` | 0.0 | 0.0 | — | 86 |
| `profile:furiosa-rngd-card-node_rngd1` | 0.0 | 0.0 | — | 12 |
| all six `link_bw:fabric-*` | 0.0 | 0.0 | — | 122 each |
| `sim_error:domain` | 41,118.4 | — | — | **SKIPPED** |

**The closed form overestimates by 52.9 %**, where on E-A1 it underestimated by
30.5 %. Both figures are like-for-like — `refine` recomputes the closed form on
the same two-point grid `{lo, hi}` the resimulation uses, so the only thing that
differs is where the metrics came from.

> **Two closed-form numbers, and they are not in conflict.** `analyze` on the
> five-point grid reports ΔR = 30,381.2 for each A40 profile; the table above
> says 33,041.1. The first is the sweep the measurement plan actually ranks by,
> the second is the same rule restricted to the two endpoints so that it can be
> compared against a resimulation of those endpoints. Quoting 30,381 against
> 21,615 would give +40.6 % and would be comparing two different grids. The
> ranking uses the five-point value; the error bar uses the two-point one.

**The two A40 items agree to the digit and were computed on different machines** —
`a40a` on the A5000 node, `a40b` on a second i9-10900X. The cluster is symmetric
in those two islands, so agreement is expected; getting it across two independent
runs is the cross-node control the sharding needed.

## The ordering survives, and the Spearman is refused

Closed form ranks `sim_error:domain` (41,118) first, the two A40 profiles
(30,381) second, and the remaining eight at exactly zero. After refinement the
profiles fall to 21,615 and **the order does not change**: `sim_error` is still
first, the profiles still second, the eight still zero.

**Spearman is `None`, by rule.** §2.5 requires at least three items with a
non-zero ΔR before a rank correlation means anything, and F2 has two. The payload
carries the reason beside the null so it cannot be read as a correlation that
came out zero:

```
not computed: 2 item(s) with a non-zero dR among the 10 checked, below the 3
§2.5 requires. A correlation over items tied at zero measures the ties.
```

This is the rule PR #86's two 1.000s earned. One of those was over four items
with three tied at zero; the other was over a single refined item, where the two
orderings were identical by construction.

`approximation` flipped `True → False` on **4 items** — every PROFILE item, once
its ΔR came from simulation rather than from the first-order scaling. The six
link items were already exact and did not move.

## The identity controls

**The ×1.0 control, and the D40 answer.** Every PROFILE refinement simulates the
multiplier 1.0 first, which must reproduce the cached prediction exactly:

| control | worst relative deviation | candidates | over 1e-6 |
| --- | ---: | ---: | ---: |
| `profile:cuda-a40-node_a40a` | **0.000e+00** | 223 | 0 |
| `profile:cuda-a40-node_a40b` | **0.000e+00** | 223 | 0 |
| `profile:furiosa-rngd-card-node_rngd0` | **0.000e+00** | 223 | 0 |
| `profile:furiosa-rngd-card-node_rngd1` | **0.000e+00** | 223 | 0 |

Exact, to the last digit, over all four. On E-A1 the two A40 controls both
deviated by **7.869e-02**, at the same candidate — the mirrored placement
`mix(a40a-tp2-dp2+a40b-tp2-dp1)` that D40 names. **F2's 52.9 % is therefore the
closed form's own error and nothing else.**

**The LINK_BW exactness control fires but says little.** `LINK_BW` re-prices a
transfer term the planner adds itself, so its closed form is exact and a
resimulation must land on it. All six agree — and all six agree at `0 = 0`,
because no link carries any regret on this fixture. The control is satisfied and
carries almost no information, which `eb2_f2.md` predicted when it found the same
six links inert under flip detection.

## Coverage — and the failure that nearly passed as a result

1,478 of 1,528 runs came back, across 20 endpoints. **No endpoint produced zero
successful runs**; 13 are partially short, which is D71 (a TP=1 decode island
raises in the simulator) and is already recorded in `f2_truth.md`.

The worst holes are on the one link that touches the winner:

| endpoint | resimulated | touched |
| --- | ---: | ---: |
| `link_bw:fabric-a40a-a40b` at 35.0 GB/s | 41 | 61 |
| `link_bw:fabric-a40a-a40b` at 12.9 GB/s | 58 | 61 |
| `link_bw:fabric-rngd0-a40a` at 35.0 GB/s | 55 | 61 |

More P/D candidates die at the *faster* link, which is consistent with D71's
mechanism: a quicker KV transfer fills the decode engine sooner, and an
over-subscribed decode instance raises instead of running at a worse TPOT.

**This count exists because of D73.** On the first sharded attempt, two helper
nodes were missing `astra-sim/inputs/system/system.json`, every run on them raised
before the simulator started, and the harness reported

```
link_bw:fabric-rngd0-a40b: closed=0 resim=0 in 10s (122 runs)
link exactness link_bw:fabric-rngd0-a40b: ... rel=0.000e+00 -> agrees
```

— a clean pass and an exact-rule agreement produced entirely by failure, because
`_endpoint` leaves the baseline's value in place for a candidate whose run raised
and reports `simulated` as the number of runs *started*. Link items are the worst
case: their true ΔR really is 0, so total failure and the correct answer are the
same number. The only symptom was that 122 simulations had taken ten seconds.
`endpoint_coverage()` now distinguishes the two by object identity, and
`coverage_gate()` stops the run when any endpoint simulated nothing. The link
rows above were re-run and took **45 minutes each**.

## How it was run

The resimulation was split across machines — one uncertain input is one
independent `resimulate` call, so the ten resimulable items divide cleanly and
`--merge` puts them back together, refusing shards that disagree about the
corpus, the penalty, the incumbent or any item's closed-form ΔR.

| shard | items | CPU | host |
| --- | ---: | ---: | --- |
| `shard_a5000` | 2 | 7,456 s | A5000 node |
| `shard_a5000b` | 2 | 6,179 s | A5000 node |
| `shard_node49` | 4 | 6,859 s | i9-10900X helper |
| `shard_node49b` | 2 | 5,468 s | i9-10900X helper |
| `shard_simerror` | 1 | 0 s | A5000 node (closed form only) |

**25,962 s of CPU work inside a 7,456 s slowest shard — 3.5× on wall clock.**
Against the closed form's 0.71 s for all eleven items, the resimulation costs
**36,566×**.

> **Provenance, and what it does and does not claim.** Two of the machines are
> not among the three nodes `scripts/whichnode.sh` classifies; it reports them as
> `cuda-other` and `unknown`, which is correct. Nothing here is a hardware
> measurement — every run is LLMServingSim against the committed F2 cache and the
> committed RNGD-CARD and A40 domains, which were measured on the nodes that own
> them. That the closed form came back at 41118.38218974412 and 30381.218319787986
> on every machine, to the last digit, is the evidence that the split changed
> nothing.
>
> A third helper (an i5-13500 under WSL2) was tried and abandoned. Its cores are
> ~2.5× this node's on a single-thread benchmark, and its throughput on this
> workload was about one seventh: 16 workers each ran at 49 % CPU against 65–73 %
> on the i9 nodes, with 27 % system time and ~553,000 context switches per second.
> The workload spawns a simulator process per candidate, and that is what WSL2
> charges for. Recorded because the per-core number would otherwise suggest the
> opposite conclusion.

## What this licenses, and what it does not

**Does:** using the closed form to decide *what to measure next*. It ranks
`sim_error:domain` first and the two A40 profiles second, and the resimulation
leaves that order intact while confirming the other eight inputs are worth
nothing to measure.

**Does not: quoting ΔR as an amount of energy.** On F2 it is **52.9 % high**; on
E-A1 it was 30.5 % low. The magnitude is not a saving, and it is not even
reliably signed.

**Does not: reading the two fixtures as a trend.** They differ in more than D40 —
F2 enables P/D and tightens TTFT to 8,000 ms. What is attributable is narrower and
firmer: **F2's +52.9 % contains no D40 contribution**, where E-A1's −30.5 % did,
of a size that run could not report.

**`sim_error` is SKIPPED, not checked**, and it is the largest ΔR on the fixture
(41,118 against the profiles' 30,381). A SIM_ERROR moves the margin rather than
the prediction, so no simulator input changes and there is nothing to disagree
with. **D72 adds a second reason to leave it alone here**: the sweep spends its
range as a flat margin while the harness restores it as an accuracy-domain policy
swap, so even a notional resimulation would not be pricing the same input. The
unchecked fraction of this ranking is the item at the top of it.

## Reproduce

```bash
# one shard; --degrade takes any subset of the pool, and shards are independent
PYTHONPATH=$PWD .venv/bin/python \
    experiments/uncertainty/eb3_closed_form_vs_resim.py \
    --fixture experiments/uncertainty/fixtures/f2.json \
    --cache-dir outputs/uncertainty/f2/cache \
    --work-dir outputs/uncertainty/eb3_f2/work \
    --kinds all --top 10 --workers 16 \
    --out-json outputs/uncertainty/eb3_f2/shard_<name>.json

# combine; runs nothing, and refuses shards that are not the same experiment
PYTHONPATH=$PWD .venv/bin/python \
    experiments/uncertainty/eb3_closed_form_vs_resim.py \
    --merge outputs/uncertainty/eb3_f2/shard_*.json \
    --out-json outputs/uncertainty/eb3_f2/eb3_f2_closed_form_vs_resim.json
```

A single-machine run is `--kinds all --top 10` with no split; it takes about
7 hours of CPU on a 20-core node.
