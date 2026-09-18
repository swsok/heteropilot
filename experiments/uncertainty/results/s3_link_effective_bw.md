# S3 — a link does not have one bandwidth, and what keying the item by traffic changes

**What this is.** `WORK_ORDER_domain_scoping.md` STEP S3; `docs/deviations.md`
D112. S2 ended by naming what was still missing: the registry priced a link as a
prefill→decode KV transfer, while the error V3 measured came from the same link
carrying a TP all-reduce inside one island. This is what happened when the item's
meaning was changed to *the effective collective bandwidth of the deployment*.

The measured table is `outputs/uncertainty/s3/s3_bucket_move_table.md`, the raw
record `outputs/uncertainty/s3/s3_bucket_move.json`. Both arms are one run of
the **same** E-A1 corpus S2 used (300 requests, seed 42, `pd-rngd-gpu-card.yaml`,
`examples/service_specs/llama31-8b.yaml`, per-point accuracy-domain margin,
`--condition-mismatch warn`, warm cache), so nothing re-simulates and the
recommendation is `cuda-a40-node_a40a-tp4-dp1-s128-t8192` throughout — V3's P1.

## The answer

| | pre-S2 | S2 (committed) | **S3** |
| --- | ---: | ---: | ---: |
| ranked | 2 | 2 | 2 |
| inert (swept, moves nothing) | 0 | 4 | **2** |
| undecidable | 4 | 0 | 0 |
| needs simulation to price | — | — | **2** |
| where `pcie-a40a-02` lands | undecidable | inert | **needs_resimulation** |

S3 splits S2's four inert inputs into the two that really are inert and the two
that were never priced at all. The two that move are both `link_bw`; the two
that stay are both `link_lat`, correctly — the planner applies a link latency to
every kind of traffic alike through `path_latency_ns`, and on a corpus with no
P/D candidate moving it changes nothing.

**Three things had to move before `--resimulate-top` actually reached the new
bucket**, and each would have made it a dead end: `refine` skipped every item
whose closed-form regret was None (a rule meant for an *unbounded* range, which
has nothing to simulate at); `Refinement.closed_form` recorded a **0.0** where
no closed form existed, which reads as "the closed form says this does not
matter" — the claim S3 removed; and E-B3's link identity control asserted a
LINK_BW item's closed form is exact and any disagreement is a bug in the rule,
which for an `all_reduce` item would have reported a bug in every one of them.
All three are fixed and pinned by tests.

**Nothing became ranked, and that is the honest outcome rather than a
disappointment.** S2's hypothesis was that the V3 link would rank first once it
had a range; S2 measured `inert` instead. S3 does not make it rank either — it
changes the claim from "measuring this would change no plan" to "the sweep cannot
say what measuring this would change", which is the difference between a wrong
answer and a missing one. `--resimulate-top` is what turns it into a rank.

## Why `inert` was wrong, and by how much

A link bandwidth reaches a predicted metric by **two** paths, and the closed form
knew one:

| path | where | closed form |
| --- | --- | --- |
| prefill→decode KV handoff | `exhaustive.apply_pd_transfer_cost`, added by the planner | exact (`perturb._reprice_transfer`) |
| **every TP collective** | `island_interconnect` → the simulator's own `link_bw` → ASTRA-Sim | **none exists** |

The E-A1 corpus was generated with `enable_pd=False`, so no candidate has a
handoff, so the only rule that existed moved nothing and every `link_bw` item
swept its whole range with `affected=0`. The plan reported them as inert.

What that verdict was worth is already measured, by V3 itself
(`experiments/p2_evidence/results/v3_verdict_accuracy.md` A.4,
`v5_resim_measured_link.json`). Re-simulating P1 with the same link at the
measured 8.8 GB/s instead of the `vendor_spec` 64.0:

| | measured | sim @ 64.0 | sim @ 8.8 |
| --- | ---: | ---: | ---: |
| p99 TPOT, ms | 64.62 | 36.55 (**−43.44 %**) | 60.09 (**−7.01 %**) |
| served concurrency | 162.26 | 127.87 (−21.19 %) | 161.07 (−0.73 %) |

So the input the measurement plan called not worth 0.114 h was worth 23.5 ms of
a 28.1 ms error. A closed form that returns "unchanged" for the path an error
actually travels does not produce a small error in the ranking; it produces a
confident zero.

## What a link's bandwidth actually is, on the one link we measured

One run over the A40 node's PCIe bridge, 8 KiB to 64 MiB, both GPU groups
(`outputs/p2_evidence/link/`, `link47/`, PR #100):

| traffic | measured | ratio to the `vendor_spec` 64.0 |
| --- | ---: | ---: |
| direct device-to-device copy | 25.0 GB/s | 0.39× |
| two-rank all-reduce | 19.29 GB/s | 0.30× |
| four-rank all-reduce (what a tp=4 island runs) | 8.8 GB/s | **0.1375×** |

Three figures, one wire, a 2.8× spread between them. The datasheet number is not
optimistic about a quantity — there is no single quantity for it to be
optimistic about. Which one is right depends on the collective, the group size,
the message size and the binding, so those four are the key.

**`world_size` is in the key and the work order did not ask for it.** §S3 names
`(link_id, collective, msg_size_class, device_binding)`. 8.8 and 19.29 differ by
nothing else, so under the four-part key they collide — the fixture carrying
both would not load, which is how this was found rather than reasoned. It is
also the field V3 asked for in as many words: *"a static per-link field cannot
hold a quantity that depends on how many devices a candidate spans and which
ones."* Recorded as a deviation in D112.

**The message-size bands come from the curve, not from round numbers.** The
four-rank all-reduce busbw climbs from 0.23 GB/s at 8 KiB and is flat only from
4 MiB up (4–64 MiB within 4 %), so `bulk` starts at 4 MiB and is the only band
that answers the simulator's asymptotic `link_bw` term. The two-rank curve is
**not** flat there — 16.05 GB/s at 4 MiB to 19.29 at 64, a 20.2 % spread — so
its filed figure is the 64 MiB endpoint and its `note` says so. It is filed
anyway, because the alternative is not "no answer" but the 64.0 datasheet
number, four times the worst point on that curve.

## What the simulator now receives

`experiments/configs/clusters/pd-rngd-gpu-card-measured-pcie.yaml` was V3's
hand-made copy with 8.8 written into `bandwidth_gbps` and `source: measured`. It
is rewritten onto the schema: the spec value and its `source` are restored and
the measurements sit beside them. Verified, and pinned by
`tests/test_link_effective_bw.py`:

| what is asked for | before (hand-edited) | after (S3) |
| --- | ---: | ---: |
| tp=4 island, intra `link_bw` | 8.8 | **8.8** |
| Level-2 intra / cross | 8.8 / 13.0 | **8.8 / 13.0** |
| tp=2 on the same island | 8.8 | **19.29** |
| tp=3 (never measured) | 8.8 | 64.0 + a recorded note |

The tp=4 column is the point: the file's purpose is to reproduce V3's candidate
and it still does, byte for byte in the compiled input. The tp=2 column is the
improvement V3 asked for — it had to report *"for TP=2 the substitution
introduces an error and changes its sign, +18.21 %"*, because the hand-edit gave
a two-rank group a four-rank figure. The tp=3 column is the refusal: no
measurement answers, so the spec value is used and the reduction's
`assumptions` say which measurements the link does carry.

**Two things this does not fix.** The simulator still has no representation of
*which* devices inside an island a TP group occupies, so it routes tp=2 over the
cross-pair link although the hardware's tp=2 stays inside the NVLink pair and
never touches it: S3 fixes the group-**size** axis, not the device-**identity**
axis, and V3's item #2 survives in that narrower form. And `nccl-tests` still
could not be used (its only build on that host needs GLIBC 2.34 against Ubuntu
20.04's 2.31), so every figure here is attributed to this repo's torch probe in
its `method` field and **S6(ii) remains open** for a vendor-tool number.

## Two consequences worth knowing before re-running anything

**A `vendor_spec` link with a measured collective now reports an uncertain
latency.** Restoring `source: vendor_spec` on those two links means their
bandwidth leaves the registry (their all-reduce is measured) while their
*latency* correctly enters it — no link latency in this repository is measured at
all. The hand-edited copy said `source: measured` and had been crediting those
latencies as measurements, which the registry's own docstring had flagged as an
overstatement it could not correct without a schema change. This is that change.

**On `pd-rngd-gpu.yaml` all 16 uncertain `link_bw` items are intra-island**, so
until `--resimulate-top` runs, `--measurement-plan` ranks no link on that
fixture at all. Read that as "cannot say without simulating", not as a
regression. Quantifying what it does to E-A1's counts belongs to STEP S4.

**The generator script gained a bucket.** `s2_default_ranges.py` classified into
four and would have silently dropped these two items, printing "inert 4 → 2"
with nothing absorbing the difference. It now reports `needs_resimulation`, and
finds the V3 row by link id rather than by the bare string, which no longer
matches any item. S2's own committed result files are untouched.
