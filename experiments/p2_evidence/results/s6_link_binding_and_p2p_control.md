# S6 — the bound half of the link key, and an NCCL-free control

*`WORK_ORDER_domain_scoping.md` STEP S6. A40 node, accelerator set
`83e3434d7696` (8 × A40), 2026-09-21. Every figure here is a measurement of that
card set and of no other machine.*

## 0. What S6 turned out to be

The work order lists three parts and the handover carried all three as open. Two
of them were already measured, by PR #104 and by V3's own appendix, and reading
that first is what kept this to one afternoon:

| part | asked for | state |
| --- | --- | --- |
| **(i)** NUMA-bound P1 (TP=4) open-loop, 3 repeats | `v3_verdict_accuracy.md` **A.3** — GPUs 4–7, `numactl --cpunodebind=1 --membind=1`, three repeats, 64.616 ms p99 TPOT at served `L` 162.26 | **already measured** |
| **(iii)** the same candidate at TP=2 inside an NV4 pair | `v3_verdict_accuracy.md` **A.2** — three repeats, bound, 67.337 ms measured against 63.831 simulated (−5.21 %) | **already measured** |
| **(ii)** `nccl-tests` + `p2pBandwidthLatencyTest`, bound and unbound, filed in S3's schema | nothing filed was bound; everything in `Link.measurements[]` was `binding: unpinned` | **this document** |

So S6 is (ii), and (ii) had two distinct gaps, only one of which was about the
tool.

## 1. `nccl-tests` was not used, and would not have been a control

Three lines, because this is the part a reader will want to challenge.

1. **The claim in A.1 is true and was re-verified here.** The only build on this
   host is `/home/ieg95/workspace/bandwidth/nccl-tests/build/all_reduce_perf`,
   and it does not run: `version 'GLIBC_2.34' not found` against this host's
   `ldd (Ubuntu GLIBC 2.31-0ubuntu9.18) 2.31`, plus `libnccl.so.2 => not found`.
   There is no `nvcc` and no system NCCL header either, so it cannot be rebuilt
   without installing a CUDA toolchain.
2. **It was not attempted, deliberately, because it is the same NCCL.**
   `nccl-tests` links the same library `link_probe.py` reaches through torch —
   NCCL 2.27.5 in both cases. Two front ends onto one implementation agree by
   construction; that is corroboration of the harness, not of the number. The
   work order asked for a vendor tool as a second opinion, and this would not
   have been one.
3. **What replaced it is a control that genuinely does not use NCCL**: a direct
   `cudaMemcpyPeer` (§3). That bounds the wire independently of the collective,
   which is the question `nccl-tests` was being asked to answer.

Recorded as **D116**.

## 2. Bound against unbound — binding does not move this wire

`experiments/p2_evidence/run_link_probe_numa.sh`, which is
`run_link_probe.sh`/`run_link_probe47.sh` with a `numactl` prefix and nothing
else changed. The probe itself is untouched on purpose: `binding` is one axis of
the measurement key, so the bound and unbound runs must differ in binding and in
nothing else. Affinity was verified with `taskset` rather than trusted
(`binding_proof.txt` beside each group's raw files; NUMA 0 → CPUs `0-15,32-47`,
NUMA 1 → `16-31,48-63`, matching `docs/nodes/a40.md`).

64 MiB `busbw`, GB/s:

| case | GPUs 0–3 unpinned | numa0 pinned | Δ | GPUs 4–7 unpinned | numa1 pinned | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tp4` (4-rank all-reduce) | 8.81 | 8.81 | +0.10 % | 8.78 | 8.73 | −0.52 % |
| `tp4` `NCCL_P2P_DISABLE=1` | 6.49 | 6.53 | +0.60 % | 6.47 | 6.50 | +0.48 % |
| `pair` inside the NV4 pair | 39.11 | 39.28 | +0.44 % | 39.10 | 39.24 | +0.35 % |
| `pair` across the bridge | 19.29 | 19.30 | +0.09 % | 19.33 | 19.38 | +0.28 % |
| the same, `NCCL_P2P_DISABLE=1` | 6.81 | 6.80 | −0.21 % | 6.78 | 6.82 | +0.66 % |

**The largest move in ten cases is 0.66 %, and the direction is not consistent.**

That is the result, and it is worth stating plainly because the same axis is
worth **1.93×** of deployment throughput on this node (`v3_verdict_accuracy.md`
A.3: 1 018.5 → 1 970.7 output tok/s on GPUs 4–7). Both are true and they are
about different things. **NUMA binding buys the host memory path — prefill and
queueing — and not the GPU-to-GPU wire.** A.3 already said TPOT barely moves
while TTFT moves 0.37×; this is the link-level confirmation of the same split,
and it means a deployment measured unbound is not thereby a link measured
wrongly.

## 3. The NCCL-free control: the ring uses a third of the wire

`experiments/p2_evidence/p2p_probe.py` — direct peer copy, no NCCL on any path,
median of 20 with the spread reported, at 64 / 256 / 512 MiB. Both one-way and
round trip, because a ring all-reduce drives the link in both directions and a
one-way figure cannot be compared with it unexamined.

| pair | path | one-way 512 MiB | round trip 512 MiB | spread |
| --- | --- | ---: | ---: | ---: |
| 0↔1, 4↔5 | inside the NV4 pair | 52.72 / 52.72 | 52.85 / 52.84 | < 0.6 % |
| 0↔2, 4↔6 | across the PCIe bridge | **25.15 / 25.15** | **25.16 / 25.15** | < 0.5 % |

Round trip agrees with one way to **0.04 %**, so the link is symmetric per
direction and the round-trip figure adds no capacity — the two legs serialise.
The bound and unbound runs agree to 0.1 %, consistent with §2.

**Put beside the collective figures on the same wire, this is the point:**

| what crosses `pcie-a40a-02` | GB/s | fraction of the wire |
| --- | ---: | ---: |
| direct peer copy (no NCCL) | **25.15** | 1.00 |
| 2-rank all-reduce | 19.3 | 0.77 |
| 4-rank all-reduce | **8.8** | **0.35** |
| datasheet `vendor_spec` | 64.0 | 2.54 × the measured wire |

D112 said one wire carries three numbers and keyed the measurement by the
traffic. This measures the wire itself with NCCL out of the path, and the three
numbers survive it: **8.8 is not a slow link, it is a four-rank ring using a
third of what the link can carry.** The datasheet 64.0 is 2.5× the wire under
any traffic at all, which is the other half of why V3's P1 was off by 43 %.

`nvidia-smi topo -m` is filed at `outputs/p2_evidence/p2p_control/nvidia_smi_topo.txt`
and matches `docs/nodes/a40.md` exactly: `NV4` for (0,1) (2,3) (4,5) (6,7),
`NODE` between pairs within a NUMA node, `SYS` across. Only GPU6 has `PHB` to
the NIC.

## 4. What was filed, and what was deliberately not

Filed into `experiments/configs/clusters/pd-rngd-gpu-card-measured-pcie.yaml`,
**appended after** the existing unpinned entries so that today's selection is
bit-for-bit unchanged (`measurement_for` prefers the first stated-binding match
when the node states `unknown`):

| link | key | GB/s |
| --- | --- | ---: |
| `pcie-a40a-02` | `(all_reduce, 4, bulk, numa_pinned)` | 8.81 |
| | `(all_reduce, 2, bulk, numa_pinned)` | 19.30 |
| | `(p2p, 2, bulk, numa_pinned)` | 25.15 |
| `pcie-a40b-02` | `(all_reduce, 4, bulk, numa_pinned)` | 8.73 |
| | `(all_reduce, 2, bulk, numa_pinned)` | 19.38 |
| | `(p2p, 2, bulk, numa_pinned)` | 25.15 |

The datasheet `bandwidth_gbps: 64.0` is untouched on both links (rule A3), and
both nodes still say `device_binding: unpinned`, so the planner still selects
the unpinned figures — verified for all three node bindings and by
`tests/test_link_effective_bw.py` (31 passed).

**The NVLink links were measured and NOT filed.** `pair_nvlink` gives 39.2–39.3
bound against a datasheet 112.5, and the peer copy 52.8. Filing those would give
`nvlink-a40*` measurements it has never had, and `island_interconnect` reduces a
`min` over island links — so a tp=2 candidate inside an NV4 pair would silently
move from 112.5 to 39.3 and every plan touching one would change. That is a
planner behaviour change, not a measurement, and it is outside what S6 was
scoped to do. The figures are here, in §3 and in
`outputs/p2_evidence/link_numa/`, for whoever takes that decision.

## 5. The residual on P1 is stated, not margined

S6(i)'s measurement exists (§0) and `v3_addendum_s4.md` §3 pairs it with the
simulator: priced at S3's measured 8.8 instead of the datasheet 64.0, the
simulator lands **−7.01 %** on p99 TPOT and **−0.73 %** on served concurrency.
**That −7.01 % is not margined by anything, and cannot be by reaching for the
domain that exists.** The committed `a40.accuracy.yaml` is fitted at tp=1; P1 is
tp=4; the mismatch is refused, which is rule (e) working.

Pinned by two tests against the committed file at V3's real operating point
(`tests/test_calibration_condition.py`, §(vii)):

* under the `refuse` default, `condition_mismatch` with `mismatch_fields ==
  ["tp"]` and `required_measurement.tp == 4`, and **no margin leaks through** —
  `tpot_percent` and `ttft_percent` are both 0.0. The operating point 162.26 is
  *inside* the domain's load axis [4.043, 170.56], so this is provably the
  configuration refusal and not the load-axis one.
* under `warn`, the tp=1 domain is applied across the mismatch, says so
  (`APPLIED ACROSS` in `basis`), and charges **1.38 %** against a residual of
  7.01 % — **about 5× too little, in the unsafe direction.**

So one residual at one operating point is not a margin: it has no slope to widen
along. Closing it needs an A40 domain *fitted* at tp=4, which is a measurement
campaign rather than a reading. The skeleton for that work is recorded in
`docs/HANDOVER.md` §2.12 rather than started here, because S6 is a half-day
optional step and that is not.

## 6. Reproduce

```bash
bash experiments/p2_evidence/run_link_probe_numa.sh        # §2, ~4 min, 8 idle GPUs
.venv-vllm/bin/python experiments/p2_evidence/p2p_probe.py \
    --pairs 0-1,0-2,4-5,4-6 --label unpinned \
    --out outputs/p2_evidence/p2p_control/unpinned.json    # §3
```

Raw: `outputs/p2_evidence/link_numa/{numa0,numa1}/`,
`outputs/p2_evidence/p2p_control/`. Both directories carry the binding proof or
the topology dump beside the numbers.
