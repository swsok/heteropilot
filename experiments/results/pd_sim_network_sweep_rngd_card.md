# Does the corrected fabric bandwidth change anything? The simulator-side test

*Run 2026-08-28 on the A40 server. Driver:
`experiments/scripts/pd_sim_network_sweep.py`. Companion to
`experiments/results/gpu_host_bandwidth.md`, which measured the fabric, and to
`rngd_parallel_bandwidth.md` (deviations D18), which retracted the NPU leg.*

## Why this run exists

Measuring both legs of the cross-vendor KV path moved the six `fabric-*` links in
the two P/D fixtures from a `placeholder` 35 GB/s to measured, composed values of
12.6–13.0 GB/s. Re-running `pd_slo_sweep.py` at those values changed **no
winner at any of 16 SLO points**, and the reason turned out to be that the sweep
cannot see the fabric at all: the simulator prices the P/D handoff at zero unless
`--pd-transfer-model bandwidth` is passed, and the planner-side add-on charges a
per-request latency term that is milliseconds against multi-second TTFTs.

`pd_sim_network_sweep.py` is the driver that *does* pass the flag. It also charges
the delay in a different bucket — **latency, ITL[0] and TPOT, not TTFT** (D15) —
so it tests the one thing the SLO sweep could not.

The config swept is the compiled cluster of the **exact candidate that matters**:
`pd(cuda-a40-node_a40a-tp1-dp1 P + furiosa-rngd-card-node_rngd0-tp1-dp1 D)-s128-t2048`,
the `A40 prefill → RNGD-card decode` row that `pd_slo_sweep.md` reports at 88 % of
the winner's energy efficiency. Both islands are tp1, which is the condition the
driver's `none`-mode control assumes. Workload is the SLO sweep's own 300-request
trace (mean 875 input / 636 output tokens), not the driver's 20-request default,
so the numbers are comparable to that sweep.

## The answer: no, not over the range that matters

| link_bw | TPOT mean (ms) | latency mean (ms) |
| ---: | ---: | ---: |
| 400 | 44.0723 | 30739.62 |
| **35** — the old placeholder | 44.0780 | 30743.41 |
| **13** — the measured value | 44.0835 | 30746.69 |
| 5 | 44.1074 | 30761.70 |

**35 → 13 GB/s moves TPOT by +0.012 % and mean latency by +0.011 %.**

The magnitude is arithmetically sound, which is worth checking rather than
assuming. The trace averages 875 input tokens and Llama-3.1-8B holds 128 KiB of KV
per token, so one request ships **114.7 MB**: 3.28 ms at 35 GB/s against 8.82 ms at
13 GB/s, a 5.5 ms difference. The observed mean-latency change is **+3.28 ms** —
the same order, the remainder absorbed because not every completed request takes
the transfer path.

**The cause is structural, not a coincidence of this bandwidth pair.** This
deployment runs at a 30.7-second mean latency and a 44 ms TPOT; it is heavily
queue-bound. A 3–9 ms per-request KV transfer disappears into that queue. For the
fabric to bind, the deployment would have to be far closer to bandwidth-limited
than any configuration in these fixtures is.

This also settles a doubt raised when the SLO-sweep result came in. That analysis
compared the planner-side penalty against p99 **TTFT**, and the simulator-side
model does not touch TTFT — so the comparison could have been against the wrong
bucket. Measured in the right bucket, the conclusion is unchanged.

## Exp 3's ~10 GB/s crossing is fixture-specific

There is nothing distinctive at 10 GB/s here: 44.0898 ms TPOT, continuous with its
neighbours. The crossing that `pd_network_sweep.py` reports (reconfirmed
byte-identically on 2026-08-28) is a property of that experiment's fixture,
workload and *planner-side* TTFT term. It should not be quoted as a general
threshold for this hardware pair.

## Three verdict FAILs, and why they are not bugs

The driver asserts pass criteria designed for a single-node tp1 config. Three fail
here, all traceable to one cause.

At **1 GB/s everything blows up** — TTFT 2886 → 32892 ms, latency 30740 → 52904,
and TPOT *falls* to 31.5 as the batch composition changes. That breaks
"TPOT monotonic", "TTFT flat" and "none-mode flat" at once.

**It is not the P/D transfer model.** The `none`-mode control reproduces the
1 GB/s TTFT as **32892.1924 ms — identical to the last digit**. Whatever `link_bw`
is doing at that value, it is doing it with the transfer model switched off.

The driver's docstring names only one condition under which the control may move:
tp > 1, where `link_bw` also slows the intra-instance TP AllReduce. **That does not
apply — both islands here are tp1.** And the committed single-node run
(`pd_sim_network_sweep_table.md`) has a flat control at 1 GB/s
(660.9796 → 660.9797 ms). The difference is that this config spans **two nodes**,
so `link_bw` prices inter-node traffic beyond the P/D handoff.

**Consequence:** `link_bw` is an overloaded scalar in multi-node configs, and the
driver's pass criteria hold only for single-node ones. This does not invalidate
D15 — the committed single-node validation still passes — but the criteria should
not be read as a D15 regression signal on a multi-node cluster. The cliff sits
between 5 and 1 GB/s; the measured operating point of 12.6–13.0 GB/s is well clear
of it, and every row from 400 down to 5 GB/s behaves as D15 describes.

---

## Generated output

- cluster: `outputs/.hp-slo-pd-rngd-gpu-card/work/pd_cuda-a40-node_a40a-tp1-dp1_P___furiosa-rngd-card-node_rngd0-tp1-dp1_D_-s128-t2048/cluster.json`  dataset: `outputs/.hp-slo-pd-rngd-gpu-card/work/sweep_trace.jsonl`  num_reqs: 300
- All numbers are simulator predictions (ms). Transfer time is a hand-computed KV_bytes/link_bw delay charged to latency/TPOT, not TTFT (docs/deviations.md D15).

## `--pd-transfer-model bandwidth`

| link_bw (GB/s) | n | TTFT mean (ms) | TPOT mean (ms) | latency mean (ms) |
| --- | --- | --- | --- | --- |
| 400 | 300 | 2886.1925 | 44.0723 | 30739.6231 |
| 200 | 300 | 2886.1937 | 44.0728 | 30740.0182 |
| 100 | 300 | 2886.1294 | 44.0738 | 30740.5962 |
| 50 | 300 | 2886.3360 | 44.0763 | 30742.1699 |
| 35 | 300 | 2886.3400 | 44.0780 | 30743.4141 |
| 25 | 300 | 2886.1344 | 44.0795 | 30744.1673 |
| 13 | 300 | 2886.1597 | 44.0835 | 30746.6900 |
| 10 | 300 | 2886.2430 | 44.0898 | 30750.8848 |
| 5 | 300 | 2886.2007 | 44.1074 | 30761.6965 |
| 1 | 300 | 32892.1924 | 31.5121 | 52903.6498 |

## `--pd-transfer-model none` (control)

| link_bw (GB/s) | n | TTFT mean (ms) | TPOT mean (ms) | latency mean (ms) |
| --- | --- | --- | --- | --- |
| 400 | 300 | 2886.1389 | 44.0610 | 30735.8076 |
| 1 | 300 | 32892.1924 | 31.3258 | 52786.8744 |

## Verdict

- latency monotonic increasing as bw drops: PASS
- TPOT monotonic increasing as bw drops: FAIL
- TTFT flat (<0.5% spread): FAIL
- none-mode latency flat across bw (control): FAIL
