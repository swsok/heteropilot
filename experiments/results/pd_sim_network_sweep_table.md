# Simulator-level P/D network-bandwidth sweep

- cluster: `configs/cluster/single_node_pd_instance.json`  dataset: `workloads/example_trace.jsonl`  num_reqs: 20
- All numbers are simulator predictions (ms). Transfer time is a hand-computed KV_bytes/link_bw delay charged to latency/TPOT, not TTFT (docs/deviations.md D15).

## `--pd-transfer-model bandwidth`

| link_bw (GB/s) | n | TTFT mean (ms) | TPOT mean (ms) | latency mean (ms) |
| --- | --- | --- | --- | --- |
| 16 | 10 | 11.6799 | 11.2453 | 660.9669 |
| 4 | 10 | 11.6769 | 11.2539 | 661.9797 |
| 1 | 10 | 11.6860 | 11.3436 | 665.0958 |

## `--pd-transfer-model none` (control)

| link_bw (GB/s) | n | TTFT mean (ms) | TPOT mean (ms) | latency mean (ms) |
| --- | --- | --- | --- | --- |
| 16 | 10 | 11.8402 | 11.2158 | 660.9796 |
| 1 | 10 | 11.8402 | 11.2158 | 660.9797 |

## Verdict

- latency monotonic increasing as bw drops: PASS
- TPOT monotonic increasing as bw drops: PASS
- TTFT flat (<0.5% spread): PASS
- none-mode latency flat across bw (control): PASS
