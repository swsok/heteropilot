# A40 GPU power measurement (roadmap step 3)

Measured 2026-08-18 on the A40x8 migration server, **GPU 2** (the profiling job
was occupying GPU 0/1 at the time; per-GPU `power.draw` on GPU 2 is isolated from
that). Method mirrors the A5000 protocol (`profiles/accelerators/a5000.yaml`,
`experiments/results/a5000_power/`): `nvidia-smi --query-gpu=power.draw` polled at
10 Hz while a real vLLM bench drives the card.

## Raw data (this directory)
- `idle_gpu2.csv` — 320 samples, GPU idle (no CUDA context), 32 s @ 10 Hz.
- `load_gpu2.csv` — continuous 10 Hz log spanning warm-up + load + post-load decay.
- `bench/meta.json`, `bench/timeseries.csv`, `bench/requests.jsonl` — the driving
  bench run (`python -m bench run`).

## Driving workload
- Model: **NousResearch/Meta-Llama-3.1-8B** (ungated mirror; weights identical to
  the gated `meta-llama/Llama-3.1-8B`, which is config-only in this cache. The
  compute — hence the power draw — is identical. Ran with `HF_HUB_OFFLINE=1`).
- Dataset: `workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl` (300 requests), TP=1,
  max_num_seqs 128, max_num_batched_tokens 2048, max_model_len 8192, bf16, seed 42.
- Bench window (UTC): started 2026-08-18T09:09:13.998Z, finished 09:12:08.512Z.

## Results (source: measured)
| quantity | value | basis |
| --- | ---: | --- |
| idle_power | **31.76 W** | mean of idle_gpu2.csv (320 samples, util 0%) |
| active_power | **297.83 W** | mean where util >= 90% during load (1737 samples; p95 301.3, max 305.6 W — near the 300 W TDP) |
| standby_power | **152.28 W** | mean of the first 2 s after load ended (20 samples) |
| standby_duration | **2 s** (2e9 ns) | A5000-protocol elevated-draw window |

Post-load decay (power vs seconds after load end): +0-2s 152.3 W → +2-5s 83.4 →
+5-10s 68.9 → +10-20s 52.5 → +20-30s 36.8 → +45-60s 34.6 W. A40 holds elevated
power markedly longer than the A5000 (which settled by ~4 s), but the sharp
standby window used by the sim power model is the first ~2 s, matching the A5000
field definition. Full decay to <=110% of idle took ~48 s.

## Not measured
- **Host/node power** (NodePower in the cluster spec) was NOT measured: `ipmitool`
  is absent on this host, so IPMI/DCMI wall power is unavailable, and the running
  profiling job on GPU 0/1 would contaminate any whole-node reading now anyway.
  NodePower therefore stays `source: placeholder`. HANDOVER §7 step 3 flags that
  the Exp 2 mixed-deployment penalty depends on real host power — still open.
- The "resident serving process" idle floor (A5000 distinguished 19 W resident vs
  8.2 W bare) was not isolated here; the bench process exits at load end, so idle
  is the bare-idle floor (31.76 W).
