# S7.2 — the furiosa open-loop route, run once end to end on the card

*`WORK_ORDER_domain_scoping.md` STEP S7.2, `docs/deviations.md` D114. Run
2026-09-18 on the **NPU node** (`whichnode.sh` → `npu`), RNGD **npu0**, PCI
`0000:03:00.0`. No holder pod was present — checked with the process list, not
`alloc_status`, which reads all-zero while a pod holds a card (`docs/nodes/npu.md`).*

> **This is a plumbing check, not a calibration point, and it must never be used
> as one.** It ran **20 requests**. The harness itself says why that is too few:
> `--num-reqs` defaults to 300 and its help names D32, which measured the drain
> tail at −31.7 % on a 20-request run. The artifacts live in
> `outputs/s72_smoke/` and are untracked and regenerable. S7.3's points are
> 300 requests, three repeats, and are a separate measurement.

## What it establishes

That the route exists and every seam holds on real hardware — which no unit test
can show, because the seams are a vendor server, a NUMA binding read from sysfs,
a power sampler and an HTTP client in four different processes.

| seam | evidence from this run |
| --- | --- |
| server launch, furiosa backend | `furiosa-llm serve --devices npu:0:*` came up; `wait_for_server` got `200 OK` on `/v1/models` |
| NUMA binding, default `auto` | point records `numa_bind: node0` — resolved through `furiosa-smi info` → BDF → `/sys/bus/pci/devices/0000:03:00.0/numa_node`, which is where npu.md says all three cards sit |
| rate rescaling | `--target-rps 0.5` against a 10.966 rps trace: "offering 0.500 rps, span 40.0s, no concurrency bound" |
| the client kept the schedule | `launch_error_ms` mean **0.78**, p99 **1.89**, max **1.95** — the client is not reporting its own lateness as the server's TTFT |
| A5(c), power with utilisation | bench window **69.86 W at 28.07 %** over 156 samples; idle window **38.67 W at 0.0 %** over 180. The idle figure matches npu.md's ~40 W RNGD idle |
| saturation test on real data | `ttft_drift_slope` = **+0.0026 s/s**, flat, `saturated: false` — the right answer at 0.5 rps on a card that serves ~100 concurrent, and the first check of this metric against something other than a fixture |
| the artifact says what it is | `protocol: open_loop`, `harness: deployed-server-over-http` on the point and in `envelope.json`'s run block |

Measured, for the record and for nothing else: served concurrency **4.378** at an
offered 0.5 rps, p99 TPOT **24.52 ms**, 203.7 output tok/s, 20/20 requests ok,
`ignore_eos: true`, warm-up 0.25 s paid before the schedule started.

## What it does not establish

* **Nothing about accuracy.** No simulator run was paired with it. The point of
  S7.3 is the pairing; this run only shows the harness can produce one side of it.
* **Nothing at the operating points S7.0 selected.** Those are served
  concurrency 37.975 and 74.498 (`v3r_candidate_selection.md`); this is 4.378.
* **Nothing about repeatability.** One repeat. The run-to-run p99 spread S7.0
  asked to be reported beside every verdict comes from S7.3's three.

## Reproduce

```bash
ART=~/.cache/huggingface/hub/models--furiosa-ai--Llama-3.1-8B-Instruct/snapshots/231d94fbc03cdd66aaeb2411697064a45f008ec7
PYTHONPATH=$PWD .venv/bin/python experiments/scripts/measure_envelope.py \
    --backend furiosa --mode open --rps 0.5 --num-reqs 20 \
    --artifact "$ART" --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
    --card 0 --port 8020 --out outputs/s72_smoke
```

The **driver** runs in `.venv` (it needs `planner.util.percentile`); the **client**
runs in `/usr/bin/python3`, which is where the vendor stack and `openai` live —
that split is the `bench_python` column of the `BACKENDS` table and not a choice
made here. `--card` is the **physical** npu index, and the labels move: npu.md
recorded `45:00.0` as `npu2` on 2026-09-17 and `furiosa-smi` calls it `npu3`
today, so pin by BDF when it matters.
