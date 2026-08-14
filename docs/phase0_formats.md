# Phase 0 — Upstream format reference

Basis for the `planner/predictor/llmservingsim.py` config compiler and result parser
(work order §5.5). Every schema here was derived from **real artifacts at the pinned commit**,
not from the work order. Where the two disagree, see `deviations.md`.

- Upstream pin: `2c2042ce` (`UPSTREAM_COMMIT`), `astra-sim` at `f82fb3d`
- Verified on: 2026-08-07
- Machine: 2 × RTX A5000 24 GB, Python 3.10.12, no NPU

## 1. Reproduction

ASTRA-Sim builds bare-metal; the simulator itself is CPU-only.

```bash
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install pyyaml pyinstrument transformers datasets msgspec scikit-learn \
  xgboost==3.1.2 matplotlib==3.5.3 pandas==1.5.3 numpy==1.23.5 rich
bash scripts/compile.sh
uv pip install ./astra-sim/extern/graph_frontend/chakra
uv pip install "protobuf>=7.35.1"

python -m serving \
  --cluster-config 'configs/cluster/single_node_power_instance.json' \
  --dtype bfloat16 --block-size 16 \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/phase0_power_repro.csv' \
  --log-interval 1.0
```

Two build-environment traps, both hit and resolved:

- `scripts/compile.sh` calls bare `pip3`. A `uv venv` ships **no `pip3`**, so the call silently
  falls through to `~/.local/bin/pip3` and installs Chakra outside the venv. Re-install Chakra
  with `uv pip install` afterwards.
- Chakra's generated `et_def_pb2.py` carries gencode 7.35.1 while the default resolved runtime is
  protobuf 6.33.6, which raises `VersionError` on import. Pin `protobuf>=7.35.1`. (System
  `protoc` is 3.12.4 and is not what generated the stub.)

Result: run succeeded, `outputs/phase0_power_repro.csv` written, power summary printed.
Re-running byte-identical output confirmed — **the simulator is deterministic run-to-run**.

## 2. Per-request CSV (`--output`)

There is no file called `sim.csv`. The path is whatever `--output` says; `{run_id}` in the path is
substituted with the active run id. Stdout only if `--output` is omitted.

```csv
instance id,request id,model,input,output,arrival,end_time,latency,queuing_delay,TTFT,TPOT,ITL
0,3,meta-llama/Llama-3.1-8B,4,12,570907776,707211372,136303596,1867613,12939416,11214925,"[11073255, ...]"
```

| Column | Type | Unit | Meaning |
| --- | --- | --- | --- |
| `instance id` | int | — | Serving instance that ran the request |
| `request id` | int | — | Monotonic id from the router |
| `model` | str | — | HF model id |
| `input` | int | tokens | Prompt tokens, including prefix-cache hits |
| `output` | int | tokens | **Decode tokens only** (see §2.1) |
| `arrival` | int | ns | Arrival on the simulator clock |
| `end_time` | int | ns | Last generated token completion |
| `latency` | int | ns | `end_time - arrival` |
| `queuing_delay` | int | ns | Arrival → first scheduling step |
| `TTFT` | int | ns | First-token completion − `arrival` |
| `TPOT` | int | ns | `(latency - TTFT) // (output - 1)`, or `0` when `output == 1` |
| `ITL` | str | ns | Serialized Python list — parse with `ast.literal_eval`, not `json` |

All times are **nanoseconds**. Column names contain spaces (`df["instance id"]`).

Not present, despite being needed downstream: any power, energy, or memory column; per-request
prefix-hit counters; `session_id` / `sub_request_index`. These exist on the in-memory `Request`
object but are never written.

### 2.1 The committed `outputs/example_*_run.csv` files are stale — do not use as golden references

The bundled example CSVs were generated before the current `main`. Verified across all 10 rows of
the power example against `workloads/example_trace.jsonl`:

| | `output` column semantics |
| --- | --- |
| Committed `example_power_run.csv` | `input_toks + output_toks` (total length) |
| Current `main` (this run) | `output_toks` only (decode tokens) |

Example, request 3: trace says `input_toks=4, output_toks=12`; committed CSV records `output=16`,
current run records `output=12`. Current behavior matches `docs/docs/simulator/reading-output.md`;
the committed CSVs do not.

Latencies moved too — request 3 TPOT went 25.54 ms → 11.21 ms — consistent with the accounting
fixes merged into `serving/` after the artifacts were committed (KV eviction/reload, multi-tier
prefix accounting, per-block weight ÷ `pp_size`, duplicate prefix-hit accounting).

Consequence for work order §9: golden-output regression tests must be generated **by us at the
pinned commit**, never adopted from `outputs/`.

## 3. Stdout structure

Four sections, in order. The `--log-level` flag (`WARNING` default / `INFO` / `DEBUG`) changes only
what surrounds them.

1. **Banner** — resolved input configuration, including `Run ID` and the ASTRA-Sim inputs root.
2. **Throughput log line** every `--log-interval` seconds. Field set varies by enabled feature
   (`P=`/`D=` for prefill/decode split, `prefix_hit=`, `alltoall=`, `pim_busy=`, `cxl_mem=`,
   `power=`).
3. **Final summary** — totals and throughput.
4. **Per-instance percentiles** — `Mean / Median / P99` for TTFT, TPOT, ITL in ms.

Observed final summary from the reproduction run:

```text
Total clocks (ns):                    1665077255
Total latency (s):                    1.665
Total input tokens:                   120
Total generated tokens:               591
Request throughput (req/s):           6.01
Average prompt throughput (tok/s):    72.07
Average generation throughput (tok/s):354.94
Total token throughput (tok/s):       427.01
```

### 3.1 Power output — stdout only

Emitted only when the node config has a `power:` block. **No machine-readable file is produced.**

```text
Total energy consumption (kJ):                  1.42
Node 0 total energy consumption (kJ):           1.42
├─ Base Node energy consumption (J):            99.90
├─ NPU energy consumption (J):                  972.14
├─ CPU energy consumption (J):                  233.13
├─ Memory energy consumption (J):               53.28
├─ Link energy consumption (J):                 8.33
├─ NIC energy consumption (J):                  33.30
└─ Storage energy consumption (J):              16.65
Power per 1.0 sec (W): [845.91]
```

Mapping to the metrics the planner must produce (work order §4):

| Planner metric | Source |
| --- | --- |
| `total_energy_j` | `Total energy consumption (kJ)` × 1000 |
| `average_power_w` | mean of the `Power per N sec (W)` list |
| `peak_power_w` | max of the `Power per N sec (W)` list |
| `tokens_per_joule` | Σ decode tokens from CSV ÷ `total_energy_j` |
| `p50/p95/p99 TTFT,TPOT` | computed from the CSV, **not** from the printed P99 |
| `slo_goodput_rps` | computed from the CSV |

Percentiles come from our own `planner/util/percentile.py` (numpy `linear`) per work order §4 — the
printed P99 uses the simulator's own method and must not be mixed in.

The power time series resolution equals `--log-interval`, so `peak_power_w` is an interval average,
not an instantaneous peak. A coarse interval understates the peak; pick the interval deliberately
when a power cap is being enforced.

Parsing this text is brittle. Options and the recommendation are in `deviations.md` (D2).

## 4. Cluster config JSON (`--cluster-config`)

Compile target for `ClusterSpecV2`. Reference: `docs/docs/reference/cluster-config.md`. Do not
edit `configs/cluster/*.json` in place — the planner writes generated files to a temp dir.

### Top level

| Field | Type | Req | Description |
| --- | --- | --- | --- |
| `num_nodes` | int | ✓ | Must equal `len(nodes)` |
| `link_bw` | float \| float[] | ✓ | ASTRA-Sim link bandwidth, **GB/s**. Scalar applies to every topology dimension; array must match `network.yml::npus_count` rank |
| `link_latency` | float \| float[] | ✓ | Link latency, **ns**. Same scalar/array rule |
| `nodes` | array | ✓ | |
| `cxl_mem` | object | | `mem_size` GB, `mem_bw` GB/s, `mem_latency` ns, `num_devices` |

**There is no link graph.** No per-link `src`/`dst`, no `contention_group`, no
`energy_per_bit`. Topology is a per-dimension bandwidth/latency vector. See `deviations.md` (D3).

### Per node

| Field | Type | Req | Description |
| --- | --- | --- | --- |
| `num_instances` | int | ✓ | Must equal `len(instances)` |
| `cpu_mem` | object | ✓ | `mem_size` GB, `mem_bw` GB/s, `mem_latency` ns, optional `pim_config` |
| `instances` | array | ✓ | |
| `power` | object | | Enables the power model on this node |

### Per instance

Required: `model_name` (must resolve to `configs/model/<model_name>.json`), `hardware` (must match
`profiler/perf/<hardware>/`), `npu_mem.{mem_size,mem_bw,mem_latency}`, `pd_type`
(`"prefill"` / `"decode"` / `null` for aggregated).

Parallelism — `num_npus == tp_size * pp_size` always. `pp_size` defaults 1; `ep_size` defaults
`tp_size` for MoE and 1 for dense, must divide `num_local_experts`, and must be `<= tp_size`
without a `dp_group`. `dp_group` is a string that makes instances share experts via cross-instance
ALLTOALL — it is **MoE expert parallelism, not data-parallel replication**.

Runtime overrides (per instance, each falling back to the matching CLI flag): `max_num_seqs`,
`max_num_batched_tokens`, `long_prefill_token_threshold`, `block_size`, `dtype`, `kv_cache_dtype`,
`enable_chunked_prefill`, `enable_prefix_caching`, `prioritize_prefill`, `enable_local_offloading`,
`enable_attn_offloading`, `enable_sub_batch_interleaving`, `enable_block_copy`. `0` means unlimited
for the two budget fields. This set covers the work order's `vllm_knobs` exactly.

`placement` optionally pins weights / `kv_loc` / `kv_evict_loc` per block range or named layer to
`npu` / `cpu` / `cxl:<id>`.

### `power` block

```json
"power": {
  "base_node_power": 60,
  "npu": {"RTXPRO6000": {"idle_power": 35, "standby_power": 300,
                         "active_power": 600, "standby_duration": 18}},
  "cpu":     {"idle_power": 10, "active_power": 200, "util": 0.15},
  "dram":    {"dimm_size": 32,  "idle_power": 2.0,   "energy_per_bit": 6.0},
  "link":    {"num_links": 1,   "idle_power": 5,     "energy_per_bit": 4.0},
  "nic":     {"num_nics": 1,    "idle_power": 20},
  "storage": {"num_devices": 2, "idle_power": 5}
}
```

`npu.<hardware>` is keyed by the instance's `hardware` string, so a multi-hardware node lists one
entry per hardware type. The planner must synthesize this from `profiles/accelerators/*.yaml`;
the field sets do not line up — see `deviations.md` (D7).

### Validation rules enforced upstream

- `num_nodes == len(nodes)`, `num_instances == len(instances)`
- `weight_per_gpu * num_npus <= npu_mem.mem_size * num_npus`
- `profiler/perf/<hardware>/<model_name>/<variant>/tp<tp_size>/` must exist
- instances sharing a `dp_group` must share `ep_size` and `tp_size`

## 5. Available hardware profiles

```
profiler/perf/RTXPRO6000/meta-llama/Llama-3.1-8B
profiler/perf/RTXPRO6000/Qwen/Qwen3-30B-A3B-Instruct-2507
profiler/perf/RTXPRO6000/Qwen/Qwen3-32B
```

**One hardware class only.** `configs/cluster/single_node_single_instance_H100.json` exists but
has no matching `profiler/perf/H100/`. Any `hardware` value without a profile directory fails
validation. See `deviations.md` (D4) — this gates Phase 3.

Also note `configs/cluster/single_node_heterogeneous.json` is **not** hardware-heterogeneous: both
instances are `RTXPRO6000`, split prefill/decode. Upstream "heterogeneous" means P/D roles.

## 6. CLI flags the planner will drive

Beyond the knobs mirrored as per-instance overrides:

| Flag | Default | Why the planner cares |
| --- | --- | --- |
| `--cluster-config` | — | Generated per candidate |
| `--dataset` | `None` | Generated workload trace |
| `--num-reqs` | `0` (all) | Trims trace length for cheap pruning passes |
| `--output` | `None` | Per-request CSV; supports `{run_id}` |
| `--request-routing-policy` | `LOAD` | `RAND` breaks reproducibility — avoid |
| `--network-backend` | `analytical` | `ns3` is detailed but WIP |
| `--run-id` | auto | **Isolates parallel runs** |
| `--inputs-root` | `astra-sim/inputs/runs/<run-id>` | Move intermediates to tmpfs |
| `--log-interval` | — | Sets power time-series resolution (§3.1) |

`--run-id` / `--inputs-root` make concurrent candidate evaluation safe without extra locking —
each run gets its own ASTRA-Sim input root, cleaned up on success.

**There is no `--seed` flag.** Determinism comes from a fixed input trace plus a deterministic
routing policy, so work order §9 reproducibility is satisfied by seeding *our* workload generator
(`planner/util/workload.py`) and recording the seed in provenance. See `deviations.md` (D5).

## 7. Workload JSONL

```json
{"input_toks": 10, "output_toks": 70, "arrival_time_ns": 46926808,
 "input_tok_ids": [...], "output_tok_ids": [...]}
```

`input_tok_ids` must be populated for prefix caching to register hits. `arrival_time_ns` is
absolute on the simulator clock. This is the format `planner/util/workload.py` must emit from the
`ServiceSpec.traffic` distribution.
