# Profiler CSV contract (work order §3.7)

The bundle layout and column schemas below are **reverse-extracted from real
artifacts** at the pinned commit (upstream RTXPRO6000 bundle) and from the
locally measured A5000 bundle — not guessed, per §3.7's explicit instruction.
Every importer (`CsvProfileImporter` for ATOM/RNGD in Phase 3, any future
externally measured data) must produce exactly this shape; the simulator's
`trace_generator.py` consumes it as-is.

New file added by HeteroPilot; upstream ships no equivalent document.

## Bundle layout

```
profiler/perf/<HARDWARE>/<org>/<model>/<variant>/tp<N>/
    dense.csv
    per_sequence.csv
    attention.csv
    moe.csv          # MoE models only
    skew.csv         # optional but strongly recommended (see below)
    skew_fit.csv     # produced by the alpha fit over skew.csv
profiler/perf/<HARDWARE>/<org>/<model>/<variant>/meta.yaml
```

- `<HARDWARE>` is a free-form label; it must match the `hardware` field of
  cluster-config instances byte-for-byte (upstream validation checks the
  directory exists).
- `<variant>` encodes dtype: `bf16`, plus `-kvfp8` when the KV cache dtype
  differs (e.g. `bf16-kvfp8`). The profiler derives it; importers must follow
  the same naming or the simulator will not find the bundle.
- Every TP degree the planner may select needs its own `tp<N>/` directory.
  A profile's `max_tp_size` must not exceed the largest measured `tp<N>`.

## Column schemas (exact, header included, no index column)

All times are **microseconds** (`time_us`); the simulator converts to ns
(`trace_generator.py:266`). Rows are keyed measurements; duplicate keys are
deduplicated downstream, last write does not win predictably — importers must
emit unique keys.

### dense.csv — token-count-scaled layers (projections, norms, activations)

| column | type | meaning |
| --- | --- | --- |
| `layer` | str | canonical layer name from `profiler/models/<model_type>.yaml` |
| `tokens` | int | batch token count of the shot |
| `time_us` | float | measured kernel time |

### per_sequence.csv — sequence-count-scaled layers (e.g. lm_head sampling)

| column | type | meaning |
| --- | --- | --- |
| `layer` | str | canonical layer name |
| `sequences` | int | sequence count of the shot |
| `time_us` | float | measured kernel time |

### attention.csv — the 4-axis attention grid

| column | type | meaning |
| --- | --- | --- |
| `prefill_chunk` | int | prefill tokens in the step (0 = decode-only shot) |
| `kv_prefill` | int | KV length behind the prefill chunk |
| `n_decode` | int | decode sequences in the step |
| `kv_decode` | int | KV length per decode sequence (uniform grid) |
| `time_us` | float | measured attention time for the whole step |

The simulator interpolates over this grid. Density matters: the ~2.2pp
end-to-end accuracy cost of a x2-only grid vs the reference's densified one is
quantified in `docs/deviations.md` D11. **Do not describe a bundle's grid from
`meta.yaml`** — resume-mode accumulation means the CSV can be denser than the
recorded factors (also D11); read the keys.

### moe.csv — MoE expert dispatch (MoE models only)

| column | type | meaning |
| --- | --- | --- |
| `tokens` | int | batch token count |
| `activated_experts` | int | experts hit by the batch |
| `time_us` | float | measured expert-path time |

### skew.csv — heterogeneous-decode batches (alpha-fit input)

| column | type |
| --- | --- |
| `regime` | str (`pure` / mixed regimes as emitted by the profiler) |
| `n`, `nb` | int |
| `ratio`, `skew` | float |
| `pc`, `kp`, `kvs`, `kv_big`, `kv_mean` | int |
| `t_mean_us`, `t_max_us`, `t_skew_us` | float |
| `alpha` | float, **may be empty** |

`alpha` is empty when the per-row fit is undefined (the profiler writes
`None` for NaN, `profiler/core/skew.py:353`). Every shipped measured bundle
contains such rows; validators must accept an empty `alpha` cell in
`skew.csv` (and only there).

### skew_fit.csv — bucketed alpha table (fit output, not a measurement)

| column | type |
| --- | --- |
| `pc` | int |
| `n_label`, `skew_rate_label`, `kv_big_label`, `kp_label` | str (bucket labels) |
| `alpha` | float |
| `n_samples` | int |

Importers that cannot produce skew data may omit `skew.csv`/`skew_fit.csv`,
but must say so in `meta.yaml`: the simulator then falls back to uniform-batch
attention, which under-represents ragged decode batches — the dominant regime
in real serving traces.

## meta.yaml — provenance, mandatory

Record at minimum: `profiler_version`, serving-stack name+version (upstream
uses `vllm_version`; importers record whatever produced the numbers, e.g.
`vllm-rbln` or `furiosa-llm` versions), `cuda_version` or backend runtime
version, `gpu` (device string as the machine reports it), `hardware`,
`profiled_at`, `model`, `variant`, `tp_degrees`, `measurement_iterations`,
and the measurement method for imported data. §3.8 provenance discipline
applies: numbers with no measurement never enter a bundle.

**Known limitation (D11)**: `meta.yaml`'s `attention_grid` reflects the *last*
profiling run's factors, not the union actually present in the CSVs. Trust the
CSVs.

## Tier and source labels (tiered profile supply)

Since the tiered-profile work (`WORK_ORDER_tiered_profiles.md`), a bundle's
`meta.yaml` carries two provenance fields:

- `source` — how the numbers entered the repository. Allowed values:
  `measured`, `imported`, `placeholder`, `analytical`, `calibrated`
  (`profiler/contract.py::ALLOWED_SOURCES` is the machine-readable list).
- `tier` — **required** for new bundles; the single source of truth for how
  trustworthy the numbers are. Allowed values: `measured` | `imported` |
  `calibrated` | `analytical` | `placeholder`. Readers that find no `tier`
  fall back to `source`, and treat a bundle with neither as *unknown* — never
  as measured.

### Hardware-label suffix rule for synthetic bundles

Synthetic bundles must never share a hardware label with measured ones:

| tier | label rule | example |
| --- | --- | --- |
| Tier 2 (`measured` / `imported`) | plain label | `A40`, `RNGD-CARD` |
| Tier 0 (`analytical`) | `<LABEL>-t0` | `A40-t0`, `ASCEND_TARGET-t0` |
| Tier 1 (`calibrated`) | `<LABEL>-t1` | `RNGD-CARD-t1` |

The suffix is a secondary signal only; `meta.yaml`'s `tier` field decides.
A measured bundle can therefore never be silently shadowed by a synthetic
one, and the label alone reveals the tier at a glance.

### Extra meta.yaml requirements for Tier 0/1 bundles

An `analytical` or `calibrated` bundle must additionally record:

- `cost_model` — the generator's cost-model identifier (e.g. `roofline-v1`)
- `datasheet_source` — where the datasheet numbers came from
- `efficiency` — the flops/mem/family efficiency values used
- `calibration_anchors` — Tier 1 only: anchor count, per-family distribution,
  and where the anchor measurements came from
- `generated_at`, `generator_version`
- `vllm_version` / `cuda_version` / `gpu` set to `null` with a `null_reason`
  — a synthetic bundle never invents runtime provenance it does not have
- `skew: omitted (...)` — synthetic bundles ship no skew data; the simulator
  falls back to the pooled constant alpha

## Importer checklist (CsvProfileImporter, Phase 3)

1. Validate the header of every CSV byte-for-byte against this contract.
2. Reject non-unique keys and non-positive `time_us`.
3. Emit `meta.yaml` with source attribution (`measured` on what machine, by
   what tool, or the external publication being imported).
4. Place the bundle so `profiler/perf/<hardware>/...` matches the cluster
   config's `hardware` string, then verify with a 20-request simulation before
   trusting any planning result built on it.
