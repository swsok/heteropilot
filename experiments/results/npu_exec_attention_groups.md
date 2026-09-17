# E-N8 — what one attention execution costs

`WORK_ORDER_npu_exec_model_spike.md` STEP C.3. Read from EDF traces in `outputs/npu_spike/c3_bucket1024/edf`, `outputs/npu_spike/c3_bucket2048/edf`.

**Measured on `npu2` (PCI `45:00.0`), not the `npu0` the committed bundle
was built from** — `npu0` was held by another tenant's pod throughout. A
per-execution cost is a property of the *card*, so these numbers must not be
merged into the `npu0` bundle without a cross-card check.

Decode executions only (`kv > 0` and `attention_size - kv == 1`). The EDF stage
name carries the compiled bucket, so `batch_size` **is** the number of sequences
that execution covered — the group size. Cycles convert at 1.6 GHz.

## attention_size = 1024

| group size | executions | median µs | p05 | p95 | at concurrency |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1536 | 48.3 | 47.7 | 49.2 | c4, c8, c16 |
| 2 | 1440 | 54.5 | 45.9 | 57.1 | c4, c8, c16 |
| 4 | 1504 | 55.0 | 47.7 | 59.5 | c4, c8, c16 |
| 8 | 2688 | 68.8 | 64.4 | 71.4 | c8, c16 |
| 16 | 3168 | 96.5 | 89.1 | 109.3 | c16 |

Least squares over the group sizes above: **45.1 µs fixed + 3.15 µs per sequence**.

## attention_size = 2048

| group size | executions | median µs | p05 | p95 | at concurrency |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2112 | 38.8 | 38.1 | 39.5 | c4, c8, c16 |
| 2 | 2848 | 54.9 | 54.1 | 57.7 | c4, c8, c16 |
| 4 | 3392 | 76.7 | 74.5 | 78.5 | c4, c8, c16 |
| 8 | 5088 | 108.8 | 104.1 | 116.3 | c8, c16 |
| 16 | 5952 | 171.1 | 159.5 | 177.8 | c16 |

Least squares over the group sizes above: **37.1 µs fixed + 8.54 µs per sequence**.

