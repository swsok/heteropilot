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


## NUMA binding: the exposure was real and the numbers were not affected

Everything above was measured **unbound** — no `numactl`, no affinity, placement
left to the kernel scheduler. That is a method debt rather than a number debt, and
this section is the check that says which.

The host has two NUMA nodes at a distance of 32 against 10 local, and the repo had
already measured what leaving placement to chance is worth: **1.93× of throughput**
on a TP=4 vLLM deployment, with every A40 measurement before that one unbound.
STEP C then ran unbound too.

`bucket1024` was re-collected on the same card with `--numa-bind auto`, which
resolves the card's node from its PCI address and pins the server *and* the bench
client to it. Verified applied: the server's CPU mask became `0-23,48-71` — node
0's CPU list — against the unrestricted `0-95` of the original runs. Same
workload, same concurrencies, same artifact.

| group size | unbound µs | bound µs | Δ |
| ---: | ---: | ---: | ---: |
| 1 | 48.3 | 48.3 | +0.02 % |
| 2 | 54.7 | 54.8 | +0.12 % |
| 4 | 50.3 | 50.2 | −0.24 % |
| 8 | 67.6 | 67.6 | −0.05 % |
| 16 | 96.5 | 96.6 | +0.05 % |
| **least squares** | **43.7 + 3.20·n** | **43.7 + 3.20·n** | fixed 0.05 %, slope 0.07 % |

Pooled over c4/c8/c16; worst single deviation **0.24 %**.

**Why the exposure was small, now measured rather than argued.** These numbers are
EDF *device* cycles for an on-device attention stage, and NUMA placement governs
host-side memory and DMA staging, not how many cycles a kernel takes once its
operands are resident. And the RNGD artifact's `tensor_parallel_size = 8` is two
fused quads *inside one card* — the intra-card reduction is on the device, so from
the host this is one engine, which is the shape that showed 0.42 % in the A40
check, not the four-worker TP=4 shape that showed 93 %.

**What this does not clear.** The wall-clock side of `measure_envelope.py` — served
concurrency, throughput, TPOT, TTFT — is host-timed and was *not* re-measured here.
C.1's hit rates are runtime counters and are unaffected; `point_c4.json`'s served
concurrency is wall-clock and nothing in this spike rests on it. The committed
RNGD envelope and accuracy domain were taken unbound by earlier sessions and are
outside this spike's scope to re-take.
