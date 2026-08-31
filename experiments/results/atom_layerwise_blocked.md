# ATOM layerwise profiling — why there is no `profiler/perf/ATOM/` bundle

*Attempted 2026-08-28 on the NPU server, rbln0 (`RBLN-CA22`). This is a negative
result with the evidence behind it. The device facts (memory, power) that **did**
land are in `atom_device_facts.md`; this page is only about the perf bundle.*

## The short answer

**ATOM's per-call host I/O is larger than the kernels being measured**, and there
is no device-span profiler to see past it. Both documented routes were tried in
full. Neither yields numbers that could sit in the same table as the RNGD or A40
bundles, so no bundle was shipped.

Committed anyway, because they are what a future attempt should start from:

| artifact | what it is |
| --- | --- |
| `outputs/atom_profile/io_cost_vs_size.csv` | the measurement that decides this |
| `outputs/atom_profile/layerwise_attempt/` | the CSVs the harness did produce, **not a bundle** |
| `experiments/scripts/profile_atom.py` | the harness, working, awaiting an instrument |

## Route B — vLLM (`python -m profiler`), blocked on two independent grounds

`docs/HANDOVER_NPU.md` §3 says that once the vendor versions are consistent, the
existing vLLM profiler drives ATOM as-is. The first half of that held: a
consistent 0.11.0 trio assembled offline in `.venv-rbln-vllm` (torch 2.11.0+cpu,
transformers 5.8.1, vllm 0.22.0, vllm_rbln 0.11.0, optimum-rbln 0.11.0.post1,
rebel 0.11.0) activates `RblnPlatform`, and `python -m profiler` loads.

The second half did not. The profiler's two engine settings, marked in
`profiler/core/config.py` as "should not be changed (changing them breaks
profiling correctness)", each collide with one of vllm-rbln's two paths:

| | Llama | `load_format: dummy` | `enforce_eager: True` |
| --- | :--: | :--: | :--: |
| optimum path (default) | registered | **no** — AOT-compiles a *real* checkpoint | — |
| native path (`VLLM_RBLN_USE_VLLM_MODEL=1`) | **not registered** | yes | **no** — needs `VLLM_RBLN_USE_DEVICE_TENSOR=1` |

`VLLM_RBLN_USE_DEVICE_TENSOR=1` cannot be satisfied either: it needs a real torch
device named `rbln`, and nothing on this machine registers one —
`torch.device("rbln")` raises `Expected one of cpu, cuda, … privateuseone`. There
is no `torch_rbln` package, unlike RNGD where `furiosa.torch` registers `rngd:`
as PrivateUse1. The native path also registers only deepseek_v2 / gpt_oss /
minimax_m2 / qwen2 / qwen3 — **not llama**, which is the model RNGD was profiled
with and therefore the only one that makes a cross-hardware comparison possible.

Getting past either means editing `profiler/`, which is upstream and pristine
until Phase 5 (absolute rule 1). Overriding `enforce_eager` would also defeat the
measurement it enables: launches would be captured graphs rather than
independently timeable events.

## Route A — a `rebel` harness like RNGD's, blocked by the instrument

`experiments/scripts/profile_atom.py` mirrors `profile_rngd.py`: compile one
canonical layer per graph, time it on device, emit §3.7 CSVs for
`CsvProfileImporter`, leaving `profiler/` untouched. It **works** — 284 shots
across all eleven layers plus the attention grid, zero compile failures.

What it cannot do is produce *device* time.

### There is no device-span profiler

RNGD's numbers are true device spans: `furiosa.torch` exposes an
`RNGDProfiler` whose `Renegade::TuExec` and DMA spans are summed as a union.
ATOM's equivalent exists but is unreadable — `rebel._C.profiler.start(path)` with
`RBLN_PROFILER=1` emits protobuf traces carrying `Neural Engine Clusters`,
`Task DMA`, `comp_cycle` and `transfer` records, but the schema is undocumented
and no decoder ships with the stack (`rebel/core/profiler_backend.py` is
Pyarmor-obfuscated and exposes only `start`/`done`). Deriving microseconds would
mean guessing a field layout *and* a clock rate, and a wrong guess would look
exactly like a measurement. **That was not done.**

### So timing is wall clock, and wall clock here is I/O

`outputs/atom_profile/io_cost_vs_size.csv` — cost of a graph that does nothing
but `x + 1`, i.e. pure per-call I/O:

| input elements | bytes | min µs |
| ---: | ---: | ---: |
| 8 | 16 | 6.4 |
| 1,024 | 2 KB | 56.3 |
| 65,536 | 128 KB | 67.5 |
| 1,048,576 | 2 MB | 300.6 |
| 4,194,304 | 8 MB | 999.7 |

Compare that with what is being measured: RNGD's device spans for the same layers
run **3–200 µs**. On ATOM the transport costs more than the computation, for
every layer in the model.

### Three subtraction schemes, and why each fails

1. **Constant floor.** The first version subtracted 6.5 µs, calibrated on a 1×8
   tensor. Every shot then carried its own transfer cost, inflating the
   elementwise layers 8–25× against RNGD — `layernorm@1` read 52.4 µs where
   RNGD's device span is 3.4 µs. *This produced a bundle that validated against
   the contract and imported cleanly.* It was wrong, and was deleted.
2. **Per-shot I/O baseline** — same input shapes, `xs[0] + 1` plus `sum()` over
   the rest. Correct for single-input layers: `o_proj` landed at 0.83–1.09× RNGD
   and `layernorm` at 0.5–1.9×, both physically sensible. But `sum()` is a full
   reduction, so for multi-input layers the baseline costs more than the layer —
   measured on one attention shape, baseline **27,919 µs** against a real layer
   of a few hundred. **145 of 284 shots came out negative** and were clamped.
   Replacing `sum()` with a single-element index fixes the cost (146 µs) but the
   compiler then elides the unread tensors — that variant is cheaper than
   transferring `k` alone (710 µs for 8.4 MB), so it under-subtracts instead.
   **The transfer only happens if the graph consumes the data, and consuming it
   costs compute.** There is no cheap full-read op.
3. **Repetition slope** — run the layer *k* times in one graph so transfer is
   paid once, take `(t_k − t_1)/(k−1)`. The accumulator defeats it: summing
   outputs per iteration is itself a reduction of the same order as the layer, so
   `layernorm@128` gave a slope of 1815 µs against RNGD's 16.2 µs. Chaining
   outputs into inputs instead would work only for shape-preserving layers.

### Why a partial bundle is not an option

`attention.csv` is `required=True` in `profiler/core/importer.py`. The dense and
per-sequence measurements are the ones the subtraction handles well; attention is
exactly the case it cannot, because a decode shot carries megabytes of K/V whose
transfer is inseparable from its compute. A bundle cannot ship without the file
that is least trustworthy.

## What was kept, and what it is not

`outputs/atom_profile/layerwise_attempt/` holds the CSVs from the scheme-2 run
with `measurement_notes.json` beside them. **This is not a profiler bundle and
must not be imported as one.** Read it only with the sidecar: 166 of 284 shots
are flagged `baseline_dominated`, and 145 of those are clamped negatives.

The single-input dense rows are the defensible part, and they agree with RNGD
where physics says they should. For scale, after correct per-shot subtraction:

| layer | tokens | ATOM µs | RNGD µs | ATOM/RNGD |
| --- | ---: | ---: | ---: | ---: |
| o_proj | 1 | 145.1 | 174.3 | 0.83 |
| o_proj | 128 | 228.5 | 209.6 | 1.09 |
| layernorm | 128 | 8.4 | 16.2 | 0.52 |

Suggestive, not citable — a three-row sample from a method whose own attention
case is broken.

## What would unblock this

In rough order of expected effort:

1. **The `.pb` trace schema from Rebellions.** The tracer already runs and
   already records `comp_cycle` and `transfer`. This is the direct analogue of
   what makes the RNGD bundle possible, and it is a documentation request.
2. **A torch backend registering device `rbln`**, which turns on
   `VLLM_RBLN_USE_DEVICE_TENSOR=1` and with it the vLLM-native path — though
   llama would still need registering there.
3. **A Llama entry in vllm-rbln's native model registry**, after (2).

Until one of them, `profiles/accelerators/rbln_atom.yaml` keeps `sim_hardware:
null` and empty `supported_models`, so ATOM stays out of candidate generation and
out of Exp 4. Measured memory and power do not change that (absolute rule 3).

## Reproduce

```bash
# the decisive measurement
.venv-rbln/bin/python - <<'PY'
import time, torch, rebel
from torch import nn
class Null(nn.Module):
    def forward(self, x): return x + 1
for n in (8, 1024, 65536, 1048576, 4194304):
    x = torch.zeros(1, n, dtype=torch.bfloat16)
    cm = rebel.compile_from_torch(Null().eval(), [("x", [1, n], "bfloat16")])
    rt = cm.create_runtime(tensor_type="pt", device=0); rt.run(x)
    ts = []
    for _ in range(30):
        s = time.perf_counter(); rt.run(x); ts.append((time.perf_counter()-s)*1e6)
    print(n, n*2, round(min(ts), 1))
PY

# the harness (produces CSVs; see above for why they are not a bundle)
.venv-rbln/bin/python experiments/scripts/profile_atom.py \
    --device 0 --reps 30 --out outputs/atom_profile/layerwise_attempt
```
