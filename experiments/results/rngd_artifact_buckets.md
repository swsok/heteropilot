# RNGD — the bucket grid the vendor runtime is compiled for

**What this is.** `furiosa-llm` is an ahead-of-time compiler: a served model is a fixed set of
compiled plans ("buckets"), each for one `(batch_size, attention_size, kv_cache_size)`. Nothing is
compiled at run time, so a step that does not match a bucket exactly is padded up to one that does.
This file records the grid of the three artifacts present on the NPU node, because the grid — not
the card — is what the simulator's vLLM-style scheduler fails to reproduce
(`WORK_ORDER_npu_exec_model_spike.md` §1.1).

**Provenance.** Extracted with `experiments/scripts/furiosa_artifact_buckets.py` on 2026-09-16,
NPU node (`scripts/whichnode.sh`: 3 RNGD cards, 4 ATOM). The script reads `artifact.json` only —
no `furiosa_llm` import, no device — so it runs under `.venv` and is reproducible on any checkout
that has the artifacts in the HuggingFace cache. Classification follows
`furiosa_llm/metadata/config_types.py` verbatim:

```
input_ids_size = attention_size - kv_cache_size
prefill : kv_cache_size == 0
decode  : kv_cache_size > 0 and input_ids_size == 1
extend  : kv_cache_size > 0 and input_ids_size > 1     # chunked prefill
```

`furiosa_llm.utils.compute_bucket_lengths` counts `extend` together with `prefill`; this file keeps
them apart because they behave differently (an extend step carries KV, a prefill step does not).

## The three artifacts — two schemas, three grids

| artifact | id | furiosa-llm | tp | pipelines | prefill | extend | decode | max bucket | schema |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `furiosa-ai/Llama-3.1-8B-Instruct` | `d6ae6a43` | `b62dbc1` | 8 | 47 | 8 | 74 | 46 | 131072 | `pipeline_metadata_list[].attention_buckets` |
| `furiosa-ai/Llama-3.3-70B-Instruct` | `5ea10eac` | `b62dbc1` | 32 | 46 | 8 | 74 | 45 | 131072 | same |
| `furiosa-ai/Llama-3.1-8B-Instruct-FP8` | `97f74da4` | `f285611` | 8 | 40 | 10 | 3 | 27 | 32768 | pipeline-name regex `-kv{K}-b{B}-attn{A}` |

The 70B snapshot appears twice in the cache (`7b640482`, `2cbb7a62`) under one `artifact_id`; they
are the same artifact.

**This is why the grid is stored beside the perf bundle and not in `profiles/accelerators/*.yaml`**
(work order A7). The same three RNGD cards serve all three artifacts. The FP8 build — a different
compiler, a different metadata schema, 10/3/27 instead of 8/74/46, and a ceiling of 32768 instead
of 131072 — would change every input of an execution model while changing no hardware fact.

The machine-readable grid for the artifact the committed measurements were taken under
(`d6ae6a43`) is
`profiler/perf/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/artifact_buckets.yaml`, regenerable with:

```bash
python experiments/scripts/furiosa_artifact_buckets.py <artifact_dir> --yaml
```

## `d6ae6a43` — the grid the committed envelope and EDF traces were measured under

### prefill — 8 buckets, every one `bs=1`

`attn ∈ {128, 256, 384, …, 1024}`, `kv=0`, i.e. 128-token steps up to 1024.

### extend (chunked prefill) — 74 buckets, every one `bs=1`

`chunk ∈ {128 … 1024}` in 128-token steps, against `kv` from 128 up to 130944; `attn = kv + chunk` holds for every one of the 74.

### decode — 46 buckets, every one `input_ids=1`

| bs | widest `attention_size` | `bs × attn` (KV budget) |
| ---: | ---: | ---: |
| 1 | 131072 | 131072 |
| 2 | 131072 | 262144 |
| 4 | 65536 | 262144 |
| 8 | 32768 | 262144 |
| 16 | 16384 | 262144 |
| 32 | 8192 | 262144 |
| 64 | 4096 | 262144 |
| 128 | 4096 | 524288 |
| 256 | 2048 | 524288 |

Within each `bs` the `attention_size` ladder is powers of two from 1024 up to the widest entry.

## Three facts that follow, and one that does not

**(i) Mixed prefill+decode steps are not compiled.** Every prefill and extend bucket is `bs=1`, and
every decode bucket is `input_ids=1`. There is no bucket in which a prefill chunk and a decode token
coexist, so one forward pass is either *one request's* prefill/extend chunk or a decode batch.
D17 recorded this question as "not decidable from these traces" because the EDF CSV carries
durations and not timestamps; the grid decides it structurally instead. See `docs/deviations.md` D90.

**(ii) The decode grid is a KV budget.** `bs × attn` is 131072 on the `bs=1` row and 262144–524288
on every other. The consequence is an admission ceiling the simulator does not have: on sharegpt
(mean KV ≈ 2200, D17) the `bs=256` row is unusable, because its widest bucket is `attn=2048 < 2200`,
so the largest decode batch a sharegpt-like workload can actually run is **128**. `rps_aware`'s E6
gave RNGD candidates `max_num_seqs 256` — a setting this artifact cannot execute.

**(iii) Padding has two axes.** A step is padded up in batch (to the next compiled batch size) and in
context (to that batch size's next `attention_size`). The batch ladder is **not** the powers of two
alone — see below.

**What does not follow: a uniform 1024-token attention grid.** The work order derives
"131072 / 128 = 1024, so the kernelwise path's attention bucket width is 1024 tokens" from the
kernelwise menu's size. That derivation does not hold. Pipeline 0 (`composition_type: kernelwise`)
carries 128 attention buckets, and 128 is exactly `8 prefill + 74 extend + 46 decode` — the menu is
the **union of the compiled buckets**, verified as a set. Its spacing is not uniform: the distinct
steps present are 128 (up to 1024) and then 1024, 2048, 4096, 8192, 16384, 32768, 65536, and the
decode edges specifically are `{1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}`. The 46
composed pipelines are each one bucket of that menu (verified: a strict subset).

This matters because two later rules are written in terms of "1024-unit KV buckets" — the R-attn
rule of STEP A.3 and P3 of STEP B.1. **Open for STEP A.1**, recorded here rather than decided:

- grouping by the real decode edges (powers of two from 1024) puts a sharegpt batch of ~29
  sequences into roughly 2–4 groups, which is what D17 *measured* (1.95 → 3.08 attention executions
  per layer as batch grows 1.95 → 29.09, saturating near 3);
- grouping on a uniform 1024-token grid would put the same batch into many more groups than 3.

So the census in STEP A.1 should count distinct groups **both ways** and compare each against D17's
1.95 / 2.40 / 2.87 / 3.03 / 3.08. This is a prediction, not a result: nothing here measures the
runtime's grouping, only what it was compiled for.

## The kernelwise menu in full

| field | value |
| --- | --- |
| `attention_buckets` | 128 — the union of all prefill/extend/decode buckets |
| `tokenwise_buckets` (`input_size`) | 1, 2, 4, 8, 16, 32, 64, 128, 256, **384**, 512, 1024 |

The tokenwise ladder is the one a decode batch's size is padded onto, and it is **not** powers of
two throughout: 384 sits between 256 and 512. A "round the batch up to the next power of two" rule
(R-pad in STEP A.3, P2 in STEP B.1) is therefore correct only at or below 256. Above it the padding
target is the next entry of this ladder.
