# Hardware roadmap — incoming cluster access (recorded 2026-08-14)

The user has confirmed access, expected from the week of 2026-08-17 onward, to:

| Resource | Count | Backend | Status |
| --- | --- | --- | --- |
| A40 x8 GPU nodes | up to 8 nodes (up to 64 GPUs) | cuda | not yet reachable |
| Rebellions ATOM server | 4 devices | rbln (new) | not yet reachable |
| FuriosaAI RNGD server | 4 devices | furiosa (new) | not yet reachable |

This changes the project's constraint structure. Until now every non-RTXPRO6000
number was either locally measured on 2xA5000 or a placeholder; the plan below
records what the new hardware unblocks and in what order to bring it up.

## What it changes, by open item

**D4 (only one shipped hardware profile; NPU data source undecided) — resolution
path changes.** The work order assumed Ascend as the NPU target and CSV import
of externally measured data as V1. The concrete NPU targets are now ATOM and
RNGD, both physically accessible: profiles can be *measured*, not imported.
The `backend` field was always declared extensible (`cuda | ascend | <향후 추가>`);
`rbln` and `furiosa` are added as identifiers. The `ascend_target.yaml` stub
stays as a schema example but is no longer the expected Phase 3 vehicle.

**Phase 4 calibration model problem — solved.** Qwen3-32B (the work order's
headline model, ~64 GB bf16) does not fit local hardware, which made Phase 4
calibration impossible here. On A40 (48 GB) it serves at TP=2. Real-vLLM
calibration on the headline model becomes possible.

**Exp 3 (network sensitivity) gains ground truth.** Up to 8 multi-GPU nodes
means real inter-node collectives to validate the Level-1/Level-2 topology
models against, instead of simulator-only sweeps.

**Exp 4 (GPU vs NPU SLO-goodput/J) becomes a measured experiment.** This was
the largest "simulator-only, placeholder inputs" caveat in the paper plan.
Both NPU vendors position their parts on efficiency, which is exactly the
tokens/J axis this project optimizes.

## Bring-up order (work order §11 V1→V2 discipline)

1. **Inventory before anything** (each system, day one):
   - GPU nodes: `nvidia-smi -L`, `nvidia-smi topo -m` (NVLink bridge vs PCIe),
     NIC inventory (`ibstat` / `ethtool`), inter-node fabric and speed.
   - NPU servers: device count/memory via vendor tools (`rbln-stat`,
     `furiosactl` or equivalents — verify actual tool names on the machine),
     host topology.
   - Write each as a `ClusterSpecV2` YAML under `examples/clusters/` with
     per-field `source:` provenance. No number enters a profile that was not
     read off the machine or a vendor document (cite which).
2. **A40 profiling** (existing profiler works as-is; CUDA sm_86 like A5000):
   `python -m profiler profile meta-llama/Llama-3.1-8B --hardware A40 --tp 1,2,4,8`
   plus Qwen/Qwen3-32B. Use the x2 grid initially; densify later if the D11
   penalty (~2.2pp) matters for the experiment at hand.
3. **A40 real bench + calibration** (Phase 4 entry): repeat the A5000
   sim-vs-real protocol (docs/phase0_bench_plan.md) on one A40, then at TP=2/4,
   then across nodes. Re-measure the D10 KV budget on 48 GB.
4. **NPU V1 — serving-stack survey and CSV import.** Verify what actually runs:
   vLLM-compatible serving stacks exist for both vendors (vllm-rbln for ATOM,
   furiosa-llm for RNGD) but versions, supported models and metrics endpoints
   must be confirmed on the machines, not assumed. Measured latency data enters
   through `CsvProfileImporter` against `profiler/CONTRACT.md` (write the
   contract first — it is the §3.7 prerequisite and needs no hardware).
5. **NPU V2 — native profiling adapters** only after V1 produces a working
   planner loop with measured NPU envelopes.

## What stays true regardless

- TP/PP never crosses a backend boundary (absolute rule 2): A40 islands,
  ATOM islands and RNGD islands are separate; heterogeneity is replica- or
  role-level.
- Anything not yet measured on these machines is `source: placeholder` and the
  profile stubs added alongside this document say so field by field.
- The simulator predicts; the new hardware measures. Results must keep the
  label they were born with.

## Open questions to resolve at first access

- A40 intra-node interconnect (NVLink bridge pairs? PCIe only?) and inter-node
  fabric (IB? 100/200G? Ethernet?) — determines the ClusterSpecV2 link graph
  and whether Exp 3 has interesting bandwidth points.
- ATOM / RNGD: device memory and TDP as reported by the machine's own tools;
  serving stack versions; which of our target models they support at which
  dtypes (fills `supported_models`, which is deliberately empty in the stubs —
  empty means excluded from candidates until verified).
- Scheduler/queue system on the cluster (SLURM? bare SSH?) — affects the
  Phase 4 `deploy/` launcher design (§5.7 allows local/SSH only).
