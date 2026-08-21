# Calibration store (work order §2.1, §5.8)

Per-hardware sim-vs-real calibration models, one YAML file per fit, written by
`planner.predictor.calibration.save_calibration` and read by `load_calibration`.

Each file is a `CalibrationModel`:

- `hardware.<label>.ttft` / `.tpot`: linear fits `real = alpha * sim + beta`.
- `hardware.<label>.errors.<workload_bucket>`: per-metric prediction-error
  distributions (`mean_error`, `p95_abs_error`, `worst_error`, `sample_count`).

Fits come only from real `bench/` validation summaries
(`outputs/phase0_bench/<hw>/vllm/validation/summary.txt`) via
`fit_from_summaries`. Absolute rule 3: with no data the model is the identity
(`alpha=1, beta=0`, `source: identity`) and every robust margin is 0.

The calibration sign is hardware-dependent (A5000 under-predicts, RTXPRO6000
over-predicts — HANDOVER.md §7), so a fit is always per-hardware, never a global
factor.

Applying calibration or robust margins to planning is opt-in: the `plan` command
does not use this store, so default plans and the golden regression outputs are
unaffected.
