# RPS-aware accelerator selection — design

*Written 2026-09-01. A design proposal, not an implementation. Grounded in the
envelope measurement of D21 (`experiments/results/rngd_concurrency_envelope.md`)
and the ATOM power measurement of `experiments/results/atom_device_facts.md`.*

## The question this answers

Given a heterogeneous cluster and a `ServiceSpec`, recommend the resource
combination and distributed-inference architecture that minimises power while
meeting the SLOs — **and choose between GPU and NPU according to the RPS the
service is expected to carry.**

The last clause is what the planner cannot currently express, and this document
sets out what has to be added.

## 1. Why the current model cannot answer it

`AcceleratorProfile` describes performance with **scalars**:
`memory_bandwidth_gbps`, a single `active_power`, and a layerwise perf bundle.
The measurement in D21 shows performance is a **function of the operating
point**. One RNGD card, Llama-3.1-8B, measured:

| served concurrency | throughput tok/s | TPOT ms | marginal exponent |
| ---: | ---: | ---: | ---: |
| 15.3 | 585.8 | 25.7 | — |
| 29.3 | 908.6 | 31.2 | 0.675 |
| 59.2 | 1277.0 | 44.5 | 0.485 |
| 107.2 | 1473.3 | 67.9 | 0.241 |

Three things move at once: throughput rises **sublinearly** and decreasingly so,
TPOT degrades **faster than linearly**, and power saturates somewhere in between
(ATOM, measured: 19.44 W idle → 68.73 W at 95.1 % utilisation).

The consequence that matters:

> **tokens/J is unimodal in concurrency.** At low load idle power dominates the
> denominator; at high load throughput roll-off starves the numerator. So
> **which accelerator is more energy-efficient depends on the RPS**, a device
> with low idle power wins the low end, a device with a better scaling exponent
> wins the high end, and there is a crossover between them.

Finding and reporting that crossover is the output this system owes the user.
The planner today cannot represent it, because a scalar profile has no crossover.

There is a second reason, and it is the one that bit us: **the predictor's own
accuracy is a function of the operating point.** The RNGD card profile agrees to
−3.1 % at the ~16.6 concurrency it was fitted on, and is 1.31× optimistic on
throughput and 18 % on TPOT at 76. That is not absorbable into a calibration
offset, because it is not an offset — it varies along the same axis.

## 2. Data model — the performance envelope

Replace the scalar with a **measured curve**, promoted to a first-class artifact:

```
profiles/envelopes/<HARDWARE>/<org>/<model>/<variant>/tp<N>.yaml
```

```yaml
envelope_id: RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/tp1
unit: card                  # the accelerator unit: PE for RNGD, card for ATOM
measured_at: 2026-08-31
source: measured
harness: experiments/scripts/rebuild_rngd_bundle_from_edf.py collect
raw: outputs/rngd_envelope/edf/real_c{16,32,64,128}.json

# SERVED concurrency (Little's law: sum(latency)/wall), never the requested
# value. Conflating the two is what produced the D21 retraction: a 24-request
# pool made "c32" run at 21.2, and the exponent computed from it read a x1.74
# interval as a x2 doubling.
concurrency_metric: served

measured_on_workload:
  dataset: workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl
  input_tokens_p50: 731
  output_tokens_p50: 632

points:
  - {conc:  15.3, tput_tok_s:  585.8, tpot_p50: 25.71, tpot_p99: 29.46, ttft_p99:  1925, util_pct: null, power_w: null}
  - {conc:  29.3, tput_tok_s:  908.6, tpot_p50: 31.18, tpot_p99: 35.52, ttft_p99:  3849, util_pct: null, power_w: null}
  - {conc:  59.2, tput_tok_s: 1277.0, tpot_p50: 44.54, tpot_p99: 47.99, ttft_p99:  6901, util_pct: null, power_w: null}
  - {conc: 107.2, tput_tok_s: 1473.3, tpot_p50: 67.88, tpot_p99: 78.22, ttft_p99: 14010, util_pct: null, power_w: null}

validity:
  conc_min: 15.3
  conc_max: 107.2
  extrapolation: refuse            # refuse | widen_error_bars
  pool_binding_above: 107.2        # the harness pool bound the top point

closed_loop: true
notes: >
  ttft_* is closed-loop (the whole request pool is fired at once, deviations
  D19) and must NOT be compared with an open-loop Poisson p99. Throughput and
  TPOT are the transferable axes; TTFT is not.
```

Four decisions, each paid for by a mistake already made:

- **`concurrency_metric: served`** is mandatory and the loader rejects an
  envelope without it. The requested/served conflation is exactly what D21
  retracts.
- **`validity` is mandatory, and `refuse` is the default.** Silent extrapolation
  past the measured range is the root cause of the whole D21 episode.
- **`closed_loop`** records which axes transfer. Different measurement regimes
  make different columns comparable, and the data must carry that rather than
  relying on a reader remembering it.
- **`measured_on_workload`** because the curve is conditional on the token
  distribution — see §10.

## 3. Power as a function of the operating point

Replace the scalar `active_power`:

```yaml
power_model:
  kind: piecewise_linear_in_util
  idle_w: 19.44                       # measured after a 45 s settle
  points:
    - {util_pct: 36.2, power_w: 44.3}
    - {util_pct: 64.4, power_w: 54.9}
    - {util_pct: 95.1, power_w: 68.7}
  saturation_knee: null               # RNGD: the 8th PE adds only +17.4 W
  additive_across_units: true         # ATOM: a neighbour moves <0.1 W
  quantisation_w: 0.000001            # rbln-smi is uW; furiosa-smi is 1 W
  source: measured
```

`additive_across_units` is load-bearing: ATOM's per-card power is additive
(measured — an unloaded card never moved more than 0.1 W while a neighbour was
saturated), while RNGD's card carries a fixed ~38 W shared by 8 PEs and is not.
Summing the wrong way misprices every multi-device plan.

The `util_pct` column exists because of the ATOM measurement: a load leaving the
card at 36 % utilisation reads 44.3 W and a saturating one reads 68.7 W. **A
power figure without the utilisation it was taken at is not a measurement**, and
the schema should make that impossible to omit.

## 4. RPS → operating point solver

New module `planner/perf_envelope.py`. (Note: `planner/envelope.py` already
exists and is the *simulation result cache* — do not overload the name.)

The operating point is the fixed point of Little's law:

```
L = lambda * W(L)
```

with `lambda` the per-instance arrival rate (RPS / replicas) and
`W(L) = TTFT(L) + output_tokens * TPOT(L)` read off the envelope.
`W` increases in `L`, so the fixed point is unique where it exists.

```python
def solve_operating_point(env: Envelope, rps_per_instance: float,
                          out_tokens: float) -> OperatingPoint | Saturated:
    """Little's law fixed point on the measured envelope.

    Saturated is a RESULT, not an error: it means this instance cannot serve
    this arrival rate, which the caller answers by adding replicas. It is
    returned in two distinct cases, and they must not be collapsed:

      * no L inside [conc_min, conc_max] satisfies L = lambda*W(L)
        -> genuine saturation
      * the fixed point lies beyond conc_max
        -> UNMEASURED. With extrapolation: refuse this becomes a rejection
           reason ("outside measured envelope"), never a silent guess.
    """
```

Returning `OperatingPoint | Saturated` rather than raising matches the planner's
existing stance that *infeasible is a diagnosis, not an error*.

## 5. The predictor declares its own accuracy domain

The most transferable lesson from D21. Extend `profiles/calibration/<hw>.yaml`:

```yaml
accuracy_domain:
  fitted_at_concurrency: 16.6
  points:
    - {conc: 16.6, tput_err_pct:  -3.1, tpot_err_pct:  -3.1}
    - {conc: 76.0, tput_err_pct: +31.0, tpot_err_pct: -18.0}
  outside_domain: widen_error_bars
```

Feasibility then derives the safety margin from the candidate's own operating
point instead of taking it as a hand-set constant:

```python
margin = calibration.tpot_error_at(op.concurrency)
robust_tpot = predicted_tpot * (1 + margin / 100.0)
```

`experiments/scripts/pd_slo_sweep.py` now exposes `--tpot-margin-percent` as a
manual stand-in for this. Automating it closes the hole structurally: the card
fixture's winner `hp-00323` passes the 50 ms TPOT SLO at a predicted 48.41 ms,
but 48.41 × 1.18 = 57.1 ms. **The winner is not merely optimistic — it is
infeasible**, and nothing in the pipeline could see that.

## 6. RPS as a search axis

Candidates are currently
`(island, tp, dp, max_num_seqs, max_num_batched_tokens)` with RPS fixed in the
`ServiceSpec`. Promote RPS to a swept axis:

```bash
python -m planner plan \
    --service examples/service_specs/llama31-8b.yaml \
    --cluster examples/clusters/heterogeneous-lab.yaml \
    --rps 1,3,10,30,100 \
    --output outputs/plans/rps-sweep.yaml
```

The output is not one plan but a **switchover table** — illustrative shape, not
measured values:

| RPS | recommended | backend | acc | operating point | tok/J | avg W | validity |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `agg[rbln:tp1]` | ATOM | 1 | 4.2 | 6.81 | 24 W | measured |
| 3 | `agg[rbln:tp1]` | ATOM | 2 | 11.6 | 6.44 | 51 W | measured |
| 10 | `agg[furiosa:tp8]` | RNGD | 2 | 29.3 | 4.96 | 481 W | measured |
| 30 | `agg[cuda:tp4]` | A40 | 4 | 62.1 | 2.96 | 1283 W | measured |
| 100 | — | — | — | — | — | — | **saturated / unmeasured** |

plus the crossovers computed explicitly, since they are the actual deliverable:

```
crossover: rbln -> furiosa at 6.4 rps   (tok/J equal at 5.71)
crossover: furiosa -> cuda  at 18.2 rps (RNGD saturates its measured envelope)
```

## 7. Where each piece goes

| component | location | note |
| --- | --- | --- |
| envelope schema + loader | `planner/perf_envelope.py` (new) | pydantic; `validity` and `concurrency_metric` required |
| Little's law solver | `planner/perf_envelope.py` | `solve_operating_point()` |
| power curve | `planner/inventory.py`, `AcceleratorProfile.power_model` | extends the existing `power:` block, back-compatible |
| accuracy domain | `planner/predictor/calibration.py` | new `accuracy_domain` block |
| operating-point margin | `planner/optimizer/feasibility.py` | automates today's manual `tpot_margin_percent` |
| `--rps` sweep | `planner/__main__.py` | extends `plan` |
| switchover + crossovers | `planner/optimizer/pareto.py` | new `switchover` field on `PlannerOutput` |
| envelope measurement harness | `experiments/scripts/measure_envelope.py` (new) | vendor-agnostic core shared like `llama_layers.py` |

Two constraints from the existing architecture:

- **A pruning stage must remain a relaxation.** An envelope-based rejection is
  *epistemic* ("outside what we measured"), not a §5.6 constraint violation, so
  it needs its own category in `rejected_summary` or the oracle-agreement test
  will correctly fail.
- **`exhaustive.py` stays the oracle.** The optimum with envelope pruning enabled
  must equal the optimum with it disabled.

## 8. What must be measured to populate this

Per (device, model, TP) combination:

1. **A concurrency sweep** — at least 4 points, doubling, with the request pool
   at ≥ 4× the concurrency. Verify non-binding by checking `served/requested ≥ 0.9`.
2. **At every point**: throughput, TTFT/TPOT p50·p95·p99, and **utilisation and
   power recorded together**.
3. **Far enough to see the roll-off**, or `conc_max` means nothing.

Current coverage:

| device | envelope | gap |
| --- | --- | --- |
| RNGD-CARD | **measured**, 4 points, eff 15–107 | power not recorded during the sweep |
| A40 | partial | committed profile, no concurrency curve |
| ATOM | **impossible today** | no perf bundle at all (D20) |
| A5000 | partial | `docs/nodes/a5000.md` is thin and says so |

So **ATOM cannot enter this system until D20 is resolved** — which needs the
`rebel._C.profiler` trace schema from Rebellions. And the RNGD envelope needs a
re-run with a power sampler attached before its `power_w` column can be filled.

## 9. Tests

Beyond the repo's three mandatory classes:

- **Envelope round-trip** — measured JSON → envelope YAML → solver → reproduces
  the measured points to < 1 % interpolation error.
- **Extrapolation refusal** — an RPS whose fixed point lies past `conc_max`
  returns `Saturated`/rejected, never a silent number.
- **Monotonicity** — the solver returns non-decreasing concurrency for
  increasing RPS; a non-monotone answer is a fixed-point bug.
- **Crossover stability** — the crossover RPS must not be sensitive to envelope
  point count, quantified the way D11 quantified profile-grid density (2.2pp).
- **Oracle agreement** — envelope pruning on/off give the same optimum.

## 10. Risks

**The envelope is conditional on the workload.** Every number above came from one
model, one dataset (sharegpt, input p50 731 / output p50 632) and one card. A
prefill-heavy workload moves the TTFT axis; a long-output one moves TPOT. The
schema records `measured_on_workload` for this reason, and the loader should
refuse to mix envelopes measured on different workloads in one comparison.
Mixing them silently is the same class of error as D21.

**The served/requested discipline is fragile and load-bearing.** Confusing them
corrupts the exponent, and the exponent is what every extrapolation rides on. It
produced 0.598 where the truth was 0.675, and "~1.6× optimistic" where the truth
was 1.31×. The schema makes it explicit; reviews should treat any envelope
without served concurrency as unusable.

**Crossovers are only as good as the envelopes on both sides.** A crossover
between a measured device and a placeholder one is not a result. The switchover
table must carry the `validity` column, and a row whose winner rests on an
unmeasured envelope must say so rather than print a number.
