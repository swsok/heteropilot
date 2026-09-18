# E-A1 under S1's application-condition refusal — rule (e)

**What this is.** `WORK_ORDER_domain_scoping.md` STEP S4. S1 (D110) made an
accuracy domain answer only for the configuration it was measured under. This
is what that costs on E-A1's own fixture, and it is the ending of the V3 case
the disclosure's §2 and §5.2 are mapped to.

Generated tables: `outputs/uncertainty/s4/ea1_s4_stated_table.md`; raw record
`outputs/uncertainty/s4/ea1_s4_stated.json`, and the intermediate arm
`outputs/uncertainty/s4/ea1_s4.json`. **The committed E-A1 record is not
regenerated** — `ea1_margin_modes.md`, its table and its JSON stay as published,
and rules (a)–(d) below are a reproduction of them, not a replacement.

Same fixture, cache and seed as E-A1: `pd-rngd-gpu-card.yaml` +
`examples/service_specs/llama31-8b.yaml`, 300 requests, seed 42, TTFT 25 000 ms
p99 / TPOT 50 ms p99, 324 candidates, warm cache, nothing simulated.

---

## 0. The thing to know before reading any of this

**On `main` today, the committed E-A1 script does not reproduce the committed
E-A1 numbers.** S1 ships `condition_mismatch="refuse"` as the planner's default,
`AccuracyDomainMargin` applies it, and every committed domain is fitted at
`tp=1`, `dp=1`, one island. Run as-is, conditions (c) and (d) both return **0
feasible** with 276 candidates held, and 50/244/30 cannot be got back at all.

That is D110 working, and it is why rules (a)–(d) are now built with
`condition_mismatch="warn"` — a setting that did not exist when they were
written. `warn` applies a domain that was measured elsewhere, under protest, and
records the mismatch. It is what keeps (a)–(d) a comparison of **margin rules**
rather than a re-run of S1, and with it they reproduce exactly:

| condition | feasible | rejected | committed E-A1 |
| --- | ---: | --- | --- |
| (a) no margin | 70 | slo_violated 254 | ✔ identical |
| (b) global 18 % | 10 | slo_violated 314 | ✔ identical |
| (c) per-point, `widen_error_bars` | 50 | slo_violated 274 | ✔ identical |
| (d) per-point, `outside_domain: refuse` | 50 | slo_violated 244, outside_calibration_domain 30 | ✔ **50/244/30** |

---

## 1. Rule (e): the answer

(e) is (d) plus S1 — the same per-operating-point margin, refusing both an
operating point outside the measured range **and** a configuration no domain
was measured under.

| | (d) | **(e)** |
| --- | ---: | ---: |
| feasible | 50 | **0** |
| slo_violated | 244 | 6 |
| outside_calibration_domain | 30 | 6 |
| **calibration_condition_mismatch** | — | **312** |
| recommended plan | `cuda-a40-node_a40a-tp4-dp1-s128-t8192` | **none** |

**There is no recommendation.** Not "no feasible plan": 312 of 324 candidates
are not judged at all, and the 12 that are judged fail.

The planner says so in its own words, and names the measurements:

> 318 candidate(s) could not be judged: no margin applies to them, so they are
> undecidable, not infeasible.
> 312 of them differ from every accuracy domain on a REQUIRED APPLICATION
> CONDITION (arrival_process (42), dp (228), islands (132), tp (66)), so no
> domain may be consulted for them at all… What would settle them is a
> measurement AT THEIR CONFIGURATION, not a wider load range.
>   measure at: {"binding": "unknown", "dp": 1, "hardware": "A40", "islands": 1, "pp": 1, "tp": 2}
>   measure at: {"binding": "unknown", "dp": 1, "hardware": "A40", "islands": 1, "pp": 1, "tp": 4}
>   measure at: {"binding": "unknown", "dp": 1, "hardware": "A40", "islands": 2, "pp": 1, "tp": 1}
>   … and 5 further configuration(s)

`binding: unknown` is not a gap in the output — it is what this fixture's nodes
say, and S3 left `pd-rngd-gpu-card.yaml` unstated on purpose. It does mean the
measurement request is under-specified on the axis worth 1.93× of throughput
here, which is one more reason S6(i) should record its binding.

Alongside these the run also emits the ordinary SLO-relaxation suggestions,
which are about the wrong thing for a run that judged nothing — a reporting
wrinkle for S5, not a wrong answer.

## 2. Why each candidate was held, and where the work order's hypothesis was wrong

The work order predicted that *"A40 `tp>=2` or multi-island candidates are all
held"*. They are — but they are a minority of the holds.

| fields that disagreed | held |
| --- | ---: |
| `dp` alone | 108 |
| `dp` + `islands` | 90 |
| `arrival_process` alone | 36 |
| `tp` alone | 24 |
| `dp` + `islands` + `tp` | 18 |
| `dp` + `tp` | 12 |
| `islands` + `tp` | 12 |
| `islands` alone | 6 |
| `arrival_process` + `islands` | 6 |
| **total** | **312** |

Counted per field instead — a candidate counts once for each field it
disagrees on — that is the tally the planner prints itself:
**`dp` 228, `islands` 132, `tp` 66, `arrival_process` 42.**

**`tp` accounts for 66 of 312.** The single largest cause is **`dp`** — 228
candidates disagree on it — because every committed domain is `dp: 1` and this
fixture generates DP replicas freely. `islands` adds another 132. Neither was
named in the hypothesis, and both are the same kind of gap as `tp`: a domain
measured on one replica of one island does not answer for two.

This matters beyond bookkeeping. The disclosure's story is "the parallelism
degree a domain was fitted at is a missing scoping axis". The measurement says
the axis is wider than parallelism: **replication and placement are refused more
often than parallelism is**, and a measurement programme aimed only at tp would
close less than a quarter of this.

## 3. The two fields S1 left unstated, and what stating them did

S1 deliberately left `arrival_process` and `model`/`variant` unstated; S4 owns
them. **`arrival_process` is now stated on all three, `variant` on the two whose
provenance records a dtype, and `model` on none** — each from the file's own
evidence, none inferred:

| domain | `arrival_process` | sourced from | `variant` |
| --- | --- | --- | --- |
| `a40.accuracy.yaml` | `open_loop` | its own header: "OPEN-LOOP, deliberately", `measure_envelope_openloop.py` + `python -m bench run` | `bf16` |
| `rngd_card_edf.yaml` | `closed_loop` | its own `why_unresolved`: "a burst against a closed-loop client… `bench_furiosa_endpoint.py` IGNORES `arrival_time_ns`" (D19) | — |
| `rngd_perpe.yaml` | `closed_loop` | its measured half is the RNGD-CARD envelope, whose file states `closed_loop: true` | `bf16` |

**`model` is withheld deliberately, and that is a finding rather than an
omission.** This repository holds **two strings for one set of weights**:
`meta-llama/Llama-3.1-8B` in the envelope paths, and
`NousResearch/Meta-Llama-3.1-8B` in the open-loop A40 domain's own
`provenance.deployment`. `check_conditions` compares strings, so stating either
would refuse a run on the **mirror name** rather than on a model difference — a
different wrong answer, not a safer one. Stating it needs an alias policy that
does not exist.

**And the measurement says nothing is bought by guessing it.** With `model` and
`variant` stated on all three the counts were **identical** — 312 / 6 / 6 — and
no candidate mismatched on either field. `rngd_card_edf.yaml` gets no `variant`
either: its one `fitted_from` artifact
(`outputs/rngd_bench/rngd-card-edf-burst_summary.txt`) is a table of TTFT and
TPOT percentiles that records no model and no dtype, and the bucket label
`sharegpt-llama31-8b-20` names a family, not a precision.

`tests/test_domain_conditions_stated.py` pins all of this, including the
control that the A40 domain must *not* refuse an open-loop candidate.

**`arrival_process` moved 36 candidates**, and the control is that it moved
exactly the right ones:

| | before stating | after |
| --- | ---: | ---: |
| calibration_condition_mismatch | 276 | **312** |
| outside_calibration_domain | 26 | 6 |
| slo_violated | 22 | 6 |
| feasible | 0 | 0 |

42 candidates disagree on `arrival_process` in the end — 12 single-card RNGD
and 30 A40+RNGD mixes — but 6 of those were already held on `islands`, so the
field **moves** 36. **Every one of the 42 is RNGD-touching, and no A40-only
candidate moved at all**, because the A40 domain is open-loop and so is `plan`.
A field that refused everything would prove nothing; this one refuses the
closed-loop half of the cluster and leaves the open-loop half where it was.

**The 12 single-card RNGD candidates are the case worth stating.** They match
every other condition — the card domain is `tp=1, dp=1, islands=1` and so are
they — and before this they were held as `outside_calibration_domain` at served
concurrency 144.6 against a measured [1.02, 76]. Now they are held as
`calibration_condition_mismatch`. **The verdict is the same and the measurement
it asks for is not**: an envelope point above c=76 no longer answers them; an
open-loop refit does. That is the disclosure's "additional measurement
condition" output changing because the condition changed.

**The 12 that survive the condition check** are the A40 `tp1-dp1` single-island
candidates on both nodes, and they are still not feasible: 6 outside the domain,
6 `slo_violated`.

## 4. The limit of this refusal, stated rather than fixed

A condition mismatch is **whole-domain**: `arrival_process` disagreeing withdraws
the TTFT *and* the TPOT error together. D19's own evidence is narrower than
that — a burst against a spread arrival process differs and *"the difference
lands entirely in TTFT"* — so a closed-loop domain's TPOT error may well survive
into an open-loop deployment, and refusing it is conservative rather than
correct.

Per-metric condition scoping would express that. It is not built, it is not in
this work order, and the counts above are what the whole-domain rule gives. A
reader deciding how much to spend on an open-loop RNGD refit should know that
the TPOT half of what it would buy may already be in hand.

## 5. Reproducing

```bash
PYTHONPATH=$PWD .venv/bin/python experiments/uncertainty/ea1_margin_modes.py \
    --cache-dir outputs/uncertainty/ea1/cache \
    --work-dir outputs/uncertainty/s4/work \
    --out outputs/uncertainty/s4/ea1_s4_stated_table.md \
    --json-out outputs/uncertainty/s4/ea1_s4_stated.json
```

Warm cache, no simulation. **Do not point `--out`/`--json-out` at
`experiments/uncertainty/results/ea1_margin_modes_table.md` or
`outputs/uncertainty/ea1/ea1_margin_modes.json`**: (a)–(d) would land on them
unchanged and rule (e) would be appended to a record whose own text describes
four rules.

To see the arm before `arrival_process` was stated, check the three domain
files out at `23ff68c` and write to `ea1_s4.json`.
