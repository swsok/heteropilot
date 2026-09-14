# E-B3 — what the closed form costs, measured against real simulation

**What this is.** B1's design bet is that a measurement plan can be built by
post-processing cached predictions instead of re-running the simulator. Two of
the perturbation rules are exact; the rest are first-order stand-ins. E-B3 runs
the plan both ways and prices the bet.

> **Run 2026-09-14 on the NPU node**, on the reconciled architecture (PRs
> #78–#80, D33) and on the same truth as E-B1/E-B2 — the nine-point D32
> RNGD-CARD domain. Fixture `pd-rngd-gpu-card`, incumbent
> `cuda-a40-node_a40a-tp4-dp1-s128-t8192`. Artifacts:
> `outputs/uncertainty/eb3/eb3_closed_form_vs_resim.json`.

## The answer

| | |
| --- | ---: |
| closed form, five items | **0.287 s** |
| resimulated, four items | **2,090.8 s** (1,104 simulator runs) |
| speed-up | **7,285×** |
| Spearman, full order | 1.000 |
| Spearman, the four checked items | 1.000 |

Per item — `closed` is `analyze`'s ΔR on the two-point grid, `resim` the same
arithmetic on metrics the simulator actually produced at both endpoints:

| input | closed | resim | runs | time |
| --- | ---: | ---: | ---: | ---: |
| `profile:cuda-a40-node_a40a` | 14,298.5 | **20,583.3** | 432 | 835 s |
| `profile:cuda-a40-node_a40b` | 0.0 | 0.0 | 432 | 836 s |
| `profile:furiosa-rngd-card-node_rngd0` | 0.0 | 0.0 | 120 | 207 s |
| `profile:furiosa-rngd-card-node_rngd1` | 0.0 | 0.0 | 120 | 212 s |
| `sim_error:domain` | 5,686.3 | — | — | SKIPPED |

**The ordering survives; the magnitude does not.** The closed form puts the same
item first and keeps the three inert ones at exactly zero, so as a *ranking* —
which is all a measurement plan reads — it is right here. But on the one item
where there was anything to get wrong it reports **14,298 against a true 20,583,
a 30.5 % underestimate**. A plan that quotes ΔR as an expected saving is quoting
a number that is low by a third.

## Why the Spearman is the weakest thing in this table

**1.000 over four items, three of which are tied at zero.** A rank correlation
on that shape is close to vacuous: it says the closed form agreed that three
inert inputs are inert and that the one live input is live. The real content of
this experiment is the single pair `14,298 → 20,583` and the three exact
agreements, not the correlation computed from them.

**The `--top 1` run is worse still and must not be quoted.** It also reported
Spearman 1.000, over five items of which **one** was refined — the other four
kept their closed-form values, so the two orderings were identical by
construction. The same file's `spearman_delta_regret_on_checked` came back
`None`, which is the harness saying it refused to compute a correlation from one
point. That run is kept as `eb3_top1_tautological_spearman.json` beside the
superseded pair and is evidence of nothing.

**`sim_error` is SKIPPED, not checked.** A SIM_ERROR moves the margin rather
than the prediction, so no simulator input changes and its closed form is exact
by construction. The harness reports it as skipped rather than counting an
agreement: "we could not check this" and "we checked it and it agreed" are
different claims. It is also the item with the second-largest ΔR, so the
unchecked fraction of the ranking is not small.

## The identity control caught D40 again, and localised it

Every refinement runs a ×1.0 endpoint first: perturbing a profile by a factor of
one must reproduce the cached prediction exactly.

| control | worst relative deviation | at |
| --- | ---: | --- |
| `profile:cuda-a40-node_a40a` | **7.869e-02** | `mix(a40a-tp2-dp2+a40b-tp2-dp1)-s128-t2048.p99_ttft_ms` |
| `profile:cuda-a40-node_a40b` | **7.869e-02** | the same candidate |
| `profile:furiosa-rngd-card-node_rngd0` | 0.000e+00 | — |
| `profile:furiosa-rngd-card-node_rngd1` | 0.000e+00 | — |

The two RNGD controls are exact to the last digit over 120 runs each. Both A40
controls deviate, by the same amount, at **the same candidate** — the mirrored
A40 placement `docs/deviations.md` **D40** names. That is an independent
confirmation of D40 and it narrows it: the defect is not diffuse numerical noise,
it is one shared cache entry serving two placements that the simulator prices
differently.

It also means **the 20,583 figure carries that defect**: `a40a`'s resimulated ΔR
is computed over a candidate set in which at least one mixed A40 placement reads
its mirror's TTFT. The 30.5 % gap is therefore an estimate of the closed form's
error *plus* whatever D40 contributes, and the two are not separated here.
Separating them needs the cache re-keyed, which D40 explains is a decision about
what to re-run rather than a patch.

## What this does and does not license

**Does:** using the closed form to decide *what to measure next*. On this fixture
it picks the same input first and correctly identifies three inputs as not worth
measuring, for 0.287 s against 34.8 minutes.

**Does not:** quoting ΔR as an amount of energy saved. It was low by 30.5 % on the
only item that tested it.

**Does not, either:** generalising from four items on one fixture, three of them
inert. The work order asks for a Spearman and gets one; what it is worth is
stated above.

## Reproduce

```bash
PYTHONPATH=$PWD:experiments/uncertainty .venv/bin/python \
    experiments/uncertainty/eb3_closed_form_vs_resim.py \
    --cache-dir outputs/uncertainty/ea1/cache --top 4 --workers 32
```

`--top 4` rather than the work order's `--top 3`: four is every refinable item in
this pool, so the run is complete rather than truncated at an arbitrary depth.
`sim_error` is skipped and does not consume a slot.
