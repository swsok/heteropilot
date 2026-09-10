# `plan --rps`: what the switchover table looks like, and how to read it

*`WORK_ORDER_rps_aware.md` rev 2 STEP 4.5. Output below is from a real run on the
NPU node, 2026-09-09, 20 requests per rate. It is an illustration of the tool, not
a result: 20 requests is a smoke-sized workload and the fixture's NPU profile is
the card bundle.*

## The command

```bash
python -m planner plan \
  --service examples/service_specs/llama31-8b.yaml \
  --cluster experiments/configs/clusters/pd-rngd-gpu-card.yaml \
  --num-requests 20 --seed 42 --workers 24 \
  --accuracy-domain \
  --rps 1,3.3,10 \
  --cache-dir outputs/.hp-rps --work-dir outputs/rps/work \
  --output outputs/rps/plan.yaml
```

One plan per rate, then the table. `--output plan.yaml` writes
`plan_rps1p0.yaml`, `plan_rps3p3.yaml`, `plan_rps10p0.yaml` and
`plan_switchover.yaml`.

**The cache is shared on purpose.** Its key includes the trace digest, and RPS
changes the arrival times, so entries separate by rate on their own —
`planner/envelope.py:key_for` has no RPS field and needs none. Sharing it means a
re-run of one rate does not re-simulate the others.

## The output

```
--- RPS switchover ------------------------------------------------------------
service : meta-llama/Llama-3.1-8B
cluster : pd-rngd-gpu-card

    rps  recommended                                  backend       acc    tok/J    avg W  p99 TPOT  margin  validity
    1.0  furiosa-rngd-card-node_rngd0-tp1-dp1-s128-t2 furiosa         1     0.73    548.4      26.3    0.0%  measured
    3.3  furiosa-rngd-card-node_rngd0-tp1-dp1-s128-t2 furiosa         1     0.98    554.5      27.6    1.4%  measured
   10.0  furiosa-rngd-card-node_rngd0-tp1-dp1-s128-t8 furiosa         1    1.048    558.5     28.39    2.6%  measured

--- Crossovers ----------------------------------------------------------------
  none: the same backend wins at every rate swept.
  - [rps 1.0] 306 candidate(s) put A40 at served concurrency 1.78-12.18, outside
    its measured accuracy domain; those margins are EXTRAPOLATED, not measured
  ...
```

## How to read it

**`margin` is not a constant, and that is the point.** It rises 0.0 → 1.4 → 2.6 %
because the winner's own operating point rises with the arrival rate, and the
simulator's measured error rises with it — the RNGD card's accuracy domain runs
from +11.6 % (pessimistic) at served concurrency 3.9 to −18 % (optimistic) at 76.
A hand-set `--tpot-margin-percent` would have been one of those numbers everywhere.
Both can be given; the larger wins and provenance records both.

**`validity` is the column to check first.** `measured` means every operating
point in that row sat inside its hardware's measured accuracy domain.
`extrapolated` means at least one did not and its margin was widened by distance
rather than measured. `unknown` means no domain applied at all. A switchover table
read without this column is the D22 mistake in miniature.

**"no crossover" is a result.** It says the same backend wins at every rate
*swept*, not that no crossover exists — a rate outside `--rps` could still flip it.

**A crossover, when there is one, is `estimated`.** Nothing is run at the crossing
point; it is interpolated linearly between the two bracketing rates that were run,
on curves that are not linear. The label travels with the number everywhere it
appears, and when either bracketing row lacks a tok/J the crossing is reported
without a rate rather than with a fabricated midpoint.

**The caveats are per rate where they differ.** Three lines each saying
"306 candidate(s)" are three rates, not 918 candidates, so rate-specific caveats
carry an `[rps N]` tag.

## Verified with `--enable-pd` too

`WORK_ORDER_rps_aware.md` §7 asks that
`plan --rps … --accuracy-domain --enable-pd` emit the switchover table and the
crossovers or their absence. It does — run 2026-09-10 on `pd-rngd-gpu-card`,
300 requests, rates 10 and 20 against the E6 cache:

```
    rps  recommended                                  backend  acc   tok/J   avg W  p99 TPOT  margin  validity
   10.0  cuda-a40-node_a40a-tp4-dp1-s128-t8192        cuda       4   2.595  1292.5     35.58   1.42%  extrapolated
   20.0  pd(cuda-a40-node_a40a-tp2-dp1 P + cuda-a40…) cuda       6   2.481  1371.9     48.52   1.42%  extrapolated

--- Crossovers ---
  none: the same backend wins at every rate swept.
```

**"none" here and a crossover in E6 are not a contradiction.** This run uses the
service spec's own TTFT SLO (25 s) and its full candidate set, where E6 swept
64 s and 8 s over a knob-fixed set — and it covers only 10 and 20 rps, both on
the same side of E6's crossover. The switchover table is conditional on the SLO
and on which candidates were asked about, which is why both travel with it.

## The two opt-in stages

Both default off, so the plain `plan` path is unchanged.

| flag | what it does | what it can cost |
| --- | --- | --- |
| `--accuracy-domain` | sizes each candidate's SLO margin from its own operating point (`deviations.md` D29) | nothing; margin 0 where no domain exists |
| `--envelope-prefilter` | rejects, **before simulating**, candidates whose predicted operating point is outside the hardware's measured envelope | **it can drop the optimum** |

The second needs care. On `pd-rngd-gpu-card` at 10 rps it rejects 114 of 324
candidates and **changes the recommendation** from an RNGD card to an A40 — because
one RNGD card at that rate would have to serve 6320 tok/s and the measured envelope
tops out at 1473. That is not "the RNGD is infeasible"; it is "nobody has measured
an RNGD card at four times the load we measured it at". The caveat says so, and the
honest response is to add replicas or to measure further, not to delete the flag.
