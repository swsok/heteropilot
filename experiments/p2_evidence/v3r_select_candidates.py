#!/usr/bin/env python
"""V3-R: find the single-card RNGD candidates whose measurement could decide a rule.

`WORK_ORDER_domain_scoping.md` STEP S7.0, rev 2. **This selects; it does not
measure.** The work order requires the table to be confirmed before the card is
touched, which is the rule V3's own CPU half followed.

WHY A SELECTION IS NEEDED AT ALL, when V3 already has P2 and P3. Both of V3's
are `mix(cuda-a40-...)` candidates and there is no NVIDIA driver on this node, so
neither can be run here (absolute rule 3). P2 and P3 are therefore re-read as
VERDICT shapes rather than topology shapes:

    P2-shape   (a) no margin passes  and  (c) per-point rejects
               -- the optimistic candidate the invention claims to CATCH
    P3-shape   (b) global 18 % rejects and (c) per-point passes
               -- the candidate it claims to RESCUE

On the A40 the (a)/(c) disagreement band is at most 0.80 ms wide and the best
candidate decided by 0.3 ms, below any repeatability (v3_candidate_selection.md
§1). On RNGD the same band is 2.3 ms at served concurrency 30 and 7.6 ms at 66,
because the margin there runs to ~21 % instead of ~1 %. That width is the whole
reason S7 exists.

WHAT THE SWEEP IS FOR. The fixture's own 10 rps puts every single-card RNGD
candidate at served concurrency 140-178, far above the domain's measured [1.02,
76], so all twelve are held before any margin is computed. `--rps` moves the
operating point; this script sweeps it and reports where each candidate lands,
which margin its own operating point earns, and what each of the five rules then
says.

CANDIDATE SCOPE, and both halves are required. Only `tp=1, pp=1, dp=1`,
single-island RNGD candidates are swept, because (i) the card domain was fitted
at that configuration, so anything else mismatches on `dp`/`islands` too and
hides what an arrival-process refusal did, and (ii) there is no furiosa deploy
backend, so a measurement is driven by hand at one engine and a multi-island
candidate would need the router Phase 4 does not have.

`node_rngd1`'s six candidates are the cache mirrors of `node_rngd0`'s: identical
card, identical profile, so `EnvelopeCache` serves one entry to both. D40 is the
reason that cannot be assumed - it is sound for single-island aggregated
candidates and unsound for mirrored `mix(...)` ones - so `--verify-mirror`
simulates one `node_rngd1` candidate with the cache disabled and compares, the
same control V3 §2.1 ran on its own two.

THE RULES ARE NOT REDEFINED HERE. `_conditions` is imported from
`experiments/uncertainty/ea1_margin_modes.py`, so (a)-(e) are literally E-A1's
and this experiment cannot drift from the one whose numbers the disclosure
quotes. That also means (a)-(d) carry `condition_mismatch="warn"` and (e)
carries `refuse` - see that module's docstring for why.

Run (about 40 s per candidate-rate, so ~6 min for the default grid at 6 workers)::

    PYTHONPATH=$PWD .venv/bin/python experiments/p2_evidence/v3r_select_candidates.py \
        --rps 0.5,1,1.5,2,3,4,6,8,10 \
        --out experiments/p2_evidence/results/v3r_candidates.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/uncertainty"))

from ea1_margin_modes import (  # noqa: E402
    _conditions,
    _IslandOperatingPoint,
    _verdicts,
)

from planner.candidate_generator import CandidateGenerator  # noqa: E402
from planner.envelope import workload_bucket, workload_shape  # noqa: E402
from planner.inventory import (  # noqa: E402
    detect_islands,
    load_cluster_spec,
    load_profiles_for,
)
from planner.optimizer import exhaustive  # noqa: E402
from planner.predictor.calibration import (  # noqa: E402
    load_accuracy_domains,
    load_calibrations,
    load_domain_index,
)
from planner.predictor.llmservingsim import LLMServingSimPredictor  # noqa: E402
from planner.spec import load_service_spec  # noqa: E402
from planner.topology import TopologyGraph  # noqa: E402
from planner.util import provenance as prov  # noqa: E402
from planner.util.tier import resolve_variant  # noqa: E402
from planner.util.workload import generate_trace  # noqa: E402

CLUSTER = "experiments/configs/clusters/pd-rngd-gpu-card.yaml"
SERVICE = "examples/service_specs/llama31-8b.yaml"
NUM_REQUESTS = 300
SEED = 42

#: E-A1's rule order, with S4's (e) last.
RULES = ("a_margin0", "b_global18", "c_accuracy_domain", "d_refuse",
         "e_condition_refuse")

#: The island whose six knob settings are swept. `node_rngd1` mirrors it.
SWEPT_ISLAND = "furiosa-rngd-card-node_rngd0"
MIRROR_ISLAND = "furiosa-rngd-card-node_rngd1"

#: The shape windows, from WORK_ORDER_domain_scoping.md S7.B. A P3-shape needs
#: the candidate's own margin BELOW the global 18 %, which the committed domain
#: reaches at served concurrency 66.52; above that the global rule is the looser
#: of the two and cannot over-reject. A P2-shape only needs a margin at all,
#: i.e. an operating point above where the domain's error crosses zero (~15.5).
P3_CONC_MAX = 66.52
P2_CONC_MIN = 15.5


def single_card_candidates(candidates: list) -> list:
    """`tp=1, pp=1, dp=1` single-island candidates on the swept RNGD island."""
    out = []
    for c in candidates:
        if len(c.assignments) != 1:
            continue
        a = c.assignments[0]
        if a.island_id != SWEPT_ISLAND:
            continue
        if (a.tp_size, a.pp_size, a.dp_replicas) != (1, 1, 1):
            continue
        out.append(c)
    return out


def classify(verdicts: dict[str, str]) -> list[str]:
    """Which verdict shapes this candidate exhibits. It can be neither or both."""
    shapes = []
    if verdicts["a_margin0"] == "feasible" and verdicts["c_accuracy_domain"] == "rejected":
        shapes.append("P2-shape")
    if verdicts["b_global18"] == "rejected" and verdicts["c_accuracy_domain"] == "feasible":
        shapes.append("P3-shape")
    return shapes


def deciding_metric(reasons: dict[str, str]) -> str:
    """Which SLO the rejections turned on, over all rules that rejected.

    A shape is only testable if TPOT decides it: the margin under test is the
    TPOT margin, and a candidate rejected on TTFT would be measured for the
    wrong reason.
    """
    seen = set()
    for reason in reasons.values():
        if "p99_tpot_ms" in reason:
            seen.add("TPOT")
        if "p99_ttft_ms" in reason:
            seen.add("TTFT")
    return "+".join(sorted(seen))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rps", default="0.5,1,1.5,2,3,4,6,8,10",
                    help="comma-separated arrival rates to sweep")
    ap.add_argument("--cluster", default=CLUSTER)
    ap.add_argument("--service", default=SERVICE)
    ap.add_argument("--num-requests", type=int, default=NUM_REQUESTS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--work-dir", type=Path,
                    default=REPO / "outputs/p2_evidence/v3r/work")
    ap.add_argument("--cache-dir", type=Path,
                    default=REPO / "outputs/p2_evidence/v3r/cache")
    ap.add_argument("--out", type=Path,
                    default=REPO / "experiments/p2_evidence/results/v3r_candidates.json")
    ap.add_argument("--accuracy-domain", nargs="*", default=None, metavar="YAML",
                    help="judge against THESE domain files instead of whatever the "
                         "default non-recursive glob finds. S7.4 needs it: the "
                         "open-loop RNGD domain lives in openloop/ precisely so the "
                         "default path keeps resolving RNGD-CARD to the closed-loop "
                         "one (D102), so the two arms of the inversion are two runs "
                         "of this script differing only in this flag.")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--verify-mirror", action="store_true",
                    help="re-simulate one node_rngd1 candidate with the cache "
                         "disabled and compare, the D40 control V3 §2.1 ran")
    args = ap.parse_args()

    rates = [float(v) for v in args.rps.split(",") if v.strip()]
    base_spec = load_service_spec(args.service)
    cluster = load_cluster_spec(args.cluster)
    profiles = load_profiles_for(cluster, REPO)
    islands = detect_islands(cluster, profiles)
    by_id = {i.id: i for i in islands}
    domains = load_accuracy_domains(str(REPO), args.accuracy_domain or None)
    calibration = load_calibrations(str(REPO))
    if args.accuracy_domain:
        print("accuracy domains (explicit): "
              + ", ".join(f"{hw}[{d.arrival_process}, "
                          f"{d.compared_metric or 'basis not stated'}]"
                          for hw, d in sorted(domains.items())), flush=True)

    from planner.optimizer.exhaustive import _profile_tiers

    report: dict = {
        "fixture": {
            "cluster": args.cluster, "service": args.service,
            "num_requests": args.num_requests, "seed": args.seed,
            "base_arrival_rate_rps": base_spec.traffic.arrival_rate_rps,
        },
        "swept_island": SWEPT_ISLAND,
        "mirror_island": MIRROR_ISLAND,
        "rules": list(RULES),
        "accuracy_domain": [str(a) for a in (args.accuracy_domain or [])] or "default glob",
        "windows": {"p3_conc_max": P3_CONC_MAX, "p2_conc_min": P2_CONC_MIN},
        "rows": [],
        "provenance": prov.collect(),
    }

    for rate in rates:
        spec = base_spec.model_copy(deep=True)
        spec.traffic.arrival_rate_rps = rate
        tag = f"rps{rate:g}".replace(".", "p")
        work = args.work_dir / tag
        work.mkdir(parents=True, exist_ok=True)

        bucket, shape = workload_bucket(spec), workload_shape(spec)
        _tiers, island_hw, _w = _profile_tiers(spec, islands, profiles)
        scope = {
            "model": spec.model,
            "variant": resolve_variant(spec.service.dtype, spec.service.kv_cache_dtype),
            "arrival_process": "open_loop",
            "device_binding": {
                i.id: cluster.node(i.node_id).device_binding for i in islands
            },
            "index": load_domain_index(str(REPO)),
        }

        trace = generate_trace(spec, work / "workload.jsonl",
                               num_requests=args.num_requests, seed=args.seed)
        reduction = TopologyGraph(cluster).reduce_for_simulator(islands)
        cache = _IslandOperatingPoint(
            args.cache_dir, spec,
            accelerator_of={i.id: i.accelerator_model for i in islands},
            link_bw_gbps=reduction.link_bw_gbps,
            trace_digest=prov.hash_file(trace.path),
            island_hw=island_hw,
        )
        predictor = LLMServingSimPredictor(
            trace, work_dir=work / "sims", timeout_s=args.timeout,
        )
        swept = single_card_candidates(
            CandidateGenerator(spec, cluster, islands, profiles,
                               enable_prefix_caching=False).generate().candidates
        )
        print(f"=== rps {rate:g}: {len(swept)} single-card candidates", flush=True)

        per_rule: dict[str, dict] = {}
        for condition in _conditions(domains, calibration, bucket, shape, scope=scope):
            if condition.key not in RULES:
                continue
            evaluation = exhaustive.evaluate_candidates(
                swept, spec, cluster, by_id, profiles, predictor,
                cache=cache,
                ttft_margin_percent=condition.ttft_percent,
                tpot_margin_percent=condition.tpot_percent,
                margin_policy=condition.policy,
                island_hw=island_hw,
                max_workers=args.workers,
            )
            per_rule[condition.key] = _verdicts(evaluation)

        for candidate in swept:
            cid = candidate.id
            v = {r: per_rule[r][cid] for r in RULES}
            entry = cache.get(candidate)
            metrics = entry.metrics if entry is not None else None
            row = {
                "rps": rate,
                "candidate_id": cid,
                "max_num_seqs": candidate.knobs.max_num_seqs,
                "max_num_batched_tokens": candidate.knobs.max_num_batched_tokens,
                "served_concurrency": getattr(metrics, "served_concurrency", None),
                "p99_tpot_ms": getattr(metrics, "p99_tpot_ms", None),
                "p99_ttft_ms": getattr(metrics, "p99_ttft_ms", None),
                "tokens_per_joule": getattr(metrics, "tokens_per_joule", None),
                "operating_point": v["c_accuracy_domain"].get("concurrency"),
                "tpot_margin_pct": {r: v[r].get("tpot_percent") for r in RULES},
                "verdicts": {r: v[r]["verdict"] for r in RULES},
                "stages": {r: v[r]["stage"] for r in RULES},
                "reasons": {r: v[r].get("reason", "") for r in RULES},
                "basis": v["c_accuracy_domain"].get("basis", ""),
            }
            row["shapes"] = classify(row["verdicts"])
            row["deciding_metric"] = deciding_metric(row["reasons"])
            report["rows"].append(row)
            print(f"  {cid.split('-tp1-dp1-')[-1]:12} L="
                  f"{_fmt(row['served_concurrency'])} tpot="
                  f"{_fmt(row['p99_tpot_ms'])} ttft="
                  f"{_fmt(row['p99_ttft_ms'], 1)} "
                  f"{'/'.join(row['verdicts'][r][:4] for r in RULES)} "
                  f"{','.join(row['shapes']) or '-'}", flush=True)

    if args.verify_mirror:
        report["mirror_control"] = _verify_mirror(
            base_spec, cluster, islands, by_id, profiles, args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out} ({len(report['rows'])} rows)")
    return 0


def _fmt(value: float | None, digits: int = 3) -> str:
    return "     n/a" if value is None else f"{value:8.{digits}f}"


def _verify_mirror(base_spec, cluster, islands, by_id, profiles, args) -> dict:
    """D40 control: does node_rngd1 really get node_rngd0's numbers, and rightly?

    The two islands are one card model on one profile, so the cache key collides
    by design and dedup is the sound case D40 identifies. Sound is not the same
    as verified, so this re-simulates the mirror with NO cache and compares.
    """
    rate = float(args.rps.split(",")[0])
    spec = base_spec.model_copy(deep=True)
    spec.traffic.arrival_rate_rps = rate
    work = args.work_dir / "mirror"
    work.mkdir(parents=True, exist_ok=True)
    trace = generate_trace(spec, work / "workload.jsonl",
                           num_requests=args.num_requests, seed=args.seed)
    predictor = LLMServingSimPredictor(trace, work_dir=work / "sims",
                                       timeout_s=args.timeout)
    candidates = CandidateGenerator(
        spec, cluster, islands, profiles, enable_prefix_caching=False,
    ).generate().candidates
    pair = {}
    for c in candidates:
        knobs = (c.knobs.max_num_seqs, c.knobs.max_num_batched_tokens)
        if len(c.assignments) != 1 or knobs != (32, 2048):
            continue
        if c.assignments[0].island_id in (SWEPT_ISLAND, MIRROR_ISLAND):
            pair[c.assignments[0].island_id] = c
    out: dict = {"rps": rate, "compared": {}}
    fields = ("p99_tpot_ms", "p99_ttft_ms", "served_concurrency", "tokens_per_joule")
    values: dict[str, dict] = {}
    for island, candidate in sorted(pair.items()):
        result = predictor.predict(candidate, spec, cluster, by_id, profiles)
        values[island] = {f: getattr(result.metrics, f, None) for f in fields}
        print(f"  mirror control {island}: {values[island]}", flush=True)
    if len(values) == 2:
        a, b = (values[SWEPT_ISLAND], values[MIRROR_ISLAND])
        out["compared"] = {
            f: {"swept": a[f], "mirror": b[f],
                "abs_pct": (None if not a[f] else abs(b[f] - a[f]) / a[f] * 100.0)}
            for f in fields
        }
        out["identical"] = all(a[f] == b[f] for f in fields)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
