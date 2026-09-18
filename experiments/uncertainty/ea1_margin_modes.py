"""E-A1: scalar margin vs per-candidate margin (uncertainty work order STEP A5).

Four conditions on ONE fixture and ONE set of simulations:

  (a) no margin                 - what the planner reported before any calibration
  (b) global 18 %               - D22's measured TPOT optimism, applied to everything
  (c) accuracy domain           - each candidate margined at its own operating point,
                                  with the committed domains' own outside policy
                                  (`widen_error_bars`: extrapolate past the points)
  (d) accuracy domain, refuse   - the same domains under `outside_domain: refuse`
                                  (the D33 default for a new domain): a candidate
                                  outside the measured range is unmeasured

Only the verdict differs between them; the predictions are identical, so the
conditions share an envelope cache and the simulator runs once. That is the
point of the experiment - a margin policy is a decision rule, not a prediction.

The domains are the committed `calibration.AccuracyDomain` curves under
profiles/calibration/ (D33 reconciled the uncertainty stack onto them):
RNGD-CARD's nine D32 points over served concurrency 1.02-76 and the A40's three.

`PlannerOutput` carries rejection COUNTS, not per-candidate reasons, so this
calls `evaluate_candidates` for the per-candidate table and `search` for the
headline recommendation. Both read the same cache, so neither costs a
simulation.

Operating points. The committed cache entries predate per-hardware
operating-point caching: they carry the served concurrency PER ISLAND
(busiest instance of each island, `served_concurrency_per_island`, uncertainty
§2.4.1 rev 2) but not the per-hardware, per-phase decomposition the margin
policy reads (`SimResult.operating_point`). `_IslandOperatingPoint` fills the
gap: each hardware kind in the candidate is placed at the busiest of its
islands' per-island figures, in phase "total" - the same aggregation
`planner/util/operating_point.py` applies to an aggregated deployment, so for
the 324 aggregated and mixed candidates here (no P/D) this is the operating
point the predictor would have recorded. Entries with no per-island figure
fall back to the run-level served concurrency and are counted separately.

Writes the GENERATED tables (`--out`) and the full per-candidate record
(`--json-out`). The authored analysis lives beside them in
`experiments/uncertainty/results/ea1_margin_modes.md`.

Usage (after the cache is warm - see the result md for the cost of filling it)::

    PYTHONPATH=$PWD .venv/bin/python experiments/uncertainty/ea1_margin_modes.py \
        --cache-dir outputs/uncertainty/ea1/cache \
        --out experiments/uncertainty/results/ea1_margin_modes_table.md \
        --json-out outputs/uncertainty/ea1/ea1_margin_modes.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from planner.envelope import EnvelopeCache, workload_bucket, workload_shape
from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.optimizer import exhaustive
from planner.optimizer.margin import AccuracyDomainMargin, MarginPolicy
from planner.plan import CandidateConfig, RejectionStage
from planner.predictor import SimResult
from planner.predictor.calibration import (
    AccuracyDomain,
    load_accuracy_domains,
    load_calibrations,
)
from planner.predictor.llmservingsim import LLMServingSimPredictor
from planner.spec import load_service_spec
from planner.topology import TopologyGraph
from planner.util import provenance as prov
from planner.util.workload import generate_trace

#: The fixture `outputs/pd_slo_sweep_margin18/` used, read from its own JSON.
MARGIN18 = Path("outputs/pd_slo_sweep_margin18/pd-rngd-gpu-card.json")
#: D22's measured TPOT optimism at the load the card fixture runs at.
D22_GLOBAL_MARGIN_PERCENT = 18.0
OUTSIDE = RejectionStage.OUTSIDE_CALIBRATION_DOMAIN.value
CONDITION_ORDER = ("a_margin0", "b_global18", "c_accuracy_domain", "d_refuse")


@dataclass
class Condition:
    key: str
    label: str
    policy: MarginPolicy | None
    ttft_percent: float = 0.0
    tpot_percent: float = 0.0


class _IslandOperatingPoint(EnvelopeCache):
    """An envelope cache that fills a missing per-hardware operating point from
    the cached per-island served concurrency (see the module docstring)."""

    def __init__(self, *args, island_hw: dict[str, str], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.island_hw = island_hw
        self.filled_per_island: list[str] = []
        self.filled_run_level: list[str] = []

    def get(self, candidate: CandidateConfig) -> SimResult | None:
        result = super().get(candidate)
        if result is None or result.operating_point or result.metrics is None:
            return result
        per_island = result.metrics.served_concurrency_per_island or {}
        run_level = result.metrics.served_concurrency
        filled = False
        for a in candidate.assignments:
            hw = self.island_hw.get(a.island_id)
            if hw is None:
                continue
            load = per_island.get(a.island_id, run_level)
            if load is None:
                continue
            current = result.operating_point.get(hw)
            if current is None or load > current["concurrency"]:
                result.operating_point[hw] = {
                    "concurrency": load, "phase": "total",
                    "requests": result.metrics.completed_requests, "wall_s": None,
                }
            filled = True
        if filled:
            (self.filled_per_island if per_island else self.filled_run_level).append(
                candidate.id
            )
        return result


def _refusing(domains: dict[str, AccuracyDomain]) -> dict[str, AccuracyDomain]:
    return {hw: d.model_copy(update={"outside_domain": "refuse"}) for hw, d in domains.items()}


def _conditions(
    domains, calibration, bucket: str, shape: str, *, scope: dict,
) -> list[Condition]:
    """The four original rules, plus S4's (e).

    (a)-(d) are frozen: they are what `ea1_margin_modes.md` records and what
    D22's revalidation is anchored to, so they are built with
    `condition_mismatch="warn"`. That is not a default they were written with -
    the argument did not exist - but it is what reproduces them now that S1
    (D110) ships `refuse`. Under the planner's own default every candidate on
    this fixture is held, (c) and (d) both report 0 feasible, and 50/244/30
    cannot be reproduced at all. `warn` applies the domain under protest and
    records the mismatch, which is what keeps (a)-(d) a comparison of MARGIN
    rules rather than a rerun of S1.

    (e) is S1's verdict: the same per-operating-point rule, refusing both an
    operating point outside the measured range AND a configuration no domain
    was measured under. `scope` carries the identifying fields the comparison
    needs - model, variant, arrival process, per-island binding - so the check
    compares everything both sides state.
    """
    warn = {"condition_mismatch": "warn"}
    return [
        Condition("a_margin0", "(a) no margin", None),
        Condition(
            "b_global18", f"(b) global {D22_GLOBAL_MARGIN_PERCENT:g} % (D22)",
            None, D22_GLOBAL_MARGIN_PERCENT, D22_GLOBAL_MARGIN_PERCENT,
        ),
        Condition(
            "c_accuracy_domain", "(c) accuracy domain (committed policy: widen)",
            AccuracyDomainMargin(domains, shape=shape, calibration=calibration,
                                 bucket=bucket, **warn),
        ),
        Condition(
            "d_refuse", "(d) accuracy domain, outside_domain: refuse",
            AccuracyDomainMargin(_refusing(domains), shape=shape,
                                 calibration=calibration, bucket=bucket, **warn),
        ),
        Condition(
            "e_condition_refuse",
            "(e) + application-condition refuse (S1, D110)",
            AccuracyDomainMargin(
                _refusing(domains), shape=shape, calibration=calibration,
                bucket=bucket, condition_mismatch="refuse", **scope,
            ),
        ),
    ]


def _conc(plan) -> float | None:
    if plan.operating_point:
        return max(p.concurrency for p in plan.operating_point)
    return plan.predicted.served_concurrency


def _verdicts(evaluation) -> dict[str, dict]:
    """candidate id -> what happened to it, with the margin that decided it."""
    out: dict[str, dict] = {}
    for plan in evaluation.feasible_plans:
        out[plan.candidate.id] = {
            "verdict": "feasible", "stage": None, "concurrency": _conc(plan),
            "ttft_percent": plan.robust_margin_ttft_percent,
            "tpot_percent": plan.robust_margin_tpot_percent,
            "margin_source": plan.margin_source,
            "extrapolated": [p.hardware for p in plan.operating_point
                             if p.in_calibration_domain is False],
            "basis": plan.margin_basis or "",
        }
    for plan, _report in evaluation.infeasible_plans:
        out[plan.candidate.id] = {
            "verdict": "rejected", "stage": None, "concurrency": _conc(plan),
            "ttft_percent": plan.robust_margin_ttft_percent,
            "tpot_percent": plan.robust_margin_tpot_percent,
            "margin_source": plan.margin_source,
            "extrapolated": [p.hardware for p in plan.operating_point
                             if p.in_calibration_domain is False],
            "basis": plan.margin_basis or "",
        }
    for candidate_id, decision in evaluation.unmeasured:
        out[candidate_id] = {
            "verdict": "rejected", "stage": OUTSIDE, "concurrency": decision.concurrency,
            "ttft_percent": decision.ttft_percent, "tpot_percent": decision.tpot_percent,
            "margin_source": decision.source, "extrapolated": [], "basis": decision.basis,
        }
    for rejection in evaluation.rejections:
        if rejection.candidate_id in out:
            out[rejection.candidate_id]["stage"] = rejection.stage.value
            out[rejection.candidate_id].setdefault("reason", rejection.reason)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/uncertainty/ea1/work"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--calibration", nargs="*", type=Path, default=None,
        help="Calibration files for (c)/(d). Default: every file under "
             "profiles/calibration/, i.e. the committed domains.")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    fixture = json.loads(MARGIN18.read_text())
    spec = load_service_spec(fixture["service"])
    cluster = load_cluster_spec(fixture["cluster"])
    profiles = load_profiles_for(cluster, Path("."))
    islands = detect_islands(cluster, profiles)
    bucket = workload_bucket(spec)
    shape = workload_shape(spec)
    domains = load_accuracy_domains(".", args.calibration)
    calibration = load_calibrations(".", args.calibration)

    # The margin policy needs each island's hardware label. Reusing the
    # planner's own resolver rather than re-deriving it, so the experiment and
    # the planner cannot disagree about which calibration an island maps to.
    from planner.optimizer.exhaustive import _profile_tiers

    _tiers, island_hw, _warnings = _profile_tiers(spec, islands, profiles)

    # (e)'s application conditions, built exactly as `python -m planner plan`
    # builds them (planner/__main__.py `_margin_policy`), so the experiment and
    # the planner cannot disagree about what this run presents to a domain.
    # `open_loop` is a statement about the planner and not a setting: `plan`
    # always replays an arrival trace, so a domain fitted closed-loop is
    # measured under a different process (D19) and says so through the same
    # match test.
    from planner.predictor.calibration import load_domain_index
    from planner.util.tier import resolve_variant

    scope = {
        "model": spec.model,
        "variant": resolve_variant(spec.service.dtype, spec.service.kv_cache_dtype),
        "arrival_process": "open_loop",
        "device_binding": {
            i.id: cluster.node(i.node_id).device_binding for i in islands
        },
        "index": load_domain_index("."),
    }

    args.work_dir.mkdir(parents=True, exist_ok=True)
    trace = generate_trace(
        spec, args.work_dir / "workload.jsonl",
        num_requests=fixture["num_requests"], seed=fixture["seed"],
    )
    reduction = TopologyGraph(cluster).reduce_for_simulator(islands)
    cache = _IslandOperatingPoint(
        args.cache_dir, spec,
        accelerator_of={i.id: i.accelerator_model for i in islands},
        link_bw_gbps=reduction.link_bw_gbps,
        trace_digest=prov.hash_file(trace.path),
        island_hw=island_hw,
    )
    predictor = LLMServingSimPredictor(
        trace, work_dir=args.work_dir / "sims", timeout_s=args.timeout,
    )

    by_id = {i.id: i for i in islands}
    results: dict[str, dict] = {}
    try:
        for condition in _conditions(domains, calibration, bucket, shape, scope=scope):
            from planner.candidate_generator import CandidateGenerator

            generation = CandidateGenerator(
                spec, cluster, islands, profiles, enable_prefix_caching=False,
            ).generate()
            evaluation = exhaustive.evaluate_candidates(
                generation.candidates, spec, cluster, by_id, profiles, predictor,
                cache=cache,
                ttft_margin_percent=condition.ttft_percent,
                tpot_margin_percent=condition.tpot_percent,
                margin_policy=condition.policy,
                island_hw=island_hw,
                max_workers=args.workers,
            )
            output = exhaustive.search(
                spec, cluster, islands, profiles, predictor,
                cache=cache,
                ttft_margin_percent=condition.ttft_percent,
                tpot_margin_percent=condition.tpot_percent,
                margin_policy=condition.policy,
                max_workers=args.workers,
            )
            best = output.recommended
            results[condition.key] = {
                "label": condition.label,
                "feasible": output.feasible,
                "recommended_plan_id": best.plan.plan_id if best else None,
                "recommended_candidate": best.plan.candidate.id if best else None,
                "objective": best.objective.value if best else None,
                "value": best.value if best else None,
                "tokens_per_joule": (
                    best.plan.predicted.tokens_per_joule if best else None
                ),
                "served_concurrency": _conc(best.plan) if best else None,
                "margin_basis": (best.plan.margin_basis if best else None),
                "rejected_summary": output.rejected_summary,
                "feasible_count": len(evaluation.feasible_plans),
                "suggestions": output.suggestions,
                "notes": evaluation.notes,
                "verdicts": _verdicts(evaluation),
            }
            print(f"{condition.label}: feasible={len(evaluation.feasible_plans)} "
                  f"rejected={output.rejected_summary}")
    finally:
        predictor.close()

    payload = {
        "fixture": fixture,
        "bucket": bucket,
        "shape": shape,
        "calibration": (
            [str(p) for p in args.calibration] if args.calibration
            else "profiles/calibration/*.yaml"
        ),
        "domains": {
            hw: {"conc_min": d.conc_min, "conc_max": d.conc_max,
                 "points": len(d.points), "outside_domain": d.outside_domain}
            for hw, d in sorted(domains.items())
        },
        "operating_point_filled_from_per_island": sorted(set(cache.filled_per_island)),
        "operating_point_filled_from_run_level": sorted(set(cache.filled_run_level)),
        "cache_stats": cache.stats(),
        "provenance": prov.collect(
            service_spec_path=fixture["service"],
            cluster_spec_path=fixture["cluster"],
            dataset_path=trace.path,
            random_seed=fixture["seed"],
        ),
        "conditions": results,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        print(f"wrote {args.json_out}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_render(payload))
    print(f"wrote {args.out}")
    return 0


def _render(payload: dict) -> str:
    conditions = payload["conditions"]
    lines = [
        "# E-A1 — scalar margin vs per-candidate margin (STEP A5)",
        "",
        f"Fixture: `{payload['fixture']['cluster']}` + `{payload['fixture']['service']}`, "
        f"{payload['fixture']['num_requests']} requests, seed {payload['fixture']['seed']}.",
        f"Canonical bucket: `{payload['bucket']}`; shape `{payload['shape']}`.",
        f"Domains ({payload['calibration']}): "
        + ", ".join(
            f"`{hw}` [{d['conc_min']:.4g}, {d['conc_max']:.4g}] "
            f"({d['points']} pts, {d['outside_domain']})"
            for hw, d in payload["domains"].items()
        ) + ".",
        "",
        "One set of simulations serves all four conditions - only the verdict rule",
        "differs - so the comparison is of decision rules, not of predictions.",
        f"Operating points filled from the cached per-island served concurrency: "
        f"{len(payload['operating_point_filled_from_per_island'])} candidates; from "
        f"the run-level figure: {len(payload['operating_point_filled_from_run_level'])} "
        f"(see the script docstring).",
        "",
        "## Headline",
        "",
        "| condition | feasible | recommended | tokens/J | rejected |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for key in CONDITION_ORDER:
        c = conditions[key]
        tpj = "n/a" if c["tokens_per_joule"] is None else f"{c['tokens_per_joule']:.4f}"
        rej = ", ".join(f"{k} {v}" for k, v in sorted(c["rejected_summary"].items()))
        lines.append(
            f"| {c['label']} | {c['feasible_count']} | "
            f"`{c['recommended_candidate'] or 'NONE'}` | {tpj} | {rej} |"
        )

    def differing(left: str, right: str) -> list[str]:
        lv, rv = conditions[left]["verdicts"], conditions[right]["verdicts"]
        return sorted(
            cid for cid in set(lv) | set(rv)
            if (lv.get(cid, {}).get("verdict"), lv.get(cid, {}).get("stage"))
            != (rv.get(cid, {}).get("verdict"), rv.get(cid, {}).get("stage"))
        )

    for left, right, title in (
        ("b_global18", "c_accuracy_domain",
         "(b) vs (c): candidates the two conditions judge differently"),
        ("c_accuracy_domain", "d_refuse", "(c) vs (d): what refusing extrapolation changes"),
    ):
        lines += ["", f"## {title}", ""]
        diff = differing(left, right)
        lv, rv = conditions[left]["verdicts"], conditions[right]["verdicts"]
        if not diff:
            lines.append("None: the two conditions agree on every candidate.")
            continue
        lines += [
            f"{len(diff)} of {len(set(lv) | set(rv))} candidates.",
            "",
            f"| candidate | served L | {left} verdict | tpot margin | {right} verdict | "
            "tpot margin |",
            "| --- | ---: | --- | ---: | --- | ---: |",
        ]
        for cid in diff[:60]:
            a, b = lv.get(cid, {}), rv.get(cid, {})
            conc = b.get("concurrency") or a.get("concurrency")
            lines.append(
                f"| `{cid}` | {'n/a' if conc is None else f'{conc:.2f}'} "
                f"| {a.get('stage') or a.get('verdict', '-')} "
                f"| {a.get('tpot_percent', 0):.2f} % "
                f"| {b.get('stage') or b.get('verdict', '-')} "
                f"| {b.get('tpot_percent', 0):.2f} % |"
            )
        if len(diff) > 60:
            lines.append(f"... {len(diff) - 60} more (see the JSON)")

    lines += ["", "## Notes and suggestions emitted", ""]
    for key in ("c_accuracy_domain", "d_refuse"):
        c = conditions[key]
        lines.append(f"**{c['label']}**")
        for s in (c.get("notes") or []) + (c["suggestions"] or []):
            lines.append(f"- {s}")
        lines.append("")
    lines += [
        "## Provenance",
        "",
        f"- cache: {payload['cache_stats']}",
        f"- git: {payload['provenance'].get('git_commit')}",
        f"- node: {payload['provenance'].get('accelerators')}",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
