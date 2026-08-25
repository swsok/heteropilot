"""Exp: baselines + ablation (work order §12).

Simulate-once, replay-select. One oracle-mode pass simulates every structurally
valid, *simulatable* candidate (islands whose profile has a real sim_hardware);
every optimizer/resource/architecture baseline and every replay-able ablation is
then a (subset -> selection-rule -> objective) decision over that shared cache,
scored on each pick's TRUE simulated metrics. Regret = (oracle_value -
strategy_value) / oracle_value on the primary objective (SLO-goodput/J).

Why this shape: the baselines are a selection problem layered on ONE expensive
simulation, not independent planners. Sharing the cache keeps every strategy on
identical physics and leaves the real optimizer (exhaustive/pareto/feasibility)
untouched, so oracle-agreement is preserved (nothing here is a pruning stage).

HONESTY (absolute rule 3, and see the architect design record):
- Every number is an LLMServingSim prediction; NPU/placeholder-profile islands are
  excluded up front (they cannot simulate) and counted, never faked.
- The objective (goodput/J) is computed from simulator `slo_attainment` /
  `completed_tokens` / energy; it does NOT read TTFT/TPOT. So calibration and
  robust margins move only the *feasibility boundary*, never a feasible plan's
  objective value. Consequently No-Calibration / No-Uncertainty / Static are
  N/A in the current phase and are emitted as labeled N/A rows, not numbers:
    * No-Calibration : calibration is not in the search path (it is opt-in and the
      `plan` command never applies it), so today's default already IS
      no-calibration; the contrast needs a non-identity Phase-4 fit.
    * No-Uncertainty : robust margins default to 0; the contrast needs non-zero
      Phase-4 calibration margins.
    * Static         : "no replanning" is a Phase-6 concept; pre-Phase-6 every
      plan is static, so there is no dynamic baseline to remove.
- fastest / most-efficient use memory_bandwidth_gbps and active_power as proxies
  (AcceleratorProfile has no compute-speed field; decode is memory-bound). Labeled.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from planner.candidate_generator import CandidateGenerator
from planner.inventory import (
    AcceleratorProfile,
    ExecutionIsland,
    detect_islands,
    load_cluster_spec,
    load_profiles_for,
)
from planner.optimizer import exhaustive, feasibility, greedy, pareto
from planner.optimizer.exhaustive import _routing_for, apply_pd_transfer_cost
from planner.plan import CandidateConfig, DeploymentPlan, Role, ServingArch
from planner.predictor import Predictor, SimResult
from planner.predictor.llmservingsim import LLMServingSimPredictor
from planner.spec import ServiceSpec, load_service_spec
from planner.topology import TopologyGraph
from planner.util import provenance as prov
from planner.util.parallel import predict_all
from planner.util.workload import generate_trace

DEFAULT_SEED = 42
DEFAULT_NUM_REQUESTS = 300


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _simulatable(candidate: CandidateConfig, islands, profiles) -> bool:
    """True iff every island the candidate touches has a real sim_hardware bundle.
    A placeholder profile (sim_hardware=None, e.g. the Ascend stub) cannot be
    simulated, so such candidates are excluded up front and counted."""
    return all(
        profiles[islands[a.island_id].accelerator_model].sim_hardware is not None
        for a in candidate.assignments
    )


def _assemble_plans(
    candidates: list[CandidateConfig],
    raw: dict[str, SimResult],
    spec: ServiceSpec,
    cluster,
    islands: dict[str, ExecutionIsland],
    *,
    apply_transfer: bool,
) -> tuple[dict[str, DeploymentPlan], dict[str, DeploymentPlan]]:
    """Build (feasible, all) plan dicts from cached SimResults, mirroring
    exhaustive.evaluate_candidates but with the P/D transfer cost toggleable.
    Used ONLY for the No-Topology ablation (apply_transfer=False); every other
    strategy goes through the real evaluate_candidates. Returns
    (feasible_by_id, all_by_id)."""
    topology = TopologyGraph(cluster)
    feasible: dict[str, DeploymentPlan] = {}
    every: dict[str, DeploymentPlan] = {}
    for index, cand in enumerate(candidates):
        sim = raw.get(cand.id)
        if sim is None or not sim.ok or sim.metrics is None:
            continue
        metrics = sim.metrics
        if apply_transfer and cand.serving_arch is ServingArch.PD_SPLIT:
            metrics, _ = apply_pd_transfer_cost(cand, metrics, spec, cluster, islands, topology)
        plan = DeploymentPlan(
            plan_id=f"plan-{index}", model=spec.model, candidate=cand,
            predicted=metrics, routing=_routing_for(cand),
        )
        every[cand.id] = plan
        if feasibility.evaluate(plan, spec).passed:
            feasible[cand.id] = plan
    return feasible, every


def _best(plans: list[DeploymentPlan], spec: ServiceSpec) -> DeploymentPlan | None:
    scorable = [p for p in plans if pareto.can_score(p, spec.objective.primary)[0]]
    if not scorable:
        return None
    return pareto.rank(scorable, spec.objective.primary, spec.objective.secondary)[0].plan


def _goodput(plan: DeploymentPlan) -> float:
    """SLO goodput proxy (tokens meeting SLO), the No-Energy selection key."""
    m = plan.predicted
    return m.completed_tokens * m.slo_attainment


# --------------------------------------------------------------------------- #
# strategy definitions -> chosen candidate id (or None)
# --------------------------------------------------------------------------- #

def _max_bandwidth_models(profiles: dict[str, AcceleratorProfile]) -> set[str]:
    bw = {m: p.memory_bandwidth_gbps for m, p in profiles.items() if p.sim_hardware}
    top = max(bw.values())
    return {m for m, v in bw.items() if v == top}


def _max_efficiency_models(profiles: dict[str, AcceleratorProfile]) -> set[str]:
    eff = {}
    for m, p in profiles.items():
        if not p.sim_hardware:
            continue
        watt = greedy._active_power_w(p)
        if watt:
            eff[m] = p.memory_bandwidth_gbps / watt
    if not eff:
        return set()
    top = max(eff.values())
    return {m for m, v in eff.items() if v == top}


def _island_models(cand: CandidateConfig, islands) -> set[str]:
    return {islands[a.island_id].accelerator_model for a in cand.assignments}


def _pd_backends(cand: CandidateConfig, islands) -> tuple[str, str] | None:
    if cand.serving_arch is not ServingArch.PD_SPLIT:
        return None
    pf = next(a for a in cand.assignments if a.role is Role.PREFILL)
    dc = next(a for a in cand.assignments if a.role is Role.DECODE)
    return islands[pf.island_id].backend, islands[dc.island_id].backend


def build_strategies(
    feasible: dict[str, DeploymentPlan],
    every: dict[str, DeploymentPlan],
    simulatable: list[CandidateConfig],
    pruned_ids: set[str],
    raw: dict[str, SimResult],
    spec: ServiceSpec,
    cluster,
    islands: dict[str, ExecutionIsland],
    profiles: dict[str, AcceleratorProfile],
) -> list[dict]:
    """Return an ordered list of {name, group, chosen_id, note} rows. chosen_id
    None means the strategy found nothing; a chosen_id not in `feasible` means the
    pick is infeasible (regret 1.0)."""
    feas_plans = list(feasible.values())
    by_cand = {c.id: c for c in simulatable}
    rows: list[dict] = []

    def add(name, group, chosen_id, note=""):
        rows.append({"name": name, "group": group, "chosen_id": chosen_id, "note": note})

    # -- optimizer ---------------------------------------------------------- #
    add("exhaustive-oracle", "optimizer",
        (_best(feas_plans, spec).candidate.id if _best(feas_plans, spec) else None),
        "argmax objective over ALL simulatable feasible candidates (reference)")
    proposed_plans = [feasible[i] for i in pruned_ids if i in feasible]
    add("proposed", "optimizer",
        (_best(proposed_plans, spec).candidate.id if _best(proposed_plans, spec) else None),
        "sim-guided pruned search + lexicographic rank")
    gpick = greedy.greedy(simulatable, spec, islands, profiles)
    add("greedy", "optimizer", gpick.id if gpick else None,
        "analytical roofline goodput/J proxy, NO simulation")

    # -- resource ----------------------------------------------------------- #
    fast_models = _max_bandwidth_models(profiles)
    fast = [p for p in feas_plans
            if _island_models(by_cand[p.candidate.id], islands) <= fast_models]
    add("fastest-only", "resource",
        (_best(fast, spec).candidate.id if _best(fast, spec) else None),
        f"only candidates on max-memory-bandwidth class {sorted(fast_models)} (proxy)")
    eff_models = _max_efficiency_models(profiles)
    effp = [p for p in feas_plans if _island_models(by_cand[p.candidate.id], islands) <= eff_models]
    add("most-efficient-only", "resource",
        (_best(effp, spec).candidate.id if _best(effp, spec) else None),
        f"only candidates on max bandwidth-per-watt class {sorted(eff_models)} (proxy)")
    least = min(feas_plans, key=lambda p: (p.active_accelerators,
                -pareto.objective_value(p, spec.objective.primary), p.candidate.id),
                default=None)
    add("least-device", "resource", least.candidate.id if least else None,
        "fewest active accelerators, tie-broken by objective")
    # simulator-blind heuristic: a fixed provisioning rule, no sim, no objective.
    #   prefer fastest class, smallest TP that fits, most replicas, aggregated.
    def _blind_key(c: CandidateConfig):
        on_fast = _island_models(c, islands) <= fast_models
        tp = max(a.tp_size for a in c.assignments)
        dp = sum(a.dp_replicas for a in c.assignments)
        is_pd = c.serving_arch is ServingArch.PD_SPLIT
        return (not on_fast, tp, -dp, is_pd, c.id)
    blind = min(simulatable, key=_blind_key, default=None)
    add("simulator-blind", "resource", blind.id if blind else None,
        "fixed provisioning rule (fastest class, smallest TP, most replicas), no sim")

    # -- architecture ------------------------------------------------------- #
    agg = [p for p in feas_plans if by_cand[p.candidate.id].serving_arch is ServingArch.AGGREGATED]
    add("aggregated", "architecture",
        (_best(agg, spec).candidate.id if _best(agg, spec) else None), "aggregated only")
    homo = [p for p in feas_plans if (b := _pd_backends(by_cand[p.candidate.id], islands))
            and b[0] == b[1]]
    add("homogeneous-P/D", "architecture",
        (_best(homo, spec).candidate.id if _best(homo, spec) else None),
        "P/D with same backend on both roles")
    hetero = [p for p in feas_plans if (b := _pd_backends(by_cand[p.candidate.id], islands))
              and b[0] != b[1]]
    add("heterogeneous-P/D", "architecture",
        (_best(hetero, spec).candidate.id if _best(hetero, spec) else None),
        "P/D with different backends across roles")

    # -- ablation ----------------------------------------------------------- #
    no_pd = [p for p in feas_plans
             if by_cand[p.candidate.id].serving_arch is not ServingArch.PD_SPLIT]
    add("No-PD-Specialization", "ablation",
        (_best(no_pd, spec).candidate.id if _best(no_pd, spec) else None),
        "P/D candidates removed from the search")
    if feas_plans:
        ne = max(feas_plans, key=lambda p: (_goodput(p), p.candidate.id))
        add("No-Energy", "ablation", ne.candidate.id,
            "select by SLO-goodput (ignore energy); scored on true goodput/J")
    else:
        add("No-Energy", "ablation", None, "no feasible candidates")
    nt_feasible, _ = _assemble_plans(simulatable, raw, spec, cluster, islands, apply_transfer=False)
    nt_best = _best(list(nt_feasible.values()), spec)
    add("No-Topology", "ablation", nt_best.candidate.id if nt_best else None,
        "P/D KV-transfer cost dropped at selection; scored on true (transfer-priced) metrics")
    # N/A ablations (labeled, never fabricated).
    add("No-Calibration", "ablation", None,
        "N/A: calibration is opt-in and not in the plan path, so the default already "
        "is no-calibration (needs a non-identity Phase-4 fit to contrast)")
    add("No-Uncertainty", "ablation", None,
        "N/A: robust margins default to 0 (needs non-zero Phase-4 calibration margins)")
    add("Static", "ablation", None,
        "N/A: no-replanning is a Phase-6 concept; pre-Phase-6 every plan is static")
    return rows


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def run(args: argparse.Namespace) -> int:
    spec = load_service_spec(args.service)
    cluster = load_cluster_spec(args.cluster)
    profiles = load_profiles_for(cluster, args.root)
    islands = detect_islands(cluster, profiles)
    islands_by_id = {i.id: i for i in islands}
    if not islands:
        print("error: no execution islands", file=sys.stderr)
        return 1

    # Full (oracle) candidate set and pruned (proposed) set — same generator,
    # only bound pruning differs. enable_pd so architecture/PD ablations exist.
    gen_all = CandidateGenerator(
        spec, cluster, islands, profiles,
        max_num_seqs=(args.max_num_seqs,), max_num_batched_tokens=(args.max_num_batched_tokens,),
        enable_prefix_caching=False, enable_bound_pruning=False, enable_pd=True,
    ).generate()
    gen_pruned = CandidateGenerator(
        spec, cluster, islands, profiles,
        max_num_seqs=(args.max_num_seqs,), max_num_batched_tokens=(args.max_num_batched_tokens,),
        enable_prefix_caching=False, enable_bound_pruning=True, enable_pd=True,
    ).generate()
    pruned_ids = {c.id for c in gen_pruned.candidates}

    simulatable = [c for c in gen_all.candidates if _simulatable(c, islands_by_id, profiles)]
    excluded = len(gen_all.candidates) - len(simulatable)
    print(f"generated {len(gen_all.candidates)} (oracle) / {len(gen_pruned.candidates)} (pruned); "
          f"{len(simulatable)} simulatable, {excluded} excluded (placeholder profile)",
          file=sys.stderr)

    work_root = Path(args.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    trace = generate_trace(spec, work_root / "workload.jsonl",
                           num_requests=args.num_requests, seed=args.seed)

    predictor = LLMServingSimPredictor(trace, work_dir=work_root / "sims", timeout_s=args.timeout)
    t_sim = time.monotonic()

    def _progress(i, total, cand):
        if not args.quiet:
            print(f"  [{i + 1}/{total}] simulating {cand.id}", file=sys.stderr)

    try:
        raw = predict_all(predictor, simulatable, spec, cluster, islands_by_id, profiles,
                          progress=_progress)
    finally:
        predictor.close()
    sim_wall = time.monotonic() - t_sim

    class _Replay(Predictor):
        def predict(self, candidate, spec, cluster, islands, profiles) -> SimResult:
            return raw[candidate.id]

    t_eval = time.monotonic()
    evaluation = exhaustive.evaluate_candidates(
        simulatable, spec, cluster, islands_by_id, profiles, _Replay()
    )
    feasible = {p.candidate.id: p for p in evaluation.feasible_plans}
    every = dict(feasible)
    for p, _r in evaluation.infeasible_plans:
        every.setdefault(p.candidate.id, p)

    oracle_best = _best(list(feasible.values()), spec)
    oracle_value = (
        pareto.objective_value(oracle_best, spec.objective.primary) if oracle_best else None
    )

    strategies = build_strategies(
        feasible, every, simulatable, pruned_ids, raw,
        spec, cluster, islands_by_id, profiles,
    )
    select_wall = time.monotonic() - t_eval

    rows = []
    for s in strategies:
        cid = s["chosen_id"]
        row = {"strategy": s["name"], "group": s["group"], "note": s["note"],
               "chosen_id": cid}
        if cid is None:
            row["status"] = "n/a"
        elif cid in feasible:
            val = pareto.objective_value(feasible[cid], spec.objective.primary)
            m = feasible[cid].predicted
            if val == float("-inf"):
                # Feasible, but the primary objective cannot score it (e.g. the
                # objective needs energy and this plan has none). Surface it as
                # unscored rather than writing a non-finite regret / invalid JSON.
                row["status"] = "unscored"
                row["note"] = (row["note"] + " | feasible but objective unscorable "
                               "(missing energy)").strip(" |")
            else:
                row["status"] = "feasible"
                row["goodput_per_joule"] = val
                row["regret"] = (
                    max(0.0, (oracle_value - val) / oracle_value)
                    if oracle_value and oracle_value > 0 else None
                )
            row["p99_ttft_ms"] = m.p99_ttft_ms
            row["p99_tpot_ms"] = m.p99_tpot_ms
            row["devices"] = feasible[cid].active_accelerators
        else:
            row["status"] = "selected_infeasible"
            row["regret"] = 1.0
        rows.append(row)

    planner_metrics = {
        "generated_oracle": len(gen_all.candidates),
        "generated_pruned": len(gen_pruned.candidates),
        "simulatable": len(simulatable),
        "excluded_placeholder": excluded,
        "feasible": len(feasible),
        "prune_ratio": round(1 - len(gen_pruned.candidates) / len(gen_all.candidates), 3)
        if gen_all.candidates else None,
        "oracle_goodput_per_joule": oracle_value,
        "sim_wall_seconds": round(sim_wall, 1),
        "select_wall_seconds": round(select_wall, 3),
    }

    profile_paths = [Path(args.root) / a.profile for node in cluster.nodes
                     for a in node.accelerators if a.profile]
    provenance = prov.collect(
        service_spec_path=args.service, cluster_spec_path=args.cluster,
        profile_paths=profile_paths, dataset_path=trace.path, random_seed=args.seed,
        extra={"experiment": "exp_baselines_ablation", "planner_metrics": planner_metrics,
               "workload": trace.as_provenance(),
               "knobs": {"max_num_seqs": args.max_num_seqs,
                         "max_num_batched_tokens": args.max_num_batched_tokens}},
    )
    result = {"provenance": provenance, "planner_metrics": planner_metrics, "rows": rows}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baselines.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str)
    )
    table = render_table(rows, planner_metrics)
    (out_dir / "baselines_table.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nwrote {out_dir / 'baselines.json'}", file=sys.stderr)
    return 0


def render_table(rows: list[dict], pm: dict) -> str:
    def fmt(v, spec=".4f"):
        return format(v, spec) if isinstance(v, (int, float)) else "-"
    out = [
        f"# Baselines + ablation (oracle goodput/J = {fmt(pm.get('oracle_goodput_per_joule'))})",
        "",
        "| group | strategy | status | goodput/J | regret | p99 TTFT (ms) "
        "| p99 TPOT (ms) | devices | note |",
        "| --- | --- | :---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        out.append(
            f"| {r['group']} | {r['strategy']} | {r['status']} "
            f"| {fmt(r.get('goodput_per_joule'))} | {fmt(r.get('regret'), '.3f')} "
            f"| {fmt(r.get('p99_ttft_ms'), ',.0f')} | {fmt(r.get('p99_tpot_ms'), '.1f')} "
            f"| {r.get('devices', '-')} | {r['note']} |"
        )
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp: baselines + ablation (§12).")
    p.add_argument("--service", required=True, type=Path)
    p.add_argument("--cluster", required=True, type=Path)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--work-dir", type=Path, default=Path("outputs/exp_baselines"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    p.add_argument("--quiet", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
