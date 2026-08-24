"""Exp: router baselines RR / RAND / LOAD (work order §12).

Routing policy is a SIMULATOR input, not a post-hoc selection, so — unlike the
optimizer/resource/architecture baselines (experiments/scripts/exp_baselines.py,
which replay one cached sim) — the router axis needs re-simulation: the same
deployment is simulated once per policy. Router choice only matters for a
MULTI-REPLICA deployment (a single engine has nothing to balance), so this driver
picks one aggregated, multi-replica candidate and sweeps the three stock policies
over it.

Metrics are the §4 headline set (TTFT/TPOT percentiles, throughput, SLO
attainment/goodput, energy, tokens/J). Every number is an LLMServingSim
prediction on the measured/vendor profiles (absolute rule 3).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from planner.candidate_generator import CandidateGenerator
from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.plan import ServingArch
from planner.predictor.llmservingsim import LLMServingSimPredictor
from planner.spec import load_service_spec
from planner.util import provenance as prov
from planner.util.workload import generate_trace

DEFAULT_SEED = 42
DEFAULT_NUM_REQUESTS = 300
POLICIES = ("RR", "RAND", "LOAD")


def _pick_multi_replica(candidates, islands_by_id, profiles):
    """One aggregated, multi-replica, simulatable candidate — deterministic.
    Most replicas first (most for the router to balance), then candidate id."""
    def ok(c):
        return (
            c.serving_arch is ServingArch.AGGREGATED
            and sum(a.dp_replicas for a in c.assignments) >= 2
            and all(profiles[islands_by_id[a.island_id].accelerator_model].sim_hardware
                    for a in c.assignments)
        )
    eligible = [c for c in candidates if ok(c)]
    if not eligible:
        return None
    return min(eligible, key=lambda c: (-sum(a.dp_replicas for a in c.assignments), c.id))


def _metrics_row(policy: str, m) -> dict:
    return {
        "policy": policy,
        "p50_ttft_ms": m.p50_ttft_ms, "p99_ttft_ms": m.p99_ttft_ms,
        "p50_tpot_ms": m.p50_tpot_ms, "p99_tpot_ms": m.p99_tpot_ms,
        "throughput_tps": m.throughput_tps,
        "slo_attainment": m.slo_attainment, "slo_goodput_rps": m.slo_goodput_rps,
        "total_energy_j": m.total_energy_j, "tokens_per_joule": m.tokens_per_joule,
    }


def _fmt(v, spec=",.1f"):
    return format(v, spec) if isinstance(v, (int, float)) else "-"


def render_table(cand_id: str, rows: list[dict]) -> str:
    out = [
        f"# Router baselines (candidate `{cand_id}`)",
        "",
        "| policy | p50/p99 TTFT (ms) | p50/p99 TPOT (ms) | throughput (tok/s) "
        "| SLO attain | goodput (rps) | tok/J |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        ttft = f"{_fmt(r.get('p50_ttft_ms'))} / {_fmt(r.get('p99_ttft_ms'))}"
        tpot = f"{_fmt(r.get('p50_tpot_ms'))} / {_fmt(r.get('p99_tpot_ms'))}"
        out.append(
            f"| {r['policy']} | {ttft} | {tpot} | {_fmt(r.get('throughput_tps'))} "
            f"| {_fmt(r.get('slo_attainment'), '.2f')} | {_fmt(r.get('slo_goodput_rps'), '.2f')} "
            f"| {_fmt(r.get('tokens_per_joule'), '.3f')} |"
        )
    return "\n".join(out)


def run(args: argparse.Namespace) -> int:
    spec = load_service_spec(args.service)
    cluster = load_cluster_spec(args.cluster)
    profiles = load_profiles_for(cluster, args.root)
    islands = detect_islands(cluster, profiles)
    islands_by_id = {i.id: i for i in islands}

    gen = CandidateGenerator(
        spec, cluster, islands, profiles,
        max_num_seqs=(args.max_num_seqs,), max_num_batched_tokens=(args.max_num_batched_tokens,),
        enable_prefix_caching=False,
    ).generate()
    cand = _pick_multi_replica(gen.candidates, islands_by_id, profiles)
    if cand is None:
        print("error: no aggregated multi-replica simulatable candidate in this cluster "
              "(router choice is a no-op for single-replica deployments)", file=sys.stderr)
        return 1
    print(f"router sweep on {cand.id} "
          f"({sum(a.dp_replicas for a in cand.assignments)} replicas)", file=sys.stderr)

    work_root = Path(args.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    trace = generate_trace(spec, work_root / "workload.jsonl",
                           num_requests=args.num_requests, seed=args.seed)

    rows = []
    for policy in POLICIES:
        if not args.quiet:
            print(f"  simulating policy={policy}", file=sys.stderr)
        predictor = LLMServingSimPredictor(
            trace, work_dir=work_root / f"sims_{policy}", timeout_s=args.timeout,
            routing_policy=policy,
        )
        try:
            sim = predictor.predict(cand, spec, cluster, islands_by_id, profiles)
        finally:
            predictor.close()
        if not sim.ok or sim.metrics is None:
            rows.append({"policy": policy, "outcome": "sim_error", "detail": sim.detail})
            continue
        rows.append(_metrics_row(policy, sim.metrics))

    profile_paths = [Path(args.root) / a.profile for node in cluster.nodes
                     for a in node.accelerators if a.profile]
    provenance = prov.collect(
        service_spec_path=args.service, cluster_spec_path=args.cluster,
        profile_paths=profile_paths, dataset_path=trace.path, random_seed=args.seed,
        extra={"experiment": "exp_router_baselines", "candidate_id": cand.id,
               "replicas": sum(a.dp_replicas for a in cand.assignments),
               "policies": list(POLICIES), "workload": trace.as_provenance(),
               "knobs": {"max_num_seqs": args.max_num_seqs,
                         "max_num_batched_tokens": args.max_num_batched_tokens}},
    )
    result = {"provenance": provenance, "candidate_id": cand.id, "rows": rows}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "router_baselines.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str)
    )
    table = render_table(cand.id, rows)
    (out_dir / "router_baselines_table.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nwrote {out_dir / 'router_baselines.json'}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp: router baselines RR/RAND/LOAD (§12).")
    p.add_argument("--service", required=True, type=Path)
    p.add_argument("--cluster", required=True, type=Path)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--work-dir", type=Path, default=Path("outputs/exp_router"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    p.add_argument("--quiet", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
