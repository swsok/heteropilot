"""P/D network-bandwidth sweep driver (Phase 5 increment 4, docs/phase5_plan.md).

Reproduces the work order §5.9 adoption crossing *at the planning level*: sweep one
inter-island fabric link's bandwidth and record where the recommended serving
architecture flips between ``pd_split`` and ``aggregated``.

Why this does not re-simulate per bandwidth
-------------------------------------------
The simulator is bandwidth-invariant for the prefill->decode KV transfer: it charges
the handoff as free (docs/phase5_plan.md increment 2, verified by the increment-2
spike). The whole sweep effect therefore comes from the *planner-side* transfer term
in ``planner.optimizer.exhaustive.apply_pd_transfer_cost`` (over
``planner.util.kv_transfer``), not from the simulator. So every candidate is
simulated exactly ONCE with the real ``LLMServingSimPredictor`` and its raw metrics
are cached to disk; for each bandwidth the driver rebuilds the cluster with that
fabric bandwidth and re-runs ONLY ``evaluate_candidates`` (planner transfer cost +
feasibility) and the ``pareto`` ranking, replaying the cached raw metrics through a
``ReplayPredictor``.

The envelope cache is deliberately bypassed: its key bands the network class
(``planner.envelope.network_class``), so sweeping bandwidth across bands would force a
re-simulation, which is exactly what we avoid here.

Honesty
-------
Every number this driver emits is a simulator prediction, and the P/D transfer cost
is a planner-side analytical add-on that the simulator does not model. Both facts are
recorded in the output provenance and repeated in the results doc.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from planner.candidate_generator import CandidateGenerator
from planner.inventory import (
    ClusterSpecV2,
    ExecutionIsland,
    detect_islands,
    load_cluster_spec,
    load_profiles_for,
)
from planner.optimizer import exhaustive, pareto
from planner.plan import (
    CandidateConfig,
    DeploymentPlan,
    PredictedMetrics,
    ServingArch,
)
from planner.predictor import Predictor, SimOutcome, SimResult
from planner.predictor.llmservingsim import LLMServingSimPredictor
from planner.spec import ServiceSpec, load_service_spec
from planner.util import provenance as prov
from planner.util.workload import WorkloadTrace, generate_trace

DEFAULT_BANDWIDTHS_GBPS = (400.0, 200.0, 100.0, 25.0, 10.0, 1.0)
DEFAULT_SEED = 42
DEFAULT_NUM_REQUESTS = 300


# --------------------------------------------------------------------------
# Replaying cached raw metrics through the Predictor ABC
# --------------------------------------------------------------------------

class ReplayPredictor(Predictor):
    """Returns raw (un-adjusted) sim metrics captured in a single earlier run.

    ``evaluate_candidates`` re-applies the P/D transfer cost every time it runs, so
    the replay must hand back the *raw* metrics the simulator produced - never the
    transfer-adjusted ones - or the sweep would double-charge the transfer.
    """

    def __init__(self, raw: dict[str, SimResult]) -> None:
        self._raw = raw

    def predict(self, candidate, spec, cluster, islands, profiles) -> SimResult:
        hit = self._raw.get(candidate.id)
        if hit is None:
            return SimResult(
                candidate.id,
                SimOutcome.CRASHED,
                detail="no cached simulation for this candidate (replay miss)",
            )
        return hit


# --------------------------------------------------------------------------
# Phase A: simulate every candidate once, cache raw metrics to disk
# --------------------------------------------------------------------------

def _raw_cache_load(path: Path) -> dict[str, SimResult] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    out: dict[str, SimResult] = {}
    for cid, entry in payload.get("candidates", {}).items():
        metrics = (
            PredictedMetrics.model_validate(entry["metrics"])
            if entry.get("metrics") is not None
            else None
        )
        out[cid] = SimResult(
            candidate_id=cid,
            outcome=SimOutcome(entry["outcome"]),
            metrics=metrics,
            detail=entry.get("detail", ""),
        )
    return out


def _raw_cache_store(path: Path, raw: dict[str, SimResult], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "candidates": {
            cid: {
                "outcome": res.outcome.value,
                "detail": res.detail,
                "metrics": res.metrics.model_dump() if res.metrics is not None else None,
            }
            for cid, res in raw.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def simulate_once(
    candidates: list[CandidateConfig],
    spec: ServiceSpec,
    cluster: ClusterSpecV2,
    islands_by_id: dict[str, ExecutionIsland],
    profiles: dict,
    trace: WorkloadTrace,
    *,
    work_dir: Path,
    timeout_s: float,
    quiet: bool,
) -> dict[str, SimResult]:
    """Run the real simulator once per candidate; return raw metrics keyed by id."""
    predictor = LLMServingSimPredictor(trace, work_dir=work_dir, timeout_s=timeout_s)
    raw: dict[str, SimResult] = {}
    try:
        for i, cand in enumerate(candidates):
            if not quiet:
                print(f"  [{i + 1}/{len(candidates)}] simulating {cand.id}", file=sys.stderr)
            raw[cand.id] = predictor.predict(cand, spec, cluster, islands_by_id, profiles)
    finally:
        predictor.close()
    return raw


# --------------------------------------------------------------------------
# Phase B: per-bandwidth planner-side re-evaluation (cheap, no simulation)
# --------------------------------------------------------------------------

def with_fabric_bandwidth(
    cluster: ClusterSpecV2, link_id: str, bandwidth_gbps: float
) -> ClusterSpecV2:
    """Deep-copy the cluster and set one link's bandwidth. Islands are unaffected:
    detection groups on intra-node links only, so the candidate set is stable."""
    clone = copy.deepcopy(cluster)
    for link in clone.links:
        if link.id == link_id:
            link.bandwidth_gbps = bandwidth_gbps
            return clone
    known = ", ".join(link.id for link in clone.links)
    raise SystemExit(f"error: fabric link '{link_id}' not found. Known links: {known}")


@dataclass
class BandwidthPoint:
    bandwidth_gbps: float
    recommended_arch: str | None = None
    recommended_id: str | None = None
    recommended_p99_ttft_ms: float | None = None
    recommended_energy_j: float | None = None
    pd_feasible: bool = False
    best_pd_id: str | None = None
    best_pd_p99_ttft_ms: float | None = None
    best_pd_energy_j: float | None = None
    best_pd_xfer_ms_p99: float | None = None
    best_pd_class_default: bool = False
    n_feasible: int = 0
    n_feasible_pd: int = 0
    notes: list[str] = field(default_factory=list)


def _rank_feasible(
    plans: list[DeploymentPlan], spec: ServiceSpec
) -> list[DeploymentPlan]:
    """Mirror exhaustive.search's selection: keep scorable plans, rank them."""
    scorable = [p for p in plans if pareto.can_score(p, spec.objective.primary)[0]]
    if not scorable:
        return []
    ranked = pareto.rank(scorable, spec.objective.primary, spec.objective.secondary)
    return [s.plan for s in ranked]


def evaluate_at_bandwidth(
    bandwidth_gbps: float,
    candidates: list[CandidateConfig],
    spec: ServiceSpec,
    cluster: ClusterSpecV2,
    islands_by_id: dict[str, ExecutionIsland],
    profiles: dict,
    link_id: str,
    replay: ReplayPredictor,
) -> BandwidthPoint:
    cluster_bw = with_fabric_bandwidth(cluster, link_id, bandwidth_gbps)
    evaluation = exhaustive.evaluate_candidates(
        candidates, spec, cluster_bw, islands_by_id, profiles, replay
    )
    point = BandwidthPoint(bandwidth_gbps=bandwidth_gbps)
    point.n_feasible = len(evaluation.feasible_plans)
    point.n_feasible_pd = sum(
        1 for p in evaluation.feasible_plans if p.candidate.serving_arch is ServingArch.PD_SPLIT
    )

    ranked = _rank_feasible(evaluation.feasible_plans, spec)
    if ranked:
        best = ranked[0]
        point.recommended_arch = best.candidate.serving_arch.value
        point.recommended_id = best.candidate.id
        point.recommended_p99_ttft_ms = best.predicted.p99_ttft_ms
        point.recommended_energy_j = best.predicted.total_energy_j

    pd_ranked = [p for p in ranked if p.candidate.serving_arch is ServingArch.PD_SPLIT]
    xfer_by_id = {t["candidate_id"]: t for t in evaluation.pd_transfers}
    if pd_ranked:
        best_pd = pd_ranked[0]
        point.pd_feasible = True
        point.best_pd_id = best_pd.candidate.id
        point.best_pd_p99_ttft_ms = best_pd.predicted.p99_ttft_ms
        point.best_pd_energy_j = best_pd.predicted.total_energy_j
        info = xfer_by_id.get(best_pd.candidate.id, {})
        point.best_pd_xfer_ms_p99 = info.get("xfer_ms_p99")
        point.best_pd_class_default = bool(info.get("class_default"))
        if point.best_pd_class_default:
            point.notes.append(
                "best P/D priced on interconnect class-default (no declared link on the "
                "prefill->decode path); the sweep does not move it - check the fixture"
            )
    return point


def find_crossing(points: list[BandwidthPoint]) -> float | None:
    """Highest bandwidth (scanning high->low) at which the recommendation stops
    being pd_split after having been pd_split at a higher bandwidth."""
    ordered = sorted(points, key=lambda p: p.bandwidth_gbps, reverse=True)
    seen_pd = False
    for point in ordered:
        if point.recommended_arch == ServingArch.PD_SPLIT.value:
            seen_pd = True
        elif seen_pd:
            return point.bandwidth_gbps
    return None


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def render_table(points: list[BandwidthPoint]) -> str:
    lines = [
        "| fabric BW (GB/s) | recommended arch | recommended id | rec p99 TTFT (ms) "
        "| best-P/D feasible | best-P/D p99 TTFT (ms) | P/D xfer p99 (ms) |",
        "| ---: | --- | --- | ---: | :---: | ---: | ---: |",
    ]
    for p in sorted(points, key=lambda x: x.bandwidth_gbps, reverse=True):
        def fmt(v: float | None) -> str:
            return f"{v:,.1f}" if v is not None else "-"

        lines.append(
            f"| {p.bandwidth_gbps:,.0f} | {p.recommended_arch or '-'} "
            f"| {p.recommended_id or '-'} | {fmt(p.recommended_p99_ttft_ms)} "
            f"| {'yes' if p.pd_feasible else 'no'} | {fmt(p.best_pd_p99_ttft_ms)} "
            f"| {fmt(p.best_pd_xfer_ms_p99)} |"
        )
    return "\n".join(lines)


def write_figure(points: list[BandwidthPoint], path: Path, slo_ttft_ms: float) -> bool:
    """P/D transfer-adjusted p99 TTFT vs fabric bandwidth. Returns False (skipped)
    when matplotlib is unavailable rather than failing the run.

    The P/D line breaks (NaN) at bandwidths where P/D is infeasible, so the plot
    shows P/D vanishing rather than appearing to speed up when the recommendation
    actually switches to a different (aggregated) plan. The SLO budget and the
    recommended-arch coloring make the crossing legible.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        # A missing plotting backend must not fail the sweep.
        return False

    nan = float("nan")
    ordered = sorted(points, key=lambda p: p.bandwidth_gbps)
    xs = [p.bandwidth_gbps for p in ordered]
    # NaN where P/D is infeasible, so the line breaks instead of jumping to another plan.
    pd_ttft = [
        (p.best_pd_p99_ttft_ms if (p.pd_feasible and p.best_pd_p99_ttft_ms is not None) else nan)
        for p in ordered
    ]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.axhline(slo_ttft_ms, color="#666666", linestyle="--", linewidth=1,
               label=f"TTFT p99 budget ({slo_ttft_ms:g} ms)")
    ax.plot(xs, pd_ttft, marker="o", color="#1b9e77",
            label="P/D p99 TTFT (transfer-adjusted; breaks where infeasible)")
    # Recommended plan, colored by the architecture actually chosen.
    for p in ordered:
        is_pd = p.recommended_arch == ServingArch.PD_SPLIT.value
        ax.scatter(
            [p.bandwidth_gbps], [p.recommended_p99_ttft_ms],
            color="#1b9e77" if is_pd else "#d95f02",
            marker="o" if is_pd else "s", s=70, zorder=5, edgecolors="black", linewidths=0.5,
        )
    # Legend proxies for the recommended-arch markers.
    ax.scatter([], [], color="#1b9e77", marker="o", edgecolors="black",
               linewidths=0.5, label="recommended: pd_split")
    ax.scatter([], [], color="#d95f02", marker="s", edgecolors="black",
               linewidths=0.5, label="recommended: aggregated")
    ax.set_xscale("log")
    ax.set_xlabel("inter-island fabric bandwidth (GB/s, log scale)")
    ax.set_ylabel("p99 TTFT (ms)")
    ax.set_title("P/D network sweep: adoption crossing vs fabric bandwidth")
    ax.legend(fontsize=8, loc="center left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    spec = load_service_spec(args.service)
    cluster = load_cluster_spec(args.cluster)
    profiles = load_profiles_for(cluster, args.root)
    islands = detect_islands(cluster, profiles)
    if not islands:
        print("error: no execution islands in this cluster", file=sys.stderr)
        return 1
    islands_by_id = {i.id: i for i in islands}

    seqs = (args.max_num_seqs,)
    tokens = (args.max_num_batched_tokens,)
    generator = CandidateGenerator(
        spec,
        cluster,
        islands,
        profiles,
        max_num_seqs=seqs,
        max_num_batched_tokens=tokens,
        enable_prefix_caching=False,
        enable_bound_pruning=not args.oracle,
        enable_pd=True,
    )
    generation = generator.generate()
    candidates = generation.candidates
    print(
        f"generated {len(candidates)} candidate(s) "
        f"(knobs restricted to max_num_seqs={args.max_num_seqs}, "
        f"max_num_batched_tokens={args.max_num_batched_tokens})",
        file=sys.stderr,
    )

    work_root = Path(args.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    trace = generate_trace(
        spec, work_root / "workload.jsonl", num_requests=args.num_requests, seed=args.seed
    )

    raw_cache = Path(args.raw_cache)
    raw = None if args.refresh else _raw_cache_load(raw_cache)
    if raw is None:
        raw = simulate_once(
            candidates, spec, cluster, islands_by_id, profiles, trace,
            work_dir=work_root / "sims", timeout_s=args.timeout, quiet=args.quiet,
        )
        _raw_cache_store(
            raw_cache, raw,
            meta={
                "note": "raw (un-adjusted) simulator metrics, simulated once for the sweep",
                "candidate_ids": sorted(raw),
                "num_requests": args.num_requests,
                "seed": args.seed,
                "trace_hash": prov.hash_file(trace.path),
            },
        )
        print(f"simulated once; raw metrics cached to {raw_cache}", file=sys.stderr)
    else:
        missing = [c.id for c in candidates if c.id not in raw]
        if missing:
            print(
                f"error: raw cache {raw_cache} is missing {len(missing)} candidate(s); "
                f"re-run with --refresh",
                file=sys.stderr,
            )
            return 1
        print(f"reusing raw metrics from {raw_cache} (no re-simulation)", file=sys.stderr)

    ok = sum(1 for r in raw.values() if r.ok)
    print(f"raw metrics: {ok}/{len(raw)} candidates simulated OK", file=sys.stderr)

    replay = ReplayPredictor(raw)
    points = [
        evaluate_at_bandwidth(
            bw, candidates, spec, cluster, islands_by_id, profiles, args.fabric_link, replay
        )
        for bw in args.bandwidths
    ]
    crossing = find_crossing(points)

    profile_paths = [
        Path(args.root) / a.profile
        for node in cluster.nodes
        for a in node.accelerators
        if a.profile
    ]
    provenance = prov.collect(
        service_spec_path=args.service,
        cluster_spec_path=args.cluster,
        profile_paths=profile_paths,
        dataset_path=trace.path,
        random_seed=args.seed,
        extra={
            "experiment": "phase5_increment4_pd_network_sweep",
            "fabric_link": args.fabric_link,
            "bandwidths_gbps": list(args.bandwidths),
            "knobs": {
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
            },
            "bound_pruning": not args.oracle,
            "workload": trace.as_provenance(),
            "pd_transfer_caveat": exhaustive.PD_TRANSFER_CAVEAT,
            "simulate_once": (
                "the simulator is bandwidth-invariant for the P/D KV transfer; every "
                "candidate was simulated once and the bandwidth sweep re-runs only the "
                "planner-side transfer cost + feasibility + ranking"
            ),
        },
    )

    result = {
        "provenance": provenance,
        "objective": {
            "primary": spec.objective.primary.value,
            "secondary": spec.objective.secondary.value if spec.objective.secondary else None,
        },
        "slo": {
            "ttft_p": spec.slo.ttft.percentile,
            "ttft_max_ms": spec.slo.ttft.max_ms,
            "tpot_p": spec.slo.tpot.percentile,
            "tpot_max_ms": spec.slo.tpot.max_ms,
        },
        "crossing_bandwidth_gbps": crossing,
        "points": [vars(p) for p in points],
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pd_network_sweep.json").write_text(json.dumps(result, indent=2, sort_keys=True))

    table = render_table(points)
    (out_dir / "pd_network_sweep_table.md").write_text(table + "\n")
    print("\n" + table)
    if crossing is not None:
        print(f"\ncrossing bandwidth (pd_split -> aggregated): {crossing:g} GB/s")
    else:
        archs = {p.recommended_arch for p in points}
        print(
            "\nNO crossing observed (null result): recommended arch never flips "
            f"pd_split -> aggregated across the swept range. Arch(es) seen: {sorted(archs)}"
        )

    fig_written = False
    if args.figure:
        fig_written = write_figure(points, Path(args.figure), spec.slo.ttft.max_ms)
        msg = f"written to {args.figure}" if fig_written else "skipped (no matplotlib)"
        print(f"figure {msg}", file=sys.stderr)
    print(f"wrote {out_dir / 'pd_network_sweep.json'}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P/D network-bandwidth sweep (Phase 5 increment 4).")
    p.add_argument("--service", required=True, type=Path)
    p.add_argument("--cluster", required=True, type=Path)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument(
        "--fabric-link", required=True,
        help="id of the inter-island link to sweep (must lie on the prefill->decode path).",
    )
    p.add_argument(
        "--bandwidths", type=float, nargs="+", default=list(DEFAULT_BANDWIDTHS_GBPS),
        help="fabric bandwidths to sweep, GB/s.",
    )
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--oracle", action="store_true", help="Disable bound-based pruning.")
    p.add_argument("--work-dir", type=Path, default=Path("outputs/.hp-pd-sweep/work"))
    p.add_argument(
        "--raw-cache", type=Path, default=Path("outputs/.hp-pd-sweep/raw_metrics.json"),
        help="where the single simulation's raw metrics are cached / reused.",
    )
    p.add_argument("--refresh", action="store_true", help="Ignore the raw cache and re-simulate.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/.hp-pd-sweep"))
    p.add_argument("--figure", type=Path, help="Optional matplotlib figure path (PNG).")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
