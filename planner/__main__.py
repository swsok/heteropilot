"""HeteroPilot CLI (work order §6).

`inspect-cluster` (Phase 1) and `plan` / `validate-plan` (Phase 2) are live.
`deploy` / `status` arrive in Phase 4 and are declared so `--help` shows the
intended shape.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

from planner.envelope import EnvelopeCache
from planner.inventory import (
    AcceleratorProfile,
    ClusterSpecV2,
    InventoryError,
    compatibility,
    detect_islands,
    load_cluster_spec,
    load_profiles_for,
)
from planner.optimizer import exhaustive
from planner.plan import PlannerOutput
from planner.predictor.llmservingsim import LLMServingSimPredictor
from planner.render import render
from planner.spec import ServiceSpec, SpecError, load_service_spec
from planner.topology import TopologyGraph
from planner.util import memory as memutil
from planner.util import provenance as prov
from planner.util.workload import generate_trace

_NOT_YET = {"deploy": "Phase 4", "status": "Phase 4"}

DEFAULT_TRACE_REQUESTS = 300
DEFAULT_SEED = 42


# --------------------------------------------------------------------------
# inspect-cluster (Phase 1)
# --------------------------------------------------------------------------

def _print_islands(
    cluster: ClusterSpecV2,
    profiles: dict[str, AcceleratorProfile],
    service: ServiceSpec | None,
) -> None:
    islands = detect_islands(cluster, profiles)

    print(f"Cluster: {cluster.cluster_id}")
    print(f"  nodes={len(cluster.nodes)} links={len(cluster.links)} islands={len(islands)}")
    print("  accelerators considered: state == FREE only")
    if service is not None:
        print(f"  service: {service.model} dtype={service.service.dtype}")
    print()

    if not islands:
        print("No execution islands. Every accelerator is non-FREE or the cluster is empty.")
        return

    for island in islands:
        profile = profiles.get(island.accelerator_model)
        link = island.interconnect_type.value if island.interconnect_type else "none (isolated)"
        print(f"[{island.id}]")
        print(f"  node          : {island.node_id}")
        print(f"  backend       : {island.backend}")
        print(f"  accelerators  : {island.size} x {island.accelerator_model} "
              f"({', '.join(island.accelerator_ids)})")
        print(f"  interconnect  : {link}")
        print(f"  total memory  : {island.total_memory_gb:g} GB")
        print(f"  TP candidates : {island.max_tp_candidates}")
        if profile is None:
            print("  profile       : MISSING - TP capped at island size")
        else:
            print(f"  profile       : {profile.profile_id} (source={profile.source.value}, "
                  f"max_tp={profile.max_tp_size})")
        if service is not None:
            _print_compatibility(island, profile, service)
        print()

    _print_summary(islands, profiles, service)


def _print_compatibility(island, profile, service: ServiceSpec) -> None:
    if profile is None:
        print("  compatibility : UNKNOWN - cannot check without a profile")
        return
    ok = compatibility(service.model, service.service.dtype, profile)
    print(f"  compatibility : {'SUPPORTED' if ok else 'UNSUPPORTED'} "
          f"for {service.model} @ {service.service.dtype}")
    if not ok:
        declared = ", ".join(
            f"{e.pattern}{list(e.dtypes)}" for e in profile.supported_models
        ) or "(none declared)"
        print(f"                  declared support: {declared}")
        return

    per_device_gb = island.total_memory_gb / island.size
    print("  memory fit    :")
    for tp in island.max_tp_candidates:
        try:
            fits, report = memutil.feasible(
                service.model, tp_size=tp, device_memory_gb=per_device_gb,
                dtype=service.service.dtype, kv_cache_dtype=service.service.kv_cache_dtype,
            )
        except memutil.MemoryError_ as exc:
            print(f"      tp={tp}: ERROR {exc}")
            continue
        print(f"      tp={tp}: {'ok  ' if fits else 'FAIL'} {report.summary()}")
        if fits and report.naive_kv_tokens > report.kv_tokens:
            delta = report.naive_kv_tokens / report.kv_tokens - 1
            print(f"              (raw simulator model would claim "
                  f"{report.naive_kv_tokens:,} tokens, +{delta:.0%} - see deviations D10)")


def _print_summary(islands, profiles, service) -> None:
    print("-" * 72)
    backends = sorted({i.backend for i in islands})
    print(f"Backends present: {', '.join(backends)}")
    if len(backends) > 1:
        print("  Heterogeneous cluster. TP/PP stays inside an island; heterogeneity is")
        print("  exploited across replicas or prefill/decode roles only (absolute rule 2).")
    missing = sorted({i.accelerator_model for i in islands if i.accelerator_model not in profiles})
    if missing:
        print(f"Models without a profile: {', '.join(missing)}")
    if service is not None:
        usable = [
            i for i in islands
            if (p := profiles.get(i.accelerator_model))
            and compatibility(service.model, service.service.dtype, p)
        ]
        print(f"Islands able to run {service.model}: {len(usable)} of {len(islands)}")


def cmd_inspect_cluster(args: argparse.Namespace) -> int:
    cluster = load_cluster_spec(args.cluster)
    profiles = load_profiles_for(cluster, args.root)
    service = load_service_spec(args.service) if args.service else None
    _print_islands(cluster, profiles, service)
    return 0


# --------------------------------------------------------------------------
# plan (Phase 2)
# --------------------------------------------------------------------------

def _write_output(output: PlannerOutput, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(output.model_dump(mode="json"), sort_keys=False))


def cmd_plan(args: argparse.Namespace) -> int:
    spec = load_service_spec(args.service)
    cluster = load_cluster_spec(args.cluster)
    profiles = load_profiles_for(cluster, args.root)
    islands = detect_islands(cluster, profiles)
    if not islands:
        print("error: no execution islands in this cluster", file=sys.stderr)
        return 1

    work_root = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="hp-"))
    work_root.mkdir(parents=True, exist_ok=True)

    trace = generate_trace(
        spec,
        work_root / "workload.jsonl",
        num_requests=args.num_requests,
        seed=args.seed,
    )

    profile_paths: list[str | Path] = [
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
            "workload": trace.as_provenance(),
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "activation_reserve_gb": args.activation_reserve_gb,
            "enable_prefix_caching": False,
            "bound_pruning": not args.oracle,
        },
    )
    missing = prov.note_missing(provenance)
    if missing:
        print(f"warning: provenance fields could not be determined: {', '.join(missing)}",
              file=sys.stderr)

    topology = TopologyGraph(cluster)
    reduction = topology.reduce_for_simulator(islands)
    provenance["topology"] = reduction.as_provenance()

    cache = None
    if args.cache_dir:
        cache = EnvelopeCache(
            args.cache_dir,
            spec,
            accelerator_of={i.id: i.accelerator_model for i in islands},
            link_bw_gbps=reduction.link_bw_gbps,
            trace_digest=prov.hash_file(trace.path),
        )

    predictor = LLMServingSimPredictor(
        trace,
        work_dir=work_root / "sims",
        timeout_s=args.timeout,
        gpu_memory_utilization=args.gpu_memory_utilization,
        activation_reserve_gb=args.activation_reserve_gb,
        keep_artifacts=args.keep_artifacts,
    )

    def progress(i: int, total: int, candidate) -> None:
        if not args.quiet:
            print(f"  [{i + 1}/{total}] simulating {candidate.id}", file=sys.stderr)

    try:
        runner = exhaustive.oracle if args.oracle else exhaustive.search
        output = runner(
            spec, cluster, islands, profiles, predictor,
            cache=cache,
            gpu_memory_utilization=args.gpu_memory_utilization,
            activation_reserve_gb=args.activation_reserve_gb,
            provenance=provenance,
            progress=progress,
        )
    finally:
        predictor.close()

    if cache is not None:
        output.provenance["envelope_cache"] = cache.stats()

    print(render(output))
    if args.output:
        _write_output(output, Path(args.output))
        print(f"\nwrote {args.output}")
    return 0 if output.feasible else 3


def cmd_validate_plan(args: argparse.Namespace) -> int:
    """Re-simulate a saved plan against a specific dataset."""
    raw = yaml.safe_load(Path(args.plan).read_text())
    output = PlannerOutput.model_validate(raw)
    if output.recommended is None:
        print("error: this plan file carries no recommendation to validate", file=sys.stderr)
        return 1

    spec = load_service_spec(args.service)
    cluster = load_cluster_spec(args.cluster)
    profiles = load_profiles_for(cluster, args.root)
    islands = {i.id: i for i in detect_islands(cluster, profiles)}

    from planner.util.workload import WorkloadTrace

    dataset = Path(args.dataset)
    num = sum(1 for _ in dataset.open())
    trace = WorkloadTrace(dataset, num, args.seed, 0, 0, 0.0)

    predictor = LLMServingSimPredictor(
        trace,
        timeout_s=args.timeout,
        gpu_memory_utilization=args.gpu_memory_utilization,
        activation_reserve_gb=args.activation_reserve_gb,
    )
    try:
        result = predictor.predict(
            output.recommended.plan.candidate, spec, cluster, islands, profiles
        )
    finally:
        predictor.close()

    print(f"plan     : {output.recommended.plan.plan_id}")
    print(f"dataset  : {dataset} ({num} requests)")
    print(f"outcome  : {result.outcome.value}")
    if not result.ok:
        print(f"detail   : {result.detail}")
        return 1

    from planner.render import render_metrics

    replayed = output.recommended.plan.model_copy(update={"predicted": result.metrics})
    print("\nre-simulated metrics:")
    print(render_metrics(replayed))
    print("\noriginal prediction:")
    print(render_metrics(output.recommended.plan))
    return 0


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m planner",
        description="HeteroPilot: plan LLM serving on heterogeneous GPU/NPU clusters.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser(
        "inspect-cluster",
        help="List execution islands, TP candidates and model compatibility.",
    )
    inspect.add_argument("--cluster", required=True, type=Path)
    inspect.add_argument("--service", type=Path,
                         help="ServiceSpec YAML; enables compatibility and memory-fit checks.")
    inspect.add_argument("--root", type=Path, default=Path("."))
    inspect.set_defaults(func=cmd_inspect_cluster)

    plan = sub.add_parser("plan", help="Generate, simulate and rank deployment candidates.")
    plan.add_argument("--service", required=True, type=Path)
    plan.add_argument("--cluster", required=True, type=Path)
    plan.add_argument("--root", type=Path, default=Path("."))
    plan.add_argument("--output", type=Path, help="Write the PlannerOutput YAML here.")
    plan.add_argument("--num-requests", type=int, default=DEFAULT_TRACE_REQUESTS,
                      help="Requests in the generated workload trace.")
    plan.add_argument("--seed", type=int, default=DEFAULT_SEED,
                      help="Workload generator seed. The simulator itself has no seed; "
                           "reproducibility comes from here (deviations D5).")
    plan.add_argument("--timeout", type=float, default=900.0,
                      help="Per-candidate simulator wall-clock budget, seconds.")
    plan.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                      help="Fraction of device memory the runtime reserves (deviations D10).")
    plan.add_argument("--activation-reserve-gb", type=float, default=0.0,
                      help="Extra per-device memory withheld for activations and CUDA graphs.")
    plan.add_argument("--cache-dir", type=Path, help="PerformanceEnvelope cache directory.")
    plan.add_argument("--work-dir", type=Path, help="Where traces and sim artifacts go.")
    plan.add_argument("--keep-artifacts", action="store_true")
    plan.add_argument("--oracle", action="store_true",
                      help="Disable bound-based pruning and simulate every candidate.")
    plan.add_argument("--quiet", action="store_true")
    plan.set_defaults(func=cmd_plan)

    validate = sub.add_parser("validate-plan",
                              help="Re-simulate a saved plan against a specific dataset.")
    validate.add_argument("--plan", required=True, type=Path)
    validate.add_argument("--service", required=True, type=Path)
    validate.add_argument("--cluster", required=True, type=Path)
    validate.add_argument("--dataset", required=True, type=Path)
    validate.add_argument("--root", type=Path, default=Path("."))
    validate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    validate.add_argument("--timeout", type=float, default=900.0)
    validate.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    validate.add_argument("--activation-reserve-gb", type=float, default=0.0)
    validate.set_defaults(func=cmd_validate_plan)

    for name, phase in _NOT_YET.items():
        p = sub.add_parser(name, help=f"({phase}) not implemented yet")
        p.set_defaults(func=_unimplemented, _name=name, _phase=phase)

    return parser


def _unimplemented(args: argparse.Namespace) -> int:
    print(f"'{args._name}' is scheduled for {args._phase} and is not implemented yet.",
          file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (SpecError, InventoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
