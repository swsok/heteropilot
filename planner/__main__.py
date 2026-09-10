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

from planner import sweep as sweep_mod
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
from planner.perf_envelope import PerfEnvelope, find_envelope
from planner.plan import DeploymentPlan, PlannerOutput
from planner.predictor.calibration import load_accuracy_domains
from planner.predictor.llmservingsim import LLMServingSimPredictor
from planner.render import render, render_deployment_handle, render_deployment_metrics
from planner.spec import ServiceSpec, SpecError, load_service_spec
from planner.topology import TopologyGraph
from planner.util import memory as memutil
from planner.util import provenance as prov
from planner.util.workload import generate_trace

_NOT_YET: dict[str, str] = {}

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


def _load_envelopes(
    cluster: ClusterSpecV2, profiles: dict[str, AcceleratorProfile], spec: ServiceSpec
) -> dict[str, PerfEnvelope]:
    """Measured envelopes for the hardware this cluster actually contains.

    Keyed by `sim_hardware`, which is what an operating point is attributed to.
    Hardware without a curve is simply absent -- the prefilter then leaves its
    candidates alone.
    """
    out: dict[str, PerfEnvelope] = {}
    dtype = spec.service.dtype
    for node in cluster.nodes:
        for accel in node.accelerators:
            profile = profiles.get(accel.model)
            if profile is None or profile.sim_hardware is None:
                continue
            hw = profile.sim_hardware
            if hw in out:
                continue
            env = find_envelope(hw, spec.model, dtype, 1)
            if env is not None:
                out[hw] = env
    return out

def cmd_plan(args: argparse.Namespace) -> int:
    """One plan, or one per rate when --rps is given."""
    if not getattr(args, "rps", None):
        return _plan_once(args)
    return _plan_sweep(args)


def _merge_caveats(rates: list[float], outputs: list[PlannerOutput]) -> list[str]:
    """One entry per distinct caveat; rate-specific ones say which rate."""
    shared, tagged = [], []
    counts: dict[str, int] = {}
    for out in outputs:
        for c in out.caveats:
            counts[c] = counts.get(c, 0) + 1
    seen: set[str] = set()
    for rate, out in zip(rates, outputs, strict=False):
        for c in out.caveats:
            if counts[c] == len(outputs):
                if c not in seen:
                    seen.add(c)
                    shared.append(c)
            else:
                tagged.append(f"[rps {rate}] {c}")
    return shared + tagged


def _plan_sweep(args: argparse.Namespace) -> int:
    """Run the planner once per arrival rate and report where the answer changes.

    Each rate gets its own output file and work directory; the envelope cache is
    shared deliberately. Its key includes the trace digest and RPS changes the
    arrival times, so entries separate by rate on their own -- verified rather
    than assumed (`planner/envelope.py:key_for` has no RPS field, and needs none).
    """
    import copy as _copy

    rates = [float(v) for v in str(args.rps).split(",") if v.strip()]
    if not rates:
        print("error: --rps parsed to no values", file=sys.stderr)
        return 2
    if sorted(rates) != rates:
        print("error: --rps must be ascending so adjacency means what the "
              "switchover table says it means", file=sys.stderr)
        return 2

    base_out = Path(args.output) if args.output else None
    rows, outputs = [], []
    for rate in rates:
        sub = _copy.copy(args)
        sub.rps = None
        sub.quiet = True
        sub._rps_override = rate
        tag = str(rate).replace(".", "p")
        sub.output = str(base_out.with_name(f"{base_out.stem}_rps{tag}{base_out.suffix}")) \
            if base_out else None
        if getattr(args, "work_dir", None):
            sub.work_dir = str(Path(args.work_dir) / f"rps{tag}")
        print(f"\n=== rps {rate} ===", file=sys.stderr)
        out = _plan_once(sub, return_output=True)
        if not isinstance(out, PlannerOutput):
            return int(out)
        outputs.append(out)
        rows.append(sweep_mod.row_for(rate, out))

    sweep = sweep_mod.SweepOutput(
        service_model=outputs[0].service_model,
        cluster_id=outputs[0].cluster_id,
        rps_values=rates,
        switchover=rows,
        crossovers=sweep_mod.find_crossovers(rows),
        provenance={"per_rps_outputs": [r.plan_id for r in rows],
                    "shared_cache_dir": getattr(args, "cache_dir", None)},
        # Per-rate caveats are tagged with their rate. Untagged, three lines each
        # saying "306 candidate(s)" read as 918.
        caveats=_merge_caveats(rates, outputs),
    )
    print(sweep_mod.render(sweep))
    if base_out:
        sweep_path = base_out.with_name(f"{base_out.stem}_switchover{base_out.suffix}")
        sweep_path.write_text(
            yaml.safe_dump(sweep.model_dump(mode="json"), sort_keys=False))
        print(f"\nwrote {sweep_path}")
    return 0 if any(r.feasible for r in rows) else 3


def _plan_once(args: argparse.Namespace, return_output: bool = False):
    if args.oracle and args.top_k is not None:
        print("error: --oracle simulates everything; --top-k is a heuristic subset - "
              "they are mutually exclusive", file=sys.stderr)
        return 1
    spec = load_service_spec(args.service)
    override = getattr(args, "_rps_override", None)
    if override is not None:
        # A ServiceSpec is frozen-ish by convention; copy rather than mutate so a
        # sweep cannot leak one rate's rate into another's provenance.
        spec = spec.model_copy(deep=True)
        spec.traffic.arrival_rate_rps = override
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
            "enable_pd": args.enable_pd,
            "surrogate": args.surrogate if args.top_k is not None else None,
            "top_k": args.top_k,
        },
    )
    missing = prov.note_missing(provenance)
    if missing:
        print(f"warning: provenance fields could not be determined: {', '.join(missing)}",
              file=sys.stderr)

    topology = TopologyGraph(cluster)
    reduction = topology.reduce_for_simulator(islands)
    if args.topology_level == 2:
        provenance["topology"] = topology.reduce_for_simulator_perdim(islands).as_provenance()
    else:
        provenance["topology"] = reduction.as_provenance()

    cache = None
    if args.cache_dir:
        cache = EnvelopeCache(
            args.cache_dir,
            spec,
            accelerator_of={i.id: i.accelerator_model for i in islands},
            link_bw_gbps=reduction.link_bw_gbps,
            trace_digest=prov.hash_file(trace.path),
            topology_level=args.topology_level,
        )

    predictor = LLMServingSimPredictor(
        trace,
        work_dir=work_root / "sims",
        timeout_s=args.timeout,
        gpu_memory_utilization=args.gpu_memory_utilization,
        activation_reserve_gb=args.activation_reserve_gb,
        keep_artifacts=args.keep_artifacts,
        topology_level=args.topology_level,
    )

    def progress(i: int, total: int, candidate) -> None:
        if not args.quiet:
            print(f"  [{i + 1}/{total}] simulating {candidate.id}", file=sys.stderr)

    surrogate = None
    if args.top_k is not None:
        from planner.optimizer.surrogate import AnalyticalRooflineRanker
        surrogate = AnalyticalRooflineRanker()

    accuracy_domains = load_accuracy_domains(args.root) if args.accuracy_domain else None
    envelopes = (_load_envelopes(cluster, profiles, spec)
                 if args.envelope_prefilter else None)
    provenance["accuracy_domain"] = sorted(accuracy_domains) if accuracy_domains else None
    provenance["envelope_prefilter"] = sorted(envelopes) if envelopes else None
    # Both margins are recorded even when only one binds, so a plan says what it
    # was checked against rather than only what won (STEP 4.3).
    provenance["manual_margin_percent"] = {
        "ttft": args.ttft_margin_percent, "tpot": args.tpot_margin_percent,
    }

    try:
        runner = exhaustive.oracle if args.oracle else exhaustive.search
        output = runner(
            spec, cluster, islands, profiles, predictor,
            enable_pd=args.enable_pd,
            cache=cache,
            gpu_memory_utilization=args.gpu_memory_utilization,
            activation_reserve_gb=args.activation_reserve_gb,
            surrogate=surrogate,
            top_k=args.top_k,
            envelopes=envelopes,
            accuracy_domains=accuracy_domains,
            ttft_margin_percent=args.ttft_margin_percent,
            tpot_margin_percent=args.tpot_margin_percent,
            max_workers=args.workers,
            provenance=provenance,
            progress=progress,
        )
    finally:
        predictor.close()

    if cache is not None:
        output.provenance["envelope_cache"] = cache.stats()

    if not args.quiet or not return_output:
        print(render(output))
    if args.output:
        _write_output(output, Path(args.output))
        if not return_output:
            print(f"\nwrote {args.output}")
    if return_output:
        return output
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
# deploy / status (Phase 4)
# --------------------------------------------------------------------------

def _load_deployment_plan(path: Path) -> DeploymentPlan:
    """Load a DeploymentPlan from a plan file.

    Accepts both a full `PlannerOutput` (what `plan --output` writes, we take its
    recommendation) and a bare `DeploymentPlan`.
    """
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise SpecError(f"{path}: expected a YAML mapping at the top level")
    if "candidate" in raw and "predicted" in raw:
        return DeploymentPlan.model_validate(raw)
    output = PlannerOutput.model_validate(raw)
    if output.recommended is None:
        raise SpecError(f"{path}: this plan file carries no recommendation to deploy")
    return output.recommended.plan


def cmd_deploy(args: argparse.Namespace) -> int:
    from planner.deploy import DeploymentError, VllmCudaBackend

    plan = _load_deployment_plan(args.plan)
    cluster = load_cluster_spec(args.cluster)
    profiles = load_profiles_for(cluster, args.root)
    islands = {i.id: i for i in detect_islands(cluster, profiles)}

    backend = VllmCudaBackend(
        root=args.root, host=args.host, port=args.port, profiles=profiles
    )
    problems = backend.validate(plan, cluster, islands)

    print(f"plan    : {plan.plan_id} ({plan.model})")
    print(f"cluster : {cluster.cluster_id}")
    print(f"backend : {backend.name} (host={args.host}, port={args.port})")
    print()

    from planner.deploy.vllm_cuda import build_serve_command

    print("resolved serve command(s):")
    for assignment in plan.candidate.assignments:
        island = islands.get(assignment.island_id)
        if island is None:
            print(f"  {assignment.island_id}: UNKNOWN island; cannot resolve")
            continue
        command = build_serve_command(plan, assignment, island, port=args.port)
        print(f"  [{island.id}] devices -> {command.env['CUDA_VISIBLE_DEVICES']}")
        print(f"    {command.as_shell()}")
    print()

    if problems:
        print("validation problems:")
        for p in problems:
            print(f"  - {p}")
        print()
    else:
        print("validation: OK")
        print()

    if args.dry_run:
        print("dry run: nothing was launched. Re-run with --no-dry-run to launch locally.")
        return 0 if not problems else 3

    if problems:
        print("error: refusing to launch a plan with validation problems", file=sys.stderr)
        return 3
    try:
        handle = backend.launch(plan, cluster, islands)
    except (DeploymentError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("launched:")
    print(render_deployment_handle(handle))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from planner.deploy import DeploymentError, VllmCudaBackend

    backend = VllmCudaBackend(root=args.root)
    try:
        handle = backend.read_handle(args.deployment)
    except DeploymentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(render_deployment_handle(handle))
    print()
    if not backend.is_running(args.deployment):
        print(f"deployment '{args.deployment}' is not running (no live process).")
        return 0
    try:
        metrics = backend.metrics(args.deployment)
    except DeploymentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("live metrics:")
    print(render_deployment_metrics(metrics))
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    from planner.deploy import DeploymentError, VllmCudaBackend

    backend = VllmCudaBackend(root=args.root)
    was_running = backend.is_running(args.deployment)
    try:
        backend.stop(args.deployment)
    except DeploymentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if was_running:
        print(f"stopped deployment '{args.deployment}'.")
    else:
        print(f"deployment '{args.deployment}' was not running; cleared its pidfile.")
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
    plan.add_argument("--topology-level", type=int, choices=[1, 2], default=1,
                      help="Network model level compiled into the simulator: 1 (default) "
                      "collapses the link graph to one scalar link_bw; 2 emits a "
                      "per-dimension list so intra-island (TP) collectives keep their real "
                      "bandwidth instead of the slowest cross-instance link. Level 2 changes "
                      "predictions only for multi-island placements (deviations D3).")
    plan.add_argument("--enable-pd", action=argparse.BooleanOptionalAction, default=False,
                      help="Also enumerate Prefill/Decode-split candidates across islands "
                           "(work order §5.3). Off by default; note it grows the candidate "
                           "space roughly quadratically in the number of islands.")
    plan.add_argument("--top-k", type=int, default=None,
                      help="Stage-6 surrogate top-K (§5.4): score all survivors with the "
                           "analytical roofline and fully simulate only the K best. HEURISTIC "
                           "- it can drop the optimum (measured by exp_surrogate.py). Default: "
                           "simulate all survivors. Mutually exclusive with --oracle.")
    plan.add_argument("--surrogate", choices=["roofline"], default="roofline",
                      help="Surrogate ranker for --top-k. 'roofline' reuses the greedy "
                           "analytical proxy (a learned/xgboost ranker is a corpus-gated "
                           "future option).")
    plan.add_argument("--workers", type=int, default=None,
                      help="Concurrent candidate simulations (default: ~half the CPUs, "
                           "capped 32). Each candidate is an isolated subprocess, so this "
                           "only speeds the search up - the result is byte-identical. Use "
                           "--workers 1 to force sequential.")
    plan.add_argument("--quiet", action="store_true")
    plan.add_argument("--accuracy-domain", action="store_true",
                      help="size each candidate's SLO margin from its own operating "
                           "point, using the measured accuracy domains in "
                           "profiles/calibration/. Opt-in: the default path applies "
                           "no automatic margin and its output is unchanged.")
    plan.add_argument("--envelope-prefilter", action="store_true",
                      help="reject, before simulating, candidates whose predicted "
                           "operating point falls outside the hardware's measured "
                           "performance envelope. Epistemic, not a sound bound: it "
                           "can drop the optimum, and the rejection says so.")
    plan.add_argument("--rps", default=None,
                      help="comma-separated arrival rates to sweep, e.g. "
                           "'1,3.3,10,20'. Runs one plan per rate and emits the "
                           "switchover table and any backend crossovers.")
    plan.add_argument("--tpot-margin-percent", type=float, default=0.0,
                      help="a manual SLO margin floor. When --accuracy-domain also "
                           "applies, the LARGER of the two is used and both are "
                           "recorded in provenance.")
    plan.add_argument("--ttft-margin-percent", type=float, default=0.0)
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

    deploy = sub.add_parser(
        "deploy",
        help="Validate a plan against a cluster and (optionally) launch it locally.",
    )
    deploy.add_argument("--plan", required=True, type=Path)
    deploy.add_argument("--cluster", required=True, type=Path)
    deploy.add_argument("--root", type=Path, default=Path("."))
    deploy.add_argument("--host", default="local",
                        help="'local' launches a subprocess; any other value is an SSH "
                             "hook point and is not implemented in this increment.")
    deploy.add_argument("--port", type=int, default=8000)
    deploy.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True,
                        help="Print the resolved command without launching (default). "
                             "Pass --no-dry-run to launch locally.")
    deploy.set_defaults(func=cmd_deploy)

    status = sub.add_parser(
        "status",
        help="Read a launched deployment's live TTFT/TPOT/throughput/power.",
    )
    status.add_argument("--deployment", required=True,
                        help="Deployment id (defaults to the plan_id at launch time).")
    status.add_argument("--root", type=Path, default=Path("."))
    status.set_defaults(func=cmd_status)

    stop = sub.add_parser("stop", help="Stop a launched deployment and free its devices.")
    stop.add_argument("--deployment", required=True, help="Deployment id.")
    stop.add_argument("--root", type=Path, default=Path("."))
    stop.set_defaults(func=cmd_stop)

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
