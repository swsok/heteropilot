"""HeteroPilot CLI (work order §6).

`inspect-cluster` (Phase 1) and `plan` / `validate-plan` (Phase 2) are live.
`deploy` / `status` arrive in Phase 4 and are declared so `--help` shows the
intended shape.
"""

from __future__ import annotations

import argparse
import dataclasses
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
from planner.predictor.accuracy_domain import served_concurrency_from_sim
from planner.predictor.calibration import load_accuracy_domains, load_calibrations
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

#: Fields the uncertainty work adds to existing structures, keyed by a field
#: that identifies the structure they hang off. The dump has no `exclude_none` -
#: other optional fields DO appear as null and the frozen outputs contain them -
#: so these are removed one by one when the caller did not opt in, which is what
#: keeps the flagless YAML byte-identical (rule A4). `served_concurrency` needs
#: this even though `uncertain_inputs` does not: the predictor fills it on every
#: real run, so without the strip it would appear with a value, not as a null,
#: in output that must not change.
_UNCERTAINTY_FIELDS: tuple[tuple[str, str], ...] = (
    ("p50_ttft_ms", "served_concurrency"),   # PredictedMetrics
    ("plan_id", "margin_basis"),             # DeploymentPlan
)


def _strip_uncertainty_fields(node: object) -> None:
    """Recursively drop the opt-in uncertainty fields from a dumped output."""
    if isinstance(node, dict):
        for marker, field in _UNCERTAINTY_FIELDS:
            if marker in node:
                node.pop(field, None)
        for value in node.values():
            _strip_uncertainty_fields(value)
    elif isinstance(node, list):
        for value in node:
            _strip_uncertainty_fields(value)


def _write_output(output: PlannerOutput, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = output.model_dump(mode="json")
    if output.uncertain_inputs is None:
        # Dropped rather than emitted as `uncertain_inputs: null`. The dump has
        # no exclude_none (other optional fields DO appear as null and the
        # golden outputs contain them), so the new keys are removed on their own
        # to keep the default path's YAML byte-identical (absolute rule A4).
        data.pop("uncertain_inputs", None)
        _strip_uncertainty_fields(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _build_uncertainty(args, spec, cluster, profiles, islands, provenance):
    """Wire up the accuracy-domain machinery for one `plan` run (STEP A4).

    Returns (accuracy_domains, margin_policy, registry). All None when
    --accuracy-domain was not given, which is what keeps the default path
    byte-identical (rule A4). With the flag and no files, every domain under
    profiles/calibration/ is read (the rps STEP 4 behaviour); with files, those
    and nothing else.
    """
    if args.accuracy_domain is None:
        return None, None, None

    from planner.envelope import is_canonical_bucket, shape_of_bucket, workload_bucket
    from planner.optimizer.margin import AccuracyDomainMargin
    from planner.uncertainty import build_registry, load_costs, load_grades

    paths = [Path(p) for p in args.accuracy_domain] or None
    # Raises on two domains for one hardware: the policy never picks silently.
    domains = load_accuracy_domains(args.root, paths)
    # The scalar calibrations too: the registry lists their error, and a
    # hardware with a fitted bucket but no domain falls back to it.
    calibration = load_calibrations(args.root, paths)

    requested = workload_bucket(spec)
    bucket = requested
    override = None
    if args.calibration_bucket:
        if not is_canonical_bucket(args.calibration_bucket):
            raise SpecError(
                f"--calibration-bucket {args.calibration_bucket!r} is not a canonical "
                f"bucket key (§2.4.1); it must look like 'in_lt1024-out_ge512-rps_lt20'"
            )
        bucket = args.calibration_bucket
        override = {"requested": requested, "used": bucket}
    shape = shape_of_bucket(bucket)

    grades, costs = load_grades(), load_costs()
    registry = build_registry(
        cluster, profiles, islands, calibration, grades, spec, costs,
    )

    provenance["uncertainty"] = {
        "policy": "accuracy_domain",
        "calibration_files": (
            [str(p) for p in paths] if paths else f"{args.root}/profiles/calibration/*.yaml"
        ),
        "bucket": bucket,
        "shape": shape,
        "grades_digest": grades.digest,
        "costs_digest": costs.digest,
        "domains": {
            hardware: {
                "conc_min": d.conc_min, "conc_max": d.conc_max,
                "outside_domain": d.outside_domain,
                "workload_shape": d.workload_shape or None,
            }
            for hardware, d in sorted(domains.items())
        },
    }
    if override is not None:
        provenance["uncertainty"]["bucket_override"] = override

    policy = AccuracyDomainMargin(
        domains, shape=shape, calibration=calibration, bucket=bucket,
        ttft_floor=args.ttft_margin_percent, tpot_floor=args.tpot_margin_percent,
    )
    return domains, policy, registry


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
    if args.calibration_bucket and args.accuracy_domain is None:
        print("error: --calibration-bucket only means something with --accuracy-domain",
              file=sys.stderr)
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

    try:
        accuracy_domains, margin_policy, registry = _build_uncertainty(
            args, spec, cluster, profiles, islands, provenance
        )
    except (SpecError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

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
        from planner.optimizer.surrogate import (
            AnalyticalRooflineRanker,
            BinnedRooflineRanker,
        )
        # `roofline` stays selectable so a published run reproduces; `binned` is
        # the default because it weakly dominates it over sixteen corpora
        # (docs/surrogate_topk_regret.md, deviations D30).
        surrogate = (BinnedRooflineRanker() if args.surrogate == "binned"
                     else AnalyticalRooflineRanker())

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
            margin_policy=margin_policy,
            max_workers=args.workers,
            provenance=provenance,
            progress=progress,
        )
        if registry is not None:
            output.uncertain_inputs = registry
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


# --------------------------------------------------------------------------
# fit-accuracy-domain (uncertainty work order STEP A2)
# --------------------------------------------------------------------------

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000


@dataclasses.dataclass(frozen=True)
class _RunStats:
    """TTFT / TPOT distributions (ms) and served concurrency for one run."""

    ttft: list[float]
    tpot: list[float]
    served: float
    wall_s: float
    n: int
    #: Only a real bench run records what the client asked for; None for sim.
    requested_concurrency: int | None = None


def _read_real_run(path: Path) -> _RunStats:
    """TTFT / TPOT / latency (ms) and served concurrency from a bench JSON.

    Schema is `bench_furiosa_endpoint.py`'s: top-level `wall_s` and
    `concurrency`, plus `per_request[]` with `ttft_ns` / `tpot_ns` /
    `latency_ns`. Failed rows are dropped, never counted as zero.
    """
    import json

    report = json.loads(path.read_text())
    ttft, tpot, latency_s = [], [], []
    for row in report["per_request"]:
        if row.get("error") or row.get("ttft_ns") is None:
            continue
        ttft.append(row["ttft_ns"] / NS_PER_MS)
        if row.get("tpot_ns"):
            tpot.append(row["tpot_ns"] / NS_PER_MS)
        latency_s.append(row["latency_ns"] / NS_PER_S)
    wall_s = float(report["wall_s"])
    return _RunStats(
        ttft=ttft, tpot=tpot,
        served=served_concurrency_from_sim(latency_s, wall_s),
        wall_s=wall_s, n=len(ttft),
        requested_concurrency=report.get("concurrency"),
    )


def _csv_float(row: dict[str, str], name: str) -> float | None:
    """A CSV cell as a float; None when the column is absent or blank."""
    raw = (row.get(name) or "").strip()
    return float(raw) if raw else None


def _read_sim_run(path: Path) -> _RunStats:
    """The same quantities from a simulator per-request CSV."""
    import csv as _csv

    ttft, tpot, latency_s, arrivals, ends = [], [], [], [], []
    with path.open(newline="") as handle:
        for row in _csv.DictReader(handle):
            t, p = _csv_float(row, "TTFT"), _csv_float(row, "TPOT")
            lat = _csv_float(row, "latency")
            arrival, end = _csv_float(row, "arrival"), _csv_float(row, "end_time")
            if t is not None:
                ttft.append(t / NS_PER_MS)
            if p:
                tpot.append(p / NS_PER_MS)
            if lat is not None:
                latency_s.append(lat / NS_PER_S)
            if arrival is not None and end is not None:
                arrivals.append(arrival)
                ends.append(end)
    wall_s = (max(ends) - min(arrivals)) / NS_PER_S if arrivals else 0.0
    return _RunStats(
        ttft=ttft, tpot=tpot,
        served=served_concurrency_from_sim(latency_s, wall_s),
        wall_s=wall_s, n=len(ttft),
    )


def _pair_rows(
    real: _RunStats, sim: _RunStats
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """`(sim, real)` summary rows per metric for one (real, sim) run pair.

    Deliberately routed through the SAME pipeline the committed calibrations
    were fitted with - `bench.core.plots.write_summary` then
    `parse_validation_summary` - which is what
    `experiments/scripts/compare_rngd_sim_vs_real.py` does, so a domain point and
    the scalar fit it sits beside come from one estimator.
    """
    import tempfile as _tempfile

    from bench.core.plots import write_summary
    from planner.predictor.calibration import parse_validation_summary

    with _tempfile.TemporaryDirectory() as tmp:
        summary = write_summary(
            Path(tmp), "pair",
            bench_ttft=real.ttft, sim_ttft=sim.ttft,
            bench_tpot=real.tpot, sim_tpot=sim.tpot,
            bench_latency=[], sim_latency=[],
        )
        pairs = parse_validation_summary(summary.read_text())
    return (pairs.ttft, pairs.tpot)


def _signed_error_pct(rows: list[tuple[float, float]]) -> float | None:
    """Mean `(sim - real) / real * 100` over the summary rows: the
    `AccuracyPoint` convention, where NEGATIVE means the simulator is optimistic
    (the direction that produced D22)."""
    vals = [(sim - real) / real * 100.0 for sim, real in rows if real > 0]
    if not vals:
        return None
    return float(sum(vals)) / len(vals)


def cmd_fit_accuracy_domain(args: argparse.Namespace) -> int:
    """Turn paired (real, sim) runs into a hardware's `accuracy_domain` (§2.4)."""
    from planner.envelope import is_canonical_shape, workload_shape
    from planner.predictor.calibration import (
        AccuracyDomain,
        AccuracyPoint,
        CalibrationModel,
        HardwareCalibration,
        load_calibration,
        save_calibration,
    )

    if args.service and args.shape:
        print("error: give at most one of --service and --shape", file=sys.stderr)
        return 1
    shape = workload_shape(load_service_spec(args.service)) if args.service else (args.shape or "")
    if shape and not is_canonical_shape(shape):
        print(f"error: --shape {shape!r} is not a canonical shape; it must look like "
              f"'in_lt1024-out_ge512'", file=sys.stderr)
        return 1

    real_paths = [Path(p) for p in args.real]
    sim_paths = [Path(p) for p in args.sim]
    if len(real_paths) != len(sim_paths):
        print(
            f"error: --real has {len(real_paths)} file(s) and --sim has {len(sim_paths)}; "
            f"they are paired by position and must match",
            file=sys.stderr,
        )
        return 1

    out = Path(args.out)
    # Absolute rule A3: a measured artifact is read-only. Refuse rather than
    # ask, so no invocation of this command can ever destroy a calibration.
    if out.exists() and not args.overwrite:
        print(f"error: {out} exists; refusing to overwrite (pass --overwrite to replace)",
              file=sys.stderr)
        return 1
    if args.base is not None and out.resolve() == Path(args.base).resolve():
        print("error: --out must differ from --base; the base calibration is read-only",
              file=sys.stderr)
        return 1

    model = load_calibration(args.base) if args.base else CalibrationModel.identity()
    cal = model.hardware.get(args.hardware) or HardwareCalibration(hardware=args.hardware)
    if cal.accuracy_domain is not None and not args.replace_domain:
        print(f"error: {args.hardware} in {args.base} already carries an accuracy domain "
              f"({len(cal.accuracy_domain.points)} points); pass --replace-domain to "
              f"replace it", file=sys.stderr)
        return 1

    points: list[AccuracyPoint] = []
    print("=== paired runs ===")
    for real_path, sim_path in zip(real_paths, sim_paths, strict=True):
        real = _read_real_run(real_path)
        sim = _read_sim_run(sim_path)
        ttft_rows, tpot_rows = _pair_rows(real, sim)
        tpot_err = _signed_error_pct(tpot_rows)
        ttft_err = None if args.metric == "tpot" else _signed_error_pct(ttft_rows)
        if tpot_err is None:
            # AccuracyPoint requires TPOT: a point without it is not a domain point.
            print(f"  SKIP {real_path.name}: no TPOT data on both sides")
            continue
        served = real.served
        if any(abs(pt.conc - served) < 1e-9 for pt in points):
            print(f"error: two pairs land at served concurrency {served:.4g}; a domain "
                  f"cannot carry two errors for one operating point", file=sys.stderr)
            return 1
        gap = (sim.served - served) / served * 100.0 if served > 0 else float("nan")
        note = (
            f"real n={real.n} wall={real.wall_s:.3f}s"
            + (f" requested c{real.requested_concurrency}"
               if real.requested_concurrency is not None else "")
            + f" ({real_path.name}); sim n={sim.n} wall={sim.wall_s:.3f}s served "
            f"L={sim.served:.3f} ({sim_path.name}); conc gap {gap:+.1f}%"
        )
        points.append(AccuracyPoint(
            conc=served, tpot_err_pct=tpot_err, ttft_err_pct=ttft_err, note=note,
        ))
        print(
            f"  L={served:8.3f}  tpot_err={tpot_err:+7.2f}%  ttft_err="
            f"{'n/a' if ttft_err is None else f'{ttft_err:+.2f}%'}  "
            f"sim served L={sim.served:.3f} (gap {gap:+.1f}%)"
        )

    if not points:
        print("error: no usable pairs; nothing to write", file=sys.stderr)
        return 1
    points.sort(key=lambda pt: pt.conc)

    fitted_at = args.fitted_at_concurrency
    if fitted_at is None:
        fitted_at = points[0].conc
        print(f"  NOTE: --fitted-at-concurrency not given; recording the lowest measured "
              f"point, {fitted_at:.4g}, as where the profile was validated")

    cal.accuracy_domain = AccuracyDomain(
        fitted_at_concurrency=fitted_at,
        points=points,
        outside_domain=args.outside_domain,
        source=args.source,
        note=args.note,
        workload_shape=shape,
        arrival_process=args.arrival_process,
    )
    model.hardware[args.hardware] = cal
    fitted = model.provenance.setdefault("accuracy_domain", {})
    fitted[args.hardware] = {
        "pairs": [f"{r} + {s_}" for r, s_ in zip(real_paths, sim_paths, strict=True)],
        "base": str(args.base) if args.base else None,
        "concurrency": "served, sum(latency)/wall of the REAL run (D22)",
        "metrics_recorded": args.metric,
        "shape": shape or None,
        "arrival_process": args.arrival_process,
        "error_convention": (
            "(sim - real) / real * 100 over write_summary's stat rows; negative = "
            "the simulator is optimistic"
        ),
    }

    save_calibration(model, out)
    lo, hi = points[0].conc, points[-1].conc
    print(f"\nwrote {out}")
    print(f"  {args.hardware}: {len(points)} point(s), domain [{lo:.4g}, {hi:.4g}], "
          f"outside_domain={args.outside_domain}"
          + (f", shape {shape}" if shape else ", unscoped"))
    if len(points) == 1:
        print("  NOTE: one point is a domain of zero width - every other concurrency "
              "is unmeasured under `refuse`. Add points before relying on this.")
    return 0


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
    plan.add_argument("--surrogate", choices=["binned", "roofline"], default="binned",
                      help="Surrogate ranker for --top-k. 'roofline' reuses the greedy "
                           "analytical proxy (a learned/xgboost ranker is a corpus-gated "
                           "future option).")
    plan.add_argument("--workers", type=int, default=None,
                      help="Concurrent candidate simulations (default: ~half the CPUs, "
                           "capped 32). Each candidate is an isolated subprocess, so this "
                           "only speeds the search up - the result is byte-identical. Use "
                           "--workers 1 to force sequential.")
    plan.add_argument("--quiet", action="store_true")
    plan.add_argument("--accuracy-domain", nargs="*", metavar="CALIBRATION_YAML",
                      default=None,
                      help="size each candidate's SLO margin from its own operating "
                           "point, using the measured accuracy domains in "
                           "profiles/calibration/ (or in exactly the files given). A "
                           "candidate whose operating point a domain under `refuse` "
                           "does not cover, or whose hardware has no calibration, is "
                           "rejected as outside_calibration_domain - unmeasured, not "
                           "infeasible (uncertainty work order §2.4). Also emits the "
                           "uncertain-input registry. Opt-in: the default path applies "
                           "no automatic margin and its output is unchanged.")
    plan.add_argument("--calibration-bucket", default=None, metavar="CANONICAL_KEY",
                      help="Override the workload bucket the scalar calibration is "
                           "looked up under (and the token-mix shape a scoped domain "
                           "must match). Must be a canonical key (in_*-out_*-rps_*); a "
                           "human label is never accepted. Recorded in provenance and "
                           "warned about in the output, because it asserts that a "
                           "different workload's error applies here.")
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

    fit_domain = sub.add_parser(
        "fit-accuracy-domain",
        help="Fit a hardware's error-vs-served-concurrency accuracy domain from paired "
             "real/sim runs.",
    )
    fit_domain.add_argument("--real", nargs="+", required=True,
                            help="bench JSON(s); paired with --sim by position")
    fit_domain.add_argument("--sim", nargs="+", required=True,
                            help="simulator per-request CSV(s), same order as --real")
    fit_domain.add_argument("--hardware", required=True,
                            help="calibration hardware label, e.g. RNGD-CARD")
    fit_domain.add_argument(
        "--service", default=None,
        help="Service spec whose token-mix shape (in_*-out_*) scopes the domain; "
             "candidates for a different shape are refused (§2.4.1).")
    fit_domain.add_argument(
        "--shape", default=None, metavar="IN_OUT",
        help="The shape directly, instead of --service. Omit both for a domain that "
             "applies to any workload on this hardware.")
    fit_domain.add_argument(
        "--arrival-process", choices=["open_loop", "closed_loop", "unknown"],
        default="unknown",
        help="How load was offered when the REAL side was measured. `closed_loop` (a "
             "fixed number of clients in flight) means there is no arrival rate and, "
             "per D19, its TTFT does not transfer to an open-loop deployment.")
    fit_domain.add_argument(
        "--metric", choices=["both", "tpot"], default="both",
        help="Record TTFT too, or TPOT only. Use `tpot` when the TTFT comparison is not "
             "like-for-like - e.g. a burst sim against a closed-loop bench, where queued "
             "requests inflate sim TTFT by orders of magnitude while TPOT stays "
             "comparable. TPOT is always recorded: a point without it is not a point.")
    fit_domain.add_argument(
        "--outside-domain", choices=["refuse", "widen_error_bars"], default="refuse",
        help="What a margin policy does past the measured range. `refuse` (default, D33) "
             "leaves the candidate unmeasured; `widen_error_bars` extrapolates and grows "
             "the margin with distance.")
    fit_domain.add_argument(
        "--fitted-at-concurrency", type=float, default=None,
        help="The served concurrency the hardware's profile was validated at. Defaults "
             "to the lowest measured point, and says so.")
    fit_domain.add_argument("--source", default="measured",
                            help="provenance label for the domain (default: measured)")
    fit_domain.add_argument("--note", default="", help="free-form note stored on the domain")
    fit_domain.add_argument("--base", default=None,
                            help="calibration YAML to extend; never modified")
    fit_domain.add_argument("--replace-domain", action="store_true",
                            help="allow replacing a domain the base already carries")
    fit_domain.add_argument("--out", required=True, help="new calibration YAML to write")
    fit_domain.add_argument("--overwrite", action="store_true",
                            help="allow --out to replace an existing file")
    fit_domain.set_defaults(func=cmd_fit_accuracy_domain)

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
