"""Candidate enumeration and the pruning pipeline (work order §5.4).

Stage order is fixed and every rejection is recorded, so `rejected_summary` can
explain *why* a search came back empty rather than just that it did:

    all available resources
      1. backend / model compatibility
      2. memory feasibility
      3. parallelism feasibility
      4. topology lower bound
      5. analytical performance lower bound
      -> survivors go to full simulation

Stages 1-3 are exact. Stages 4-5 are *lower bounds*: they may only reject a
candidate when even the most optimistic arithmetic misses the SLO. A bound that
prunes an achievable configuration is a bug, which is what the oracle-agreement
test exists to catch (§9).
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

from planner.inventory import (
    AcceleratorProfile,
    ClusterSpecV2,
    ExecutionIsland,
    compatibility,
)
from planner.plan import (
    CandidateConfig,
    IslandAssignment,
    Rejection,
    RejectionStage,
    Role,
    ServingArch,
    VllmKnobs,
)
from planner.spec import ServiceSpec
from planner.topology import TopologyGraph
from planner.util import memory as memutil

#: Discrete knob values enumerated in the MVP (§5.4). Kept small deliberately -
#: the search space multiplies, and the work order asks for "a few discrete
#: candidates" rather than a sweep.
DEFAULT_MAX_NUM_SEQS = (32, 128, 256)
DEFAULT_MAX_NUM_BATCHED_TOKENS = (2048, 8192)

_log = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    candidates: list[CandidateConfig]
    rejections: list[Rejection]
    generated: int

    @property
    def survivors(self) -> int:
        return len(self.candidates)


class CandidateGenerator:
    """Enumerate placements, then prune in the fixed §5.4 order."""

    def __init__(
        self,
        spec: ServiceSpec,
        cluster: ClusterSpecV2,
        islands: list[ExecutionIsland],
        profiles: dict[str, AcceleratorProfile],
        *,
        topology: TopologyGraph | None = None,
        max_num_seqs: tuple[int, ...] = DEFAULT_MAX_NUM_SEQS,
        max_num_batched_tokens: tuple[int, ...] = DEFAULT_MAX_NUM_BATCHED_TOKENS,
        gpu_memory_utilization: float = 0.90,
        activation_reserve_gb: float = 0.0,
        enable_prefix_caching: bool = False,
        enable_bound_pruning: bool = True,
        enable_mixed: bool = True,
        enable_pd: bool = False,
    ) -> None:
        self.spec = spec
        self.cluster = cluster
        self.islands = islands
        self.profiles = profiles
        self.topology = topology or TopologyGraph(cluster)
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.activation_reserve_gb = activation_reserve_gb
        self.enable_prefix_caching = enable_prefix_caching
        #: When False, stages 4-5 (the *bound-based* filters) are skipped and
        #: every structurally valid candidate goes to simulation. This is the
        #: oracle mode of §9: if a bound ever prunes the true optimum, the
        #: pruned and unpruned searches disagree and the test fails. Stages 1-3
        #: are exact, not bounds, so they always run.
        self.enable_bound_pruning = enable_bound_pruning
        #: Enumerate two-island replica placements (work order §1.3). Pairs
        #: only, uniform devices-per-replica - see _mixed_candidates and D14.
        self.enable_mixed = enable_mixed
        #: Enumerate Prefill/Decode-split placements (work order §5.3, Phase 5
        #: increment 1). Cross-island, both directions, uniform instance size -
        #: see _pd_candidates and D14. Off by default so Phase 2 behaviour and
        #: the frozen golden outputs are unchanged.
        self.enable_pd = enable_pd
        self._rejections: list[Rejection] = []
        self._generated = 0

    # -- entry point -------------------------------------------------------

    def generate(self) -> GenerationResult:
        self._rejections = []
        self._generated = 0
        survivors: list[CandidateConfig] = []

        island_opts: dict[str, tuple[ExecutionIsland, AcceleratorProfile, list]] = {}
        for island in self.islands:
            profile = self.profiles.get(island.accelerator_model)
            if not self._stage1_compatible(island, profile):
                continue
            assert profile is not None
            opts = self._parallelism_options(island, profile)
            survivors.extend(self._single_island_candidates(island, profile, opts))
            if opts:
                island_opts[island.id] = (island, profile, opts)

        single_count = len(survivors)
        if self.enable_mixed:
            survivors.extend(self._mixed_candidates(island_opts))
        mixed_count = len(survivors) - single_count
        if self.enable_pd:
            survivors.extend(self._pd_candidates(island_opts))
        pd_count = len(survivors) - single_count - mixed_count

        # Make the P/D blow-up visible: P/D roughly squares the island-pairing
        # space (work order §5.3 risk note), so log the breakdown for anyone
        # watching the candidate count grow.
        _log.info(
            "generated %d candidate(s): %d single-island, %d mixed, %d pd-split "
            "(%d structurally enumerated)",
            len(survivors), single_count, mixed_count, pd_count, self._generated,
        )

        return GenerationResult(survivors, list(self._rejections), self._generated)

    # -- stage 1: backend / model compatibility ----------------------------

    def _stage1_compatible(
        self, island: ExecutionIsland, profile: AcceleratorProfile | None
    ) -> bool:
        """Rejects the whole island, so it is charged once rather than per candidate."""
        cid = f"{island.id}/*"
        if profile is None:
            self._reject(
                cid,
                RejectionStage.BACKEND_INCOMPATIBLE,
                f"island {island.id}: accelerator model "
                f"'{island.accelerator_model}' has no profile",
            )
            return False
        if not compatibility(self.spec.model, self.spec.service.dtype, profile):
            self._reject(
                cid,
                RejectionStage.BACKEND_INCOMPATIBLE,
                f"island {island.id}: profile {profile.profile_id} does not declare "
                f"support for {self.spec.model} @ {self.spec.service.dtype}",
            )
            return False
        if profile.sim_hardware is None:
            self._reject(
                cid,
                RejectionStage.BACKEND_INCOMPATIBLE,
                f"island {island.id}: profile {profile.profile_id} has no sim_hardware, "
                f"so no profiler/perf/<hardware>/ bundle exists to simulate it",
            )
            return False
        return True

    # -- per-island enumeration -------------------------------------------

    def _parallelism_options(
        self, island: ExecutionIsland, profile: AcceleratorProfile
    ) -> list[tuple[int, memutil.MemoryReport]]:
        """Knob-independent (tp, memory report) options for one island.

        Runs the *role-agnostic* stages 2-3 (parallelism + memory) once per TP
        degree; the surviving options feed single-island, mixed and P/D
        enumeration so they can never disagree on what an island supports. The
        remaining bounds (stage 4 topology, stage 5 analytical) are role- and
        assignment-specific - a prefill engine runs no decode steps, so its
        decode-oriented TPOT floors do not apply - and are therefore evaluated
        per assignment in `_assignment_ok`, not here.
        """
        out: list[tuple[int, memutil.MemoryReport]] = []
        per_device_gb = island.total_memory_gb / island.size

        for tp in island.max_tp_candidates:
            # stage 3: parallelism feasibility. TP must divide the island and
            # respect the profile cap; detect_islands already applies both, so a
            # failure here means the island and profile disagree.
            if island.size % tp != 0 or tp > profile.max_tp_size:
                self._generated += 1
                self._reject(
                    f"{island.id}/tp{tp}",
                    RejectionStage.PARALLELISM_INFEASIBLE,
                    f"tp={tp} does not divide island size {island.size} "
                    f"or exceeds profile max_tp_size {profile.max_tp_size}",
                )
                continue

            fits, report = memutil.feasible(
                self.spec.model,
                tp_size=tp,
                device_memory_gb=per_device_gb,
                dtype=self.spec.service.dtype,
                kv_cache_dtype=self.spec.service.kv_cache_dtype,
                gpu_memory_utilization=self.gpu_memory_utilization,
                activation_reserve_gb=self.activation_reserve_gb,
            )
            if not fits:
                self._generated += 1
                self._reject(
                    f"{island.id}/tp{tp}",
                    RejectionStage.MEMORY_INFEASIBLE,
                    f"tp={tp}: weights {report.weight_bytes / (1024**3):.2f} GiB/gpu "
                    f"leave no usable KV space on a {per_device_gb:g} GB device",
                )
                continue

            out.append((tp, report))
        return out

    def _single_island_candidates(
        self,
        island: ExecutionIsland,
        profile: AcceleratorProfile,
        opts: list[tuple[int, memutil.MemoryReport]],
    ) -> list[CandidateConfig]:
        out: list[CandidateConfig] = []
        for tp, report in opts:
            max_replicas = island.size // tp
            for dp in range(1, max_replicas + 1):
                for seqs, tokens in itertools.product(
                    self.max_num_seqs, self.max_num_batched_tokens
                ):
                    self._generated += 1
                    cand = self._build(island, tp, dp, seqs, tokens)
                    if not self._assignment_ok(
                        cand, cand.assignments[0], island, profile, report, seqs
                    ):
                        continue
                    out.append(cand)
        return out

    # -- mixed (cross-island) enumeration -----------------------------------

    def _mixed_candidates(
        self,
        island_opts: dict[str, tuple[ExecutionIsland, AcceleratorProfile, list]],
    ) -> list[CandidateConfig]:
        """Replica placements spanning two islands (work order §1.3).

        Heterogeneity at replica granularity: each island hosts its own
        replicas at its own parallelism, one engine per replica, requests
        spread by the router. TP never crosses an island (absolute rule 2 is
        structural here - each assignment carries its own tp).

        Two restrictions, both deliberate:
        - Pairs only. Exp 2 compares A-only / B-only / mixed; k>2 islands adds
          combinatorial volume the MVP does not need.
        - Equal devices-per-replica across the two islands. The simulator
          infers its network topology as [npus_per_group, num_instances] with
          integer division over the total device count, which silently
          mis-scopes collectives when instance sizes differ (deviations D14).
          Unequal-size mixes are unrepresentable, so they are not enumerated -
          same treatment as cross-backend TP.
        """
        out: list[CandidateConfig] = []
        for a_id, b_id in itertools.combinations(sorted(island_opts), 2):
            island_a, prof_a, opts_a = island_opts[a_id]
            island_b, prof_b, opts_b = island_opts[b_id]
            for (tp_a, rep_a), (tp_b, rep_b) in itertools.product(opts_a, opts_b):
                if tp_a != tp_b:  # D14: uniform instance size required
                    continue
                for dp_a in range(1, island_a.size // tp_a + 1):
                    for dp_b in range(1, island_b.size // tp_b + 1):
                        for seqs, tokens in itertools.product(
                            self.max_num_seqs, self.max_num_batched_tokens
                        ):
                            self._generated += 1
                            cand = self._build_mixed(
                                (island_a, tp_a, dp_a), (island_b, tp_b, dp_b), seqs, tokens
                            )
                            ok = True
                            for i, (isl, prof, rep) in enumerate(
                                ((island_a, prof_a, rep_a), (island_b, prof_b, rep_b))
                            ):
                                if not self._assignment_ok(
                                    cand, cand.assignments[i], isl, prof, rep, seqs
                                ):
                                    ok = False
                                    break
                            if ok:
                                out.append(cand)
        return out

    def _build_mixed(
        self,
        a: tuple[ExecutionIsland, int, int],
        b: tuple[ExecutionIsland, int, int],
        seqs: int,
        tokens: int,
    ) -> CandidateConfig:
        knobs = VllmKnobs(
            max_num_seqs=seqs,
            max_num_batched_tokens=tokens,
            enable_prefix_caching=self.enable_prefix_caching,
            kv_cache_dtype=self.spec.service.kv_cache_dtype,
        )
        parts = []
        assignments = []
        for island, tp, dp in (a, b):
            parts.append(f"{island.id}-tp{tp}-dp{dp}")
            assignments.append(
                IslandAssignment(
                    island_id=island.id, role=Role.AGGREGATED, tp_size=tp, dp_replicas=dp
                )
            )
        return CandidateConfig(
            id=f"mix({'+'.join(parts)})-s{seqs}-t{tokens}",
            model=self.spec.model,
            dtype=self.spec.service.dtype,
            assignments=assignments,
            serving_arch=ServingArch.AGGREGATED,
            knobs=knobs,
        )

    # -- P/D-split (cross-island role) enumeration -------------------------

    def _pd_candidates(
        self,
        island_opts: dict[str, tuple[ExecutionIsland, AcceleratorProfile, list]],
    ) -> list[CandidateConfig]:
        """Prefill/Decode-split placements across two islands (work order §5.3).

        Heterogeneity at *role* granularity rather than replica granularity: the
        prefill role runs on island A, the decode role on island B, each an
        independent vLLM engine, with the KV cache shipped A->B per request. The
        simulator models this natively via per-instance `pd_type`
        (serving/core/config_builder.py), so no upstream edit is needed - the
        generator only has to emit the right role pairs.

        Restrictions, all deliberate:
        - Cross-island only, and BOTH directions. Prefill-on-A/decode-on-B is a
          different deployment from prefill-on-B/decode-on-A (different hardware
          does the prefill vs the decode work), so ordered pairs are enumerated -
          `itertools.permutations`, not the unordered `combinations`
          `_mixed_candidates` uses. Cross-vendor P/D (GPU-P + NPU-D, etc.)
          therefore emerges naturally whenever two islands run different
          backends; on a single-backend cluster only same-backend P/D appears.
        - Same-island P/D is deferred: prefill and decode engines would then
          share one island's devices, which needs device-overlap accounting the
          MVP does not do. Enumerating it here would double-book hardware.
        - Equal devices-per-replica across the two islands (`tp_p == tp_d`). Same
          reason as `_mixed_candidates`: the simulator infers its network
          topology as [npus_per_group, num_instances] by integer division over
          the total device count, which silently mis-scopes collectives when
          instance sizes differ (deviations D14). Unequal sizes are
          unrepresentable, so they are not enumerated.

        Bounds run per assignment via `_assignment_ok`, and that check is
        role-aware: the prefill assignment skips the decode-oriented TPOT floors
        (stage-4 all-reduce, stage-5 memory roofline) because a disaggregated
        prefill engine runs no decode steps - TPOT is set by the decode island.
        Charging the prefill side a decode TPOT floor would reject
        slow-prefill/fast-decode candidates that the real simulator accepts,
        violating "a pruning stage must be a relaxation of feasibility". The
        bounds that do apply stay gated by `enable_bound_pruning`, so oracle mode
        sees the same P/D candidates minus only the bound rejections.
        """
        out: list[CandidateConfig] = []
        for p_id, d_id in itertools.permutations(sorted(island_opts), 2):
            island_p, prof_p, opts_p = island_opts[p_id]
            island_d, prof_d, opts_d = island_opts[d_id]
            for (tp_p, rep_p), (tp_d, rep_d) in itertools.product(opts_p, opts_d):
                if tp_p != tp_d:  # D14: uniform instance size required
                    continue
                # dp_p == dp_d for now. Asymmetric prefill/decode replica counts
                # produce a prefill<->decode instance pairing that is not yet
                # validated against the simulator's config_builder (a D14-style
                # representability limit); unlock it in Phase 5 increment 2 once
                # the Level-2 network model can express the pairing.
                for dp in range(1, min(island_p.size // tp_p, island_d.size // tp_d) + 1):
                    for seqs, tokens in itertools.product(
                        self.max_num_seqs, self.max_num_batched_tokens
                    ):
                        self._generated += 1
                        cand = self._build_pd(
                            (island_p, tp_p, dp), (island_d, tp_d, dp), seqs, tokens
                        )
                        ok = True
                        for i, (isl, prof, rep) in enumerate(
                            ((island_p, prof_p, rep_p), (island_d, prof_d, rep_d))
                        ):
                            if not self._assignment_ok(
                                cand, cand.assignments[i], isl, prof, rep, seqs
                            ):
                                ok = False
                                break
                        if ok:
                            out.append(cand)
        return out

    def _build_pd(
        self,
        prefill: tuple[ExecutionIsland, int, int],
        decode: tuple[ExecutionIsland, int, int],
        seqs: int,
        tokens: int,
    ) -> CandidateConfig:
        knobs = VllmKnobs(
            max_num_seqs=seqs,
            max_num_batched_tokens=tokens,
            enable_prefix_caching=self.enable_prefix_caching,
            kv_cache_dtype=self.spec.service.kv_cache_dtype,
        )
        island_p, tp_p, dp_p = prefill
        island_d, tp_d, dp_d = decode
        return CandidateConfig(
            id=(
                f"pd({island_p.id}-tp{tp_p}-dp{dp_p} P + "
                f"{island_d.id}-tp{tp_d}-dp{dp_d} D)-s{seqs}-t{tokens}"
            ),
            model=self.spec.model,
            dtype=self.spec.service.dtype,
            assignments=[
                IslandAssignment(
                    island_id=island_p.id, role=Role.PREFILL, tp_size=tp_p, dp_replicas=dp_p
                ),
                IslandAssignment(
                    island_id=island_d.id, role=Role.DECODE, tp_size=tp_d, dp_replicas=dp_d
                ),
            ],
            serving_arch=ServingArch.PD_SPLIT,
            knobs=knobs,
        )

    def _build(
        self, island: ExecutionIsland, tp: int, dp: int, seqs: int, tokens: int
    ) -> CandidateConfig:
        knobs = VllmKnobs(
            max_num_seqs=seqs,
            max_num_batched_tokens=tokens,
            enable_prefix_caching=self.enable_prefix_caching,
            kv_cache_dtype=self.spec.service.kv_cache_dtype,
        )
        return CandidateConfig(
            id=f"{island.id}-tp{tp}-dp{dp}-s{seqs}-t{tokens}",
            model=self.spec.model,
            dtype=self.spec.service.dtype,
            assignments=[
                IslandAssignment(
                    island_id=island.id, role=Role.AGGREGATED, tp_size=tp, dp_replicas=dp
                )
            ],
            serving_arch=ServingArch.AGGREGATED,
            knobs=knobs,
        )

    # -- per-assignment bounds (stages 4-5) --------------------------------

    def _assignment_ok(
        self,
        cand: CandidateConfig,
        assignment: IslandAssignment,
        island: ExecutionIsland,
        profile: AcceleratorProfile,
        report: memutil.MemoryReport,
        seqs: int,
    ) -> bool:
        """Run the bound-based lower bounds (stages 4-5) for one assignment.

        Both stages read the assignment's own role and tp rather than
        `cand.assignments[0]`, so a decode assignment is scored against its own
        island/tp and a prefill assignment is exempted from the decode-oriented
        TPOT floors. Stages 2-3 already ran (role-agnostic) in
        `_parallelism_options`.
        """
        if not self._stage4_topology_ok(cand, assignment, island):
            return False
        return self._stage5_analytical_ok(cand, assignment, island, profile, report, seqs)

    # -- stage 4: topology lower bound -------------------------------------

    def _stage4_topology_ok(
        self, cand: CandidateConfig, assignment: IslandAssignment, island: ExecutionIsland
    ) -> bool:
        """Reject only if TP collectives alone already blow the TPOT budget.

        Per decode step every transformer block runs two all-reduces over the
        hidden state. A ring all-reduce moves 2(N-1)/N x bytes. Compute is
        assumed free here - that is what makes it a lower bound.

        This is a *decode* floor, so it does not apply to a disaggregated prefill
        engine (`Role.PREFILL`): prefill runs no decode steps, and pruning it on
        a decode-step cost would remove candidates the simulator accepts. For
        `Role.DECODE` and `Role.AGGREGATED` it applies as before.
        """
        if not self.enable_bound_pruning:
            return True
        if assignment.role is Role.PREFILL:
            return True
        tp = assignment.tp_size
        if tp == 1:
            return True

        bw_gbps, lat_ns, _ = self.topology.island_interconnect(island)
        if bw_gbps == float("inf"):
            return True

        cfg = memutil.model_config(self.spec.model)
        hidden = cfg["hidden_size"]
        layers = cfg["num_hidden_layers"]
        bytes_per_elem = memutil.dtype_bits(self.spec.service.dtype) // 8

        payload = hidden * bytes_per_elem
        ring_factor = 2 * (tp - 1) / tp
        per_allreduce_ns = lat_ns + (payload * ring_factor) / bw_gbps
        floor_ms = (2 * layers * per_allreduce_ns) / 1e6

        budget = self.spec.slo.tpot.max_ms
        if floor_ms > budget:
            self._reject(
                cand.id,
                RejectionStage.TOPOLOGY_INFEASIBLE,
                f"TP={tp} all-reduce floor {floor_ms:.1f}ms on {island.id} exceeds the "
                f"TPOT budget {budget:.1f}ms even with zero compute "
                f"({bw_gbps:g} GB/s, {lat_ns:g} ns per hop)",
            )
            return False
        return True

    # -- stage 5: analytical performance lower bound -----------------------

    def _stage5_analytical_ok(
        self,
        cand: CandidateConfig,
        assignment: IslandAssignment,
        island: ExecutionIsland,
        profile: AcceleratorProfile,
        report: memutil.MemoryReport,
        seqs: int,
    ) -> bool:
        """Memory-roofline floor on TPOT, and a KV-capacity check on concurrency.

        A decode step must at minimum stream the weights and the live KV through
        HBM once. `memory_bandwidth_gbps` is a peak figure, so the resulting time
        is a genuine lower bound.

        Role-aware: the roofline-vs-TPOT floor is a *decode*-step cost and is
        skipped for `Role.PREFILL` (a disaggregated prefill engine runs no decode
        steps). The KV-capacity check is exact and always runs, but a prefill
        engine holds only the prompt KV, so it sizes against the input length
        alone; decode and aggregated engines hold input+output.
        """
        tp = assignment.tp_size
        is_prefill = assignment.role is Role.PREFILL
        if is_prefill:
            median_len = self.spec.traffic.input_tokens.p50
        else:
            median_len = (
                self.spec.traffic.input_tokens.p50 + self.spec.traffic.output_tokens.p50
            )

        # Concurrency the KV budget actually supports, versus what the knob asks
        # for. This one is exact rather than a bound - a placement with no room
        # for a single median request cannot run at all - so it survives oracle
        # mode along with stages 1-3.
        kv_capacity = report.kv_tokens // max(1, median_len)
        if kv_capacity < 1:
            self._reject(
                cand.id,
                RejectionStage.MEMORY_INFEASIBLE,
                f"KV space holds {report.kv_tokens:,} tokens, below one median "
                f"{'prompt' if is_prefill else 'request'} of {median_len:,} tokens",
            )
            return False

        if not self.enable_bound_pruning or is_prefill:
            return True

        active = min(seqs, kv_capacity)
        bytes_per_step = report.weight_bytes + active * report.kv_bytes_per_token
        # GB/s against GiB-denominated sizes: convert to the same base.
        bw_bytes_per_s = profile.memory_bandwidth_gbps * 1e9
        floor_ms = (bytes_per_step / bw_bytes_per_s) * 1e3

        if floor_ms > self.spec.slo.tpot.max_ms:
            self._reject(
                cand.id,
                RejectionStage.ANALYTICAL_LOWER_BOUND,
                f"memory-roofline TPOT floor {floor_ms:.1f}ms exceeds budget "
                f"{self.spec.slo.tpot.max_ms:.1f}ms at tp={tp}, "
                f"{active} concurrent sequences",
            )
            return False

        # A throughput bound used to live here: reject when the roofline ceiling
        # falls below the offered token rate. It was removed because it is
        # *stricter* than the feasibility test it is supposed to approximate.
        # §5.6 declares hard constraints on P99 TTFT/TPOT, power and tokens/J -
        # not on throughput - so a candidate rejected here could still have
        # passed every declared constraint, and the pruned search disagreed with
        # the oracle. A pruning stage must be a relaxation of feasibility, never
        # an extra condition. Under-provisioned candidates now reach the
        # simulator, which surfaces the saturation as a TTFT blow-up.
        #
        # Restoring it requires first adding a goodput/attainment hard
        # constraint to feasibility, so that both sides agree on what "cannot
        # absorb the load" means.
        return True

    # -- bookkeeping -------------------------------------------------------

    def _reject(self, candidate_id: str, stage: RejectionStage, reason: str) -> None:
        self._rejections.append(
            Rejection(candidate_id=candidate_id, stage=stage, reason=reason)
        )
