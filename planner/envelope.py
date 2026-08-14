"""PerformanceEnvelope cache (work order §3.6).

A file-backed cache of simulation results. The work order is explicit that this
stays on the filesystem - no database server - because the point is that a
result set can be committed, diffed and shipped with a paper.

The key extends §3.6 to describe the entire deployment (every island
assignment, dp included) - see EnvelopeKey for why. Two runs that agree on the
full key are interchangeable; anything else is a different experiment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from planner.plan import CandidateConfig, PredictedMetrics
from planner.predictor import SimOutcome, SimResult
from planner.spec import ServiceSpec
from planner.util import provenance as prov

#: Workload bucket boundaries (§3.6). Hardcoded by the work order, isolated here
#: so a later change is one edit and shows up in every cache key at once.
INPUT_BUCKETS = (1024, 4096)
OUTPUT_BUCKETS = (128, 512)
RATE_BUCKETS = (5.0, 20.0)


def _bucket(value: float, edges: tuple[float, ...]) -> str:
    for edge in edges:
        if value < edge:
            return f"lt{int(edge)}"
    return f"ge{int(edges[-1])}"


def workload_bucket(spec: ServiceSpec) -> str:
    """Coarse workload identity: (input p50, output p50, arrival rate)."""
    return "-".join(
        (
            f"in_{_bucket(spec.traffic.input_tokens.p50, INPUT_BUCKETS)}",
            f"out_{_bucket(spec.traffic.output_tokens.p50, OUTPUT_BUCKETS)}",
            f"rps_{_bucket(spec.traffic.arrival_rate_rps, RATE_BUCKETS)}",
        )
    )


def network_class(link_bw_gbps: float) -> str:
    """Bandwidth band, so a 398 GB/s and a 400 GB/s fabric share an entry."""
    for edge, label in ((25, "lt25"), (100, "lt100"), (200, "lt200"), (400, "lt400")):
        if link_bw_gbps < edge:
            return label
    return "ge400"


@dataclass(frozen=True)
class EnvelopeKey:
    model: str
    dtype: str
    #: Canonical description of the ENTIRE deployment, one segment per island
    #: assignment: "accelerator|role|tp|pp|ep|dp", sorted and ';'-joined.
    #:
    #: This deliberately extends the work order's §3.6 key in two ways. `dp`
    #: was added after omitting it served 12 dp=2 candidates their dp=1 metrics
    #: (deviations D13). Then mixed candidates made the per-field form
    #: insufficient outright: a placement spanning two islands must never
    #: collide with a single-island placement that happens to share the first
    #: assignment. Our predictor simulates whole deployments, so the key must
    #: describe whole deployments. §3.6's field list would only suffice for a
    #: *per-replica* envelope that the planner composes arithmetically.
    placement: str
    scheduler_config_hash: str
    network_class: str
    workload_bucket: str

    def digest(self) -> str:
        return prov.hash_object(self.__dict__)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class EnvelopeKeyError(ValueError):
    """Raised when a candidate cannot be keyed (unknown island)."""


def key_for(
    candidate: CandidateConfig,
    spec: ServiceSpec,
    *,
    accelerator_of: dict[str, str],
    link_bw_gbps: float,
) -> EnvelopeKey:
    segments = []
    for a in candidate.assignments:
        accel = accelerator_of.get(a.island_id)
        if accel is None:
            raise EnvelopeKeyError(
                f"candidate {candidate.id}: island '{a.island_id}' has no accelerator "
                f"mapping; refusing to build a cache key that ignores an assignment"
            )
        segments.append(f"{accel}|{a.role.value}|{a.tp_size}|{a.pp_size}|1|{a.dp_replicas}")
    return EnvelopeKey(
        model=candidate.model,
        dtype=candidate.dtype,
        placement=";".join(sorted(segments)),
        scheduler_config_hash=prov.hash_object(candidate.knobs.model_dump()),
        network_class=network_class(link_bw_gbps),
        workload_bucket=workload_bucket(spec),
    )


class EnvelopeCache:
    """One JSON file per entry, under `root`.

    Reads are best-effort: a corrupt or stale entry is ignored rather than
    fatal, because a cache that can break a planning run is worse than no cache.
    Writes record the full key and provenance so an entry can always be traced
    back to the run that produced it.
    """

    def __init__(
        self,
        root: str | Path,
        spec: ServiceSpec,
        *,
        accelerator_of: dict[str, str],
        link_bw_gbps: float,
        trace_digest: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.root = Path(root)
        self.spec = spec
        self.accelerator_of = accelerator_of
        self.link_bw_gbps = link_bw_gbps
        self.trace_digest = trace_digest
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, candidate: CandidateConfig) -> Path | None:
        try:
            key = key_for(
                candidate, self.spec,
                accelerator_of=self.accelerator_of, link_bw_gbps=self.link_bw_gbps,
            )
        except EnvelopeKeyError:
            return None
        name = key.digest()
        if self.trace_digest:
            name = prov.hash_object([name, self.trace_digest])
        return self.root / f"{name}.json"

    def get(self, candidate: CandidateConfig) -> SimResult | None:
        if not self.enabled:
            return None
        path = self._path(candidate)
        if path is None or not path.exists():
            self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text())
            metrics = PredictedMetrics.model_validate(payload["metrics"])
        except Exception:
            # Unreadable entry: treat as a miss and let it be overwritten.
            self.misses += 1
            return None
        self.hits += 1
        return SimResult(
            candidate_id=candidate.id,
            outcome=SimOutcome.OK,
            metrics=metrics,
            warnings=["metrics served from the envelope cache"],
        )

    def put(self, candidate: CandidateConfig, result: SimResult) -> None:
        if not self.enabled or not result.ok:
            return
        path = self._path(candidate)
        if path is None:
            return
        key = key_for(
            candidate, self.spec,
            accelerator_of=self.accelerator_of, link_bw_gbps=self.link_bw_gbps,
        )
        assert result.metrics is not None
        payload = {
            "key": key.as_dict(),
            "candidate_id": candidate.id,
            "trace_digest": self.trace_digest,
            "metrics": result.metrics.model_dump(),
            "provenance": {
                "git_commit": prov.git_commit(),
                "llmservingsim_commit": prov.upstream_commit(),
                "timestamp": prov.collect()["timestamp"],
            },
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(path)  # atomic, so a killed run cannot leave a half-written entry

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}
