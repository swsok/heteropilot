"""PerformanceEnvelope cache (work order §3.6).

A file-backed cache of simulation results. The work order is explicit that this
stays on the filesystem - no database server - because the point is that a
result set can be committed, diffed and shipped with a paper.

The key follows §3.6: (model, dtype, accelerator, tp, pp, ep, pd_role,
scheduler_config_hash, network_class, workload_bucket). Two runs that agree on
all of those are interchangeable; anything else is a different experiment.
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
    accelerator: str
    tp: int
    pp: int
    ep: int
    #: NOT in the work order's §3.6 key list, and deliberately added.
    #:
    #: Our predictor simulates the *whole deployment*, replicas included, so a
    #: 2-replica result is nothing like a 1-replica one - roughly double the
    #: throughput and a fraction of the queueing delay. Omitting `dp` made
    #: dp=1 and dp=2 collide: in a 30-candidate run, all 12 dp=2 candidates
    #: were served the dp=1 entry and never simulated, and some were then
    #: wrongly marked slo_violated.
    #:
    #: §3.6's list would only be sufficient for an envelope describing
    #: *per-replica* performance that the planner then composes. That is not
    #: what this cache stores.
    dp: int
    pd_role: str
    scheduler_config_hash: str
    network_class: str
    workload_bucket: str

    def digest(self) -> str:
        return prov.hash_object(self.__dict__)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def key_for(
    candidate: CandidateConfig,
    spec: ServiceSpec,
    *,
    accelerator: str,
    link_bw_gbps: float,
) -> EnvelopeKey:
    first = candidate.assignments[0]
    return EnvelopeKey(
        model=candidate.model,
        dtype=candidate.dtype,
        accelerator=accelerator,
        tp=first.tp_size,
        pp=first.pp_size,
        ep=1,
        dp=first.dp_replicas,
        pd_role=first.role.value,
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
        accel = self.accelerator_of.get(candidate.assignments[0].island_id)
        if accel is None:
            return None
        key = key_for(
            candidate, self.spec, accelerator=accel, link_bw_gbps=self.link_bw_gbps
        )
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
        accel = self.accelerator_of[candidate.assignments[0].island_id]
        key = key_for(
            candidate, self.spec, accelerator=accel, link_bw_gbps=self.link_bw_gbps
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
