"""PerformanceEnvelope cache (work order §3.6).

A file-backed cache of simulation results. The work order is explicit that this
stays on the filesystem - no database server - because the point is that a
result set can be committed, diffed and shipped with a paper.

The key extends §3.6 to describe the entire deployment (every island
assignment, dp included) - see EnvelopeKey for why. Two runs that agree on the
full key are interchangeable; anything else is a different experiment.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
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


#: A canonical bucket key, e.g. "in_lt1024-out_ge512-rps_lt20". Built only by
#: `workload_bucket`; anything else is a human label and must not be used as a
#: lookup key (uncertainty work order §2.4.1).
_CANONICAL_BUCKET = re.compile(
    r"^in_(lt\d+|ge\d+)-out_(lt\d+|ge\d+)-rps_(lt\d+|ge\d+)$"
)


#: The (input, output) half of a canonical key: "in_lt1024-out_ge512".
_CANONICAL_SHAPE = re.compile(r"^in_(lt\d+|ge\d+)-out_(lt\d+|ge\d+)$")


def workload_shape(spec: ServiceSpec) -> str:
    """The token-mix half of a canonical bucket, without the arrival rate.

    An accuracy domain is indexed by served concurrency, which already carries
    per-device load; the rate component of a full bucket key is a SERVICE-level
    quantity that gets divided across replicas, so it is the wrong axis for a
    per-hardware error (uncertainty work order §2.4.1, amended). What must still
    match is the token mix, because that sets the compute/memory balance.
    """
    return "-".join(
        (
            f"in_{_bucket(spec.traffic.input_tokens.p50, INPUT_BUCKETS)}",
            f"out_{_bucket(spec.traffic.output_tokens.p50, OUTPUT_BUCKETS)}",
        )
    )


def shape_of_bucket(canonical: str) -> str:
    """The shape prefix of a full canonical key, or "" if it is not one."""
    if not is_canonical_bucket(canonical):
        return ""
    return "-".join(canonical.split("-")[:2])


def is_canonical_shape(value: str) -> bool:
    return bool(_CANONICAL_SHAPE.match(value))


def is_canonical_bucket(value: str) -> bool:
    """Does this string have the shape `workload_bucket` produces?

    The uncertainty work order forbids fuzzy bucket matching, so every place
    that accepts a bucket from a human checks the shape here first rather than
    discovering the mismatch as a silent lookup miss.
    """
    return bool(_CANONICAL_BUCKET.match(value))


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


def _metrics_schema_digest() -> str:
    """Digest of `PredictedMetrics`' field set, stored inside every new entry.

    An entry written before a field existed still VALIDATES - a new optional
    field just defaults to None - so without this a schema change is served
    silently from a stale cache. That happened (uncertainty STEP B2): E-A1
    re-ran to byte-identical numbers off entries that predated
    `served_concurrency_per_island`, which reads as "the change had no effect"
    when it means "the change was never applied". An entry whose stored digest
    differs from the running schema is a miss and re-simulates.

    Stored IN the payload rather than folded into the file name so the committed
    replay caches (`outputs/perf/topk/cache_*`, E6) keep their names and their
    consumers keep working; an entry with no digest at all predates this and is
    served with a warning that says so.
    """
    from planner.plan import PredictedMetrics

    return prov.hash_object(sorted(PredictedMetrics.model_fields))[:16]


_METRICS_SCHEMA = _metrics_schema_digest()


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
        topology_level: int = 1,
        graph_signature: str | None = None,
        signature_of: Callable[[CandidateConfig], str | None] | None = None,
    ) -> None:
        self.root = Path(root)
        self.spec = spec
        self.accelerator_of = accelerator_of
        self.link_bw_gbps = link_bw_gbps
        self.trace_digest = trace_digest
        self.enabled = enabled
        # Level-2 compiles the same candidate to different per-dimension link_bw
        # than Level 1, so the two must not share cache entries. Folded into the
        # key only when != 1, keeping existing (Level-1) cache files valid.
        self.topology_level = topology_level
        # Structure the key cannot otherwise see. The envelope key describes a
        # candidate's parallelism and hardware, not the GRAPH it was placed on,
        # so two graph-search representatives that differ only in which shared
        # uplink they cross would collide here. None - every path but the
        # graph-search driver - leaves the file names exactly as they were, so
        # the committed replay caches keep working. Same precedent as
        # `topology_level` above.
        self.graph_signature = graph_signature
        # PER-CANDIDATE structure, where one signature for the whole cache is
        # not enough. A batch of graph-search representatives differs candidate
        # by candidate, and `EnvelopeKey` cannot see any of it: two placements
        # alike in parallelism and hardware but crossing different shared
        # uplinks produce the same key, so the second would be served the
        # first's metrics with nothing in the output to show it (D126).
        # Takes precedence over `graph_signature` when both are set.
        self.signature_of = signature_of
        # Counters live in a dict so `with_graph_signature` can hand out a
        # sibling cache that SHARES them: the caller wants one hit/miss tally
        # for the run, not one per representative.
        self._stats = {"hits": 0, "misses": 0}
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @property
    def hits(self) -> int:
        return self._stats["hits"]

    @property
    def misses(self) -> int:
        return self._stats["misses"]

    def with_signature_of(
        self, signature_of: Callable[[CandidateConfig], str | None]
    ) -> EnvelopeCache:
        """A sibling keyed per candidate, sharing this cache's counters."""
        sibling = copy.copy(self)
        sibling.signature_of = signature_of
        sibling._stats = self._stats
        return sibling

    def with_graph_signature(self, sig: str) -> EnvelopeCache:
        """A sibling reading the same root under a different graph signature.

        Shares this cache's hit/miss counters by reference, so a driver that
        makes one per representative still reports a single tally.
        """
        sibling = copy.copy(self)
        sibling.graph_signature = sig
        sibling._stats = self._stats
        return sibling

    def cache_key(self, candidate: CandidateConfig) -> str | None:
        """Stable string identifying a candidate's cache entry, or None if it has
        no computable key. Two candidates share an entry iff this is equal - used
        to dedup within-run same-key simulations so a homogeneous multi-island
        cluster memoizes instead of re-simulating (parallel evaluate_candidates)."""
        path = self._path(candidate)
        return str(path) if path is not None else None

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
        if self.topology_level != 1:
            name = prov.hash_object([name, f"topology_level={self.topology_level}"])
        signature = self.graph_signature
        if self.signature_of is not None:
            per_candidate = self.signature_of(candidate)
            if per_candidate is not None:
                signature = per_candidate
        if signature:
            name = prov.hash_object([name, f"graph={signature}"])
        return self.root / f"{name}.json"

    def get(self, candidate: CandidateConfig) -> SimResult | None:
        if not self.enabled:
            return None
        path = self._path(candidate)
        if path is None or not path.exists():
            self._stats["misses"] += 1
            return None
        try:
            payload = json.loads(path.read_text())
            metrics = PredictedMetrics.model_validate(payload["metrics"])
        except Exception:
            # Unreadable entry: treat as a miss and let it be overwritten.
            self._stats["misses"] += 1
            return None
        stored_schema = payload.get("metrics_schema")
        if stored_schema is not None and stored_schema != _METRICS_SCHEMA:
            # Written under a different PredictedMetrics field set: a stale
            # entry that would validate anyway. A miss, so it re-simulates.
            self._stats["misses"] += 1
            return None
        self._stats["hits"] += 1
        op = payload.get("operating_point") or {}
        warnings = ["metrics served from the envelope cache"]
        if stored_schema is None:
            warnings.append(
                "this cache entry carries no metrics-schema digest, so a field added "
                "since it was written would read as absent rather than as stale"
            )
        if not op:
            warnings.append(
                "this cache entry predates operating-point caching, so no "
                "accuracy-domain margin can be applied to it"
            )
        return SimResult(
            candidate_id=candidate.id,
            outcome=SimOutcome.OK,
            metrics=metrics,
            warnings=warnings,
            operating_point=op,
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
            "metrics_schema": _METRICS_SCHEMA,
            # Cached with the metrics because it is the same kind of thing: a
            # property of the run. Leaving it out meant a warm cache silently
            # dropped the accuracy-domain margin to zero (STEP 4.3).
            "operating_point": result.operating_point,
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
