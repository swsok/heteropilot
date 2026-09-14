"""STEP C1's three premises, each of which is cheap to break silently.

`WORK_ORDER_uq_stage_b_plus.md` STEP C1. F2 exists to give the Stage-B
experiments a corpus where P/D is on and the TTFT SLO sits in the middle regime,
so more than one uncertain input can decide a plan. Three things make that true
rather than intended, and none of them announces itself when it stops holding:
the fixture really does turn P/D on and really does move the SLO; D40's mirrored
placements are collapsed to one representative each; and the E-A1 cache is
reusable under the new SLO, which is what keeps the build minutes rather than
hours.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.uncertainty.build_truth_cache import (
    E_A1_TTFT_MAX_MS,
    load_fixture,
    mirror_groups,
)
from planner.envelope import EnvelopeCache
from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.plan import CandidateConfig, IslandAssignment, Role, ServingArch
from planner.spec import load_service_spec
from planner.util import provenance as prov
from planner.util.workload import generate_trace

F2 = Path("experiments/uncertainty/fixtures/f2.json")
EA1_SERVICE = Path("examples/service_specs/llama31-8b.yaml")


def _write(tmp_path: Path, **overrides) -> Path:
    fixture = json.loads(F2.read_text()) | overrides
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture))
    return path


# -- the fixture ----------------------------------------------------------

def test_the_shipped_fixture_is_pd_and_out_of_the_ea1_regime() -> None:
    fixture = load_fixture(F2)
    assert fixture["enable_pd"] is True
    spec = load_service_spec(Path(fixture["service"]))
    assert spec.slo.ttft.max_ms != E_A1_TTFT_MAX_MS


@pytest.mark.parametrize("overrides", [{"enable_pd": False}, {}])
def test_a_fixture_without_pd_is_refused(tmp_path: Path, overrides: dict) -> None:
    """Both the explicit `false` and the forgotten key. An aggregated corpus is
    not a smaller F2, it is E-A1 again under a different name."""
    path = _write(tmp_path, **overrides)
    if not overrides:                       # {} means: drop the key entirely
        data = json.loads(path.read_text())
        del data["enable_pd"]
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="enable_pd"):
        load_fixture(path)


def test_the_ea1_service_spec_is_refused(tmp_path: Path) -> None:
    """Pointing F2 at E-A1's spec would rebuild E-A1's truth with P/D bolted on
    and prove nothing about the regime the work order asks for."""
    path = _write(tmp_path, service=str(EA1_SERVICE))
    with pytest.raises(ValueError, match="E-A1"):
        load_fixture(path)


# -- D40 mirror detection -------------------------------------------------

def _pd(prefill: str, decode: str, tp: int) -> CandidateConfig:
    return CandidateConfig(
        id=f"pd({prefill}-tp{tp} P + {decode}-tp{tp} D)",
        model="meta-llama/Llama-3.1-8B",
        dtype="bfloat16",
        serving_arch=ServingArch.PD_SPLIT,
        assignments=[
            IslandAssignment(island_id=prefill, role=Role.PREFILL, tp_size=tp),
            IslandAssignment(island_id=decode, role=Role.DECODE, tp_size=tp),
        ],
    )


@pytest.fixture(scope="module")
def cluster_world():
    cluster = load_cluster_spec(Path("experiments/configs/clusters/pd-rngd-gpu-card.yaml"))
    profiles = load_profiles_for(cluster, Path("."))
    islands = detect_islands(cluster, profiles)
    return cluster, profiles, islands


def test_two_mirror_pairs_are_found_and_the_representative_is_stable(
    tmp_path: Path, cluster_world
) -> None:
    """Four candidates, two mirrored pairs: exactly two groups, and the pick is
    the same on a re-run. `mirror_groups` asks the cache rather than re-deriving
    the key, so this also pins D40 itself -- swapping which A40 island takes
    prefill produces the SAME cache entry."""
    _cluster, _profiles, islands = cluster_world
    spec = load_service_spec(Path("experiments/uncertainty/fixtures/llama31-8b-ttft8000.yaml"))
    a, b = "cuda-a40-node_a40a", "cuda-a40-node_a40b"
    candidates = [_pd(a, b, 2), _pd(b, a, 2), _pd(a, b, 4), _pd(b, a, 4)]

    cache = EnvelopeCache(
        tmp_path / "cache", spec,
        accelerator_of={i.id: i.accelerator_model for i in islands},
        link_bw_gbps=12.5, trace_digest="deadbeef",
    )
    groups = mirror_groups(candidates, cache)

    assert len(groups) == 2
    assert all(len(ids) == 2 for ids in groups.values())
    # Every representative is its own group's smallest id, so a re-run keeps it.
    assert all(rep == min(ids) for rep, ids in groups.items())
    assert mirror_groups(list(reversed(candidates)), cache) == groups
    # Every candidate is accounted for: two kept, two excluded.
    excluded = {i for ids in groups.values() for i in ids[1:]}
    assert len(excluded) == 2


def test_a_candidate_with_no_mirror_forms_no_group(
    tmp_path: Path, cluster_world
) -> None:
    """The guard against a grouping that would collapse everything: two
    candidates that differ in TP are not mirrors."""
    _cluster, _profiles, islands = cluster_world
    spec = load_service_spec(Path("experiments/uncertainty/fixtures/llama31-8b-ttft8000.yaml"))
    a, b = "cuda-a40-node_a40a", "cuda-a40-node_a40b"
    cache = EnvelopeCache(
        tmp_path / "cache", spec,
        accelerator_of={i.id: i.accelerator_model for i in islands},
        link_bw_gbps=12.5, trace_digest="deadbeef",
    )
    assert mirror_groups([_pd(a, b, 2), _pd(a, b, 4)], cache) == {}


# -- cache reuse ----------------------------------------------------------

def test_the_ttft_slo_does_not_enter_the_cache_key(tmp_path: Path, cluster_world) -> None:
    """The reuse is a claim about `EnvelopeKey`, so it is tested as one: the same
    candidate keys to the same entry under E-A1's spec and F2's, whose only
    difference is `slo.ttft.max_ms`. If a future key field picks the SLO up, the
    E-A1 corpus stops being reusable and this fails instead of quietly
    re-simulating 162 candidates."""
    _cluster, _profiles, islands = cluster_world
    ea1 = load_service_spec(EA1_SERVICE)
    f2 = load_service_spec(Path("experiments/uncertainty/fixtures/llama31-8b-ttft8000.yaml"))
    assert ea1.slo.ttft.max_ms != f2.slo.ttft.max_ms

    accelerator_of = {i.id: i.accelerator_model for i in islands}
    candidate = _pd("cuda-a40-node_a40a", "cuda-a40-node_a40b", 2)

    def entry_name(spec, root: str) -> str:
        cache = EnvelopeCache(tmp_path / root, spec, accelerator_of=accelerator_of,
                              link_bw_gbps=12.5, trace_digest="deadbeef")
        key = cache.cache_key(candidate)
        assert key is not None
        return Path(key).name

    assert entry_name(ea1, "ea1") == entry_name(f2, "f2")


def test_the_two_specs_generate_the_same_trace(tmp_path: Path) -> None:
    """The other half of the reuse: the cache entry is also keyed on the trace
    digest, and the traffic section the trace comes from is byte-identical."""
    ea1 = load_service_spec(EA1_SERVICE)
    f2 = load_service_spec(Path("experiments/uncertainty/fixtures/llama31-8b-ttft8000.yaml"))
    digests = [
        prov.hash_file(generate_trace(
            spec, tmp_path / f"{name}.jsonl", num_requests=300, seed=42,
        ).path)
        for spec, name in ((ea1, "ea1"), (f2, "f2"))
    ]
    assert digests[0] == digests[1]
