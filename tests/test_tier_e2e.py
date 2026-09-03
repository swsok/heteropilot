"""Tier 0 bundles drive plans end to end with propagated labels (STEP 10)."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.optimizer import exhaustive
from planner.plan import PlannerOutput
from planner.spec import load_service_spec
from profiler.core.config import load_architecture
from profiler.synth.backend import AnalyticalProfileBackend
from profiler.synth.device import DeviceSpec
from profiler.synth.dims import ModelDims
from profiler.synth.emit import BundleEmitter, GridParams

from .conftest import MockPredictor

REPO = Path(__file__).resolve().parents[1]
HETERO_CLUSTER = REPO / "examples" / "clusters" / "hetero-gpu-ascend.yaml"
QWEN_SPEC = REPO / "examples" / "service_specs" / "qwen3-32b.yaml"


@pytest.fixture(scope="module")
def hetero():
    cluster = load_cluster_spec(HETERO_CLUSTER)
    profiles = load_profiles_for(cluster, REPO)
    islands = detect_islands(cluster, profiles)
    spec = load_service_spec(QWEN_SPEC)
    return spec, cluster, islands, profiles


@pytest.fixture(scope="module")
def tier0_perf_root(tmp_path_factory, hetero) -> Path:
    """A perf root holding a small ASCEND_TARGET-t0 bundle plus the measured
    RTXPRO6000 tier signal (meta-level; tier resolution reads meta.yaml only)."""
    _, _, _, profiles = hetero
    out = tmp_path_factory.mktemp("perf")
    dims = ModelDims.from_hf_config(REPO / "configs" / "model" / "Qwen" / "Qwen3-32B.json", "bf16")
    arch = load_architecture(REPO / "profiler" / "models" / f"{dims.model_type}.yaml")
    device = DeviceSpec.from_profile(profiles["ASCEND_TARGET"], "bf16")
    BundleEmitter(
        dims=dims, arch=arch,
        backend_for_tp={1: AnalyticalProfileBackend(dims, arch, device, 1),
                        2: AnalyticalProfileBackend(dims, arch, device, 2)},
        hardware_label="ASCEND_TARGET-t0", variant="bf16", out_root=out,
        grid=GridParams(max_num_batched_tokens=32, max_num_seqs=4, attention_max_kv=32),
        datasheet_source="test", generated_at="2026-09-02T00:00:00+00:00",
    ).emit()
    # Measured GPU tier signal: only meta.yaml matters for resolution.
    gpu_meta = out / "RTXPRO6000" / "Qwen/Qwen3-32B" / "bf16"
    gpu_meta.mkdir(parents=True)
    (gpu_meta / "meta.yaml").write_text("tier: measured\n")
    return out


def test_ascend_profile_loads_with_datasheet(hetero):
    """ascend_target.yaml loads with a datasheet and the -t0 sim_hardware."""
    _, _, _, profiles = hetero
    ascend = profiles["ASCEND_TARGET"]
    assert ascend.sim_hardware == "ASCEND_TARGET-t0"
    assert ascend.datasheet is not None
    assert ascend.datasheet.flops_efficiency is not None  # transferred, sourced


def test_ascend_island_appears_in_candidates(hetero):
    """Ascend islands are not filtered out at candidate generation."""
    from planner.candidate_generator import CandidateGenerator

    spec, cluster, islands, profiles = hetero
    generation = CandidateGenerator(spec, cluster, islands, profiles).generate()
    ascend_islands = {i.id for i in islands if i.backend == "ascend"}
    assert ascend_islands
    used = {a.island_id for c in generation.candidates for a in c.assignments}
    assert ascend_islands & used, (
        "no candidate uses the Ascend island - the -t0 bundle path failed to "
        "unlock it (was it rejected as sim_hardware-less?)"
    )


def test_heterogeneous_plan_tier_is_analytical(hetero, tier0_perf_root):
    """A hetero (measured GPU + analytical Ascend) plan reports analytical."""
    spec, cluster, islands, profiles = hetero
    output = exhaustive.search(
        spec, cluster, islands, profiles, MockPredictor(), perf_root=tier0_perf_root
    )
    tiers = output.profile_tiers
    assert any(v == "analytical" for v in tiers.values())
    assert any(v == "measured" for v in tiers.values())
    assert output.profile_tier == "analytical"


def test_heterogeneous_plan_carries_caveat(hetero, tier0_perf_root):
    """The exact simulator-only caveat wording travels with the plan."""
    spec, cluster, islands, profiles = hetero
    output = exhaustive.search(
        spec, cluster, islands, profiles, MockPredictor(), perf_root=tier0_perf_root
    )
    assert (
        "simulator-only (analytical inputs): ASCEND_TARGET-t0 profile is "
        "datasheet-derived, not measured"
    ) in output.caveats


def test_all_gpu_plan_has_no_tier_caveat(spec, cluster, islands, profiles):
    """The all-measured example cluster gains no tier caveat (A4)."""
    output = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    assert output.profile_tier == "measured"
    assert not any("simulator-only" in c for c in output.caveats)


def test_plan_yaml_roundtrip_includes_tier(hetero, tier0_perf_root):
    """profile_tier survives the YAML round trip the CLI performs."""
    import yaml

    spec, cluster, islands, profiles = hetero
    output = exhaustive.search(
        spec, cluster, islands, profiles, MockPredictor(), perf_root=tier0_perf_root
    )
    dumped = yaml.safe_dump(output.model_dump(mode="json"), sort_keys=False)
    restored = PlannerOutput.model_validate(yaml.safe_load(dumped))
    assert restored.profile_tier == "analytical"
    assert restored.profile_tiers == output.profile_tiers


def test_regenerate_script_is_deterministic(tmp_path):
    """Two runs of the Tier 0 generator produce identical CSVs.

    Uses the same emitter path as scripts/gen-tier0-bundles.sh with a small
    grid (the script itself regenerates the full grid; determinism is a
    property of the generator, not the grid size). Catches dict-order and
    float-format nondeterminism.
    """
    def emit_once(dest: Path) -> dict[str, str]:
        from planner.inventory import load_accelerator_profile

        profile = load_accelerator_profile(
            REPO / "profiles" / "accelerators" / "ascend_target.yaml"
        )
        dims = ModelDims.from_hf_config(
            REPO / "configs" / "model" / "Qwen" / "Qwen3-32B.json", "bf16"
        )
        arch = load_architecture(
            REPO / "profiler" / "models" / f"{dims.model_type}.yaml"
        )
        device = DeviceSpec.from_profile(profile, "bf16")
        BundleEmitter(
            dims=dims, arch=arch,
            backend_for_tp={1: AnalyticalProfileBackend(dims, arch, device, 1)},
            hardware_label="ASCEND_TARGET-t0", variant="bf16", out_root=dest,
            grid=GridParams(max_num_batched_tokens=64, max_num_seqs=8,
                            attention_max_kv=64),
            datasheet_source="test", generated_at="1970-01-01T00:00:00+00:00",
        ).emit()
        return {
            str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(dest.rglob("*"))
            if p.is_file()
        }

    assert emit_once(tmp_path / "run1") == emit_once(tmp_path / "run2")


def test_gen_script_exists_and_is_executable():
    """scripts/gen-tier0-bundles.sh exists, is executable, targets -t0 labels."""
    script = REPO / "scripts" / "gen-tier0-bundles.sh"
    assert script.exists()
    assert script.stat().st_mode & 0o111
    text = script.read_text()
    assert "ASCEND_TARGET-t0" in text
    assert "profiler.synth emit" in text
    # And bash agrees it parses.
    subprocess.run(["bash", "-n", str(script)], check=True)
