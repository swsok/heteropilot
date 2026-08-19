"""Deploy backend tests: command building, device resolution, validation.

No test launches a process, opens a socket or needs a GPU (work order §9): the
backend is exercised through its pure `build_serve_command` and `validate`.
"""

from __future__ import annotations

import pytest

from planner.deploy import (
    DeviceMap,
    VllmCudaBackend,
    build_serve_command,
    device_index_from_id,
    resolve_devices,
)
from planner.deploy.base import DeploymentError
from planner.inventory import ExecutionIsland
from planner.plan import (
    CandidateConfig,
    DeploymentPlan,
    IslandAssignment,
    PredictedMetrics,
    Role,
    VllmKnobs,
)

A5000_ISLAND = "cuda-rtx-a5000-node0"
ASCEND_ISLAND = "ascend-ascend-target-node2"

_METRICS = PredictedMetrics(
    p50_ttft_ms=10.0, p95_ttft_ms=20.0, p99_ttft_ms=30.0,
    p50_tpot_ms=1.0, p95_tpot_ms=2.0, p99_tpot_ms=3.0,
    throughput_tps=100.0, slo_goodput_rps=5.0, slo_attainment=1.0,
    completed_requests=100, completed_tokens=1000,
)


def _plan(
    island_id: str = A5000_ISLAND,
    *,
    tp: int = 2,
    role: Role = Role.AGGREGATED,
    dtype: str = "bfloat16",
    knobs: VllmKnobs | None = None,
) -> DeploymentPlan:
    assignment = IslandAssignment(island_id=island_id, role=role, tp_size=tp)
    candidate = CandidateConfig(
        id="cand-1",
        model="meta-llama/Llama-3.1-8B",
        dtype=dtype,
        assignments=[assignment],
        knobs=knobs or VllmKnobs(),
    )
    return DeploymentPlan(
        plan_id="hp-test", model=candidate.model, candidate=candidate, predicted=_METRICS
    )


def _island_map(islands: list[ExecutionIsland]) -> dict[str, ExecutionIsland]:
    return {i.id: i for i in islands}


# ---------------------------------------------------------------------------
# device resolution
# ---------------------------------------------------------------------------

def test_device_index_from_id() -> None:
    assert device_index_from_id("gpu0") == 0
    assert device_index_from_id("gpu1") == 1
    assert device_index_from_id("npu13") == 13


def test_device_index_from_id_rejects_unnumbered() -> None:
    with pytest.raises(DeploymentError):
        device_index_from_id("accelerator")


def test_resolve_devices_and_visible_string(islands: list[ExecutionIsland]) -> None:
    island = _island_map(islands)[A5000_ISLAND]
    dmap = resolve_devices(island)
    assert isinstance(dmap, DeviceMap)
    assert dmap.node_id == "node0"
    assert dmap.device_indices == [0, 1]
    assert dmap.visible_devices == "0,1"


def test_resolve_devices_honours_override(islands: list[ExecutionIsland]) -> None:
    island = _island_map(islands)[A5000_ISLAND]
    dmap = resolve_devices(island, index_of=lambda _id: 7)
    assert dmap.device_indices == [7, 7]


# ---------------------------------------------------------------------------
# build_serve_command
# ---------------------------------------------------------------------------

def test_build_serve_command_argv_and_env(islands: list[ExecutionIsland]) -> None:
    island = _island_map(islands)[A5000_ISLAND]
    plan = _plan(tp=2)
    command = build_serve_command(plan, plan.candidate.assignments[0], island, port=8010)

    assert command.argv[:3] == ["vllm", "serve", "meta-llama/Llama-3.1-8B"]
    assert command.env == {"CUDA_VISIBLE_DEVICES": "0,1"}

    argv = command.argv
    assert "--tensor-parallel-size" in argv
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert argv[argv.index("--port") + 1] == "8010"
    assert argv[argv.index("--max-num-seqs") + 1] == "128"
    assert argv[argv.index("--max-num-batched-tokens") + 1] == "2048"
    assert argv[argv.index("--block-size") + 1] == "16"


def test_build_serve_command_boolean_knob_flags(islands: list[ExecutionIsland]) -> None:
    island = _island_map(islands)[A5000_ISLAND]
    # default knobs: prefix caching off, chunked prefill on.
    default_plan = _plan()
    default = build_serve_command(
        default_plan, default_plan.candidate.assignments[0], island
    )
    assert "--no-enable-prefix-caching" in default.argv
    assert "--enable-chunked-prefill" in default.argv

    on = _plan(knobs=VllmKnobs(enable_prefix_caching=True, enable_chunked_prefill=False))
    cmd = build_serve_command(on, on.candidate.assignments[0], island)
    assert "--enable-prefix-caching" in cmd.argv
    assert "--no-enable-chunked-prefill" in cmd.argv


def test_build_serve_command_omits_pp_when_one(islands: list[ExecutionIsland]) -> None:
    island = _island_map(islands)[A5000_ISLAND]
    plan = _plan()
    cmd = build_serve_command(plan, plan.candidate.assignments[0], island)
    assert "--pipeline-parallel-size" not in cmd.argv


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------

def test_validate_accepts_clean_cuda_plan(cluster, islands, profiles) -> None:
    backend = VllmCudaBackend(profiles=profiles)
    problems = backend.validate(_plan(tp=2), cluster, _island_map(islands))
    assert problems == []


def test_validate_rejects_backend_mixing(cluster, islands, profiles) -> None:
    # An ascend island served by the CUDA backend is a backend mix (rule 2).
    backend = VllmCudaBackend(profiles=profiles)
    problems = backend.validate(_plan(island_id=ASCEND_ISLAND, tp=2), cluster, _island_map(islands))
    assert any("backend" in p and "cuda" in p for p in problems)


def test_validate_rejects_non_aggregated_role(cluster, islands, profiles) -> None:
    backend = VllmCudaBackend(profiles=profiles)
    problems = backend.validate(_plan(role=Role.DECODE), cluster, _island_map(islands))
    assert any("aggregated" in p for p in problems)


def test_validate_rejects_bad_tp(cluster, islands, profiles) -> None:
    backend = VllmCudaBackend(profiles=profiles)
    problems = backend.validate(_plan(tp=3), cluster, _island_map(islands))
    assert any("does not divide" in p for p in problems)


def test_validate_rejects_unknown_island(cluster, islands, profiles) -> None:
    backend = VllmCudaBackend(profiles=profiles)
    problems = backend.validate(_plan(island_id="cuda-nope-node9"), cluster, _island_map(islands))
    assert any("not in this cluster" in p for p in problems)


def test_validate_rejects_incompatible_model(cluster, islands, profiles) -> None:
    backend = VllmCudaBackend(profiles=profiles)
    problems = backend.validate(_plan(dtype="float8"), cluster, _island_map(islands))
    assert any("no support" in p for p in problems)


def test_validate_without_profiles_skips_model_check(cluster, islands) -> None:
    backend = VllmCudaBackend()  # no profiles supplied
    problems = backend.validate(_plan(dtype="float8"), cluster, _island_map(islands))
    assert all("no support" not in p for p in problems)


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------

def test_ascend_and_k8s_backends_refuse(cluster, islands) -> None:
    from planner.deploy import KubernetesBackend, VllmAscendBackend

    for backend in (VllmAscendBackend(), KubernetesBackend()):
        problems = backend.validate(_plan(), cluster, _island_map(islands))
        assert problems  # non-empty: refuses
        with pytest.raises(NotImplementedError):
            backend.launch(_plan(), cluster, _island_map(islands))


# ---------------------------------------------------------------------------
# launch lifecycle guard (review: reused-id overwrite would orphan a run)
# ---------------------------------------------------------------------------

def test_launch_refuses_when_already_running(tmp_path, cluster, islands, profiles) -> None:
    import os

    backend = VllmCudaBackend(root=tmp_path, profiles=profiles)
    dep_id = _plan().plan_id
    pidfile = backend._pidfile(dep_id)
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(str(os.getpid()))  # a live pid -> is_running() True
    with pytest.raises(DeploymentError, match="already running"):
        backend.launch(_plan(), cluster, _island_map(islands))


def test_stop_is_idempotent_on_dead_pid(tmp_path, profiles) -> None:
    backend = VllmCudaBackend(root=tmp_path, profiles=profiles)
    dep_id = "hp-test"
    pidfile = backend._pidfile(dep_id)
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    # 2**31-1 is not a live pid; stop must not raise and must clear the pidfile.
    pidfile.write_text("2147483647")
    backend.stop(dep_id)
    assert not pidfile.exists()


def test_build_serve_command_max_model_len(islands) -> None:
    # --max-model-len is emitted only when the knob is set; None -> omitted
    # (vLLM then uses the model default, which is also the simulator's context).
    with_len = build_serve_command(
        _plan(knobs=VllmKnobs(max_model_len=8192)),
        _plan(knobs=VllmKnobs(max_model_len=8192)).candidate.assignments[0],
        _island_map(islands)[A5000_ISLAND], port=8000,
    )
    assert "--max-model-len" in with_len.argv
    assert with_len.argv[with_len.argv.index("--max-model-len") + 1] == "8192"

    without = build_serve_command(
        _plan(), _plan().candidate.assignments[0],
        _island_map(islands)[A5000_ISLAND], port=8000,
    )
    assert "--max-model-len" not in without.argv
