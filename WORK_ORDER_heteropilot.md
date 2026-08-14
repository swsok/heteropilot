# HeteroPilot 구현 작업지시서 (Claude Code용)

> 문서 버전: v1.0 (2026-08-07)
> 대상: Claude Code (구현 담당 AI 에이전트)
> 기준 코드베이스: `casys-kaist/LLMServingSim` (https://github.com/casys-kaist/LLMServingSim, `main` 브랜치)
> 상위 문서: "LLMServingSim 기반 이기종 GPU/NPU LLM Serving Orchestrator 연구·구현 제안서"

---

## 이 문서의 사용법

이 문서는 Claude Code가 순서대로 수행해야 할 **구현 작업 지시서**다. 각 Phase는 명시된 **완료 조건(Definition of Done)** 을 만족해야 다음 Phase로 진행한다. 스키마·인터페이스·디렉터리 구조는 본 문서에 정의된 형식을 그대로 따른다.

**절대 규칙 (모든 Phase에 적용):**

1. 기존 LLMServingSim의 `serving/` core 코드는 **Phase 5 이전까지 수정하지 않는다.** 모든 신규 기능은 새 디렉터리(`planner/`, `profiles/`, `experiments/`, `examples/`)에 추가한다.
2. 서로 다른 backend(CUDA GPU와 Ascend NPU 등)를 **하나의 Tensor Parallel 그룹에 절대 섞지 않는다.** 이기종성은 execution island 단위(replica 또는 Prefill/Decode 역할)로만 활용한다.
3. 각 기능은 별도 feature branch + 단일 목적 PR로 구현한다 (§10 참조).
4. 모든 실험 결과 파일에는 §3.8의 metadata를 반드시 기록한다.
5. 하드웨어 수치(bandwidth, latency, power 등)를 임의로 만들어내지 않는다. 측정값이 없으면 profile 파일에 `source: placeholder`를 명시하고, 실측했다고 표기하지 않는다.
6. 첫 optimizer는 RL이 아니라 **exhaustive enumeration + pruning**으로 구현한다. RL, Kubernetes operator, cross-vendor TP, live migration은 이 작업지시서 범위 밖이다.

---

# 1. 프로젝트 목표

## 1.1 한 줄 정의

> 현재 가용한 heterogeneous GPU/NPU cluster에서 workload와 network topology를 고려하여 TTFT/TPOT SLO를 보장하고, power cap 하에서 SLO-goodput/J를 최대화하도록 execution island, parallelism, replica count, Prefill/Decode placement, vLLM runtime configuration을 공동 최적화한다. LLMServingSim은 후보 configuration의 performance/energy predictor로 사용하고, 실제 vLLM 측정으로 지속 calibration한다.

## 1.2 시스템 구성 (4-Plane)

```text
+------------------------------------------------------------+
|                      User / API                            |
|  model, traffic, TTFT, TPOT, power, tokens/J, objective    |
+------------------------------+-----------------------------+
                               |
                               v
+------------------------------------------------------------+
|                    Control Plane   <-- 이번 구현의 핵심     |
| Requirement Normalizer                                     |
| Cluster Inventory -> Candidate Generator -> Optimizer      |
| Deployment Planner -> Replanner                            |
+----------------------+------------------+------------------+
                       |                  |
              prediction|                  |deployment
                       v                  v
+--------------------------------+   +------------------------+
|      Simulation Plane          |   |      Data Plane        |
| LLMServingSim (기존 코드 활용) |   | vLLM CUDA islands      |
| ASTRA-Sim                      |   | vLLM-Ascend islands    |
| Surrogate / Pareto DB          |   | request router         |
+---------------+----------------+   +-----------+------------+
                ^                                |
                | calibration                    | metrics
                |                                v
+------------------------------------------------------------+
|                   Profiling Plane                          |
| GPU/NPU profiling | network profiling | power profiling    |
+------------------------------------------------------------+
```

## 1.3 핵심 추상화: Execution Island

**Execution Island**는 다음을 모두 만족하는 accelerator 집합이다.

- 동일 또는 호환 runtime backend (예: cuda, ascend)
- 서로 collective communication이 가능
- 대상 model의 kernel이 지원됨
- 하나의 vLLM engine instance를 안정적으로 실행 가능

예시:

```text
Island 0: 4 x H100, backend=cuda, interconnect=NVLink
Island 1: 8 x RTX PRO 6000, backend=cuda, interconnect=PCIe
Island 2: 8 x Ascend, backend=ascend, interconnect=HCCS
```

TP/PP는 island 내부에서만 허용:

```text
H100 x 4        -> TP=4   허용
Ascend x 2      -> TP=2   backend가 지원하면 허용
H100 + Ascend   -> TP=2   금지 (candidate 생성 단계에서 자동 배제)
```

이기종성은 replica/phase 단위로만 활용:

```text
Prefill replicas: H100 island
Decode replicas : Ascend island
```

---

# 2. 저장소 구조

기존 LLMServingSim 저장소를 fork/clone한 뒤, 아래 디렉터리를 **신규 추가**한다. 기존 디렉터리(`serving/`, `profiler/`, `bench/`, `configs/`)는 유지한다.

## 2.1 신규 디렉터리 전체 구조

```text
LLMServingSim/                      # 기존 저장소 루트
├── serving/                        # (기존) simulator core — Phase 5까지 수정 금지
├── profiler/                       # (기존) vLLM layerwise profiler
├── bench/                          # (기존) 실제 vLLM vs sim 검증 도구
├── configs/cluster/                # (기존) simulator 입력 형식 — 그대로 유지
│
├── planner/                        # [신규] Control Plane
│   ├── __init__.py
│   ├── __main__.py                 # CLI entry point (§6)
│   ├── spec.py                     # ServiceSpec 정의 및 로더 (§3.1)
│   ├── inventory.py                # ClusterSpecV2, island detection (§3.2, §5.2)
│   ├── topology.py                 # topology graph, path/bandwidth 계산 (§5.3)
│   ├── candidate_generator.py      # 후보 열거 + pruning pipeline (§5.4)
│   ├── plan.py                     # DeploymentPlan / PlannerOutput 자료구조 (§3.4, §3.5)
│   ├── predictor/
│   │   ├── __init__.py
│   │   ├── llmservingsim.py        # 시뮬레이터 subprocess wrapper + 결과 parser (§5.5)
│   │   ├── surrogate.py            # (Phase 2 이후) 빠른 근사 예측기
│   │   └── calibration.py          # (Phase 4) sim vs real 보정 모델 (§5.8)
│   ├── optimizer/
│   │   ├── __init__.py
│   │   ├── feasibility.py          # hard constraint 검사 (§5.6)
│   │   ├── exhaustive.py           # exhaustive oracle (§5.6)
│   │   ├── pareto.py               # Pareto frontier 계산 (§5.6)
│   │   ├── greedy.py               # (Phase 2 이후) greedy baseline
│   │   └── cpsat.py                # (선택, 후순위) CP-SAT solver
│   ├── deploy/
│   │   ├── base.py                 # ServingBackend 추상 인터페이스 (§5.7)
│   │   ├── vllm_cuda.py            # CUDA vLLM launcher (Phase 4)
│   │   ├── vllm_ascend.py          # vLLM-Ascend launcher (Phase 4)
│   │   └── kubernetes.py           # (범위 외 stub만, 구현 금지)
│   └── monitor/
│       ├── metrics.py              # 배포 인스턴스 metric 수집 (Phase 4)
│       └── replanner.py            # (Phase 6) online replanning
│
├── profiles/                       # [신규] 하드웨어/네트워크 profile catalog
│   ├── accelerators/
│   │   ├── h100.yaml
│   │   ├── rtxpro6000.yaml
│   │   └── ascend_target.yaml
│   ├── networks/
│   │   ├── nvlink.yaml
│   │   ├── pcie_gen5.yaml
│   │   ├── ib_100g.yaml
│   │   └── ib_400g.yaml
│   └── calibration/                # (Phase 4) calibration 결과 저장
│
├── experiments/                    # [신규] 실험 설정·스크립트·결과
│   ├── configs/
│   │   ├── services/               # ServiceSpec YAML들
│   │   ├── clusters/               # ClusterSpecV2 YAML들
│   │   └── sweeps/                 # sweep 정의 (예: network 1G~400G)
│   ├── scripts/                    # one-command 실험 runner
│   ├── results/                    # raw 결과 (metadata 포함, git-lfs 또는 .gitignore 판단)
│   └── figures/                    # 그림 생성 스크립트 및 산출물
│
├── examples/                       # [신규] 최소 동작 예제
│   ├── service_specs/
│   │   └── qwen3-32b.yaml
│   └── clusters/
│       └── heterogeneous-lab.yaml
│
└── tests/                          # [신규] planner 단위/통합 테스트
    ├── test_spec.py
    ├── test_inventory.py
    ├── test_topology.py
    ├── test_candidate_generator.py
    ├── test_feasibility.py
    ├── test_exhaustive.py
    └── test_pareto.py
```

## 2.2 기존 파일과 신규 기능의 대응 관계

| 기존 파일/모듈 | 현재 역할 | 신규 기능에서의 활용 |
|---|---|---|
| `serving/core/request.py` | 요청 상태, TTFT/TPOT/ITL | SLO 측정 소스. Phase 5+에서 per-request SLO metadata 확장 |
| `serving/core/router.py` | RR/RAND/LOAD/CUSTOM routing | Phase 5+에서 `_custom_select()` hook으로 SLO-aware routing 추가 |
| `serving/core/scheduler.py` | vLLM-style continuous batching | 수정하지 않고 그대로 사용 |
| `serving/core/trace_generator.py` | profiled latency 기반 trace | 후보 구성의 compute latency 예측에 그대로 사용 |
| `serving/core/config_builder.py` | cluster config → ASTRA-Sim config | Phase 5에서 ClusterSpecV2 → 기존 config compile adapter 추가 |
| `serving/core/power_model.py` | node/component 에너지 | power cap, J/request, tokens/J 계산 소스 |
| `serving/core/memory_model.py` | weight/KV 메모리 | memory feasibility pruning의 근거 |
| `profiler/` | vLLM layerwise profiler | profile registry의 데이터 소스. Phase 3에서 NPU importer 추가 |
| `bench/` | 실제 vLLM vs sim 비교 | Phase 0 재현 대상, Phase 4 calibration 데이터 소스 |
| `configs/cluster/*.json` | simulator 입력 | **planner의 컴파일 타깃 형식**으로 유지 (직접 수정 금지) |

핵심 데이터 흐름: planner는 ClusterSpecV2(풍부한 스키마)로 추론하고, 시뮬레이션 실행 시점에 기존 `configs/cluster/*.json` 형식으로 **compile해서** 시뮬레이터에 넘긴다. 시뮬레이터 입력 형식 자체는 바꾸지 않는다.

---

# 3. 데이터 스키마 정의

아래 스키마는 모두 Pydantic(v2) 모델로 구현하고, YAML 로더/밸리데이터/JSON schema export를 제공한다. 필드명은 아래와 정확히 일치시킨다.

## 3.1 ServiceSpec (`planner/spec.py`)

사용자 입력. 파일 위치 예: `examples/service_specs/qwen3-32b.yaml`

```yaml
service:
  model: Qwen/Qwen3-32B          # HuggingFace model id
  dtype: bfloat16
  kv_cache_dtype: auto

traffic:                          # 자원 크기 결정에 필수 — 생략 불가
  arrival_rate_rps: 15
  input_tokens:
    p50: 512
    p95: 4096
    p99: 8192
  output_tokens:
    p50: 128
    p95: 512
  burstiness: 2.0                 # 선택, 기본 1.0
  prefix_share_ratio: 0.20        # 선택, 기본 0.0

slo:
  ttft:
    percentile: 99                # 50/95/99만 허용
    max_ms: 500
  tpot:
    percentile: 99
    max_ms: 40
  max_cluster_power_w: 3000       # peak power 상한
  min_tokens_per_joule: 0.40      # 에너지 효율 하한

objective:
  primary: minimize_energy        # 허용값: minimize_energy |
                                  #        maximize_slo_goodput_per_joule |
                                  #        minimize_active_accelerators
  secondary: minimize_active_accelerators
```

검증 규칙:

- `traffic` 블록이 없으면 로드 실패 + "SLO만으로는 자원 크기를 결정할 수 없음" 오류 메시지 출력.
- percentile은 50/95/99만 허용.
- `objective.primary`와 `secondary`가 같으면 오류.

## 3.2 ClusterSpecV2 (`planner/inventory.py`)

파일 위치 예: `examples/clusters/heterogeneous-lab.yaml`

### 3.2.1 Accelerator inventory

```yaml
cluster_id: lab-cluster

nodes:
  - id: node0
    accelerators:
      - id: gpu0
        type: GPU                  # GPU | NPU
        vendor: NVIDIA
        model: H100-80GB
        backend: cuda              # cuda | ascend | <향후 추가>
        memory_gb: 80
        state: FREE                # FREE | ALLOCATED | RESERVED | DEGRADED
        profile: profiles/accelerators/h100.yaml   # 성능/전력 profile 참조
      - id: gpu1
        type: GPU
        vendor: NVIDIA
        model: H100-80GB
        backend: cuda
        memory_gb: 80
        state: FREE
        profile: profiles/accelerators/h100.yaml
    nics:
      - id: mlx5_0
        type: infiniband
        speed_gbps: 400

  - id: node1
    accelerators:
      - id: npu0
        type: NPU
        vendor: Huawei
        model: ASCEND_TARGET
        backend: ascend
        memory_gb: 64
        state: FREE
        profile: profiles/accelerators/ascend_target.yaml
    nics:
      - id: mlx5_0
        type: infiniband
        speed_gbps: 400
```

동적 상태 필드(Phase 4 이후 monitor가 채움, 그 전까지는 선택 필드):

```yaml
        utilization: 0.0          # 0~1
        memory_used_gb: 0.0
        health: OK                # OK | WARN | FAIL
        power_state: ACTIVE       # ACTIVE | IDLE | SLEEP
        queue_depth: 0
```

### 3.2.2 Topology graph (links)

```yaml
links:
  - id: nvlink-0
    src: node0/gpu0               # "<node_id>/<device_or_nic_id>" 형식
    dst: node0/gpu1
    type: NVLINK                  # NVLINK | PCIE | INFINIBAND | ETHERNET | HCCS
    bandwidth_gbps: 900
    latency_ns: 1000
    energy_per_bit_pj: 1.0        # 선택
    duplex: full
    contention_group: nvswitch0   # 같은 물리 자원을 공유하는 link 집합 식별자

  - id: pcie-gpu0
    src: node0/gpu0
    dst: node0/mlx5_0
    type: PCIE
    bandwidth_gbps: 256
    latency_ns: 1500
    contention_group: pcie-root0

  - id: ib-node0-node1
    src: node0/mlx5_0
    dst: node1/mlx5_0
    type: INFINIBAND
    bandwidth_gbps: 400
    latency_ns: 5000
    energy_per_bit_pj: 5.0
```

주의: 위 수치는 **스키마 예시일 뿐**이다. 실제 예제 파일에는 profile 출처(`source: measured | vendor_spec | placeholder`)를 각 link에 기록한다.

`contention_group`의 의미: 같은 그룹에 속한 link들의 동시 flow는 nominal bandwidth를 나눠 쓴다. Phase 2의 fast model에서는 "동일 contention_group 내 활성 flow 수로 bandwidth를 균등 분할"하는 단순 모델을 사용한다. 표현 대상: shared PCIe root, NVSwitch domain, NIC, ToR switch, IB fabric path.

## 3.3 Accelerator Profile (`profiles/accelerators/*.yaml`)

```yaml
profile_id: h100-80gb
vendor: NVIDIA
model: H100-80GB
backend: cuda
memory_gb: 80
memory_bandwidth_gbps: 3350
tdp_w: 700
idle_power_w: 90
source: vendor_spec               # measured | vendor_spec | placeholder
perf_data: profiler/perf/h100/    # profiler CSV 위치 (§3.7 contract)
supported_models:                 # compatibility matrix (§11 Risk 1)
  - pattern: "meta-llama/Llama-3.1-*"
    dtypes: [bfloat16, fp8]
  - pattern: "Qwen/Qwen3-*"
    dtypes: [bfloat16]
max_tp_size: 8
notes: ""
```

## 3.4 DeploymentPlan (`planner/plan.py`)

planner의 출력이자 deployer의 입력.

```yaml
plan_id: hp-00042
model: Qwen/Qwen3-32B

instances:
  - id: prefill-0
    backend: vllm-cuda            # vllm-cuda | vllm-ascend
    node: node0
    devices: [gpu0, gpu1]
    island: cuda-h100-node0
    role: prefill                 # prefill | decode | aggregated
    tp_size: 2
    pp_size: 1
    max_num_seqs: 32
    max_num_batched_tokens: 8192
    enable_chunked_prefill: true
    enable_prefix_caching: true
    kv_cache_dtype: auto
    block_size: 16

  - id: decode-0
    backend: vllm-ascend
    node: node2
    devices: [npu0, npu1, npu2, npu3]
    island: ascend-node2
    role: decode
    tp_size: 4
    pp_size: 1
    max_num_seqs: 256

routing:
  policy: pd_split                # single | rr | load | pd_split
  prefill_instances: [prefill-0]
  decode_instances: [decode-0]

predicted:
  p99_ttft_ms: 420
  p99_tpot_ms: 35
  average_power_w: 2015
  peak_power_w: 2450
  tokens_per_joule: 0.68
  slo_goodput_rps: 13.2

robust_margin:                    # Phase 4 calibration 이후 채움, 그 전엔 0
  ttft_percent: 6.5
  tpot_percent: 5.0
```

## 3.5 PlannerOutput (`planner/plan.py`)

### Feasible한 경우

```yaml
feasible: true
recommended:
  plan_id: hp-00042
  score:
    objective: maximize_slo_goodput_per_joule
    value: 0.61
  # ... DeploymentPlan 전체 embed ...

alternatives:                     # Pareto frontier상의 대안들
  - plan_id: hp-00037
    note: lower latency, higher power
    predicted: { ... }
  - plan_id: hp-00051
    note: lower energy, smaller SLO margin
    predicted: { ... }

rejected_summary:                 # pruning 단계별 탈락 통계
  backend_incompatible: 14
  memory_infeasible: 22
  topology_infeasible: 3
  slo_violated: 9
  power_violated: 4
```

### Infeasible한 경우 (단순 실패로 끝내지 말 것)

```yaml
feasible: false
reason: no currently available configuration satisfies all constraints

closest_plan:                     # 위반 크기가 가장 작은 plan
  plan_id: hp-00061
  p99_ttft_ms: 580
  p99_tpot_ms: 38
  tokens_per_joule: 0.71

violated_constraints:
  - metric: p99_ttft_ms
    target: 500
    predicted: 580

suggestions:                      # 규칙 기반으로 생성
  - add 2 high-compute GPUs for prefill
  - relax TTFT SLO from 500ms to 600ms
  - lower admitted request rate from 18 to 14 rps
```

## 3.6 PerformanceEnvelope DB (Phase 2 후반)

시뮬레이션 결과 캐시. 초기 구현은 디렉터리 기반 YAML/parquet 저장으로 충분하다 (DB 서버 금지 — 파일 기반 유지).

키 구조:

```text
PerformanceEnvelope[
    model, dtype, accelerator, tp, pp, ep, pd_role,
    scheduler_config_hash, network_class, workload_bucket
]
```

각 entry 값:

```yaml
metrics:
  p50_ttft_ms: 210
  p95_ttft_ms: 320
  p99_ttft_ms: 460
  p50_tpot_ms: 22
  p99_tpot_ms: 37
  throughput_tps: 5200
  slo_goodput_rps: 13.2
  average_power_w: 1810
  peak_power_w: 2260
  total_energy_j: 12300
  tokens_per_joule: 0.73
  peak_memory_gb: 71.4
  network_bytes: 8.1e9
  network_busy_ratio: 0.64
provenance:                       # §12.3 metadata 그대로
  git_commit: ...
  timestamp: ...
```

`workload_bucket` 정의: (input p50 구간, output p50 구간, arrival rate 구간)의 3차원 버킷. 초기 구간 경계는 input {<1k, 1k-4k, >4k}, output {<128, 128-512, >512}, rps {<5, 5-20, >20}로 하드코딩하되 상수 모듈로 분리한다.

## 3.7 Profiler CSV Contract (Phase 3)

기존 profiler output 형식을 **공통 contract**로 삼는다. NPU 데이터도 같은 schema로 import한다.

```text
profiler/perf/<hardware>/<model>/<variant>/tp<N>/
    dense.csv
    per_sequence.csv
    attention.csv
    moe.csv          # MoE 모델만
    meta.yaml        # 측정 환경: backend 버전, driver, 날짜, source
```

구현 착수 시 첫 작업: 기존 `profiler/` 코드와 실제 output CSV의 컬럼을 확인하고, 그 컬럼 정의를 `profiler/CONTRACT.md`로 문서화한 뒤 NPU importer가 이를 따르게 한다. **컬럼 스키마를 추측으로 정의하지 말고 반드시 기존 산출물에서 역추출할 것.**

## 3.8 실험 metadata (모든 결과 파일에 필수)

```yaml
git_commit:                # 이 저장소의 commit hash
llmservingsim_commit:      # upstream 기준 commit
vllm_version:
backend_version:           # torch/torch-npu/CANN 등
model_revision:
hardware_profile_hash:     # 사용한 profiles/ 파일들의 hash
cluster_spec_hash:
service_spec_hash:
dataset_hash:
random_seed:
command:                   # 실행한 전체 커맨드라인
timestamp:
```

`planner/util/provenance.py`(신규)에 이 metadata를 자동 수집·기록하는 helper를 만들고 모든 runner가 사용한다.

---

# 4. 핵심 지표 정의 (구현 시 정확히 이 정의를 따를 것)

```text
energy_efficiency = completed_tokens / total_energy_joule    # tokens/J
                  (tokens/sec ÷ watt 와 차원 동일)

SLO attainment  = TTFT와 TPOT SLO를 "모두" 만족한 요청 비율
SLO goodput     = 단위 시간에 TTFT/TPOT를 모두 만족하며 완료된 token 수 (또는 요청 수)
SLO-goodput/J   = SLO 만족 요청들의 token 수 / 전체 소비 joule
```

추가로 항상 함께 기록: `J/request`, `J/output-token`, `average W`, `peak W`, `SLO-goodput/W`.

주 지표는 평균이 아니라 **P50/P95/P99**다. metric parser는 percentile 계산을 단일 유틸 함수(`planner/util/percentile.py`)로 통일한다 (보간 방식 차이로 인한 불일치 방지 — numpy `percentile`의 `linear` interpolation으로 고정).

---

# 5. 모듈별 구현 명세

## 5.1 `planner/spec.py`

- `ServiceSpec` Pydantic 모델 (§3.1 스키마).
- `load_service_spec(path) -> ServiceSpec`: YAML 로드 + 검증.
- 검증 실패 시 어떤 필드가 왜 잘못됐는지 명확한 한국어/영어 메시지.

## 5.2 `planner/inventory.py`

- `ClusterSpecV2`, `Node`, `Accelerator`, `Nic`, `Link` Pydantic 모델 (§3.2).
- `load_cluster_spec(path) -> ClusterSpecV2`.
- `detect_islands(cluster: ClusterSpecV2) -> list[ExecutionIsland]`:
  - 같은 node 내에서 backend가 동일하고, links 그래프상 서로 연결된(직접 또는 같은 고속 interconnect domain을 통해) accelerator들을 하나의 island로 묶는다.
  - island id 규칙: `"{backend}-{model_slug}-{node_id}"` (예: `cuda-h100-node0`).
  - `ExecutionIsland` 필드: `id`, `backend`, `accelerator_ids`, `node_id`, `interconnect_type`, `total_memory_gb`, `max_tp_candidates` (island 크기의 약수 중 profile의 `max_tp_size` 이하).
- `compatibility(model, dtype, accelerator_profile) -> bool`: profile의 `supported_models` 패턴 매칭. 미지원이면 candidate 생성에서 자동 배제.

## 5.3 `planner/topology.py`

- `TopologyGraph`: links로부터 만든 무방향 그래프 (networkx 사용 가능).
- `path(src_device, dst_device) -> list[Link]`: 최소 hop 경로.
- `effective_bandwidth(path, concurrent_flows: dict[contention_group, int]) -> float`:
  - 경로상 각 link의 `bandwidth / flows_in_group`을 계산해 최솟값(bottleneck)을 반환.
- `path_latency(path) -> float`: link latency 합.
- `transfer_time(bytes, path, flows) -> float`, `transfer_energy(bytes, path) -> float` (energy_per_bit_pj 합산; 값 없는 link는 0 + warning).
- Two-level 정책:
  - **Level 1 (fast)**: 후보 대량 평가용. TP 통신은 island interconnect class 대표값, cross-node는 NIC-IB 경로 대표값 사용.
  - **Level 2 (path-aware)**: Top-K 후보에만 실제 경로 + contention 반영. Phase 5에서 구현.

## 5.4 `planner/candidate_generator.py`

Pruning pipeline (순서 고정, 각 단계에서 탈락 사유를 기록):

```text
All available resources
  -> 1. Backend/model compatibility filter
  -> 2. Memory feasibility filter          # weight + KV + activation 추정
  -> 3. Parallelism feasibility filter     # TP는 island 크기의 약수, max_tp_size 이하
  -> 4. Topology/network lower-bound filter  # 이론적 최선 통신 시간으로도 SLO 불가면 배제
  -> 5. Analytical performance lower-bound filter  # profile 기반 이론 최소 latency로 배제
  -> 6. (Phase 2 후반) Surrogate predictor -> Top-K
  -> 7. Full LLMServingSim simulation
```

- Memory feasibility는 기존 `serving/core/memory_model.py`의 로직을 **호출**해서 계산한다 (복제 금지). subprocess 경계 때문에 직접 import가 어려우면 동일 수식을 `planner/util/memory.py`로 옮기되 출처 주석을 남긴다.
- Candidate 자료구조 `CandidateConfig`: `island_assignments` (role -> island), `tp`, `pp`, `dp_replicas`, `serving_arch` (aggregated | pd_split), `vllm_knobs` (max_num_seqs, max_num_batched_tokens, chunked_prefill, prefix_caching, kv_cache_dtype, block_size).
- MVP에서 열거하는 결정 변수 (§8의 MVP 범위와 일치):
  - island 선택, accelerator 수, TP degree, DP replica count, `max_num_seqs`, `max_num_batched_tokens`.
  - vllm_knobs는 초기에 소수의 이산 후보만 (예: max_num_seqs ∈ {32, 128, 256}).
- 탈락 candidate는 `(candidate, rejection_stage, reason)`으로 모두 보존 → `rejected_summary` 생성.

## 5.5 `planner/predictor/llmservingsim.py`

- `compile_to_sim_config(candidate, cluster) -> Path`: CandidateConfig + ClusterSpecV2를 기존 `configs/cluster/*.json` 형식으로 변환해 임시 디렉터리에 기록. **기존 config 파일의 실제 스키마를 먼저 읽고 그 형식을 정확히 따를 것** (저장소의 `configs/cluster/README.md`와 기존 예제 참조).
- `run_simulation(sim_config, workload) -> SimResult`: LLMServingSim을 subprocess로 실행. timeout, stdout/stderr 캡처, 실패 시 재시도 1회.
- `parse_results(output_dir) -> SimResult`: `sim.csv` 등 출력에서 TTFT/TPOT percentile, throughput, power/energy를 추출. 출력 파일의 실제 컬럼명은 **Phase 0에서 실행해본 결과물에서 확인**하고 하드코딩하되 상수로 분리.
- workload 생성: ServiceSpec의 traffic 분포로부터 arrival/token-length trace 생성 (`planner/util/workload.py`). random_seed 필수.

## 5.6 `planner/optimizer/`

### `feasibility.py` — hard constraints

```text
Memory(x)          <= AvailableMemory(x)
RequiredDevices(x) <= FreeDevices(x)      # state == FREE인 장치만
BackendCompatible(x) == true
ModelSupported(x)  == true
TopologyCompatible(x) == true
P{p}_TTFT(x)       <= slo.ttft.max_ms
P{p}_TPOT(x)       <= slo.tpot.max_ms
PeakPower(x)       <= slo.max_cluster_power_w
TokensPerJoule(x)  >= slo.min_tokens_per_joule
```

각 검사 함수는 `(passed: bool, violation: Violation | None)`을 반환해 infeasible 출력 생성에 재사용한다.

### 최적화 방식 — weighted sum 금지, lexicographic 고정

```text
Stage 1. Feasibility: hard constraint를 모두 만족하는 plan만 남김
Stage 2. Primary objective로 정렬 (예: maximize SLO_goodput_per_joule)
Stage 3. Tie-break (순서 고정):
         active accelerator 수 최소화
         -> fragmentation 최소화
         -> reconfiguration cost 최소화 (Phase 6 전까지는 0)
```

### `exhaustive.py` — oracle

- 작은 cluster(장치 ≤ 16개 수준)에서 pruning 후 모든 candidate를 full simulation.
- 존재 이유: heuristic optimality gap 측정, optimizer 버그 검출, 논문 oracle baseline, surrogate error와 search error 분리. **어떤 경우에도 삭제하지 말 것.**

### `pareto.py`

- 목적 차원: (P99 TTFT, P99 TPOT, peak power, tokens/J, active device count).
- 지배(dominance) 판정으로 frontier 계산, recommended + alternatives 생성.
- 출력 예시는 §3.5. 사용자에게는 항상 recommended 1개 + Pareto 대안들을 함께 제시.

## 5.7 `planner/deploy/` (Phase 4)

```python
class ServingBackend:                      # base.py
    def validate(self, plan: DeploymentPlan) -> list[str]: ...   # 문제 목록, 빈 리스트면 OK
    def launch(self, plan: DeploymentPlan) -> DeploymentHandle: ...
    def stop(self, deployment_id: str) -> None: ...
    def metrics(self, deployment_id: str) -> DeploymentMetrics: ...

class VllmCudaBackend(ServingBackend): ...     # vllm serve 커맨드 조립, CUDA_VISIBLE_DEVICES 설정
class VllmAscendBackend(ServingBackend): ...   # vLLM-Ascend, ASCEND_RT_VISIBLE_DEVICES 설정
```

- launcher는 **local/SSH 실행**만 지원한다. Kubernetes는 adapter 자리(stub)만 두고 구현하지 않는다.
- `metrics()`는 vLLM의 Prometheus endpoint(`/metrics`)에서 TTFT/TPOT/throughput을 수집.
- 전력: NVIDIA는 `nvidia-smi`/NVML 폴링, Ascend는 `npu-smi` 폴링. 수집 주기와 적분 방식(단순 사다리꼴)을 문서화.

## 5.8 `planner/predictor/calibration.py` (Phase 4)

첫 버전은 hardware별 선형 보정:

```text
real_ttft ≈ alpha_hw * sim_ttft + beta_hw
real_tpot ≈ gamma_hw * sim_tpot + delta_hw
```

- `bench/`의 sim vs real 비교 산출물을 학습 데이터로 사용.
- (hardware, workload_bucket)별 prediction error 분포 저장: `mean_error, p95_abs_error, worst_error, sample_count`.
- Robust planning: `robust_metric = predicted * (1 + p95_error)`. feasibility 검사는 robust 값으로 수행. 예: 예측 TTFT 450ms, P95 오차 +8% → robust 486ms → SLO 500ms이면 feasible; 예측 480ms면 robust 518ms → 배제.
- 발전형(후순위): `correction = f(accelerator, model, TP, batch_size, prompt_len, decode_len, network_utilization)` residual 모델.

---

# 6. CLI 명세 (`planner/__main__.py`)

```bash
# 1. 클러스터 점검: island 목록과 가능한 TP 후보 출력 (Phase 1 완료 조건)
python -m planner inspect-cluster --cluster examples/clusters/heterogeneous-lab.yaml

# 2. 계획 수립 (Phase 2 완료 조건)
python -m planner plan \
  --service examples/service_specs/qwen3-32b.yaml \
  --cluster examples/clusters/heterogeneous-lab.yaml \
  --output outputs/plans/qwen3-plan.yaml

# 3. 선택된 plan을 특정 dataset으로 재시뮬레이션 검증
python -m planner validate-plan \
  --plan outputs/plans/qwen3-plan.yaml \
  --dataset workloads/sharegpt-qwen3.jsonl

# 4. 실제 배포 (Phase 4)
python -m planner deploy --plan outputs/plans/qwen3-plan.yaml

# 5. 배포 상태/metric 조회 (Phase 4)
python -m planner status --deployment hp-00042
```

`plan` 커맨드의 stdout은 반드시 다음을 포함한다:

```text
Feasible candidates (개수 + 상위 목록)
Rejected candidates + 단계별 reason 집계
Recommended plan
Pareto alternatives
Predicted metrics (TTFT/TPOT percentiles, power, tokens/J)
```

---

# 7. 구현 단계 (Phase 0~6) — 순서 엄수

**중요: Phase 0~2의 정적 planner를 완성하기 전에 topology graph, P/D placement, replanning(Phase 5~6)을 시작하지 않는다.**

## Phase 0. Baseline 고정 (가장 먼저)

작업:

1. `casys-kaist/LLMServingSim`을 clone하고 특정 commit으로 pin (commit hash를 `UPSTREAM_COMMIT` 파일에 기록).
2. README의 기존 example 시뮬레이션을 1개 이상 재현.
3. `bench/`의 실제 vLLM validation 예제를 재현 (GPU가 없는 환경이면 시뮬레이션 부분만 재현하고, real vLLM 부분은 실행 커맨드와 필요 환경을 `docs/phase0_bench_plan.md`로 문서화).
4. 시뮬레이터의 실제 출력 파일 형식(sim.csv 컬럼 등)과 `configs/cluster/*.json` 실제 스키마를 조사해 `docs/phase0_formats.md`에 기록. **이 문서가 §5.5 parser/compiler 구현의 근거가 된다.**

완료 조건:

```text
same dataset + same engine settings 기준으로
  sim.csv 생성 확인
  (가능 환경이면) real vLLM 결과 생성 확인
  validation report 생성 확인
docs/phase0_formats.md 작성 완료
```

## Phase 1. Static Requirement + Inventory

작업: `spec.py`, `inventory.py`(island detection, compatibility), memory feasibility filter, `examples/` 예제 2개, 단위 테스트.

아직 하지 않는 것: 실제 배포, 시뮬레이션 자동 실행.

완료 조건: `python -m planner inspect-cluster ...`가 island 목록, 각 island의 TP 후보, model 호환성 결과를 출력. `tests/test_spec.py`, `test_inventory.py` 통과.

## Phase 2. Offline Simulator-Guided Planner ★ 첫 논문용 핵심 결과

작업: candidate enumeration + pruning(§5.4), sim config compiler + subprocess wrapper + parser(§5.5), feasibility(§5.6), exhaustive oracle, Pareto, `plan`/`validate-plan` CLI, PerformanceEnvelope 캐시.

완료 조건 (= MVP 성공 조건):

1. ServiceSpec 입력 → 후보 자동 생성.
2. 각 후보에 대해 LLMServingSim 결과 획득.
3. SLO/power/tokens-J constraint 자동 검증.
4. 최적 plan + Pareto alternatives 출력 (infeasible 시 §3.5의 진단 출력).
5. 작은 cluster에서 exhaustive oracle과 동일한 최적 plan 도출을 테스트로 검증.
6. 동일 입력 + 동일 seed → 동일 출력 (재현성 테스트).

## Phase 3. Heterogeneous Hardware Profiles

작업: 두 번째 GPU profile 추가, `profiler/CONTRACT.md` 작성(§3.7), NPU CSV importer (`CsvProfileImporter`), power/network profile 파일들.

```python
class HardwareProfilerBackend:
    def profile(self, spec): ...

class CudaVllmProfiler(HardwareProfilerBackend): ...   # 기존 profiler 래핑
class AscendVllmProfiler(HardwareProfilerBackend): ... # V2에서 구현
class CsvProfileImporter(HardwareProfilerBackend): ... # V1: 외부 측정 CSV import
```

NPU bring-up 순서: **V1** 외부 benchmark 데이터 CSV import → **V2** vLLM-Ascend native profiling adapter → **V3** backend별 kernel/communication 특성. V1만으로도 planner 연구를 진행할 수 있게 한다.

완료 조건: 서로 다른 accelerator class 2~3개가 포함된 cluster spec으로 `plan`이 동작하고, 후보에 이기종 island 선택이 나타남.

## Phase 4. Real Deployment + Calibration

작업: `deploy/vllm_cuda.py`, `deploy/vllm_ascend.py`, `monitor/metrics.py`, `predictor/calibration.py`(§5.8), `bench/` 확장(planner가 선택한 deployment config 자동 replay).

완료 조건: `deploy`로 plan 실행 → `status`로 실측 TTFT/TPOT/power 수집 → sim 예측과 비교한 calibration report 생성 → robust margin이 이후 `plan` 실행에 반영됨.

## Phase 5. Topology-Aware P/D ★ 가장 추천하는 논문 핵심

작업: Level 2 path-aware model(§5.3), KV transfer estimator(크기 = KV bytes/token × prompt tokens, 경로 bandwidth/latency/energy 반영), Prefill/Decode 독립 placement candidate generator, `config_builder.py`에 ClusterSpecV2 → ASTRA config adapter 추가(이 시점부터 기존 파일 수정 허용), network-aware routing.

P/D split 채택 조건을 candidate 평가에 명시적으로 포함:

```text
Benefit_of_split > KV_transfer_latency + KV_transfer_energy + queueing_penalty
```

완료 조건: `GPU P + GPU D / NPU P + NPU D / GPU P + NPU D / NPU P + GPU D` 4조합이 후보로 평가되고, network sweep(예: 25/100/200/400G)에서 P/D 이득이 사라지는 bandwidth 지점을 재현하는 실험 스크립트가 동작.

## Phase 6. Online Replanning (후속 논문 범위 — 착수 전 사용자 승인 필요)

작업: workload estimator, replan trigger(arrival rate 변화, queue 증가, SLO violation 확률, device failure, power budget 변경), migration/warmup cost 모델, admission control.

재구성 결정 조건: `gain_after_replan > migration_cost + warmup_cost + KV_movement_cost`.

## Phase별 기존 파일 수정 허용 시점 요약

| 파일 | 수정 허용 시점 | 내용 |
|---|---|---|
| `serving/core/config_builder.py` | Phase 5 | ClusterSpecV2 → ASTRA config adapter |
| `serving/core/router.py` | Phase 5+ | `_custom_select()` hook에 SLO-aware routing |
| `serving/core/request.py` | Phase 5+ | `slo_class, ttft_deadline, tpot_target, priority, tenant_id` 추가 |
| `serving/core/power_model.py` | Phase 4 | per-instance/per-request/network/phase별 energy 외부 API 노출 (읽기 전용 확장) |
| `profiler/` | Phase 3 | backend interface + NPU importer 추가 |
| `bench/` | Phase 4 | deployment config 자동 replay 확장 |

---

# 8. MVP 범위 (Phase 2 완료 시점 기준)

## 포함

1 model/service · 2~3 accelerator classes · execution island 단위 선택 · TP + DP replica count · aggregated serving · TTFT P99 · TPOT P99 · power cap · tokens/J · static traffic 분포 · static cluster availability · exhaustive + pruning · LLMServingSim 예측.

## 제외 (구현 금지 — 발견 시 즉시 중단하고 사용자에게 보고)

GPU+NPU mixed TP · dynamic migration · multi-tenant · RL · Kubernetes operator · cross-vendor P/D(Phase 5 전) · full switch-level congestion 모델.

---

# 9. 테스트·검증 요구사항

- 모든 신규 모듈에 pytest 단위 테스트. Phase 2 종료 시 최소 커버 대상: spec 검증, island detection, memory filter, TP 열거, feasibility 각 constraint, Pareto dominance, exhaustive vs greedy 일치성(작은 케이스).
- **Oracle 일치 테스트**: 소형 synthetic cluster(예: GPU 4개 + NPU 2개)에서 pruning을 끈 exhaustive 결과와 pruning을 켠 결과의 최적 plan이 동일함을 검증. pruning이 optimal을 제거하면 버그다.
- **재현성 테스트**: 동일 spec + seed로 두 번 실행 시 byte-identical plan 출력.
- **Golden output 테스트**: `examples/`의 spec 2개에 대한 PlannerOutput을 golden 파일로 고정하고 회귀 검증.
- 시뮬레이터 subprocess는 테스트에서 mock 가능하게 predictor를 인터페이스(`Predictor` ABC) 뒤에 둔다.
- CI 수준: `pytest` + `ruff` + `mypy`(planner/ 한정)를 통과해야 PR merge.

---

# 10. Git 운영 규칙

## 10.1 Branch/PR

PR 하나 = 연구 기능 하나. branch 이름:

```text
feat/service-spec          feat/cluster-inventory     feat/candidate-generator
feat/sim-predictor         feat/pareto-optimizer      feat/npu-profile-importer
feat/vllm-deployer         feat/topology-graph        feat/pd-placement
```

## 10.2 Milestones

```text
M0 Baseline        : upstream 재현 + vLLM validation + 형식 조사 문서
M1 Planner schema  : ServiceSpec, ClusterSpecV2, execution islands
M2 Static optimizer: candidate generation, feasibility, exhaustive oracle, Pareto
M3 Hetero profiles : GPU A/B, NPU importer, power/network profiles
M4 Real service    : launcher, monitor, calibration
M5 Topology/P-D    : topology graph, KV transfer, P/D planning
M6 Paper artifact  : scripts, configs, figure 재현, 문서화
```

---

# 11. 위험 요소 대응 규칙 (구현 중 판단 기준)

| 위험 | 대응 (구현에 반영할 것) |
|---|---|
| NPU가 model/kernel을 vLLM로 미지원 | profile의 `supported_models` compatibility matrix로 candidate 단계에서 제거. `(model, dtype, feature, backend) -> supported/unsupported` |
| Cross-GPU/NPU 통신 구현 난도 | 첫 논문은 island 간 replica placement만. cross-backend P/D는 Phase 5 이후 simulator-only로 먼저 평가 |
| Search space 폭발 | §5.4 pruning pipeline + Pareto DB + (후반) surrogate + top-K full sim |
| Simulator 오차로 SLO 위반 | `bench/` 지속 validation + §5.8 calibration + uncertainty margin + 보수적 admission |
| 논문 novelty 부족 | "heterogeneous GPU selection" 단독을 기여로 쓰지 않음. backend 이기종성 + explicit topology + phase-aware placement + power/tokens-J + calibrated sim-in-the-loop의 조합 유지 |
| 실물 hardware 부족 | 소규모 실측 validation + 시뮬레이터 대규모 sensitivity study. **미보유 hardware를 실측했다고 절대 표기하지 않음** |

---

# 12. 실험 계획 (Phase 2 이후 experiments/에 구현)

우선순위 순서대로 실험 스크립트를 작성한다. 각 실험은 one-command runner(`experiments/scripts/run_exp{N}.sh`)로 재현 가능해야 한다.

```text
Exp 1. 동일 GPU, TP=1/2/4 변화        -> planner pipeline 검증 (TTFT/TPOT/power)
Exp 2. GPU-A vs GPU-B vs mixed replicas -> heterogeneous resource-selection MVP
Exp 3. Network sensitivity 25/100/200/400G (+1/10G stress) -> topology model 검증
Exp 4. GPU vs NPU execution island     -> 동일 model/workload에서 SLO-goodput/J 비교
Exp 5. Heterogeneous P/D 4조합          -> 핵심 hypothesis (Phase 5)
        (미지원 backend 조합은 simulator-only 결과로 명확히 구분 표기)
```

Baseline (실험 결과 비교 대상으로 반드시 구현):

- Router: RR, RAND, LOAD (기존 코드 활용)
- Resource: fastest-only, most-efficient-only, least-device, topology-blind best-fit, simulator-blind heuristic
- Architecture: aggregated, homogeneous P/D, heterogeneous P/D
- Optimizer: exhaustive oracle vs greedy vs proposed

Ablation (Phase 5 이후): No-Topology / No-Energy / No-PD-Specialization / No-Uncertainty / No-Calibration / Static.

평가 지표는 §4 정의를 따르고, planner 자체 지표(planning latency, 평가 후보 수, prune ratio, oracle 대비 regret, prediction error)도 함께 기록한다.

---

# 13. 즉시 착수 목록 (첫 세션에서 이 순서대로)

```text
[ ] 1. upstream repo clone + commit pin (UPSTREAM_COMMIT 파일)
[ ] 2. 기존 example 시뮬레이션 1개 재현
[ ] 3. sim.csv / configs/cluster/*.json 실제 형식 조사 -> docs/phase0_formats.md
[ ] 4. bench/ baseline 재현 (또는 실행 계획 문서화)
[ ] 5. planner/spec.py        — ServiceSpec
[ ] 6. planner/inventory.py   — ClusterSpecV2 + island detector + compatibility matrix
[ ] 7. planner/topology.py    — graph + Level 1 model
[ ] 8. planner/candidate_generator.py — 열거 + memory/TP pruning
[ ] 9. planner/predictor/llmservingsim.py — config compiler + subprocess runner + parser
[ ] 10. planner/optimizer/exhaustive.py + pareto.py + feasibility.py
[ ] 11. examples/service_specs/qwen3-32b.yaml, examples/clusters/heterogeneous-lab.yaml
[ ] 12. python -m planner plan --service ... --cluster ... 동작 (§6의 출력 요건 충족)
[ ] 13. tokens/J·percentile 계산 유틸 + provenance metadata 기록
[ ] 14. tests/ 작성 및 통과 (oracle 일치, 재현성, golden output 포함)
```

첫 CLI 목표는 하나로 충분하다:

```bash
python -m planner plan \
  --service examples/service_specs/qwen3-32b.yaml \
  --cluster examples/clusters/heterogeneous-lab.yaml
```

출력: Feasible candidates / Rejected candidates + reason / Recommended plan / Pareto alternatives / Predicted metrics.

---

# 14. 참고 자료

## 기준 코드베이스

- Repository: https://github.com/casys-kaist/LLMServingSim
- Serving architecture: `serving/README.md`
- Cluster configuration: `configs/cluster/README.md`
- Benchmark/validation: `bench/README.md`
- Profiler: `profiler/README.md`

## vLLM / backend

- vLLM Disaggregated Prefilling: https://docs.vllm.ai/en/stable/features/disagg_prefill/
- vLLM Production Stack P/D: https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/disaggregated-prefill.html
- vLLM Ascend: https://docs.vllm.ai/projects/ascend/en/latest/

## 관련 연구 (novelty 비교 대상 — 논문 작성 시 상세 비교 필수)

Helix (2406.01566) · DistServe (2401.09670) · Splitwise (2311.18677) · Llumnix (2406.03243) · TaiChi (2508.01989) · ShuntServe (2606.18600) · Festina (2606.30391) · GreenLLM (2508.16449) · NeuScale (2607.16488) · Fast Heterogeneous Serving (2604.07472) · AccelGen (2503.13737)

---

## 마지막 지침

구현 중 이 문서와 실제 upstream 코드가 충돌하면(파일명, config 스키마, 출력 형식 등), **실제 코드가 우선**이다. 차이를 발견하면 `docs/deviations.md`에 기록하고 계속 진행한다. 범위(§8 제외 목록)를 벗어나는 작업이 필요해 보이면 임의로 진행하지 말고 사용자에게 보고한다.
