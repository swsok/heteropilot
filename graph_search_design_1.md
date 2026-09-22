# 그래프 기반 배치 탐색 — 소프트웨어 구조 설계서

> HeteroPilot 위에 「공유 통신 자원을 보존한 동등성 압축 + 증명 가능한 제거 + 적응형 Top-K」 탐색을 구현하기 위한 구조 설계.
> 구현 저장소: **`github.com/swsok/heteropilot-graphsearch`(신규, 이 연구 전용)** · 의존 저장소: `github.com/swsok/heteropilot`(서브모듈 `vendor/heteropilot`로 고정) · 코드 근거 커밋: `a27469a3b14c2be7def379ef5aab5a14eb0e0689` (2026-09-21, PR #130 머지) · 작성일: 2026-09-22 (rev 2: 두 저장소 구조로 개정)
> 상위 문서: `HeteroPilot_그래프기반_배치탐색_연구설계.md`(연구 설계, 이하 "연구설계서"). 이 문서는 그 연구설계서 §2~§11을 **코드 구조**로 옮긴 것이며, 구현 순서와 시험 절차는 `WORK_ORDER_graph_search.md`가 담당한다.
> 범위: **첫 논문 MVP.** 정규화 그래프, 템플릿·임베딩 생성, 정확 동등성 압축, 증명 가능한 제거, 서비스 여유도 랭커, 적응형 Top-K, 복원·예약 검사, 완전탐색 oracle, MVP 시뮬레이터 어댑터. flow 단위 공유 자원 경합 시뮬레이터는 **인터페이스만** 정의한다.

---

## 0. 한 문단 요약

현재 HeteroPilot은 `ExecutionIsland`(같은 노드·같은 backend·같은 모델의 가속기 묶음) 단위로 후보를 열거하고, 섬 안의 가속기를 서로 교환 가능한 것으로 취급한다. 이것은 이미 일종의 **암묵적 동등성 압축**이지만 (a) 섬 내부의 물리 장치 선택(어느 GPU 쌍인가)을 구분하지 않고, (b) 섬 경계 밖의 공유 자원(NIC uplink, PCIe root)을 후보의 속성으로 보지 않으며, (c) 압축이 정확한지 증명하지 않는다. 또 `EnvelopeCache`의 twin 병합은 "가속기 모델·역할·tp·pp·dp·네트워크 대역 클래스"만 같으면 결과를 공유하므로 근사 압축에 해당한다. 이 설계는 기존 파이프라인(`CandidateGenerator` → `SurrogateRanker` → `evaluate_candidates` → `feasibility` → `pareto`)을 **그대로 두고**, 그 앞뒤에 다섯 계층을 끼워 넣는다: ① `ClusterSpecV2`를 정규화 자원 그래프로 바꾸는 계층, ② 섬 단위 후보를 물리 임베딩으로 펼치는 계층, ③ 임베딩을 정확 동등성으로 다시 접는 계층, ④ 증명 가능한 하한만으로 제거하는 계층, ⑤ 대표를 K를 늘리며 평가하고 마지막에 물리 장치로 복원하는 계층. 다섯 계층은 전부 **새 저장소 `heteropilot-graphsearch`의 `graphsearch/` 패키지**에 두고, heteropilot에는 골든 출력을 바꾸지 않는 **세 개의 작은 훅 PR(H1~H3)**만 넣는다(§6).

---

## 1. 두 저장소 구조와 그 근거

### 1.1 왜 새 저장소 하나로는 안 되는가

- **물리 계산이 heteropilot 밖에서 재현되지 않는다.** `planner/util/memory.py`는 상류 `serving/core/memory_model.py`를 호출하고, 실제 예측은 `serving/`(LLMServingSim) + `astra-sim` 서브모듈이 필요하다. `planner/`만 복사하면 MockPredictor 시험까지만 돈다.
- **기존 파일 수정은 heteropilot의 golden 테스트가 지키는 자리다.** `ServiceSpec`의 처리량·비용 계약, `RejectionStage` 값, `evaluate_candidates`의 plan_id 부여, `EnvelopeCache` 키, predictor의 컴파일 경로는 heteropilot 안에서 바꾸고 "기본 경로 바이트 동일"을 거기서 증명해야 한다. 새 저장소의 몽키패치는 그 보장을 우회한다.

### 1.2 선례: ScenarioLab 분리

`WORK_ORDER_consolidation.md` STEP 3과 `docs/deviations.md` D24가 기록한 방식 그대로 따른다. 새 저장소는 heteropilot을 `vendor/heteropilot` 서브모듈로 특정 sha에 고정하고, `planner.*`를 import만 하며 역방향 의존은 두지 않는다. 경로는 `HETEROPILOT_ROOT`(환경변수, 기본 `vendor/heteropilot`) 하나로 모으고, 실행은 `PYTHONPATH=$PWD:$PWD/vendor/heteropilot`. heteropilot은 pip 설치 가능한 패키지가 아니므로(`pyproject.toml`에 `[project]` 테이블 없음) 이 방식이 유일하다.

### 1.3 경계 규칙

| 항목 | heteropilot (`vendor/heteropilot`) | heteropilot-graphsearch |
|---|---|---|
| 들어가는 코드 | 훅 PR H1(서비스 계약), H2(v2 인벤토리), H3(plan_id_base·`_assemble_output`·`RejectionStage` 3값·`EnvelopeCache.graph_signature`·`Predictor` 컴파일 훅) | `graphsearch/` 13개 모듈, 자체 CLI·렌더, toy fixture, 실험 스크립트, 이 문서와 작업지시서 |
| 의존 방향 | 없음(graphsearch를 모른다) | `planner.*` import |
| 테스트 | 기존 golden + 훅별 회귀 | `tests/`(MockPredictor·GraphAwareMockPredictor) + 서브모듈 sha에서 heteropilot `pytest` 통과 확인 |
| D-번호·실험 태그 | **D120–D129, `E-G*`** (훅 PR이 `docs/deviations.md`에 기록) | 자체 `docs/decisions.md`(GS-1, GS-2 …) |
| 시뮬레이터 실행 | `vendor/heteropilot/.venv` 재사용 또는 `scripts/compile.sh` | 없음(서브모듈 것을 호출) |

---

## 2. 현재 코드의 관련 구조 (사실 확인)

아래는 커밋 `a27469a3`에서 확인한 사실이다. 설계는 이 사실 위에 세운다. 경로는 heteropilot 기준.

| 위치 | 사실 | 설계에 미치는 영향 |
|---|---|---|
| `planner/inventory.py:379-411` `ClusterSpecV2` | 정점은 `Node.accelerators`와 `Node.nics`뿐. CPU socket·PCIe switch·네트워크 switch 정점 없음. `Link.src/dst`는 `<node>/<device_or_nic>` 형식 강제(`_endpoint_format`). | 정점 종류 확장은 `schema_version`으로 구분(H2, §4.1). |
| `planner/inventory.py:289-306` `Link` | `bandwidth_gbps`는 이름과 달리 **GB/s**(`topology.py:292` 주석). `contention_group: str \| None`이 공유 자원 식별자 역할. 무방향. `measurements`에 `(collective, world_size, msg_size_class, binding)` 키의 실효 대역폭. | 단위 어댑터는 v1을 재해석하지 않고 v2에서만 `bandwidth_unit`. `contention_group`은 v2의 `SharedResource.id`로 승격. |
| `planner/inventory.py:675-724` `detect_islands` | 노드 내부, 같은 `(backend, model)`, `INTRA_ISLAND_LINKS`로 연결된 컴포넌트. `max_tp_candidates`는 섬 크기의 약수. | 섬은 템플릿 생성의 시작 단위로 유지. 임베딩 계층이 섬 내부 부분집합을 펼친다. |
| `planner/candidate_generator.py:115-148` `generate` | 단일 섬(tp×dp×knobs), 두 섬 mixed(`tp_a == tp_b`, D14), 두 섬 P/D(`tp_d ∈ {tp_p, 2tp_p}`, D28). 단계 1~3 정확, 4~5 하한. 처리량 하한은 §5.6에 제약이 없어 **제거된 기록**(`:637-650`). | 후보 골격 재사용(`enable_bound_pruning=False`로 호출). 처리량 하한 복원은 H1이 `min_goodput_rps`를 넣은 뒤에만. |
| `planner/plan.py:99-135` `CandidateConfig` | `assignments: list[IslandAssignment]`, `knobs`, `topology_mode`. 물리 장치 id 없음. | 물리 임베딩은 이를 **감싸는** 새 타입 `EmbeddedCandidate`(§4.3). `CandidateConfig`는 수정하지 않음. |
| `planner/plan.py:138-190` `RejectionStage` | 정확/하한/휴리스틱/인식론적 거절이 별도 값. `SURROGATE_PRUNED`는 "최적을 놓칠 수 있음" 명문화. | 연구설계서 §6의 상태 5종을 기존 값에 매핑하고 부족한 셋만 H3에서 추가(§4.6). |
| `planner/spec.py:79-84` `Slo` / `:20-23` `Objective` | 금전 비용·처리량 하한·완료율 없음. | H1에서 확장. |
| `planner/optimizer/feasibility.py:121-155` `evaluate` | 지연·전력·효율만 검사. | H1에서 `check_throughput` 추가. |
| `planner/optimizer/surrogate.py:35-50` `SurrogateRanker` | `order(candidates, spec, islands, profiles, *, gpu_memory_utilization)`. 순서만 반환. `BinnedRooflineRanker` 기본. | 새 랭커는 이 ABC 구현(graphsearch). 기존 랭커는 baseline. |
| `planner/optimizer/exhaustive.py:333-584` `evaluate_candidates` | 인덱스 순서로 `_plan_id(index)`. 결과 조립 순차. | 배치 평가용 `plan_id_base` 인자(H3). |
| `planner/optimizer/exhaustive.py:922-1182` `search` | 생성→필터→surrogate top-K(한 번)→평가→랭킹→`PlannerOutput` 조립. | 조립 부분을 `_assemble_output()`로 추출(H3)해 graphsearch 드라이버가 재사용. `oracle()`은 그대로 정답 기준. |
| `planner/envelope.py:108-162` `EnvelopeKey` | 그래프 구조·공유 자원 미포함. `topology_level != 1`일 때만 키에 접어 넣는 선례(`:243-244`). | `graph_signature: str \| None` 옵션 추가(H3). 기존 캐시 파일 이름 불변. |
| `planner/topology.py:131-513` `TopologyGraph` | 무방향 BFS, Level-1/2 축약, `path_aware=False`, `contention_modeled=False` 명시. | 변경 없음. graphsearch의 `ResourceGraph`가 방향·공유자원·경계 문맥 담당. |
| `planner/predictor/llmservingsim.py:182-351` `compile_to_sim_config`, `:354-` `LLMServingSimPredictor` | 섬 단위 인스턴스 생성, `topology_level=2`면 두 차원. | predictor에 **컴파일 함수 교체 훅**만 추가(H3). 실제 임베딩 컴파일은 graphsearch(§4.9). |
| `planner/optimizer/greedy.py:51-110` `estimate` | 메모리 roofline proxy. | 랭커 특징에 재사용. |
| `pyproject.toml`, `topology.py:134` | networkx 미사용. | graphsearch가 networkx 의존(heteropilot은 무관). |

---

## 3. 목표와 비목표

**목표**
1. 후보 하나가 "물리 장치 집합 + 역할·shard + 통신 경로 + 공유 자원 사용량"을 모두 갖는 **실행 가능한 전체 계획**이 되게 한다(연구설계서 §2).
2. 정확 동등성으로 병합된 후보는 정의된 모형 아래에서 **같은 제약 판정과 같은 목적값**을 갖는다는 불변식을 테스트로 강제한다(§7.1).
3. 제거는 증명 가능한 필요조건 위반에만 쓰고, 나머지는 보류 상태로 남긴다. 어떤 후보도 "미평가"가 "불가능"으로 세어지지 않는다(§7.4).
4. 적응형 Top-K는 예산 모드와 인증 모드를 갖고, 종료 근거와 미평가 수를 감사 가능하게 출력한다.
5. heteropilot 기본 경로(`python -m planner plan`)의 출력은 훅 PR 뒤에도 **바이트 동일**. graphsearch는 자체 CLI(`python -m graphsearch plan`)로 동작.

**비목표(이번 MVP)** 임의 fabric의 packet/flow 경합 시뮬레이션(`ContentionModel` ABC만) · GPU–NPU 혼합 TP · KV 변환 P/D · 학습 랭커 · Kubernetes 배포 · 상류 `serving/` 수정.

---

## 4. 전체 구조

### 4.0 패키지 배치

```
heteropilot-graphsearch/                 # 신규 저장소
  vendor/heteropilot/                    # git submodule, sha 고정(H3 머지 후 커밋)
  graphsearch/
    __init__.py
    paths_root.py          # HETEROPILOT_ROOT 해석(기본 vendor/heteropilot), examples/profiles 경로 helper
    schema.py              # ResourceGraph, Vertex, DirectedEdge, SharedResource, 단위 어댑터 (v2 인벤토리 필드는 H2가 heteropilot에 추가)
    paths.py               # 허용 경로 집합, 컷 용량, 경계 문맥(BoundaryContext)
    demand.py              # CommFlow: 후보를 통신 수요로 변환
    embeddings.py          # PlacementTemplate -> EmbeddedCandidate 열거, canonical key
    equivalence.py         # 후보 그래프 직렬화, WL hash, VF2 정확 검사, Representative, 충돌 행렬
    bounds.py              # 증명 가능한 제거(BoundProof), 후보 상태(CandidateStatus)
    cost.py                # 시간당 비용 모델
    ranker.py              # ServiceMarginRanker(SurrogateRanker), 다양성 배분
    adaptive.py            # AdaptiveSearch 드라이버(예산/인증 모드), SearchAudit
    restore.py             # 대표 -> 물리 임베딩 복원, 용량 충돌·snapshot 재확인
    adapter.py             # compile_embedded, TopologyLossReport, apply_pd_transfer_cost_embedded, predictor 훅 바인딩
    contention.py          # ContentionModel ABC + NullContentionModel
    oracle.py              # 소규모 완전탐색 oracle 하네스
    render.py              # "Graph search:" 블록 렌더(heteropilot render 출력 뒤에 덧붙임)
    __main__.py            # python -m graphsearch plan|oracle|compare
  fixtures/
    clusters/graph-toy-abcde.yaml, graph-toy-abcde.v2.yaml, graph-toy-shared-nic.yaml, graph-toy-shared-nic.v2.yaml, graph-toy-asym.v2.yaml
    profiles/toy_gpu.yaml, toy_npu.yaml
    service_specs/graph-toy-llama31-8b.yaml
  tests/                   # conftest.py(ROOT=HETEROPILOT_ROOT), graph_fixtures.py, test_*.py
  experiments/scripts/, experiments/results/
  docs/graph_search_design.md (이 문서), docs/decisions.md (GS-n)
  WORK_ORDER_graph_search.md, CLAUDE.md, README.md, pyproject.toml, .gitmodules

heteropilot/ (훅 PR만)
  H1  planner/spec.py, planner/plan.py(PredictedMetrics.offered_requests, DeploymentPlan.cost_*), planner/optimizer/feasibility.py, planner/optimizer/pareto.py, planner/__main__.py(_write_output 키 드롭), planner/predictor/llmservingsim.py(_parse: offered_requests)
  H2  planner/inventory.py(schema_version, v2 정점·공유자원·단위·가격·runtime_capabilities), docs/docs/reference/cluster-config.md
  H3  planner/plan.py(RejectionStage 3값), planner/optimizer/exhaustive.py(plan_id_base, _assemble_output), planner/envelope.py(graph_signature), planner/predictor/llmservingsim.py(compile 훅), CLAUDE.md(D-블록·E-태그·문서 표), docs/deviations.md(D120~D124)
```

### 4.1 데이터 흐름

```
ClusterSpecV2 (v1|v2, planner.inventory)      ServiceSpec (H1 확장)
        │                                            │
        ▼                                            │
  [schema.py] ResourceGraph ◀── 단위 정규화, 공유자원 보조정점, 외부 예약 스냅샷
        │                                            │
        ▼                                            ▼
  detect_islands ──▶ CandidateGenerator(enable_bound_pruning=False).generate()   (heteropilot)
        │                    │  list[CandidateConfig]  ← 템플릿(섬 단위)
        │                    ▼
        │            [embeddings.py] ──▶ list[EmbeddedCandidate]
        │                    ▼
        │            [demand.py] CommFlow ──▶ [paths.py] 허용 경로·컷·경계 문맥
        │                    ▼
        │            [equivalence.py] WL → bucket → VF2 ──▶ list[Representative] + ConflictMatrix
        │                    ▼
        │            [bounds.py] 증명 가능한 제거 ──▶ CandidateStatus + BoundProof
        │                    ▼
        │            [ranker.py] ServiceMarginRanker.order() + 다양성 배분
        │                    ▼
        │            [adaptive.py] K 배치 ──▶ planner.optimizer.exhaustive.evaluate_candidates(plan_id_base=…)   (heteropilot, H3)
        │                    │            ▲               │
        │                    │            └── [adapter.py] compile_embedded (predictor 훅 경유) + TopologyLossReport
        │                    ▼
        │            feasibility.evaluate(+check_throughput, H1) → rank_plans(+cost, H1) → _assemble_output (H3)
        │                    ▼
        │            [restore.py] 대표 → 물리 임베딩 복원, 용량 충돌·snapshot 재확인
        │                    ▼
        └────────▶  PlannerOutput (+ provenance["graph_search"] = SearchAudit) ──▶ [render.py]
```

### 4.2 기존 파이프라인과의 관계

- heteropilot의 `search()`·`oracle()`은 그대로. `AdaptiveSearch.run()`은 `CandidateGenerator`, `evaluate_candidates`, `judge`, `rank_plans`, `_assemble_output`을 **호출**하는 graphsearch 쪽 드라이버다. 논문 실험은 (a) 전 임베딩 완전 평가(정답), (b) heteropilot `search(surrogate=BinnedRooflineRanker, top_k=K)`(baseline), (c) `AdaptiveSearch`(제안)를 비교한다.
- 압축이 임베딩당 1대표만 남기는 클러스터(모든 섬이 정확히 필요 장치 수)에서는 후보 집합이 heteropilot의 섬 단위 후보와 1:1이어야 한다(회귀, §7.5).

---

## 5. 모듈별 설계

각 모듈은 (역할 / 입력 / 출력 / 주요 자료구조 / 재사용 / 결정론)을 적는다. 자료구조는 pydantic `_Strict`(extra=forbid) 또는 `@dataclass(frozen=True)`. 코드 주석·docstring은 영어(heteropilot `AGENTS.md` 규칙을 새 저장소도 따른다).

### 5.1 `graphsearch/schema.py` — 정규화 자원 그래프

**역할** `ClusterSpecV2`를 방향 간선·bytes/s·공유 자원 보조 정점을 가진 속성 그래프로 정규화한다. 후보 비교와 하한 계산이 참조하는 **유일한 물리 표현**.

**입력** `ClusterSpecV2`(v1 또는 H2의 v2), `dict[str, AcceleratorProfile]`. **출력** `ResourceGraph`.

```python
class VertexKind(str, enum.Enum):
    ACCELERATOR = "accelerator"; CPU_SOCKET = "cpu_socket"; PCIE_SWITCH = "pcie_switch"
    NIC = "nic"; NET_SWITCH = "net_switch"; SHARED_RESOURCE = "shared_resource"

@dataclass(frozen=True)
class Vertex:
    id: str                     # "<node>/<device>" 또는 "res:<shared_resource_id>" 또는 "<net_switch_id>"
    kind: VertexKind
    node_id: str | None
    attrs: Mapping[str, Any]    # accelerator: model, backend, memory_bytes, runtime_capabilities, dtypes, price_per_hour_usd, active_power_w, state, profile_id
                                # shared_resource: capacity_bytes_per_s, reserved_bytes_per_s, res_kind

@dataclass(frozen=True)
class DirectedEdge:
    id: str                     # "<link_id>:fwd" / "<link_id>:rev"
    src: str; dst: str
    link_type: LinkType
    capacity_bytes_per_s: float; latency_ns: float
    rdma_capable: bool | None; p2p_capable: bool | None
    shared_resource_id: str | None
    source: Source
    measurements: tuple[LinkMeasurement, ...]

@dataclass(frozen=True)
class SharedResource:
    id: str; capacity_bytes_per_s: float; reserved_bytes_per_s: float; kind: str; node_id: str | None

@dataclass(frozen=True)
class ResourceGraph:
    schema_version: int
    vertices: Mapping[str, Vertex]; edges: Mapping[str, DirectedEdge]; shared_resources: Mapping[str, SharedResource]
    snapshot_version: str        # sha256(snapshot_id, reserved 값들, accelerator state들)
    unit_notes: tuple[str, ...]
    def out_edges(self, v: str) -> Sequence[DirectedEdge]
    def accelerators(self) -> Sequence[Vertex]
    def subgraph(self, vertex_ids: Collection[str]) -> "ResourceGraph"
    def digest(self) -> str
def build_resource_graph(cluster: ClusterSpecV2, profiles: dict[str, AcceleratorProfile]) -> ResourceGraph
def bytes_per_s(value: float, unit: str) -> float
```

**인벤토리 v2(H2가 heteropilot에 추가)** `ClusterSpecV2.schema_version: int = 1`; v2 전용 필드 `Node.cpu_sockets`, `Node.pcie_switches`, `ClusterSpecV2.net_switches`, `ClusterSpecV2.shared_resources`, `ClusterSpecV2.snapshot_id`, `Link.bandwidth_unit`, `Link.direction`, `Link.rdma`, `Link.p2p`, `Link.shared_resource`, `Accelerator`/`AcceleratorProfile.price_per_hour_usd`, `Node.host_price_per_hour_usd`, `AcceleratorProfile.runtime_capabilities`. v1 파일에 v2 필드가 있으면 `InventoryError`. **v1 `bandwidth_gbps`는 GB/s로 계속 읽는다.**

**단위** 내부는 `bytes/s`·`ns`. `Gbit/s` → `×1e9/8`. **공유 자원** 같은 uplink를 쓰는 간선은 같은 `shared_resource_id`, `res:<id>` 보조 정점(incidence). v1 `contention_group`은 용량을 알 수 없으므로 멤버 간선 최소 용량을 placeholder로 두고 `unit_notes`에 기록. **결정론** 정렬 키 순서, `digest()`는 `json.dumps(sort_keys=True)`.

### 5.2 `graphsearch/paths.py` — 경로·컷·경계 문맥

**역할** 허용 경로 집합과 컷의 낙관적 총 용량(max-flow). 하한이 "한 경로의 느린 링크"가 아니라 컷 전체 용량을 쓰게 하는 근거.

```python
@dataclass(frozen=True)
class PathPolicy: max_hops: int = 8; require_rdma: bool = False; allowed_types: frozenset[LinkType] | None = None
@dataclass(frozen=True)
class Path: edges: tuple[str, ...]; bottleneck_bytes_per_s: float; latency_ns: float; shared_resources: frozenset[str]
@dataclass(frozen=True)
class PathSet: src: str; dst: str; paths: tuple[Path, ...]          # 정렬 (hops, -bottleneck, edge ids)
@dataclass(frozen=True)
class CutCapacity: src_set: frozenset[str]; dst_set: frozenset[str]; bytes_per_s: float; saturating_resources: frozenset[str]; assumptions: tuple[str, ...]
@dataclass(frozen=True)
class BoundaryContext: transit_vertices: frozenset[str]; shared_resources: frozenset[str]; reserved_bytes_per_s: Mapping[str, float]
def path_set(graph, src, dst, policy=PathPolicy()) -> PathSet
def cut_capacity(graph, src_set, dst_set, policy=PathPolicy()) -> CutCapacity
def boundary_context(graph, devices, paths: Iterable[PathSet]) -> BoundaryContext
def to_networkx(graph, policy) -> "nx.DiGraph"     # 공유 자원은 res_in→res_out 간선(용량 capacity-reserved)으로 분할
```

### 5.3 `graphsearch/embeddings.py` — 템플릿과 물리 임베딩

```python
@dataclass(frozen=True)
class RankPlacement: assignment_index: int; replica: int; ranks: tuple[str, ...]
@dataclass(frozen=True)
class EmbeddedCandidate:
    template: CandidateConfig; embedding_key: str; placements: tuple[RankPlacement, ...]
    devices: frozenset[str]; transit_closure: frozenset[str]
    flows: tuple["CommFlow", ...]; boundary: "BoundaryContext"; resource_demand: Mapping[str, float]
    @property
    def id(self) -> str: return f"{self.template.id}@{self.embedding_key[:12]}"
class EmbeddingPolicy(_Strict):
    max_embeddings_per_template: int | None = None; hierarchical: bool = True; canonical_only: bool = True
@dataclass
class EmbeddingStats: templates: int; embeddings: int; skipped_symmetric: int; truncated_by_policy: int; truncated_template_ids: list[str]
def enumerate_embeddings(templates, islands, graph, spec, policy=EmbeddingPolicy(), *, path_policy=PathPolicy()) -> tuple[list[EmbeddedCandidate], EmbeddingStats]
```
**열거 규칙** assignment별로 섬의 FREE 장치에서 `total_devices`개 조합 → `dp_replicas` 그룹. TP 그룹 내부·복제본 사이는 순서 무관(정렬). 같은 섬을 참조하는 두 assignment는 장치 겹침 금지. `hierarchical`이면 같은 pcie_switch/cpu_socket 조합을 먼저. 잘린 수는 `EXCLUDED_BY_SCOPE`로 보고(연구설계서 §4 "완전성의 범위를 정직하게").

### 5.4 `graphsearch/demand.py` — 통신 수요

```python
class FlowKind(str, enum.Enum): TP_ALLREDUCE="tp_allreduce"; PP_ACTIVATION="pp_activation"; PD_KV_TRANSFER="pd_kv_transfer"; INGRESS="ingress"; EGRESS="egress"
@dataclass(frozen=True)
class CommFlow:
    flow_id: str; kind: FlowKind; participants: tuple[str, ...]
    bytes_per_event: float; events_per_request: float
    on_critical_path: Literal["ttft","tpot","both","none"]
    allowed_paths: tuple[PathSet, ...]; assumptions: tuple[str, ...]
def tp_allreduce_bytes(model, dtype, tp) -> float          # hidden*bytes_per_elem*2(tp-1)/tp — candidate_generator._stage4_topology_ok(:550-558)와 같은 식을 독립 구현
def tp_allreduces_per_output_token(model) -> int           # 2*num_hidden_layers
def pd_kv_bytes(model, dtype, kv_cache_dtype, prompt_tokens) -> float   # planner.util.kv_transfer._kv_bytes_per_token 재사용
def flows_for(template, placements, spec, graph, policy=PathPolicy()) -> tuple[CommFlow, ...]
```
**일치 보장** heteropilot 코드를 리팩터하지 않는 대신, 테스트가 `tp_allreduce_bytes`로 계산한 stage-4 floor가 `CandidateGenerator`의 `TOPOLOGY_INFEASIBLE` 거절 사유 문자열의 수치와 일치함을 확인한다(두 구현이 갈라지면 테스트가 잡는다).

### 5.5 `graphsearch/equivalence.py` — 정확 동등성 압축

```python
class EquivalenceLevel(str, enum.Enum): EXACT="exact"; HASH_ONLY="hash_only"; APPROX="approx"
@dataclass(frozen=True)
class Signature: wl_hash: str; attr_histogram: str; tool_version: str
@dataclass
class Representative:
    rep_id: str; signature: Signature; level: EquivalenceLevel; template_id: str
    exemplar: EmbeddedCandidate; embeddings: list[EmbeddedCandidate]
    role_mappings: dict[str, dict[str, str]]; resource_usage: dict[str, float]
    @property
    def multiplicity(self) -> int
@dataclass(frozen=True)
class ConflictMatrix:
    conflicts: frozenset[frozenset[str]]
    def max_concurrent(self, rep: Representative) -> int   # 그리디 하한(정확값 아님을 문서화)
class CompressionPolicy(_Strict): enabled: bool=True; wl_iterations: int=3; max_vf2_seconds: float|None=None; include_boundary: bool=True; include_prices: bool=True
class CompressionReport(_Strict): embeddings_in: int; representatives_out: int; exact_merges: int; hash_only_groups: int; vf2_calls: int; vf2_seconds: float
def candidate_graph(emb, graph, policy) -> "nx.DiGraph"; def signature(emb, graph, policy) -> Signature
def compress(embs, graph, policy=CompressionPolicy()) -> tuple[list[Representative], ConflictMatrix, CompressionReport]
```
**후보 그래프** `devices ∪ transit_closure ∪ boundary.transit_vertices ∪ res:*`로 유도된 부분그래프에 역할·tp·pp·slot·가격·전력·프로파일 id를 정점 라벨로, 링크 속성·측정 키를 간선 라벨로. 정점 id·노드 id는 라벨에서 제거. **알고리즘** WL 해시 bucket → 같은 bucket 안에서 VF2(`DiGraphMatcher`, categorical match) 확인 후에만 병합 → 예산 초과분은 `HASH_ONLY`(병합 금지). `include_boundary=False`는 오병합 ablation 전용.

### 5.6 `graphsearch/bounds.py` — 증명 가능한 제거

```python
class CandidateStatus(str, enum.Enum): IMPOSSIBLE_PROVEN; EXCLUDED_BY_SCOPE; DEFERRED_HEURISTIC; UNKNOWN_MEASUREMENT; EVALUATED
@dataclass(frozen=True)
class BoundProof: check: Literal["compat","memory","comm_latency","throughput_capacity","cost_lower_bound"]; bound_value: float; threshold: float; unit: str; relaxations: tuple[str, ...]; inputs: Mapping[str, float]; safe: bool
@dataclass
class BoundVerdict: status: CandidateStatus; stage: RejectionStage | None; proofs: list[BoundProof]
class BoundPolicy(_Strict): compat=True; memory=True; comm_latency=True; throughput_capacity=True; cost_lower_bound=False; demoted: frozenset[str]=frozenset()
def prune(reps, spec, graph, profiles, stats, *, policy=BoundPolicy(), incumbent_cost=None) -> tuple[dict[str, BoundVerdict], list[Rejection]]
```
상태→`RejectionStage` 매핑: IMPOSSIBLE_PROVEN → `BACKEND_INCOMPATIBLE`/`MEMORY_INFEASIBLE`/`TOPOLOGY_INFEASIBLE`/`ANALYTICAL_LOWER_BOUND`/`THROUGHPUT_UPPER_BOUND`(H3 신규); EXCLUDED_BY_SCOPE → `EXCLUDED_BY_SCOPE`(H3 신규); DEFERRED_HEURISTIC → `SURROGATE_PRUNED`(기존) 또는 `NOT_EVALUATED_BUDGET`(H3 신규); UNKNOWN_MEASUREMENT → `OUTSIDE_CALIBRATION_DOMAIN`/`CALIBRATION_CONDITION_MISMATCH`/`SIM_ERROR`(기존). 검사 5종의 정확한 식은 작업지시서 G8. `comm_latency`는 컷 용량 기반이며 그래프 모드에서 생성기 stage 4·5를 대신한다. 측정 실효 대역폭은 하한에 쓰지 않는다.

### 5.7 `graphsearch/ranker.py` — 서비스 여유도 랭커

`planner.optimizer.surrogate.SurrogateRanker`를 구현. `RankFeatures(ttft_ratio, tpot_ratio, goodput_ratio, cost_per_hour, shared_nic_util, cut_margin, memory_margin, outside_calibration, structure_key)`, `risk_proxy = max(세 비율)`. 1군(`risk_proxy ≤ 1`)은 비용 오름차순, 2군은 `risk_proxy` 오름차순, `cost None`은 1군 뒤·2군 앞. `DiversityQuota(by, reserved_fraction, round_robin_when_short)`로 `k_hint` 안에서 구조별 자리 보장, 예산 초과 금지. 순서만 반환(ABC 계약).

### 5.8 `graphsearch/adaptive.py` — 적응형 Top-K 드라이버

```python
class SearchMode(str, enum.Enum): BUDGET="budget"; CERTIFY="certify"
class AdaptiveConfig(_Strict): k_schedule: tuple[int, ...]=(4,8,16); mode=SearchMode.BUDGET; max_simulations: int|None=None; max_wall_seconds: float|None=None; epsilon: float=0.0; split_approx_groups_on_residual: float=0.25
@dataclass
class SearchAudit:
    generated_templates: int; embeddings: int; representatives: int
    impossible_proven: int; excluded_by_scope: int; deferred_heuristic: int; unknown_measurement: int; evaluated: int
    simulations_run: int; cache_hits: int; k_reached: int
    termination: Literal["k_exhausted","budget_sims","budget_wall","certified","all_evaluated"]
    certificate: dict | None; unevaluated_ids: list[str]
    def as_provenance(self) -> dict
class AdaptiveSearch:
    def __init__(self, spec, cluster, islands, profiles, predictor, *, graph, reps, verdicts, ranker, config, cache=None, ...)
    def run(self) -> tuple[PlannerOutput, SearchAudit]
```
루프는 heteropilot의 `evaluate_candidates(batch, ..., plan_id_base=누적)`(H3)를 배치마다 호출하고, 최종 `PlannerOutput`은 `exhaustive._assemble_output(...)`(H3)로 조립한다. 인증 모드는 `bounds.prune(cost_lower_bound=True, incumbent_cost=…)`를 재실행해 `min LB ≥ incumbent×(1-ε)`일 때만 종료.

### 5.9 `graphsearch/adapter.py` — MVP 시뮬레이터 어댑터

```python
class TopologyLossReport(_Strict):
    model_level: int = 2; path_aware: bool = False; contention_modeled: bool = False
    dropped_shared_resources: list[str]; flows_priced_analytically: list[str]
    per_dim_bw_bytes_per_s: list[float]; per_dim_latency_ns: list[float]; basis: str
def compile_embedded(emb, cluster, islands, profiles, *, graph, topology, gpu_memory_utilization=0.90, activation_reserve_gb=0.0) -> tuple[dict, TopologyLossReport]
def apply_pd_transfer_cost_embedded(emb, metrics, spec, graph) -> tuple[PredictedMetrics, dict]
def bind(predictor: LLMServingSimPredictor, embeddings_by_template_id: dict[str, EmbeddedCandidate], graph) -> None
    # predictor.set_compile_hook(fn) (H3) 에 compile_embedded를 감싼 클로저를 등록
```
`compile_embedded`는 heteropilot `compile_to_sim_config(topology_level=2)`를 호출한 뒤 `link_bw`/`link_latency`를 flow 경로 병목으로 덮어쓰고, 시뮬레이터가 표현하지 못한 공유 자원 전부를 `dropped_shared_resources`에 적는다. 공유 uplink 유무만 다른 두 대표가 같은 sim 입력이 되면 이 목록이 다르고 caveat가 붙는다(연구설계서 §8). **H3의 predictor 훅** `LLMServingSimPredictor.set_compile_hook(fn: Callable[[CandidateConfig, ClusterSpecV2, dict, dict], tuple[dict, TopologyReduction] | None] | None)`: 훅이 `None`을 반환하면 기존 경로. heteropilot은 `EmbeddedCandidate`를 알 필요가 없다.

**캐시 키** `EnvelopeCache(graph_signature=…)`(H3). 값은 `f"{rep.signature.wl_hash}:{graph.schema_version}:{rep.signature.tool_version}"`. 대표별로 다르므로 `EnvelopeCache.with_graph_signature(sig)`(H3, 같은 root·카운터 공유) 사용.

### 5.10 `graphsearch/restore.py` — 복원과 예약 검사

```python
@dataclass(frozen=True)
class RestoredPlan: plan: DeploymentPlan; embedding: EmbeddedCandidate; device_ids: tuple[str, ...]; shared_resource_reservations: Mapping[str, float]; snapshot_version: str
class RestoreError(ValueError): ...
def restore(rep, plan, graph, *, occupied=frozenset(), prefer=None) -> RestoredPlan
def restore_many(reps_plans, graph, conflicts, *, count=1) -> list[RestoredPlan]
def check_capacity(restored, graph) -> list[str]; def recheck_snapshot(restored, current) -> list[str]
```
`multiplicity ≠ max_concurrent`. 대표 하나의 성능을 복수 배치에 곱하지 않는다.

### 5.11 `graphsearch/cost.py`, 5.12 `graphsearch/contention.py`, 5.13 `graphsearch/oracle.py`

`CostBreakdown(accelerator, host, total|None, missing, basis)`, `cost_of_devices()`, `cost_lower_bound()`; 가격 결측은 None(활성 장치 수로 대체 금지). `ContentionModel` ABC + `NullContentionModel`(latency + bytes/bottleneck). `run_oracle()`(전 임베딩, 압축·제거·top-K 없음), `run_proposed()`, `compare() -> OracleComparison(feasible_recall, cost_regret, false_infeasible, mismerged_pairs)`.

### 5.14 `graphsearch/__main__.py`, `render.py`

`python -m graphsearch plan --service … --cluster … [--k-schedule 4,8,16 --search-mode budget|certify --budget-sims N --budget-seconds S --epsilon E --max-embeddings-per-template N --compression exact|off --ranker service_margin|binned|roofline --bounds all|none|<list> --oracle --cache-dir … --output …]`. heteropilot의 `planner/__main__.py`는 **수정하지 않는다**; graphsearch CLI가 `planner.render`의 텍스트 뒤에 "Graph search:" 블록을 덧붙이고 YAML에 `provenance.graph_search`를 넣는다. 서브커맨드 `oracle`, `compare`는 §5.13 하네스.

---

## 6. heteropilot 훅 PR 명세 (H1~H3)

모든 훅 PR은 golden 불변(`tests/test_search.py`, `tests/test_render.py`)을 통과해야 하며, 새 필드의 기본값은 기존 출력을 바꾸지 않는다(`_write_output`이 None 키를 떨어뜨리는 A4 관례).

### H1 — 서비스 계약 확장
- `spec.py`: `Objective.MINIMIZE_COST_PER_HOUR`; `Slo.min_goodput_rps`, `Slo.min_completion_ratio`, `Slo.observation_window_s`.
- `plan.py`: `PredictedMetrics.offered_requests: int | None`; `DeploymentPlan.cost_per_hour_usd: float | None`, `cost_basis: str | None`.
- `feasibility.py`: `check_throughput()`; `evaluate()`에서 지연 다음, 전력 앞에 호출. 위반은 `SLO_VIOLATED`.
- `pareto.py`: 비용 목적의 `can_score`/`objective_value`; `_DIMENSIONS`에 비용.
- `llmservingsim.py::_parse`: `offered_requests` 채움. `__main__.py::_write_output`: None 키 드롭.
- **D121** `offered_requests` 추가로 `_metrics_schema_digest`가 바뀌어 기존 캐시 전체 miss(의도).

### H2 — v2 인벤토리 스키마
§5.1의 v2 필드와 검증 규칙. `docs/docs/reference/cluster-config.md`에 v2 절. **D120** "v1 `bandwidth_gbps`는 GB/s 유지; v2에서만 단위·정점 확장".

### H3 — 탐색 훅
- `plan.py`: `RejectionStage.THROUGHPUT_UPPER_BOUND`, `EXCLUDED_BY_SCOPE`, `NOT_EVALUATED_BUDGET`.
- `exhaustive.py`: `evaluate_candidates(..., plan_id_base: int = 0)`; `search()` 후반부를 `_assemble_output(spec, cluster, generation_counts, evaluation, all_rejections, caveats, prov, island_tiers, island_hw) -> PlannerOutput`로 추출(동작 불변).
- `envelope.py`: `EnvelopeCache(..., graph_signature: str | None = None)`, `with_graph_signature(sig)`.
- `llmservingsim.py`: `LLMServingSimPredictor.set_compile_hook(fn | None)`; `predict()`는 훅이 있으면 먼저 호출, `None` 반환 시 기존 `compile_to_sim_config`.
- `CLAUDE.md`: D-블록 `D120–D129 | WORK_ORDER_graph_search.md (heteropilot-graphsearch)`, 실험 태그 `E-G*`, 문서 표에 "graph search는 `swsok/heteropilot-graphsearch`로 분리, heteropilot `<sha>`에 pin" 한 줄. **D122**(그래프 모드에서 stage 4·5 대신 컷 하한 — graphsearch 쪽 결정이지만 heteropilot 생성기의 `enable_bound_pruning=False` 사용을 기록), **D123**(networkx는 graphsearch 의존; heteropilot 무관), **D124**(MVP 어댑터의 공유 자원 손실은 D3의 연장).

---

## 7. 불변식과 검증 전략

| # | 불변식 | 테스트(graphsearch `tests/`) |
|---|---|---|
| 7.1 압축 보존 | 같은 대표의 두 임베딩은 같은 판정·같은 목적값. 노드 재라벨은 서명 보존; 공유 NIC 추가·역할 교환은 서명 변경. | `test_equivalence.py::test_node_relabel_preserves_signature`, `::test_shared_uplink_changes_signature`, `::test_role_swap_changes_signature`, `test_oracle_agreement.py::test_merged_embeddings_agree_under_oracle` |
| 7.2 제거 보존 | 각 안전 필터 on/off의 완전 탐색에서 feasible 최선 동일. 모든 rejection에 `BoundProof`. | `test_bounds.py::test_each_bound_is_a_relaxation`, `::test_every_rejection_carries_a_proof` |
| 7.3 mapping 보존 | 복원 사용량 ≤ 가용량. `multiplicity ≠ max_concurrent`. | `test_restore.py::test_capacity_never_exceeded`, `::test_multiplicity_is_not_concurrency` |
| 7.4 결론 보존 | `SIM_ERROR`, `OUTSIDE_CALIBRATION_DOMAIN`, `SURROGATE_PRUNED`, `NOT_EVALUATED_BUDGET`, `EXCLUDED_BY_SCOPE`는 infeasible 수에 합산되지 않음. | `test_adaptive.py::test_unevaluated_is_not_infeasible` |
| 7.5 기본 경로 불변 | heteropilot `plan` golden 바이트 동일(훅 PR마다); 임베딩당 1대표면 후보 집합 1:1. | heteropilot `tests/test_search.py`, `tests/test_render.py`; `test_embeddings.py::test_single_embedding_reduces_to_island_candidates` |
| 7.6 재현성 | 같은 입력 두 번 → `PlannerOutput` 바이트 동일. | `test_adaptive.py::test_reproducible` |
| 7.7 단위 | v1 64 GB/s → 64e9; v2 100 Gbit/s → 12.5e9; v1에 v2 필드 → 오류. | `test_schema.py::test_units` (H2 쪽 `tests/test_inventory.py`도) |
| 7.8 손실 보고 | 공유 uplink 유무만 다른 두 대표: sim 설정이 같아도 `dropped_shared_resources`가 다름. | `test_adapter.py::test_loss_report_names_dropped_resources` |
| 7.9 수요 일치 | `demand.tp_allreduce_bytes`로 계산한 floor == heteropilot stage-4 거절 사유의 수치. | `test_demand.py::test_matches_heteropilot_stage4` |

**MockPredictor** graphsearch `tests/graph_fixtures.py`의 `GraphAwareMockPredictor`는 heteropilot `tests/conftest.MockPredictor`를 상속(import 경로 `vendor/heteropilot/tests`)하고, `EmbeddedCandidate.flows`의 경로 병목과 공유 자원 예약을 읽어 TTFT/TPOT를 계산한다. 하한보다 빠른 예측을 하지 않는다(`MOCK_ROOFLINE_SLACK ≥ 1`).

---

## 8. 결정 사항과 기록

1. **두 저장소.** 구현은 `heteropilot-graphsearch`, heteropilot에는 H1~H3 훅만. ScenarioLab 선례(`vendor/heteropilot`, `HETEROPILOT_ROOT`, `PYTHONPATH`) 준수. graphsearch `docs/decisions.md` **GS-1**.
2. **D-블록 D120–D129, 실험 태그 `E-G*`**는 heteropilot에서 선점(훅 PR이 deviations를 쓰기 때문). graphsearch 자체 결정은 GS-n.
3. **networkx 도입**(graphsearch만). 버전은 `Signature.tool_version`에 기록. GS-2.
4. **비용 목적 우선**, 전력 최소화는 후속.
5. **그래프 모드에서 heteropilot stage 4·5와 `bounds.py`를 중복 실행하지 않는다.** `CandidateGenerator(enable_bound_pruning=False)`.
6. **heteropilot 코드 리팩터 최소화.** `_stage4_topology_ok`는 건드리지 않고 테스트로 일치를 확인(§7.9). predictor는 컴파일 훅만 받는다.
7. **대표는 exemplar 하나로 시뮬레이션**, 최종 추천 전 `restore` + `recheck_snapshot` 필수.

---

## 9. 위험과 완화

| 위험 | 완화 |
|---|---|
| 두 저장소 커밋 동기화 | 훅 PR H1~H3을 **먼저** 머지하고 그 sha에 서브모듈 고정. graphsearch CI가 `vendor/heteropilot`에서 heteropilot `pytest`도 돌린다. |
| heteropilot이 private이면 서브모듈 clone에 인증 필요 | `.gitmodules`는 HTTPS URL; CI·에이전트에는 읽기 토큰 또는 SSH 키. README에 명시. |
| 임베딩 폭발 | `EmbeddingPolicy.max_embeddings_per_template` + hierarchical canonical; 잘린 양은 `EXCLUDED_BY_SCOPE`로 보고. |
| VF2 비용 > sim 절감 | `max_vf2_seconds`, `HASH_ONLY` 보류, `CompressionReport.vf2_seconds`를 실험 표에. |
| 비대칭 클러스터 압축률 0 | 실패 조건으로 실험 포함(`graph-toy-asym.v2.yaml`). |
| 어댑터가 공유 자원 차이 미표현 | `TopologyLossReport`로 명시, 경합 재현 실험은 `ContentionModel` 이후. |
| `PredictedMetrics` 필드 추가로 캐시 전체 miss | D121로 기록, 실험 전 캐시 재생성. |
