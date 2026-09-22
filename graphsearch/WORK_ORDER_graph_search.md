# 작업지시서 — 그래프 기반 배치 탐색 (heteropilot-graphsearch)

> 연구설계서 `HeteroPilot_그래프기반_배치탐색_연구설계.md`와 설계서 `docs/graph_search_design.md`(이하 "설계서")를 코드로 옮기는 지시서. 정규화 자원 그래프 → 임베딩 열거 → 정확 동등성 압축 → 증명 가능한 제거 → 서비스 여유도 랭킹 → 적응형 Top-K → 복원, 그리고 이 전부를 검증하는 완전탐색 oracle까지가 범위다(첫 논문 MVP).
> **두 저장소.** 구현 저장소 `github.com/swsok/heteropilot-graphsearch`(신규, 이 연구 전용)와 의존 저장소 `github.com/swsok/heteropilot`(서브모듈 `vendor/heteropilot`로 sha 고정). heteropilot에는 골든 출력을 바꾸지 않는 훅 PR **H1~H3**만 들어가고, 나머지 **G0~G13**은 전부 graphsearch에 들어간다.
> 작성일: 2026-09-22 (rev 2: 두 저장소 구조) · heteropilot 기준 `main` = `a27469a3` (PR #130 머지 후)
> 도구: Claude Code. **전 STEP CPU 노드**(MockPredictor). 실측은 이 지시서 밖.
> 선점: heteropilot 편차 번호 **D120–D129**, 실험 태그 **`E-G*`**(H3에서 `CLAUDE.md` 표에 등록). graphsearch 자체 결정은 `docs/decisions.md`의 **GS-n**.
> 독자: 코딩 에이전트. 각 STEP은 그 STEP만 읽고 착수할 수 있게 저장소·입력·출력·자료구조·테스트·시험 절차·완료 조건을 자족적으로 적었다.

---

## 0. 사용법과 규칙

1. **순서** H1 → H2 → H3 (heteropilot, 순서대로 머지) → 서브모듈 sha 고정 → G0 → G1 → … → G13. 병행 가능: H1‖H2; G2‖G3(둘 다 G1 뒤); G7‖G8(둘 다 G6 뒤). G0는 H3 머지 전에 시작할 수 있으나(브랜치 sha에 임시 고정) H3 머지 후 sha를 갱신하는 커밋을 반드시 남긴다.
2. **경로 표기** 아무 접두가 없는 경로는 **graphsearch 저장소** 기준(`graphsearch/schema.py`, `tests/test_schema.py`, `fixtures/…`). `hp:` 접두는 **heteropilot** 기준(`hp:planner/spec.py`). graphsearch 안에서 heteropilot 파일을 가리킬 때는 `vendor/heteropilot/…`.
3. **한 STEP = 한 브랜치 = 한 PR.** heteropilot 브랜치 `feat/gs-h<N>-<이름>`, graphsearch 브랜치 `feat/g<N>-<이름>`. heteropilot에서는 머지 전 `git rev-list --count origin/main..<branch>`가 0인지 확인(CLAUDE.md 규칙 7).
4. **품질 게이트** heteropilot PR: `pytest && ruff check . && mypy planner/` + golden(`tests/test_search.py`, `tests/test_render.py`) 바이트 불변. graphsearch PR: `pytest -q && ruff check . && mypy graphsearch/` + `(cd vendor/heteropilot && pytest -q)`가 고정 sha에서 통과(서브모듈이 손상되지 않았다는 확인; G0에서 CI 잡으로 둔다).
5. **절대 규칙(heteropilot CLAUDE.md) 재확인** 상류 `serving/` 수정 금지 · 하드웨어 수치 발명 금지(`source: placeholder`) · `exhaustive.py` 삭제 금지 · 코드 주석·docstring·로그 영어 · graphsearch → planner 단방향 import(역방향 금지, ScenarioLab 선례).
6. **하한은 feasibility의 완화.** 어떤 필터도 `feasibility.evaluate`가 선언하지 않은 제약으로 거절하지 않는다. 처리량 하한은 **H1이 `min_goodput_rps`를 spec에 넣은 뒤에만** 켠다.
7. **미평가 ≠ 불가능.** 다섯 상태(`impossible_proven / excluded_by_scope / deferred_heuristic / unknown_measurement / evaluated`)를 섞지 않는다(G9 테스트가 강제).
8. **결정론.** 집합은 정렬 순회, 해시는 `json.dumps(sort_keys=True)`, networkx 그래프는 정렬된 순서로 정점 추가. 같은 입력 두 번 → 바이트 동일.
9. **기록.** heteropilot 쪽 결정은 `hp:docs/deviations.md` D120~D129(`tests/test_deviations_numbering.py`가 중복·미참조를 잡는다). graphsearch 쪽 결정은 `docs/decisions.md` GS-n(형식: 번호·날짜·결정·근거·영향 5줄).
10. **결과가 가설과 다르면 그대로 기록한다.** 압축률이 낮거나 VF2가 느리면 그 수치가 논문 수치다.

---

## 1. STEP 요약과 의존

| STEP | 저장소 | 이름 | 주요 산출물 | 의존 | 예상 |
|---|---|---|---|---|---|
| H1 | heteropilot | 서비스 계약 확장 | `spec.py`, `plan.py`, `feasibility.check_throughput`, `pareto` 비용, D121 | — | 1일 |
| H2 | heteropilot | v2 인벤토리 스키마 | `inventory.py`, cluster-config 문서, D120 | — | 1일 |
| H3 | heteropilot | 탐색 훅 | `RejectionStage` 3값, `plan_id_base`, `_assemble_output`, `EnvelopeCache.graph_signature`, predictor 컴파일 훅, `CLAUDE.md` 표, D122~D124 | H1, H2 | 1일 |
| G0 | graphsearch | 저장소 부트스트랩·fixture·mock | 서브모듈, `pyproject`, `CLAUDE.md`, `fixtures/`, `tests/graph_fixtures.py`, CI | H3 | 1일 |
| G1 | graphsearch | 자원 그래프 스키마 | `graphsearch/schema.py` | G0 | 1일 |
| G2 | graphsearch | 경로·컷·경계 문맥 | `graphsearch/paths.py` | G1 | 1일 |
| G3 | graphsearch | 통신 수요 | `graphsearch/demand.py` | G1 | 1일 |
| G4 | graphsearch | 비용 모델 | `graphsearch/cost.py` | G1 | 0.5일 |
| G5 | graphsearch | 임베딩 열거 | `graphsearch/embeddings.py` | G2, G3 | 1.5일 |
| G6 | graphsearch | 정확 동등성 압축 | `graphsearch/equivalence.py` | G5 | 2일 |
| G7 | graphsearch | 증명 가능한 제거 | `graphsearch/bounds.py` | G4, G5 | 1.5일 |
| G8 | graphsearch | 서비스 여유도 랭커 | `graphsearch/ranker.py` | G4, G5 | 1일 |
| G9 | graphsearch | 적응형 Top-K 드라이버 | `graphsearch/adaptive.py` | G6, G7, G8 | 2일 |
| G10 | graphsearch | 복원·예약 검사 | `graphsearch/restore.py` | G6 | 1일 |
| G11 | graphsearch | MVP 어댑터·캐시 키·손실 보고·경합 인터페이스 | `graphsearch/adapter.py`, `contention.py` | G5, G6 | 1.5일 |
| G12 | graphsearch | 완전탐색 oracle 하네스·통합 불변식 | `graphsearch/oracle.py`, `tests/test_oracle_agreement.py` | G9, G10, G11 | 1.5일 |
| G13 | graphsearch | CLI·렌더·문서·E-G1 파일럿 | `graphsearch/__main__.py`, `render.py`, `docs/`, `experiments/` | G12 | 1.5일 |

합계 약 21 작업일(CPU).

---

## 2. 공통 자료구조 색인

| 타입 | 파일 | STEP |
|---|---|---|
| `Objective.MINIMIZE_COST_PER_HOUR`, `Slo.min_goodput_rps/min_completion_ratio/observation_window_s`, `PredictedMetrics.offered_requests`, `DeploymentPlan.cost_per_hour_usd/cost_basis`, `feasibility.check_throughput` | `hp:planner/spec.py`, `hp:planner/plan.py`, `hp:planner/optimizer/feasibility.py` | H1 |
| `ClusterSpecV2.schema_version`, `CpuSocket, PcieSwitch, NetSwitch, SharedResourceSpec, RuntimeCapabilities`, v2 `Link` 필드, 가격 필드 | `hp:planner/inventory.py` | H2 |
| `RejectionStage.THROUGHPUT_UPPER_BOUND/EXCLUDED_BY_SCOPE/NOT_EVALUATED_BUDGET`, `evaluate_candidates(plan_id_base)`, `_assemble_output`, `EnvelopeCache(graph_signature)`, `with_graph_signature`, `LLMServingSimPredictor.set_compile_hook` | `hp:planner/plan.py`, `hp:planner/optimizer/exhaustive.py`, `hp:planner/envelope.py`, `hp:planner/predictor/llmservingsim.py` | H3 |
| `VertexKind, Vertex, DirectedEdge, SharedResource, ResourceGraph, build_resource_graph, bytes_per_s` | `graphsearch/schema.py` | G1 |
| `PathPolicy, Path, PathSet, CutCapacity, BoundaryContext` | `graphsearch/paths.py` | G2 |
| `FlowKind, CommFlow, flows_for, tp_allreduce_bytes, pd_kv_bytes` | `graphsearch/demand.py` | G3 |
| `CostBreakdown, cost_of_devices, cost_lower_bound` | `graphsearch/cost.py` | G4 |
| `RankPlacement, EmbeddedCandidate, EmbeddingPolicy, EmbeddingStats, enumerate_embeddings` | `graphsearch/embeddings.py` | G5 |
| `EquivalenceLevel, Signature, Representative, ConflictMatrix, CompressionPolicy, CompressionReport, compress` | `graphsearch/equivalence.py` | G6 |
| `CandidateStatus, BoundProof, BoundVerdict, BoundPolicy, prune` | `graphsearch/bounds.py` | G7 |
| `RankFeatures, DiversityQuota, ServiceMarginRanker, features_for` | `graphsearch/ranker.py` | G8 |
| `SearchMode, AdaptiveConfig, SearchAudit, AdaptiveSearch` | `graphsearch/adaptive.py` | G9 |
| `RestoredPlan, RestoreError, restore, restore_many, check_capacity, recheck_snapshot` | `graphsearch/restore.py` | G10 |
| `TopologyLossReport, compile_embedded, apply_pd_transfer_cost_embedded, bind`, `ContentionModel, NullContentionModel` | `graphsearch/adapter.py`, `graphsearch/contention.py` | G11 |
| `OracleComparison, run_oracle, run_proposed, compare` | `graphsearch/oracle.py` | G12 |

heteropilot에서 import하는 기존 타입: `planner.inventory`(`ClusterSpecV2, Node, Accelerator, Link, LinkType, Source, ExecutionIsland, AcceleratorProfile, detect_islands, load_profiles_for, compatibility`), `planner.plan`(`CandidateConfig, IslandAssignment, Role, ServingArch, RejectionStage, Rejection, PredictedMetrics, DeploymentPlan, PlannerOutput`), `planner.spec`(`ServiceSpec, Slo, Objective`), `planner.optimizer.surrogate.SurrogateRanker`, `planner.optimizer.exhaustive`(`evaluate_candidates, judge, rank_plans, SearchResult, _assemble_output`), `planner.predictor`(`Predictor, SimResult, SimOutcome`), `planner.envelope.EnvelopeCache`, `planner.util.memory`(`feasible, MemoryReport, model_config, dtype_bits`), `planner.util.kv_transfer._kv_bytes_per_token`, `planner.optimizer.greedy.estimate`, `planner.candidate_generator.CandidateGenerator`, `planner.topology.TopologyGraph`, `planner.predictor.llmservingsim`(`compile_to_sim_config, LLMServingSimPredictor`).

---

## 3. heteropilot 훅 PR

### STEP H1 — 서비스 계약 확장: 처리량·완료율·비용 목적 (heteropilot, 1일)

**역할** 연구설계서 §2 "먼저 확장할 인터페이스". 그래프 쪽 처리량 하한(G7)과 최종 feasibility가 **같은 정의**를 쓰게 하는 전제.

**변경** (`hp:planner/spec.py`)
```python
class Objective(str, enum.Enum):
    ...  # 기존 3개 유지
    MINIMIZE_COST_PER_HOUR = "minimize_cost_per_hour"
class Slo(_Strict):
    ...  # 기존
    min_goodput_rps: float | None = Field(default=None, gt=0)
    min_completion_ratio: float | None = Field(default=None, gt=0, le=1)
    observation_window_s: float | None = Field(default=None, gt=0)
```
(`hp:planner/plan.py`) `PredictedMetrics.offered_requests: int | None = None`; `DeploymentPlan.cost_per_hour_usd: float | None = None`, `cost_basis: str | None = None`.
(`hp:planner/optimizer/feasibility.py`)
```python
def check_throughput(metrics: PredictedMetrics, spec: ServiceSpec) -> tuple[list[Violation], list[str]]:
    """min_goodput_rps -> Violation(metric='slo_goodput_rps'); min_completion_ratio ->
    Violation(metric='completion_ratio', predicted=completed/offered). offered_requests None -> note, never a violation."""
```
`evaluate()`에서 `check_latency` 뒤, `check_power` 앞. 위반 stage는 `SLO_VIOLATED`.
(`hp:planner/optimizer/pareto.py`) `can_score`/`objective_value`에 비용 분기(None → `(False, "objective 'minimize_cost_per_hour' needs a price model; this plan has none")`, 값 `-cost`); `_DIMENSIONS`에 `cost_per_hour_usd`(lower better; None이면 건너뜀).
(`hp:planner/__main__.py::_write_output`) 세 새 필드가 None이면 키 드롭(기존 `measurement_plan` 드롭과 같은 자리).
(`hp:planner/predictor/llmservingsim.py::_parse`) `offered_requests` = per-request CSV 행 수(없으면 None).

**테스트** `hp:tests/test_spec_contract.py`
- (i) `min_goodput_rps=12`, `slo_goodput_rps=10` → `evaluate().passed is False`, metric `slo_goodput_rps`, stage `SLO_VIOLATED`.
- (ii) `min_completion_ratio=0.99`, `completed=98, offered=100` → violation `completion_ratio`, predicted 0.98.
- (iii) `min_completion_ratio` 설정 + `offered_requests=None` → passed(다른 위반 없음), `notes`에 "could not be checked". Violation 아님.
- (iv) 둘 다 None → `tests/test_optimizer.py` 전부 불변.
- (v) 비용 목적: `cost None` → `can_score` False, reason에 "price"; `3.5` vs `2.0` → `rank` 첫 원소 2.0.
- (vi) golden: `tests/test_search.py`, `tests/test_render.py` 바이트 동일; 새 None 필드가 YAML에 없다.
- (vii) `slo.min_goodput_rps: -1` → `SpecError`.

**시험 절차** `pytest tests/test_spec_contract.py tests/test_optimizer.py tests/test_spec.py tests/test_search.py tests/test_render.py -q && ruff check . && mypy planner/`.
**산출물** 코드 + `hp:docs/deviations.md` **D121** "`PredictedMetrics.offered_requests` 추가로 `_metrics_schema_digest`가 바뀌어 기존 envelope cache 전부 miss; 의도된 것".
**완료 조건** 통과, golden 불변, D121.

---

### STEP H2 — v2 인벤토리 스키마 (heteropilot, 1일)

**역할** 그래프가 필요로 하는 정점 종류·공유 자원·단위·가격·runtime 능력을 `ClusterSpecV2`에 **v2 전용**으로 추가한다. v1 파일은 한 글자도 다르게 읽히지 않는다.

**변경** (`hp:planner/inventory.py`)
```python
class ClusterSpecV2(_Strict):
    cluster_id: str
    schema_version: int = 1                              # NEW; 1 = today's file, 2 = graph-aware
    nodes: list[Node]; links: list[Link]
    net_switches: list[NetSwitch] = []                   # v2
    shared_resources: list[SharedResourceSpec] = []      # v2
    snapshot_id: str | None = None                       # v2
class CpuSocket(_Strict):  id: str; numa_node: int | None = None
class PcieSwitch(_Strict): id: str; upstream: str | None = None
class NetSwitch(_Strict):  id: str; ports: int | None = None
class SharedResourceSpec(_Strict):
    id: str; kind: Literal["pcie_uplink","nic","switch_port","other"]
    capacity: float; unit: Literal["GB/s","Gbit/s"] = "GB/s"; reserved: float = 0.0
    node: str | None = None; source: Source = Source.PLACEHOLDER
class Node(_Strict):
    ...; cpu_sockets: list[CpuSocket] = []; pcie_switches: list[PcieSwitch] = []   # v2
    host_price_per_hour_usd: float | None = Field(default=None, ge=0)             # v2
class Link(_Strict):
    ...  # bandwidth_gbps는 v1 의미(GB/s) 그대로
    bandwidth_unit: Literal["GB/s","Gbit/s"] = "GB/s"   # v2; v1에서 GB/s 이외 → 오류
    direction: Literal["bidir","src_to_dst"] = "bidir"  # v2
    rdma: bool | None = None; p2p: bool | None = None   # v2
    shared_resource: str | None = None                  # v2; contention_group과 동시 지정 금지
class RuntimeCapabilities(_Strict):
    collectives: list[str] = []; max_world_size: int | None = None; kv_transfer: bool | None = None; source: Source = Source.PLACEHOLDER
class Accelerator(_Strict): ...; price_per_hour_usd: float | None = Field(default=None, ge=0)   # v2
class AcceleratorProfile(_Strict):
    ...; runtime_capabilities: RuntimeCapabilities | None = None
    price_per_hour_usd: float | None = Field(default=None, ge=0); price_source: Source | None = None
```
검증: `schema_version == 1`에서 v2 필드가 기본값 아닌 값을 가지면 `InventoryError`(메시지에 "schema_version 2 required for <field>"). v2에서 `Link.src/dst`는 `<node>/<accel|nic|cpu_socket|pcie_switch>` 또는 `<net_switch_id>`(슬래시 없음, `net_switches`에 존재) 허용; `_consistent`가 확인. `shared_resource`는 `shared_resources[].id`에 존재해야 함. `AcceleratorProfile.price_per_hour_usd`가 있으면 `price_source` 필수(절대 규칙 3의 가격 버전).

**테스트** `hp:tests/test_inventory_v2.py`
- (i) 기존 `examples/clusters/*.yaml` 전부 `schema_version == 1`로 로드, `tests/test_inventory.py` 불변.
- (ii) v1 파일에 `bandwidth_unit: Gbit/s` / `cpu_sockets` / `shared_resources` → `InventoryError`, 메시지에 "schema_version".
- (iii) v2 최소 예제(`tests/data/cluster_v2_min.yaml`: 노드 2, cpu_socket 1씩, net_switch 1, shared_resource 1, Gbit/s 링크 1) 로드 성공; `contention_group`+`shared_resource` 동시 → 오류; 미존재 `shared_resource` id → 오류; `net_switch` 미등록 endpoint → 오류.
- (iv) `price_per_hour_usd` 있고 `price_source` 없음 → 오류.
- (v) golden 불변.

**시험 절차** `pytest tests/test_inventory_v2.py tests/test_inventory.py tests/test_search.py -q && ruff check . && mypy planner/`.
**산출물** 코드 + `hp:docs/docs/reference/cluster-config.md` v2 절 + **D120** "v1 `bandwidth_gbps`는 GB/s로 계속 읽음; 단위·정점·공유자원 확장은 `schema_version: 2` 전용".
**완료 조건** 통과, D120.

---

### STEP H3 — 탐색 훅 (heteropilot, 1일)

**역할** graphsearch 드라이버가 heteropilot의 평가·조립·캐시·predictor를 **수정 없이 재사용**할 수 있게 하는 최소 훅. 기본값에서 전부 no-op.

**변경**
1. `hp:planner/plan.py`
   ```python
   class RejectionStage(str, enum.Enum):
       ...
       THROUGHPUT_UPPER_BOUND = "throughput_upper_bound"   # sound bound; only meaningful when slo.min_goodput_rps is set
       EXCLUDED_BY_SCOPE = "excluded_by_scope"             # not a verdict: enumeration cap or caller filter
       NOT_EVALUATED_BUDGET = "not_evaluated_budget"       # adaptive search never reached it; NOT a verdict
   ```
2. `hp:planner/optimizer/exhaustive.py`
   - `evaluate_candidates(..., plan_id_base: int = 0)`; `_plan_id(plan_id_base + index)`.
   - `search()`의 `all_rejections = …` 이하(1044~1182행)를 `_assemble_output(*, spec, cluster, generation_generated: int, generation_survivors: int, evaluation: SearchResult, all_rejections: list[Rejection], caveats: list[str], prov: dict, island_tiers, island_hw, enable_bound_pruning: bool) -> PlannerOutput`로 추출. `search()`는 그것을 호출. **동작 불변**(golden).
3. `hp:planner/envelope.py` `EnvelopeCache.__init__(..., graph_signature: str | None = None)`; `_path()`에 `if self.graph_signature: name = prov.hash_object([name, f"graph={self.graph_signature}"])`(`topology_level` 분기 뒤); `with_graph_signature(self, sig: str) -> EnvelopeCache`(같은 root/spec/accelerator_of/link_bw/trace_digest/topology_level, `hits/misses` 카운터를 **공유**하도록 얕은 복사 후 참조 연결 — 구현은 `copy.copy(self)` 후 `graph_signature` 교체, 카운터는 `_stats` dict로 옮겨 공유).
4. `hp:planner/predictor/llmservingsim.py`
   ```python
   CompileHook = Callable[[CandidateConfig, ClusterSpecV2, dict[str, ExecutionIsland], dict[str, AcceleratorProfile]], "tuple[dict, TopologyReduction] | None"]
   class LLMServingSimPredictor(Predictor):
       def set_compile_hook(self, hook: CompileHook | None) -> None
   ```
   `_run_once`(또는 컴파일을 호출하는 자리)에서 훅이 있으면 먼저 호출; `None`이면 기존 `compile_to_sim_config`. 훅이 준 `TopologyReduction`은 기존과 같은 provenance 자리에 기록.
5. `hp:CLAUDE.md` D-블록 표 `| D120–D129 | WORK_ORDER_graph_search.md in swsok/heteropilot-graphsearch (claimed 2026-09-22) |`, 실험 태그 표 `| E-G* | heteropilot-graphsearch |`, 문서 표에 "Graph search는 `swsok/heteropilot-graphsearch`로 분리, heteropilot `<H3 sha>`에 pin, planner를 import만" 한 줄.
6. `hp:docs/deviations.md` **D122** "그래프 모드(graphsearch)는 `CandidateGenerator(enable_bound_pruning=False)`로 템플릿만 받고 컷 기반 하한을 자체 적용; heteropilot stage 4·5는 기본 경로에서 불변", **D123** "networkx는 graphsearch 의존; heteropilot은 계속 hand-rolled BFS", **D124** "graphsearch MVP 어댑터는 공유 자원을 시뮬레이터에 전달하지 못함(D3의 연장); 손실은 graphsearch `TopologyLossReport`가 명시".

**테스트** `hp:tests/test_search_hooks.py`
- (i) `plan_id_base=5`로 3후보 평가 → plan_id `hp-00005..hp-00007`; 기본값 0 → `tests/test_search.py` 불변.
- (ii) `_assemble_output` 추출 후 golden 바이트 동일(`tests/test_search.py`, `tests/test_render.py`).
- (iii) `graph_signature=None` → `_path` 기존과 동일; `"x"`와 `"y"` → 서로 다르고 None과도 다름; `with_graph_signature` 후 `get` miss가 원본 `stats()`에 반영.
- (iv) `set_compile_hook(lambda *a: None)` → 기존 컴파일 호출(monkeypatch 카운트); 훅이 dict를 반환 → 그 dict가 sim config로 기록되고 `compile_to_sim_config` 미호출.
- (v) 새 `RejectionStage` 값 문자열이 `summarize_rejections` 키로 동작; 기존 값 문자열 불변.
- (vi) `tests/test_deviations_numbering.py`, `tests/test_experiment_ids.py` 통과.

**시험 절차** `pytest tests/test_search_hooks.py tests/test_search.py tests/test_render.py tests/test_envelope.py tests/test_parallel.py tests/test_deviations_numbering.py tests/test_experiment_ids.py -q && ruff check . && mypy planner/`.
**완료 조건** 통과, golden 불변, D122~D124, `CLAUDE.md` 표. **머지 후 sha를 graphsearch G0에 전달.**

---

## 4. graphsearch STEP

### STEP G0 — 저장소 부트스트랩, fixture, 그래프 인지 mock, CI (graphsearch, 1일)

**역할** 이후 모든 STEP이 공유하는 뼈대. 코드 로직 없음.

**전제** 사용자가 GitHub에 `swsok/heteropilot-graphsearch`(빈 저장소, `main`)를 만들어 두었고, H3가 heteropilot `main`에 머지되어 sha `<H3>`를 안다. H3 전에 시작하면 `feat/gs-h3-*` 브랜치 sha에 임시 고정하고 이 STEP 안에서 갱신 커밋을 남긴다.

**작업**
1. 뼈대
   ```bash
   git init && git submodule add https://github.com/swsok/heteropilot.git vendor/heteropilot
   (cd vendor/heteropilot && git checkout <H3 sha> && git submodule update --init --recursive)   # astra-sim까지
   ```
   `pyproject.toml`: heteropilot의 것을 참고해 `[tool.ruff] line-length=100 target-version=py310 select=["E","F","W","I","UP","B","C4","RUF","SIM"] extend-exclude=["vendor"]`, `[tool.mypy] files=["graphsearch"] ignore_missing_imports=true`, `[tool.pytest.ini_options] testpaths=["tests"]`. `requirements.txt`: `networkx>=3.2 pydantic pyyaml numpy pytest ruff mypy`(시뮬레이터 실행은 `vendor/heteropilot/.venv` 재사용을 README에 안내).
   `graphsearch/paths_root.py`: `HETEROPILOT_ROOT = Path(os.environ.get("HETEROPILOT_ROOT", Path(__file__).resolve().parents[1] / "vendor/heteropilot"))`, `examples_dir()`, `profiles_dir()`, `ensure_importable()`(`sys.path`에 root 삽입; `__main__`과 `tests/conftest.py`가 첫 줄에서 호출).
   `README.md`: 목적, 실행법(`export PYTHONPATH=$PWD:$PWD/vendor/heteropilot`, `python -m graphsearch plan …`), pin sha와 이유, "이 저장소의 숫자는 heteropilot의 provenance 라벨을 그대로 전파하며 자체 측정을 하지 않는다", private 서브모듈 인증 안내.
   `CLAUDE.md`(graphsearch): 이 저장소의 목적, 두 저장소 경계 규칙(§0.2·0.5), heteropilot `CLAUDE.md`·`AGENTS.md`를 함께 읽으라는 포인터, 게이트 명령, "vendor/heteropilot은 읽기 전용 — 수정은 heteropilot PR로", GS-n 기록 규칙.
   `docs/decisions.md`: **GS-1** 두 저장소 결정, **GS-2** networkx 도입.
   `docs/graph_search_design.md`, `WORK_ORDER_graph_search.md` 커밋.
2. **toy fixture**(전부 `source: placeholder`, 머리에 "가상 구성, 연구설계서 §9" 주석) — `fixtures/`:
   - `clusters/graph-toy-abcde.yaml`(v1) / `graph-toy-abcde.v2.yaml`(v2: cpu_socket, NIC, `sw0` net_switch, uplink를 `shared_resources`로): 노드 A,B(GPU 2개, 10 GB/s uplink), C,D(GPU 2개, 5 GB/s), E(NPU 2개). 노드 내부 GPU pair PCIE 64 GB/s.
   - `clusters/graph-toy-shared-nic.yaml` / `.v2.yaml`: 노드 X,Y 각각 GPU 2개+NVLINK, 동일 가격. X의 uplink `reserved: 6`(v2) — 외부 서비스 공유. v1 사본은 `contention_group`만.
   - `clusters/graph-toy-asym.v2.yaml`: 5노드, 가격·uplink 전부 다름(압축률 0 실패 조건용).
   - `profiles/toy_gpu.yaml`(`sim_hardware: A5000`로 기존 perf 번들 재사용, `supported_models: Llama-3.1-8B@bfloat16`, `max_tp_size: 2`, `price_per_hour_usd: 2.0, price_source: placeholder`, `runtime_capabilities: {collectives:[all_reduce,p2p], max_world_size: 8, kv_transfer: true, source: placeholder}`), `profiles/toy_npu.yaml`(`sim_hardware` 없음 → stage 1에서 자연 제외 = §9 표의 "runtime 미지원").
   - `service_specs/graph-toy-llama31-8b.yaml`: heteropilot `llama31-8b.yaml` 복사 + `ttft.max_ms: 50, tpot.max_ms: 50`(§9 예시 재현용) + `objective.primary: minimize_cost_per_hour`.
   fixture 파일 안의 `profile:` 경로는 graphsearch 루트 기준(`fixtures/profiles/toy_gpu.yaml`)이며, `load_profiles_for(cluster, root=GRAPHSEARCH_ROOT)`로 읽는다.
3. `tests/conftest.py`: `paths_root.ensure_importable()`; heteropilot `tests/conftest.py`의 `MockPredictor`·`MOCK_ROOFLINE_SLACK`를 `vendor/heteropilot/tests`를 `sys.path`에 넣어 import(`tests/__init__.py`가 heteropilot에 있으므로 `from tests.conftest import MockPredictor`는 이름이 충돌 — `importlib.util.spec_from_file_location("hp_conftest", HETEROPILOT_ROOT/"tests/conftest.py")`로 로드).
   `tests/graph_fixtures.py`: `toy_cluster, toy_cluster_v2, toy_shared_cluster, toy_shared_cluster_v2, toy_asym_cluster, toy_profiles, toy_islands, toy_spec` fixture + 
   ```python
   class GraphAwareMockPredictor(hp_conftest.MockPredictor):
       """Deterministic; TTFT = base TTFT + sum over PD_KV_TRANSFER flows of bytes / (path bottleneck - shared reservation);
       TPOT = base TPOT + TP all-reduce time on the flow's path bottleneck. Never faster than the bounds."""
       def bind_embeddings(self, by_template_id: dict[str, "EmbeddedCandidate"]) -> None: ...   # G5에서 본문
       calls: int   # 호출 카운터(예산 테스트용)
   ```
4. CI(`.github/workflows/ci.yml`): checkout with submodules(읽기 토큰 secret), `pip install -r requirements.txt`, `pytest -q`, `ruff check .`, `mypy graphsearch/`, 그리고 `(cd vendor/heteropilot && pip install pyyaml pydantic numpy pytest && pytest tests/test_spec.py tests/test_inventory.py tests/test_optimizer.py -q)`(시뮬레이터 없이 도는 부분집합으로 서브모듈 건강 확인).
5. `tests/test_fixtures.py::test_fixtures_load`: 다섯 클러스터 로드, abcde 섬 5개, shared-nic 2개; `GraphAwareMockPredictor`가 `Predictor` 인스턴스.

**완료 조건** CI 녹색 · `python -c "import graphsearch, planner"` 성공 · fixture 테스트 통과.
**금지** `vendor/heteropilot` 안의 파일 수정.

---

### STEP G1 — 정규화 자원 그래프 스키마와 단위 어댑터 (graphsearch, 1일)

**역할** `ClusterSpecV2`(v1|v2)를 `ResourceGraph`로 바꾼다. 이후 모든 그래프 연산의 유일한 물리 표현(설계서 §5.1).

**입력** `ClusterSpecV2`, `dict[str, AcceleratorProfile]`. **출력** `ResourceGraph`.

**API** 설계서 §5.1의 `VertexKind, Vertex, DirectedEdge, SharedResource, ResourceGraph, build_resource_graph, bytes_per_s`.
**규칙** `direction="bidir"` → 간선 2개(`:fwd`,`:rev`) 각각 전체 용량; 기존 `duplex="half"`면 두 간선이 `shared_resource_id="halfdup:<link_id>"`로 묶인 자동 SharedResource(capacity=링크 용량). v1 `contention_group` → `shared_resource_id`로 옮기되 용량은 멤버 간선 최소값(placeholder, `unit_notes` 기록). 가속기 attrs: `model, backend, memory_bytes, state, profile_id, price_per_hour_usd(accelerator 우선, 없으면 profile), active_power_w(profile.power.active_power or tdp_w), runtime_capabilities(dict|None)`. `snapshot_version = sha256(json(snapshot_id, sorted reserved, sorted accelerator states))`.

**테스트** `tests/test_schema.py`
- (i) `vendor/heteropilot/examples/clusters/heterogeneous-lab.yaml`(v1) → `bandwidth_gbps=64` 간선 용량 64e9; 간선 수 = 링크 수×2; `contention_group` 두 링크가 같은 `shared_resource_id`.
- (ii) v2 toy에서 `bandwidth_unit: Gbit/s, 100` → 12.5e9(`approx`).
- (iii) 공유 자원 보조 정점 `res:<id>` 존재; 독립 링크 2개 vs 공유 링크 1개 클러스터의 `digest()` 다름.
- (iv) 결정론: 링크 순서를 섞은 두 YAML → 같은 `digest()`.
- (v) `snapshot_version`은 `reserved`가 바뀌면 달라진다.
- (vi) `bytes_per_s(1,"GB/s")==1e9`, `bytes_per_s(8,"Gbit/s")==1e9`, 미지 단위 → `ValueError`.

**시험 절차** `pytest tests/test_schema.py -q && ruff check . && mypy graphsearch/`.
**완료 조건** 통과.

---

### STEP G2 — 경로 집합, 컷 용량, 경계 문맥 (graphsearch, 1일)

**역할** 하한(G7)이 컷의 낙관적 총 용량을 쓰고, 동등성(G6)이 후보 바깥의 공유 자원까지 보게 하는 근거(설계서 §5.2).

**API** 설계서 §5.2. **구현** `to_networkx`는 공유 자원 보조 정점을 `res:<id>_in → res:<id>_out`(용량 `capacity - reserved`)으로 분할하고 그 자원을 쓰는 물리 간선을 `src → res_in`, `res_out → dst`로 대체. `cut_capacity`는 `nx.maximum_flow`(정렬된 노드 순으로 super-source/sink 연결). `path_set`은 `nx.shortest_simple_paths`를 `max_hops`로 잘라 `(hops, -bottleneck, edge ids)` 정렬. `CutCapacity.assumptions`에 "max-flow over nominal capacities minus external reservation; no per-flow contention; filters: …".

**테스트** `tests/test_paths.py`
- (i) toy-abcde v2: A/gpu0→B/gpu0 병목 10e9, C/gpu0→D/gpu0 5e9, A/gpu0→C/gpu0 5e9.
- (ii) `cut_capacity({A/gpu0,A/gpu1},{B/gpu0,B/gpu1})` = 10e9; NIC 2개 노드 fixture(테스트 내 생성)에서는 합.
- (iii) 공유 자원 `reserved=6e9, capacity=10e9` → 컷 4e9, `saturating_resources`에 id.
- (iv) `require_rdma=True`, `rdma=False` 링크만 → `paths == ()`, 컷 0 + 사유.
- (v) `boundary_context`: A 내부 pair → PCIe 공유 자원만; A–B pair → NIC·uplink 포함.
- (vi) 결정론.

**시험 절차** `pytest tests/test_paths.py -q && ruff check . && mypy graphsearch/`.
**완료 조건** 통과.

---

### STEP G3 — 통신 수요 모델 (graphsearch, 1일)

**역할** 후보의 통신을 `CommFlow`로 기술. 하한·랭커·어댑터가 한 정의를 공유(설계서 §5.4). **heteropilot 코드는 건드리지 않고** 일치를 테스트로 보장.

**API** 설계서 §5.4. `flows_for` 규칙: replica의 tp>1 그룹 → `TP_ALLREDUCE`(participants=ranks, `bytes_per_event=tp_allreduce_bytes`, `events_per_request=tp_allreduces_per_output_token*output_tokens.p50`, `"tpot"`, allowed_paths=모든 rank 쌍 `path_set`); pp>1 → `PP_ACTIVATION`(`"both"`); `PD_SPLIT` → prefill replica i ↔ decode replica i `PD_KV_TRANSFER`(`bytes_per_event`=p50 프롬프트, assumptions에 p95·p99 값, `"ttft"`); replica 첫 rank ↔ 노드 NIC `INGRESS/EGRESS`(`"none"`). G5 이전에는 `placements_from_islands(template, islands)`(섬 첫 장치들) helper로 테스트.

**테스트** `tests/test_demand.py`
- (i) `tp_allreduce_bytes("meta-llama/Llama-3.1-8B","bfloat16",2)==8192`(hidden 4096×2 B×1.0), tp=4 → 12288.
- (ii) **heteropilot 일치**(설계서 §7.9): heterogeneous-lab에서 `CandidateGenerator(enable_bound_pruning=True)`가 낸 `TOPOLOGY_INFEASIBLE` 거절 사유 문자열의 "floor X.Yms"를 파싱해, `tp_allreduce_bytes`·`island_interconnect` 값으로 재계산한 floor와 `abs(diff) < 0.05 ms`. 거절이 없는 스펙이면 `tpot.max_ms=0.01`로 강제 유발.
- (iii) 단일 섬 tp=2 dp=2 → `TP_ALLREDUCE` 2개, participants disjoint.
- (iv) P/D 후보 → `PD_KV_TRANSFER` 1개, `bytes_per_event == pd_kv_bytes(..., input_tokens.p50)`.
- (v) tp=1 dp=1 aggregated → INGRESS/EGRESS만.
- (vi) 결정론·`flow_id` 유일.

**시험 절차** `pytest tests/test_demand.py -q && ruff check . && mypy graphsearch/`.
**완료 조건** 통과.

---

### STEP G4 — 비용 모델 (graphsearch, 0.5일)

**역할** 시간당 금전 비용과 하한. 가격이 없으면 None(설계서 §5.11).

**API** `CostBreakdown(accelerator_usd_per_hour, host_usd_per_hour, total_usd_per_hour|None, missing, basis)`, `cost_of_devices(devices, graph)`, `cost_lower_bound(template, islands, graph)`(assignment별 섬의 최저가 `total_devices`개 + 닿는 노드 호스트 비용 전액; 결측 있으면 None).

**테스트** `tests/test_cost.py`
- (i) 장치 2.0, 호스트 1.0, 2장치 1노드 → 5.0.
- (ii) 한 장치 가격 None → `total None`, `missing`에 정점.
- (iii) `cost_lower_bound ≤ cost_of_devices(임의 임베딩)` — 가격 다른 두 노드 fixture 3케이스(G5 이후 property로 승격).
- (iv) 가격 전무(v1 예제) → None, 예외 없음.

**시험 절차** `pytest tests/test_cost.py -q`. **금지** 활성 장치 수를 비용으로 대체.

---

### STEP G5 — 임베딩 열거 (graphsearch, 1.5일)

**역할** 섬 단위 `CandidateConfig`를 물리 장치가 확정된 `EmbeddedCandidate`로 펼친다(설계서 §5.3).

**입력** `CandidateGenerator(spec, cluster, islands, profiles, enable_bound_pruning=False, enable_pd=…).generate().candidates`, `islands`, `ResourceGraph`, `ServiceSpec`, `EmbeddingPolicy`. **출력** `list[EmbeddedCandidate]`, `EmbeddingStats`.

**열거 규칙** ① assignment별 섬의 `accelerator_ids`에서 `total_devices`개 조합(정렬 id, `itertools.combinations`) → `dp_replicas` 그룹(`tp*pp`씩). TP 그룹 내부·복제본 사이 순서 무관(정렬). pp>1은 stage 순서 유지(MVP 생성기는 pp=1). ② 같은 섬 참조 두 assignment는 장치 겹침 금지. ③ `hierarchical=True`: `(같은 pcie_switch 수, 같은 cpu_socket 수)` 내림차순으로 열거 → 상한에 걸릴 때 지역성 높은 것이 남음. ④ `canonical_only=True`: `canonical_embedding_key`(정렬 placements JSON 해시) 중복 건너뛰기, `skipped_symmetric`. ⑤ `flows = demand.flows_for`, `boundary = paths.boundary_context`, `transit_closure`, `resource_demand[res] = Σ bytes_per_event*events_per_request*arrival_rate_rps`(첫 허용 경로가 그 자원을 지날 때). ⑥ `GraphAwareMockPredictor.bind_embeddings` 본문 구현.

**테스트** `tests/test_embeddings.py`
- (i) toy-abcde, 단일 섬 A tp=2 dp=1 → 1개; tp=1 dp=1 → 2개.
- (ii) mixed(A,B) tp=1 dp=1 → 4개.
- (iii) `max_embeddings_per_template=1`에서 (ii) → 1개, `truncated_by_policy=3`, id 기록.
- (iv) 4장치 섬 fixture(테스트 내 생성) dp=2 tp=1 → 대칭 제거 후 3개, `skipped_symmetric>0`.
- (v) 회귀(설계서 §7.5): heterogeneous-lab(v1)에서 임베딩 수 == 템플릿 수(모든 섬이 필요 장치 수와 같은 후보에 대해), `embedding.template is template`.
- (vi) `id` 유일, 결정론.
- (vii) A–B 걸친 tp=2의 `boundary.shared_resources`에 두 uplink 포함, A 내부 pair 미포함.

**시험 절차** `pytest tests/test_embeddings.py -q && ruff check . && mypy graphsearch/`.
**완료 조건** 통과. **주의** `CandidateConfig` 미수정.

---

### STEP G6 — 정확 동등성 압축 (graphsearch, 2일)

**역할** 임베딩을 대표로 접는다. WL 해시 bucket → VF2 확정. **해시만으로 병합하지 않는다**(설계서 §5.5, 연구설계서 §5).

**API** 설계서 §5.5의 `EquivalenceLevel, Signature, Representative, ConflictMatrix, CompressionPolicy, CompressionReport, candidate_graph, signature, compress`.
**라벨 규칙** 정점 라벨 = JSON(sort_keys) — 가속기 `{kind, model, backend, memory_bytes, profile_id, price(옵션), power_w, role, assignment_index, tp, pp, slot:"tp_member"}`; transit `{kind}`+속성; 공유 자원 `{kind:"shared_resource", capacity, reserved, res_kind}`. 간선 라벨 `{link_type, capacity, latency, rdma, p2p, measurement_keys}`. 정점 id·노드 id 제외. `include_boundary=False`면 boundary 정점·간선 제외(ablation).
**알고리즘** ① `nx.weisfeiler_lehman_graph_hash(G, node_attr="label", edge_attr="label", iterations=policy.wl_iterations)` + `attr_histogram` + `tool_version=f"networkx=={nx.__version__};labels=v1"`. ② bucket `(template.id, wl_hash, attr_histogram)`. ③ `DiGraphMatcher(node_match=categorical_node_match("label",None), edge_match=categorical_edge_match("label",None)).is_isomorphic()` 성공 시 `.mapping` 저장. ④ 실패 → 새 대표. ⑤ `max_vf2_seconds` 초과분 `HASH_ONLY`(병합 금지). ⑥ 충돌: 장치 공유 또는 어떤 공유 자원에서 `demand_a+demand_b > capacity-reserved`.

**테스트** `tests/test_equivalence.py`
- (i) 노드 재라벨(A↔B 사본) → 대응 임베딩 `Signature` 동일, 대표 수 동일.
- (ii) **§9 표 재현**: 두 가속기 tp=2 템플릿(GPU) 임베딩 28개 → 대표 5개, `sorted(multiplicity)==[2,2,4,4,16]`.
- (iii) **공유 uplink**: toy-shared-nic X-pair·Y-pair를 P/D 상대와 묶은 임베딩 → 서명 다름, 대표 2; `include_boundary=False` → 대표 1.
- (iv) 역할 교환(P/D 방향) → 서명 다름.
- (v) 해시 충돌 방어: `wl_hash`를 monkeypatch로 동일하게 강제한 구조 다른 두 그래프 → 병합되지 않음.
- (vi) `max_vf2_seconds=0` → 전부 `HASH_ONLY`, `exact_merges=0`.
- (vii) 충돌 행렬: 장치 공유 쌍; uplink 10e9에 6e9+6e9 충돌, 4e9+4e9 비충돌.
- (viii) 결정론.

**시험 절차** `pytest tests/test_equivalence.py -q && ruff check . && mypy graphsearch/`.
**산출물** 코드 + `docs/decisions.md` GS-2 갱신(networkx 버전 고정 방식).
**완료 조건** 통과.

---

### STEP G7 — 증명 가능한 제거와 후보 상태 (graphsearch, 1.5일)

**역할** 연구설계서 §6의 표. 모든 거절에 재계산 가능한 `BoundProof`(설계서 §5.6).

**입력** `list[Representative]`(exemplar 검사 = 전 임베딩 판정), `ServiceSpec`, `ResourceGraph`, `profiles`, `EmbeddingStats`, `incumbent_cost`. **출력** `dict[rep_id, BoundVerdict]`, `list[Rejection]`.

**검사 정의(정확히 이 식)**
1. `compat`: `planner.inventory.compatibility(spec.model, dtype, profile)` False → IMPOSSIBLE(`BACKEND_INCOMPATIBLE`). `runtime_capabilities`가 있으면 tp>1에 `"all_reduce" in collectives`, `tp<=max_world_size`, PD_SPLIT에 `kv_transfer is True` 요구; **None(미등록)은 검사 생략 + relaxations "runtime_capabilities unstated: not checked"**.
2. `memory`: assignment별 `memutil.feasible`; `fits False` 또는 `kv_tokens < median_len`(prefill은 input p50, 그 외 in+out p50) → IMPOSSIBLE(`MEMORY_INFEASIBLE`). `relaxations=("no offload","no quantization beyond spec dtype","no recompute")`.
3. `comm_latency`: TPOT — `TP_ALLREDUCE` flow마다 `t = min path latency + bytes_per_event / cut_capacity(participants 쌍 최악 컷)`; `floor_ms = 2*layers*t/1e6`; flows 간 **max**(동시 진행 가정) `> tpot.max_ms` → IMPOSSIBLE(`TOPOLOGY_INFEASIBLE`). TTFT — `PD_KV_TRANSFER`(p50) + `PP_ACTIVATION` 순차 합 `> ttft.max_ms` → IMPOSSIBLE. `relaxations=("compute free","cut = max-flow nominal minus reservation","no contention","ring all-reduce 2(n-1)/n")`. 측정 실효 대역폭 **사용 금지**.
4. `throughput_capacity`: `spec.slo.min_goodput_rps is None`이면 실행·proof 모두 없음. 있으면 `ub_tps = Σ_decode_replicas kv_capacity_active / (weight_bytes/mem_bw)`(knob 무시·KV 스트리밍 무시, 낙관), `ub_rps = ub_tps/output_tokens.p50`, `ub_rps_cut = min_flow cut_capacity/(bytes_per_event*events_per_request)`; `min(ub_rps, ub_rps_cut) < min_goodput_rps` → IMPOSSIBLE(`THROUGHPUT_UPPER_BOUND`).
5. `cost_lower_bound`: `policy.cost_lower_bound and incumbent_cost is not None and objective.primary is MINIMIZE_COST_PER_HOUR`에서 `cost.cost_lower_bound(template) >= incumbent_cost` → IMPOSSIBLE(`ANALYTICAL_LOWER_BOUND`, `check="cost_lower_bound"`). 가격 결측 → 생략.
6. `stats.truncated_template_ids` → `Rejection(candidate_id=f"{template_id}/*", stage=EXCLUDED_BY_SCOPE, reason=...)` 1건/템플릿.
7. `policy.demoted`의 검사는 실행하되 `DEFERRED_HEURISTIC`(stage `SURROGATE_PRUNED`), `proof.safe=False`.

**테스트** `tests/test_bounds.py`(`GraphAwareMockPredictor`)
- (i) **완화 property**: toy-abcde·toy-shared-nic, 각 검사 on/off 전 대표 평가 → feasible 최선 `candidate id`·목적값 동일. 실패 메시지에 검사 이름.
- (ii) `min_goodput_rps=None` → `THROUGHPUT_UPPER_BOUND` 거절·proof 0.
- (iii) §9 추가 필터: `ttft.max_ms=50`, PD 프롬프트 바이트가 0.4 GB가 되는 `input_tokens.p50` → 5 GB/s 컷(C–D, fast–slow) IMPOSSIBLE(80 ms), A–B(40 ms)·내부 pair 생존 → 대표 3개.
- (iv) 모든 `Rejection`에 대응 `BoundProof`; `inputs`로 `bound_value` 재계산 일치.
- (v) `runtime_capabilities=None` → compat 거절 없음 + relaxations; `collectives=[]`·tp=2 → 거절.
- (vi) `demoted={"comm_latency"}` → `DEFERRED_HEURISTIC`, `safe=False`.

**시험 절차** `pytest tests/test_bounds.py -q && ruff check . && mypy graphsearch/`.
**산출물** 코드 + `docs/decisions.md` **GS-3** "그래프 모드는 heteropilot stage 4·5를 끄고 컷 하한을 쓴다(heteropilot D122와 쌍)".
**완료 조건** 통과.

---

### STEP G8 — 서비스 여유도 랭커와 다양성 배분 (graphsearch, 1일)

**역할** `planner.optimizer.surrogate.SurrogateRanker`를 구현하는 설명 가능한 규칙 랭커(설계서 §5.7). 순서만 반환.

**API** `RankFeatures, DiversityQuota, ServiceMarginRanker(features_of, quota=None, k_hint=None)`, `features_for(rep, spec, graph, profiles, *, gpu_memory_utilization=0.90)`:
`ttft_ratio` = (PD/PP flow 시간 — `NullContentionModel` 경유, 측정 실효 대역폭 허용: `TopologyGraph.link_bandwidth_gbps`의 measurement 우선 규칙 재사용) / `ttft.max_ms`; `tpot_ratio` = (`greedy.estimate(template).roofline_tpot_ms` + TP flow 시간)/`tpot.max_ms`; `goodput_ratio` = (`min_goodput_rps or arrival_rate_rps`)/G7의 `ub_rps`; `shared_nic_util` = max_res demand/(capacity−reserved); `cut_margin` = min_flow cut/(요구 bytes/s); `memory_margin` = kv_tokens/(median_len×max_num_seqs); `outside_calibration` = `calibration.load_domain_index(HETEROPILOT_ROOT)`에 `sim_hardware` 도메인 없음(실패 시 True); `structure_key=(tp, node_count, accel_model_set, is_pd)`.
**정렬** 1군 `risk_proxy≤1` → `(cost, risk_proxy, id)`; 2군 → `(risk_proxy, cost, id)`; `cost None`은 1군 뒤·2군 앞. 다양성: `k_hint` 안에서 `structure_key` 그룹별 `ceil(K*reserved_fraction/그룹수)` 자리 선점, 예산<그룹수면 정렬 순 순환, **정확히 K개**.

**테스트** `tests/test_ranker.py`
- (i) ABC 계약(길이·집합·결정론).
- (ii) 1군 앞, 1군 내 비용 오름차순, `cost None` 위치.
- (iii) tp=1 10개(최저가) + tp=2 2개, `k_hint=4, reserved_fraction=0.5` → 상위 4에 tp=2 ≥1.
- (iv) 예산 3·그룹 5 → 정확히 3, 순환 순서 = `structure_key` 정렬.
- (v) property: `tpot_ratio*tpot.max_ms ≥ G7 comm_latency floor`.
- (vi) `BinnedRooflineRanker`를 같은 후보에 돌려 길이·집합 동일(baseline 호환).

**시험 절차** `pytest tests/test_ranker.py -q`. **금지** 랭커가 `PredictedMetrics` 생성.

---

### STEP G9 — 적응형 Top-K 드라이버 (graphsearch, 2일)

**역할** 대표를 K 스케줄로 배치 평가, 순위·근사 그룹 갱신, 예산/인증 종료(설계서 §5.8). heteropilot의 `evaluate_candidates(plan_id_base)`·`judge`·`rank_plans`·`_assemble_output`(H3)을 재조립.

**루프(정확히 이 순서)**
1. `eligible` = status ∉ {IMPOSSIBLE_PROVEN, EXCLUDED_BY_SCOPE}, `rep_id` 정렬.
2. `for k in k_schedule`: a. `ranker.order(미평가 exemplar templates, k_hint=k-evaluated)` 앞 `k-evaluated`개 → 배치. b. `max_simulations`·`max_wall_seconds` 검사(배치 잘라 정확히 남은 수). c. `adapter.bind(predictor, {template.id: exemplar}, graph)` 후 `evaluate_candidates(batch, spec, cluster, islands, profiles, predictor, cache=cache.with_graph_signature(sig) if cache else None, plan_id_base=evaluated_count)`. PD_SPLIT 대표는 `apply_pd_transfer_cost_embedded`로 후처리(heteropilot의 `apply_pd_transfer_cost`는 `evaluate_candidates` 안에서 이미 호출되므로, 이중 부과를 막기 위해 **candidate `serving_arch`가 PD_SPLIT이면 heteropilot 쪽 조정을 되돌린 뒤** graphsearch 값을 더한다 — `pd_transfers` info의 `xfer_ms_*`를 빼고 embedded 값을 더함; 테스트 (x)). d. feasible plan에 `cost_per_hour_usd`, `cost_basis` 부착(`model_copy`). e. incumbent = `rank_plans(feasible_so_far).best`. f. 잔차 `features_for` 예측 대 실측 비율 저장; `structure_key` 그룹 잔차 표준편차 > `split_approx_groups_on_residual`이면 그 그룹 미평가 대표를 다음 순서 앞으로. g. `CERTIFY`: `bounds.prune(미평가, policy(cost_lower_bound=True), incumbent_cost)` 재실행 → 새 IMPOSSIBLE 제외; 미평가 전무 또는 `min LB ≥ incumbent×(1-ε)` → `"certified"` + `certificate`. h. 미평가 없음 → `"all_evaluated"`.
3. 스케줄 소진 `"k_exhausted"`; 예산 `"budget_sims"/"budget_wall"`.
4. 미평가 대표 → `Rejection(NOT_EVALUATED_BUDGET, "not reached by K schedule/budget; NOT a verdict")`. `feasible=False` reason에 "N representatives (M embeddings) were never evaluated" 명시.
5. `PlannerOutput = exhaustive._assemble_output(...)` + `provenance["graph_search"]=audit.as_provenance()` + caveat `GRAPH_SEARCH_CAVEAT`.

**테스트** `tests/test_adaptive.py`
- (i) `k_schedule=(2,4)`, 대표 6 → 평가 4, `NOT_EVALUATED_BUDGET` 2, `"k_exhausted"`.
- (ii) `max_simulations=3` → predictor `calls==3`, `"budget_sims"`.
- (iii) **미평가 ≠ 불가능**: 전 평가 후보 SLO 위반 fixture → `feasible=False`, reason에 "never evaluated", `rejected_summary["not_evaluated_budget"]` 별도.
- (iv) `CERTIFY` 가격 있는 fixture → `"certified"`, `certificate["min_unevaluated_lower_bound"] >= incumbent_cost`; 가격 결측 대표 있으면 인증 불가·스케줄 계속.
- (v) plan_id 연속·유일(`hp-00000..`), 배치 경계 충돌 없음.
- (vi) 재현성: 두 실행 `model_dump()` 동일.
- (vii) 캐시: `EnvelopeCache(graph_signature)` 두 번째 실행 `cache_hits==evaluated`.
- (viii) 그룹 분리: 잔차 큰 그룹의 미평가 대표가 다음 배치 첫 원소.
- (ix) `audit.as_provenance()`의 아홉 카운트 합 == representatives.
- (x) PD 이중 부과 방지: PD 대표의 최종 `p99_ttft_ms` == sim 원값 + embedded 전송(heteropilot 전송값이 아님) — `approx`.

**시험 절차** `pytest tests/test_adaptive.py -q && ruff check . && mypy graphsearch/`.
**완료 조건** 통과.

---

### STEP G10 — 복원과 예약 검사 (graphsearch, 1일)

**역할** 추천 대표를 물리 장치·링크로 되돌리고 용량·snapshot을 검사(설계서 §5.10).

**API** `restore(rep, plan, graph, *, occupied=frozenset(), prefer=None)`, `restore_many(reps_plans, graph, conflicts, *, count=1)`, `check_capacity`, `recheck_snapshot`. `prefer` 기본 `(cost_of_devices.total or inf, min node id)`; `shared_resource_reservations[res]=exemplar.resource_demand[res]`; 부족 시 `RestoreError("representative has multiplicity M but only N embeddings can coexist")`.

**테스트** `tests/test_restore.py`
- (i) multiplicity 4(A–B pair) 1개 복원 → `device_ids` 2개 ⊂ A∪B.
- (ii) `restore_many(count=2)` on multiplicity 16 → disjoint 2세트; `count=9` → `RestoreError`.
- (iii) uplink 10e9, 수요 6e9 → `count=2` `RestoreError`; 4e9 → `check_capacity==[]`.
- (iv) `recheck_snapshot`: reserved 바꾼 v2 사본 → 사유 1건.
- (v) `multiplicity != max_concurrent` fixture에서 두 값 다름 assert.
- (vi) `plan.predicted`가 복제되지 않고 참조만(`is`).

**시험 절차** `pytest tests/test_restore.py -q`. **금지** 대표 성능을 복수 배치에 곱하는 코드 경로.

---

### STEP G11 — MVP 어댑터, 캐시 키, 손실 보고, 경합 인터페이스 (graphsearch, 1.5일)

**역할** 대표 exemplar를 시뮬레이터 설정으로 컴파일하되 **무엇을 잃었는지** 함께 반환. H3의 `set_compile_hook`으로 heteropilot predictor에 주입(설계서 §5.9, §5.12).

**변경**
1. `graphsearch/adapter.py`: `TopologyLossReport`, `compile_embedded(emb, cluster, islands, profiles, *, graph, topology, ...)` — `compile_to_sim_config(topology_level=2)` 호출 뒤 `link_bw`/`link_latency`를 flow 경로 병목(bytes/s→GB/s)으로 덮어씀(intra=min `TP_ALLREDUCE` 첫 경로 병목, cross=min `PD_KV_TRANSFER`/`PP_ACTIVATION`; 해당 flow 없으면 유지); `dropped_shared_resources=sorted(emb.boundary.shared_resources)`; `flows_priced_analytically=[PD flow ids]`. `apply_pd_transfer_cost_embedded(emb, metrics, spec, graph)` — heteropilot 산식, 경로는 PD flow 첫 허용 경로. `bind(predictor, embeddings_by_template_id, graph)` — `predictor.set_compile_hook(closure)`; closure는 `candidate.id`가 바인딩에 없으면 `None`(기존 경로). 보고서는 closure가 `predictor`에 `last_loss_reports[candidate.id]`로 남기고 `AdaptiveSearch`가 provenance에 옮긴다.
2. `graphsearch/contention.py`: `ContentionModel` ABC(`transfer_times_ns(flows, graph, start_ns) -> Mapping[str, float]`), `NullContentionModel`(latency + bytes/bottleneck). `features_for`(G8)·`GraphAwareMockPredictor`가 이를 경유하도록 리팩터(동작 불변).
3. 캐시 서명 `graph_signature = f"{rep.signature.wl_hash}:{graph.schema_version}:{rep.signature.tool_version}"` helper.

**테스트** `tests/test_adapter.py`
- (i) 단일 섬 tp=2 대표 → config `link_bw`가 flow 병목(GB/s)과 일치, 차원 수 동일.
- (ii) **손실 보고**(설계서 §7.8): toy-shared-nic X/Y P/D 대표 → config dict 같아도 `dropped_shared_resources` 다름; `warnings`에 "not modeled".
- (iii) 바인딩에 없는 candidate → closure `None` → heteropilot 기존 컴파일 호출(monkeypatch 카운트).
- (iv) `apply_pd_transfer_cost_embedded`: 같은 경로면 heteropilot 함수와 `approx` 동일; 경로 다르면 병목 차이만.
- (v) 서명 다른 두 대표 → `EnvelopeCache` 키 다름.
- (vi) `NullContentionModel` 값 검증; ABC 미구현 → `TypeError`.
- (vii) heteropilot `tests/test_pd_transfer.py`, `tests/test_sim_pd_transfer.py`, `tests/test_topology_perdim.py`가 서브모듈 sha에서 통과(변경 없음 확인).

**시험 절차** `pytest tests/test_adapter.py -q && ruff check . && mypy graphsearch/ && (cd vendor/heteropilot && pytest tests/test_pd_transfer.py tests/test_sim_pd_transfer.py tests/test_topology_perdim.py -q)`.
**산출물** 코드 + `docs/decisions.md` **GS-4** "MVP 어댑터는 공유 자원을 전달하지 못함(heteropilot D124와 쌍); 경합 재현 실험은 `ContentionModel` 구현 이후".
**완료 조건** 통과. **금지** `vendor/heteropilot` 수정.

---

### STEP G12 — 완전탐색 oracle 하네스와 통합 불변식 (graphsearch, 1.5일)

**역할** 압축·제거·적응형 결과를 "전 임베딩 완전 평가"와 같은 predictor로 비교(설계서 §5.13). 논문 표(연구설계서 §12 1·3행)의 데이터 생산자.

**API** `run_oracle(spec, cluster, islands, profiles, predictor, *, graph, policy=EmbeddingPolicy())`(전 임베딩, 압축·하한·top-K 없음, `EmbeddedCandidate.id` 키), `run_proposed(...)`, `compare(oracle, proposed, reps, spec) -> OracleComparison`.
`feasible_recall` = 제안이 feasible로 판정한 임베딩(대표 전개) ∩ oracle feasible / oracle feasible; `cost_regret` = (제안 최선 − oracle 최선)/|oracle 최선|; `false_infeasible` = IMPOSSIBLE_PROVEN인데 oracle feasible인 임베딩 수(**0이어야 함 — 하한 버그**); `mismerged_pairs` = 같은 대표 두 임베딩이 oracle에서 다른 판정 또는 목적값(`rel=1e-9`)(**0이어야 함 — 동등성 버그**).

**테스트** `tests/test_oracle_agreement.py`
- (i) toy-abcde: `false_infeasible==0`, `mismerged_pairs==[]`, K≥대표 수면 `cost_regret==0`, `feasible_recall==1`.
- (ii) toy-shared-nic: 동일 + `include_boundary=False`에서는 `mismerged_pairs` **비어 있지 않음**.
- (iii) `audit.simulations_run < embeddings` and `== 평가된 representatives`.
- (iv) toy-asym: `representatives==embeddings`(압축률 0)이어도 (i) 정확성 유지.
- (v) heterogeneous-lab(v1)에서 heteropilot `oracle()` 추천 template id == 그래프 oracle 최선 임베딩의 template id.
- (vi) `experiments/scripts/graph_oracle_compare.py --predictor mock --out experiments/results/graph_oracle_toy.md`가 표 생성(열: fixture, embeddings, representatives, exact_merges, vf2_seconds, sims_run, feasible_recall, cost_regret, false_infeasible, mismerged). 수치는 실행 결과 그대로.

**시험 절차** `pytest -q`(전체) ; 위 스크립트 실행.
**완료 조건** 통과, md 생성. 0이 아닌 `false_infeasible`/`mismerged`는 해당 하한 `demoted` 또는 라벨 규칙 수정으로 해결 — **테스트를 완화하지 않는다**.

---

### STEP G13 — CLI·렌더·문서·E-G1 파일럿 (graphsearch, 1.5일)

**CLI** `graphsearch/__main__.py` — heteropilot `planner/__main__.py`는 **수정하지 않는다**.
```
python -m graphsearch plan --service … --cluster … [--profiles-root …] --output out.yaml
    --k-schedule 4,8,16  --search-mode budget|certify  --budget-sims N  --budget-seconds S  --epsilon 0.0
    --max-embeddings-per-template N  --compression exact|off  --ranker service_margin|binned|roofline
    --bounds all|none|<comma list>   (none은 하한만 끔; compat·memory 정확 검사는 항상)
    --oracle          압축·제거·적응 끄고 전 임베딩 평가(G12 run_oracle)
    --predictor sim|mock   (mock은 GraphAwareMockPredictor; 실험·시연용, 결과에 "MOCK" 배너)
    --cache-dir …  --num-requests … --seed …   (heteropilot plan과 동일 의미로 전달)
python -m graphsearch oracle …   /   python -m graphsearch compare --oracle a.yaml --proposed b.yaml
```
`--predictor sim`은 `LLMServingSimPredictor`를 heteropilot과 같은 인자로 만들고(`vendor/heteropilot/.venv` 안내), `adapter.bind`로 훅 등록.
**렌더** `graphsearch/render.py::render_graph_block(audit, compression, restored) -> str`; `planner.render`의 텍스트 뒤에 덧붙임:
```
Graph search: 128 templates -> 412 embeddings -> 37 representatives (exact merges 375, hash-only 0, VF2 1.8 s)
  impossible_proven 9 | excluded_by_scope 0 | deferred/not-evaluated 12 | unknown_measurement 0 | evaluated 16
  simulations 16 (cache hits 0) | K reached 16 | termination: k_exhausted
  certificate: none (budget mode)     restored: A/gpu0, A/gpu1 ; reservations uplink-A 3.2e9 B/s
```
YAML: `provenance.graph_search`, `.compression`, `.restored`, `.topology_loss`.
**문서** `README.md` 실행 예시·플래그 표; `CLAUDE.md`(graphsearch) 갱신; `docs/decisions.md` GS-5 "CLI는 heteropilot CLI를 감싸지 않고 함수 레벨(`search` 부품)로 호출".
**E-G1 파일럿(CPU, mock)** `experiments/scripts/e_g1_toy_pilot.py`: {toy-abcde, toy-shared-nic, toy-asym, heteropilot heterogeneous-lab(v1)} × {oracle, heteropilot `search(BinnedRooflineRanker, top_k∈{4,8,16})`, `AdaptiveSearch(service_margin, (4,8,16))`} → `experiments/results/e_g1_toy_pilot.md`(압축률, sim 수, regret, recall, false_infeasible, wall). 머리에 "MockPredictor 결과, 성능 수치 아님". heteropilot `docs/CLAIMS.md`에는 넣지 않는다.

**테스트** `tests/test_cli.py`
- (i) `plan --predictor mock --k-schedule 2,4` toy-abcde → YAML `provenance.graph_search` 존재, stdout "Graph search:" 블록.
- (ii) `--oracle` → caveat oracle 문구, `compression.exact_merges==0`.
- (iii) `--compression off` → representatives==embeddings.
- (iv) `--search-mode certify` 가격 없는 클러스터 → caveat "certificate impossible: price missing", 예외 없음.
- (v) `--bounds none` → `impossible_proven`은 compat·memory 거절만.
- (vi) `--predictor sim`은 시뮬레이터 없는 CI에서 명확한 오류 메시지로 종료(exit 2, "vendor/heteropilot/.venv not found").

**시험 절차** `pytest -q && ruff check . && mypy graphsearch/ && python experiments/scripts/e_g1_toy_pilot.py --out experiments/results/e_g1_toy_pilot.md`.
**완료 조건** 통과, md 생성, README/CLAUDE.md/decisions 갱신.

---

## 5. 통합 시험 절차 (G12·G13 이후, 릴리스 태그 전 1회)

1. graphsearch: `pytest -q && ruff check . && mypy graphsearch/`.
2. 서브모듈 건강: `(cd vendor/heteropilot && git status --porcelain)`가 빈 출력(수정 없음) 이고 sha == H3 머지 sha; `(cd vendor/heteropilot && pytest -q)`(시뮬레이터 있는 노드) 또는 CI 부분집합.
3. 재현성: `e_g1_toy_pilot.py` 두 번 → wall 열 제외 diff 없음.
4. 불변식 표(설계서 §7) 9개 테스트 `-k "relabel or shared_uplink or role_swap or relaxation or proof or capacity or multiplicity or unevaluated or reproducible or units or loss_report or stage4"` 한 번에 통과.
5. heteropilot: D120~D124 존재, `tests/test_deviations_numbering.py`·`tests/test_experiment_ids.py` 통과, `CLAUDE.md` 표에 분리 저장소 한 줄.
6. graphsearch `docs/decisions.md` GS-1~GS-5.

## 6. 이 지시서가 하지 않는 것

실측(연구설계서 §12 2단계; 별도 지시서 `E-G2`~) · `ContentionModel` flow 기반 구현 · 전력 최소화 목적 · GPU–NPU 혼합 TP · 학습 랭커 · PP 다단 생성기 확장 · 상류 `serving/` 수정 · heteropilot CLI 변경.

## 7. 수용 기준 요약 (논문 1차 게이트)

| 기준 | 측정 | 판정 |
|---|---|---|
| 압축 정확성 | `mismerged_pairs`(G12) | 0 (toy 3종) |
| 제거 안전성 | `false_infeasible`(G12), 완화 property(G7) | 0 |
| 미평가 분리 | `not_evaluated_budget`/`excluded_by_scope`가 infeasible 계수와 분리 | G9 (iii) |
| heteropilot 기본 경로 불변 | H1~H3 각 PR golden 바이트 동일 | 통과 |
| 절감 구조 | `simulations_run ≤ representatives < embeddings` on toy-abcde | 통과(수치 기록) |
| 재현성 | 두 실행 diff 없음 | 통과 |
| 저장소 경계 | graphsearch→planner 단방향; `vendor/heteropilot` 무수정 | `git status` 빈 출력 |
