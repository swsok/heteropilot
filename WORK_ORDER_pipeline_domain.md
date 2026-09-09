# HeteroPilot 작업지시서 — 파이프라인 도메인과 아일랜드 간 파이프라인 병렬

> 특허 1 명세서 개정(2026-09-04)에 맞추어, 텐서 병렬 경계(실행 아일랜드)와 파이프라인 병렬 경계(파이프라인 도메인)를 분리하고 아일랜드 간 PP 후보를 플래너에 추가하기 위한 작업지시서.
> 대상 저장소: `github.com/swsok/heteropilot` (upstream: `casys-kaist/LLMServingSim`, 핀 `2c2042ce`)
> 작성일: 2026-09-04 · 작업 도구: Claude Code

---

## 0. 이 문서의 사용법

1. **STEP 순서를 반드시 지킬 것.** 각 STEP은 앞 STEP의 산출물에 의존한다.
2. **한 STEP = 한 브랜치 = 한 PR.** 브랜치 이름은 `feat/pp-stepNN-<slug>`.
3. **각 STEP의 "테스트" 절을 통과하지 못하면 다음 STEP으로 넘어가지 않는다.** 테스트를 약화시켜 통과시키는 것은 금지.
4. **"조사 필요"로 표시된 항목은 추측하지 말고 실제 코드·산출물을 읽어 확인**하고 결과를 PR 설명과 `docs/deviations.md`에 적을 것.
5. 이 문서와 `WORK_ORDER_heteropilot.md`, `CLAUDE.md`, `AGENTS.md`가 충돌하면 **기존 문서가 우선**한다. 충돌 발견 시 즉시 중단하고 사용자에게 보고.

### 절대 규칙 (기존 규칙 재확인 + 이 작업의 추가 규칙)

- **A1. 프루닝 단계는 실행가능성 검사의 완화(relaxation)여야 한다.** 새 하한 S4′는 계산 시간 0·병목 대역폭 가정의 하한이어야 하고, §5.6이 선언하지 않은 제약(처리량 등)을 조건으로 쓰면 안 된다. 오라클 일치 테스트가 이를 강제한다.
- **A2. `planner/optimizer/exhaustive.py`를 삭제하지 않는다.** PP 후보가 늘어나 전수 탐색이 느려져도 오라클은 유지한다(픽스처를 작게 할 것).
- **A3. 하드웨어 숫자를 만들어내지 않는다.** 노드 간 링크(InfiniBand/이더넷) 대역폭·지연이 클러스터 명세에 없으면 인터커넥트 클래스 기본값을 쓰고 `assumptions`에 기록한다. 실측값처럼 표기 금지.
- **A4. 기본 동작을 바꾸지 않는다.** PP 열거는 **opt-in**(`ServiceSpec.parallelism.max_pp` 기본 1). 기존 golden/재현성 테스트 출력은 바이트 동일해야 한다.
- **A5. `serving/` 업스트림 코드는 STEP 6까지 수정하지 않는다.** 시뮬레이터가 표현하지 못하는 것(비대칭 계층 분할, 단계별 상이 하드웨어 라벨)은 플래너 측 근사로 처리하고 deviations에 기록한다. 업스트림 수정은 STEP 7에서만, 사용자 승인 후.
- **A6. 코드 주석·docstring·로그는 영어.** 이 지시서는 한국어지만 산출 코드는 아니다.

---

## 1. 배경과 목표

### 1.1 문제

현재 플래너는 `IslandAssignment.pp_size = 1`로 고정되어 있고(`planner/plan.py:85`), 후보 생성부는 TP × DP × 노브만 열거한다. `CLAUDE.md`와 `WORK_ORDER_heteropilot.md`는 "TP/PP는 아일랜드 내부에서만"이라고 적었으나, 이는 TP에는 옳고 PP에는 불필요하게 강한 제약이다.

- vLLM v0.28.0(2026-08-26) 공식 문서는 멀티노드에서 **TP = 노드당 GPU 수, PP = 노드 수**를 권장한다. 즉 PP는 노드 간 네트워크(InfiniBand 등)를 넘는 것이 정상 사용법이다.
- PP 단계 경계는 스텝당 `토큰 수 × hidden × dtype` 바이트만 전송한다(Llama-3.1-8B bf16, 디코드 64 시퀀스: 512 KB; 4K 프리필 청크: 32 MB). TP는 계층마다 all-reduce 2회이므로 대역폭 요구가 두 자릿수 높다. **느린 링크에서 TP는 거절되지만 PP는 실용적**인 영역이 존재한다.
- 업스트림 시뮬레이터는 `pp_size`를 지원한다(`configs/cluster/*.json`의 `instances[i].pp_size`, `num_npus == tp_size * pp_size`; 단계 경계는 Chakra `COMM_SEND/RECV`로 모델, `inflight` 큐 깊이 = `pp_size`). **조사 확인 사항**: 계층 분할은 균등(`num_layers // pp_size`)만, 인스턴스당 하드웨어 라벨은 하나, 마이크로배치 없음, 스케줄(1F1B 등) 선택 불가 (`docs/docs/simulator/parallelism-mechanics.md`).
- vLLM 제약(조사 확인): 모든 PP 단계에 **동일 TP** 필요(Issue #27239, not planned), 노드 간 **동일 실행 환경**(모델 경로·패키지), 이기종 GPU 세대 혼합은 비공식(소스 빌드 `TORCH_CUDA_ARCH_LIST`, `NCCL_P2P_DISABLE=1`, `VLLM_PP_LAYER_PARTITION`으로 비대칭 분할; Issue #34437은 FP8에서 수치 불일치).

### 1.2 목표

1. **파이프라인 도메인**(pipeline domain) 개념 도입: 런타임 백엔드가 같고 임의 링크 클래스로 도달 가능한 아일랜드 집합. 정책 `strict`(장치 모델까지 동일) / `backend_only`.
2. **후보 유형 (d) 아일랜드 간 파이프라인** 열거: 도메인 내 아일랜드 순서열에 PP 단계 할당, 단계 간 동일 TP(교집합에서 선택), 균등 계층 분할 기본 + 용량 비례 비대칭 분할 옵션.
3. **하한 S4′**(PP 단계 경계 전송 지연 하한) 추가 — 완화 조건 유지, 프리필 역할 면제, 오라클 일치 검증.
4. 컴파일부에서 `pp_size`·단계 간 링크 값 방출, 시뮬레이터 표현력 한계를 deviations로 기록·근사.
5. 배포부(vLLM CUDA)에서 멀티노드 PP 실행 명령 생성.
6. 실험: (E-PP1) 8×A40 노드에서 TP=4 vs PP=2×TP=2(NVLink 쌍 간 PCIe) 비교, (E-PP2) 2노드 InfiniBand 픽스처, (E-PP3) 프루닝 실효성 — 빡빡한 TPOT SLO에서 S4가 PCIe TP를 거절하고 S4′가 PP를 통과시키는지 + 오라클 일치.

### 1.3 만들지 않는 것

- PP 단계별 상이 TP (vLLM 미지원). 열거 금지, `rejected_summary`에 "uniform TP required" 사유로 기록만.
- 벤더 간(백엔드 상이) PP. 도메인 정의에서 구조적으로 배제.
- 시뮬레이터 마이크로배치/1F1B 스케줄 모델링 변경 (업스트림 범위).
- 온라인 재계획·라이브 마이그레이션.

---

## 2. 설계 개요

### 2.1 데이터 흐름

```
ClusterSpecV2 ──▶ detect_islands()  ──▶ detect_pipeline_domains(policy)
                                              │
ServiceSpec.parallelism.max_pp ───────────────┤
                                              ▼
                 candidate_generator: (a)(b)(c) + (d) pipeline candidates
                                              │  stages=[(island, tp, layer_range)], pp_size
                                              ▼
                 pruning: S1 S2(stage-wise weights) S3 S5' | S4  S4'(new)  S5 | S6 | S7
                                              ▼
                 compiler: pp_size, num_npus=tp*pp, inter-stage link dims, assumptions
                                              ▼
                 simulator (pp_size, equal partition) ──▶ post-hoc partition scaling (if uneven)
                                              ▼
                 feasibility → lexicographic rank → Pareto → diagnosis (unchanged)
```

### 2.2 스키마 변경 (요약, STEP 1에서 확정)

| 위치 | 필드 | 의미 |
| --- | --- | --- |
| `ServiceSpec.parallelism` (신규 섹션) | `max_pp: int = 1`, `allow_uneven_partition: bool = False` | PP 열거 상한; 비대칭 분할 후보 생성 여부 |
| `ClusterSpecV2.planning` (신규 또는 기존 policy 블록) | `pipeline_domain_policy: Literal["strict","backend_only"] = "strict"` | 도메인 구성 정책 |
| `planner/inventory.py` | `PipelineDomain(id, backend, island_ids, policy, link_class_between)` | 도메인 객체 |
| `IslandAssignment` | `pp_size: int`, `pp_stage: int | None`, `layer_range: tuple[int,int] | None` | 단계 인덱스·계층 범위 |
| `CandidateConfig` | `kind` 에 `"pipeline"` 추가, `stages: list[IslandAssignment]` | 후보 유형 |
| `RejectionStage` | `PIPELINE_BOUNDARY_BOUND` (S4′), `PIPELINE_UNIFORM_TP` (열거 불가 사유) | 거절 원장 |
| `TopologyReduction` | `pp_boundary: list[{stage_i, bw_gbps, lat_ns, source}]` | 단계 경계 링크 축약 기록 |

### 2.3 S4′ 하한 공식 (STEP 4에서 구현)

```
active        = min(max_num_seqs, kv_capacity)            # S5와 동일
payload_bytes = active × hidden_size × bytes_per_element
per_boundary_i = lat_ns_i + payload_bytes / bw_bytes_per_s_i
floor_ms      = Σ_{i=1..PP-1} per_boundary_i / 1e6
reject if floor_ms > slo.tpot.max_ms      (Role.PREFILL 은 면제)
```

`bw_i`, `lat_i`는 단계 i−1 아일랜드 ↔ 단계 i 아일랜드 경로의 병목 값(`topology.island_path()`; 없으면 클래스 기본값 + assumption). 이 값은 S4(all-reduce)의 `2·L·(…)`와 달리 계층 수에 곱해지지 않는다는 점이 핵심이며, 테스트로 두 하한의 순서 관계를 고정한다.

---

## 3. 사전 조사 결과 (이미 있는 것 / 없는 것)

| 항목 | 상태 | 위치 |
| --- | --- | --- |
| 시뮬레이터 `pp_size` 지원 | 있음 (균등 분할, 단일 hardware 라벨) | `serving/core/config_builder.py:56-96`, `docs/docs/simulator/parallelism-mechanics.md` |
| `IslandAssignment.pp_size` | 필드는 있으나 항상 1 | `planner/plan.py:85-90` |
| 아일랜드 간 경로·대역폭 축약 | Level 1/2 구현 (P/D용) | `planner/topology.py:266-367`, `island_path()` |
| 노드 간 링크 클래스 (`InfiniBand`, `Ethernet`) | 스키마에 있음, 아일랜드 병합엔 미사용 | `planner/inventory.py` `INTRA_ISLAND_LINK_KINDS` |
| 오라클 일치 테스트 | 있음 | `tests/test_search.py` |
| 모의 예측기 물리 모델 | 가중치·KV·대역폭 기반; PP p2p 항 없음 → STEP 4에서 추가 | `tests/conftest.py` MockPredictor (조사 필요: 정확한 위치 확인) |
| vLLM 배포 백엔드 | 단일 노드 CUDA만 | `planner/deploy/vllm_cuda.py` |
| `VLLM_PP_LAYER_PARTITION` | vLLM 환경 변수, 단계별 계층 수 콤마 목록 | 조사 필요: v0.28.0에서 유효한지 문서/코드 확인 후 PR에 기록 |

---

## 4. 공통 규칙

### 4.1 Git
- 브랜치 `feat/pp-step00-baseline` … `feat/pp-step08-docs`. `main`에 직접 커밋 금지.
- 커밋 메시지 접두 `pp-stepNN:`. 각 PR 설명에 "조사 확인" 결과와 deviations 번호를 적는다.

### 4.2 테스트 게이트 (모든 STEP 공통)
```bash
pytest                      # 전체 통과
ruff check .
mypy planner/
pytest tests/test_search.py::test_oracle_agreement -x   # 프루닝 소건성
pytest tests/test_render.py tests/test_optimizer.py      # golden/재현성 (바이트 동일)
```
STEP 3 이후에는 `--max-pp 2`를 켠 PP 픽스처에서도 오라클 일치·재현성 테스트가 통과해야 한다.

### 4.3 테스트 작성 원칙
- 하한 테스트는 "하한 ≤ 모의 예측기 값"을 **모든 후보**에 대해 검사한다(단조성 검사).
- PP 픽스처는 작게: 4 GPU(NVLink 2쌍) + 노드 간 링크 1개 정도. 오라클 전수 탐색이 1분 내에 끝나야 한다.
- 기본값(`max_pp=1`)에서 기존 테스트 산출물이 바이트 동일함을 별도 테스트로 고정한다.

---

# STEP 0. 준비 — baseline 고정과 조사 기록

## 목표
PP 도입 전 상태를 고정하고, 시뮬레이터·vLLM의 PP 표현력을 deviations에 기록한다.

## 지시
1. `docs/consolidation_baseline.md` 방식으로 현재 `plan` golden 출력(두 examples 스펙)의 해시를 `docs/pp_baseline.md`에 기록.
2. `docs/deviations.md`에 **D25** 추가: "시뮬레이터 PP는 균등 계층 분할·단일 hardware 라벨·마이크로배치 없음. 플래너는 비대칭 분할을 후처리 스케일로 근사하고, 상이 하드웨어 단계는 가장 느린 단계 라벨로 보수 시뮬레이션한다." 근거 파일 경로 인용.
3. **D26** 추가: "vLLM은 PP 단계 간 동일 TP·동일 실행 환경을 요구(공식 문서 v0.28.0, Issue #27239). 도메인 정책 `backend_only`는 비공식 지원 영역이며 배포부는 경고를 출력한다." 출처 URL 기록.
4. `CLAUDE.md`의 "TP/PP are permitted only within an island" 문장을 "TP only within an island; PP may span islands inside one pipeline domain (this work order)"로 수정. `WORK_ORDER_heteropilot.md`는 수정하지 않고 본 문서가 보완함을 각주로 명시.

## 테스트
- 기존 전체 테스트 통과. 변경은 문서만.

## 완료 조건
- D25, D26 기록. baseline 해시 파일 커밋.

---

# STEP 1. 스키마 — `max_pp`, 도메인 정책, 단계 필드

## 목표
§2.2의 스키마 변경을 pydantic 모델에 추가한다. 기본값에서 동작 불변.

## 지시
1. `planner/spec.py`: `ParallelismSpec(max_pp: int = 1 (ge=1, le=8), allow_uneven_partition: bool = False)`를 `ServiceSpec.parallelism`(optional, 기본 인스턴스)로 추가.
2. `planner/inventory.py`: `ClusterSpecV2`(또는 planning 정책 블록)에 `pipeline_domain_policy: Literal["strict","backend_only"] = "strict"`.
3. `planner/plan.py`: `IslandAssignment`에 `pp_stage: int | None = None`, `layer_range: tuple[int,int] | None = None` 추가. `num_devices` 프로퍼티가 `tp_size * pp_size`인지 확인(조사 필요) — 단계별 할당에서는 각 단계가 `tp_size`개 장치를 쓰므로 `CandidateConfig.total_devices = Σ_stage tp_size × dp`로 재정의.
4. `CandidateConfig.kind`에 `"pipeline"` 추가. YAML 직렬화 순서 결정성 유지.

## 테스트
- `tests/test_spec.py`: `max_pp` 기본 1, 범위 검증, `allow_uneven_partition` 기본 False.
- `tests/test_inventory.py`: 정책 기본 `strict`, 잘못된 값 거절.
- golden: examples 두 스펙의 `plan` 출력 바이트 동일.

## 완료 조건
- 스키마 추가, 기본값 불변, mypy 통과.

---

# STEP 2. 파이프라인 도메인 검출

## 목표
`detect_pipeline_domains(islands, cluster, policy) -> list[PipelineDomain]` 구현.

## 지시
1. `planner/inventory.py`에 `PipelineDomain` dataclass/pydantic 모델: `id`, `backend`, `island_ids: list[str]`(결정적 정렬), `policy`, `edges: list[(island_a, island_b, link_kind, bw_gbps|None, lat_ns|None)]`.
2. 알고리즘: 아일랜드를 정점으로, **모든 링크 클래스**(intra 클래스 + `InfiniBand`/`Ethernet` 등 노드 간 클래스)를 간선으로 하는 그래프에서 연결요소. 정점 분할 키는 `strict`: `(backend, model_slug)`, `backend_only`: `(backend,)`.
3. 노드 간 링크가 클러스터 명세에 없으면 **간선을 만들지 않는다**(A3). 단, 동일 노드의 아일랜드들은 PCIe 루트 컴플렉스 경유로 도달 가능하다고 보고 `link_kind="PCIe"`(클래스 기본값)로 간선을 만들되 `source="class_default"`로 표시.
4. 도메인 id: `pd-{backend}-{index}`; 크기 1 도메인도 생성(PP 후보는 없지만 출력 일관성).
5. `inspect-cluster` CLI 출력에 도메인 표를 추가(아일랜드 표 아래).

## 테스트
- `tests/test_pipeline_domain.py` (신규):
  - 도 2 픽스처(노드0: NVLink 쌍×2 + PCIe 단독 + NPU 8PE, 노드1: PCIe×4, InfiniBand): `strict` → D1={I1,I2,I3}, D2={I5}, D3={I4}; `backend_only` → D1={I1,I2,I3,I5}, D2={I4}.
  - 노드 간 링크가 명세에 없으면 노드 간 병합 없음.
  - 백엔드가 다르면 어떤 정책에서도 병합 없음.
  - 결정성: 두 번 실행 결과 동일, id 정렬.

## 완료 조건
- `inspect-cluster`가 도메인을 출력하고 테스트 통과.

---

# STEP 3. 후보 유형 (d) — 아일랜드 간 파이프라인 열거

## 목표
`max_pp ≥ 2`일 때 도메인 내 아일랜드 순서열에 PP 단계를 할당한 후보를 열거한다.

## 지시
1. `planner/candidate_generator.py`에 `_pipeline_candidates(domain)`:
   - `pp ∈ {2..min(max_pp, |domain|)}`; 도메인 아일랜드의 **순서 있는** `pp`-순열을 열거(순서열 (I1,I5) ≠ (I5,I1)). 조합 폭발 방지: `|domain| > 6`이면 아일랜드를 (모델, 크기) 서명으로 묶고 서명 동치 순열은 대표 하나만(결정적).
   - `tp ∈ ∩_stage tp_candidates(I_stage)`; 교집합이 비면 `RejectionStage.PIPELINE_UNIFORM_TP`로 기록하고 건너뜀.
   - `dp ∈ {1 .. min_stage(|I_stage| // tp)}`.
   - 계층 분할: 기본 균등 `layers_per_stage = L // pp`(나머지는 마지막 단계). `allow_uneven_partition`이면 단계 아일랜드의 derated 메모리 용량 비례 분할을 **추가 후보**로 생성(정수 반올림, 합 = L, 각 단계 ≥ 1).
   - 각 단계를 `IslandAssignment(role=AGGREGATED, tp, pp, pp_stage=i, layer_range, dp)`로.
2. 역할 결합: `(c) P/D`의 프리필 또는 디코드 측에 파이프라인 후보를 두는 것은 **이 STEP에서는 제외**(D14 `tp_p == tp_d` 제약과의 상호작용을 STEP 8 이후 별도 지시서로). `rejected_summary`에 사유 없이 단순히 열거하지 않음을 docstring에 명시.
3. 정확 단계 적용:
   - S1: 모든 단계 아일랜드가 호환.
   - S2: 단계별 가중치 = 해당 `layer_range`의 계층 가중치 + (단계 0: 임베딩, 마지막 단계: lm_head/norm). `planner/util/memory.py`에 `weight_bytes_for_layers(model, layer_range, tp)` 추가 — 시뮬레이터 메모리 모델 산식과 정합(조사 필요: `serving/core/memory_model.py`의 계층별 분해 가능 여부; 불가하면 계층 비례 근사 + 주석).
   - S3: 각 단계 `|I| % tp == 0`.
   - S5′: KV 용량은 단계별로 계산하되 **최소값**을 후보의 KV 용량으로.
4. 후보 id에 `pp{pp}-s{stage_signature}`를 포함하여 결정성 유지.

## 테스트
- `tests/test_candidates_pp.py`:
  - `max_pp=1`(기본)이면 pipeline 후보 0개, 기존 후보 집합 바이트 동일.
  - 도 2 픽스처 `backend_only`, `max_pp=2`: (I1,I5),(I5,I1),(I1,I2),… 열거, tp ∈ {1,2}(I5 {1,2,4}와 I1 {1,2}의 교집합).
  - 교집합 공집합 케이스 → `PIPELINE_UNIFORM_TP` 사유 기록.
  - 비대칭 분할: 24 GB + 48 GB 단계 → 계층 1:2 비율, 합 = L.
  - 재현성: 동일 입력 두 번 → 동일 후보 목록.

## 완료 조건
- PP 후보 열거·정확 단계 통과, 기본 동작 불변.

---

# STEP 4. 하한 S4′와 오라클 일치

## 목표
§2.3의 S4′를 추가하고, 완화성을 오라클 일치와 단조성 테스트로 검증한다.

## 지시
1. `candidate_generator._stage4p_pipeline_boundary_ok(cand)`:
   - `Role.PREFILL` 면제, `pp == 1` 통과.
   - 단계 쌍마다 `topology.island_path(I_{i-1}, I_i)` → `(bw_gbps, lat_ns, source)`. 경로 없음 → 클래스 기본값(`InfiniBand`/`PCIe`) + `assumptions` 기록. `bw == inf` → 통과.
   - 공식 §2.3. 거절 사유 문자열에 단계 쌍·bw·lat·floor 포함.
2. 순서: S4 → **S4′** → S5 (도 5). `enable_bound_pruning=False`(오라클)에서 S4′도 비활성.
3. **모의 예측기 갱신**(조사 필요: 위치): PP 후보의 TPOT 예측에 단계 경계 항 `Σ (lat + payload/bw)`를 **하한보다 크게**(예: ×1.2 + 계산 시간) 더한다. 하한 위반 후보가 모의 예측기에서 실행가능으로 되돌아오지 않도록.
4. `docs/deviations.md`에 S4와 S4′의 관계 메모: S4는 `2·L`배, S4′는 `(PP−1)`배 — 같은 링크에서 S4′ ≤ S4임을 수식으로.

## 테스트
- `tests/test_search.py::test_oracle_agreement_pp`: PP 픽스처(4 GPU NVLink 2쌍 + PCIe 링크 5 GB/s, TPOT SLO 20 ms)에서 프루닝 on/off 추천·목적함수 일치.
- `tests/test_bounds_pp.py`:
  - 모든 후보에 대해 `floor_S4p ≤ mock_TPOT`.
  - 같은 링크·같은 tp에서 `floor_S4p(pp=2) < floor_S4(tp=2 over that link)` (계층 수 ≥ 2일 때).
  - 프리필 역할 면제.
  - 링크 미선언 시 assumption 기록 및 기본값 사용.
- golden 바이트 동일(기본값).

## 완료 조건
- 오라클 일치 통과, 단조성 통과, 거절 원장에 `PIPELINE_BOUNDARY_BOUND` 집계.

---

# STEP 5. 컴파일부 — `pp_size` 방출과 표현력 근사

## 목표
PP 후보를 시뮬레이터 설정으로 변환하고, 표현 불가 항목을 기록·근사한다.

## 지시
1. `planner/predictor/llmservingsim.py`: 인스턴스에 `pp_size`, `num_npus = tp × pp`, `tp_size` 방출. 단계 경계 링크 값을 인스턴스 네트워크 차원에 기록 — Level 2가 있으면 `pp` 차원 추가, 없으면 스칼라 `link_bw`에 병목 `min(intra_tp_bw, boundary_bw)`를 쓰고 `assumptions`에 "pp boundary bw folded into scalar link_bw" 기록. **조사 필요**: 시뮬레이터 네트워크 설정에서 단계 경계 p2p가 어느 링크 파라미터를 읽는지(`configs/network/*`, Chakra `COMM_SEND` 크기) 확인 후 PR에 기록.
2. 단계 하드웨어 라벨이 상이한 경우(`backend_only`): 가장 느린 단계(메모리 대역폭 최소)의 라벨로 시뮬레이션, `assumptions`에 "heterogeneous stages simulated with slowest label <X>" 기록. 메모리 크기는 단계 최소 derated 값.
3. 비대칭 분할: 시뮬레이터는 균등만 지원하므로 균등으로 시뮬레이션한 뒤 후처리로 단계 시간 스케일 — 조사 필요: 시뮬레이터 출력에서 단계별 시간 분해가 가능한지. 불가하면 TPOT·TTFT에 `max_stage_layers / (L/pp)` 배율을 곱한 **보수적** 근사를 적용하고 `assumptions`에 기록, 결과에 `approximation: uneven_partition_scaled` 플래그.
4. `PlannerOutput.provenance`에 도메인 정책, PP 후보 수, 근사 플래그 수 기록. `render.py` 배너: 근사가 적용된 추천 계획에는 경고 1줄.

## 테스트
- `tests/test_compile_pp.py`: `pp_size`·`num_npus` 정합, `num_npus == tp*pp`(업스트림 `config_builder` 제약), assumptions 기록, 상이 라벨 → 느린 라벨 선택.
- 실제 시뮬레이터 스모크: `python -m serving`을 PP=2 설정으로 20요청 실행해 정상 종료·결과 파싱(A40 프로파일 사용, 노드 무관).

## 완료 조건
- PP 후보가 실제 시뮬레이터로 평가되고 결과가 판정부까지 흐른다.

---

# STEP 6. 배포부 — 멀티노드 PP 실행 명령

## 목표
`VllmCudaBackend`가 아일랜드 간 PP 계획을 멀티노드 vLLM 실행으로 변환한다. (실제 실행은 하드웨어 가용 시)

## 지시
1. `planner/deploy/vllm_cuda.py`: 계획의 단계 아일랜드가 서로 다른 노드에 있으면 `multi_node=True`. 헤드 노드 명령(`--tensor-parallel-size tp --pipeline-parallel-size pp`, `--distributed-executor-backend {ray|mp}`)과 워커 노드 조인 명령(vLLM `run_cluster.sh` 상당)을 생성. 각 노드의 `CUDA_VISIBLE_DEVICES`는 단계 아일랜드 구성원.
2. 비대칭 분할이면 `VLLM_PP_LAYER_PARTITION="l0,l1,…"` 환경 변수 방출 (조사 필요: v0.28.0 유효성; 무효하면 균등 분할 계획만 배포 가능하도록 거절 + 사유).
3. `backend_only` 도메인의 상이 GPU 세대 계획에는 경고와 권고 옵션(`TORCH_CUDA_ARCH_LIST`, `NCCL_P2P_DISABLE=1`)을 배포 노트에 기록. 기본은 **배포 거절**(`--allow-heterogeneous-pp` 플래그로만 허용).
4. `deploy` dry-run 출력에 노드별 명령을 표로.

## 테스트
- `tests/test_deploy.py` 확장: 2노드 PP 계획 → 헤드/워커 명령 생성, 환경 변수, 플래그 없이 이기종 세대 → 거절.
- 실제 배포는 A40 노드 접근 시 `docs/a40_live_deploy_loop.md` 절차로 1회 검증(선택).

## 완료 조건
- dry-run 명령 생성 테스트 통과.

---

# STEP 7. 실험 — PP 후보의 효과와 프루닝 실효성

## 목표
특허 명세서 §6(효과)에 인용할 수치를 커밋 산출물로 남긴다.

## 지시
1. **E-PP1 (단일 노드, A40 프로파일)**: `experiments/configs/clusters/a40x8-pp.yaml` — NVLink 쌍 4개(I1..I4), 쌍 간 PCIe(측정값 있으면 사용, 없으면 클래스 기본값+assumption). 후보: `agg tp4`(불가, NVLink 쌍 넘음 → 실제로는 PCIe 아일랜드가 아니므로 열거 안 됨을 확인), `agg tp2`, `pp2×tp2 (I1→I2)`, `pp4×tp2`. Llama-3.1-8B, 300요청, 시드 42. 표: p99 TTFT/TPOT, goodput, tok/J, 활성 GPU 수.
2. **E-PP2 (2노드 픽스처)**: 노드0 A40×4(NVLink 쌍 2) + 노드1 A5000×2(PCIe) + InfiniBand 100 Gb/s(클래스 기본값, assumption). `strict` vs `backend_only` 도메인에서 후보 수·추천 차이. 상이 라벨 근사(느린 라벨) 플래그 확인.
3. **E-PP3 (프루닝 실효성 + 오라클 일치)**: TPOT SLO를 {50, 30, 20, 15} ms로 스윕. 각 점에서 S4 거절 수(PCIe TP), S4′ 거절 수, S5 거절 수, 시뮬레이션 횟수 절감률, 프루닝 on/off 추천 일치 여부. **기대**: 느린 링크 구간에서 S4는 TP 후보를 거절하고 S4′는 PP 후보를 통과시킴; 일치는 전 구간 유지.
4. 스크립트 `experiments/scripts/exp_pp.py`, 결과 `experiments/results/exp_pp_{1,2,3}.{json,md}`, 그림 `experiments/figures/pp_*.png`(make_figures.py에 추가). 모든 결과에 provenance(§3.8) 포함.

## 테스트
- `tests/test_experiments_pp.py`: 스크립트가 mock predictor로 end-to-end 실행되고 결과 JSON 스키마 검증.

## 완료 조건
- 세 실험 결과·표·그림 커밋. `docs/PROJECT_REPORT.md`에 §4.9 "아일랜드 간 파이프라인" 추가(측정/시뮬 라벨 준수).

---

# STEP 8. 문서·명세서 동기화

## 목표
코드와 특허 명세서·핸드오버가 일치하게 한다.

## 지시
1. `docs/HANDOVER.md`에 PP 작업 상태·트랩(예: 노드 간 링크 미선언 시 기본값, 상이 라벨 근사) 기록.
2. `docs/deviations.md` D25/D26 상태 갱신(Resolved/Open).
3. 특허 명세서 초안(`HeteroPilot_특허1_명세서_배치플래너.docx`) §5.3(d)·§5.4 S4′·§5.5 근사·§6 일곱째 항목의 수치 자리에 E-PP1~3 결과를 채운다. 명세서 문구가 코드와 다르면 **코드가 아니라 명세서를** 고치고 대리인에 전달.
4. `README`/CLI 도움말에 `--max-pp`, `--pipeline-domain-policy` 추가.

## 완료 조건
- 문서 갱신, 전체 테스트 통과, golden 바이트 동일(기본값).

---

## 부록 A. 위험과 대응

| 위험 | 대응 |
| --- | --- |
| PP 순열로 후보 폭발 → 오라클 느려짐 | 서명 동치 대표화, `max_pp` 상한, 픽스처 소형화 |
| 시뮬레이터가 단계 경계 p2p에 어느 링크 값을 쓰는지 불명 | STEP 5 조사 필수; 확인 전 실험 수치 인용 금지 |
| 노드 간 링크 미측정 | 클래스 기본값 + assumption; 특허 2(측정 계획)의 첫 대상 항목으로 등록 |
| vLLM 이기종 세대 PP 비공식 | `backend_only`는 opt-in, 배포 기본 거절, D26 기록 |
| 비대칭 분할 근사가 낙관적일 가능성 | 보수적 배율(최대 단계 기준) 사용, 플래그 노출, 실측 시 재검증 |

## 부록 B. 특허 명세서와의 대응표

| 명세서 항목 | 이 지시서 STEP |
| --- | --- |
| §5.2 절차 8(파이프라인 도메인, 정책) | STEP 2 |
| §5.3 (d) 후보, 동일 TP 교집합, 비대칭 분할 | STEP 3 |
| §5.4 S4′ 하한, 역할 면제, 오라클 일치 | STEP 4 |
| §5.5 pp_size 방출·느린 라벨 근사·비대칭 후처리 | STEP 5 |
| §5.8 (7) 파이프라인 배포 실시예 | STEP 6 |
| §6 일곱째 효과 수치 | STEP 7 |
| 청구항 1(b)(c), 3, 4, 5 | STEP 2–4 |
