# ScenarioLab Workspace (대화형 워크스페이스) 구현·시험 보고서

- 작성일: 2026-09-02
- 대상 작업지시서: `WORK_ORDER_scenariolab_workspace.md` v1.0
- 선행: ScenarioLab P1~P4 + topology v2 (main 머지 완료 상태에서 착수)
- 브랜치: `feat/scenariolab-workspace-core` → `-ui` → `-extras` (stacked)

## 1. 구현 범위 (작업지시서 §8 단계별)

### P1 — Workspace 코어 (API/CLI로 완결)

| 기능 | 구현 |
|---|---|
| F1 ClusterBuilder | `scenariolab/generator/cluster_builder.py` — ClusterBuildRequest → 기존 topology-v2의 `_make_node`를 그대로 재사용(복제 없음)해 ClusterSpecV2 YAML 생성. placeholder class 거부(가용 목록 포함), 노드당 상한, 총 64대 상한, >1600Gbps 경고(거부 아님), 요청 hash 기반 멱등(FR-CB4). CLI `scenariolab build-cluster --spec req.yaml` |
| user_defined 라벨 | planner `Source` enum에 `USER_DEFINED` 추가(의존 변경, 커밋 `cb690b4`). 직접 입력한 링크 수치는 YAML→DB link_summary→UI 뱃지까지 전파 |
| 스키마 v3 | `workspaces`/`placements` 테이블, `clusters.origin/build_request_json/link_summary`, `services.origin`. v1/v2→v3 자동(가산적) 마이그레이션 |
| F3 오버레이 배치 | `scenariolab/runner/workspace.py` — PLACED 장치를 ALLOCATED로 마킹한 **in-memory 사본** 위에서 기존 fast path(`plan_fast`) 실행. planner 수정 없음(§5.1의 환원 그대로). preview(PLANNING)→confirm(PLACED)는 BEGIN IMMEDIATE + 장치 재검사로 원자화(FR-P2) |
| FR-P5/P6/P7 | REJECTED에 진단 + 점유 장치 요약; fidelity/calibrated/npu/shared_fabric 라벨 기록; 랜덤 SLO는 (seed,index)로 완전 결정(seed가 service id에 포함) |
| shared_fabric_warning | 신규 plan 장치쌍 최단경로 링크들의 contention_group ∩ 기존 PLACED 집합 (§5.4, `planner/topology.py` 경로 함수 사용). 간섭 수치는 만들지 않음 — 경고만 |
| workspace 전력 cap | `total_power_cap_w` 설정 시 "기존 peak 합 + 신규 peak > cap" → `workspace_power_cap` 위반으로 거부 (FR-P3) |

### P2 — 웹 UI (화면 ⑥⑦⑧)

- ⑥ Builder: 노드 그룹 행 추가/삭제, preset/custom 링크(+user_defined 고지), 거부 메시지 표시, 생성 결과(island·TP 후보·경고·토폴로지) + Workspace 시작
- ⑦ Clusters: origin 탭(all/random/custom), fabric 라벨(user_defined 빨간 뱃지), 행 클릭 미니 토폴로지, Workspace 시작
- ⑧ Workspace: 간섭 미모델링 상시 고지(§9), 서비스별 색 토폴로지(색=seq 고정, FR-W2), 클래스별 잔여 게이지, 랜덤/직접 SLO 폼, preview→확정/취소(FR-W4), avg/peak 합 구분 + "동시 peak 가정 보수적 상한" 라벨 + cap 게이지(FR-W3), SLO ✓/✗ per-metric(비보정 시 회색, FR-W1), REJECTED 이력 유지+진단 펼침(FR-W6), 상세 펼침은 `GET .../placements/{id}` 재사용(FR-W5), `#/workspace/{id}` 라우팅(FR-W7)
- 수동 체크리스트에 ⑥⑦⑧ 항목 추가 (`docs/scenariolab_ui_checklist.md`)

### P3 — 보강

- **replan-all** (§5.5): `POST /replan?order=seq|rps_desc&apply=` — preview는 무상태, apply는 `replace_all_placements` 단일 트랜잭션으로 원자 스왑. "joint 최적화가 아님" 문구를 코드 주석·API 응답·본 문서에 3중 명시
- **ARCHIVED**: archive 후 신규 배치 409 거부
- **`scenariolab verify --workspace <id>`**: PLACED plan을 **계획 당시와 동일한 점유 뷰**(다른 PLACED를 ALLOCATED로) 위에서 실제 LLMServingSim으로 재평가, `<document>.verify.json`에 오차 기록. fast-path 라벨은 절대 덮어쓰지 않음

## 2. FR-A6 개정 (deviation 기록, 작업지시서 §7 지시)

배치 모드의 "DB는 mode=ro" 원칙은 **읽기 endpoint에 한해 유지**된다. workspace 계열
endpoint는 상태를 만들므로 전용 read-write store를 요청 단위로 연다. 동시성은 기존
WAL + BEGIN IMMEDIATE로 담보하며, 결과(results/verifications) 테이블은 여전히 웹
계층이 만지지 않는다.

## 3. 시험 결과 (2026-09-02)

```
$ pytest -q            → 393 passed (workspace 신규 32: builder 9, placement 13,
                          API 6, migration 갱신 등)
$ ruff check . / mypy  → clean
```

| 요구 (작업지시서) | 테스트 | 결과 |
|---|---|---|
| §3.3 builder 전 항목 (거부/라벨/멱등/intra-node 강제) | `test_cluster_builder.py` 9건 | PASS |
| §5.6 순차 배치: 장치 교집합 없음·잔여 정확 | `test_sequential_placement_disjoint_devices` | PASS |
| §5.6 자원 소진 → REJECTED + 점유 요약, 오버레이 불변 | `test_exhaustion_rejected_with_occupancy_summary` | PASS |
| §5.6 제거 → FREE 복귀 → 재배치 | `test_remove_returns_devices_and_allows_replacement` | PASS |
| FR-P2 원자성: 동일 장치 동시 confirm 경쟁 → 정확히 한쪽 승리 | `test_confirm_atomicity_under_race` (스레드 2) | PASS |
| FR-P3 workspace cap | `test_workspace_power_cap` | PASS |
| FR-CAT2 workspace 상호 격리 | `test_workspaces_are_isolated` | PASS |
| FR-P7 랜덤 시퀀스 재현성 | `test_random_sequence_reproducible` | PASS |
| FR-W4 preview 무점유 | `test_preview_does_not_occupy` | PASS |
| §8 P1 게이트 시나리오 (HTTP 전 구간) | `test_workspace_api.py::test_gate_scenario` | PASS |
| §5.5 replan preview 무상태 + apply 원자 스왑 + 순서 옵션 | `test_replan_preview_and_apply` | PASS |
| ARCHIVED 거부 | `test_archived_workspace_refuses_placements` | PASS |
| §9 verify --workspace (기지 편향 가짜 sim → 오차 정확히 20%) | `test_verify_workspace_with_fake_sim` | PASS |

### 라이브 게이트 재현 (실서버 HTTP, §8 P1 완료 조건)

```
1. POST /api/clusters/build  a40×8×2 + rngd×4×1, ib_100g
   → custom-demo-d11f2f0a | accels 20 | islands 3 | INFINIBAND 100Gbps (placeholder)
2. POST /api/workspaces → ws0001
3. POST /placements {"slo":"random","count":3,"seed":123}
   → p0001~p0003 PLACED, SLO ✓/✓, avg 214.6/228.2/211.3 W
4. GET /summary → free 17/20, by_class {A40 16/16, RNGD-CARD 1/4},
   power avg 654.1W / peak 1369.3W, overlay 3 devices
5. DELETE p0001 → free 18/20
```

CLI: `build-cluster`로 custom ETHERNET 50Gbps(user_defined 라벨) 클러스터 생성 확인.
`verify --workspace ws0001`은 실제 LLMServingSim으로 실행 (결과는 아래 §4).

## 4. 실환경 verify --workspace — 실제 발견 포함

`scenariolab verify --workspace ws0001 --db …` (실제 LLMServingSim, 2 verified / 0 skipped):

| placement | 장치 | fast p99 TPOT | sim p99 TPOT | sim SLO 판정 |
|---|---|---|---|---|
| ws0001-p0002 | RNGD-CARD ×1 | 11.0 ms | **212.1 ms** | **✗/✗ — fast의 ✓/✓가 뒤집힘** |
| ws0001-p0003 | A40 ×1 | ~18 ms | 30.7 ms | ✓/✓ (판정 유지, 오차 64%) |

p0002는 작업지시서 §9가 경고한 "fast path 예측 오류로 잘못된 ✓ 표시"의 **실제
사례다**: surrogate의 상수 기반 큐잉 모델이 RNGD 카드의 포화 상태(주지된 동시성
낙관, HANDOVER §2.1)를 크게 과소평가했다. 이 훅이 존재하는 이유이며, 결과는
`<document>.verify.json`으로 남고 fast-path 행의 fidelity 라벨은 그대로다 —
사용자는 UI의 surrogate 뱃지 + 이 검증 기록으로 판단한다. RNGD가 낀 배치는
verify --workspace를 돌려보는 것을 권장 (P5+ 후속: 이 오차를 workspace UI에
직접 표시하는 verify 패널).

## 5. 남은 것 / 범위 밖 재확인

- co-location(장치 공유), 서비스 간 간섭 수치화, joint 최적화: **범위 밖** (작업지시서
  §0.1) — 요청이 오면 거부하고 보고한다.
- UI 브라우저 수동 체크리스트(⑥⑦⑧)는 연구자 확인 대상.
