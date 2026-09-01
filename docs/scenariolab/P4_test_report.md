# ScenarioLab P4 (feat/scenariolab-interactive) 구현·시험 보고서

- 작성일: 2026-09-01
- 대상 설계서: `DESIGN_scenariolab.md` §2.2, §9.2 `/api/plan`, §10 화면 ④⑤, §12 P4

## 1. 구현 범위

| 항목 | 파일 | 상태 |
|---|---|---|
| 대화형 fast path | `scenariolab/runner/interactive.py` | 완료 |
| `POST /api/plan` | `scenariolab/api/server.py` | 완료 |
| 질의 이력 | `store/schema.sql` v2 `plan_queries` + `record_plan_query()` | 완료 |
| 화면 ④ Interactive Planner | `scenariolab/web/app.js` `renderPlanner` | 완료 |
| 화면 ⑤ Verification | `scenariolab/web/app.js` `renderVerification` | 완료 |

## 2. 설계 요구사항 반영과 확정 사항

- **FR-T5 (time budget)**: 대화형 경로는 full sim이 원천 금지(envelope read-only +
  SurrogatePredictor만). 후보 폭발은 **선제적 top-K 절단**으로 제어 — surrogate 순위
  상위 `INTERACTIVE_TOP_K`(64)만 평가하고 절단이 발생하면 `truncated: true`.
  경과 시간은 `elapsed_s`로 항상 보고. 실측: smoke 클러스터에서 **~0.06초**
  (예산 10초 대비 상시 여유; wall-clock 강제 종료는 필요 시 후속).
- **FR-A2**: 응답에 `fidelity`·`calibrated`·`npu_extrapolated`·`truncated`·`elapsed_s`
  상시 포함 + 재현용 `seed`/`num_requests`.
- **FR-A3**: traffic(rps/input_p50/output_p50) 없는 요청 → 400
  "traffic is required: an SLO alone cannot size a deployment".
- **FR-A4**: infeasible은 200 + reason/violated_constraints/suggestions (진단은 오류가 아님).
- **FR-A6 절충(확정)**: 질의 이력은 설계대로 같은 DB의 **별도 테이블**(`plan_queries`)에
  기록하되, API의 결과 조회는 계속 `mode=ro`. 이력 기록만 요청당 단명 read-write
  연결로 수행하며, 이력 실패가 응답을 깨지 않도록 격리.
- **스키마 v1→v2**: 순수 추가형(테이블 1개)이라 read-write open 시 자동 마이그레이션,
  read-only open은 마이그레이션 방법을 안내하는 명시적 오류 (FR-D2).
  `serve`는 기동 시 1회 read-write로 열어 마이그레이션 후 read-only 서빙.
- **화면 ④**: 클러스터 선택 + 미니 토폴로지 미리보기, SLO 입력 폼, 결과는 ③과 동일
  구성 + "이 결과는 surrogate 예측이며 full sim 검증을 거치지 않았습니다" 고지 상시 표시.
- **화면 ⑤**: fast vs sim 산점도(avg power / p99 TTFT, y=x 기준선), 오차 p50/p95 카드,
  selection-flip 목록 → 클릭 시 ③ 이동.

## 3. 시험 결과 (2026-09-01)

```
$ pytest -q            → 362 passed (scenariolab 78)
$ ruff check . / mypy  → clean
```

| 요구 | 테스트 | 결과 |
|---|---|---|
| 유효 질의 → 200 + honesty 블록 + plan overlay 그래프 | `test_api.py::test_plan_endpoint_feasible_and_history` | PASS |
| 질의 이력이 `plan_queries`에 기록 | 〃 | PASS |
| traffic 누락 → 400 + 명시 메시지 (FR-A3) | `test_plan_endpoint_missing_traffic_is_400` | PASS |
| 미존재 cluster → 404 | `test_plan_endpoint_unknown_cluster_is_404` | PASS |
| infeasible SLO → 200 + 진단 (FR-A4) | `test_plan_endpoint_infeasible_is_200_with_diagnosis` | PASS |
| top-K 절단 → `truncated: true` + `surrogate_pruned` 카운트 | `test_interactive.py::test_top_k_truncation_flagged` | PASS |
| 동일 질의 2회 → 동일 recommended (결정론) | `test_fast_path_labels_and_determinism` | PASS |
| v1→v2 마이그레이션 (자동/안내 오류) | `test_store.py::test_v1_to_v2_migration` | PASS |
| OpenAPI golden 갱신 (endpoint 계약 회귀) | `test_openapi_contract_golden` (재생성 후 통과) | PASS |

라이브 smoke (v1 smoke DB → 자동 마이그레이션 후):

```
POST /api/plan {"cluster_id":"c0000", slo relaxed}
  → feasible: True | fidelity: surrogate | truncated: False | elapsed: 0.055 s
POST /api/plan {slo만, traffic 없음}
  → 400 "traffic is required: an SLO alone cannot size a deployment"
```

## 4. 남은 것 (P5, 후순위 — 설계서 명시)

정적 리포트 export (`scenariolab export`) · 다크 테마 · Playwright headless smoke.
