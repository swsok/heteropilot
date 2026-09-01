# ScenarioLab P3 (feat/scenariolab-web) 구현·시험 보고서

- 작성일: 2026-09-01
- 대상 설계서: `DESIGN_scenariolab.md` §9 (M6), §10 (M7), §12 P3
- 선행: P1 (`docs/scenariolab/P1_test_report.md`), P2 (`docs/scenariolab/P2_test_report.md`)

## 1. 구현 범위

| 모듈 | 파일 | 상태 |
|---|---|---|
| M6 Web API | `scenariolab/api/{server,schemas,graph}.py` | 완료 (`/api/plan`은 P4) |
| M7 Web UI | `scenariolab/web/{index.html,app.js,style.css}` | 화면 ①②③ 완료 (④⑤는 P4) |
| CLI | `python -m scenariolab serve --config <lab.yaml> [--db --host --port]` | 동작 |
| 저장소 확장 | `store/db.py` — read-only 모드(`mode=ro`) + 조회/집계/카운트 API | 완료 |
| 수동 smoke 게이트 | `docs/scenariolab_ui_checklist.md` | 커밋됨 (§10.4) |

### Endpoint (구현분)

`GET /api/summary` (배치 목록 + 대시보드용 서버측 binning) · `GET /api/scenarios`
(필터·정렬·페이지네이션 + total) · `GET /api/scenarios/{id}` (결과 문서 + 토폴로지
그래프 + 검증 레코드) · `GET /api/clusters[/{id}]` · `GET /api/services[/{id}]` ·
`GET /api/verification` · `GET /api/batches/{id}/progress` · `GET /` (정적 SPA)

## 2. 설계 요구사항 반영

- **FR-A1**: 모든 응답이 `api/schemas.py`의 Pydantic 모델. `/docs` OpenAPI 자동 문서 활성.
- **FR-A5/FR-U3**: 토폴로지 그래프 JSON은 UI가 그대로 그리는 형태. plan을 겹쳐 볼 때만
  `role`/`in_plan` 채움 (`api/graph.py`). infeasible이면 closest_plan을 overlay.
- **FR-A6**: 결과 DB는 `sqlite mode=ro`로만 접근. 읽기 전용 연결에서 DELETE 시도가
  실제로 거부되는 것을 테스트로 고정.
- **FR-U1**: fidelity(envelope/surrogate/sim), `⚠ NPU extrapolated`,
  `calibrated: false` 뱃지를 모든 수치 옆에 상시 렌더. 뱃지 데이터는 API 응답에 포함.
- **FR-U4**: hash 라우팅 (`#/scenario/sc0007x0031` 공유 가능).
- **FR-U5**: 서버 부재/오류 시 빨간 배너 (fetch 실패·비 2xx 공통 처리).
- **FR-U6**: 서버측 페이지네이션. 대시보드 차트도 서버측 binning — 브라우저는
  전체 테이블을 절대 당기지 않음.
- 기술 선택은 설계 §10.1 그대로: 빌드 도구 없는 vanilla JS + ECharts CDN, 라이트 테마.

## 3. 시험 결과 (2026-09-01)

```
$ pytest tests/scenariolab -q   → (P3 시점) 79 passed
$ pytest -q                     → 전체 회귀 통과
$ ruff check . / mypy           → clean (mypy는 planner+scenariolab)
```

| 요구 | 테스트 | 결과 |
|---|---|---|
| 전 endpoint 왕복 (실제 미니 배치 fixture DB) | `test_api.py` 전반 | PASS |
| 필터·정렬·페이지네이션의 SQL-응답 일치 (total vs rows) | `test_scenarios_filters_and_pagination` | PASS |
| 부당한 정렬 키 → 422 (SQL injection 차단) | 〃 | PASS |
| 404 (scenario/cluster/service) | `test_scenario_detail`, `test_clusters_and_services` | PASS |
| DB 부재 → 503 (빈 응답 금지) | `test_missing_db_is_503` | PASS |
| read-only 강제 (FR-A6) | `test_readonly_store_rejects_writes` | PASS |
| plan overlay: in_plan 노드에 role 존재 | `test_scenario_detail` | PASS |
| 검증 산점도 데이터 (fast/sim 쌍) | `test_verification_endpoint` | PASS |
| 정적 SPA 서빙 | `test_web_ui_served` | PASS |
| OpenAPI 계약 golden (UI 회귀 1차 방어선, §10.4) | `test_openapi_contract_golden` + `golden/openapi-surface.json` | PASS |

라이브 smoke: `python -m scenariolab serve --config .../smoke.yaml` 기동 후
`/api/summary`·`/`·`/app.js`·`/api/scenarios` 정상 응답 확인.
UI 수동 체크리스트는 `docs/scenariolab_ui_checklist.md` — 브라우저 확인은
연구자가 수행 (Playwright 자동화는 설계상 P5 후순위).

## 4. 비고

- venv에 `fastapi`/`uvicorn`/`httpx` 추가 설치 (serve·테스트 전용; 시뮬레이터/planner
  경로는 영향 없음).
- ruff에 FastAPI `Depends`/`Query` 관용구 예외(`extend-immutable-calls`) 추가.
