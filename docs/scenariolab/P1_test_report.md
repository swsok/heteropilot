# ScenarioLab P1 (feat/scenariolab-core) 구현·시험 보고서

- 작성일: 2026-09-01
- 대상 설계서: `DESIGN_scenariolab.md` v1.0 (§12 P1 — 생성기 + 실행기)
- 기반: `main` = `0316c29`, upstream pin `2c2042ce`
- 시험 환경: 이 저장소의 개발 머신 (20 cores, GPU 불필요 — P1은 시뮬레이터를 사용하지 않음)

## 1. 구현 범위

| 모듈 | 파일 | 상태 |
|---|---|---|
| LabConfig (§3.1) | `scenariolab/config.py` | 완료 — pool placeholder 거부, 모델 커버리지 검증, max_scenarios 상한 |
| Seed 파생 (§3.2) | `scenariolab/generator/sampling.py` | 완료 — `sha256-low8-63bit` 계약 고정 |
| M1 ClusterGenerator (§4) | `scenariolab/generator/cluster_gen.py` | 완료 — FR-C1~C9 |
| M2 SLOGenerator (§5) | `scenariolab/generator/slo_gen.py` | 완료 — FR-S1~S7 |
| M3 BatchRunner (§6) | `scenariolab/runner/batch.py` | 완료 — FR-B1~B7, B9 (B8 검증 표본은 P2) |
| M4 TieredPredictor (§7) | `scenariolab/runner/tiers.py` | **부분** — Tier-2 surrogate만. Tier-1 envelope / calibration / Tier-3 검증은 P2 |
| M5 ResultStore (§8) | `scenariolab/store/{schema.sql,db.py}` | 완료 — FR-D1~D3, WAL, 스키마 버전 |
| CLI (§6.4) | `scenariolab/__main__.py` | `generate`/`run` 동작, `verify`/`serve`/`export`는 명시적 미구현 안내 |
| 배치 설정 | `experiments/configs/lab/{smoke,default}.yaml` | 완료 |
| 링크 클래스 | `profiles/networks/{pcie_gen4,nvlink,ib_100g,ib_400g}.yaml` | 신규 (아래 편차 D-S1) |

## 2. 설계 대비 편차 (구현 중 확정한 사항)

| ID | 편차 | 결정 |
|---|---|---|
| D-S1 | 설계서는 `profiles/networks/`에서 링크 풀을 읽는다고 했으나 실제로는 빈 디렉터리였음 | 링크 클래스 YAML 4종을 신규 작성. 모든 수치는 `examples/clusters/heterogeneous-lab.yaml`의 placeholder 값을 복사(ib_100g 대역폭만 공칭 100G)하고 전부 `source: placeholder`로 라벨 |
| D-S2 | §3.2의 seed는 uint64였으나 SQLite INTEGER(signed 64-bit)에 저장 불가(OverflowError) | 파생 seed를 63-bit로 마스크. 계약 문자열 `sha256-low8-63bit`로 고정 |
| D-S3 | planner의 `SurrogateRanker`는 의도적으로 **순서만** 내고 metric을 내지 않음 (planner 내부 규율) | ScenarioLab 쪽에 별도 `SurrogatePredictor`를 구현. greedy/stage-5와 동일한 memory-roofline 물리 + slack 1.2, 전력은 프로파일의 idle/active 실측치를 utilization 블렌딩. 모든 결과에 `fidelity: surrogate` 라벨을 DB까지 전파 (FR-T3의 정직성 계약을 P1부터 적용) |
| D-S4 | FR-C4의 interconnect 지정이 프로파일 파일에 없음 | `cluster_gen.CLASS_FACTS` 레지스트리로 명시: a40/a5000→PCIE, rtxpro6000→NVLINK, furiosa_rngd_card→링크 없음(카드 내부 fabric, 싱글턴 island). 노드당 카드 수 상한도 여기서 (RNGD 4, GPU 8) |
| D-S5 | `power_saving_pct` 저장 단위 | 설계 수식(비율)에 ×100한 퍼센트 값으로 저장 (컬럼명 `_pct`와 일치) |
| D-S6 | 생성 클러스터의 NodePower | 모든 노드에 upstream 기본값 `power:` 블록(placeholder)을 부여 — 없으면 시뮬레이터/서로게이트 모두 에너지를 내지 않아 `minimize_energy` 목적함수가 계산 불능이 되기 때문 (deviations D2/D14와 일관) |
| D-S7 | resume 시 ERROR 재시도 | `pending_scenarios`는 PENDING + ERROR + (중단으로 남은) RUNNING을 반환. 한 번의 `run` 호출 안에서는 ERROR 자동 재시도 정확히 1회 (FR-B5) |
| D-S8 | 재현성 vs provenance | workload trace는 per-run temp 디렉터리에 생성되므로 provenance에 trace **경로**를 기록하면 byte-identical 재현이 깨짐 → 경로 대신 `dataset_hash`(내용 해시)로 정체성 기록 |

## 3. 시험 결과

실행 명령과 결과 (2026-09-01):

```
$ pytest tests/scenariolab -q        → 46 passed
$ pytest -q                          → 330 passed (기존 284 + 신규 46, 회귀 없음)
$ ruff check .                       → All checks passed!
$ mypy (planner + scenariolab)       → Success: no issues found in 45 source files
```

### 3.1 시험 항목별 매핑 (설계서 §4.5/§5.4/§6.5/§8.3/§7.5)

| 요구 | 테스트 | 결과 |
|---|---|---|
| 결정론: 같은 seed → byte-identical YAML | `test_cluster_gen.py::test_determinism_byte_identical`, `test_slo_gen.py::test_determinism_byte_identical` | PASS |
| 독립성: 배치 크기를 늘려도 기존 인덱스 불변 | `test_*::test_independence_from_batch_size` | PASS |
| 전수 유효성: `load_cluster_spec` + `detect_islands ≥ 1`, `load_service_spec` | `test_all_generated_clusters_valid`, `test_all_specs_valid_and_in_range` | PASS |
| placeholder 거부 (즉시 오류) | `test_placeholder_pool_rejected` (ascend_target 투입) | PASS |
| FR-C4 회귀: 클래스별 interconnect 강제 | `test_interconnect_dictated_by_profile` (rtxpro6000=NVLINK만 / a40=PCIE만 / RNGD=링크 없음) | PASS |
| FR-C5 contention group | `test_contention_groups_assigned` | PASS |
| 경계값 (1 노드 × 1 accel) | `test_minimal_ranges_still_valid` | PASS |
| 통계적 클래스 커버리지 (seed 고정) | `test_statistical_class_coverage` (40개 생성 시 4클래스 전부 등장) | PASS |
| FR-S4 p50≤p95≤p99 | `test_all_specs_valid_and_in_range` | PASS |
| FR-S5 objective 고정 | 〃 | PASS |
| 미니 배치 E2E (2×3, 30초 이내) | `test_mini_batch_e2e` (~2초) | PASS |
| Resume: DONE 불변, 누락분만 재실행 | `test_resume_skips_done` (predictor 호출 횟수로 검증) | PASS |
| 오류 격리 + 자동 재시도 1회 | `test_error_isolation_and_retry` (1건 ERROR, 5건 DONE, attempts=2) | PASS |
| FR-B3 workers=1 ≡ workers=2 (DB 내용) | `test_worker_count_does_not_change_results` | PASS |
| FR-B7 baseline: feasible → saving 기록 / SLO 위반 → NULL+사유 | `test_baseline_feasible_records_saving`, `test_baseline_infeasible_yields_null_saving` | PASS |
| infeasible 진단 2경로: bound-pruned(진단=rejected_summary) vs 예측 후 위반(violated_constraints+closest_plan) | `test_baseline_infeasible_yields_null_saving`, `test_slo_violation_after_prediction_diagnosed` | PASS |
| 재현성: 동일 시나리오 2회 → timestamp 제외 동일 JSON | `test_reproducibility_byte_identical_modulo_timestamp` | PASS |
| Golden 배치 (seed 고정 DB 이미지) | `test_golden_batch` + `tests/scenariolab/golden/lab-test-db.json` | PASS |
| 저장소: 왕복 무손실 / 필터·정렬·페이지네이션 / 스키마 버전 오류 / 16-worker 병렬 무유실 | `test_store.py` 전체 (8 프로세스 × 10건) | PASS |
| Surrogate: 결정론·내적 일관성·프로파일 전력 복사(수치 발명 금지) | `test_tiers.py` (A5000 peak = 실측 active 227.6 W 정확 일치) | PASS |

### 3.2 P1 완료 게이트 (설계서 §12)

```
$ python -m scenariolab run --config experiments/configs/lab/smoke.yaml
  → 15/15 시나리오 DONE_FEASIBLE, DB 적재 15건, 1.4초, 오류 0
```

추가로 default 배치(설계 규모)도 실행:

```
$ python -m scenariolab run --config experiments/configs/lab/default.yaml
  → 1,500 시나리오 68초 (wall, workers=16) — 목표 "≥10 시나리오/분"의 ~130배
  → DONE_FEASIBLE 1,398 / DONE_INFEASIBLE 102 / ERROR 0
  → median power saving 19.2% (saving 계산 가능 1,104건; NULL 396건 = baseline이
    SLO를 위반했거나 부재한 시나리오, 사유는 행별 baseline_note에 기록)
  → NPU(RNGD) 포함 클러스터 18/30
```

## 4. 정직성 라벨의 현재 상태

- 모든 P1 결과의 fidelity는 **surrogate**다. 시뮬레이션된 결과는 아직 하나도 없고,
  DB(`results.fidelity`)와 시나리오 JSON, provenance에 그렇게 기록되어 있다.
- surrogate 오차는 **아직 측정되지 않았다**. 위 feasible율/절감률은 P2의 full-sim
  교차 검증(§7.4) 전까지 신뢰 구간이 없는 수치로 취급해야 한다.
- `calibrated`/`npu_extrapolated` 컬럼은 P1에서 항상 0 — P2에서 실제 판정 로직이 들어간다.

## 5. 다음 단계 (P2, branch `feat/scenariolab-tiers`)

1. Tier-1: `planner/envelope.py` 캐시 키 그대로 조회 (D13의 dp_replicas 포함 키 준수)
2. Tier-2 보강: `profiles/calibration/*` bucket 커버 판정 → `calibrated` 플래그
3. FR-T6: RNGD 동시 시퀀스 추정 > 32 → `npu_extrapolated` 플래그
4. FR-B8/§7.4: verify_seed 기반 층화 표본 추출 + full sim 재평가 → `verifications` 테이블
5. 오라클 정합 회귀: tiered(full_sim=top_k=전체) ≡ `plan --oracle` (ASTRA-Sim 빌드 노드 필요)
