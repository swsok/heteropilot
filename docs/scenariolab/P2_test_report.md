# ScenarioLab P2 (feat/scenariolab-tiers) 구현·시험 보고서

- 작성일: 2026-09-01
- 대상 설계서: `DESIGN_scenariolab.md` §7 (M4), §7.4 검증, §12 P2
- 선행: P1 (`docs/scenariolab/P1_test_report.md`)
- 실환경 검증 노드: a5000 (ASTRA-Sim 빌드됨, `scripts/whichnode.sh` 확인)

## 1. 구현 범위

| Tier | 내용 | 파일 |
|---|---|---|
| Tier-1 | `SharedEnvelope` — planner의 envelope 키(dp 포함, D13 준수)로 **trace digest 없이** 조회. 같은 (placement, knobs, network class, workload bucket)이면 시나리오·배치를 넘어 재사용. surrogate 경로는 **read-only**로 열어 분석 수치가 캐시에 절대 못 들어가게 강제. hit는 fidelity=`envelope`로 재라벨 | `runner/tiers.py` |
| Tier-2 | calibration margin (FR-T2): `profiles/calibration/*`가 클러스터의 **모든** hardware class × workload bucket을 커버할 때만 robust margin 적용 + `calibrated: true`; 아니면 raw + false (§0.4) | 〃 |
| Tier-3 | `tier_policy.full_sim: top_k` — planner stage-6 surrogate top-K를 실제 LLMServingSim으로. `verification_only` — 검증 표본만 sim | `runner/{batch,verify}.py` |
| FR-T6 | `npu_concurrency_extrapolated`: plan의 RNGD 카드당 추정 동시 시퀀스 > 실측 최대 32 (HANDOVER §2.1) → 플래그, DB·UI 전파 | `runner/tiers.py` |
| FR-B8/§7.4 | verify_seed 기반 층화 표본(feasible × 크기 bucket × NPU) → recommended + Pareto 대안 K개 full sim 재평가 → 오차·selection flip·feasibility flip·regret 기록, sim 결과는 envelope에 적재 | `runner/verify.py` |
| CLI | `python -m scenariolab verify --config … [--fraction --min-count]` | `__main__.py` |

## 2. planner 쪽 의존 변경 (설계 §1.2 원칙에 따라 기록)

**`LLMServingSimPredictor`에 `run_id_prefix` 파라미터 추가** (커밋 `e8218e2`).

- 발견 경위: 첫 실환경 검증에서 4개 시나리오를 병렬 sim하자 chakra converter /
  trace_generator가 무작위로 붕괴. 원인은 run-id가 `candidate.id`에서만 파생되는 것 —
  후보 id는 **한 탐색 안에서만** 유일하므로, 서로 다른 클러스터의 동일 placement가
  같은 `astra-sim/inputs/runs/<run-id>/`를 공유·오염했다.
- 수정: 호출자가 접두어를 주면 격리, 기본값(빈 문자열)은 기존 run-id와 byte-identical.
  ScenarioLab은 시나리오 id를 접두어로 사용. 회귀 테스트
  `test_sim_predictor_gets_scenario_scoped_run_ids`로 고정.

## 3. 시험 결과 (단위/통합, mock 기반)

```
$ pytest tests/scenariolab -q   → (P2 시점) 59 passed
$ ruff check . / mypy           → clean
```

| 요구 (설계 §7.5) | 테스트 | 결과 |
|---|---|---|
| Tier-1 hit: 캐시 적중 시 sim 0회 + fidelity=envelope | `test_envelope_tier1_hit_and_readonly`, `test_batch.py::test_envelope_reuse_across_scenarios` (2번째 시나리오에서 predictor 호출 0) | PASS |
| Tier-1 키가 trace에 독립 (bucket 수준 재사용) | `test_envelope_key_ignores_trace` | PASS |
| read-only 캐시에 put 불가 (fidelity 오염 방지) | `test_envelope_tier1_hit_and_readonly` | PASS |
| FR-T2 calibrated 판정 (커버/하드웨어 미커버/버킷 미커버) | `test_calibration_margins_covered_and_not` | PASS |
| FR-T6: 동시성 128→true, 16→false, GPU-only→false | `test_npu_concurrency_flag`, `test_npu_flag_false_on_gpu_only` | PASS |
| **오라클 정합**: tiered(full_sim=top_k, K 무제한) ≡ `exhaustive.oracle` (동일 predictor) | `test_batch.py::test_oracle_agreement_tiered_path` | PASS |
| 층화 표본: 결정론·전 계층 대표·min_count·빈 풀 | `test_verify.py::test_stratified_sample_*` | PASS |
| 오차 통계 정확성: 편향 1.25×/1.1× 가짜 sim → err 정확히 20%/9.09% | `test_verification_pass_records_known_error` | PASS |
| plan 없는 시나리오는 skip + 사유 (충돌 아님) | `test_verification_skips_scenario_without_plans` | PASS |
| 균일 편향에서 selection flip 0 / regret 0 기록 | `test_selection_flip_fields_populated` | PASS |

## 4. 실환경 검증 (P2 완료 게이트)

`experiments/configs/lab/verify_smoke.yaml`: 클러스터 2(a5000/a40) × SLO 2,
50 requests, fraction 1.0 → 4/4 시나리오를 실제 LLMServingSim으로 교차 검증.

```
[batch lab-verify-smoke] 4 done · feasible 100% · errors 0        (fast path ~1s)
[verify lab-verify-smoke] 4 verified · 0 selection flips · 0 feasibility flips
  scenario당 검증 벽시계 306~574s (recommended + 대안 ≤2개 순차 sim)
  envelope cache에 sim 결과 3건 적재 (후속 배치의 Tier-1 재사용분)
```

측정된 fast-path 오차 (err = (sim − fast)/sim):

| 지표 | 값 (4건) | p95 |비고 |
|---|---|---|---|
| p99 TPOT | +7.2 ~ +34.3% | 31.0% | surrogate가 과소예측 (낙관) |
| p99 TTFT | −37.0 ~ −86.0% | 79.7% | surrogate가 과대예측 (비관, queue 상수의 한계) |
| avg power | +58.6 ~ +60.6% | 60.6% | 아래 개선 후에도 남은 잔차 |

**검증 루프가 실제로 개선을 만든 사례** — 첫 검증에서 avg power 오차가 **+95.5%**로
측정됐다. 원인은 surrogate가 장치 전력만 계산하고 시뮬레이터가 과금하는 노드(호스트)
전력을 누락한 것. NodePower 블록의 정적 성분(base + CPU util 블렌드 + DRAM/link/NIC/
storage idle)을 **그대로 복사**해 합산하도록 수정(커밋 `4b32b4f`, 수치 발명 없음) 후
재검증에서 **+60.5%**로 감소. 잔여 오차(호스트 동적 per-bit 항, accelerator standby
거동)는 calibration(§5.8)이 흡수할 몫으로 남겨두고 라벨(`calibrated: false`)로 표시.

## 5. 정직성 라벨 상태

- 검증 표본을 거친 시나리오만 status=`VERIFIED`, sim 수치는 `verifications` 테이블에.
- fast-path 수치의 fidelity 라벨은 그대로 유지 — 검증은 오차를 **측정**하지, fast 수치를
  sim 수치로 바꿔치기하지 않는다.
- 기존 calibration 파일(`profiles/calibration/*.yaml`)의 bucket 키는 명명형
  (`sharegpt-llama31-8b-300`)이라 랜덤 시나리오의 `workload_bucket()` 키와 매칭되지
  않는다 → 현재 모든 시나리오가 `calibrated: false` (정직한 상태). 랜덤 SLO용
  calibration을 만들려면 이 검증 파이프라인의 (sim, fast) 쌍으로 bucket별 재적합이
  필요하다 — P5+ 후보 작업.

## 6. 남은 리스크

- TTFT surrogate는 상수 기반(§P1 D-S3)이라 오차가 크고 방향도 비관적. selection flip이
  이번 표본(4건)에서 0이었지만 표본이 작다 — default 배치의 5% 검증(75건)을 실행해
  flip율을 측정해야 한다 (수 시간짜리 작업, 명령은 P1 보고서 §3.2와 동일 + fraction 반영).
