# HeteroPilot 작업지시서 — Stage B 증거 강화 (uq-B+)

> `WORK_ORDER_uncertainty_planner.md` Stage A·B 완료(PR #78–#84, `main` = `fcf8ba5`) 이후, 측정 계획(ΔR/비용 순위·판정 보류·`measure-apply`)의 **우월성 증거**를 만들기 위한 후속 작업지시서. 결과는 세 번째 직무발명(측정 계획) 신고서 §6 효과와 논문 한 절의 근거가 된다.
> 대상 저장소: `github.com/swsok/heteropilot` · 작성일: 2026-09-14 · 도구: Claude Code (CPU 노드로 충분, 가속기 불필요)
> 선행 문서: `docs/uncertainty_planner.md`, `experiments/uncertainty/results/eb1_regret_vs_budget.md`, `eb2_flip_detection.md`, `eb3_closed_form_vs_resim.md`, `docs/deviations.md` D33·D34·D40
>
> **개정 rev 2 (2026-09-14, `main` = `201f99e`).** 초안 작성 중 PR #86(E-B3 실행, `exp/uq-eb3`)과 PR #85(rps 작업지시서 STEP 1.5 정리)가 머지되었다. E-B3는 **이미 E-A1 fixture에서 실행되었으므로** §1 표와 STEP C4를 그 결과 위에서 다시 정의했다: 닫힌 형식은 순서를 보존했지만(같은 항목 1위, 비활성 3항목 정확히 0) 유일한 활성 항목의 ΔR 크기를 **30.5 % 과소**(14,298 vs 20,583)했고, 그 20,583에는 D40 미러 결함이 섞여 있어 분리되지 않았다. 따라서 C4의 목표는 "E-B3를 돌리는 것"이 아니라 **활성 항목이 둘 이상인 F2에서 크기 오차를 D40과 분리해 다시 재는 것**이다. `BinnedRooflineRanker`가 `--surrogate` 기본값이 된 것(PR #76, D30)은 이 지시서와 무관하다.

---

## 0. 이 문서의 사용법

1. STEP 순서를 지킨다. C1의 진실 캐시가 없으면 C2~C4는 실행할 수 없다.
2. 한 STEP = 한 브랜치 = 한 PR. 브랜치 이름 `feat/uqb-c<N>-<짧은이름>`.
3. "조사 필요" 항목은 추측하지 않고 코드·산출물을 읽어 확인한 뒤 PR 설명에 적는다.
4. 이 문서와 `CLAUDE.md`, `AGENTS.md`, 기존 `WORK_ORDER_*.md`가 충돌하면 기존 문서가 우선한다.
5. **결과가 가설과 다르면 그대로 기록한다.** fixture·시드·penalty를 조정해 결과를 좋게 만드는 것은 금지. 설정을 바꾸면 이유와 바꾸기 전 결과를 함께 남긴다(E-B1 결과 문서의 "discarded run" 표가 좋은 선례다).

### 절대 규칙 (재확인)

- **A1** 측정하지 않은 숫자를 측정값으로 표기하지 않는다. 열화(degrade)는 등급 라벨과 범위만 바꾸는 것이며, 진실은 항상 실측·시뮬레이션 캐시에서 온다.
- **A2** 데이터가 없으면 null. unbounded 항목은 ΔR=None, `undecidable` 목록.
- **A3** `profiles/`, `outputs/` 아래 실측 파일은 읽기 전용. 열화는 메모리 사본에.
- **A4** 기본 `plan` 경로의 golden 출력 불변. 이 지시서는 실험 하네스와 문서만 바꾸며, `planner/` 코드 변경은 C4의 `resimulate` 보강에 한정한다.

---

## 1. 배경 — 지금 결과가 "동작 증거"이지 "우월성 증거"가 아닌 이유

E-B1(2026-09-14 재실행)에서 `ours`는 모든 예산에서 오라클과 일치하고, 결정이 움직인 37개 집합에서 후회 0 도달 시간이 0.041 h(무작위 0.452 h, 라운드로빈 1.097 h)였다. E-B2의 전환 검출은 재현율 0.983·정밀도 0.773이었다. 그러나 결과 문서가 스스로 적은 대로:

| 사실 | 결과에 미치는 영향 |
|---|---|
| E-A1 진실 캐시는 `enable_pd=False`로 생성되어 후보 324개가 모두 aggregated/mixed. **P/D 후보가 없다** | `link_bw`가 어떤 결정도 뒤집지 못한다(E-B2에서 336건 전부 TN). 레지스트리 11항목 중 `sim_error:domain` 하나가 후회를 거의 전부 갖는다 |
| 후회를 갖는 항목이 하나뿐 | 그 항목을 첫째로 놓는 어떤 규칙도 오라클과 일치한다. `ours = oracle`은 순위 규칙의 증거가 아니라 "가장 싼 항목이 가장 중요했다"는 우연의 증거다 |
| E-B2 정밀도 0.773은 `sim_error` 56건에 대한 수치(TP 38 / FP 17 / FN 1) | 검출기 일반의 성능이 아니다. FP 17건의 원인이 분류되지 않았다 |
| 격자 밀도 m=3/5/9가 결과를 전혀 바꾸지 않음 | "구간이 교차점을 포함하는가"로 전환이 결정되므로 격자는 후회 크기에만 영향. 이 사실은 발명 설명에 유리하니 명시한다 |
| E-B3(PR #86, E-A1 fixture, `--top 4`): 닫힌 형식 0.287 s vs 재시뮬 2,090.8 s(1,104 런, **7,285×**). 순서 보존(활성 1항목 1위, 비활성 3항목 = 0). 그러나 활성 항목 `profile:cuda-a40-node_a40a`의 ΔR은 **14,298 vs 재시뮬 20,583, 30.5 % 과소**. `sim_error`는 정의상 재시뮬 불가(SKIPPED). Spearman 1.000은 4항목 중 3개가 0으로 묶여 **사실상 공허**하며, `--top 1` 실행의 Spearman은 항진(tautological)이라 인용 금지 | "닫힌 형식은 **무엇을 측정할지** 정하는 데는 충분하지만 ΔR을 **절약량**으로 인용하기엔 3할 낮다"가 현재 결론. 크기 오차가 닫힌 형식의 1차 근사 때문인지 D40 때문인지 **분리되지 않음**(×1.0 identity control이 A40 미러 후보에서 7.87e-2 편차, RNGD는 0). `docs/uncertainty_planner.md` §5·§6의 "`--resimulate-top` not implemented"는 오기 |
| D40: `EnvelopeCache`가 미러 대칭 P/D 분할을 하나로 병합 | P/D 후보 진실 캐시를 만들 때 진실 자체가 오염될 수 있음 |

이 지시서의 목표는 **둘 이상의 입력 종류가 동시에 결정에 관여하는 fixture**에서 E-B1·E-B2·E-B3를 다시 돌려, 순위 규칙이 기준선과 실제로 갈리는지(또는 갈리지 않는지)를 측정하는 것이다.

---

## 2. 설계

### 2.1 새 fixture F2 — P/D 후보와 촘촘한 TTFT

| 항목 | 값 | 이유 |
|---|---|---|
| 클러스터 | `experiments/configs/clusters/pd-rngd-gpu-card.yaml` (E-A1과 동일) | 실측 링크(13 GB/s), RNGD-CARD·A40 도메인이 모두 있는 유일한 fixture. 도메인이 없는 하드웨어는 `refuse`에서 전부 보류되어 진실이 성립하지 않으므로 다른 fixture는 쓸 수 없다 |
| 서비스 명세 | `examples/service_specs/llama31-8b.yaml`의 사본 `experiments/uncertainty/fixtures/llama31-8b-ttft4000.yaml`: **TTFT p99 4 000 ms**, TPOT 50 ms, 나머지 동일 | 25 s TTFT에서는 KV 전송 시간이 판정에 들어오지 않는다. 4 s는 D23 재검증의 3-regime 표에서 "중간" 구간 — 링크 대역폭 35 vs 13 GB/s가 P/D 후보의 TTFT 판정을 실제로 바꿀 수 있는 첫 SLO. **조사 필요**: `docs/d23_revalidation.md`의 regime 경계값을 읽어 4 000이 맞는지 확인하고, 아니면 그 표의 중간 regime 값을 쓴다 |
| 후보 생성 | `enable_pd=True`, 300 요청, seed 42 | E-A1 캐시와 같은 트레이스(캐시 키의 `trace_digest`가 같아야 aggregated 후보 162개 시뮬레이션을 **재사용**할 수 있다 — **조사 필요**: TTFT SLO는 `EnvelopeKey`에 들어가지 않으므로 재사용이 가능해야 한다. 확인 후 재사용 수를 기록) |
| 예상 규모 | 이전 sweep 기준 생성 ~496, 평가 ~490 → 신규 시뮬레이션 ~330 | 8 워커에서 E-A1 162개가 24 m 26 s였으므로 ~50 m. P/D는 더 느리므로 1.5~2 h로 잡는다 |
| 진실 도메인 | `main`의 커밋된 도메인(D32 9점 RNGD-CARD, A40 3점), `outside_domain=refuse` | E-B1 재실행과 동일 |

**D40 대응(필수).** P/D 후보에는 미러 대칭 분할이 존재한다. 캐시 생성 전에 `EnvelopeCache._path`의 키에 island id 순서가 반영되도록 **하지 않는다**(A4, 캐시 스키마 변경은 이 지시서 범위 밖). 대신 하네스가 후보 목록에서 미러 쌍을 식별해 목록으로 남기고, 진실 평가 시 미러 쌍 중 캐시가 대표하는 쪽만 후보로 인정하고 다른 쪽을 `excluded_mirror`로 제외한다. 제외 수를 결과 md에 적는다. (D40을 "측정만 하고 고치지 않은" 상태로 두는 결정과 일관.)

### 2.2 열화 집합의 구성 — 종류가 섞이게

E-B1의 열화는 세 종류(링크 → placeholder 35 GB/s, 프로파일 → Tier 0 배율 1.3888의 analytical 등급, 도메인 → 스칼라 모드)였다. F2에서는 다음을 추가한다.

- 열화 집합 추출을 **종류 층화(stratified)** 로 한다: k∈{2,3}에서 서로 다른 종류가 최소 둘 포함되는 집합만 뽑는다(k=1은 종류별 전수). 순수 무작위 추출은 링크 항목이 많아(16개) 링크만 뽑히는 집합이 대부분이 되기 때문이다. 층화 여부와 시드는 provenance에 기록.
- 링크 열화 대상을 "모든 `fabric-*` 링크 일괄"에서 **링크별 개별**로 바꾼다. 그래야 "어느 링크가 결정에 들어오는가"가 항목 단위로 분리된다.
- 스칼라 모드 정의는 E-B1의 것을 그대로 쓴다(각 도메인을 적합점 하나로 붕괴). E-B1 결과 문서의 "첫 시도가 도메인을 전혀 넘기지 않아 전부 unmeasured가 된" 사고를 회귀 테스트로 고정한다: 열화 상태에서도 실행가능 후보가 0이면 하네스가 **중단하고 오류**를 낸다(조용히 ΔR=None으로 진행 금지).

### 2.3 측정 지표 — 무엇이 "우월성"인가

| 지표 | 정의 | 통과 기준(가설) |
|---|---|---|
| **후회 소지 항목 수** `n_active` | 열화 집합별로 ΔR>0인 항목 수, 그리고 fixture 전체에서 ΔR>0을 한 번이라도 가진 (종류, 항목) 수 | **종류 2개 이상**에서 ΔR>0 발생. 그렇지 않으면 F2도 퇴화 fixture이며 그 사실이 결과다 |
| R(B) 곡선 | E-B1과 동일. 결정이 움직인 집합만으로 별도 표 | `ours`가 `random`·`round_robin`·`widest`보다 빠르게 0 도달. **`oracle`과의 차이가 0이 아니어야 정상** — 0이면 다시 퇴화 의심 |
| 후회 0 도달 예산 비율 | `mean h(ours) / mean h(baseline)` | 보고만 |
| penalty 민감도 | `--slo-penalty` 기본값과 그 1/10에서 ΔR 순위의 Spearman ρ, 순위 뒤집힘 항목 목록 | 보고만(§2.5 지시) |
| 전환 검출 정밀도·재현율 | E-B2와 동일 + §2.4의 재정의 | 종류별로 보고. `link_bw`·`profile`이 0건이 아니어야 함 |
| E-B3 순위 상관 | 닫힌 형식 ΔR vs `--resimulate-top` 양끝점 ΔR의 Spearman ρ, 소요 시간 | 보고만 |

### 2.4 E-B2 오탐의 분류와 지표 재정의

E-B2의 FP 17건(`outputs/uncertainty/eb2/eb2_flip_detection.json`)을 **개별로 읽어** 다음 셋으로 분류한다.

- **(α) 구조적 오탐**: 항목의 범위 안에 추천 전환 교차점이 있어 "뒤집힐 수 있다"고 예측했으나, 진실값이 nominal과 같은 쪽에 있어 실제 복원 시 뒤집히지 않음. 이것은 검출기의 오류가 아니라 채점 방식의 문제다 — 계획이 예측하는 것은 "이 입력을 측정하면 결정이 **바뀔 수 있다**"이고, 단일 진실값에 대한 실현 여부로 채점하면 범위가 넓은 입력은 구조적으로 FP가 된다.
- **(β) 근사 오탐**: 닫힌 형식 규칙의 오차(PROFILE·POWER)로 교차점 위치가 어긋남. E-B2에서는 근사 규칙이 224건 전부 정확했으므로 0건일 것으로 예상하지만 확인한다.
- **(γ) 동치·타이**: 목적함수 값이 같은 동치 후보 사이의 id 변경. `equivalent_candidates` 병합으로 해소되는 것.

분류 결과에 따라 두 지표를 **병기**한다: (1) 기존 "실현 전환" 기준 정밀도·재현율, (2) **"가능 전환" 기준** — 진실을 nominal에서 항목 범위 내 여러 값(격자점)으로 바꾸어 각 값에서의 실현 여부와 예측을 비교한 정밀도·재현율. (2)가 검출기 자체의 정확도이고, (1)은 "이 fixture의 진실이 어디 있었나"에 좌우된다. 신고서 §6에는 두 값을 모두 적고 차이의 이유를 한 문장으로 쓴다.

### 2.5 E-B3 재실행 — 크기 오차를 D40과 분리하고, 활성 항목 둘 이상에서 재기

E-A1 fixture의 E-B3(PR #86)는 순서는 맞고 크기는 30.5 % 낮다는 결론을 냈지만, (i) 활성 항목이 하나라 순위 지표가 공허하고, (ii) 재시뮬 ΔR에 D40 미러 결함이 섞여 크기 오차의 원인이 분리되지 않았다. F2에서는 다음과 같이 다시 잰다.

- **D40 분리가 먼저다.** C1의 미러 쌍 목록으로 미러 후보를 재시뮬·ΔR 계산 **양쪽에서 제외**한 뒤 크기 비교를 한다. 그러면 남는 차이는 닫힌 형식(1차 근사)만의 오차다. 제외 전/후 두 값을 모두 보고하고, 차이가 D40의 기여다.
- **헤드라인 지표는 Spearman이 아니라 항목별 크기 오차**(`closed/resim − 1`)와 **활성 항목 사이의 순서 보존 여부**다. Spearman은 활성(ΔR>0) 항목이 3개 이상일 때만 계산하고, 그 미만이면 "계산 안 함"으로 기록한다(PR #86의 교훈).
- `--top N`은 재시뮬 가능한 항목 전부(PROFILE·POWER·LINK; `sim_error`는 정의상 SKIPPED로 별도 표기). F2에서 링크 항목이 활성이면 LINK_BW 재시뮬은 닫힌 형식과 **일치해야 한다**(정확 규칙) — 이것이 하네스의 두 번째 identity control이 된다.
- 소요 시간: E-A1에서 4항목 1,104 런 35 분(32 워커). F2는 후보 ~1.5배, 항목 ~2배로 **2~3 h**(32 워커) 예상. 재시뮬 결과는 별도 캐시 디렉터리에 저장해 재실행 시 재사용.
- `resimulate.py` docstring의 비대칭(PROFILE 항목은 island, 번들은 모델 단위)이 F2의 어느 후보에 영향을 주는지 표로 남긴다.

**결론 문장의 형태를 미리 정해 둔다.** "닫힌 형식은 측정 우선순위(순서)를 보존한다 / 보존하지 않는다; ΔR 크기는 항목별로 x~y % 낮으며 그중 D40 기여는 z %p이다." 이 문장이 특허 3 §5의 "섭동은 캐시 예측의 후처리로 수행한다"는 구성의 한계 서술이 된다.

---

## 3. STEP

### STEP C0 — 문서 동기화 (0.5일, 코드 변경 없음)

- `docs/uncertainty_planner.md` §4 "E-B1–E-B3 Not run" → E-B1·E-B2 결과 요약과 링크로 교체, §5·§6의 "`--resimulate-top` not implemented" 삭제(구현되어 있음), §6에 이 지시서를 후속으로 명시.
- D35: `README.md`와 `CHANGELOG.md`에 `--accuracy-domain`, `--measurement-plan`, `fit-accuracy-domain`, `measure-apply` 한 단락씩(이 fork가 소유하지 않는 파일이면 `docs/README.md`로).
- **잔차 분모 정리(코드·데이터, 작음).** `profiles/calibration/rngd_card_edf.yaml`은 부호 규약을 `(sim − measured)/measured`로 선언하지만, `conc: 76.0`의 `tpot_err_pct: -18.0`은 D22의 `(57.1 − 48.41)/48.41 = 17.95 %`(예측 분모)를 부호만 바꾼 값이다. 파일 규약으로는 `(48.41 − 57.1)/57.1 = −15.2 %`. (i) 그 점을 −15.2로 고치고 note에 환산 근거를 남긴다(다른 8점은 300요청 재측정으로 파일 규약이 맞는지 **조사 필요**). (ii) 마진 적용식 검토: 규약 `e = (sim − meas)/meas`를 `sim × (1 − e)`로 쓰면 실측을 복원하지 못한다(48.41 × 1.152 = 55.8 ≠ 57.1). 정확한 마진은 `m = −e/(1+e)`. `planner/optimizer/margin.py`가 어느 식을 쓰는지 확인하고, 후자로 바꾸되 E-A1·E6 결과 재현 여부를 회귀로 확인해 `docs/deviations.md`에 항목을 남긴다(판정 변화는 없을 것으로 예상하지만 확인이 결과다).
- 테스트: 링크 깨짐 확인 + 마진식 단위 테스트(`e=−0.152` → `m=0.1793`, `sim×(1+m)`이 실측을 1e-9 이내로 복원).

### STEP C1 — F2 진실 캐시 (1일 + 시뮬레이션 1.5~2 h)

**만드는 것**
- `experiments/uncertainty/fixtures/llama31-8b-ttft4000.yaml`(§2.1), `experiments/uncertainty/fixtures/f2.json`(service·cluster·num_requests·seed·enable_pd 기록; E-A1의 `FIXTURE` json과 같은 형식).
- `experiments/uncertainty/build_truth_cache.py`: fixture json → 후보 생성(`enable_pd=True`) → `EnvelopeCache`로 시뮬레이션(E-A1 캐시 디렉터리를 **읽기 소스로 연결**해 aggregated 후보 재사용) → `outputs/uncertainty/f2/cache`. D40 미러 쌍 목록 `outputs/uncertainty/f2/mirror_pairs.json`. 캐시 히트/미스/미러 제외 수를 표준 출력과 `f2_cache_summary.json`에.
- 진실 평가: `main`의 도메인 + `refuse`로 판정한 실행가능 후보 수, 승자, 목적값 → `experiments/uncertainty/results/f2_truth.md`.

**테스트** `tests/test_uq_f2_fixture.py`
- fixture 로더가 `enable_pd=True`를 강제하고, 서비스 명세의 TTFT SLO가 25 000이 아님을 확인(E-A1과의 혼동 방지).
- 미러 쌍 식별 함수: 합성 후보 4개(미러 2쌍)에서 정확히 2쌍 검출, 대표 선택이 결정론적.
- 캐시 재사용: E-A1 캐시의 키 하나로 F2 캐시 조회가 히트함(트레이스 digest 동일성 확인).

**완료 기준**: `f2_truth.md`에 실행가능 후보 수 ≥ 20, P/D 후보 중 실행가능 ≥ 1. P/D 실행가능이 0이면 TTFT SLO를 3-regime 표의 다음 값으로 **한 번만** 바꾸고 그 사실을 기록.

### STEP C2 — E-B1 재실행 on F2 (1일)

**만드는 것**
- `eb1_regret_vs_budget.py`에 `--fixture <json>`(기본 E-A1 json 유지 → 기존 결과 재현 가능), 층화 추출 `--stratified`, 링크 개별 열화 `--per-link`, "열화 상태에서 실행가능 0이면 오류" 게이트.
- 결과: `experiments/uncertainty/results/eb1_f2.md` — §2.3의 표 전부, 종류별 ΔR>0 항목 표(`n_active`), penalty 두 값, 그림 `figures/eb1_f2_regret_vs_budget.png`(x: 예산 h, y: 후회, 5곡선, 시드 평균±표준편차; 결정이 움직인 집합만).

**테스트** `tests/test_uq_eb1_harness.py`
- 층화 추출: k=2, 종류 3개 풀에서 100회 추출해 단일 종류 집합이 0개.
- 실행가능 0 게이트: 도메인을 넘기지 않은 상태로 호출하면 `SystemExit`/예외.
- 기존 E-A1 fixture로 실행 시 `eb1_summary.json`의 `ours` 곡선이 커밋된 값과 동일(회귀).

**완료 기준**: `eb1_f2.md`에 `n_active` 종류 수 기록. 2 이상이면 "우월성" 표 작성; 1이면 "F2도 퇴화, 이유(어느 종류가 왜 비활성인지)" 기록 후 C3로 진행.

### STEP C3 — E-B2 오탐 분류와 재정의 (1일)

- `eb2_flip_detection.py`에 `--fixture`, FP/FN 개별 행에 §2.4 분류 열(`alpha|beta|gamma`) 자동 부여(α: 진실이 nominal과 같은 쪽 & 범위 내 교차 존재; γ: `equivalent_candidates` 병합 후 동일; β: 나머지) + "가능 전환" 기준 채점 `--truth-sweep`.
- E-A1 fixture의 기존 FP 17건 분류 결과와 F2 결과를 `eb2_f2.md`에. 종류별 표 필수.
- 테스트: 합성 케이스 3개(α, β, γ 각 1)가 올바르게 분류됨.

### STEP C4 — E-B3 재실행 on F2 (0.5일 + 재시뮬 2~3 h)

- `eb3_closed_form_vs_resim.py`에 `--fixture`, `--exclude <mirror_pairs.json>`(C1 산출물), 활성 항목 수 < 3이면 Spearman을 `None`으로 두는 규칙, LINK_BW 재시뮬 일치 검사(정확 규칙 identity control).
- ×1.0 identity control은 그대로 선행 게이트. 1e-6 이내 일치하지 않는 후보 목록은 C1의 미러 쌍 목록과 **정확히 일치해야** 하며, 불일치가 있으면 D40 외의 원인이므로 중단하고 보고.
- 결과 `eb3_f2.md`: 항목별 (닫힌 형식 ΔR, 재시뮬 ΔR — 미러 제외 전/후, 크기 오차 %), 활성 항목 순서 보존 여부, Spearman(조건부), 소요 시간과 배속, `approximation` 플래그가 `False`로 바뀐 항목 수, `sim_error` SKIPPED 표기. E-A1 fixture의 14,298 → 20,583 결과는 비교용으로 병기.
- `planner/` 변경은 `resimulate.py`에 후보 제외 인자를 추가하는 것으로 한정(A4). 캐시 키(D40 자체) 수정 금지 — 그 결정은 별도.

### STEP C5 — 결과 정리와 특허 3 증거 매핑 (0.5일)

- `docs/uncertainty_planner.md` §4에 F2 결과 요약 추가, §5 한계 갱신(F2에서도 비활성인 종류가 있으면 명시).
- `docs/deviations.md`에 새 항목: F2 fixture 정의와 SLO 선택 근거, 층화 추출 결정, D40 미러 제외.
- 아래 §5 표를 채운 `experiments/uncertainty/results/patent3_evidence.md`.

---

## 4. 일정

| STEP | 예상 | 비고 |
|---|---|---|
| C0 | 0.5일 | |
| C1 | 1일 + 시뮬 2 h | CPU 8 워커 이상 권장 |
| C2 | 1일 | |
| C3 | 1일 | |
| C4 | 0.5일 + 재시뮬 ~8 코퍼스 | |
| C5 | 0.5일 | |

총 4.5일 + 시뮬레이션. 특허 2 신고서 보완과 병행 가능(서로 파일이 겹치지 않음).

---

## 5. 특허 3(측정 계획) 증거 매핑

| 신고서 항목(예정) | 근거 산출물 | STEP |
|---|---|---|
| 배경: 오차가 크더라도 결정을 바꾸지 않는 입력이 있다 | E-B2 `link_bw` 336건 TN(E-A1 fixture), F2에서의 종류별 활성/비활성 표 | C2·C3 |
| 구성: 캐시 예측의 닫힌 형식 후처리로 섭동 (속도) | E-B3(E-A1): 0.287 s vs 2,090.8 s, **7,285×**; F2 재실행 값 병기 | 완료·C4 |
| 한계: 닫힌 형식은 순서 보존, 크기는 과소 | E-B3(E-A1): 활성 항목 −30.5 %(D40 미분리); F2에서 D40 분리 후 항목별 크기 오차 | C4 |
| 구성: ΔR_i 정의(실행불가 = 최선 실행가능 − penalty×overshoot) | `docs/uncertainty_planner.md` §2.5, B2 테스트 | 완료 |
| 구성: ΔR/비용 정렬·예산·undecidable | `measurement_plan.py`, `costs.yaml` | 완료 |
| 효과: 동일 예산에서의 후회, 후회 0 도달 예산 | `eb1_f2.md` 표·그림 (+ E-A1 fixture 결과는 "단일 활성 항목 사례"로 병기) | C2 |
| 효과: 전환 예측의 정밀도·재현율 (실현/가능 기준 병기) | `eb2_f2.md` | C3 |
| 한계 명시 | penalty 민감도, 균등 가중, 1차원 도메인 | C2·C5 |

**작성 규칙**: 신고서 §6 수치는 이 표의 파일에서만 인용한다.

---

## 6. 위험

| 위험 | 대응 |
|---|---|
| F2에서도 활성 종류가 1개 | 그대로 기록. 특허 3 §6은 "단일 활성 항목에서 오라클과 일치, 다중 활성은 시뮬레이션 사례로 제시"로 쓰고, 다중 활성 fixture는 합성 클러스터(Stage D 범위)로 넘긴다 |
| P/D 시뮬 타임아웃·KV 할당 실패(HANDOVER 2.7의 68–114 후보) | `SIM_ERROR`로 분류되어 진실에서 빠짐. 개수를 `f2_truth.md`에 기록. 고치지 않음 |
| 캐시 재사용 실패(trace digest 불일치) | 162개 재시뮬(+25 m). 원인을 기록 |
| D40 미러 쌍이 승자에 포함 | 대표 하나만 인정하고 기록. 캐시 키 수정은 별도 PR로 제안만 |
| F2에서도 재시뮬 가능한 활성 항목이 1개 | 크기 오차만 보고하고 순위 지표는 "계산 안 함". 특허 3에는 "닫힌 형식은 우선순위 결정용, 절약량 인용 불가"로 한계를 명시 |
