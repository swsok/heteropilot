# HeteroPilot 작업지시서 — 정확도 도메인 적용 조건 스코핑과 측정 계획 등급 기본 범위 (domain-scoping)

> V3(P1, A40 TP=4)가 드러낸 두 결함을 코드에 반영한다. (1) `AccuracyDomain`이 병렬화·장치 배치 축을 표현하지 못해 TP=1로 적합한 도메인이 TP=4 후보에 조회되었고, 발명의 보류 계약이 발화하지 못했다. (2) 측정 계획이 출처 범위 없는 VENDOR_SPEC 항목을 "판정 불가"로 분류해 ΔR 순위 밖으로 밀어냈는데, 그 항목이 오차 전부를 설명하는 링크였다. 두 결함은 각각 특허 2(필수 적용 조건 → 판정 보류)와 특허 3 후보(측정 계획)의 실시예와 구현을 일치시키는 일이다.
> 대상 저장소: `github.com/swsok/heteropilot` · 작성일: 2026-09-17 · 기준 `main` = PR #98·#99·#104 머지 후
> 도구: Claude Code. **S1~S5 전부 CPU 노드.** S6만 A40 반나절(선택), S7은 RNGD 노드가 잡힐 때(선택).
> 선행 문서: `experiments/p2_evidence/results/{v0_source_reconciliation,v1_validation_region,v2_openloop_concurrency,v3_verdict_accuracy,patent2_evidence_map}.md`, `docs/uncertainty_planner.md`(stale — S5에서 갱신), `WORK_ORDER_uncertainty_planner.md` §2.3(등급)·§2.4.1 rev2(도메인 키), `WORK_ORDER_uq_stage_b_plus.md`, `docs/deviations.md` D32·D33·D40·D70
> 관련 결정(사용자, 2026-09-17): 병렬화 축은 **독립항에 넣지 않고 종속항·상세설명**에 둔다. 코드는 그 상세설명과 일치해야 한다. 라우터/다중 엔진(Phase 4)은 이 지시서 범위 밖.

---

## 0. 사용법과 규칙

1. S1 → S2 → S3 → S4 → S5 순서. S1과 S2는 코드가 겹치지 않으므로 병행 가능. S6·S7은 노드가 잡힐 때.
2. 한 STEP = 한 브랜치 = 한 PR. 브랜치 `feat/ds-s<N>-<짧은이름>`.
3. 절대 규칙 A1~A4(`WORK_ORDER_uncertainty_planner.md`) 적용. 특히 **A3**: `profiles/calibration/*.yaml`의 `points` 값은 바꾸지 않는다. 이 지시서가 요구하는 것은 **메타데이터 필드 추가**뿐이며, 추가 시 `note`에 "적용 조건 필드 2026-09 추가, 측정 당시 구성 기록" 한 줄을 남긴다. 클러스터 설정의 vendor 값도 덮어쓰지 않고 `source: measured` 항목을 새로 둔다.
4. 용어는 신고서 v5를 따른다: **필수 적용 조건**(모델·수치 정밀도·병렬화 구성·장치 배치·도착 과정 등 일치해야 조회 가능한 조건), **검증 영역**(동작점 좌표 구간), **판정 보류**. 코드에서 두 가지 보류 사유를 **구분**한다 — 적용 조건 불일치는 검증 영역 밖과 다른 사유다(§S1).
5. 결과가 가설과 다르면 그대로 기록한다. 특히 S4의 E-A1 재실행에서 통과 수가 어떻게 바뀌든 그 수가 명세서 수치가 된다.

---

## 1. 무엇을 고치는가

| 결함 | 어디서 드러났나 | 고칠 곳 | STEP |
|---|---|---|---|
| 도메인 스코핑에 병렬화·장치 배치 축 없음 | V3 P1: TP=1 도메인(마진 1.13 %)이 TP=4 후보에 조회, 실측 오차 −44.6 %(NUMA 미고정), refuse 미발화 | `planner/predictor/calibration.py` `AccuracyDomain`, `planner/uncertainty` 도메인 키, `RejectionStage` | S1 |
| 적용 조건 불일치와 검증 영역 밖이 한 사유(`OUTSIDE_CALIBRATION_DOMAIN`)로 뭉개짐 | E-A1 refuse 30건은 전부 좌표 상한 초과. 조건 불일치 사유는 코드에 없음 | `RejectionStage`, `exhaustive.judge` 출력, 결과 md 집계 | S1 |
| 출처 범위 없는 항목이 측정 계획에서 순위 밖 | V3: `pcie-a40a-02` VENDOR_SPEC → "판정 불가", 비용 0.114 h인데 오차 전부 설명 | `planner/uncertainty/{grades,measurement_plan}.py`, `grades.yaml` | S2 |
| LINK_BW 항목의 의미가 "명세 대역폭"이라 배포 상태(NUMA 고정)를 반영 못 함 | 같은 링크, 고정 여부로 처리량 1.93배(PR #104) | `planner/uncertainty/registry.py` LINK_BW 키·의미, 클러스터 설정 스키마 | S3 |
| 구현·문서·실시예 불일치 | `docs/uncertainty_planner.md`가 B4 미실행·resimulate 미구현으로 기술(stale); evidence map에 V3 후속 결론 미반영 | 문서 | S5 |

---

## 2. STEP

### STEP S1 — AccuracyDomain 적용 조건 필드와 조건 불일치 보류 (CPU, 1.5일)

**설계**
- `AccuracyDomain`에 필드 추가: `parallelism: {tp: int, pp: int, dp: int}`, `placement: {islands: int, device_binding: "numa_pinned" | "unpinned" | "unknown"}`, `hardware: str`(기존 파일명에 암시된 것을 명시). 기존 필드(`model, variant, workload_shape, arrival_process, fitted_at_concurrency, points, outside_domain, source, note`)는 유지.
- **일치 규칙**(적용 조건 검사): 후보의 `(hardware, model, variant, tp, pp, dp, islands, arrival_process)`가 도메인과 **정확히** 같아야 조회한다. `device_binding`은 도메인이 `unknown`이면 검사하지 않고 결과에 경고 플래그를 남긴다(기존 파일은 전부 `unknown`으로 시작). `dp`는 섬 수와 함께 비교 — 미러 문제(D40)와 분리해 다룬다.
- **새 보류 사유** `RejectionStage.CALIBRATION_CONDITION_MISMATCH`(적용 조건 불일치). 기존 `OUTSIDE_CALIBRATION_DOMAIN`은 **검증 영역 밖(좌표)** 전용으로 남긴다. 판정 결과에 `mismatch_fields: [...]`(어느 조건이 어긋났는지)와 `required_measurement: {tp, pp, dp, islands, binding}`(신고서의 "추가 측정 조건")을 기록.
- **정책 키** `calibration.condition_mismatch: refuse | warn` — 기본 **`refuse`**(D33의 `outside_domain: refuse`와 대칭). `warn`은 실험 비교용.
- **도메인 파일 선택**: 후보의 조건으로 도메인 파일을 **찾는** 것이 아니라(그러면 후보마다 파일 탐색), 후보 조건과 등록된 도메인들의 조건을 비교해 일치하는 도메인이 없으면 불일치 보류. `profiles/calibration/index.yaml`(신규)에 도메인 목록과 조건을 둔다.
- 도메인 키(`WORK_ORDER_uncertainty_planner.md` §2.4.1 rev2의 `(hw, model, dtype, in, out)`)에 `(tp, pp, dp, islands)`를 추가. 캐시 무효화 영향 확인(EnvelopeCache 키는 이미 placement를 포함하므로 도메인 키만).

**기존 파일 메타데이터 보강**(A3: `points` 불변)
- `a40.accuracy.yaml`, `a40.accuracy.openloop.yaml`: `tp:1, pp:1, dp:1, islands:1, device_binding: unpinned`(PR #104 사다리는 고정 상태로 재검증됐으므로 `note`에 "고정 유무 차이 <0.5 %, PR #104" 기재; 값은 `unpinned`로 두고 재검증 사실만 적음).
- `rngd_card_edf.yaml`: 단일 카드 `tp:1, pp:1, dp:1, islands:1, device_binding: unknown`.
- 파일별로 원자료(측정 로그·배포 명령)에서 병렬 구성을 **확인한 뒤** 적는다. 확인 안 되면 `unknown`과 그 이유.

**테스트** `tests/test_calibration_condition.py`
- (i) 도메인 tp=1, 후보 tp=4 → `CALIBRATION_CONDITION_MISMATCH`, `mismatch_fields=["tp"]`, 좌표가 검증 영역 안이어도 보류.
- (ii) 조건 일치 + 좌표 상한 초과 → `OUTSIDE_CALIBRATION_DOMAIN`(기존 동작 유지).
- (iii) 조건 일치 + 좌표 안 → 마진 조회 정상(E-A1 (c) 회귀 0 변경).
- (iv) `device_binding: unknown` 도메인 → 검사 생략 + 경고 플래그.
- (v) `condition_mismatch: warn` → 통과하되 경고 기록.
- (vi) 기존 yaml 로딩: 필드 없는 옛 파일도 로드되고 `unknown` 처리(하위 호환).

**산출물** 코드 + `profiles/calibration/index.yaml` + 보강된 yaml 3개 + `docs/deviations.md`에 D-번호(적용 조건 불일치 사유 분리).

### STEP S2 — 등급별 기본 출처 범위와 측정 계획 순위 복원 (CPU, 1일)

**할 일**
- `grades.yaml`에 등급별 **기본 상대 범위** 추가: 항목에 출처 범위가 없을 때 쓰는 `[lo, hi]`(명세값 대비 배수). 초기값과 근거를 파일에 주석으로: VENDOR_SPEC은 **[1/8, 1]**(V3: 유효 all-reduce 대역폭이 명세의 약 1/8 — n=1 근거임을 명시, 보수적 폭), SIM_PROXY·PROFILE 등은 기존 E-B2 관측 오차에서 유도, MEASURED는 반복 스프레드(기본 ±2 %). 값은 설정이며 이 STEP은 값의 정당화가 아니라 **"범위 없음 → 순위 밖" 경로를 없애는 것**이 목적.
- `measurement_plan.py`: 범위 없는 항목은 기본 범위를 적용해 ΔR을 계산하고 `range_source: default`로 표기. "판정 불가" 분류는 **기본 범위마저 정의되지 않은 등급**에만 남긴다.
- 회귀: E-B1·E-B2·E-B3 재실행 — 기본 범위가 없던 항목이 순위에 들어오며 상위 순위가 어떻게 바뀌는지 표. 기존 결과 md는 덮어쓰지 않고 `*_s2.md`로.
- **V3 재현 검사**: P1 후보의 레지스트리 6개 항목에 대해 새 계획을 돌려 `pcie-a40a-02`가 ΔR/비용 기준 **몇 위**인지 기록(가설: 1위. 아니면 그대로 기록하고 왜 아닌지).

**테스트** `tests/test_default_range.py` — (i) 범위 없는 VENDOR_SPEC 항목이 순위에 포함, `range_source=default`; (ii) 범위 있는 항목은 기본 범위를 쓰지 않음; (iii) 기본 범위 미정 등급만 "판정 불가"; (iv) E-B2 recall/precision 회귀 변화 없음(기본 범위는 순위에만 영향, flip 판정에는 비영향 — 아니면 기록).

**산출물** `experiments/uncertainty/results/s2_default_ranges.md`(V3 P1 항목 순위 표 포함).

### STEP S3 — LINK_BW 항목 의미 재정의: 배포 상태의 유효 집합통신 대역폭 (CPU, 1일)

**할 일**
- 레지스트리 LINK_BW 항목의 의미를 "명세 링크 대역폭"에서 **"해당 배포(병렬 구성·장치 배치·바인딩)에서의 유효 집합통신 대역폭"**으로 바꾸고, 키에 `(link_id, collective, msg_size_class, device_binding)`을 포함. 같은 링크라도 고정/미고정이 다른 항목.
- 클러스터 설정 스키마: 링크에 `measurements: [{source: measured, method: nccl-tests all_reduce_perf, binding, msg_sizes, bus_bw, date, raw}]` 배열 추가. vendor 값은 그대로(A3). PR #104 및 V3 후속의 nccl-tests·p2pBandwidthLatencyTest 결과를 이 형식으로 **적재**(측정은 이미 되어 있으면 값만 옮김; 없으면 S6).
- 시뮬레이터 입력: 후보의 배치·바인딩에 맞는 측정값이 있으면 그것을, 없으면 vendor 값을 쓰고 레지스트리 항목 등급을 VENDOR_SPEC으로 둔다(그래야 S2의 기본 범위가 붙어 측정 계획에 오른다).
- `measure-apply` 경로가 이 새 항목을 소비하는지 확인(Stage B `measure-apply`는 측정 결과를 레지스트리에 반영하는 명령).

**테스트** `tests/test_link_effective_bw.py` — (i) 바인딩이 다르면 다른 항목, (ii) 측정값 존재 시 시뮬레이터 입력이 측정값, (iii) vendor 값 파일 불변.

**산출물** 스키마 변경 + 적재된 측정값 + `docs/uncertainty_planner.md` 해당 절.

### STEP S4 — E-A1 재실행과 V3 P1 재판정 (CPU, 0.5일)

**할 일**
- E-A1(fixture `pd-rngd-gpu-card`, TPOT SLO 50 ms)을 S1 반영 상태로 재실행. 네 규칙 (a)(b)(c)(d)에 **(e) 동작점별 + 검증 영역 밖 보류 + 적용 조건 불일치 보류**를 추가. 표: 규칙별 통과 / 미통과 / 보류(사유별: 검증 영역 밖, 적용 조건 불일치). 기존 E-A1의 50/244/30은 **그대로 두고** 새 표를 병기(캐시 진실 불변).
- 가설: A40 `tp≥2` 또는 다중 섬 후보가 전부 적용 조건 불일치로 보류되고, 통과 수가 50에서 줄어든다. 줄어든 수와 어떤 후보가 빠졌는지(P1 포함 여부) 기록. RNGD 후보는 단일 카드 도메인과 조건이 일치하는지 후보 구성별로 확인.
- V3 P1 재판정: SLO 스윕 38~50 ms 표에 규칙 (e) 열 추가 — 예상 결과는 모든 SLO에서 "보류(적용 조건 불일치: tp)". 이것이 신고서 §2·§5.2에 매핑된 V3 사례의 **결말 문장**이 된다.
- PR #104의 NUMA 고정 후 P1 실측이 있으면(TPOT p99, L) 시뮬레이터(측정 링크값 적용) 예측과의 오차를 한 줄 병기. 없으면 S6.

**산출물** `experiments/uncertainty/results/ea1_s4_condition_refuse.md`, `experiments/p2_evidence/results/v3_addendum_s4.md`.

### STEP S5 — 문서 정합 (CPU, 0.5일)

- `docs/uncertainty_planner.md` 전면 갱신(stale 항목 제거: B4 미실행, resimulate 미구현; S1~S3 반영).
- `patent2_evidence_map.md` 갱신: (i) 신고서 §5.2 적용 조건 ↔ S1 구현·S4 결과, (ii) §8(3) 판정 보류 사유 구분 ↔ `CALIBRATION_CONDITION_MISMATCH`, (iii) V3 P1 결말 문장, (iv) 특허 3 후보 실시예 한 줄: "레지스트리 지목 → 0.114 h 측정 → 원인 확정(NUMA 배치·유효 대역폭 ≈ 명세 1/8) → 이전 측정 재검증(PR #104, <0.5 %)". 이 표 밖의 수치는 명세서에 쓰지 않는다는 규칙 유지.
- `docs/deviations.md`: S1(사유 분리), S2(기본 범위), S3(LINK_BW 의미) 각 D-항목.

### STEP S6 (선택, A40 반나절) — 고정 배포 P1 재실측과 nccl-tests 적재

PR #104에 아래가 이미 있으면 생략. 없는 것만: (i) NUMA 고정 P1(TP=4) 개루프 실측 3회 → S3 측정 링크값 적용 시뮬레이터 예측과 오차; (ii) `nccl-tests all_reduce_perf` 4 GPU(TP 메시지 크기 범위)·`p2pBandwidthLatencyTest`, 고정/미고정 각각 → S3 형식으로 적재; (iii) 같은 후보 TP=2를 NV4 쌍(GPU0↔1) 안에서 실측 → 시뮬레이터가 NVLink 경로에서 맞는지(TP 연산 모델 가설 배제). 워크로드는 정상상태 구간이 생기도록 길이를 늘리고 워밍업 제외(V3의 과도상태 교훈).

### STEP S7 (선택, RNGD 노드) — V3-R

RNGD 후보로 P2(무마진 통과·동작점별 미통과)·P3(전역 18 % 미통과·동작점별 통과) 각 1개, 정상상태 워크로드, p99/p99. 판정을 바꾸는 사례는 RNGD 영역(마진 ≈21 %)에만 있으므로, 이것이 신고서 §6 수치의 유일한 공급원이다. 노드가 잡히면 이 STEP만 별도 rev로 상세화한다.

---

## 3. 일정과 자원

| STEP | 노드 | 예상 |
|---|---|---|
| S1 | CPU | 1.5일 |
| S2 | CPU | 1일(S1과 병행 가능) |
| S3 | CPU | 1일 |
| S4 | CPU | 0.5일 |
| S5 | CPU | 0.5일 |
| S6 | A40 | 반나절(선택) |
| S7 | RNGD | 반나절~1일(선택) |

CPU 합계 약 4일. 정규 명세서 초안(변리사 전달용 자료)은 S5 evidence map이 나온 뒤 작성한다.

---

## 4. 위험

| 위험 | 대응 |
|---|---|
| S1의 정확 일치 규칙이 너무 엄격해 대부분 후보가 보류됨(도메인이 tp=1뿐) | 그것이 현 측정 상태의 정직한 결과. 보류 결과에 `required_measurement`가 붙으므로 "무엇을 재야 하는가"가 나온다 — 이것이 신고서의 추가 측정 조건 출력이다. 완화(예: tp 무시)는 하지 않는다 |
| 기존 yaml의 병렬 구성을 원자료에서 확인 못 함 | `unknown`으로 두고 이유 기록. `unknown`은 검사 생략 + 경고이므로 기존 동작 유지 |
| S2 기본 범위 값이 자의적 | 값은 설정이고 근거는 주석에. 목적은 순위 밖 경로 제거. 값 정당화는 후속(측정 누적) |
| E-A1 재실행에서 통과 수가 크게 줄어 신고서 §6의 50/244/30과 어긋남 | 기존 표는 규칙 (d)까지의 결과로 유효. 새 표는 규칙 (e)로 병기. 명세서에는 둘 다 쓰되 규칙 정의를 명시 |
| S3 키 변경으로 Stage B 결과(E-B1~3) 재현 불가 | 기존 결과 md 불변, `*_s2.md`·`*_s3.md`로 새 결과. 키 변경은 deviations에 기록 |
| S6·S7 노드 미확보 | S1~S5만으로 지시서 완결. S6·S7은 evidence map에 "미실시"로 남김 |
