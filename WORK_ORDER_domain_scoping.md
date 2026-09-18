# HeteroPilot 작업지시서 — 정확도 도메인 적용 조건 스코핑과 측정 계획 등급 기본 범위 (domain-scoping)

> V3(P1, A40 TP=4)가 드러낸 두 결함을 코드에 반영한다. (1) `AccuracyDomain`이 병렬화·장치 배치 축을 표현하지 못해 TP=1로 적합한 도메인이 TP=4 후보에 조회되었고, 발명의 보류 계약이 발화하지 못했다. (2) 측정 계획이 출처 범위 없는 VENDOR_SPEC 항목을 "판정 불가"로 분류해 ΔR 순위 밖으로 밀어냈는데, 그 항목이 오차 전부를 설명하는 링크였다. 두 결함은 각각 특허 2(필수 적용 조건 → 판정 보류)와 특허 3 후보(측정 계획)의 실시예와 구현을 일치시키는 일이다.
> 대상 저장소: `github.com/swsok/heteropilot` · 작성일: 2026-09-17 · 기준 `main` = PR #98·#99·#104 머지 후
> 도구: Claude Code. **S1~S5 전부 CPU 노드.** S6만 A40 반나절(선택), S7은 RNGD 노드 약 2.5~3.5일 + CPU 1일.
> **개정 rev 2 (2026-09-18).** S1~S5 완료(PR #106 S1 · #107 S2 · #109 S3 · #112 S4·S5) 후 RNGD 노드가 잡혀 S7을 상세화했고, 상세화 과정에서 초안 S7의 전제 세 개가 이미 무효임이 드러났다 — (1) S4가 명시한 `arrival_process: closed_loop`(D113) 때문에 기본 정책에서 **RNGD 후보는 마진이 계산되기 전에 전부 보류**되어 ≈21 % 마진을 조회할 수 없고, (2) V3의 P2·P3은 둘 다 A40 혼합 후보여서 이 노드에서 글자 그대로 실행할 수 없으며, (3) `planner/deploy/`에 **furiosa 백엔드가 없다.** S7은 반나절짜리 측정이 아니라 **하네스 포팅 + 개루프 재적합 + 본측정 약 4일**이 되고, §S7이 그것을 하위 STEP으로 나눈다. S1~S6과 §1 표는 손대지 않았다.
> 선행 문서: `experiments/p2_evidence/results/{v0_source_reconciliation,v1_validation_region,v2_openloop_concurrency,v3_verdict_accuracy,patent2_evidence_map}.md`, `docs/uncertainty_planner.md`(stale — S5에서 갱신), `WORK_ORDER_uncertainty_planner.md` §2.3(등급)·§2.4.1 rev2(도메인 키), `WORK_ORDER_uq_stage_b_plus.md`, `docs/deviations.md` D32·D33·D40·D70
> 관련 결정(사용자, 2026-09-17): 병렬화 축은 **독립항에 넣지 않고 종속항·상세설명**에 둔다. 코드는 그 상세설명과 일치해야 한다. 라우터/다중 엔진(Phase 4)은 이 지시서 범위 밖.

---

## 0. 사용법과 규칙

1. S1 → S2 → S3 → S4 → S5 순서. S1과 S2는 코드가 겹치지 않으므로 병행 가능. S6·S7은 노드가 잡힐 때.
   **S7은 rev 2에서 하위 STEP으로 나뉘었고 순서가 고정이다**: S7.0 → (S7.1은 결정) → S7.2 → S7.3 → S7.4 → S7.5. S7.0은 CPU만 필요하므로 카드 없이 먼저 끝내며, **선정 결과가 확정되기 전에는 장치를 만지지 않는다**(V3의 관례).
2. 한 STEP = 한 브랜치 = 한 PR. 브랜치 `feat/ds-s<N>-<짧은이름>`. **S7의 하위 STEP도 각각 한 브랜치·한 PR**이며 `feat/ds-s7-<0..5>-<짧은이름>`을 쓴다 — 하네스 포팅(S7.2)과 측정(S7.3·S7.4)이 한 PR에 섞이면 회귀 근거가 분리되지 않는다.
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

### STEP S7 (RNGD 노드) — V3-R: 판정을 바꾸는 사례를 RNGD 영역에서 측정한다

> **rev 2에서 상세화.** 초안의 S7은 한 문단이었고 "노드가 잡히면 별도 rev로
> 상세화한다"고만 적었다. 노드가 잡혔고(2026-09-18, `whichnode.sh` → `npu`,
> RNGD 3장), 상세화해 보니 **초안의 전제 세 개가 이미 무효**다. 아래 §S7.A가
> 그것이고, §S7.0~S7.5가 무효가 아닌 부분을 실행 가능한 절차로 바꾼 것이다.

#### S7.A 초안이 쓰인 뒤 무효가 된 전제

**(1) ≈21 % 마진은 오늘 어느 후보에도 적용되지 않는다.** S4가
`rngd_card_edf.yaml`에 `arrival_process: closed_loop`을 명시했고(D113), `python -m
planner plan`은 언제나 도착 트레이스를 재생한다 — 즉 `open_loop`이다. 기본 정책
`--condition-mismatch refuse`에서 **모든 RNGD 후보는 마진이 계산되기 전에**
`calibration_condition_mismatch`로 보류된다(E-A1 재실행: 42건이
`arrival_process`에서 불일치, 그중 12건이 단일 카드). 초안은 "판정을 바꾸는
사례는 RNGD 영역에만 있다"고 옳게 적었지만, **그 영역을 조회할 방법이 지금
없다.** 이것은 S7의 목표가 아니라 경로가 무효가 된 것이다.

**(2) P2·P3은 이 노드에서 글자 그대로 실행할 수 없다.** V3이 고른 두 후보는
`mix(a40a-tp2-dp1+a40b-tp2-dp2)`와 `mix(a40a-tp1-dp1+a40b-tp1-dp4)` — **둘 다
A40 혼합**이고 이 노드에는 NVIDIA 드라이버가 닿지 않는다. 절대 규칙 3에 따라
여기서 재현할 대상이 아니다. **P2·P3은 위상(topology) 형태가 아니라 판정
형태(verdict shape)로 읽어야 한다** — A40에서 혼합이었던 것은 그쪽에서 분리폭이
거기밖에 없었기 때문이고(A40 per-point 마진 0.37~1.63 %), RNGD에서는 §S7.B가
보이는 대로 단일 카드 후보로 두 형태가 모두 성립한다:

| 형태 | 정의 | 무엇을 보이는가 |
| --- | --- | --- |
| **P2-형** | (a) 무마진 통과 · (c) 동작점별 미통과 | 발명이 **잡아내는** 후보 |
| **P3-형** | (b) 전역 18 % 미통과 · (c) 동작점별 통과 | 발명이 **구제하는** 후보 |

**(3) `planner deploy`로는 RNGD 후보를 띄울 수 없다.** `planner/deploy/`에는
`vllm_cuda`, `vllm_ascend`(스텁), `kubernetes`뿐이고 **furiosa 백엔드가 없다.**
V3의 한계 ①(라우터·다중 엔진 미구현)의 RNGD 대응이며 그보다 더 나쁘다 — 형태가
아니라 벤더가 없다. 따라서 S7은 `deploy`를 쓰지 않고 `furiosa-llm serve` +
`experiments/scripts/replay_to_endpoint.py --open-loop`으로 측정한다. 백엔드
작성은 S7 범위 밖이고, 이 사실은 결과 파일에 한계로 적는다.

**(4) 기존 단일 카드 RNGD 후보는 검증 영역 밖이기도 하다.** E-A1의 12건은 서비스
동시성 **144.6**, 도메인 측정 범위는 **[1.02, 76]**이다. 조건을 맞춰도 좌표가
밖이면 규칙 (d)·(e)는 여전히 보류한다. **판정이 뒤집히는 사례는 영역 안에
있어야 하므로 후보를 다시 골라야 한다** — 초안에 없던 단계다.

#### S7.B 왜 RNGD여야 하는가 — 커밋된 값에서 나오는 산술

(a)와 (c)가 TPOT에서 갈리는 구간은 `(50/(1+m), 50]`이다. `m`은 후보 자신의
동작점에서 나오고, `rngd_card_edf.yaml`의 점들을 `calibration._err_at`이 하는 대로
선형 보간해 `m = -e/(1+e)`(한쪽 방향, D70)로 환산하면:

| 서비스 동시성 L | 보간된 e (%) | 마진 m (%) | 불일치 구간 (ms) | 구간 폭 |
| ---: | ---: | ---: | --- | ---: |
| 25.181 | −3.28 | 3.391 | (48.360, 50] | 1.640 |
| 30 | −4.68 | 4.905 | (47.662, 50] | 2.338 |
| 40 | −7.57 | 8.193 | (46.214, 50] | 3.786 |
| 50 | −10.47 | 11.693 | (44.766, 50] | 5.234 |
| **66.52** | −15.25 | **18.000** | (42.373, 50] | 7.627 |
| 76 | −18.00 | 21.951 | (41.000, 50] | 9.000 |

*표는 도출값이며 측정값이 아니다.* `[25.181, 76]`에서만 유효하다 — 그 밖에서는
보간 구간이 다른 두 점이고, L < 15.5에서 e는 양수여서 m = 0이다.

두 가지가 여기서 나온다:

1. **A40의 불일치 구간은 최대 0.80 ms였고**(`v3_candidate_selection.md` §1) 실제
   최선 후보는 0.3 ms로 갈렸다 — 측정 반복성 아래다. RNGD에서는 같은 구간이
   L ≈ 40에서 3.8 ms, L ≈ 66에서 7.6 ms다. **측정이 판정할 수 있는 폭이 처음으로
   생긴다**, 이것이 §6 수치가 RNGD에서만 나오는 이유의 정량적 진술이다.
2. **P3-형에는 상한이 있다.** (b)의 18 %가 미통과시키고 (c)가 통과시키려면
   `m < 18 %`, 즉 **L < 66.52**여야 한다. P2-형은 m > 0, 즉 L ≳ 15.5면 성립한다.
   **두 형태가 공존하는 창은 L ∈ 약 [25, 66]** 이고, 목표 창을 **L ∈ [30, 66]**으로
   둔다(양 끝에서 폭이 2.3~7.6 ms).

**창은 현 fixture의 10 rps가 만드는 지점이 아니다.** 거기서 단일 카드는 L = 144.6
이다. 창으로 내리는 수단은 `plan --rps`(rps_aware)다.

> **창 자체가 성립하지 않을 수 있고, 그것이 이 STEP의 정직한 실패 모드다.**
> `rngd_card_edf.yaml`의 `note`는 카드 모델이 **서비스 동시성 ≈44에서 포화**한다고
> 적는다(실측 c59.2·c107.2에 대응하는 rps에서 시뮬레이터는 300 요청에서도 37.67·
> 44.47에 머문다). 그 진술과 §S7.A(4)의 144.6은 **표면상 충돌하며**, 후자는 대기열이
> 자라는 과부하 지점의 L(체류시간 기반)일 것으로 보이지만 **확인되지 않았다.**
> S7.0은 이 충돌을 해소하고 기록해야 한다. 창의 상한이 실제로 ≈44라면 P3-형의
> 상한 66.52와 겹치는 구간은 [30, 44]로 좁아지고 폭은 2.3~4.3 ms다. 창이 L < 16
> 으로만 열린다면 **m = 0이고 어떤 형태도 존재하지 않는다** — 그 경우 S7은 "이
> fixture에서 판정 뒤집기 사례는 없다"를 결과로 보고하고 종료한다. 완화하지 않는다.

#### S7.0 후보 선정 (CPU, 어느 노드에서나, 0.5일) — **장치를 만지기 전에 확정**

V3의 구조를 그대로 따른다: 선정은 CPU 절반이고, 승인 전에는 배포하지 않는다
(`v3_candidate_selection.md`가 선례).

- fixture는 **`experiments/configs/clusters/pd-rngd-gpu-card.yaml`을 그대로 쓴다.**
  새 RNGD 전용 fixture를 만들지 않는다 — E-A1·V3과 같은 코퍼스여야 수치가
  비교 가능하고, 이 파일의 프로비넌스는 이미 검수되어 있다. A40 후보는 선정
  단계에서 걸러낸다(측정하지 않으므로).
- 후보 조건은 **`tp=1, pp=1, dp=1, islands=1`(단일 카드)로 고정**한다. 이유는 두
  가지이고 둘 다 필수다: (i) 카드 도메인이 그 구성에서 적합됐으므로 다른 구성은
  `arrival_process` 외에 `dp`·`islands`에서도 불일치해 §S7.1이 무엇을 고쳤는지
  보이지 않는다; (ii) furiosa 백엔드가 없어 손으로 띄우는데, 다중 섬은 라우터가
  필요하다(§S7.A(3)).
- `--rps`를 스윕해 단일 카드 후보의 **L과 예측 p99 TPOT를 표로 뽑고**, §S7.B의
  창 `L ∈ [30, 66]`(또는 포화 확인 후 좁혀진 창) 안에서 **P2-형 1개, P3-형 1개**를
  고른다. 각 후보에 대해 네 규칙 (a)~(d)의 판정과 분리폭을 함께 적는다.
- **분리폭을 V2의 실행 간 p99 산포와 나란히 적는다.** V3의 사용자 지시가 그대로
  적용된다: 산포보다 좁은 통과는 잡음 안이며, 그렇게 보고해야 한다. RNGD의 산포는
  아직 개루프로 측정된 적이 없으므로 §S7.3의 반복이 그것을 처음 제공한다.
- 스크립트 `experiments/p2_evidence/v3r_select_candidates.py`,
  산출물 `experiments/p2_evidence/results/v3r_candidate_selection.md`
  (+ `v3r_candidates.json`), 대안 후보 각 2개 포함(배포 실패 대비, V3 관례).
- **§S7.A(4)의 충돌을 여기서 해소한다**: 144.6과 포화 ≈44가 같은 양인지, 다르면
  어느 정의인지 한 단락으로 기록. 가설과 다르면 그대로 적는다(§0.5).

**S7.0은 지금 이 노드에서 실행 가능하다**(장치 불필요, 웜 캐시 + `--rps` 스윕은
시뮬레이션이 필요할 수 있으므로 `livelock_watch.sh`로 감싼다).

#### S7.1 게이트 — 조회 가능한 도메인을 어떻게 얻는가 (결정: **(A) 개루프 재적합 먼저**)

§S7.A(1) 때문에 S7에는 두 경로가 있고, `docs/HANDOVER.md` §2.10.1이 "프롬프트에서
결정할 일이 아니다"라고 남겨둔 선택이다. **이 rev가 (A)로 정한다.**

| | (A) 개루프 재적합 후 기본 정책으로 실행 | (B) `--condition-mismatch warn`으로 실행 |
| --- | --- | --- |
| 비용 | 하네스 포팅 + 재측정 (§S7.2·S7.3) | 없음 |
| §6에 들어갈 수치의 성격 | **출하 정책이 내는 판정** | 조건 검사를 **끈 상태**의 판정 |
| 위험 | 포팅이 이 STEP 비용의 대부분 | 신고서 §8(3)이 주장하는 기능을 끄고 얻은 근거 |

**(A)를 고르는 이유**는 비용이 아니라 근거의 일관성이다. 신고서 §8(3)의 주장은
두 보류 사유가 구분된다는 것이고, §6은 동작점별 마진이 판정을 바꾼다는 것이다.
조건 검사를 끄고 얻은 §6 수치는 **한 표 안에서 마진 주장을 지지하면서 스코핑
주장을 반증한다.** 게다가 (A)는 §2.10.1의 항목 1·2가 이미 요구하는 작업이고
같은 출장이다.

**단, (B)는 라벨된 대조군으로 함께 돌린다.** S4의 E-A1 표에서 규칙 (a)~(d)가
`warn`으로만 재현되는 것과 같은 구조다. 신고서가 인용하는 것은 **기본 정책 arm**
이고, `warn` arm은 "조건 검사가 없었다면 무엇을 판정했을 것인가"를 보이는 열로만
쓴다. 두 arm을 같은 표에 두고 어느 쪽이 기본인지 명시한다.

#### S7.2 furiosa 개루프 하네스 (카드 필요, 1.5~2일) — **실행됨, D114로 경로 변경**

> **개정 rev 3 (2026-09-18).** 아래 초안은 "`measure_envelope_openloop.py`에
> `furiosa` 백엔드를 추가한다"였고 **그 파일은 그 변경을 담을 수 없다.** 그 파일의
> 본체는 `python -m bench run`이고 `bench/core/runner.py`는
> `vllm.v1.engine.async_llm.AsyncLLM`을 **인프로세스로** 구동한다 — furiosa-llm은
> 서버이고 AsyncLLM 등가물이 없다. §2.2가 찾은 세 이음매(서버 기동·샘플러·인터프리터)가
> 이 파일에는 **존재하지 않는다**: 엔진이 파이썬 객체이므로 바꿀 기동 줄이 없다.
> `bench/`에 두 번째 엔진 경로를 넣는 것은 절대 규칙 1(Phase 5 전 upstream 불변)이
> 막는다. **전체 근거는 `docs/deviations.md` D114.**
>
> 실제로 한 일: **개루프 *프로토콜*을 `measure_envelope.py`에 추가**했다. 그 파일이
> 이미 백엔드별 기동 줄(`server_command`), 서버·클라이언트 양쪽 NUMA 바인딩
> (`numa_prefix`, 기본 `auto`), 전력 샘플러, settle/idle 창, 프로세스 그룹 종료를
> 갖고 있어서, `--mode open`이 바꾸는 것은 **부하 생성기 하나**다 —
> `bench_furiosa_endpoint.py --concurrency N` → `replay_to_endpoint.py --open-loop
> --target-rps R`. `--mode closed`가 기본이므로 커밋된 호출은 전부 그대로 돈다.

| 경로 | 프로토콜 | 하네스 | 백엔드 |
| --- | --- | --- | --- |
| `measure_envelope.py --mode closed` | 폐루프 | 배포 서버 + HTTP | `furiosa`, `cuda` |
| **`measure_envelope.py --mode open`** | **개루프** | **배포 서버 + HTTP** | **`furiosa`, `cuda`** |
| `measure_envelope_openloop.py` | 개루프 | 인프로세스 `bench run` | `cuda`만 (이식 불가) |

**이 경로가 A40의 어느 도메인과 비교 가능한지가 S7.3을 좌우한다.** A40에는 개루프
도메인이 **둘** 있고 D102가 하네스가 달라서 파일을 분리해 둔다 —
`a40.accuracy.yaml`(bench-run)과 `openloop/a40.accuracy.openloop.yaml`(배포 서버 +
HTTP). RNGD 재적합은 **후자와** 프로토콜·하네스가 모두 같아지므로, 그 파일의 실측
주의사항을 물려받는다: HTTP 경유에서 **TPOT는 전송 비용이 없고**(클라이언트 36.64 ms
대 엔진 36.62, +0.06 %) **TTFT는 클라이언트측 전송 +20.26 ms**(그 부하에서 +9.5 %)를
포함하며 값은 보정하지 않고 원시로 기록된다. 엔진측 수치와 비교하는 소비자는 이
오프셋을 감안해야 한다.

실행 내역:

- 클라이언트는 새로 쓰지 않았다. `replay_to_endpoint.py --open-loop`이 이미 요청별
  도착·최초 토큰·완료 타임스탬프와 `L_meas = Σ residency / window`를 기록하고,
  평범한 OpenAI 호환 클라이언트이므로 `furiosa-llm serve`를 그대로 구동한다.
  `--target-rps`가 트레이스 자신의 간격을 재조정하므로 rate별 재간격 사본을 쓰지
  않는다. **`--ignore-eos`는 선택이 아니다**(V2 §1) — 테스트가 고정한다.
- **`--numa-bind`는 공짜로 따라왔다.** 이미 구현돼 있고 기본이 `auto`다. §2.10.1
  항목 2가 요구한 "첫 커밋부터"를 규율이 아니라 **구조로** 만족한다 — 카드를 두 번
  재지 않는다. 노드 해석은 `furiosa-smi info`의 BDF →
  `/sys/bus/pci/devices/<bdf>/numa_node`(`/sys/class/rngd_mgmt/*`에는 PCI 부모가
  없어 조용히 실패한다), `taskset -cp`로 검증. RNGD 3장은 모두 **NUMA 노드 0**.
- **포화 판정은 다른 양이고 이름도 다르다.** A5(b)는 폐루프 규칙이다. bench-run
  경로는 엔진의 `scheduled_ts − queued_ts`로 대체했지만 HTTP 서버는 그것을 내주지
  않으므로, `ttft_drift_slope`는 **클라이언트측 TTFT**(대기 + prefill + 전송)의
  기울기를 읽는다. 절편은 비교 불가, **기울기는 비교 가능**이며 임계값
  `QUEUE_GROWTH_SLOPE = 0.05`은 일부러 공유한다. 크지만 평평한 TTFT가 포화로 읽히면
  안 되고 그것을 고정하는 테스트가 있다.
- 지연·동시성은 **재계산하지 않는다**. `replay_to_endpoint`가 이미
  `planner/util/percentile.py`로 산출하므로 두 번째 보간을 만들지 않는다. 이 파일이
  더하는 것은 클라이언트가 볼 수 없는 것 — 창별 전력·이용률, 비교용 idle 창, 포화
  판정이다.
- `envelope.json`의 run 블록과 각 point가 `protocol`·`harness`를 싣는다.
  `closed_loop: true`는 모드가 하나일 때 하드코딩된 리터럴이었고, 그대로 두면 모든
  개루프 아티팩트를 잘못 라벨했을 것이다.

**테스트** `tests/test_measure_envelope.py` — 기존 18개 무수정 통과(폐루프 경로 불변),
신규 14개: 기울기(평평/증가/표본 부족/실패 요청 제외), point 요약(포화 플래그 + 주석,
정상 point 미플래그, 지연은 클라이언트 값 통과, protocol/harness 기록, 실패 요청 계수,
성공 0건이면 point 아님), 클라이언트 명령(`--ignore-eos` 항상, numa 접두, replay 호출),
양쪽 백엔드에서 서버 절반 공유, 모순된 플래그 조합 거부.

**하드웨어 1회 검증(실행됨).** 단위 테스트가 보일 수 없는 것 — 벤더 서버·sysfs에서
읽은 NUMA 바인딩·전력 샘플러·HTTP 클라이언트가 네 개의 프로세스로 맞물리는 것 — 을
카드에서 한 번 확인했다. npu0(`0000:03:00.0`), 20 요청, `--rps 0.5`:
`numa_bind: node0`, `launch_error_ms` 평균 0.78 / 최대 1.95, bench 창 69.86 W @
28.07 % 대 idle 38.67 W @ 0.0 %(A5(c) 충족), `ttft_drift_slope` +0.0026 s/s로
`saturated: false`. **이것은 배관 점검이고 적합점이 아니다** — 20 요청은 D32의 배수
꼬리(−31.7 %) 아래이므로 도메인 점으로 쓰면 안 된다. 기록:
`experiments/results/s72_openloop_smoke.md`, 아티팩트는 `outputs/s72_smoke/`(비추적).

**산출물** `--mode open`을 갖춘 `measure_envelope.py` + 테스트 + `docs/deviations.md`
D114 + `experiments/results/s72_openloop_smoke.md` + `docs/HANDOVER.md` §2.10.1
항목 1·2 갱신.

#### S7.3 RNGD-CARD 개루프 재적합 (카드 필요, 반나절~1일)

- **새 도메인 파일** `profiles/calibration/openloop/rngd_card.accuracy.openloop.yaml`.
  **기존 폐루프 도메인에 점을 추가하지 않는다** — 한 보간축에 두 프로토콜을 올리는
  것은 D22의 오류 유형이고, `a40.accuracy.yaml`의 헤더가 선례다(개루프로 간 이유가
  이어붙일 점이 도착 재생이었기 때문). 절대 규칙 A3: 기존 파일의 `points` 불변.
  `index.yaml`은 손으로 고치지 않고 재생성한다
  (`tests/test_calibration_condition.py::test_the_index_matches_the_domain_files`).
- **어느 카드인가**: **npu0**, BDF **`0000:03:00.0`** — 커밋된 모든 RNGD 측정이 그
  실리콘에서 나왔으므로 재적합이 같은 카드여야 폐루프/개루프 차이가 카드 차이와
  섞이지 않는다. **`npuN` 라벨이 아니라 BDF로 지정한다**: 2026-09-17에
  `45:00.0`이 `npu2`였고 오늘 `npu3`이다. 오늘 확인된 것은 npu0=`03:00.0`,
  npu1=`04:00.0`, npu3=`45:00.0`.
- **점유 확인은 프로세스 목록으로 한다.** `alloc_status`·`furiosa-smi ps`·전력
  판독은 **모두 pod 점유를 못 본다**(2026-09-17: 16시간 넘게 점유된 카드 2장이 세
  검사 전부에서 비어 보였다). `ps -eo pid,etime,cmd | grep "[r]ngd_pd.serving.cluster"`
  의 `--chip N`을 읽는다. 2026-09-18 현재 점유 pod 없음.
- **측정 지점**: S7.0이 고른 두 후보의 동작점을 **덮는** rps들 + 창의 양 끝. 지점
  수는 S7.0 결과에 달렸으므로 여기서 숫자를 박지 않는다. 각 지점 3회 반복
  (실행 간 p99 산포가 §S7.0의 분리폭 판정에 필요하다).
- **`--ignore-eos`는 선택이 아니다**(V2 §1): 없으면 양쪽이 다른 워크로드를 돈다.
- 워크로드는 정상상태 구간이 생기도록 길이를 늘리고 워밍업을 제외한다(V3의 과도상태
  교훈). D32의 배수 꼬리(drain tail) — 20요청 런에서 −31.7 %를 만든 그것 — 때문에
  300요청 미만으로 내리지 않는다.
- `outside_domain` 정책은 새 도메인에서 **`refuse`가 기본**이다(D33: 새 도메인은
  refuse, 커밋된 세 도메인만 `widen_error_bars`를 명시적으로 opt-in). 이 선택이
  S7.4의 규칙 (c)/(d) 구분을 만든다.

**산출물** 새 도메인 파일 + 재생성된 `index.yaml` + 원자료 +
`experiments/results/rngd_openloop_envelope.md`.

#### S7.4 V3-R 본체 (카드 필요, 반나절)

- S7.0이 고른 **P2-형·P3-형 각 1개**를 개루프로 3회씩 돌린다. 예측은 `plan`이
  이미 캐시에 가진 값이고, D40을 가정하지 않는다 — **캐시를 끄고 재시뮬레이션해
  바이트 동일을 확인**한 뒤 비교한다(V3 §2.1의 절차).
- 판정 표: 후보 × 규칙 **(a) 무마진 · (b) 전역 18 % · (c) 동작점별 · (d) (c)+좌표
  refuse · (e) (d)+조건 refuse**, 그리고 **측정 ≤ SLO?** 열. V3와 같은 형태여야
  두 파일이 한 표로 읽힌다.
- **두 arm**: 기본(`refuse`, 새 개루프 도메인 조회 가능) / 대조(`warn`). 어느 쪽이
  기본인지 표 제목에 쓴다(§S7.1).
- TPOT SLO 스윕을 함께 돌려 한 배포를 격자로 만든다(V3 §6.2의 수단). **E-A1의
  50 ms 집계는 재채점하지 않는다** — 사후 파라미터 민감도 분석이고
  `outputs/uncertainty/ea1/`는 건드리지 않는다.
- **실행 간 p99 산포를 판정 옆에 적는다.** 분리폭이 산포보다 좁으면 통과는
  잡음 안이라고 쓴다. 이것이 V3에서 P3이 0.623 ms로 통과한 것을 보고할 때
  요구된 규율이고, RNGD의 폭(2.3~7.6 ms)이 그 규율을 처음 통과할 수 있는 폭이다.
- `--rps`로 내린 동작점이므로 **서비스 스펙의 사본**을 쓴다(원본 불변, `measure-apply`
  관례와 같다).

**산출물** `experiments/p2_evidence/results/v3r_verdict_accuracy.md` + 원자료.

#### S7.5 문서·신고서 매핑 (0.5일)

- `patent2_evidence_map.md` §6 행을 갱신한다. 현재 "**not from here.** … S7 on the
  RNGD node is the only source, and it has not run" 이며, `v3_addendum_s4.md` §4가
  그것을 "the standing limit"으로 적는다. S7.4가 그 행을 채우거나, §S7.B의 실패
  모드대로 **"이 fixture에 사례 없음"으로 닫는다.** 둘 다 결론이다.
- `docs/deviations.md`에 D-항목: 개루프 재적합이 폐루프 도메인을 **대체하지 않고
  병존**한다는 것(두 프로토콜 = 두 도메인), 그리고 furiosa 배포 백엔드 부재가
  S7의 측정 경로를 결정했다는 것. **번호는 이 지시서 블록의 다음 빈 번호**를
  쓴다(CLAUDE.md의 D-블록 표에서 `WORK_ORDER_domain_scoping.md` 행; 지금까지
  D110·D111·D112·D113이 쓰였다). 번호를 이 문서에 미리 적어두지 않는 이유는
  `tests/test_deviations_numbering.py`가 **항목 없는 D-참조를 실패로 잡기**
  때문이다 — 엔트리를 쓰는 커밋에서 번호를 확정한다.
- 실험 id는 **`V3-R`을 유지한다.** V3의 애드덤이며 이름은 그것이 닫는 파일에서
  온다. `E-` 태그를 새로 claim하지 않는다 — CLAUDE.md의 태그 표에 도메인 스코핑
  행이 없고, 새 태그를 만드는 것보다 V 계열을 잇는 것이 충돌을 만들지 않는다.
- `docs/HANDOVER.md` §2.10.1의 항목 1·2·3 상태 갱신.
- **`docs/CLAIMS.md`**: §6 수치가 나오면 라벨과 아티팩트를 붙여 등재한다. 나오지
  않으면 "Not established"로 명시한다.

#### S7 일정

| 하위 STEP | 노드 | 예상 |
| --- | --- | --- |
| S7.0 후보 선정 | CPU(어디서나) | 0.5일 |
| S7.1 게이트 | — | 이 rev가 결정 |
| S7.2 furiosa 개루프 포팅 | RNGD | 1.5~2일 |
| S7.3 개루프 재적합 | RNGD | 반나절~1일 |
| S7.4 V3-R 본체 | RNGD | 반나절 |
| S7.5 문서 | CPU | 0.5일 |

**합계 약 4~4.5일.** 초안의 "반나절~1일"은 §S7.A(1)·(3)을 모르고 쓴 값이며 이
rev가 그것을 정정한다. 카드가 필요한 구간은 S7.2~S7.4 약 2.5~3.5일이다.

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
| S7 | RNGD + CPU | **약 4~4.5일** (§S7 일정 표가 하위 STEP별 내역, rev 2) |

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
| **S7: 판정 뒤집기 창이 열리지 않음** (단일 카드 후보가 L < 16에만 존재하면 m = 0) | §S7.B가 예상 실패 모드로 명시. "이 fixture에 사례 없음"을 결과로 보고하고 종료한다. 창을 만들려고 SLO나 도메인을 조정하지 않는다 — 그러면 측정이 아니라 구성이 판정을 만든다 |
| **S7: 개루프 재적합이 사는 것이 비용보다 적음** | S4 §4가 이미 적은 한계: 조건 불일치는 **도메인 전체**를 철회하지만 D19의 근거는 "차이가 전부 TTFT에 떨어진다"로 더 좁다. 폐루프 도메인의 TPOT 오차는 개루프에서 살아남을 수 있고, 그러면 재적합이 사는 것의 TPOT 절반은 이미 손에 있다. 지표별 조건 스코핑은 미구현·범위 밖 — S7.3 착수 전에 이 단락을 읽고 예상 이득 크기를 적는다 |
| **S7: `warn` arm의 수치가 기본 arm의 수치로 인용됨** | §S7.1이 두 arm을 라벨하고 표 제목에 기본을 쓰게 한다. 신고서가 인용하는 것은 기본 arm뿐이며, 이 규칙은 S4가 E-A1 (a)~(d)를 `warn`으로 재현하며 이미 한 번 필요했던 것이다 |
| **S7: 카드가 다른 테넌트에게 점유됨** | `alloc_status`·`furiosa-smi ps`·전력 판독은 **전부 pod 점유를 못 본다**(docs/nodes/npu.md, 2026-09-17). `ps`로 `rngd_pd.serving.cluster`의 `--chip N`을 읽는다. 카드는 BDF로 지정(`npuN` 라벨은 재열거로 바뀐다) |
| **S7: furiosa 배포 백엔드를 쓰고 싶어짐** | 없다(`planner/deploy/`는 cuda·ascend 스텁·k8s). 작성은 S7 범위 밖이고 Phase 4다. `furiosa-llm serve` + `replay_to_endpoint.py --open-loop`으로 측정하고, 백엔드 부재를 결과 파일의 한계로 적는다 |
