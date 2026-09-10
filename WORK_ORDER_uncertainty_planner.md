# HeteroPilot 작업지시서 — 불확실성 인식 플래너 (Stage A · Stage B)

> 특허 2("예측기의 정확도 도메인과 결정 민감도 기반 측정 계획을 이용한 불확실성 인식 LLM 서빙 배치 계획")의 구현·검증 작업지시서.
> 대상 저장소: `github.com/swsok/heteropilot` (upstream: `casys-kaist/LLMServingSim`)
> 작성일: 2026-09-09 · 작업 도구: Claude Code
> 선행 문서: `docs/patent_review.md`, `docs/patent_future_ideas.md`, `WORK_ORDER_tiered_profiles.md`, `WORK_ORDER_pipeline_domain.md`, `docs/deviations.md` (특히 D22, D23)

---

## 0. 이 문서의 사용법

1. **STEP 순서를 지킬 것.** Stage A(A1→A5)를 모두 끝내고 Stage B(B1→B5)로 넘어간다. Stage B는 Stage A의 레지스트리와 정확도 도메인 위에서만 동작한다.
2. **한 STEP = 한 브랜치 = 한 PR.** 브랜치 이름은 `feat/uq-a1-registry`처럼 `feat/uq-<step>-<짧은이름>`.
3. **각 STEP의 "테스트" 절을 통과하지 못하면 다음 STEP으로 넘어가지 않는다.** 테스트를 약화시켜 통과시키는 것은 금지. 구현이 틀렸다면 구현을 고칠 것.
4. **"조사 필요"라고 표시된 항목은 추측하지 말고 실제 코드/산출물을 읽어서 확인**한 뒤 결과를 PR 설명에 적을 것.
5. 이 문서와 `CLAUDE.md`, `AGENTS.md`, 기존 `WORK_ORDER_*.md`가 충돌하면 **기존 문서가 우선**한다. 충돌을 발견하면 즉시 중단하고 사용자에게 보고할 것.
6. 이 작업의 결과는 **특허 출원의 실시예·효과 근거**가 된다. 따라서 §7의 "특허 증거 매핑" 표에 적힌 산출물 파일명·수치 항목은 임의로 바꾸지 말 것. 바꿔야 한다면 §7도 함께 갱신하고 PR 설명에 명시한다.

### 절대 규칙 (기존 프로젝트 규칙의 재확인 — 위반 시 작업 중단)

- **A1. 측정하지 않은 숫자를 측정값으로 표기하지 않는다.** 이 작업지시서는 "불확실한 숫자"를 다루는 기능 자체를 만드는 것이므로 이 규칙이 특히 중요하다. 레지스트리의 모든 오차 범위(§2.2)는 **출처가 있는 값**이어야 하고, 출처가 없으면 `unbounded`로 기록한다. 그럴듯한 기본 범위를 채우는 것은 금지.
- **A2. 데이터가 없으면 identity / null이다.** 정확도 도메인 밖의 동작점은 보간·외삽하지 않고 `unmeasured`로 처리한다.
- **A3. 실측 데이터를 덮어쓰지 않는다.** 섭동(perturbation)은 항상 메모리 안의 복사본에 적용한다. `profiles/`, `outputs/` 아래의 실측 파일은 읽기 전용.
- **A4. 기존 golden 회귀 출력이 바뀌면 안 된다.** 모든 신기능은 opt-in(CLI 플래그 또는 인자)이다. 플래그 없이 `python -m planner plan`을 실행했을 때의 출력은 바이트 단위로 이전과 같아야 한다(`tests/test_search.py`의 golden 비교가 그대로 통과해야 한다).
- **A5. 예측기를 교체하지 않는다.** LLMServingSim은 그대로 두고, 그 위에 레이어를 얹는다.

---

## 1. 배경과 목표

### 1.1 문제

플래너의 입력에는 본질적으로 불확실한 값이 섞여 있고, 현재 코드는 그 불확실성을 **세 곳에서 서로 다른 방식으로, 그리고 서로 연결되지 않은 채로** 다룬다.

| 불확실성의 출처 | 현재 코드의 처리 | 문제 |
|---|---|---|
| 시뮬레이터 예측 오차 | `planner/predictor/calibration.py` — `CalibrationModel.errors[workload_bucket]`의 `p95_abs_error`를 **workload bucket 하나당 스칼라**로 저장. `exhaustive.search(ttft_margin_percent=, tpot_margin_percent=)`로 **모든 후보에 같은 마진** 적용 | D22가 보여준 대로 오차는 동작점(서비스 동시성)의 함수다. RNGD에서 동시성 16.6일 때 TPOT −3.1 %, 동시성 76일 때 −18 %. 스칼라 마진은 저동시성 후보에 과도하고 고동시성 후보에 부족하다. 또 마진이 측정된 동시성 범위 밖의 후보에도 그대로 적용된다 |
| 하드웨어 프로파일 등급 | `planner/util/tier.py` — `ProfileTier`(analytical/calibrated/measured/imported)를 plan에 전파, 렌더러가 배너 표시 | 등급이 **경고**일 뿐 판정에 영향을 주지 않는다. Tier 0 프로파일의 10~15 % 오차가 SLO 판정에 반영되지 않는다 |
| 링크 대역폭·지연 | `planner/inventory.py` — `Link.source: Source`(measured/vendor_spec/placeholder/user_defined) | 출처가 provenance에만 기록된다. 35 GB/s 플레이스홀더가 실측 13 GB/s로 바뀌어도(D16) 어느 후보의 판정이 뒤집힐지는 다시 시뮬레이션해야 안다 |

그 결과 운영자는 "무엇을 먼저 측정해야 하는가"를 알 수 없다. 프로파일링은 가속기당 약 2.1 시간, 동시성 엔벌로프 측정은 서버를 점유하는 수 시간 작업이므로 측정 순서는 실무의 핵심 비용이다.

### 1.2 목표

세 출처의 불확실성을 **하나의 레지스트리**로 모으고, 예측 오차를 **동작점의 함수(정확도 도메인)** 로 선언하게 하고, 후보별로 **자기 동작점의 오차 마진**을 적용하고, 도메인 밖은 **`unmeasured`로 거절**하고(Stage A), 레지스트리의 각 항목을 오차 범위 안에서 **섭동**하여 추천이 뒤집히는지 검출하고 **결정 후회 감소량 ΔR_i** 를 산출하여 **측정 계획**을 배치 계획과 함께 출력한다(Stage B).

### 1.3 이 작업지시서가 만드는 것

| Stage | 산출물 |
|---|---|
| A | `planner/uncertainty/` 신규 패키지(`registry.py`, `grades.py`), `planner/predictor/accuracy_domain.py`, `planner/optimizer/margin.py`(후보별 마진 정책), `RejectionStage.UNMEASURED`, `PlannerOutput.uncertain_inputs`, 렌더러 섹션, `profiles/uncertainty/grades.yaml`, `profiles/calibration/*.yaml` 스키마 확장(동작점별 오차), CLI `--accuracy-domain`, 실험 E-A1·E-A2 |
| B | `planner/uncertainty/perturb.py`(닫힌 형식 섭동), `planner/uncertainty/sensitivity.py`(전환 검출·ΔR_i), `planner/uncertainty/measurement_plan.py`, CLI `--measurement-plan`, `python -m planner measure-apply`, `experiments/uncertainty/`(진실 열화 실험 E-B1~E-B3), `docs/uncertainty_planner.md` |

### 1.4 이 작업지시서가 만들지 않는 것 (구현 금지)

- **베이지안 최적화·가우시안 프로세스·능동학습 라이브러리 도입.** ΔR_i는 격자점 평균의 결정론적 계산이다(§2.5). 확률 모델은 Stage D(선택) 범위.
- **리플랜 게이트·폐루프 모니터 연동.** `planner/monitor/`는 이 지시서에서 건드리지 않는다(Stage C 범위).
- **conformal 하한 기반 서로게이트 프루닝.** Stage D 범위.
- **`serving/` 하위 upstream 코드 수정.**
- **특허1 관련 S4/S5 하한 수식 변경.** A/K 최소 보장치 수정은 `WORK_ORDER_pipeline_domain.md`의 후속 항목으로 별도 처리한다(§8 참조). 이 지시서에서 `planner/optimizer/greedy.py`, `surrogate.py`의 하한 수식을 바꾸면 중단.

---

## 2. 설계 개요

### 2.1 용어

- **불확실 입력(uncertain input)** `u_i`: 플래너 입력 중 실측이 아닌 값 또는 실측이라도 오차 분포를 갖는 값. 식별자 `id`, **종류(kind)**, **출처 등급(grade)**, **현재값(nominal)**, **오차 범위(range)**, **측정 비용(cost)**, **영향 후보 집합(affects)** 을 가진다.
- **동작점(operating point)**: 후보가 실제로 운용될 때의 서비스 동시성 `L`(평균 in-flight 요청 수). 시뮬레이션 결과에서는 요청별 (도착, 완료) 시각으로 직접 계산하고(`Σ latency / wall`, D22의 계산법과 동일), 서로게이트 단계에서는 Little's law 고정점 `L = λ · W(L)`, `W(L) = TTFT(L) + n_out · TPOT(L)`로 푼다.
- **정확도 도메인(accuracy domain)**: 하드웨어 라벨 × workload bucket 별로, 오차가 측정된 동작점들의 집합 `{(L_k, err_ttft_k, err_tpot_k)}`과 그 유효 범위 `[L_min, L_max]`. 범위 안은 구간 선형 보간, 밖은 `unmeasured`.
- **후보별 마진** `m_c = err(L_c)`: 후보 `c`의 동작점 `L_c`에서 보간한 오차. 강건 지표 `= 예측 × (1 + max(0, m_c))`. (`ErrorStats` 규약과 같이 양수 = 시뮬레이터가 낙관.)
- **전환(flip)**: 불확실 입력 하나를 범위 안에서 움직였을 때 추천 계획의 id 또는 상위 K 순서가 바뀌는 사건.
- **결정 후회 감소량** `ΔR_i`: §2.5.

### 2.2 출처 등급과 오차 범위 — `profiles/uncertainty/grades.yaml`

레지스트리는 각 항목의 `grade`로부터 `range`를 얻는다. **범위 값은 전부 출처를 갖는다.** 초기 파일은 다음만 담고, 각 행에 `source:` 필드를 채운다.

| kind | grade | range 결정 규칙 | 출처 |
|---|---|---|---|
| `sim_error` | `measured` | 정확도 도메인의 보간값 ± (해당 bucket의 `p95_abs_error` − `|mean_error|`) | `profiles/calibration/<hw>.yaml` |
| `sim_error` | (도메인 밖) | `unbounded` → 후보는 `unmeasured` | — |
| `profile` | `imported`/`measured` (번들 tier **와** 프로파일 `source` 둘 다) | `sim_error` 항목에 흡수(별도 항목 만들지 않음). `measured_count["profile"] += 1` | — |
| `profile` | `calibrated` (Tier 1) | ±`calibration_report.operator_error_p95` | `experiments/tier_validation/` E2 결과 파일 (**조사 필요**: 실제 파일명·필드명 확인) |
| `profile` | `analytical` (Tier 0) | ±`tier_validation` E1의 operator 오차 p95 | 같음 |
| `profile` | `placeholder` — 번들이 없거나, **프로파일 yaml의 `source: placeholder`** (번들 tier와 무관. 예: `ascend-sim-proxy`는 RTXPRO6000 실측 번들을 빌려 쓰지만 프로파일 자체가 placeholder) | `unbounded` | — |
| `profile` | `user_defined` | 사용자가 `range`를 함께 주지 않으면 `unbounded` | — |
| `link_bw` | `measured` | 레지스트리 항목을 만들지 않음. `UncertainInputRegistry.measured_count["link_bw"]`에 개수만 집계 | — |
| `link_bw` | `vendor_spec` | `[spec × r_min, spec]` 여기서 `r_min` = 저장소에 있는 실측/스펙 비율의 최솟값 | A40 D2H 25.71 GB/s vs PCIe Gen4 x16 스펙(**조사 필요**: 스펙값·측정 로그 `experiments/scripts/gpu_host_bandwidth.py` 출력 확인), RNGD host↔PE 26.27 GB/s vs 스펙 |
| `link_bw` | `placeholder` | `unbounded` | — |
| `link_bw` | `user_defined` | 사용자가 `range`를 함께 주지 않으면 `unbounded` | — |
| `power` | `vendor_spec` (TDP 기반) | ±(`phase0_power_repro.csv`의 측정 대비 TDP 오차 p95) | `outputs/phase0_power_repro.csv` (**조사 필요**) |

규칙: `grades.yaml`에 없는 (kind, grade) 조합은 `unbounded`다. 코드에 숫자를 하드코딩하지 않는다.

### 2.3 레지스트리 자료구조 — `planner/uncertainty/registry.py`

```python
class UncertainKind(str, enum.Enum):
    SIM_ERROR = "sim_error"      # hw × bucket 의 예측 오차 (정확도 도메인)
    PROFILE = "profile"          # 가속기 성능 프로파일 등급 (Tier 0/1/placeholder)
    LINK_BW = "link_bw"          # inventory.Link.bandwidth_gbps
    LINK_LAT = "link_lat"        # inventory.Link.latency_ns
    POWER = "power"              # 가속기 전력 모델 (TDP/측정)

class Range(_Strict):
    lo: float | None            # None = unbounded (해당 방향)
    hi: float | None
    unit: str                   # "fraction" | "gbps" | "ns" | "w"
    source: str                 # grades.yaml 의 행 id 또는 calibration 파일 경로

class UncertainInput(_Strict):
    id: str                     # 예: "link_bw:gpu0->npu0", "sim_error:RNGD/sharegpt-llama31-8b-300"
    kind: UncertainKind
    grade: str                  # Source / ProfileTier 문자열 그대로
    nominal: float
    range: Range
    cost: MeasurementCost       # §2.6
    affects: list[str]          # 영향받는 island_id 또는 link id
    note: str = ""

class UncertainInputRegistry(_Strict):
    items: list[UncertainInput]                 # 불확실한 것만. measured 입력은 넣지 않는다
    measured_count: dict[str, int]              # kind -> 실측이라 제외된 입력 수 (렌더러의 커버리지 한 줄용)
    grades_digest: str          # grades.yaml 해시 (provenance)
    def unbounded(self) -> list[UncertainInput]: ...
    def by_kind(self, kind) -> list[UncertainInput]: ...
```

수집 함수 `build_registry(cluster, profiles, islands, calibration, grades) -> UncertainInputRegistry`는 다음에서 항목을 만든다.

- `cluster.links`의 각 링크: `source`가 `measured`가 아니면 `LINK_BW`(및 `latency_ns`에 대해 `LINK_LAT`) 항목. `measured`면 항목 없이 `measured_count["link_bw"] += 1`. **원칙: 레지스트리에는 폭이 0인 항목이 존재하지 않는다** — 섭동·정렬 코드가 폭 0을 특수 처리할 필요가 없어야 한다. `affects`는 `TopologyGraph.path()`로 그 링크를 지나는 island 쌍.
- 각 island의 프로파일: **등급은 두 신호의 약한 쪽**으로 정한다 — (i) 번들 tier `tier.resolve_bundle_tier(sim_hardware, ...)`, (ii) 프로파일 yaml의 `AcceleratorProfile.source`. 순위 `placeholder = user_defined (0) < analytical (1) < calibrated (2) < vendor_spec (3) < measured = imported (4)`; `grade = 순위가 낮은 쪽`, 순위 4가 아니면 `PROFILE` 항목을 만든다. 번들 tier만 보면 `ascend-sim-proxy`(프로파일 `source: placeholder`, 번들은 RTXPRO6000 실측)가 `measured`로 통과해 **저장소에서 가장 오해를 부르는 입력이 레지스트리에서 빠진다** — 이것이 두 신호를 모두 보는 이유다. `note`에는 두 신호를 모두 적는다(`"profile source=placeholder; bundle RTXPRO6000 tier=measured"`). 추가로 `profile.model`(또는 `profile_id`)과 `sim_hardware` 라벨이 다르면 `note`에 `proxy: <model> simulated as <sim_hardware>`를 붙인다. `ascend_target`(프로파일 `vendor_spec`, 번들 `ASCEND_TARGET-t0` analytical)은 `analytical`이 된다. 전력 블록 `power.source`도 같은 규칙으로 `POWER` 항목을 만든다.
- `CalibrationModel.hardware[hw]`가 있는 hw × bucket: `SIM_ERROR` 항목 하나. 도메인은 §2.4. **calibration이 없는 hw는 `SIM_ERROR` 항목을 `unbounded`로 만든다** — 지금까지는 마진 0으로 통과시켰지만(`CalibrationModel.margins()`의 `(0.0, 0.0)`), 이 지시서 이후 `--accuracy-domain` 모드에서는 그 hw의 후보가 `unmeasured`가 된다. 기본 모드는 A4에 따라 그대로 둔다.

### 2.4 정확도 도메인 — `profiles/calibration/*.yaml` 확장과 `accuracy_domain.py`

기존 `BucketError`에 **동작점별 오차**를 추가한다(하위 호환: 필드가 없으면 기존 스칼라 동작).

```yaml
errors:
  sharegpt-llama31-8b-300:
    workload_bucket: sharegpt-llama31-8b-300
    ttft: {mean_error: ..., p95_abs_error: ..., worst_error: ..., sample_count: 5}
    tpot: {...}
    operating_points:                    # 신규. 비어 있으면 스칼라 모드
      - concurrency: 16.6                # Little's law 로 계산한 실제 서빙 동시성 (요청 동시성이 아님, D22)
        ttft_error: -0.031
        tpot_error: -0.031
        sample_count: 128
        source: outputs/rngd_envelope/edf/real_c16.json + outputs/envcheck/rngd_verify_card_edf.csv
      - concurrency: 76.0
        tpot_error: -0.18
        ...
    domain: {concurrency_min: 16.6, concurrency_max: 107.2}   # operating_points 로부터 자동 산출, 명시하면 검증
```

`accuracy_domain.py`:

```python
class AccuracyDomain:
    def __init__(self, bucket_error: BucketError): ...
    def contains(self, concurrency: float) -> bool
    def error_at(self, concurrency: float) -> tuple[float, float] | None   # (ttft, tpot); 밖이면 None
    # 보간: 구간 선형. 외삽 금지. 점이 하나뿐이면 그 점 ±0 폭의 도메인.

def served_concurrency_from_sim(per_request: Sequence[RequestRecord], wall_s: float) -> float
def served_concurrency_little(rps: float, ttft_ms: float, tpot_ms: float, mean_out_tokens: float) -> float
```

`served_concurrency_from_sim`은 `LLMServingSimPredictor._parse()`가 이미 읽는 요청별 CSV에서 계산한다. **조사 필요**: `_parse()`가 요청별 도착·완료 시각을 버리는지 확인. 버린다면 `PredictedMetrics`에 `served_concurrency: float | None = None` 필드를 추가하고 `_parse()`에서 채운다(기존 필드는 건드리지 않음, golden 유지).

`operating_points`를 채우는 도구: `python -m planner fit-accuracy-domain --real outputs/rngd_envelope/edf/real_c*.json --sim <대응 sim csv> --hardware RNGD-CARD --bucket ...`. 기존 `compare_rngd_sim_vs_real.py`의 페어링 로직을 재사용한다(**조사 필요**: 각 real_cN에 대응하는 sim CSV가 `outputs/envcheck/`에 있는지; 없으면 E-A2에서 생성).

### 2.5 전환 검출과 결정 후회 감소량 — Stage B

레지스트리 항목 `i`에 대해 격자 `G_i = {g_1..g_m}` (기본 m=5: `lo`, `lo+¼`, nominal, `hi−¼`, `hi`; `unbounded`면 격자 없음 → ΔR_i 정의 불가, "측정 전 결정 불가"로 표기). 각 격자점 `g`에서:

1. `perturb.py`로 **모든 후보의 예측 지표를 닫힌 형식으로 재계산**한다(재시뮬레이션 없음, §2.7).
2. 후보별 마진·SLO 판정·순위를 다시 매겨 그 격자점에서의 최적 계획 `π*(g)`와 목적함수 값 `V(π*(g))`, 그리고 현재 추천 `π̂`의 그 격자점에서의 값 `V_g(π̂)`을 얻는다.
3. `flip_i = any_g [ id(π*(g)) != id(π̂) ]`.
4. `ΔR_i = mean_g [ V_g(π*(g)) − V_g(π̂) ]` — 격자점 균등 가중(가중 방식은 provenance에 기록). `V`는 `ObjectiveSpec`의 1차 목적(기본 `slo_goodput_per_joule`), 실행불가는 `V = −∞`가 아니라 **SLO 위반 비용 항**으로 처리: `V_g(π̂) = value − penalty × overshoot_ratio` (penalty는 `--slo-penalty`, 기본은 목적값의 최댓값 — provenance에 기록).

`ΔR_i > 0`인 항목만 측정 계획 후보. 정렬 키 `ΔR_i / cost_i`.

### 2.6 측정 비용 — `profiles/uncertainty/costs.yaml`

| kind | 측정 방법 | 비용(시간) | 출처 |
|---|---|---|---|
| `profile` | `python -m profiler profile` 전체 | ≈ 2.1 h / (hw, model, tp) | `WORK_ORDER_tiered_profiles.md` §1.2 |
| `sim_error` (동작점 1개 추가) | 서빙 스택 기동 + `bench_furiosa_endpoint.py` 또는 vLLM bench 1회 | **조사 필요**: `outputs/rngd_envelope/edf/serve_c*.log`의 타임스탬프로 산출 | 로그 |
| `link_bw` | `gpu_host_bandwidth.py` / `rngd_collective_probe.py` | **조사 필요** | 로그 |
| `power` | `profiler/power/profile_gpu_power.sh` | **조사 필요** | 스크립트 |

비용 단위는 시간(h). 서버 점유 여부(`exclusive: true/false`)도 기록해 예산 B는 "시간"과 "점유 시간"을 따로 셀 수 있게 한다. 비용이 없는 kind는 `cost: null` → 정렬 시 맨 뒤.

### 2.7 닫힌 형식 섭동 — `perturb.py`

핵심 설계 결정: **섭동은 후보별 재시뮬레이션이 아니라 캐시된 예측 지표의 후처리**다. 그래야 레지스트리 항목 수 × 격자점 수 × 후보 수의 조합이 초 단위로 끝난다. 각 kind별 규칙:

| kind | 영향받는 지표 | 규칙 |
|---|---|---|
| `SIM_ERROR` | 마진 `m_c` | 정확도 도메인 보간값을 범위 안에서 이동. 예측 지표 자체는 불변 |
| `PROFILE` (Tier 0/1) | TTFT, TPOT | 해당 island를 쓰는 후보의 TTFT·TPOT에 `(1+δ)` 곱. 처리량·goodput은 `1/(1+δ)` 스케일. 에너지는 불변(전력 모델 별도) |
| `LINK_BW` | TTFT (P/D 후보), 하한 S4′(PP 후보) | `kv_transfer.kv_transfer_cost()`로 nominal과 섭동값의 전송 시간 차를 계산해 TTFT 백분위에 더하고 뺀다. 이 항은 `compile_to_sim_config`가 넣는 값과 같은 수식이어야 한다(**조사 필요**: 시뮬레이터가 KV 전송을 지연에 어떻게 반영하는지 — 단순 가산이 아니면 이 규칙은 근사이며 `approximation: true` 플래그를 결과에 붙인다) |
| `LINK_LAT` | TTFT | 링크 지연 × 홉 수 가산 |
| `POWER` | energy, tokens/J | 평균 전력 `(1+δ)` 곱 |

각 규칙은 `Perturbation.apply(metrics: PredictedMetrics, delta) -> PredictedMetrics` 인터페이스로 구현하고, **근사 여부를 `PerturbResult.approximation: bool`로 반환**한다. 근사 규칙이 사용된 항목은 측정 계획 출력에 "(근사)"를 붙인다. 근사가 아닌 정확한 재계산이 필요하면 `--resimulate-top N`으로 ΔR 상위 N개 항목에 대해서만 `lo`/`hi` 두 점을 실제 시뮬레이션한다(선택 기능, B2).

### 2.8 CLI 표면

```
python -m planner plan ... --accuracy-domain           # Stage A: 후보별 마진 + unmeasured 거절 + 레지스트리 출력
python -m planner plan ... --accuracy-domain --measurement-plan [--budget-hours H] [--grid 5] [--slo-penalty X]
                                                       # Stage B: 측정 계획까지 출력
python -m planner fit-accuracy-domain ...              # calibration yaml 에 operating_points 채우기
python -m planner measure-apply --plan out.yaml --input <id> --value V --source measured [--range lo,hi]
                                                       # 측정값 반영 → 레지스트리 갱신 → 캐시 재사용 재계획
```

`--accuracy-domain` 없이 실행하면 **어떤 출력도 바뀌지 않는다**(A4). `--measurement-plan`은 `--accuracy-domain`을 요구한다.

---

## 3. Stage A — 레지스트리 · 정확도 도메인 · 후보별 마진 · 미측정 거절

### STEP A1 — 등급표와 레지스트리

**만드는 것**
- `profiles/uncertainty/grades.yaml` (§2.2), `profiles/uncertainty/costs.yaml` (§2.6). 조사 필요 항목은 조사 결과로 채우고, 못 채운 행은 `range: unbounded` / `cost: null`로 두고 `note`에 이유를 쓴다.
- `planner/uncertainty/__init__.py`, `grades.py`(yaml 로더 + 스키마 검증, `extra="forbid"`), `registry.py`(§2.3).
- `PlannerOutput.uncertain_inputs: UncertainInputRegistry | None = None` 필드 추가(기본 None → golden 불변).
- `planner/render.py`에 "불확실 입력" 섹션: `--accuracy-domain`일 때만. 첫 줄에 커버리지(`링크 26개 중 16개 불확실 (placeholder 14, vendor_spec 2); 프로파일 3개 중 1개; ...`), 이어서 표: id / 종류 / 등급 / 현재값 / 범위 / 영향 island. `unbounded` 항목은 맨 위, 굵게.

**테스트** `tests/test_uncertainty_registry.py`
- `experiments/configs/clusters/pd-rngd-gpu.yaml`(링크 26개: measured 10, placeholder 14, vendor_spec 2 — 테스트 안에서 yaml을 읽어 이 수를 다시 세고 하드코딩하지 않는다)로 레지스트리를 만들면 `LINK_BW` 항목 16개, `measured_count["link_bw"] == 10`. placeholder 14개는 `range unbounded`, vendor_spec 2개는 `grades.yaml`의 `r_min` 규칙으로 유한 범위(`r_min` 행이 아직 비어 있으면 unbounded — 어느 쪽인지 테스트가 grades.yaml을 읽어 판정). 폭 0인 항목은 0개(`all(i.range.lo != i.range.hi for i in items)`).
- `a40x8.yaml` + `profiles/calibration/a40.yaml`로 만들면 `SIM_ERROR:A40/<bucket>` 항목 1개, 도메인 스칼라 모드.
- `ascend-sim-proxy.yaml`로 만들면 NPU island에 `PROFILE` 항목 1개, `grade == "placeholder"`, `range unbounded`, `note`에 `proxy:`와 `bundle RTXPRO6000 tier=measured`가 모두 포함. 같은 fixture의 GPU island(RTXPRO6000, 프로파일 `source: measured`)는 항목 없음, `measured_count["profile"] == GPU island 수`. **이 테스트가 "번들 tier만으로 판정하면 실패"하도록**, 테스트 안에서 `resolve_bundle_tier("RTXPRO6000", ...)`가 `MEASURED`임을 먼저 단언한다(회귀 방지용 문서화).
- `ascend_target.yaml`(프로파일 `vendor_spec`, 번들 `-t0`)을 쓰는 fixture(없으면 테스트용 최소 cluster yaml을 `tests/fixtures/`에 추가)로 만들면 `grade == "analytical"` — 두 신호 중 약한 쪽이 선택됨을 확인.
- `grades.yaml`에 없는 조합 → `unbounded`, 예외 없음. 숫자가 있는데 `source`가 비어 있으면 로더가 거부.
- golden: `--accuracy-domain` 없이 기존 `tests/test_search.py` 전부 통과, `PlannerOutput` YAML 덤프에 `uncertain_inputs` 키가 **없어야** 한다(None은 덤프에서 제외 — **조사 필요**: `_write_output`의 직렬화 옵션 확인).

### STEP A2 — 정확도 도메인과 서빙 동시성

**만드는 것**
- `BucketError.operating_points: list[OperatingPoint]`, `BucketError.domain: ConcurrencyDomain | None` (§2.4). 기존 yaml은 필드 없이 그대로 로드되어야 한다(`tests/test_calibration.py` 불변).
- `planner/predictor/accuracy_domain.py` (§2.4).
- `PredictedMetrics.served_concurrency: float | None = None`, `_parse()`에서 채움(**조사 필요** 결과에 따라).
- `python -m planner fit-accuracy-domain`: real JSON들과 sim CSV들을 짝지어 `operating_points`를 계산하고 calibration yaml을 **새 파일로** 쓴다(`--out`; 기존 파일 덮어쓰기 금지, A3). 동시성은 반드시 `Σ latency / wall`로 계산하고 요청 동시성(`concurrency` 필드)은 `note`에만 남긴다 — D22의 교훈.
- `profiles/calibration/rngd_card_edf.yaml`에 대응하는 `rngd_card_edf.domain.yaml`을 생성해 커밋: c16/c32/c64/c128 real과 sim의 짝(**조사 필요**: sim 쪽 4점이 없으면 E-A2에서 만든 뒤 커밋).

**테스트** `tests/test_accuracy_domain.py`
- 점 2개 `(16.6, −0.031), (76, −0.18)`에서 `error_at(46.3)`은 선형 중간값, `error_at(10)`과 `error_at(100)`은 `None`.
- 점 1개면 도메인이 그 점 하나. `contains(그 점 ± ε)`는 False.
- `served_concurrency_from_sim`: 합성 요청 4개(각 10 s, wall 20 s) → 2.0.
- `served_concurrency_little`: `rps=2, ttft=100 ms, tpot=20 ms, n_out=500` → `2 × 10.1 s = 20.2`.
- 기존 스칼라 yaml 로드 시 `operating_points == []`, `AccuracyDomain.error_at(any)`가 스칼라 `mean_error`를 반환하고 `contains()`는 항상 True(하위 호환 모드, 명시적으로 `scalar_mode=True` 속성 노출).

### STEP A3 — 후보별 마진 정책과 `UNMEASURED` 거절

**만드는 것**
- `planner/optimizer/margin.py`:
  ```python
  class MarginDecision(_Strict):
      ttft_percent: float; tpot_percent: float
      status: Literal["in_domain", "scalar", "unmeasured"]
      concurrency: float | None; basis: str      # 어떤 도메인·어떤 점 사이 보간인지
  class MarginPolicy(Protocol):
      def decide(self, candidate: CandidateConfig, metrics: PredictedMetrics, island_hw: dict[str,str]) -> MarginDecision
  class GlobalMargin(MarginPolicy)        # 기존 동작: 스칼라 두 개. 기본값
  class AccuracyDomainMargin(MarginPolicy) # hw별 도메인에서 metrics.served_concurrency 로 보간.
                                           # 혼합/PD 후보는 island 별 마진의 max. 어느 island 라도 밖이면 unmeasured
  ```
- `exhaustive.search(..., margin_policy: MarginPolicy | None = None)`. `None`이면 기존 `ttft_margin_percent/tpot_margin_percent`로 `GlobalMargin`을 만든다 → 출력 불변.
- `RejectionStage.UNMEASURED = "unmeasured"` — 순서상 `SLO_VIOLATED` 앞. 사유 문자열에 동작점과 도메인 범위를 넣는다: `"served concurrency 92.4 outside RNGD-CARD accuracy domain [16.6, 76.0] for bucket sharegpt-...; measure envelope at c≥92 or accept scalar margin"`.
- `DeploymentPlan`에 `margin_basis: str = ""` 추가(기본값 빈 문자열 → YAML 불변 여부 **조사 필요**; 바뀌면 `None` 기본 + 덤프 제외).
- `PlannerOutput.suggestions`에 unmeasured 후보가 최적 후보였을 때 "가장 좋은 후보 hp-xxxx는 미측정 도메인에 있음 — 동시성 L에서 엔벌로프 측정 권고" 추가. **infeasible-is-a-diagnosis 원칙과 일관.**

**테스트** `tests/test_margin_policy.py`
- 합성 후보 3개(동시성 20/60/110), 도메인 `[16.6, 76]`: 첫 둘은 `in_domain`으로 각각 다른 마진, 셋째는 `unmeasured` 거절, `rejected_summary["unmeasured"] == 1`.
- 같은 입력에 `GlobalMargin(18,18)` → 셋 모두 18 %, 거절 없음 (기존 동작 재현).
- D22 재현 테스트: TPOT 예측 48.41 ms, SLO 50 ms, 동시성 76, 도메인 점 `(76, −0.18)` → 강건 TPOT 57.1 ms, `SLO_VIOLATED`; 동시성 16.6 → 마진 3.1 %, 49.9 ms, 통과. **이 테스트가 특허 §5.7 동작 예의 근거다.**
- golden: `tests/test_search.py`, `tests/test_pd_sweep_margin.py` 무변경 통과.

### STEP A4 — CLI · 렌더 · provenance

**만드는 것**
- `--accuracy-domain [PATH...]`: calibration yaml(들)을 읽어 `AccuracyDomainMargin` 구성, 레지스트리 생성, `PlannerOutput.uncertain_inputs` 채움, `provenance["uncertainty"] = {grades_digest, costs_digest, domains: {hw: [L_min, L_max]}, policy: "accuracy_domain"}`.
- `--ttft-margin-percent/--tpot-margin-percent`와 `--accuracy-domain`을 동시에 주면 오류(둘은 배타).
- 렌더러: 추천 계획 아래 "마진 근거" 한 줄(`margin_basis`), 거절 요약에 `unmeasured` 행, 레지스트리 표(A1).
- `experiments/scripts/pd_slo_sweep.py`에 `--accuracy-domain` 패스스루.

**테스트** `tests/test_cli_accuracy_domain.py` — CLI 스모크(작은 fixture, 시뮬레이터는 `tests/conftest.py`의 fake predictor 사용, **조사 필요**: fake predictor에 `served_concurrency`를 넣을 수 있는지).

### STEP A5 — 실험 E-A1 · E-A2

**E-A1 — 스칼라 마진 vs 후보별 마진 (재현 실험, 시뮬레이션만)**
- 입력: `outputs/pd_slo_sweep_margin18/` 가 사용한 service/cluster fixture(JSON의 `service`, `cluster` 필드), 같은 `--cache-dir`(캐시가 있으면 재시뮬레이션 0).
- 세 조건: (a) 마진 0, (b) 전역 18 %(D22 재현), (c) `--accuracy-domain rngd_card_edf.domain.yaml`.
- 산출: `experiments/uncertainty/results/ea1_margin_modes.md` — 조건별 추천 plan_id, tokens/J, `rejected_summary`(특히 `unmeasured` 수), (b)와 (c)에서 판정이 다른 후보 목록과 각 후보의 동시성·마진.
- 기대(가설, 결과로 확인): (c)에서 저동시성 RNGD 후보 일부가 (b)의 일괄 거절에서 살아나고, 고동시성 후보는 `unmeasured` 또는 더 큰 마진으로 거절된다. 결과가 가설과 다르면 그대로 기록한다.

**E-A2 — RNGD 정확도 도메인 채우기 (시뮬레이션만; 실측 JSON은 이미 커밋됨, 가속기 불필요)**
- `outputs/rngd_envelope/edf/real_c{16,32,64,128}.json`에 대응하는 sim CSV 4개를 같은 워크로드로 생성(`serving/` 직접 실행, CPU. 기존 `envcheck` 절차 재사용). **조사 필요**: `outputs/envcheck/rngd_verify_card_edf.csv`가 이 중 한 점인지 확인하고 있으면 재사용.
- `fit-accuracy-domain`으로 `rngd_card_edf.domain.yaml` 생성, 4점의 (서빙 동시성, TTFT 오차, TPOT 오차) 표를 `experiments/uncertainty/results/ea2_rngd_domain.md`에 기록. 이 표가 **특허 도 3의 실측 대체 데이터**가 된다.
- A40도 가능하면: `outputs/phase0_bench/A40/vllm/`의 validation 5점의 서빙 동시성을 계산해 `a40.domain.yaml` 생성. 5점이 거의 같은 동시성이면 도메인이 좁다는 사실 자체를 기록(A40에서 c-sweep 추가 측정은 Stage C 항목).

**Stage A 완료 기준**: A1~A4 테스트 전부 통과, golden 불변, E-A1 결과 md 커밋, E-A2의 RNGD 도메인 yaml 커밋(A40은 선택).

---

## 4. Stage B — 섭동 · 전환 검출 · 후회 감소량 · 측정 계획

### STEP B1 — 닫힌 형식 섭동 엔진

**만드는 것** `planner/uncertainty/perturb.py` (§2.7). 후보 → 예측 지표 맵과 레지스트리 항목, δ를 받아 새 지표 맵을 돌려주는 순수 함수들. 입력을 절대 변경하지 않는다(`model_copy`).

**조사 필요(B1 시작 전 PR 설명에 결론을 적을 것)**
1. `compile_to_sim_config`와 시뮬레이터가 P/D KV 전송 시간을 TTFT에 어떻게 반영하는가. 가산이면 `LINK_BW` 규칙은 정확(`approximation=False`), 아니면 근사.
2. `kv_transfer_cost()`가 프롬프트 길이 백분위별로 돌려주는 값과 `PredictedMetrics`의 p50/p95/p99 TTFT의 대응 관계.

**테스트** `tests/test_perturb.py`
- δ=0이면 모든 지표가 바이트 동일.
- `PROFILE` δ=+0.1: TTFT·TPOT ×1.1, throughput ×1/1.1, energy 불변.
- `LINK_BW` 35→13 GB/s: TTFT 증가량이 `kv_transfer_cost(35) − kv_transfer_cost(13)`의 부호 반대·크기 동일. 단일 island 후보(전송 없음)는 불변.
- 섭동 후 원본 객체가 변하지 않았음을 `id`/해시로 확인.

### STEP B2 — 전환 검출과 ΔR_i

**만드는 것** `planner/uncertainty/sensitivity.py`
```python
class GridPoint(_Strict): value: float; best_plan_id: str; best_value: float; current_value: float
class Sensitivity(_Strict):
    input_id: str; flip: bool; delta_regret: float | None   # None = unbounded
    grid: list[GridPoint]; approximation: bool; note: str
def analyze(output: PlannerOutput, registry, metrics_by_candidate: dict[str, PredictedMetrics],
            policy: MarginPolicy, spec, *, grid: int = 5, slo_penalty: float | None = None) -> list[Sensitivity]
```
- 판정·순위 재계산은 `feasibility.evaluate`와 `exhaustive`의 랭킹 함수를 **그대로 호출**한다(복제 금지 — **조사 필요**: 랭킹이 `search()` 내부에 인라인이면 먼저 `rank_plans()`로 추출하는 순수 리팩터 PR을 B2 앞에 둔다. 동작 변경 0, golden 불변).
- `metrics_by_candidate`는 `EnvelopeCache`에서 읽는다. 캐시가 없는 후보는 분석에서 제외하고 `note`에 기록.
- 선택: `--resimulate-top N` (§2.7).

**테스트** `tests/test_sensitivity.py`
- 합성 후보 2개(A: 단일 island, B: P/D). `LINK_BW` 범위 `[13, 35]`, nominal 35에서 B가 추천. 격자 하단에서 A가 최적으로 바뀌면 `flip=True`, `delta_regret > 0`, 격자 상단 `current_value == best_value`.
- 어느 격자점에서도 순위가 안 바뀌는 항목: `flip=False`, `delta_regret == 0.0`.
- `unbounded` 항목: `delta_regret is None`, 격자 비어 있음.
- 항목 두 개 중 비용 대비 ΔR이 큰 쪽이 먼저 정렬됨.

### STEP B3 — 측정 계획 출력과 `measure-apply`

**만드는 것**
- `planner/uncertainty/measurement_plan.py`: `Sensitivity` 목록 + `costs.yaml` + 예산 → `MeasurementPlan(items: [ {rank, input_id, kind, method, cost_h, exclusive, delta_regret, flip, approximation, how_to_measure} ], budget_h, covered_regret, uncovered: [...] )`. `how_to_measure`는 costs.yaml의 `method` 문자열(실행할 스크립트 이름).
- `PlannerOutput.measurement_plan: MeasurementPlan | None = None`, 렌더러 섹션 "측정 계획"(순위 표 + "이 예산으로 제거되는 후회 합계" 한 줄 + unbounded 항목 목록 "측정 전 결정 불가").
- `python -m planner measure-apply`: 저장된 PlannerOutput YAML을 읽고 `--input <id> --value V --source measured`를 반영해 (1) 링크면 cluster yaml의 **사본**을 `<name>.measured.yaml`로 쓰고, (2) sim_error면 calibration yaml 사본에 점을 추가하고, (3) 캐시 디렉터리를 재사용해 `plan`을 다시 실행한다. 원본 yaml은 건드리지 않는다(A3).

**테스트** `tests/test_measurement_plan.py` — 예산 0이면 항목 0개·`uncovered` 전부; 예산 무한이면 ΔR>0 항목 전부 순서대로; `measure-apply` 후 레지스트리에서 해당 항목의 grade가 `measured`, range 폭 0.

### STEP B4 — 진실 열화 실험 (E-B1 ~ E-B3, 시뮬레이션만)

이 실험이 **특허 §6 효과의 정량 근거**다. 실측 서버가 필요 없고 기존 캐시로 돈다.

**공통 설정**
- **진실(truth)**: 완전 측정 상태의 fixture. 후보 = 실측 프로파일(A40, RNGD-CARD imported), 링크 = `pd_slo_sweep_measured_fabric`의 실측 13 GB/s, calibration = Stage A의 도메인 yaml. 이 상태에서 `--oracle`로 낸 추천 `π_truth`와 목적값이 기준.
- **열화(degrade)**: 진실의 입력 중 `k`개를 등급만 낮춘다 — 링크를 `placeholder` 35 GB/s로, 프로파일을 Tier 0 합성 번들(`profiler/synth`)로, 도메인을 스칼라 모드로. 열화 집합은 `k ∈ {1,2,3}`, 무작위 시드 10개.
- **측정 시뮬레이션**: 계획이 항목 `i`를 고르면 그 항목만 진실값으로 되돌리고(비용 `c_i` 차감) 재계획. 예산이 소진될 때까지 반복.
- **지표**: 예산 `B`에 따른 후회 `R(B) = V(π_truth) − V_truth(π̂_B)` (진실 환경에서 평가한 현재 추천의 손실), 후회 0에 도달하는 최소 예산, 전환 검출의 정밀도/재현율(계획이 `flip=True`라 한 항목이 실제 진실 복원 시 추천을 바꿨는가).
- **비교 대상**: (1) 본 발명 `ΔR_i/c_i` 순, (2) 무작위 순, (3) 균등(모든 항목 비용 비례 분할 — 실측 불가하므로 "라운드 로빈"으로 구현), (4) 최대 오차폭 우선(`range` 폭 큰 순 — Sampling-Where-It-Matters류 정확도 기준의 대리), (5) 오라클(진실을 알고 실제 후회 감소가 가장 큰 항목부터 — 상한).

**E-B1** 단일 fixture(`pd-rngd-gpu-card`), 위 전체 → `experiments/uncertainty/results/eb1_regret_vs_budget.md` + `figures/eb1_regret_vs_budget.png`(x: 예산 h, y: 후회, 5개 곡선, 시드 평균 ± 표준편차).
**E-B2** 전환 검출 정밀도/재현율 표(격자 m=3/5/9 비교) → `eb2_flip_detection.md`. 근사 규칙(`approximation=True`)이 쓰인 항목의 오탐률을 따로 보고.
**E-B3** `--resimulate-top 3` 켰을 때 vs 안 켰을 때의 ΔR 순위 상관(Spearman)과 소요 시간 → `eb3_closed_form_vs_resim.md`. 닫힌 형식 근사가 순위를 얼마나 보존하는지가 결론.

**금지**: 결과가 기대와 다를 때 열화 집합·시드·penalty를 조정해 결과를 좋게 만드는 것. 설정을 바꾸면 이유와 바꾸기 전 결과를 함께 기록.

### STEP B5 — 문서화

- `docs/uncertainty_planner.md`: 설계(§2 요약), CLI, 결과 표(E-A1, E-A2, E-B1~3), 한계(닫힌 형식 근사, 균등 가중 격자, 도메인 1차원(동시성만)).
- `docs/deviations.md`에 새 항목: 도메인 밖 후보 처리 정책 변경(opt-in), 근사 규칙이 근사인 kind 목록.
- `README.md`에 `--accuracy-domain`, `--measurement-plan` 한 단락.
- `CHANGELOG.md`.

**Stage B 완료 기준**: B1~B3 테스트 통과, golden 불변, E-B1~E-B3 결과 md·png 커밋, `docs/uncertainty_planner.md` 커밋.

---

## 5. 일정 추정 (Claude Code 작업 기준, 서버 점유 시간 제외)

| STEP | 예상 | 비고 |
|---|---|---|
| A1 | 1일 | 조사 필요 항목(스펙값·비용 로그) 포함 |
| A2 | 1.5일 | `_parse()` 확장 여부에 따라 |
| A3 | 1.5일 | 랭킹 리팩터가 필요하면 +0.5일 |
| A4 | 0.5일 | |
| A5 | E-A1 0.5일 / E-A2 0.5일 | 둘 다 시뮬레이션만(CPU). E-A2는 기존 real_c*.json에 대응하는 sim CSV 3~4개 생성 |
| B1 | 1일 | 조사 필요 2건 |
| B2 | 1.5일 | |
| B3 | 1일 | |
| B4 | 2일 | 시뮬레이션은 캐시 재사용, 재시뮬 옵션(E-B3)만 시간 소요 |
| B5 | 0.5일 | |

Stage A 종료 시점에 **임시명세서**(실시예 = A3 D22 재현 테스트 + E-A1/E-A2 표), Stage B 종료 시점에 **정규 명세서**(효과 = E-B1 후회-예산 곡선, E-B2 정밀도/재현율)를 쓸 수 있다.

---

## 6. 위험과 대응

| 위험 | 징후 | 대응 |
|---|---|---|
| 정확도 도메인 데이터가 RNGD 4점밖에 없음 | A40 validation 5점이 같은 동시성에 몰려 있음 | 사실대로 기록. A40 c-sweep은 Stage C(폐루프)에서 측정. 특허 실시예는 RNGD 4점으로 충분 |
| `LINK_BW` 닫힌 형식 규칙이 근사 | B1 조사에서 시뮬레이터가 전송을 큐잉과 결합 | `approximation=True`로 표기하고 E-B3에서 재시뮬 대비 순위 상관을 보고. 상관이 낮으면 측정 계획 상위 N은 재시뮬을 기본으로 |
| 캐시에 없는 후보 | E-A1에서 `pd_slo_sweep_margin18` 캐시 키가 현재 코드와 안 맞음(키 스키마 변경) | 재시뮬레이션(캐시 재생성) — 시간 소요를 결과 md에 기록. 키 스키마를 옛 것으로 되돌리지 말 것 |
| D23 livelock 후보 | tight-TTFT 후보가 시뮬 타임아웃 | `SIM_ERROR` 스테이지 그대로. 분석에서 제외하고 개수 보고. D23 진단은 이 지시서 범위 밖 |
| golden 변경 | `tests/test_search.py` 실패 | 기본 경로에 새 필드가 덤프되는지 확인 → `None` 기본 + 덤프 제외 |

---

## 7. 특허 증거 매핑

| 특허 2 명세서 항목 | 근거 산출물 | STEP |
|---|---|---|
| §5.2 레지스트리(도 2) | `render` 출력의 "불확실 입력" 표, `experiments/uncertainty/results/ea1_margin_modes.md`의 레지스트리 덤프 | A1, A4 |
| §5.3 정확도 도메인(도 3) | `profiles/calibration/rngd_card_edf.domain.yaml`, `ea2_rngd_domain.md` 4점 표 | A2, A5 |
| §5.4 후보별 마진·미측정 거절(도 4), §5.7 동작 예 | `tests/test_margin_policy.py`의 D22 재현 테스트, `ea1_margin_modes.md`의 (b) vs (c) 판정 차이 표 | A3, A5 |
| §5.5 섭동·전환 검출(도 5) | `eb2_flip_detection.md` 정밀도/재현율 | B1, B2, B4 |
| §5.6 측정 계획(도 6), 청구항 2의 ΔR_i | `eb1_regret_vs_budget.md/.png` — 본 발명 vs 무작위/균등/최대오차 우선/오라클 | B2, B3, B4 |
| §6 효과 | E-B1 곡선의 "동일 예산에서 후회 감소율", "후회 0 도달 예산 비율"; E-B3의 소요 시간(닫힌 형식 vs 재시뮬) | B4 |
| §5.8 관련 기능(리플랜·conformal) | 없음 — 언급만. 구현 금지(§1.4) | — |

**작성 규칙**: 명세서 §6의 수치는 이 표의 파일에서만 인용한다. 파일에 없는 수치는 명세서에 쓰지 않는다.

---

## 8. 이 지시서 이후의 후속 항목 (별도 지시서)

- **A/K 최소 보장치 수정** (`WORK_ORDER_pipeline_domain.md` §2.3·STEP 4, 특허1 §5.5 대응): S5 메모리 루프라인 하한의 `active`를 `min(max_num_seqs, kv_capacity)`에서 "최소 보장 동시 요청 수 × 최소 입력 길이"로 바꾸거나 포화 가정을 provenance에 기록. 오라클 일치 테스트가 그대로 통과해야 한다.
- **Stage C** — A40 폐루프: 실제 배치 → 잔차 측정 → `measure-apply` → 재계획. 리플랜 게이트(잔차 > p95 → 재적합; 이득 > 재구성 비용 + 불확실성 → 재배치).
- **Stage D** — conformal 하한 서로게이트, 다차원 도메인(동시성 × 프롬프트 길이 백분위).
- **CXL KV 풀** — `WORK_ORDER_cxl_kv_pool.md` (서버 확보 후).
