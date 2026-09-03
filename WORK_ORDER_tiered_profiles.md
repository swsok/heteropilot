# HeteroPilot 작업지시서 — Tiered Profile Supply (Tier 0 / Tier 1)

> 조사 보고서 `profiling_alternatives_report.md` §7의 제안(3-Tier Profile 공급 체계)을 구현하기 위한 작업지시서.
> 대상 저장소: `github.com/swsok/heteropilot` (upstream: `casys-kaist/LLMServingSim`)
> 작성일: 2026-09-02 · 작업 도구: Claude Code

---

## 0. 이 문서의 사용법

1. **STEP 순서를 반드시 지킬 것.** 각 STEP은 앞 STEP의 산출물에 의존한다.
2. **한 STEP = 한 브랜치 = 한 PR.** STEP을 건너뛰거나 합치지 말 것.
3. **각 STEP의 "테스트" 절을 통과하지 못하면 다음 STEP으로 넘어가지 않는다.** 테스트를 통과시키기 위해 테스트를 약화시키는 것은 금지. 구현이 틀렸다면 구현을 고칠 것.
4. **"조사 필요"라고 표시된 항목은 추측하지 말고 실제 코드/산출물을 읽어서 확인**한 뒤 그 결과를 PR 설명에 적을 것.
5. 이 문서와 기존 `WORK_ORDER_heteropilot.md`, `CLAUDE.md`, `AGENTS.md`가 충돌하면 **기존 문서가 우선**한다. 충돌을 발견하면 즉시 중단하고 사용자에게 보고할 것.

### 절대 규칙 (기존 프로젝트 규칙의 재확인 — 위반 시 작업 중단)

- **A1. 측정하지 않은 숫자를 측정값으로 표기하지 않는다.** Tier 0/1이 만든 숫자는 항상 `analytical` / `calibrated` 라벨을 달고, 그 라벨은 plan 결과까지 전파되어야 한다.
- **A2. 데이터가 없으면 identity / null이다.** 기본값을 그럴듯하게 채우지 않는다.
- **A3. 실측 데이터를 덮어쓰지 않는다.** 합성 번들은 항상 별도 hardware 라벨에 쓴다(§2.4).
- **A4. 기존 golden 회귀 출력이 바뀌면 안 된다.** Tier 0/1 도입은 opt-in이며, 기본 `plan` 동작을 바꾸지 않는다.

---

## 1. 배경과 목표

### 1.1 문제

LLMServingSim은 시뮬레이션을 돌리기 위해 `profiler/perf/<HARDWARE>/<org>/<model>/<variant>/tp<N>/` 번들을 요구한다. 이 번들은 **실제 하드웨어에서 vLLM을 돌려 측정**해야 얻어진다. 따라서:

- 보유하지 않은 accelerator(Ascend, Rebellions ATOM)는 번들을 만들 수 없다 → `sim_hardware: null` → cluster config가 upstream validation에서 실패.
- 미측정 config 축(fp8 KV-cache, TP≥4)도 같은 문제를 갖는다.

`profiles/accelerators/ascend_target.yaml` 이 이 gap을 명시적으로 기록하고 있다:

```yaml
sim_hardware: null
source: placeholder
notes: >
  Shape-only stub so heterogeneous candidate generation can be exercised before
  real Ascend data exists.
```

이것이 `docs/deviations.md` D4다.

### 1.2 해결 전략

**시뮬레이터를 교체하지 않는다.** 시뮬레이터는 `profiler/perf/**/*.csv` 만 읽으므로, **그 CSV를 무엇이 만들었는지는 시뮬레이터의 관심사가 아니다.** 따라서 CSV를 만드는 경로를 3계층으로 늘린다.

| Tier | 라벨 | 타깃 HW 측정 | 방법 | 기대 오차 | 용도 |
|---|---|---|---|---|---|
| **Tier 0** | `analytical` | **0회** | 데이터시트 + roofline(compute/memory) × efficiency | operator 10~15% | 미보유 HW candidate 탐색, sensitivity study |
| **Tier 1** | `calibrated` | 소량 (앵커 수~수십 점) | Tier 0 + per-kernel-family scaling fit | operator 4~8% | 실물 접근이 제한적인 HW |
| **Tier 2** | `measured` / `imported` | 전체 (~2.1h) | 현행 `python -m profiler profile` 또는 `CsvProfileImporter` | E2E ~1% | 최종 논문 수치 |

### 1.3 이 작업지시서가 만드는 것

1. `profiler/synth/` — Tier 0/1 번들 생성기 (신규 패키지)
2. `profiler/core/contract.py` — CSV 스키마의 공용 단일 출처 (importer에서 승격)
3. `planner/util/tier.py` — `ProfileTier` 해석 + plan 전파
4. `AcceleratorProfile.datasheet` — 데이터시트 필드 확장
5. `experiments/tier_validation/` — E1~E4 검증 하네스
6. 각 단계의 pytest 스위트

### 1.4 이 작업지시서가 만들지 않는 것 (구현 금지)

- **시뮬레이터 교체** (Vidur / Frontier / LLMCompass로의 이전). 발견 시 즉시 중단.
- **NeuSight / KernelSight-LM 코드의 직접 이식.** Tier 0의 정식화(roofline을 상한으로 두고 efficiency만 학습/보정)만 차용한다. 외부 학습 모델 도입은 이 지시서 범위 밖.
- **`serving/` 하위 upstream 코드 수정.** STEP 9까지는 `serving/`을 건드리지 않는다.
- **`profiler/core/` 의 측정 로직 변경.** STEP 1의 스키마 승격 리팩터만 예외이며, 동작 변경 없는 순수 이동이어야 한다.

---

## 2. 설계 개요

### 2.1 핵심 설계 결정 — 키는 기존 그리드 생성기에서 가져온다

Tier 0의 가장 큰 리스크는 **합성 CSV의 키가 실측 CSV의 키와 달라져 시뮬레이터의 보간이 깨지는 것**이다. 이를 원천적으로 막기 위해:

> **`profiler/core/categories.py` 의 그리드 생성기를 그대로 재사용해 키를 열거하고, "측정" 단계만 "해석적 추정"으로 대체한다.**

`categories.py`는 순수 Python이며 `torch`/`vllm`을 import하지 않는다(조사 확인). 재사용 대상:

- `categories_for(arch: Architecture, tp: int) -> list[Category]`
- `DenseCategory` / `SequenceCategory` / `AttentionCategory` / `ExpertCategory`
- `DensePoint` / `SequencePoint` / `AttentionPoint` / `ExpertPoint`
- `_token_grid(max_tokens)`, `_geometric_grid(max_value, start, factor)`

**반대로 `profiler/core/writer.py` 는 재사용하지 않는다.** 조사 결과 `writer.py`는 모듈 레벨에서 `torch`, `vllm`을 import하고 `_vllm_version()` / `_cuda_version()` / `_gpu_name()` 를 호출한다. Tier 0은 GPU도 vLLM도 없는 CPU 머신에서 돌아야 하므로 writer를 import하면 안 된다. 대신 CSV를 직접 쓰고, **`profiler/core/contract.py`(STEP 1에서 만들 공용 스키마)로 자기 출력을 검증**한다.

### 2.2 데이터 흐름

```
profiles/accelerators/<accel>.yaml   (datasheet: peak_flops, bw, pe_count, clock, l2, efficiency)
              │
              ├──> DeviceSpec  ─────────────┐
                                            │
configs/model/<org>/<model>.json  (HF config: hidden_size, intermediate_size, heads, vocab)
              │                             │
              ├──> ModelDims ───────┐        │
                                    ▼        ▼
profiler/models/<model_type>.yaml ─> ShapeResolver  ──> per-layer (FLOPs, bytes)
   (sequence + catalog)                    │
                                           ▼
profiler/core/categories.py ──> 키 열거 ──> RooflineModel.estimate(key) ──> time_us
                                                        │
                                                        ▼
                                              BundleEmitter (csv 직접 쓰기)
                                                        │
                                                        ▼
                              profiler/perf/<HW>-t0/<org>/<model>/<variant>/tp<N>/*.csv
                                                        │
                                              contract.validate_bundle()  ← 자기 검증
                                                        │
                                                        ▼
                                         planner/util/tier.py  ──> ProfileTier.ANALYTICAL
                                                        │
                                                        ▼
                                    PlannerOutput.provenance + .caveats
```

### 2.3 신규 디렉터리

```
profiler/
  core/
    contract.py          # NEW (STEP 1) — CSV 스키마 단일 출처. importer가 여기서 import.
  synth/                 # NEW 패키지
    __init__.py
    __main__.py          # CLI: python -m profiler.synth emit / calibrate / diff
    backend.py           # ProfileBackend ABC + AnalyticalProfileBackend + CalibratedProfileBackend
    device.py            # DeviceSpec: 데이터시트 → 계산에 쓰는 정규화된 값
    dims.py              # ModelDims: HF config + architecture yaml → 레이어별 shape
    shapes.py            # ShapeResolver: (layer, key) → (flops, bytes_moved)
    roofline.py          # RooflineModel: (flops, bytes) → time_us
    attn.py              # AttentionCostModel: 4축 그리드 전용
    emit.py              # BundleEmitter: 키 열거 + CSV/meta.yaml 쓰기
    calibrate.py         # Tier 1: 앵커 측정 → per-kernel-family scaling fit
    diff.py              # 두 번들 비교 리포트 (STEP 7)

planner/
  util/
    tier.py              # NEW — ProfileTier enum, 번들 tier 해석, 최저 tier 전파

experiments/
  tier_validation/       # NEW — E1~E4 하네스
    __init__.py
    e1_plan_agreement.py
    e2_budget_pareto.py
    e3_shape_overlap.py
    e4_sensitivity.py

tests/
  test_contract.py       # STEP 1
  test_tier.py           # STEP 2
  test_datasheet.py      # STEP 3
  test_synth_dims.py     # STEP 4
  test_synth_shapes.py   # STEP 4
  test_synth_roofline.py # STEP 5
  test_synth_attn.py     # STEP 6
  test_synth_emit.py     # STEP 7
  test_synth_diff.py     # STEP 8
  test_synth_calibrate.py# STEP 9
  test_tier_e2e.py       # STEP 10
  test_experiments_tier.py # STEP 11
```

### 2.4 하드웨어 라벨 규칙 (절대 규칙 A3의 구현)

합성 번들은 **반드시** 실측 번들과 다른 hardware 라벨을 쓴다.

| Tier | 라벨 규칙 | 예 |
|---|---|---|
| Tier 2 (measured) | 라벨 그대로 | `A40`, `RTXPRO6000`, `RNGD-CARD` |
| Tier 0 | `<LABEL>-t0` | `A40-t0`, `ASCEND_TARGET-t0` |
| Tier 1 | `<LABEL>-t1` | `RNGD-CARD-t1` |

- `profiles/accelerators/*.yaml` 의 `sim_hardware` 가 이 라벨을 명시적으로 가리킨다.
- 이렇게 하면 **실측 번들을 합성 번들이 조용히 가릴 수 없고**, 라벨만 보고 tier를 알 수 있다.
- 라벨 접미사는 tier 판정의 **보조 신호일 뿐**이며, 판정의 단일 출처는 `meta.yaml` 의 `tier` 필드다(§STEP 2).

---

## 3. 사전 조사 결과 (이미 있는 것 / 없는 것)

Claude Code는 이 표를 **먼저 직접 확인**하고 다르면 보고할 것. (2026-09-02 기준 확인 결과)

| 항목 | 상태 | 위치 |
|---|---|---|
| CSV 컨트랙트 문서 | **이미 존재** | `profiler/CONTRACT.md` (128행, work order §3.7 완료) |
| CSV 스키마 검증 코드 | **이미 존재하지만 private** | `profiler/core/importer.py` 의 `_CsvSchema`, `_SCHEMAS` |
| 외부 CSV 임포터 | **이미 존재** | `profiler/core/importer.py::CsvProfileImporter` + `tests/test_csv_importer.py` |
| 허용 source 값 | `("imported","measured","placeholder")` | `importer.py::_ALLOWED_SOURCES` — **`analytical`/`calibrated` 추가 필요** |
| 하드웨어 numbers provenance enum | **이미 존재** | `planner/inventory.py::Source` = MEASURED / VENDOR_SPEC / PLACEHOLDER |
| provenance가 plan까지 전파되는가 | **아니오** | `Source`는 `planner/__main__.py:79` 의 `inspect` 출력에만 쓰인다 |
| PlannerOutput의 확장 슬롯 | **이미 존재** | `plan.py::PlannerOutput.provenance: dict`, `.caveats: list[str]` |
| 실험 metadata 수집기 | **이미 존재** | `planner/util/provenance.py` (§3.8) |
| memory roofline 하한 | **이미 존재** | `candidate_generator.py::_stage5_analytical_ok` (memory BW만, decode만, 하한만) |
| roofline goodput proxy | **이미 존재** | `optimizer/greedy.py::rank`, `optimizer/surrogate.py::AnalyticalRooflineRanker` |
| 그리드 생성기 (순수 Python) | **이미 존재** | `profiler/core/categories.py` |
| CSV writer | **존재하나 재사용 불가** | `profiler/core/writer.py` — 모듈 레벨에서 `torch`, `vllm` import |
| 데이터시트 필드 | **없음** | `AcceleratorProfile` 에는 `memory_gb`, `memory_bandwidth_gbps` 만 있음 |
| Tier 0/1 생성기 | **없음** | 본 작업지시서의 산출물 |

---

## 4. 공통 규칙

### 4.1 Git

```
브랜치: feat/tier-step<N>-<slug>      예: feat/tier-step04-shape-resolver
커밋:   step<N>: <무엇을 했는가>
PR:     한 STEP 당 하나. 본문에 (a) 무엇을 만들었는지 (b) 테스트 결과 (c) 조사 필요 항목의 결론
```

### 4.2 테스트 게이트 (모든 STEP 공통)

```bash
pytest -q                       # 전체 통과
ruff check .                    # 통과
mypy                            # 통과
pytest -q tests/test_<step>.py  # 해당 STEP 스위트
```

**중요 — 신규 코드가 린트/타입 검사에서 제외되어 있다.** 조사 결과 `pyproject.toml`은:

```toml
[tool.ruff]
extend-exclude = ["astra-sim", "serving", "profiler", "bench", "workloads", "scripts", "docs"]

[tool.mypy]
files = ["planner", "scenariolab"]
[[tool.mypy.overrides]]
module = ["serving.*", "profiler.*", "bench.*"]
ignore_errors = true
```

→ `profiler/synth/` 와 `profiler/core/contract.py` 를 검사 대상으로 되돌리는 것이 **STEP 0의 작업**이다.

### 4.3 테스트 작성 원칙

- **GPU / vLLM / 네트워크에 의존하는 테스트를 만들지 않는다.** 모든 STEP의 테스트는 CPU-only CI에서 돌아야 한다.
- **실측 번들을 픽스처로 쓴다.** 레포에 동봉된 `profiler/perf/RTXPRO6000/meta-llama/Llama-3.1-8B/bf16/tp1/` 이 유일한 ground truth다. 임의 숫자를 기대값으로 하드코딩하지 말 것.
- **수치 비교는 반드시 허용 오차(tolerance)와 그 근거를 함께 적는다.** `assert t == 12.3` 금지, `assert t == pytest.approx(12.3, rel=0.05)  # 근거: ...` 형태.
- **각 테스트 함수에 한 줄 docstring으로 "무엇을 보장하는가"를 적는다.**
- `tmp_path` 픽스처를 쓰고, 저장소 안에 테스트 산출물을 남기지 않는다.

---

# STEP 0. 준비 — baseline 고정과 검사 범위 확장

## 목표
Tier 작업 전 상태를 고정하고, 신규 코드가 실제로 린트/타입 검사를 받게 만든다.

## 지시

1. **baseline 확인.** 현재 `main`에서 아래를 실행하고 결과를 `docs/tier_baseline.md` 에 기록한다.
   ```bash
   pytest -q 2>&1 | tail -5
   ruff check . 2>&1 | tail -3
   mypy 2>&1 | tail -3
   git rev-parse HEAD
   ```
2. **`pyproject.toml` 검사 범위 조정.**
   - `[tool.ruff] extend-exclude` 에서 `"profiler"` 를 제거하고, 대신 `"profiler/core"`, `"profiler/v0"`, `"profiler/models"`, `"profiler/power"`, `"profiler/perf"` 를 나열한다. → `profiler/synth/` 는 검사 대상이 된다.
   - `profiler/core/contract.py` 는 STEP 1에서 만들 파일이므로, 이 파일만 예외적으로 검사 대상에 넣기 위해 `extend-exclude` 대신 `[tool.ruff.lint.per-file-ignores]` 를 쓰지 말고, `extend-exclude` 에 `"profiler/core"` 를 넣은 뒤 STEP 1에서 `!profiler/core/contract.py` 형태의 negation을 추가한다. **ruff가 negation 패턴을 지원하는지 조사 필요** — 지원하지 않으면 `contract.py` 를 `profiler/contract.py`(core 밖)에 두는 대안을 택하고 그 결정을 PR에 기록한다.
   - `[tool.mypy] files` 에 `"profiler/synth"` 를 추가한다.
   - `[[tool.mypy.overrides]]` 의 `module` 을 `["serving.*", "profiler.core.*", "profiler.v0.*", "bench.*"]` 로 좁힌다.
3. **`profiler/synth/__init__.py` 를 빈 패키지로 생성**하고, `torch`/`vllm` 을 import하지 않는다는 규칙을 docstring에 적는다.

## 테스트

`tests/test_synth_import_hygiene.py`

```python
"""profiler.synth 가 GPU 스택 없이 import되는지 보장 (STEP 0)."""

def test_synth_imports_without_torch_or_vllm(monkeypatch):
    """torch / vllm 이 sys.modules 에서 차단된 상태에서도 profiler.synth 는 import된다."""
    # 지시: sys.meta_path 에 torch/vllm 을 ImportError 로 만드는 finder 를 심고,
    #       importlib.reload 로 profiler.synth 하위 모든 모듈을 import 시도.
    #       STEP 0 시점에는 __init__ 만 존재하므로 그것만 검사.

def test_no_writer_import_in_synth():
    """profiler.synth 의 어떤 모듈도 profiler.core.writer 를 import 하지 않는다."""
    # 지시: ast 로 profiler/synth/**/*.py 를 파싱해 import 목록을 수집하고
    #       {"torch", "vllm", "profiler.core.writer"} 와 교집합이 비어 있음을 assert.
    #       이 테스트는 이후 모든 STEP에서 회귀 방지 역할을 한다 — 절대 삭제 금지.
```

추가 검증 명령:

```bash
ruff check profiler/synth          # "no files to check" 가 아니어야 한다 (실제로 검사되는지 확인)
mypy profiler/synth                # 대상에 포함되었는지 확인
pytest -q                          # baseline 과 동일한 결과
```

## 완료 조건

- [ ] `docs/tier_baseline.md` 에 baseline 기록됨
- [ ] `ruff check profiler/synth` 가 실제로 파일을 검사한다 (빈 결과가 아님)
- [ ] `mypy` 출력에 `profiler/synth` 가 포함된다
- [ ] `pytest -q` 결과가 baseline과 동일 (신규 2개 테스트만 증가)

---

# STEP 1. CSV 컨트랙트를 공용 모듈로 승격

## 목표
`importer.py` 안에 private으로 갇혀 있는 CSV 스키마를 `profiler/core/contract.py` 로 옮겨, importer와 synth 양쪽이 **하나의 출처**를 쓰게 한다. **동작 변경 없는 순수 리팩터**여야 한다.

## 배경
`profiler/core/importer.py` 에는 다음이 private으로 존재한다 (조사 확인):
- `class _CsvSchema` (filename, columns, int_columns, float_columns, str_columns, key_columns, time_column, required, moe_only)
- `_SCHEMAS: tuple[_CsvSchema, ...]` — 6개 파일 스키마
- `_as_int`, `_as_float`, `_cast`
- `_TP_DIR_RE`, `_ALLOWED_SOURCES`
- `class ProfileContractError`

## 지시

1. `profiler/core/contract.py` 를 만들고 위 심볼을 **public 이름으로** 옮긴다.
   - `_CsvSchema` → `CsvSchema`
   - `_SCHEMAS` → `SCHEMAS`
   - `_as_int` / `_as_float` → `as_int` / `as_float`
   - `_cast` → `cast_value`
   - `_TP_DIR_RE` → `TP_DIR_RE`
   - `_ALLOWED_SOURCES` → `ALLOWED_SOURCES`
   - `ProfileContractError` 도 이동
   - **`torch`/`vllm`/`pandas` 를 import하지 않는다.** 표준 라이브러리 + `yaml` 만 허용.
2. `ALLOWED_SOURCES` 에 `"analytical"`, `"calibrated"` 를 추가한다.
   → 현재 `("imported", "measured", "placeholder")` → `("imported", "measured", "placeholder", "analytical", "calibrated")`
3. `importer.py` 는 `from profiler.core.contract import ...` 로 바꾸고, **하위 호환을 위해 기존 private 이름을 alias로 남긴다** (`_SCHEMAS = SCHEMAS` 등). 기존 테스트가 private 이름을 참조할 수 있으므로 먼저 `grep -n "_SCHEMAS\|_CsvSchema\|_ALLOWED_SOURCES" tests/` 로 확인할 것.
4. `contract.py` 에 스키마를 이용하는 **읽기 전용 검증 함수**를 추가한다. 이 함수는 importer의 검증 로직에서 순수 부분만 추출한 것이어야 한다.
   ```python
   def schema_for(filename: str) -> CsvSchema: ...
   def validate_csv(path: Path, schema: CsvSchema) -> None:
       """헤더 byte-for-byte, 타입, 키 유일성, time_us > 0 검사. 위반 시 ProfileContractError."""
   def validate_bundle(variant_root: Path, tp_degrees: list[int], is_moe: bool) -> None:
       """번들 전체 검증. importer 와 synth 가 공용으로 쓴다."""
   ```
   **importer의 기존 검증 동작을 바꾸지 말 것.** importer가 이미 하는 검사를 `contract.validate_*` 가 재현하되, importer 쪽 리팩터가 부담되면 importer는 그대로 두고 `contract` 에 함수만 추가해도 된다. 단 두 구현이 같은 `SCHEMAS` 를 참조해야 한다.
5. `profiler/CONTRACT.md` 에 절을 추가한다.
   - `meta.yaml` 의 `source` 허용값에 `analytical`, `calibrated` 추가
   - **신규 필수 필드 `tier`** 를 문서화: `measured` | `imported` | `calibrated` | `analytical` | `placeholder`
   - 합성 번들의 hardware 라벨 접미사 규칙(§2.4)
   - Tier 0/1 번들이 `meta.yaml` 에 반드시 기록해야 하는 것: `cost_model`(예: `roofline-v1`), `datasheet_source`, `efficiency` 값들, `calibration_anchors`(Tier 1), `generated_at`, `generator_version`

## 테스트

`tests/test_contract.py`

```python
"""profiler.core.contract 가 CSV 컨트랙트의 단일 출처임을 보장 (STEP 1)."""

def test_schemas_cover_all_contract_files():
    """SCHEMAS 가 CONTRACT.md 에 문서화된 6개 파일을 모두 덮는다."""
    # 지시: CONTRACT.md 를 파싱해 "### <name>.csv" 헤딩을 수집하고
    #       {s.filename for s in SCHEMAS} 와 정확히 일치함을 assert.
    #       (문서와 코드가 갈라지는 것을 CI가 잡게 만드는 것이 목적)

def test_columns_match_real_bundle_headers():
    """SCHEMAS 의 columns 가 레포 동봉 실측 번들의 헤더와 byte-for-byte 일치한다."""
    # 지시: profiler/perf/RTXPRO6000/meta-llama/Llama-3.1-8B/bf16/tp1/*.csv 의
    #       첫 줄을 읽어 schema.columns 와 비교. 이것이 STEP 7 emit 정확성의 근거가 된다.

def test_allowed_sources_extended():
    """analytical / calibrated 가 허용 source 에 포함된다."""

def test_validate_csv_accepts_real_bundle():
    """실측 번들의 모든 CSV 가 validate_csv 를 통과한다."""
    # 지시: 동봉된 RTXPRO6000 / A40 / RNGD-CARD 번들 전부에 대해 루프.
    #       RNGD-CARD 는 skew*.csv 가 없으므로 optional 처리가 맞는지도 함께 확인.

def test_validate_csv_rejects_renamed_column(tmp_path):
    """헤더 컬럼명을 하나 바꾸면 ProfileContractError."""

def test_validate_csv_rejects_duplicate_key(tmp_path):
    """같은 키가 두 번 나오면 ProfileContractError."""

def test_validate_csv_rejects_nonpositive_time(tmp_path):
    """time_us <= 0 이면 ProfileContractError."""

def test_validate_csv_allows_negative_alpha_in_skew(tmp_path):
    """skew.csv 의 alpha / t_*_us 는 음수가 허용된다 (기존 importer 동작 보존)."""

def test_importer_behaviour_unchanged():
    """리팩터 후에도 importer 의 공개 동작이 동일하다."""
    # 지시: tests/test_csv_importer.py 를 그대로 재실행하는 것으로 충분.
    #       이 테스트는 "pytest tests/test_csv_importer.py 가 통과한다" 는 사실을
    #       PR 설명에 적는 것으로 대체 가능.
```

실행:

```bash
pytest -q tests/test_contract.py tests/test_csv_importer.py
git diff --stat            # importer.py 의 변경이 import 문 + alias 로 한정되는지 육안 확인
```

## 완료 조건

- [ ] `tests/test_csv_importer.py` **26개 테스트가 하나도 깨지지 않는다** (리팩터 무해성 증명)
- [ ] `tests/test_contract.py` 전체 통과
- [ ] `contract.py` 의 import 목록에 `torch`, `vllm`, `pandas` 가 없다
- [ ] `profiler/CONTRACT.md` 에 `tier` 필드와 라벨 규칙이 문서화됨

---

# STEP 2. ProfileTier 도입과 plan 전파

## 목표
번들의 tier를 읽어 plan 결과까지 전파한다. **이 STEP만으로 절대 규칙 A1이 구조적으로 지켜진다.** Tier 0 생성기가 없어도 독립적으로 가치가 있으므로 먼저 만든다.

## 지시

1. `planner/util/tier.py` 생성.
   ```python
   class ProfileTier(str, enum.Enum):
       MEASURED = "measured"       # tier 2 — 실측
       IMPORTED = "imported"       # tier 2 — 외부 실측 import
       CALIBRATED = "calibrated"   # tier 1
       ANALYTICAL = "analytical"   # tier 0
       PLACEHOLDER = "placeholder" # 번들 없음 / 신뢰 불가
       UNKNOWN = "unknown"         # meta.yaml 에 tier 도, source 도 없음

       @property
       def rank(self) -> int:
           """신뢰도 순위. 낮을수록 신뢰도 낮음. 최저 tier 전파에 쓴다."""

       @property
       def is_measurement(self) -> bool:
           """MEASURED / IMPORTED 만 True."""
   ```
2. 번들 tier 해석 함수.
   ```python
   def resolve_bundle_tier(perf_root: Path, hardware: str, model: str, variant: str) -> ProfileTier:
       """meta.yaml 의 tier 필드를 읽는다.

       판정 우선순위 (A2 준수):
         1. meta.yaml 의 `tier` 필드가 있으면 그것.
         2. 없으면 `source` 필드로 매핑 (measured/imported/placeholder).
         3. 둘 다 없으면 UNKNOWN. **절대 MEASURED 로 추측하지 않는다.**
       번들 디렉터리가 없으면 PLACEHOLDER.
       하드웨어 라벨의 -t0/-t1 접미사는 보조 신호로만 쓰고,
       meta.yaml 과 모순되면 경고를 반환한다(예외를 던지지는 않는다).
       """

   def min_tier(tiers: Iterable[ProfileTier]) -> ProfileTier:
       """가장 신뢰도 낮은 tier. 빈 입력은 UNKNOWN."""

   def caveat_for(tier: ProfileTier) -> str | None:
       """Tier 0/1/placeholder/unknown 에 대해 리포트에 실을 경고 문구. tier 2 는 None."""
   ```
3. `caveat_for` 문구는 다음 형태로 고정한다 (테스트가 이 문자열을 검사한다).
   ```
   ANALYTICAL   → "simulator-only (analytical inputs): <hardware> profile is datasheet-derived, not measured"
   CALIBRATED   → "simulator-only (calibrated inputs): <hardware> profile is analytical + limited anchors"
   PLACEHOLDER  → "no profile bundle for <hardware>: this plan cannot be reported as a result"
   UNKNOWN      → "profile provenance unknown for <hardware>: meta.yaml records neither tier nor source"
   ```
4. **plan 파이프라인에 배관.** `planner/plan.py::PlannerOutput` 에 필드를 추가한다.
   ```python
   profile_tier: str = ProfileTier.UNKNOWN.value   # plan 을 구성한 모든 island 의 최저 tier
   profile_tiers: dict[str, str] = Field(default_factory=dict)  # island_id -> tier
   ```
   `DeploymentPlan` 에도 같은 정보를 넣을지는 **조사 필요**: `ScoredPlan`/`DeploymentPlan` 중 어디에 두는 것이 렌더러/YAML 직렬화에 자연스러운지 `planner/render.py` 를 읽고 결정하고, 결정 근거를 PR에 적을 것.
5. 최저 tier가 `MEASURED`/`IMPORTED` 가 아니면 `PlannerOutput.caveats` 에 `caveat_for(...)` 결과를 **반드시** 넣는다.
6. `planner/render.py` 에 배너를 추가한다. Tier 0/1/placeholder/unknown plan은 출력 최상단에 눈에 띄게 표시한다.
7. `planner/util/provenance.py` 가 만드는 §3.8 metadata 블록에 `profile_tier` 와 `profile_tiers` 를 포함시킨다.

## 테스트

`tests/test_tier.py`

```python
"""ProfileTier 해석과 plan 전파를 보장 (STEP 2)."""

# --- 해석 ---
def test_resolve_reads_tier_field(tmp_path):
    """meta.yaml 에 tier: analytical 이면 ANALYTICAL."""

def test_resolve_falls_back_to_source(tmp_path):
    """tier 필드가 없고 source: measured 면 MEASURED."""

def test_resolve_returns_unknown_when_no_provenance(tmp_path):
    """tier 도 source 도 없으면 UNKNOWN. MEASURED 로 추측하지 않는다 (A2)."""

def test_resolve_returns_placeholder_when_bundle_missing(tmp_path):
    """번들 디렉터리가 없으면 PLACEHOLDER."""

def test_resolve_real_bundles_are_measured():
    """레포 동봉 실측 번들 3종이 MEASURED 로 해석된다."""
    # 지시: RTXPRO6000 / A40 / RNGD-CARD. meta.yaml 에 tier 필드가 없다면
    #       source 로 폴백되는지, 혹은 source 도 없어 UNKNOWN 이 되는지 확인하고
    #       UNKNOWN 이면 STEP 2의 부수 작업으로 실측 번들 meta.yaml 에
    #       tier: measured 를 추가한다 (측정 사실을 기록하는 것이므로 A1 위반 아님).

def test_label_suffix_mismatch_warns_not_raises(tmp_path):
    """하드웨어 라벨이 -t0 인데 meta.yaml 이 measured 면 경고를 반환하고 예외는 없다."""

# --- 최저 tier ---
@pytest.mark.parametrize("tiers,expected", [...])
def test_min_tier(tiers, expected):
    """혼합 tier 집합에서 가장 신뢰도 낮은 것이 선택된다."""
    # 지시: 최소 케이스 — [MEASURED, ANALYTICAL] -> ANALYTICAL,
    #       [MEASURED, IMPORTED] -> IMPORTED 또는 MEASURED (rank 정의에 따라, 근거를 적을 것),
    #       [] -> UNKNOWN

# --- 전파 ---
def test_planner_output_carries_min_tier(monkeypatch):
    """이기종 plan(GPU measured + NPU analytical)의 profile_tier 가 analytical 이다."""
    # 지시: MockPredictor(기존 tests/conftest.py 의 패턴 재사용)로 시뮬레이터를 대체.
    #       cluster spec 은 tmp_path 에 만들고 두 island 를 서로 다른 tier 로 둔다.

def test_caveat_present_for_analytical_plan(monkeypatch):
    """analytical 이 섞인 plan 의 caveats 에 정확한 문구가 포함된다."""
    # 지시: §STEP 2 지시 3의 문자열을 그대로 검사.

def test_no_caveat_for_all_measured_plan(monkeypatch):
    """전부 measured 인 plan 에는 tier caveat 이 붙지 않는다 (기존 동작 보존)."""

def test_render_shows_banner(capsys):
    """render 가 analytical plan 에 배너를 출력한다."""

def test_provenance_block_includes_tier():
    """§3.8 metadata 에 profile_tier / profile_tiers 가 들어간다."""
```

**골든 회귀 테스트 (A4 확인)**

```bash
pytest -q                     # 기존 golden 테스트가 전부 통과해야 한다
git diff --stat tests/        # 기존 golden 파일이 수정되었다면 그 이유를 PR 에 명시
```

> 기존 golden 출력에 `profile_tier` 필드가 추가되어 깨지면, **golden을 갱신하는 것이 맞다** (새 필드 추가는 의도된 변경). 단 값이 `measured` 여야 하며 `unknown` 이 나오면 실측 번들 meta.yaml 보강이 필요하다는 신호다.

## 완료 조건

- [ ] `tests/test_tier.py` 전체 통과
- [ ] 실측 번들 3종이 `MEASURED` 로 해석된다 (필요 시 meta.yaml 에 `tier: measured` 추가)
- [ ] `plan` 출력 YAML에 `profile_tier` 가 나타난다
- [ ] `pytest -q` 전체 통과 (golden 갱신은 필드 추가에 한정)

---

# STEP 3. 데이터시트 스키마 확장

## 목표
Tier 0가 필요한 하드웨어 파라미터를 `profiles/accelerators/*.yaml` 로 표현할 수 있게 한다. **숫자를 채우는 것이 아니라 채울 자리를 만드는 것**이 이 STEP의 목표다.

## 지시

1. `planner/inventory.py` 에 `Datasheet` 모델을 추가한다.
   ```python
   class Datasheet(_Strict):
       """Tier 0 roofline 계산에 필요한 데이터시트 값.

       모든 필드가 optional 이다. 없으면 Tier 0 생성이 실패해야 하며(A2),
       기본값을 그럴듯하게 채우지 않는다.
       """
       #: dtype -> dense peak TFLOP/s. 키는 'bf16','fp16','fp8','int8' 등.
       peak_tflops: dict[str, float] = Field(default_factory=dict)
       #: 연산 유닛 수 (GPU: SM, NPU: PE cluster / core). 이름은 backend 중립으로.
       compute_units: int | None = Field(default=None, gt=0)
       clock_mhz: float | None = Field(default=None, gt=0)
       l2_cache_mb: float | None = Field(default=None, gt=0)
       #: 커널 실행 오버헤드 하한 (us). KernelSight-LM 의 t_0 항에 해당.
       kernel_launch_us: float | None = Field(default=None, gt=0)
       #: roofline derating. 없으면 Tier 0 생성 불가 (A2).
       flops_efficiency: float | None = Field(default=None, gt=0, le=1)
       mem_efficiency: float | None = Field(default=None, gt=0, le=1)
       #: kernel family 별 override. 키 예: 'gemm','attention','elementwise','moe'.
       family_efficiency: dict[str, float] = Field(default_factory=dict)
       #: 이 숫자들의 출처. A1 준수를 위해 필수.
       datasheet_source: str = ""
   ```
2. `AcceleratorProfile` 에 `datasheet: Datasheet | None = None` 를 추가한다. **기존 필드는 건드리지 않는다.**
3. `Source` enum에 값을 추가하지 **않는다.** `Source` 는 accelerator YAML 숫자의 출처(measured/vendor_spec/placeholder)이고, `ProfileTier` 는 번들의 tier다. 두 개념을 섞지 말 것. 대신 `Datasheet.datasheet_source` 를 필수 기술로 삼는다.
4. **검증 규칙 추가.** `AcceleratorProfile` 에 model validator를 추가:
   - `sim_hardware` 가 `-t0` / `-t1` 로 끝나면 `datasheet` 가 반드시 있어야 한다.
   - `datasheet` 가 있으면 `datasheet_source` 가 빈 문자열이어서는 안 된다.
5. **기존 YAML 파일에 숫자를 채우는 것은 이 STEP의 범위가 아니다.** 단 하나의 예외로, `profiles/accelerators/a40.yaml` 에 **공개 데이터시트에서 확인 가능한 값만** 채운다(A40: bf16 dense peak, SM 수, clock, L2). 각 값 옆에 출처 URL을 주석으로 남긴다. 이것이 STEP 8 검증의 대조군이 된다.
   - A40의 `flops_efficiency` / `mem_efficiency` 는 **채우지 않는다.** 이 값은 측정에서 나와야 하며 STEP 8에서 fit한다. 지금 채우면 A1 위반이다.
6. `docs/deviations.md` 에 항목을 추가한다: "Tier 0 도입 — datasheet 필드는 vendor spec이며 실측이 아니다."

## 테스트

`tests/test_datasheet.py`

```python
"""Datasheet 스키마와 검증 규칙을 보장 (STEP 3)."""

def test_all_shipped_profiles_still_load():
    """profiles/accelerators/*.yaml 전부가 여전히 파싱된다 (하위 호환)."""
    # 지시: 7개 파일 전부 루프. datasheet 없는 파일도 정상 로드되어야 한다.

def test_datasheet_optional():
    """datasheet 없는 프로필도 유효하다."""

def test_t0_sim_hardware_requires_datasheet():
    """sim_hardware 가 -t0 로 끝나는데 datasheet 가 없으면 ValidationError."""

def test_datasheet_requires_source():
    """datasheet 가 있는데 datasheet_source 가 비면 ValidationError."""

def test_efficiency_bounds():
    """flops_efficiency / mem_efficiency 는 (0, 1] 범위만 허용."""

def test_a40_datasheet_has_source_urls():
    """a40.yaml 의 datasheet 값에 출처가 기록되어 있다."""
    # 지시: datasheet_source 가 비어 있지 않고, YAML 원문에 URL 문자열이 있음을 확인.

def test_a40_efficiency_not_prefilled():
    """A1: A40 의 efficiency 는 측정 전이므로 채워져 있지 않다."""
    # 지시: flops_efficiency is None and mem_efficiency is None
    #       STEP 8 에서 fit 된 이후에는 이 테스트를 갱신하고 그 사실을 PR 에 적는다.

def test_inventory_golden_unchanged():
    """기존 tests/test_inventory.py 가 깨지지 않는다."""
```

실행:

```bash
pytest -q tests/test_datasheet.py tests/test_inventory.py
python -m planner inspect --cluster <기존 예제 cluster>   # 회귀 확인 (출력이 같아야 한다)
```

## 완료 조건

- [ ] 기존 7개 accelerator YAML이 모두 로드된다
- [ ] `-t0` 라벨 + datasheet 부재 조합이 거부된다
- [ ] `a40.yaml` 에 출처 주석과 함께 데이터시트 값이 들어갔고 efficiency는 비어 있다
- [ ] `tests/test_inventory.py` 무회귀

---

# STEP 4. ModelDims + ShapeResolver — 레이어별 shape 해석

## 목표
`(모델, TP, 레이어명, 키)` → `(FLOPs, moved_bytes)` 를 계산한다. 이것이 Tier 0의 물리적 기초다.

## 배경 — 왜 이게 가능한가

Transformer decoder의 모든 dense 레이어 shape은 소수의 아키텍처 상수로 결정된다. 런타임에 변하는 차원은 `T`(배치 총 토큰 수) 하나뿐이다.

| 레이어 (`dense.csv` 의 `layer` 값) | vLLM 클래스 | shape |
|---|---|---|
| `qkv_proj` | QKVParallelLinear | `[T × d_model] × [d_model × (n_q + 2·n_kv)·d_head / TP]` |
| `o_proj` | RowParallelLinear | `[T × (n_q·d_head/TP)] × [(n_q·d_head/TP) × d_model]` |
| `gate_up_proj` | MergedColumnParallelLinear | `[T × d_model] × [d_model × 2·d_ff/TP]` |
| `down_proj` | RowParallelLinear | `[T × d_ff/TP] × [d_ff/TP × d_model]` |
| `layernorm`, `final_layernorm` | RMSNorm | elementwise, `O(T · d_model)` |
| `act_fn` | SiluAndMul | elementwise, 입력 `2·d_ff/TP`, 출력 `d_ff/TP` |
| `rotary_emb` | Llama3RotaryEmbedding | elementwise, `O(T · (n_q + n_kv)·d_head / TP)` |
| `embedding` | VocabParallelEmbedding | gather, `O(T · d_model)` |
| `lm_head` (per_sequence) | LogitsProcessor | `[S × d_model] × [d_model × V/TP]` |
| `sampler` (per_sequence) | Sampler | `O(S · V/TP)` |
| `moe` (moe.csv) | FusedMoE | `[T·k × d_model] × [d_model × 2·d_ff_moe/TP]` + down |

레이어 목록과 카테고리는 **추측하지 말고** `profiler/models/<model_type>.yaml` 의 `sequence:` / `catalog:` 에서 읽는다. `tp_stable: true` 인 레이어는 TP로 나뉘지 않는다.

## 지시

1. `profiler/synth/dims.py` — `ModelDims`.
   ```python
   @dataclass(frozen=True)
   class ModelDims:
       model: str
       model_type: str          # HF config 의 model_type (architecture yaml 선택 키)
       num_hidden_layers: int
       hidden_size: int         # d_model
       intermediate_size: int   # d_ff
       num_attention_heads: int # n_q
       num_key_value_heads: int # n_kv
       head_dim: int
       vocab_size: int
       dtype_bytes: int         # 가중치/활성 dtype
       kv_dtype_bytes: int      # KV cache dtype (fp8 variant 대응)
       # MoE
       num_experts: int | None = None
       experts_per_token: int | None = None
       moe_intermediate_size: int | None = None

       @classmethod
       def from_hf_config(cls, path: Path, variant: str) -> "ModelDims": ...
   ```
   - `configs/model/<org>/<model>.json` 의 HF config를 읽는다.
   - **`head_dim` 은 HF config에 없다** (2026-09-02 확인: `Llama-3.1-8B.json`, `Llama-3.1-70B.json` 모두 부재). `hidden_size // num_attention_heads` 로 유도하고, 유도했다는 사실과 확인 날짜를 docstring에 적는다. 단 **`head_dim` 키가 존재하는 config가 오면 그 값을 우선**한다 (일부 모델은 `hidden_size / n_heads` 와 다르다).
   - 참고 실측값 (테스트 기대값의 출처로 사용): Llama-3.1-8B = `d_model 4096`, `d_ff 14336`, `n_q 32`, `n_kv 8`, `head_dim 128`(유도), `vocab 128256`, `layers 32`. MoE 필드명은 `Qwen3-30B-A3B` config를 직접 열어 확인할 것.
   - `variant` 문자열(`bf16`, `bf16-kvfp8`)에서 dtype을 파싱한다. **파싱 규칙은 `profiler/CONTRACT.md` 의 variant 정의를 따를 것.**
   - 필수 필드가 없으면 예외를 던진다. 기본값으로 때우지 않는다(A2).
2. `profiler/synth/shapes.py` — `ShapeResolver`.
   ```python
   @dataclass(frozen=True)
   class OpCost:
       flops: float
       bytes_moved: float
       family: str      # 'gemm' | 'elementwise' | 'gather' | 'attention' | 'moe'

   class ShapeResolver:
       def __init__(self, dims: ModelDims, arch: Architecture, tp: int) -> None: ...

       def dense(self, layer: str, tokens: int) -> OpCost: ...
       def per_sequence(self, layer: str, sequences: int) -> OpCost: ...
       def expert(self, tokens: int, activated_experts: int) -> OpCost: ...
       def layers(self) -> list[str]:
           """architecture yaml 의 catalog 에서 읽은 dense 레이어 목록."""
   ```
   - GEMM: `flops = 2 · M · N · K`, `bytes = (M·K + K·N + M·N) · dtype_bytes`
   - elementwise: `flops = c · elems` (c는 레이어별 상수, 근거를 주석으로), `bytes = (in + out) · dtype_bytes`
   - **`tp_stable` 레이어는 TP로 나누지 않는다.** architecture yaml의 플래그를 읽어서 처리.
   - 알 수 없는 레이어명이 오면 예외를 던진다. **조용히 0을 반환하지 말 것** — 그러면 Tier 0가 그 레이어를 무료로 취급해 결과가 조용히 틀린다.
3. **가중치 재사용을 명시적으로 다룰 것.** `bytes_moved` 에 가중치를 매번 포함하면 큰 `T`에서 memory-bound로 잘못 판정된다. `T`가 클 때는 가중치가 캐시/재사용되므로, `bytes_moved` 를 `max(weight_bytes, activation_bytes)` 로 볼 것인지 합으로 볼 것인지 **두 변형을 모두 구현하고 STEP 8에서 실측으로 고른다.** 선택 가능한 옵션으로 두고 기본값을 문서화할 것.

## 테스트

`tests/test_synth_dims.py`

```python
"""ModelDims 가 HF config 를 정확히 읽는지 보장 (STEP 4)."""

def test_llama31_8b_dims():
    """Llama-3.1-8B 의 차원이 HF config 와 일치한다."""
    # 지시: configs/model/meta-llama/Llama-3.1-8B.json 을 직접 열어 기대값을 뽑고,
    #       하드코딩하지 말고 json 을 읽어 비교 (파일이 바뀌면 테스트도 따라간다).

def test_qwen3_moe_dims():
    """Qwen3-30B-A3B 의 MoE 필드(num_experts, experts_per_token)가 채워진다."""

def test_kv_dtype_from_variant():
    """variant 'bf16-kvfp8' 이면 dtype_bytes=2, kv_dtype_bytes=1."""

def test_missing_field_raises(tmp_path):
    """필수 필드가 없는 config 는 예외. 기본값으로 때우지 않는다 (A2)."""

def test_head_dim_derivation_documented():
    """head_dim 유도 규칙이 docstring 에 기록되어 있다."""
    # 지시: inspect.getdoc 으로 문자열 존재 확인 — 결정을 코드에 남기게 강제.
```

`tests/test_synth_shapes.py`

```python
"""ShapeResolver 의 FLOPs/bytes 계산을 보장 (STEP 4)."""

def test_layer_list_matches_architecture_yaml():
    """layers() 가 profiler/models/llama.yaml 의 catalog.dense 와 정확히 일치한다."""
    # 지시: yaml 을 직접 파싱해 비교. 하드코딩 금지.

def test_dense_csv_layers_are_all_resolvable():
    """실측 dense.csv 에 등장하는 모든 layer 값을 ShapeResolver 가 처리한다."""
    # 지시: 실측 번들의 dense.csv 에서 layer 컬럼을 unique 로 뽑아 전부 resolver 에 넣는다.
    #       예외가 나면 Tier 0 가 그 레이어를 못 만든다는 뜻이므로 반드시 통과해야 한다.
    #       per_sequence.csv, moe.csv 에도 같은 검사를 한다.

def test_qkv_proj_flops_formula():
    """qkv_proj 의 FLOPs 가 2·T·d_model·((n_q+2·n_kv)·d_head/TP) 다."""
    # 지시: TP=1 과 TP=2 두 경우를 손계산 기대값과 비교.

def test_row_parallel_layers_shard_input_not_output():
    """o_proj / down_proj 는 입력 차원이 TP 로 나뉜다."""
    # 지시: TP=1 대비 TP=2 의 FLOPs 가 정확히 절반인지 확인.

def test_tp_stable_layers_do_not_shard():
    """tp_stable: true 인 layernorm / sampler 는 TP 에 따라 변하지 않는다."""

def test_lm_head_scales_with_sequences_not_tokens():
    """per_sequence 레이어는 sequences 에 선형이다."""

def test_moe_cost_scales_with_activated_experts():
    """expert() 의 FLOPs 가 activated_experts 에 선형이다."""

def test_unknown_layer_raises():
    """catalog 에 없는 레이어명은 예외. 조용히 0 을 반환하지 않는다."""

def test_flops_monotonic_in_tokens():
    """모든 레이어에서 tokens 증가 시 FLOPs 가 단조 증가한다."""
    # 지시: hypothesis 를 쓰지 않고 tokens in [1,2,4,...,4096] 루프로 충분.

def test_weight_reuse_variants_differ():
    """bytes_moved 의 두 변형(sum / max)이 큰 T 에서 다른 값을 준다."""
    # 지시: 두 변형이 실제로 구현되어 선택 가능한지 확인.
```

실행:

```bash
pytest -q tests/test_synth_dims.py tests/test_synth_shapes.py
pytest -q tests/test_synth_import_hygiene.py   # torch/vllm 미의존 회귀 확인
```

## 완료 조건

- [ ] 실측 `dense.csv` / `per_sequence.csv` / `moe.csv` 의 모든 `layer` 값이 예외 없이 해석된다
- [ ] TP 샤딩 방향(column vs row parallel)이 테스트로 고정되었다
- [ ] `head_dim` 유도 등 조사 항목의 결론이 docstring과 PR에 기록되었다
- [ ] `torch`/`vllm` 미의존 유지

---

# STEP 5. RooflineModel — dense / per_sequence 비용 모델

## 목표
`OpCost` + `DeviceSpec` → `time_us`. Tier 0의 심장.

## 지시

1. `profiler/synth/device.py` — `DeviceSpec`.
   ```python
   @dataclass(frozen=True)
   class DeviceSpec:
       label: str
       peak_flops: float          # FLOP/s, dtype 선택 후의 값
       mem_bandwidth_bytes: float # B/s
       flops_efficiency: float
       mem_efficiency: float
       family_efficiency: dict[str, float]
       kernel_launch_us: float
       source: str

       @classmethod
       def from_profile(cls, profile: AcceleratorProfile, dtype: str) -> "DeviceSpec":
           """datasheet 가 없거나 필수 값이 비면 예외. 기본값을 만들지 않는다 (A2)."""

       def efficiency(self, family: str) -> tuple[float, float]:
           """family_efficiency override 를 적용한 (flops_eff, mem_eff)."""
   ```
2. `profiler/synth/roofline.py` — `RooflineModel`.
   ```python
   class RooflineModel:
       """roofline 을 상한으로 두고 efficiency 로 derating 하는 cost model.

       t = max( flops / (peak_flops · eff_c),
                bytes / (bw · eff_m),
                kernel_launch_us · 1e-6 )

       설계 근거: roofline 을 '예측기'로 쓰면 안 되고 '제약'으로 쓰고 그 안에
       efficiency 항을 두어야 한다. NeuSight(ASPLOS'25), KernelSight-LM(2026),
       GenZ 가 서로 독립적으로 같은 결론에 도달했다. kernel_launch_us 는
       KernelSight-LM 의 t_0 항에 해당하며, 작은 T 에서 지배적이다.
       """
       def __init__(self, device: DeviceSpec, scaling: ScalingTable | None = None): ...
       def estimate_us(self, cost: OpCost) -> float: ...
   ```
3. `ScalingTable` 은 Tier 1(STEP 9)이 채우는 per-kernel-family 배율표다. **STEP 5에서는 인터페이스만 만들고 `None` 이면 배율 1.0** 으로 둔다.
4. **단위를 실수하기 쉬우므로 명시적으로 관리할 것.** CSV는 `time_us`(마이크로초), `memory_bandwidth_gbps` 는 GB/s, `peak_tflops` 는 TFLOP/s다. 변환 상수를 모듈 상수로 두고 테스트로 고정한다.
5. **quantization / sparsity 를 모델링하지 않는다.** 이 지시서 범위 밖. `variant` 에 알 수 없는 dtype이 오면 예외.

## 테스트

`tests/test_synth_roofline.py`

```python
"""RooflineModel 의 수치와 경계 동작을 보장 (STEP 5)."""

def test_compute_bound_branch():
    """arithmetic intensity 가 높으면 compute term 이 선택된다."""
    # 지시: 손계산한 기대값과 pytest.approx 로 비교. 근거를 주석에.

def test_memory_bound_branch():
    """arithmetic intensity 가 낮으면 memory term 이 선택된다."""

def test_launch_floor_dominates_at_tiny_size():
    """flops/bytes 가 0 에 가까우면 kernel_launch_us 가 결과가 된다."""

def test_unit_conversions():
    """GB/s, TFLOP/s, us 변환 상수가 정확하다."""
    # 지시: peak 1 TFLOP/s, 2e12 FLOPs, eff=1.0 -> 2.0 s = 2e6 us

def test_efficiency_scales_linearly():
    """eff 를 절반으로 하면 해당 term 의 시간이 두 배가 된다."""

def test_family_efficiency_override():
    """family_efficiency['attention'] 가 기본 efficiency 를 덮어쓴다."""

def test_missing_datasheet_raises():
    """datasheet 없는 프로필로 DeviceSpec 을 만들면 예외 (A2)."""

def test_missing_efficiency_raises():
    """flops_efficiency 가 None 이면 예외. 1.0 으로 가정하지 않는다 (A2)."""

def test_unknown_dtype_raises():
    """peak_tflops 에 없는 dtype 을 요청하면 예외."""

def test_scaling_table_none_is_identity():
    """ScalingTable=None 이면 배율 1.0."""

def test_monotonic_in_cost():
    """flops 또는 bytes 가 증가하면 시간이 감소하지 않는다."""

# --- 실측 대조 (핵심) ---
def test_order_of_magnitude_against_real_bundle():
    """A40 실측 dense.csv 와 Tier 0 추정이 같은 자릿수 안에 있다."""
    # 지시: A40 datasheet(STEP 3)로 DeviceSpec 을 만들되 efficiency 는 아직 없으므로
    #       eff=1.0 을 명시적으로 주입한 '이론 하한' 모드로 계산한다.
    #       eff=1.0 이므로 추정은 실측보다 항상 작아야 한다(하한 성질).
    #       assert 는 (a) 모든 점에서 est <= measured, (b) 중위 비율이 0.05~1.0.
    #       (b) 가 깨지면 shape 공식이나 단위가 틀렸다는 강한 신호다.
    #       이 테스트가 STEP 5 의 존재 이유다 — 반드시 통과시킬 것.
```

실행:

```bash
pytest -q tests/test_synth_roofline.py -v
```

## 완료 조건

- [ ] 이론 하한 성질(`eff=1.0` 일 때 추정 ≤ 실측)이 A40 실측 번들 전체에서 성립한다
- [ ] 중위 비율이 0.05~1.0 범위 (즉 20배 이내로 근접)
- [ ] 단위 변환이 테스트로 고정되었다
- [ ] 필수 datasheet 값 부재 시 예외 (A2)

> **이 완료 조건이 깨지면 STEP 6으로 넘어가지 말 것.** shape 공식이나 단위가 틀린 상태로 진행하면 이후 모든 STEP이 무의미해진다.

---

# STEP 6. AttentionCostModel — 4축 그리드

## 목표
`attention.csv` 의 `(prefill_chunk, kv_prefill, n_decode, kv_decode)` 4축 키에 대한 Tier 0 비용 모델.

## 배경 — 왜 별도 STEP인가

세 독립 출처가 **attention kernel만 세대 간 이식이 되지 않는다**고 보고한다.
- KernelSight-LM: attention efficiency factor의 cross-device 분산이 최대 3.8배
- LLMCompass: GELU 5.0% vs softmax 12.0% — attention 관련 연산 오차가 2배 이상
- Vidur: attention을 애초에 sequence-level 별도 bucket으로 분리 설계

또한 `attention.csv` 는 19,365행으로 실측 번들 전체 행 수의 대부분을 차지한다. **하드웨어당 하나만 실측한다면 그것은 attention이어야 한다** — 이 판단이 STEP 9(Tier 1)와 STEP 11(E2 실험)의 근거다.

## 지시

1. `profiler/synth/attn.py` — `AttentionCostModel`.
   ```python
   class AttentionCostModel:
       """혼합 배치 한 스텝의 attention 비용.

       한 스텝은 prefill chunk 와 decode 시퀀스를 동시에 담을 수 있다
       (chunked prefill). 커널이 융합되어 있으므로 두 위상을 더하지 않고
       총 FLOPs / 총 bytes 에 roofline 을 적용한다. 이 선택의 타당성은
       STEP 8 diff 로 검증하며, 틀리면 sum 변형으로 바꾼다.
       """
       def estimate_us(self, point: AttentionPoint) -> float: ...
   ```
2. 비용 항 (초안 — STEP 8에서 검증 후 확정):
   - **prefill causal attention FLOPs**: `4 · n_q_local · d_head · pc · (kv_prefill + pc/2)`
     - `n_q_local = n_q / TP`
     - `4 =` QK^T(2) + PV(2)
     - `(kv_prefill + pc/2)` — causal 마스크로 chunk 내부는 평균 절반만 본다
   - **decode attention bytes** (memory-bound): `n_decode · kv_decode · 2 · n_kv_local · d_head · kv_dtype_bytes`
     - `2 =` K와 V
     - Vidur가 decode attention을 "가져오는 총 KV 바이트"로 모델링한 것과 동일한 근거
   - **decode attention FLOPs**: `4 · n_q_local · d_head · n_decode · kv_decode`
   - **prefill bytes**: KV 쓰기 + 활성 읽기
   - `t = max(total_flops/(F·eff_attn), total_bytes/(BW·eff_attn_mem), launch_us)`
3. `n_decode == 0 and prefill_chunk == 0` 같은 퇴화 키를 어떻게 처리하는지 **조사 필요**: 실측 `attention.csv` 의 첫 행이 `0,0,1,16` 이므로 `prefill_chunk=0` 은 decode-only를 의미한다(CONTRACT.md 확인). 두 축이 모두 0인 행이 실제로 존재하는지 확인하고, 존재하면 그 의미를 CONTRACT.md에서 찾아 처리하고, 없으면 예외를 던진다.
4. **`skew.csv` / `skew_fit.csv` 는 Tier 0에서 생성하지 않는다.** CONTRACT.md가 "importer가 skew를 만들 수 없으면 omit하되 meta.yaml에 명시"를 허용한다. Tier 0 번들은 skew를 omit하고 `meta.yaml` 에 `skew: omitted (analytical bundle has no heterogeneous-decode measurement)` 를 기록한다. 시뮬레이터는 pooled constant alpha로 폴백한다. **이 한계를 `caveat_for(ANALYTICAL)` 문구와 별도로 meta.yaml과 PR에 기록할 것.**

## 테스트

`tests/test_synth_attn.py`

```python
"""AttentionCostModel 의 4축 스케일링과 실측 대조를 보장 (STEP 6)."""

def test_decode_only_point():
    """prefill_chunk=0 이면 decode 항만 남는다."""

def test_prefill_only_point():
    """n_decode=0 이면 prefill 항만 남는다."""

def test_prefill_quadratic_in_chunk():
    """kv_prefill=0 고정 시 prefill_chunk 를 2배로 하면 FLOPs 가 약 4배."""
    # 지시: causal 이므로 정확히 4배는 아니다. pc·(0+pc/2) = pc^2/2 이므로 정확히 4배.
    #       기대 관계를 손계산해 approx 로 고정.

def test_prefill_linear_in_kv_prefill():
    """prefill_chunk 고정 시 kv_prefill 에 선형 (kv_prefill >> pc 영역에서)."""

def test_decode_linear_in_n_decode():
    """kv_decode 고정 시 n_decode 에 선형."""

def test_decode_linear_in_kv_decode():
    """n_decode 고정 시 kv_decode 에 선형."""

def test_gqa_uses_kv_heads_for_bytes():
    """n_kv < n_q 인 GQA 모델에서 decode bytes 가 n_kv 로 계산된다."""
    # 지시: Llama-3.1-8B 는 n_q=32, n_kv=8 (확인됨). n_q 로 계산하면 정확히 4배 과대추정된다.
    #       이 테스트가 GQA 오류를 잡는다.

def test_tp_shards_heads():
    """TP=2 면 head 가 절반이므로 시간이 감소한다."""

def test_degenerate_point_handling():
    """prefill_chunk=0, n_decode=0 인 키에 대한 동작이 정의되어 있다."""
    # 지시: 실측 CSV 에 그런 행이 있으면 그 의미대로, 없으면 예외.
    #       어느 쪽인지 조사 결과를 docstring 에 적는다.

# --- 실측 대조 ---
def test_lower_bound_property_on_real_attention_csv():
    """eff=1.0 일 때 추정이 실측 attention.csv 의 하한이다."""
    # 지시: A40 tp1 번들 전체(수천 행)에 대해 est <= measured 를 검사.
    #       위반 행이 있으면 그 행을 출력해 실패시킨다 — 어떤 축에서 공식이
    #       틀렸는지 즉시 드러난다.

def test_relative_shape_correlation():
    """추정과 실측의 순위 상관(Spearman)이 0.9 이상이다."""
    # 지시: 절대값이 틀려도 '어느 키가 더 비싼가'의 순서가 맞으면
    #       efficiency 보정(STEP 9)으로 절대값을 맞출 수 있다.
    #       scipy 가 없으면 순위를 직접 계산할 것 (의존성 추가 금지).
    #       0.9 미달이면 공식이 구조적으로 틀렸다는 뜻이므로 STEP 7 로 가지 말 것.
```

실행:

```bash
pytest -q tests/test_synth_attn.py -v
```

## 완료 조건

- [ ] 4축 각각의 스케일링 관계가 테스트로 고정되었다
- [ ] GQA(`n_kv < n_q`)가 올바르게 처리된다
- [ ] A40 실측 `attention.csv` 전체에서 하한 성질이 성립한다
- [ ] 순위 상관 ≥ 0.9

---

# STEP 7. BundleEmitter + CLI — 번들 생성과 자기 검증

## 목표
키를 열거해 완전한 Tier 0 번들을 쓰고, **자기 출력을 컨트랙트로 검증**한다.

## 지시

1. `profiler/synth/emit.py` — `BundleEmitter`.
   ```python
   class BundleEmitter:
       """Tier 0/1 번들을 CONTRACT.md 레이아웃으로 쓴다.

       키 열거는 profiler.core.categories 의 그리드 생성기를 재사용한다.
       이것이 실측 번들과 키 호환성을 보장하는 유일한 방법이다 (§2.1).
       profiler.core.writer 는 torch/vllm 을 import 하므로 쓰지 않는다.
       """
       def emit(self, out_root: Path) -> BundleReport: ...
   ```
2. 키 열거 규칙:
   - `categories_for(arch, tp)` 로 카테고리를 얻고, 각 카테고리의 그리드 생성기로 키를 만든다.
   - **그리드 파라미터(`max_num_batched_tokens`, `max_num_seqs`, `attention_max_kv`, chunk/kv factor)를 CLI 인자로 노출**하고 `meta.yaml` 에 기록한다.
   - **대안 모드 `--mirror-keys <실측번들경로>`** 를 제공한다: 실측 번들의 CSV에서 키를 그대로 읽어와 같은 키 집합으로 합성한다. **STEP 8의 diff는 반드시 이 모드로 만든 번들을 쓴다** (키가 다르면 비교가 무의미하다).
3. CSV 쓰기:
   - `contract.SCHEMAS` 의 `columns` 순서를 그대로 따른다.
   - `time_us` 포맷은 실측 CSV와 동일해야 한다. **조사 필요**: `profiler/core/writer.py::_format_time_us` 의 포맷을 읽고 동일하게 재현할 것 (writer를 import하지 않고 로직만 복제하며, 복제했다는 사실과 원본 위치를 주석에 남긴다).
   - `time_us` 가 0 이하가 되면 예외 — 컨트랙트 위반이다. `kernel_launch_us` 하한이 있으므로 정상 경로에서는 발생하지 않아야 한다.
4. `meta.yaml`:
   - CONTRACT.md의 필수 필드 전부
   - `tier: analytical` (또는 `calibrated`)
   - `source: analytical` (또는 `calibrated`)
   - `cost_model: roofline-v1`
   - `datasheet_source`, `efficiency: {flops, mem, family: {...}}`
   - `generator_version`, `generated_at`
   - `skew: omitted (...)`
   - `grid`: 사용한 그리드 파라미터
   - **`vllm_version` / `cuda_version` / `gpu` 는 `null` 로 두고 `null_reason` 을 적는다.** 없는 값을 만들지 않는다(A2).
5. `profiler/synth/__main__.py` — CLI.
   ```
   python -m profiler.synth emit \
       --accelerator profiles/accelerators/ascend_target.yaml \
       --model meta-llama/Llama-3.1-8B \
       --variant bf16 \
       --tp 1,2,4 \
       --hardware-label ASCEND_TARGET-t0 \
       [--mirror-keys profiler/perf/A40/meta-llama/Llama-3.1-8B/bf16] \
       [--max-num-batched-tokens 2048] [--max-num-seqs 256] [--attention-max-kv 16384] \
       --out profiler/perf
   ```
   - `--hardware-label` 이 `-t0`/`-t1` 로 끝나지 않으면 **거부**한다 (A3).
   - 출력 디렉터리가 이미 존재하면 `--force` 없이 거부한다 (importer의 overwrite guard와 동일한 규율).
6. **emit 직후 `contract.validate_bundle()` 을 호출**하고 실패하면 쓴 파일을 지운 뒤 예외를 던진다. 반쯤 쓴 번들을 남기지 않는다.

## 테스트

`tests/test_synth_emit.py`

```python
"""BundleEmitter 가 컨트랙트를 만족하는 번들을 만드는지 보장 (STEP 7)."""

def test_emitted_bundle_passes_contract_validation(tmp_path):
    """합성 번들이 contract.validate_bundle 을 통과한다."""

def test_emitted_bundle_passes_csv_importer(tmp_path):
    """합성 번들을 CsvProfileImporter 가 검증 모드로 받아들인다."""
    # 지시: 이것이 이 STEP 의 핵심 테스트다. importer 는 헤더 byte-for-byte,
    #       키 유일성, time_us>0 을 독립적으로 검사하므로,
    #       통과하면 시뮬레이터가 읽을 수 있는 번들이라는 강한 보장이 된다.
    #       importer 가 source: analytical 을 허용하는지도 함께 확인 (STEP 1 결과).

def test_headers_match_real_bundle_byte_for_byte(tmp_path):
    """합성 CSV 의 첫 줄이 실측 CSV 의 첫 줄과 완전히 같다."""

def test_mirror_keys_reproduces_exact_key_set(tmp_path):
    """--mirror-keys 모드의 키 집합이 원본과 정확히 일치한다."""
    # 지시: A40 tp1 번들을 미러링하고, 4개 CSV 각각에서 키 튜플 집합을
    #       set 비교. 하나라도 다르면 실패. STEP 8 diff 의 전제조건.

def test_default_grid_keys_are_subset_of_categories(tmp_path):
    """기본 그리드 모드의 키가 categories.py 의 그리드와 일치한다."""

def test_moe_bundle_emits_moe_csv(tmp_path):
    """MoE 모델은 moe.csv 를 쓰고, dense 모델은 쓰지 않는다."""
    # 지시: Qwen3-30B-A3B (MoE) 와 Llama-3.1-8B (dense) 두 케이스.

def test_meta_yaml_records_tier_and_nulls(tmp_path):
    """meta.yaml 에 tier/source/cost_model/efficiency 가 있고
    vllm_version 등은 null + null_reason 으로 기록된다 (A2)."""

def test_skew_omitted_and_declared(tmp_path):
    """skew*.csv 가 없고 meta.yaml 이 그 사실을 기록한다."""

def test_label_without_suffix_rejected(tmp_path):
    """--hardware-label 이 -t0/-t1 이 아니면 거부 (A3)."""

def test_existing_output_not_overwritten(tmp_path):
    """--force 없이 기존 디렉터리에 쓰려 하면 거부."""

def test_partial_bundle_cleaned_on_validation_failure(tmp_path, monkeypatch):
    """검증 실패 시 쓴 파일이 남지 않는다."""
    # 지시: monkeypatch 로 validate_bundle 을 강제 실패시키고 디렉터리가 비었는지 확인.

def test_all_times_positive(tmp_path):
    """모든 time_us 가 > 0 이다."""

def test_resolve_bundle_tier_reads_analytical(tmp_path):
    """STEP 2 의 resolve_bundle_tier 가 합성 번들을 ANALYTICAL 로 읽는다."""
    # 지시: STEP 2 와 STEP 7 이 실제로 연결되었는지 확인하는 통합 테스트.

def test_cli_smoke(tmp_path, capsys):
    """python -m profiler.synth emit 이 종료코드 0 으로 번들을 만든다."""
    # 지시: subprocess 대신 __main__ 의 main(argv) 를 직접 호출.
```

실행:

```bash
pytest -q tests/test_synth_emit.py -v

# 수동 확인 — Ascend Tier 0 번들 실제 생성 (D4 해소의 첫 단계)
python -m profiler.synth emit \
  --accelerator profiles/accelerators/ascend_target.yaml \
  --model meta-llama/Llama-3.1-8B --variant bf16 --tp 1,2 \
  --hardware-label ASCEND_TARGET-t0 --out /tmp/perf_test
python -c "
from pathlib import Path
from profiler.core.importer import CsvProfileImporter
# 검증 모드로 통과하는지 확인
"
```

> **주의**: `ascend_target.yaml` 에는 아직 datasheet가 없다(STEP 3에서 A40만 채웠다). 이 수동 확인을 하려면 Ascend의 vendor spec을 `datasheet:` 에 채워야 하며, 그 값은 **반드시 출처와 함께** 넣고 `datasheet_source` 에 기록한다. 출처를 찾을 수 없으면 채우지 말고, 이 수동 확인을 STEP 10으로 미룬다.

## 완료 조건

- [ ] 합성 번들이 `CsvProfileImporter` 검증을 통과한다
- [ ] `--mirror-keys` 모드가 실측 키 집합을 정확히 재현한다
- [ ] 헤더가 실측 CSV와 byte-for-byte 동일하다
- [ ] `meta.yaml` 이 tier/cost_model/efficiency/null_reason 을 기록한다
- [ ] 검증 실패 시 잔여 파일이 없다

---

# STEP 8. Tier 0 정확도 검증 하네스 (R4)

## 목표
**문헌 수치가 아니라 자기 환경 수치로** Tier 0의 오차를 정량화한다. 그리고 STEP 4~6에서 미결로 남긴 설계 선택(가중치 재사용 변형, attention max vs sum)을 실측으로 결정한다.

## 지시

1. `profiler/synth/diff.py` — 두 번들 비교.
   ```python
   @dataclass(frozen=True)
   class DiffReport:
       """실측 번들과 합성 번들의 키별 비교."""
       n_keys_compared: int
       n_keys_only_measured: int
       n_keys_only_synth: int
       per_file: dict[str, FileDiff]   # dense/per_sequence/attention/moe
       per_layer: dict[str, LayerDiff] # dense.csv 의 layer 별
       per_family: dict[str, LayerDiff]# gemm/elementwise/attention/moe

   @dataclass(frozen=True)
   class LayerDiff:
       n: int
       mape: float           # mean absolute percentage error
       median_ratio: float   # synth / measured
       p95_abs_error: float
       spearman: float
       max_over_key: tuple   # 최악 오차를 낸 키

   def diff_bundles(measured: Path, synth: Path, tp: int) -> DiffReport: ...
   ```
2. CLI 서브커맨드:
   ```
   python -m profiler.synth diff \
       --measured profiler/perf/A40/meta-llama/Llama-3.1-8B/bf16 \
       --synth    profiler/perf/A40-t0/meta-llama/Llama-3.1-8B/bf16 \
       --tp 1 --out outputs/tier_validation/a40_tp1.json
   ```
3. **효율 계수 fit 서브커맨드.** 이것이 A40 datasheet의 빈 `flops_efficiency`/`mem_efficiency` 를 채우는 유일한 정당한 경로다.
   ```
   python -m profiler.synth fit-efficiency \
       --measured profiler/perf/A40/... --tp 1 \
       --out profiles/accelerators/a40.efficiency.yaml
   ```
   - family별로 `measured / theoretical_lower_bound` 의 중위값을 efficiency로 삼는다.
   - 결과는 **별도 파일**에 쓰고, `a40.yaml` 에 손으로 병합할지는 사람이 결정한다. 자동으로 accelerator YAML을 수정하지 말 것.
   - 출력 YAML에 `derived_from` (번들 경로, 커밋, 날짜)을 기록한다.
4. **미결 설계 선택을 실험으로 결정한다.** 아래 4개 변형에 대해 diff를 돌려 표를 만들고 `docs/tier0_calibration.md` 에 기록한다.
   | 변형 | 옵션 |
   |---|---|
   | V1 | bytes = sum(weight, act), attention = max |
   | V2 | bytes = max(weight, act), attention = max |
   | V3 | bytes = sum, attention = sum(prefill, decode) |
   | V4 | bytes = max, attention = sum |
   가장 낮은 전체 MAPE를 주는 변형을 기본값으로 채택하고, **채택 근거와 나머지 3개의 수치를 문서에 남긴다.**
5. 리포트를 사람이 읽을 형태로도 출력한다 (`--format table`).

## 테스트

`tests/test_synth_diff.py`

```python
"""diff 하네스의 정확성을 보장 (STEP 8)."""

def test_identical_bundles_have_zero_error(tmp_path):
    """같은 번들을 두 번 비교하면 MAPE=0, median_ratio=1, spearman=1."""
    # 지시: 실측 번들을 복사해 비교. 하네스 자체의 버그를 잡는다.

def test_scaled_bundle_reports_known_error(tmp_path):
    """모든 time_us 를 1.5배 한 번들은 MAPE=50%, median_ratio=1.5, spearman=1."""
    # 지시: 합성 오차를 주입해 지표 계산이 맞는지 확인. 이것이 하네스의 단위 테스트다.

def test_key_mismatch_is_reported_not_silently_dropped(tmp_path):
    """한쪽에만 있는 키가 n_keys_only_* 로 보고된다."""
    # 지시: 조용히 교집합만 비교하면 오차가 과소평가된다.

def test_spearman_implementation():
    """자체 구현한 순위 상관이 알려진 케이스에서 정확하다."""
    # 지시: 완전 일치=1, 완전 역순=-1, 알려진 중간 케이스 1개.

def test_per_layer_and_per_family_breakdown(tmp_path):
    """dense.csv 의 layer 별, family 별 집계가 모두 채워진다."""

def test_fit_efficiency_writes_provenance(tmp_path):
    """fit-efficiency 출력에 derived_from(경로/커밋/날짜)이 기록된다."""

def test_fit_efficiency_does_not_touch_accelerator_yaml(tmp_path):
    """fit-efficiency 가 profiles/accelerators/*.yaml 을 수정하지 않는다."""

def test_fit_efficiency_bounds():
    """fit 된 efficiency 가 (0, 1] 범위 밖이면 경고하고 클램프하지 않는다."""
    # 지시: >1 이 나오면 이론 하한이 깨졌다는 뜻이므로 실패해야 한다.
```

**실제 검증 실행 (이 STEP의 산출물)**

```bash
# 1) 미러 키로 A40 Tier 0 번들 생성
python -m profiler.synth emit \
  --accelerator profiles/accelerators/a40.yaml --model meta-llama/Llama-3.1-8B \
  --variant bf16 --tp 1 --hardware-label A40-t0 \
  --mirror-keys profiler/perf/A40/meta-llama/Llama-3.1-8B/bf16 \
  --out outputs/tier_validation/perf

# 2) diff
python -m profiler.synth diff \
  --measured profiler/perf/A40/meta-llama/Llama-3.1-8B/bf16 \
  --synth outputs/tier_validation/perf/A40-t0/meta-llama/Llama-3.1-8B/bf16 \
  --tp 1 --format table

# 3) RTXPRO6000 에도 반복 (두 번째 하드웨어로 일반성 확인)
# 4) 4개 변형 × 2개 하드웨어 표를 docs/tier0_calibration.md 에 기록
```

## 완료 조건

- [ ] `docs/tier0_calibration.md` 에 2개 하드웨어 × 4개 변형의 MAPE / median_ratio / spearman 표가 기록되었다
- [ ] 기본 변형이 근거와 함께 채택되었다
- [ ] `a40.efficiency.yaml` / `rtxpro6000.efficiency.yaml` 이 provenance와 함께 생성되었다
- [ ] efficiency 병합 후 A40 Tier 0의 dense MAPE와 attention MAPE가 문서에 기록되었다

> **이 STEP의 결과가 나쁘더라도(예: MAPE 50%) 실패가 아니다.** 자기 환경의 실제 수치를 아는 것이 목표다. 다만 순위 상관이 0.9 미만이면 STEP 4~6의 공식에 구조적 오류가 있으므로 되돌아가 고칠 것.

---

# STEP 9. Tier 1 — per-kernel-family 보정

## 목표
소량의 앵커 측정으로 Tier 0를 보정한다. **attention에 예산을 몰아넣는 전략을 코드로 표현**한다.

## 지시

1. `profiler/synth/calibrate.py`.
   ```python
   @dataclass(frozen=True)
   class ScalingTable:
       """per-kernel-family 배율. Tier 1 의 산출물.

       t_tier1 = t_tier0 · scale(family, feature)

       가장 단순한 형태(family 별 스칼라)부터 시작하고, STEP 8 diff 가
       feature 의존성(예: 작은 T 에서 배율이 다름)을 보이면 구간별 배율로 확장한다.
       KernelSight-LM 은 이 구조를 'measured efficiency grid + interpolation,
       학습 모델은 외삽 경계의 fallback' 으로 기술한다.
       """
       scalars: dict[str, float]
       #: family -> [(feature_lo, feature_hi, scale)] 구간표 (optional)
       piecewise: dict[str, list[tuple[float, float, float]]]
       anchors: list[AnchorRecord]
       derived_from: dict[str, str]

   def fit_from_anchors(
       anchors: Path,            # 앵커 측정 CSV (실측 번들의 부분집합 형식)
       tier0: RooflineModel,
       resolver: ShapeResolver,
   ) -> ScalingTable: ...
   ```
2. **앵커 입력 형식**은 실측 번들 CSV와 동일한 스키마의 **부분집합**이다. 즉 `dense.csv` 에 100행만 있어도 유효한 앵커다. `contract.validate_csv` 로 검증한다. 새 형식을 만들지 말 것.
3. **앵커 선택 도구**를 함께 제공한다.
   ```
   python -m profiler.synth pick-anchors \
       --model meta-llama/Llama-3.1-8B --variant bf16 --tp 1 \
       --budget 200 --out anchors_plan.csv
   ```
   - family별로 feature 공간(로그 스케일 T, attention 4축)에서 고르게 퍼진 키를 고른다.
   - `--attention-share 0.7` 로 attention에 예산 비중을 줄 수 있게 한다 (기본값 0.7 — §STEP 6 배경의 근거).
   - 출력은 "이 키들을 측정하라"는 계획표이며, 실제 측정은 사람이 `python -m profiler profile` 로 수행한다.
4. `RooflineModel` 이 `ScalingTable` 을 받아 적용한다 (STEP 5에서 만든 인터페이스).
5. `BundleEmitter` 가 `--scaling <table.yaml>` 을 받으면 `tier: calibrated`, `source: calibrated` 로 쓰고, `meta.yaml` 에 `calibration_anchors`(앵커 수, family별 분포, 출처)를 기록한다.

## 테스트

`tests/test_synth_calibrate.py`

```python
"""Tier 1 보정의 정확성과 규율을 보장 (STEP 9)."""

def test_fit_recovers_known_scale(tmp_path):
    """Tier 0 출력을 정확히 2배 한 인공 앵커로 fit 하면 scale≈2.0."""
    # 지시: 하네스의 단위 테스트. family 별로 서로 다른 배율(gemm=2, elementwise=3)을
    #       주입해 독립적으로 복원되는지 확인.

def test_fit_is_per_family_not_global(tmp_path):
    """family 별 배율이 섞이지 않는다."""

def test_empty_anchors_yields_identity(tmp_path):
    """앵커가 없으면 배율 1.0 (A2 — identity)."""

def test_anchor_csv_validated_by_contract(tmp_path):
    """잘못된 헤더의 앵커 CSV 는 ProfileContractError."""

def test_partial_anchor_csv_accepted(tmp_path):
    """전체 그리드의 부분집합만 있는 앵커 CSV 가 받아들여진다."""

def test_tier1_improves_on_tier0_with_real_anchors():
    """실측 A40 번들의 부분집합(예: 5%)을 앵커로 쓰면 나머지 95% 에 대한
    MAPE 가 Tier 0 보다 낮아진다."""
    # 지시: 이것이 이 STEP 의 핵심 테스트다. hold-out 검증이다.
    #       앵커/검증 분할은 random_state 고정. 개선폭을 테스트 이름이 아니라
    #       docs/tier0_calibration.md 에 기록한다.
    #       개선되지 않으면 Tier 1 의 설계가 틀렸다는 뜻이므로 실패해야 한다.

def test_attention_share_respected():
    """pick-anchors --attention-share 0.7 이면 앵커의 약 70% 가 attention 키다."""

def test_pick_anchors_covers_feature_range():
    """선택된 앵커가 T 축의 로그 스케일 전 구간에 퍼져 있다."""

def test_calibrated_bundle_tier_is_calibrated(tmp_path):
    """--scaling 을 준 번들의 resolve_bundle_tier 가 CALIBRATED 다."""

def test_calibrated_meta_records_anchors(tmp_path):
    """meta.yaml 에 calibration_anchors 가 기록된다 (A1)."""
```

실행:

```bash
pytest -q tests/test_synth_calibrate.py -v

# hold-out 검증 실측 실행 — 앵커 비율을 바꿔가며 곡선을 만든다
for share in 0.01 0.02 0.05 0.10 0.20; do
  python -m profiler.synth diff --holdout-fit $share \
    --measured profiler/perf/A40/meta-llama/Llama-3.1-8B/bf16 --tp 1
done
```

## 완료 조건

- [ ] 인공 배율 복원 테스트 통과
- [ ] 실측 hold-out에서 Tier 1이 Tier 0보다 MAPE가 낮다
- [ ] 앵커 비율(1%/2%/5%/10%/20%) vs MAPE 곡선이 `docs/tier0_calibration.md` 에 기록되었다
- [ ] `pick-anchors` 가 attention 비중을 지킨다

---

# STEP 10. planner 통합 — D4 해소

## 목표
Ascend / ATOM island을 포함한 cluster spec으로 `plan` 이 **실패하지 않고** Tier 라벨과 함께 동작한다.

## 지시

1. `profiles/accelerators/ascend_target.yaml` 을 갱신한다.
   - `datasheet:` 를 **출처와 함께** 채운다. 출처를 찾을 수 없는 값은 비워 둔다.
   - `sim_hardware: ASCEND_TARGET-t0`
   - `source: vendor_spec` (accelerator YAML 숫자의 출처. `placeholder` 에서 승격)
   - `notes` 를 갱신: placeholder가 아니라 vendor-spec 기반 Tier 0임을 명시
   - **`supported_models` 는 확장하지 않는다.** 지원 모델 주장은 실측/벤더 문서 근거가 필요하다.
2. `rbln_atom.yaml`, `furiosa_rngd.yaml` 도 같은 방식으로 처리한다. 단 **출처 없는 숫자는 채우지 않는다** — 채울 수 없으면 그 파일은 이 STEP에서 건드리지 않고 그 사실을 PR에 적는다.
3. Tier 0 번들을 실제로 생성해 커밋한다.
   - **조사 필요 / 사용자 확인 필요**: 합성 CSV(수만 행)를 저장소에 커밋할 것인가? 대안은 (a) 커밋한다 (b) `.gitignore` 하고 `make tier0-bundles` 같은 재생성 타깃을 제공한다. **(b)를 권장**한다 — 합성 데이터는 결정론적으로 재생성 가능하고, 실측 데이터와 저장소에서 섞이지 않는 편이 A1/A3에 부합한다. 최종 결정은 사용자에게 확인할 것.
   - 재생성 스크립트를 `scripts/gen-tier0-bundles.sh` 로 제공한다.
4. `examples/clusters/` 에 이기종 예제를 추가한다: GPU island(measured) + Ascend island(Tier 0).
5. **candidate generator의 stage-5 roofline을 Tier 0 모델과 통일할지는 이 STEP에서 결정하지 않는다.** 지금은 두 경로가 공존한다 (stage-5는 pruning용 하한, Tier 0는 번들 생성용 추정). 통일은 STEP 12(선택)로 미룬다. **이 공존 사실과 이유를 `docs/deviations.md` 에 기록할 것.**

## 테스트

`tests/test_tier_e2e.py`

```python
"""Tier 0 번들로 plan 이 동작하고 라벨이 전파됨을 보장 (STEP 10)."""

def test_ascend_profile_loads_with_datasheet():
    """ascend_target.yaml 이 datasheet 와 -t0 sim_hardware 로 로드된다."""

def test_ascend_island_appears_in_candidates(tmp_path, monkeypatch):
    """Ascend island 가 candidate 생성 단계에서 배제되지 않는다."""
    # 지시: MockPredictor 사용. 기존 tests/conftest.py 패턴 재사용.
    #       "sim_hardware 디렉터리 없음" 으로 걸러지지 않는 것이 핵심.

def test_heterogeneous_plan_tier_is_analytical(tmp_path, monkeypatch):
    """GPU(measured) + Ascend(analytical) plan 의 profile_tier 가 analytical."""

def test_heterogeneous_plan_carries_caveat(tmp_path, monkeypatch):
    """caveats 에 'simulator-only (analytical inputs)' 가 포함된다."""

def test_all_gpu_plan_has_no_tier_caveat(tmp_path, monkeypatch):
    """GPU 전용 plan 에는 tier caveat 이 없다 (A4 무회귀)."""

def test_plan_yaml_roundtrip_includes_tier(tmp_path, monkeypatch):
    """plan 결과 YAML 을 다시 읽어도 profile_tier 가 보존된다."""

def test_golden_plans_unchanged():
    """기존 golden plan 출력이 tier 필드 추가 외에는 동일하다."""
    # 지시: golden 파일 diff 를 확인하고 tier 관련 키만 추가되었음을 assert.

def test_regenerate_script_is_deterministic(tmp_path):
    """scripts/gen-tier0-bundles.sh 를 두 번 돌리면 같은 CSV 가 나온다."""
    # 지시: 파일 해시 비교. 비결정론(딕셔너리 순서, 부동소수 포맷)을 잡는다.
```

**시뮬레이터 실동작 확인 (수동, GPU 불필요)**

```bash
# Tier 0 번들로 20-request 시뮬레이션이 끝까지 도는지 확인
# (CONTRACT.md 임포터 체크리스트 4번의 규율)
python -m planner plan --cluster examples/clusters/hetero_gpu_ascend.yaml \
  --service examples/service_specs/<적당한 spec> --out /tmp/plan_t0.yaml
grep -n "profile_tier\|simulator-only" /tmp/plan_t0.yaml
```

## 완료 조건

- [ ] Ascend island를 포함한 `plan` 이 성공하고 결과에 `profile_tier: analytical` 이 있다
- [ ] 리포트에 `simulator-only (analytical inputs)` 배너가 나타난다
- [ ] GPU 전용 plan의 golden 출력이 tier 필드 추가 외에 변하지 않았다
- [ ] Tier 0 번들 재생성이 결정론적이다
- [ ] `docs/deviations.md` 의 D4 항목이 "해소(Tier 0 경로로)" 로 갱신되고, stage-5와의 공존이 기록되었다

---

# STEP 11. 검증 실험 하네스 E1~E4

## 목표
보고서 §7.6의 실험을 재현 가능한 스크립트로 만든다. **E1이 가장 중요하다** — planner의 목적은 절대 latency 예측이 아니라 configuration 순위 결정이므로, "Tier 0가 plan 결정을 바꾸는가"가 핵심 질문이다.

## 지시

각 실험은 `experiments/tier_validation/e<N>_*.py` 로 만들고, 다음을 공통으로 지킨다.
- 결과는 `outputs/tier_validation/e<N>/` 에 JSON + 사람이 읽는 표로 쓴다.
- `planner/util/provenance.py` 의 §3.8 metadata 블록을 결과 파일에 포함한다.
- 시뮬레이터 실행이 필요한 실험은 `--dry-run` 으로 조합 수만 세는 모드를 제공한다 (CI에서 이 모드만 돈다).

### E1 — Tier 0는 planner 결정을 바꾸는가 (`e1_plan_agreement.py`)

4점 비교. work order §12가 이미 greedy vs 제안 vs oracle 비교를 요구하므로, Tier 0 한 점을 끼우면 논문 그림 하나가 완성된다.

```
① greedy (roofline proxy, 시뮬레이션 없음)
② Tier 0 번들 + 시뮬레이터
③ Tier 2 번들 + 시뮬레이터        ← 기준(ground truth)
④ exhaustive oracle
```

- 대상: 보유 GPU 2종(A40, RTXPRO6000) × 모델 1~2종 × service spec 2~3종
- 지표: **top-1 일치율**, **top-3 포함율**, 선택된 plan의 실제(③ 기준) SLO-goodput/J **상대 오차**, Kendall tau (후보 순위 전체)
- 출력 표: 행=조건, 열=①②③④

### E2 — 프로파일링 예산을 어디에 써야 하는가 (`e2_budget_pareto.py`)

```
조건 A: Tier 0 전체 (측정 0)
조건 B: Tier 0 + attention 만 앵커 측정 (STEP 9 pick-anchors --attention-share 1.0)
조건 C: Tier 0 + 균등 앵커
조건 D: Tier 2 전체
```

- 지표: 측정 점수(≈GPU-hour 대용) vs E2E 오차 Pareto
- KernelSight-LM의 "캘리브레이션 1회 = per-kernel 3.2배 개선, E2E는 1.1~2배" 관찰을 자기 환경에서 재현하는지 확인

### E3 — shape 중복은 실제로 얼마인가 (`e3_shape_overlap.py`)

- 레포에 있는 모든 실측 번들의 `(layer, tokens)` / attention 4축 키를 모아 **모델 간 중복률**을 센다.
- 나아가 `ShapeResolver` 로 각 키를 `(op_family, M, N, K)` 로 정규화해 **shape 레벨 중복률**을 센다. 이것이 shape 캐시 도입 시 절감 상한이다 (Dooly의 56.4% 대비).
- 시뮬레이터 실행 불필요 → CI에서 항상 돌릴 수 있다.

### E4 — 미보유 HW 결론의 견고성 (`e4_sensitivity.py`)

- Ascend를 Tier 0으로 두고 `peak_tflops`, `memory_bandwidth_gbps`, `flops_efficiency`, `mem_efficiency` 를 각각 ±30% 스윕한다.
- 지표: **plan이 뒤집히는 파라미터 임계값**. 미보유 하드웨어를 정직하게 다루는 표준 방식이다.
- 출력: 파라미터별 tornado 표 + 뒤집힘 임계값

## 테스트

`tests/test_experiments_tier.py`

```python
"""실험 하네스의 정확성을 보장 (STEP 11). 시뮬레이터는 mock 한다."""

def test_e1_agreement_metrics_on_synthetic_rankings():
    """top-1 / top-3 / Kendall tau 계산이 알려진 순위 쌍에서 정확하다."""
    # 지시: 완전 일치, 완전 역순, 1개만 다른 경우 3케이스.

def test_e1_dry_run_counts_combinations(tmp_path):
    """--dry-run 이 시뮬레이터를 호출하지 않고 조합 수만 보고한다."""

def test_e1_uses_tier2_as_ground_truth(tmp_path, monkeypatch):
    """③ Tier 2 가 기준이고 ①②④ 가 그것과 비교된다."""

def test_e2_budget_accounting():
    """조건별 '측정 점수' 가 앵커 수와 일치한다."""

def test_e2_attention_only_condition_has_only_attention_anchors():
    """조건 B 의 앵커가 전부 attention 키다."""

def test_e3_overlap_on_real_bundles():
    """실측 번들에서 계산한 중복률이 [0,1] 이고 결정론적이다."""
    # 지시: 실제 값을 하드코딩하지 말고 성질만 검사.
    #       실제 값은 outputs/ 리포트에 남긴다.

def test_e3_normalized_overlap_ge_raw_overlap():
    """shape 정규화 후 중복률이 원시 키 중복률보다 크거나 같다."""
    # 지시: 정규화는 서로 다른 모델의 같은 shape 을 합치므로 중복이 늘어야 한다.
    #       줄어들면 정규화 로직이 틀렸다.

def test_e4_sweep_grid_is_complete():
    """±30% 스윕이 지정한 파라미터 전부 × 지정한 스텝 수를 만든다."""

def test_e4_reports_flip_threshold(monkeypatch):
    """plan 이 뒤집히는 지점이 임계값으로 보고된다."""
    # 지시: MockPredictor 가 파라미터에 따라 순위를 바꾸도록 구성.

def test_all_reports_include_provenance(tmp_path, monkeypatch):
    """E1~E4 결과 JSON 에 §3.8 provenance 블록이 있다."""
```

실행:

```bash
pytest -q tests/test_experiments_tier.py -v

# E3 는 시뮬레이터 불필요 — 즉시 실제 실행
python -m experiments.tier_validation.e3_shape_overlap --out outputs/tier_validation/e3

# E1 은 시뮬레이터 필요 — dry-run 으로 규모 확인 후 실행
python -m experiments.tier_validation.e1_plan_agreement --dry-run
python -m experiments.tier_validation.e1_plan_agreement --out outputs/tier_validation/e1
```

## 완료 조건

- [ ] E1~E4 하네스가 mock 시뮬레이터로 전부 테스트된다
- [ ] E3의 실제 결과(모델 간 shape 중복률)가 리포트로 존재한다
- [ ] E1의 실제 결과(top-1 일치율)가 리포트로 존재한다
- [ ] 모든 결과 파일에 §3.8 provenance가 있다

---

# STEP 12. (선택) 후속 항목

사용자 승인 없이 착수하지 말 것. 각 항목은 별도 작업지시서가 필요하다.

| 항목 | 내용 | 선행 조건 |
|---|---|---|
| **S1. stage-5 통일** | `candidate_generator._stage5_analytical_ok` 의 memory-only roofline을 `RooflineModel` 로 대체하고 compute term을 추가 | STEP 10 완료 + golden 갱신 계획 |
| **S2. shape 캐시** | `profiler/perf/<hw>/_shapecache/` 도입. 캐시 키에 `(op, M, N, K, dtype, tp, vllm_version, cuda_version)` 포함 | E3 결과가 유의미한 중복률을 보일 때 |
| **S3. NPU bucket-E2E predictor** | LENS 방식. `planner/predictor/bucket_e2e.py` 를 `llmservingsim.py` 와 나란히 두고 backend별 dispatch. **operator 가시성이 없어 Phase 5의 P/D·KV transfer 연구에는 못 쓴다** — NPU island의 feasibility/throughput 검증 전용 | 벤더 SDK 접근 |
| **S4. 학습 기반 efficiency** | NeuSight 정식화(per-tile utilization을 roofline으로 bound한 상태에서 학습)를 이식. Tier 0의 정확도가 부족할 때만 | STEP 8 결과가 MAPE 30% 초과일 때 |
| **S5. fp8 / TP≥4 축** | Tier 0로 미측정 config 축을 메운다 | STEP 8에서 dtype 처리가 검증된 뒤 |

---

# 13. 전체 완료 조건

- [ ] STEP 0~11의 개별 완료 조건 전부
- [ ] `pytest -q` 전체 통과, `ruff check .` 통과, `mypy` 통과
- [ ] `profiler/synth/**` 가 `torch` / `vllm` / `profiler.core.writer` 를 import하지 않는다 (STEP 0의 회귀 테스트가 이를 지킨다)
- [ ] Ascend island를 포함한 `plan` 이 동작하고 `profile_tier: analytical` + caveat이 붙는다 → **D4 해소**
- [ ] `docs/tier0_calibration.md` 에 자기 환경의 Tier 0 / Tier 1 오차가 기록되었다 (문헌 수치 인용이 아님)
- [ ] E1의 top-1 plan 일치율이 리포트로 존재한다
- [ ] 기존 golden 회귀 출력이 tier 필드 추가 외에 변하지 않았다 (A4)

---

# 14. 리스크 대응 규칙

| 상황 | 판단 기준 |
|---|---|
| STEP 5/6의 실측 대조에서 **하한 성질이 깨진다** (추정 > 실측, eff=1.0) | shape 공식 또는 단위가 틀렸다. **다음 STEP으로 가지 말고** STEP 4로 돌아간다 |
| STEP 6의 **순위 상관이 0.9 미만** | attention 모델이 구조적으로 틀렸다. `max` vs `sum`, GQA head 수, causal 계수를 재검토 |
| STEP 8의 **MAPE가 50% 초과** | 실패는 아니다. 자기 환경 수치로 기록하고, Tier 0를 "candidate 넓게 훑기" 용도로만 쓴다. E1에서 plan 일치율이 높으면 여전히 유용하다 |
| STEP 9의 **Tier 1이 Tier 0를 개선하지 못한다** | 보정 구조가 틀렸다. family 정의를 세분화하거나 piecewise로 전환. 개선 없이 STEP 10으로 가지 말 것 |
| E1의 **top-1 일치율이 낮다** (예: 40%) | 이것도 결과다. "Tier 0는 plan 결정에 쓸 수 없다"는 결론을 기록하고, Tier 0의 용도를 sensitivity study로 한정한다. **결과를 좋게 보이려고 조건을 바꾸지 말 것** |
| 데이터시트 값의 **출처를 찾을 수 없다** | 채우지 않는다(A2). 그 하드웨어는 Tier 0 대상에서 제외하고 PR에 기록 |
| 합성 번들 CSV의 **저장소 커밋 여부** 판단이 필요 | 사용자에게 확인. 기본 권고는 `.gitignore` + 재생성 스크립트 |
| upstream `serving/` 수정이 필요해 보인다 | 중단하고 사용자에게 보고. 이 지시서 범위 밖 |

---

# 부록 A. Tier 0 비용 모델 수식 (초안 — STEP 8에서 확정)

```
# 공통
t_us = max( flops / (peak_flops · eff_c) ,
            bytes / (bw     · eff_m) ,
            kernel_launch_us · 1e-6 ) · 1e6

# dense GEMM (M=T)
qkv_proj      : M=T, K=d_model,        N=(n_q + 2·n_kv)·d_head / TP
o_proj        : M=T, K=n_q·d_head/TP,  N=d_model
gate_up_proj  : M=T, K=d_model,        N=2·d_ff / TP
down_proj     : M=T, K=d_ff/TP,        N=d_model
  flops = 2·M·N·K
  bytes = variant V1: (M·K + K·N + M·N)·dtype_bytes
          variant V2: max(K·N·dtype_bytes, (M·K + M·N)·dtype_bytes)

# per_sequence (M=S)
lm_head       : M=S, K=d_model,        N=vocab / TP

# elementwise (tp_stable 이면 /TP 없음)
layernorm     : elems = T · d_model            ; flops = 5·elems (근거: mean/var/norm/scale)
act_fn        : elems_in = T · 2·d_ff/TP       ; elems_out = elems_in/2 ; flops = 3·elems_out
rotary_emb    : elems = T · (n_q+n_kv)·d_head/TP ; flops = 6·elems
embedding     : gather, bytes = T·d_model·dtype_bytes ; flops ≈ 0
sampler       : elems = S · vocab/TP           ; flops = 3·elems
  # 계수(5, 3, 6, 3)는 근거를 코드 주석에 남기고 STEP 8 diff 로 조정한다.

# attention (한 스텝, 융합 커널 가정)
n_q_l  = n_q  / TP
n_kv_l = n_kv / TP
flops_prefill = 4 · n_q_l · d_head · pc · (kv_prefill + pc/2)
flops_decode  = 4 · n_q_l · d_head · n_decode · kv_decode
bytes_decode  = n_decode · kv_decode · 2 · n_kv_l · d_head · kv_dtype_bytes   # ← GQA: n_kv
bytes_prefill = (pc · (n_q_l + 2·n_kv_l) · d_head · dtype_bytes)              # QKV 읽기 + KV 쓰기
  variant "max" : t = max( (flops_p+flops_d)/(F·eff), (bytes_p+bytes_d)/(BW·eff), launch )
  variant "sum" : t = t(prefill) + t(decode)

# MoE
moe : M = T · experts_per_token, K = d_model, N = 2·moe_intermediate_size / TP  (+ down)
```

**단위**: `peak_tflops` [TFLOP/s] → `peak_flops = peak_tflops · 1e12`; `memory_bandwidth_gbps` [GB/s] → `bw = gbps · 1e9`; CSV는 `time_us`.

---

# 부록 B. 설계 근거 문헌

| 설계 요소 | 출처 |
|---|---|
| roofline을 **예측기가 아닌 제약**으로 쓰고 그 안에 efficiency를 둔다 | NeuSight, "Forecasting GPU Performance for Deep Learning Training and Inference", ASPLOS 2025, arXiv:2407.13853 · github.com/sitar-lab/NeuSight (MIT) |
| kernel launch 하한 항 `t₀`, Tier A/B 분리, 캘리브레이션 1회의 가치 정량화 | KernelSight-LM, arXiv:2606.28565 (AWS + Boston Univ., **preprint**) |
| operator를 token-level / sequence-level / communication으로 3분류; RF 선택 이유(tile·wave quantization) | Vidur, MLSys 2024, arXiv:2405.05465 · github.com/microsoft/vidur (MIT) |
| `T_op = max(C/(F·Eff_C), M/(BW·Eff_mem))` 형태와 하드웨어별 efficiency factor 실측값 | GenZ-LLM Analyzer, arXiv:2406.01698 (MIT, `pip install genz-llm`) |
| HW description만으로 operator latency; softmax(12.0%) vs GELU(5.0%) 오차 격차 | LLMCompass, ISCA 2024, arXiv:2312.03134 (BSD-3) |
| 프로파일링 중복 제거로 GPU-hour 56.4% 절감; "drop-in backend for existing simulators" | Dooly, arXiv:2605.07985 (UT Austin, **preprint**) |
| single-GPU sharded 프로파일링, collective stub out, operator class별 회귀 전략 | Frontier, arXiv:2605.21312 · github.com/NetX-lab/Frontier |
| NPU black-box + bucket당 E2E 2회, 평균 2.15%; GPU 방법론 직접 적용 시 최대 493% 오차 | LENS, arXiv:2606.18042 (한양대, **preprint**) |
| 외부 하드웨어 시뮬레이터 profile ingest 경로; single decode block, ~2.1h | LLMServingSim 2.0, ISPASS 2026, DOI 10.1109/ISPASS69572.2026.00012 |
| 소수 계수의 affine 모델로 prefill <4% / decode <5% | Aladdin, arXiv:2405.06856 |

**주의**: `preprint` 표시된 항목은 peer review를 통과하지 않았고 대부분 코드가 공개되지 않았다. 아이디어 차용은 가능하나, 논문/보고서에서 수치를 인용할 때는 preprint임을 반드시 명시할 것.
