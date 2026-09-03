# HeteroPilot 작업지시서 — 통합 스프린트 (Consolidation Sprint)

> 목적: 2026-08-31 이후 세 갈래로 벌어진 작업(Tiered Profiles / D21 envelope 재측정 / ScenarioLab workspace)을
> `main` 하나로 수렴시키고, 논문 작업이 출발할 수 있는 "현재 주장 가능한 것" 한 페이지를 만든다.
> 대상 저장소: `github.com/swsok/heteropilot` (upstream: `casys-kaist/LLMServingSim`, pin `2c2042ce`)
> 작성일: 2026-09-03 · 기준 `main` = `f6d5f20` (PR #43 머지 직후) · 작업 도구: Claude Code CLI, NPU 노드
> 후속 작업지시서: `WORK_ORDER_rps_aware.md` (미작성 — 이 스프린트 완료 후 착수)

---

## 0. 이 문서의 사용법

1. **STEP 순서를 지킬 것.** 단 STEP 2(시뮬레이션 재실행)는 수 시간이 걸리므로 STEP 1 머지 직후 백그라운드로 띄우고 STEP 3~4를 병행한다. 그 외의 병행은 금지.
2. **한 STEP = 한 브랜치 = 한 PR.** 브랜치명 `chore/consol-step<N>-<slug>`, 커밋 `consol-step<N>: <무엇을 했는가>`.
3. **이 스프린트는 새 숫자를 만들지 않는다.** 유일한 예외는 STEP 2 — 이미 실행된 sweep의 미완 구간을 같은 파라미터로 마저 돌리는 것이며, 새 실험이 아니다. 새 측정(저부하 envelope, 전력 샘플링 등)은 `WORK_ORDER_rps_aware.md`의 일이다. 하고 싶어져도 하지 말 것.
4. **"조사 필요" 항목은 추측하지 말고 실제 코드/산출물을 읽어 확인**하고 결론을 PR 본문에 적는다.
5. 이 문서는 `WORK_ORDER_heteropilot.md` → `docs/deviations.md` → `CLAUDE.md`의 **하위 문서**다. 충돌 시 상위 문서 우선, 충돌 발견 시 중단·보고.

### 절대 규칙 (재확인 — 위반 시 작업 중단)

- **A1. 측정하지 않은 숫자를 측정값으로 표기하지 않는다.** `scripts/whichnode.sh`가 나열하지 않는 하드웨어의 결과를 주장하지 않는다.
- **A2. 철회는 공개적으로.** 틀린 숫자를 발견하면 덮어쓰지 않고 위에 "superseded" 표기를 얹고 포인터를 남긴다 (D18/D19/D21 방식). 이 스프린트의 문서 작업 전부가 이 규칙 아래 있다.
- **A3. upstream 코드(`serving/`, `profiler/core`, `bench/`, `configs/`, `astra-sim/`) 수정 금지.**
- **A4. `planner/optimizer/exhaustive.py` 삭제 금지. golden 회귀 출력 변경 금지.** 이 스프린트는 planner 동작을 바꾸지 않는다 — STEP 3의 5줄 enum 추가가 유일한 `planner/` 변경이며, 동작 변경이 없음을 golden 테스트로 증명한다.

---

## 1. 배경 — 왜 이 스프린트가 필요한가

2026-09-03 시점의 리포는 다음 상태다 (이 문서 작성 시 클론을 직접 읽어 확인).

| 트랙 | 위치 | 상태 |
| --- | --- | --- |
| Tiered Profiles STEP 0~11 | `main` (PR #32~#46, 9/2~9/3) | 완료·머지. D4 해소. `docs/tier0_calibration.md`에 E1~E4 결과 기록 |
| RNGD envelope 재측정 + 18% margin 재실행 (구 "D21") | `origin/feat/pd-slo-margin-rerun` 외 3개 브랜치 | **미머지.** `pd_slo_sweep.md`의 3-regime 헤드라인 중 loose-TTFT 절반이 infeasible로 판정됨. tight-TTFT 구간은 타임아웃 아티팩트로 미결 |
| RPS-aware planning 설계서 | `origin/docs/rps-aware-planning` | 미머지. 문서만, 코드 없음 |
| ScenarioLab workspace P1~P3 | `origin/feat/scenariolab-workspace-{core,ui,extras}` | 미머지. ~3,800줄. 과제 시연용으로 **별도 리포로 분리 결정** (사용자 결정 2026-09-03) |
| 문서 기준 시점 | `PROJECT_REPORT.md` = 8/27, `HANDOVER.md` = 8/31 (`3d59035`), `main` = 9/3 | 세 문서가 서로 다른 시점을 가리킴. HANDOVER는 "pytest 284 passed"라고 쓰여 있으나 STEP 0 baseline은 366, 현재 테스트 함수는 484개 |

문제의 핵심은 기술이 아니라 서사다. 프로젝트의 헤드라인 결과("RNGD가 loose TTFT에서 에너지 1.67× 승리")가 프로젝트 자신의 검증 규율에 의해 무너졌는데, 그 사실이 `main`에도 보고서에도 없고, 그 사이 표면적만 넓어졌다. 이 스프린트의 산출물은 **"지금 확실히 주장할 수 있는 것" 한 페이지(STEP 5)**이며, 나머지 STEP은 그 페이지를 쓸 수 있는 상태를 만드는 일이다.

---

## 2. 사전 조사 결과 (2026-09-03, `f6d5f20` 클론에서 직접 확인)

작업 전 각 항목을 노드에서 재확인할 것. 틀린 항목이 있으면 PR 본문에 기록하고 이 문서를 고친다.

### 2.1 미머지 브랜치 8개의 조상 관계

D21 계열 4개는 **일직선 체인**이다:

```
docs/d18-close (1b9a291)
  ⊂ feat/rngd-concurrency-envelope (0ae4d3f)
      ⊂ docs/rps-aware-planning (c2fbb50)
          ⊂ feat/pd-slo-margin-rerun (fab7491)      ← 이것 하나만 머지하면 넷이 모두 들어온다
```

ScenarioLab workspace 3개도 일직선이다:

```
feat/scenariolab-workspace-core (fb2d85d) ⊂ -ui (dc50b4d) ⊂ -extras (fc3609d)   ← extras 하나로 충분
```

`docs/slide-deck-ko` (0e32750, PR #13)는 `docs/slide_deck_ko.html` 1파일 추가만 있는 독립 브랜치. 머지 베이스는 `0316c29`.

### 2.2 `feat/pd-slo-margin-rerun` → `main` 머지 시 충돌 (dry-run 결과)

정확히 두 파일, 둘 다 **양쪽 유지(additive)**로 해결된다:

| 파일 | HEAD(main) 쪽 | 브랜치 쪽 | 해결 |
| --- | --- | --- | --- |
| `.gitignore` | Tier 0/1 합성 번들 무시 규칙 (`profiler/perf/*-t0/` 등) | `outputs/rngd_envelope/edf/*.csv` 무시 (166 MB 트레이스) | 두 블록 모두 유지 |
| `docs/deviations.md` | `## D21 — Tier 0 introduction: datasheet fields are vendor spec` (1023~1063행) | `## D21 — The RNGD scaling curve's top point was request-pool-limited…` | **번호 충돌.** §2.3 참조 |

ScenarioLab-extras는 `main`에 충돌 없이 머지된다(`planner/inventory.py` auto-merge). 그러나 이 스프린트에서는 머지하지 않는다 — STEP 3에서 새 리포로 간다.

### 2.3 D21 번호 충돌 — 브랜치 쪽을 D22로 재번호

`main`의 D21(Tier 0 datasheet, PR #45로 머지됨)이 canonical이다. 브랜치의 "D21"(RNGD envelope 철회+재측정)은 **D22**가 된다. 브랜치에서 `D21`을 언급하는 파일과 횟수:

| 파일 | 횟수 |
| --- | ---: |
| `docs/rps_aware_planning_design.md` | 7 |
| `experiments/results/pd_slo_sweep_margin.md` | 4 |
| `experiments/results/pd_slo_sweep.md` | 3 |
| `docs/PROJECT_REPORT.md` | 2 |
| `docs/deviations.md` | 2 (헤딩 + Open items summary 행) |
| `experiments/results/rngd_concurrency_envelope.md` | 1 |
| `experiments/scripts/pd_slo_sweep.py` | 1 (help text) |

주의: 이 파일들 안에서 "D21 §4.4"처럼 **하위 절 번호를 함께 인용**하는 곳이 있다. 절 번호는 브랜치 D21 본문의 내부 구조이므로 D22로 옮겨도 그대로 유효하다. 단순 치환하되, `main`의 D21(Tier 0)을 가리키는 문맥이 아닌지 각 사이트를 눈으로 확인한다. 커밋 메시지 히스토리의 "D21"은 고치지 않는다(rewrite 금지).

### 2.4 브랜치 내부의 자기모순 하나

`feat/pd-slo-margin-rerun`의 `PROJECT_REPORT.md` §4.8.7 caveat 블록은 두 단락이다. 첫 단락(9/2)은 "loose-TTFT winner가 `agg[cuda:tp4]`로 바뀜"이라 쓰고, 둘째 단락(8/31 작성)은 여전히 "*ordering* of the regimes below is still unaffected"라고 쓴다. 둘째 단락은 첫 단락 이전 시점의 텍스트다. STEP 1에서 둘째 단락의 해당 문장을 "loose-TTFT regime은 §pd_slo_sweep_margin.md에 의해 뒤집혔고, tight-TTFT regime만 미결"로 고친다. A2에 따라 삭제가 아니라 수정+날짜 표기.

### 2.5 ScenarioLab의 heteropilot 결합 지점 (분리 설계의 입력)

ScenarioLab → planner 방향 import (main 기준): `planner.inventory`(7), `planner.spec`(5), `planner.util.workload`(4), `planner.util`(3), `planner.topology`(3), `planner.predictor`(3), `planner.plan`(3), `planner.optimizer`(3), `planner.util.percentile`(2), `planner.optimizer.surrogate`(2), `planner.predictor.calibration`(1), `planner.envelope`(1). **역방향(planner → scenariolab) import는 없다.**

ScenarioLab 때문에 heteropilot 쪽에 들어간 변경:

| 위치 | 내용 | 처분 |
| --- | --- | --- |
| `planner/predictor/llmservingsim.py` `run_id_prefix` (PR #26 계열, 머지됨) | 동일 placement를 두 클러스터에서 평가할 때 run id 충돌 방지 | **유지.** 범용 기능, 주석의 "ScenarioLab" 예시만 남겨도 무해 |
| `planner/inventory.py` `Source.USER_DEFINED` (workspace-extras에만 존재, 5줄) | 사용자가 타이핑한 what-if 값의 라벨 | **heteropilot에 수용** (STEP 3.1). 라벨 enum 값 하나이며 `planner/`의 어떤 로직도 `Source` 값으로 분기하지 않음을 확인했다(`grep 'Source\.' planner`는 inventory.py의 기본값 4곳만). golden 불변으로 증명 |
| `pyproject.toml` `[tool.mypy] files`에 `"scenariolab"`, `[tool.ruff.lint.flake8-bugbear] extend-immutable-calls = ["fastapi.Depends", "fastapi.Query"]` | 린트/타입 설정 | heteropilot에서 제거 |
| `.gitignore` `outputs/scenariolab/` | | heteropilot에서 제거 |
| `profiles/networks/{ib_100g,ib_400g,nvlink,pcie_gen4}.yaml` | 랜덤 클러스터 생성기 전용 (헤더 주석에 명시) | 새 리포로 이동. **조사 필요:** `planner/`·`examples/`·`experiments/`에서 참조하는지 grep — 이 문서 작성 시 참조 없음 |
| `experiments/configs/lab/*.yaml` (5개) | ScenarioLab 배치 설정 | 새 리포로 이동 |
| `docs/scenariolab/` (5개 리포트), `docs/scenariolab_ui_checklist.md`, `DESIGN_scenariolab.md`, (extras 브랜치의) `WORK_ORDER_scenariolab_workspace.md`, `docs/scenariolab/workspace_test_report.md` | 문서 | 새 리포로 이동 |
| `tests/scenariolab/` (78 test functions), `tests/scenariolab/conftest.py` | **상위 `tests/conftest.py`의 픽스처 `cluster`(54회)·`profiles`(29)·`islands`(36)·`spec`(66)를 사용한다.** 그 픽스처들은 `examples/clusters/*.yaml`, `examples/service_specs/*.yaml`을 읽는다 | 새 리포로 이동 + `tests/conftest.py`도 filter-repo 경로에 포함, 픽스처의 `ROOT`를 `vendor/heteropilot`으로 |
| `scenariolab/config.py` `PROFILE_DIR = Path("profiles/accelerators")`, `calibration_dir = Path("profiles/calibration")`, `runner/interactive.py:101` 같은 기본값, `generator/cluster_gen.py` `NETWORK_DIR = Path("profiles/networks")` | **CWD 상대 경로**로 heteropilot 루트를 가정 | 새 리포에서 `HETEROPILOT_ROOT` 하나로 모은다 (3.2 지시 4) |
| `WORK_ORDER_tiered_profiles.md` §4.2 인용문 `files = ["planner", "scenariolab"]` | 역사 기록 | 그대로 둔다 |

### 2.6 미완 시뮬레이션 — tight-TTFT 구간

`pd_slo_sweep_margin.md`에 기록된 대로 `--timeout 1080`은 tp4 fixture의 `pd_cuda-a40-tp4` 후보 72개 전부를 타임아웃시켰고, 그중 하나가 committed tight-TTFT winner(`P[cuda:tp4] D[cuda:tp4]`, p99 TPOT 37.27 ms → ×1.18 = 43.98 ms, **통과 예상**)다. 재실행 커맨드는 그 문서 "Reproduce" 절에 있고, `--timeout 1800`으로 고쳐 써 있다. loose 지점(64000)은 완료되었으므로 **`--ttft-ms 500,8000`만** 다시 돈다.

### 2.7 이미 머지되어 삭제 가능한 원격 브랜치 17개

`main..origin/<b>`가 0인 브랜치: `chore/node-detection`, `docs/html-slide-deck`, `docs/project-report-and-npu-handover`, `docs/slide-outline`, `feat/a40-inventory`, `feat/baselines-ablation`, `feat/exp1-tp-sweep`, `feat/exp25-figures`, `feat/figure-packaging`, `feat/gpu-host-bandwidth`, `feat/rngd-parallel-bandwidth`, `feat/router-baselines`, `feat/sim-pd-transfer`, `feat/surrogate-and-parallel`, `feat/topology-perdim`, `fix/rngd-ttft-validation-arrivals`, `fix/scaling-curve-provenance-and-npu-envelope`.

HANDOVER §2.1은 `fix/scaling-curve-provenance-and-npu-envelope`를 "pushed, not merged"라 쓰지만 **이미 머지되어 있다.** HANDOVER 재작성(STEP 4) 때 반영.

### 2.8 옛 헤드라인 숫자가 살아 있는 `main` 사이트

`4.956` / `3.164` / `1.67×`를 담은 파일: `docs/PROJECT_REPORT.md`, `experiments/results/pd_slo_sweep.md` — 둘뿐이며 둘 다 브랜치가 superseded 표기를 이미 얹는다. `docs/slide_deck.html`, `docs/SLIDE_OUTLINE.md`, 브랜치의 `docs/slide_deck_ko.html`에는 이 숫자가 **없다**(grep 0건). 단, 숫자가 없어도 "RNGD wins on energy" 서사가 문장으로 들어 있을 수 있으므로 STEP 4에서 세 파일을 눈으로 훑는다.

---

## 3. 공통 규칙

### 3.1 첫 명령

```bash
bash scripts/whichnode.sh          # NPU 노드여야 한다. 아니면 이 문서 §2.6의 STEP 2는 실행 불가
export PYTHONPATH=$PWD && export PATH="$PWD/.venv/bin:$PATH"   # python -m chakra가 PATH의 python을 쓴다
git fetch --all --prune
```

### 3.2 테스트 게이트 (모든 STEP 공통)

```bash
pytest -q          # 전체 통과. 통과 수를 PR 본문에 적는다 — STEP별로 달라지는 것이 정상이며 그 이유를 함께 적는다
ruff check .
mypy
```

골든 회귀(`tests/`의 golden 파일 비교 테스트)가 어떤 STEP에서도 갱신되어서는 안 된다. 갱신이 필요해 보이면 중단·보고.

### 3.3 문서 수정 규칙

- 결과 파일·보고서의 틀린 숫자는 **삭제하지 않는다.** 해당 표/단락 바로 위에 `> **SUPERSEDED <날짜>.** …` 인용 블록을 얹고 새 산출물 경로를 가리킨다.
- 날짜와 커밋 sha를 반드시 적는다.
- 영어 파일은 영어로, 한국어 파일은 한국어로. 코드 주석·docstring·로그는 영어(`AGENTS.md`).

---

# STEP 0. 준비 — baseline 고정과 원격 정리

## 목표
스프린트 전 상태를 기록하고, 판단이 필요 없는 정리(머지된 브랜치 삭제)를 먼저 끝낸다.

## 지시

1. `main`에서 게이트 3종을 돌리고 결과를 `docs/consolidation_baseline.md`에 기록한다 (`docs/tier_baseline.md` 형식). `git rev-parse HEAD`, `bash scripts/whichnode.sh` 출력 포함.
2. §2.1~§2.8의 사전 조사를 노드에서 재확인한다. 특히 dry-run 머지:
   ```bash
   git checkout -b tmp/dryrun main
   git merge --no-commit --no-ff origin/feat/pd-slo-margin-rerun; git diff --name-only --diff-filter=U
   git merge --abort; git checkout main; git branch -D tmp/dryrun
   ```
   충돌 파일이 `.gitignore`, `docs/deviations.md` 두 개가 아니면 중단·보고.
3. §2.7의 17개 원격 브랜치를 삭제한다. 삭제 전 각 브랜치에 대해 `git rev-list --count main..origin/<b>`가 0임을 다시 확인하고, 그 출력을 PR 본문에 붙인다.
   ```bash
   git push origin --delete <b1> <b2> ...
   ```
4. 미머지 8개 브랜치는 건드리지 않는다.

## 완료 조건
- [ ] `docs/consolidation_baseline.md` 커밋
- [ ] 원격 브랜치가 `main` + 미머지 8개만 남음 (`git branch -r`)
- [ ] 게이트 3종 통과

---

# STEP 1. D22 정식화 — envelope 체인 머지

## 목표
`feat/pd-slo-margin-rerun`(체인 4개 포함)을 `main`에 머지하고, 번호 충돌과 자기모순을 해결해 D22를 canonical 기록으로 만든다.

## 지시

1. `main`에서 `chore/consol-step1-d22` 브랜치를 만들고 `origin/feat/pd-slo-margin-rerun`을 머지한다.
2. 충돌 해결:
   - `.gitignore`: 양쪽 블록 모두 유지.
   - `docs/deviations.md`: `main`의 D21(Tier 0) 절을 그대로 두고, 브랜치의 D21 절을 그 **뒤에** `## D22 — …`로 붙인다. `main` 쪽 D21 절 아래에 있는 `### D4 addendum (2026-09-02)`의 위치가 D21 절 안에 유지되는지 확인.
3. §2.3의 7개 파일에서 브랜치 유래 `D21` → `D22` 치환. 각 사이트를 열어 문맥이 envelope 철회를 가리키는지 확인한다. `experiments/scripts/pd_slo_sweep.py`의 help text 1건 포함.
4. `docs/deviations.md`의 **Open items summary 표**를 갱신한다: D21(Tier 0, Decided) 행과 D22(envelope, Resolved + tight-TTFT open) 행이 둘 다 있어야 한다. **D4 행은 `main`에서도 아직 "Resolution path secured"로 남아 있다** — D4 헤딩은 "Resolved 2026-09-02 (Tier 0 synthetic-bundle path)"인데 표가 따라오지 못한 것. 표를 헤딩에 맞춘다.
5. §2.4의 자기모순 수정: `PROJECT_REPORT.md` §4.8.7 caveat 둘째 단락의 "The *ordering* of the regimes below is still unaffected … until `pd_slo_sweep.py` is re-run at a defensible load" 문장을, 첫 단락과 일치하도록 고친다. 원문 뒤에 `(corrected 2026-09-0X: the loose-TTFT regime WAS overturned by the margin re-run above; only the tight-TTFT regime is undetermined)` 형태로 덧붙이는 방식으로 — 삭제 금지.
6. `docs/npu_concurrency_envelope_work_order.md` 상단에 "실행 완료 2026-08-31, 결과 `experiments/results/rngd_concurrency_envelope.md`, D22" 블록을 얹는다.
7. `outputs/rngd_envelope/edf/real_c*.json` (총 ~9,000줄)이 커밋에 포함된다. 이는 브랜치가 이미 내린 결정이며 이 프로젝트는 원시 측정 JSON을 커밋해 온 관례(`outputs/rngd_edf_bundle/`)가 있으므로 따른다. `.csv` 트레이스는 `.gitignore`로 제외되어 있음을 확인.
8. PR 본문에 (a) 체인 4개가 모두 포함됨을 `git log --oneline main..HEAD`로 보이고, (b) 치환 사이트 목록, (c) 게이트 결과를 적는다. 머지 후 원격 브랜치 4개(`docs/d18-close`, `feat/rngd-concurrency-envelope`, `docs/rps-aware-planning`, `feat/pd-slo-margin-rerun`) 삭제.

## 테스트

`pd_slo_sweep.py`에 `--tpot-margin-percent`/`--ttft-margin-percent` 플래그가 추가되었으나 브랜치에 테스트가 없다. 다음을 `tests/test_pd_sweep_margin.py`로 추가한다 (시뮬레이터 불필요):

```python
"""pd_slo_sweep.py의 margin 플래그가 exhaustive.search()의 기존 파라미터로 전달되는지 보장 (STEP 1)."""

def test_margin_default_is_zero():
    """플래그 미지정 시 margin 0.0 — 기존 호출이 byte-identical하다는 브랜치 주장의 회귀 방지."""

def test_margin_reaches_feasibility(monkeypatch):
    """--tpot-margin-percent 18 이 feasibility의 robust = predicted*(1+0.18) 계산에 도달한다.
    지시: exhaustive.search 를 monkeypatch 로 가로채 kwargs 를 캡처하고 값을 assert.
    """
```

## 완료 조건
- [ ] `docs/deviations.md`에 D21(Tier 0)과 D22(envelope)가 모두 존재, Open items summary 일치
- [ ] `grep -rn 'D21' docs experiments --include=*.md --include=*.py`의 결과가 전부 Tier 0 문맥
- [ ] §4.8.7 caveat 두 단락이 서로 모순되지 않음
- [ ] 게이트 3종 통과, 신규 테스트 2개 추가
- [ ] 원격 브랜치 4개 삭제

---

# STEP 2. tight-TTFT 구간 재실행 (이 스프린트의 유일한 시뮬레이션)

## 목표
`pd_slo_sweep_margin.md`가 "미결"로 남긴 TTFT ≤ 500 ms, ≤ 8 s 두 지점을 `--timeout 1800`으로 마저 돌려, 3-regime 표의 나머지 절반이 서는지 무너지는지 확정한다.

## 지시

1. STEP 1 머지 직후, 별도 셸에서 시작한다 (`nohup`/`tmux`). STEP 3·4를 병행한다.
2. 커맨드는 `pd_slo_sweep_margin.md` "Reproduce" 절 그대로, 단 `--ttft-ms 500,8000`과 새 `--output-dir`:
   ```bash
   for fx in pd-rngd-gpu-card pd-rngd-gpu; do
     PYTHONPATH=$PWD .venv/bin/python experiments/scripts/pd_slo_sweep.py \
         --service examples/service_specs/llama31-8b.yaml \
         --cluster experiments/configs/clusters/$fx.yaml \
         --ttft-ms 500,8000 --num-requests 300 --seed 42 --workers 32 \
         --timeout 1800 \
         --tpot-margin-percent 18 \
         --output-dir outputs/.hp-slo-margin18-tight-$fx
   done
   ```
   `--workers`는 노드의 CPU 수에 맞춘다(NPU 노드는 96 코어; 시뮬레이션은 CPU-only). 기존 `outputs/pd_slo_sweep_margin18/`은 **덮어쓰지 않는다.**
3. **소요 시간 추정을 먼저 적는다.** 타임아웃은 캐시되지 않고 TTFT 지점마다 재시도된다. 72개 tp4 후보가 전부 1800 s를 채우면 지점당 72×1800/32 ≈ 68분, 2지점×2 fixture면 최악 ~5시간 + 완료되는 후보들. 예상치와 실측 소요를 결과 파일에 함께 적는다.
4. 종료 후 결과 JSON을 `outputs/pd_slo_sweep_margin18/tight/` 아래로 복사해 커밋하고, `experiments/results/pd_slo_sweep_margin.md`에 **"Tight-TTFT regime (re-run <날짜>, timeout 1800)"** 절을 추가한다. 반드시 포함할 것:
   - fixture별·지점별 winner, p99 TTFT/TPOT, tok/J, **타임아웃 후보 수와 그 목록**.
   - committed winner `P[cuda:tp4] D[cuda:tp4]`가 실제로 완료·통과했는지. 예상(43.98 ms 통과)과 다르면 그 사실을 먼저 쓴다.
   - 결론 문장은 세 가지 중 하나여야 한다: (i) tight regime 유지 — pd_split이 여전히 승자, (ii) tight regime도 뒤집힘 — 무엇으로, (iii) **여전히 미결** — 어떤 후보가 1800 s에도 타임아웃했는지. (iii)이면 결과를 좋게 보이려 타임아웃을 더 올려 반복하지 않는다. 단, **committed winner 한 후보만** `--timeout 3600`으로 1회 재시도하는 것은 허용하며, 그 경우 "1 candidate, 3600 s, single retry"라고 명기한다.
5. `pd_slo_sweep.md`의 3-regime 표 위 superseded 블록을 STEP 2 결과에 맞게 한 문장 갱신하고, `PROJECT_REPORT.md` §4.8.7 caveat에도 한 문장 추가한다.
6. 이 STEP은 문서·산출물 커밋만 있고 코드 변경이 없어야 한다. 있다면 중단·보고.

## 완료 조건
- [ ] `outputs/pd_slo_sweep_margin18/tight/` 산출물 커밋 (provenance 포함)
- [ ] `pd_slo_sweep_margin.md`에 tight 절, 타임아웃 목록, 세 결론 중 하나
- [ ] `pd_slo_sweep.md`·`PROJECT_REPORT.md` §4.8.7 갱신
- [ ] 게이트 3종 통과 (테스트 수 불변)

---

# STEP 3. ScenarioLab 분리

## 목표
ScenarioLab을 히스토리를 보존한 채 별도 리포로 옮기고, heteropilot에서 제거해 논문 리포의 표면을 줄인다. 시연은 heteropilot의 특정 커밋에 pin되어 계속 동작해야 한다.

## 사전 결정 (사용자 확인 필요 — 기본값 제시)

| 항목 | 기본값 | 대안 |
| --- | --- | --- |
| 새 리포 이름 | `swsok/heteropilot-scenariolab` | `swsok/scenariolab` |
| heteropilot 참조 방식 | **git submodule** `vendor/heteropilot` + `PYTHONPATH` (heteropilot은 `[project]` 테이블이 없어 pip 설치 불가 — `pyproject.toml` 주석이 그 이유를 밝힘) | heteropilot에 `[project]`를 추가해 `pip install -e` — **비추천**, upstream 빌드에 영향 |
| pin 커밋 | STEP 3.1 머지 직후의 `main` sha (USER_DEFINED 포함) | — |

## 지시

### 3.1 heteropilot: `Source.USER_DEFINED` 수용 (선행 PR, 브랜치 `chore/consol-step3a-user-defined-source`)

1. `origin/feat/scenariolab-workspace-extras`의 `planner/inventory.py` 변경(5줄, `Source.USER_DEFINED = "user_defined"` + 주석)을 cherry-pick이 아니라 **수동으로 동일하게** 적용한다 (그 커밋은 scenariolab 파일도 함께 건드림).
2. **조사 필요:** `grep -rn 'Source\.' planner tests`로 `Source` 값에 따라 분기하는 로직이 없음을 재확인. 있으면 USER_DEFINED의 취급을 결정하고 PR에 기록.
3. 게이트 통과 + golden 불변. 머지 후 sha를 기록 — 이것이 새 리포의 pin이다.

### 3.2 새 리포 생성 (heteropilot 밖에서)

1. 별도 디렉터리에 heteropilot을 **새로 클론**하고(filter-repo는 fresh clone을 요구) workspace 브랜치를 로컬 브랜치로 만든다:
   ```bash
   git clone https://github.com/swsok/heteropilot.git scenariolab-split && cd scenariolab-split
   git branch workspace origin/feat/scenariolab-workspace-extras
   ```
2. `git filter-repo`(설치 필요: `pip install git-filter-repo`)로 경로를 추린다. 모든 ref(main, workspace)에 적용된다:
   ```bash
   git filter-repo \
     --path scenariolab --path tests/scenariolab --path tests/conftest.py --path tests/__init__.py \
     --path docs/scenariolab --path docs/scenariolab_ui_checklist.md \
     --path DESIGN_scenariolab.md --path WORK_ORDER_scenariolab_workspace.md \
     --path experiments/configs/lab --path profiles/networks
   ```
   결과에서 `main`은 P1~P4+topology v2까지, `workspace`는 그 위에 workspace P1~P3다. `planner/inventory.py` 변경은 경로 필터에 걸려 **떨어진다** — 그래서 3.1이 선행이다.
3. `workspace`를 `main`에 머지(fast-forward여야 한다 — 아니면 중단·보고).
4. 리포 뼈대 추가:
   - `git submodule add https://github.com/swsok/heteropilot.git vendor/heteropilot && (cd vendor/heteropilot && git checkout <3.1 sha>)`
   - `pyproject.toml`: heteropilot의 것에서 `[tool.mypy] files = ["scenariolab"]`, ruff 설정(`extend-immutable-calls` 포함), `[tool.pytest.ini_options] testpaths = ["tests"]`만 남긴다.
   - `README.md`: 목적(과제 시연), 실행법(`export PYTHONPATH=$PWD:$PWD/vendor/heteropilot`, `python -m scenariolab serve …`), pin된 heteropilot sha와 그 이유, 그리고 **"이 리포의 숫자는 heteropilot의 provenance 라벨을 그대로 전파하며, 자체 측정을 하지 않는다"**는 문장.
   - **루트 경로 통합.** `scenariolab/`은 heteropilot 루트를 CWD 상대 경로로 가정한다 — 확인된 사이트: `config.py:27 PROFILE_DIR = Path("profiles/accelerators")`, `config.py:107 calibration_dir = Path("profiles/calibration")`, `runner/interactive.py:101` 동일 기본값, `generator/cluster_gen.py:53 NETWORK_DIR = Path("profiles/networks")`. 그리고 `tests/conftest.py`·`tests/scenariolab/conftest.py`의 `ROOT = Path(__file__).resolve().parents[N]`. 이들을 하나의 `HETEROPILOT_ROOT`(환경변수, 기본값 `vendor/heteropilot`)로 모은다. `profiles/networks`는 새 리포 루트에 남으므로 `NETWORK_DIR`만 새 리포 기준. `git grep -nE 'Path\("(profiles|examples|experiments)' scenariolab tests`로 빠진 곳이 없는지 확인.
   - `experiments/configs/lab/*.yaml`이 heteropilot 경로를 담고 있으면 같은 루트 설정을 따르게 한다.
5. 새 리포에서 `pytest -q`(78 함수 기준), `ruff check .`, `mypy` 통과. `python -m scenariolab serve`로 UI가 뜨고 `verify --workspace` 경로가 pin된 heteropilot의 시뮬레이터를 찾는지 1회 확인(시뮬레이터 빌드는 `vendor/heteropilot` 안에서 `scripts/compile.sh` — 또는 heteropilot 체크아웃의 `.venv`를 재사용하는 방법을 README에 적는다).
6. GitHub에 리포 생성·push. workspace 원격 브랜치 3개는 새 리포에 반영되었으므로 heteropilot에서 삭제.

### 3.3 heteropilot: ScenarioLab 제거 (브랜치 `chore/consol-step3b-remove-scenariolab`)

1. §2.5 표의 "heteropilot에서 제거/이동" 항목을 `git rm`. `profiles/networks/`와 `experiments/configs/lab/`은 §2.5의 grep 조사 결과 참조가 없을 때만 제거.
2. `pyproject.toml`에서 `"scenariolab"`과 `extend-immutable-calls` 블록 제거. `.gitignore`에서 `outputs/scenariolab/` 블록 제거.
3. `docs/HANDOVER.md`·`CLAUDE.md`·`README.md`(fork 부분이 있다면)에 "ScenarioLab은 `<새 리포 URL>`로 이전, heteropilot `<sha>`에 pin" 한 줄.
4. `CHANGELOG.md`는 upstream 파일이다 — 건드리지 않는다. fork의 변경 기록은 `docs/deviations.md`나 HANDOVER가 담당한다.
5. 게이트 통과. `pytest -q` 통과 수가 정확히 78개 함수분만큼 줄었는지(parametrize 때문에 함수 수 ≠ 통과 수일 수 있음 — 실제 감소량과 그 이유를 PR에 적는다). golden 불변.

## 완료 조건
- [ ] 새 리포에서 게이트 3종 통과, UI 1회 기동 확인
- [ ] heteropilot에 `scenariolab` 문자열이 남은 곳이 문서 포인터와 `llmservingsim.py` 주석, `WORK_ORDER_tiered_profiles.md` 인용문만임 (`git grep -i scenariolab`)
- [ ] heteropilot `mypy` 대상이 `planner`, `profiler/synth`, `profiler/contract.py`
- [ ] workspace 원격 브랜치 3개 삭제

---

# STEP 4. 문서 재정렬 — 세 문서를 한 시점으로

## 목표
`HANDOVER.md`, `PROJECT_REPORT.md`, `CLAUDE.md`가 모두 이 스프린트 종료 시점의 `main`을 가리키게 한다.

## 지시

### 4.1 `docs/HANDOVER.md` 재작성

기존 파일은 `docs/HANDOVER_2026-08-31.md`로 옮겨 역사 기록으로 남기고(상단에 "historical" 표기), 새 HANDOVER를 쓴다. 필수 내용:

- 기준 커밋·날짜, 게이트 결과(실제 pytest 통과 수), `whichnode.sh` 안내(§0 그대로 유지).
- §1 Status 표에 **Tiered Profiles 행 추가**(완료, D4 해소, `docs/tier0_calibration.md`), ScenarioLab 이전 한 줄.
- "What the last cycle established"를 이 스프린트 기준으로: Tier 0 E1~E4 핵심 숫자(τ 0.90–0.91, top-1 오답 비용 0.4 %/11.3 %, E2 attention-anchoring 38.9→29.5 %), D22(1.31×/18 %, loose-TTFT 뒤집힘, tight-TTFT는 STEP 2 결론), 그리고 "현재 어떤 실험에서도 이종 구성이 이기지 않는다"는 사실을 **한 문장으로 명시**한다. 이것이 다음 작업지시서의 출발점이다.
- §2 Next work: 1순위를 `WORK_ORDER_rps_aware.md`(미작성)로 두고, 그 안에 들어갈 저부하 envelope 측정을 "이 스프린트에서 의도적으로 하지 않았다"고 적는다. 2.2(ATOM/D20), 2.3(TTFT tail), 2.4(D14) 유지. 2.1의 "branch ready" 문장은 삭제하고 D22 완료로 대체. 2.5의 PR #13은 4.3의 결정으로 대체.
- §3 Traps, §4 Invariants는 유지하되 D22가 추가한 trap 두 개를 넣는다: **"요청 풀 크기가 동시성을 제한한다 — 실효 동시성(Little's law)을 기록하고 requested를 기록하지 말 것"**, **"sweep의 INFEASIBLE은 타임아웃 카운트를 확인하기 전에는 결과가 아니다"**.

### 4.2 `docs/PROJECT_REPORT.md` 갱신

전면 재작성이 아니다. 상단 snapshot 날짜·sha 갱신, §3 phase 표에 Tiered Profiles 추가, **§4.9 "Tier 0/1 — planning on hardware we do not own"** 절 신설(`tier0_calibration.md`의 E1~E4 표 요약 + 경로), §6 "What remains" 표에서 D22로 해소된 행 정리, ScenarioLab 이전 각주. §4.8.7은 STEP 1·2에서 이미 처리됨.

### 4.3 PR #13 (`docs/slide-deck-ko`) 처분

`docs/slide_deck_ko.html`을 열어 RNGD 에너지 승리 서사(숫자가 아니라 문장)가 있는지 확인한다.
- 없으면: 머지하되 첫 슬라이드에 "2026-08-25 기준, D22 이전 — `docs/PROJECT_REPORT.md` §4.8.7 참조" 배너를 넣는다.
- 있으면: 해당 슬라이드에 superseded 표기를 넣어 머지한다. **닫지 않는다** — 빌더가 bilingual이라 영문 덱과 함께 관리되어야 한다.
`docs/slide_deck.html`과 `docs/SLIDE_OUTLINE.md`도 같은 기준으로 훑고, 서사가 있으면 같은 표기를 넣는다.

### 4.4 `CLAUDE.md`

"Authoritative documents" 표에 `WORK_ORDER_tiered_profiles.md`, `docs/tier0_calibration.md`, 이 문서를 추가한다. "What this repository is"의 fork 디렉터리 목록(`planner/`, `profiles/`, `experiments/`, `examples/`, `tests/`)에 `profiler/synth/`, `profiler/contract.py`를 추가한다(STEP 0 baseline 문서가 이들이 fork 소유임을 확인했다). ScenarioLab 언급이 있으면 이전 포인터로.

## 완료 조건
- [ ] HANDOVER·PROJECT_REPORT·CLAUDE.md의 기준 sha가 동일
- [ ] PR #13 머지 완료, 세 덱 파일에 필요한 superseded 표기
- [ ] 게이트 3종 통과 (문서만 변경)

---

# STEP 5. `docs/CLAIMS.md` — 지금 주장할 수 있는 것

## 목표
논문 아웃라인의 입력. 리뷰어가 물을 순서로 "무엇을, 어떤 근거로, 어떤 라벨로" 주장할 수 있는지 한 페이지. **새 숫자 없음** — 전부 커밋된 산출물 경로를 가리킨다.

## 지시

영어로 쓴다(논문 입력). 세 절:

1. **Established (measured or simulated on measured profiles, reproducible from committed artifacts).** 각 항목: 한 문장 주장 / 숫자 / 라벨(measured·sim-on-measured·analytical) / 경로. 최소 포함:
   - Tier 0+sim 순위 보존 (τ, top-1 비용), E2 앵커 배분, E3 shape 중복 0.8–16 %, E4 Ascend 민감도.
   - RNGD envelope: eff 15.3→107.2에서 585.8→1473.3 tok/s, exponent 0.675→0.241, TPOT 25.7→67.9 ms. 시뮬레이터 오차 1.31×/18 % at eff 76.
   - D22 결과: loose-TTFT에서 RNGD 전 후보 infeasible, margin 3.3 % 이상이면 동일. fixture 독립.
   - tight-TTFT: STEP 2의 결론 그대로.
   - 측정된 fabric(12.6–13.0 GB/s composed), RNGD 전력 모델(38.01+32.71×PE, R² 0.996), A40/A5000 프로파일 정확도, TTFT −5.1 %(matched arrivals).
2. **Not established — and why.** 이종 P/D가 이기는 구성은 없음(D14로 열거 불가한 형태 포함), ATOM(D20), RNGD 저부하 구간(미측정 — 다음 작업지시서), 전력 crossover(설계서의 가설, 측정 아님), Tier 0 fp8/TP≥4(S5 미착수).
3. **Retracted, with the correction.** D18, D19, D22 각 한 줄: 무엇을 주장했고, 무엇이 측정됐고, 무엇이 바뀌었나.

각 절의 항목은 `docs/`·`experiments/results/`의 문서와 숫자가 일치해야 한다. 작성 후 스크립트로 인용한 모든 경로가 존재하는지 확인한다(`grep -oE '[a-zA-Z0-9_./-]+\.(md|json|yaml|csv)' docs/CLAIMS.md | sort -u | xargs ls`).

## 완료 조건
- [ ] `docs/CLAIMS.md` 커밋, 인용 경로 전부 존재
- [ ] HANDOVER §1에서 링크

---

# 6. 전체 완료 조건

- [ ] STEP 0~5 개별 완료 조건 전부
- [ ] `git branch -r`이 `origin/main`만 (또는 진행 중인 consol 브랜치만) 보임
- [ ] `docs/deviations.md`에 D21(Tier 0)·D22(envelope) 공존, Open items summary 최신
- [ ] `git grep -c '4\.956\|3\.164\|1\.67'`의 모든 사이트에 superseded 표기가 인접
- [ ] heteropilot에 `scenariolab/` 없음, 새 리포가 pin된 heteropilot으로 기동
- [ ] `docs/CLAIMS.md` 존재, HANDOVER·PROJECT_REPORT·CLAUDE.md가 같은 sha를 가리킴
- [ ] 이 스프린트에서 `planner/` 변경은 `Source.USER_DEFINED` 5줄뿐, golden 불변

# 7. 리스크 대응 규칙

| 상황 | 판단 |
| --- | --- |
| dry-run 머지 충돌이 §2.2와 다르다 | 브랜치가 이 문서 작성 후 바뀌었다. 중단, `git log origin/feat/pd-slo-margin-rerun`으로 차이를 확인하고 §2를 고친 뒤 진행 |
| `Source` 값으로 분기하는 로직이 발견된다 | USER_DEFINED를 PLACEHOLDER와 같게 취급(candidate 생성에서 제외)하는 것이 안전한 기본값. 결정과 이유를 PR에 기록 |
| STEP 2에서 committed tight winner가 1800 s에도 타임아웃 | 3600 s 단일 재시도 1회만. 그래도 미결이면 "미결"로 기록. 타임아웃을 계속 올리거나 `--num-requests`를 줄여 결과를 만들지 말 것 — 줄이면 committed sweep과 비교 불가 |
| STEP 2 결과가 tight regime도 뒤집는다 | 결과다. §4.8.7과 CLAIMS.md에 그대로. "P/D가 sub-second를 산다"는 문장이 사라지면 그것도 적는다 |
| filter-repo 후 `workspace`가 `main`에 fast-forward되지 않는다 | 경로 필터가 두 ref에서 다른 충돌을 남긴 것. 강제 머지하지 말고 어느 커밋에서 갈라졌는지 보고 |
| 새 리포에서 scenariolab이 heteropilot 루트를 하드코딩으로 가정하는 곳이 많다 | 하나의 `HETEROPILOT_ROOT` 설정(환경변수 또는 LabConfig 필드)으로 모은다. `vendor/heteropilot` 기본값. 코드 변경은 새 리포에서만 |
| 문서를 고치다 새 실험을 하고 싶어진다 | 하지 않는다. `WORK_ORDER_rps_aware.md`의 "후보 실험" 목록에 한 줄 적어 두고 넘어간다 |
| upstream 파일(`CHANGELOG.md`, `AGENTS.md`, `serving/`)을 건드려야 할 것처럼 보인다 | 중단·보고 |
