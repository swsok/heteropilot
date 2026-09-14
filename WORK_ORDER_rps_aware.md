# HeteroPilot 작업지시서 — RPS-aware 계획: 성능·전력·오차를 운영점의 곡선으로

> 목적: planner가 "이 서비스가 실을 RPS에서 어느 가속기·어느 구조가 SLO를 지키며 에너지를 가장 적게 쓰는가"에 답하게 만든다. 그러기 위해 (1) 스칼라 프로파일을 **측정된 envelope 곡선**으로 보강하고, (2) 시뮬레이터의 오차를 **운영점의 함수**로 선언해 feasibility margin을 자동 유도하고, (3) RPS를 탐색 축으로 올려 **switchover 표와 crossover**를 산출하고, (4) D14 스파이크가 표현 가능함을 보인 **비대칭 TP P/D**를 후보 집합에 넣는다. 검증 실험 E5·E6가 논문의 positive/negative result를 결정한다.
> 대상 저장소: `github.com/swsok/heteropilot` · 기준 `main` = `de4d235` · 작성일: 2026-09-07 · 도구: Claude Code CLI, NPU 노드(측정 STEP 3 포함)
> 설계 근거: `docs/rps_aware_planning_design.md`(2026-09-01). 이 작업지시서는 그 설계를 구현 단위로 자르고, 이후 세 작업지시서(통합·스파이크·D23 수정)가 바꾼 사실을 반영한다.
> 선행 완료: D22(envelope c16–c128), D23/D25/D26(harness), D14 스파이크(`slab3d`), 3-regime 표 확정(`docs/d23_revalidation.md`).
> 예산: **약 3.5주** (STEP 0 0.5일 · 1 1일 · 2 3일 · 3 2일 · 4 5일 · 5 4일 · 6 2일). 11월 학회 마감 기준 10월 중순에 E5·E6 결과가 있어야 한다.
>
> **개정 rev 2 (2026-09-08, `main` = `a7b06eb`, `origin/feat/rps-step1-sim-cost` = `2578575`).** STEP 0·1 완료 결과를 반영해 다음을 바꿨다:
> (a) STEP 0·1을 "완료"로 표기하고 결과 요약을 남김. STEP 1의 30 % 규칙은 CLI가 근거를 들어 넘었고 **그 판단을 승인**한다(§STEP 1).
> (b) **STEP 1.5 신설** — `step1` 브랜치 머지 전 정리: §4.7 "regret 0 at every K" 부분 철회 표기, 브랜치 삭제.
> (c) **E6a 재설계** — 실측 비용(1 rps = 10 rps의 10.3×, 축 합계 21.4×)으로 218 h → D27 후 ~140 h. `--top-k`는 **사용 불가**(surrogate 발견). 대신 64 워커 + 구조별 knob 고정 + RPS {1, 3.3, 10, 20}으로 **~9 h**.
> (d) STEP 3에 STEP 0이 확인한 노드 사실(`furiosa-smi status`의 PE별 이용률, 1 W 양자화, 장치명 불안정)을 반영.
> (e) STEP 3은 STEP 2와 **병행 시작** — 하드웨어 측정이 코드와 독립이라 먼저 끝내는 것이 일정상 유리.

---

## 0. 이 문서의 사용법

1. **STEP 순서를 지킨다.** 단 STEP 3(측정, NPU 하드웨어)은 STEP 1·2와 독립이므로 노드 일정에 맞춰 앞당길 수 있다. STEP 4는 STEP 2·3의 산출물(calibration 파일, envelope 파일)을 소비한다.
2. **한 STEP = 한 브랜치 = 한 PR** (`feat/rps-step<N>-<slug>`). STEP 2·4는 하위 PR로 쪼개도 된다 — 각각 게이트를 통과할 것.
3. **결과를 좋게 보이려고 조건을 바꾸지 않는다.** crossover가 측정 범위 안에 없으면 그것이 결과다. E5가 실패하면 그것이 결과다. 이 프로젝트가 D18·D19·D22를 철회하며 지킨 규율이다.
4. **"조사 필요"는 추측하지 말고 코드·산출물을 읽어 확인**하고 결론을 PR 본문에 적는다.
5. 상위 문서 우선: `WORK_ORDER_heteropilot.md` → `docs/deviations.md` → `CLAUDE.md`. `docs/CLAIMS.md`를 바꾸는 결과는 즉시 반영한다.

### 절대 규칙 (재확인 + 이 문서의 추가)

- **A1.** `scripts/whichnode.sh`가 나열하지 않는 하드웨어의 결과를 주장하지 않는다. A40 envelope은 A40 노드 없이는 측정하지 않는다 — 시뮬레이션(validated ±2 %)으로 대신하고 그렇게 라벨한다.
- **A2.** 철회는 공개적으로.
- **A3.** `serving/` 편집은 이 문서가 명시한 것만(STEP 1의 D27, STEP 2의 D28), 각각 D15 방식 — **opt-in 또는 동작 동일, 기본 출력 byte-identical을 회귀 앵커로 증명**. `astra-sim/` 불가.
- **A4.** golden 회귀 출력 불변. 새 기능은 전부 opt-in(`--rps`, `--enable-pd`, `topology_mode`, envelope 존재 시에만 margin 자동화).
- **A5 — 측정 규율 (D18·D22에서 배운 것, 이 문서에서 강제).** (a) 동시성은 항상 **served**(Little's law `Σlatency / wall`)로 기록하고 requested는 참고값. (b) 요청 풀은 동시성의 **4배 이상**, `served/requested ≥ 0.9`를 매 점에서 확인. (c) 전력은 **sustained 평균**(best-of-N 금지)이고 **`util_pct`와 같은 표본에서** 기록 — 이용률 없는 전력값은 측정이 아니다. (d) idle은 45 s settle 후 60 s 평균. (e) 모든 점은 같은 워크로드(`measured_on_workload`)에서.
- **A6 — envelope 밖은 숫자가 아니다.** `validity.extrapolation: refuse`가 기본. 측정 범위 밖 운영점은 `rejected_summary`의 별도 범주로 기록되며, 값이 아니다.

---

## 1. 배경 — 지금 서 있는 곳

세 작업지시서 뒤의 사실:

- **3-regime 표는 닫혔다** (`docs/d23_revalidation.md`): tight(≤0.5/8 s) → 균일 cuda P/D 2.205 tok/J, loose → `agg[cuda:tp4]` 2.595. **RNGD는 어느 regime에서도 SLO를 통과하지 못한다**(0/45 단독, 0/18 혼합) — 10 rps에서 TTFT와 TPOT 사이에 끼어서다(D22).
- **이기종 구성이 이기는 실험은 없다** (`docs/CLAIMS.md` §2). 남은 positive result 후보는 둘: **(P1) 저부하 crossover** — tokens/J가 동시성에 단봉이라면 idle 전력이 낮은 RNGD가 저RPS에서 이긴다는 가설(설계서 §1), **(P2) 비대칭 TP P/D** — `A40 tp4 prefill + RNGD tp8 decode`, 업계 권장 형태, 지금까지 열거조차 못 했고 스파이크가 표현 가능·완주를 확인. 둘 다 아직 측정·시뮬레이션되지 않았다.
- **측정 공백**: RNGD envelope은 eff 15.3부터다. 그 아래(P1이 사는 곳)는 미측정. 전력은 envelope 측정 때 기록되지 않았다.
- **planner 공백**: `AcceleratorProfile`은 스칼라. `feasibility.py`의 `ttft/tpot_margin_percent`는 수동 상수. RPS는 `ServiceSpec.traffic.arrival_rate_rps` 고정값. `_pd_candidates`는 `tp_p == tp_d`.
- **harness는 이제 신뢰할 수 있다**: D25·D26 이후 32 워커 병렬이 0 타임아웃·소수점 동일. 이 문서의 스윕은 그 위에서 돈다.

이 문서가 끝나면 논문은 다음 중 하나를 갖는다: P1 또는 P2가 성립하는 **positive result**, 또는 "측정 범위 안에서 이 하드웨어 쌍에는 crossover가 없고 비대칭 P/D도 이기지 못한다"는 **정직한 negative result + 방법론**. 어느 쪽이든 §6의 문서가 그것을 적는다.

---

## 2. 사전 조사 (2026-09-07, `de4d235`에서 확인)

### 2.1 이미 있는 것

| 필요한 것 | 현재 | 위치 |
| --- | --- | --- |
| margin을 받는 feasibility | 있음, 수동 상수 | `planner/optimizer/feasibility.py:50-67` `robust = predicted × (1 + margin/100)` |
| `rejected_summary` 범주 | `RejectionStage` enum 5+1개 | `planner/plan.py:129-143`; 새 범주 추가 시 oracle-agreement 테스트가 relaxation 규칙을 강제한다 |
| 시뮬레이션 결과에서 **served 동시성** | 계산 가능 — CSV에 `instance id, arrival, end_time, latency` | `outputs/**/sim*.csv` 헤더; 인스턴스별 `Σlatency / (max end − min arrival)` |
| 시뮬레이터 오차 측정점 | 2점: eff 16.6에서 TPOT −3.1 %, eff 76에서 throughput +31 %/TPOT −18 % | `experiments/results/rngd_concurrency_envelope.md`, `profiles/calibration/rngd_card_edf.yaml`(`errors` 블록은 bucket-scoped, 운영점 축 없음) |
| RNGD envelope 측정 harness | `rebuild_rngd_bundle_from_edf.py collect --concurrency --num-reqs` | `experiments/scripts/`; 풀 크기(`--num-reqs`)가 동시성을 제한했던 것이 D22의 원인 — **풀은 동시성의 4배 이상** |
| 전력 샘플러 | 별도 스크립트(upstream 프로파일러 `llm_profile/profiler/power/profile_rngd_power.sh`): `furiosa-smi info --format json` 을 `sleep 1` 루프로 폴링, `dev_name, power`만 기록 | 이 문서 §STEP 3이 개선판을 만든다 — 타임스탬프 ms, 드리프트 없는 간격, **util 필드 동시 기록**(조사 필요: `furiosa-smi info --format json`에 이용률 필드가 있는가; 없으면 `furiosa-smi status`/`top` 계열에서 같은 순간 값을 얻는 방법) |
| RNGD 카드 전력 모델 | 스칼라 `idle 39.35 / standby 265 / active 290.93 W` | `profiles/accelerators/furiosa_rngd_card.yaml:175-180`; PE 분해 `board = 38.01 + 32.71×PEs`(R² 0.996) |
| 비대칭 TP 표현 | 프로토타입 `topology_mode: slab3d`, 미머지 | `origin/spike/d14-asym-tp`의 `serving/core/config_builder.py`(+141줄), `docs/d14_spike.md` |
| `slab3d` 정확도 보정 | dim-1 `link_latency` = 스칼라의 **4×**(80,000 ns)에서 flat 대비 0.008 % | 한 대역폭(16)·한 분할(`[4,2]`)·한 모델에서 측정 — calibration domain 없음 |
| trace 열 폭 버그 | `ALLREDUCE:1,1,0`(15자)이 `_FMT` comm_type 열(15자)과 정확히 같아 `comm_size`와 붙음 | `serving/core/utils.py:11 _FMT`; `generate_trace`가 자기 파일을 `re.findall(r'\S+')`로 재파싱 |
| 시뮬레이션 비용 | 재검증 한 TTFT 지점(≈250 후보, 300 req, 32 워커) **1 h 43 m** | `docs/d23_revalidation.md` §2; RPS 스윕은 지점 수 × fixture 수 배 |
| Chakra 변환기 호출 | iteration마다 인스턴스별 **subprocess**(`sys.executable -m chakra…`, D26 이후) | `serving/core/graph_generator.py:30-43` |

### 2.2 D26이 소급해서 흔드는 두 결론 — STEP 0에서 재확인

1. **"arrival rate를 낮추면 RNGD 후보가 종료하지 않는다"** (D22 §4.4, 2026-09-01, 3.3 rps 시도: RNGD P/D 24개 시도 0 완료, cuda-only 84개 중 60 완료). 이것은 **D26의 서명**(다중 인스턴스만 hang, 단일 인스턴스 완료)과 일치한다. 그때의 결론 "느린 drain"은 아마 틀렸고, E6가 저RPS를 스윕해야 하므로 **반드시 재시험**한다.
2. **Exp 5 4-combo·`pd_slo_sweep` committed 표**(8/27–28, NPU 노드, PATH 함정 기록 전). D25는 거짓말을 못 하지만 **D26은 완주하면서 다른 값을 낼 수 있다**(MoE bench 앵커가 그 예). D22 근거 행은 재검증에서 소수점 동일이었으므로 그 sweep 환경은 올발랐을 가능성이 높지만, 같은 후보의 tok/J가 당시 2.206·지금 2.2051로 미세하게 다른 것은 설명이 필요하다.

### 2.3 설계서와 이 문서의 차이 — 결정 사항

- **시뮬레이터는 여전히 예측기다.** envelope은 시뮬레이터를 대체하지 않는다. 역할은 셋: (a) `accuracy_domain`으로 시뮬레이터 오차를 운영점의 함수로 선언 → margin 자동화, (b) 운영점이 측정 범위 밖이면 **인식론적 거부**(A6), (c) **시뮬레이터와 독립적으로** 측정 곡선+전력에서 tokens/J(RPS)를 직접 계산해 시뮬레이션 결과와 대조(E6b). 설계서의 "envelope 기반 solver로 plan을 낸다"는 (c)의 대조 축으로 격하한다 — 측정 곡선이 있는 하드웨어가 RNGD 하나뿐이어서 planner의 주 경로가 될 수 없다.
- **A40 쪽 곡선은 시뮬레이션이다.** A40 노드에 접근할 수 없으면 A40의 RPS 곡선은 validated(±2 %, `docs/a40_sim_vs_real_plan.md`) 시뮬레이터가 만들고 `sim-on-measured`로 라벨한다. crossover는 "RNGD 측정 vs A40 시뮬레이션"으로 보고하고, A40 노드가 생기면 STEP 3b로 보강.
- **운영점은 시뮬레이션 출력에서 읽는다.** feasibility margin에 쓰는 동시성은 Little's law solver의 추정치가 아니라 그 후보의 **시뮬레이션 CSV에서 계산한 인스턴스별 served 동시성**이다(§2.1). solver는 시뮬레이션 전 사전 거부(측정 범위 밖)와 E6b에만 쓴다.

---

## 3. 공통 규칙

```bash
bash scripts/whichnode.sh
export PYTHONPATH=$PWD && export PATH="$PWD/.venv/bin:$PATH"   # D26 이후에도 습관으로 유지
pytest -q && ruff check . && mypy
```

**회귀 앵커** (모든 `serving/` 편집 PR):
- **R1** `bench/examples/` 3종(Llama, Qwen3-32B, MoE) — D26 이후 4/4 재현되는 값. `outputs/d23fix/evidence/anchor_after_d25_d26/SHA256SUMS`가 기준.
- **R2** `P[cuda:tp4] D[cuda:tp4] -s256-t8192` 300 req — 재검증의 sim1.csv.
- **R3 (STEP 2 이후)** colocated tp4×2를 `auto`와 `slab3d`로 — byte-identical(스파이크 B.2-2).

**측정 산출물 규칙**: 원시 JSON/로그는 `outputs/rngd_envelope_lowload/`에 커밋(선례: `outputs/rngd_envelope/edf/`), 전력 원시 로그(1 Hz CSV)도 커밋(수 MB). provenance(`planner/util/provenance.py`)에 `whichnode` 인벤토리 포함.

---

# STEP 0. D26 소급 점검 — ✅ 완료 (PR #60, `2c373b9`)

> **결과.** (1) 3.3 rps "종료하지 않음"은 **D26 아티팩트** — 고친 harness에서 RNGD P/D 74 s, cross-vendor 82 s(20 req)에 완주. E6의 저RPS 축 사용 가능. D22 §4.4 reason 2(목적함수 논거)는 유지. (2) Exp 5 4-combo 5행 모두 소수점까지 재현 — D26 소급 오염 없음. (3) 2.206 vs 2.2051은 planner 쪽이 아니라 **fixture의 `link_bw` 35.0/35.2 차이**(이 문서 초판의 추정이 틀렸음). (4) STEP 3용 노드 사실: `furiosa-smi status --format json`에 PE별 `pe_utilizations`와 DRAM `used_ratio`가 있음(`info`에는 없음); 전력 1 W 양자화(idle ~38 W에서 1 quantum = 2.6 %); 장치명 `npuN`이 재열거로 바뀜 — **`device_sn`/`pci_bdf`로 식별**.
>
> 아래 원문은 기록용.


1. **3.3 rps 재시험.** D22 §4.4의 3.3 rps 서비스 spec(`examples/` spec의 `arrival_rate_rps`만 3.3)으로, 그때 종료하지 않았던 RNGD P/D 후보 하나와 cross-vendor 후보 하나를 **20 요청**으로 `livelock_watch.sh` 아래에서 실행. 완주하면 D22 §4.4의 "종료하지 않는다"는 결론 위에 "D26 아티팩트였음, <날짜>" 표기를 얹고, E6의 저RPS 스윕이 가능함을 기록. 완주하지 않으면 exit code(3=틱 정지 / 4=자식 사망)와 로그로 새 결함인지 판정 — 새 D 번호.
2. **Exp 5 대표 4개 재실행** (`experiments/results/pd_4combo_table.md`(`pd_4combo.json`)의 4 representatives, 같은 seed·요청 수·knobs, 20이 아니라 **원래 요청 수**). committed 값과 소수점 대조. 다르면 D26 소급 오염 → 해당 결과 파일 위에 superseded 표기 + `CLAIMS.md` 갱신; 같으면 "D26 소급 점검 통과" 한 줄.
3. **tok/J 2.206 vs 2.2051.** `outputs/.hp-pd-slo/pd_slo_sweep.json`의 해당 후보 레코드와 `outputs/d23fix/step3/evidence/tight_pd-rngd-gpu.json`의 hp-00365를 필드별로 diff. CSV는 동일(R2)이므로 차이는 planner 쪽 — 전력 블록(8/27 갱신) 또는 `link_bw`(D18) 유래일 것. 원인을 `docs/d23_revalidation.md`에 한 단락으로.

완료 조건: 세 항목의 결론이 각각 한 문장으로 기록됨. 코드 변경 없음.

---

# STEP 1. 시뮬레이션 비용 — ✅ 완료 (`2d9c8f4`, PR #61로 머지됨)

> **결과.** 프로파일: `read_wait`(ASTRA-Sim 대기) 53.9 %, Chakra subprocess 28.9 % (10 rps; 3.3 rps에서 51.6/29.6 %). 규칙상 30 % 미만이지만 **D27 시행** — 이유: 저RPS 배수(300 req 기준 3.3 rps = 2.63×, 1 rps = **10.32×**)로 E6a가 60 h가 아니라 **218 h**였고, `--top-k 20`은 FEASIBLE plan을 INFEASIBLE로 바꾸는 것이 확인되어 비용 절감 수단으로 쓸 수 없었다. D27은 정확도 손실 없는 유일한 수단. **이 판단을 승인한다.** 구현: `LLMConverter` 직접 호출(`main()`의 `setup_logging` 부작용 회피), 모듈 전역 상태 없음 확인. 1.55–1.72× 빨라짐, R1×3·R2·3개 rate CSV **7건 byte-identical**. D26은 subsumed(해석할 인터프리터가 없어짐; venv에 chakra가 없으면 ImportError — 더 나은 실패). 테스트 483.
>
> **같은 브랜치에 surrogate 발견(PR #62 → step1 브랜치로 머지됨)이 포함되어 있다** — §STEP 1.5.
>
> 아래 원문은 기록용.


## 지시

1. R2 후보를 `pyinstrument -r html -o outputs/perf/r2_profile.html -- python -m serving …`로 1회 프로파일. 시간 분해: (a) scheduler/trace 생성, (b) `graph_generator.generate_graph`의 chakra **subprocess**(기동+변환), (c) `controller.read_wait`(ASTRA-Sim 대기), (d) 파일 I/O. 표로 `docs/sim_cost_profile.md`에.
2. **결정 규칙**: (b)가 벽시계의 **30 % 이상**이면 D27을 시행한다. 미만이면 D27을 하지 않고 이유를 기록한다 — 이후 STEP의 스윕 시간 추정에 (a)~(d)를 쓴다.
3. **D27 (조건부, sanctioned edit)** — `graph_generator.py`에서 변환기를 subprocess 대신 **in-process 함수 호출**로. `chakra.src.converter.converter`의 `main`/클래스를 import해 같은 인자로 호출. `sys.executable`과 같은 인터프리터이므로(D26) protobuf 버전 문제 없음. `--no-in-process-chakra` 같은 opt-out은 두지 않는다 — 대신 **출력 `.et` 바이트 동일**을 테스트로: 같은 trace를 두 경로로 변환해 `sha256` 비교(`tests/test_chakra_inprocess.py`). 주의: 변환기가 모듈 전역 상태(카운터, 로거)를 가지면 두 번째 호출이 오염된다 — 조사 필요, 있으면 호출마다 리셋하거나 subprocess를 유지하고 그 사실을 D27에 기록.
4. R1·R2 byte-identical. 20 req R2의 벽시계 before/after를 PR에.

## 완료 조건
- [ ] `docs/sim_cost_profile.md`(분해 표, 결정, 근거)
- [ ] D27 시행 시: `.et` sha256 동일 테스트, R1·R2 동일, 속도 향상 수치, `docs/deviations.md` D27

---

# STEP 1.5. 브랜치 정리와 §4.7 부분 철회 (0.5일) — **STEP 2 착수 전 필수**

## 배경 — surrogate 발견 (`0659675`, `docs/surrogate_topk_regret.md`)

> **결론이 2026-09-11에 뒤집혔다 (PR #76).** 아래 본문의 *"랭커를 바꾸지 않는다"* 는
> 더 이상 유효하지 않다. 당시의 판단 근거는 대안 랭커(`floor`)가 두 fixture를 고치고
> **세 번째를 깨뜨린다**는 것이었고, 그 판단 자체는 옳았다 — 틀린 것은 "그러므로 고칠
> 수 있는 랭커가 없다"는 암묵적 결론이었다.
>
> **수정은 이 절의 증거 안에 있었다.** `tpj_then_floor`(병렬성 항으로 명시적 tie-break)가
> `roofline`과 **바이트 동일**하게 측정됐다는 사실이 기록돼 있었는데, 그것은 무결과가
> 아니라 원인이다. tp·dp 상쇄는 대수적이지만 연산은 부동소수점이라 약 1만분의 1이
> 남고(4.65에서 0.000463), 따라서 정확히 같은 값이 없어 tie-break가 한 번도 발동하지
> 않는다. 랭커는 병렬성 축을 **무시한** 것이 아니라 반올림 오차가 대신 고르게 두고
> 있었다.
>
> `BinnedRooflineRanker`는 tie를 명시화한다 — proxy tok/J가 상대 허용오차 안이면 동률로
> 묶고 그 안에서 `roofline_tpot_ms`로 정렬. 가속기·`max_num_seqs`처럼 *배수*로 갈리는
> 거친 순서는 건드리지 않으므로 `floor`가 깨뜨린 fixture를 깨지 않는다. **16개 코퍼스**
> (세 개가 아니라 — E6의 두 스윕을 도착률별로 읽어 부하 축을 추가)에서 측정: 96개
> (코퍼스, K) 셀 중 **11개 개선, 0개 악화, 85개 동일**, K=20 false-infeasible이 8개
> 코퍼스에서 **2개**로. 지금은 `plan --surrogate`의 기본값이고 `roofline`은 재현용으로
> 남아 있다. 허용오차는 맞춘 값이 아니다(0.001·0.01·0.05가 동일한 곡선과 동일한 top-K
> 멤버십).
>
> **살아남은 절반:** *"`--top-k`를 일반적 비용 레버로 쓰지 않는다"* 는 여전히 유효하다.
> **20 rps에서는 효율 기반 랭커가 K=50까지 false-infeasible**이고 `floor`만 답을 찾는다 —
> 최고 부하에서는 feasibility를 TPOT floor가 전적으로 결정한다. K=5·K=10도 그대로다.
> 그리고 16개 코퍼스는 여전히 **클러스터 fixture 2개**다.
>
> 시도했다가 측정으로 기각한 것 하나: **rank fusion**(두 순서의 라운드로빈 병합, 가중치
> 없음)은 K=20에서 7개 코퍼스가 false-infeasible로 `binned`의 2개보다 나쁘다. 포기한
> 깊이가 상보성 이득보다 크다. `exp_surrogate.py --rankers`에 남겨 두었고 출하하지 않는다.
>
> `docs/deviations.md` D30, `docs/surrogate_topk_regret.md`. 아래 본문과 지시는 2026-09-08
> 당시의 기록으로 그대로 둔다.

`AnalyticalRooflineRanker`가 쓰는 proxy tok/J는 **TP·DP에 대해 대수적으로 불변**(throughput과 power가 모두 `tp·dp`에 비례해 비율이 소거) — 랭커는 가속기와 `max_num_seqs`만 보고 병렬성을 보지 못한다. 병렬성이 feasibility를 가르는 fixture(`tp4-dp1` 49.40 ms vs `tp2-dp2` 53.47 ms, SLO 50 ms)에서 `--top-k 20`은 두 fixture 중 둘에서 false-INFEASIBLE. 대안 랭커(roofline floor)는 그 둘을 고치고 **세 번째 fixture를 깨뜨린다**. 결론: **랭커를 바꾸지 않고, `--top-k`를 E6에서 쓰지 않는다.** K=30은 세 fixture에서 깨끗하지만 세 fixture로 임계값을 정하지 않는다. `PROJECT_REPORT.md` §4.7 "regret is 0 even at K=1"은 N=78·aggregated 위주 fixture 하나의 결과였다 → **부분 철회**. 부수 수정: `exp_surrogate.py`의 regret 공식(최소화 목적에서 `oracle_value > 0` 가드로 항상 `None`이던 것 → `abs()`), `--cache-dir` 재생 모드.

## 지시

1. **§4.7 superseded 표기** (A2): `docs/PROJECT_REPORT.md` §4.7 headline 위에 `> **SUPERSEDED IN PART 2026-09-08.** regret 0 at every K held on one aggregated-heavy fixture (N=78). On the three P/D + heterogeneous corpora of docs/surrogate_topk_regret.md the shipped ranker is false-infeasible at K=20 on two of three, because its proxy is TP/DP-invariant. top-K is not a cost lever for those sweeps.` 같은 블록. `docs/SLIDE_OUTLINE.md`의 같은 문장에도. `docs/CLAIMS.md` §3(Retracted)에 항목 추가, §1에는 넣지 않는다(surrogate는 기여 주장이 아님).
2. `docs/deviations.md`: 브랜치는 **D27만** 추가했다(확인). surrogate 발견은 D 번호가 없으므로 **D30 — roofline surrogate의 proxy는 TP/DP에 불변; top-K는 P/D·이종 corpus에서 비용 절감 수단이 아니다** 절을 추가하고 `docs/surrogate_topk_regret.md`를 가리킨다. (D28·D29는 이 문서의 STEP 2·4가 쓴다.)
3. `feat/rps-step1-sim-cost`를 `main`에 머지(PR #61 또는 새 PR). 머지 후 `feat/rps-step0-d26-retro`, `feat/rps-step1-sim-cost`, `feat/rps-step4-surrogate-regret` 원격 브랜치 삭제. `spike/d14-asym-tp`는 STEP 2.1 완료 시 삭제.
4. `docs/HANDOVER.md` §2.1에 "rev 2" 반영: E6a 예산과 설계 변경 한 단락.
5. 브랜치가 커밋한 `outputs/**/cache/*.json`(envelope cache 재생용 ~200개 소파일)은 유지 — 재생 근거. `.gitignore` 변경 14줄이 무엇을 풀었는지 PR에서 확인.

## 완료 조건 — 전부 이행됨 (확인 2026-09-14)
- [x] `main`에 D27 + surrogate 발견 — D27·D30 모두 `docs/deviations.md`에 있음
- [x] §4.7·SLIDE_OUTLINE·CLAIMS §3 표기 — `PROJECT_REPORT.md:237`,
      `SLIDE_OUTLINE.md:117`의 `SUPERSEDED IN PART 2026-09-08` 블록, `CLAIMS.md:319`의
      D30 항목
- [x] 원격 브랜치 정리 — `spike/d14-asym-tp`는 STEP 2.1 완료 시 삭제됨. 이후 다른
      작업지시서의 브랜치들이 새로 생겼으므로 이 조건은 당시 기준으로만 참이다.

---

# STEP 2. 비대칭 TP P/D 정식화 — `slab3d` (3일) — STEP 3과 병행

## 2.1 `serving/` sanctioned edit — D28 (스파이크 프로토타입의 정리)

1. `origin/spike/d14-asym-tp`의 `config_builder.py` 변경을 **cherry-pick하지 않고** 읽어서 다시 쓴다(스파이크는 `split2` 같은 실험용 모드와 임시 로그를 포함한다). 최종 형태:
   - cluster config 최상위 `"topology_mode": "auto" | "slab3d"`, 기본/부재 = `auto`, **`auto` 경로 코드 불변**.
   - `slab3d`: `g` = 최소 compute tp; 모든 인스턴스는 반 슬랩(`g`, colocated) 또는 전체 슬랩(prefill tp=g는 compute+sender로 2g, decode/colocated tp=2g); 반 슬랩 수 짝수; MoE/EP 인스턴스 존재 시 `ValueError`(범위 밖). dims `[g, 2, n_slabs]`, **인스턴스별 `tp_dim`은 collective 크기로 키잉** — 반 슬랩·prefill → `[T,F,F]`, decode/colocated tp=2g → `[T,T,F]`(스파이크의 "footprint로 키잉" 실수를 테스트로 고정).
   - `_resolve_dp_groups`의 균일 `local_dim` 할당을 인스턴스별로.
2. **`_FMT` 열 폭** (`serving/core/utils.py`): comm_type 열을 3-D 태그가 들어가도록 넓힌다(예: 15→24). `auto` 경로 출력은 trace **텍스트**가 바뀌지만 파싱 결과가 같아야 한다 — R1·R2 CSV byte-identical이 그 증명. 재파싱을 없애는 근본 수정은 upstream issue 초안(`docs/upstream_issues/llmservingsim-trace-column-overflow.md`)으로 남기고 여기서는 폭만.
3. 테스트: `tests/test_slab3d_config.py` — `auto` 무변경(기존 fixture들의 dims·tp_dim 스냅샷), `slab3d`의 dims·tp_dim 표(§1.2 of `WORK_ORDER_spikes.md`), 홀수 반 슬랩·MoE 거부; R1·R2·**R3** byte-identical(시뮬레이터 필요, PR 본문에 기록).

## 2.2 calibration domain — dim-1 `link_latency` 보정 계수

스파이크는 `[4,2]`·bw 16에서 4×를 얻었다. **최소 6점**을 측정한다(각 20 req, 단일 인스턴스, tp8 또는 tp4 colocated, flat vs split):

| 분할 | `link_bw` | 대상 |
| --- | ---: | --- |
| `[4,2]` (tp8) | 16 | RNGD tp8 — 스파이크 재현(회귀) |
| `[4,2]` | 35.2 | P/D fixture의 값 |
| `[4,2]` | 100 | NVLink급 (A40 페어 상한) |
| `[2,2]` (tp4) | 16 / 35.2 / 100 | `A40 tp2 prefill + tp4 decode` 형태를 위해 |

각 점에서 dim-1 latency 계수 {1,2,4,8}× 스윕 → flat 대비 TPOT p50 차이 최소인 계수와 그 잔차. 결과를 `profiles/calibration/slab3d_latency.yaml`에 `{split, link_bw, factor, residual_pct, fitted_on: <sha>}` 리스트 + `validity`(측정한 (split, bw) 조합만; 밖은 `refuse`)로 저장. 계수가 (split, bw)에 따라 달라지면 **법칙을 세우지 말고 표를 쓴다**. 문서: `docs/slab3d_calibration.md`.

## 2.3 planner 측

1. `planner/candidate_generator.py::_pd_candidates`: `tp_p == tp_d` → `tp_d ∈ {tp_p, 2·tp_p}`. `2·tp_p`일 때 후보에 `topology_mode: slab3d` 표시. **`--enable-pd`가 기본 False이므로 golden 불변.**
2. 컴파일러(`planner/predictor/llmservingsim.py`): `slab3d` 후보에 `topology_mode`와 **dim별 `link_latency` 리스트**(dim 1 = 스칼라 × calibration 계수)를 emit. 계수가 `validity` 밖이면 emit하지 않고 후보를 `RejectionStage` 새 범주 `OUTSIDE_CALIBRATION_DOMAIN`으로 기록(§STEP 4의 envelope 거부와 같은 범주 계열 — 인식론적).
3. `tests/test_mixed.py`의 균일성 단언 갱신; oracle-agreement 테스트(`--enable-pd` on, 작은 합성 cluster)에 비대칭 후보 포함; 재현성 테스트.
4. `docs/deviations.md` D14·D16(b)·(c)에 "Lifted <날짜> for tp_d ∈ {tp_p, 2tp_p} (D28)" 블록. D16(b) fixture의 size-4 bridging 우회는 유지하되 "더 이상 필수 아님" 표기.

## 완료 조건
- [ ] D28 머지, R1·R2·R3 동일, `spike/d14-asym-tp` 브랜치 삭제(내용이 정식화됨을 PR에 명시)
- [ ] `slab3d_latency.yaml` 6점 이상 + validity, `docs/slab3d_calibration.md`
- [ ] `plan --enable-pd`가 `pd-asym-a40tp4-rngdtp8` fixture에서 비대칭 후보를 열거·시뮬레이션·랭킹(20 req 1회, 결과는 정합성만 — 정확도 주장 아님)
- [ ] 게이트 통과, golden 불변

---

# STEP 3. 측정 — RNGD 저부하 envelope + 전력 (2일, NPU 노드) — **STEP 1.5 직후 바로 시작, STEP 2와 병행**

> rev 2 — STEP 0이 확인한 노드 사실을 반영: 이용률은 `furiosa-smi status --format json`(`pe_utilizations` PE별, DRAM `used_ratio`)에 있고 `info`에는 없다 → **샘플러는 두 서브커맨드를 같은 틱에서 호출**하고 두 타임스탬프 차를 기록한다. 전력은 **1 W 양자화** → 저부하 점의 반복 편차 판정 임계 5 %는 idle 근처에서 약 2 quanta; 벤치 구간이 60 s 이상이어야 평균이 의미를 갖는다. 장치는 **`device_sn`·`pci_bdf`로 식별**(dev_name은 재열거로 바뀜 — 9/4→9/7 사이 `npu0/1/3`→`npu0/1/2`). card-as-device 점은 PE별 이용률의 **평균과 분산**을 모두 기록 — 일부 PE만 바쁜 카드와 균일 부하 카드는 평균이 같아도 다른 운영점이다.


## 3.1 harness

1. `experiments/scripts/power_sampler.sh` — upstream 스크립트의 개선판: 매 틱 `furiosa-smi info --format json`(전력)과 `furiosa-smi status --format json`(PE별 이용률, DRAM used_ratio)을 **정확한 1 Hz**(`sleep`이 아니라 다음 초 경계까지 대기, 드리프트 방지)로 폴링, 타임스탬프 `date +%s.%N`, 출력 CSV `ts_info, ts_status, device_sn, pci_bdf, dev_name, power_w, util_mean_pct, util_min_pct, util_max_pct, dram_used_ratio, <원시 json 두 줄>`. 시작 시 `furiosa-smi` 버전과 필드 목록을 헤더로. 시리얼·UUID는 산출물 커밋 시 **redact**(STEP 0의 관례).
2. `experiments/scripts/measure_envelope.py` — 설계서 §7의 vendor-agnostic core. 입력: 엔드포인트, 동시성 목록, 풀 크기(기본 `4 × max(conc)`, 최소 300), 데이터셋(D22와 **같은** sharegpt 트레이스). 각 점마다: 샘플러 시작 → 45 s settle(idle 구간 기록) → 벤치 → 60 s 후행 idle → 샘플러 정지. 출력 JSON: served conc, requested, `served/requested`, tput, TTFT/TPOT p50/p95/p99, 벤치 구간의 **전력 평균·p5·p95와 util 평균**, idle 평균. `rebuild_rngd_bundle_from_edf.py collect`의 벤치 호출부를 재사용(조사 필요: 그 스크립트의 `bench_python=/usr/bin/python3` 의존 — FuriosaAI 런타임은 system python).
3. 테스트(시뮬레이터·하드웨어 불필요): 가짜 벤치 결과+가짜 샘플러 CSV에서 served 동시성·전력 구간 평균이 맞게 계산되는지; `served/requested < 0.9`면 그 점을 `pool_binding: true`로 표시하는지.

## 3.2 측정

RNGD 카드 1장(card-as-device, TP=8 내부), Llama-3.1-8B, D22와 같은 아티팩트·데이터셋:

- 동시성 **1, 2, 4, 8, 16**(16은 D22의 15.3점과의 이음새 확인용), 풀 ≥ 300.
- 각 점 **2회 반복**(독립 프로세스), 편차 기록. 편차 5 % 초과면 3회. 전력 양자화(1 W)를 감안해 **벤치 구간 ≥ 60 s**가 되도록 풀 크기를 조정(c1은 요청 300개로 수 분이 걸리므로 자연히 충족; c16은 확인).
- idle: 서버 기동 후 45 s settle → 60 s 평균. **standby**(모델 로드됨, 요청 없음)도 같은 방법 — 현 프로파일의 `standby_power 265 W`가 맞는지.
- 결과: `outputs/rngd_envelope_lowload/{bench_c*.json, power_c*.csv, idle.csv}` 커밋.

## 3.3 산출물

1. `profiles/envelopes/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/tp1.yaml` — 설계서 §2 스키마, **D22의 4점 + 이번 5점**, `power_w`·`util_pct` 채움(D22 점들은 `power_w: null` 유지 — 재측정하지 않은 것을 채우지 않는다, A2). `validity.conc_min`이 1.x로 내려간다.
2. `furiosa_rngd_card.yaml`의 `power:` 블록 옆에 설계서 §3 `power_model`(piecewise in util) 추가 — 기존 스칼라는 유지(back-compat), `source: measured`, 원시 경로.
3. `experiments/results/rngd_lowload_envelope.md`: 표, 곡선 그림(tput·TPOT·power·**tokens/J** vs served conc), 단봉 여부의 **관찰**(가설 확인/반증), D22 곡선과의 이음새.
4. **tokens/J(RPS) 측정 곡선**: 각 점의 `tput / power_w`와, 그 점이 대응하는 RPS(`= served_conc / W`, W = 평균 요청 지연). 이것이 E6b의 RNGD 측 입력이다.

## 완료 조건
- [ ] 9점 envelope YAML(served, validity, closed_loop, measured_on_workload 필수 필드)
- [ ] power_model 측정치, idle/standby 재확인
- [ ] 결과 문서 + 원시 산출물 커밋

---

# STEP 4. planner — envelope, accuracy domain, RPS 축 (5일)

## 4.1 `planner/perf_envelope.py` (신규)

설계서 §2·§4. pydantic 스키마(`concurrency_metric: served`와 `validity` 필수 — 없으면 로드 거부), 로더, 보간(log-linear), `solve_operating_point(env, rps_per_instance, out_tokens) -> OperatingPoint | Saturated` — `Saturated`는 `genuine` / `unmeasured` 두 종류를 구분하는 필드를 갖는다. 이름 충돌 주의: `planner/envelope.py`는 시뮬레이션 결과 캐시.

테스트(설계서 §9): 라운드트립 <1 %, 외삽 거부, RPS 단조성, `Saturated` 두 종류.

## 4.2 `accuracy_domain` (`planner/predictor/calibration.py`)

`profiles/calibration/rngd_card_edf.yaml`에 설계서 §5 블록 추가 — 점: (16.6, tpot −3.1, tput −3.1), (76, tpot −18, tput +31); STEP 3에서 저부하 점의 시뮬레이터 오차도 측정 가능하면(해당 동시성의 20-req 시뮬레이션 vs 측정) 추가. `tpot_error_at(conc)`: 점 사이 선형, **밖은 `outside_domain` 정책**(기본 `widen_error_bars` = 가장 가까운 점의 오차에 |기울기|×거리 가산, 상한 없음 — 보수적). A40 calibration에는 현재 bucket 오차(~2 %)를 단일 점으로 넣고 `fitted_at_concurrency`를 그 validation run의 served conc로(조사 필요: `docs/a40_sim_vs_real_plan.md`의 run에서 계산).

## 4.3 운영점 기반 margin (`planner/optimizer/feasibility.py`, `exhaustive.py`)

1. `planner/util/operating_point.py`: 시뮬레이션 CSV → 인스턴스별 served 동시성(Little's law), 인스턴스→하드웨어 매핑으로 **하드웨어별 운영점**.
2. `exhaustive.search()`: 후보의 SimResult에 운영점을 붙이고, 하드웨어별 calibration의 `accuracy_domain`에서 margin을 읽어 feasibility에 전달. **P/D 후보는 prefill 측 하드웨어의 TTFT 오차, decode 측의 TPOT 오차**를 각각. 수동 `--tpot-margin-percent`는 유지하되 둘이 동시에 주어지면 **큰 쪽**을 쓰고 provenance에 둘 다 기록.
3. `accuracy_domain`이 없는 하드웨어 → margin 0 + `caveat`(현행 동작, byte-identical). 있는 하드웨어에만 자동 적용 → **golden(A40 only, calibration에 domain 없음) 불변** — 단 4.2에서 A40에 점을 넣으면 golden이 바뀐다. **결정**: A40 domain은 별도 파일 `profiles/calibration/a40.accuracy.yaml`로 두고 `plan --accuracy-domain` 플래그로 opt-in. 기본 경로 불변.
4. PlannerOutput에 후보별 `operating_point`, `applied_margin`, 출처 기록.

## 4.4 envelope 사전 거부 — 새 `RejectionStage`

`OUTSIDE_MEASURED_ENVELOPE`: 후보의 예상 운영점(4.1 solver, RPS/replicas)이 그 하드웨어 envelope의 `validity` 밖이고 정책이 `refuse`면 **시뮬레이션 전에** 거부. envelope 없는 하드웨어에는 적용 안 함. **oracle-agreement**: 이 단계 on/off로 최적해가 같아야 한다 — 같지 않으면 이 단계가 최적해를 잘랐다는 뜻이고, 그건 "측정 범위 밖의 최적해"라는 **결과**이므로 테스트는 두 경우를 구분해 보고한다(거부된 후보가 오라클 최적해였는지 `unscored`에 명시).

## 4.5 `--rps` 스윕과 switchover

`python -m planner plan --rps 1,2,3,5,10,20 …`: RPS마다 plan을 돌리고(`--cache-dir` 공유 — 후보·knobs가 같고 트레이스만 다르므로 캐시 키에 RPS 포함 여부 조사 필요), `PlannerOutput.switchover`(설계서 §6 표: RPS, 추천, 백엔드, 가속기 수, 운영점, tok/J, 평균 W, **validity 라벨**)와 `crossovers`(인접 RPS에서 승자 백엔드가 바뀌는 지점; tok/J 곡선 교차는 선형 보간으로 추정하고 "estimated" 라벨). 렌더러·YAML 출력. 재현성 테스트(같은 seed 두 번 → byte-identical).

## 4.6 power_model 소비

`inventory.py`: `power_model` 블록 파싱(옵션). 현재 에너지 계산은 시뮬레이터의 node-level 전력(§4.8.7 "~558 W는 node 전력")에서 오므로, `power_model`은 **E6b의 측정 곡선 계산에만** 쓴다. planner의 에너지 정의를 바꾸지 않는다(정의 변경은 모든 과거 tok/J와의 비교를 끊는다 — 하지 않음, 이유 기록).

## 완료 조건
- [ ] 4.1~4.6 각 테스트 + oracle-agreement(envelope 단계 on/off, 비대칭 P/D 포함) + 재현성 + golden 불변
- [ ] `plan --rps` 출력 예시를 `docs/`에
- [ ] D29: "accuracy domain과 운영점 기반 margin" 결정 기록(수동 margin과의 관계, A40 opt-in 이유)

---

# STEP 5. 검증 실험 (4일, 시뮬레이션 대부분 백그라운드 — E6a ~10–20 h, rev 2 설계)

## E5 — planner가 D22의 infeasible plan을 스스로 거부하는가

- 입력: `pd-rngd-gpu-card.yaml`, TTFT ≤ 64 s, **수동 margin 없이**, `--accuracy-domain` on. 캐시: 재검증의 `outputs/.hp-reval-margin18-*`(같은 후보·knobs·seed면 시뮬레이션 재사용 — 캐시 키 확인).
- 기대: committed winner `hp-00323`(`agg[furiosa:tp1]` n=2, 예측 TPOT 48.41)의 운영점 ≈ 76 → domain에서 −18 % → robust 57.1 > 50 → `SLO_VIOLATED`, `applied_margin` 18 ± 보간 오차. 최종 추천 = `agg[cuda:tp4]` 2.595.
- **회귀 테스트로 고정**: `tests/test_e5_self_rejection.py`가 committed 시뮬레이션 레코드(JSON)를 mock predictor로 먹여 같은 판정을 내는지 — 시뮬레이터 불필요, CI에서 영구 보호.
- 결과 문서 `experiments/results/e5_self_rejection.md`. 실패하면(예: 운영점이 76이 아니거나 margin이 3.3 % 미만) 그것이 결과 — 왜인지 적고 4.2·4.3을 고치되 **결과에 맞추기 위해 domain 점을 조정하지 않는다**.

## E6 — RPS 축 스윕: crossover는 존재하는가

- **E6a (시뮬레이션) — rev 2 설계.** fixture `pd-rngd-gpu-card`, `pd-rngd-gpu`, STEP 2의 비대칭 fixture; **RPS {1, 3.3, 10, 20}**; TTFT SLO 두 점(8 s, 64 s); 300 req; `--accuracy-domain`, `--enable-pd`; **`--top-k` 사용 금지**(STEP 1.5); **64 워커**(96코어 노드, `read_wait` 54 %는 자식 프로세스의 CPU — 32→64에서 CPU load가 ~2배로 오르는지 첫 지점에서 확인하고 아니면 48로).
  - **구조별 knob 고정(heuristic, 라벨 필수).** 후보 수를 1/6로 줄이는 유일한 정당한 수단. 규칙: `outputs/.hp-reval-*`(10 rps 전수 결과)에서 `(fixture, arch, backend_mix)`별로 **10 rps에서 feasible이었고 tok/J가 가장 높은 knob 조합** `(max_num_seqs, max_num_batched_tokens)` 하나를 고른다. **10 rps에서 feasible 후보가 없던 조합(모든 RNGD 관련 구성)은 knob 6개를 전부 유지** — 저RPS에서 어느 knob이 살아나는지가 바로 E6의 질문이므로 고정할 근거가 없다. 고정 규칙과 고정된 knob 표를 `experiments/results/e6_rps_sweep.md` §Method에 적고, 결과 표의 cuda 행에 `knob: fixed@10rps` 라벨을 단다. 구현: `pd_slo_sweep.py`(또는 `plan`)에 `--knob-policy fixed-from:<reval json>` 옵션 — `planner/`가 아니라 `experiments/scripts/`에.
  - **시간 추정을 먼저 적고 실측과 병기.** STEP 1 실측: 10 rps 전수 지점 ≈ 65 min(D27 후, 32 워커). knob 고정으로 cuda 후보 1/6, RNGD 후보 유지 → 후보 ≈ 40 % → ~26 min; 64 워커 → ~14 min; RPS 가중 합 10.3 + 2.6 + 1 + ~0.7(20 rps, **첫 지점에서 실측해 채움**) ≈ 14.6 → ~3.4 h per (fixture, TTFT); × 3 fixture × 2 TTFT ≈ **~20 h → 캐시 공유로 TTFT 두 점이 같은 시뮬레이션을 재사용하면 ~10 h**(TTFT SLO는 feasibility 판정에만 쓰이고 시뮬레이션 입력이 아니다 — **확인됨**: `planner/envelope.py`의 키는 `placement, scheduler_config_hash, network_class, workload_bucket` + trace digest이고 SLO는 없다. RPS는 트레이스 도착 시각을 바꾸므로 digest가 달라져 RPS별로는 올바르게 분리된다). 하룻밤. 24 h를 넘기면 1 rps 지점의 RNGD knob를 STEP 3 envelope이 가리키는 운영점 근처 2개로 줄이고 그 사실을 적는다 — 1 rps 자체는 버리지 않는다(P1이 사는 곳).
  - 저RPS에서 시뮬레이션 시간이 길어지는 것(drain)은 STEP 0-1이 확인한 범위에서 허용; 타임아웃은 D25-b 이후 진짜 타임아웃만 남으므로 그 수를 기록. `nohup`/`tmux`로 detach.
  - 산출: switchover 표 × 조건, crossover 목록(있으면), **RNGD가 어느 RPS에서 처음 SLO를 통과하는지와 그때의 knob**, 비대칭 P/D가 어느 조건에서 순위 몇 위인지.
- **E6b (측정 대조, 시뮬레이터 독립)**: STEP 3의 RNGD 측정 곡선에서 tokens/J(RPS)를 직접 계산하고, E6a의 RNGD 후보 예측치(같은 운영점)와 나란히. 차이가 `accuracy_domain`의 선언 범위 안인지. A40 측은 E6a의 시뮬레이션만 있으므로 crossover 문장은 "RNGD **측정** 대 A40 **시뮬레이션(±2 %)**"로 쓴다.
- 결과 문서 `experiments/results/e6_rps_sweep.md` + 그림(tok/J vs RPS, 백엔드별, validity 음영). 세 결론 중 하나: (i) crossover 존재 — RPS 값과 양쪽 라벨, (ii) 측정 범위 안에 없음 — RNGD가 SLO를 통과하는 RPS 구간과 그때의 tok/J 격차, (iii) 비대칭 P/D가 어느 조건에서 이김/못 이김.

## E7 (선택, 하루 이내면) — crossover의 envelope 점 수 민감도

envelope에서 점을 하나씩 빼고(leave-one-out) E6b의 crossover RPS가 얼마나 움직이는지 — D11이 프로파일 그리드 밀도를 2.2 pp로 정량화한 방식. crossover가 없으면 생략.

## 완료 조건
- [ ] E5 회귀 테스트 + 결과 문서
- [ ] E6a/E6b 결과 문서, 산출물 JSON 커밋, 시간 예측/실측 병기, **knob 고정 표와 `knob: fixed@10rps` 라벨**, 1 rps 지점의 knob-고정 regret
- [ ] 결론이 (i)/(ii)/(iii) 중 무엇인지 첫 문단에

---

# STEP 6. 문서와 논문 뼈대 (2일)

1. `docs/CLAIMS.md` 갱신 — §1에 E5·E6·`slab3d`·저부하 envelope 행, §2 첫 문장("no heterogeneous configuration is shown to win")을 E6 결과대로 유지/수정, §3에 STEP 0의 D26 소급 결과.
2. `docs/PROJECT_REPORT.md` §4.10 "RPS-aware selection" 절, §6 remains 갱신.
3. `docs/deviations.md` D27~D29 + D14/D16/D22 갱신, Open items summary.
4. `docs/HANDOVER.md` 재작성(기준 sha, 게이트 수, 다음 일).
5. **`docs/PAPER_OUTLINE.md`** — 논문 뼈대: 문제(D22: 오차는 운영점의 함수, 스칼라 프로파일은 infeasible plan을 추천) → 해법(accuracy domain + envelope + RPS 축; 비대칭 P/D 표현) → 일반화(Tier 0/1) → 평가(E1~E6, 3-regime, retraction 규율) → 한계(ATOM D20, A40 envelope 미측정, 단일 워크로드). 각 절에 인용할 산출물 경로. 이것이 다음 대화의 입력이다.

---

# 7. 전체 완료 조건

- [ ] STEP 0~6 개별 완료 조건
- [ ] `serving/` 변경은 D27(조건부)·D28만, 각 R1·R2(·R3) byte-identical 증명
- [ ] `plan --rps … --accuracy-domain --enable-pd`가 switchover 표와 crossover(또는 부재)를 낸다
- [ ] E5 회귀 테스트가 CI에 있다
- [ ] E6 결론이 CLAIMS.md 첫 페이지에 있다
- [ ] §4.7 top-K regret 주장의 부분 철회가 PROJECT_REPORT·SLIDE_OUTLINE·CLAIMS §3에 있다 (STEP 1.5)
- [ ] 게이트 통과, golden 불변, oracle-agreement·재현성 통과

# 8. 리스크 대응 규칙

| 상황 | 판단 |
| --- | --- |
| STEP 0-1에서 3.3 rps RNGD P/D가 D26 이후에도 종료하지 않음 | 새 결함. `livelock_watch.sh` exit code로 분류, 새 D 번호. E6의 RPS 하한을 종료하는 값으로 올리고 그 사실을 결과에 적는다 — 조용히 축을 줄이지 않는다 |
| STEP 0-2에서 Exp 5 값이 다름 | D26 소급 오염. 해당 문서 superseded 표기, CLAIMS.md §3에 항목, 나머지 8/25–31 다중 인스턴스 결과 목록화 후 재실행 여부 사용자 결정 |
| ~~STEP 1 프로파일에서 chakra subprocess가 30 % 미만~~ | rev 2: 발생했고(28.9 %) 저RPS 배수를 근거로 D27을 시행 — 승인됨. 행은 기록용 |
| 2.2에서 계수가 (split, bw)마다 크게 다름 | 표로 기록, validity를 측정 조합으로 한정. 법칙을 만들지 않는다 |
| 2.2에서 어떤 계수로도 잔차가 3.1 % 이상 | `slab3d`는 "순위용"으로만 라벨, 비대칭 P/D의 절대 tok/J는 인용 금지. E6 결과 문서에 그 라벨을 전파 |
| STEP 3에서 `furiosa-smi`에 이용률 필드가 없음 | 전력만 기록하되 `util_pct: null`로 두고, 대신 벤치의 served 동시성을 부하 지표로 병기. A5(c)의 "이용률 없는 전력은 측정이 아니다"는 **완화하지 않고** 결과 문서에 한계로 명시 |
| STEP 3에서 저부하 점의 반복 편차 > 5 % | 3회로 늘리고 중앙값·범위 기록. 편차의 원인(전력 양자화 1 W, 짧은 벤치)을 적는다 |
| E5에서 운영점이 76이 아님 | 운영점 계산(`Σlatency/wall`, 인스턴스별)을 D22 문서의 정의와 대조. 정의 차이면 문서를 고치는 것이 아니라 코드를 D22 정의에 맞춘다 |
| E5에서 margin이 3.3 % 미만이라 winner가 통과 | 결과다. `accuracy_domain`의 보간이 D22 측정과 어떻게 다른지 적고, **domain 점을 옮기지 않는다** |
| E6에서 crossover 없음 | 결과다 — (ii). RNGD가 SLO를 통과하는 RPS와 격차를 그대로 보고. 논문은 negative result + 방법론 |
| E6에서 비대칭 P/D가 이김 | 2.2의 라벨을 확인하고 "순위 주장" 수준인지 "절대 수치" 수준인지 명시. 이기는 이유(decode KV 용량 tp8 246k vs tp4 62k, D16(c))가 결과에서 보이는지 확인 |
| E6a 총 시간 > 24 h | `--top-k`는 쓰지 않는다(STEP 1.5). 1 rps 지점의 RNGD knob를 envelope 운영점 근처 2개로 줄이고 기록. RPS 축·300 req·1 rps 지점은 유지 |
| 64 워커에서 CPU load가 32 워커의 ~2배로 오르지 않음 | 메모리·I/O 병목. 48로 내리고 `docs/sim_cost_profile.md`에 스케일링 표 추가 |
| knob 고정이 cuda의 저RPS 최적을 놓쳤을 가능성 | 10 rps 승자 knob 외의 후보가 1 rps에서 이길 수 있다. E6 결과 문서에 heuristic임을 명시하고, **1 rps 지점 하나에서만** cuda 전 knob를 돌려 고정 규칙의 regret을 측정해 표로 (추가 ~1 h) |
| oracle-agreement가 envelope 단계 on/off에서 갈림 | 잘린 후보가 오라클 최적해면 "최적해가 측정 범위 밖"이라는 결과. 테스트는 이를 실패가 아닌 **명시적 보고**로 분류하되, 그 외 이유의 불일치는 실패 |
| A40 노드에 접근 가능해짐 | STEP 3b: A40 tp4 envelope + 전력(`nvidia-smi --query-gpu=power.draw,utilization.gpu`, 1 Hz) 같은 프로토콜로 측정하고 E6b의 A40 측을 측정으로 교체. 별도 PR, 사용자 승인 후 |
| upstream 파일 수정이 필요해 보임 | 중단·보고 |
