# HeteroPilot 작업지시서 — 스파이크: NPU 실행 모델이 시뮬레이터 오차의 얼마를 설명하는가

> 질문: RNGD 카드의 accuracy domain이 보이는 오차 곡선(TPOT +11 % at c2 → 0 at c16 → −18 % at c76; TTFT는 D17·`rngd_sim_vs_real_summary.md`의 −32.6 %)은 **LLMServingSim이 vLLM 스케줄러를 모사하고 Furiosa-LLM은 정적 bucket AOT 실행기라는 구조 차이**로 얼마나 설명되는가. 그 답이 "대부분"이면 시뮬레이터에 opt-in **실행 모델**을 넣는 후속 작업지시서를 쓰고, "일부"면 그 일부만 모델로 흡수하고 나머지는 accuracy domain에 남긴다.
> 이 문서는 **분해(decomposition)** 가 목적이다. 실행 모델을 완성하는 것은 후속 작업지시서의 일이다.
> 대상 저장소: `github.com/swsok/heteropilot` · 기준 `main` = `fcf8ba5` 이후(UQ Stage B+ 머지 후의 `main`에서 시작할 것 — 시작 시 sha를 §STEP 0에 기록) · 작성일: 2026-09-15 · 도구: Claude Code CLI · 노드: 시뮬레이션은 아무 노드, STEP C는 **NPU 노드**
> 예산: **5일** (STEP 0 0.5 · A 1.5 · B 2 · C 1(NPU) · D 0.5). STEP C는 노드 일정에 맞춰 A·B와 병행 가능.
> D-번호: 이 작업지시서는 **D90–D99 블록**을 쓴다. 초안은 D70–D79라고 적었으나 그 블록은 `WORK_ORDER_uq_stage_b_plus.md`가 2026-09-14 STEP C0에서 claim했고 D70–D73이 이미 기록되어 있다(D80–D89는 작업지시서 없는 일회성 작업 몫이므로 비어 있어도 쓰지 않는다). 블록과 실험 id 태그 **`E-N*`** 는 CLAUDE.md 표에 **이미 등록되어 있다**(2026-09-15) — STEP 0에서 다시 추가하지 말고 존재만 확인할 것. **블록 안의 개별 번호는 이 문서에 미리 적지 않는다**: `tests/test_deviations_numbering.py`가 아직 항목이 없는 `D<n>` 인용을 실패로 잡으므로, 번호는 항목을 쓰는 커밋에서 정한다.

---

## 0. 이 문서의 사용법

1. **STEP A → B → D 순서. C는 NPU 노드가 잡히는 대로 병행.** B는 A의 분해 결과로 프로토타입 범위를 정하므로 A가 먼저다.
2. **한 STEP = 한 브랜치 = 한 PR.** `spike/npu-exec-a-decompose`, `spike/npu-exec-b-prototype`, `spike/npu-exec-c-hardware`, `spike/npu-exec-d-memo`. B의 PR은 **"spike — do not merge"** 라벨로 리뷰 기록만 남긴다(D14 스파이크와 같은 규칙).
3. **모든 실험은 확인 / 반증 / 미결 중 하나로 끝난다.** "아마"는 결론이 아니다. 오차의 기여도는 **퍼센트포인트**로 적는다("c76의 −18 % 중 −11 pp는 KV-bucket 그룹 attention").
4. **결과를 좋게 보이려고 조건을 바꾸지 않는다.** 특히 §B.3의 hold-out 워크로드는 프로토타입을 만든 뒤에 처음 본다.
5. 상위 문서 우선: `WORK_ORDER_heteropilot.md` → `docs/deviations.md` → `CLAUDE.md`.

### 절대 규칙 (재확인 + 이 문서의 추가)

- **A1.** `scripts/whichnode.sh`가 나열하지 않는 하드웨어의 결과를 주장하지 않는다. STEP C만 하드웨어를 만진다.
- **A2.** 철회는 공개적으로. D17의 "혼합 스텝 여부는 결정 불가" 문장은 이 스파이크가 뒤집는다 — 덮어쓰지 않고 위에 얹는다.
- **A3 — `serving/` 편집은 STEP B의 프로토타입 하나, 스파이크 브랜치 안에서만.** `main`에 머지하지 않는다. 머지 가능한 형태(opt-in, 기본 byte-identical, R1·R2 앵커, 테스트)는 후속 작업지시서의 일이다. STEP A·C·D는 `serving/`·`planner/`를 건드리지 않는다(분석 스크립트는 `experiments/scripts/`).
- **A4.** golden 회귀 출력 불변.
- **A5 — 측정 규율**(`WORK_ORDER_rps_aware.md` A5 그대로): served 동시성, 풀 ≥ 4×, sustained, 같은 워크로드.
- **A7 — bucket 격자는 하드웨어가 아니라 아티팩트의 속성이다.** 세 아티팩트가 두 스키마·다른 격자를 쓴다(FP8은 10/3/27). 격자는 `profiles/accelerators/*.yaml`이 아니라 **그 격자로 측정된 perf bundle의 `meta.yaml` 옆**(`profiler/perf/RNGD-CARD/…/bf16/`)에 둔다. 다른 아티팩트로 서빙하면 격자도, 실행 모델의 입력도 바뀐다.

---

## 1. 사전 조사 — 이미 확인된 사실 (2026-09-14/15, NPU 노드 + 커밋된 로그)

### 1.1 아티팩트의 bucket 격자 (`d6ae6a43`, tp=8, furiosa-llm `b62dbc1`)

`artifact.json` → `model.pipeline_metadata_list[i].attention_buckets`, 분류 규칙은 `furiosa_llm/metadata/config_types.py`: `input_ids = attention − kv`, prefill `kv==0`, decode `kv>0 ∧ input_ids==1`, extend `kv>0 ∧ input_ids>1`.

| 종류 | 수 | 형상 |
| --- | ---: | --- |
| prefill | 8 | **bs=1**, attn ∈ {128, 256, …, 1024}(128 스텝), kv=0 |
| extend (chunked prefill) | 74 | **bs=1**, chunk ∈ {128…1024} × attn ≤ 131072 |
| decode | 46 | input_ids=1, (bs, attn): 1→…131072, 2→131072, 4→65536, 8→32768, 16→16384, 32→8192, 64/128→4096, 256→2048 |

파생 사실 세 가지. **(i) 혼합 스텝은 컴파일되어 있지 않다** — prefill·extend가 전부 bs=1이고 decode는 input_ids=1이므로 한 forward는 "한 요청의 prefill/extend chunk" 또는 "decode 배치"다. D17이 "트레이스에 타임스탬프가 없어 결정 불가"라 남긴 질문은 격자가 닫는다. **(ii) decode 격자는 KV 예산이다** — `bs × attn`이 131072~524288. sharegpt(KV ≈ 2200)에서 bs 256 bucket(attn 2048)은 쓸 수 없고 실질 최대 decode 배치는 **128**. E6가 RNGD 후보에 준 `max_num_seqs 256`은 실행 불가능한 설정이었다. **(iii) 패딩은 두 축** — 배치는 다음 2의 거듭제곱, 컨텍스트는 그 bs의 attn bucket.

composed 파이프라인 1–46이 decode bucket과 1:1이고, **pipeline 0은 kernelwise**(composable IR)로 **attention bucket 128개 × tokenwise 12개(input_size 1, 2, 4, …, 1024)** 메뉴를 갖는다. 131072 / 128 = **1024** → kernelwise 경로의 attention bucket 폭은 1024 토큰.

### 1.2 어느 경로를 타는가 — `Wire pipeline hit rate` (커밋된 serve 로그, first/last/min/max)

| 동시성 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wire(composed) hit % (last) | 99.8 | 47.5 | 12.0 | 1.8 | 0.5 | 0.0 | 0.0 | 0.0 |

반복런 0.2 pp 이내. **결론: 계획에 중요한 모든 부하(c8 이상)에서 런타임은 kernelwise 경로다.** composed는 사실상 c1 전용. D17의 per-layer EDF 트레이스(c16–c32)가 kernelwise 비용이라는 것과 정합. `wire_hit_rate`는 네이티브 게이지라 윈도우가 문서화되어 있지 않다 — 평균 내지 말고 last/min/max만 쓴다(조사 결과 그대로).

**hit rate 붕괴의 원인은 미결 — 두 가설.** (H-a, CLI) running 배치 크기가 흔들려 bs가 정확히 맞는 일이 드물다. (H-b) composed는 `(bs, attn)` **하나**의 attention bucket으로 컴파일되어 배치 안 모든 시퀀스의 KV가 같은 bucket일 때만 쓸 수 있고, 섞이면 그룹 attention이 가능한 kernelwise로 떨어진다. c2 47 % ≈ 둘이 같은 bucket일 확률, c4 12 % ≈ 넷이 같을 확률이라는 점에서 H-b가 숫자에 더 맞는다. STEP C.1이 판정한다.

### 1.3 D17이 이미 측정해 둔 것 (`docs/deviations.md` D17)

레이어당 attention 실행 수가 배치의 KV 다양성을 따른다: 시퀀스 1.95/3.91/8.91/15.16/29.09에서 1.95/2.40/2.87/3.03/3.08회, 레이어당 88→330 µs. 지금 bundle은 이것을 **sharegpt에 보정된 total-preserving 행 하나**로 흡수한다(워크로드가 바뀌면 깨짐). batch 1은 fused `Composed` 그래프라 `tokens=1` 행은 별도 유도. **"혼합 prefill+decode 스텝이 트레이스에 없다"** — 이제 §1.1(i)로 구조적으로 확인됨.

### 1.4 시뮬레이터가 하는 것 (`serving/`)

- `scheduler.py:101-107`: `prioritize_prefill ∧ ¬chunked`이면 prefill 요청들만 모아 **`available_slots`까지 여러 개**를 한 배치로 — bs=1이 아니다. `:110-120`: chunked prefill이면 decode + prefill chunk를 `max_num_batched_tokens` 예산 안에서 **한 배치에 섞는다**(vLLM식). 즉 knob만으로는 "prefill bs=1, decode와 배타"를 정확히 재현할 수 없다 — 근사만 가능(§A.2).
- `trace_generator.py`: `_load_perf_db`(:326), `_lookup_attention`(:759) / `_attn_slice_lookup`(:561) / `_lookup_attention_with_skew`(:714), `_emit_layer`(:924). 비용은 실제 토큰 수·실제 KV로 lookup — 패딩도 그룹도 없다.
- 시뮬레이션 CSV(`instance id, request id, input, output, arrival, end_time, latency, queuing_delay, TTFT, TPOT, ITL`)와 `--log-level INFO`의 배치 로그로 **스텝별 배치 구성**을 재구성할 수 있다(조사 필요: INFO 로그가 배치의 요청 id·토큰 수를 남기는지 — `scheduler.py:296-335`의 `Scheduling new batch #%d` 주변; 없으면 `--no-cleanup-inputs`로 남긴 trace 파일의 `input_size` 열에서 복원).

### 1.5 검증 기준 — 이미 있는 것

- **9점 accuracy domain**(`profiles/calibration/rngd_card_edf.yaml`, 300 req, `--match offered`): c1.02 +2.25, c2.19 +11.0, c4.31 +10.8, c8.21 +7.33, c14.8 +3.26, c15.2 +2.47, c16.6 −3.1, c25.2 −3.28, c76 −18.0 (TPOT %). 이것이 **"분해해야 할 오차"의 정의**다.
- **9점 envelope**(`profiles/envelopes/RNGD-CARD/…/tp1.yaml`, served 1.00–107.2, 전력 포함) — throughput·TPOT·TTFT(closed-loop) 실측.
- 하네스 `experiments/scripts/lowload_sim_error.py`(`--envelope --cluster --dataset --num-reqs --match`) — 시뮬레이션을 envelope 점의 도착률로 돌려 오차를 내는 파이프라인. 프로토타입 cluster json으로 재사용.
- 시뮬레이터가 c59·c107 도착률에서 served 37.7·44.5에 머무는 것(domain 노트) — "sim의 admission 상한"이 아니라 **Little's law**(sim의 지연이 짧아 in-flight가 적음)다. 이 문서에서 그 표현을 쓰지 않는다.

---

## 2. 공통 규칙

```bash
bash scripts/whichnode.sh
export PYTHONPATH=$PWD && export PATH="$PWD/.venv/bin:$PATH"
pytest -q && ruff check . && mypy
```

**분해 표의 형식** (STEP A·B의 산출물은 전부 이 표에 채워진다):

| 운영점(served) | 실측 TPOT | sim TPOT(현행) | 오차 % | 메커니즘별 기여(pp): 스텝정책 / 배치·KV 패딩 / 그룹 attention / prefill 128패딩 / 잔차 | sim TPOT(프로토타입) | 잔여 오차 % |

TTFT는 같은 표를 별도로(closed-loop 측정이라 D19 caveat 명시, TPOT 열만 프로토콜 경계를 넘을 수 있음). 잔차 = 어떤 메커니즘으로도 설명되지 않는 부분 — **0으로 만들려 하지 말 것**, 그것이 accuracy domain에 남을 몫이다.

---

# STEP 0. 준비 (0.5일)

1. UQ Stage B+ 머지 후 `main` sha, 게이트 결과, `whichnode.sh` 출력을 `docs/npu_exec_spike.md` §0에 기록.
2. **CLI가 이미 만든 두 조사를 커밋**: `furiosa_artifact_buckets.py`를 `experiments/scripts/`로, 결과를 `experiments/results/rngd_artifact_buckets.md`(아티팩트 sha `d6ae6a43`, furiosa-llm 빌드 `b62dbc1`, 분류 규칙 출처 `config_types.py`, 세 아티팩트·두 스키마 표)와 `experiments/results/rngd_pipeline_hit_rate.md`(§1.2 표 + 두 스택의 통계 정의 차이 + H-a/H-b)로. 격자는 **A7**대로 `profiler/perf/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/artifact_buckets.yaml`에 기계가 읽는 형태로도 저장(출처·sha 헤더). prefix cache hit(7.6–9.0 %)이 서비스 spec `prefix_share_ratio: 0.07`과 맞는 것을 교차 확인으로 한 줄.
3. `CLAUDE.md`: D90–D99 블록 배정과 `E-N*` 태그 행은 2026-09-15에 이미 들어갔다. 두 행이 그대로 있는지 확인만 하고, 없으면 추가한다.
4. `deviations.md` **새 항목(블록 D90–D99의 첫 번호, 쓰는 커밋에서 확정)** — "RNGD 런타임은 c8 이상에서 kernelwise 경로만 탄다; composed(wire) 파이프라인은 c1 전용" (측정 사실, §1.2). D17의 "혼합 스텝 결정 불가" 문장 위에 "Resolved by artifact buckets 2026-09-15: 구조적으로 불가능(이 항목/§1.1)" 표기.

---

# STEP A. 코드 변경 없는 분해 (1.5일, 시뮬레이션만, 아무 노드)

## A.1 sim 스텝 재구성 — 실측과 무엇이 다른지 세기 (E-N1)

envelope 5점(c1/2/4/8/16)의 도착률로 현행 시뮬레이터를 300 req로 돌리되 `--log-level INFO --no-cleanup-inputs`로 **스텝별 배치 구성**을 남긴다(§1.4 조사 결과에 따라 로그 또는 trace 파일에서 복원). 스크립트 `experiments/scripts/npu_exec_step_census.py`가 각 run에서 다음을 센다:

- 스텝 수, 그중 **혼합 스텝**(prefill 토큰과 decode 토큰이 함께 있는 스텝)의 비율과 그 스텝들의 시간 비중
- prefill 스텝의 배치 크기 분포(bs=1인 비율)
- decode 스텝의 배치 크기 분포와 **다음 2의 거듭제곱까지의 패딩 비율**(Σ pad / Σ bs)
- decode 스텝마다 배치 안 시퀀스들의 KV를 1024 단위 bucket으로 나눈 **distinct bucket 수**의 분포(= D17의 "attention 실행 수" 예측치) — D17 측정값 1.95/2.40/2.87/3.03/3.08과 나란히
- prefill 토큰의 128 배수 패딩 비율(Σ ceil128(t)/Σ t − 1) — D17의 +10.9 %와 대조

이 표가 "시뮬레이터가 실제 런타임과 다르게 행동하는 양"의 첫 정량화다. 실측이 아니라 sim의 스텝 구조를 세는 것이므로 하드웨어 불필요.

## A.2 knob 근사 — 스텝 정책의 기여 (E-N2)

같은 5점 + c25(EDF 도착률 1.39 rps) + c76 도착률에 대해, 현행 knob(E6가 쓴 것: chunked on, s128/s256, t2048/t8192) 대신 **런타임에 가장 가까운 knob**으로 재시뮬레이션: `--no-enable-chunked-prefill --prioritize-prefill --max-num-batched-tokens 1024 --max-num-seqs 128`. §1.4대로 prefill bs=1은 강제되지 않으므로 이것은 **상한이 아니라 근사**임을 결과에 적는다. 각 점의 TPOT·TTFT 오차가 현행 대비 몇 pp 움직였는지 = "스텝 정책" 열의 1차 추정. 300 req, `lowload_sim_error.py --match offered`로 오차 계산(조사 필요: 그 스크립트가 knob을 인자로 받는지 — 안 받으면 cluster json 사본으로).

## A.3 오프라인 재청구 — 패딩·그룹·128의 기여 (E-N3)

A.1에서 남긴 스텝별 배치 구성과 현행 perf DB(`_load_perf_db`를 import해서 **읽기만**)로, 스텝 비용을 규칙별로 다시 계산한다 — 시뮬레이터를 다시 돌리지 않고 후처리:

- **R-pad**: decode 배치 bs → 다음 2의 거듭제곱 tokenwise bucket으로 올려 dense/tokenwise 비용 재lookup
- **R-attn**: attention 비용을 "n_decode·평균 KV 한 값" 대신 **KV 1024-bucket 그룹별 합**으로. 그룹 비용 함수 attn(n_group, L_bucket)은 (조사 필요) EDF 원시 CSV의 attention stage에 shape이 있으면 거기서 fit, 없으면 STEP C.3의 측정 전까지 D17의 per-execution 중앙값(86 µs @ n16)을 임시로 쓰고 **임시임을 표시**
- **R-128**: prefill 토큰을 128 배수로 올려 prefill 레이어 재lookup (TTFT 쪽)
- **R-c1**: c1 점은 composed 경로 — 현행 `tokens=1` 행 그대로(변경 없음, 대조군)

각 규칙을 **하나씩** 켜고 스텝 비용 합(≈ 요청 지연)의 변화를 pp로 기록. 스케줄링 피드백(스텝이 길어지면 배치 구성이 바뀜)은 이 방법이 못 보므로 **1차 근사**임을 표에 명시 — 그 피드백까지 보는 것이 STEP B다.

## A.4 산출물

`docs/npu_exec_spike.md` §A: 분해 표(§2 형식) 초안 — 스텝정책(A.2), 패딩·그룹·128(A.3) 열이 채워지고 잔차가 계산됨. 각 점에서 "어느 메커니즘이 지배적인가"를 한 줄로. **예측(반증 가능하도록 미리 적는다)**: 저부하(c2–c8) 비관은 R-pad가, 고부하(c25–c76) 낙관은 R-attn과 스텝 정책이 지배할 것.

## 완료 조건
- [ ] E-N1 census 표(혼합 스텝 비율, 패딩 비율, distinct bucket 분포 vs D17)
- [ ] E-N2·E-N3 기여 pp, 예측과 대조
- [ ] `serving/`·`planner/` 무변경

---

# STEP B. 프로토타입 — 스파이크 브랜치의 `execution_model: bucketed_aot` (2일)

## B.1 범위 — STEP A가 정한다

A의 표에서 **기여가 2 pp 미만인 메커니즘은 구현하지 않는다**(문서에 "무시 가능, 측정값 n pp"로). 나머지를 `serving/`에 opt-in으로 넣는다. 예상 최대 범위:

| 규칙 | 위치 | 내용 |
| --- | --- | --- |
| P1 스텝 정책 | `scheduler.py` | `execution_model == bucketed_aot`이면: prefill/extend 스텝은 **bs=1, chunk ≤ 1024(128 배수)**, decode 스텝과 배타. prefill과 decode의 교대 순서는 **knob**(`prefill_priority: strict|alternate`) — 어느 쪽이 맞는지는 C.2가 판정 |
| P2 decode 양자화 | `trace_generator.py` lookup 앞 | bs → 다음 2의 거듭제곱; 배치의 최대 KV가 그 bs의 attn bucket을 넘으면 배치를 줄임(→ 자연스러운 admission 상한 128) |
| P3 그룹 attention | `_lookup_attention` 분기 | 배치의 KV를 1024-bucket으로 묶어 그룹별 attn 합 |
| P4 c1 경로 | lookup 분기 | bs=1 decode는 `tokens=1`(fused) 행 — 현행 유지 |
| P5 KV 예약 | `memory_model` 호출부 | (조사 필요) 실측 KV max가 c1에서도 100 %인 것이 bucket 단위 예약이면 admission 계산에 반영; 아니면 제외 |

입력: `profiler/perf/RNGD-CARD/…/artifact_buckets.yaml`(STEP 0). cluster json 최상위 `"execution_model": "bucketed_aot"`(부재 = 현행, **byte-identical**).

## B.2 등가성과 재현 (E-N4)

- **R1·R2 byte-identical**(`execution_model` 부재 시) — `outputs/d23fix/evidence/anchor_after_d25_d26/SHA256SUMS` 기준.
- 9점 accuracy-domain 재실행: `lowload_sim_error.py`를 프로토타입 cluster json으로, 300 req, `--match offered`. 결과를 §2 표의 "sim TPOT(프로토타입) / 잔여 오차" 열에. **목표는 정해두지 않는다** — 잔여 오차가 얼마든 그것이 결과다. 다만 비교 기준 두 개는 미리 적는다: (a) 현행 오차 폭 [−18, +11] 대비 잔여 폭, (b) c16 근처의 **부호 반전이 사라지는가**(vLLM 모사의 지문이 없어졌는가).
- TTFT는 별도 표(closed-loop caveat). c59·c107 도착률에서 sim의 served 동시성이 실측(59.2/107.2)에 얼마나 가까워지는가 — 대기열 재현의 지표.

## B.3 hold-out — 프로토타입을 만든 뒤 처음 보는 워크로드 (E-N5)

sharegpt와 토큰 분포가 다른 워크로드 하나(`planner/util/workload.py`로 생성: 예 input p50 256 / output p50 1024 "long-output", 또는 input p50 2048 / output p50 128 "prefill-heavy" — **하나만**, 미리 고정하고 STEP C.4에서 측정)에 대해 c4·c16 두 점의 실측 vs 프로토타입 sim 오차. 이것이 "모델인가 보정인가"의 판정이다: sharegpt에서 잔여 오차가 줄고 hold-out에서 비슷하게 줄면 모델, hold-out에서 다시 벌어지면 sharegpt 보정.

## B.4 산출물

프로토타입 diff(`git diff --stat`을 PR에), E-N4·E-N5 표, `docs/npu_exec_spike.md` §B. `deviations.md` **블록의 다음 번호** — "실행 모델 프로토타입이 설명한 오차와 남긴 잔차"(숫자 표 포함). PR은 do-not-merge.

## 완료 조건
- [ ] 기본 경로 byte-identical(R1·R2)
- [ ] 9점 잔여 오차 표, 부호 반전 여부
- [ ] hold-out 2점(STEP C.4 측정 후) — C가 늦으면 "미결, 측정 대기"로 표기하고 D로 넘어간다

---

# STEP C. 하드웨어 — 세 가지 판정 실험 + hold-out 측정 (1일, NPU 노드)

전부 `measure_envelope.py` + `power_sampler.sh`(A5 프로토콜) 아래에서, 카드 1장, 아티팩트 `d6ae6a43`.

## C.1 composed 선택 규칙 — H-a vs H-b (E-N6, ~20분)

프롬프트 길이가 **전부 같은** 합성 워크로드(예: 512 토큰, 출력 128)로 c4 1회 → wire hit rate가 ~100 %면 H-b(같은 KV bucket이면 composed), 여전히 ~12 %면 H-a. 이어 같은 워크로드로 **c3** 1회 → hit ~0 %면 bs 정확 일치 필요, ~100 %면 다음 2의 거듭제곱으로 패딩. 두 결과로 composed 선택 규칙을 확정해 `rngd_pipeline_hit_rate.md`에 추가. (c1 경로의 정확한 규칙이라 우선순위는 낮지만 20분이면 끝난다.)

## C.2 스텝 교대 순서 — prefill 우선인가 (E-N7, ~30분)

c16 정상 워크로드 1회를 **Furiosa 프로파일러의 타임스탬프 모드**(조사 필요: `furiosa-llm` 또는 EDF profiler에 스텝 시작 시각을 남기는 옵션이 있는가; 없으면 `/metrics`를 100 ms로 폴링해 running/waiting 카운터의 변화로 추정)로 찍어, prefill 스텝이 대기 중일 때 decode 스텝이 끼어드는지(alternate) 아니면 prefill 큐가 빌 때까지 prefill만 도는지(strict)를 판정. B.1 P1의 knob 값이 여기서 정해진다. 판정 불가면 "미결"로 두고 프로토타입은 두 값 모두 돌려 어느 쪽이 실측에 가까운지로 **간접** 추정(그 사실을 명시).

## C.3 그룹 attention 비용 함수 (E-N8, ~1시간)

R-attn의 attn(n_group, L_bucket)을 **깨끗하게** 얻는다: 프롬프트 길이를 한 bucket 안(예: 전부 900~1000 → bucket 1)으로 맞춘 워크로드로 c4·c8·c16을 각 1회 EDF 프로파일 → 레이어당 attention 실행 1회의 비용을 (n, L=1024)에서 직접 읽는다. 두 번째 bucket(1900~2000)으로 한 번 더 → L 의존성. 결과를 bundle 옆 `attention_groups.csv`(A7)로. STEP A.3의 임시값을 이것으로 교체하고 A.3 표를 갱신.

## C.4 hold-out 워크로드 측정 (E-N5의 실측 절반, ~1시간)

B.3에서 고정한 워크로드로 c4·c16, 각 2회, 300 req. envelope YAML은 **별도 파일**(`measured_on_workload`가 다르므로 같은 파일에 섞지 않는다 — 설계서 §10).

## C.5 staircase (선택, ~40분)

c3·c5·c6·c12에서 TPOT 4점. 2의 거듭제곱 사이가 평평한 계단이면 R-pad 확정. 시간이 남을 때만.

## 완료 조건
- [ ] C.1 규칙 확정, C.2 판정(또는 미결), C.3 `attention_groups.csv`, C.4 hold-out 4 run
- [ ] 원시 로그·전력 CSV 커밋(시리얼 redact), provenance

---

# STEP D. 메모와 결정 (0.5일)

`docs/npu_exec_spike.md` 완성:

1. **분해 표 최종본**(§2 형식) — 현행 오차, 메커니즘별 pp, 프로토타입 잔여 오차, hold-out.
2. **결론 세 가지 중 하나**: (i) 실행 모델이 오차의 대부분(잔여 폭이 현행의 1/3 이하, hold-out에서도)을 설명 → 후속 작업지시서 `WORK_ORDER_npu_exec_model.md`(정식 opt-in 구현, 테스트, calibration domain 재측정, upstream PR 초안) 작성; (ii) 일부(특정 메커니즘만) → 그 메커니즘만 정식화하고 나머지는 accuracy domain에 남긴다는 결정과 이유; (iii) 설명 못 함 → 무엇이 남는지(잔차의 운영점 의존성)와 다음 가설.
3. **accuracy domain과의 관계** 한 단락: 프로토타입 아래에서 domain을 다시 측정하면 어떻게 바뀌는지(예상이 아니라 B.2의 숫자), 그리고 planner의 margin이 얼마나 줄어드는지(E5의 winner `hp-00323`이 프로토타입 sim에서는 몇 ms로 예측되고 domain이 몇 % 청구하는지 — **재계산 1회**).
4. `docs/CLAIMS.md` §2에 "시뮬레이터 오차의 n %는 vLLM-vs-bucketed 실행 모델 차이로 설명된다(스파이크, 프로토타입, 미머지)" 한 줄 — 라벨은 `sim-on-measured, prototype`. `PAPER_OUTLINE.md`의 시뮬레이터 fidelity 절에 포인터.
5. `docs/HANDOVER.md` §2에 후속 항목. 블록의 다음 번호(있으면: hold-out 결과), Open items summary.

---

# 3. 전체 완료 조건

- [ ] STEP 0 산출물 커밋(격자 yaml, 두 결과 문서, 블록 첫 deviation 항목; CLAUDE.md 블록·태그는 확인만)
- [ ] 분해 표(sharegpt 9점 + hold-out 2점) — 각 셀에 숫자 또는 "미결"
- [ ] `main`의 `serving/`·`planner/` 무변경; 프로토타입은 스파이크 브랜치에만
- [ ] 결론 (i)/(ii)/(iii) 명시, 후속 작업지시서 필요 여부 결정
- [ ] 게이트 통과, golden 불변

# 4. 리스크 대응 규칙

| 상황 | 판단 |
| --- | --- |
| A.1에서 INFO 로그가 배치 구성을 남기지 않음 | `--no-cleanup-inputs`의 trace 파일(`input_size` 열)에서 복원. 그것도 부족하면 **읽기 전용** 디버그 print를 스파이크 브랜치에만 넣고 A3 예외로 기록 |
| A.3의 그룹 비용 함수를 EDF 원시 CSV에서 못 얻음 | D17 중앙값을 임시로 쓰고 표에 `provisional` 표기, C.3 후 갱신. C.3 전에 결론을 내지 않는다 |
| A의 분해에서 어떤 메커니즘도 5 pp를 넘지 않음 | 결과다 — 결론 (iii) 쪽. B는 P1(스텝 정책)만 최소로 만들어 스케줄링 피드백이 A.3의 1차 근사를 얼마나 바꾸는지만 확인하고 닫는다 |
| B.2에서 프로토타입이 sharegpt 잔여 오차를 줄이는데 hold-out에서 벌어짐 | 보정이지 모델이 아니다. 어느 규칙이 워크로드 의존적인지(R-attn일 가능성) 적고, 그 규칙은 후속 작업지시서에서 **워크로드 독립 형태로 재설계**하기 전까지 정식화하지 않는다 |
| C.2를 판정할 도구가 없음 | 두 knob 값을 모두 돌려 간접 추정, "미결(간접)"로 표기. Furiosa에 스텝 타임스탬프 로깅 옵션을 문의하는 항목을 HANDOVER에 |
| C.1이 H-a·H-b 어느 쪽도 아님 | 관찰값을 그대로 기록. c1 경로는 프로토타입에서 현행 유지(P4)라 이 스파이크의 결론에는 영향 없음 |
| 프로토타입이 R1·R2를 깨뜨림 | 기본 경로가 바뀐 것. 등가성 실험을 진행하지 말고 고친다 |
| 결과를 좋게 만들기 위해 hold-out 워크로드를 바꾸고 싶어짐 | 하지 않는다. B.3에서 고정한 것 하나만 |
| 5일을 넘김 | 그 시점의 분해 표로 STEP D를 쓴다. 미결 셀은 미결로. 실행 모델의 완성은 어차피 후속 작업지시서의 일이다 |
