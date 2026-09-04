# HeteroPilot 작업지시서 — 진단 스파이크 2건 (D23 livelock · D14 비대칭 TP)

> 목적: `WORK_ORDER_rps_aware.md`를 쓰기 전에, 그 내용을 바꾸는 두 미지수를 확정한다.
> (1) D23 — tight-TTFT P/D 후보가 왜 livelock하는가. 이것이 풀리지 않으면 P/D에 관한 어떤 문장도 쓸 수 없다.
> (2) D14 — 비대칭 TP(`A40 tp4 prefill + RNGD tp8 decode`)가 정말 시뮬레이터 수준에서 막혀 있는가. 이 문서의 사전 조사는 "아니다"라고 답하며, 스파이크는 그것을 실행으로 확인한다.
> 대상 저장소: `github.com/swsok/heteropilot` · 기준 `main` = `08518e6` (통합 스프린트 종료) · 작성일: 2026-09-04 · 작업 도구: Claude Code CLI, NPU 노드(시뮬레이션 전용 — 하드웨어 측정 없음)
> 예산: **3일.** 스파이크는 답을 내는 것이 목표이고, 고치는 것은 다음 작업지시서의 일이다.

---

## 0. 이 문서의 사용법

1. **STEP A(D23)를 먼저 끝낸다.** STEP B의 비대칭 P/D 실행은 P/D 후보이므로 D23이 미해결이면 같은 livelock에 걸릴 수 있다. B.1~B.2(균일 구성 등가성 검증)는 D23과 독립이라 A와 병행 가능하고, B.3은 A의 결론을 기다린다.
2. **한 STEP = 한 브랜치 = 한 PR.** `spike/d23-livelock`, `spike/d14-asym-tp`. 스파이크 브랜치의 산출물은 **문서와 실험 산출물**이며, 코드 변경은 아래에 명시된 것만 허용된다.
3. **시뮬레이션은 `--num-requests 20`이 기본이다.** 300은 재현 확인 1회에만 쓴다. 각 실험은 20분을 넘기지 않는다.
4. **모든 실험 결과는 세 가지 결론 중 하나로 끝난다** — 확인 / 반증 / 미결(무엇이 부족했는가). "아마"는 결론이 아니다.
5. 상위 문서 우선: `WORK_ORDER_heteropilot.md` → `docs/deviations.md` → `CLAUDE.md`. `docs/CLAIMS.md`를 바꾸는 결과가 나오면 즉시 갱신한다.

### 절대 규칙 (재확인)

- **A1.** `scripts/whichnode.sh`가 나열하지 않는 하드웨어의 결과를 주장하지 않는다. 이 스파이크는 시뮬레이션만 한다.
- **A2.** 철회는 공개적으로 — D23·D14·D16(b)·CLAIMS.md의 문장이 바뀌면 덮어쓰지 않고 위에 얹는다.
- **A3. `serving/` 수정은 STEP B.2의 프로토타입 패치 하나만, 스파이크 브랜치 안에서만 허용**된다. `main`에 머지하지 않는다. 머지 가능한 형태(opt-in, 기본 byte-identical, 테스트)로 만드는 것은 다음 작업지시서의 STEP이다. STEP A에서 `serving/` 수정이 필요해 보이면 **수정하지 말고 원인과 제안을 D23에 기록**한다.
- **A4.** golden 회귀 출력 불변. 스파이크는 `planner/`를 건드리지 않는다.

---

## 1. 사전 조사 결과 (2026-09-04, `main`=`08518e6` 클론과 `casys-kaist/astra-sim`·`casys-kaist/chakra` 소스에서 직접 확인)

### 1.1 D23 — 알고 있는 것

`docs/deviations.md` D23과 `outputs/pd_slo_sweep_margin18/tight/retry3600_livelock_evidence.txt`:

- 후보 `P[cuda:tp4] D[cuda:tp4]`, knobs `-s256-t8192`, `--no-enable-prefix-caching`. 1080/1800/3600 s 모두 미완.
- 시뮬레이터 시계는 52,903틱 진행. `Instance[0]`(prefill) **running 1, Waiting 7→299 단조 증가**, `Instance[1]`(decode) **running 0, waiting 0**. 양쪽 메모리 **9.304 % / 9.234 %로 평평** — 3858.51 MB는 tp4 bf16 가중치 크기와 일치하므로 **KV 블록이 한 번도 할당되지 않았다.** 즉 "running 1"인 요청은 토큰을 한 개도 계산하지 못하고 있다.
- 같은 후보가 `outputs/.hp-pd-slo/`에 280.6 s 완료로 기록(p99 TTFT 361.6, TPOT 37.32). 컴파일된 `cluster.json`의 유일한 차이는 `link_bw` 35.0→35.2 (D18 재계산).
- `serving/`의 마지막 변경은 2026-08-21 `8f17612`(D15). `.hp-pd-slo` sweep은 8/27–28 실행 → **D15는 완료 시점에 이미 들어가 있었다.** 코드 회귀라면 `serving/` 밖(planner 컴파일러, 워크로드 생성, 환경)이다.
- P/D 핸드오프 경로: `serving/__main__.py:623` `router.transfer_prefill_request(finished_reqs)` → `serving/core/router.py:340`. `pd_transfer_bw_gbps=None`(기본, sweep도 이 경로)이면 즉시 `add_decode`. planner의 전송 비용은 `planner/optimizer/exhaustive.py::apply_pd_transfer_cost`에서 **사후 해석적으로** 얹으므로 시뮬레이터 플래그와 무관.
- `d48c401`(tier step2)이 `profiler/perf/A40/**/meta.yaml`에 `tier: measured` 한 줄을 추가했다. CSV는 불변. 배제 가능하지만 STEP A.2에서 한 번 확인한다.

증상이 가리키는 가설, 검증 비용 순:

| # | 가설 | 위치 | 예측되는 관찰 |
| --- | --- | --- | --- |
| H1 | **환경 차이** — 완료 run과 현재 노드의 chakra / protobuf / Python 버전이 다름 | venv | 동일 입력을 현재 노드에서 재생하면 livelock, 옛 환경에서는 완료 |
| H2 | **`link_bw` 임계값 버그** — 35.2가 어딘가에서 정수 변환·비교를 뒤집음 | ASTRA-Sim 입력 | 35.0으로 되돌리면 완료. 그 경우 ASTRA-Sim 쪽(`network.yml` 파싱)이거나 frontend는 무관 |
| H3 | **zero-progress 스케줄링** — prefill 인스턴스가 매 iteration 토큰 0개(또는 KV 할당 없는) 배치를 만들어 ASTRA-Sim에 넘기고, 요청은 영원히 running | frontend `scheduler.py` | `--log-level INFO`에서 prefill 인스턴스의 batch가 매 틱 같은 `total_len`(0 또는 극소)으로 반복; `--no-cleanup-inputs`로 남긴 trace 파일이 매 iteration 거의 동일 |
| H4 | **ASTRA-Sim 내부 deadlock** — prefill의 COMM_SEND(compute rank → sender rank, `llm_converter.py:862`)가 짝 RECV를 못 만나 `sys[prefill]`의 iteration finished가 영원히 안 나옴 | ASTRA-Sim / .et | frontend는 그 인스턴스의 응답을 기다리며 decode 쪽 idle 틱만 진행; `--log-level INFO`에서 `NPU[k] iteration N finished`가 prefill 랭크에 대해 멈춤 |
| H5 | **워크로드 생성 변화** — `planner/util/workload.py`가 seed 42에서 8/28과 다른 trace를 만듦 | planner | trace 파일 해시가 `.hp-pd-slo` 시점과 다름 (D23은 "same trace"라고 썼으나 해시 비교로 못 박는다) |

H3과 H4는 **한 번의 `--log-level INFO` 실행으로 서로 구분**된다: H3이면 prefill 인스턴스의 iteration이 계속 "finished"되며 배치가 반복되고, H4이면 어느 순간부터 prefill 랭크의 finished 로그가 끊긴다.

### 1.2 D14 — 사전 조사의 결론: 막힌 곳은 ASTRA-Sim이 아니라 `config_builder.py`의 30줄이다

D14/D16(b)는 "시뮬레이터가 균일한 instance 크기를 요구한다"고 쓰지만, 실제 구조는 다음과 같다.

**ASTRA-Sim analytical은 N차원(최대 5) 토폴로지와 collective별 `involved_dim` 벡터를 지원한다.** `astra-sim/workload/Workload.cc:275-296`이 ET 노드의 `involved_dim` bool_list를 읽어 collective가 참여하는 차원을 결정한다(없으면 `[1,1,1,1,1]`). 이 fork의 Chakra 변환기 `chakra/src/converter/llm_converter.py:226-233`은 임의 길이의 `involved_dim`을 그대로 ET에 쓴다. frontend `serving/core/trace_generator.py:980 _with_dim`은 `ALLREDUCE:1,0` 형식으로 인스턴스별 `ctx.tp_dim`을 인코딩하고, **`tp_dim`은 이미 인스턴스별 필드**다(`config_builder.py:188 inst["tp_dim"] = local_dim`). EP 경로는 이미 2차원 관여(`ep_dim = [True, True]`)를 쓴다.

**균일성을 강제하는 곳은 딱 하나** — `serving/core/config_builder.py:192 _compute_network_dims`가 토폴로지를 `[npus_per_group, num_instances]`로, `npus_per_group = total_npu // num_instances` 정수 나눗셈으로 정하고, 모든 인스턴스에 같은 `local_dim = [True, False]`를 준다.

**랭크 배치 규칙** (`config_builder.py:540-556`, `llm_converter.py:753-755, 862`): 인스턴스는 연속 랭크 블록을 차지한다. prefill 인스턴스는 `2·tp` 랭크(compute `tp`개 + sender `tp`개, sender = `npu_id + num_npus`), decode/colocated는 `tp` 랭크. ASTRA-Sim의 랭크 번호는 `i0 + d0·i1 + d0·d1·i2`로 선형이다.

**따라서 3차원 인코딩이 비대칭을 표현한다.** 토폴로지 `[g, 2, n]`에서:

| 인스턴스 | 랭크 수 | 슬랩 점유 | `tp_dim` | collective 범위 |
| --- | ---: | --- | --- | ---: |
| prefill tp=g (compute + sender) | 2g | 슬랩 1개 전체 (i1=0 compute, i1=1 sender) | `[T, F, F]` | g ✓ |
| decode tp=2g | 2g | 슬랩 1개 전체 | `[T, T, F]` | 2g ✓ |
| colocated tp=g | g | 반 슬랩 (둘이 한 슬랩 공유) | `[T, F, F]` | g ✓ |

`A40 tp4 prefill + RNGD tp8 decode` = `[4, 2, 2]`, 16랭크, 유휴 랭크 없음. prefill의 compute→sender COMM_SEND는 오늘의 `[4, 3]` 배치에서와 똑같이 dim 1을 건넌다. **균일 P/D(`tp_p == tp_d`)는 3-D로 가지 않는다**: prefill(전체 슬랩) + decode tp4(반 슬랩 1개, 홀수)는 슬랩 격자에 맞지 않으므로 오늘의 `[4, 3]` 경로(`auto`)에 그대로 남는다. 즉 **3-D는 비대칭 전용 opt-in 확장이고 기본 경로는 불변**이다.

**커버 범위와 한계.** `tp_d = 2·tp_p`(업계 권장 방향이자 RNGD가 필요로 하는 방향)를 유휴 랭크 없이 표현하고, colocated tp=g 인스턴스는 짝수 개일 때 함께 들어간다. 균일 P/D는 `auto`에 남는다. 다른 비율(4×)이나 반 슬랩 인스턴스가 홀수 개인 경우는 유휴 랭크 패딩이 필요하다 — `llm_converter.py:398`의 경고("Some npus won't do anything!")는 변환기가 유휴 랭크를 만들 수는 있음을 시사하지만, frontend의 iteration 배리어가 그 랭크를 어떻게 다루는지는 **미확인**. 이 스파이크 범위 밖.

**정확도 함의 — 스파이크가 측정할 것.** decode tp8의 allreduce가 flat ring(8)에서 계층 ring(4×2)으로 바뀐다. ring의 대역폭 항은 두 경우 모두 `2·(7/8)·size/bw`로 동일하고(RS+AG, 계층: 3/4 + (1/4)(1/2) = 7/8), **지연 항만 7단계→4단계로 준다.** RNGD tp8 프로파일의 `link_latency`는 flat ring 기준으로 보정되었으므로(`experiments/configs/clusters/rngd-llama31-8b-tp8.json`, PROJECT_REPORT §4.8.5의 115 µs/layer), 3-D 인코딩은 레이어당 `2·3·link_latency`만큼 decode를 빠르게 본다. 이 차이가 TPOT에서 프로파일 자체 오차(−3.1 %)보다 작으면 그대로 쓰고, 크면 dim별 `link_latency`로 보정한다 — `_normalize_network_dim_values`가 dim별 리스트를 이미 받는다.

**planner 쪽에 필요한 것** (구현은 다음 작업지시서): `planner/candidate_generator.py::_pd_candidates`의 `tp_p == tp_d`를 `tp_d ∈ {tp_p, 2·tp_p}`로; 컴파일러(`planner/predictor/llmservingsim.py`)가 서로 다른 `num_npus`의 인스턴스를 내보내고 `link_bw`/`link_latency`를 dim별 리스트로 쓸 수 있게; `tests/test_mixed.py`의 균일성 단언 갱신; D16(b) fixture의 "size-4 island bridging" 우회가 더 이상 필요 없음을 기록.

---

## 2. 공통 규칙

```bash
bash scripts/whichnode.sh                                   # NPU 노드. 시뮬레이션만 한다
export PYTHONPATH=$PWD && export PATH="$PWD/.venv/bin:$PATH"
pytest -q && ruff check . && mypy                           # 시작·종료 시 각 1회. 스파이크는 테스트 수를 바꾸지 않는다(A.1의 스크립트 테스트 제외)
```

**livelock 감시기.** 모든 P/D 시뮬레이션은 `experiments/scripts/livelock_watch.sh`(STEP A.1에서 작성) 아래에서 돈다. 이 스크립트는 `python -m serving --log-interval 1.0` 출력을 tail하며, **prefill 인스턴스가 running 1 · Waiting 단조 증가 · 메모리 % 불변 상태를 연속 N=300틱 유지하면 프로세스를 죽이고 `LIVELOCK`를 exit code 3으로 보고**한다. 완료면 0, 일반 타임아웃이면 124. 이것으로 3600 s짜리 실험이 2~3분에 끝난다.

---

# STEP A. D23 — livelock 진단 (예산 1.5일)

## 목표
§1.1의 H1~H5 중 어느 것인지 확정하고, 고치는 위치(planner / 환경 / `serving/` sanctioned edit)를 D23에 기록한다.

## A.0 재현 최소화

1. `outputs/pd_slo_sweep_margin18/tight/timeouts_pd-rngd-gpu.txt`에서 후보 `P[cuda:tp4] D[cuda:tp4] -s256-t8192`의 run 디렉터리 이름을 찾고, 그 `cluster.json`을 `outputs/d23/cluster_livelock.json`으로 복사한다(없으면 planner로 재컴파일: `python -m planner plan … --keep-artifacts` 계열 옵션 조사 필요 — `LLMServingSimPredictor(work_dir=…, keep_artifacts=True)`가 `run_dir/cluster.json`을 남긴다, `llmservingsim.py:347-349`).
2. **300 → 20 요청으로 재현되는지** 확인한다. 워크로드는 `.hp-pd-slo` 실행과 같은 생성기·seed로 만들되 `--num-requests 20`. 감시기 아래에서:
   ```bash
   experiments/scripts/livelock_watch.sh -- python -m serving \
     --cluster-config outputs/d23/cluster_livelock.json --dataset <trace20.jsonl> \
     --num-reqs 20 --dtype bfloat16 --kv-cache-dtype auto --block-size 16 \
     --max-num-seqs 256 --max-num-batched-tokens 8192 \
     --request-routing-policy LOAD --network-backend analytical \
     --log-level INFO --log-interval 1.0 --no-enable-prefix-caching \
     --no-cleanup-inputs --run-id d23-repro20 --output outputs/d23/repro20.csv
   ```
   (플래그 목록은 `planner/predictor/llmservingsim.py:395-415`가 planner가 실제로 넘기는 것이다 — 거기서 복사하고 임의로 바꾸지 않는다.)
3. 20에서 재현되면 이후 모든 실험은 20으로. 재현되지 않으면 50, 100으로 올려 **재현되는 최소 N**을 기록 — 그 자체가 정보다(요청 수 의존 = 도착 타이밍·큐 길이 관련, H3 쪽).

## A.1 H3 vs H4 분리 — 로그 한 번으로

A.0의 `--log-level INFO` 출력에서:
- `NPU[k] iteration N finished` 로그(`controller.py:50`)가 **prefill 랭크(0..3)에 대해 계속 나오는가** — 나오면 ASTRA-Sim은 정상 응답 중 → H3(frontend). 끊기면 → H4(ASTRA-Sim/ET).
- H3이면 `--no-cleanup-inputs`가 남긴 `astra-sim/inputs/runs/d23-repro20/trace/` 아래 prefill 인스턴스의 연속 trace 파일 3개를 diff — 같은 배치가 반복되는지, `total_len`이 0인지, 어느 요청 id인지.
- H4이면 마지막으로 생성된 prefill `.et`를 `chakra` 도구로 덤프해 COMM_SEND/RECV 쌍을 확인한다.

이 단계에서 `experiments/scripts/livelock_watch.sh`와 그 pytest(`tests/test_livelock_watch.py`: 가짜 로그 스트림에 대해 exit 3 / 0 을 내는지, 시뮬레이터 불필요)를 커밋한다.

## A.2 입력 vs 환경 분리

1. **trace 해시.** A.0의 20-요청 trace와, 같은 방법으로 만든 300-요청 trace의 sha256을 `.hp-pd-slo` 시점 것과 비교한다. 옛 파일이 없으면 `git stash`/worktree로 **8/28 시점 커밋**(`experiments/results/pd_slo_sweep.md`를 커밋한 sha)을 체크아웃해 같은 seed로 생성하고 해시 비교. 다르면 H5 확정 → `planner/util/workload.py` 히스토리 bisect.
2. **환경 기록.** `.venv/bin/python -c "import chakra, google.protobuf, sys; print(sys.version, google.protobuf.__version__)"`, `pip list | grep -iE 'chakra|protobuf|pyyaml|msgspec'`, `git -C astra-sim rev-parse HEAD`, ASTRA-Sim 바이너리의 mtime. `docs/phase0_formats.md` §1의 pin 시점 값과 비교해 `docs/d23_spike.md`에 표로 기록. 차이가 있으면 H1 후보.
3. **`link_bw` 되돌리기 (H2).** `cluster_livelock.json`의 `link_bw`를 35.0으로 바꿔 1회. 완료되면 H2 확정 → 35.1로 한 번 더 좁히고, `config_builder.py::_normalize_network_dim_values`와 ASTRA-Sim analytical의 bandwidth 파싱에서 정수 변환을 찾는다(**읽기만**). 완료되지 않으면 H2 기각.
4. **meta.yaml `tier:` 줄 (배제 확인).** `git stash`가 아니라 `git show d48c401~1:profiler/perf/A40/meta-llama/Llama-3.1-8B/bf16/meta.yaml`로 옛 파일을 임시 복원해 1회. 예상: 무관.

## A.3 원인 확정과 기록

`docs/deviations.md` D23을 갱신한다(A2 — 기존 본문 위에 "Diagnosed <날짜>" 블록):
- 확정된 가설, 재현 최소 조건(N, knobs), 결정적 로그 발췌.
- **고치는 위치**: (a) planner/환경이면 다음 작업지시서의 STEP으로 명시; (b) `serving/`이면 D15 형식의 sanctioned edit 제안 — 어느 파일 어느 함수, opt-in 방식, byte-identical 증명 방법(`.hp-pd-slo`의 완료 후보를 회귀 기준으로).
- **`--log-level WARNING`이 이 증상을 숨긴다**는 점과 감시기 사용법을 "How to adapt"에 추가.
- `docs/CLAIMS.md` §2의 D23 단락에 한 줄: 원인과 tight-TTFT regime 판정 재개 조건.

## 완료 조건
- [ ] H1~H5 각각 확인/반증/미결 표
- [ ] 재현 최소 조건 기록, `outputs/d23/` 산출물 커밋(provenance 포함)
- [ ] `experiments/scripts/livelock_watch.sh` + 테스트 커밋
- [ ] D23 갱신, CLAIMS.md 한 줄, `docs/d23_spike.md`
- [ ] `serving/` 무변경

---

# STEP B. D14 — 3-D 토폴로지 인코딩으로 비대칭 TP 검증 (예산 1.5일)

## 목표
§1.2의 주장 — "`config_builder.py`만 바꾸면 `tp_d = 2·tp_p`가 표현된다" — 를 (1) 균일 구성에서 등가성으로, (2) 비대칭 P/D 20-요청 완주로 확인하고, 정확도 대가(allreduce 지연 항)를 숫자로 적는다. 결과는 go/no-go 메모.

## B.1 확인 — ASTRA-Sim이 3차원 `involved_dim`을 받는가 (읽기, 30분)

- `astra-sim/workload/Workload.cc:275-296`(서브모듈 체크아웃에서)과 analytical 네트워크 백엔드의 dim 수 상한을 읽어 기록. `astra-sim/inputs/network/` 예시가 1-D뿐이라 **3-D `network.yml`을 ASTRA-Sim 단독으로 1회 실행**해(예: `[2,2,2]` FullyConnected, 아무 예시 workload) 파싱·실행이 되는지 확인. 실패하면 STEP B 중단·보고.

## B.2 프로토타입 패치 + 균일 구성 등가성 (핵심)

**허용되는 유일한 `serving/` 수정. 스파이크 브랜치에만.**

1. `serving/core/config_builder.py`에 opt-in 키를 추가한다: cluster config 최상위 `"topology_mode": "auto" | "slab3d"`. `auto`(기본, 키 없음)는 현행 `_compute_network_dims` 그대로 — **byte-identical**. `slab3d`는:
   - `g` = 인스턴스들의 최소 tp(compute 기준), 모든 인스턴스가 `g`(반 슬랩) 또는 `2g`(전체 슬랩: prefill tp=g, 또는 decode/colocated tp=2g)이어야 하며 반 슬랩 인스턴스 수는 짝수 — 아니면 `ValueError`.
   - dims = `[g, 2, n_slabs]`, 인스턴스별 `tp_dim`: 반 슬랩·prefill → `[T,F,F]`, 전체 슬랩 decode/colocated → `[T,T,F]`. `ep_dim`은 `tp_dim`과 같게(EP는 이 스파이크 범위 밖 — MoE 인스턴스가 있으면 `ValueError`).
   - 랭크 할당(`current_npu_start` 누적)은 그대로. 반 슬랩 두 개가 한 슬랩을 이루도록 **인스턴스 순서를 정렬**하는 대신, 순서가 맞지 않으면 `ValueError`로 사용자에게 맡긴다(스파이크에서는 fixture를 손으로 맞춘다).
   - `_create_network_config`·`_sync_system_collective_dims`는 `len(dims)`를 따라가므로 변경 불필요 — 확인만.
2. **등가성 실험 1 — 균일 colocated tp4 ×2.** `exp1-a40-tp-sweep`류 cluster에서 `agg[cuda:tp4]` 인스턴스 2개(P/D 아님)를 `auto`와 `slab3d`로 각 1회, 20 요청. `slab3d`에서 tp4 두 인스턴스는 반 슬랩 둘 = `[4,2,1]` → trailing 1 제거로 `[4,2]`, `tp_dim [T,F]` — 이것은 오늘 `auto`가 만드는 `[4,2]`·`[T,F]`와 **같은 dims·같은 tp_dim**이어야 하므로 **CSV가 byte-identical**해야 한다. 아니면 패치가 틀렸다. (균일 P/D는 §1.2대로 `slab3d` 대상이 아니므로 여기서 쓰지 않는다 — `ValueError`가 나는지만 확인.)
3. **등가성 실험 2 — 균일 tp8 단일 인스턴스, flat vs 계층.** `rngd-llama31-8b-tp8.json`(colocated tp8 1개)을 `auto`(dims `[8]`)와, 강제로 `[4,2]`·`tp_dim [T,T]`가 되도록 한 변형(스파이크용 두 번째 opt-in 값 `"topology_mode": "split2"`도 허용)으로 각 1회, 20 요청. 비교:
   - iteration별 cycle 수(`--log-level INFO`의 `iteration N finished, C cycles`) 차이의 분포
   - 최종 CSV의 p50/p99 TPOT, TTFT 차이(%)
   - 예측: 레이어당 `2·3·link_latency` 만큼 `[4,2]`가 빠름. 예측치와 실측 차이를 표로.
   - **판정 기준:** TPOT 차이가 RNGD 프로파일 자체 오차 3.1 %(D22, `rngd_concurrency_envelope.md`) 미만이면 "보정 없이 사용 가능", 이상이면 "dim별 `link_latency` 보정 필요 — 보정값 제시".

## B.3 비대칭 P/D 완주 (D23 결론 이후)

1. fixture 작성: `experiments/configs/clusters/pd-asym-a40tp4-rngdtp8.json` — node 0: A40 ×4 prefill tp4(`hardware: A40`), node 1: RNGD-CARD 또는 RNGD ×8 decode tp8, `topology_mode: slab3d` → dims `[4,2,2]`. `link_bw`/`link_latency`는 우선 스칼라(D3 관례). 두 인스턴스가 서로 다른 `hardware`인 것은 기존 `pd-rngd-gpu` fixture와 같다.
2. 감시기 아래에서 20 요청 1회. **완주하면** CSV의 TTFT를 `agg[cuda:tp4]` 20-요청 run의 TTFT와, TPOT를 `rngd tp8` 20-요청 run의 TPOT와 나란히 놓는다 — 각각 prefill·decode 담당 하드웨어의 단독 값과 같은 자릿수여야 한다(정합성 검사, 정확도 주장 아님).
3. 완주하지 않으면 livelock 감시기 결과(exit 3 vs 124)와 로그로 **D23과 같은 증상인지** 판정. 같으면 D23 의존으로 기록하고 B.3을 미결로 닫는다.

## B.4 메모

`docs/d14_spike.md`:
- §1.2의 주장 중 확인된 것 / 반증된 것.
- B.2 표(등가성, 지연 항 대가), B.3 결과.
- **go/no-go**: 비대칭 TP P/D를 `WORK_ORDER_rps_aware.md`의 후보 집합에 넣는가. go이면 필요한 작업 목록(§1.2 마지막 단락)과 sanctioned edit 제안(D25: `config_builder.py` `topology_mode`, opt-in, B.2-2의 byte-identical 테스트를 회귀 테스트로).
- `docs/deviations.md` D14와 D16(b)에 "Spike 2026-09-0X" 블록: "시뮬레이터가 아니라 `_compute_network_dims`의 제약이며 3-D 인코딩으로 `tp_d ∈ {tp_p, 2·tp_p}` 표현 가능(검증 결과 링크)". D16(c)의 "D14 forbids it" 문장 위에 갱신 표기. `docs/CLAIMS.md` §2 "The shape the industry recommends cannot even be enumerated" 단락에 한 줄.

## 완료 조건
- [ ] B.1 확인 기록
- [ ] B.2-2 byte-identical 확인(diff 결과 커밋), B.2-3 표
- [ ] B.3 완주 여부와 정합성 표, 또는 D23 의존 기록
- [ ] `docs/d14_spike.md`, D14/D16/CLAIMS 갱신
- [ ] 프로토타입 패치는 `spike/d14-asym-tp` 브랜치에만 존재. PR은 **머지하지 않고** "spike — do not merge" 라벨로 열어 리뷰 기록만 남긴다. `main`의 `serving/`은 불변

---

# 3. 전체 완료 조건

- [ ] STEP A·B 개별 완료 조건
- [ ] `main`에 `serving/` 변경 없음, `planner/` 변경 없음, golden 불변
- [ ] `docs/HANDOVER.md` §2.1에 두 스파이크의 결론 한 줄씩 + `WORK_ORDER_rps_aware.md` 작성 시 반영할 항목
- [ ] 게이트 3종 통과 (신규 테스트: `tests/test_livelock_watch.py`만)

# 4. 리스크 대응 규칙

| 상황 | 판단 |
| --- | --- |
| A.0에서 20·50·100 요청으로 재현되지 않고 300에서만 재현 | 그것이 결론이다(도착 타이밍/큐 길이 의존, H3 쪽 강화). 300으로 A.1을 1회만 수행하고 감시기로 3분 안에 끊는다 |
| A.1에서 H3과 H4가 동시에 성립하는 것처럼 보임 | prefill 랭크의 `iteration finished` 로그가 **있는가 없는가**로만 판정한다. 있으면 H3 |
| H2가 확정됨 (35.0 완료, 35.2 livelock) | ASTRA-Sim 쪽 정수 변환을 찾더라도 **고치지 않는다.** D23에 위치와 제안을 적고, 임시 우회로 fixture의 `link_bw`를 정수로 반올림하는 것도 **금지** — D18이 측정한 값을 결과를 위해 바꾸는 것이다 |
| H5가 확정됨 (trace가 달라짐) | `planner/util/workload.py` bisect는 하루를 넘기지 않는다. 넘기면 "어느 커밋 사이"까지만 기록 |
| B.1에서 ASTRA-Sim이 3-D를 거부 | STEP B 중단. `docs/d14_spike.md`에 "no-go: ASTRA-Sim 제약"으로 기록하고 §1.2를 그에 맞게 수정 |
| B.2-2가 byte-identical하지 않음 | 패치의 `auto` 경로가 바뀐 것이다. 등가성 실험을 진행하지 말고 패치를 고친다 |
| B.2-3의 TPOT 차이가 3.1 % 이상 | 실패가 아니다. dim별 `link_latency` 보정값을 제시하고 "보정 필요"로 go |
| B.3이 D23과 같은 증상으로 미완 | B.3 미결·D23 의존으로 기록. **다른 knob으로 바꿔 완주시키려 하지 말 것** |
| 스파이크 도중 `serving/`의 버그를 발견했고 한 줄이면 고칠 수 있음 | 고치지 않는다. 위치·증거·제안을 D23/D14에 적는다 |
| 3일을 넘김 | 그 시점의 확인/반증/미결 표로 닫고 `WORK_ORDER_rps_aware.md` 작성으로 넘어간다. 스파이크는 답을 내는 것이 목표이고, 미결도 답이다 |
