# HeteroPilot 작업지시서 — D23 근본 수정과 병렬 sweep 결과 재검증

> 목적: D23 스파이크가 찾은 harness 결함 두 개(ASTRA-Sim 공유 임시 파일 경쟁, frontend의 dead-child 무한 루프)를 `serving/`에서 **sanctioned edit**으로 고치고, 그 결함 아래에서 병렬로 생산된 과거 결과 — 특히 D22의 근거인 `pd_slo_sweep_margin18` — 가 신뢰할 수 있는지 재검증한다. 그리고 결함 때문에 4시간 타임아웃으로 끝났던 tight-TTFT regime을 마저 돌려 3-regime 표를 닫는다.
> 대상 저장소: `github.com/swsok/heteropilot` · 기준 `main` = `47d702f`(PR #54) · 작성일: 2026-09-04 · 작업 도구: Claude Code CLI, NPU 노드(시뮬레이션 전용)
> 선행: `WORK_ORDER_spikes.md` STEP A 완료(`docs/d23_spike.md`). 후속: `WORK_ORDER_rps_aware.md`(이 문서의 STEP 3 결과를 입력으로 받는다).
> 예산: 코드 1일 + 시뮬레이션(대부분 백그라운드) 1~2일.

---

## 0. 이 문서의 사용법

1. **STEP 1 → 2 → 3 순서.** STEP 3의 모든 시뮬레이션은 STEP 1·2가 머지된 `main`에서 돈다 — 결함이 있는 frontend로 재검증하는 것은 의미가 없다.
2. **한 STEP = 한 브랜치 = 한 PR.** `fix/d23-astra-cwd`, `fix/d23-dead-child`, `exp/d23-revalidation`.
3. STEP 3의 긴 실행은 **`nohup`/`tmux`로 detach**한다(통합 스프린트 STEP 2에서 세션 종속 실행으로 84개 시뮬레이션을 잃은 전례).
4. 상위 문서 우선: `WORK_ORDER_heteropilot.md` → `docs/deviations.md` → `CLAUDE.md`.

### 절대 규칙 (재확인)

- **A1.** 하드웨어 측정 없음. 이 문서는 시뮬레이션과 코드만 다룬다.
- **A2.** 철회는 공개적으로. 재검증에서 과거 숫자가 바뀌면 덮어쓰지 않고 위에 얹는다.
- **A3′ — `serving/` 편집은 이 문서가 명시한 두 곳만.** `serving/__main__.py`의 `Popen` 호출 1건과 `serving/core/controller.py`의 `read_wait`/`check_end`. 각각 D15 방식: **기본 출력 byte-identical을 회귀 앵커로 증명**하고 `docs/deviations.md`에 기록한다. `astra-sim/`은 건드리지 않는다(upstream issue 초안이 `docs/upstream_issues/`에 있다).
- **A4.** golden 회귀 출력 불변. `planner/`는 이 문서에서 변경하지 않는다(STEP 3.1의 분석 스크립트는 `experiments/`에 둔다).

---

## 1. 사전 조사 (2026-09-04, `47d702f` 클론과 `casys-kaist/astra-sim` 소스에서 확인)

### 1.1 결함의 정확한 위치

- **ASTRA-Sim** `astra-sim/network_frontend/analytical/congestion_unaware/main.cc`: `save_json_to_tmp()`가 `tmp__mem/<name>.json`(cwd 상대, pid·run id 없음)에 메모리 설정을 쓰고 `AnalyticalMemory(path)`가 읽은 뒤 `std::remove`. `local_mem`/`remote_mem`/`cxl_mem` 세 번(main.cc:125-149). 마지막의 `::rmdir("tmp_mem")`은 오타(`tmp_mem` ≠ `tmp__mem`)라 디렉터리는 남는다.
- **frontend** `serving/__main__.py:199 os.chdir(astra_sim)` → 모든 동시 프로세스의 cwd가 `astra-sim/` 하나. `:548 Popen(astra_args, stdin=PIPE, stdout=PIPE, stderr=PIPE)` — `cwd` 인자 없음, stderr는 읽는 곳이 없음.
- **frontend** `serving/core/controller.py:12-19 read_wait`: `while "Waiting" not in out[-1] and out[-1] != "Checking Non-Exited Systems ...\n": line = p.stdout.readline()` — 자식이 죽으면 `readline()`이 `""`을 반환하고 어느 조건에도 걸리지 않아 무한 루프, `out` 리스트 무한 증가(측정: 30 s에 RSS +170 MB). `:21-29 check_end`도 같은 구조.
- **네 인자는 이미 절대 경로**다: `serving/core/run_paths.py:39-45`가 `inputs_root`와 그 아래 `network/system/memory` 경로를 `abspath`로 만들고, `workload`는 `input_path()`(`:49-50`)로 절대화된다. 즉 자식의 cwd는 `tmp__mem/` 이외에 아무 데도 쓰이지 않는다 — **`cwd=run_paths.inputs_root` 한 인자로 충분**하다(스파이크가 64/64로 측정). `--cleanup-inputs`(기본 on)가 성공 후 `inputs_root`를 지우므로 `tmp__mem/`도 함께 정리된다.
- **planner 쪽은 이미 준비되어 있다**: `planner/predictor/llmservingsim.py:440-445`가 `returncode != 0`이면 stderr/stdout 꼬리 3줄을 `detail`에 담아 실패 outcome을 기록한다. frontend가 빨리, 메시지와 함께 죽기만 하면 `rejected_summary`에 원인이 남는다.

### 1.2 "조용한 오염"은 가능한가 — 재검증이 답해야 할 질문

경쟁의 결과는 두 가지다: (a) 파일이 없거나 반쯤 쓰인 상태에서 읽어 **crash**(스파이크가 관찰: `Unable to open file` 5건, SIGABRT 8건 / 64), (b) **다른 프로세스가 완성해 둔 파일을 읽어** 그 프로세스의 메모리 설정으로 **에러 없이** 시뮬레이션. (b)는 스파이크가 원리적으로 볼 수 없었다 — 동일 후보 64개였으므로 바꿔 읽어도 내용이 같다. 실제 sweep은 서로 다른 후보 64개였다.

(b)가 결과를 바꾸는지는 **파일 내용이 후보 간에 실제로 다른가**에 달려 있다. `config_builder.py:488-493`을 보면 `remote_mem` json은 `{memory-type, mem-bw: cpu_mem.mem_bw, mem-latency: cpu_mem.mem_latency, num-devices: num_nodes}`이고, planner 컴파일러는 `cpu_mem`을 노드의 `cpu_memory_bw_gbps`(기본 256)·`cpu_memory_latency_ns`(기본 0)에서 채운다(`llmservingsim.py:221-224`, `inventory.py:152-154`). `pd-rngd-gpu*.yaml` fixture는 이 값을 재정의하지 않으므로 **bw/latency는 모든 후보에서 같고, 다를 수 있는 필드는 `num-devices = num_nodes`뿐**이다. `local_mem`은 local offloading 시에만, `cxl_mem`은 CXL 설정 시에만 쓰이며 sweep fixture에는 둘 다 없다.

그러므로 예측은: 후보 간 내용 차이는 `num-devices`에 한정되고, 노드 수가 더 많은 후보가 더 적은 후보의 파일을 읽으면 `REMOTE:{node_id}` 접근이 범위를 벗어나 **crash 또는 정의되지 않은 동작**(원래 D23 증상 — 죽지도 끝나지도 않음 — 의 유력한 설명), 반대 방향은 bw가 같으니 **수치 동일**. 이 예측이 맞으면 **완료된 과거 결과는 신뢰할 수 있다**(오염은 crash/hang으로만 나타남). STEP 3.1이 이것을 시뮬레이션 없이 확인하고, STEP 3.2가 실제 재실행으로 못 박는다.

### 1.3 재검증 대상 — 무엇이 병렬·이종으로 돌았는가

| 결과 | 병렬 | 이종 하드웨어 동시 실행 | 위험 | 처분 |
| --- | --- | --- | --- | --- |
| `pd_slo_sweep_margin18`(D22 근거) | workers 32 | A40 + RNGD | 있음 | **STEP 3.2 재실행** |
| `pd_slo_sweep`(committed 3-regime, 이미 D22로 superseded) | 32 | A40 + RNGD | 있음 | 재실행하지 않음 — superseded. STEP 3.2 결과가 그 표의 A40 행과 일치하는지만 대조 |
| tight-TTFT(500/8000) | 32 | A40 + RNGD | 미완 | **STEP 3.3 실행** |
| Exp 2 이종 선택, Exp 5 4-combo | 병렬 | 있음 | 있음 | STEP 3.1의 결론이 "내용 동일 또는 crash-only"면 재실행 불필요로 기록; 아니면 `WORK_ORDER_rps_aware.md`에 재실행 항목 |
| Exp 1 TP sweep, E1(A40 only), §4.7 78후보 병렬-순차 대조 | 병렬 | **없음**(단일 하드웨어) | 없음 — 내용 동일 | 기록만 |

---

## 2. 공통 규칙

```bash
bash scripts/whichnode.sh
export PYTHONPATH=$PWD && export PATH="$PWD/.venv/bin:$PATH"
pytest -q && ruff check . && mypy
```

**회귀 앵커 두 개**, 모든 `serving/` 편집 PR이 둘 다 통과해야 한다:
- **R1 (upstream 앵커)** `bench/examples/`의 기준 run — `docs/phase0_formats.md` §2.1이 "pin에서 정확히 재현"이라 보증하는 것. 출력을 `outputs/d23fix/anchor/`로 리다이렉트(**`bench/examples/`에 쓰지 말 것**, CLAUDE.md). 편집 전후 CSV `sha256` 동일.
- **R2 (P/D 앵커)** `outputs/.hp-pd-slo/`의 완료 후보 `P[cuda:tp4] D[cuda:tp4] -s256-t8192` — `docs/d23_spike.md`가 단독 343 s 완주를 확인한 그 입력. 편집 전후 CSV `sha256` 동일.

---

# STEP 0. 준비 (반나절)

1. baseline 게이트 결과와 `whichnode.sh` 출력을 `docs/d23fix_baseline.md`에 기록.
2. `origin/docs/d14-spike-findings`(문서·증거만, `serving/` 없음 — 확인 후) 를 `main`에 머지한다. `origin/spike/d14-asym-tp`는 **머지하지 않고** 그대로 둔다(PR은 "spike — do not merge").
3. R1·R2를 **편집 전 `main`에서 1회씩** 돌려 기준 CSV와 `sha256`을 `outputs/d23fix/anchor/before/`에 저장한다. R2는 단독 실행(격리 불필요).

---

# STEP 1. sanctioned edit ① — ASTRA-Sim 자식에 run별 cwd (D25-a)

## 지시

1. `serving/__main__.py:548`의 `Popen(...)`에 `cwd=run_paths.inputs_root`를 추가한다. 그 외 변경 없음. 주석 한 줄: 왜(main.cc의 cwd 상대 `tmp__mem/`), 그리고 네 인자가 절대 경로라 안전한 이유(`run_paths.py`).
2. `network_backend == 'ns3'` 경로의 `--logical-topology-configuration`은 `astra_sim` 절대 경로로 조립되므로 영향 없음 — 확인만.
3. `docs/deviations.md`에 **D25** 절 추가: "ASTRA-Sim의 cwd 상대 임시 파일 경쟁 — frontend가 run별 cwd를 준다 · Resolved (second sanctioned `serving/` edit)". D23의 "Where to fix" 단락 위에 "Fixed by D25-a" 표기. D23 heading은 STEP 3 종료 시 갱신.
4. `CLAUDE.md` 절대 규칙 1의 예외 목록에 D15와 나란히 D25를 추가하고, D23 스파이크가 넣은 "parallel simulations need no extra locking is wrong" 문구를 "D25 이후에는 다시 참(run별 cwd)"로 갱신.
5. `experiments/scripts/astra_isolated.sh` 상단에 "D25-a 이후 필수 아님. 다른 노드에서 D25가 적용된 바이너리인지 확실치 않을 때의 이중 안전장치"로 표기. 삭제하지 않는다.

## 테스트

- `tests/test_astra_cwd.py`(시뮬레이터 불필요): `serving.__main__`의 Popen 호출을 monkeypatch로 가로채 `cwd`가 `run_paths.inputs_root`(절대 경로)인지 assert. `--inputs-root` 재정의 시에도 따라가는지.
- **R1·R2 byte-identical** (`outputs/d23fix/anchor/after_step1/`), 결과를 PR 본문에.
- **동시성 증명**: `docs/d23_spike.md`의 H6 하네스(64-way, 동일 입력)를 `astra_isolated.sh` **없이** 재실행 → 64/64 완료, `astra-sim/tmp__mem/`이 비어 있고 각 `inputs/runs/<id>/tmp__mem/`이 생겼다가 cleanup으로 사라짐. 로그를 `outputs/d23fix/h6_cwd.log`로 커밋.

## 완료 조건
- [ ] `serving/__main__.py` 변경이 `cwd=` 한 인자 + 주석뿐 (`git diff --stat`으로 PR에 표시)
- [ ] R1·R2 sha256 동일, H6 64/64
- [ ] D25-a, CLAUDE.md, `astra_isolated.sh` 표기

---

# STEP 2. sanctioned edit ② — dead-child 감지와 stderr 보존 (D25-b)

## 지시

1. `serving/core/controller.py`:
   - `read_wait`·`check_end` 루프 안에서 `line == ""`이면 `p.poll()`을 확인하고, 자식이 종료되었으면 `RuntimeError`를 던진다. 메시지에 `returncode`와 stderr 꼬리(아래 3)를 포함. 자식이 살아 있는데 `""`이 온 경우(EOF 없는 빈 줄은 `readline`이 `"\n"`을 주므로 이론상 없음)도 N회 연속이면 같은 예외.
   - `out` 리스트 무한 증가 방지는 위 예외로 충분하다. 별도 상한을 두지 않는다(정상 run의 출력량을 가정하지 않기 위해).
2. `serving/__main__.py:548`: `stderr=subprocess.PIPE` → `stderr=<inputs_root>/astra_stderr.log` 파일 핸들. 파이프 버퍼(64 KB)가 차서 자식이 막히는 잠재 결함도 함께 제거된다. 예외 경로에서 그 파일의 마지막 20줄을 읽어 메시지에 붙이고 **stderr로 재출력**한다 — planner의 `llmservingsim.py:440-445`가 그 꼬리를 `detail`로 담는다. `_cleanup_inputs_root`는 성공 경로의 마지막(`__main__.py:1076-1077`, `save_output` 뒤)에서만 호출되므로 예외로 종료한 run에는 파일이 남는다 — 확인됨.
3. 예외는 잡지 않고 전파해 **frontend가 non-zero로 종료**하게 한다. planner는 이미 non-zero를 실패 outcome으로 분류한다.
4. D25 절에 (b) 항목 추가.

## 테스트

- `tests/test_controller_dead_child.py`(시뮬레이터 불필요): `stdout.readline()`이 `""`을 돌려주고 `poll()`이 1인 stub 프로세스에 대해 `read_wait`가 1 s 이내에 `RuntimeError`를 던지고 메시지에 returncode가 있는지; 정상 스트림("…Waiting…")에서는 기존과 같은 `out` 리스트를 돌려주는지(기존 동작 회귀).
- **R1·R2 byte-identical** (`after_step2/`).
- **수동 확인 1회, 기록 필수**: R2를 돌리다 `AnalyticalAstra` 프로세스를 `kill -9` → frontend가 **5 s 이내** non-zero 종료, 메시지에 stderr 꼬리. 그리고 planner 경로로도 1회: 스파이크의 bare-binary 실패를 유발하는 조건(격리 없이 64-way — STEP 1 이후에는 재현이 안 되므로, 대신 `cwd`를 임시로 공유 디렉터리로 되돌린 브랜치 로컬 실험)에서 `pd_slo_sweep.py`가 그 후보를 **TIMEOUT이 아니라 exit-code 실패**로 `rejected_summary`에 남기는지. 출력 발췌를 `outputs/d23fix/dead_child_demo.txt`로 커밋.

## 완료 조건
- [ ] `controller.py`·`__main__.py` 변경이 위 항목뿐
- [ ] R1·R2 sha256 동일, 단위 테스트, 수동 확인 기록
- [ ] D25-b

---

# STEP 3. 재검증 (시뮬레이션, 대부분 백그라운드)

## 3.1 후보 간 메모리 json 내용 비교 — 시뮬레이션 불필요 (1시간)

1. `outputs/.hp-slo-margin18-*/`(margin18 sweep의 work dir; 없으면 `pd_slo_sweep.py`를 `--dry-run`류 옵션 또는 `LLMServingSimPredictor`로 **컴파일만** 해서 후보별 `cluster.json`을 다시 만든다 — `llmservingsim.py:347-349`, 조사 필요)에서 후보별 `cluster.json`을 모은다.
2. `experiments/scripts/d23_memjson_census.py`: 각 `cluster.json`에 대해 `serving.core.config_builder.build_cluster_config(astra_sim, path, inputs_root=<tmp>)`를 호출해 `memory/memory_expansion.json`을 얻고, `remote_mem`/`local_mem`/`cxl_mem` 각각의 정규화된 내용을 해시. 출력: 고유 내용 수, 어떤 필드가 다른지, 후보 수 분포. `build_cluster_config`가 `.venv`에서 torch 없이 import되는지 먼저 확인(조사 필요 — 안 되면 json 필드 규칙을 §1.2대로 스크립트에 복제하고 출처 주석).
3. 결과를 `docs/d23_revalidation.md` §1에 기록. **예측(§1.2): bw/latency 동일, `num-devices`만 다름.** 예측과 다르면 어느 필드가 다른지와 그 필드가 시뮬레이션 시간에 영향을 주는 경로(ASTRA-Sim의 `AnalyticalMemory`가 `REMOTE:*` MEM_LOAD에 쓰는 값)를 적는다.

## 3.2 D22 근거 행의 재실행 (병렬, 수정된 frontend, 1~2시간)

`experiments/results/pd_slo_sweep_margin.md`의 표가 근거하는 행들을 **같은 스크립트·같은 파라미터**로 다시 돈다 — `--workers 32`, 격리 wrapper 없이(D25가 고쳤음을 증명하는 것이 목적의 절반이다):

```bash
for fx in pd-rngd-gpu-card pd-rngd-gpu; do
  nohup .venv/bin/python experiments/scripts/pd_slo_sweep.py \
      --service examples/service_specs/llama31-8b.yaml \
      --cluster experiments/configs/clusters/$fx.yaml \
      --ttft-ms 64000 --num-requests 300 --seed 42 --workers 32 \
      --timeout 1800 --tpot-margin-percent 18 \
      --output-dir outputs/.hp-reval-margin18-$fx > outputs/d23fix/reval_$fx.log 2>&1 &
done
```

기록할 것:
- **타임아웃 수** (예측 0) 와 exit-code 실패 수(예측 0; 있으면 stderr 꼬리와 함께).
- 완료 후보 각각에 대해 committed `outputs/pd_slo_sweep_margin18/<fx>.json`의 같은 후보와 **p99 TTFT / p99 TPOT / tok/J / SLO attainment** 대조. 시뮬레이터는 결정적이므로 **동일 후보는 소수점까지 같아야 한다**; 다른 행이 하나라도 있으면 그 행의 두 CSV를 diff하고 §1.2의 예측(bw 동일 → 수치 동일)과 대조한다.
- 과거 run에서 **타임아웃했던 후보들이 이번에 완료**되면(예측: 그렇다) 그 후보들이 순위를 바꾸는지 — 즉 D22의 "모든 RNGD 후보 거부, 승자 `agg[cuda:tp4]` 2.595 tok/J"가 유지되는지. **이것이 이 STEP의 핵심 판정**이다.

## 3.3 tight-TTFT regime 완결 (백그라운드, 수 시간)

3.2와 같은 커맨드로 `--ttft-ms 500,8000`, `--output-dir outputs/.hp-reval-tight-$fx`. 스파이크가 단독 343 s를 확인했으니 타임아웃은 0이어야 한다. 결과:
- fixture별·지점별 승자, p99 TTFT/TPOT, tok/J.
- committed winner `P[cuda:tp4] D[cuda:tp4]`(p99 TPOT 37.27 → ×1.18 = 43.98 ms, 통과 예상)의 실제 판정.
- 3-regime 표의 tight 행을 **확정**: (i) 유지, (ii) 뒤집힘, (iii) 여전히 미결(왜). 타임아웃이 남으면 `livelock_watch.sh`로 재현해 D23과 같은지 다른지 구분하고, 다르면 새 D 번호.

## 3.4 (조건부) 메모리 모델 민감도

3.1에서 bw/latency가 후보 간에 **다르다**고 나왔을 때만: `pd-rngd-gpu.yaml` 사본에서 한 노드의 `cpu_memory_bw_gbps`를 25.6으로 바꿔 후보 하나를 20요청으로 돌리고 기본값 run과 CSV 비교. 같으면 이 config군에서 `remote_mem` 모델은 수치에 영향 없음 → 과거 결과 안전. 다르면 과거 이종 병렬 sweep(Exp 2, Exp 5)의 재실행을 `WORK_ORDER_rps_aware.md` 항목으로 올린다.

## 3.5 기록

- `docs/d23_revalidation.md`: 3.1 census, 3.2 대조표(행별 동일/상이), 3.3 tight 결과, 3.4(있으면), 그리고 **§1.3 표의 "처분" 열을 결론으로 갱신**.
- `experiments/results/pd_slo_sweep_margin.md`: "Re-validated <날짜> under D25" 블록 + tight 절 갱신(A2 — 기존 텍스트 유지).
- `docs/deviations.md`: **D22** 행에 "근거 재검증 완료/변경" 한 줄; **D23** heading을 "Resolved (harness fault, D25; original symptom explained/unexplained)"로 — 원래 증상(52,903틱, prefill 고정)이 3.1·3.2로 설명되면 그 문장을, 안 되면 "unexplained, not reproduced after D25"를 정직하게.
- `docs/PROJECT_REPORT.md` §4.8.7 caveat에 tight regime 확정 문장. `docs/CLAIMS.md` §1.4·§2의 D22/D23 단락 갱신 — 특히 §2 첫 문장("no heterogeneous configuration is shown to win")이 3.3 결과로 바뀌는지 확인.
- `docs/HANDOVER.md` §2.1: 이 문서 완료, 다음은 `WORK_ORDER_rps_aware.md`, 그 입력으로 3.1~3.3의 결론 세 줄.

## 완료 조건
- [ ] 3.1 census 결과와 예측 대조
- [ ] 3.2: 타임아웃 0, 과거 완료 행과 소수점 일치 여부 표, D22 판정 유지/변경 명시
- [ ] 3.3: tight regime 확정(세 결론 중 하나), 3-regime 표 갱신
- [ ] 문서 5종 갱신, 산출물 JSON 커밋(provenance 포함), 로그·trace는 untracked

---

# 4. 전체 완료 조건

- [ ] `main`의 `serving/` 변경이 D25-a·b 두 곳뿐, R1·R2 byte-identical 증명이 두 PR 본문에 있음
- [ ] 64-way 동시 실행이 격리 wrapper 없이 64/64
- [ ] D22의 판정이 재검증 후 "유지" 또는 "변경(무엇으로)"로 명시됨
- [ ] tight-TTFT regime이 확정되어 3-regime 표가 닫힘
- [ ] D23 Resolved, D25 기록, CLAIMS.md 최신
- [ ] 게이트 3종 통과, golden 불변

# 5. 리스크 대응 규칙

| 상황 | 판단 |
| --- | --- |
| R1 또는 R2가 편집 후 byte-identical하지 않음 | 편집이 출력에 영향을 줬다. 머지하지 말고 원인을 찾는다. `cwd`가 어떤 상대 경로에 쓰이고 있었다는 뜻이면 그 경로를 절대화하는 것이 아니라 **어디인지 D25에 기록**하고 보고 |
| 3.2에서 과거 완료 후보와 수치가 다른 행이 있음 | 조용한 오염의 증거다. 해당 후보 두 CSV diff와 3.1 census를 대조해 어느 필드가 원인인지 적고, **§1.3의 Exp 2·Exp 5 재실행을 필수 항목으로 격상**해 `WORK_ORDER_rps_aware.md`에 넘긴다. D22 판정은 재실행 결과 기준으로 다시 쓴다 |
| 3.2에서 과거 타임아웃 후보가 완료되어 RNGD 후보 중 하나가 SLO를 통과 | D22의 "모든 RNGD 후보 거부"가 바뀐다. 결과다 — A2대로 D22·CLAIMS.md·§4.8.7을 갱신하고, "margin 3.3 % 이상이면 거부"가 그 후보에도 성립하는지 계산해 적는다 |
| 3.3에서도 타임아웃이 남음 | `livelock_watch.sh`로 재현. exit 3(틱 정지)이면 D23과 다른 새 결함 → 새 D 번호, 원인 추적은 다음 작업지시서. exit 4(자식 사망)면 D25-b의 메시지에 원인이 있어야 한다 — 없으면 D25-b가 불완전 |
| `build_cluster_config`가 `.venv`에서 import되지 않음 | 3.1은 json 규칙을 스크립트에 복제한다(출처 줄 번호 주석). 결론의 신뢰도를 한 단계 낮춰 기록 |
| 시뮬레이션 총 시간이 2일을 넘김 | 3.2는 반드시 끝내고, 3.3은 fixture 하나(card)만 끝낸 상태로 닫아 미결을 기록한다. 결과를 위해 `--num-requests`나 타임아웃을 조정하지 않는다 |
| upstream(`astra-sim/`, `AGENTS.md`, `CHANGELOG.md`) 수정이 필요해 보임 | 중단·보고. `docs/upstream_issues/`의 초안을 제출하는 것은 사용자 결정 |
