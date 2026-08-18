# HeteroPilot 진행 현황 및 인계 문서

> 작성일: 2026-08-18 · 기준 커밋: `a5d88f3` (main) · 기준 문서: `WORK_ORDER_heteropilot.md` v1.0
>
> 용도: 2×RTX A5000 개발 머신 → **A40×8 서버**로 소스 이전 시 인계.
> 이 문서는 요약이며, 충돌 시 상세 문서(`docs/deviations.md`, `docs/phase0_formats.md`,
> `docs/phase0_bench_plan.md`, `docs/hardware_roadmap.md`)와 작업지시서가 우선한다.

---

## 1. 한눈에 보는 현황

| 작업지시서 Phase | 상태 | 완료 조건 대비 |
| --- | --- | --- |
| Phase 0 — Baseline 고정 | ✅ 완료 | §7 조건 4/4. upstream `2c2042ce` pin, 예제 재현(바이트 동일), bench 3종 재현(커밋본과 바이트 동일), 형식 조사 문서화 |
| Phase 1 — Spec + Inventory | ✅ 완료 | `inspect-cluster`가 island/TP후보/호환성 출력, 테스트 통과 |
| Phase 2 — Offline Planner (MVP ★) | ✅ 완료 | §7 조건 6/6 (아래 §3 참조) |
| Phase 3 — Hetero Profiles | 🟡 부분 선행 | A5000 실측 프로파일 + 전력, `profiler/CONTRACT.md`(§3.7), mixed-replica 후보. **NPU importer 미구현** |
| Phase 4 — Real Deploy + Calibration | ⬜ 미착수 | 단, sim-vs-real 검증 데이터·방법론은 확보(§4 참조) |
| Phase 5 — Topology-aware P/D | ⬜ 미착수 | 순서상 정상 |
| Phase 6 — Online Replanning | ⬜ 미착수 | **착수 전 사용자 승인 필요** (작업지시서 명시) |

품질 게이트(§9): **pytest 140개 / ruff / mypy 전부 통과.** upstream 코드 무수정
(절대 규칙 1 준수 — D12 수정 시도 2회는 승인하에 진행했으나 실패하여 전량 원복, `serving/` pristine).

§13 즉시 착수 목록: 14항목 중 **13개 완료**, 미완 1개는 §5.5의 surrogate predictor
(§5.4 6단계 — Phase 2 후반 항목, 현재 규모에선 불필요해 보류).

---

## 2. 저장소 구조 (신규 추가분)

작업지시서 §2.1 구조를 따르되 실제 구현은 다음과 같다:

```
planner/                     # Control Plane (전부 신규)
├── __main__.py              # CLI: inspect-cluster, plan, validate-plan 동작 / deploy, status는 Phase 4
├── spec.py                  # ServiceSpec (§3.1)
├── inventory.py             # ClusterSpecV2, island 탐지, NodePower (§3.2, §5.2)
├── topology.py              # Level-1 모델 + 손실 압축 기록 (§5.3, D3)
├── candidate_generator.py   # §5.4 고정 순서 pruning + mixed-replica 열거
├── plan.py                  # CandidateConfig/DeploymentPlan/PlannerOutput (§3.4, §3.5)
├── envelope.py              # PerformanceEnvelope 파일 캐시 (§3.6, 키는 확장됨 — D13)
├── render.py                # §6 stdout 요건
├── predictor/
│   ├── __init__.py          # Predictor ABC + SimResult(OK/CRASHED/TIMEOUT/UNPARSEABLE)
│   └── llmservingsim.py     # config 컴파일러 + subprocess 러너 + 파서 (§5.5)
├── optimizer/
│   ├── feasibility.py       # hard constraints, robust margin 배관 포함 (§5.6, §5.8)
│   ├── pareto.py            # lexicographic 랭킹 + frontier (weighted-sum 금지 준수)
│   └── exhaustive.py        # 검색 드라이버 + oracle (절대 규칙 5: 삭제 금지)
└── util/
    ├── memory.py            # upstream memory_model 호출 + D10 derating
    ├── percentile.py        # 단일 percentile 유틸 (§4)
    ├── power_parse.py       # stdout 전력 파서 (D2)
    ├── provenance.py        # §3.8 metadata 수집
    └── workload.py          # traffic 분포 → JSONL trace (seed는 여기 — D5)

profiles/accelerators/       # a5000(실측), rtxpro6000, a40/rbln_atom/furiosa_rngd(스텁), ascend_target(스키마 예시)
profiler/perf/A5000/         # 로컬 실측 프로파일 번들 (Llama-3.1-8B bf16 tp1)
profiler/perf/RTXPRO6000X2/  # 파생 아티팩트 (밀도 대조군, 측정 아님 — D11)
profiler/CONTRACT.md         # §3.7 CSV 계약 (NPU importer 기준)
examples/                    # 서비스 스펙 3종 + heterogeneous-lab 클러스터
experiments/                 # Exp 2 설정·스크립트·결과, A5000 전력 원시 데이터
tests/                       # 140개 (oracle 일치·재현성·golden 요건 충족)
docs/                        # phase0_formats, phase0_bench_plan, deviations(D1–D14), hardware_roadmap, 본 문서
```

## 3. Phase 2 MVP — 완료 증거 (§7 완료 조건 대비)

1. **후보 자동 생성** — 예제 클러스터에서 54개 (단일 island 30 + mixed 24)
2. **후보별 LLMServingSim 결과** — 54/54 실제 시뮬레이션 (cold cache 히트 0 검증)
3. **SLO/power/tokens-J 자동 검증** — §5.6 hard constraints, 측정 불가 제약은 통과가 아닌 노트로 처리
4. **최적 plan + Pareto 대안 / infeasible 진단** — closest_plan + violated_constraints + 규칙 기반 suggestions
5. **oracle 일치** — `--oracle` 모드와 pruning 모드가 동일 최적해 (테스트로 상시 검증)
6. **재현성** — 동일 spec+seed → 바이트 동일 출력 (실행으로 검증, 테스트로 고정)

주요 실험 결과 (전부 **시뮬레이션 예측값**, 실측 아님):

- **Exp 1** (TP 스윕, 300req): TP=2가 DP=2를 전 지표에서 지배. `outputs/plans/llama31-8b-plan-300.yaml`
- **Exp 2** (이기종 자원 선택, 54후보): RTXPRO6000 1장 1.697 tok/J > A5000 2장 1.634 (−3.7%, 오차 대역 내)
  > mixed 1.043 (−39%). 수요가 한 클래스에 담기면 right-sizing이 scale-out을 이김.
  `experiments/results/exp2_summary.md`
- **목적함수 스터디**: 유한 폐쇄 트레이스에서 goodput/J ≡ 1/E로 퇴화 (attainment가 갈릴 때만 변별)

## 4. sim-vs-real 검증 실적 (Phase 4의 기반)

| 조건 | 평균 절대오차 (15지표) |
| --- | ---: |
| RTXPRO6000, 전체 그리드 (upstream 재현) | 1.23% |
| RTXPRO6000, ×2 그리드 (밀도 대조군, D11) | 3.05% |
| A5000, nominal `mem_size: 24` | 22.54% |
| **A5000, KV-matched `mem_size: 20.81`** | **9.26%** |

핵심 교훈: **메모리 회계(D10)가 메모리 제약 장치에서 최대 단일 오차 항** (−13.3pp).
플래너는 `gpu_memory_utilization`(기본 0.9)과 `activation_reserve_gb`를 명시적으로 derating한다.
A5000 실측 vLLM 데이터 2세트(prefix on/off)는 `outputs/phase0_bench/A5000*/vllm/`에 커밋됨.

## 5. 반드시 알아야 할 괴리 (deviations.md D1–D14 중 핵심)

| ID | 내용 | 상태 |
| --- | --- | --- |
| D2 | 전력/에너지는 stdout에만 존재 → `power_parse.py`로 파싱 (golden 테스트 고정) | 결정 |
| D3 | cluster config에 링크 그래프 없음 → Level-1 손실 압축, provenance에 `path_aware: false` 기록 | 결정 (Phase 5에서 Level-2) |
| D5 | 시뮬레이터에 `--seed` 없음 → 재현성은 workload 생성기 소유 | 해결 |
| D6 | `outputs/example_*_run.csv`는 **stale** — golden으로 쓰지 말 것. `bench/examples/`가 안전한 앵커 | 해결 |
| **D10** | 시뮬레이터 메모리 모델에 utilization/예약 없음 → 24GB 카드에서 KV +55~71% 과대평가 | 해결+실증 |
| D11 | 프로파일 그리드 밀도 ≈ 2.2pp 정확도. `meta.yaml`은 그리드를 과소 기술 — CSV 키가 진실 | 정량화 |
| **D12** | **prefix cache 메모리 단조 증가 → 포화 장치에서 런 사망. 미해결.** Phase 2는 전 후보 prefix caching OFF (모든 출력에 caveat 동반). 승인받은 수정 시도 2회 실패·원복 — 재시도 전 D12 기록 필독 | **미해결** |
| D13 | §3.6 envelope 키에 dp 누락 → 충돌 사고. 키를 전체 배치 서술로 확장 | 해결 |
| D14 | 시뮬레이터 토폴로지 추론이 균일 인스턴스 크기 가정 → mixed는 replica당 장치 수 동일 조합만 열거 | 해결 (제약 회피) |

**깨지기 쉬운 불변식 2개** (CLAUDE.md에도 기재):
pruning 단계는 feasibility의 완화여야지 추가 조건이면 안 됨(처리량 bound 사건) /
mock predictor는 bound와 같은 물리를 따라야 함.

## 6. A40 서버에서의 환경 구축

```bash
git clone <this-repo> && cd heteropilot
git submodule update --init --recursive        # astra-sim (~30분, 572MB+)

# 시뮬레이터 venv (bare-metal — planner가 subprocess로 띄우므로 Docker 아님)
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install pyyaml pyinstrument transformers datasets msgspec scikit-learn \
  xgboost==3.1.2 matplotlib==3.5.3 pandas==1.5.3 numpy==1.23.5 rich \
  "pydantic>=2" pytest ruff mypy types-PyYAML
bash scripts/compile.sh                        # ASTRA-Sim 빌드 (cmake/g++/protoc 필요)
uv pip install ./astra-sim/extern/graph_frontend/chakra   # 함정 1: compile.sh의 pip3가 venv를 놓침
uv pip install "protobuf>=7.35.1"                          # 함정 2: Chakra gencode 버전
pytest && ruff check planner/ tests/ && mypy   # 140 passed가 나와야 정상

# vLLM venv (별도! — 함정 3: scripts/install-vllm.sh는 CWD에 venv를 만들어 .venv를 덮어씀)
uv venv --python 3.12 .venv-vllm
VLLM_USE_PRECOMPILED=1 uv pip install --python .venv-vllm/bin/python vllm==0.19.0 --no-build-isolation
uv pip install --python .venv-vllm/bin/python datasets matplotlib pandas
```

함정 4: 시뮬레이터에 넘기는 모든 경로는 **저장소 루트 기준 상대경로**여야 한다
(`serving/__main__.py:199`가 astra-sim으로 chdir 후 `../` 접두). predictor는 이를 처리하므로
`--work-dir`를 저장소 안(`outputs/…`)에 두면 된다.

## 7. A40 서버에서의 작업 순서 (docs/hardware_roadmap.md 상세판의 요약)

**1단계 — 인벤토리 (첫날, 코드 작성 전):**
`nvidia-smi -L`, `nvidia-smi topo -m`(NVLink 브리지 여부가 island 구조를 결정),
NIC/패브릭 조사 → 실측 기반 `ClusterSpecV2` YAML 작성 (모든 필드에 `source:` 표기, 절대 규칙 3).
`python -m planner inspect-cluster`로 island 탐지 확인.

**2단계 — A40 프로파일링 (기존 프로파일러 그대로 동작, CUDA):**
```bash
CUDA_VISIBLE_DEVICES=0 .venv-vllm/bin/python -m profiler profile \
  meta-llama/Llama-3.1-8B --hardware A40 --tp 1,2,4,8 \
  --max-num-batched-tokens 2048 --max-num-seqs 256 --measurement-iterations 3
# Qwen/Qwen3-32B 동일 (작업지시서 대표 모델 — 48GB에서 TP=2로 서빙 가능)
```
소요 예상: TP당 attention ~1h + skew 1–2h (A5000 실적 기준). resume 모드라 중단 후 재개 가능.
완료 후 `profiles/accelerators/a40.yaml`에 `sim_hardware: A40` 기입 (스텁에 이유 주석 있음).

**3단계 — A40 전력 실측 (D7):** A5000 때의 절차 재사용 —
`nvidia-smi --query-gpu=power.draw -lms 100` 폴링 + bench 부하, idle/active/standby 산출.
원시 데이터는 `experiments/results/`에 커밋. **호스트 전력도 측정**(IPMI 등) — Exp 2 mixed
페널티의 크기가 placeholder 호스트 전력에 걸려 있음.

**4단계 — sim-vs-real 재검증 (Phase 4 진입):** A5000 프로토콜 반복
(`docs/phase0_bench_plan.md`) — bench 실측 → KV 예산 재측정(48GB에서 D10 재확인) →
nominal/KV-matched 비교. 그다음 TP=2/4, 멀티 노드.

**5단계 — Phase 4 본체:** `deploy/vllm_cuda.py`(§5.7 — local/SSH만),
`monitor/metrics.py`, `calibration.py`(§5.8 — 선형 보정. 주의: A5000은 과소예측,
RTXPRO6000은 과대예측으로 **부호가 반대** — 하드웨어별 보정 필수).
서버의 스케줄러(SLURM 여부)가 launcher 설계에 영향 — 확인 필요.

**병행 가능 — NPU (ATOM/RNGD) V1:** 서빙 스택(vllm-rbln / furiosa-llm) 버전·모델 지원
실기 확인 → 측정 → `CsvProfileImporter` 구현(`profiler/CONTRACT.md` 준수) →
스텁 프로파일의 `supported_models` 채움 (빈 목록 = 후보 제외가 현재 의도된 동작).

## 8. 미해결 사안 (사용자 결정 필요)

1. **D12** — prefix cache 포화 사망. 선택지: upstream 이슈 제기 / 계측 기반 KV 생애주기
   조사 / 현행 유지(prefix off). 실측 prefix hit 7.4% 워크로드가 있으므로 방치 비용 있음.
2. **원격 저장소** — push할 origin이 없음. fork 생성 후 §10.1의 feat/* + PR 흐름 적용 가능.
3. **Phase 6** — 착수 전 명시적 승인 필요 (작업지시서).
4. **surrogate predictor** (§5.4 6단계) — 후보 수가 수백 규모가 되면(A40×64 조합) 필요해짐.
   A40 8장×8노드에서는 탐색 공간이 커지므로 우선순위 재평가 권장.

## 9. 범위 제한 리마인더 (작업지시서 §8 — 발견 시 중단·보고)

GPU+NPU mixed TP · dynamic migration · multi-tenancy · RL · Kubernetes operator ·
cross-vendor P/D(Phase 5 전) · full switch-level congestion. 지금까지 위반 없음.
Ascend 스텁은 스키마 예시로만 존치 (실 NPU 타깃은 ATOM/RNGD로 변경 — D4).
