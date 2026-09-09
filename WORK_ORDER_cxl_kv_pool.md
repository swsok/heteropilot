# 연구 노트 겸 작업지시서 — CXL 메모리 풀을 이기종 가속기의 KV 교환·보관 계층으로 쓰는 배치 계획

> HeteroPilot의 다음 연구 축 후보. 2026-09-09 논의 정리.
> 대상 저장소: `github.com/swsok/heteropilot`. 실험 서버: 부서 CXL 서버(삼성 CXL Type 3, 256 GB, CPU 없는 NUMA 노드로 인식). 서버 사용 가능 시 실행 예정.
> 상태: **설계·계획 단계. 아래 수치 중 실측값은 없다.** 실측 전에는 어떤 숫자도 결과로 인용하지 않는다(HeteroPilot 절대 규칙 3).

---

## 0. 한 문단 요약

가속기가 CXL.mem을 직접 로드/스토어하는 하드웨어는 당분간 없다. 그러나 CXL Type 3 확장기는 호스트에 메모리 전용 NUMA 노드로 보이므로, GPU·NPU가 이미 사용하는 **호스트 메모리 DMA 경로**(cudaMemcpy D2H/H2D, RNGD 호스트↔PE 전송)를 그 노드로 향하게 하면 CXL 메모리를 **이기종 가속기가 공유하는 벤더 중립 KV 캐시 풀**로 쓸 수 있다. 크로스 벤더 P/D 핸드오프에서 호스트 측 memcpy 한 번이 사라지고 DRAM 용량을 소모하지 않으며, 256 GB는 다중 턴 프리픽스 재사용 풀로도 충분하다. HeteroPilot에는 **KV 계층(HBM / DRAM / CXL 풀)** 이 후보 축으로, **계층 경계 통과 바이트 하한(S5″)** 이 프루닝 단계로, **CXL 대역폭 경합 엔벨로프**가 특허 2의 정확도 도메인 변수로 들어간다. 특허는 "CXL에 KV를 둔다"가 아니라 **핸드오프 경로·계층을 측정 엔벨로프로 계획하는 방식**과 **CPU 없는 풀에서의 레이아웃 변환 배치**에서 나온다.

---

## 1. 배경과 동기

### 1.1 HeteroPilot에서 확인된 사실 (실측, 커밋 산출물)

| 사실 | 값 | 출처 |
| --- | --- | --- |
| A40 D2H 지속 대역폭 (pinned, 단일 스트림) | 25.71 GB/s; 8 GPU 합산 82.63 GB/s | `experiments/results/gpu_host_bandwidth.md` |
| RNGD 호스트↔PE 지속 대역폭 | 3.77 GB/s 단일, 26.27 GB/s 8스트림 | `experiments/results/rngd_parallel_bandwidth.md` |
| 크로스 벤더 KV 경로(GPU→호스트→NPU) 합성 대역폭 | 12.6–13.0 GB/s (35 GB/s 플레이스홀더는 2.7–4.5× 낙관) | 같은 문서 |
| P/D 채택 임계 fabric 대역폭 | ~10 GB/s (해석식 `KV_p99/(SLO−TTFT_base)`) | `exp_pd_summary.md` |
| Llama-3.1-8B bf16 요청당 KV | 토큰당 128 KB (32계층 × 8 KV헤드 × 128 × 2 B × 2) → 1K 토큰 = 128 MB | 모델 config에서 산출 |

핵심 관찰: 크로스 벤더 핸드오프의 병목은 두 번의 호스트 복사와 DRAM 경합이며, 이 경로의 대역폭(13 GB/s)이 P/D 채택 임계(10 GB/s)에 가깝다. 호스트 측 복사를 한 번으로 줄이고 경합을 분리하면 임계를 넘길 여지가 있다.

### 1.2 왜 CXL 풀인가

- **DMA 대상으로서의 CXL**: Type 3 메모리는 시스템 물리 주소 공간에 매핑되므로, 그 주소 범위에 잡은 버퍼를 `cudaHostRegister`로 고정하면 GPU DMA가 동작한다(PCIe → 루트 컴플렉스 → CXL 링크). NPU도 호스트 버퍼 DMA는 같은 원리. **직접 접근이 아니라 DMA 목적지**라는 점이 이 연구의 전제이자 한계다.
- **CPU 없는 노드**: 풀 트래픽이 일반 부하와 섞이지 않아 풀의 성질(대역폭·경합·지연)만 분리 측정 가능. 반면 풀 위에서 CPU 연산(레이아웃 변환)을 하려면 원격 노드 CPU가 읽어야 하므로 비효율 → **변환을 가속기 DMA에 흡수**해야 한다(§5 발명 요소).
- **용량**: 256 GB ≈ Llama-8B 기준 2,000개 세션 × 1K 토큰 프리픽스. 다중 턴 에이전트 워크로드의 재사용 풀로 의미 있는 크기.

### 1.3 이 연구가 아닌 것 (범위 제외)

- 가속기의 CXL 직접 접근(CXL.cache/Type 2) 가정. 로드맵 의존.
- 삼성/SK hynix가 이미 시연한 "vLLM KV를 CXL로 오프로드" 자체의 재현. 우리는 **이기종 핸드오프와 배치 계획**이 대상.
- Mooncake/LMCache 수준의 범용 KV 저장 계층 구현. 필요한 만큼의 최소 풀 API만 만든다.

---

## 2. 연구 프레임과 가설

**연구 질문.** 이기종 가속기 클러스터에서 KV 캐시를 어느 계층(장치 HBM / DRAM / CXL 풀)에 두고, 프리필→디코드 핸드오프와 다중 턴 프리픽스 재사용을 어느 계층을 경유시켜야 SLO와 tok/J가 최적인가. 그리고 그 선택이 측정되지 않은 CXL 경합 특성에 얼마나 민감한가.

**가설.**
- H1. GPU→CXL 풀→NPU 핸드오프는 GPU→DRAM→NPU 대비 호스트 복사 1회가 줄어 종단 핸드오프 시간이 짧거나 같고, DRAM 대역폭 경합을 유발하지 않는다.
- H2. CXL 링크 대역폭(장치당 실효 수십 GB/s)은 가속기 1~2장으로 포화하므로, 여러 가속기가 동시에 핸드오프하면 처리량이 포화·지연이 급증하는 **경합 엔벨로프**가 존재하며, 이는 KV 계층 선택을 동시성의 함수로 만든다.
- H3. 다중 턴 워크로드에서 CXL 풀 프리픽스 재사용은 프리필 재계산을 줄여 TTFT와 에너지를 동시에 개선하되, 풀 읽기 시간이 프리필 재계산 시간보다 짧은 프롬프트 길이 구간에서만 이득이다(교차점 존재).
- H4. 위 세 효과를 플래너에 넣으면 특정 SLO·요청률 구간에서 추천 계획이 바뀐다(측정 계획 발명의 정례).

---

## 3. 실험 사다리

각 단계는 앞 단계의 산출물에 의존한다. 첫 단계는 하루, 전체는 3–4주 규모. 모든 결과는 `experiments/results/cxl_*.md` + JSON, provenance(§3.8) 포함.

### E0. 서버 준비와 사실 확인 (반나절)

목표: 커널·드라이버·토폴로지를 기록하고 CXL 노드를 식별한다.

```bash
uname -r                       # CXL Type 3 NUMA 인식은 6.x 계열 기대. 5.4였던 기존 노드와 다름을 기록
numactl -H                     # CPU 없는 노드 번호 N_cxl, 용량 256 GB 확인
lscpu | grep -i numa
ls /sys/bus/cxl/devices/       # mem0, decoder, region 등
cat /sys/bus/cxl/devices/region*/size 2>/dev/null
daxctl list 2>/dev/null; ndctl list 2>/dev/null
lspci -vv | grep -i -A3 cxl    # 링크 폭·속도 (x8/x16, Gen5) → 이론 대역폭 산출
nvidia-smi topo -m             # GPU가 어느 루트 컴플렉스/NUMA에 붙었는지
lspci -tv                      # CXL 장치와 가속기의 PCIe 트리 위치 — 같은 루트 컴플렉스인지
```

기록: `docs/nodes/cxl.md` 신설 (기존 `docs/nodes/a40.md` 형식). **함정**: `scripts/whichnode.sh`가 이 노드를 모른다 → CXL 노드 검출 분기 추가. 기존 노드들의 hostname이 모두 `s8`이었던 문제를 기억하고 hostname 대신 `/sys/bus/cxl` 존재로 판별.

**조사 필요**: A40 드라이버와 Furiosa 런타임이 이 커널에서 동작하는지. 안 되면 E1 이전에 해결.

### E1. 마이크로벤치 — GPU ↔ CXL 노드 DMA (하루)

목표: CXL 노드에 잡은 pinned 버퍼로의 D2H/H2D 지속 대역폭·지연을 DRAM 노드와 비교. 포화 곡선 획득.

절차:
1. 버퍼 할당을 NUMA로 제어. `numactl --membind=N_cxl --cpunodebind=<GPU 인접 CPU 노드> python bench.py`. 프로세스 전체가 CXL에 잡히는 부작용을 피하려면 `numa.h`/`libnuma`(`numa_alloc_onnode`)로 대상 버퍼만 할당.
2. GPU 등록: `cupy.cuda.runtime.hostRegister(ptr, nbytes, 0)` (또는 torch `cudaHostRegister` via ctypes). `pin_memory()`는 새로 할당·복사하므로 **쓰지 않는다**(NUMA 위치가 바뀜).
3. 측정: 크기 1 MB ~ 4 GB 스윕, 단일 스트림 / 다중 스트림(1,2,4,8), D2H·H2D, 지속(반복 평균) 값. 기존 `experiments/scripts/gpu_host_bandwidth.py` 방법론 재사용(지속 vs 피크 구분, D18 교훈).
4. 대조군: 같은 스크립트를 `--membind=<DRAM 노드>`로.
5. 기준선: CPU에서 CXL 노드 STREAM/`mlc` 대역폭·지연(가속기 없이 링크 자체 한계).

지표: GB/s(지속), µs(지연, 4 KB 전송), 포화 스트림 수. 성공 기준: 곡선이 재현되고(3회 편차 < 5 %), 이론 링크 대역폭 대비 실효 비율이 기록됨.

**함정**: (a) GPU와 CXL 장치가 다른 루트 컴플렉스에 있으면 UPI/인피니티 패브릭을 건너 대역폭이 반토막 — `lspci -tv`로 먼저 확인하고 슬롯을 고른다. (b) 커널 CXL 메모리를 `movable` 존으로 올리면 pinning이 실패할 수 있음 — `daxctl reconfigure-device --mode=system-ram` 옵션과 `memmap`/`nosoftreserve` 설정 확인. (c) 첫 접근 페이지 폴트 비용 → 워밍업 후 측정.

### E2. 마이크로벤치 — RNGD ↔ CXL 노드 (하루)

목표: RNGD 호스트↔PE 전송을 CXL 노드 버퍼로 반복. `rngd_parallel_bandwidth.md`의 방법(1/8 스트림, 지속) 재사용.

**조사 필요**: Furiosa 런타임이 호스트 버퍼 NUMA 위치를 지정할 수 있는가. 없으면 `numactl --membind` 프로세스 전체 바인딩 → 런타임 메타데이터·컴파일 아티팩트도 CXL에 올라가 오염. 이 경우 (i) `--preferred` 대신 `--membind`로 강제한 뒤 메타데이터 오버헤드를 별도 측정해 빼거나, (ii) `LD_PRELOAD`로 특정 크기 이상의 `mmap`만 CXL 노드로 보내는 얇은 셤을 쓴다. 어느 쪽이든 오염 여부를 `numastat -p <pid>`로 확인.

지표·성공 기준: E1과 동일. 추가로 GPU와 RNGD가 **동시에** CXL을 칠 때의 합산 대역폭(E4의 예비).

### E3. 핸드오프 실험 — A40 프리필 → CXL 풀 → RNGD 디코드 (일주일)

목표: 크로스 벤더 KV 핸드오프를 세 경로로 비교하고 H1 검증.

경로:
- P-DRAM: GPU D2H(DRAM) → CPU memcpy/레이아웃 변환 → NPU H2D. 현재 방식.
- P-CXL-cpuconv: GPU D2H(CXL) → 원격 CPU가 CXL에서 읽어 변환 후 CXL에 다시 씀 → NPU H2D. (CPU 없는 노드의 대가를 보이는 대조군)
- P-CXL-zerocopy: GPU가 **소비자 레이아웃으로** CXL에 기록 → NPU가 그대로 H2D. 또는 GPU가 자기 레이아웃으로 기록 → NPU가 gather DMA로 변환하며 읽기. **핵심 후보.**

전제 작업:
- 레이아웃 정의: vLLM CUDA paged KV 블록 레이아웃 vs Furiosa 런타임 KV 레이아웃 조사(**조사 필요** — furiosa-llm의 KV 버킷/블록 배열, dtype). 변환이 순열+dtype 캐스트인지, 재패킹이 필요한지에 따라 "생산자 측 소비자 레이아웃 기록"의 가능 여부가 갈린다.
- 최소 풀 API: `pool.alloc(nbytes, numa=N_cxl) -> (ptr, handle)`, `pool.register_cuda(handle)`, `pool.export(handle) -> fd/offset` (다른 프로세스·런타임이 같은 물리 버퍼를 열 수 있게 — `memfd`/`hugetlbfs`+`mbind` 또는 `dax` 장치 파일 mmap), `pool.free`. 두 프로세스(vLLM 워커, furiosa-llm 워커)가 **같은 물리 페이지**를 보는지 `/proc/<pid>/pagemap`으로 검증.
- 서빙 엔진 연동은 **최소로**: 실제 vLLM/furiosa-llm 내부에 끼우기 전, 두 독립 프로세스가 (i) 프리필을 돌려 KV를 뽑고 (ii) 풀에 쓰고 (iii) 다른 쪽이 읽어 디코드를 이어가는 **오프라인 하네스**로 시작. 정합성은 디코드 출력 토큰 일치(동일 시드, greedy)로 검증.

지표: 핸드오프 시간(프리필 종료 → 디코드 첫 스텝 시작), 경로별 바이트 이동 횟수, 호스트 CPU 사용, DRAM 대역폭 소모(`pcm-memory` 또는 `perf`), 전력(노드 전력 측정 방법론 재사용). 요청 길이 {256, 1K, 4K, 16K} 토큰.

성공 기준(H1): P-CXL-zerocopy 핸드오프 시간 ≤ P-DRAM, DRAM 대역폭 소모 유의하게 감소. 실패해도 **왜**(CXL 지연? 페이지 폴트? 변환 비용?)가 분해되어 기록되면 성공.

### E4. 경합 엔벨로프 (2–3일)

목표: 가속기 k대가 동시에 풀을 사용할 때의 처리량·지연 곡선(H2). 특허 2의 정확도 도메인 데이터 형식으로 저장.

절차: E1/E2 하네스를 k ∈ {1,2,3,4}(A40 2 + RNGD 1~2)로 동시 실행, 총 GB/s와 각 장치 p99 전송 지연. 읽기/쓰기 혼합 비율 {100/0, 50/50, 0/100}.

산출물: `profiles/links/cxl_pool.yaml` — `sustained_bw_gbps(k)`, `latency_us(k)`, `validity: {k_min, k_max}`, `source: measured`. 이 파일이 §4의 `LinkKind.CXL_POOL` 프로파일이 된다.

### E5. 플래너 통합과 결정 민감도 (일주일)

목표: KV 계층 축과 S5″ 하한을 HeteroPilot에 넣고, CXL 경합 엔벨로프가 추천을 바꾸는 SLO·요청률 구간을 찾는다(H4). 상세는 §4.

산출물: `experiments/results/cxl_plan_sensitivity.md` — 계층별 추천표, 전환점, 엔벨로프 불확실성에 대한 flip 여부(특허 2 §5.5 격자 섭동 재사용).

### E6 (선택). 프리픽스 재사용 풀 (2주)

목표: 다중 턴 워크로드에서 CXL 풀 프리픽스 히트 시 TTFT·에너지 이득과 교차점(H3). 워크로드: ShareGPT 다중 턴을 세션 단위로 재구성. 풀 읽기 시간 vs 프리필 재계산 시간의 교차 프롬프트 길이를 A40·RNGD 각각에 대해 구한다. 이 결과는 세 번째 연구 축(에이전트 워크로드)의 입구가 된다.

---

## 4. HeteroPilot 통합 설계

### 4.1 스키마

| 위치 | 추가 | 의미 |
| --- | --- | --- |
| `ClusterSpecV2.nodes[].memory_tiers[]` | `{id, kind: dram|cxl_pool, numa_node, capacity_gb, profile}` | 노드의 호스트 측 메모리 계층 |
| `LinkKind` | `CXL_POOL` (가속기 ↔ 풀 경로; 실효 대역폭은 E1/E2 실측) | 링크 클래스 |
| `profiles/links/cxl_pool.yaml` | E4 엔벨로프 | 동시성 의존 대역폭·지연 |
| `IslandAssignment.kv_tier` | `hbm | dram | cxl_pool` (디코드 KV 상주 계층; P/D 핸드오프 경유 계층은 `CandidateConfig.handoff_via`) | 후보 축 |
| `ServiceSpec.workload.reuse` | `{multi_turn: bool, prefix_hit_ratio_prior}` | E6용, 기본 없음 |

### 4.2 후보 생성

- P/D 분리 후보에 `handoff_via ∈ {dram, cxl_pool, network}` 축 추가 (경로가 클러스터 명세에 있을 때만).
- 집약형·디코드 후보에 `kv_tier` 축 추가. `hbm`이 기본이고 나머지는 opt-in(`--kv-tiers`). 골든 출력 불변.

### 4.3 하한 S5″ — 계층 경계 통과 바이트

디코드 스텝당 KV 계층 경계를 **반드시** 통과하는 바이트로 하한을 둔다. 특허 1 v4 §5.5의 규칙을 그대로 따른다: A·K는 **최소 보장치**(요청 1개, 최소 입력 길이), 대역폭은 엔벨로프의 **최대값**(k=1), 지연은 최소값.

```
kv_tier = cxl_pool 인 디코드 할당:
  bytes_min   = kv_bytes_per_token × tokens_min(요청 1개, 최소 입력 길이)   # 스텝마다 풀에서 읽어야 하는 최소량
  floor_ms    = lat_min_us/1e3 + bytes_min / bw_max(cxl_pool)
  reject if floor_ms > slo.tpot.max_ms      (Role.PREFILL 면제)
P/D 핸드오프(handoff_via = cxl_pool):
  시뮬레이션 후 TTFT 가산 항 = KV_p{50,95,99} / bw(k_expected) + lat   # 기존 kv_transfer.py 확장, 하한이 아님
```

S5″는 S5(HBM 루프라인)와 **직렬 성분이 아니므로 합산하지 않고 각각 SLO와 비교**(v3의 max/sum 규칙). 오라클 일치 테스트에 `kv_tier` 픽스처 추가.

### 4.4 시뮬레이터 표현력 (조사 필요 → deviations D27)

업스트림 LLMServingSim은 CPU/CXL/PIM 메모리 계층을 모델링한다고 문서화되어 있다(`README`: "disaggregated memory tiers (CPU / CXL / PIM)"). **확인할 것**: (i) 계층별 KV 상주를 인스턴스 설정으로 지정할 수 있는지, (ii) 계층 대역폭 파라미터가 어디에 있는지, (iii) 두 인스턴스가 하나의 계층을 공유하는 경합이 모델되는지. 안 되는 부분은 플래너 후처리로 근사하고 가정 기록. E4 엔벨로프로 시뮬레이터 계층 파라미터를 보정하는 것이 곧 특허 2의 정확도 도메인 첫 사례.

### 4.5 특허 2와의 연결

CXL 엔벨로프는 처음에는 `placeholder`(E4 전) → `measured`(E4 후)로 등급이 올라가는 불확실 입력의 전형이다. E5에서 "엔벨로프를 오차 범위 내에서 섭동했을 때 추천이 뒤집히는가"를 보이면 측정 계획 발명의 실증 사례가 된다. 즉 이 작업은 특허 2의 효과 절 데이터를 함께 만든다.

---

## 5. 특허 아이디어 (실험 결과에 따라 결정)

### 5.1 후보 P-A: 이기종 가속기 간 KV 핸드오프 경로·계층의 엔벨로프 기반 배치 계획

- 청구 골격: 클러스터 명세에 호스트 부착 확장 메모리 풀을 메모리 계층으로 선언 → 프리필/디코드 후보에 핸드오프 경유 계층과 디코드 KV 상주 계층을 후보 축으로 열거 → 계층 경계 통과 최소 바이트로 디코드 하한을 산출(완화 조건) → 측정된 동시성 의존 엔벨로프로 핸드오프 시간을 예측 지표에 가산 → SLO·에너지로 순위화. 종속항: 엔벨로프의 유효 범위 밖은 "미측정" 거절; 엔벨로프 등급(플레이스홀더/측정) 전파.
- 위치: 특허 1·2의 종속항으로 흡수 가능, 또는 소형 독립 출원.
- 선행 위험 **중**: Mooncake/LMCache(호스트 계층), 삼성·SK hynix CXL KV 오프로드 데모(2025), CXL 메모리 풀 KV 논문들. "계획 축 + 완화 보장 하한 + 엔벨로프 등급 전파"가 차별점. **E4·E5 결과 없이는 출원하지 않는다.**

### 5.2 후보 P-B: CPU 없는 공유 메모리 풀에서 벤더 간 KV 레이아웃 변환의 배치

- 청구 골격: 생산 가속기(프리필)가 KV를 공유 풀에 기록하고 소비 가속기(디코드)가 읽되, 풀에 연산 능력이 없으므로 레이아웃·dtype 변환을 (a) 생산자가 소비자 레이아웃으로 기록(생산 측 변환), (b) 소비자가 gather/scatter DMA 서술자로 읽으며 변환(소비 측 변환), (c) 두 레이아웃의 공통 상위 블록 단위로 기록하고 하위 순열만 소비 측에서 처리(혼합) 중 **레이아웃 거리와 두 장치의 DMA 특성(서술자 수 한계, 최소 버스트)으로 선택**하는 방법. 종속항: 선택 결과를 배치 계획의 핸드오프 비용에 반영.
- 위치: 하드웨어 냄새가 있어 순수 소프트웨어 선행에 덜 흔들림. ETRI 권리행사 용이.
- 선행 위험 **중**: arXiv 2509.17542(멀티벤더 P/D의 레이아웃 변환 — CPU/RDMA 기반), Gimlet/Moreh 크로스 벤더 보고(호스트 경유 변환). "CPU 없는 풀 + 변환 위치 선택 규칙"이 차별점. **E3의 P-CXL-zerocopy가 실제로 동작한 뒤에만** 출원.

### 5.3 출원 순서

특허 2(불확실성 인식 플래너)를 먼저 내고, E3–E5 결과가 나오면 P-A/P-B를 국내우선권 기간 안에 추가 출원 또는 특허 2의 우선권주장출원에 실시예로 편입. 이 문서와 실험 로그는 착상 시점 증거로 보존.

---

## 6. 선행기술 확인 목록 (실험 전 반나절)

조사 없이 적은 항목이므로 실험 시작 전 확인한다.

| 항목 | 확인할 것 |
| --- | --- |
| 삼성 CMM-D / SK hynix CMM CXL KV 오프로드 데모·논문 (2024–2026) | vLLM 통합 방식, 이기종 가속기 다루는지, 계획 요소 유무 |
| Mooncake Transfer Engine, LMCache, NIXL | CXL/NUMA 계층 지원 여부, 크로스 벤더 레이아웃 변환 위치 |
| arXiv 2509.17542 (멀티벤더 P/D), HMA-Serve (2606.29986) | 변환이 어디서 이루어지는지(CPU? 장치?) |
| CXL 메모리 풀 위 LLM 추론 논문 (예: "CXL-based KV cache", "Pond", "TPP" 계열) | 티어링 정책과 우리 계획 축의 차이 |
| vLLM `--kv-transfer-config`, `LMCacheConnector`, CPU offload 옵션 (v0.28.0) | 호스트 버퍼의 NUMA 제어 가능 여부 — E3 연동 방식 결정 |
| furiosa-llm KV 레이아웃·호스트 버퍼 API | E2/E3 전제 |

---

## 7. 서버 사용 전 체크리스트

- [ ] 커널 버전·CXL 노드 번호·용량·링크 폭 기록 (`docs/nodes/cxl.md`)
- [ ] A40 2장(NVLink 쌍) + RNGD 1장 장착 위치가 CXL 장치와 같은 루트 컴플렉스인지 `lspci -tv`로 확인. 아니면 슬롯 재배치 요청
- [ ] NVIDIA 드라이버·Furiosa 런타임이 해당 커널에서 로드되는지
- [ ] `libnuma`, `numactl`, `daxctl/ndctl`, `pcm`(또는 `perf` uncore) 설치
- [ ] CXL 메모리 존이 pinning 가능한지(`cudaHostRegister` 소형 테스트)
- [ ] `scripts/whichnode.sh`에 CXL 노드 분기 추가
- [ ] 다른 사용자와 시간 분리 — 경합 실험(E4)은 서버 단독 점유 필수
- [ ] 모든 산출물에 provenance(커밋, 커널, NUMA 맵, 장치 인벤토리) 포함

---

## 8. 위험과 대응

| 위험 | 징후 | 대응 |
| --- | --- | --- |
| GPU DMA→CXL 대역폭이 DRAM 대비 크게 낮음(루트 컴플렉스 교차, 링크 폭) | E1에서 < 50 % | 슬롯 재배치; 그래도 낮으면 "핸드오프 경유"보다 "보관 계층(E6)"으로 연구 초점 이동 |
| Furiosa 런타임이 호스트 버퍼 NUMA 제어 불가 | E2 오염 | `LD_PRELOAD` mmap 셤; 최악의 경우 RNGD는 DRAM 경유로 두고 GPU↔GPU 세대 간 핸드오프로 P-B 검증 |
| 두 런타임이 같은 물리 버퍼를 공유 못 함 | E3 zero-copy 불가 | `dax` 장치 mmap 또는 hugetlbfs + `mbind`; 그래도 안 되면 풀 내부 1회 복사(CXL→CXL)로 절충하고 비용 기록 |
| 시뮬레이터가 계층 공유 경합을 표현 못 함 | §4.4 조사 | 플래너 후처리 근사 + deviations D27 기록 |
| 선행기술이 P-A/P-B를 이미 덮음 | §6 조사 | 청구를 "엔벨로프 등급 전파·완화 보장 하한" 쪽으로 좁힘, 또는 특허 2 종속항으로만 편입 |

---

## 부록 A. E1 벤치 스크립트 골격 (검증 전, 실행 시 수정 필요)

```python
# experiments/scripts/cxl_gpu_bandwidth.py  (sketch)
import ctypes, time, numpy as np, cupy as cp
libnuma = ctypes.CDLL("libnuma.so.1")
libnuma.numa_alloc_onnode.restype = ctypes.c_void_p
def alloc_on_node(nbytes, node):
    p = libnuma.numa_alloc_onnode(ctypes.c_size_t(nbytes), ctypes.c_int(node))
    assert p, "numa_alloc_onnode failed"
    cp.cuda.runtime.hostRegister(p, nbytes, 0)          # pin in place; do NOT use pin_memory()
    return p
def bench(node, nbytes, streams, direction, iters=20):
    host = alloc_on_node(nbytes, node)
    dev = cp.empty(nbytes, dtype=cp.uint8)
    chunk = nbytes // streams
    ss = [cp.cuda.Stream(non_blocking=True) for _ in range(streams)]
    t0 = time.perf_counter()
    for _ in range(iters):
        for i, s in enumerate(ss):
            src, dst = (dev.data.ptr + i*chunk, host + i*chunk) if direction == "d2h" else (host + i*chunk, dev.data.ptr + i*chunk)
            cp.cuda.runtime.memcpyAsync(dst, src, chunk, cp.cuda.runtime.memcpyDefault, s.ptr)
        for s in ss: s.synchronize()
    return nbytes * iters / (time.perf_counter() - t0) / 1e9   # GB/s sustained
# sweep: node in {N_dram, N_cxl}, nbytes in {1MB..4GB}, streams in {1,2,4,8}, direction in {d2h,h2d}
# record: json with provenance (kernel, numactl -H, lspci -tv excerpt, driver versions)
```

## 부록 B. 이 문서가 참조하는 기존 산출물

`experiments/results/gpu_host_bandwidth.md`, `rngd_parallel_bandwidth.md`, `exp_pd_summary.md`, `planner/util/kv_transfer.py`, `planner/topology.py`(LinkKind, island_path), `docs/deviations.md` D16·D18·D22, `WORK_ORDER_pipeline_domain.md`(스키마 확장 방식 참고), 특허 1 v4 §5.5(A·K 최소 보장치 규칙), 특허 2 초안 §5.2–5.6(불확실 입력 레지스트리·정확도 도메인).
