# ScenarioLab 토폴로지 v2 — 서버 내부 연결망 모델링 및 대규모 시나리오 보고서

- 작성일: 2026-09-01
- 계기: Explorer(화면 ②→③)에서 일부 GPU/NPU가 연결망에 붙어 있지 않은 문제 보고
  + PCIe 버스의 서버 내부 구조(루트 콤플렉스당 ~4장치) 반영 요구
  + 최대 128 accelerator 규모 시나리오 요구

## 1. 무엇이 문제였나

기존 생성기(topology v1)의 두 가지 비현실성:

1. **고립 장치**: NIC 연결은 노드의 첫 번째 accelerator에만 있었고, RNGD 카드는
   peer 링크가 아예 없어서(설계 v1의 FR-C4 해석) 카드 1~3번이 그래프에서 완전히
   고립되어 보였다. 실제 서버에서 호스트 버스에 연결되지 않은 카드는 존재할 수 없다.
2. **평면적인 PCIe**: 노드 내 모든 장치를 하나의 contention group으로 전부 연결
   (all-pairs)했다. 실제 서버는 CPU 소켓별 루트 콤플렉스에 ~4장치가 매달리고,
   소켓 간에는 CPU interconnect를 거친다.

## 2. 토폴로지 v2 (scenariolab/generator/cluster_gen.py)

노드 내부를 호스트 구조로 모델링:

```
[root0: dev0..dev3 full-mesh PCIE, cg=node-pcie-root0] ── nic0 (root0에 부착)
        │ cpu-interconnect (PCIE, cg=node-cpu-interconnect)
[root1: dev4..dev7 full-mesh PCIE, cg=node-pcie-root1]
 + NVLink 클래스(rtxpro6000)는 위에 all-pairs NVLINK 메시 추가 (cg=node-nvlink-switch)
```

- `DEVICES_PER_ROOT = 4`: 루트 콤플렉스당 최대 4장치.
- **모든 클래스가 호스트 PCIe 트리에 올라간다** — RNGD 카드 포함. 카드 내부 fabric은
  여전히 내부(ONPACKAGE 링크 없음)지만, 카드는 여느 장치처럼 호스트 버스에 있다.
- NIC은 root0에 부착 → 모든 장치가 fabric까지 경로를 가진다.
- 생성 시 **전체 연결성 자체 검증** 추가: accelerator+NIC 전체가 하나의 연결 성분이
  아니면 해당 클러스터를 폐기·재샘플링 (v2 불변식).

### Island 구조에 미친 영향

RNGD 노드의 카드들이 PCIe로 이어지면서 **카드 4개 = island 1개**(이전: 싱글턴 4개)가
됐다. TP는 프로파일의 `max_tp_size=1` 그대로 카드 경계를 넘지 않고(후보 TP=[1]),
DP replicas가 island 안에서 카드 수만큼 허용된다 — 후보 공간의 표현이 바뀔 뿐
물리 제약(절대규칙 2)은 동일하다. 회귀 테스트
`test_rngd_cards_share_one_island_via_host_bus`로 고정.

## 3. 검증

```
pytest 366 passed (신규: 연결성 전수, 루트 콤플렉스 구조, RNGD island,
                   API 그래프 무고립) · ruff clean · mypy clean
```

- `test_every_device_connected`: 랜덤 12개 클러스터 전수 — 고립 장치 0, 단일 성분.
- `test_pcie_root_complex_structure`: 8-GPU 노드 → root0(peer 6 + NIC 1),
  root1(peer 6), cpu-interconnect 1로 정확히 분해.
- `test_graph_has_no_isolated_devices`(API): UI가 받는 그래프 기준으로도 무고립.
- **real-sim sanity**: verify_smoke를 새 토폴로지로 재실행 — 4/4 시나리오 full sim
  정상 완주, 오차 프로파일은 이전과 동일 (TPOT |err| p95 31%, power 60.6%,
  flip 0건). 새 링크 구조가 시뮬레이터 컴파일 경로를 깨지 않음을 확인.

## 4. 재실행 결과 (2026-09-01, 새 토폴로지)

| 배치 | 규모 | 소요(wall) | feasible | median 절감 | 비고 |
|---|---|---|---|---|---|
| smoke | 3×5=15 | 1.4 s | 15/15 | – | |
| default | 30×50=1,500 | 71 s | 1,377 (91.8%) | 24.5% | NPU 외삽 경고 626건, 오류 0 |
| **large** (신규) | 8×10=80, 클러스터당 41~91 accel | 69 s | 80/80 | 37.4% | 노드 8~16 × 장치 4~8 |
| **max128** (신규) | 2×5=10, 클러스터당 정확히 **128 accel** | 3 m 28 s | 10/10 | 50.4% | 16노드 × GPU 8, plan 최대 10 devices |

- 신규 설정: `experiments/configs/lab/large.yaml`(32~128 accel 혼합 규모),
  `experiments/configs/lab/max128.yaml`(항상 128 accel — RNGD는 노드당 4장 캡이라
  GPU 클래스만 사용).
- `run --skip-verify` 플래그 추가: fast path만 빠르게 돌리고 검증은
  `scenariolab verify`로 나중에 명시적으로 실행.
- 위 수치는 전부 fidelity=`surrogate`(default의 envelope hit 제외) — 검증 표본의
  오차 통계(§3)와 함께 읽어야 한다.

## 5. 사용법 요약

```bash
python -m scenariolab run --config experiments/configs/lab/large.yaml     # 32~128 accel
python -m scenariolab run --config experiments/configs/lab/max128.yaml    # 항상 128 accel
python -m scenariolab run --config experiments/configs/lab/default.yaml --skip-verify
python -m scenariolab serve --config experiments/configs/lab/large.yaml   # UI에서 토폴로지 확인
```
