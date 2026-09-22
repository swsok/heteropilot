# 이종 가속기 클러스터의 그래프 기반 배치 탐색 연구 설계

HeteroPilot 코드 기반 실행안  |  2026년 9월 22일

### 1 제안에 대한 판단

제안한 흐름은 타당하며 HeteroPilot 위에서 단계적으로 구현할 수 있다. 특히 반복되는 하드웨어 구조를 압축해 시뮬레이션 횟수를 줄이겠다는 방향이 유망하다. 다만 연구의 핵심은 모든 서브그래프를 일단 나열하는 것이 아니라, 필요한 배치 구조를 생성하면서 중복을 없애고 좋은 후보를 잃지 않는 데 있다.

**권장 연구 질문** “공유 통신 자원을 보존한 동등성 압축과 단계적 후보 평가로, 이종 클러스터에서 서비스 조건을 만족하는 저비용 배치를 얼마나 적은 시뮬레이션으로 찾을 수 있는가?”로 구체화한다.

| **사용자 아이디어** | **실행 가능한 구체화** |
| --- | --- |
| 가속기 연결망을 그래프로 표현 | CPU·NIC·스위치·공유 버스까지 포함하는 속성 그래프 |
| 연결 구성을 후보로 나열 | TP·PP·복제·P/D 역할과 경로를 포함하는 배치 템플릿 생성 |
| 같은 서브그래프를 하나로 압축 | 정확 동등성은 병합하고, 근사 유사성은 우선순위 그룹으로만 활용 |
| 확실히 불가능한 후보 제거 | 실행 계약에 따른 필요조건과 증명 가능한 낙관적 경계 사용 |
| 간단한 알고리즘으로 랭킹 | 서비스 여유도·목적 비용·통신 병목을 분리한 순위 |
| Top K 시뮬레이션 | 구조별 후보를 보존하고 K를 늘리며 평가하는 적응형 탐색 |

### 논문 기여로 삼을 부분

**기여 A** 동일한 모양을 넘어, 공유 링크·경계 연결·실행 역할을 보존하는 후보 압축 규칙과 안전 조건을 정의한다.

**기여 B** 확실한 제거와 휴리스틱 보류를 구별하며, 후보 누락 위험을 측정하는 적응형 Top K 탐색을 만든다.

**기여 C** 다양한 토폴로지에서 압축률, 시뮬레이션 절감, 최적해 손실과 실제 SLO 충족을 함께 검증한다. 기존 코드 재사용과 새 연구 기여를 분명히 나눈다.

**검토 범위** 코드 근거는 a27469a3b14c2be7def379ef5aab5a14eb0e0689에 고정한다. 설명용 예시는 가상 구성이고, 아래 알고리즘의 효과는 앞으로 검증할 가설이다.

## 2 최적화 문제와 후보의 정의

### 입력은 연결망과 SLO만으로 충분하지 않다

**필수 입력** 모델과 정밀도, 입력·출력 길이 분포, 도착률과 burst, prefix cache 조건, runtime 지원 범위, 가용 장치와 링크, 가격 또는 전력 모델을 받는다. 같은 그래프도 요청 패턴에 따라 최선의 배치가 달라진다.

**용어** TTFT는 요청 후 첫 토큰까지의 시간이다. TPOT는 이후 출력 토큰당 시간이며 요청별 계산 정의를 고정한다. p99는 요청의 99%가 그 값 이하라는 뜻이다. Goodput은 정한 지연 기준을 만족한 요청의 초당 처리 수로 정의한다.

### 하나의 후보는 실행 가능한 전체 계획이다

**후보 구성** 선택한 물리 장치 집합, 모델 shard와 역할 배치, TP·PP·복제 수, 배칭 설정, 요청 routing, 통신 경로와 예약 자원을 함께 저장한다. 같은 장치 서브그래프라도 TP4와 TP2 두 복제본은 다른 후보다.

**목적함수** 첫 구현은 시간당 비용 최소화를 권한다. 평균 전력 최소화는 별도 실행 모드로 둔다. 동일 관측 시간과 처리 요구를 고정하지 않으면 최소 에너지, 최소 전력, 최대 tok/J는 서로 다른 목표가 된다. 가격이 없을 때 활성 장치 수를 비용이라고 부르지 않는다.

| **제약** | **판정 기준의 예** |
| --- | --- |
| 지연 | p99 TTFT ≤ TTFT 목표, p99 TPOT ≤ TPOT 목표 |
| 처리량 | 완료 요청/s 또는 유효 요청/s ≥ 명시한 하한 |
| 완료율 | 입력 요청 대비 성공 완료 비율과 timeout 기준 고정 |
| 자원 | 장치별 메모리, 장치 수, 공유 링크와 호스트 용량 |
| 실행 가능성 | 모델·dtype·backend·collective·KV 교환 지원 |

**주의** TTFT와 TPOT 각각의 p99 조건은 “동일 요청 99%가 두 조건을 동시에 만족”한다는 보장과 다르다. 후자를 원한다면 요청별 공동 충족률을 별도로 검사한다.

### HeteroPilot에서 먼저 확장할 인터페이스

현재 ServiceSpec의 목적에는 에너지·goodput/J·활성 가속기 수가 있고 일반적인 금전 비용 항목은 없다. 최종 feasibility에도 처리량 하한이 명시되어 있지 않다. min_goodput_rps, completion_ratio, cost_per_hour와 관측 창을 먼저 추가하고, 생성기의 제거 기준과 최종 판정을 같은 의미로 맞춘다. [\[C6\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/spec.py) [\[C7\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/optimizer/feasibility.py)

## 3 물리 그래프와 통신 수요를 함께 표현한다

### 권장 데이터 모델

**물리 그래프** 가속기, CPU socket 또는 NUMA 영역, PCIe switch·root complex, NIC, 네트워크 switch를 정점으로 둔다. 링크에는 방향별 용량과 지연을 기록한다. QPI·UPI는 CPU socket 사이 경로이므로 GPU 간 직접 링크처럼 단순화하지 않는다.

| **객체** | **필수 속성** | **사용 목적** |
| --- | --- | --- |
| 가속기 | 종류·모델·메모리·runtime·정밀도·가격·전력·사용 상태 | 실행 호환성과 자원 비용 |
| 링크 | bytes/s·지연·방향·duplex·P2P/RDMA 가능 여부 | 경로와 통신 시간 |
| 공유 자원 | PCIe uplink·NIC·switch port·용량·예약량 | 여러 흐름이 함께 쓰는 병목 |
| 측정 프로파일 | 메시지 크기·collective·참여 수·NUMA·측정 시각 | 명세 속도와 실효 속도 구분 |
| 실행 통신 | 역할·의존성·전송량·빈도·허용 경로 | 배치를 통신 부하로 변환 |

**공유 경합** 여러 간선이 같은 uplink를 사용하면 같은 resource_id에 연결한다. 보조 정점을 둔 incidence graph로 표현하면 일반 그래프 도구를 활용할 수 있다. 독립 링크 두 개와 하나의 공유 링크를 두 번 표기한 경우를 반드시 구분한다.

**외부 부하** 배치 외부 서비스가 쓰는 용량은 snapshot 또는 예약 계약으로 포함한다. 후보 내부 연결만 같아도 외부 트래픽이 다르면 성능 동등성이 성립하지 않을 수 있다. 실행 직전에 자원 예약과 상태 버전을 다시 확인한다.

### 속도 단위와 경로의 함정

**단위** 100 Gbit/s는 명목상 12.5 GB/s이며 실제 payload 속도는 별도다. HeteroPilot의 bandwidth_gbps 필드는 이름과 달리 코드에서 GB/s로 처리하므로 수집 단계에서 단위를 명시해 bytes/s로 정규화하는 adapter를 둔다. 단위 변경은 기존 YAML을 자동으로 재해석하지 말고 schema version으로 구분한다. [\[C1\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/inventory.py) [\[C2\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/topology.py)

**경로** 현재 TopologyGraph.path는 최소 hop BFS다. 실제 routing 정책, 병목 용량, RDMA/P2P 가능 여부와 다를 수 있다. 허용 경로 집합을 저장하고, 같은 메시지라도 collective 알고리즘과 배치에 맞게 경로를 선택한다. [\[C2\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/topology.py) [\[R8\]](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)

## 4 모든 부분집합 대신 배치 구조를 생성한다

### 탐색 범위를 먼저 선언한다

가속기 n개의 부분집합만 2의 n승이며, 역할·병렬화·경로까지 붙이면 더 커진다. 실제 연결 경로가 중간 CPU·switch를 지나므로 가속기만의 induced subgraph도 충분하지 않다. 가속기 선택 뒤 필요한 transit 정점과 공유 자원을 닫힘 형태로 포함한다.

**초기 범위** 같은 backend의 TP 그룹, 지원되는 PP 그룹, 서로 다른 장치의 독립 복제본부터 시작한다. GPU와 NPU의 요청 단위 혼합은 가능성이 있지만, 하나의 TP communicator를 공유하거나 KV를 넘기는 P/D는 별도 호환성 확인이 있어야 후보가 된다.

| **생성 단계** | **방법** |
| --- | --- |
| 1 실행 그룹 | 기존 execution island에서 가능한 장치 부분집합과 TP 차수 생성 |
| 2 모델 배치 | TP 그룹 또는 PP stage에 layer와 메모리를 배정 |
| 3 서비스 구성 | 한 복제본, 여러 복제본, 지원되는 P/D 역할을 조합 |
| 4 통신 연결 | 필요한 경로·공유 자원·runtime 통신 알고리즘 선택 |
| 5 중복 제거 | 생성 즉시 동등성 키를 검사하고 대표·embedding을 저장 |

**복제본은 별개 그룹일 수 있다** DP 또는 요청 단위 복제는 장치 간 collective가 없을 수 있으므로, 모든 후보를 하나의 연결된 가속기 집합으로 제한하면 유효한 계획을 놓친다. 각 실행 그룹 내부 연결과 ingress·egress 요구를 따로 검사한다.

**계층적 열거** GPU pair, 같은 socket의 GPU 묶음, 노드 내 그룹, 노드 간 stage의 순서로 확장한다. 부분 후보에서도 메모리·backend의 필요조건을 검사하고, 대칭인 확장 경로를 canonical key로 막는다.

### 완전성의 범위를 정직하게 유지한다

TP 차수나 노드 수 상한을 구현 편의로 제한하면 “허용한 템플릿 안에서 탐색”한 것이다. 이것을 전체 클러스터의 전역 최적화라고 부르지 않는다. 비용이 낮다는 이유만으로 적은 장치 후보만 남기는 것도 성능 제약 때문에 안전하지 않다.

**HeteroPilot 연결** CandidateGenerator는 단일 island, 두 island의 mixed 및 P/D 조합을 생성한다. 이 골격을 유지하고, 내부 장치 embedding과 여러 그룹을 생성하는 계층을 앞에 추가한다. 기존 코드가 이미 임의 그래프의 모든 배치를 열거한다고 가정하지 않는다. [\[C1\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/inventory.py) [\[C3\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/candidate_generator.py)

## 5 후보 압축의 정확한 조건

### 같은 모양과 같은 실행 특성은 다르다

**정확 병합의 조건** 정점·간선·공유 자원의 속성, 선택한 역할과 shard, 통신 경로, 외부 예약 상태, 비용·전력·보정 프로파일까지 보존하는 일대일 대응이 있어야 한다. 수치 반올림으로 같은 class가 된 경우는 정확 동등성이라고 부르지 않는다.

**가장 강한 기준** 클러스터 전체의 속성과 자원 예약을 보존하는 대칭 변환이 한 배치를 다른 배치로 옮기면, 같은 실행 모델 아래에서 대표 평가를 재사용할 수 있다. 실제 측정 편차까지 0이라는 뜻은 아니므로, 하드웨어 잔차는 별도 검증한다.

**실용적인 기준** 전체 그래프 검사 비용이 크면 후보와 boundary context를 사용한다. 단, 외부로 이어지는 NIC·uplink·공유 그룹·예약을 포함한 context가 평가에 충분하다는 가정을 문서화한다. 보존성을 증명하지 못하면 근사 압축으로 표시한다.

### 구현 절차

**1차 묶기** 속성 histogram과 WL hash로 같은 가능성이 있는 후보만 묶는다. 방향·다중 간선·공유 자원은 보조 정점으로 명시하고 속성 직렬화와 도구 버전을 고정한다. 해시 일치만으로 동형이라고 판정하지 않는다. [\[R6\]](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash.html)

**2차 확인** 동일 bucket 안에서 VF2 등 정확 속성 동형 검사를 수행한다. 일치할 때 canonical role mapping과 실제 device ID 목록을 저장한다. 작은 후보부터 시작하고 전체 그래프 대칭은 오프라인으로 계산한다. [\[R7\]](https://networkx.org/documentation/stable/reference/algorithms/isomorphism.vf2.html)

**압축 뒤에도 남길 정보** 대표 signature, 실제 embedding 목록, 동시 배치 가능 수, 공유 자원 사용량, 대표를 원래 장치에 되돌리는 대응표를 저장한다. 두 embedding이 GPU나 uplink를 겹쳐 쓰면 동시에 선택할 수 없다는 제약도 유지한다.

### 중요한 반례

GPU 두 쌍이 모두 “GPU 2개 + NVLink”여도 한 쌍의 NIC uplink가 다른 서비스와 공유되면 P/D 전송 성능이 다르다. 후보 내부 모양만 보고 합치면 이 차이가 사라진다. 공유 경합과 경계 연결을 보존하는 압축이 이 연구의 핵심 기여가 될 수 있다.

**정확 압축과 근사 그룹을 분리한다** 비슷한 속도의 링크들을 묶어 랭킹이나 측정 순서를 공유하는 것은 가능하다. 이때 그룹 내 최악값 또는 잔차를 검증하고, 상위 후보는 원래 속성으로 다시 평가한다. 근사 그룹에서 대표 하나만 남겨 나머지를 영구 삭제하지 않는다.

## 6 확실한 제거와 추정에 따른 보류

### 안전한 제거는 최선의 가정에서도 실패할 때만 한다

| **검사** | **확실한 제거가 가능한 조건** | **주의할 가정** |
| --- | --- | --- |
| 호환성 | 선택 runtime이 모델·정밀도·통신 구성을 지원하지 않음 | 미등록과 실제 미지원은 구분 |
| 메모리 | 불가피한 최소 메모리가 가용량을 초과 | offload·양자화·재계산 허용 여부 고정 |
| 통신 지연 | 필수 통신의 낙관적 하한도 지연 목표를 초과 | 경로·collective·overlap 선택 범위 명시 |
| 처리 용량 | 낙관적 처리량 상한조차 요구량보다 낮음 | 최종 feasibility에도 같은 처리량 제약 필요 |
| 비용 하한 | 인증된 가능 후보보다 더 싸질 수 없음 | 같은 목표, 비용 범위와 유효 incumbent 필요 |

**메모리** 총 메모리 합계만 보지 말고 shard별 배정과 비분할 텐서를 검사한다. weight 크기 하한은 비교적 명확하지만 KV cache는 workload·스케줄러에 의존한다. 중앙값 길이를 담지 못한다는 사실을 모든 p99 서비스 계약의 일반적 불가능 증명으로 쓰지 않는다.

**통신 하한** 특정 cut을 반드시 지나는 최소 bytes를 그 cut의 낙관적 총 용량으로 나눈다. 여러 cut이나 compute와 통신이 겹칠 수 있으면 하한들의 최대값을 사용한다. 실제로 순차 의존성이 있는 구간만 더한다. 한 경로의 느린 링크로 고정해 다른 허용 경로를 무시하면 안전한 하한이 아니다.

**실측 속도의 용도** 측정된 평균 실효 대역폭은 예측과 랭킹에는 유용하다. 그러나 가능한 최고 속도의 상한임이 보장되지 않으면, 그것으로 계산한 시간이 “절대 더 빠를 수 없는 하한”은 아니다. 확실한 제거와 경험적 제거는 별도 상태로 기록한다.

### 기존 bound를 그대로 증명이라고 부르지 않는다

HeteroPilot은 단계 4·5를 낙관적 필터로 설계했지만, ring all reduce 비용과 메모리 roofline에는 알고리즘·역할·캐시 관련 가정이 있다. 이 가정이 실제 허용 runtime에서도 성립하는지 재검토한다. 특히 throughput 필터는 과거 최종 제약과 불일치해 제거된 기록이 있어 먼저 spec을 확장해야 한다. [\[C3\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/candidate_generator.py) [\[C7\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/optimizer/feasibility.py)

**출력 상태** 각 후보를 impossible_proven, excluded_by_scope, deferred_heuristic, unknown_measurement, evaluated로 구분한다. 측정이 없다는 이유나 낮은 순위 때문에 빠진 후보를 “실행 불가능”으로 세지 않는다.

## 7 간단한 랭킹과 적응형 Top K

### 첫 랭커는 설명 가능한 규칙으로 만든다

**특징** 예측 TTFT/SLO, TPOT/SLO, 요구 goodput/예측 goodput, 시간당 비용, 공유 NIC 이용률, 링크 cut 여유, 메모리 여유, 보정 범위 밖 여부를 쓴다. 단일 tok/J만 정렬하면 TP나 복제 수에 따른 feasibility 차이를 놓칠 수 있다.

**간단한 정렬** 지연·처리량 비율 중 최댓값을 risk_proxy로 두고, 추정상 요구를 만족하는 후보를 먼저 보되 그 안에서는 목표 비용 순으로 정렬한다. 나머지는 risk_proxy와 비용을 기준으로 탐색한다. 이 값은 확률이나 보장된 경계가 아니라 우선순위용 휴리스틱이다.

**다양성 보존** TP 차수, 노드 수, 장치 종류, P/D 여부별로 일부 자리를 남겨 하나의 cheap proxy가 같은 구조만 고르지 않게 한다. 예산이 그룹 수보다 작으면 순환 선택 규칙을 명시하고, 그룹 수가 많다는 이유로 예산을 몰래 초과하지 않는다.

### 평가 결과를 보고 다음 후보를 고른다

**초기 K** 예를 들어 4개로 시작해 8개, 16개로 확대한다. 이 숫자는 시작 설정이며 실험으로 조정한다. 시뮬레이션 결과를 받은 뒤 예측 잔차와 순위를 갱신하고, 근사 동등 그룹에서 편차가 크면 그룹을 분리한다.

**예산 모드** 시간 또는 실행 횟수를 다 쓰면 “평가한 후보 중 최선”과 미평가 수를 반환한다. 좋은 후보를 찾았다는 이유만으로 전체 최적이라고 선언하지 않는다. 선택한 후보는 정확도 도메인 검사와 실측 확인을 거친다.

**인증 모드** 유효한 하한이 모든 미평가 후보에 존재하고, 그 비용 하한이 현재 가능한 해의 비용보다 높거나 같을 때만 최적성 종료가 가능하다. ε 허용 오차를 쓰면 그 격차도 보고한다. 이 인증도 선언한 후보 공간과 평가 모델에 한정되며 실제 하드웨어 최적성 인증은 아니다.

### 현재 코드에서 재사용할 부분

BinnedRooflineRanker는 이미 존재하고 기본값으로 표기되어 있다. 기존 proxy의 수치상 동률 문제를 개선했으므로 이를 baseline에 포함한다. 기록상 높은 부하에서는 K=50까지도 가능한 후보를 놓치는 사례가 남아 있다. 따라서 과거의 단순 랭커만 비교해 개선 효과를 부풀리지 않는다. [\[C4\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/optimizer/surrogate.py) [\[C10\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/docs/surrogate_topk_regret.md)

**학습 기반 랭커는 후속이다** 충분한 서로 다른 토폴로지 데이터가 쌓인 뒤 gradient boosting 같은 모형을 비교한다. 처음부터 GNN을 핵심으로 삼으면 데이터와 일반화 검증 부담이 커진다.

## 8 시뮬레이터가 그래프 차이를 보존해야 한다

### 현재 가장 큰 구현 간극

현재 Level 1은 링크를 대표 bandwidth·latency로 줄인다. Level 2는 intra와 cross dimension을 구분하지만, 코드상 path_aware와 contention_modeled는 false다. planner의 경합 보정 helper가 있어도 simulator 전체가 동시에 흐르는 전송의 공유 자원을 재현한다는 뜻은 아니다. [\[C2\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/topology.py) [\[C8\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/predictor/llmservingsim.py)

**연구에 미치는 영향** 공유 uplink 유무가 다른 두 그래프가 같은 시뮬레이터 입력으로 축약되면, 새로운 압축 알고리즘이 차이를 보존해도 성능 실험에서 효과를 평가할 수 없다. 그래프와 simulator adapter를 함께 바꿔야 한다.

### 두 단계로 확장한다

**MVP** 평가 범위를 경합이 통제된 노드 내부와 단순 노드 간 구조로 제한한다. Level 2를 재사용하고, P/D 전송은 명시한 간단한 경로 비용으로 모델링한다. 이 결과로 임의의 공유 fabric을 해결했다고 주장하지 않는다.

**연구형 확장** 전송 이벤트에 flow ID, message bytes, route, shared resource ID, 시작 의존성을 붙인다. 공유 자원 용량을 동시에 활성화된 flow들이 사용하는 event 기반 네트워크 모형을 추가하거나 해당 기능이 있는 backend로 연결한다. fluid 모델도 packet 단위와 동일하지 않으므로 범위를 명시한다.

**필요한 검증** 단일 전송, 두 흐름의 동일 NIC 공유, 독립 NIC, bidirectional 전송, collective의 참여 수와 메시지 크기를 바꾼 microbenchmark로 맞춘다. 그 뒤 실제 TTFT·TPOT가 어떻게 변하는지 종단 trace로 확인한다.

### 예측 정확도와 탐색 정확도를 분리한다

**탐색 정확도** 같은 시뮬레이터로 전체 평가한 최선과 압축·Top K 결과를 비교한다. 그래프 압축이나 후보 선택 때문에 잃은 품질을 본다.

**예측 정확도** 시뮬레이터가 고른 후보와 경계 근처 대안을 실제 실행한다. p99를 p99와 비교하고 workload·runtime·배치가 일치하는 보정만 쓴다. 보정 데이터로 다시 검증하는 순환 평가를 피한다. [\[C9\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/predictor/calibration.py)

**최종 배포** 대표 signature에서 물리 embedding을 복원한 뒤 장치와 링크를 예약한다. 새 snapshot에서 병목이나 가용량이 바뀌면 재평가한다. 대표 하나의 성능 결과를 중복 배치 여러 개에 무조건 곱하지 않는다.

## 9 작은 가상 예제로 보는 후보 압축

이 절의 장치와 수치는 설명용 가정이다. 실제 HeteroPilot 실험 결과가 아니다. 논문 성능 수치로 사용하지 않는다.

### 예시 클러스터와 탐색 범위

A·B 노드는 각각 동일 GPU 2개와 10 GB/s uplink, C·D 노드는 각각 같은 GPU 2개와 5 GB/s uplink를 갖는다. 각 노드 안의 GPU pair는 동일한 빠른 연결을 갖는다. E 노드에는 NPU 2개가 있다. 노드 쌍 A/B와 C/D는 가격·외부 부하 등 모든 관련 속성이 각각 같다고 가정한다.

**범위** 가속기 두 개를 사용하는 단일 TP 그룹만 생성한다. GPU 8개와 NPU 2개에서 선택하는 장치 쌍은 45개다. 동일 backend의 GPU TP2만 지원하고, NPU TP2와 GPU–NPU TP2는 이 예시의 runtime 계약에서 미지원으로 둔다.

| **대표 유형** | **실제 장치 쌍 수** | **판정** |
| --- | --- | --- |
| A 또는 B 내부 GPU pair | 2 | GPU TP2 가능 |
| C 또는 D 내부 GPU pair | 2 | GPU TP2 가능 |
| A와 B에 걸친 GPU pair | 4 | GPU TP2 가능 |
| C와 D에 걸친 GPU pair | 4 | GPU TP2 가능 |
| A/B와 C/D에 걸친 GPU pair | 16 | GPU TP2 가능 |
| A/B GPU와 NPU pair | 8 | 예시 runtime 미지원 |
| C/D GPU와 NPU pair | 8 | 예시 runtime 미지원 |
| NPU pair | 1 | 예시 runtime 미지원 |

**압축 결과** 45개 물리 쌍이 8개 대표 유형으로 줄고, 호환성 검사 뒤에는 28개 GPU 쌍에 해당하는 5개 대표가 남는다. 압축 후에도 물리 쌍 28개의 mapping은 유지한다.

**추가 필터의 예** 노드 간 필수 cut traffic이 한 단계에 0.4 GB이고 예시 지연 한도가 50 ms라고 가정한다. 5 GB/s cut의 낙관적 하한은 80 ms이므로 C–D 및 fast–slow 유형을 제외한다. A–B는 40 ms여서 남지만 실제 통과가 보장되지는 않는다. 예시의 대표는 3개가 남는다.

**해석** 동등성의 기준을 엄격히 하면 압축률이 낮아질 수 있다. 대신 잘못 합친 후보 때문에 최적 배치를 놓치는 위험을 통제할 수 있다. 실제 압축률과 통과율은 workload·장치·토폴로지를 바꿔 측정해야 한다.

## 10 HeteroPilot 수정 위치와 개발 인터페이스

| **현재 파일** | **재사용 기능** | **제안하는 변경** |
| --- | --- | --- |
| inventory.py [\[C1\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/inventory.py) | 장치·링크·측정·island | CPU/switch 및 shared resource 모델, 단위 adapter, runtime capability |
| topology.py [\[C2\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/topology.py) | 경로·link 측정·dimension 축약 | 허용 경로, boundary context, graph signature |
| candidate_generator.py [\[C3\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/candidate_generator.py) | 단계별 후보 생성과 rejection | 템플릿 생성, embedding 열거, exact와 heuristic 상태 분리 |
| surrogate.py [\[C4\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/optimizer/surrogate.py) | Analytical 및 Binned 랭커 | 서비스 여유도·통신 특성·다양성 랭커 |
| exhaustive.py [\[C5\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/optimizer/exhaustive.py) | Top K·평가·최종 추천 | K 확대, 단계별 재정렬, cache 및 종료 근거 |
| spec.py / feasibility.py [\[C6\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/spec.py) [\[C7\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/optimizer/feasibility.py) | 지연·전력 조건과 목적 | 금전 비용, goodput·완료율, 결측 지표 unknown 처리 |
| predictor/llmservingsim.py [\[C8\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/predictor/llmservingsim.py) | sim config와 결과 변환 | 흐름·경로·공유 자원 전달, 손실 정보 보고 |
| predictor/calibration.py [\[C9\]](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/predictor/calibration.py) | 보정 적용 조건 | 토폴로지 signature와 runtime 버전, p99 일치 |

### 새로 나눌 모듈의 예

graph_schema.py는 정규화된 자원 그래프를, candidate_templates.py는 역할 포함 배치를, equivalence.py는 hash·동형 검사·embedding을 담당하게 한다. bound_pruning.py는 증명 가능한 제거만, adaptive_search.py는 휴리스틱 평가 순서와 예산을 담당하게 한다. 이 이름은 신규 구현 제안이다.

**후보 레코드** candidate_id, template_id, role_mapping, paths, resource_demands, exact_signature, embeddings, assumptions, rejection_reason, lower_bounds, score, prediction_status를 저장한다. simulator cache key에는 모델·trace·runtime·보정·topology schema 버전까지 넣는다.

**감사 가능한 결과** 총 생성 수, 대표 수, 확실한 제거 수, 보류 수, 시뮬레이션 수, 미평가 수, 반환 계획의 물리 자원과 인증 범위를 함께 출력한다. 한 장치의 capacity를 여러 대표가 중복 사용할 수 없도록 최종 복원 검사를 둔다.

## 11 구현 절차와 검증해야 할 불변식

### 전체 탐색 절차

1  snapshot과 service contract를 읽고 단위·호환성 정보를 정규화한다.

2  허용한 배치 템플릿을 확장하며 역할과 통신 경로를 붙인다.

3  확실한 필요조건 위반을 기록하고 후보를 제거한다.

4  정확 동등 후보는 대표에 mapping을 추가하고 근사 후보는 그룹으로만 묶는다.

5  남은 대표를 여유도·목표 비용으로 정렬하고 구조별로 초기 후보를 선택한다.

6  시뮬레이션 결과로 같은 서비스 계약의 feasibility와 목표값을 계산한다.

7  예산이 남으면 순위와 근사 그룹을 갱신하고 다음 평가 batch를 선택한다.

8  종료 근거와 미평가 후보를 기록하고 대표를 물리 embedding으로 복원한다.

9  snapshot·예약·보정 조건을 재확인하고 상위 후보를 실측 검증한다.

### 반드시 지켜야 할 불변식

**압축 보존** 정확히 합친 후보는 정의된 모형에서 같은 제약 판정과 목적값을 가진다. 노드 ID를 바꾸어도 signature는 같고, 공유 NIC 구조나 역할을 바꾸면 필요할 때 달라져야 한다.

**제거 보존** 안전 필터를 켜고 끈 작은 전체 탐색에서 feasible 최선이 유지되어야 한다. 모든 rejection에 사용한 경계와 가정이 남아야 한다. 반례 하나가 발견되면 그 필터는 heuristic으로 낮춘다.

**mapping 보존** 복원한 장치와 공유 자원의 사용량이 실제 가용량을 넘지 않는다. 대표의 개수와 동시에 배치 가능한 개수는 다르다.

**결론 보존** 시뮬레이션 실패, 미측정, 낮은 순위, scope 제외를 infeasible과 합치지 않는다. 실제 실행 지원과 시뮬레이션 지원도 구분한다.

### 복잡도와 구현 절충

동형 검사와 전체 대칭 계산도 비용이 든다. hash를 전처리로 쓰고 작은 실행 그룹부터 cache한다. 최악의 조합 폭발이 사라진다고 주장하지 말고, 반복 구조가 많은 실제 cluster에서 전체 wall time이 줄어드는지 측정한다. 정확 병합이 거의 없는 비대칭 cluster도 실패 조건으로 포함한다.

## 12 실험 설계와 논문 기여의 검증

| **질문** | **비교 실험** | **핵심 지표** |
| --- | --- | --- |
| 압축이 정확한가 | 전체 후보 대 정확 압축, 경계 속성 제거 ablation | 최선 보존, 오병합, mapping 충돌 |
| 검색이 빨라지는가 | 전체 sim, 기존 랭커, 제안 랭커와 적응 K | 전체 시간, sim 수, feasible 발견 시간 |
| 좋은 후보를 잃는가 | 작은 후보 공간의 전체 평가 oracle | feasible recall, 비용 regret, false infeasible |
| 공유 경합을 재현하는가 | 독립/공유 NIC와 background load | 전송 시간, TTFT·TPOT·goodput 오차 |
| 새 topology로 전이되는가 | 학습/보정에 쓰지 않은 node·fabric 구성 | 추천 실패율, 비용, 미검증 비율 |

### 실험 범위를 현실적으로 확대한다

**1단계** 4~8개 가속기의 작은 범위에서 전체 후보 평가를 수행해 정확 압축과 필터를 검증한다. 공유 uplink가 있는 경우와 없는 경우를 반드시 함께 둔다.

**2단계** 실측 가능한 GPU/NPU 구성에서 상위 계획과 경계 대안을 실행한다. 2개 모델, 3개 부하 패턴, 4개 토폴로지 조건, 4개 부하 수준, 3회 반복이면 288개 조건 실행이다. 모든 비교 방법이 매번 별도 실측을 요구하는 것은 아니며, 미지원 조합은 적용 범위에서 제외한다.

**3단계** 32~128개 장치의 합성·확장 그래프로 탐색 확장성을 평가한다. 이 단계의 시뮬레이션 결과를 대규모 실장비 정확도 검증으로 표현하지 않는다. trace는 도착 과정과 길이의 상관성을 보존하며 ServeGen 등을 참고한다. [\[R9\]](https://www.usenix.org/conference/nsdi26/presentation/xiang-servegen)

### 성공 기준과 실패할 가능성

**성공 기준** 작은 완전 탐색에서 정확 압축·필터의 최선 손실 0, 같은 wall time에서 baseline보다 낮은 regret 또는 더 빠른 feasible 발견, 독립 실측에서 요구 goodput과 지연 충족을 함께 보이는 것이다. 수치 개선 목표는 파일럿 후 별도 사전 등록한다.

**주요 실패 조건** cluster가 비대칭이라 압축률이 낮거나, 동형 검사 비용이 sim 절감보다 크거나, 경로 모형 오차가 ranking 차이를 압도할 수 있다. 이때 exact 압축을 cache 기능으로 줄이고 shared contention 모델 또는 적응 탐색 하나에 논문을 집중한다.

**통계** 요청과 실행의 상관성을 고려해 반복 실행·신뢰구간을 보고한다. 예측기 전체 평가 oracle과 실제 장비 최선은 다르다. 임의의 작은 K에서 regret 0이 나와도 일반적 최적성으로 확장하지 않는다.

## 13 선행연구와 차별화 및 실행 일정

| **가까운 연구** | **중복되는 부분** | **권장 차별화** |
| --- | --- | --- |
| Helix [\[R1\]](https://www.pdl.cmu.edu/PDL-FTP/BigLearning/helix_abs.shtml) | 이종 GPU·network 그래프와 배치/routing 최적화 | 공유 자원을 보존하는 후보 동등성 및 제한 sim 예산 탐색 |
| Vidur [\[R2\]](https://arxiv.org/abs/2405.05465) | 프로파일 기반 sim과 배포 설정 탐색 | 그래프 압축의 정확 조건과 누락 위험 |
| DistServe [\[R3\]](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) | P/D 역할·병렬화·대역폭을 고려한 배치 | GPU/NPU runtime 계약과 후보 공간 압축 |
| LLMServingSim 2.0 [\[R4\]](https://arxiv.org/abs/2602.23036)<br>Frontier [\[R5\]](https://arxiv.org/abs/2605.21312) | 이종·분리형 서빙과 topology 관련 sim | 새 simulator 주장보다 검색 품질·계산 비용의 동시 개선 |

**논문에서 피할 주장** “처음으로 클러스터를 그래프로 표현했다” 또는 “Top K로 전역 최적을 보장한다”는 주장은 어렵다. Helix는 이미 그래프 기반 placement·scheduling을 다루며, HeteroPilot 자체에 Top K가 있다. 새로움은 정확한 중복 정의, 공유 병목을 잃지 않는 압축, 제한된 평가 예산의 품질 관리에 둔다.

### 권장 8주 파일럿

**1~2주** service contract에 goodput과 비용을 추가한다. topology 단위를 통일하고 작은 가상 graph 및 완전 열거 oracle을 만든다.

**3~4주** 역할 포함 signature와 정확 동형 검사를 구현한다. 현재 island 구조에서 embedding 복원과 공유 자원 충돌을 검증한다. 정확 압축만으로 절감이 있는지 먼저 본다.

**5~6주** 기존 BinnedRooflineRanker를 baseline으로 적응 K를 붙인다. 동시에 두 flow가 공유 NIC를 지나는 microbenchmark로 네트워크 표현의 최소 요건을 확인한다.

**7~8주** 새 topology와 workload에서 holdout 평가를 하고 상위 배치를 실제로 실행한다. 임의 graph의 경합 모델 확장이 길어지면 MVP 범위를 제한하고 topology adapter는 후속 단계로 분리한다.

### 첫 논문에 권하는 범위

노드 내 동종 TP와 노드 간 이종 요청 복제부터 시작하고, 지원이 확인된 PP 또는 P/D만 추가한다. 첫 논문은 “그래프 동등성에 기반한 시뮬레이션 탐색 비용 절감”에 집중하는 구성이 명확하다. GPU–NPU 간 단일 TP나 KV 변환 시스템까지 동시에 해결하려 하면 논문 범위와 구현 위험이 급격히 커진다.

**예상 산출물** 정규화 그래프 schema, 후보/embedding 라이브러리, 경계 증명 기록, 적응형 평가기, 작은 전체 탐색 oracle, 반복 가능한 실측 harness와 논문용 ablation 표다. 일정은 장비·runtime 접근이 가능한 경우의 추정이다.

## 14 논문 및 공식 자료 링크

**R1  Helix** ASPLOS 2025<br>[https://www.pdl.cmu.edu/PDL-FTP/BigLearning/helix_abs.shtml](https://www.pdl.cmu.edu/PDL-FTP/BigLearning/helix_abs.shtml)

**R2  Vidur** MLSys 2024<br>[https://arxiv.org/abs/2405.05465](https://arxiv.org/abs/2405.05465)

**R3  DistServe** OSDI 2024<br>[https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)

**R4  LLMServingSim 2.0** ISPASS 2026<br>[https://arxiv.org/abs/2602.23036](https://arxiv.org/abs/2602.23036)

**R5  Frontier** 2026 사전공개 논문<br>[https://arxiv.org/abs/2605.21312](https://arxiv.org/abs/2605.21312)

**R6  NetworkX WL graph hash** 공식 문서<br>[https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash.html](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash.html)

**R7  NetworkX VF2 graph isomorphism** 공식 문서<br>[https://networkx.org/documentation/stable/reference/algorithms/isomorphism.vf2.html](https://networkx.org/documentation/stable/reference/algorithms/isomorphism.vf2.html)

**R8  NCCL GPU troubleshooting** NVIDIA 공식 문서<br>[https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)

**R9  ServeGen** NSDI 2026<br>[https://www.usenix.org/conference/nsdi26/presentation/xiang-servegen](https://www.usenix.org/conference/nsdi26/presentation/xiang-servegen)

문헌 확인일은 2026년 9월 22일이다. Frontier는 사전공개 논문으로 구분한다. 자료의 링크는 비교 논문과 구현 출발점을 제공하며, 제안 알고리즘의 신규성이나 성능을 확정하는 근거는 아니다.

## 15 코드 근거와 읽는 순서

**C1  하드웨어 및 링크 스키마와 실행 island<br>**[planner/inventory.py](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/inventory.py)

**C2  그래프 경로와 시뮬레이터 토폴로지 축약<br>**[planner/topology.py](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/topology.py)

**C3  후보 열거와 단계별 제거<br>**[planner/candidate_generator.py](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/candidate_generator.py)

**C4  기본 및 개선된 대리 랭커<br>**[planner/optimizer/surrogate.py](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/optimizer/surrogate.py)

**C5  Top K 평가와 시뮬레이션 탐색<br>**[planner/optimizer/exhaustive.py](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/optimizer/exhaustive.py)

**C6  서비스 요구사항과 목적함수<br>**[planner/spec.py](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/spec.py)

**C7  최종 제약 조건 검사<br>**[planner/optimizer/feasibility.py](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/optimizer/feasibility.py)

**C8  시뮬레이션 설정으로 변환하는 코드<br>**[planner/predictor/llmservingsim.py](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/predictor/llmservingsim.py)

**C9  보정 조건과 정확도 영역<br>**[planner/predictor/calibration.py](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/planner/predictor/calibration.py)

**C10  Top K의 누락 위험 실험<br>**[docs/surrogate_topk_regret.md](https://github.com/swsok/heteropilot/blob/a27469a3b14c2be7def379ef5aab5a14eb0e0689/docs/surrogate_topk_regret.md)

### 추천 확인 순서

inventory와 topology로 현재 물리 표현을 확인한 뒤 candidate_generator의 단계별 필터를 읽는다. 이어 surrogate의 BinnedRooflineRanker와 exhaustive의 top_k 경로를 비교한다. 마지막으로 spec과 feasibility에서 추가할 처리량·비용 계약을 확정한다.

**중요한 구분** 제안한 새 모듈과 입력 필드는 아직 구현 제안이며, 기존 코드의 기능으로 표시하지 않았다. 코드 검토와 저장소의 결과 기록에 근거한 설계이며 이 보고서 작성 과정에서 새 하드웨어 성능 실험을 수행한 것은 아니다.
