"""Build self-contained HTML slide decks from the committed figures.

Reads experiments/figures/*.png, base64-embeds them, and writes two
self-contained decks (no external assets except Google Fonts):
docs/slide_deck.html (English) and docs/slide_deck_ko.html (Korean).
Reproducible: re-run after regenerating figures. Content mirrors
docs/SLIDE_OUTLINE.md / docs/PROJECT_REPORT.md; the Korean deck is a
translation of the same slides, with figures, numbers and identifiers shared.
"""
# ruff: noqa: E501 - this file is mostly inline HTML/CSS template strings.

from __future__ import annotations

import base64
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIGDIR = REPO / "experiments" / "figures"
OUT = REPO / "docs" / "slide_deck.html"
OUT_KO = REPO / "docs" / "slide_deck_ko.html"


def data_uri(name: str) -> str:
    b = (FIGDIR / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()


def fig(name: str, alt: str) -> str:
    return f'<img src="{data_uri(name)}" alt="{alt}" />'


# Each slide: (eyebrow, title, body_html, is_data_slide)
SLIDES: list[tuple[str, str, str, bool]] = [
    ("", "HeteroPilot",
     """<p class="lede">SLO-goodput/J-optimal LLM serving on heterogeneous GPU/NPU clusters</p>
        <p class="sub">A calibrated, simulator-in-the-loop control plane &middot; fork of LLMServingSim</p>
        <p class="meta">Project report &middot; 2026-08-25</p>
        <p class="meta"><b class="warn">As of 2026-08-25, before D22 and D23.</b>
        The RNGD arm was still unmeasured when these slides were built. What the
        measurements later retracted, and what is still undetermined, is in
        <code>docs/PROJECT_REPORT.md</code> &sect;4.8.7 and &sect;4.9.</p>""", False),

    ("The problem", "Choosing how to serve is hard, and getting it wrong wastes energy",
     """<ul>
        <li>Serving clusters are increasingly <b>heterogeneous</b> — GPU generations, NPUs.</li>
        <li>Every deployment is a choice: <b>which accelerator, TP degree, replica count,
            aggregated vs Prefill/Decode split</b> — under TTFT/TPOT SLOs and a power cap.</li>
        <li>The trade-offs (latency &times; energy &times; SLO) are non-obvious; a wrong pick
            burns joules or misses SLOs.</li>
        </ul>
        <p class="pull">We optimize <b>energy per SLO-satisfying token</b>, not raw throughput.</p>""",
     False),

    ("Approach", "Enumerate, predict, rank",
     """<div class="flow">
          <span class="node">ServiceSpec + ClusterSpecV2</span><span class="arr">&rarr;</span>
          <span class="node">enumerate candidates</span><span class="arr">&rarr;</span>
          <span class="node accent">predict via LLMServingSim</span><span class="arr">&rarr;</span>
          <span class="node">rank (lexicographic)</span><span class="arr">&rarr;</span>
          <span class="node good">DeploymentPlan</span>
        </div>
        <ul>
        <li><b>Input:</b> model + traffic + SLOs + power cap; accelerator inventory + topology.</li>
        <li><b>Objective:</b> maximize <b>SLO-goodput / J</b> under the power cap; Pareto
            alternatives; a diagnosis when infeasible.</li>
        <li><b>Percentiles</b> (P50/P95/P99) gate the SLO — never means.</li>
        </ul>""", False),

    ("Key abstraction", "The execution island",
     """<ul>
        <li>An <b>island</b> = accelerators sharing one backend, mutually reachable by
            collectives, hosting one vLLM engine.</li>
        <li><b>TP/PP live only inside an island.</b> Heterogeneity is exploited across
            <b>replicas</b> or <b>Prefill/Decode roles</b> — never cross-vendor TP.</li>
        <li>Why: cross-backend collectives are impractical; this keeps every candidate
            realizable and the search space enumerable.</li>
        </ul>
        <p class="pull">The island is the unit of placement — and the reason the space is tractable.</p>""",
     False),

    ("Method &amp; trust", "Why you can trust the numbers",
     """<ul>
        <li><b>Simulator-in-the-loop, calibrated.</b> A40 sim-vs-real fit
            <span class="mono">real = &alpha;&middot;sim + &beta;</span> (TTFT &alpha;=1.02/&beta;=111ms,
            TPOT &alpha;=1.01) &mdash; <b class="good">~1.3&ndash;2% mean error (measured)</b>.</li>
        <li><b>Provenance</b> on every result: git, versions, spec hash, seed, command line.</li>
        <li><b>Honesty rule:</b> unmeasured hardware is labelled placeholder / SIM-PROXY,
            never presented as measured.</li>
        <li><b>Reproducible:</b> same spec + seed &rArr; byte-identical plan (tested).</li>
        </ul>""", False),

    ("Result &middot; Exp 1", "TP scaling validates the pipeline",
     f"""<div class="figrow"><div class="figbox">{fig('exp1_tp_sweep.png', 'TP sweep')}</div>
        <div class="callout"><span class="ktag">Finding</span>
        Same A40, TP 1/2/4 (TP=4 <b>profiled on real hardware</b>). Monotonic:
        p99 TTFT 106s&rarr;35s&rarr;<b>4.5s</b>, p99 TPOT 225&rarr;100&rarr;<b>52ms</b>,
        tok/J 1.86&rarr;<b>2.88</b>. Under saturation more TP is <b>also</b> more efficient;
        TP=4 is the only config near the SLO.</div></div>""", True),

    ("Result &middot; Exp 2", "Right-sizing beats scale-out",
     f"""<div class="figrow"><div class="figbox">{fig('exp2_selection.png', 'per-class goodput/J')}</div>
        <div class="callout"><span class="ktag">Finding</span>
        Best goodput/J per class: <b>RTXPRO6000 1.697 &gt; A5000 1.634 &gt; mixed 1.043</b>.
        One big GPU edges two small ones; mixing is worst when demand fits a single class.
        Heterogeneity pays only <b>past a capacity threshold</b>.</div></div>""", True),

    ("Result &middot; Exp 3 + 5", "When does Prefill/Decode split pay?",
     f"""<div class="figrow"><div class="figbox">{fig('pd_network_sweep.png', 'network sweep')}</div>
        <div class="callout"><span class="ktag">Finding</span>
        <b>Exp 3:</b> a bandwidth threshold below which P/D stops paying (reproduced
        planner-side <i>and</i> sim-level). <b>Exp 5:</b> the aggregated baseline is more
        efficient (1.655 tok/J) but <b class="bad">infeasible</b> &mdash; P/D pays
        <b>by meeting the SLO</b> (1.081, feasible).
        <span class="warn">NPU combos are SIM-PROXY.</span>
        <span class="warn">Superseded 2026-09-03: the SLO sweeps that were to replace
        these proxy rows retracted the RNGD energy win (D22), and the sub-second
        regime where P/D paid is undetermined &mdash; those candidates livelock
        (D23).</span></div></div>""", True),

    ("Result &middot; Exp 5", "Heterogeneous P/D &mdash; the four combos",
     f"""<div class="figrow"><div class="figbox">{fig('pd_4combo.png', '4-combo')}</div>
        <div class="callout"><span class="ktag">Honesty</span>
        The four P/D combos are <b>byte-identical</b> because the NPU is a relabelled GPU
        (<b class="warn">SIM-PROXY</b>) &mdash; that identity is the proof, not a result.
        A real GPU-vs-NPU efficiency story <b>needs NPU hardware</b>.</div></div>""", True),

    ("Result &middot; baselines", "What the sim-guided planner buys",
     f"""<div class="figrow"><div class="figbox">{fig('baselines_regret.png', 'regret')}</div>
        <div class="callout"><span class="ktag">Finding</span>
        Regret vs the exhaustive oracle. <b>proposed = 0.000</b> (pruning is sound);
        <b class="bad">No-Energy = 0.470</b> &mdash; energy-blind over-provisions, the
        <b>biggest lever</b>. simulator-blind &amp; homogeneous-P/D &asymp; 0.33.
        N/A rows are honest, not fabricated.</div></div>""", True),

    ("Result &middot; routing", "Load-aware routing wins the tail",
     f"""<div class="figrow"><div class="figbox">{fig('router_baselines.png', 'router')}</div>
        <div class="callout"><span class="ktag">Finding</span>
        Heterogeneous 4-replica fleet: <b class="good">LOAD p99 TTFT 314ms</b> vs
        <b class="bad">RAND 644ms</b> (~2&times; worse &mdash; it overloads the slow replicas).
        Energy is flat: routing moves latency, not joules.</div></div>""", True),

    ("Scaling &middot; 1", "Surrogate top-K &mdash; fewer simulations, no loss",
     f"""<div class="figrow"><div class="figbox">{fig('surrogate.png', 'surrogate')}</div>
        <div class="callout"><span class="ktag">Finding</span>
        A roofline ranker scores all candidates; only the top-K are simulated.
        <b>Regret 0.000 at every K down to K=1 (78&times; fewer sims)</b>; recall reaches 1
        only at K=20 &mdash; the objective has ties, so an equal-value candidate is chosen.
        Accuracy is <b>measured, never asserted</b>.</div></div>""", True),

    ("Scaling &middot; 2", "Parallel candidate simulation",
     """<ul>
        <li>Each candidate is an <b>isolated subprocess</b> &rArr; parallelize with no locking.</li>
        <li><b class="good">~5&ndash;7&times; wall speedup</b>: 78 sims ~8 min vs ~40&ndash;60 min;
            64-core load 2.7&rarr;21.</li>
        <li><b>Byte-identical</b> to sequential (sim parallel, assembly in candidate order).</li>
        </ul>
        <div class="statrow">
          <div class="stat"><div class="statnum bad">~40&ndash;60<span>min</span></div><div class="statlab">sequential</div></div>
          <div class="stat"><div class="statnum good">~8<span>min</span></div><div class="statlab">parallel (32-wide)</div></div>
          <div class="stat"><div class="statnum accent">78&times;</div><div class="statlab">surrogate cuts sim count</div></div>
        </div>
        <p class="pull">Surrogate cuts sim <i>count</i>; parallelism cuts sim <i>wall time</i> &mdash; complementary.</p>""",
     True),

    ("Status", "Implementation vs the work order",
     """<table class="phase">
        <tr><th>Phase</th><th>Scope</th><th>Status</th></tr>
        <tr><td>0</td><td>Baseline reproduction</td><td class="good">done</td></tr>
        <tr><td>1</td><td>Spec / inventory / islands</td><td class="good">done</td></tr>
        <tr><td>2</td><td>Offline planner (MVP)</td><td class="good">done</td></tr>
        <tr><td>3</td><td>Heterogeneous profiles</td><td class="good">done</td></tr>
        <tr><td>4</td><td>Real deployment + calibration</td><td class="good">done (CUDA)</td></tr>
        <tr><td>5</td><td>Topology-aware P/D</td><td class="good">core done</td></tr>
        <tr><td>6</td><td>Online replanning</td><td class="muted">not started (needs approval)</td></tr>
        </table>
        <p class="pull">11 merged PRs &middot; 284 tests green &middot; ruff + mypy clean &middot;
        end-to-end runnable on real GPU hardware today.</p>""", False),

    ("Limits", "What we don't claim",
     """<ul>
        <li><b class="warn">No NPU hardware yet</b> &rarr; Exp 4 (GPU vs NPU) not run; the
            Exp-5 NPU rows are SIM-PROXY.</li>
        <li>The analytical backend can't model per-flow link contention (needs ns3).</li>
        <li>Prefix caching off (upstream memory bug); calibration not yet in the plan path.</li>
        </ul>
        <p class="pull">These are stated on every figure &mdash; credibility through disclosure.</p>""",
     False),

    ("Future", "Close the last honesty gap: measure the NPUs",
     """<ul>
        <li><b>Measure Rebellions ATOM (&times;4) &amp; FuriosaAI RNGD (&times;4).</b> Converts
            Exp 4 and the Exp-5 NPU rows from SIM-PROXY to <b class="good">measured</b> &mdash;
            the biggest credibility win.</li>
        <li>NPU sim-vs-real calibration &rarr; unblocks the No-Calibration / No-Uncertainty ablations.</li>
        <li>Learned surrogate once a corpus exists; network-aware routing; Phase 6 replanning.</li>
        </ul>
        <p class="sub">The continuation path is sequenced in <span class="mono">docs/HANDOVER_NPU.md</span>.</p>""",
     False),

    ("Takeaways", "Three things to remember",
     """<div class="takeaways">
          <div class="tk"><div class="tknum">1</div><div>Heterogeneity is a <b>tool</b>: it
             pays only past a capacity / bandwidth threshold.</div></div>
          <div class="tk"><div class="tknum">2</div><div><b>Energy-awareness is the dominant
             lever</b> &mdash; ~47% of goodput/J.</div></div>
          <div class="tk"><div class="tknum">3</div><div>The planner <b>matches the oracle</b>
             while a surrogate makes it cheap; real NPU numbers close the last gap.</div></div>
        </div>""", False),
]


# Korean translation of SLIDES, same shape. Figures, numbers and identifiers
# (TP, TTFT/TPOT, SLO, goodput/J, SIM-PROXY, spec names) stay in English -
# they are shared vocabulary with the code and the figures' axis labels.
SLIDES_KO: list[tuple[str, str, str, bool]] = [
    ("", "HeteroPilot",
     """<p class="lede">이기종 GPU/NPU 클러스터에서 SLO-goodput/J 최적 LLM 서빙</p>
        <p class="sub">보정된 simulator-in-the-loop 제어 평면 &middot; LLMServingSim fork</p>
        <p class="meta">프로젝트 보고 &middot; 2026-08-25</p>
        <p class="meta"><b class="warn">2026-08-25 기준, D22&middot;D23 이전.</b>
        이 슬라이드를 만들 당시 RNGD는 아직 측정 전이었다. 이후 측정이 무엇을
        철회했고 무엇이 미결로 남았는지는 <code>docs/PROJECT_REPORT.md</code>
        &sect;4.8.7과 &sect;4.9에 있다.</p>""", False),

    ("문제", "서빙 구성 선택은 어렵고, 잘못 고르면 에너지를 낭비한다",
     """<ul>
        <li>서빙 클러스터는 점점 <b>이기종화</b>되고 있다 &mdash; GPU 세대, NPU.</li>
        <li>모든 배포는 선택이다: <b>어떤 가속기, TP degree, replica 수,
            aggregated vs Prefill/Decode 분리</b> &mdash; TTFT/TPOT SLO와 전력 상한 아래에서.</li>
        <li>지연 &times; 에너지 &times; SLO 트레이드오프는 자명하지 않다;
            잘못 고르면 joule을 태우거나 SLO를 놓친다.</li>
        </ul>
        <p class="pull">우리는 원시 처리량이 아니라 <b>SLO 만족 토큰당 에너지</b>를 최적화한다.</p>""",
     False),

    ("접근", "열거하고, 예측하고, 랭킹한다",
     """<div class="flow">
          <span class="node">ServiceSpec + ClusterSpecV2</span><span class="arr">&rarr;</span>
          <span class="node">후보 열거</span><span class="arr">&rarr;</span>
          <span class="node accent">LLMServingSim으로 예측</span><span class="arr">&rarr;</span>
          <span class="node">랭킹 (lexicographic)</span><span class="arr">&rarr;</span>
          <span class="node good">DeploymentPlan</span>
        </div>
        <ul>
        <li><b>입력:</b> 모델 + 트래픽 + SLO + 전력 상한; 가속기 인벤토리 + 토폴로지.</li>
        <li><b>목적함수:</b> 전력 상한 아래에서 <b>SLO-goodput / J</b> 최대화; Pareto
            대안 제시; infeasible이면 진단을 출력.</li>
        <li>SLO 판정은 평균이 아니라 <b>percentile</b> (P50/P95/P99).</li>
        </ul>""", False),

    ("핵심 추상화", "Execution island",
     """<ul>
        <li><b>Island</b> = 동일 backend를 공유하고, collective로 상호 도달 가능하며,
            vLLM 엔진 하나를 호스팅하는 가속기 집합.</li>
        <li><b>TP/PP는 island 내부에만 존재.</b> 이기종성은 <b>replica</b> 또는
            <b>Prefill/Decode 역할</b> 단위로만 활용 &mdash; cross-vendor TP는 없다.</li>
        <li>이유: backend 간 collective는 비현실적이다. 이 제약이 모든 후보를
            실현 가능하게 만들고, 탐색 공간을 열거 가능하게 유지한다.</li>
        </ul>
        <p class="pull">Island가 배치의 단위이며, 탐색이 감당 가능해지는 이유다.</p>""",
     False),

    ("방법론 &middot; 신뢰", "이 수치를 믿을 수 있는 이유",
     """<ul>
        <li><b>Simulator-in-the-loop + 보정.</b> A40 sim-vs-real 적합
            <span class="mono">real = &alpha;&middot;sim + &beta;</span> (TTFT &alpha;=1.02/&beta;=111ms,
            TPOT &alpha;=1.01) &mdash; <b class="good">평균 오차 ~1.3&ndash;2% (실측)</b>.</li>
        <li>모든 결과에 <b>provenance</b>: git, 버전, spec 해시, seed, 커맨드라인.</li>
        <li><b>정직성 규칙:</b> 미측정 하드웨어는 placeholder / SIM-PROXY로 표기하고,
            실측으로 제시하지 않는다.</li>
        <li><b>재현성:</b> 동일 spec + seed &rArr; 바이트 동일 plan (테스트로 검증).</li>
        </ul>""", False),

    ("결과 &middot; Exp 1", "TP 스케일링이 파이프라인을 검증한다",
     f"""<div class="figrow"><div class="figbox">{fig('exp1_tp_sweep.png', 'TP sweep')}</div>
        <div class="callout"><span class="ktag">발견</span>
        동일 A40, TP 1/2/4 (TP=4는 <b>실물에서 프로파일</b>). 단조 개선:
        p99 TTFT 106s&rarr;35s&rarr;<b>4.5s</b>, p99 TPOT 225&rarr;100&rarr;<b>52ms</b>,
        tok/J 1.86&rarr;<b>2.88</b>. 포화 상태에서는 TP를 높이는 것이 <b>효율까지</b> 개선한다;
        SLO 근처에 있는 구성은 TP=4뿐.</div></div>""", True),

    ("결과 &middot; Exp 2", "Right-sizing이 scale-out을 이긴다",
     f"""<div class="figrow"><div class="figbox">{fig('exp2_selection.png', 'per-class goodput/J')}</div>
        <div class="callout"><span class="ktag">발견</span>
        클래스별 최고 goodput/J: <b>RTXPRO6000 1.697 &gt; A5000 1.634 &gt; mixed 1.043</b>.
        큰 GPU 한 장이 작은 두 장을 근소하게 앞서고, 수요가 단일 클래스에 담기면
        혼합이 최악이다. 이기종성은 <b>용량 임계를 넘어야</b> 비로소 이득이 된다.</div></div>""", True),

    ("결과 &middot; Exp 3 + 5", "Prefill/Decode 분리는 언제 이득인가?",
     f"""<div class="figrow"><div class="figbox">{fig('pd_network_sweep.png', 'network sweep')}</div>
        <div class="callout"><span class="ktag">발견</span>
        <b>Exp 3:</b> 그 아래에서는 P/D의 이득이 사라지는 대역폭 임계가 존재
        (planner 수준과 sim 수준 <i>모두</i>에서 재현). <b>Exp 5:</b> aggregated 기준선이
        더 효율적(1.655 tok/J)이지만 <b class="bad">infeasible</b> &mdash; P/D는
        <b>SLO를 맞추는 방식으로</b> 이득을 낸다 (1.081, feasible).
        <span class="warn">NPU 조합은 SIM-PROXY.</span>
        <span class="warn">2026-09-03 supersede: 이 proxy 행을 대체할 SLO sweep이
        RNGD 에너지 우위를 철회했고(D22), P/D가 이득을 내던 sub-second 구간은
        미결이다 &mdash; 해당 후보들이 livelock한다(D23).</span></div></div>""", True),

    ("결과 &middot; Exp 5", "이기종 P/D &mdash; 네 가지 조합",
     f"""<div class="figrow"><div class="figbox">{fig('pd_4combo.png', '4-combo')}</div>
        <div class="callout"><span class="ktag">정직성</span>
        네 P/D 조합이 <b>바이트 동일</b>한 이유는 NPU가 GPU의 라벨만 바꾼 것
        (<b class="warn">SIM-PROXY</b>)이기 때문 &mdash; 이 동일성은 증명이지 결과가 아니다.
        진짜 GPU-vs-NPU 효율 스토리에는 <b>NPU 하드웨어가 필요하다</b>.</div></div>""", True),

    ("결과 &middot; 기준선", "시뮬레이터 기반 플래너가 사주는 것",
     f"""<div class="figrow"><div class="figbox">{fig('baselines_regret.png', 'regret')}</div>
        <div class="callout"><span class="ktag">발견</span>
        Exhaustive oracle 대비 regret. <b>proposed = 0.000</b> (pruning이 건전);
        <b class="bad">No-Energy = 0.470</b> &mdash; 에너지를 무시하면 과잉 프로비저닝,
        <b>가장 큰 레버</b>다. simulator-blind와 homogeneous-P/D는 &asymp; 0.33.
        N/A 행은 정직한 공백이지 조작이 아니다.</div></div>""", True),

    ("결과 &middot; 라우팅", "부하 인지 라우팅이 tail을 이긴다",
     f"""<div class="figrow"><div class="figbox">{fig('router_baselines.png', 'router')}</div>
        <div class="callout"><span class="ktag">발견</span>
        이기종 4-replica 구성: <b class="good">LOAD p99 TTFT 314ms</b> vs
        <b class="bad">RAND 644ms</b> (~2&times; 악화 &mdash; 느린 replica를 과부하시킨다).
        에너지는 평평하다: 라우팅은 지연을 움직이지, joule을 움직이지 않는다.</div></div>""", True),

    ("스케일링 &middot; 1", "Surrogate top-K &mdash; 시뮬레이션은 줄이고 손실은 없이",
     f"""<div class="figrow"><div class="figbox">{fig('surrogate.png', 'surrogate')}</div>
        <div class="callout"><span class="ktag">발견</span>
        Roofline 랭커가 전체 후보를 점수화하고, top-K만 시뮬레이션한다.
        <b>K=1(시뮬 78&times; 감소)까지 모든 K에서 regret 0.000</b>; recall은 K=20에서야
        1이 되는데, 목적함수에 동점이 있어 같은 값의 다른 후보가 선택되기 때문이다.
        정확도는 <b>단언이 아니라 측정</b>이다.</div></div>""", True),

    ("스케일링 &middot; 2", "후보 병렬 시뮬레이션",
     """<ul>
        <li>후보 하나가 <b>격리된 subprocess</b> &rArr; 락 없이 병렬화.</li>
        <li><b class="good">~5&ndash;7&times; wall 단축</b>: 78 시뮬 ~8분 vs ~40&ndash;60분;
            64코어 load 2.7&rarr;21.</li>
        <li>순차 실행과 <b>바이트 동일</b> (시뮬은 병렬, 조립은 후보 순서).</li>
        </ul>
        <div class="statrow">
          <div class="stat"><div class="statnum bad">~40&ndash;60<span>분</span></div><div class="statlab">순차</div></div>
          <div class="stat"><div class="statnum good">~8<span>분</span></div><div class="statlab">병렬 (32-wide)</div></div>
          <div class="stat"><div class="statnum accent">78&times;</div><div class="statlab">surrogate의 시뮬 수 절감</div></div>
        </div>
        <p class="pull">Surrogate는 시뮬 <i>개수</i>를, 병렬화는 <i>wall time</i>을 줄인다 &mdash; 상보적.</p>""",
     True),

    ("현황", "작업지시서 대비 구현 현황",
     """<table class="phase">
        <tr><th>Phase</th><th>범위</th><th>상태</th></tr>
        <tr><td>0</td><td>Baseline 재현</td><td class="good">완료</td></tr>
        <tr><td>1</td><td>Spec / inventory / island</td><td class="good">완료</td></tr>
        <tr><td>2</td><td>오프라인 플래너 (MVP)</td><td class="good">완료</td></tr>
        <tr><td>3</td><td>이기종 프로파일</td><td class="good">완료</td></tr>
        <tr><td>4</td><td>실배포 + 보정</td><td class="good">완료 (CUDA)</td></tr>
        <tr><td>5</td><td>토폴로지 인지 P/D</td><td class="good">핵심 완료</td></tr>
        <tr><td>6</td><td>온라인 replanning</td><td class="muted">미착수 (승인 필요)</td></tr>
        </table>
        <p class="pull">PR 11건 병합 &middot; 284개 테스트 green &middot; ruff + mypy clean &middot;
        오늘 기준 실물 GPU에서 end-to-end 실행 가능.</p>""", False),

    ("한계", "우리가 주장하지 않는 것",
     """<ul>
        <li><b class="warn">아직 NPU 하드웨어 없음</b> &rarr; Exp 4 (GPU vs NPU) 미실행;
            Exp-5의 NPU 행은 SIM-PROXY.</li>
        <li>Analytical 백엔드는 per-flow 링크 경합을 모델링하지 못한다 (ns3 필요).</li>
        <li>Prefix caching off (upstream 메모리 버그); 보정은 아직 plan 경로에 미반영.</li>
        </ul>
        <p class="pull">모든 그림에 이 사실이 명시돼 있다 &mdash; 공개를 통한 신뢰.</p>""",
     False),

    ("향후", "마지막 정직성 격차를 닫는다: NPU 측정",
     """<ul>
        <li><b>Rebellions ATOM (&times;4)과 FuriosaAI RNGD (&times;4)를 측정.</b> Exp 4와
            Exp-5의 NPU 행을 SIM-PROXY에서 <b class="good">measured</b>로 전환 &mdash;
            가장 큰 신뢰성 이득.</li>
        <li>NPU sim-vs-real 보정 &rarr; No-Calibration / No-Uncertainty ablation의 차단 해제.</li>
        <li>코퍼스가 쌓이면 학습 surrogate; 네트워크 인지 라우팅; Phase 6 replanning.</li>
        </ul>
        <p class="sub">이어지는 경로는 <span class="mono">docs/HANDOVER_NPU.md</span>에 정리돼 있다.</p>""",
     False),

    ("정리", "기억할 세 가지",
     """<div class="takeaways">
          <div class="tk"><div class="tknum">1</div><div>이기종성은 <b>도구</b>다:
             용량 / 대역폭 임계를 넘어야 이득이 된다.</div></div>
          <div class="tk"><div class="tknum">2</div><div><b>에너지 인지가 지배적 레버</b>
             &mdash; goodput/J의 ~47%.</div></div>
          <div class="tk"><div class="tknum">3</div><div>플래너는 <b>oracle과 일치</b>하고
             surrogate가 이를 저렴하게 만든다; 실측 NPU 수치가 마지막 격차를 닫는다.</div></div>
        </div>""", False),
]

_HONESTY = {
    "en": "LLMServingSim prediction &mdash; measured / SIM-PROXY labelled inline",
    "ko": "LLMServingSim 예측 &mdash; 실측 / SIM-PROXY 여부는 본문에 표기",
}


def render(slides: list[tuple[str, str, str, bool]], lang: str) -> str:
    parts = []
    total = len(slides)
    for i, (eyebrow, title, body, is_data) in enumerate(slides):
        cls = "slide" + (" title-slide" if i == 0 else "")
        eb = f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ""
        foot = f'<div class="honesty">{_HONESTY[lang]}</div>' if is_data else ""
        num = f'<div class="pagenum">{i + 1:02d} / {total:02d}</div>'
        parts.append(
            f'<section class="{cls}" data-i="{i}">{eb}'
            f'<h1>{title}</h1><div class="body">{body}</div>{foot}{num}</section>'
        )
    slides_html = "\n".join(parts)

    html = _TEMPLATE.replace("__SLIDES__", slides_html).replace("__TOTAL__", str(total))
    if lang == "ko":
        html = _localize_ko(html)
    return html


def _localize_ko(html: str) -> str:
    """Chrome-level localization applied only to the Korean build.

    Kept as post-render replacements (not template placeholders) so the English
    deck's bytes cannot change: the EN path never touches this function. Each
    replacement targets a string that exists exactly once in the template.
    """
    subs = [
        ('<html lang="en">', '<html lang="ko">'),
        # IBM Plex Sans has no Hangul; add the KR family and put it in the stacks.
        ("family=IBM+Plex+Mono:wght@400;500&display=swap",
         "family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap"),
        ('--sans:"IBM Plex Sans",system-ui,sans-serif;',
         '--sans:"IBM Plex Sans","IBM Plex Sans KR",system-ui,sans-serif;'),
        ('--cond:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;',
         '--cond:"IBM Plex Sans Condensed","IBM Plex Sans KR","IBM Plex Sans",system-ui,sans-serif;'),
        # Korean line breaking: never split inside a word.
        ("-webkit-font-smoothing:antialiased;line-height:1.5}",
         "-webkit-font-smoothing:antialiased;line-height:1.5;word-break:keep-all}"),
        ('<button id="prev" aria-label="Previous slide">&larr; Prev</button>',
         '<button id="prev" aria-label="이전 슬라이드">&larr; 이전</button>'),
        ('<button id="next" aria-label="Next slide">Next &rarr;</button>',
         '<button id="next" aria-label="다음 슬라이드">다음 &rarr;</button>'),
        ('<span class="hint">&larr;/&rarr; or Space</span>',
         '<span class="hint">&larr;/&rarr; 또는 Space</span>'),
    ]
    for old, new in subs:
        assert html.count(old) == 1, f"localization anchor not unique: {old[:50]!r}"
        html = html.replace(old, new)
    return html


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>HeteroPilot</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" />
<style>
:root{
  --bg:#eef0f3; --surface:#ffffff; --text:#171d29; --muted:#5f6b7c;
  --border:#e0e4ea; --faint:#f2f4f7;
  --accent:#c47d22; --accent-soft:#f3e2c6;
  --good:#1c8577; --bad:#c9564c;
  --sans:"IBM Plex Sans",system-ui,sans-serif;
  --cond:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,monospace;
}
:root:not([data-theme="light"]){ @media (prefers-color-scheme:dark){
  --bg:#080b11; --surface:#121821; --text:#e6ebf2; --muted:#93a0b1;
  --border:#242f3d; --faint:#0f151d;
  --accent:#e9a544; --accent-soft:#3a2e1a; --good:#3fb8a4; --bad:#e0736a;
}}
:root[data-theme="dark"]{
  --bg:#080b11; --surface:#121821; --text:#e6ebf2; --muted:#93a0b1;
  --border:#242f3d; --faint:#0f151d;
  --accent:#e9a544; --accent-soft:#3a2e1a; --good:#3fb8a4; --bad:#e0736a;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--text);font-family:var(--sans);
  -webkit-font-smoothing:antialiased;line-height:1.5}
.deck{height:100vh;display:grid;place-items:center;padding:2.5vh 2.5vw}
.stage{width:min(100%,1180px);aspect-ratio:16/9;position:relative}

.slide{position:absolute;inset:0;background:var(--surface);
  border:1px solid var(--border);border-radius:14px;
  padding:clamp(28px,4.2vw,60px) clamp(32px,5vw,72px);
  display:none;flex-direction:column;overflow:hidden;
  box-shadow:0 18px 50px -28px rgba(10,20,40,.45)}
.slide::before{content:"";position:absolute;left:0;top:0;bottom:0;width:6px;background:var(--accent)}
.slide.on{display:flex;animation:in .4s ease both}
@keyframes in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@media (prefers-reduced-motion:reduce){.slide.on{animation:none}}

.eyebrow{font-family:var(--mono);font-size:clamp(11px,1.05vw,14px);
  letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin-bottom:.7em}
.slide h1{font-family:var(--cond);font-weight:700;
  font-size:clamp(26px,3.9vw,50px);line-height:1.04;letter-spacing:-.01em;
  text-wrap:balance;margin:0 0 .6em}
.body{font-size:clamp(15px,1.55vw,21px);flex:1;min-height:0}
.body ul{margin:.2em 0;padding-left:1.1em}
.body li{margin:.5em 0;max-width:60ch}
.body li::marker{color:var(--accent)}
b{font-weight:600}
.good{color:var(--good);font-weight:600}
.bad{color:var(--bad);font-weight:600}
.warn{color:var(--accent);font-weight:600}
.mono{font-family:var(--mono);font-size:.92em}
.muted{color:var(--muted)}
.pull{margin-top:1em;padding:.6em .9em;background:var(--faint);
  border-left:3px solid var(--accent);border-radius:0 8px 8px 0;
  font-size:.98em;max-width:64ch}
.sub{color:var(--muted);font-size:.95em}

/* title slide */
.title-slide{justify-content:center}
.title-slide h1{font-size:clamp(46px,8.5vw,110px);margin:0}
.title-slide .lede{font-family:var(--cond);font-weight:600;
  font-size:clamp(20px,2.7vw,34px);margin:.35em 0 .1em;max-width:22ch;color:var(--text)}
.title-slide .sub{margin:.6em 0 0}
.title-slide .meta{font-family:var(--mono);font-size:13px;color:var(--muted);margin-top:2.2em;letter-spacing:.05em}

/* flow diagram */
.flow{display:flex;flex-wrap:wrap;align-items:center;gap:.5em;margin:.2em 0 1.2em}
.node{font-family:var(--mono);font-size:clamp(11px,1.15vw,15px);padding:.45em .7em;
  background:var(--faint);border:1px solid var(--border);border-radius:7px;white-space:nowrap}
.node.accent{border-color:var(--accent);color:var(--accent);background:var(--accent-soft)}
.node.good{border-color:var(--good);color:var(--good)}
.arr{color:var(--muted)}

/* figure slides */
.figrow{display:grid;grid-template-columns:1.55fr 1fr;gap:clamp(18px,2.4vw,34px);
  align-items:center;height:100%}
.figbox{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:10px;display:grid;place-items:center;min-height:0}
.figbox img{max-width:100%;max-height:52vh;object-fit:contain;border-radius:4px}
.callout{font-size:clamp(14px,1.4vw,19px);align-self:center}
.ktag{display:inline-block;font-family:var(--mono);font-size:11px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--accent);border:1px solid var(--accent);
  border-radius:20px;padding:.2em .7em;margin-bottom:.7em}

/* stats */
.statrow{display:flex;gap:1.2em;margin:1.2em 0}
.stat{flex:1;background:var(--faint);border:1px solid var(--border);border-radius:10px;
  padding:1em;text-align:center}
.statnum{font-family:var(--cond);font-weight:700;font-size:clamp(26px,3.6vw,44px);
  line-height:1;font-variant-numeric:tabular-nums}
.statnum span{font-family:var(--sans);font-weight:500;font-size:.4em;color:var(--muted);margin-left:.15em}
.statlab{font-size:.8em;color:var(--muted);margin-top:.5em}

/* phase table */
table.phase{border-collapse:collapse;width:100%;font-size:clamp(13px,1.35vw,18px)}
table.phase th,table.phase td{text-align:left;padding:.42em .7em;border-bottom:1px solid var(--border)}
table.phase th{font-family:var(--mono);font-size:.72em;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted)}
table.phase td:first-child{font-family:var(--mono);color:var(--accent);width:3em}
table.phase td.good{color:var(--good);font-weight:600}
table.phase td.muted{color:var(--muted)}

/* takeaways */
.takeaways{display:flex;flex-direction:column;gap:1em;margin-top:.4em;
  font-size:clamp(16px,1.7vw,23px)}
.tk{display:flex;gap:.8em;align-items:baseline}
.tknum{font-family:var(--cond);font-weight:700;font-size:1.7em;color:var(--accent);
  line-height:1;min-width:1.1em}

.honesty{position:absolute;left:0;right:0;bottom:14px;text-align:center;
  font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;color:var(--muted);opacity:.85}
.pagenum{position:absolute;right:22px;bottom:14px;font-family:var(--mono);
  font-size:11px;color:var(--muted)}

/* progress + nav */
.progress{position:fixed;left:0;top:0;height:3px;background:var(--accent);
  width:0;transition:width .3s ease;z-index:5}
.nav{position:fixed;bottom:14px;left:50%;transform:translateX(-50%);
  display:flex;gap:8px;align-items:center;z-index:5}
.nav button{font-family:var(--mono);font-size:13px;background:var(--surface);
  color:var(--text);border:1px solid var(--border);border-radius:7px;
  padding:.35em .7em;cursor:pointer}
.nav button:hover{border-color:var(--accent);color:var(--accent)}
.nav button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.hint{font-family:var(--mono);font-size:11px;color:var(--muted)}

@media (max-width:720px){
  .figrow{grid-template-columns:1fr;overflow-y:auto}
  .figbox img{max-height:32vh}
  .statrow{flex-direction:column}
}

@media print{
  @page{size:landscape;margin:0}
  body{background:#fff}
  .deck{display:block;height:auto;padding:0}
  .stage{width:100%;aspect-ratio:auto}
  .slide{position:static;display:flex;page-break-after:always;
    border:none;border-radius:0;box-shadow:none;height:100vh;aspect-ratio:16/9}
  .progress,.nav{display:none}
}
</style>
</head>
<body>
<div class="progress" id="progress"></div>
<div class="deck"><div class="stage" id="stage">
__SLIDES__
</div></div>
<div class="nav">
  <button id="prev" aria-label="Previous slide">&larr; Prev</button>
  <span class="hint">&larr;/&rarr; or Space</span>
  <button id="next" aria-label="Next slide">Next &rarr;</button>
</div>
<script>
(function(){
  var slides=[].slice.call(document.querySelectorAll('.slide'));
  var total=slides.length, cur=0;
  var progress=document.getElementById('progress');
  function show(n){
    cur=Math.max(0,Math.min(total-1,n));
    slides.forEach(function(s,i){s.classList.toggle('on',i===cur);});
    progress.style.width=((cur+1)/total*100)+'%';
    try{location.hash=cur+1;}catch(e){}
  }
  function go(d){show(cur+d);}
  document.getElementById('next').onclick=function(){go(1)};
  document.getElementById('prev').onclick=function(){go(-1)};
  document.addEventListener('keydown',function(e){
    if(e.key==='ArrowRight'||e.key===' '||e.key==='PageDown'){e.preventDefault();go(1);}
    else if(e.key==='ArrowLeft'||e.key==='PageUp'){e.preventDefault();go(-1);}
    else if(e.key==='Home'){show(0);} else if(e.key==='End'){show(total-1);}
  });
  var start=parseInt((location.hash||'').replace('#',''),10);
  show(isFinite(start)&&start>0?start-1:0);
})();
</script>
</body>
</html>
"""


def main() -> int:
    assert len(SLIDES) == len(SLIDES_KO), (
        f"deck translations out of sync: en={len(SLIDES)} ko={len(SLIDES_KO)}"
    )
    for out, slides, lang in ((OUT, SLIDES, "en"), (OUT_KO, SLIDES_KO, "ko")):
        out.write_text(render(slides, lang))
        kb = out.stat().st_size / 1024
        print(f"wrote {out.relative_to(REPO)} ({kb:.0f} KB, {len(slides)} slides, {lang})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
