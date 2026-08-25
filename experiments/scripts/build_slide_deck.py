"""Build a self-contained HTML slide deck from the committed figures.

Reads experiments/figures/*.png, base64-embeds them, and writes a single
self-contained docs/slide_deck.html (no external assets except Google Fonts).
Reproducible: re-run after regenerating figures. Content mirrors
docs/SLIDE_OUTLINE.md / docs/PROJECT_REPORT.md.
"""
# ruff: noqa: E501 - this file is mostly inline HTML/CSS template strings.

from __future__ import annotations

import base64
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIGDIR = REPO / "experiments" / "figures"
OUT = REPO / "docs" / "slide_deck.html"


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
        <p class="meta">Project report &middot; 2026-08-25</p>""", False),

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
        <span class="warn">NPU combos are SIM-PROXY.</span></div></div>""", True),

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


def render() -> str:
    parts = []
    total = len(SLIDES)
    for i, (eyebrow, title, body, is_data) in enumerate(SLIDES):
        cls = "slide" + (" title-slide" if i == 0 else "")
        eb = f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ""
        foot = ('<div class="honesty">LLMServingSim prediction &mdash; '
                'measured / SIM-PROXY labelled inline</div>') if is_data else ""
        num = f'<div class="pagenum">{i + 1:02d} / {total:02d}</div>'
        parts.append(
            f'<section class="{cls}" data-i="{i}">{eb}'
            f'<h1>{title}</h1><div class="body">{body}</div>{foot}{num}</section>'
        )
    slides_html = "\n".join(parts)

    return _TEMPLATE.replace("__SLIDES__", slides_html).replace("__TOTAL__", str(total))


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
    OUT.write_text(render())
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(REPO)} ({kb:.0f} KB, {len(SLIDES)} slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
