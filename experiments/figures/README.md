# Figures (work order §12, M6 paper artifacts)

Every figure here is an **LLMServingSim prediction, not a live measurement**, and
each is regenerated from a committed result artifact — no figure embeds numbers
that are not also in `experiments/results/`.

Regenerate the JSON-derived figures with one command (pure plotting, no sim):

```bash
experiments/scripts/make_figures.py             # all JSON-derived figures
experiments/scripts/make_figures.py --only exp1
```

| Figure | Experiment | Source data | Generator |
| --- | --- | --- | --- |
| `exp1_tp_sweep.png` | Exp 1 — same-GPU (A40) TP=1/2/4 sweep | `experiments/results/exp1_tp_sweep.json` | `make_figures.py` |
| `exp2_selection.png` | Exp 2 — heterogeneous selection (best goodput/J per class) | `experiments/results/exp2_selection.json` | `make_figures.py` |
| `baselines_regret.png` | §12 baselines + ablation | `experiments/results/baselines.json` | `make_figures.py` |
| `router_baselines.png` | §12 router baselines (RR/RAND/LOAD) | `experiments/results/router_baselines.json` | `make_figures.py` |
| `pd_4combo.png` | Exp 5 — P/D 4-combo vs aggregated | `experiments/results/pd_4combo.json` | `make_figures.py` |
| `surrogate.png` | §5.4 stage-6 surrogate top-K accuracy | `experiments/results/surrogate.json` | `make_figures.py` |
| `pd_network_sweep.png` | Exp 3 — P/D network bandwidth sweep | `experiments/results/pd_network_sweep_table.md` | `pd_network_sweep.py` (produced inline by `run_exp_pd.sh`) |

## What each figure shows

- **`exp1_tp_sweep.png`** — (a) p99 TTFT/TPOT falling with TP on a log axis;
  (b) tokens/J rising while average power rises. The pipeline reproduces the
  expected TP curve; under this saturating load higher TP is also more efficient
  (see `experiments/results/exp1_summary.md`).
- **`baselines_regret.png`** — regret vs the oracle optimum per strategy, colored
  by group (optimizer / resource / architecture / ablation). `proposed` and
  `greedy` sit at 0 (the optimum is analytically obvious here); the
  differentiators are `No-Energy` (0.47), `homogeneous-P/D` (0.33) and
  `simulator-blind` (0.33). N/A strategies (`most-efficient-only`,
  `heterogeneous-P/D`, `No-Calibration`, `No-Uncertainty`, `Static`) are shown as
  labeled zero-length bars, never fabricated (see `exp_baselines_summary.md`).
- **`exp2_selection.png`** — best SLO-goodput/J per placement class. A single
  RTXPRO6000 (1.697) edges two A5000s (1.634), and the mixed placement (1.043) is
  worst: right-sizing beats scale-out when demand fits one class (see
  `experiments/results/exp2_summary.md`). Reproduced from all feasible candidates
  because the committed plan YAML keeps only the Pareto frontier, which omits the
  dominated A5000-only best.
- **`router_baselines.png`** — p99 TTFT bars + SLO-goodput line per routing policy
  on a heterogeneous 4-replica deployment; LOAD wins the tail, RAND is worst (see
  `exp_router_summary.md`).
- **`pd_4combo.png`** — the four P/D role×backend combos vs the aggregated
  baseline. The four combos are byte-identical (1.081, feasible) because every
  NPU-touching combo is SIM-PROXY (RTXPRO6000 model — labeled, absolute rule 3);
  the aggregated baseline has higher tokens/J (1.655) but is INFEASIBLE, so P/D
  pays here by meeting the SLO (see `pd_4combo_table.md`).
- **`surrogate.png`** — stage-6 surrogate top-K: recall@K (does the exact optimum
  survive) and regret@K (goodput/J lost) vs K, with the N/K speedup each K buys.
  On this workload regret is 0 even at K=1 (78x) although recall only reaches 1 at
  K=20 — the objective has ties, so the surrogate picks a different candidate of
  equal value (see `exp_surrogate_summary.md`).
- **`pd_network_sweep.png`** — the §5.9 adoption crossing at the planning level
  (see `exp_pd_summary.md`).
