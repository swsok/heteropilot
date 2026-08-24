# Figures (work order §12, M6 paper artifacts)

Every figure here is an **LLMServingSim prediction, not a live measurement**, and
each is regenerated from a committed result artifact — no figure embeds numbers
that are not also in `experiments/results/`.

Regenerate the JSON-derived figures with one command (pure plotting, no sim):

```bash
experiments/scripts/make_figures.py             # exp1 + baselines
experiments/scripts/make_figures.py --only exp1
```

| Figure | Experiment | Source data | Generator |
| --- | --- | --- | --- |
| `exp1_tp_sweep.png` | Exp 1 — same-GPU (A40) TP=1/2/4 sweep | `experiments/results/exp1_tp_sweep.json` | `make_figures.py` |
| `baselines_regret.png` | §12 baselines + ablation | `experiments/results/baselines.json` | `make_figures.py` |
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
- **`pd_network_sweep.png`** — the §5.9 adoption crossing at the planning level
  (see `exp_pd_summary.md`).

## Not yet packaged as PNG

Exp 2 (heterogeneous selection) and Exp 5 (P/D 4-combo) currently ship as result
tables (`exp2_summary.md`, `pd_4combo_table.md`) rather than committed JSON, so
they have no `make_figures.py` entry yet. Re-running their drivers to emit a
structured JSON is the prerequisite for adding those figures — deferred to avoid
re-simulation here.
