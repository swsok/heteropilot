# Baselines + ablation (oracle goodput/J = 3.1376)

| group | strategy | status | goodput/J | regret | p99 TTFT (ms) | p99 TPOT (ms) | devices | note |
| --- | --- | :---: | ---: | ---: | ---: | ---: | ---: | --- |
| optimizer | exhaustive-oracle | feasible | 3.1376 | 0.000 | 267 | 29.2 | 1 | argmax objective over ALL simulatable feasible candidates (reference) |
| optimizer | proposed | feasible | 3.1376 | 0.000 | 267 | 29.2 | 1 | sim-guided pruned search + lexicographic rank |
| optimizer | greedy | feasible | 3.1376 | 0.000 | 267 | 29.2 | 1 | analytical roofline goodput/J proxy, NO simulation |
| resource | fastest-only | feasible | 3.1376 | 0.000 | 267 | 29.2 | 1 | only candidates on max-memory-bandwidth class ['RTXPRO6000'] (proxy) |
| resource | most-efficient-only | n/a | - | - | - | - | - | only candidates on max bandwidth-per-watt class ['RTX-A5000'] (proxy) |
| resource | least-device | feasible | 3.1376 | 0.000 | 267 | 29.2 | 1 | fewest active accelerators, tie-broken by objective |
| resource | simulator-blind | feasible | 2.1117 | 0.327 | 146 | 20.0 | 2 | fixed provisioning rule (fastest class, smallest TP, most replicas), no sim |
| architecture | aggregated | feasible | 3.1376 | 0.000 | 267 | 29.2 | 1 | aggregated only |
| architecture | homogeneous-P/D | feasible | 2.1001 | 0.331 | 4,942 | 22.2 | 2 | P/D with same backend on both roles |
| architecture | heterogeneous-P/D | n/a | - | - | - | - | - | P/D with different backends across roles |
| ablation | No-PD-Specialization | feasible | 3.1376 | 0.000 | 267 | 29.2 | 1 | P/D candidates removed from the search |
| ablation | No-Energy | feasible | 1.6616 | 0.470 | 983 | 18.0 | 4 | select by SLO-goodput (ignore energy); scored on true goodput/J |
| ablation | No-Topology | feasible | 3.1376 | 0.000 | 267 | 29.2 | 1 | P/D KV-transfer cost dropped at selection; scored on true (transfer-priced) metrics |
| ablation | No-Calibration | n/a | - | - | - | - | - | N/A: calibration is opt-in and not in the plan path, so the default already is no-calibration (needs a non-identity Phase-4 fit to contrast) |
| ablation | No-Uncertainty | n/a | - | - | - | - | - | N/A: robust margins default to 0 (needs non-zero Phase-4 calibration margins) |
| ablation | Static | n/a | - | - | - | - | - | N/A: no-replanning is a Phase-6 concept; pre-Phase-6 every plan is static |
