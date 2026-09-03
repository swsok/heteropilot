"""Tiered-profile validation experiments E1-E4 (WORK_ORDER_tiered_profiles.md STEP 11).

E1  plan agreement    does Tier 0 change the planner's DECISION?
E2  budget Pareto     where should a limited measurement budget go?
E3  shape overlap     how much do models share (layer, key) / GEMM shapes?
E4  sensitivity       how robust are unowned-hardware conclusions to +-30%?

Common discipline: results land under outputs/tier_validation/e<N>/ as JSON
(with the §3.8 provenance block) plus a human-readable table; experiments
needing the simulator offer --dry-run, which only counts combinations (the
mode CI exercises).
"""
