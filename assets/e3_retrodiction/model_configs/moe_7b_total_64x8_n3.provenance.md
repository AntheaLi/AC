---
ac_version: 0.4.0
quality_model_version: effective_capacity_v2
git_commit: c170cda
experiment_date: 2026-07-17
agent: gate2-wave1
config: moe_7b_total_64x8_n3.json
---

# Provenance — moe_7b_total_64x8_n3.json

N3-literal negative control: plan-specified 7B-total MoE (64 experts, top-8).
Plan expects INFEASIBLE from expert spill; audit arithmetic (bf16 ≈ 14 GB
total, TP8 → 1.75 GB/GPU ≪ 640 GB spill sentinel) predicts FEASIBLE — the
discrepancy is the point of the control and is double-recorded in
`validation/prereg/e3_cases.md`. Measured by AC ledger: total = 6.91 B,
active = 1.27 B.

| Field | Value | Source | Tier | Date accessed |
|---|---|---|---|---|
| 32 layers, d_model 2048, 16 heads / 4 KV, 64 experts top-8, expert_dim 512, no shared expert, vocab 32000 | — | synthetic plan-literal shape (7B-total MoE) | T3 (synthetic) | 2026-07-17 |
| parallelism tp8/pp1/dp1, ep=1 | — | plan-literal: experts sharded by TP only | T3 (synthetic) | 2026-07-17 |
