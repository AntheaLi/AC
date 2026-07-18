---
ac_version: 0.4.0
quality_model_version: effective_capacity_v2
git_commit: c170cda
experiment_date: 2026-07-17
agent: gate2-wave1
config: moe_430b_total_64x8_n3.json
---

# Provenance — moe_430b_total_64x8_n3.json

N3-scaled negative control: intent-preserving scale-up of the N3 MoE-spill
idea to 430B total (64 experts, top-8). bf16 weights ≈ 848 GB: TP8EP1 ≈
106 GB/GPU = 1.33× HBM80 → soft spill penalty + hbm_spill_warning expected;
TP1EP1 ≈ 848 GB > 10× HBM80 → hard INFEASIBLE sentinel expected. Measured by
AC ledger: total = 424.08 B, active = 63.31 B.

| Field | Value | Source | Tier | Date accessed |
|---|---|---|---|---|
| 64 layers, d_model 8192, 64 heads / 8 KV, 64 experts top-8, expert_dim 4096, no shared expert, vocab 128256, rope base 500000, eps 1e-5 | — | synthetic scale-up composed from Llama-3-70B shape vocabulary (arXiv:2407.21783 Table 3) to hit the spill regime | T3 (synthetic) | 2026-07-17 |
| parallelism tp8/pp1/dp1, ep=1 | — | soft-spill arm; wave 2 also runs TP1 arm for the hard sentinel | T3 (synthetic) | 2026-07-17 |
