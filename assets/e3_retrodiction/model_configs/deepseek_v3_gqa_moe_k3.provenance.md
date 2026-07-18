---
ac_version: 0.4.0
quality_model_version: effective_capacity_v2
git_commit: c170cda
experiment_date: 2026-07-17
agent: gate2-wave1
config: deepseek_v3_gqa_moe_k3.json
---

# Provenance — deepseek_v3_gqa_moe_k3.json

Case-1b counterfactual: V3 with GQA-8 attention instead of MLA. Wave 2 applies
`swap_attention_to_mla:latent_dim=512,d_rope=64`. Measured by AC ledger:
total = 674.73 B, active = 41.26 B.

| Field | Value | Source | Tier | Date accessed |
|---|---|---|---|---|
| shape (61 layers, d_model 7168, MoE 256e top-8, dense FFN 18432, vocab 129280, rope 10000 + YaRN 40/4096) | — | same sources as deepseek_v3_mla_moe_k0.provenance.md (arXiv:2412.19437; HF deepseek-ai/DeepSeek-V3) | T2 | 2026-07-17 |
| first-3-dense bands [0,1,2] dense + [3..60] MoE | — | arXiv:2412.19437: "We substitute all FFNs except for the first three layers with MoE layers" | T2 | 2026-07-17 |
| attention full/GQA: 128 q heads, 8 KV heads, d_head 128 | — | counterfactual GQA-8 group size matching Llama-3-70B GQA ratio (arXiv:2407.21783 Table 3); head count 128 kept from V3 MLA | T3 (counterfactual) | 2026-07-17 |
| parallelism tp8/pp1/dp8/ep8 | — | synthetic storage default | T3 | 2026-07-17 |
