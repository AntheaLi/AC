---
ac_version: 0.4.0
quality_model_version: effective_capacity_v2
git_commit: c170cda
experiment_date: 2026-07-17
agent: gate2-wave1
config: deepseek_v3_mha_moe_k3.json
---

# Provenance — deepseek_v3_mha_moe_k3.json

Case-1a counterfactual: V3 with full MHA (128 q / 128 KV heads) instead of MLA.
Wave 2 applies `swap_attention_to_mla:latent_dim=512,d_rope=64`. Measured by
AC ledger: total = 688.16 B, active = 54.69 B.

| Field | Value | Source | Tier | Date accessed |
|---|---|---|---|---|
| shape (61 layers, d_model 7168, MoE 256e top-8, dense FFN 18432, vocab 129280, rope 10000 + YaRN 40/4096, first-3-dense) | — | same sources as deepseek_v3_mla_moe_k0.provenance.md (arXiv:2412.19437; HF deepseek-ai/DeepSeek-V3) | T2 | 2026-07-17 |
| attention full MHA: 128 q heads, 128 KV heads, d_head 128 | — | counterfactual MHA (pre-MLA baseline); head count from V3 MLA spec | T3 (counterfactual) | 2026-07-17 |
| parallelism tp8/pp1/dp8/ep8 | — | synthetic storage default | T3 | 2026-07-17 |
