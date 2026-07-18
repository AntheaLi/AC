---
ac_version: 0.4.0
quality_model_version: effective_capacity_v2
git_commit: c170cda
experiment_date: 2026-07-17
agent: gate2-wave1
config: deepseek_v3_mla_moe_k0.json
---

# Provenance — deepseek_v3_mla_moe_k0.json

Counterfactual all-MoE V3 (first_k_dense_replace=0) used as the case-4 base;
wave 2 applies `densify_first_k k=3` (actual V3) and `k=1` (K2 choice).
Measured by AC ledger: total = 703.686 B, active = 37.45 B (paper states
671 B total / 37 B active; gap is MTP module + ledger conventions, noted).

| Field | Value | Source | Tier | Date accessed |
|---|---|---|---|---|
| n_layers = 61 | 61 | arXiv:2412.19437v2 (DeepSeek-V3 report: 61 layers) | T2 | 2026-07-17 |
| d_model = 7168 | 7168 | arXiv:2412.19437 (hidden 7168); HF config deepseek-ai/DeepSeek-V3 | T2 | 2026-07-17 |
| all 61 layers MoE (k=0) | — | counterfactual construction; actual V3: "We substitute all FFNs except for the first three layers with MoE layers" (arXiv:2412.19437) | T2 (base fact) / T3 (counterfactual) | 2026-07-17 |
| n_experts = 256 routed + 1 shared, top_k = 8 | — | arXiv:2412.19437 (256 routed experts, top-8, 1 shared; ≤4 nodes per token) | T2 | 2026-07-17 |
| expert_dim = 2048, shared ffn_dim = 2048 | 2048 | HF config `moe_intermediate_size: 2048` | T2 | 2026-07-17 |
| attention mla 128 heads | 128 | arXiv:2412.19437 (MLA, n_h=128, d_h=128, d_c=512, d_c'=1536, d_rope=64) | T2 | 2026-07-17 |
| kv_latent 512 / q_latent 1536 / rope_head 64 / nope_head 128 | — | arXiv:2412.19437; HF config `kv_lora_rank/q_lora_rank/qk_rope_head_dim/qk_nope_head_dim` | T2 | 2026-07-17 |
| vocab_size = 129280 | 129280 | HF config deepseek-ai/DeepSeek-V3 `vocab_size: 129280` | T2 | 2026-07-17 |
| rope base 10000, YaRN factor 40 from 4096 | — | HF config `rope_theta: 10000`, yarn factor 40, original 4096, max_position 163840 | T2 | 2026-07-17 |
| router aux_loss_coef 0.001 | — | synthetic template default (V3 uses aux-loss-free balancing) | T3 (synthetic) | 2026-07-17 |
| parallelism tp8/pp1/dp8/ep8 | — | inference-shaped storage default consistent with V3 §3.4 prefill unit (TP4+EP32 there); storage choice is synthetic, wave-2 may override | T3 (synthetic) | 2026-07-17 |
