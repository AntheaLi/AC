---
ac_version: 0.4.0
quality_model_version: effective_capacity_v2
git_commit: c170cda
experiment_date: 2026-07-17
agent: gate2-wave1
config: kimi_k2_like_1t_moe.json
---

# Provenance — kimi_k2_like_1t_moe.json

Measured by AC ledger on 2026-07-17: total = 1026.243 B, active = 32.70 B,
n_dense_ffn_layers = 1. Loader warnings (expected): dominant-band collapse
("60 of 61 layers"), MoE active-FFN scoring note.

| Field | Value | Source | Tier | Date accessed |
|---|---|---|---|---|
| n_layers = 61 | 61 | arXiv:2507.20534 Table 2 (K2 vs V3: layers 61→61) | T2 | 2026-07-17 |
| d_model = 7168 | 7168 | arXiv:2507.20534 Table 2 (hidden 7168); HF config hidden_size | T2 | 2026-07-17 |
| dense band = layer [0] | first_k_dense_replace=1 | HF config moonshotai/Kimi-K2-Instruct `first_k_dense_replace: 1`; arXiv:2507.20534 Table 2 (V3 3 → K2 1) | T2 | 2026-07-17 |
| dense FFN dim = 18432 | 18432 | HF config `intermediate_size: 18432`; GLM-4.5 report Table 1 lists K2 dense intermediate 18432 | T2 | 2026-07-17 |
| MoE band = layers 1..60 | — | derived: 61 layers minus 1 dense | T2 | 2026-07-17 |
| n_experts = 384, top_k = 8 | 384 / 8 | arXiv:2507.20534 (384 routed experts, top-8 + 1 shared); HF config `n_routed_experts: 384`, `num_experts_per_tok: 8` | T2 | 2026-07-17 |
| expert_dim = 2048 | 2048 | HF config `moe_intermediate_size: 2048` | T2 | 2026-07-17 |
| shared_expert ffn_dim = 2048 | 2048 | arXiv:2507.20534 (1 shared expert); shared dim = expert dim (HF `n_shared_experts: 1`) | T2 | 2026-07-17 |
| router aux_loss_coef = 0.001 | 0.001 | synthetic default matching mai_thinking_1 template; K2 uses aux-free (noaux_tc) balancing — recorded as approximation | T3 (synthetic) | 2026-07-17 |
| attention type = mla | mla | arXiv:2507.20534 (MLA inherited from V3); HF config `kv_lora_rank: 512`, `q_lora_rank: 1536` | T2 | 2026-07-17 |
| n_heads = 64, n_kv_heads = 64 | 64 | arXiv:2507.20534 (64 attention heads; halved from 128 to cut 128k inference FLOPs by 83%); HF `num_attention_heads: 64` | T2 | 2026-07-17 |
| d_head = 128 | 128 | HF config `v_head_dim: 128` / qk_nope 128 | T2 | 2026-07-17 |
| kv_latent_dim = 512 | 512 | HF config `kv_lora_rank: 512` | T2 | 2026-07-17 |
| q_latent_dim = 1536 | 1536 | HF config `q_lora_rank: 1536` | T2 | 2026-07-17 |
| rope_head_dim = 64, nope_head_dim = 128 | 64 / 128 | HF config `qk_rope_head_dim: 64`, `qk_nope_head_dim: 128` | T2 | 2026-07-17 |
| vocab_size = 163840 | 163840 | HF config `vocab_size: 163840` (K2 README rounds to "160K vocab") | T2 | 2026-07-17 |
| rope base = 50000, YaRN factor 32 from 4096 | — | HF config `rope_theta: 50000`, rope_scaling yarn factor 32, original_max_position_embeddings 4096; max_position 131072 | T2 | 2026-07-17 |
| normalization rmsnorm eps 1e-6 | — | HF config `rms_norm_eps: 1e-6` (sic: 1e-06 in file) | T2 | 2026-07-17 |
| parallelism stored = tp1/pp1/dp1/ep8 | — | neutral storage default; ep=8 chosen because schema requires n_experts % ep == 0 and 384 % 72 != 0 (nearest legal deployment-shaped values 64/96 applied in wave 2) | T3 (synthetic) | 2026-07-17 |
| FP8 weights not modeled | — | HF config shows FP8 (e4m3, 128×128 blocks); AC config precision bf16 — recorded approximation, wave-2 notes | T3 (deviation) | 2026-07-17 |

Context length 128K (K2 GitHub README, moonshotai/Kimi-K2-Instruct, T2,
2026-07-17) is exercised via `--workload long_context` / `--context-length`,
not stored in the config.
