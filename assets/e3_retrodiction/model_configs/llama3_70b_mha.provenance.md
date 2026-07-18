---
ac_version: 0.4.0
quality_model_version: effective_capacity_v2
git_commit: c170cda
experiment_date: 2026-07-17
agent: gate2-wave1
config: llama3_70b_mha.json
---

# Provenance — llama3_70b_mha.json

Case-2 counterfactual: Llama-3-70B shape with MHA (64 KV heads). Wave 2 applies
`swap_attention_to_gqa:group_size` ∈ {1, 8, 64}; group_size=8 reproduces the
actual GQA-8 choice. Measured by AC ledger: total = 79.95 B (MHA variant is
larger than the GQA 70.6B; expected — KV projection width differs).

| Field | Value | Source | Tier | Date accessed |
|---|---|---|---|---|
| n_layers = 80, d_model = 8192, FFN 28672 | — | arXiv:2407.21783 Table 3 (70B: 80 layers, d_model 8192, FFN 28672) | T2 | 2026-07-17 |
| n_heads = 64 | 64 | arXiv:2407.21783 Table 3 (64 attention heads) | T2 | 2026-07-17 |
| n_kv_heads = 64 (MHA) | — | counterfactual; actual: "We use grouped query attention (GQA) with 8 key-value heads to improve inference speed and to reduce the size of key-value caches during decoding" (arXiv:2407.21783 §3.2) | T2 (actual=8) / T3 (counterfactual) | 2026-07-17 |
| d_head = 128 | 128 | derived: 8192 / 64 heads | T2 | 2026-07-17 |
| vocab_size = 128256 | 128256 | arXiv:2407.21783 (128K vocab); HF config meta-llama/Meta-Llama-3-70B | T2 | 2026-07-17 |
| rope base = 500000, no scaling | — | arXiv:2407.21783 Table 3 (RoPE θ=500000); 8K pretraining context | T2 | 2026-07-17 |
| rmsnorm eps 1e-5 | — | HF config meta-llama/Meta-Llama-3-70B `rms_norm_eps: 1e-5` | T2 | 2026-07-17 |
| parallelism tp8/pp1/dp1 | — | synthetic inference default | T3 | 2026-07-17 |
