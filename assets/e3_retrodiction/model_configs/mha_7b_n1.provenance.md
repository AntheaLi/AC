---
ac_version: 0.4.0
quality_model_version: effective_capacity_v2
git_commit: c170cda
experiment_date: 2026-07-17
agent: gate2-wave1
config: mha_7b_n1.json
---

# Provenance — mha_7b_n1.json

N1 negative control: Mistral-7B shape with pure MHA (32 KV heads). Wave 2
stresses at 32k context vs a GQA-4 comparator. Measured by AC ledger:
total = 8.05 B.

| Field | Value | Source | Tier | Date accessed |
|---|---|---|---|---|
| d_model 4096, 32 layers, head_dim 128, FFN 14336, vocab 32000, context 8192 | — | arXiv:2310.06825 Table 1 (dim 4096, n_layers 32, head_dim 128, hidden_dim 14336, n_heads 32, n_kv_heads 8, window 4096, context 8192, vocab 32000) | T2 | 2026-07-17 |
| n_heads = 32, n_kv_heads = 32 (MHA) | — | counterfactual MHA; actual Mistral uses GQA-8 "for faster inference" (mistral.ai/news/announcing-mistral-7b) | T2 (actual) / T3 (counterfactual) | 2026-07-17 |
| rope base 10000, no scaling | — | arXiv:2310.06825 (RoPE, theta 10000) | T2 | 2026-07-17 |
| parallelism tp1/pp1/dp1 | — | single-GPU N1 default | T3 | 2026-07-17 |
