---
ac_version: 0.4.0
quality_model_version: effective_capacity_v2
git_commit: c170cda
experiment_date: 2026-07-17
agent: gate2-wave1
config: dmodel_misaligned_7b_n2.json
---

# Provenance — dmodel_misaligned_7b_n2.json

N2 negative control: Mistral-7B with d_model = 4104 (divisible by 8 only —
breaks the 256-element tensor-core lattice). Aligned twin is the frozen
`configs/mistral_7b.json` (read-only). Measured by AC ledger: total = 7.26 B.

| Field | Value | Source | Tier | Date accessed |
|---|---|---|---|---|
| base shape (32 layers, FFN 14336, vocab 32000, GQA-8, d_head 128) | — | arXiv:2310.06825 Table 1, same as configs/mistral_7b.json | T2 | 2026-07-17 |
| d_model = 4104 | — | synthetic: 4096 + 8; gcd with 256-lattice is 8; attention width 32×128 = 4096 stays within schema's [0.25, 4]× d_model bound | T3 (synthetic) | 2026-07-17 |
| parallelism tp1/pp1/dp1 | — | single-GPU default | T3 | 2026-07-17 |
