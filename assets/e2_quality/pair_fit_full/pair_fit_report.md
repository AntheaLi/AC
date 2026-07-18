# Paired-ablation residual fit (Wave 18h)

Pairs: 11 | within tolerance: 6 | outside: 5

| pair | term | observed Δ% | predicted Δ% | residual | ok |
|---|---|---:|---:|---:|---|
| waleffe2024_pure_mamba_vs_hybrid_8b | state_residual | +1.50 | +0.85 | +0.65 | ✓ |
| waleffe2024_hybrid_vs_transformer_8b | state_residual | -0.40 | +0.07 | -0.47 | ✓ |
| jamba2024_ratio_parity_256k | state_residual | +0.00 | -2.72 | +2.72 | ✗ |
| ainslie2023_gqa8_vs_mha | architecture_residual | +0.10 | +0.11 | -0.01 | ✓ |
| ainslie2023_mqa_vs_mha | architecture_residual | +0.55 | +2.17 | -1.62 | ✗ |
| deepseek2024_mla_vs_mha | architecture_residual | -0.10 | +1.31 | -1.41 | ✗ |
| gemma2_2024_local_global_parity | attention_locality | +0.00 | +0.01 | -0.01 | ✓ |
| mistral2023_swa_ppl_hit | attention_locality | +0.55 | +0.53 | +0.02 | ✓ |
| deepseekmoe2024_capacity | effective_capacity | -2.50 | -0.11 | -2.39 | ✗ |
| deepseekv3_2024_mtp | mtp_residual | -0.60 | -0.09 | -0.51 | ✗ |
| peng2024_yarn_vs_pi | context_utility | -0.30 | -0.16 | -0.14 | ✓ |

## Per-term fit

| term | pairs | scale | bias % | rms % |
|---|---:|---:|---:|---:|
| state_residual | 3 | 1.652 | +0.96 | 1.64 |
| architecture_residual | 3 | 0.218 | -1.01 | 1.24 |
| attention_locality | 2 | 1.032 | +0.00 | 0.02 |
| effective_capacity | 1 | 3.833 | -2.39 | 2.39 |
| moe_residual | 0 | — | +0.00 | 0.00 |
| mtp_residual | 1 | — | -0.51 | 0.51 |
| context_utility | 1 | 1.847 | -0.14 | 0.14 |
| precision_residual | 0 | — | +0.00 | 0.00 |
| large_shape_stability_prior | 0 | — | +0.00 | 0.00 |
| risk_residual | 0 | — | +0.00 | 0.00 |

## Coverage audit

- **architecture_residual**: fitted scale 0.22 is far from 1.0 — the term's magnitude disagrees with published evidence; inspect before trusting decisions dominated by 'architecture_residual'.
- **effective_capacity**: fitted scale 3.83 is far from 1.0 — the term's magnitude disagrees with published evidence; inspect before trusting decisions dominated by 'effective_capacity'.
- **effective_capacity**: only 1 pair(s) — scale is a point estimate with no cross-validation; treat as directional.
- **moe_residual**: UNCOVERED: no published pair constrains 'moe_residual' — its constants are hand priors; any decision it dominates is extrapolation.
- **mtp_residual**: 'mtp_residual' pairs exist but the model predicts ~0 delta for all of them — the term is UNIDENTIFIABLE from this corpus (shape error, or the pairs sit where the term is flat).
- **mtp_residual**: only 1 pair(s) — scale is a point estimate with no cross-validation; treat as directional.
- **context_utility**: only 1 pair(s) — scale is a point estimate with no cross-validation; treat as directional.
- **precision_residual**: UNCOVERED: no published pair constrains 'precision_residual' — its constants are hand priors; any decision it dominates is extrapolation.
- **large_shape_stability_prior**: UNCOVERED: no published pair constrains 'large_shape_stability_prior' — its constants are hand priors; any decision it dominates is extrapolation.
- **risk_residual**: UNCOVERED: no published pair constrains 'risk_residual' — its constants are hand priors; any decision it dominates is extrapolation.
- state_residual: anchored p_attn range [0.00, 1.00], contexts [8192, 8192, 262144]. No anchor above ctx 262,144 — the long-context benefit magnitude at 1M+ is extrapolation.
- attention_locality: anchored local fractions [0.5, 1.0], windows [4096, 4096]; other regions are priors.
- effective_capacity: anchored sparsity ratios [5.9]; ratios beyond this range (e.g. >8x) lean on the N_eff functional form, not on paired evidence.

Fitted scales are cross-paper (datamix/tokenizer confounded); use as priors and ingest lab pairs via the same format to sharpen.