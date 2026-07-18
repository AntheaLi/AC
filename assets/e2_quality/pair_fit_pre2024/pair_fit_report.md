# Paired-ablation residual fit (Wave 18h)

Pairs: 4 | within tolerance: 3 | outside: 1

| pair | term | observed Δ% | predicted Δ% | residual | ok |
|---|---|---:|---:|---:|---|
| ainslie2023_gqa8_vs_mha | architecture_residual | +0.10 | +0.11 | -0.01 | ✓ |
| ainslie2023_mqa_vs_mha | architecture_residual | +0.55 | +2.17 | -1.62 | ✗ |
| mistral2023_swa_ppl_hit | attention_locality | +0.55 | +0.53 | +0.02 | ✓ |
| peng2024_yarn_vs_pi | context_utility | -0.30 | -0.16 | -0.14 | ✓ |

## Per-term fit

| term | pairs | scale | bias % | rms % |
|---|---:|---:|---:|---:|
| state_residual | 0 | — | +0.00 | 0.00 |
| architecture_residual | 2 | 0.222 | -0.81 | 1.15 |
| attention_locality | 1 | 1.032 | +0.02 | 0.02 |
| effective_capacity | 0 | — | +0.00 | 0.00 |
| moe_residual | 0 | — | +0.00 | 0.00 |
| mtp_residual | 0 | — | +0.00 | 0.00 |
| context_utility | 1 | 1.847 | -0.14 | 0.14 |
| precision_residual | 0 | — | +0.00 | 0.00 |
| large_shape_stability_prior | 0 | — | +0.00 | 0.00 |
| risk_residual | 0 | — | +0.00 | 0.00 |

## Coverage audit

- **state_residual**: UNCOVERED: no published pair constrains 'state_residual' — its constants are hand priors; any decision it dominates is extrapolation.
- **architecture_residual**: fitted scale 0.22 is far from 1.0 — the term's magnitude disagrees with published evidence; inspect before trusting decisions dominated by 'architecture_residual'.
- **attention_locality**: only 1 pair(s) — scale is a point estimate with no cross-validation; treat as directional.
- **effective_capacity**: UNCOVERED: no published pair constrains 'effective_capacity' — its constants are hand priors; any decision it dominates is extrapolation.
- **moe_residual**: UNCOVERED: no published pair constrains 'moe_residual' — its constants are hand priors; any decision it dominates is extrapolation.
- **mtp_residual**: UNCOVERED: no published pair constrains 'mtp_residual' — its constants are hand priors; any decision it dominates is extrapolation.
- **context_utility**: only 1 pair(s) — scale is a point estimate with no cross-validation; treat as directional.
- **precision_residual**: UNCOVERED: no published pair constrains 'precision_residual' — its constants are hand priors; any decision it dominates is extrapolation.
- **large_shape_stability_prior**: UNCOVERED: no published pair constrains 'large_shape_stability_prior' — its constants are hand priors; any decision it dominates is extrapolation.
- **risk_residual**: UNCOVERED: no published pair constrains 'risk_residual' — its constants are hand priors; any decision it dominates is extrapolation.
- attention_locality: anchored local fractions [1.0], windows [4096]; other regions are priors.

Fitted scales are cross-paper (datamix/tokenizer confounded); use as priors and ingest lab pairs via the same format to sharpen.