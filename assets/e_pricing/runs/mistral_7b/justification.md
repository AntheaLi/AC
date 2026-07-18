# Modifier Justification

This run used the baseline-aware Pareto modifier ability. The original greenfield compiler remains available when no baseline config is supplied.
Default modifier selection is same-quality: it only selects changes that preserve the learned model topology and numerics.

## Selected Config

- Change: n_layers: 32 -> 31; ffn_dim: 14336 -> 15040
- d_model=4096, layers=31, heads=32, d_head=128, kv_heads=8, ffn_dim=15040
- FFN precision=bf16, KV cache=16-bit, TP=8
- Relative loss-proxy delta: -0.004%
- Quality-preserving: False
- Move class: architecture
- Risk label: medium

## Why This Moved The Baseline

- n_layers: 32 -> 31: trades depth against latency, memory, and scaling-law shape residual.
- ffn_dim: 14336 -> 15040: adjusts FFN capacity while keeping the dimension tile-friendly.

## Pareto Context

- Local candidates evaluated: 1681
- Feasible candidates: 1681
- Risk-aware Pareto frontier size: 141
- Performance-dominating variants found: 1

## Representative Alternatives

| Change | Risk | Loss Risk | TBT Improvement | Train TPS Improvement | Mem Improvement | Modeled KV (per request) Improvement | Reason |
|---|---:|---:|---:|---:|---:|---:|---|
| n_layers: 32 -> 33; n_kv_heads: 8 -> 16; ffn_dim: 14336 -> 15744 | medium | -0.171% | -37.5% | -8.8% | -55.8% | 2.1× larger | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 33; n_kv_heads: 8 -> 32 | low | -0.169% | -99.7% | -8.4% | 2.6× larger | 4.1× larger | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 34; ffn_dim: 14336 -> 15744 | medium | -0.165% | -8.5% | -9.8% | -6.6% | -6.2% | trades depth quality proxy against latency |
| n_kv_heads: 8 -> 32; ffn_dim: 14336 -> 15040 | low | -0.160% | -94.7% | -8.0% | 2.5× larger | 4× larger | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 33; ffn_dim: 14336 -> 15744 | medium | -0.136% | -5.3% | -7.1% | -4.1% | -3.1% | trades depth quality proxy against latency |
| n_layers: 32 -> 33; n_kv_heads: 8 -> 16; ffn_dim: 14336 -> 15040 | medium | -0.135% | -36.4% | -7.5% | -55.0% | 2.1× larger | changes KV bandwidth/capacity pressure |
| n_kv_heads: 8 -> 16; ffn_dim: 14336 -> 15744 | medium | -0.132% | -33.3% | -5.9% | -51.7% | -100.0% | changes KV bandwidth/capacity pressure |
| n_kv_heads: 8 -> 32 | low | -0.130% | -93.6% | -5.5% | 2.5× larger | 4× larger | changes KV bandwidth/capacity pressure |

## Uncertainty

- Confidence: medium
- Total residual: 4.25%
- Residual range: +1.71% to +6.80% (negative values = predicted quality improvement; this is the loss-residual band, not a perplexity CI)
- Treat this as an architecture-ranking signal, not a final perplexity prediction.