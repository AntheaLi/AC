# Modifier Justification

This run used the baseline-aware Pareto modifier ability. The original greenfield compiler remains available when no baseline config is supplied.
Default modifier selection is same-quality: it only selects changes that preserve the learned model topology and numerics.

## Selected Config

- Change: n_layers: 36 -> 34; kv_cache_bits: 16 -> 8; expert_dim: 2880 -> 3008
- d_model=2880, layers=34, heads=64, d_head=64, kv_heads=8, ffn_dim=2880
- FFN precision=bf16, KV cache=8-bit, TP=8
- Relative loss-proxy delta: -0.041%
- Quality-preserving: False
- Move class: precision
- Risk label: medium

## Why This Moved The Baseline

- n_layers: 36 -> 34: trades depth against latency, memory, and scaling-law shape residual.
- kv_cache_bits: 16 -> 8: reduces KV cache bytes and decode KV bandwidth pressure, with a heuristic quantization residual.
- expert_dim: 2880 -> 3008: adjusts routed-expert capacity while keeping the dense reference FFN unchanged.

## Pareto Context

- Local candidates evaluated: 1933
- Feasible candidates: 1933
- Risk-aware Pareto frontier size: 116
- Performance-dominating variants found: 8

## Representative Alternatives

| Change | Risk | Loss Risk | TBT Improvement | Train TPS Improvement | Mem Improvement | Modeled KV (per request) Improvement | Reason |
|---|---:|---:|---:|---:|---:|---:|---|
| n_layers: 36 -> 34; n_kv_heads: 8 -> 64; expert_dim: 2880 -> 3200 | high | -0.653% | -68.5% | -3.6% | 2.1× larger | 7.6× larger | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 35; n_kv_heads: 8 -> 64; expert_dim: 2880 -> 3200 | medium | -0.649% | -75.9% | -6.7% | 2.2× larger | 7.8× larger | changes KV bandwidth/capacity pressure |
| n_kv_heads: 8 -> 64; expert_dim: 2880 -> 3200 | medium | -0.638% | -83.4% | -9.7% | 2.3× larger | 8× larger | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 34; n_kv_heads: 8 -> 64; kv_cache_bits: 16 -> 8; expert_dim: 2880 -> 3200 | high | -0.555% | -29.7% | -3.6% | -47.8% | 3.8× larger | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 35; n_kv_heads: 8 -> 64; kv_cache_bits: 16 -> 8; expert_dim: 2880 -> 3200 | medium | -0.551% | -34.7% | -6.7% | -53.4% | 3.9× larger | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 34; n_kv_heads: 8 -> 64; expert_dim: 2880 -> 3008 | medium | -0.548% | -66.5% | +1.5% | 2.1× larger | 7.6× larger | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 35; n_kv_heads: 8 -> 64; expert_dim: 2880 -> 3008 | medium | -0.543% | -73.9% | -1.8% | 2.1× larger | 7.8× larger | changes KV bandwidth/capacity pressure |
| n_kv_heads: 8 -> 64; kv_cache_bits: 16 -> 8; expert_dim: 2880 -> 3200 | medium | -0.540% | -39.8% | -9.7% | -59.0% | 4× larger | changes KV bandwidth/capacity pressure |

## Uncertainty

- Confidence: medium
- Total residual: 1.25%
- Residual range: +0.34% to +2.15% (negative values = predicted quality improvement; this is the loss-residual band, not a perplexity CI)
- Treat this as an architecture-ranking signal, not a final perplexity prediction.