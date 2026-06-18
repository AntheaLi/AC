# Modifier Justification

This run used the baseline-aware Pareto modifier ability. The original greenfield compiler remains available when no baseline config is supplied.
Default modifier selection is same-quality: it only selects changes that preserve the learned model topology and numerics.

## Selected Config

- Change: ffn_precision: bf16 -> fp8
- d_model=4096, layers=32, heads=32, d_head=128, kv_heads=8, ffn_dim=14336
- FFN precision=fp8, KV cache=16-bit, TP=8
- Relative loss-proxy delta: +0.998%
- Quality-preserving: False
- Move class: precision
- Risk label: low

## Why This Moved The Baseline

- ffn_precision: bf16 -> fp8: uses lower-precision FFN matmuls for throughput/memory gains with a precision residual.

## Pareto Context

- Local candidates evaluated: 1242
- Feasible candidates: 1242
- Risk-aware Pareto frontier size: 316

## Representative Alternatives

| Change | Risk | Loss Risk | TBT Improvement | Train TPS Improvement | Mem Improvement | Modeled KV Improvement | Reason |
|---|---:|---:|---:|---:|---:|---:|---|
| n_kv_heads: 8 -> 32; ffn_dim: 14336 -> 15040 | low | -0.627% | -8.1% | -12.3% | -31.9% | -300.0% | changes KV bandwidth/capacity pressure |
| n_kv_heads: 8 -> 32 | low | -0.530% | -7.3% | -10.2% | -28.6% | -300.0% | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 31; n_kv_heads: 8 -> 32; ffn_dim: 14336 -> 15040 | medium | -0.529% | -4.7% | -9.5% | -28.1% | -287.5% | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 33; n_kv_heads: 8 -> 16; ffn_dim: 14336 -> 15744 | medium | -0.523% | -7.3% | -9.4% | -19.4% | -106.2% | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 30; n_kv_heads: 8 -> 32; ffn_dim: 14336 -> 15744 | high | -0.516% | -2.1% | -8.7% | -27.4% | -275.0% | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 34; n_kv_heads: 8 -> 16; ffn_dim: 14336 -> 15040 | medium | -0.505% | -9.7% | -9.7% | -19.2% | -112.5% | changes KV bandwidth/capacity pressure |
| n_kv_heads: 8 -> 16; ffn_dim: 14336 -> 15744 | medium | -0.432% | -4.1% | -6.5% | -16.1% | -100.0% | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 31; n_kv_heads: 8 -> 32 | low | -0.432% | -3.9% | -7.3% | -24.9% | -287.5% | changes KV bandwidth/capacity pressure |

## Uncertainty

- Confidence: high
- Total residual: 1.21%
- Uncertainty interval: +0.00% to +3.21% residual range
- Treat this as an architecture-ranking signal, not a final perplexity prediction.