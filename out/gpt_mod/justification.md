# Modifier Justification

This run used the baseline-aware Pareto modifier ability. The original greenfield compiler remains available when no baseline config is supplied.
Default modifier selection is same-quality: it only selects changes that preserve the learned model topology and numerics.

## Selected Config

- Change: baseline
- d_model=4096, layers=36, heads=64, d_head=64, kv_heads=8, ffn_dim=11520
- FFN precision=bf16, KV cache=16-bit, TP=8
- Relative loss-proxy delta: +0.000%
- Quality-preserving: True
- Move class: baseline
- Risk label: baseline

## Why This Moved The Baseline

The baseline was retained because no local modification improved resource use inside the configured quality-risk budget.

## Pareto Context

- Local candidates evaluated: 690
- Feasible candidates: 690
- Risk-aware Pareto frontier size: 329

## Representative Alternatives

| Change | Risk | Loss Delta | TBT Delta | Train TPS Delta | Mem Delta | Modeled KV Delta | Reason |
|---|---:|---:|---:|---:|---:|---:|---|
| n_layers: 36 -> 34; n_kv_heads: 8 -> 64; ffn_dim: 11520 -> 12672 | high | +10.004% | +27.1% | +11.0% | +85.8% | -655.6% | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 35; n_kv_heads: 8 -> 64; ffn_dim: 11520 -> 12096 | medium | +10.026% | +25.5% | +10.2% | +85.7% | -677.8% | changes KV bandwidth/capacity pressure |
| n_kv_heads: 8 -> 64 | low | +10.059% | +23.8% | +9.5% | +85.7% | -700.0% | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 34; n_kv_heads: 8 -> 64; ffn_dim: 11520 -> 12096 | medium | +10.113% | +27.6% | +13.4% | +86.1% | -655.6% | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 35; n_kv_heads: 8 -> 64 | low | +10.140% | +25.9% | +12.6% | +86.0% | -677.8% | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 37; n_kv_heads: 8 -> 32; ffn_dim: 11520 -> 12672 | medium | +10.140% | +24.3% | +12.5% | +86.9% | -311.1% | changes KV bandwidth/capacity pressure |
| n_kv_heads: 8 -> 64; ffn_dim: 11520 -> 10944 | low | +10.180% | +24.3% | +12.0% | +86.0% | -700.0% | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 38; n_kv_heads: 8 -> 32; ffn_dim: 11520 -> 12096 | medium | +10.193% | +22.8% | +12.2% | +87.0% | -322.2% | changes KV bandwidth/capacity pressure |

## Uncertainty

- Confidence: medium
- Total residual: -5.93%
- Uncertainty interval: +0.00% to +-2.25% residual range
- Treat this as an architecture-ranking signal, not a final perplexity prediction.