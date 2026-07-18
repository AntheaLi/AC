# Shadow Price Report

Local shadow prices are estimated by directly evaluating nearby architecture perturbations from the baseline.

## Binding Constraints

- Serving regime: decode-heavy — Decode 3222ms (512 tokens × 6.3ms) dominates prefill 72ms
- Training-layer bottleneck proxy: compute
- Baseline meets configured serving and memory constraints.

## Marginal Tradeoffs

| Perturbation | Feasible | Throughput Impact | Quality Proxy Impact | Decision |
|---|---:|---:|---:|---|
| n_layers: 32 -> 30; ffn_precision: bf16 -> fp8; kv_cache_bits: 16 -> 8 | True | TBT +32.6%, train +34.3% | +0.398% | accepted within risk budget (medium) |
| n_layers: 32 -> 31; ffn_precision: bf16 -> fp8; kv_cache_bits: 16 -> 8 | True | TBT +30.3%, train +30.0% | +0.337% | accepted within risk budget (medium) |
| ffn_precision: bf16 -> fp8; kv_cache_bits: 16 -> 8 | True | TBT +28.1%, train +25.9% | +0.288% | accepted within risk budget (low) |
| n_layers: 32 -> 30; ffn_dim: 14336 -> 13632; ffn_precision: bf16 -> fp8; kv_cache_bits: 16 -> 8 | True | TBT +32.7%, train +35.1% | +0.461% | accepted within risk budget (high) |
| n_layers: 32 -> 30; ffn_precision: bf16 -> fp8; kv_cache_bits: 16 -> 4 | True | TBT +39.7%, train +34.3% | +1.456% | rejected: outside quality-risk budget |
| n_layers: 32 -> 31; ffn_dim: 14336 -> 13632; ffn_precision: bf16 -> fp8; kv_cache_bits: 16 -> 8 | True | TBT +30.4%, train +30.8% | +0.400% | accepted within risk budget (medium) |
| n_layers: 32 -> 31; ffn_precision: bf16 -> fp8; kv_cache_bits: 16 -> 4 | True | TBT +37.7%, train +30.0% | +1.393% | rejected: outside quality-risk budget |
| ffn_precision: bf16 -> fp8; kv_cache_bits: 16 -> 4 | True | TBT +35.6%, train +25.9% | +1.343% | rejected: outside quality-risk budget |

## Interpretation

- Positive throughput deltas mean the variant is faster than the baseline.
- Positive loss-proxy deltas mean expected quality risk increased.
- Accepted/rejected is based on feasibility plus the configured relative loss-proxy budget.