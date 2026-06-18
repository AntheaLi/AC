# Shadow Price Report

Local shadow prices are estimated by directly evaluating nearby architecture perturbations from the baseline.

## Binding Constraints

- Serving regime: decode-heavy — Decode 4648ms (512 tokens × 9.1ms) dominates prefill 42ms
- Training-layer bottleneck proxy: alltoall
- Baseline meets configured serving and memory constraints.

## Marginal Tradeoffs

| Perturbation | Feasible | Throughput Impact | Quality Proxy Impact | Decision |
|---|---:|---:|---:|---|
| ffn_precision: bf16 -> fp8 | True | TBT +36.4%, train +125.0% | +12.056% | rejected: outside quality-risk budget |
| n_kv_heads: 8 -> 16; ffn_precision: bf16 -> fp8 | True | TBT +35.7%, train +124.6% | +11.852% | rejected: outside quality-risk budget |
| n_layers: 36 -> 35; ffn_precision: bf16 -> fp8 | True | TBT +38.2%, train +131.3% | +12.141% | rejected: outside quality-risk budget |
| n_kv_heads: 8 -> 32; ffn_precision: bf16 -> fp8 | True | TBT +34.2%, train +120.8% | +11.568% | rejected: outside quality-risk budget |
| n_layers: 36 -> 35; n_kv_heads: 8 -> 16; ffn_precision: bf16 -> fp8 | True | TBT +37.4%, train +130.9% | +11.937% | rejected: outside quality-risk budget |
| n_layers: 36 -> 35; n_kv_heads: 8 -> 32; ffn_precision: bf16 -> fp8 | True | TBT +36.0%, train +126.9% | +11.652% | rejected: outside quality-risk budget |
| n_layers: 36 -> 34; ffn_precision: bf16 -> fp8 | True | TBT +39.9%, train +138.0% | +12.232% | rejected: outside quality-risk budget |
| n_layers: 36 -> 34; n_kv_heads: 8 -> 16; ffn_precision: bf16 -> fp8 | True | TBT +39.2%, train +137.5% | +12.027% | rejected: outside quality-risk budget |

## Interpretation

- Positive throughput deltas mean the variant is faster than the baseline.
- Positive loss-proxy deltas mean expected quality risk increased.
- Accepted/rejected is based on feasibility plus the configured relative loss-proxy budget.