# Shadow Price Report

Local shadow prices are estimated by directly evaluating nearby architecture perturbations from the baseline.

## Binding Constraints

- Serving regime: decode-heavy — Decode 3187ms (512 tokens × 6.2ms) dominates prefill 55ms
- Training-layer bottleneck proxy: compute
- Baseline meets configured serving and memory constraints.

## Marginal Tradeoffs

| Perturbation | Feasible | Throughput Impact | Quality Proxy Impact | Decision |
|---|---:|---:|---:|---|
| n_layers: 36 -> 34; kv_cache_bits: 16 -> 8 | True | TBT +11.0%, train +6.8% | +0.073% | accepted within risk budget (medium) |
| n_layers: 36 -> 34; kv_cache_bits: 16 -> 8; expert_dim: 2880 -> 2752 | True | TBT +12.3%, train +6.8% | +0.197% | accepted within risk budget (medium) |
| n_layers: 36 -> 35; kv_cache_bits: 16 -> 8 | True | TBT +8.2%, train +3.3% | +0.082% | accepted within risk budget (medium) |
| n_layers: 36 -> 35; kv_cache_bits: 16 -> 8; expert_dim: 2880 -> 2560 | True | TBT +11.7%, train +9.4% | +0.416% | accepted within risk budget (medium) |
| n_layers: 36 -> 35; kv_cache_bits: 16 -> 8; expert_dim: 2880 -> 2752 | True | TBT +9.6%, train +3.3% | +0.207% | accepted within risk budget (medium) |
| kv_cache_bits: 16 -> 8 | True | TBT +5.4%, train +0.0% | +0.099% | accepted within risk budget (low) |
| n_layers: 36 -> 34; kv_cache_bits: 16 -> 8; expert_dim: 2880 -> 3008 | True | TBT +9.7%, train +6.8% | -0.041% | accepted: improves decode latency with no loss-proxy increase |
| kv_cache_bits: 16 -> 8; expert_dim: 2880 -> 2560 | True | TBT +9.0%, train +5.9% | +0.434% | accepted within risk budget (medium) |

## Interpretation

- Positive throughput deltas mean the variant is faster than the baseline.
- Positive loss-proxy deltas mean expected quality risk increased.
- Accepted/rejected is based on feasibility plus the configured relative loss-proxy budget.