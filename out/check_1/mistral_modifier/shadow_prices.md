# Shadow Price Report

Local shadow prices are estimated by directly evaluating nearby architecture perturbations from the baseline.

## Binding Constraints

- Serving regime: decode-heavy — Decode 3028ms (512 tokens × 5.9ms) dominates prefill 34ms
- Training-layer bottleneck proxy: compute
- Baseline meets configured serving and memory constraints.

## Marginal Tradeoffs

| Perturbation | Feasible | Throughput Impact | Quality Proxy Impact | Decision |
|---|---:|---:|---:|---|
| n_layers: 32 -> 30; ffn_precision: bf16 -> fp8 | True | TBT +16.1%, train +98.6% | +1.206% | rejected: outside quality-risk budget |
| n_layers: 32 -> 30; ffn_dim: 14336 -> 13632; ffn_precision: bf16 -> fp8 | True | TBT +16.5%, train +103.5% | +1.336% | rejected: outside quality-risk budget |
| n_layers: 32 -> 31; ffn_precision: bf16 -> fp8 | True | TBT +13.3%, train +92.2% | +1.098% | rejected: outside quality-risk budget |
| n_layers: 32 -> 30; ffn_dim: 14336 -> 12928; ffn_precision: bf16 -> fp8 | True | TBT +16.9%, train +108.7% | +1.475% | rejected: outside quality-risk budget |
| n_layers: 32 -> 31; ffn_dim: 14336 -> 13632; ffn_precision: bf16 -> fp8 | True | TBT +13.7%, train +96.9% | +1.227% | rejected: outside quality-risk budget |
| ffn_precision: bf16 -> fp8 | True | TBT +10.5%, train +86.2% | +0.998% | accepted within risk budget (low) |
| n_layers: 32 -> 31; ffn_dim: 14336 -> 12928; ffn_precision: bf16 -> fp8 | True | TBT +14.1%, train +101.9% | +1.366% | rejected: outside quality-risk budget |
| ffn_dim: 14336 -> 13632; ffn_precision: bf16 -> fp8 | True | TBT +11.0%, train +90.8% | +1.126% | rejected: outside quality-risk budget |

## Interpretation

- Positive throughput deltas mean the variant is faster than the baseline.
- Positive loss-proxy deltas mean expected quality risk increased.
- Accepted/rejected is based on feasibility plus the configured relative loss-proxy budget.