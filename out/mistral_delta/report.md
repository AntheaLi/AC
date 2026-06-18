# Delta Influence — `swap_attention_to_gqa`

**Baseline:** mistral-7b  
**Hardware:** h100  
**Delta args:** group_size=8

## Summary

Baseline binding stresses: HBM-BW-decode=1.00 pressured, training-memory=1.17 binding. Applying GQA drops HBM-BW-decode by 0.01; no binding axis was relieved. Quality cost: +0.0025 attention, +0.0000 shape-law (+0.0025 total residual change).

## Field-level changes

| Field | Baseline | Candidate |
|---|---|---|
| `n_kv_heads` | `8` | `4` |
| `total_params_b` | `7.24` | `7.11` |

## Evaluation metrics

| Metric | Baseline | Candidate | Δ | Δ% | Direction |
|---|---:|---:|---:|---:|---|
| Predicted loss | 2.025 | 2.031 | +0.006 | +0.31% | ↑ worsens |
| Serving TBT (ms) | 6.200 | 6.200 | +0.000 | +0.00% | · neutral |
| Prefill / TTFT (ms) | 33.641 | 33.641 | +0.000 | +0.00% | · neutral |
| Training TPS (tok/s) | 11877.895 | 11889.095 | +11.200 | +0.09% | ↓ improves |
| Memory / GPU (GB) | 2.342 | 2.061 | -0.281 | -12.01% | ↓ improves |
| KV cache (GB) | 0.500 | 0.250 | -0.250 | -50.00% | ↓ improves |
| Total params (B) | 7.240 | 7.110 | -0.130 | -1.80% | ↓ improves |

## Quality residual decomposition

| Quality term | Baseline | Candidate | Δ | Δ% |
|---|---:|---:|---:|---:|
| architecture_residual | 0.00208 | 0.00457 | +0.002 | +119.51% |

## Stress influence

| Stress axis | Baseline | Candidate | Δ | Baseline band | Candidate band |
|---|---:|---:|---:|---|---|
| HBM bandwidth (decode) | 0.997 | 0.992 | -0.005 | pressured | pressured |
| HBM bandwidth (prefill) | 0.002 | 0.002 | -0.000 | relaxed | relaxed |
| HBM capacity | 0.321 | 0.321 | -0.000 | relaxed | relaxed |
| KV footprint | 0.050 | 0.050 | +0.000 | relaxed | relaxed |
| Tensor-core util (prefill) | 0.686 | 0.686 | +0.000 | relaxed | relaxed |
| Tensor-core util (decode) | 0.034 | 0.034 | +0.000 | relaxed | relaxed |
| All-reduce traffic | 0.009 | 0.009 | +0.000 | relaxed | relaxed |
| Training memory | 1.169 | 1.165 | -0.003 | binding | binding |

**Baseline binding axes:** HBM bandwidth (decode), Training memory

**Stress relief score:** +0.009

## Pareto position

**Verdict:** Expands the Pareto frontier
**Signed distance to frontier (normalized):** +0.0095
**Frontier size:** 286 candidates
**Frontier points dominated by this delta:** 4
**Frontier points that dominate this delta:** 0
**Axes that moved:** Predicted loss, Memory / GPU (GB), Training TPS (tok/s)
