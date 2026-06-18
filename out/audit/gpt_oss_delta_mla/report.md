# Delta Influence — `swap_attention_to_mla`

**Baseline:** gpt-oss-120b  
**Hardware:** h100  
**Delta args:** latent_dim=256

## Summary

Baseline binding stresses: HBM-BW-decode=3.86 violated. Applying MLA drops HBM-BW-decode by 0.01; no binding axis was relieved. Quality cost: +0.0179 attention, -0.0001 shape-law (+0.0178 total residual change).

## Field-level changes

| Field | Baseline | Candidate |
|---|---|---|
| `n_kv_heads` | `8` | `1` |
| `total_params_b` | `8.09` | `7.96` |

## Evaluation metrics

| Metric | Baseline | Candidate | Δ | Δ% | Direction |
|---|---:|---:|---:|---:|---|
| Predicted loss | 1.845 | 1.881 | +0.035 | +1.90% | ↑ worsens |
| Serving TBT (ms) | 9.037 | 9.037 | +0.000 | +0.00% | · neutral |
| Prefill / TTFT (ms) | 42.185 | 42.185 | +0.000 | +0.00% | · neutral |
| Training TPS (tok/s) | 9524.728 | 9531.816 | +7.088 | +0.07% | ↓ improves |
| Memory / GPU (GB) | 19.856 | 19.810 | -0.046 | -0.23% | ↓ improves |
| KV cache (GB) | 0.018 | 0.002 | -0.015 | -87.50% | ↓ improves |
| Total params (B) | 8.090 | 7.960 | -0.130 | -1.61% | ↓ improves |

## Quality residual decomposition

| Quality term | Baseline | Candidate | Δ | Δ% |
|---|---:|---:|---:|---:|
| architecture_residual | 0.01691 | 0.03468 | +0.018 | +105.10% |
| moe_residual | -0.07625 | -0.07625 | +0.000 | +0.00% |

## Stress influence

| Stress axis | Baseline | Candidate | Δ | Baseline band | Candidate band |
|---|---:|---:|---:|---|---|
| HBM bandwidth (decode) | 3.861 | 3.855 | -0.006 | violated | violated |
| HBM bandwidth (prefill) | 0.275 | 0.274 | -0.000 | relaxed | relaxed |
| HBM capacity | 0.248 | 0.248 | -0.000 | relaxed | relaxed |
| KV footprint | 0.000 | 0.000 | +0.000 | relaxed | relaxed |
| Tensor-core util (prefill) | 0.315 | 0.315 | +0.000 | relaxed | relaxed |
| Tensor-core util (decode) | 0.002 | 0.002 | +0.000 | relaxed | relaxed |
| SRAM tile fit | 0.022 | 0.022 | +0.000 | relaxed | relaxed |
| All-reduce traffic | 0.001 | 0.001 | +0.000 | relaxed | relaxed |
| All-to-all traffic | 0.002 | 0.002 | +0.000 | relaxed | relaxed |
| Training memory | 0.419 | 0.416 | -0.003 | relaxed | relaxed |

**Baseline binding axes:** HBM bandwidth (decode)

**Stress relief score:** +0.006

## Pareto position

**Verdict:** On the Pareto frontier
**Signed distance to frontier (normalized):** +0.0159
**Frontier size:** 336 candidates
**Frontier points dominated by this delta:** 0
**Frontier points that dominate this delta:** 0
**Axes that moved:** Predicted loss, Memory / GPU (GB), Training TPS (tok/s)
