# Delta Influence — `swap_attention_to_mla+add_state_layers`

**Baseline:** gpt-oss-120b  
**Hardware:** h100  
**Delta args:** sequence=[{'name': 'swap_attention_to_mla', 'args': {'latent_dim': 256}}, {'name': 'add_state_layers', 'args': {'ratio': '1:3'}}]

## Summary

Baseline binding stresses: HBM-BW-decode=3.86 violated. Applying swap_attention_to_mla+add_state_layers relieves HBM-BW-decode 3.86→0.39.

## Field-level changes

| Field | Baseline | Candidate |
|---|---|---|
| `n_kv_heads` | `8` | `1` |
| `total_params_b` | `8.09` | `7.96` |
| `state_enabled` | `False` | `True` |

## Evaluation metrics

| Metric | Baseline | Candidate | Δ | Δ% | Direction |
|---|---:|---:|---:|---:|---|
| Predicted loss | 1.845 | 1.898 | +0.053 | +2.86% | ↑ worsens |
| Serving TBT (ms) | 9.037 | 9.055 | +0.018 | +0.19% | ↑ worsens |
| Prefill / TTFT (ms) | 42.185 | 44.398 | +2.213 | +5.25% | ↑ worsens |
| Training TPS (tok/s) | 9524.728 | 9077.046 | -447.682 | -4.70% | ↑ worsens |
| Memory / GPU (GB) | 19.856 | 2.047 | -17.809 | -89.69% | ↓ improves |
| KV cache (GB) | 0.018 | 0.002 | -0.015 | -87.50% | ↓ improves |
| Total params (B) | 8.090 | 7.960 | -0.130 | -1.61% | ↓ improves |

## Quality residual decomposition

| Quality term | Baseline | Candidate | Δ | Δ% |
|---|---:|---:|---:|---:|
| architecture_residual | 0.01691 | 0.03468 | +0.018 | +105.10% |
| moe_residual | -0.07625 | -0.07625 | +0.000 | +0.00% |
| state_residual | 0.00000 | 0.00903 | +0.009 | +0.00% |

## Stress influence

| Stress axis | Baseline | Candidate | Δ | Baseline band | Candidate band |
|---|---:|---:|---:|---|---|
| HBM bandwidth (decode) | 3.861 | 0.392 | -3.469 | violated | relaxed |
| HBM bandwidth (prefill) | 0.275 | 0.027 | -0.248 | relaxed | relaxed |
| HBM capacity | 0.248 | 0.026 | -0.222 | relaxed | relaxed |
| KV footprint | 0.000 | 0.000 | -0.000 | relaxed | relaxed |
| Tensor-core util (prefill) | 0.315 | 0.224 | -0.092 | relaxed | relaxed |
| Tensor-core util (decode) | 0.002 | 0.002 | -0.001 | relaxed | relaxed |
| SRAM tile fit | 0.022 | 0.022 | +0.000 | relaxed | relaxed |
| All-reduce traffic | 0.001 | 0.001 | -0.000 | relaxed | relaxed |
| All-to-all traffic | 0.002 | 0.002 | -0.000 | relaxed | relaxed |
| Training memory | 0.419 | 0.194 | -0.225 | relaxed | relaxed |

**Baseline binding axes:** HBM bandwidth (decode)

**Relieved by delta:** HBM bandwidth (decode)

**Stress relief score:** +3.469

## Pareto position

**Verdict:** On the Pareto frontier
**Signed distance to frontier (normalized):** +0.2617
**Frontier size:** 336 candidates
**Frontier points dominated by this delta:** 0
**Frontier points that dominate this delta:** 0
**Axes that moved:** Predicted loss, Serving TBT (ms), Memory / GPU (GB), Training TPS (tok/s)
