# Delta Influence — `swap_attention_to_gqa`

**Baseline:** mistral-7b  
**Hardware:** gb200_nvl72  
**Delta args:** group_size=8

**Resolved workload** (all predictions below use these settings):

- preset: `long_context`  
- serving_batch: 8  
- prompt_len: 32768  context_length: 32768  
- TP / PP / DP: 8 / 1 / 8  
- TBT budget: 80.0 ms

## Summary

Baseline has no binding stresses; transformation is exploratory. Applying GQA lowers HBM-BW-decode by 0.00; no binding axis was relieved. Quality cost: +0.0014 attention, +0.0000 shape-law (+0.0015 total residual change). NOTE: the residual figures above exclude the capacity effect — this delta moves scaling-spine active params by -1.9%, and the full predicted-loss move is +0.0034 (+0.0012 of it scaling-spine); see the ACTIVE-PARAM SHIFT note under Topology notes.

## Field-level changes

| Field | Baseline | Candidate |
|---|---|---|
| `n_kv_heads` | `8` | `4` |
| `total_params_b` | `7.241728` | `7.10751` |
| `applied_deltas` | `None` | `['swap_attention_to_gqa']` |

## Evaluation metrics

| Metric | Baseline | Candidate | Δ | Δ% | Direction |
|---|---:|---:|---:|---:|---|
| Predicted loss | 2.033 | 2.036 | +0.003 | +0.17% | worsens |
| Scaling-law loss (predicted − residual terms) | 1.950 | 1.952 | +0.001 | +0.06% | worsens |
| Serving TBT (ms) | 3.397 | 3.397 | +0.000 | +0.00% | neutral |
| Prefill / TTFT (ms) | 174.346 | 174.346 | +0.000 | +0.00% | neutral |
| Training TPS (tok/s) | 104723.041 | 104725.375 | +2.334 | +0.00% | improves |
| Training memory / GPU (GB) | 2.290 | 2.258 | -0.031 | -1.36% | improves |
| Memory / GPU (GB) | 8.100 | 8.069 | -0.031 | -0.39% | improves |
| KV cache / GPU, per request (GB) | 0.500 | 0.500 | +0.000 | +0.00% | neutral |
| Total params (B) | 7.242 | 7.108 | -0.134 | -1.85% | improves |
| Active non-embedding params used by scaling spine (B) | 6.980 | 6.845 | -0.134 | -1.92% | improves |

_KV-cache figures above are per concurrent request; multiply by the resolved workload's `serving_batch` for the steady-state batch total._

## Topology notes

- ACTIVE-PARAM SHIFT: this delta changes the scaling-spine active params 6.98B → 6.85B (-1.9%). Attribution: of the +0.0034 predicted-loss move, +0.0012 is the scaling-law baseline responding to the spine-param change and ~+0.0022 is residual-term/interaction effects. The scaling-law share is a capacity effect (more parameters active per token), not an architecture-quality effect — do not read it as 'this mixer is better'. Size the new mixer dims to hold active params constant to isolate the architecture question.
- KV heads changed, but modeled per-GPU KV cache stayed flat because the current TP/KV placement assumes at least one KV head resident per rank. This is expected when TP is greater than or equal to the candidate KV-head count; use a different KV-sharding policy or lower TP to realize per-rank KV-cache savings.
- Decode TBT is neutral for the same reason: local decode bandwidth still reads one KV head per rank.

## Quality residual decomposition

| Quality term | Baseline | Candidate | Δ | Δ% |
|---|---:|---:|---:|---:|
| architecture_residual | 0.00000 | 0.00146 | +0.001 | (no baseline, +0.00145) |
| vocab_residual | 0.04235 | 0.04199 | -0.000 | -0.84% |

## Stress influence

| Stress axis | Baseline | Candidate | Δ | Baseline band | Candidate band |
|---|---:|---:|---:|---|---|
| HBM bandwidth (decode) | 0.687 | 0.683 | -0.004 | relaxed | relaxed |
| HBM bandwidth (prefill) | 0.000 | 0.000 | -0.000 | relaxed | relaxed |
| HBM capacity | 0.042 | 0.042 | -0.000 | relaxed | relaxed |
| KV footprint | 0.021 | 0.021 | +0.000 | relaxed | relaxed |
| Tensor-core util (prefill) | 0.744 | 0.744 | +0.000 | loaded | loaded |
| Tensor-core util (decode) | 0.025 | 0.025 | +0.000 | relaxed | relaxed |
| All-reduce traffic | 0.007 | 0.007 | +0.000 | relaxed | relaxed |
| Training memory | 0.059 | 0.059 | -0.000 | relaxed | relaxed |

**Stress relief score:** +0.000

## Pareto position

**Verdict:** _Not classified — the local Pareto frontier was empty._

The classifier had no other candidates to compare against, so no dominance verdict could be reached. This usually means the local neighborhood around the baseline produced no feasible alternatives under the current TP/PP/CP and constraint settings; try `--no-pareto` to skip the classification, or widen the modifier sweep upstream to populate the comparison frontier.

**Axes that moved:** Predicted loss, Memory / GPU (GB), Training TPS (tok/s)
