# Architecture Justification

**Target**: 35.0B active parameters, 8.0T training tokens, B200 (TP=8 PP=4 DP=4), context=131072.
Serving TBT ≤ 60.0ms, TTFT ≤ Nonems, batch=8.

## Predicted Performance

- **Training throughput**: 19,099 tokens/sec
- **Serving TBT**: 4.2ms (93% under budget)
- **Serving TTFT**: 14.4ms
- **Memory per GPU**: 25.7 GB
- **Predicted loss**: 1.7526 (scaling-law spine: 1.8955, total residual: -7.54%)
- **Confidence**: medium

## Quality Proxy Backbone

The quality proxy is a modular compiler scaling-law backbone: a spine over active non-embedding parameters and training tokens, plus residuals for width/depth, MLP-attention allocation, coupled attention-head variables, precision, MoE, state/memory hooks, risk, and data-quality hooks.
The compiler treats query heads as a weak, saturating architecture prior derived from width, not as a monotonic quality law. KV heads are treated as a direct memory/latency tradeoff with uncertain GQA-sharing quality risk.
- **Model version**: quality_v1_modular_backbone
- **Spine active proxy**: 35.334B active non-embedding params
- **Total uncertainty**: ±8.58%
- **architecture_residual**: 0.21% residual, ±0.05% uncertainty, confidence=medium
- **precision_residual**: 0.00% residual, ±0.00% uncertainty, confidence=high
- **moe_residual**: -7.75% residual, ±3.10% uncertainty, confidence=low
- **risk_residual**: 0.00% residual, ±0.00% uncertainty, confidence=high

## Design Decisions

### d_model = 9216

Lattice constraint: must be divisible by TP=8 and tile-aligned for BF16 CTA tiles on B200. d_model=9216 lies on the lattice with n_heads=72 × d_head=128 = 9216.

### n_layers = 52

Chinchilla-derived aspect ratio target for 309.75B is approximately 52 layers at d_model=9216.
Divisible by PP=4 for pipeline parallelism.

### n_kv_heads = 72 (MHA)

Multi-head attention (MHA) selected. No KV sharing.

### Weight precision: FFN=bf16, attention=bf16

BF16 baseline — no weight precision penalty.

### KV cache: 16-bit

Full-precision KV cache — no quantization penalty.

## Search Statistics

- Candidates generated: 8,163
- Candidates evaluated: 8,163
- Feasible candidates: 8,163
- Pareto frontier size: 949
- Search time: 53.3s

## Caveats

- Quality predictions are *relative* within this parameter band (compiler scaling-law spine + modular residuals; not validated against absolute PPL).
- Throughput predictions assume calibrated system efficiency factors (training ~37%, decode ~42% on H100); actual performance may vary ±20%.
- KV quantization penalties calibrated from KIVI on 7B-scale models; transfer to other model families assumed.
- N=35.3B is above Chinchilla's calibration range (70M-16B); extrapolation
- D=8.0T tokens is above Chinchilla's calibration range; extrapolation
- Contains low-confidence residual values.
