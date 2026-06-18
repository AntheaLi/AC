# Architecture Justification

**Target**: 1.0B active parameters, 0.2T training tokens, H100 (TP=1 PP=1 DP=1), context=2048.
Serving TBT ≤ 100.0ms, TTFT ≤ Nonems, batch=4.

## Predicted Performance

- **Training throughput**: 13,112 tokens/sec
- **Serving TBT**: 4.4ms (96% under budget)
- **Serving TTFT**: 29.6ms
- **Memory per GPU**: 2.2 GB
- **Predicted loss**: 2.3624 (scaling-law spine: 2.3449, total residual: 0.75%)
- **Confidence**: high

## Quality Proxy Backbone

The quality proxy is a modular compiler scaling-law backbone: a spine over active non-embedding parameters and training tokens, plus residuals for width/depth, MLP-attention allocation, coupled attention-head variables, precision, MoE, state/memory hooks, risk, and data-quality hooks.
The compiler treats query heads as a weak, saturating architecture prior derived from width, not as a monotonic quality law. KV heads are treated as a direct memory/latency tradeoff with uncertain GQA-sharing quality risk.
- **Model version**: quality_v1_modular_backbone
- **Spine active proxy**: 0.853B active non-embedding params
- **Total uncertainty**: ±3.01%
- **architecture_residual**: 0.75% residual, ±0.26% uncertainty, confidence=medium
- **precision_residual**: 0.00% residual, ±0.00% uncertainty, confidence=high
- **risk_residual**: 0.00% residual, ±0.00% uncertainty, confidence=high

## Design Decisions

### d_model = 2048

Lattice constraint: must be divisible by TP=1 and tile-aligned for BF16 CTA tiles on H100. d_model=2048 lies on the lattice with n_heads=32 × d_head=64 = 2048.
Pareto alternatives: 2176, 2304, 2496.

### n_layers = 17

Chinchilla-derived aspect ratio target for 0.98B is approximately 17 layers at d_model=2048.

### n_kv_heads = 32 (MHA)

Multi-head attention (MHA) selected. No KV sharing.

### Weight precision: FFN=bf16, attention=bf16

BF16 baseline — no weight precision penalty.

### KV cache: 16-bit

Full-precision KV cache — no quantization penalty.

## Search Statistics

- Candidates generated: 20
- Candidates evaluated: 20
- Feasible candidates: 20
- Pareto frontier size: 12
- Search time: 0.1s

## Caveats

- Quality predictions are *relative* within this parameter band (compiler scaling-law spine + modular residuals; not validated against absolute PPL).
- Throughput predictions assume calibrated system efficiency factors (training ~37%, decode ~42% on H100); actual performance may vary ±20%.
- KV quantization penalties calibrated from KIVI on 7B-scale models; transfer to other model families assumed.
