# Architecture Justification

**Target**: 5.1B active parameters, 4.0T training tokens, H100 (TP=8 PP=1 DP=8), context=32768.
Serving TBT ≤ 80.0ms, TTFT ≤ Nonems, batch=16.

## Predicted Performance

- **Training throughput**: 16,190 tokens/sec
- **Serving TBT**: 6.2ms (92% under budget)
- **Serving TTFT**: 24.2ms
- **Memory per GPU**: 11.5 GB
- **Predicted loss**: 1.8845 (scaling-law spine: 2.0225, total residual: -6.82%)
- **Confidence**: medium

## Quality Proxy Backbone

The quality proxy is a modular compiler scaling-law backbone: a spine over active non-embedding parameters and training tokens, plus residuals for width/depth, MLP-attention allocation, coupled attention-head variables, precision, MoE, state/memory hooks, risk, and data-quality hooks.
The compiler treats query heads as a weak, saturating architecture prior derived from width, not as a monotonic quality law. KV heads are treated as a direct memory/latency tradeoff with uncertain GQA-sharing quality risk.
- **Model version**: quality_v1_modular_backbone
- **Spine active proxy**: 4.587B active non-embedding params
- **Total uncertainty**: ±4.32%
- **architecture_residual**: 0.93% residual, ±0.30% uncertainty, confidence=medium
- **precision_residual**: 0.00% residual, ±0.00% uncertainty, confidence=high
- **moe_residual**: -7.75% residual, ±3.10% uncertainty, confidence=low
- **risk_residual**: 0.00% residual, ±0.00% uncertainty, confidence=high

## Design Decisions

### d_model = 4608

Lattice constraint: must be divisible by TP=8 and tile-aligned for BF16 CTA tiles on H100. d_model=4608 lies on the lattice with n_heads=72 × d_head=64 = 4608.

### n_layers = 27

Chinchilla-derived aspect ratio target for 40.43B is approximately 27 layers at d_model=4608.

### n_kv_heads = 72 (MHA)

Multi-head attention (MHA) selected. No KV sharing.

### Weight precision: FFN=bf16, attention=bf16

BF16 baseline — no weight precision penalty.

### KV cache: 16-bit

Full-precision KV cache — no quantization penalty.

## Search Statistics

- Candidates generated: 1,917
- Candidates evaluated: 1,917
- Feasible candidates: 1,917
- Pareto frontier size: 329
- Search time: 17.3s

## Caveats

- Quality predictions are *relative* within this parameter band (compiler scaling-law spine + modular residuals; not validated against absolute PPL).
- Throughput predictions assume calibrated system efficiency factors (training ~37%, decode ~42% on H100); actual performance may vary ±20%.
- KV quantization penalties calibrated from KIVI on 7B-scale models; transfer to other model families assumed.
- Contains low-confidence residual values.
