# Architecture Justification

**Target**: 1.0B active parameters, 0.2T training tokens, B200 (TP=1 PP=1 DP=1), context=2048.
Serving TBT ≤ 100.0ms, TTFT ≤ Nonems, batch=4.

## Predicted Performance

- **Training throughput**: 86,661 tokens/sec
- **Serving TBT**: 2.1ms (98% under budget)
- **Serving TTFT**: 4.1ms
- **Memory per GPU**: 0.8 GB
- **Predicted loss**: 3.1557 (scaling-law spine: 2.3367, total residual: 35.05%)
- **Confidence**: medium

## Quality Proxy Backbone

The quality proxy is a modular compiler scaling-law backbone: a spine over active non-embedding parameters and training tokens, plus residuals for width/depth, MLP-attention allocation, coupled attention-head variables, precision, MoE, state/memory hooks, risk, and data-quality hooks.
The compiler treats query heads as a weak, saturating architecture prior derived from width, not as a monotonic quality law. KV heads are treated as a direct memory/latency tradeoff with uncertain GQA-sharing quality risk.
- **Model version**: quality_v1_modular_backbone
- **Spine active proxy**: 0.911B active non-embedding params
- **Total uncertainty**: ±14.53%
- **architecture_residual**: 0.05% residual, ±0.01% uncertainty, confidence=medium
- **precision_residual**: 35.00% residual, ±14.21% uncertainty, confidence=low
- **risk_residual**: 0.00% residual, ±0.00% uncertainty, confidence=high

## Design Decisions

### d_model = 2048

Lattice constraint: must be divisible by TP=1 and tile-aligned for BF16 CTA tiles on B200. d_model=2048 lies on the lattice with n_heads=16 × d_head=128 = 2048.
Pareto alternatives: 1920, 2304, 2560.

### n_layers = 18

Chinchilla-derived aspect ratio target for 1.04B is approximately 18 layers at d_model=2048.

### n_kv_heads = 16 (MHA)

Multi-head attention (MHA) selected. No KV sharing.

### Weight precision: FFN=fp4, attention=fp4

Quality model penalty: 35.00% relative PPL (source: component_precision_sensitivity_with_hardware_feasibility).
FFN at FP4 is well-tolerated (~0.1% relative PPL per FP8-LM); throughput gain outweighs quality cost.

### KV cache: 16-bit

Full-precision KV cache — no quantization penalty.

## Search Statistics

- Candidates generated: 30
- Candidates evaluated: 30
- Feasible candidates: 30
- Pareto frontier size: 15
- Search time: 0.0s

## Caveats

- Quality predictions are *relative* within this parameter band (compiler scaling-law spine + modular residuals; not validated against absolute PPL).
- Throughput predictions assume calibrated system efficiency factors (training ~37%, decode ~42% on H100); actual performance may vary ±20%.
- Some penalty values (e.g., FP4) are low-confidence — derived from early literature with sparse per-component data.
- KV quantization penalties calibrated from KIVI on 7B-scale models; transfer to other model families assumed.
- Contains low-confidence residual values.
