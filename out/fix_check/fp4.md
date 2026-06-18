# Architecture Justification

**Target**: 7.0B active parameters, 2.0T training tokens, B200 (TP=8 PP=1 DP=8), context=8192.
Serving TBT ≤ 50.0ms, TTFT ≤ Nonems, batch=32.

## Predicted Performance

- **Training throughput**: 77,275 tokens/sec
- **Serving TBT**: 2.2ms (96% under budget)
- **Serving TTFT**: 4.8ms
- **Memory per GPU**: 0.6 GB
- **Predicted loss**: 2.7521 (scaling-law spine: 2.0158, total residual: 36.52%)
- **Confidence**: medium

## Quality Proxy Backbone

The quality proxy is a modular compiler scaling-law backbone: a spine over active non-embedding parameters and training tokens, plus residuals for width/depth, MLP-attention allocation, coupled attention-head variables, precision, MoE, state/memory hooks, risk, and data-quality hooks.
The compiler treats query heads as a weak, saturating architecture prior derived from width, not as a monotonic quality law. KV heads are treated as a direct memory/latency tradeoff with uncertain GQA-sharing quality risk.
- **Model version**: quality_v1_modular_backbone
- **Spine active proxy**: 7.531B active non-embedding params
- **Total uncertainty**: ±14.53%
- **architecture_residual**: 1.52% residual, ±0.40% uncertainty, confidence=medium
- **precision_residual**: 35.00% residual, ±14.21% uncertainty, confidence=low
- **risk_residual**: 0.00% residual, ±0.00% uncertainty, confidence=high

## Design Decisions

### d_model = 6144

Lattice constraint: must be divisible by TP=8 and tile-aligned for BF16 CTA tiles on B200. d_model=6144 lies on the lattice with n_heads=48 × d_head=128 = 6144.
Pareto alternatives: 3072.

### n_layers = 19

Chinchilla-derived aspect ratio target for 7.92B is approximately 19 layers at d_model=6144.

### n_kv_heads = 12 (GQA-4)

GQA-4 reduces KV cache by 4x vs MHA. Coupled GQA-sharing residual: 0.21% with uncertainty from KV-head sharing.
GQA-8 at d_model ≥ 2048 is within seed variance per published ablations.

### Weight precision: FFN=fp4, attention=fp4

Quality model penalty: 35.00% relative PPL (source: component_precision_sensitivity_with_hardware_feasibility).
FFN at FP4 is well-tolerated (~0.1% relative PPL per FP8-LM); throughput gain outweighs quality cost.

### KV cache: 16-bit

Full-precision KV cache — no quantization penalty.

## Search Statistics

- Candidates generated: 387
- Candidates evaluated: 387
- Feasible candidates: 387
- Pareto frontier size: 87
- Search time: 0.1s

## Caveats

- Quality predictions are *relative* within this parameter band (compiler scaling-law spine + modular residuals; not validated against absolute PPL).
- Throughput predictions assume calibrated system efficiency factors (training ~37%, decode ~42% on H100); actual performance may vary ±20%.
- Some penalty values (e.g., FP4) are low-confidence — derived from early literature with sparse per-component data.
- KV quantization penalties calibrated from KIVI on 7B-scale models; transfer to other model families assumed.
- Contains low-confidence residual values.
