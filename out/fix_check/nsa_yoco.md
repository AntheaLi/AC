# Architecture Justification

**Target**: 1.0B active parameters, 0.2T training tokens, H100 (TP=1 PP=1 DP=1), context=2048.
Serving TBT ≤ 50.0ms, TTFT ≤ Nonems, batch=4.

## Predicted Performance

- **Training throughput**: 12,740 tokens/sec
- **Serving TBT**: 4.0ms (92% under budget)
- **Serving TTFT**: 30.8ms
- **Memory per GPU**: 2.4 GB
- **Predicted loss**: 2.3293 (scaling-law spine: 2.3262, total residual: 0.13%)
- **Confidence**: high

## Quality Proxy Backbone

The quality proxy is a modular compiler scaling-law backbone: a spine over active non-embedding parameters and training tokens, plus residuals for width/depth, MLP-attention allocation, coupled attention-head variables, precision, MoE, state/memory hooks, risk, and data-quality hooks.
The compiler treats query heads as a weak, saturating architecture prior derived from width, not as a monotonic quality law. KV heads are treated as a direct memory/latency tradeoff with uncertain GQA-sharing quality risk.
- **Model version**: quality_v1_modular_backbone
- **Spine active proxy**: 0.993B active non-embedding params
- **Total uncertainty**: ±3.00%
- **architecture_residual**: 0.13% residual, ±0.04% uncertainty, confidence=medium
- **precision_residual**: 0.00% residual, ±0.00% uncertainty, confidence=high
- **risk_residual**: 0.00% residual, ±0.00% uncertainty, confidence=high

## Design Decisions

### d_model = 2304

Lattice constraint: must be divisible by TP=1 and tile-aligned for BF16 CTA tiles on H100. d_model=2304 lies on the lattice with n_heads=18 × d_head=128 = 2304.
Pareto alternatives: 2048, 2176, 2432.

### n_layers = 17

Chinchilla-derived aspect ratio target for 1.14B is approximately 17 layers at d_model=2304.

### n_kv_heads = 9 (GQA-2)

GQA-2 reduces KV cache by 2x vs MHA. Coupled GQA-sharing residual: 0.10% with uncertainty from KV-head sharing.
GQA-8 at d_model ≥ 2048 is within seed variance per published ablations.

### Weight precision: FFN=bf16, attention=bf16

BF16 baseline — no weight precision penalty.

### KV cache: 16-bit

Full-precision KV cache — no quantization penalty.

## Search Statistics

- Candidates generated: 8,820
- Candidates evaluated: 8,820
- Feasible candidates: 8,820
- Pareto frontier size: 274
- Search time: 11.3s

## Caveats

- Quality predictions are *relative* within this parameter band (compiler scaling-law spine + modular residuals; not validated against absolute PPL).
- Throughput predictions assume calibrated system efficiency factors (training ~37%, decode ~42% on H100); actual performance may vary ±20%.
- KV quantization penalties calibrated from KIVI on 7B-scale models; transfer to other model families assumed.

## Post-search Stamps
- NSA was applied as a post-search emission stamp. The optimizer selected the lattice point before this sparse-attention block was added.
- YOCO was applied as a post-search emission stamp. The optimizer selected the lattice point before this KV-sharing block was added.
