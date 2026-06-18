# Baseline Delta Report

**Baseline**: gpt-oss-120b
**Hardware**: H100
**Selection mode**: quality-spending allowed up to +2.00% relative loss proxy

## Baseline Status

- Baseline loss proxy: 1.8455 (+0.000% vs baseline)
- Baseline training throughput: 9,525 tok/s (+0.0%)
- Baseline TBT: 9.08 ms (+0.0% faster)
- Baseline TTFT: 42.19 ms (+0.0% faster)
- Baseline memory/GPU: 19.91 GB (+0.0% lower)
- Baseline modeled KV cache/GPU: 0.070 GB (+0.0% lower)

- Risk-aware Pareto status: baseline remains on the frontier because lower-risk/no-change is itself an objective.
- Performance-only status: no local variant strictly dominates the baseline across quality, latency, throughput, memory, and KV footprint.

## Selected Modification

No local modification beat the baseline inside the configured quality-risk budget. The baseline config is retained.

## Same-Quality Hardware-Fit Modifications

No non-baseline same-quality modeled deployment variant is on the current Pareto frontier.

Reserved same-quality hardware-fit hooks for the next modifier layer:
- Tensor/pipeline/data parallel placement search without changing weights or layer shapes.
- GQA-aware head sharding and KV-group placement across TP ranks.
- Paged KV cache block size, allocator locality, and scheduler residency policy.
- Static shape buckets / CUDA graph capture for decode and prefill.
- Fused BF16 kernels, tensor-core weight swizzles, and sequence-parallel activation layout.
- Chunked prefill scheduling for long-context prompts.

## Optional Quality-Spending Modifications

These candidates change architecture or numerics and should be treated as retraining/calibration options, not same-quality hardware-fit edits.

| Change | Risk | Loss Delta | TBT Delta | Train TPS Delta | Mem Delta | Modeled KV Delta | Reason |
|---|---:|---:|---:|---:|---:|---:|---|
| n_layers: 36 -> 34; n_kv_heads: 8 -> 64; ffn_dim: 11520 -> 12672 | high | +10.004% | +27.1% | +11.0% | +85.8% | -655.6% | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 35; n_kv_heads: 8 -> 64; ffn_dim: 11520 -> 12096 | medium | +10.026% | +25.5% | +10.2% | +85.7% | -677.8% | changes KV bandwidth/capacity pressure |
| n_kv_heads: 8 -> 64 | low | +10.059% | +23.8% | +9.5% | +85.7% | -700.0% | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 34; n_kv_heads: 8 -> 64; ffn_dim: 11520 -> 12096 | medium | +10.113% | +27.6% | +13.4% | +86.1% | -655.6% | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 35; n_kv_heads: 8 -> 64 | low | +10.140% | +25.9% | +12.6% | +86.0% | -677.8% | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 37; n_kv_heads: 8 -> 32; ffn_dim: 11520 -> 12672 | medium | +10.140% | +24.3% | +12.5% | +86.9% | -311.1% | changes KV bandwidth/capacity pressure |
| n_kv_heads: 8 -> 64; ffn_dim: 11520 -> 10944 | low | +10.180% | +24.3% | +12.0% | +86.0% | -700.0% | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 38; n_kv_heads: 8 -> 32; ffn_dim: 11520 -> 12096 | medium | +10.193% | +22.8% | +12.2% | +87.0% | -322.2% | changes KV bandwidth/capacity pressure |

## Baseline-Dominating Candidates

No local candidate strictly dominates the baseline on performance/resource axes.

## Near-Dominating Candidates

No additional near-dominating variants found inside the risk budget.

## Decode KV Bandwidth

Baseline uses 8 KV heads (GQA group size 8) and 16-bit KV. No accepted local change reduced KV bandwidth inside the risk budget.

## Baseline Ingestion Notes

- MoE baseline: 128 experts × top-4. Modifier scoring uses active-FFN compute mass.

## Caveats

- Same-quality mode means the learned model topology and numerical precision are unchanged in the v0.5 schema.
- Quality values are proxy estimates for ranking nearby candidates, not measured perplexity.
- Risk labels are heuristic placeholders until empirical per-component sensitivity and coupling data are available.
- MoE, state/hybrid, MLA, and heterogeneous layers are reserved hooks and are not modeled in v0.5 modifier mode.