# Baseline Delta Report

**Baseline**: mistral-7b
**Hardware**: H100
**Selection mode**: quality-spending allowed up to +1.00% relative loss proxy

## Baseline Status

- Baseline loss proxy: 2.0247 (+0.000% vs baseline)
- Baseline training throughput: 11,878 tok/s (+0.0%)
- Baseline TBT: 5.91 ms (+0.0% faster)
- Baseline TTFT: 33.64 ms (+0.0% faster)
- Baseline memory/GPU: 1.97 GB (+0.0% lower)
- Baseline modeled KV cache/GPU: 0.125 GB (+0.0% lower)

- Risk-aware Pareto status: baseline remains on the frontier because lower-risk/no-change is itself an objective.
- Performance-only status: no local variant strictly dominates the baseline across quality, latency, throughput, memory, and KV footprint.

## Selected Modification

Selected change: **ffn_precision: bf16 -> fp8**
Move class: **precision**
Quality-preserving: **False**
Risk: **low** (score 0.5)
- Selected loss proxy: 2.0449 (+0.998% vs baseline)
- Selected training throughput: 22,117 tok/s (+86.2%)
- Selected TBT: 5.29 ms (+10.5% faster)
- Selected TTFT: 17.23 ms (+48.8% faster)
- Selected memory/GPU: 1.05 GB (+46.8% lower)
- Selected modeled KV cache/GPU: 0.125 GB (+0.0% lower)

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

| Change | Risk | Loss Risk | TBT Improvement | Train TPS Improvement | Mem Improvement | Modeled KV Improvement | Reason |
|---|---:|---:|---:|---:|---:|---:|---|
| n_kv_heads: 8 -> 32; ffn_dim: 14336 -> 15040 | low | -0.627% | -8.1% | -12.3% | -31.9% | -300.0% | changes KV bandwidth/capacity pressure |
| n_kv_heads: 8 -> 32 | low | -0.530% | -7.3% | -10.2% | -28.6% | -300.0% | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 31; n_kv_heads: 8 -> 32; ffn_dim: 14336 -> 15040 | medium | -0.529% | -4.7% | -9.5% | -28.1% | -287.5% | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 33; n_kv_heads: 8 -> 16; ffn_dim: 14336 -> 15744 | medium | -0.523% | -7.3% | -9.4% | -19.4% | -106.2% | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 30; n_kv_heads: 8 -> 32; ffn_dim: 14336 -> 15744 | high | -0.516% | -2.1% | -8.7% | -27.4% | -275.0% | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 34; n_kv_heads: 8 -> 16; ffn_dim: 14336 -> 15040 | medium | -0.505% | -9.7% | -9.7% | -19.2% | -112.5% | changes KV bandwidth/capacity pressure |
| n_kv_heads: 8 -> 16; ffn_dim: 14336 -> 15744 | medium | -0.432% | -4.1% | -6.5% | -16.1% | -100.0% | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 31; n_kv_heads: 8 -> 32 | low | -0.432% | -3.9% | -7.3% | -24.9% | -287.5% | changes KV bandwidth/capacity pressure |

## Baseline-Dominating Candidates

No local candidate strictly dominates the baseline on performance/resource axes.

## Near-Dominating Candidates

These variants improve at least one resource axis while staying inside the quality-risk budget.

| Change | Risk | Loss Risk | TBT Improvement | Train TPS Improvement | Mem Improvement | Modeled KV Improvement | Reason |
|---|---:|---:|---:|---:|---:|---:|---|
| ffn_precision: bf16 -> fp8 | low | +0.998% | +10.5% | +86.2% | +46.8% | +0.0% | uses lower-precision FFN matmuls |
| n_layers: 32 -> 31; ffn_dim: 14336 -> 15040; ffn_precision: bf16 -> fp8 | medium | +0.978% | +12.9% | +87.7% | +46.7% | +3.1% | uses lower-precision FFN matmuls |
| n_layers: 32 -> 30; n_kv_heads: 8 -> 16; ffn_precision: bf16 -> fp8 | medium | +0.985% | +14.2% | +95.4% | +42.4% | -87.5% | changes KV bandwidth/capacity pressure |
| ffn_dim: 14336 -> 15040; ffn_precision: bf16 -> fp8 | low | +0.878% | +10.1% | +81.8% | +45.2% | +0.0% | uses lower-precision FFN matmuls |
| n_layers: 32 -> 31; n_kv_heads: 8 -> 16; ffn_precision: bf16 -> fp8 | medium | +0.878% | +11.4% | +89.2% | +40.6% | -93.8% | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 31; n_kv_heads: 8 -> 16; ffn_dim: 14336 -> 13632; ffn_precision: bf16 -> fp8 | medium | +0.998% | +11.8% | +93.8% | +42.2% | -93.8% | changes KV bandwidth/capacity pressure |
| n_kv_heads: 8 -> 16; ffn_precision: bf16 -> fp8 | low | +0.778% | +8.5% | +83.3% | +38.9% | -100.0% | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 30; ffn_dim: 14336 -> 15744; ffn_precision: bf16 -> fp8 | high | +0.973% | +15.4% | +89.4% | +46.7% | +6.2% | uses lower-precision FFN matmuls |

## Decode KV Bandwidth

Baseline uses 8 KV heads (GQA group size 4) and 16-bit KV. Selected config uses 8 KV heads (group size 4) and 16-bit KV. The estimated per-GPU KV cache footprint changes from 0.125 GB to 0.125 GB under the current single-stream decode memory proxy.

## Caveats

- Same-quality mode means the learned model topology and numerical precision are unchanged in the modifier schema.
- Quality values are proxy estimates for ranking nearby candidates, not measured perplexity.
- Risk labels are heuristic placeholders until empirical per-component sensitivity and coupling data are available.
- MoE, state/hybrid, MLA, and heterogeneous layer edits are reserved hooks and are not selected by the current local modifier mode.