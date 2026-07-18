# Baseline Delta Report

**Baseline**: gpt-oss-120b
**Hardware**: H100
**Selection mode**: quality-spending allowed up to +1.00% relative loss proxy

## Modifier Scope

Before reading the results below, note which moves this modifier did **not** enumerate for your baseline:

- **MoE baseline detected.** The local modifier currently sweeps dense-side moves (width / depth / FFN precision / KV bits). It does NOT sweep `n_experts`, `top_k`, expert-parallel degree, or first-K-dense prefix. Use `ac-delta-eval --apply change_moe_topology` or `--apply densify_first_k` to evaluate those.

## Baseline Status

- Baseline loss proxy: 1.9838 (+0.000% vs baseline)
- Baseline training throughput: 36,788 tok/s (+0.0%)
- Baseline TBT: 6.22 ms (+0.0% faster)
- Baseline TTFT: 55.13 ms (+0.0% faster)
- Baseline memory/GPU: 6.73 GB (+0.0% lower)
- Baseline modeled KV cache/GPU (per request): 0.070 GB (+0.0% lower)

- Risk-aware Pareto status: baseline remains on the frontier because lower-risk/no-change is itself an objective.
- Performance-only status: baseline is dominated by local variants when risk is excluded from the objective axes.

## Selected Modification

Selected change: **n_layers: 36 -> 34; kv_cache_bits: 16 -> 8; expert_dim: 2880 -> 3008**
Move class: **precision**
Model-preserving (deployment-only, no retraining): **False**
Risk: **medium** (score 2.5)
- Selected loss proxy: 1.9830 (-0.041% vs baseline)
- Selected training throughput: 39,282 tok/s (+6.8%)
- Selected TBT: 5.62 ms (+9.7% faster)
- Selected TTFT: 53.19 ms (+3.5% faster)
- Selected memory/GPU: 6.01 GB (+10.7% lower)
- Selected modeled KV cache/GPU (per request): 0.033 GB (+52.8% lower)

## Same-Quality Hardware-Fit Modifications

| Change | Risk | Loss Risk | TBT Improvement | Train TPS Improvement | Mem Improvement | Modeled KV (per request) Improvement | Reason |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | low | +0.000% | +0.0% | +0.0% | +0.0% | +0.0% | local architecture perturbation |

## Optional Quality-Spending Modifications

These candidates change architecture or numerics and should be treated as retraining/calibration options, not same-quality hardware-fit edits.

| Change | Risk | Loss Risk | TBT Improvement | Train TPS Improvement | Mem Improvement | Modeled KV (per request) Improvement | Reason |
|---|---:|---:|---:|---:|---:|---:|---|
| n_layers: 36 -> 34; n_kv_heads: 8 -> 64; expert_dim: 2880 -> 3200 | high | -0.653% | -68.5% | -3.6% | 2.1× larger | 7.6× larger | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 35; n_kv_heads: 8 -> 64; expert_dim: 2880 -> 3200 | medium | -0.649% | -75.9% | -6.7% | 2.2× larger | 7.8× larger | changes KV bandwidth/capacity pressure |
| n_kv_heads: 8 -> 64; expert_dim: 2880 -> 3200 | medium | -0.638% | -83.4% | -9.7% | 2.3× larger | 8× larger | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 34; n_kv_heads: 8 -> 64; kv_cache_bits: 16 -> 8; expert_dim: 2880 -> 3200 | high | -0.555% | -29.7% | -3.6% | -47.8% | 3.8× larger | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 35; n_kv_heads: 8 -> 64; kv_cache_bits: 16 -> 8; expert_dim: 2880 -> 3200 | medium | -0.551% | -34.7% | -6.7% | -53.4% | 3.9× larger | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 34; n_kv_heads: 8 -> 64; expert_dim: 2880 -> 3008 | medium | -0.548% | -66.5% | +1.5% | 2.1× larger | 7.6× larger | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 35; n_kv_heads: 8 -> 64; expert_dim: 2880 -> 3008 | medium | -0.543% | -73.9% | -1.8% | 2.1× larger | 7.8× larger | changes KV bandwidth/capacity pressure |
| n_kv_heads: 8 -> 64; kv_cache_bits: 16 -> 8; expert_dim: 2880 -> 3200 | medium | -0.540% | -39.8% | -9.7% | -59.0% | 4× larger | changes KV bandwidth/capacity pressure |

## Baseline-Dominating Candidates

These variants dominate the baseline on performance/resource axes before accounting for risk as a separate objective.

| Change | Risk | Loss Risk | TBT Improvement | Train TPS Improvement | Mem Improvement | Modeled KV (per request) Improvement | Reason |
|---|---:|---:|---:|---:|---:|---:|---|
| n_layers: 36 -> 34; kv_cache_bits: 16 -> 8; expert_dim: 2880 -> 3008 | medium | -0.041% | +9.7% | +6.8% | +10.7% | +52.8% | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 34 | medium | -0.026% | +6.1% | +6.8% | +5.2% | +5.6% | trades depth quality proxy against latency |
| n_layers: 36 -> 35; kv_cache_bits: 16 -> 8; expert_dim: 2880 -> 3008 | medium | -0.033% | +6.8% | +3.3% | +8.5% | +51.4% | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 35 | low | -0.017% | +3.1% | +3.3% | +2.6% | +2.8% | trades depth quality proxy against latency |
| n_layers: 36 -> 34; expert_dim: 2880 -> 3008 | medium | -0.140% | +4.8% | +6.8% | +3.2% | +5.6% | trades depth quality proxy against latency |
| n_layers: 36 -> 34; n_kv_heads: 8 -> 16; kv_cache_bits: 16 -> 8 | medium | -0.001% | +5.7% | +6.0% | +4.9% | +5.6% | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 34; kv_cache_bits: 16 -> 8; expert_dim: 2880 -> 3200 | high | -0.194% | +7.7% | +1.1% | +7.6% | +52.8% | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 35; expert_dim: 2880 -> 3008 | medium | -0.131% | +1.7% | +3.3% | +0.5% | +2.8% | trades depth quality proxy against latency |

## Near-Dominating Candidates

These variants improve at least one resource axis while staying inside the quality-risk budget.

| Change | Risk | Loss Risk | TBT Improvement | Train TPS Improvement | Mem Improvement | Modeled KV (per request) Improvement | Reason |
|---|---:|---:|---:|---:|---:|---:|---|
| n_layers: 36 -> 34; kv_cache_bits: 16 -> 8 | medium | +0.073% | +11.0% | +6.8% | +12.8% | +52.8% | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 34; kv_cache_bits: 16 -> 8; expert_dim: 2880 -> 2752 | medium | +0.197% | +12.3% | +6.8% | +14.9% | +52.8% | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 35; kv_cache_bits: 16 -> 8 | medium | +0.082% | +8.2% | +3.3% | +10.6% | +51.4% | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 35; kv_cache_bits: 16 -> 8; expert_dim: 2880 -> 2560 | medium | +0.416% | +11.7% | +9.4% | +16.0% | +51.4% | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 35; kv_cache_bits: 16 -> 8; expert_dim: 2880 -> 2752 | medium | +0.207% | +9.6% | +3.3% | +12.8% | +51.4% | changes KV bandwidth/capacity pressure |
| kv_cache_bits: 16 -> 8 | low | +0.099% | +5.4% | +0.0% | +8.5% | +50.0% | changes KV bandwidth/capacity pressure |
| kv_cache_bits: 16 -> 8; expert_dim: 2880 -> 2560 | medium | +0.434% | +9.0% | +5.9% | +14.0% | +50.0% | changes KV bandwidth/capacity pressure |
| n_layers: 36 -> 34; ffn_precision: bf16 -> fp8; kv_cache_bits: 16 -> 8 | medium | +0.271% | +11.0% | +6.8% | +12.8% | +52.8% | changes KV bandwidth/capacity pressure |

## Decode KV Bandwidth

Baseline uses 8 KV heads (GQA group size 8) and 16-bit KV. Selected config uses 8 KV heads (group size 8) and 8-bit KV. The estimated per-GPU KV cache footprint changes from 0.070 GB to 0.033 GB under the current single-stream decode memory proxy.

## Baseline Ingestion Notes

- Local:global interleave baseline: 18 of 36 layers are sliding-window (window=128); shape read from the dominant global band.
- MoE baseline: 128 experts × top-4. Modifier scoring uses active-FFN compute mass.

## Caveats

- Same-quality mode means the learned model topology and numerical precision are unchanged in the modifier schema.
- Quality values are proxy estimates for ranking nearby candidates, not measured perplexity.
- Risk labels are heuristic placeholders until empirical per-component sensitivity and coupling data are available.
- MoE, state/hybrid, MLA, and heterogeneous layer edits are reserved hooks and are not selected by the current local modifier mode.