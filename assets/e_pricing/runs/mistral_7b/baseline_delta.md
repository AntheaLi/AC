# Baseline Delta Report

**Baseline**: mistral-7b
**Hardware**: H100
**Selection mode**: quality-spending allowed up to +1.00% relative loss proxy

## Baseline Status

- Baseline loss proxy: 2.0329 (+0.000% vs baseline)
- Baseline training throughput: 47,479 tok/s (+0.0%)
- Baseline TBT: 6.29 ms (+0.0% faster)
- Baseline TTFT: 72.16 ms (+0.0% faster)
- Baseline memory/GPU: 8.10 GB (+0.0% lower)
- Baseline modeled KV cache/GPU (per request): 0.125 GB (+0.0% lower)

- Risk-aware Pareto status: baseline remains on the frontier because lower-risk/no-change is itself an objective.
- Performance-only status: baseline is dominated by local variants when risk is excluded from the objective axes.

## Selected Modification

Selected change: **n_layers: 32 -> 31; ffn_dim: 14336 -> 15040**
Move class: **architecture**
Model-preserving (deployment-only, no retraining): **False**
Risk: **medium** (score 1.25)
- Selected loss proxy: 2.0328 (-0.004% vs baseline)
- Selected training throughput: 47,645 tok/s (+0.3%)
- Selected TBT: 6.16 ms (+2.1% faster)
- Selected TTFT: 71.70 ms (+0.6% faster)
- Selected memory/GPU: 7.96 GB (+1.7% lower)
- Selected modeled KV cache/GPU (per request): 0.121 GB (+3.1% lower)

## Same-Quality Hardware-Fit Modifications

| Change | Risk | Loss Risk | TBT Improvement | Train TPS Improvement | Mem Improvement | Modeled KV (per request) Improvement | Reason |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | low | +0.000% | +0.0% | +0.0% | +0.0% | +0.0% | local architecture perturbation |

## Optional Quality-Spending Modifications

These candidates change architecture or numerics and should be treated as retraining/calibration options, not same-quality hardware-fit edits.

| Change | Risk | Loss Risk | TBT Improvement | Train TPS Improvement | Mem Improvement | Modeled KV (per request) Improvement | Reason |
|---|---:|---:|---:|---:|---:|---:|---|
| n_layers: 32 -> 33; n_kv_heads: 8 -> 16; ffn_dim: 14336 -> 15744 | medium | -0.171% | -37.5% | -8.8% | -55.8% | 2.1× larger | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 33; n_kv_heads: 8 -> 32 | low | -0.169% | -99.7% | -8.4% | 2.6× larger | 4.1× larger | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 34; ffn_dim: 14336 -> 15744 | medium | -0.165% | -8.5% | -9.8% | -6.6% | -6.2% | trades depth quality proxy against latency |
| n_kv_heads: 8 -> 32; ffn_dim: 14336 -> 15040 | low | -0.160% | -94.7% | -8.0% | 2.5× larger | 4× larger | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 33; ffn_dim: 14336 -> 15744 | medium | -0.136% | -5.3% | -7.1% | -4.1% | -3.1% | trades depth quality proxy against latency |
| n_layers: 32 -> 33; n_kv_heads: 8 -> 16; ffn_dim: 14336 -> 15040 | medium | -0.135% | -36.4% | -7.5% | -55.0% | 2.1× larger | changes KV bandwidth/capacity pressure |
| n_kv_heads: 8 -> 16; ffn_dim: 14336 -> 15744 | medium | -0.132% | -33.3% | -5.9% | -51.7% | -100.0% | changes KV bandwidth/capacity pressure |
| n_kv_heads: 8 -> 32 | low | -0.130% | -93.6% | -5.5% | 2.5× larger | 4× larger | changes KV bandwidth/capacity pressure |

## Baseline-Dominating Candidates

These variants dominate the baseline on performance/resource axes before accounting for risk as a separate objective.

| Change | Risk | Loss Risk | TBT Improvement | Train TPS Improvement | Mem Improvement | Modeled KV (per request) Improvement | Reason |
|---|---:|---:|---:|---:|---:|---:|---|
| n_layers: 32 -> 31; ffn_dim: 14336 -> 15040 | medium | -0.004% | +2.1% | +0.3% | +1.7% | +3.1% | trades depth quality proxy against latency |

## Near-Dominating Candidates

These variants improve at least one resource axis while staying inside the quality-risk budget.

| Change | Risk | Loss Risk | TBT Improvement | Train TPS Improvement | Mem Improvement | Modeled KV (per request) Improvement | Reason |
|---|---:|---:|---:|---:|---:|---:|---|
| n_layers: 32 -> 30; ffn_precision: bf16 -> fp8; kv_cache_bits: 16 -> 8 | medium | +0.398% | +32.6% | +34.3% | +35.6% | +53.1% | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 31; ffn_precision: bf16 -> fp8; kv_cache_bits: 16 -> 8 | medium | +0.337% | +30.3% | +30.0% | +34.2% | +51.6% | changes KV bandwidth/capacity pressure |
| ffn_precision: bf16 -> fp8; kv_cache_bits: 16 -> 8 | low | +0.288% | +28.1% | +25.9% | +32.8% | +50.0% | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 30; ffn_dim: 14336 -> 13632; ffn_precision: bf16 -> fp8; kv_cache_bits: 16 -> 8 | high | +0.461% | +32.7% | +35.1% | +36.0% | +53.1% | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 31; ffn_dim: 14336 -> 13632; ffn_precision: bf16 -> fp8; kv_cache_bits: 16 -> 8 | medium | +0.400% | +30.4% | +30.8% | +34.6% | +51.6% | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 30; ffn_dim: 14336 -> 15040; ffn_precision: bf16 -> fp8; kv_cache_bits: 16 -> 8 | high | +0.345% | +32.5% | +31.9% | +35.3% | +53.1% | changes KV bandwidth/capacity pressure |
| ffn_dim: 14336 -> 13632; ffn_precision: bf16 -> fp8; kv_cache_bits: 16 -> 8 | medium | +0.351% | +28.2% | +26.7% | +33.2% | +50.0% | changes KV bandwidth/capacity pressure |
| n_layers: 32 -> 31; ffn_dim: 14336 -> 15040; ffn_precision: bf16 -> fp8; kv_cache_bits: 16 -> 8 | medium | +0.284% | +30.3% | +27.6% | +33.8% | +51.6% | changes KV bandwidth/capacity pressure |

## Decode KV Bandwidth

Baseline uses 8 KV heads (GQA group size 4) and 16-bit KV. Selected config uses 8 KV heads (group size 4) and 16-bit KV. The estimated per-GPU KV cache footprint changes from 0.125 GB to 0.121 GB under the current single-stream decode memory proxy.

## Caveats

- Same-quality mode means the learned model topology and numerical precision are unchanged in the modifier schema.
- Quality values are proxy estimates for ranking nearby candidates, not measured perplexity.
- Risk labels are heuristic placeholders until empirical per-component sensitivity and coupling data are available.
- MoE, state/hybrid, MLA, and heterogeneous layer edits are reserved hooks and are not selected by the current local modifier mode.