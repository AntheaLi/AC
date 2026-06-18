# Architecture Compiler v0 — Assumptions

## Quality Proxy

- No training sweeps were performed. The quality proxy is based on a scaling-law spine over active non-embedding parameters (Hoffmann et al. 2022 defaults) plus modular residuals for architecture, precision, MoE, state/memory hooks, risk, and data quality.
- The quality proxy predicts relative expected loss among nearby architecture candidates; it is not an absolute perplexity predictor.
- Architecture residuals couple width/depth, MLP-to-attention allocation, d_head, query heads, KV heads, and GQA sharing. Query heads are a weak width-derived prior, not a monotonic quality law; KV heads carry the direct memory/latency tradeoff and GQA-sharing uncertainty.
- Precision residuals use a configurable per-component sensitivity table plus hardware feasibility checks from the legacy penalty table.
- MoE and state/hybrid residuals are explicit high-uncertainty hooks. Dense v0 search does not enumerate those families yet.
- Residual composition is additive. In practice, residuals interact (e.g., GQA + FP4 KV may be worse than the sum). The coupling matrix is deferred to a later calibrated model.
- State/hybrid residuals (v2) model two penalties: compression (effective memory horizon vs context) and composition (state fraction × d_state ratio × context scaling). Recall-intensive tasks incur a 3x compression penalty multiplier. Quality saturation caps d_state at 256 regardless of hardware capacity.
- Uncertainty intervals are approximate. They capture residual confidence levels, scaling-law regime uncertainty, and placeholder uncertainty for uncalibrated hooks; data distribution effects are disabled by default.

## MoE Quality Residual (v1)

- The MoE residual is calibrated against Krajewski et al. (2024) plus published Mixtral 8×7B and DeepSeek-V2/V3 priors. It is not fit against measured loss deltas from training runs the compiler can run itself.
- Capacity bonus: `-0.05 × min(log(N_total/N_active), log(4))`. The `log(4)` cap is deliberate so that highly sparse top_k=1 configs are not over-rewarded purely by their `N_total/N_active` ratio.
- Granularity bonus: `-0.005 × log(max(G/8, 1))` where `G = N_total / top_k`. Applied above the Krajewski reference granularity of 8.
- Shared-expert adjust: `-0.005 × shared_ratio` (DeepSeek-V2 shared-expert ablation).
- Top-k=1 (Switch) penalty: `+0.015` — Switch vs ≥top_k=2 ablations in the Mixture-of-Experts literature.
- Router-fp8 penalty: `+0.005` when router precision is fp8 (DeepSeekMoE router-stability ablation).
- Routing-imbalance penalty: `+0.10 × max(0, 1 - load_balance)`. **Defaults to zero** because v1 assumes balanced routing under a load-balance loss. Production deployments with degraded balance (e.g., LB loss removed, no dropless routing) should set `load_balance` explicitly.
- The MoE residual is currently a smooth function of (`n_experts`, `top_k`, `expert_dim`, `shared_dim`). It does NOT model expert collapse, dropless-vs-dropping scheduler tradeoffs, or token-choice vs expert-choice routing. These are out of scope for v1.
- First-K-dense FFN pattern (v1-fix Part B) attenuates the capacity bonus linearly by `n_moe_layers / n_layers` and adds a small stability bonus (`-0.003` per dense prefix layer, up to 3 layers). Source: DeepSeek-V3 / Qwen3-MoE conventions.

## State / Hybrid Quality Residual (v2)

- Only **Mamba-2** has a measured-empirical reference implementation in `ac-base/` (PyTorch and JAX). Other state families (Mamba-1, Gated DeltaNet, KDA, GLA, RWKV-7, sliding-window, generic linear attention) are validated by the schema and routed to the correct family in the quality residual, but their reference models are research-stub: shapes are correct and `verify_forward()` round-trips, but they are not production-tuned and should not be used to benchmark wall-clock throughput.
- The 5-term family-specific decomposition (`f_hybrid_ratio`, `f_state_capacity`, `f_kv_cost`, `f_recall_risk`, `f_family_uncertainty`) uses the band-pass priors in `configs/quality/quality_v1_defaults.yaml:state_residual.families`. Bands and recall minima are anchored on Jamba / Nemotron-H / Zamba / DeepSeek-V3 ablations; they should be re-fit when more sweeps land.
- Family resolution rules (`_resolve_hybrid_family`):
  - `mamba_sequential` — Mamba-1/2, S4/S5/S6.
  - `gated_delta_or_kda_linear` — DeltaNet, Gated DeltaNet, Kimi Delta Attention (KDA), GLA (when used with gated linear attention).
  - `generic_linear_attention` — RWKV-7, RetNet, generic linear attention; also the fallback when a per-family band hasn't been calibrated.
  - `parallel_hybrid_heads` — MoH-style parallel splits where attention/state mixers run in parallel within a layer rather than alternating across layers.
  - `recurrent_local_attention` — Sliding Window Attention (SWA), local-attention + recurrent state hybrids.
- `recall_intensive` workloads pay a 3× compression-penalty multiplier (heuristic from RULER / passkey ablations); other workloads pay 1×.
- The 5-term residual is uncertainty-heavy on purpose: family confidence ranges from `medium` (mamba_sequential, generic_linear_attention) to `medium-low` (parallel_hybrid_heads, recurrent_local_attention).

## Throughput Model

- Throughput is estimated via analytic roofline modeling: per-operation cost = max(compute_time, memory_time).
- Kernel calibration, when available, overrides analytic GEMM efficiency estimates. If calibration data is absent, public spec estimates are used (source: "public_estimate").
- No compute-communication overlap is assumed. In production systems, overlap can improve throughput by 10-30%.
- Pipeline bubble fraction uses the simple (pp-1)/(M+pp-1) formula. Interleaved schedules are not modeled.

## Serving Model

- Serving workload is modeled using three canonical regimes (prefill-heavy, decode-heavy, mixed). This does not capture continuous batching scheduler dynamics.
- KV cache memory is computed assuming static allocation for the full context length. PagedAttention-style dynamic allocation is not modeled.
- Latency predictions are single-request estimates. Queuing effects under concurrency are not modeled in v0.

## Search

- Brute-force enumeration over the lattice. No gradient-based or Bayesian optimization. Tractable because the lattice constrains the space to ~1K-9K candidates per configuration.
- Parallelism (TP, PP, DP) is treated as a fixed input, not a search variable. Joint architecture-parallelism search is deferred to v3.

## Hardware

- Hardware specs are from public documentation. Actual performance may vary based on firmware, driver version, cooling, and system configuration.
- Results are intended for architecture ranking, not final production deployment validation.
