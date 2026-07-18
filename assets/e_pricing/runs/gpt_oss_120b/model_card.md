# Architecture Compiler v0 — Model Card

## Intended Use

Dense-transformer architecture ranking under explicit training and serving constraints. Designed to produce internally consistent, falsifiable architecture recommendations with hardware-derived justifications.

## Not Intended For

- Final production architecture selection without training validation.
- Absolute perplexity prediction.
- Cross-architecture-family comparisons (e.g., dense vs. MoE).

## Hardware Scope

NVIDIA H100 SXM (primary, with calibration), NVIDIA B200, Google TPU v5p, Google TPU v5e (analytic only).

## Model Scope

Decoder-only Transformers with MHA/GQA/MQA. Supports dense (v0), MoE (v1), and hybrid attention/state architectures (v2). v2 adds Mamba-2 structured SSM layers with SRAM-derived d_state and hardware-derived hybrid ratios. v1-fix Part J extends the schema validator to accept additional state families — Mamba-1/S4/S5/S6, Sliding Window Attention, DeltaNet, Gated DeltaNet, Kimi Delta Attention (KDA), GLA, RWKV-7, and generic linear attention — with reference-model stubs in PyTorch and JAX.

## Quality Model

Relative expected-loss proxy based on a configurable compiler scaling-law spine plus modular residuals for coupled architecture variables, precision, MoE, state/memory mechanisms, risk, and optional data quality. Dense v0 continuity is preserved with compatibility aliases for shape and GQA, but the architecture residual models width/depth, MLP-attention ratio, d_head, query heads, KV heads, and GQA sharing together.

## Throughput Model

Analytic roofline with optional lightweight kernel calibration. Covers GEMM, fused attention, KV cache bandwidth, TP all-reduce, and PP bubble. System efficiency factors calibrated against published benchmarks.

## Known Failure Modes

- May mis-rank architectures when actual kernel efficiency differs significantly from calibration data.
- May underestimate quality loss from aggressive GQA at large group sizes (>8).
- Additive residual composition breaks down when stacking 3+ large quality risks.
- Data quality defaults to disabled and optimizer stability/training dynamics are not modeled.
- MoE quality terms are hooks with moderate uncertainty calibrated from published routing ablations (Krajewski + Mixtral + DeepSeek priors). Routing imbalance defaults to zero — production deployments with degraded balance must set `load_balance` explicitly.
- MoE all-to-all cost assumes ring-AllToAll efficiency of 0.67 of peak NVLink BW within the NVLink domain and 0.80 of single-axis ICI BW on TPU; cross-axis ICI multiplies by 2.5. These efficiencies are heuristics, not measured.
- State/hybrid quality residuals (v2) model compression and composition penalties with medium confidence; the recall-intensive task multiplier (3x) is empirically motivated but not sweep-calibrated.
- Only Mamba-2 has a measured-empirical reference implementation in `ac-base/`. Other state families (Mamba-1, Gated DeltaNet, KDA, GLA, RWKV-7, sliding-window, generic linear attention) have research-stub references that round-trip shape but are not production-tuned — do not use them to benchmark wall-clock throughput.
- Does not model RL post-training workloads or production scheduler details.
- Does not model compute-communication overlap.

## Validation Target

Within 25% throughput prediction error on known public baselines (Llama-2/3, Mistral). Stable pairwise ranking among architectures within the same parameter band.

## Minimum Claim

Under the stated workload and quality proxy assumptions, the compiler finds a dense architecture that improves predicted serving throughput over a hand-copied Llama-style baseline at similar expected loss. The result is a falsifiable architecture recommendation with explicit hardware-derived justifications, not a claim of global optimality.
