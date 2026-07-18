# Preregistration — Amendment 2 (serving operating point fix)

- ac_version: 0.4.0
- quality_model_version: effective_capacity_v2
- git_commit: c170cda
- experiment_date: 2026-07-17
- agent wave: "gate2-wave1"
- status: FROZEN before the official V1–V4 runs. Supersedes the serving
  batch choice in amendment 1; PASS criteria unchanged.

## What happened

The first official CLI run at the amendment-1 operating point
(--serving-batch 256, context 8192, no MLA) produced a pathological
serving regime: KV cache alone is ~524 GB/GPU (bf16, GQA-8, tp=1), so
EVERY candidate on BOTH targets served from deep NVLink/PCIe spill
(598–1803 GB/GPU footprint, TBT 100–6000 ms). At that point the TBT axis
measures the spill tier, not the EP economics, and the frontier is
uninformative for V1/V2. The run is archived unchanged at
`runs/v2_nvl72_pathological_batch256/` for the record.

## Fix (operating point only, criteria untouched)

- Add `--allow-mla` (MLA is the K2/DeepSeek-V3 attention family; KV
  latent 512 + rope 64 cuts KV ~30–60× and is what makes 1T-class serving
  physically possible at all). Without it the experiment silently forbids
  the attention family the real systems use.
- Serving batch 256 → 64 (heavy but plausible large-MoE serving batch
  once MLA is active; KV ≈ 37 GB bf16 at 8k context, fits both targets'
  HBM without spill).
- The frozen PASS criteria from amendment 1 are unchanged: they are
  directional EP-economics statements (ratios, monotonicity, frontier
  composition), not absolute TBT values, so the operating-point fix does
  not touch them.

## Already-known consequence (disclosed pre-run)

From the archived probe: at sane operating points, on h100 a 1.15T-total
fp8 MoE does not fit 80 GB HBM at EP=8 (≈144 GB/rank) — small EP is
memory-infeasible, and EP ≥ 32 becomes mandatory on h100 while paying
the cross-node all-to-all tax. On gb200_nvl72 every EP fits and EP=72 is
TBT-optimal. V1(a) is therefore evaluated as frozen: "every 1T-class
(g=1.0) EP>8 candidate is infeasible or Pareto-dominated" — the
amendment-1 text already anticipated the dominance reading.
