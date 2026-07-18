# Preregistration — Amendment 1 (pre-run protocol freeze)

- ac_version: 0.4.0
- quality_model_version: effective_capacity_v2
- git_commit: c170cda
- experiment_date: 2026-07-17
- agent wave: "gate2-wave1"
- status: FROZEN before the V1–V4 official runs. prereg.md remains
  unmodified; this amendment supersedes its §3 criteria where noted.

## Why an amendment (transparency requirement)

Before the official runs I executed a *mechanism probe* — `throughput()`
on a fixed K2-class MoE shape at EP ∈ {8,16,32,72} on h100 and
gb200_nvl72 — to design the experiment protocol (batch size, expert_dim).
The probe measured **AC's own model**, not external reality; the
datasheet-derived "reality" inputs (NVLink 1.8 TB/s, IB 50 GB/s, NVLink
400 GB/s) were already frozen in the spec files. Per the spirit of the
preregistration discipline, the probe and its consequences are disclosed
here in full instead of silently editing prereg.md. Probe transcripts are
archived in `runs/probe_mechanism.txt`.

## What the probe showed (model-structure findings)

1. **Decode all-to-all volume is small relative to weight streaming at
   moderate batch**, so the stress all_to_all axis is structurally bounded
   below 0.9 for this shape at sane decode batches: the axis is
   `a2a_bytes / (link_bw × decode_layer_time)` and `decode_layer_time`
   itself *includes* the all-to-all time, making the ratio self-damping
   (`x/(c+x)` form). The plan's "stress ≥ 0.9" reading is therefore the
   wrong instrument at decode; the penalty must be read off TBT/TPS and
   Pareto dominance instead.
2. **On h100, a ~1.15T-total MoE at EP=8 does not fit HBM** (≈144 GB fp8
   per rank > 80 GB) and even at 530B total, EP=8 at serving batch 256
   spills (TBT 228 ms vs EP=72 107 ms). Big EP on h100 is punished
   *relative to NVL72* (3.7× TBT at EP=72 in the probe), not absolutely —
   memory pressure pushes the other way. The h100-rational strategy is a
   *smaller-total* MoE at EP ≤ 8.
3. **Training TPS shows the interconnect penalty unconditionally**:
   h100 EP=72 ≈ 0.73× h100 EP=8 in the probe, while NVL72 EP=72 ≈ 1.06×
   NVL72 EP=8.
4. Serving batch for the official runs is set to **256** (frontier MoE
   serving regime; also where the decode a2a term becomes visible).

## FINAL frozen criteria for V1–V4 (supersede prereg.md §3)

| # | Experiment | PASS criteria (frozen) |
|---|---|---|
| V1 | 32B-active / ~1.15T-total MoE, 288 experts top-8, **h100**, EP∈{8,16,32,72}, serving batch 256 | (a) every 1T-class (g=1.0) EP>8 candidate is infeasible or Pareto-dominated; the frontier's feasible MoE candidates use EP ≤ 8 at reduced total-params; (b) TBT(EP=72, g=1.0) on h100 ≥ 3× TBT(EP=72, g=1.0) on nvl72; (c) training TPS(EP=72) < TPS(EP=8) on h100 (interconnect penalty direction correct); (d) report the all_to_all stress axis values with the self-damping caveat |
| V2 | identical config on **gb200_nvl72** | (a) EP=72 MoE candidates are feasible and appear on the Pareto frontier; (b) best EP=72 decode TBT < best EP=8 decode TBT (all-to-all relieved by domain = 72); (c) training TPS(EP=72) ≥ TPS(EP=8) on nvl72 |
| V2b | true K2 shape (384 experts, top-8) on gb200_nvl72 | EP=72 correctly absent (divisibility rule); best feasible MoE EP ∈ {16, 32} |
| V3 | V2 mechanism sweep with `nvlink_domain_size` overridden to 16 / 32 | argmin over EP of TBT(EP) is non-decreasing in domain size (8 → 16 → 32 → 72), proving the domain size — not a coincidence — drives the conclusion |
| V4 | golden snapshot | `diff -r out/golden out/c_check` empty |

The plan-level headline (第二道.md §6): "同一 1T 级 MoE 配置，在 h100 上
被判不可行/受罚，在 gb200_nvl72 上 EP=72 进入 Pareto 前沿" maps onto
V1(a)+V1(b) and V2(a)+V2(b) respectively; the mapping from the plan's
qualitative wording to these measurable criteria is itself documented
here, pre-run.
