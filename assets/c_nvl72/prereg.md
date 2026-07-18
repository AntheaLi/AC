# Preregistration — Task C: NVL72 rack-scale modeling (V1–V4)

- ac_version: 0.4.0
- quality_model_version: effective_capacity_v2
- git_commit: c170cda
- experiment_date: 2026-07-17
- agent wave: "gate2-wave1"
- task: 第二道 子任务 C（GB200 NVL72 / H800 / EP=72）

> Written BEFORE any experiment run. Predictions below are restatements of
> 第二道-C-NVL72机架级建模.md §2, made numeric where the plan left them
> qualitative. No result in this directory may be edited to match reality
> after the fact; mismatches are reported as mismatches in report.md.

## 1. What is being added (pre-registered scope)

- `ac/hardware_specs/gb200_nvl72.json` — B200 chip params reused
  (`"chip": "b200"`), plus a `system` block:
  `nvlink_domain_size=72`, `intra_domain_bandwidth_gbps=1800`,
  `inter_domain_bandwidth_gbps=50`, `gpus_per_rack=72`.
  - **Unit deviation (documented):** the plan's example wrote
    `inter_domain_bandwidth_gbps: 400`. 400 is the datasheet figure in
    **Gb/s** (ConnectX-7 NDR, one 400 Gb/s port per GPU). AC's
    `interconnect.*_bw_gb_s` fields are in **GB/s** everywhere, so the spec
    stores 50 (= 400/8). Storing 400 would make cross-rack bandwidth 4×
    faster than DGX-B200's existing 100 GB/s — physically wrong.
- `ac/hardware_specs/h800.json` — exact copy of `h100_sxm.json` except
  `accelerator_name` and `interconnect.intra_node_bw_gb_s: 900 → 400`
  (NVLink 4 reduced on the export SKU). Everything else identical silicon.
- Loader: optional `system` block parsed into existing fields; absent →
  single-node semantics, old specs byte-identical. New optional
  `gpus_per_rack` / `chip` attributes default to `None`.
- Registration (pure-additive, no existing-target behavior change):
  `load_hardware` mapping, CLI `VALID_HARDWARE` (compile / delta-eval /
  stress), `lattice_engine.HARDWARE` + `NVLINK_DOMAIN` (gb200_nvl72→72,
  h800→8), `get_precision_configs_for_hardware` (nvl72=b200 set,
  h800=h100 set), `PRECISION_HARDWARE_SUPPORT` (same), `_get_hbm_gb`
  (192 / 80), calibration aliases (gb200_nvl72→b200, h800→h100 measured
  tables; bandwidth differences come from the spec JSON, compute-side
  efficiency transfers across the same silicon).
- **Pre-existing capability note:** the hierarchical all-to-all branch
  (`ep <= nvlink_domain_size` → intra-domain BW; `ep > domain` → split
  intra/inter traffic, `max(t_intra, t_inter)`) ALREADY exists in
  `throughput_model._moe_alltoall_cost` (Wave 19 P0-1), and the stress
  all_to_all axis already routes bandwidth by domain
  (`stress._link_bw_bytes_s`). Task C therefore adds NO new formula — it
  wires the new system targets into the existing branch and pins the
  behavior with numeric unit tests. This keeps the zero-regression gate
  achievable by construction.

## 2. Protocol decisions (deviations from plan text, with reasons)

1. **n_experts 288 instead of 384 for the EP-sweep runs.** AC's enumerator
   requires `n_experts % ep == 0` (lattice_engine.compute_moe_options).
   384 % 72 ≠ 0, so EP=72 candidates are never generated for 384 experts —
   the plan's "384 experts × EP∈{8,16,32,72}" grid is empty at EP=72.
   lcm(8,16,32,72)=288, so 288 experts (top_k=8, ratio 36 ≥ 8 fine-grained
   floor) makes all four EP degrees enumerable. A companion run with the
   true K2 shape (384 experts) is included as V2b, where EP=72 is expected
   to be correctly absent and EP=32 to dominate on NVL72.
2. **`--training-cluster-gpus 2304` on both targets** (288 H100 nodes /
   32 NVL72 racks). Default `--dp 8` would filter EP∈{16,32,72} out of the
   search on BOTH targets (`_filter_ep_options_by_dp`), making V2 impossible
   by construction. The cluster floor unlocks the EP axis so the
   interconnect economics — not a search-space artifact — decide.
3. **`--max-total-params-b 1200` instead of 1100.** With 288 experts the
   g=1.0 granularity candidates land at ~1.15T total params; an 1100B cap
   would filter exactly the 1T-class candidates the experiment is about.
4. `--tp 1 --pp 1`: large-EP MoE serving (K2-style) uses EP in place of TP
   for expert layers; TP=1 keeps the serving instance = EP ranks so the
   all-to-all domain comparison is clean. Attention stays replicated.
5. Small budgets (`--max-candidates 200`, granularity sweep 1.0/0.5) so
   each run finishes in minutes.

## 3. Pre-registered expected outcomes

| # | Experiment | Expected (must hold for PASS) |
|---|---|---|
| V1 | 32B-active / ~1.15T-total MoE, 288 experts top-8, **h100**, EP∈{8,16,32,72} | (a) EP=72 candidate decode TBT is ≥ 5× worse than EP=8 at comparable loss, OR EP>8 candidates are infeasible; (b) the all_to_all stress axis for EP=72 on h100 ≥ 0.9; (c) Pareto-optimal pick uses EP ≤ 8 |
| V2 | Identical config on **gb200_nvl72** | (a) EP=72 candidates appear ON the Pareto frontier; (b) best EP=72 decode TBT < best EP=8 decode TBT (all-to-all relieved by the 72-GPU domain); (c) best EP=72 TBT on nvl72 is ≥ 5× better than EP=72 on h100 |
| V2b | True K2 shape (384 experts, top-8) on gb200_nvl72 | EP=72 correctly absent (divisibility); EP=32 on frontier; documents the padding nuance honestly |
| V3 | V2 rerun with `nvlink_domain_size` overridden to 16 and 32 (spec-copy via AC_HARDWARE_SPEC_DIR) | Best-EP-by-TBT moves monotonically with domain size: domain 16 → EP≤16 beats EP=72; domain 32 → EP≤32 beats EP=72; i.e. argmin_Ep TBT is non-decreasing in domain size |
| V4 | `scripts/golden_snapshot.sh out/c_check && diff -r out/golden out/c_check` | Empty diff (zero regression) |

## 4. Secondary acceptance checks

- `ac-compile --hardware gb200_nvl72 ...` and `--hardware h800 ...` run
  end-to-end (exit 0); same for `ac-delta-eval`.
- New tests `test_nvl72_system_spec.py`, `test_ep_domain_routing.py`,
  `test_ep72_pareto.py` pass; full `python -m pytest tests/ -q` stays green
  at 796+3-file additions, 1 protocol-gated skip.
- `git diff --stat` touches only: the two new specs, the additive
  registration points listed in §1, the three new test files, and
  `validation/c_nvl72/*`. `ac/quality_defaults.yaml`, `ac/calibration/`,
  `configs/`, existing spec fields, existing test assertions: untouched.

## 5. Fact provenance (to be embedded in spec `notes` fields)

- GB200 NVL72: 72 GPUs + 36 Grace CPUs, one NVLink domain; NVLink 5 =
  1.8 TB/s per GPU, 130 TB/s rack aggregate — NVIDIA GB200 NVL72 product
  page, https://www.nvidia.com/en-us/data-center/gb200-nvl72/ ,
  accessed 2026-07-17, tier T2.
- GB200 NVL72 scale-out: ConnectX-7 / Quantum-2 NDR InfiniBand, 400 Gb/s
  per GPU (= 50 GB/s) — NVIDIA GB200 NVL72 reference configuration as
  listed by NVIDIA partner spec sheets (e.g.
  https://www.spheron.network/gpu-rental/gb200/ ,
  https://pcbstore.com.bd/product/nvidia-gb200-nvl72-48u-rack-solution),
  accessed 2026-07-17, tier T3 (exact per-GPU NIC count is
  deployment-configurable; 400 Gb/s/GPU is the reference).
- B200 per-GPU params reused from ac/hardware_specs/b200.json (192 GB
  HBM3e, 8 TB/s, datasheet peaks 2250/4500/9000 TFLOPS) — NVIDIA B200 /
  DGX B200 datasheet, https://www.nvidia.com/en-us/data-center/dgx-b200/ ,
  tier T2 (as already recorded in b200.json notes).
- H800 SXM: H100 SXM silicon, 80 GB HBM3, 3.35 TB/s, BF16 1979 TFLOPS
  (sparse) / 989 dense, FP8 3958 (sparse) / 1979 dense — NVIDIA H800 GPU
  datasheet (mirrored copy,
  https://www.scribd.com/document/777167019/NVIDIA-H800-GPU-Datasheet ,
  accessed 2026-07-17), tier T2.
- H800 NVLink = 400 GB/s (vs H100 900 GB/s) — NVIDIA H800 datasheet
  interconnect row; corroborated by TrendForce/Reuters reporting and
  DeepSeek-V3 deployment write-ups, accessed 2026-07-17, tier T2/T3.
