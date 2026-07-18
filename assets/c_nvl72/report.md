# Task C — NVL72 rack-scale modeling: validation report (V1–V4)

- ac_version: 0.4.0
- quality_model_version: effective_capacity_v2
- git_commit: c170cda
- experiment_date: 2026-07-17
- agent wave: "gate2-wave1"
- hardware spec hashes (sha256): gb200_nvl72.json = `bed277b9…fe14`,
  h800.json = `3401fbfd…01b6`
- preregistration: `prereg.md` + `prereg_amendment1.md` +
  `prereg_amendment2.md` (all written before the official runs; criteria
  frozen in amendment 1 §"FINAL frozen criteria")

## 0. TL;DR

| Exp | Claim tested | Result | Verdict |
|---|---|---|---|
| V1(a) | 1T-class MoE on h100: infeasible/penalized | **0 of 148** feasible MoE rows fit 80 GB HBM (spill mandatory); **0 of 7** 1T-class rows fit | PASS (strict no-spill reading) / PARTIAL by the frozen letter (AC's continuous-spill model keeps them "feasible-but-penalized") |
| V1(b) | TBT(EP=72) h100 ≥ 3× NVL72 | measured **1.99×** (40.33 ms vs 20.24 ms) at the frozen operating point; 3.7× at the probe point | **FAIL by the frozen letter** — decomposition below |
| V1(c) | train TPS(EP=72) < TPS(EP=8) on h100 | 1392 vs 1762 tok/s (**−21.0%**) | PASS |
| V1(d) | all_to_all stress axis reported | h100 EP=72 = **0.240** vs NVL72 EP=72 = **0.015** (16.1×) | reported (self-damping caveat, amendment 1) |
| V2(a) | EP=72 feasible + on NVL72 Pareto frontier | **11 EP=72 frontier rows**, incl. 1.128T @ 59 GB/GPU, TBT 12.9 ms | PASS |
| V2(b) | best EP=72 TBT < best EP=8 TBT on NVL72 | search: **9.0 vs 33.9 ms**; mechanism: **20.24 vs 47.70 ms** | PASS |
| V2(c) | train TPS(EP=72) ≥ TPS(EP=8) on NVL72 | 4098 vs 3748 tok/s (**+9.4%**) | PASS |
| V2b | 384 experts: EP=72 correctly absent | evaluated EPs = {8,16,32}; best = EP=32 (10.8 ms) | PASS |
| V3 | optimum moves monotonically with domain size | argmax train-TPS EP: **8 → 16 → 32 → 32** for domain 8/16/32/72 | PASS |
| V4 | golden snapshot zero-regression | `diff -r out/golden out/c_check` **empty** | PASS |

**Bottom line:** the plan-level headline (第二道.md §6) holds — the same
1T-class MoE is undeployable-without-spill on h100 (and pays a −21%
training-TPS all-to-all tax at EP=72), while on gb200_nvl72 EP=72 enters
the Pareto frontier with the best TBT of any EP. One frozen sub-criterion
(V1(b), the ≥3× decode-TBT ratio) did not hold at the frozen operating
point; it was miscalibrated against decode-phase all-to-all share, and
the failure is analyzed in §3.2 — it does not change the headline.

## 1. What was built (all pure-additive)

- `ac/hardware_specs/gb200_nvl72.json` — B200 chip params reused
  (`"chip": "b200"`), system block: `nvlink_domain_size=72`,
  `intra_domain_bandwidth_gbps=1800`, `inter_domain_bandwidth_gbps=50`,
  `gpus_per_rack=72`. Every number carries provenance (URL + access date
  + tier) in `notes`.
- `ac/hardware_specs/h800.json` — verbatim copy of `h100_sxm.json` except
  `accelerator_name` and `interconnect.intra_node_bw_gb_s: 900 → 400`.
- Loader (`throughput_model.HardwareConfig.from_json`): optional `system`
  block parsed onto legacy fields; new optional `gpus_per_rack` / `chip`
  attributes default to `None`; absent block → single-node semantics,
  old specs byte-identical (pinned by `test_nvl72_system_spec.py`).
- All-to-all routing: **no new formula** — the hierarchical branch
  (`ep <= domain` → intra; `ep > domain` → split intra/inter, concurrent
  legs, `max(t_intra, t_inter)`) already existed (Wave 19 P0-1). Task C
  wires the new targets into it and pins both legs numerically
  (`test_ep_domain_routing.py`).
- Registrations (each is a new-key-only addition; no existing target's
  code path or constant changed): `load_hardware` mapping;
  `VALID_HARDWARE` in cli_compile / cli_delta_eval / cli_stress;
  `lattice_engine.HARDWARE` (+B200/H100 tile specs) and `NVLINK_DOMAIN`
  (gb200_nvl72→72, h800→8); `get_precision_configs_for_hardware`
  (nvl72 = b200 set, h800 = h100 set); `PRECISION_HARDWARE_SUPPORT`;
  `_get_hbm_gb` (192/80); calibration alias resolution in
  `load_calibration` (gb200_nvl72→b200, h800→h100 measured tables; the
  alias map previously only silenced the warning — now it also resolves
  the lookup, which is a no-op for every pre-existing alias because no
  trainium calibration files exist).
- cli_matrix18b has no hardware enum (free-form list) — no change needed.

## 2. Protocol (as frozen)

- Grid: `--allow-moe --allow-mla --moe-n-experts 288 --moe-top-k 8
  --moe-granularity 1.0,0.5 --max-total-params-b 1200
  --ep-options 8,16,32,72 --training-cluster-gpus 2304 --tp 1 --pp 1
  --params 32 --tokens 15 --context 8192 --serving-batch 64
  --max-candidates 200`
- Erratum (fixed post-Wave-3): this section originally printed
  `--tokens 15000`; the generating harness actually used 15T tokens
  (`training_tokens=int(15e12)`). Verified empirically: re-evaluating the
  archived 723B EP=72 frontier row at 15T reproduces loss 1.9957 /
  TBT 17.33 exactly (at 15000T the loss would be 1.9055). TBT is
  token-independent, so no headline number changes.
- Deviations from plan §2 (all pre-registered in prereg.md /
  amendments): n_experts 288 not 384 (enumerator requires
  `n_experts % ep == 0`; 384 % 72 ≠ 0 — V2b runs the true 384 shape
  separately); `--training-cluster-gpus 2304` (default `--dp 8` would
  filter EP>8 out of the search on both targets); cap 1200 not 1100
  (keeps the ~1.15T g=1.0 candidates); serving batch 64 + `--allow-mla`
  (batch 256 without MLA is a pathological 524 GB/GPU KV regime — the
  run is archived at `runs/v2_nvl72_pathological_batch256/`).

## 3. Results

### 3.1 Mechanism table (fixed 1.08T-total / 45.7B-active arch, 288×2816
experts top-8, fp8, batch 64; `runs/mechanism_table.json`)

| hardware | EP | TBT (ms) | train TPS | mem/GPU (GB) | spill |
|---|---|---|---|---|---|
| h100 | 8 | 244.45 | 1761.6 | 154.6 | nvlink |
| h100 | 16 | 94.99 | 1555.0 | 92.7 | nvlink |
| h100 | 32 | 50.73 | 1446.4 | 61.7 | fits |
| h100 | 72 | 40.33 | 1392.3 | 44.5 | fits |
| gb200_nvl72 | 8 | 47.70 | 3747.5 | 154.6 | fits |
| gb200_nvl72 | 16 | 32.26 | 3915.0 | 92.7 | fits |
| gb200_nvl72 | 32 | 24.53 | 4100.3 | 61.7 | fits |
| gb200_nvl72 | 72 | 20.24 | 4098.4 | 44.5 | fits |

Reading: on h100 the 1.08T model cannot be served at EP ≤ 16 without
spill (154.6 / 92.7 GB > 80 GB) — big EP is mandatory, and at EP=72 it
pays a −21% training-TPS all-to-all tax (vs +9.4% *gain* on NVL72). On
NVL72 every EP fits and TBT improves monotonically with EP.

### 3.2 The V1(b) failure, decomposed (honest reporting)

Frozen criterion: TBT(EP=72) on h100 ≥ 3× TBT(EP=72) on NVL72.
Measured 40.33 / 20.24 = **1.99×**.

Decomposition of the cross-hardware gap at decode batch 64:
- decode all-to-all share on h100 EP=72: **0.42%** of layer time (the
  36× intra/inter bandwidth ratio acts on a term that is tiny at decode
  volumes) — so the interconnect contributes almost nothing to the ratio
  at this operating point;
- the 1.99× is compute+HBM generation gap (H100 495 vs B200 1125 TF
  bf16, 3.35 vs 8.0 TB/s HBM);
- at the archived probe point (batch 256, 530B shape, where h100 EP=72
  also spilled) the same ratio was 3.71× — i.e. the ≥3× expectation was
  calibrated to a spill-dominated regime, not to the all-to-all term.

Where the interconnect penalty *does* show up decisively: training TPS
(h100 EP=72 −21% vs EP=8; NVL72 EP=72 +9.4%), the stress all_to_all
axis (0.240 vs 0.015, 16.1×), and the no-spill deployability of the
whole model class (§3.3). The frozen criterion picked the wrong phase
(decode) for the ≥3× bar; the plan's qualitative claim ("受罚") stands
on the other three instruments.

### 3.3 Search-level evidence (`runs/search_h100.json`,
`runs/search_nvl72.json`; in-process optimize(), 296 candidates each)

h100:
- 148 feasible MoE rows; **0 fit 80 GB HBM** (spill mandatory for the
  whole class at this workload). 7 1T-class (≥1000B) rows, **0 fit**.
- Best feasible TBT by EP: 8→160.6 ms, 16→79.4, 32→31.5, 72→45.9 —
  big EP is h100's *only* non-spill path, and it is uniformly ~4–5×
  slower than the same shapes on NVL72 (same-shape contrast, e.g.
  d6144/L11/ed16384: EP=32 88.3 ms h100 vs 20.0 ms NVL72).

gb200_nvl72:
- Frontier EP composition includes **EP=72 (11 rows)**; best-by-EP TBT:
  8→33.9, 16→16.3, 32→10.3, **72→9.0 ms** (best overall).
- Frontier EP=72 highlights: 1.128T @ 59 GB/GPU TBT 12.9 ms;
  723B @ 121 GB loss 1.996 TBT 17.3 ms.
- All 8 feasible 1T-class rows fit 192 GB HBM, EPs {8: 3, 16: 2, 32: 2,
  **72: 1**}.

### 3.4 V2b — true K2 shape (384 experts, `runs/v2b_384_experts.json`)

EP=72 correctly absent (384 % 72 ≠ 0); evaluated/frontier EPs =
{8,16,32}; best feasible: EP=32, TBT 10.8 ms. This documents the
padding nuance honestly: AC will not fabricate a fractional
experts-per-rank layout; with K2's actual expert count the best
enumerable EP on NVL72 is 32.

### 3.5 V3 — domain sensitivity (`runs/v3_sensitivity.json`; same fixed
arch, `nvlink_domain_size` overridden via AC_HARDWARE_SPEC_DIR copies)

| domain | argmax train-TPS EP | train TPS by EP (8/16/32/72) |
|---|---|---|
| 8  | **8**  | 3747 / 2834 / 2493 / 2336 |
| 16 | **16** | 3747 / 3915 / 2884 / 2476 |
| 32 | **32** | 3747 / 3915 / 4005 / 2833 |
| 72 | **32** (72 within 0.05%) | 3747 / 3915 / 4100 / 4098 |

The training-optimal EP tracks the NVLink domain size exactly and
saturates — monotone non-decreasing (8→16→32→32): the domain size, not
a coincidence, drives the conclusion. (Decode-TBT argmin is 72 at every
domain — memory-driven, as predicted in amendment 1; the domain signal
lives in the all-to-all-heavy training phase.)

### 3.6 V4 — zero-regression

- `scripts/golden_snapshot.sh out/c_check && diff -r out/golden
  out/c_check` → **empty** (15 files, byte-identical).
- Full suite: `python -m pytest tests/ -q` → **830 passed, 1 skipped**
  (796 baseline + 22 new Task-C tests + 12 tests from another agent's
  untracked `tests/test_pricing.py` that happened to be present in this
  shared checkout; 0 failures).
- `git diff --stat` (tracked files): exactly 7 modified files —
  `ac/throughput_model.py` (loader system block + spec mapping +
  calibration alias resolution), `ac/cli_compile.py`,
  `ac/cli_delta_eval.py`, `ac/cli_stress.py` (enum registration),
  `ac/lattice_engine.py` (tile spec + NVLINK_DOMAIN entries),
  `ac/optimizer.py` (precision configs + HBM map), `ac/penalties.py`
  (precision support sets). New untracked: the 2 spec files, 3 test
  files, `validation/c_nvl72/*`. **No** existing spec field, constant,
  formula, or test assertion touched; `ac/quality_defaults.yaml`,
  `ac/calibration/`, `configs/` untouched.

## 4. Fact provenance (spec `notes` fields carry the same)

| fact | value | source (accessed 2026-07-17) | tier |
|---|---|---|---|
| NVL72 = 72 GPUs + 36 Grace, one NVLink domain | — | nvidia.com/en-us/data-center/gb200-nvl72/ | T2 |
| NVLink 5 per-GPU bandwidth | 1.8 TB/s (130 TB/s rack) | same NVIDIA page | T2 |
| Scale-out per GPU | 400 Gb/s = **50 GB/s** (ConnectX-7 NDR) | NVIDIA partner spec sheets (spheron.network/gpu-rental/gb200/, pcbstore NVL72 listing) | T3 |
| B200 chip params (192 GB HBM3e, 8 TB/s, peaks) | — | NVIDIA DGX B200 datasheet (as already cited in b200.json) | T2 |
| H800 SXM = H100 silicon (80 GB HBM3, 3.35 TB/s, 989 BF16 dense) | — | NVIDIA H800 GPU datasheet (mirrored), scribd.com/document/777167019 | T2 |
| H800 NVLink | **400 GB/s** (vs 900 on H100) | NVIDIA H800 datasheet interconnect row; corroborated by TrendForce/Reuters and DeepSeek-V3 (arXiv:2412.19437) | T2/T3 |

**Unit deviation (documented in prereg):** the plan's example value
`inter_domain_bandwidth_gbps: 400` is the datasheet figure in Gb/s; AC
interconnect fields are GB/s everywhere, so the spec stores **50**.
Storing 400 would have made NVL72 cross-rack fabric 4× faster than
DGX-B200's existing 100 GB/s — physically wrong.

## 5. Acceptance checklist mapping (plan §5)

- [x] `gb200_nvl72` and `h800` run as `--hardware` targets through
  compile (`runs/v1_h100`, `runs/v2_nvl72`, CLI smokes in
  `tests/test_ep72_pareto.py`) and delta-eval
  (`runs/acceptance/delta_gqa_{nvl72,h800}`, exit 0). ac-stress also
  verified manually (`--hw gb200_nvl72 --ep 72`).
- [x] V1–V4 run, results in `validation/c_nvl72/` (this file +
  `runs/*.json`, `runs/v1_h100`, `runs/v2_nvl72`,
  `runs/v2_nvl72_pathological_batch256`, `runs/acceptance`,
  `runs/probe_mechanism.txt`, `runs/run_experiments.py`).
- [x] EP=72 contrast holds: h100 = mandatory spill + −21% training tax;
  NVL72 = frontier with best TBT (§3.1–3.3). One frozen sub-criterion
  (V1(b) decode-TBT ≥3×) failed and is decomposed in §3.2.
- [x] New tests in CI (3 files, 22 tests, green); golden diff empty;
  whitelist respected (§3.6).
- [x] All spec numbers carry provenance (§4).
- [x] README rows: `readme_rows_snippet.md` (orchestrator applies).

## 6. Known limitations / honesty notes

- V1(a) is a PASS only under the strict no-spill deployability reading.
  AC's Wave-2a continuous-spill model never says a hard "infeasible" for
  serving — it prices the spill. The plan's "被判不可行" wording predates
  that design; both readings are reported with numbers.
- Decode-phase all-to-all at batch 64 is a small term for this shape;
  the interconnect domain's signature is strongest in training /
  large-batch prefill economics. Future anchors should read the penalty
  off training TPS or the stress axis, not decode TBT.
- gb200_nvl72/h800 predictions reuse the b200/h100 *measured*
  calibration tables (same silicon) — bandwidth effects come from the
  spec JSON. Lab traces on the actual systems should override via
  ac-auto-calibrate.
- Rack power/cooling/failure rates intentionally unmodeled (plan §1.4).
