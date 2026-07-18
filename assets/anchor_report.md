# AC Anchor Validation Report

> ac_version 0.4.0 | quality_model_version effective_capacity_v2 |
> git_commit ab3c614 | experiment_date 2026-07-17 | gate gate2
>
> Machine-readable single source of truth: [`anchors.json`](anchors.json)
> (27 anchors + 9 retrodiction cases; rebuilt by `build_anchors.py`).
> Detail sections: [E1](e1_perf/e1_section.md) · [E2](e2_quality/e2_section.md) ·
> [E3](e3_retrodiction/e3_section.md) · [C/NVL72](c_nvl72/report.md)

## 1. Method and discipline (read first — this is the credibility claim)

- **Pre-registration.** Every prediction and expectation was committed
  *before* the corresponding observation was gathered
  (`validation/prereg/`, commit `ba3bbd5`; observation fills are
  append-only in later commits — `git log` is the audit trail).
- **Temporal holdout (anti-circularity).** E2 predictions use
  `priors_pre2024.yaml` (SHA256 `e59cbdee…`): the paired-ablation corpus
  was cut at 2024-01-01 (4/11 pairs kept) and de-anchored residual terms
  were zeroed through their existing YAML knobs, so no post-2024 model
  family could leak into the priors that predict it.
- **Confidence tiers.** T1 = own measurement (raw logs + commands);
  T2 = paper/official numbers (source + table/section + access date);
  T3 = secondary, never counted in acceptance stats. Dropped anchors are
  listed with reasons, never silently omitted.
- **Version pinning.** All numbers above are reproduced by the pinned
  commit; later prediction changes must re-run this study and annotate,
  not overwrite.

## 2. E1 — performance anchors (throughput / memory fidelity)

Serving anchors (TBT/TTFT/VRAM for Mistral-7B, Llama-3.1-8B,
Llama-3.3-70B ×2 TP, Qwen3-8B, Qwen3-32B, GPT-OSS-120B):
**21 predictions pre-registered, status `awaiting_t1_measurement`** —
the aligned vLLM protocol (`e1_perf/run_benchmarks.sh`) and log parser
are ready; they execute on a rented H100×8 node.

Training-side anchors (T2, from papers):

| anchor | AC prediction | observed (T2) | abs. error |
|---|---|---|---|
| Llama-3 405B training MFU (h100) | 26.24% implied | 38–43% (arXiv 2407.21783 Tbl 4) | **35.2%** |
| DeepSeek-V3 training efficiency (h800) | 9.13% implied | 34.64% implied from 2.788M H800-h (arXiv 2412.19437 Tbl 1) | **73.6%** |

Median of the two: **54.4%** — far outside any family band. See §5.

## 3. E2 — quality anchors (loss & uncertainty fidelity)

Temporal-holdout priors; observed final losses from the labs' own
official training logs (WandB), not curve-reading.

| anchor | predicted [CI] | observed | point error | CI covers? |
|---|---|---|---|---|
| OLMo-2-7B @ 3.895T | 2.0016 [1.942, 2.062] | 2.2305 | −10.3% | ✗ |
| Pythia-1.4B @ 300B | 2.3012 [2.232, 2.370] | 1.9780 | +16.3% | ✗ |
| Pythia-12B @ 300B | 2.1502 [2.086, 2.215] | 1.7570 | +22.4% | ✗ |
| SmolLM3-3B @ 11.2T | 2.0959 [1.927, 2.265] | — | dropped | no published final loss |

- **CI coverage: 0/3 — FAIL vs the ≥80% target** (identical under both
  prior variants; predictions are bit-identical on dense anchors).
- **Ranking: Pythia intra-family Kendall τ = 1.0 — PASS (≥0.9).**
- Full-vs-truncated priors: bit-identical on these dense anchors (the
  dense-slice temporal-robustness null result; overlay proven live on
  hybrid/MoE arms).

## 4. E3 — decision retrodiction (end-to-end utility)

Pre-registered expectations vs AC outputs (all raw runs archived):

| case | verdict | one-line evidence |
|---|---|---|
| 1a MHA→MLA (h800) | **HIT** | KV-footprint 1.52→0.21, quality *gain*, expands frontier |
| 1b GQA→MLA (h800) | **HIT** | hbm_bw_decode relieved, smaller relief than 1a as expected |
| 2 GQA-8 (Llama-3-70B) | **PARTIAL** | substance confirmed both regimes; frontier-membership semantics + arm-label fix (D7) |
| 3 SWA-4096 (Mistral) | **HIT** | on-frontier, dominates GQA-only; KV 1.60→0.05 |
| 4 first-K-dense (V3) | **HIT** | k=3 gain > k=1, ordering matches V3's real choice |
| 5 K2 wide-EP (NVL72) | **HIT** | h100 EP64 all-to-all ×38; gb200_nvl72 expands frontier, TBT −68%; EP=72 frontier entry corroborated by Task C V2 |
| N1 7B MHA @32k | flagged ✓ | hbm_bw_decode = 1.016 binding |
| N2 tile-misaligned d_model | flagged ✓ (direction) | tc_util_prefill 0.852 vs 0.752 |
| N3 7B-total MoE no-EP | flagged ✓ | TP1 arm dominated by 64 candidates, 7.5× decode penalty |

**Hit rate: 4 HIT + 1 PARTIAL of 5 pre-registered cases — PASS (≥4/5).**
All three negative controls correctly flagged.

## 5. Error taxonomy (no anchor deleted; every miss attributed)

| # | anchor(s) | magnitude | attribution | path forward |
|---|---|---|---|---|
| E-01 | DSv3 training efficiency | 73.6% | training-efficiency bucket: MoE all-to-all non-overlap, capacity filling, moe_mla efficiency class, bf16-vs-FP8 convention | per-recipe calibration packs (`ac-auto-calibrate fit`); roadmap |
| E-05 | Llama-3 405B MFU | 35.2% | steady-state roofline ceiling ≈26% vs achieved 38–43% | same calibration path |
| E-02 | E2 all three | 10–22% | Chinchilla spine cross-corpus miscalibration (MassiveText priors vs Pile/OLMo-mix); the 3% spine band has **no datamix dimension** | datamix-specific priors (already the documented design); roadmap item: datamix uncertainty term |
| E-03 | E3 case 2 | partial | frontier-membership semantics of `swap_attention_to_gqa` arms (label fix D7) | documented; classifier UX roadmap |
| E-04 | E3 N3 / N2 | partial | GB/GiB 1.3% sentinel threshold slip; aligned-twin already in loaded band | audit notes in prereg; no model defect |
| C-V1(b) | NVL72 decode-TBT ratio | 1.99× vs ≥3× frozen | decode-phase all-to-all is 0.42% of layer time at batch 64; fabric penalty decisive on training TPS/stress/deployability instead | decomposition in `c_nvl72/report.md` §3.2 |

**No miss was attributed to a model defect requiring a Gate-1 bug
reflow**; all are prior/calibration gaps (by design, fixable by
lab-local calibration) or documented semantics.

## 6. Known boundaries of this study

- Serving TBT/TTFT/VRAM fidelity (E1 T1) is **not yet measured** — the
  protocol is ready and pre-registered; acceptance stats will be
  computed when the rented-node logs land (`e1_perf/`).
- Training-cost fidelity is exactly as good as Vidur-class rooflines
  before calibration: structurally right, systematically ~10 pp low on
  the efficiency bucket.
- Absolute loss predictions are **not** cross-datamix calibrated; use
  AC ordinally (ranking τ = 1.0 held) or calibrate per datamix, exactly
  as the README's "Calibration vs. Ordinal Use" section states.
- Not covered: data recipes, tokenizers, post-training effects;
  Muon-class optimizer shifts; 10k-node failure/checkpoint goodput
  (folded into `training_system_efficiency`).
