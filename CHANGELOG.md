# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
semantic versioning. Any change that moves prediction numbers must ride a
version bump and pass the anchor re-verification protocol (Gate-2).

## [0.0.0] - 2026-07-17

### Gate 2 — credibility layer (validation, rack-scale hardware, USD costs)

- **Anchor validation study** (`validation/`): pre-registered predictions
  (commit-then-measure), temporal-holdout priors (`priors_pre2024.yaml`,
  4/11 ablation pairs kept), T1/T2/T3 provenance on every record, and a
  merged machine-readable `anchors.json`. Results, stated as measured:
  decision retrodiction **4 hit + 1 partial of 5** with all 3 negative
  controls flagged; Pythia intra-family ranking **Kendall τ = 1.0**;
  E2 absolute-loss **CI coverage 0/3 (FAIL)** published with a datamix
  attribution; E1 T2 training-efficiency anchors show a systematic
  ~10 pp-low bucket (35.2% / 73.6% errors, taxonomy included); 21 serving
  anchors pre-registered with the vLLM measurement protocol ready.
- **Rack-scale hardware targets**: `gb200_nvl72` (72-GPU NVLink-5 domain)
  and `h800` (NVLink 400 GB/s) as pure-additive system specs; EP up to 72
  priceable. Headline: a 1T-class MoE that spills on single-node H100
  puts EP=72 on the Pareto frontier on NVL72 (9.0 ms vs 33.9 ms TBT at
  EP=8), with domain-size sensitivity monotonic.
- **USD cost layer** (`--cost-usd`): optional `cost_estimate_usd` block
  (training_total / serving_per_1m_tokens / annual_serving_at_load) from
  hot-updatable, provenance-dated price books. Pure-add; default-off
  output is byte-identical; Pareto ranking untouched.
- **Case studies** (`docs/case_studies/`): DeepSeek-V3 MLA, Llama-3
  GQA-8 (group-size sweep 1–16), EP=72/NVL72, Mistral SWA — each with
  archived recipes and verbatim AC-output quotes.
- README: TL;DR, Validation, USD-cost, and Roadmap/Known-gaps sections;
  hardware table gains the two system targets. New tests: NVL72 specs,
  EP domain routing, EP=72 Pareto smoke, pricing (34 total; suite now
  830 passed, 1 protocol skip).

### Highlights (Waves 30–47, already on main)

- Search/pruning overhaul: canonical picker used everywhere (family rollup,
  joint optimum, budgets as bounds); production MoE axis with fine-grained
  experts and granularity targets; two-stage cheap-rank cap with local
  refinement around per-class Pareto leaders.
- Compressed attention (CSA / IndexShare / MSA) wired end-to-end: CLI flags,
  enumeration in MoE/hybrid generators, sparse prefill pricing, serializer,
  dedupe keys.
- Quality-model physics fixes: head penalties for sparse attention types,
  compression floors, context-aware recall risk, MLA context-payoff
  multiplier removed.
- Release verification: deterministic family tie-break (exact-tie ordering no
  longer depends on shard-merge order), reproducible regen scripts
  (PYTHONHASHSEED=0 re-exec guard), `max_training_cluster_gpus` hard cap
  threaded through constraints / feasibility / Pareto fingerprints.
- 134-check physics invariant audit (`scripts/probe_invariants.py`).

### Fixed

- Golden decision-matrix fixture (`tests/fixtures/golden_h100_decision_matrix.json`)
  regenerated: it predated the Waves 30–47 overhaul and was never re-baselined,
  so the release-gate regression (`AC_RUN_GOLDEN_MATRIX=1`) failed on 96 drift
  records. Triage: family flips / shape collapses at 70B–500B ← canonical
  picker + max-training-cluster cap; loss moves at 1M/2M-context cells ←
  context-aware quality physics; TBT shifts ← throughput repricing. Drift
  acceptance recorded in `golden_h100_decision_matrix.regen-history.txt`.
- Wheel installs: the modifier and delta-eval quickstarts failed in a clean
  `pip install` environment because top-level `configs/` is not packaged.
  The three reference base configs now ship inside the wheel
  (`ac/packaged_configs/`); a basename-conservative fallback resolves missing
  `configs/<name>.json` paths to the bundled copy with a stderr notice.
  `configs/` remains the frozen source of truth.

### Added

- CI (GitHub Actions): full pytest matrix (Python 3.10/3.11/3.12) + physics
  invariant audit; golden decision-matrix regression; quickstart snapshot
  determinism (two runs, byte-diff); ruff in report-only mode.
- `scripts/golden_snapshot.sh`: quickstart byte-diff regression harness
  (scrubs `generated_at`, `search_time_sec`, and the wall-clock report line).
- `tests/test_packaged_configs_sync.py`: byte-identity guard between
  `configs/` and the wheel-bundled copies.
- Repo hygiene: CHANGELOG, CONTRIBUTING, issue templates, CI badge.
- Packaging: PyPI distribution renamed to `archc` (import package stays
  `ac`); Demo and Changelog project URLs.

### Contract / behavior changes

- The Waves 30–47 overhaul intentionally moved golden-matrix cells (family
  picks, shapes, TBT, and long-context loss predictions) relative to 0.3.0
  snapshots. Per the prediction-freeze rule, that behavior change ships
  under this version bump rather than silently.
- No Gate-1 fix touched the frozen zone: `ac/quality_defaults.yaml`,
  `ac/hardware_specs/`, `ac/calibration/`, and `configs/` are byte-unchanged.

## [0.3.0] - 2026-07-08

- Initial public release candidate: greenfield / modifier / delta-eval
  compilers, stress diagnostics, auto-calibration, trust audit, six CLIs.
