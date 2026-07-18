#!/usr/bin/env python3
# validation/e2_quality/build_priors_pre2024.py
# ----------------------------------------------------------------------------
# Gate 2 wave-1 header:
#   ac_version:            0.4.0
#   quality_model_version: effective_capacity_v2
#   git_commit:            c170cda
#   experiment_date:       2026-07-17
#   agent_wave:            gate2-wave1
#
# Temporal-holdout prior builder for the E2 quality-anchor study
# (Kimi_Agent_GitHub评审/第二道-A-锚点验证研究.md §3.2).
#
# What it does, deterministically (fixed inputs -> byte-identical outputs):
#   1. Loads the shipped paired-ablation corpus
#      (tests/fixtures/public_ablation_pairs_v1.json). That corpus carries NO
#      publication-date metadata, so dates come from the curated annotation
#      file validation/e2_quality/pair_dates.yaml (arXiv v1 dates, tier T2;
#      every corpus pair MUST have an entry and vice versa — a mismatch fails
#      the build rather than silently dropping evidence).
#   2. Keeps only pairs with publication_date strictly before 2024-01-01 and
#      writes the truncated corpus public_ablation_pairs_pre2024.json.
#   3. Runs AC's own fit-pairs machinery (ac.ablation_fit.run_fit_pairs) on the
#      FULL corpus and on the TRUNCATED corpus, into pair_fit_full/ and
#      pair_fit_pre2024/.
#   4. Emits the temporal-holdout prior overlay priors_pre2024.yaml, consumed
#      via AC_QUALITY_DEFAULTS=validation/e2_quality/priors_pre2024.yaml.
#
# Refit design (why the overlay looks the way it does — see prereg §Design):
#   AC's stock quality constants are hand priors; the fit-pairs corpus is the
#   reproducibility/coverage layer on top of them. The corpus format constrains
#   ten residual TERMS; it has no knob-level provenance (a pair constrains a
#   term, not an individual weight), and the machinery's fitted per-term scales
#   are flagged by the fitter itself as cross-paper confounded / directional
#   (n<=2 pairs per term). Applying such scales to individual weights would
#   mix a second experimental factor (magnitude recalibration) into the
#   temporal-holdout factor, so fitted scales are recorded as DIAGNOSTICS in
#   split_summary.json, not applied.
#   What the temporal split DOES change mechanically: terms whose anchoring
#   pairs are ALL post-cutoff become UNCOVERED in the truncated fit. For those
#   terms the overlay zeroes the term's effect through the term's OWN existing
#   YAML knob — the same "disabled until evidence exists" encoding AC already
#   uses (cf. effective_data.overfit_penalty_weight: 0.0 and the
#   downstream_head / data_quality blocks in ac/quality_defaults.yaml):
#     state_residual     -> min_delta: 0.0, max_delta: 0.0  (clamp to zero; the
#                           term has no `enabled` consumer, the [min,max] clamp
#                           at quality_model._state_residual is the only clean
#                           zeroing knob)
#     effective_capacity -> enabled: false  (consumed at
#                           quality_model._effective_capacity_transform;
#                           recovers N_eff = N_active)
#     mtp_residual       -> bonus_per_depth: 0.0  (the block has no `enabled`
#                           consumer; zeroing the per-depth bonus zeroes the
#                           term)
#   Everything else is left at stock constants: terms that KEEP >=1 pre-2024
#   pair (architecture_residual, attention_locality, context_utility) and
#   terms uncovered in BOTH corpora (moe_residual, precision_residual,
#   vocab_residual, large_shape_stability_prior, risk_residual) are identical
#   across the two prior variants, which isolates the temporal effect.
#
# Usage:
#   python validation/e2_quality/build_priors_pre2024.py
# ----------------------------------------------------------------------------
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml  # PyYAML is a hard AC dependency (load_quality_constants)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ac.ablation_fit import KNOWN_TERMS, run_fit_pairs  # noqa: E402

CORPUS_PATH = REPO_ROOT / "tests" / "fixtures" / "public_ablation_pairs_v1.json"
DATES_PATH = HERE / "pair_dates.yaml"
FILTERED_CORPUS_PATH = HERE / "public_ablation_pairs_pre2024.json"
PRIORS_PATH = HERE / "priors_pre2024.yaml"
SUMMARY_PATH = HERE / "split_summary.json"
FIT_FULL_DIR = HERE / "pair_fit_full"
FIT_PRE2024_DIR = HERE / "pair_fit_pre2024"

# Term -> zeroing overlay for terms that lose every anchoring pair.
# Each entry is the YAML subtree deep-merged over DEFAULT_QUALITY_CONSTANTS.
DEANCHOR_OVERLAYS = {
    "state_residual": {
        "min_delta": 0.0,
        "max_delta": 0.0,
    },
    "effective_capacity": {
        "enabled": False,
    },
    "mtp_residual": {
        "bonus_per_depth": 0.0,
    },
}

DEANCHOR_RATIONALE = {
    "state_residual": (
        "All three anchoring pairs are post-2023 (Waleffe 2024-06-12 x2, "
        "Jamba 2024-03-28). The term has no `enabled` consumer; clamping "
        "[min_delta, max_delta] to 0 zeroes its value for hybrid/state "
        "architectures. Dense anchors are unaffected (term is n/a for them)."
    ),
    "effective_capacity": (
        "Sole anchoring pair (DeepSeekMoE, 2024-01-11) is post-cutoff. "
        "`enabled: false` is consumed by _effective_capacity_transform and "
        "recovers N_eff = N_active. Dense anchors are unaffected (identity "
        "transform either way)."
    ),
    "mtp_residual": (
        "Sole anchoring pair (DeepSeek-V3, 2024-12-27) is post-cutoff. The "
        "block has no `enabled` consumer; bonus_per_depth: 0.0 zeroes the "
        "bonus. Dense non-MTP anchors are unaffected."
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_dates() -> dict:
    with DATES_PATH.open() as f:
        doc = yaml.safe_load(f)
    assert isinstance(doc, dict) and "pairs" in doc and "cutoff" in doc, (
        f"{DATES_PATH} must contain top-level 'pairs' and 'cutoff'")
    return doc


def main() -> int:
    cutoff = str(_load_dates()["cutoff"])
    dates_doc = _load_dates()
    pair_dates = {
        pid: str(entry["publication_date"])
        for pid, entry in dates_doc["pairs"].items()
    }

    with CORPUS_PATH.open() as f:
        corpus = json.load(f)
    pairs = corpus["pairs"]
    corpus_ids = sorted(p["id"] for p in pairs)
    date_ids = sorted(pair_dates)
    if corpus_ids != date_ids:
        missing = sorted(set(corpus_ids) - set(date_ids))
        extra = sorted(set(date_ids) - set(corpus_ids))
        raise SystemExit(
            "pair_dates.yaml does not match the corpus 1:1 — refusing to "
            f"build (missing dates for {missing}; stale entries {extra})."
        )

    kept, dropped = [], []
    for p in pairs:
        d = pair_dates[p["id"]]
        (kept if d < cutoff else dropped).append((p["id"], d, p["term"]))

    truncated = {
        "schema_version": corpus["schema_version"],
        "notes": [
            "TEMPORAL-HOLDOUT SUBSET of tests/fixtures/"
            "public_ablation_pairs_v1.json.",
            f"Kept only pairs with publication_date strictly before {cutoff}",
            "(earliest public availability = arXiv v1; see pair_dates.yaml).",
            "Generated by validation/e2_quality/build_priors_pre2024.py ",
            "for the Gate-2 E2 quality-anchor study (gate2-wave1, ",
            "ac_version 0.4.0, quality_model_version effective_capacity_v2, ",
            "git_commit c170cda, experiment_date 2026-07-17).",
        ],
        "pairs": [p for p in pairs if pair_dates[p["id"]] < cutoff],
    }
    with FILTERED_CORPUS_PATH.open("w") as f:
        json.dump(truncated, f, indent=2, sort_keys=True)
        f.write("\n")

    # Run AC's own fit-pairs machinery on both corpora.
    full_fit = run_fit_pairs(str(CORPUS_PATH), str(FIT_FULL_DIR))
    trunc_fit = run_fit_pairs(str(FILTERED_CORPUS_PATH), str(FIT_PRE2024_DIR))

    def _n_by_term(fit_payload):
        return {
            tf["term"]: tf["n_pairs"] for tf in fit_payload["term_fits"]
        }

    full_n = _n_by_term(full_fit)
    trunc_n = _n_by_term(trunc_fit)
    deanchored = sorted(
        t for t in KNOWN_TERMS if full_n.get(t, 0) > 0 and trunc_n.get(t, 0) == 0
    )
    unknown_deanchored = sorted(set(deanchored) - set(DEANCHOR_OVERLAYS))
    if unknown_deanchored:
        raise SystemExit(
            f"terms lost all anchors but have no zeroing overlay defined: "
            f"{unknown_deanchored} — extend DEANCHOR_OVERLAYS deliberately."
        )

    def _scale(fit_payload, term):
        for tf in fit_payload["term_fits"]:
            if tf["term"] == term:
                return tf["scale"], tf["bias_pct"], tf["rms_pct"]
        return None, 0.0, 0.0

    # Render the prior overlay. Hand-rendered (not yaml.dump) so comments and
    # key order are stable and the file is reviewable line-by-line.
    full_sha = _sha256(CORPUS_PATH)
    trunc_sha = _sha256(FILTERED_CORPUS_PATH)
    lines = [
        "# validation/e2_quality/priors_pre2024.yaml",
        "# ----------------------------------------------------------------------",
        "# Gate 2 wave-1 header:",
        "#   ac_version:            0.4.0",
        "#   quality_model_version: effective_capacity_v2",
        "#   git_commit:            c170cda",
        "#   experiment_date:       2026-07-17",
        "#   agent_wave:            gate2-wave1",
        "#",
        "# TEMPORAL-HOLDOUT quality prior for the E2 quality-anchor study.",
        "# Consumed via: AC_QUALITY_DEFAULTS=<repo>/validation/e2_quality/"
        "priors_pre2024.yaml",
        "# Deep-merged over ac.quality_model.DEFAULT_QUALITY_CONSTANTS by",
        "# load_quality_constants(); only the keys below differ from stock.",
        "#",
        f"# Split rule: keep pairs with publication_date < {cutoff} (arXiv v1;",
        "# earliest public availability). Dates: pair_dates.yaml (tier T2).",
        f"# Kept {len(kept)} of {len(pairs)} pairs; dropped {len(dropped)}.",
        "# De-anchored terms (pairs>0 in the full corpus, 0 pre-cutoff) are",
        "# zeroed through their own existing knobs — see build_priors_pre2024.py",
        "# for the design rationale. Fitted per-term scales are NOT applied",
        "# (cross-paper confounded, n<=2; recorded in split_summary.json).",
        "#",
        "# Generated deterministically by build_priors_pre2024.py; do not edit",
        "# by hand — edit the script and re-run.",
        "",
    ]
    for term in deanchored:
        overlay = DEANCHOR_OVERLAYS[term]
        lines.append(f"# --- de-anchored: {term} "
                     f"(full-corpus n={full_n[term]}, pre-2024 n=0) ---")
        for i, rationale_line in enumerate(
                _wrap(DEANCHOR_RATIONALE[term], 74)):
            lines.append(f"# {rationale_line}")
        lines.append(f"{term}:")
        for k, v in overlay.items():
            v_yaml = "false" if v is False else ("true" if v is True else v)
            lines.append(f"  {k}: {v_yaml}")
        lines.append("")

    lines.append("")
    audit = {
        "temporal_holdout": {
            "cutoff": cutoff,
            "dating_rule": (
                "publication_date = earliest public availability (arXiv v1, "
                "UTC); pair of two papers dated at the later of the two"
            ),
            "date_annotations": "validation/e2_quality/pair_dates.yaml",
            "corpus_full": "tests/fixtures/public_ablation_pairs_v1.json",
            "corpus_full_sha256": full_sha,
            "corpus_pre2024": (
                "validation/e2_quality/public_ablation_pairs_pre2024.json"
            ),
            "corpus_pre2024_sha256": trunc_sha,
            "pairs_total": len(pairs),
            "pairs_kept": [pid for pid, _, _ in kept],
            "pairs_dropped": [pid for pid, _, _ in dropped],
            "deanchored_terms": deanchored,
            "fitted_scales_applied": False,
            "generator": "validation/e2_quality/build_priors_pre2024.py",
            "ac_version": "0.4.0",
            "quality_model_version": "effective_capacity_v2",
            "git_commit": "c170cda",
            "experiment_date": "2026-07-17",
            "agent_wave": "gate2-wave1",
        }
    }
    rendered = "\n".join(lines) + yaml.safe_dump(
        audit, sort_keys=False, default_flow_style=False)
    PRIORS_PATH.write_text(rendered)

    priors_sha = _sha256(PRIORS_PATH)
    summary = {
        "_header": {
            "ac_version": "0.4.0",
            "quality_model_version": "effective_capacity_v2",
            "git_commit": "c170cda",
            "experiment_date": "2026-07-17",
            "agent_wave": "gate2-wave1",
        },
        "cutoff": cutoff,
        "pairs_total": len(pairs),
        "pairs_kept_n": len(kept),
        "pairs_dropped_n": len(dropped),
        "kept": [
            {"id": pid, "publication_date": d, "term": t}
            for pid, d, t in kept
        ],
        "dropped": [
            {"id": pid, "publication_date": d, "term": t}
            for pid, d, t in dropped
        ],
        "term_pair_counts": {
            t: {"full": full_n.get(t, 0), "pre2024": trunc_n.get(t, 0)}
            for t in KNOWN_TERMS
        },
        "deanchored_terms": deanchored,
        "fitted_scales_diagnostic": {
            t: {
                "full": _scale(full_fit, t),
                "pre2024": _scale(trunc_fit, t),
            }
            for t in KNOWN_TERMS
            if full_n.get(t, 0) > 0 or trunc_n.get(t, 0) > 0
        },
        "artifacts_sha256": {
            "corpus_full": full_sha,
            "corpus_pre2024": trunc_sha,
            "priors_pre2024_yaml": priors_sha,
            "pair_fit_full_json": _sha256(FIT_FULL_DIR / "pair_fit.json"),
            "pair_fit_pre2024_json": _sha256(FIT_PRE2024_DIR / "pair_fit.json"),
        },
        "priors_path": "validation/e2_quality/priors_pre2024.yaml",
    }
    with SUMMARY_PATH.open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"cutoff: {cutoff}")
    print(f"pairs: {len(pairs)} total -> {len(kept)} kept, {len(dropped)} dropped")
    print("kept:")
    for pid, d, t in kept:
        print(f"  {pid} ({d}, {t})")
    print("dropped:")
    for pid, d, t in dropped:
        print(f"  {pid} ({d}, {t})")
    print(f"de-anchored terms: {deanchored}")
    print(f"priors: {PRIORS_PATH}")
    print(f"priors SHA256: {priors_sha}")
    return 0


def _wrap(text: str, width: int):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
