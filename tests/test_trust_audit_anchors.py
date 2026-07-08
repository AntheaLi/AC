"""Trust-audit CLI behavior, public-anchor registry, and family bias.

(Pins from Waves 20, 21, and 29.)

  * `ac-trust-audit --out DIR` implies `--all`; family_bias.json is
    schema v2 (loss + TBT/TTFT/mem per family), and the shipped table
    matches.
  * Anchor workloads are honest: MoE anchors whose published benchmarks
    ran TP-only default to EP=1 (not a fabricated EP=2); GPT-OSS-120B's
    registry workload is the reconcilable tp=2 recipe.
  * The audited loss error excludes the cross-tokenizer vocab design
    prior, and the mxfp4 GPT-OSS anchor is not sentineled.
  * The audit report names the TTFT measurement basis (serving-stack
    floor included, load-dependent queueing excluded).
"""

import json
import os
import sys

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO)
AC_DIR = os.path.join(REPO, "ac")
if AC_DIR not in sys.path:
    sys.path.insert(0, AC_DIR)

from ac.trust_audit import (  # noqa: E402
    _build_candidate_from_anchor,
    load_public_model_registry,
    run_public_anchor,
)


def _registry():
    anchors, default_tol, _post = load_public_model_registry()
    return anchors, default_tol


# ---------------------------------------------------------------------------
# trust-audit CLI accepts --out alone (implies --all)
# ---------------------------------------------------------------------------

def test_trust_audit_out_implies_all(tmp_path):
    from cli_trust_audit import build_parser, main
    # No stage flag and no --out must still error.
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
        main([])
    out = tmp_path / "audit"
    rc = main(["--out", str(out)])
    # Exit code reflects block_publication (nonzero is fine); artifacts must exist.
    assert (out / "audit.json").exists()
    assert (out / "report.md").exists()
    assert (out / "family_bias.json").exists()
    audit = json.loads((out / "audit.json").read_text())
    # --all implied: both stages ran.
    assert "public_anchors" in audit
    assert "frontier_feasibility" in audit


# ---------------------------------------------------------------------------
# per-metric family bias (schema v2) + serving floors
# ---------------------------------------------------------------------------

def test_family_bias_v2_has_serving_metrics(tmp_path):
    from cli_trust_audit import main
    out = tmp_path / "audit"
    main(["--public-anchors", "--out", str(out)])
    fb = json.loads((out / "family_bias.json").read_text())
    assert fb["schema_version"] == "wave20.family_bias.v2"
    for fam in ("dense", "moe"):
        entry = fb["families"][fam]
        # v2 per-metric block
        for metric in ("loss", "tbt_ms", "ttft_ms", "mem_gb"):
            assert metric in entry["metrics"], (fam, metric)
            assert entry["metrics"][metric]["n"] >= 1
        # v1 compatibility keys preserved
        assert "mean_signed_loss_err_pct" in entry
        assert entry["mean_signed_loss_err_pct"] == \
            entry["metrics"]["loss"]["mean_signed_err_pct"]


def test_shipped_family_bias_table_is_v2():
    path = os.path.join(AC_DIR, "calibration", "family_bias_v1.json")
    fb = json.loads(open(path).read())
    assert fb["schema_version"] == "wave20.family_bias.v2"
    assert "tbt_ms" in fb["families"]["moe"]["metrics"]


# ---------------------------------------------------------------------------
# anchor-workload honesty
# ---------------------------------------------------------------------------

def test_audit_defaults_moe_anchor_to_ep1_not_ep2():
    anchors, _ = _registry()
    mixtral = [a for a in anchors if a.id == "mixtral-8x22b"][0]
    assert "ep" not in mixtral.workload  # published bench is TP-only
    cand = _build_candidate_from_anchor(mixtral)
    assert cand.ep_degree == 1


def test_gpt_oss_registry_tp_is_2_and_documents_the_correction():
    path = os.path.join(
        REPO, "tests", "fixtures", "public_model_anchors_v1.json")
    with open(path) as f:
        reg = json.load(f)
    e = [x for x in reg["entries"] if x["id"] == "gpt-oss-120b"][0]
    assert e["workload"]["tp"] == 2
    assert "Wave 29" in e["published_metric"]["source"]


def test_gpt_oss_anchor_mem_error_back_inside_family_regime():
    from trust_audit import load_public_model_registry, run_public_anchor
    anchors, tol, _ = load_public_model_registry()
    a = [x for x in anchors if x.id == "gpt-oss-120b"][0]
    r = run_public_anchor(a, tol)
    by = {m.metric: m for m in r.metrics}
    # Was -58.2% at the fabricated tp=4; the corrected workload puts
    # it at ~-16%, inside the MoE family's serving-overhead regime.
    assert by["mem_gb"].rel_err > -0.25
    assert by["mem_gb"].rel_err < 0.0
    # TBT similarly rejoins the family band (was -54.6%).
    assert by["tbt_ms"].rel_err > -0.45


def test_gpt_oss_anchor_not_sentineled():
    anchors, tol = _registry()
    gptoss = [a for a in anchors if a.id == "gpt-oss-120b"][0]
    assert gptoss.arch["ffn_precision"] == "mxfp4"
    res = run_public_anchor(gptoss, tol)
    loss = [m for m in res.metrics if m.metric == "loss"][0]
    # Was a ~1e8% sentinel when mxfp4 storage was marked infeasible.
    assert abs(loss.rel_err) < 0.5


def test_vocab_design_prior_excluded_from_anchor_loss():
    # Mixtral's 32k tokenizer draws a large undersized-vocab design
    # prior; its published loss is measured in its OWN tokenizer's
    # units, so the audit must not charge the counterfactual.
    anchors, tol = _registry()
    mixtral = [a for a in anchors if a.id == "mixtral-8x22b"][0]
    res = run_public_anchor(mixtral, tol)
    loss = [m for m in res.metrics if m.metric == "loss"][0]
    vocab_terms = [b for b in res.breakdown
                   if b.term_name == "vocab_residual"]
    if vocab_terms:
        # The term exists in the breakdown (it is real for sweeps)...
        assert vocab_terms[0].value > 0.0
    # ...but the audited loss error is not inflated by it: with the
    # Wave-21 vocab weight (0.022, 32k at 39B active => ~5% capped),
    # including it would push Mixtral's loss error past +10%.
    assert abs(loss.rel_err) < 0.10


# ---------------------------------------------------------------------------
# report wording
# ---------------------------------------------------------------------------

def test_audit_report_names_the_ttft_basis():
    from trust_audit import render_public_anchor_markdown
    md = render_public_anchor_markdown({
        "tolerances_kind": "pre_calibration",
        "tolerances_in_use": {"loss": 0.1, "tbt_ms": 0.25,
                              "ttft_ms": 0.3, "mem_gb": 0.15},
        "counts": {"pass": 0, "fail": 0, "skipped": 0,
                   "unrepresentable": 0, "error": 0, "blocking": 0},
        "block_publication": False,
        "anchors": [],
    })
    assert "TTFT basis" in md
    assert "queueing" in md
