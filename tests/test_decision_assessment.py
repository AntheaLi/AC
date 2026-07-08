"""Wave 18d — Decision confidence, abstention, and evidence provenance.

Tests the acceptance-gate scenarios from the spec:

* A 1% gap under 8% uncertainty is unresolved.
* A >8% stable, in-domain gap can produce a winner.
* An operationally extreme candidate can still be a scientific contender.
* An out-of-domain candidate cannot be a unique winner.
* Missing 18e stability data leaves the decision unresolved in trust mode.
* Every nonzero residual has evidence provenance (structural).
* JSON and Markdown adapters explain the same decision.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ac.decision import (
    DecisionAssessment,
    EVIDENCE_KINDS,
    EvidenceProvenance,
    assess_decision,
    decision_to_json,
    decision_to_markdown,
    evidence_breakdown,
    make_provenance,
    DECISION_MODEL_FAILURE,
    DECISION_NO_PHYSICAL,
    DECISION_OUT_OF_DOMAIN,
    DECISION_UNRESOLVED,
    DECISION_WINNER,
)


# ---------------------------------------------------------------------------
# Test fixture helpers
# ---------------------------------------------------------------------------


def _term(name, value, uncertainty_pct, evidence_kind="scaling_law",
          domain_status="in_domain"):
    """Build a TermResult-shaped fixture with provenance."""
    return SimpleNamespace(
        name=name,
        value=value,
        uncertainty=uncertainty_pct / 100.0,
        provenance=make_provenance(
            evidence_kind=evidence_kind,
            source=f"test:{name}",
            uncertainty=uncertainty_pct / 100.0,
            domain_status=domain_status,
        ),
    )


def _quality(loss, terms=None, in_chinchilla_regime=True,
             uncertainty_total_pct=None):
    q = SimpleNamespace(
        predicted_loss=loss,
        chinchilla_baseline=loss * 0.95,
        terms={t.name: t for t in (terms or [])},
        in_chinchilla_regime=in_chinchilla_regime,
        uncertainty_total=(uncertainty_total_pct or 0.0) / 100.0,
        uncertainty_low_pct=0.0,
        uncertainty_high_pct=0.0,
    )
    return q


def _candidate(loss, uncertainty_pct=3.0, meets_constraints=True,
               domain_status="in_domain", d_model=4096, n_layers=32,
               n_heads=32, d_head=128, n_kv_heads=8, ffn_dim=14336,
               tp=1, pp=1, moe=None, state_config=None,
               attention_type="gqa"):
    """Build an EvaluatedCandidate-shaped fixture."""
    arch = SimpleNamespace(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_head=d_head,
        n_kv_heads=n_kv_heads, ffn_dim=ffn_dim,
        tp_degree=tp, pp_degree=pp,
        moe=moe, state_config=state_config,
        attention_type=attention_type,
    )
    terms = [_term("spine", 0.0, uncertainty_pct,
                   evidence_kind="scaling_law",
                   domain_status=domain_status)]
    q = _quality(loss, terms=terms)
    return SimpleNamespace(
        arch=arch,
        quality=q,
        predicted_loss=loss,
        meets_constraints=meets_constraints,
    )


# ---------------------------------------------------------------------------
# 1. Evidence provenance construction and validation
# ---------------------------------------------------------------------------


class EvidenceProvenanceTests(unittest.TestCase):

    def test_all_evidence_kinds_accepted(self):
        for kind in EVIDENCE_KINDS:
            p = make_provenance(evidence_kind=kind, source="test")
            self.assertEqual(p.evidence_kind, kind)

    def test_unknown_evidence_kind_rejected(self):
        with self.assertRaises(ValueError):
            make_provenance(evidence_kind="magic", source="test")

    def test_unknown_domain_status_rejected(self):
        with self.assertRaises(ValueError):
            make_provenance(evidence_kind="heuristic", domain_status="unknown")

    def test_negative_uncertainty_rejected(self):
        with self.assertRaises(ValueError):
            make_provenance(evidence_kind="heuristic", uncertainty=-0.01)

    def test_calibrated_and_widens_only_labels(self):
        cal = make_provenance(evidence_kind="calibrated_measurement")
        heur = make_provenance(evidence_kind="heuristic")
        lit = make_provenance(evidence_kind="literature_prior")
        scaling = make_provenance(evidence_kind="scaling_law")
        self.assertTrue(cal.is_calibrated())
        self.assertFalse(heur.is_calibrated())
        self.assertTrue(heur.widens_only())
        self.assertTrue(lit.widens_only())
        self.assertFalse(scaling.widens_only())


# ---------------------------------------------------------------------------
# 2. Winner/unresolved band tests (the core spec scenarios)
# ---------------------------------------------------------------------------


class UniqueWinnerRuleTests(unittest.TestCase):

    def test_small_gap_under_high_uncertainty_is_unresolved(self):
        """A 1% loss gap under 8% combined quality uncertainty must be
        `unresolved`. This is the acceptance-gate example from Wave 18d."""
        # Two candidates with 1% loss gap but 6% per-candidate uncertainty
        # → combined ≈ 8.5%; gap < threshold → unresolved.
        a = _candidate(loss=2.00, uncertainty_pct=6.0)
        b = _candidate(loss=2.02, uncertainty_pct=6.0, d_model=5120)
        d = assess_decision([a, b])
        self.assertEqual(d.status, DECISION_UNRESOLVED,
                         f"1% gap under ~8% uncertainty must be unresolved, got {d.status}")
        # Both should appear in the contender list.
        self.assertEqual(len(d.contender_ids), 2)
        self.assertIn(d.selected_candidate_id, (None,))  # unresolved leaves it None

    def test_large_stable_in_domain_gap_produces_winner(self):
        """A gap larger than both the practical threshold and the combined
        uncertainty must fire the winner rule."""
        a = _candidate(loss=1.80, uncertainty_pct=2.0)
        b = _candidate(loss=2.05, uncertainty_pct=2.0, d_model=5120)
        # 13.9% gap over ~2.8% combined uncertainty
        d = assess_decision([a, b], stability_fraction=0.90,
                            contender_retention_fraction=0.97)
        self.assertEqual(d.status, DECISION_WINNER)
        self.assertIsNotNone(d.selected_candidate_id)
        self.assertGreater(d.runner_up_gap_pct or 0, 5.0)

    def test_out_of_domain_top_candidate_cannot_win(self):
        """If the best candidate carries an out-of-domain term, the cell is
        `out_of_domain` regardless of loss."""
        a = _candidate(loss=1.80, uncertainty_pct=2.0, domain_status="out_of_domain")
        b = _candidate(loss=2.05, uncertainty_pct=2.0, d_model=5120)
        d = assess_decision([a, b])
        self.assertEqual(d.status, DECISION_OUT_OF_DOMAIN)
        self.assertEqual(d.domain_status, "out_of_domain")

    def test_no_feasible_candidate_returns_no_physical(self):
        """When Wave 18c physical guards leave nothing, status is
        `no_physically_feasible_candidate` — a distinct signal from
        `unresolved`."""
        a = _candidate(loss=1.80, meets_constraints=False)
        b = _candidate(loss=2.05, meets_constraints=False, d_model=5120)
        d = assess_decision([a, b])
        self.assertEqual(d.status, DECISION_NO_PHYSICAL)
        self.assertIsNone(d.selected_candidate_id)

    def test_trust_mode_without_stability_data_unresolved(self):
        """`require_stability_for_winner=True` + missing stability data →
        unresolved even for a big gap. Matches the spec: 'Missing 18e
        stability data leaves the decision unresolved in trust mode.'"""
        a = _candidate(loss=1.80, uncertainty_pct=2.0)
        b = _candidate(loss=2.05, uncertainty_pct=2.0, d_model=5120)
        d = assess_decision([a, b], require_stability_for_winner=True)
        self.assertEqual(d.status, DECISION_UNRESOLVED)
        self.assertTrue(any("stability data" in r for r in d.reasons))

    def test_low_stability_fraction_forces_unresolved(self):
        """Stability fraction < 0.80 must prevent a winner."""
        a = _candidate(loss=1.80, uncertainty_pct=2.0)
        b = _candidate(loss=2.05, uncertainty_pct=2.0, d_model=5120)
        d = assess_decision([a, b], stability_fraction=0.65,
                            contender_retention_fraction=0.99)
        self.assertEqual(d.status, DECISION_UNRESOLVED)
        self.assertTrue(any("stability" in r for r in d.reasons))

    def test_low_contender_retention_forces_unresolved(self):
        """Contender retention < 0.95 must prevent a winner even with high
        stability fraction."""
        a = _candidate(loss=1.80, uncertainty_pct=2.0)
        b = _candidate(loss=2.05, uncertainty_pct=2.0, d_model=5120)
        d = assess_decision([a, b], stability_fraction=0.90,
                            contender_retention_fraction=0.60)
        self.assertEqual(d.status, DECISION_UNRESOLVED)

    def test_practical_effect_threshold_dominates_when_uncertainty_tiny(self):
        """When combined uncertainty is well below the 5% practical floor,
        the practical floor still applies."""
        a = _candidate(loss=2.00, uncertainty_pct=0.5)
        # 2% gap — above tiny uncertainty band but below 5% practical floor
        b = _candidate(loss=2.04, uncertainty_pct=0.5, d_model=5120)
        d = assess_decision([a, b], stability_fraction=0.90,
                            contender_retention_fraction=0.99)
        self.assertEqual(d.status, DECISION_UNRESOLVED,
                         "2% gap must be unresolved when the practical "
                         "floor is 5% even though uncertainty is tiny")

    def test_model_validation_failure_short_circuits(self):
        """Passing `model_validation_failure` produces
        `model_validation_failure` regardless of candidate content."""
        a = _candidate(loss=1.80, uncertainty_pct=2.0)
        b = _candidate(loss=2.05, uncertainty_pct=2.0, d_model=5120)
        d = assess_decision([a, b],
                            model_validation_failure="18e anchor Llama-3-70B failed")
        self.assertEqual(d.status, DECISION_MODEL_FAILURE)
        self.assertTrue(any("Llama-3-70B" in r for r in d.reasons))


# ---------------------------------------------------------------------------
# 3. Contender set semantics
# ---------------------------------------------------------------------------


class ContenderSetTests(unittest.TestCase):

    def test_multiple_contenders_all_within_threshold(self):
        """Three candidates within the noise band all appear as contenders."""
        a = _candidate(loss=2.00, uncertainty_pct=4.0, d_model=4096)
        b = _candidate(loss=2.02, uncertainty_pct=4.0, d_model=5120)
        c = _candidate(loss=2.04, uncertainty_pct=4.0, d_model=6144)
        d = assess_decision([a, b, c])
        self.assertEqual(d.status, DECISION_UNRESOLVED)
        self.assertEqual(len(d.contender_ids), 3)

    def test_out_of_band_candidates_excluded_from_contenders(self):
        """A candidate 20% worse than the best is not a contender."""
        a = _candidate(loss=2.00, uncertainty_pct=2.0, d_model=4096)
        b = _candidate(loss=2.02, uncertainty_pct=2.0, d_model=5120)
        c = _candidate(loss=2.40, uncertainty_pct=2.0, d_model=6144)  # far away
        d = assess_decision([a, b, c])
        self.assertLessEqual(len(d.contender_ids), 2,
                             "20%-worse candidate should not be in contender set")

    def test_operationally_extreme_candidate_still_a_contender(self):
        """From the spec: 'An operationally extreme candidate can still be
        a scientific contender.' We simulate this by leaving
        `meets_constraints=True` (Wave 18c operational flags never remove
        candidates) and asserting the candidate appears in the contender set
        when its loss is comparable."""
        # Both feasible, both within the noise band; the second has
        # 'operational flags' (would be attached externally) but is still
        # a scientific contender.
        a = _candidate(loss=2.00, uncertainty_pct=4.0)
        b = _candidate(loss=2.02, uncertainty_pct=4.0, d_model=5120)
        # Attach a fake operational-flag list to demonstrate that presence
        # of flags doesn't touch decision.
        b.operational_flags = ["extreme_tbt", "cross_node_collective_risk"]
        d = assess_decision([a, b])
        self.assertEqual(d.status, DECISION_UNRESOLVED)
        self.assertEqual(len(d.contender_ids), 2)


# ---------------------------------------------------------------------------
# 4. Rendering adapters
# ---------------------------------------------------------------------------


class RenderingTests(unittest.TestCase):

    def test_json_and_markdown_explain_same_decision(self):
        """The JSON and Markdown adapters must convey the same decision
        (status, selected id, reason set)."""
        a = _candidate(loss=1.80, uncertainty_pct=2.0)
        b = _candidate(loss=2.05, uncertainty_pct=2.0, d_model=5120)
        d = assess_decision([a, b], stability_fraction=0.90,
                            contender_retention_fraction=0.97)
        j = decision_to_json(d)
        m = decision_to_markdown(d)
        # JSON captures the status.
        self.assertEqual(j["status"], d.status)
        # Markdown mentions the status word ("winner" here).
        self.assertIn("winner", m.lower())
        # Both surface the runner-up gap in some form.
        self.assertIsNotNone(j["runner_up_gap_pct"])
        self.assertIn("runner-up", m.lower())

    def test_display_line_covers_all_statuses(self):
        """Every status renders to a one-liner without raising."""
        # winner
        a = _candidate(loss=1.80, uncertainty_pct=2.0)
        b = _candidate(loss=2.05, uncertainty_pct=2.0, d_model=5120)
        winner = assess_decision([a, b], stability_fraction=0.9,
                                 contender_retention_fraction=0.99)
        self.assertTrue(winner.to_display_line(lambda cid: "sigA").startswith("winner:"))
        # unresolved
        u = assess_decision([_candidate(2.00, 6.0),
                             _candidate(2.02, 6.0, d_model=5120)])
        self.assertTrue(u.to_display_line(lambda cid: "sigX").startswith("unresolved:"))
        # out_of_domain
        ood = assess_decision([_candidate(1.8, 2.0, domain_status="out_of_domain"),
                               _candidate(2.0, 2.0, d_model=5120)])
        self.assertEqual(ood.to_display_line(lambda cid: "?"), "out-of-domain")
        # no physical
        nop = assess_decision([_candidate(1.8, meets_constraints=False),
                               _candidate(2.0, meets_constraints=False, d_model=5120)])
        self.assertEqual(nop.to_display_line(lambda cid: "?"),
                         "no physically feasible candidate")
        # model validation failure
        mvf = assess_decision([a, b], model_validation_failure="anchor tripped")
        self.assertEqual(mvf.to_display_line(lambda cid: "?"),
                         "model validation failure")

    def test_evidence_breakdown_returns_all_terms(self):
        """Per-term breakdown surfaces evidence_kind for downstream renderers."""
        ev = _candidate(loss=1.80, uncertainty_pct=2.0)
        # Add extra terms of varied provenance
        ev.quality.terms["arch_res"] = _term(
            "arch_res", 0.02, 3.0,
            evidence_kind="literature_prior",
        )
        ev.quality.terms["moe_res"] = _term(
            "moe_res", -0.04, 8.0,
            evidence_kind="heuristic",
        )
        eb = evidence_breakdown(ev)
        names = {row["name"] for row in eb}
        self.assertEqual(names, {"spine", "arch_res", "moe_res"})
        # widens_only flag propagates correctly
        by_name = {row["name"]: row for row in eb}
        self.assertTrue(by_name["moe_res"]["widens_only"])
        self.assertFalse(by_name["spine"]["widens_only"])


# ---------------------------------------------------------------------------
# 5. Integration with OptimizationResult
# ---------------------------------------------------------------------------


class IntegrationTests(unittest.TestCase):
    """Verify the optimizer populates OptimizationResult.decision."""

    def test_optimize_populates_decision_field(self):
        from ac.optimizer import DeploymentConstraints, optimize
        c = DeploymentConstraints(
            target_params_b=1.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            tp_options=[1],
            serving_tbt_ms=None, serving_ttft_ms=None,
            serving_batch=8, allow_quality_sentinel=True,
            max_full_evaluations=20, max_candidates=200,
        )
        r = optimize("h100", c)
        self.assertIsNotNone(r.decision,
                             "optimize() must populate OptimizationResult.decision")
        # decision status is one of the defined enums
        from ac.decision import ALL_DECISION_STATUSES
        self.assertIn(r.decision.status, ALL_DECISION_STATUSES)


if __name__ == "__main__":
    unittest.main()
