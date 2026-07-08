"""Zero-compute calibration: paired decisions, pair fitting, ladder plans.

(Pins from Waves 18h and 19.)

  1. Paired-decision math: correlated errors cancel; identical configs
     bottom out at the run-noise floor; the surviving uncertainty lives in
     the terms that actually differ.
  2. Paired-ablation residual fitting: the shipped public corpus fits,
     validates the locality residual (~1.0 scale), and flags uncovered
     terms and unanchored regions.
  3. Ladder planning: all three verdict classes (resolvable from priors /
     resolvable with proposed runs / unresolvable below the noise floor),
     with runs priced and sigma monotone, and rung costs kept sane
     (no 40k GPU-day ladders).
"""

import os
import sys
import unittest

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ac.quality_model import (
    ArchConfig as QArch, estimate_quality, paired_loss_uncertainty,
    load_quality_constants)

_TR = {"training_tokens": int(20e12)}
_W = {"context_length": 8192}
_BASE = dict(d_model=4096, n_layers=32, n_heads=32, d_head=128,
             n_kv_heads=8, ffn_dim=14336, vocab_size=128256)


def _q(**over):
    kw = dict(_BASE)
    kw.update(over)
    return estimate_quality(QArch(**kw), _TR, workload_spec=_W)


class PairedDecisionTests(unittest.TestCase):
    def test_identical_configs_cancel_to_floor(self):
        qa, qb = _q(), _q()
        p = paired_loss_uncertainty(qa, qb)
        floor = float(load_quality_constants()["paired_decision"]
                      ["run_noise_floor_pct"]) / 100.0
        self.assertAlmostEqual(p["sigma_rel"], floor, delta=0.1 * floor,
            msg="identical configs must cancel to the run-noise floor")

    def test_paired_sigma_below_naive_and_terms_attributed(self):
        qa = _q()
        qb = _q(ffn_dim=2048, model_type="moe",
                moe_config={"n_experts": 32, "top_k": 4, "expert_dim": 2048})
        p = paired_loss_uncertainty(qa, qb)
        self.assertLess(p["sigma_rel"], 0.5 * p["naive_rel"],
            msg="pairing must cancel at least half of the naive sigma")
        self.assertGreater(p["cancelled_fraction"], 0.5)
        # The decision-driving terms must be the ones that differ.
        driving = sorted(p["per_term"].items(),
                         key=lambda kv: -kv[1]["sigma_pair"])[:3]
        names = {k for k, _ in driving}
        self.assertTrue(names & {"effective_capacity", "moe_residual", "spine"},
            msg=f"MoE-vs-dense driving terms wrong: {names}")

    def test_spine_correlation_grounded_in_n_eff_distance(self):
        # Same-size reshape shares more spine error than a 4x size change.
        qa = _q()
        near = _q(d_model=4608, n_layers=28, n_heads=36, ffn_dim=16384)
        far = _q(d_model=2048, n_layers=22, n_heads=16, ffn_dim=8192)
        p_near = paired_loss_uncertainty(qa, near)
        p_far = paired_loss_uncertainty(qa, far)
        self.assertLess(p_near["sigma_rel"], p_far["sigma_rel"],
            msg="a nearby-N reshape must retain less spine uncertainty "
                "than a 4x size change")

    def test_decision_module_uses_pairing(self):
        from ac.decision import _combined_uncertainty_pct

        class _Ev:
            def __init__(self, q):
                self.quality = q
                self.predicted_loss = q.predicted_loss
        qa, qb = _q(), _q(d_model=4608, n_layers=28, n_heads=36,
                          ffn_dim=16384)
        paired_pct = _combined_uncertainty_pct(_Ev(qa), _Ev(qb))
        naive_pct = (2 ** 0.5) * qa.uncertainty_total * 100.0
        self.assertLess(paired_pct, 0.6 * naive_pct,
            msg="decision layer must exploit correlated-error cancellation")


class AblationPairFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        from ac.ablation_fit import evaluate_pairs, fit_terms, coverage_gaps
        path = os.path.join(os.path.dirname(__file__), "fixtures",
                            "public_ablation_pairs_v1.json")
        with open(path) as f:
            corpus = json.load(f)
        cls.results = evaluate_pairs(corpus)
        cls.fits = {f.term: f for f in fit_terms(cls.results)}
        cls.gaps = coverage_gaps(list(cls.fits.values()))

    def test_corpus_evaluates(self):
        self.assertGreaterEqual(len(self.results), 10)
        # A majority of pairs should predict within tolerance (the ones
        # that don't are the fit's actionable findings, not failures).
        ok = sum(1 for r in self.results if r.within_tolerance)
        self.assertGreaterEqual(ok, len(self.results) // 2)

    def test_locality_residual_validated_by_corpus(self):
        f = self.fits["attention_locality"]
        self.assertEqual(f.n_pairs, 2)
        self.assertIsNotNone(f.scale)
        self.assertAlmostEqual(f.scale, 1.0, delta=0.15,
            msg="Wave 18g locality residual should match Gemma-2/Mistral "
                "published deltas at ~1.0 scale")

    def test_uncovered_terms_flagged(self):
        uncovered = [t for t, f in self.fits.items() if f.n_pairs == 0]
        self.assertIn("precision_residual", uncovered)
        for t in uncovered:
            self.assertTrue(any("UNCOVERED" in w for w in self.fits[t].warnings))

    def test_state_coverage_range_reported(self):
        joined = " ".join(self.gaps)
        self.assertIn("state_residual", joined)
        # p_attn=0 IS anchored (Waleffe pure-SSM pair), so the audit must
        # report the anchored range rather than a low-end extrapolation
        # flag; the long-context end (no anchor above 256k) must be
        # flagged as extrapolation.
        self.assertIn("anchored p_attn range [0.00,", joined)
        self.assertIn("extrapolation", joined)


class LadderPlanTests(unittest.TestCase):
    def test_resolvable_from_priors(self):
        from ac.ladder_plan import plan_ladder
        plan = plan_ladder("dense", "hybrid", 13.0, 20.0,
                           context=1048576)
        self.assertTrue(plan.resolvable_now)
        self.assertEqual(len(plan.runs), 0)

    def test_below_floor_proposes_no_runs(self):
        from ac.ladder_plan import plan_ladder
        plan = plan_ladder("dense", "moe", 13.0, 20.0, context=8192)
        self.assertFalse(plan.resolvable_now)
        if plan.delta_pct <= plan.z * plan.sigma_floor_pct:
            self.assertEqual(len(plan.runs), 0)
            self.assertIn("UNRESOLVABLE", plan.verdict)

    def test_ladder_resolves_mid_gap_with_priced_runs(self):
        from ac.ladder_plan import plan_ladder
        plan = plan_ladder("dense", "moe", 3.0, 20.0, context=8192)
        self.assertFalse(plan.resolvable_now)
        self.assertTrue(plan.resolves, msg=plan.verdict)
        self.assertGreater(len(plan.runs), 0)
        sig = [r.marginal_sigma_after_pct for r in plan.runs]
        self.assertEqual(sig, sorted(sig, reverse=True),
                         msg="posterior sigma must shrink monotonically")
        for r in plan.runs:
            self.assertGreater(r.gpu_days_pair, 0.0)
            self.assertGreaterEqual(r.transfer_factor, 1.0)

    def test_budget_cap_respected(self):
        from ac.ladder_plan import plan_ladder
        plan = plan_ladder("dense", "moe", 3.0, 20.0, context=8192,
                           max_gpu_days=700.0)
        if plan.runs:
            self.assertLessEqual(plan.runs[-1].cumulative_gpu_days, 700.0)


class TestLadderCostSanity:
    def test_small_ladder_costs_are_sane(self):
        from ac.ladder_plan import plan_ladder
        plan = plan_ladder("dense", "moe", target_params_b=7.0,
                           target_tokens_t=20.0, seeds_per_point=2)
        if not plan.runs:
            pytest.skip("plan resolved from priors / floor — no runs")
        total = plan.runs[-1].cumulative_gpu_days
        assert total < 5000, (
            f"0.5-3B ladder priced at {total:.0f} GPU-days — the rung "
            "over-training cap or MoE throughput fix has regressed")
        per_pair = max(r.gpu_days_pair for r in plan.runs)
        assert per_pair < 2000


if __name__ == "__main__":
    unittest.main()
