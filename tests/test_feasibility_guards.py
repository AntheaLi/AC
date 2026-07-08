"""Wave 13 — Feasibility dataclass with named guards.

Tests:
  * EvaluatedCandidate.feasibility is populated by evaluate_candidate
  * Feasibility.is_feasible matches meets_constraints (back-compat)
  * violated_guards / warning_guards return correct named subsets
  * each guard's metric_value and threshold are populated
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FeasibilityStructureTests(unittest.TestCase):

    def setUp(self):
        from ac.optimizer import (
            DeploymentConstraints, evaluate_candidate, CandidateArch,
        )
        self.DC = DeploymentConstraints
        self.eval = evaluate_candidate
        self.Cand = CandidateArch

    def _make_candidate(self, d_model=4096, n_layers=32, n_heads=32, d_head=128,
                        n_kv_heads=8, ffn_dim=14336, kv_cache_bits=16):
        return self.Cand(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_head=d_head,
            n_kv_heads=n_kv_heads, ffn_dim=ffn_dim, vocab_size=128000,
            weight_precision="bf16", ffn_precision="bf16",
            attn_precision={"q":"bf16","k":"bf16","v":"bf16","o":"bf16"},
            kv_cache_bits=kv_cache_bits, moe=None, state_config=None,
            n_dense_ffn_layers=0, attention_type="gqa",
            tp_degree=1, cp_degree=1, ep_degree=1,
        )

    def test_feasibility_field_populated(self):
        """Every EvaluatedCandidate from evaluate_candidate must have a
        Feasibility object."""
        c = self.DC(
            target_params_b=1.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            serving_tbt_ms=None, serving_ttft_ms=None,
            serving_batch=8, allow_quality_sentinel=True,
        )
        ev = self.eval(self._make_candidate(), "h100", c)
        self.assertIsNotNone(ev.feasibility,
                             "Feasibility object must be populated by evaluate_candidate")
        self.assertEqual(ev.feasibility.is_feasible, ev.meets_constraints,
                         "Feasibility.is_feasible must match meets_constraints (back-compat)")

    def test_named_guards_present(self):
        """All five expected named guards must be in feasibility.guards."""
        c = self.DC(
            target_params_b=1.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            serving_tbt_ms=None, serving_ttft_ms=None,
            serving_batch=8, allow_quality_sentinel=True,
        )
        ev = self.eval(self._make_candidate(), "h100", c)
        expected_guards = {
            "memory_extreme_overflow",
            "quality_sentinel_tripped",
            "tbt_budget_warning",
            "ttft_budget_warning",
            "hbm_spill_warning",
        }
        self.assertEqual(
            set(ev.feasibility.guards.keys()), expected_guards,
            f"Expected guards: {expected_guards}; got {set(ev.feasibility.guards.keys())}")

    def test_tbt_soft_budget_warning_only(self):
        """Setting an aggressive serving_tbt_ms budget triggers a warning
        guard but does NOT fail feasibility (per Wave 2a continuous-serving
        contract)."""
        c = self.DC(
            target_params_b=1.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            serving_tbt_ms=0.001,  # impossibly aggressive — guaranteed to "exceed"
            serving_ttft_ms=None,
            serving_batch=8, allow_quality_sentinel=True,
        )
        ev = self.eval(self._make_candidate(), "h100", c)
        g = ev.feasibility.guards["tbt_budget_warning"]
        # Should trigger (TBT will exceed 0.001ms for any real arch)
        self.assertTrue(g.triggered, "TBT budget at 0.001ms must trigger")
        # But it's a warning — must not fail feasibility
        self.assertFalse(g.fails_feasibility, "TBT budget is a soft warning")
        self.assertTrue(g.is_warning)
        # And feasibility overall should still be True (no hard guards triggered)
        if not ev.feasibility.is_feasible:
            # If feasibility failed, it should be from a different guard
            self.assertNotIn("tbt_budget_warning", ev.feasibility.violated_guards)

    def test_warning_guards_property(self):
        """warning_guards returns the list of triggered warning guards."""
        c = self.DC(
            target_params_b=1.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            serving_tbt_ms=0.001, serving_ttft_ms=0.001,
            serving_batch=8, allow_quality_sentinel=True,
        )
        ev = self.eval(self._make_candidate(), "h100", c)
        warnings = ev.feasibility.warning_guards
        # Both tbt and ttft budgets should be in warnings
        self.assertIn("tbt_budget_warning", warnings)
        self.assertIn("ttft_budget_warning", warnings)

    def test_metric_value_and_threshold_populated(self):
        """Each triggered guard must carry metric_value and threshold."""
        c = self.DC(
            target_params_b=1.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            serving_tbt_ms=0.001, serving_ttft_ms=None,
            serving_batch=8, allow_quality_sentinel=True,
        )
        ev = self.eval(self._make_candidate(), "h100", c)
        g = ev.feasibility.guards["tbt_budget_warning"]
        if g.triggered:
            self.assertIsNotNone(g.metric_value,
                                 "triggered guard must carry metric_value")
            self.assertIsNotNone(g.threshold,
                                 "triggered guard must carry threshold")
            self.assertGreater(g.metric_value, g.threshold,
                               "triggered TBT guard: metric must exceed threshold")


if __name__ == "__main__":
    unittest.main()
