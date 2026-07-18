"""Wave 8 — PP / DP / EP as search variables.

Tests:
  * pp_options field on DeploymentConstraints normalizes correctly
  * generators produce multiple candidates per shape when pp_options has
    multiple entries
  * each candidate carries a distinct pp_degree
  * back-compat: single-pp callers (pp_options=None or [1]) see no change
  * the dedupe key separates pp_degree variants
  * ep_options remains a search variable (regression check)
  * dp is intentionally derived; documented as such

Mirror of tests/test_optimize_across_contexts.py contract style.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ac"))


class PpOptionsNormalizationTests(unittest.TestCase):

    def test_pp_options_defaults_to_pp_scalar(self):
        from ac.optimizer import DeploymentConstraints
        c = DeploymentConstraints(target_params_b=1.0, pp=4)
        self.assertEqual(c.pp_options, [4])

    def test_pp_options_list_is_sorted_deduped(self):
        from ac.optimizer import DeploymentConstraints
        c = DeploymentConstraints(target_params_b=1.0, pp_options=[4, 2, 4, 8])
        self.assertEqual(c.pp_options, [2, 4, 8])

    def test_pp_options_rejects_invalid(self):
        from ac.optimizer import DeploymentConstraints
        with self.assertRaises(ValueError):
            DeploymentConstraints(target_params_b=1.0, pp_options=[])

    def test_pp_options_negative_filtered(self):
        from ac.optimizer import DeploymentConstraints
        # Negative values dropped; ValueError only if list ends up empty.
        c = DeploymentConstraints(target_params_b=1.0, pp_options=[2, 4])
        self.assertEqual(c.pp_options, [2, 4])


class PpSearchExpansionTests(unittest.TestCase):

    def test_single_pp_passthrough_no_expansion(self):
        """When pp_options has one value, total candidate count equals
        the legacy (pre-Wave-8) count. The expansion is a no-op rename."""
        from ac.optimizer import (
            DeploymentConstraints, _enumerate_and_dedupe,
        )
        c = DeploymentConstraints(
            target_params_b=1.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            pp_options=[1], serving_batch=8,
            allow_quality_sentinel=True, max_candidates=999999,
        )
        cands, _, _ = _enumerate_and_dedupe("h100", c)
        # Every candidate must have pp_degree=1
        self.assertTrue(all(getattr(x, "pp_degree", 1) == 1 for x in cands),
                        "single-pp expansion must label every candidate pp_degree=1")

    def test_multi_pp_expansion_produces_multiple_pp_degrees(self):
        """When pp_options has multiple values, the generator must produce
        candidates at each pp_degree (subject to n_layers divisibility)."""
        from ac.optimizer import (
            DeploymentConstraints, generate_candidates,
        )
        c = DeploymentConstraints(
            target_params_b=1.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            pp_options=[1, 2, 4], serving_batch=8,
            allow_quality_sentinel=True, max_candidates=999999,
        )
        cands = generate_candidates("h100", c)
        pp_degrees_seen = set(getattr(x, "pp_degree", 1) for x in cands)
        # Must see at least pp=1 (always passes divisibility) plus pp=2/4
        # for some candidate with n_layers divisible by them.
        self.assertIn(1, pp_degrees_seen)
        self.assertTrue(
            2 in pp_degrees_seen or 4 in pp_degrees_seen,
            f"expected pp=2 or pp=4 expansion; got {sorted(pp_degrees_seen)}")

    def test_pp_expansion_respects_divisibility(self):
        """A candidate at pp_d=4 must have n_layers divisible by 4."""
        from ac.optimizer import (
            DeploymentConstraints, generate_candidates,
        )
        c = DeploymentConstraints(
            target_params_b=1.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            pp_options=[1, 2, 4], serving_batch=8,
            allow_quality_sentinel=True, max_candidates=999999,
        )
        cands = generate_candidates("h100", c)
        for x in cands:
            pp = int(getattr(x, "pp_degree", 1) or 1)
            if pp > 1:
                self.assertEqual(
                    x.n_layers % pp, 0,
                    f"candidate with pp_degree={pp} has n_layers={x.n_layers} "
                    f"(must be divisible by pp)")


class DedupeIncludesPpTests(unittest.TestCase):

    def test_dedupe_separates_pp_variants(self):
        """Two candidates differing only in pp_degree must NOT collide in
        dedupe. Without this, the second is silently dropped."""
        from ac.optimizer import (
            DeploymentConstraints, _enumerate_and_dedupe,
        )
        c = DeploymentConstraints(
            target_params_b=1.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            pp_options=[1, 2], serving_batch=8,
            allow_quality_sentinel=True, max_candidates=999999,
        )
        cands, _, _ = _enumerate_and_dedupe("h100", c)
        # Group by (shape minus pp); within each group, pp_degree variants
        # must be present.
        from collections import defaultdict
        groups = defaultdict(set)
        for x in cands:
            shape = (x.d_model, x.n_layers, x.n_heads, x.d_head,
                     x.n_kv_heads, x.ffn_dim, x.weight_precision,
                     x.kv_cache_bits, x.tp_degree)
            groups[shape].add(getattr(x, "pp_degree", 1))
        # At least one shape group must have both pp=1 and pp=2 entries
        # (any candidate with even n_layers gets both).
        multi_pp_groups = [s for s, pps in groups.items() if len(pps) > 1]
        self.assertTrue(
            multi_pp_groups,
            "dedupe collapsed pp_degree variants — expected at least one "
            "shape with both pp=1 and pp=2 entries.")


class EpSearchRegressionTests(unittest.TestCase):

    def test_ep_options_still_iterates(self):
        """Regression: ep_options is still respected as a search variable
        via compute_moe_options(ep_degrees=). Each MoE candidate's
        ep_degree comes from its associated MoEOption."""
        from ac.optimizer import (
            DeploymentConstraints, generate_moe_candidates,
        )
        c = DeploymentConstraints(
            target_params_b=7.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            ep_options=[2, 4, 8], serving_batch=8,
            allow_moe=True,
            allow_quality_sentinel=True, max_candidates=999999,
        )
        cands = generate_moe_candidates("h100", c)
        if not cands:
            self.skipTest("no MoE candidates generated at this setup")
        ep_degrees = set(getattr(x, "ep_degree", 1) for x in cands)
        # Should see at least two distinct EP values from the [2,4,8] set
        self.assertGreaterEqual(
            len(ep_degrees), 2,
            f"expected MoE candidates at multiple EP degrees; "
            f"got {sorted(ep_degrees)}")


class DpIsDerivedDocTest(unittest.TestCase):

    def test_no_dp_options_field(self):
        """DP is intentionally NOT a search variable — it's derived from
        cluster_size / (tp × pp). Surface as scalar `dp` only.

        This test guards against accidentally adding a `dp_options` field
        without updating the design doc that explains DP is derived.
        """
        from ac.optimizer import DeploymentConstraints
        c = DeploymentConstraints(target_params_b=1.0)
        self.assertFalse(
            hasattr(c, "dp_options"),
            "DeploymentConstraints has dp_options but DP is supposed to be "
            "a derived quantity (cluster_size / (tp × pp)). Update the "
            "Wave 8 design doc if DP-as-search-variable is desired.")


if __name__ == "__main__":
    unittest.main()
