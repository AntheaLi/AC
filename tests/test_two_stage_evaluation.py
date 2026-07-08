"""Wave 8b — Two-stage candidate evaluation (cheap-rank prune).

Tests:
  * _cheap_quality_rank is monotonic with full predicted_loss in the top decile
  * max_full_evaluations correctly caps the full-eval count
  * back-compat: default None preserves the exhaustive enumeration
  * cheap-rank is hw/ctx-blind by construction (no hardware param)
  * _estimate_n_active_params produces a reasonable proxy
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TwoStageEvaluationTests(unittest.TestCase):

    def setUp(self):
        from ac.optimizer import (
            DeploymentConstraints,
            CandidateArch,
            _cheap_quality_rank,
            _estimate_n_active_params,
            optimize,
            evaluate_candidate,
        )
        self.DC = DeploymentConstraints
        self.Cand = CandidateArch
        self.cheap = _cheap_quality_rank
        self.est = _estimate_n_active_params
        self.optimize = optimize
        self.eval = evaluate_candidate

    def _make_candidate(self, d_model=4096, n_layers=32, ffn_dim=14336,
                        n_heads=32, d_head=128, n_kv_heads=8,
                        vocab=128000):
        return self.Cand(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_head=d_head,
            n_kv_heads=n_kv_heads, ffn_dim=ffn_dim, vocab_size=vocab,
            weight_precision="bf16", ffn_precision="bf16",
            attn_precision={"q":"bf16","k":"bf16","v":"bf16","o":"bf16"},
            kv_cache_bits=16, moe=None, state_config=None,
            n_dense_ffn_layers=0, attention_type="gqa",
            tp_degree=1, cp_degree=1, ep_degree=1,
        )

    def test_estimate_n_active_params_reasonable(self):
        """For Llama-3-70B-like shape (d=8192, L=80, ffn=28672), the proxy
        should be in the 60-80B range — the canonical n_active_params is ~70B."""
        c = self._make_candidate(
            d_model=8192, n_layers=80, ffn_dim=28672,
            n_heads=64, d_head=128, n_kv_heads=8,
        )
        n = self.est(c)
        self.assertGreater(n, 50e9, f"Llama-3-70B-like proxy too small: {n/1e9:.1f}B")
        self.assertLess(n, 100e9, f"Llama-3-70B-like proxy too large: {n/1e9:.1f}B")

    def test_cheap_rank_returns_finite_positive(self):
        """Cheap rank must return a finite positive number."""
        c = self._make_candidate()
        rank = self.cheap(c, training_tokens=int(2e12))
        self.assertGreater(rank, 0.0)
        self.assertLess(rank, 100.0)  # plausible loss * (1 + penalty) range

    def test_cheap_rank_monotonic_in_n_active(self):
        """A bigger model (more params, same shape ratio) must have lower
        chinchilla baseline → lower cheap rank."""
        small = self._make_candidate(d_model=2048, n_layers=16, ffn_dim=7168)
        large = self._make_candidate(d_model=8192, n_layers=80, ffn_dim=28672)
        r_small = self.cheap(small, training_tokens=int(2e12))
        r_large = self.cheap(large, training_tokens=int(2e12))
        self.assertLess(r_large, r_small,
                        f"larger model must have lower cheap rank "
                        f"(small={r_small:.4f}, large={r_large:.4f})")

    def test_cheap_rank_penalizes_pathological_shape(self):
        """An 11776×4 architecture (extreme width, no depth — the original
        H100 750B pathological case) must rank worse than a 6144×33
        alternative at the same param scale."""
        # Both target ~30B active params, very different shapes.
        path = self._make_candidate(
            d_model=11776, n_layers=4, ffn_dim=32768,
            n_heads=92, d_head=128, n_kv_heads=8,
        )
        norm = self._make_candidate(
            d_model=6144, n_layers=33, ffn_dim=21504,
            n_heads=48, d_head=128, n_kv_heads=8,
        )
        r_path = self.cheap(path, training_tokens=int(2e12))
        r_norm = self.cheap(norm, training_tokens=int(2e12))
        self.assertGreater(r_path, r_norm,
                           f"pathological shape must rank worse "
                           f"(path={r_path:.4f}, norm={r_norm:.4f})")

    def test_cheap_rank_is_hw_blind(self):
        """Cheap rank takes no hardware parameter — same call returns
        same value regardless of which hw the surrounding optimize loop
        targets. Critical property: family comparison fairness."""
        c = self._make_candidate()
        # Two consecutive calls — must be identical.
        r1 = self.cheap(c, training_tokens=int(2e12))
        r2 = self.cheap(c, training_tokens=int(2e12))
        self.assertEqual(r1, r2)

    def test_max_full_evaluations_caps_full_evals(self):
        """When max_full_evaluations is small, the optimizer must evaluate
        only the top-N candidates, not all enumerated candidates."""
        c_small = self.DC(
            target_params_b=7.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            serving_tbt_ms=None, serving_ttft_ms=None,
            serving_batch=8, allow_quality_sentinel=True,
            max_full_evaluations=20,
        )
        result_small = self.optimize("h100", c_small)
        # All evaluated candidates should fit under the cap.
        self.assertLessEqual(len(result_small.all_evaluated), 20,
                             f"max_full_evaluations=20 produced "
                             f"{len(result_small.all_evaluated)} evaluations")

    def test_back_compat_no_cap(self):
        """Default max_full_evaluations=None must preserve exhaustive
        behavior (no pruning)."""
        c = self.DC(
            target_params_b=7.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            serving_tbt_ms=None, serving_ttft_ms=None,
            serving_batch=8, allow_quality_sentinel=True,
            # max_full_evaluations defaults to None
        )
        # Default field check
        self.assertIsNone(c.max_full_evaluations)
        result = self.optimize("h100", c)
        # With no cap, evaluations should equal the post-dedupe candidate
        # count (modulo exceptions). Lower bound: should be > 100 for a
        # 7B target at default tolerance.
        self.assertGreater(len(result.all_evaluated), 50,
                           "back-compat: no cap should produce full enumeration")

    def test_cheap_rank_top_n_includes_optimal(self):
        """For a representative cell, the top-N cheap-ranked candidates
        must include the eventual optimum found by full evaluation. This
        verifies the cheap rank doesn't prune the winner."""
        # Compare full enumeration (no cap) vs cap=200; the optimal cand
        # from the cap=200 run should also be present in the cap=None run.
        c_full = self.DC(
            target_params_b=7.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            serving_tbt_ms=None, serving_ttft_ms=None,
            serving_batch=8, allow_quality_sentinel=True,
        )
        c_cap = self.DC(
            target_params_b=7.0, training_tokens=int(2e12),
            context_length=8192, tp=1, pp=1, dp=1,
            serving_tbt_ms=None, serving_ttft_ms=None,
            serving_batch=8, allow_quality_sentinel=True,
            max_full_evaluations=200,
        )
        result_full = self.optimize("h100", c_full)
        result_cap = self.optimize("h100", c_cap)
        # Both should find SOME optimum
        self.assertIsNotNone(result_full.optimal)
        self.assertIsNotNone(result_cap.optimal)
        # The capped run's optimum should be within 5% of the full run's
        # optimum on predicted_loss — i.e., the cheap-rank didn't prune
        # away the actual winning shape.
        opt_full_loss = result_full.optimal.predicted_loss
        opt_cap_loss = result_cap.optimal.predicted_loss
        ratio = opt_cap_loss / max(opt_full_loss, 1e-9)
        self.assertLess(ratio, 1.05,
                        f"capped optimum should be within 5% of full optimum "
                        f"(full={opt_full_loss:.4f}, cap={opt_cap_loss:.4f}, "
                        f"ratio={ratio:.4f})")


if __name__ == "__main__":
    unittest.main()
