"""Wave 6 — optimize_across_contexts.

Locks in the four invariants the spec calls out:

  1. test_one_shape_per_row
     Every per-ctx evaluation carries identical (d_model, n_layers, n_heads,
     d_head, ffn_dim, weight_precision, kv_cache_bits). No shape drift can
     sneak across the row — the architecture is selected ONCE.

  2. test_pareto_dominance_is_joint
     A candidate dominated at every ctx is excluded from the frontier; a
     candidate that wins at one ctx but loses at another is on the frontier
     (i.e., the dominance test compares per-ctx vectors, not scalars).

  3. test_weighted_optimum_picks_long_ctx_when_weighted
     Passing uneven ctx_weights shifts the picked optimum toward the
     heavier ctxs. We don't lock the exact arch — only that the picker
     respects the weights (weighting the long ctx more never makes the
     short-ctx-only candidate the winner when a different candidate has
     materially lower loss at the long ctx).

  4. test_back_compat_single_ctx
     Calling optimize_across_contexts with a single-element ctx_list returns
     a result whose `optimal` matches what plain optimize() would have
     picked at that ctx (modulo per-cell sentinel handling).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ac"))

from optimizer import (  # noqa: E402
    DeploymentConstraints, MultiCtxResult, EvaluatedCandidate,
    optimize, optimize_across_contexts,
    _is_jointly_dominated, _compute_joint_pareto, _pick_joint_optimum,
)


# A small set of constraints reused across most tests — keeps each test
# under a couple of seconds even with the joint per-ctx evaluation.
def _small_constraints(**overrides) -> DeploymentConstraints:
    base = dict(
        target_params_b=1.0,
        training_tokens=int(2e11),
        context_length=8192,
        serving_tbt_ms=None,
        serving_ttft_ms=None,
        serving_batch=4,
        tp_options=[1, 2],
        cp=1,
        cp_options=[1],
        pp=1,
        dp=8,
        max_candidates=20,
        allow_quality_sentinel=True,
    )
    base.update(overrides)
    return DeploymentConstraints(**base)


SHAPE_FIELDS = (
    "d_model", "n_layers", "n_heads", "d_head",
    "ffn_dim", "weight_precision", "kv_cache_bits",
    "ep_degree", "n_kv_heads", "tp_degree",
)


class OneShapePerRowTests(unittest.TestCase):
    """Invariant 1: every per-ctx eval carries identical arch fields."""

    def test_one_shape_per_row(self):
        c = _small_constraints()
        ctx_list = [8192, 32768, 131072]
        r = optimize_across_contexts(
            "h100", c, ctx_list=ctx_list, reference_ctx=131072,
        )
        self.assertIsNotNone(r.optimal,
                             "no row-feasible candidate at 1B / [8k, 32k, 128k]")
        # All per-ctx records must share every shape field.
        for field in SHAPE_FIELDS:
            values = {
                getattr(r.per_ctx_metrics[ctx].arch, field)
                for ctx in ctx_list
            }
            self.assertEqual(
                len(values), 1,
                f"{field} drifted across ctxs: {values}",
            )

    def test_per_ctx_metrics_cover_every_requested_ctx(self):
        c = _small_constraints()
        ctx_list = [8192, 32768]
        r = optimize_across_contexts(
            "h100", c, ctx_list=ctx_list, reference_ctx=8192,
        )
        self.assertIsNotNone(r.optimal)
        self.assertEqual(sorted(r.per_ctx_metrics.keys()), sorted(ctx_list))


class JointParetoDominanceTests(unittest.TestCase):
    """Invariant 2: dominance test is across per-ctx vectors, not scalars."""

    def _mock_eval(self, loss, tps=100.0, tbt=10.0, prefill=20.0,
                   mem=10.0, total_params=1_000_000_000):
        """Build a minimal EvaluatedCandidate stub that satisfies the fields
        _is_jointly_dominated reads. We don't need real arches — the helper
        only touches the 6 numeric objectives."""
        tput = type("Tput", (), {})()
        tput.prefill_time_ms = prefill
        arch = type("Arch", (), {})()
        arch.total_params = total_params
        ev = type("Ev", (), {})()
        ev.predicted_loss = loss
        ev.training_tps = tps
        ev.serving_tbt_ms = tbt
        ev.throughput = tput
        ev.memory_per_gpu_gb = mem
        ev.arch = arch
        return ev

    def test_candidate_dominated_everywhere_is_excluded(self):
        # Two ctxs. Candidate B is strictly better than A at both.
        a_evals = [self._mock_eval(loss=2.0), self._mock_eval(loss=2.5)]
        b_evals = [self._mock_eval(loss=1.5), self._mock_eval(loss=2.0)]
        self.assertTrue(_is_jointly_dominated(a_evals, b_evals))
        pareto = _compute_joint_pareto([a_evals, b_evals])
        self.assertEqual(pareto, [1])  # only B on the frontier

    def test_candidate_winning_at_one_ctx_survives(self):
        # A wins at ctx 0 (loss 1.5 vs B's 2.0); B wins at ctx 1.
        # Neither dominates → both on the frontier.
        a_evals = [self._mock_eval(loss=1.5), self._mock_eval(loss=2.5)]
        b_evals = [self._mock_eval(loss=2.0), self._mock_eval(loss=2.0)]
        self.assertFalse(_is_jointly_dominated(a_evals, b_evals))
        self.assertFalse(_is_jointly_dominated(b_evals, a_evals))
        pareto = _compute_joint_pareto([a_evals, b_evals])
        self.assertEqual(sorted(pareto), [0, 1])

    def test_real_optimizer_returns_nonempty_joint_pareto(self):
        """Smoke test: a real call should produce a non-empty Pareto frontier
        (every row-feasible candidate is on the frontier when there's no
        global winner)."""
        c = _small_constraints()
        r = optimize_across_contexts(
            "h100", c, ctx_list=[8192, 32768], reference_ctx=8192,
        )
        self.assertGreater(len(r.pareto_frontier), 0)


class WeightedOptimumPickerTests(unittest.TestCase):
    """Invariant 3: ctx_weights shifts the picked optimum."""

    def _mock_eval(self, loss):
        tput = type("Tput", (), {})()
        tput.prefill_time_ms = 20.0
        arch = type("Arch", (), {})()
        arch.total_params = 1_000_000_000
        ev = type("Ev", (), {})()
        ev.predicted_loss = loss
        ev.training_tps = 100.0
        ev.serving_tbt_ms = 10.0
        ev.throughput = tput
        ev.memory_per_gpu_gb = 10.0
        ev.arch = arch
        return ev

    def test_uniform_weights_pick_lower_short_ctx(self):
        # Candidate A wins short ctx by a lot, ties long ctx.
        # Candidate B is roughly even.
        ctx_list = [8192, 1048576]
        per_cand = [
            [self._mock_eval(1.0), self._mock_eval(2.0)],  # A: ratio 1.0, 1.0
            [self._mock_eval(2.0), self._mock_eval(2.0)],  # B: ratio 2.0, 1.0
        ]
        pareto = _compute_joint_pareto(per_cand)
        picked = _pick_joint_optimum(pareto, per_cand, ctx_list,
                                     {8192: 1.0, 1048576: 1.0})
        self.assertEqual(picked, 0)  # A wins under uniform weights

    def test_long_ctx_heavy_weights_shift_optimum(self):
        # A wins short by 50%; B wins long by 50%. With long-ctx weight 10x,
        # B should win.
        ctx_list = [8192, 1048576]
        per_cand = [
            [self._mock_eval(1.0), self._mock_eval(3.0)],  # A
            [self._mock_eval(1.5), self._mock_eval(2.0)],  # B
        ]
        pareto = _compute_joint_pareto(per_cand)
        picked_short_heavy = _pick_joint_optimum(
            pareto, per_cand, ctx_list, {8192: 10.0, 1048576: 1.0})
        picked_long_heavy = _pick_joint_optimum(
            pareto, per_cand, ctx_list, {8192: 1.0, 1048576: 10.0})
        # The weight changes must change the winner.
        self.assertNotEqual(picked_short_heavy, picked_long_heavy,
                            "weights must steer the picker")
        # Long-heavy must pick B (the better long-ctx candidate).
        self.assertEqual(picked_long_heavy, 1)


class BackCompatTests(unittest.TestCase):
    """Invariant 4: single-element ctx_list reproduces single-cell optimize()."""

    def test_back_compat_single_ctx(self):
        # Use a determinate, narrow search so the two paths can be expected
        # to land on the same arch.
        c = _small_constraints(max_candidates=15)
        r_multi = optimize_across_contexts(
            "h100", c, ctx_list=[8192], reference_ctx=8192,
        )
        r_single = optimize("h100", c)
        self.assertIsNotNone(r_multi.optimal,
                             "multi-ctx returned no optimum")
        self.assertIsNotNone(r_single.optimal,
                             "single-cell returned no optimum")
        # Same enumeration → same shape pick.
        for field in ("d_model", "n_layers", "n_heads", "d_head",
                      "ffn_dim", "weight_precision", "kv_cache_bits",
                      "tp_degree"):
            self.assertEqual(
                getattr(r_multi.optimal.arch, field),
                getattr(r_single.optimal.arch, field),
                f"{field} diverged between single-ctx and multi-ctx paths",
            )


if __name__ == "__main__":
    unittest.main()
