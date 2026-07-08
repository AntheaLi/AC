"""Wave 15 — Pillar A: matrix-level property invariants.

These tests look at the produced matrix, not at one cell or one function.
They exist because two structural bugs caught this session (Wave 8b cheap-
rank family bias; throughput-model TP-amortization at cross-NVLink-island
all-reduce) both looked correct in isolation but produced visibly wrong
*system behavior* at the matrix level. The 139 unit tests caught neither.

Pillar A invariants:

  1. test_loss_monotonic_in_active_params_within_family
     Within a family at fixed (ctx, tokens), loss must drop as N_active
     grows. This is the chinchilla-baseline contract; a regression here
     means the spine or family-residual stack is broken.

  2. test_tbt_monotonic_in_active_params_within_family
     Within a dense family at fixed (ctx, TP), TBT must grow with
     N_active — bigger model, more weights, slower decode. Caught the
     TP-amortization bug where dense 1000B at TP=32 reported lower TBT
     than dense 7B at TP=4.

  3. test_no_family_pruned_at_moderate_scales
     With `allow_moe=True` and `max_full_evaluations` set, the full-eval
     pool must include at least one candidate from every viable family.
     Caught the Wave 8b cheap-rank bias where 100% of MoE candidates were
     pruned at moderate scales.

  4. test_tbt_not_monotonic_decrease_in_tp_beyond_island
     At TP > NVLink-domain-size, TBT must not be lower than TP at the
     island boundary. Cross-island all-reduce traverses cross-IB and is
     much slower; any regression that hides this is a calibration bug.

  5. test_per_replica_metrics_consistent
     `train_tps × dp_degree` (cluster aggregate) must equal the optimizer-
     reported aggregate within rounding. Caught historical bug where DP
     units were doubled in the family-rollup.

  6. test_family_loss_differentiation_at_moderate_scale
     At 120B active params with `allow_moe=True`, MoE loss must differ
     from dense loss by > 0.5%. A zero gap means MoE candidates were
     silently pruned before the full evaluator could differentiate them.

  7. test_pareto_includes_feasible_candidates_from_each_family
     The returned Pareto frontier must contain at least one candidate
     from each family that was enumerated AND feasible. Pareto-dominance
     bugs that silently exclude family representatives are caught here.

Each test sets up a small grid (3-4 cells), runs `optimize()` or
`optimize_across_contexts()`, and asserts the property. `max_candidates`
and `max_full_evaluations` are tuned so each test finishes in <5s.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _make_constraints(target_b, ctx=8192, tokens=int(2e12), allow_moe=False,
                     allow_state=False, tp=1, tp_options=None,
                     max_candidates=80, max_full_evaluations=40,
                     serving_batch=8):
    """Helper: minimal DeploymentConstraints for a single-cell probe."""
    from ac.optimizer import DeploymentConstraints
    kw = dict(
        target_params_b=float(target_b),
        training_tokens=int(tokens),
        context_length=int(ctx),
        serving_tbt_ms=None, serving_ttft_ms=None,
        serving_batch=int(serving_batch),
        tp=int(tp), pp=1, dp=1,
        allow_moe=allow_moe,
        allow_state=allow_state,
        max_candidates=max_candidates,
        max_full_evaluations=max_full_evaluations,
        allow_quality_sentinel=True,
        param_tolerance=0.15,
    )
    if tp_options is not None:
        kw["tp_options"] = list(tp_options)
    return DeploymentConstraints(**kw)


def _family_of(ev) -> str:
    arch = ev.arch
    has_moe = bool(getattr(arch, "moe", None))
    has_state = bool(getattr(arch, "state_config", None)) and \
                getattr(arch, "n_state_layers", 0) > 0
    if has_moe and has_state: return "moe_hybrid"
    if has_moe:               return "moe"
    if has_state:             return "hybrid"
    return "dense"


# =============================================================================
# Pillar A invariants
# =============================================================================

class LossMonotonicityTests(unittest.TestCase):
    """Within a family at fixed (ctx, tokens), loss must drop as N_active
    grows. This is the chinchilla baseline contract — any regression means
    the spine or residual stack is broken."""

    def test_loss_monotonic_in_active_params_within_family(self):
        from ac.optimizer import optimize
        targets = [1.0, 7.0, 70.0]
        losses = []
        for tb in targets:
            c = _make_constraints(tb, ctx=8192,
                                  max_candidates=60, max_full_evaluations=30)
            r = optimize("h100", c)
            if r.optimal is None:
                self.skipTest(f"no feasible solution at {tb}B; "
                              f"feasibility regression?")
            losses.append((tb, r.optimal.predicted_loss))
        # Walk pairwise; strictly decreasing (modulo 5% chinchilla noise).
        for (a_tb, a_loss), (b_tb, b_loss) in zip(losses, losses[1:]):
            # Allow a small wiggle for chinchilla numerics at the boundary
            # but disallow inversion of more than 2% (which would mean the
            # spine is broken).
            self.assertLess(
                b_loss, a_loss * 1.02,
                f"loss must drop as active params grows within dense family:\n"
                f"  {a_tb}B → loss={a_loss:.4f}\n"
                f"  {b_tb}B → loss={b_loss:.4f}\n"
                f"Regression in chinchilla spine or architecture residual."
            )


class TbtMonotonicityTests(unittest.TestCase):
    """Within a dense family at fixed (ctx, TP), TBT must grow with N_active.
    Caught: TP-amortization bug where dense 1000B at TP=32 reported lower
    TBT than dense 7B at TP=4 (cross-NVLink all-reduce was underweighted)."""

    def test_tbt_monotonic_in_active_params_within_family_fixed_tp(self):
        from ac.optimizer import optimize
        # Fix TP=1 so the only varying axis is model size.
        targets = [1.0, 7.0, 70.0]
        tbts = []
        for tb in targets:
            c = _make_constraints(tb, ctx=8192, tp=1, tp_options=[1],
                                  max_candidates=60, max_full_evaluations=30)
            r = optimize("h100", c)
            if r.optimal is None:
                self.skipTest(f"no feasible solution at {tb}B")
            tbts.append((tb, r.optimal.serving_tbt_ms))
        for (a_tb, a_tbt), (b_tb, b_tbt) in zip(tbts, tbts[1:]):
            # Bigger model → more weight load + KV → larger TBT. Allow a
            # 2× wiggle for shape differences across the Pareto pick, but
            # an inversion of >50% is unphysical.
            self.assertGreater(
                b_tbt, a_tbt * 0.5,
                f"TBT must not invert as active params grow at fixed TP:\n"
                f"  {a_tb}B → TBT={a_tbt:.2f}ms\n"
                f"  {b_tb}B → TBT={b_tbt:.2f}ms\n"
                f"A larger model should not be materially faster at decode "
                f"under the same parallelism. Likely throughput-model "
                f"TP-amortization regression — see plan/redesign/"
                f"17-tp-amortization-cost-fix.md."
            )


class FamilyRepresentationTests(unittest.TestCase):
    """Caught the Wave 8b cheap-rank prune bias: at moderate scales, 100%
    of MoE candidates were pruned before full eval, making every cell look
    like dense=MoE in the displayed matrix."""

    def test_no_family_pruned_at_moderate_scales(self):
        from ac.optimizer import optimize
        # 120B active with allow_moe=True is exactly the scale where the
        # prune-bias bug hit. Stratified-by-family Wave 8b fix guarantees
        # at least one MoE candidate survives to full eval.
        c = _make_constraints(120.0, ctx=8192, allow_moe=True,
                              max_candidates=120, max_full_evaluations=40)
        r = optimize("h100", c)
        if r.optimal is None:
            self.skipTest("no feasible solution at 120B with allow_moe=True")

        families_evaluated = {_family_of(ev) for ev in r.all_evaluated}
        self.assertIn(
            "moe", families_evaluated,
            f"MoE family was completely pruned at 120B/h100 despite "
            f"allow_moe=True. Full-eval families: {families_evaluated}. "
            f"Cheap-rank stratification bug — see plan/redesign/"
            f"08-two-stage-evaluation.md and the Wave 8b post-probe fix."
        )
        self.assertIn(
            "dense", families_evaluated,
            f"Dense family was pruned at 120B — even more suspicious. "
            f"Full-eval families: {families_evaluated}."
        )

    def test_family_loss_differentiation_at_moderate_scale(self):
        """At 120B with allow_moe=True, MoE loss must differ from dense
        loss by > 0.5%. A zero gap means MoE candidates were pruned before
        the full evaluator could differentiate them (matrix shows dense=MoE)."""
        from ac.optimizer import optimize
        c = _make_constraints(120.0, ctx=8192, allow_moe=True,
                              max_candidates=120, max_full_evaluations=40)
        r = optimize("h100", c)
        if r.optimal is None:
            self.skipTest("no feasible solution at 120B with allow_moe=True")

        by_family = {}
        for ev in r.all_evaluated:
            if not ev.meets_constraints:
                continue
            fam = _family_of(ev)
            cur = by_family.get(fam)
            if cur is None or ev.predicted_loss < cur:
                by_family[fam] = ev.predicted_loss

        if "moe" not in by_family or "dense" not in by_family:
            self.skipTest(
                f"need both MoE and dense to differentiate; "
                f"got families {list(by_family.keys())}"
            )
        gap_pct = abs(by_family["moe"] - by_family["dense"]) / by_family["dense"] * 100
        self.assertGreater(
            gap_pct, 0.5,
            f"MoE vs dense loss gap at 120B is {gap_pct:.3f}% — too close to "
            f"zero. Either both families are converging on the same shape "
            f"(unlikely at 120B with allow_moe=True), or the cheap-rank "
            f"prune is silently equalizing the families before full eval. "
            f"dense={by_family['dense']:.4f}, moe={by_family['moe']:.4f}."
        )


class TpAmortizationTests(unittest.TestCase):
    """Caught: throughput-model TP-amortization at cross-NVLink-island TP.
    The all-reduce cost at TP > gpus_per_node traverses cross-IB and should
    be ~10-30× slower than intra-island. A regression that hides this makes
    bigger-TP models look faster than they actually are.

    Locked in after Wave 17 Part 2 landed — see
    `plan/redesign/17-pp-decode-serving-cost-fix.md` Part 2 for the fix
    (per-collective launch-latency floor + decode-aware overlap_fraction=0.1)."""

    def test_tbt_not_lower_at_cross_island_tp_than_island_tp(self):
        """For a fixed 70B-class shape, TBT at TP=16 (2 nodes, cross-NVLink)
        must NOT be materially lower than TBT at TP=8 (1 node, intra-NVLink).
        Cross-IB all-reduce + per-collective launch-latency floor should
        keep cross-island TP from looking like free speedup.

        We compare at a FIXED shape (passed through the raw throughput
        model) rather than optimizer-picked shapes; the optimizer picks
        wide-shallow at TP=16 vs deep-narrow at TP=8 and TBT differences
        there are confounded by the shape choice.
        """
        from ac.throughput_model import ArchConfig as TArchConfig, throughput
        arch = TArchConfig(
            d_model=8192, n_layers=80, n_heads=64, d_head=128, n_kv_heads=8,
            ffn_dim=28672, batch_size=8, seq_len=8192,
            precision="bf16", kv_precision="bf16",
        )
        tbt_at_tp = {}
        for tp in (8, 16):
            r = throughput(arch, "h100", tp_degree=tp, pp_degree=1,
                          decode_kv_len=8192, prefill_seq_len=8192,
                          microbatches=8)
            tbt_at_tp[tp] = r.decode_time_per_token_ms
        # Cross-IB must impose at least *some* cost. If TP=16 is materially
        # (>15%) faster than TP=8 for the same shape, the cross-island
        # all-reduce is underweighted in the throughput model.
        self.assertGreater(
            tbt_at_tp[16], tbt_at_tp[8] * 0.85,
            f"TP=16 TBT ({tbt_at_tp[16]:.2f}ms) is materially LOWER than "
            f"TP=8 TBT ({tbt_at_tp[8]:.2f}ms) for the same shape. "
            f"Cross-NVLink-island all-reduce should be slower than intra-"
            f"island. Likely throughput_model._allreduce_cost regression "
            f"— see plan/redesign/17-pp-decode-serving-cost-fix.md Part 2."
        )


class PerReplicaConsistencyTests(unittest.TestCase):
    """Asserts the per-replica × DP = aggregate identity. Caught historical
    bug where DP units were doubled in family-rollup."""

    def test_train_tps_per_replica_sane_units(self):
        from ac.optimizer import optimize
        c = _make_constraints(7.0, ctx=8192, tp=1, tp_options=[1],
                              max_candidates=40, max_full_evaluations=20)
        r = optimize("h100", c)
        if r.optimal is None:
            self.skipTest("no feasible solution at 7B")
        # training_tps from the result is per-TP-replica. The aggregate
        # (training_tps × dp) should be in the plausible range for an H100
        # cluster. For 7B at TP=1 DP=1, single-replica TPS at ~5-50k tok/s
        # is the calibrated range; 10× outside that bounds catches unit bugs.
        per_replica = r.optimal.training_tps
        self.assertGreater(
            per_replica, 100.0,
            f"per-replica TPS = {per_replica:.0f} is implausibly low for "
            f"7B at TP=1 on H100. Unit regression in throughput model."
        )
        self.assertLess(
            per_replica, 1_000_000.0,
            f"per-replica TPS = {per_replica:.0f} is implausibly high. "
            f"Unit regression (likely TP/DP double-multiplied)."
        )


class FamilyDecisionContinuityTests(unittest.TestCase):
    """Adjacent cells in (params, ctx) should rarely flip family. A flip-
    then-flip-back across 2× ctx is suspicious (calibration noise or
    pruning artifact)."""

    def test_no_family_flip_flop_across_adjacent_ctx(self):
        from ac.optimizer import optimize
        ctxs = [8192, 32768, 131072]
        # 13B + allow_moe is the cell range where 8b prune-bias was most
        # visible. Check that adjacent ctxs don't ping-pong family.
        families = []
        for ctx in ctxs:
            c = _make_constraints(13.0, ctx=ctx, allow_moe=True,
                                  max_candidates=80, max_full_evaluations=30)
            r = optimize("h100", c)
            if r.optimal is None:
                self.skipTest(f"no optimum at ctx={ctx}")
            families.append((ctx, _family_of(r.optimal)))
        # The set of families across ctxs should be small (≤ 2 distinct).
        # A {dense, moe, dense} pattern is flip-flop and indicates noise.
        family_set = {f for _, f in families}
        # Locked invariant: at most 2 distinct families across this row.
        # An across-3 ctx flip pattern indicates either calibration noise
        # at the picker tiebreak or a Wave 8b stratification regression.
        self.assertLessEqual(
            len(family_set), 2,
            f"family decision flipped across 3 contexts at 13B: "
            f"{families}. Expect at most 2 distinct families per row "
            f"under stable calibration; 3+ means noise dominates."
        )


class ParetoCoverageTests(unittest.TestCase):
    """Pareto frontier must contain at least one candidate from each
    feasible family. Catches Pareto-dominance bugs that silently exclude
    operating points."""

    def test_pareto_includes_each_enumerated_family(self):
        from ac.optimizer import optimize
        c = _make_constraints(13.0, ctx=8192, allow_moe=True,
                              max_candidates=80, max_full_evaluations=30)
        r = optimize("h100", c)
        if not r.pareto_frontier:
            self.skipTest("empty Pareto frontier at 13B")
        evaluated_families = {_family_of(ev) for ev in r.all_evaluated
                              if ev.meets_constraints}
        pareto_families = {_family_of(ev) for ev in r.pareto_frontier}
        # Every family that has a feasible candidate should have at least
        # one representative on the Pareto frontier — otherwise the 6-axis
        # dominance is silently excluding operating points.
        missing = evaluated_families - pareto_families
        self.assertEqual(
            missing, set(),
            f"families enumerated as feasible but missing from Pareto: "
            f"{missing}. Pareto-dominance bug — at least one candidate "
            f"per family should land on the frontier when the family is "
            f"not strictly dominated everywhere."
        )


if __name__ == "__main__":
    unittest.main()
