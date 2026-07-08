"""Wave 15 — specific-bug regression tests.

Each test below corresponds to a numbered bug from `plan/redesign/
15-scrutinization-and-validation.md`. They're separate from the
Pillar-A property tests so a single test failure points at exactly one
historical bug class.

  1. Wave 8b cheap-rank family bias
     → test_each_family_represented_in_top_n
  2. TP-amortization TBT inversion
     → test_tbt_not_monotonic_in_tp_beyond_island
  3. Dense-tie at moderate scale
     → test_family_loss_differentiation_at_moderate_scale
  4. Throughput model MoE wiring KeyError
     → test_moe_candidate_evaluates_without_keyerror
  5. Cheap-rank monotonicity within family
     → test_cheap_rank_within_family_monotonic
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _family_of_arch(arch) -> str:
    has_moe = bool(getattr(arch, "moe", None))
    has_state = bool(getattr(arch, "state_config", None)) and \
                getattr(arch, "n_state_layers", 0) > 0
    if has_moe and has_state: return "moe_hybrid"
    if has_moe:               return "moe"
    if has_state:             return "hybrid"
    return "dense"


# -----------------------------------------------------------------------------
# Bug 1 — Wave 8b cheap-rank family bias
# -----------------------------------------------------------------------------

class CheapRankFamilyBiasTests(unittest.TestCase):
    """The original Wave 8b cheap-rank was family-blind: MoE candidates
    carry a negative `moe_residual.capacity_bonus` in the full evaluator
    that the cheap rank didn't account for, so MoE was systematically
    pruned at moderate scales (e.g. 120B target → 0 MoE candidates in
    top-40 cheap-ranked). The stratified-by-family fix splits the budget
    so every family gets equal full-eval representation.

    This test runs the optimizer at the canonical 120B/allow_moe=True
    cell and asserts at least one MoE candidate survives to full eval.
    """

    def test_each_family_represented_in_top_n(self):
        from ac.optimizer import DeploymentConstraints, optimize
        c = DeploymentConstraints(
            target_params_b=120.0,
            training_tokens=int(2e12),
            context_length=8192,
            tp=1, pp=1, dp=1,
            serving_tbt_ms=None, serving_ttft_ms=None,
            serving_batch=8,
            allow_moe=True, allow_state=False,
            max_candidates=200,
            max_full_evaluations=40,
            allow_quality_sentinel=True,
            param_tolerance=0.15,
        )
        r = optimize("h100", c)
        if r.optimal is None:
            self.skipTest("no feasible solution")

        families = {_family_of_arch(ev.arch) for ev in r.all_evaluated}
        # Stratified-by-family Wave 8b fix guarantees both dense and MoE
        # land in the full-eval pool when allow_moe=True at moderate scale.
        self.assertIn("dense", families,
                      f"dense family pruned away; full-eval families={families}")
        self.assertIn("moe", families,
                      f"MoE family pruned away despite allow_moe=True; "
                      f"full-eval families={families}. Regression of the "
                      f"Wave 8b family-blind cheap rank.")


# -----------------------------------------------------------------------------
# Bug 2 — TP-amortization TBT inversion
# -----------------------------------------------------------------------------

class TpAmortizationTbtTests(unittest.TestCase):
    """The throughput model's `_allreduce_cost` switches to inter-node
    bandwidth at TP > gpus_per_node. The original Wave 1 fix for TP
    overlap (0.5 default) was applied uniformly, making cross-island
    all-reduces look much cheaper than physics allows. Result: dense 1000B
    at TP=32 reported lower TBT than dense 7B at TP=4.

    Locked in after Wave 17 Part 2 landed: per-collective launch latency
    floor (~6 µs intra, ~14 µs cross-IB) + decode-aware overlap_fraction
    (0.1 vs 0.5 default) — see `plan/redesign/17-pp-decode-serving-cost-fix.md`
    Part 2 for the implementation."""

    def test_tbt_not_monotonic_in_tp_beyond_island(self):
        """Same-shape comparison: pinning the model to 70B-class, TBT at
        TP=16 (2 nodes, cross-NVLink) must not undercut TBT at TP=8 (1
        node). We use the raw throughput model with a fixed shape so the
        comparison isn't confounded by the optimizer picking different
        shapes per TP."""
        from ac.throughput_model import ArchConfig as TArchConfig, throughput
        arch = TArchConfig(
            d_model=8192, n_layers=80, n_heads=64, d_head=128, n_kv_heads=8,
            ffn_dim=28672, batch_size=8, seq_len=8192,
            precision="bf16", kv_precision="bf16",
        )
        def _run(tp):
            r = throughput(arch, "h100", tp_degree=tp, pp_degree=1,
                          decode_kv_len=8192, prefill_seq_len=8192,
                          microbatches=8)
            return r.decode_time_per_token_ms
        tbt_8 = _run(8)
        tbt_16 = _run(16)
        self.assertGreater(
            tbt_16, tbt_8 * 0.85,
            f"70B@TP=16 TBT ({tbt_16:.2f}ms) materially under-cuts "
            f"70B@TP=8 TBT ({tbt_8:.2f}ms) for the same shape. "
            f"Cross-NVLink-island AR should be slower than intra-island; "
            f"see plan/redesign/17-pp-decode-serving-cost-fix.md Part 2."
        )


# -----------------------------------------------------------------------------
# Bug 3 — Dense-tie at moderate scale (covered in Pillar A, included here
# for completeness so this file is the canonical bug-by-bug list)
# -----------------------------------------------------------------------------

class DenseTieAtModerateScaleTests(unittest.TestCase):
    """When the cheap-rank prune was family-blind, 120B + allow_moe=True
    showed dense and MoE with identical loss (within 0.01%) — the matrix
    silently labeled both families as dead-ties. The stratified fix
    surfaces real family differences."""

    def test_family_loss_differentiation_at_moderate_scale(self):
        from ac.optimizer import DeploymentConstraints, optimize
        c = DeploymentConstraints(
            target_params_b=120.0,
            training_tokens=int(2e12),
            context_length=8192,
            tp=1, pp=1, dp=1,
            serving_tbt_ms=None, serving_ttft_ms=None,
            serving_batch=8,
            allow_moe=True, allow_state=False,
            max_candidates=200,
            max_full_evaluations=40,
            allow_quality_sentinel=True,
            param_tolerance=0.15,
        )
        r = optimize("h100", c)
        if r.optimal is None:
            self.skipTest("no feasible solution")
        by_family = {}
        for ev in r.all_evaluated:
            if not ev.meets_constraints:
                continue
            fam = _family_of_arch(ev.arch)
            if fam not in by_family or ev.predicted_loss < by_family[fam]:
                by_family[fam] = ev.predicted_loss
        if "dense" not in by_family or "moe" not in by_family:
            self.skipTest(f"need both families; got {list(by_family.keys())}")
        gap_pct = abs(by_family["moe"] - by_family["dense"]) / by_family["dense"] * 100
        # Wave 18h: the signed data-sufficiency gate makes dense and MoE
        # legitimately CONVERGE near the tokens-per-total-param parity point
        # (120B active at 2T sits close to it), so demanding a 0.5% gap now
        # over-constrains correct physics. The bug this test guards against
        # — a family-blind cheap-rank prune returning IDENTICAL losses —
        # has a <0.01% signature; 0.1% keeps that guard intact.
        self.assertGreater(
            gap_pct, 0.1,
            f"dense and MoE loss within {gap_pct:.3f}% at 120B — dense-tie "
            f"regression. dense={by_family['dense']:.4f}, "
            f"moe={by_family['moe']:.4f}."
        )


# -----------------------------------------------------------------------------
# Bug 4 — Throughput model MoE wiring KeyError
# -----------------------------------------------------------------------------

class MoeWiringTests(unittest.TestCase):
    """When constructing CandidateArch directly with a populated moe
    dict, the throughput model used to KeyError on `expert_dim` because
    the schema-v1 nested shape (n_experts, top_k, expert_dim) was
    inconsistently accessed. Fixed by Wave 3; this test guards against
    regression."""

    def test_moe_candidate_evaluates_without_keyerror(self):
        from ac.optimizer import DeploymentConstraints, evaluate_candidate, CandidateArch
        c = DeploymentConstraints(
            target_params_b=7.0,
            training_tokens=int(2e12),
            context_length=8192,
            tp=1, pp=1, dp=1,
            serving_tbt_ms=None, serving_ttft_ms=None,
            serving_batch=8,
            allow_moe=True,
            allow_quality_sentinel=True,
        )
        cand = CandidateArch(
            d_model=2048, n_layers=24, n_heads=16, d_head=128, n_kv_heads=8,
            ffn_dim=8192, vocab_size=128000,
            weight_precision="bf16", ffn_precision="bf16",
            attn_precision={"q":"bf16","k":"bf16","v":"bf16","o":"bf16"},
            kv_cache_bits=16,
            moe={
                "n_experts": 8, "top_k": 2, "expert_dim": 8192,
                "shared_expert": None,
                "router": {"precision": "bf16"},
                "capacity_factor": 1.0,
                "precision": "bf16",
            },
            moe_style="fine",
            state_config=None, n_dense_ffn_layers=0,
            attention_type="gqa",
            tp_degree=1, cp_degree=1, ep_degree=2,
        )
        try:
            ev = evaluate_candidate(cand, "h100", c)
        except KeyError as exc:
            self.fail(
                f"evaluate_candidate KeyError on MoE candidate: {exc}. "
                f"Wave 3 MoE wiring regression — likely missing/renamed "
                f"key in the moe dict's nested schema."
            )
        self.assertIsNotNone(ev)


# -----------------------------------------------------------------------------
# Bug 5 — Cheap-rank monotonicity within family
# -----------------------------------------------------------------------------

class CheapRankWithinFamilyTests(unittest.TestCase):
    """Within a single family, the cheap-rank order should roughly match
    the full predicted_loss order. The analog of the existing cross-family
    test (`test_two_stage_evaluation.test_cheap_rank_monotonic_in_n_active`)
    but within-family — confirms that the family-conditional bonus didn't
    accidentally break the same-family ordering."""

    def test_cheap_rank_within_family_monotonic(self):
        from ac.optimizer import (
            DeploymentConstraints, CandidateArch,
            _cheap_quality_rank, evaluate_candidate,
        )
        c = DeploymentConstraints(
            target_params_b=7.0,
            training_tokens=int(2e12),
            context_length=8192,
            tp=1, pp=1, dp=1,
            serving_tbt_ms=None, serving_ttft_ms=None,
            serving_batch=8,
            allow_quality_sentinel=True,
        )
        # Two dense candidates at very different sizes.
        small = CandidateArch(
            d_model=2048, n_layers=16, n_heads=16, d_head=128, n_kv_heads=8,
            ffn_dim=8192, vocab_size=128000,
            weight_precision="bf16", ffn_precision="bf16",
            attn_precision={"q":"bf16","k":"bf16","v":"bf16","o":"bf16"},
            kv_cache_bits=16,
            attention_type="gqa",
            tp_degree=1, cp_degree=1, ep_degree=1,
        )
        large = CandidateArch(
            d_model=8192, n_layers=80, n_heads=64, d_head=128, n_kv_heads=8,
            ffn_dim=28672, vocab_size=128000,
            weight_precision="bf16", ffn_precision="bf16",
            attn_precision={"q":"bf16","k":"bf16","v":"bf16","o":"bf16"},
            kv_cache_bits=16,
            attention_type="gqa",
            tp_degree=1, cp_degree=1, ep_degree=1,
        )
        rank_small = _cheap_quality_rank(small, int(2e12))
        rank_large = _cheap_quality_rank(large, int(2e12))
        self.assertLess(
            rank_large, rank_small,
            f"larger dense model should cheap-rank lower (better) than "
            f"smaller: small_rank={rank_small:.4f}, "
            f"large_rank={rank_large:.4f}. Within-family monotonicity "
            f"regression."
        )

        # And the full eval should agree on direction (modulo tie-breaks).
        ev_small = evaluate_candidate(small, "h100", c)
        ev_large = evaluate_candidate(large, "h100", c)
        if ev_small.meets_constraints and ev_large.meets_constraints:
            self.assertLess(
                ev_large.predicted_loss, ev_small.predicted_loss,
                f"full eval disagrees with cheap rank on within-family "
                f"order: small_loss={ev_small.predicted_loss:.4f}, "
                f"large_loss={ev_large.predicted_loss:.4f}."
            )


if __name__ == "__main__":
    unittest.main()
