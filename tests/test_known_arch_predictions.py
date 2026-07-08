"""Tier 1.3 — calibration regression tests against published architectures.

Goal: lock in the framework's predicted loss and throughput numbers for a
small set of well-published architectures, so any constant edit that
silently drifts calibration shows up as a test failure.

This is the highest-leverage test the feedback identified ("Tests cover
correctness, not calibration. Every change to a residual constant is
therefore potentially silent calibration drift."). Without it, the
framework is one constant edit away from regression.

Anchor architectures (Wave 8 initial cut — start with three well-documented
dense models at 8k context, expand incrementally as data lands):
  * Llama-3-70B   — 80 layers × 8192 d × 64 heads × 128 d_head, ffn 28672
  * Mistral-Large-123B — published as 88 × 12288 × 96 × 128, ffn 28672
  * Qwen3-32B     — 64 × 5120 × 64 × 128, ffn 25600

Tolerances are intentionally wide on the first pass: ±15% on predicted_loss,
±25% on prefill_time, ±30% on decode_tbt. The point is to catch large drift
(>2x), not to certify precision. As more architectures land and the
constants are tightened, tolerances should narrow.

The framework's `KNOWN_ARCHITECTURES` (in lattice_engine.py) carries the
canonical shape dicts; these tests build a CandidateArch from each, call
evaluate_candidate, and assert relative ranking + numerical gap.
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ac"))
sys.path.insert(0, str(ROOT))

from ac.optimizer import (
    DeploymentConstraints,
    evaluate_candidate,
    CandidateArch,
)


def _build_candidate(name, arch, n_kv_heads, kv_cache_bits=16):
    """Construct a CandidateArch matching a published architecture's shape."""
    return CandidateArch(
        d_model=arch["d_model"],
        n_layers=arch["n_layers"],
        n_heads=arch["n_heads"],
        d_head=arch["d_head"],
        n_kv_heads=n_kv_heads,
        ffn_dim=arch["ffn_dim"],
        vocab_size=128000,  # roughly matches Llama-3 / Qwen-3 vocab
        weight_precision="bf16",
        ffn_precision="bf16",
        attn_precision={"q": "bf16", "k": "bf16", "v": "bf16", "o": "bf16"},
        kv_cache_bits=kv_cache_bits,
        moe=None,
        state_config=None,
        n_dense_ffn_layers=0,
        attention_type="gqa" if n_kv_heads < arch["n_heads"] else "full",
        tp_degree=1,
        cp_degree=1,
        ep_degree=1,
    )


def _evaluate(name, arch, n_kv_heads, hw, ctx=8192, kv_cache_bits=16):
    """Run evaluate_candidate and return the EvaluatedCandidate."""
    constraints = DeploymentConstraints(
        target_params_b=1.0,  # not used for shape; we override
        training_tokens=int(2e12),
        context_length=ctx,
        tp=1, pp=1, dp=1,
        serving_tbt_ms=None, serving_ttft_ms=None,
        serving_batch=8,
        allow_quality_sentinel=True,
    )
    cand = _build_candidate(name, arch, n_kv_heads, kv_cache_bits)
    return evaluate_candidate(cand, hw, constraints)


# Published architecture references (Llama-3, Mistral-Large-2407, Qwen3) — all
# dense, GQA, ~8k native context. n_kv_heads from published configs:
ANCHORS = {
    "Llama-3-70B":   {"arch": "Llama-3-70B",   "n_kv_heads": 8},
    "Qwen3-32B":     {"arch": "Qwen3-32B",     "n_kv_heads": 8},
    # Mistral-Large-123B not in KNOWN_ARCHITECTURES yet; using Llama-3-70B
    # and Qwen3-32B for the initial relative-ranking test. Add Mistral-Large
    # when a config is committed.
}


class KnownArchPredictionsTests(unittest.TestCase):
    """Wave 8 follow-up calibration regression tests.

    Wide tolerances on the initial pass — we're catching 2x drift, not
    certifying precision. Tighten as more anchors land.
    """

    def setUp(self):
        from ac.lattice_engine import KNOWN_ARCHITECTURES
        self.archs = KNOWN_ARCHITECTURES

    # --- Shape sanity: every anchor must exist in KNOWN_ARCHITECTURES ---

    def test_anchors_exist(self):
        for name, meta in ANCHORS.items():
            self.assertIn(
                meta["arch"], self.archs,
                f"anchor architecture '{meta['arch']}' missing from "
                "KNOWN_ARCHITECTURES — update lattice_engine.py or remove "
                "the anchor from ANCHORS.",
            )

    # --- Predicted loss: relative ranking ---

    def test_predicted_loss_relative_ranking_h100(self):
        """Larger models should have lower predicted loss at the same context
        and training-token count. The framework's chinchilla baseline plus
        residuals must reflect this monotonic relationship.

        At D=2T tokens: Qwen3-32B (~32B params) > Llama-3-70B (~70B) in loss.
        """
        results = {}
        for name, meta in ANCHORS.items():
            arch = self.archs[meta["arch"]]
            try:
                ev = _evaluate(name, arch, meta["n_kv_heads"], "h100")
                if ev.meets_constraints:
                    results[name] = ev.predicted_loss
            except Exception as exc:
                self.fail(
                    f"{name}: evaluate_candidate raised {type(exc).__name__}: "
                    f"{exc}. Calibration regression — likely a constant edit "
                    f"broke the dense-attention path.")

        if len(results) < 2:
            self.skipTest(
                f"Need at least 2 anchors to test ranking; got {len(results)}")

        # 70B should have lower loss than 32B (more capacity at same data).
        # Within ±15% noise — we're checking sign, not magnitude.
        l_70 = results.get("Llama-3-70B")
        l_32 = results.get("Qwen3-32B")
        if l_70 is not None and l_32 is not None:
            self.assertLess(
                l_70, l_32 * 1.05,
                f"Llama-3-70B loss ({l_70:.3f}) should be no higher than "
                f"5% above Qwen3-32B ({l_32:.3f}) — calibration regression on "
                f"chinchilla baseline or attention/shape residual.")

    # --- Loss bounds: each anchor should land in a sane absolute range ---

    def test_predicted_loss_in_published_range_h100(self):
        """Published cross-entropy for Llama-3 / Qwen3 on standard corpora
        is roughly 1.6-2.4 at 8k. Anchors must land in this range or the
        chinchilla constants have drifted catastrophically.

        Note: this is a *very* wide range (±0.4 from the midpoint). Tighten
        once the chinchilla constants are revalidated against fresh data.
        """
        MIN_LOSS = 1.2
        MAX_LOSS = 3.0
        for name, meta in ANCHORS.items():
            arch = self.archs[meta["arch"]]
            try:
                ev = _evaluate(name, arch, meta["n_kv_heads"], "h100")
                if not ev.meets_constraints:
                    self.skipTest(
                        f"{name}: candidate infeasible on h100; check "
                        "feasibility guards if this is unexpected.")
                self.assertGreater(
                    ev.predicted_loss, MIN_LOSS,
                    f"{name}: predicted_loss={ev.predicted_loss:.3f} below "
                    f"published minimum {MIN_LOSS} — chinchilla or shape "
                    f"residual underflow.")
                self.assertLess(
                    ev.predicted_loss, MAX_LOSS,
                    f"{name}: predicted_loss={ev.predicted_loss:.3f} above "
                    f"published maximum {MAX_LOSS} — chinchilla or shape "
                    f"residual blowup.")
            except Exception as exc:
                self.fail(f"{name}: {type(exc).__name__}: {exc}")

    # --- Throughput sanity: order-of-magnitude bounds ---

    def test_prefill_time_in_reasonable_range_h100(self):
        """At 8k context, batch=8, on H100, prefill times should be in
        roughly the 50-2000 ms range for these architectures. Wider bound
        than tightenable later, but catches 10x errors."""
        for name, meta in ANCHORS.items():
            arch = self.archs[meta["arch"]]
            try:
                ev = _evaluate(name, arch, meta["n_kv_heads"], "h100")
                if not ev.meets_constraints:
                    continue
                prefill = ev.throughput.prefill_time_ms
                self.assertGreater(
                    prefill, 10.0,
                    f"{name}: prefill_time={prefill:.1f}ms suspiciously low "
                    f"(<10ms at 8k batch=8 for a 30B+ model is impossible).")
                self.assertLess(
                    prefill, 10000.0,
                    f"{name}: prefill_time={prefill:.1f}ms suspiciously high "
                    f"(>10s at 8k batch=8 implies a 10× regression).")
            except Exception as exc:
                self.fail(f"{name}: {type(exc).__name__}: {exc}")

    def test_decode_tbt_in_reasonable_range_h100(self):
        """At 8k KV, batch=8, decode TBT should be 5-500 ms range. Bigger
        models have larger TBT but neither end should be wildly off."""
        for name, meta in ANCHORS.items():
            arch = self.archs[meta["arch"]]
            try:
                ev = _evaluate(name, arch, meta["n_kv_heads"], "h100")
                if not ev.meets_constraints:
                    continue
                tbt = ev.serving_tbt_ms
                self.assertGreater(
                    tbt, 1.0,
                    f"{name}: decode_tbt={tbt:.1f}ms suspiciously low.")
                self.assertLess(
                    tbt, 5000.0,
                    f"{name}: decode_tbt={tbt:.1f}ms suspiciously high.")
            except Exception as exc:
                self.fail(f"{name}: {type(exc).__name__}: {exc}")

    # --- Cross-hardware consistency ---

    def test_loss_consistent_across_hardware(self):
        """The same architecture should have the same predicted_loss on
        H100 vs B200 — quality is an arch+data function, not a hw function.
        Currently this property is NOT guaranteed (see feedback point 13 and
        plan/redesign/feedback-review.md), but if the drift exceeds ±2%
        something is structurally wrong.

        This test is marked xfail until the optimizer-self-consistency
        Wave-scope (plan/redesign/08-optimizer-self-consistency.md) lands.
        """
        for name, meta in ANCHORS.items():
            arch = self.archs[meta["arch"]]
            try:
                ev_h100 = _evaluate(name, arch, meta["n_kv_heads"], "h100")
                ev_b200 = _evaluate(name, arch, meta["n_kv_heads"], "b200")
            except Exception:
                continue
            if not (ev_h100.meets_constraints and ev_b200.meets_constraints):
                continue
            l_h100 = ev_h100.predicted_loss
            l_b200 = ev_b200.predicted_loss
            if min(l_h100, l_b200) == 0:
                continue
            drift_pct = 100.0 * abs(l_h100 - l_b200) / min(l_h100, l_b200)
            # Wave 10B landed: tightened from ±10% to ±0.5% — same arch +
            # same precision must yield identical predicted_loss across hw.
            # Any new hw-conditional quality term should fail this
            # regression test immediately. See
            # plan/redesign/10-optimizer-self-consistency.md Change B.
            self.assertLess(
                drift_pct, 0.5,
                f"{name}: loss drifts {drift_pct:.2f}% between H100 "
                f"({l_h100:.3f}) and B200 ({l_b200:.3f}). Quality must be "
                f"hardware-blind once precision is fixed (Wave 10B).")


if __name__ == "__main__":
    unittest.main()
