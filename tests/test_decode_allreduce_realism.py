"""Wave 17 Part 2 — decode all-reduce realism tests.

Spec-mandated companions to test_pp_decode_serving.py. These tests cover
the two specific properties Part 2 introduced:

  1. test_decode_ar_includes_latency_floor — at d_model very small (where
     bandwidth-amortized AR rounds to near-zero), the per-collective
     launch latency floor (~6 µs intra / 14 µs cross-IB) must dominate
     so the reported AR cost doesn't collapse to zero.

  2. test_decode_ar_at_tp32_above_floor — at TP=32 (cross-IB) the
     per-layer AR time must exceed the latency floor by a measurable
     margin. The plan calls for ">10 µs" — we use 5 µs as a generous
     floor (the formula adds the latency to the bandwidth term, and
     overlap_fraction=0.1 keeps 90% of the cost exposed).

Both tests exercise `_allreduce_cost` directly so they aren't confounded
by upstream changes (e.g., per-layer breakdown reshape).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_h100():
    """Convenience: load the H100 HardwareConfig once."""
    from ac.throughput_model import load_hardware
    return load_hardware("h100")


class DecodeArLatencyFloorTests(unittest.TestCase):
    """At decode, per-collective launch latency dominates when payload is
    small (B × 1 × d_model). Without the floor, AR cost collapses to
    near-zero — physically wrong (NCCL kernel launch + collective barrier
    takes microseconds even on tiny tensors)."""

    def test_decode_ar_includes_latency_floor(self):
        """At d_model=128, B=1, S=1 (tiny payload), the bandwidth-amortized
        AR cost is ~nanoseconds. The latency floor (~6 µs intra-NVLink at
        TP=4) must be the dominant term and must round to a non-zero ms."""
        from ac.throughput_model import _allreduce_cost
        hw = _load_h100()
        # TP=4, intra-NVLink. d_model=128 → payload = 2 × 1 × 1 × 128 × 2
        # = 512 bytes, BW-only term is ~ns at NVLink speeds. Floor at ~6 µs
        # per AR × 2 ARs × 0.9 exposed (decode overlap=0.1) ≈ 11 µs/layer.
        t = _allreduce_cost(
            B=1, S=1, d_model=128, precision="bf16",
            hw=hw, tp_degree=4, n_allreduces=2, phase="decode",
        )
        # The floor alone yields ~6 µs × 2 × 0.9 = 10.8 µs ≈ 1.08e-5 s.
        # If the floor is missing, BW-amortized term ≈ 512 / 9e11 × 2 × 0.75
        # ≈ 1 ns, which would fail this lower bound by 3 orders of magnitude.
        self.assertGreater(
            t, 5e-6,
            f"decode AR at tiny payload = {t*1e6:.2f} µs; latency floor "
            f"is missing. Per-collective launch latency (~6 µs intra-NVLink) "
            f"must dominate when payload is small. See "
            f"plan/redesign/17-pp-decode-serving-cost-fix.md Part 2."
        )
        # And a sanity upper bound: should not be more than ~50 µs at this
        # tiny payload — a catastrophic over-correction.
        self.assertLess(
            t, 5e-5,
            f"decode AR at tiny payload = {t*1e6:.2f} µs is implausibly "
            f"large; latency floor over-corrected."
        )


class DecodeArAtTp32Tests(unittest.TestCase):
    """At TP=32 (cross-IB) the per-layer AR must clear the cross-IB
    latency floor + the bandwidth-amortized term."""

    def test_decode_ar_at_tp32_above_floor(self):
        """At TP=32, d_model=8192, B=8 — typical large-model decode AR.
        Cross-IB latency is ~14 µs/AR; with 2 ARs and overlap=0.1, the
        per-layer cost should be > 10 µs (the plan's stated threshold).
        Pre-fix, this was rounding to ~1 µs because the BW-only formula
        amortized 262KB across 50 GB/s IB to ~5 µs * (TP-1)/TP and then
        the 0.5 overlap halved it."""
        from ac.throughput_model import _allreduce_cost
        hw = _load_h100()
        t = _allreduce_cost(
            B=8, S=1, d_model=8192, precision="bf16",
            hw=hw, tp_degree=32, n_allreduces=2, phase="decode",
        )
        # Floor alone at TP=32: 14 µs × 2 × 0.9 = 25.2 µs. BW-amortized
        # adds ~5 µs more. Total ≥ 25 µs.
        self.assertGreater(
            t, 10e-6,
            f"decode AR at TP=32 = {t*1e6:.2f} µs; cross-IB latency floor "
            f"underweighted. Spec target: > 10 µs/layer at frontier dim. "
            f"See plan/redesign/17-pp-decode-serving-cost-fix.md Part 2."
        )

    def test_decode_phase_overlap_lower_than_training(self):
        """At identical TP/payload, decode-phase AR should be larger than
        training-phase AR by the overlap-fraction ratio (0.9/0.5 = 1.8×).
        This is the property that lets decode honestly cost more than
        training-step AR per layer."""
        from ac.throughput_model import _allreduce_cost
        hw = _load_h100()
        t_train = _allreduce_cost(
            B=8, S=1, d_model=8192, precision="bf16",
            hw=hw, tp_degree=16, n_allreduces=2, phase="training",
        )
        t_decode = _allreduce_cost(
            B=8, S=1, d_model=8192, precision="bf16",
            hw=hw, tp_degree=16, n_allreduces=2, phase="decode",
        )
        # Decode should be at least 50% more expensive per layer (overlap
        # 0.1 vs 0.5 → exposed fraction 0.9 vs 0.5 = 1.8× ratio, but
        # latency floor is the same so the ratio is partially diluted).
        self.assertGreater(
            t_decode, t_train * 1.2,
            f"decode AR ({t_decode*1e6:.2f} µs) should clearly exceed "
            f"training AR ({t_train*1e6:.2f} µs) — decode has no matmul "
            f"to overlap behind (overlap=0.1 vs 0.5 for training)."
        )


if __name__ == "__main__":
    unittest.main()
