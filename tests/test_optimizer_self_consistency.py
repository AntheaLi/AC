"""Wave 10 — optimizer self-consistency regression tests.

Per plan/redesign/10-optimizer-self-consistency.md. These tests lock in:

  * Wave 10A — `train_tps` does not depend on serving SLO. The throughput
    model uses a training micro-batch separate from `serving_batch`, so
    two evaluations of the same arch under different serving budgets must
    produce identical training throughput.

  * Wave 10B — `predicted_loss` does not depend on hardware once
    precision is feasible. The quality residual stack reads only the
    abstract precision name; hw-conditional checks are confined to
    `precision_supported()` (feasibility, not quality).

  * Wave 10C — `_harmonize_serving_train_metrics` and
    `_harmonize_loss_across_hw` are no-ops on data where the optimizer
    already self-agrees. The post-processing pipeline no longer calls
    them; this test confirms they don't move cells when invoked manually.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _make_cand(**extras):
    from ac.optimizer import CandidateArch
    base = dict(
        d_model=4096, n_layers=32, n_heads=32, d_head=128, n_kv_heads=8,
        ffn_dim=14336, vocab_size=128000,
        weight_precision="bf16", ffn_precision="bf16",
        attn_precision={"q": "bf16", "k": "bf16", "v": "bf16", "o": "bf16"},
        kv_cache_bits=16, attention_type="gqa",
        tp_degree=1, cp_degree=1, ep_degree=1,
    )
    base.update(extras)
    return CandidateArch(**base)


def _make_constraints(**extras):
    from ac.optimizer import DeploymentConstraints
    base = dict(
        target_params_b=7.0, training_tokens=int(2e12),
        context_length=8192,
        tp=1, pp=1, dp=1,
        serving_tbt_ms=None, serving_ttft_ms=None,
        serving_batch=8,
        allow_quality_sentinel=True,
    )
    base.update(extras)
    return DeploymentConstraints(**base)


class TrainingIndependentOfServingTests(unittest.TestCase):
    """Wave 10A — training metrics must not depend on serving SLO."""

    def test_train_tps_independent_of_serving_slo(self):
        """Same arch under (tbt=20ms, serving_batch=4) and (tbt=None,
        serving_batch=32) must produce identical training_tps. Bug B
        regression guard."""
        from ac.optimizer import evaluate_candidate
        cand = _make_cand()
        ev_tight = evaluate_candidate(
            cand, "h100",
            _make_constraints(serving_tbt_ms=20, serving_batch=4),
        )
        ev_loose = evaluate_candidate(
            cand, "h100",
            _make_constraints(serving_tbt_ms=None, serving_batch=32),
        )
        self.assertEqual(
            round(ev_tight.training_tps), round(ev_loose.training_tps),
            f"training_tps differs under different serving SLOs: "
            f"tight={ev_tight.training_tps:.0f}, loose={ev_loose.training_tps:.0f}. "
            f"Wave 10A regression — see plan/redesign/10-optimizer-self-consistency.md."
        )

    def test_explicit_training_micro_batch_pins_train_tps(self):
        """When `constraints.training_micro_batch` is set, training_tps
        is computed at that batch regardless of serving_batch."""
        from ac.optimizer import evaluate_candidate
        cand = _make_cand()
        ev_small = evaluate_candidate(
            cand, "h100",
            _make_constraints(training_micro_batch=4, serving_batch=32),
        )
        ev_large = evaluate_candidate(
            cand, "h100",
            _make_constraints(training_micro_batch=16, serving_batch=32),
        )
        self.assertNotEqual(
            round(ev_small.training_tps), round(ev_large.training_tps),
            f"explicit training_micro_batch should change training_tps: "
            f"mb=4 → {ev_small.training_tps:.0f}, mb=16 → {ev_large.training_tps:.0f}"
        )


class QualityHwBlindTests(unittest.TestCase):
    """Wave 10B — predicted_loss is hw-blind once precision is feasible."""

    def test_quality_independent_of_hardware_bf16(self):
        """BF16 is supported on every hardware. Same arch + same precision
        must produce identical predicted_loss across H100 / B200 / TPU."""
        from ac.optimizer import evaluate_candidate
        cand = _make_cand()
        losses = {}
        for hw in ("h100", "b200", "tpu_v5p"):
            ev = evaluate_candidate(cand, hw, _make_constraints())
            losses[hw] = ev.predicted_loss
        baseline = losses["h100"]
        for hw, loss in losses.items():
            drift = abs(loss - baseline) / max(baseline, 1e-9) * 100
            self.assertLess(
                drift, 0.5,
                f"predicted_loss for {hw}={loss:.6f} drifts {drift:.3f}% "
                f"from h100={baseline:.6f}. Wave 10B: quality must be "
                f"hw-blind once precision is feasible."
            )

    def test_quality_independent_of_hardware_kv4(self):
        """KV=4 with per-channel scaling is supported on H100/B200/TPU.
        Loss must be identical across hw (the historic 2× uplift on TPU
        was a quality-layer hw conditionality — now removed)."""
        from ac.optimizer import evaluate_candidate
        cand = _make_cand(kv_cache_bits=4)
        losses = {}
        for hw in ("h100", "b200", "tpu_v5p"):
            ev = evaluate_candidate(cand, hw, _make_constraints())
            if ev.meets_constraints:
                losses[hw] = ev.predicted_loss
        if len(losses) < 2:
            self.skipTest(f"need >=2 feasible hw; got {list(losses.keys())}")
        baseline = next(iter(losses.values()))
        for hw, loss in losses.items():
            drift = abs(loss - baseline) / max(baseline, 1e-9) * 100
            self.assertLess(
                drift, 0.5,
                f"predicted_loss for KV=4 on {hw}={loss:.6f} drifts "
                f"{drift:.3f}% from baseline={baseline:.6f}. Wave 10B "
                f"regression — kv_quant_quality must be hw-blind."
            )


class HarmonizationPassesNoOpTests(unittest.TestCase):
    """Wave 10C — when the optimizer is already self-consistent, the
    harmonization passes do not move cells."""

    def test_harmonization_passes_are_noops_on_self_consistent_grid(self):
        """Build a synthetic 4-row grid where train_tps and loss are
        already consistent within the same (params, tokens, ctx, arch_mode);
        invoking the harmonization passes manually must not change any row."""
        import copy
        from _generator_payload import (
            _harmonize_serving_train_metrics, _harmonize_loss_across_hw,
        )
        # Synthetic grid: same arch_mode across hw with already-consistent
        # train_tps and loss values.
        grid = {
            "grid": [
                {"hw": "h100", "params_B": 7, "tokens_T": 2.0,
                 "context_length": 8192, "arch_mode": "dense",
                 "state_type": None, "serving": "unconstrained",
                 "tp": 1, "pp": 1, "dp": 1,
                 "optimal": {
                     "d_model": 4096, "n_layers": 32, "n_heads": 32,
                     "d_head": 128, "n_kv_heads": 8, "ffn_dim": 14336,
                     "weight_prec": "bf16", "kv_bits": 16,
                     "active_params_B": 7.0, "params_B": 7.0,
                     "loss": 2.020, "train_tps": 8000,
                     "tbt_ms": 12.0, "mem_gb": 30.0,
                 }},
                {"hw": "b200", "params_B": 7, "tokens_T": 2.0,
                 "context_length": 8192, "arch_mode": "dense",
                 "state_type": None, "serving": "unconstrained",
                 "tp": 1, "pp": 1, "dp": 1,
                 "optimal": {
                     "d_model": 4096, "n_layers": 32, "n_heads": 32,
                     "d_head": 128, "n_kv_heads": 8, "ffn_dim": 14336,
                     "weight_prec": "bf16", "kv_bits": 16,
                     "active_params_B": 7.0, "params_B": 7.0,
                     "loss": 2.020, "train_tps": 15000,
                     "tbt_ms": 6.0, "mem_gb": 14.0,
                 }},
            ],
        }
        snapshot = copy.deepcopy(grid)
        _harmonize_serving_train_metrics(grid)
        _harmonize_loss_across_hw(grid)
        for new_row, old_row in zip(grid["grid"], snapshot["grid"]):
            for key in ("loss", "train_tps", "d_model", "n_layers"):
                self.assertEqual(
                    new_row["optimal"][key], old_row["optimal"][key],
                    f"{new_row['hw']}: {key} moved under harmonization "
                    f"despite a self-consistent grid: "
                    f"{old_row['optimal'][key]} → {new_row['optimal'][key]}"
                )


if __name__ == "__main__":
    unittest.main()
