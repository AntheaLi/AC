"""Wave 7b unit tests for the Wave 1-5 calibration fitters.

These tests synthesize measurement rows with known ground truth, hand them
to the new fitters in `auto_calibrate.py`, and assert the fitters recover
the ground truth within a tolerance. The fitters are pure functions —
they do not require pytest fixtures, network access, or real cluster
measurements.

The final test (`test_calibrated_throughput_uses_fitted_overlap`) goes the
other direction: it constructs a `CalibrationTable` with a Wave 5 override
and verifies the throughput model's DP grad sync term picks it up.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ac"))

from ac import auto_calibrate  # noqa: E402
from ac.auto_calibrate import (  # noqa: E402
    _WAVE5_DEFAULTS,
    _fit_dp_overlap,
    _fit_tp_overlap,
    _fit_pp_queue,
    _fit_state_long_ctx,
    _fit_hbm_spill,
    _solve_overlap_from_ratio,
)
from ac.throughput_model import CalibrationTable  # noqa: E402


def _row(**kw):
    """Tiny convenience for building measurement rows."""
    row = {"hardware": "h100"}
    row.update(kw)
    return row


class FitDpOverlapTests(unittest.TestCase):
    """Synthesize rows where (1-measured)/(1-default) is a known constant
    and verify the fitter recovers `measured`."""

    def test_recovers_measured_overlap_within_tolerance(self):
        # Ground truth: lab measured overlap is 0.5 (vs default 0.7).
        # ratio = (1 - 0.5) / (1 - 0.7) = 0.5 / 0.3 = 1.667
        rows = []
        for i in range(8):
            # Predicted is the model output assuming default overlap; observed
            # is the lab measurement. Both >0; the fitter only takes the
            # median ratio.
            pred = 10.0
            obs = 10.0 * (1.0 - 0.5) / (1.0 - 0.7)  # = 16.67 ms
            rows.append(_row(predicted_dp_grad_allreduce_ms=pred,
                              observed_dp_grad_allreduce_ms=obs))
        fit = _fit_dp_overlap(rows, default_hardware="h100")
        self.assertIn("h100", fit)
        recovered = fit["h100"]["overlap_fraction"]
        self.assertIsNotNone(recovered)
        self.assertAlmostEqual(recovered, 0.5, delta=0.05)
        self.assertEqual(fit["h100"]["n"], 8)

    def test_returns_none_when_no_measurements(self):
        rows = [_row(predicted_serving_tbt_ms=10.0)]  # different field
        fit = _fit_dp_overlap(rows, default_hardware="h100")
        # With no DP-grad observations, h100 still appears (it's in hardware
        # set) but the overlap_fraction must be None.
        self.assertIsNone(fit["h100"]["overlap_fraction"])
        self.assertEqual(fit["h100"]["n"], 0)


class FitTpOverlapTests(unittest.TestCase):
    def test_recovers_overlap_for_default_05(self):
        # Default overlap is 0.5; lab measured 0.7.
        # ratio = (1-0.7) / (1-0.5) = 0.3 / 0.5 = 0.6
        rows = [
            _row(predicted_tp_allreduce_ms_per_layer=1.0,
                  observed_tp_allreduce_ms_per_layer=0.6)
            for _ in range(6)
        ]
        fit = _fit_tp_overlap(rows, default_hardware="h100")
        recovered = fit["h100"]["overlap_fraction"]
        self.assertAlmostEqual(recovered, 0.7, delta=0.05)


class FitPpQueueTests(unittest.TestCase):
    """Per-(hw, schedule) bucketing test. Verify the output shape matches
    the spec — `{hardware: {schedule: {multiplier, n}}}` — and the
    multiplier reflects the supplied observed/predicted ratio."""

    def test_groups_by_hardware_and_schedule(self):
        rows = [
            _row(pp_schedule="1f1b", pp_degree=4,
                  predicted_activation_peak_mem_gb=40.0,
                  observed_activation_peak_mem_gb=42.0),
            _row(pp_schedule="1f1b", pp_degree=4,
                  predicted_activation_peak_mem_gb=42.0,
                  observed_activation_peak_mem_gb=44.0),
            _row(pp_schedule="gpipe", pp_degree=8,
                  predicted_activation_peak_mem_gb=80.0,
                  observed_activation_peak_mem_gb=72.0),
            _row(hardware="b200", pp_schedule="1f1b", pp_degree=8,
                  predicted_activation_peak_mem_gb=50.0,
                  observed_activation_peak_mem_gb=51.0),
        ]
        fit = _fit_pp_queue(rows)
        self.assertIn("h100", fit)
        self.assertIn("1f1b", fit["h100"])
        self.assertIn("gpipe", fit["h100"])
        # 1f1b multiplier ~1.04, gpipe ~0.9
        self.assertAlmostEqual(fit["h100"]["1f1b"]["multiplier"], 1.045, delta=0.02)
        self.assertAlmostEqual(fit["h100"]["gpipe"]["multiplier"], 0.9, delta=0.02)
        self.assertIn("b200", fit)
        self.assertIn("1f1b", fit["b200"])

    def test_pp_degree_one_is_skipped(self):
        rows = [
            _row(pp_schedule="1f1b", pp_degree=1,
                  predicted_activation_peak_mem_gb=40.0,
                  observed_activation_peak_mem_gb=44.0),
        ]
        fit = _fit_pp_queue(rows)
        # PP=1 contributes nothing — no schedule-specific bucket exists.
        self.assertNotIn("h100", fit)


class FitStateLongCtxTests(unittest.TestCase):
    """Anchor against the Wave 5 design: when predicted_loss / observed_loss
    equals 1.0 (model is dead-on), the fitted weight must equal the default
    0.030. When predicted is 5% higher than observed (model under-predicts
    the benefit), the fitter shifts the weight upward."""

    def test_fit_against_anchor_returns_default_when_unbiased(self):
        rows = [
            _row(arch_mode="hybrid", context_length=131072,
                  predicted_long_ctx_loss=2.5, observed_long_ctx_loss=2.5),
            _row(arch_mode="hybrid", context_length=131072,
                  predicted_long_ctx_loss=2.6, observed_long_ctx_loss=2.6),
            _row(arch_mode="hybrid", context_length=1048576,
                  predicted_long_ctx_loss=2.7, observed_long_ctx_loss=2.7),
        ]
        fit = _fit_state_long_ctx(rows)
        self.assertEqual(fit["n"], 3)
        self.assertAlmostEqual(fit["weight"], 0.030, delta=0.001)

    def test_fit_against_jamba_like_anchor_lifts_weight(self):
        # NVIDIA-Empirical Study: at 128k+ context, hybrid models beat
        # attention baselines by ~2.65 pt; our model under-predicts by ~5%.
        rows = [
            _row(arch_mode="hybrid", context_length=131072,
                  predicted_long_ctx_loss=2.10, observed_long_ctx_loss=2.00),
            _row(arch_mode="hybrid", context_length=131072,
                  predicted_long_ctx_loss=2.05, observed_long_ctx_loss=1.95),
            _row(arch_mode="hybrid_state", context_length=1048576,
                  predicted_long_ctx_loss=2.20, observed_long_ctx_loss=2.10),
        ]
        fit = _fit_state_long_ctx(rows)
        self.assertGreater(fit["weight"], _WAVE5_DEFAULTS["state_long_context_weight"])
        self.assertLessEqual(fit["weight"], 0.10)  # band cap from the fitter

    def test_excludes_short_ctx_rows(self):
        rows = [
            _row(arch_mode="hybrid", context_length=4096,   # < 32k floor
                  predicted_long_ctx_loss=2.0, observed_long_ctx_loss=2.0),
        ]
        fit = _fit_state_long_ctx(rows)
        self.assertEqual(fit["n"], 0)
        self.assertIsNone(fit["weight"])


class FitHbmSpillTests(unittest.TestCase):
    def test_fits_only_spill_rows(self):
        rows = [
            _row(hbm_spill_gb=4.0,
                  predicted_tbt_ms_with_spill=12.0,
                  observed_tbt_ms_with_spill=15.0),
            _row(hbm_spill_gb=8.0,
                  predicted_tbt_ms_with_spill=18.0,
                  observed_tbt_ms_with_spill=22.5),
            _row(hbm_spill_gb=0.0,  # no spill — must be skipped
                  predicted_tbt_ms_with_spill=10.0,
                  observed_tbt_ms_with_spill=20.0),
        ]
        fit = _fit_hbm_spill(rows, default_hardware="h100")
        self.assertEqual(fit["h100"]["n"], 2)
        self.assertAlmostEqual(fit["h100"]["factor"], 1.25, delta=0.02)


class SolveOverlapTests(unittest.TestCase):
    """Direct unit test of the overlap inversion helper."""

    def test_round_trip(self):
        # measured == 1.0 → ratio == 0 → the inverter returns None on purpose
        # (an exposed cost of 0 means observed == 0 which we treat as bogus
        # input rather than valid data).
        for measured in (0.1, 0.3, 0.5, 0.8, 0.95):
            for default in (0.5, 0.7):
                ratio = (1.0 - measured) / max(1e-6, (1.0 - default))
                recovered = _solve_overlap_from_ratio(ratio, default)
                self.assertAlmostEqual(recovered, measured, delta=1e-6)

    def test_invalid_ratio_returns_none(self):
        self.assertIsNone(_solve_overlap_from_ratio(0.0, 0.7))
        self.assertIsNone(_solve_overlap_from_ratio(None, 0.7))


class CalibrationTableWave5Tests(unittest.TestCase):
    """Sanity-check that the CalibrationTable loads Wave 5 fields from both
    the flat and the nested JSON shapes."""

    def _write(self, tmp_path: Path, payload: dict) -> Path:
        import json
        p = tmp_path / "h100_calibration.json"
        p.write_text(json.dumps(payload))
        return p

    def test_flat_shape(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(Path(tmp), {
                "hardware": "h100",
                "dp_grad_overlap_fraction": 0.6,
                "tp_allreduce_overlap_fraction": 0.45,
                "state_long_context_weight": 0.028,
                "hbm_spill_decode_factor": 1.10,
                "pp_queue_multipliers": {"1f1b": 1.05, "gpipe": 0.97},
            })
            ct = CalibrationTable.from_json(str(p))
            self.assertAlmostEqual(ct.dp_grad_overlap_fraction, 0.6, delta=1e-6)
            self.assertAlmostEqual(ct.tp_allreduce_overlap_fraction, 0.45, delta=1e-6)
            self.assertAlmostEqual(ct.state_long_context_weight, 0.028, delta=1e-6)
            self.assertAlmostEqual(ct.hbm_spill_decode_factor, 1.10, delta=1e-6)
            self.assertEqual(ct.pp_queue_multipliers["1f1b"], 1.05)
            self.assertEqual(ct.pp_queue_multipliers["gpipe"], 0.97)

    def test_nested_wave5_shape(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(Path(tmp), {
                "hardware": "h100",
                "wave5": {
                    "dp_grad_overlap": 0.55,
                    "tp_overlap": 0.40,
                    "state_long_context_weight": 0.032,
                    "hbm_spill_factor": 0.95,
                    "pp_queue": {"1f1b": {"multiplier": 1.07}},
                },
            })
            ct = CalibrationTable.from_json(str(p))
            self.assertAlmostEqual(ct.dp_grad_overlap_fraction, 0.55, delta=1e-6)
            self.assertAlmostEqual(ct.tp_allreduce_overlap_fraction, 0.40, delta=1e-6)
            self.assertAlmostEqual(ct.state_long_context_weight, 0.032, delta=1e-6)
            self.assertAlmostEqual(ct.hbm_spill_decode_factor, 0.95, delta=1e-6)
            self.assertEqual(ct.pp_queue_multipliers["1f1b"], 1.07)


class CalibratedThroughputUsesFittedOverlapTests(unittest.TestCase):
    """Verify the runtime path: when a CalibrationTable carries a fitted DP
    grad overlap, the throughput model's training step uses it. We exercise
    this through the synchronous helper and the module-level holder."""

    def test_dp_grad_cost_changes_with_calibration(self):
        from ac.throughput_model import (
            _dp_grad_allreduce_cost,
            load_hardware,
        )
        hw = load_hardware("h100")
        # Same params, same dp_degree, two different overlap fractions
        # should produce strictly different sync costs.
        cost_default = _dp_grad_allreduce_cost(
            total_params=int(7e9), precision="bf16", hw=hw,
            dp_degree=64, zero_stage=3,
        )
        cost_lab = _dp_grad_allreduce_cost(
            total_params=int(7e9), precision="bf16", hw=hw,
            dp_degree=64, zero_stage=3, overlap_fraction=0.5,
        )
        # Default overlap is 0.7 → exposed = 0.3; lab override 0.5 → exposed
        # = 0.5. Lab cost must be strictly higher than default.
        self.assertGreater(cost_lab, cost_default)
        # And the ratio must equal 0.5/0.3 = 1.667.
        self.assertAlmostEqual(cost_lab / cost_default, 0.5 / 0.3, delta=0.01)

    def test_allreduce_cost_uses_module_calibration(self):
        from ac.throughput_model import (
            _allreduce_cost, load_hardware,
            CalibrationTable, _CURRENT_CALIBRATION,  # noqa: F401
        )
        import ac.throughput_model as tm
        hw = load_hardware("h100")
        # Baseline: no calibration table, default overlap 0.5.
        tm._CURRENT_CALIBRATION = None
        baseline = _allreduce_cost(
            B=4, S=2048, d_model=4096, precision="bf16",
            hw=hw, tp_degree=8, n_allreduces=2,
        )
        # Lab calibration: overlap fitted at 0.7 → less exposed comm, smaller cost.
        tm._CURRENT_CALIBRATION = CalibrationTable(
            tp_allreduce_overlap_fraction=0.7,
        )
        try:
            with_cal = _allreduce_cost(
                B=4, S=2048, d_model=4096, precision="bf16",
                hw=hw, tp_degree=8, n_allreduces=2,
            )
        finally:
            tm._CURRENT_CALIBRATION = None
        self.assertLess(with_cal, baseline)
        # Ratio: exposed_cal / exposed_default = (1-0.7)/(1-0.5) = 0.6
        self.assertAlmostEqual(with_cal / baseline, 0.6, delta=0.01)


if __name__ == "__main__":
    unittest.main()
