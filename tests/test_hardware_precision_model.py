"""Hardware spec conventions: datasheet peaks, MFU basis, precision rules.

(Pins from Waves 20, 21, and 22.)

  * Every shipped spec carries a display-only `datasheet_peak_flops_tf`;
    the internal roofline baseline stays halved.
  * Implied MFU (6·N·tps/GPU over the datasheet peak for the training
    precision) lands in a plausible band for a standard dense shape.
  * Weight STORAGE precision is feasibility-checked separately from
    native COMPUTE precision (mxfp4 weights on H100 are deployable; fp4
    activations are not), and `bytes_per_elem` falls back to canonical
    byte widths instead of silently pricing unknown formats as bf16.
  * The console `≈N% MFU … tok/s/GPU` line divides by the same
    TP × PP × CP training replica the optimizer emits into arch.json.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO)
AC_DIR = os.path.join(REPO, "ac")
if AC_DIR not in sys.path:
    sys.path.insert(0, AC_DIR)

from ac.penalties import precision_supported, weight_storage_supported  # noqa: E402
from ac.throughput_model import load_hardware  # noqa: E402


def test_datasheet_peaks_present_and_sane():
    from throughput_model import load_hardware
    hw = load_hardware("h100")
    assert hw.datasheet_peak_flops_s("bf16") == pytest.approx(989e12)
    assert hw.datasheet_peak_flops_s("fp8") == pytest.approx(1979e12)
    # Internal roofline baseline stays halved — the MFU display must not
    # have touched the model's own numbers.
    assert hw.peak_flops_s("bf16") == pytest.approx(495e12)
    # Every shipped spec carries the field.
    for name in ("b200", "tpu_v5e", "tpu_v5p", "trainium2", "trainium3"):
        h = load_hardware(name)
        assert h.datasheet_peak_flops_tf, name


def test_implied_mfu_in_plausible_band():
    """6·N·tps/GPU over the datasheet peak lands in [0.15, 0.6] for a
    standard dense shape — guards against future basis mix-ups in either
    direction (68% was the misread; 5% would mean a broken model)."""
    from throughput_model import ArchConfig, throughput, load_hardware
    arch = ArchConfig(d_model=8192, n_layers=46, n_heads=64, d_head=128,
                      n_kv_heads=8, ffn_dim=22016, vocab_size=32000,
                      batch_size=8, seq_len=8192, precision="fp8")
    r = throughput(arch, "h100", tp_degree=8, dp_degree=8)
    tps_gpu = r.training_throughput_tokens_per_sec / 8
    hw = load_hardware("h100")
    n_active = 32.36e9
    mfu = 6 * n_active * tps_gpu / hw.datasheet_peak_flops_s("fp8")
    assert 0.15 < mfu < 0.6, mfu


class PrecisionFeasibilityTests(unittest.TestCase):
    def test_weight_storage_vs_native_compute_split(self):
        # mxfp4 weights on H100: deployable storage, not native compute.
        self.assertTrue(weight_storage_supported("mxfp4", "h100"))
        self.assertTrue(weight_storage_supported("fp4", "h100"))
        self.assertFalse(precision_supported("mxfp4", "h100"))
        self.assertFalse(precision_supported("fp4", "h100"))
        # Native-compute parts unchanged.
        self.assertTrue(precision_supported("mxfp4", "b200"))
        self.assertTrue(precision_supported("fp8", "h100"))

    def test_bytes_per_elem_canonical_fallback(self):
        hw = load_hardware("h100")
        # h100_sxm.json has no mxfp4 entry; the old fallback priced it at
        # 2 B/elem in the roofline while the memory path used 0.53 —
        # same weights, two physics.
        self.assertLess(hw.bytes_per_elem("mxfp4"), 1.0)
        self.assertLess(hw.bytes_per_elem("mxfp6"), 1.0)
        self.assertEqual(hw.bytes_per_elem("bf16"), 2)


class ConsoleReplicaTests(unittest.TestCase):
    def test_console_tok_per_gpu_matches_artifact_under_cp(self):
        # End-to-end pin: a cp=2 dense run's console `tok/s/GPU` must
        # equal the JSON artifact's per-GPU number (tp×pp×cp replica).
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run(
                [sys.executable, "-m", "ac.cli_compile",
                 "--hardware", "h100", "--params", "7", "--tokens", "2",
                 "--cp", "2", "--max-candidates", "60",
                 "--no-shadow-prices", "--no-family-view",
                 "--out", td],
                capture_output=True, text=True, cwd=REPO,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            m = [ln for ln in r.stderr.splitlines() if "tok/s/GPU" in ln]
            if not m:
                self.skipTest("MFU line not printed (no datasheet peak)")
            console_tps = float(
                m[0].split("tok/s/GPU")[0].split(",")[-1].strip()
            )
            with open(os.path.join(td, "arch.json")) as f:
                predicted = json.load(f)["metadata"]["predicted"]
            self.assertAlmostEqual(
                console_tps,
                predicted["training_throughput_tokens_per_sec_per_gpu"],
                delta=1.5,  # console rounds to whole tokens
            )
            unit = predicted["training_throughput_unit"]
            self.assertIn("×2 = ", unit.replace("× 2", "×2"))


if __name__ == "__main__":
    unittest.main()
