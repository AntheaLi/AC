"""Integrity checks for the H100 decision-matrix driver."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "ac"))

import _generator_payload as payload  # noqa: E402


class MatrixDriverIntegrityTests(unittest.TestCase):
    def test_family_rollup_uses_actual_candidate_family(self):
        data = {
            "grid": [{
                "hw": "h100",
                "params_B": 3.0,
                "tokens_T": 2.0,
                "context_length": 8192,
                "arch_mode": "moe_hybrid",
                "state_type": "mamba2",
                "candidates": 1,
                "feasible": 1,
                "optimal": {
                    "arch_family": "moe",
                    "loss": 2.0,
                    "tbt_ms": 5.0,
                    "ttft_ms": 100.0,
                    "mem_gb": 20.0,
                    "train_tps": 1000.0,
                    "active_params_B": 3.0,
                    "params_B": 10.0,
                    "tp": 2,
                    "ep": 2,
                },
                "pareto": [],
            }]
        }
        payload._build_family_rollup(data)
        family = data["cells"][0]["families"][0]
        self.assertEqual(family["arch_mode"], "moe")
        self.assertEqual(family["display"], "MoE")
        self.assertIsNone(family["state_type"])

    def test_batch_picker_uses_real_hbm_capacity_field(self):
        constraints = SimpleNamespace(serving_batch=32)
        optimal = SimpleNamespace(arch=object())

        def fake_eval(_arch, _hw, copied):
            # 20 GB per request: batch 4 is above 90% of H100 HBM,
            # batch 2 is the largest that fits.
            return SimpleNamespace(
                memory_per_gpu_gb=20.0 * copied.serving_batch,
                serving_tbt_ms=1.0,
            )

        mode = {"batches": [32, 16, 8, 4, 2, 1], "tbt": None}
        with patch.object(payload, "evaluate_candidate", side_effect=fake_eval):
            picked = payload._pick_serving_batch(
                optimal, "h100", constraints, mode
            )
        self.assertEqual(picked._chosen_batch, 2)


if __name__ == "__main__":
    unittest.main()
