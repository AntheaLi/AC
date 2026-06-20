import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args, env=None):
    child_env = None
    if env is not None:
        import os
        child_env = os.environ.copy()
        child_env.update(env)
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=child_env,
    )


class CliSmokeTests(unittest.TestCase):
    def test_stress_quality_known_model_uses_default_vocab(self):
        result = run_cli(
            "ac/cli_stress.py",
            "quality",
            "--known",
            "Llama-3-70B",
            "--tokens",
            "15000000000000",
            "--prefill-seq",
            "4096",
            "--decode-kv",
            "4096",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("QualityStressVector", result.stdout)
        self.assertIn("Llama-3-70B", result.stdout)

    def test_compile_accepts_trainium2(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "trn2.json"
            result = run_cli(
                "ac/cli_compile.py",
                "--hardware",
                "trainium2",
                "--params",
                "1",
                "--tokens",
                "0.2",
                "--context",
                "2048",
                "--serving-tbt",
                "100",
                "--serving-batch",
                "4",
                "--tp",
                "1",
                "--pp",
                "1",
                "--dp",
                "1",
                "--max-candidates",
                "30",
                "--output-config",
                str(output),
                "--output-justification",
                str(tmp_path / "trn2.md"),
                "--output-pareto",
                str(tmp_path / "trn2.csv"),
                "--no-shadow-prices",
                "--quiet",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(output.read_text())
            self.assertEqual(config["metadata"]["input_hardware"], "trainium2")

    def test_compile_accepts_b200_fp4_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "fp4.json"
            result = run_cli(
                "ac/cli_compile.py",
                "--hardware",
                "b200",
                "--params",
                "1",
                "--tokens",
                "0.2",
                "--context",
                "2048",
                "--serving-tbt",
                "100",
                "--serving-batch",
                "4",
                "--tp",
                "1",
                "--pp",
                "1",
                "--dp",
                "1",
                "--precision-modes",
                "fp4",
                "--max-candidates",
                "30",
                "--output-config",
                str(output),
                "--output-justification",
                str(tmp_path / "fp4.md"),
                "--output-pareto",
                str(tmp_path / "fp4.csv"),
                "--no-shadow-prices",
                "--quiet",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(output.read_text())
            layer = config["architecture"]["layer_configs"][0]
            self.assertEqual(layer["ffn"]["precision"], "fp4")

    def test_compile_can_emit_pytorch_implementation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "arch.json"
            impl = tmp_path / "ac_model.py"
            result = run_cli(
                "ac/cli_compile.py",
                "--hardware",
                "h100",
                "--params",
                "1",
                "--tokens",
                "0.2",
                "--context",
                "2048",
                "--serving-tbt",
                "100",
                "--serving-batch",
                "4",
                "--tp",
                "1",
                "--pp",
                "1",
                "--dp",
                "1",
                "--max-candidates",
                "20",
                "--output-config",
                str(output),
                "--output-implementation",
                str(impl),
                "--implementation-class-name",
                "MyACModel",
                "--output-justification",
                str(tmp_path / "arch.md"),
                "--output-pareto",
                str(tmp_path / "pareto.csv"),
                "--no-shadow-prices",
                "--quiet",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            text = impl.read_text()
            self.assertIn("class MyACModel", text)
            self.assertIn("flash_attn", text)
            self.assertIn("component_overrides", text)
            syntax = run_cli(
                "-m",
                "py_compile",
                str(impl),
                env={"PYTHONPYCACHEPREFIX": str(tmp_path / "pycache")},
            )
            self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_delta_report_direction_is_plain_text(self):
        result = run_cli(
            "ac/cli_delta_eval.py",
            "--baseline-config",
            "configs/mistral_7b.json",
            "--hardware",
            "h100",
            "--tp",
            "8",
            "--workload",
            "long_context",
            "--apply",
            "swap_attention_to_gqa",
            "--apply-args",
            "group_size=8",
            "--stdout",
            "--no-pareto",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("| Training TPS (tok/s)", result.stdout)
        self.assertIn("| improves |", result.stdout)
        self.assertIn("| KV cache (GB) | 0.500 | 0.500 |", result.stdout)
        self.assertIn("at least one KV head resident per rank", result.stdout)
        self.assertNotIn("-50.00%", result.stdout)
        self.assertNotIn("↓", result.stdout)
        self.assertNotIn("↑", result.stdout)

    def test_auto_calibrate_pack_feeds_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            measurements = tmp_path / "measurements.json"
            measurements.write_text(json.dumps([
                {
                    "id": "h100_case_a",
                    "hardware": "h100",
                    "predicted_loss": 2.0,
                    "observed_loss": 2.12,
                    "predicted_uncertainty_total_pct": 2.0,
                    "predicted_training_tps": 100.0,
                    "observed_training_tps": 80.0,
                    "predicted_serving_tbt_ms": 10.0,
                    "observed_serving_tbt_ms": 12.5,
                },
                {
                    "id": "h100_case_b",
                    "hardware": "h100",
                    "predicted_loss": 2.5,
                    "observed_loss": 2.6,
                    "predicted_uncertainty_total_pct": 2.0,
                    "predicted_training_tps": 120.0,
                    "observed_training_tps": 96.0,
                    "predicted_serving_tbt_ms": 8.0,
                    "observed_serving_tbt_ms": 10.0,
                },
            ]))
            pack_dir = tmp_path / "pack"
            result = run_cli(
                "ac/auto_calibrate.py",
                "fit",
                "--measurements",
                str(measurements),
                "--out",
                str(pack_dir),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            overlay = json.loads((pack_dir / "quality_overrides.json").read_text())
            self.assertGreater(overlay["uncertainty"]["calibration_multiplier"], 1.0)
            h100 = json.loads((pack_dir / "hardware_specs" / "h100_sxm.json").read_text())
            self.assertLess(h100["calibration"]["training_system_efficiency"], 0.37)

            output = tmp_path / "calibrated_compile.json"
            result = run_cli(
                "ac/cli_compile.py",
                "--hardware",
                "h100",
                "--params",
                "1",
                "--tokens",
                "0.2",
                "--context",
                "2048",
                "--serving-tbt",
                "100",
                "--serving-batch",
                "4",
                "--tp",
                "1",
                "--pp",
                "1",
                "--dp",
                "1",
                "--max-candidates",
                "20",
                "--output-config",
                str(output),
                "--output-justification",
                str(tmp_path / "calibrated_compile.md"),
                "--output-pareto",
                str(tmp_path / "calibrated_compile.csv"),
                "--no-shadow-prices",
                "--quiet",
                env={
                    "AC_QUALITY_DEFAULTS": str(pack_dir / "quality_overrides.json"),
                    "AC_HARDWARE_SPEC_DIR": str(pack_dir / "hardware_specs"),
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.exists())
            config = json.loads(output.read_text())
            self.assertIn(
                "confidence_envelope",
                config["metadata"]["predicted"],
            )

    def test_auto_calibrate_eval_models_feed_compile_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = []
            for i, family in enumerate(["dense", "dense", "moe", "moe", "hybrid", "hybrid"]):
                params_b = 1.0 + i * 0.4
                tokens_t = 0.2 + i * 0.05
                predicted_loss = 2.8 - i * 0.08
                observed_loss = predicted_loss * 1.02
                base_score = 0.32 + i * 0.035 + (0.03 if family == "moe" else 0.0)
                rows.append({
                    "id": f"{family}_{i}",
                    "architecture_family": family,
                    "model_type": family,
                    "active_params_b": params_b,
                    "total_params_b": params_b * (2.0 if family == "moe" else 1.0),
                    "training_tokens": tokens_t,
                    "context_length": 2048,
                    "predicted_loss": predicted_loss,
                    "observed_loss": observed_loss,
                    "predicted_uncertainty_total_pct": 3.0,
                    "eval_scores": {
                        "mmlu_pro": round(base_score, 4),
                    },
                    "predicted_evals": {
                        "mmlu_pro": round(base_score - 0.01, 4),
                    },
                })
            measurements = tmp_path / "eval_measurements.json"
            measurements.write_text(json.dumps(rows))
            pack_dir = tmp_path / "eval_pack"
            result = run_cli(
                "ac/auto_calibrate.py",
                "fit",
                "--measurements",
                str(measurements),
                "--out",
                str(pack_dir),
                "--min-quality-rows",
                "2",
                "--min-eval-rows",
                "2",
                "--min-eval-families",
                "3",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            pack = json.loads((pack_dir / "calibration_pack.json").read_text())
            self.assertEqual(pack["eval_models"]["status"], "validated")
            self.assertIn("mmlu_pro", pack["eval_models"]["evals"])
            self.assertIsNotNone(
                pack["eval_models"]["evals"]["mmlu_pro"]["heldout_family_rmse"]
            )

            output = tmp_path / "eval_calibrated_compile.json"
            pareto = tmp_path / "eval_calibrated_compile.csv"
            result = run_cli(
                "ac/cli_compile.py",
                "--hardware",
                "h100",
                "--params",
                "1",
                "--tokens",
                "0.2",
                "--context",
                "2048",
                "--serving-tbt",
                "100",
                "--serving-batch",
                "4",
                "--tp",
                "1",
                "--pp",
                "1",
                "--dp",
                "1",
                "--max-candidates",
                "20",
                "--output-config",
                str(output),
                "--output-justification",
                str(tmp_path / "eval_calibrated_compile.md"),
                "--output-pareto",
                str(pareto),
                "--no-shadow-prices",
                "--quiet",
                env={
                    "AC_QUALITY_DEFAULTS": str(pack_dir / "quality_overrides.json"),
                    "AC_HARDWARE_SPEC_DIR": str(pack_dir / "hardware_specs"),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(output.read_text())
            predicted = config["metadata"]["predicted"]
            self.assertIn("mmlu_pro", predicted["eval_predictions"])
            self.assertIn("confidence_envelope", predicted)
            self.assertIn("loss_ci_low", pareto.read_text().splitlines()[0])

    def test_default_selection_prefers_best_loss_moe_frontier_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "moe.json"
            pareto = tmp_path / "moe.csv"
            result = run_cli(
                "ac/cli_compile.py",
                "--hardware",
                "h100",
                "--params",
                "1",
                "--tokens",
                "0.2",
                "--context",
                "2048",
                "--serving-tbt",
                "100",
                "--serving-batch",
                "4",
                "--tp",
                "1",
                "--pp",
                "1",
                "--dp",
                "1",
                "--allow-moe",
                "--moe-n-experts",
                "16",
                "--moe-top-k",
                "2",
                "--ep-options",
                "4",
                "--max-candidates",
                "200",
                "--output-config",
                str(output),
                "--output-justification",
                str(tmp_path / "moe.md"),
                "--output-pareto",
                str(pareto),
                "--no-shadow-prices",
                "--quiet",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(output.read_text())
            predicted = config["metadata"]["predicted"]
            self.assertEqual(predicted["selection_diagnostics"]["objective_profile"], "research_quality")
            self.assertEqual(predicted["selection_diagnostics"]["selected_pareto_rank"], 1)
            self.assertEqual(predicted["selection_diagnostics"]["best_loss_pareto_rank"], 1)
            self.assertEqual(config["parallelism"]["expert_parallel"], 4)
            self.assertEqual(config["architecture"]["layer_configs"][0]["ffn"]["type"], "moe")


if __name__ == "__main__":
    unittest.main()
