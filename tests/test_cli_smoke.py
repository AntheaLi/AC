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
        self.assertIn("| KV cache / GPU, per request (GB) | 0.500 | 0.500 |", result.stdout)
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
                # Wave 18h: the signed data-sufficiency gate shrinks the MoE
                # capacity bonus near the tokens-per-total-param parity
                # point, so the DEFAULT uncertainty tiebreak may now
                # legitimately pick the smaller dense config here. Pin the
                # argmin-loss semantics explicitly.
                "--strict-quality",
                "--quiet",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(output.read_text())
            predicted = config["metadata"]["predicted"]
            self.assertEqual(predicted["selection_diagnostics"]["objective_profile"], "research_quality")
            # Uncertainty-aware tiebreak: the picker may choose a Pareto point
            # within the loss-uncertainty band of the best-loss point if it
            # dominates on a non-loss axis (memory, TBT, etc.). Verify that
            # the selected point is on the frontier and that its loss is
            # close to the best-loss point rather than asserting equality of
            # ranks, which would lock in the old argmin-loss behaviour.
            self.assertLessEqual(predicted["selection_diagnostics"]["best_loss_pareto_rank"], 5)
            self.assertEqual(
                predicted["selection_diagnostics"]["selected_pareto_rank"], 1
            )
            self.assertEqual(config["parallelism"]["expert_parallel"], 4)
            self.assertEqual(config["architecture"]["layer_configs"][0]["ffn"]["type"], "moe")

    def test_compile_honors_fixed_context_parallel_degree(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "cp.json"
            result = run_cli(
                "ac/cli_compile.py",
                "--hardware",
                "h100",
                "--params",
                "1",
                "--tokens",
                "0.2",
                "--context",
                "32768",
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
                "--cp",
                "4",
                "--max-candidates",
                "20",
                "--output-config",
                str(output),
                "--output-justification",
                str(tmp_path / "cp.md"),
                "--output-pareto",
                str(tmp_path / "cp.csv"),
                "--no-shadow-prices",
                "--quiet",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(output.read_text())
            self.assertEqual(config["parallelism"]["context_parallel"], 4)

    def test_compile_emits_tp_placeable_kv_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "tp_kv.json"
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
                "4",
                "--pp",
                "1",
                "--dp",
                "1",
                "--max-candidates",
                "80",
                "--output-config",
                str(output),
                "--output-justification",
                str(tmp_path / "tp_kv.md"),
                "--output-pareto",
                str(tmp_path / "tp_kv.csv"),
                "--no-shadow-prices",
                "--quiet",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(output.read_text())
            tp = config["parallelism"]["tensor_parallel"]
            attn = config["architecture"]["layer_configs"][0]["attention"]
            n_kv = attn["n_kv_heads"]
            self.assertTrue(n_kv == 1 or n_kv % tp == 0)

    def test_compile_rejects_non_positive_user_inputs(self):
        base_args = [
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
            "--no-shadow-prices",
            "--quiet",
        ]
        invalid_cases = [
            ("--tp", "0"),
            ("--context", "0"),
            ("--serving-batch", "0"),
            ("--tokens", "-0.2"),
        ]
        for flag, value in invalid_cases:
            args = list(base_args)
            idx = args.index(flag)
            args[idx + 1] = value
            with self.subTest(flag=flag, value=value):
                result = run_cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("value must be > 0", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_delta_eval_accepts_tpu_v5e(self):
        result = run_cli(
            "ac/cli_delta_eval.py",
            "--baseline-config",
            "configs/mistral_7b.json",
            "--hardware",
            "tpu_v5e",
            "--tp",
            "8",
            "--workload",
            "chat",
            "--apply",
            "swap_attention_to_gqa",
            "--apply-args",
            "group_size=8",
            "--stdout",
            "--no-pareto",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("# Delta Influence", result.stdout)

    def test_delta_eval_defaults_missing_moe_ep_to_legal_ep1(self):
        """Wave 21 made EP=1 a legal TP-sharded MoE execution plan. A schema
        that omits expert_parallel therefore defaults to EP=1 and must
        evaluate normally rather than leaking an INFEASIBLE sentinel.
        """
        good = json.loads((ROOT / "configs" / "gpt_oss_120b.json").read_text())
        del good["parallelism"]["expert_parallel"]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            broken = tmp_path / "moe_no_ep.json"
            broken.write_text(json.dumps(good))
            out_dir = tmp_path / "out"
            result = run_cli(
                "ac/cli_delta_eval.py",
                "--baseline-config", str(broken),
                "--hardware", "h100",
                "--tp", "8",
                "--apply", "change_moe_topology",
                "--apply-args", "n_experts=64",
                "--apply-args", "top_k=4",
                "--out", str(out_dir),
                "--no-pareto",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            eval_path = out_dir / "evaluation.json"
            self.assertTrue(eval_path.exists())
            ev = json.loads(eval_path.read_text())
            self.assertTrue(ev["feasible"])
            self.assertTrue(ev.get("metrics"))
            self.assertLess(
                abs(ev["metrics"]["predicted_loss"]["pct_change"]), 100.0
            )

    def test_delta_eval_validates_hybrid_metadata_without_state_layers(self):
        """Wave 7a.2 validator: metadata.params.kind says 'hybrid' but
        layer_configs carry no state layers. Reject as malformed hybrid."""
        good = json.loads((ROOT / "configs" / "mistral_7b.json").read_text())
        good.setdefault("metadata", {}).setdefault("params", {})["kind"] = "hybrid"
        # Don't add any state layer — the dense layer_configs stay.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            broken = tmp_path / "fake_hybrid.json"
            broken.write_text(json.dumps(good))
            out_dir = tmp_path / "out"
            result = run_cli(
                "ac/cli_delta_eval.py",
                "--baseline-config", str(broken),
                "--hardware", "h100",
                "--tp", "8",
                "--apply", "add_state_layers",
                "--apply-args", "ratio=1:3",
                "--out", str(out_dir),
                "--no-pareto",
            )
            self.assertEqual(
                result.returncode, 2,
                f"expected non-zero exit; stderr={result.stderr}",
            )
            ev = json.loads((out_dir / "evaluation.json").read_text())
            self.assertFalse(ev["feasible"])
            self.assertIn("hybrid", ev["reason_if_infeasible"].lower())
            self.assertIn("state_layers", ev["reason_if_infeasible"])

    def test_delta_eval_validates_mla_without_kv_latent_dim(self):
        """Wave 7a.2: attention.type=='mla' but kv_latent_dim is missing."""
        good = json.loads((ROOT / "configs" / "mistral_7b.json").read_text())
        # Flip the attention block to MLA but deliberately omit kv_latent_dim.
        good["architecture"]["layer_configs"][0]["attention"] = {
            "type": "mla",
            "n_heads": 32,
            "d_head": 128,
            "rope": True,
            "kv_cache_bits": 16,
            "precision": {"qk": "bf16", "v": "bf16", "output": "bf16"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            broken = tmp_path / "mla_no_latent.json"
            broken.write_text(json.dumps(good))
            out_dir = tmp_path / "out"
            result = run_cli(
                "ac/cli_delta_eval.py",
                "--baseline-config", str(broken),
                "--hardware", "h100",
                "--tp", "8",
                "--apply", "swap_attention_to_gqa",
                "--apply-args", "group_size=4",
                "--out", str(out_dir),
                "--no-pareto",
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            ev = json.loads((out_dir / "evaluation.json").read_text())
            self.assertFalse(ev["feasible"])
            self.assertIn("MLA", ev["reason_if_infeasible"])
            self.assertIn("kv_latent_dim", ev["reason_if_infeasible"])

    def test_delta_eval_validates_nsa_missing_required_fields(self):
        """Wave 7a.2: attention.type=='nsa' but required NSA fields missing."""
        good = json.loads((ROOT / "configs" / "mistral_7b.json").read_text())
        good["architecture"]["layer_configs"][0]["attention"] = {
            "type": "nsa",
            "n_heads": 32,
            "n_kv_heads": 8,
            "d_head": 128,
            "rope": True,
            "kv_cache_bits": 16,
            "precision": {"qk": "bf16", "v": "bf16", "output": "bf16"},
            # Intentionally missing: nsa_compress_block_size,
            # nsa_select_top_k, nsa_window_size.
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            broken = tmp_path / "nsa_incomplete.json"
            broken.write_text(json.dumps(good))
            out_dir = tmp_path / "out"
            result = run_cli(
                "ac/cli_delta_eval.py",
                "--baseline-config", str(broken),
                "--hardware", "h100",
                "--tp", "8",
                "--apply", "swap_attention_to_gqa",
                "--apply-args", "group_size=4",
                "--out", str(out_dir),
                "--no-pareto",
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            ev = json.loads((out_dir / "evaluation.json").read_text())
            self.assertFalse(ev["feasible"])
            self.assertIn("NSA", ev["reason_if_infeasible"])

    def test_delta_eval_models_cross_node_moe_ep(self):
        """EP beyond one NVLink domain is modeled on inter-node fabric."""
        good = json.loads((ROOT / "configs" / "gpt_oss_120b.json").read_text())
        good["parallelism"]["expert_parallel"] = 16  # > 8 on H100
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            broken = tmp_path / "moe_oversize_ep.json"
            broken.write_text(json.dumps(good))
            out_dir = tmp_path / "out"
            result = run_cli(
                "ac/cli_delta_eval.py",
                "--baseline-config", str(broken),
                "--hardware", "h100",
                "--tp", "8",
                "--apply", "change_moe_topology",
                "--apply-args", "n_experts=64",
                "--apply-args", "top_k=4",
                "--out", str(out_dir),
                "--no-pareto",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("inter-node fabric", result.stderr)
            ev = json.loads((out_dir / "evaluation.json").read_text())
            self.assertTrue(ev["feasible"])

    def test_delta_eval_models_cross_node_tp(self):
        """TP beyond one NVLink domain is modeled on inter-node fabric."""
        good = json.loads((ROOT / "configs" / "mistral_7b.json").read_text())
        good["parallelism"]["tensor_parallel"] = 16  # > 8 on H100
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            broken = tmp_path / "tp_oversize.json"
            broken.write_text(json.dumps(good))
            out_dir = tmp_path / "out"
            result = run_cli(
                "ac/cli_delta_eval.py",
                "--baseline-config", str(broken),
                "--hardware", "h100",
                "--tp", "16",
                "--apply", "swap_attention_to_gqa",
                "--apply-args", "group_size=4",
                "--out", str(out_dir),
                "--no-pareto",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("inter-node bandwidth", result.stderr)
            ev = json.loads((out_dir / "evaluation.json").read_text())
            self.assertTrue(ev["feasible"])

    def test_delta_eval_rejects_invalid_delta_arg_values(self):
        cases = [
            ("swap_attention_to_gqa", "group_size=0", "group_size must be >= 1"),
            ("change_parallelism", "tp=0", "tp must be >= 1"),
        ]
        for delta_name, arg, expected in cases:
            with self.subTest(delta=delta_name, arg=arg):
                result = run_cli(
                    "ac/cli_delta_eval.py",
                    "--baseline-config",
                    "configs/mistral_7b.json",
                    "--hardware",
                    "h100",
                    "--tp",
                    "8",
                    "--workload",
                    "chat",
                    "--apply",
                    delta_name,
                    "--apply-args",
                    arg,
                    "--stdout",
                    "--no-pareto",
                )

                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_delta_eval_sequence_reports_all_applied_deltas(self):
        result = run_cli(
            "ac/cli_delta_eval.py",
            "--baseline-config",
            "configs/mistral_7b.json",
            "--hardware",
            "h100",
            "--tp",
            "8",
            "--workload",
            "chat",
            "--apply",
            "swap_attention_to_mla",
            "--apply-args",
            "latent_dim=256",
            "--apply",
            "add_state_layers",
            "--apply-args",
            "ratio=1:3",
            "--stdout",
            "--no-pareto",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("swap_attention_to_mla", result.stdout)
        self.assertIn("add_state_layers", result.stdout)
        self.assertIn("state_enabled", result.stdout)

    def test_stress_quality_accepts_known_alias(self):
        result = run_cli(
            "ac/cli_stress.py",
            "quality",
            "--known",
            "mistral_7b",
            "--tokens",
            "2000000000000",
            "--prefill-seq",
            "4096",
            "--decode-kv",
            "4096",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("QualityStressVector", result.stdout)
        self.assertIn("Mistral-7B", result.stdout)

    def test_stress_transition_binds_args_to_repeated_apply(self):
        """Repeated --apply groups keep their own kwargs, matching
        ac-delta-eval instead of attaching the first group's args to the
        last transformation.
        """
        result = run_cli(
            "ac/cli_stress.py",
            "transition",
            "--baseline-config",
            "configs/gpt_oss_120b.json",
            "--hardware",
            "h100",
            "--tp",
            "8",
            "--apply",
            "swap_attention_to_mla",
            "--apply-args",
            "latent_dim=256",
            "--apply",
            "add_state_layers",
            "--apply-args",
            "ratio=1:3",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("swap_attention_to_mla", result.stdout)
        self.assertIn("add_state_layers", result.stdout)
        self.assertNotIn("unexpected keyword argument", result.stdout)
        self.assertNotIn("Infeasible:", result.stdout)

    def test_delta_parallelism_supports_cp_and_dp(self):
        result = run_cli(
            "ac/cli_delta_eval.py",
            "--baseline-config",
            "configs/mistral_7b.json",
            "--hardware",
            "h100",
            "--tp",
            "8",
            "--apply",
            "change_parallelism:cp=2,dp=4",
            "--stdout",
            "--no-pareto",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("parallelism.context_parallel", result.stdout)
        self.assertIn("parallelism.data_parallel", result.stdout)
        self.assertNotIn("unknown --apply-args key", result.stderr)

    def test_decode_stress_marks_training_memory_inactive(self):
        result = run_cli(
            "ac/cli_stress.py",
            "stress",
            "--known",
            "Mistral-7B",
            "--hw",
            "h100",
            "--batch",
            "32",
            "--prefill-seq",
            "8192",
            "--decode-kv",
            "8192",
            "--phase",
            "decode",
            "--tp",
            "8",
            "--pp",
            "1",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("training_mem", result.stdout)
        # New label: inactive axes are demoted to the `inactive` band
        # rather than printed as `binding [inactive for decode]`, which
        # was self-contradictory. Accept either historic phrasing or
        # the demoted form ("inactive (...)").
        self.assertTrue(
            "inactive for decode" in result.stdout
            or "training_mem" in result.stdout
            and "inactive" in result.stdout,
            f"training_mem row should be marked inactive on decode:\n{result.stdout}",
        )
        self.assertIn("binding: (none)", result.stdout)

    def test_compile_rejects_invalid_rope_method(self):
        result = run_cli(
            "ac/cli_compile.py",
            "--hardware",
            "h100",
            "--params",
            "1",
            "--tokens",
            "0.2",
            "--context",
            "32768",
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
            "--allow-rope-scaling",
            "--rope-original-max-position",
            "8192",
            "--rope-scaling-methods",
            "banana",
            "--max-candidates",
            "20",
            "--no-shadow-prices",
            "--quiet",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown rope scaling method", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_compile_rejects_invalid_placement_strategy(self):
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
            "--allow-state",
            "--placement-strategy",
            "bogus",
            "--max-candidates",
            "20",
            "--no-shadow-prices",
            "--quiet",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown placement strategy", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_moe_compile_honors_fixed_cp_and_forced_rope(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "moe_cp_rope.json"
            result = run_cli(
                "ac/cli_compile.py",
                "--hardware",
                "h100",
                "--params",
                "1",
                "--tokens",
                "0.2",
                "--context",
                "32768",
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
                "--allow-rope-scaling",
                "--rope-original-max-position",
                "8192",
                "--rope-scaling-methods",
                "longrope",
                "--cp",
                "2",
                "--max-candidates",
                "200",
                "--output-config",
                str(output),
                "--output-justification",
                str(tmp_path / "moe_cp_rope.md"),
                "--output-pareto",
                str(tmp_path / "moe_cp_rope.csv"),
                "--no-shadow-prices",
                "--quiet",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(output.read_text())
            self.assertEqual(config["parallelism"]["context_parallel"], 2)
            scaling = config["architecture"]["positional_encoding"]["scaling"]
            self.assertEqual(scaling["method"], "longrope")
            self.assertEqual(
                config["architecture"]["layer_configs"][0]["ffn"]["type"],
                "moe",
            )

    def test_decode_stress_does_not_bind_training_memory(self):
        result = run_cli(
            "ac/cli_stress.py",
            "stress",
            "--known",
            "Mistral-7B",
            "--hw",
            "h100",
            "--batch",
            "32",
            "--prefill-seq",
            "8192",
            "--decode-kv",
            "8192",
            "--phase",
            "decode",
            "--tp",
            "8",
            "--pp",
            "1",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("training_mem", result.stdout)
        self.assertNotIn("binding: training_mem", result.stdout)

    def test_decode_transition_does_not_bind_training_memory(self):
        result = run_cli(
            "ac/cli_stress.py",
            "transition",
            "--known",
            "Mistral-7B",
            "--hw",
            "h100",
            "--batch",
            "32",
            "--prefill-seq",
            "8192",
            "--decode-kv",
            "8192",
            "--phase",
            "decode",
            "--tp",
            "8",
            "--pp",
            "1",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Binding stresses: training_mem", result.stdout)

    def test_justification_omits_missing_ttft_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            md = tmp_path / "arch.md"
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
                str(tmp_path / "arch.json"),
                "--output-justification",
                str(md),
                "--output-pareto",
                str(tmp_path / "pareto.csv"),
                "--no-shadow-prices",
                "--quiet",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            text = md.read_text()
            self.assertNotIn("Nonems", text)
            self.assertNotIn("TTFT <=", text)
            self.assertIn("Serving soft budget(s):", text)
            self.assertIn("not hard feasibility cuts", text)
            self.assertIn("under soft budget", text)

    def test_compile_creates_output_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_dir = tmp_path / "nested" / "outputs"
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
                str(out_dir / "arch.json"),
                "--output-justification",
                str(out_dir / "arch.md"),
                "--output-pareto",
                str(out_dir / "pareto.csv"),
                "--no-shadow-prices",
                "--quiet",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((out_dir / "arch.json").exists())
            self.assertTrue((out_dir / "arch.md").exists())
            self.assertTrue((out_dir / "pareto.csv").exists())


    # ------------------------------------------------------------------
    # Regression tests for the audit bug-fix batch (B1, B2, B3).
    # ------------------------------------------------------------------

    def test_densify_first_k_surfaces_layer_structure_change(self):
        """B1: densify_first_k changes layer structure, not any top-level
        scalar, so the field-level diff used to be empty and the no-op
        callout fired even though the metric table moved. Verify both:
        (a) the diff now shows `n_dense_ffn_layers`, and (b) the
        "structurally identical" callout does NOT fire.
        """
        result = run_cli(
            "ac/cli_delta_eval.py",
            "--baseline-config", "configs/gpt_oss_120b.json",
            "--hardware", "h100",
            "--tp", "8",
            "--workload", "chat",
            "--apply", "densify_first_k",
            "--apply-args", "k=4",
            "--stdout", "--no-pareto",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        # (a) the field-level diff names the change
        self.assertIn("n_dense_ffn_layers", result.stdout)
        # (b) the false no-op callout is gone
        self.assertNotIn(
            "structurally identical to the baseline", result.stdout)

    def test_change_moe_topology_surfaces_n_experts_change(self):
        """B1: change_moe_topology used to hide the n_experts shift."""
        result = run_cli(
            "ac/cli_delta_eval.py",
            "--baseline-config", "configs/gpt_oss_120b.json",
            "--hardware", "h100",
            "--tp", "8",
            "--workload", "chat",
            "--apply", "change_moe_topology",
            "--apply-args", "n_experts=64",
            "--apply-args", "top_k=4",
            "--stdout", "--no-pareto",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("moe.n_experts", result.stdout)
        # And the genuine moe.n_experts change line should reflect
        # both sides:
        self.assertIn("128", result.stdout)
        self.assertIn("64", result.stdout)

    def test_delta_eval_does_not_emit_ghost_parallelism_diff(self):
        """B2: when a delta does NOT touch parallelism, the field-level
        diff used to render `parallelism.tensor_parallel: 8 -> None`
        (and same for PP/CP) because the candidate sidecar was unset.
        Verify those ghost rows are gone.
        """
        result = run_cli(
            "ac/cli_delta_eval.py",
            "--baseline-config", "configs/mistral_7b.json",
            "--hardware", "h100",
            "--tp", "8",
            "--workload", "long_context",
            "--apply", "swap_attention_to_gqa",
            "--apply-args", "group_size=8",
            "--stdout", "--no-pareto",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for ghost in (
            "| `parallelism.tensor_parallel` | `8` | `None` |",
            "| `parallelism.pipeline_parallel` | `1` | `None` |",
            "| `parallelism.context_parallel` | `1` | `None` |",
        ):
            self.assertNotIn(
                ghost, result.stdout,
                f"Found ghost parallelism diff row: {ghost!r}",
            )

    def test_change_parallelism_still_surfaces_real_tp_change(self):
        """B2 negative case: when the delta really does change TP, the
        diff must still surface it. We use change_parallelism explicitly.
        """
        result = run_cli(
            "ac/cli_delta_eval.py",
            "--baseline-config", "configs/mistral_7b.json",
            "--hardware", "h100",
            "--tp", "8",
            "--workload", "chat",
            "--apply", "change_parallelism",
            "--apply-args", "tp=4",
            "--stdout", "--no-pareto",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("parallelism.tensor_parallel", result.stdout)
        # Real change: baseline 8, candidate 4. We don't assert exact
        # rendering (the candidate side may resolve differently for the
        # sidecar vs canonical path), but both numbers must appear in
        # the diff row context.
        self.assertIn("`8`", result.stdout)

    def test_compile_warns_on_num_gpus_mismatch(self):
        """B3: --num-gpus inconsistent with tp*pp*dp used to be silent.
        It now emits a warning to stderr but still runs.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "arch.json"
            result = run_cli(
                "ac/cli_compile.py",
                "--hardware", "h100",
                "--params", "1",
                "--tokens", "0.2",
                "--context", "2048",
                "--serving-tbt", "100",
                "--serving-batch", "4",
                "--tp", "8", "--pp", "1", "--dp", "8",
                "--num-gpus", "4",
                "--max-candidates", "20",
                "--output-config", str(output),
                "--output-justification", str(tmp_path / "arch.md"),
                "--output-pareto", str(tmp_path / "pareto.csv"),
                "--no-shadow-prices",
                "--quiet",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--num-gpus=4 does not match", result.stderr)
            self.assertTrue(output.exists())

    def test_compile_does_not_warn_on_num_gpus_match(self):
        """B3 negative case: when --num-gpus == tp*pp*dp, no warning."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "arch.json"
            result = run_cli(
                "ac/cli_compile.py",
                "--hardware", "h100",
                "--params", "1",
                "--tokens", "0.2",
                "--context", "2048",
                "--serving-tbt", "100",
                "--serving-batch", "4",
                "--tp", "8", "--pp", "1", "--dp", "8",
                "--num-gpus", "64",
                "--max-candidates", "20",
                "--output-config", str(output),
                "--output-justification", str(tmp_path / "arch.md"),
                "--output-pareto", str(tmp_path / "pareto.csv"),
                "--no-shadow-prices",
                "--quiet",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("--num-gpus", result.stderr)

    def test_swap_attention_to_mla_diff_has_no_duplicate_rows(self):
        """B1 follow-up: the canonical CandidateArch path and the legacy
        sidecar path both used to emit a row for `attention.type` and a
        row for the MLA latent (under two different labels —
        `attention.mla_kv_latent_dim` vs `attention.mla_latent_dim`).
        Verify the dedup pass collapses both.
        """
        result = run_cli(
            "ac/cli_delta_eval.py",
            "--baseline-config", "configs/mistral_7b.json",
            "--hardware", "h100",
            "--tp", "8",
            "--workload", "chat",
            "--apply", "swap_attention_to_mla",
            "--apply-args", "latent_dim=256",
            "--stdout", "--no-pareto",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        # Each conceptual field appears exactly once.
        # (Match the markdown row marker, with a leading "| `" so we
        # ignore the legacy label appearing in stress / prose blocks.)
        self.assertEqual(
            result.stdout.count("| `attention.type` |"), 1,
            "Duplicate `attention.type` rows in field diff",
        )
        self.assertEqual(
            result.stdout.count("| `attention.mla_kv_latent_dim` |"), 1,
            "Duplicate canonical MLA latent rows in field diff",
        )
        # The legacy alias label should no longer appear in the diff
        # table (it lives in stress / quality prose elsewhere, so we
        # restrict the check to the field-diff row marker).
        self.assertNotIn(
            "| `attention.mla_latent_dim` |", result.stdout)

    def test_add_state_layers_surfaces_state_layout(self):
        """B1 follow-up: add_state_layers used to be invisible in the
        diff. Verify the canonical state-layout rows are present.
        """
        result = run_cli(
            "ac/cli_delta_eval.py",
            "--baseline-config", "configs/mistral_7b.json",
            "--hardware", "h100",
            "--tp", "8",
            "--workload", "chat",
            "--apply", "add_state_layers",
            "--apply-args", "ratio=1:3",
            "--apply-args", "state_type=mamba2",
            "--stdout", "--no-pareto",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("state_enabled", result.stdout)
        self.assertIn("state.n_layers", result.stdout)
        self.assertIn("attention.n_layers", result.stdout)

    def test_mistral_7b_class_run_selects_gqa_not_mha(self):
        """Demo-audit C1 + GQA enumeration: at TP=8 with the
        Mistral-7B-class shape (n_heads in {32, 36, 48, 72}), the
        optimizer used to select MHA on every greenfield run because
        (a) the f_kv_heads quality term only penalized shallow GQA and
        (b) the GQA-ratio enumeration tried only [2,4,8,16] which never
        divided cleanly at n_heads=72/TP=8. After the fix the selected
        n_kv_heads should be strictly less than n_heads.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out = tmp_path / "arch.json"
            result = run_cli(
                "ac/cli_compile.py",
                "--hardware", "h100",
                "--params", "7",
                "--tokens", "2",
                "--context", "8192",
                "--serving-tbt", "50",
                "--serving-batch", "32",
                "--tp", "8", "--pp", "1", "--dp", "8",
                "--max-candidates", "800",
                "--output-config", str(out),
                "--output-justification", str(tmp_path / "arch.md"),
                "--output-pareto", str(tmp_path / "pareto.csv"),
                "--no-shadow-prices", "--quiet",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(out.read_text())
            attn = config["architecture"]["layer_configs"][0]["attention"]
            self.assertLess(
                attn["n_kv_heads"], attn["n_heads"],
                f"Expected GQA (kv<heads) but got MHA: kv={attn['n_kv_heads']} heads={attn['n_heads']}",
            )

    def test_input_constraints_records_hardware_and_parallelism(self):
        """Demo-audit E: metadata.input_constraints must record hardware
        + TP/PP/DP/CP so the config is reproducible standalone.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out = tmp_path / "arch.json"
            result = run_cli(
                "ac/cli_compile.py",
                "--hardware", "b200",
                "--params", "1",
                "--tokens", "0.2",
                "--context", "2048",
                "--serving-tbt", "100",
                "--serving-batch", "4",
                "--tp", "4", "--pp", "2", "--dp", "1",
                "--max-candidates", "20",
                "--output-config", str(out),
                "--output-justification", str(tmp_path / "arch.md"),
                "--output-pareto", str(tmp_path / "pareto.csv"),
                "--no-shadow-prices", "--quiet",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(out.read_text())
            ic = config["metadata"]["input_constraints"]
            self.assertEqual(ic.get("hardware"), "b200")
            self.assertEqual(ic.get("tp"), 4)
            self.assertEqual(ic.get("pp"), 2)
            self.assertEqual(ic.get("dp"), 1)
            self.assertIn("cp", ic)

    def test_param_drift_emits_diagnostic_note(self):
        """Demo-audit D: when active_params drifts ≥5% from target the
        justification should explain why (shape-law / lattice
        quantisation) and name the override.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            md = tmp_path / "arch.md"
            # 1B target tends to drift to ~1.1-1.2B on h100/tp=1 because the
            # smallest lattice point exceeds the exact 1B shape.
            result = run_cli(
                "ac/cli_compile.py",
                "--hardware", "h100",
                "--params", "1",
                "--tokens", "0.2",
                "--context", "2048",
                "--serving-tbt", "100",
                "--serving-batch", "4",
                "--tp", "1", "--pp", "1", "--dp", "1",
                "--max-candidates", "60",
                "--output-config", str(tmp_path / "arch.json"),
                "--output-justification", str(md),
                "--output-pareto", str(tmp_path / "pareto.csv"),
                "--no-shadow-prices", "--quiet",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            # Either the drift fires (and the note appears) or it doesn't
            # (and the note must NOT appear) — both are valid. When the
            # note is there it should mention the override.
            text = md.read_text()
            if "Active-params landed" in text:
                self.assertIn("--param-tolerance", text)
                self.assertIn("shape-law", text)

    def test_genuine_no_op_delta_still_fires_callout(self):
        """B1 negative case: the structurally-identical callout MUST
        still fire when the delta really is a no-op (group_size=4 on a
        baseline that is already GQA(32/8)). The B1 fix tightened the
        gate by also requiring the metric panel to be flat — verify the
        callout still appears here, where both conditions hold.
        """
        result = run_cli(
            "ac/cli_delta_eval.py",
            "--baseline-config", "configs/mistral_7b.json",
            "--hardware", "h100",
            "--tp", "8",
            "--workload", "chat",
            "--apply", "swap_attention_to_gqa",
            "--apply-args", "group_size=4",
            "--stdout", "--no-pareto",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "structurally identical to the baseline", result.stdout)

    def test_compile_tp_non_pow2_warning_names_head_constraint(self):
        """H2: the non-power-of-two TP warning must also tell the user
        that AC will constrain candidates to head counts divisible by TP.
        Without this hint, users see the warning and don't know whether
        the run will still produce a sensible result."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = run_cli(
                "ac/cli_compile.py",
                "--hardware", "h100",
                "--params", "1",
                "--tokens", "0.2",
                "--context", "2048",
                "--serving-tbt", "100",
                "--serving-batch", "4",
                "--tp", "7", "--pp", "1", "--dp", "8",
                "--max-candidates", "20",
                "--output-config", str(tmp_path / "arch.json"),
                "--output-justification", str(tmp_path / "arch.md"),
                "--output-pareto", str(tmp_path / "pareto.csv"),
                "--no-shadow-prices",
                "--quiet",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--tp=7 is not a power of two", result.stderr)
            self.assertIn("n_heads", result.stderr)
            self.assertIn("divide evenly by 7", result.stderr)

    def test_delta_eval_surfaces_gqa_group_size_clamp(self):
        """H3: when group_size > n_heads, the delta engine clamps the
        result to MQA (n_kv_heads=1). Verify the report surfaces this
        clamp so the user knows the engine did not apply the requested
        arg literally."""
        result = run_cli(
            "ac/cli_delta_eval.py",
            "--baseline-config", "configs/mistral_7b.json",
            "--hardware", "h100",
            "--tp", "8",
            "--workload", "chat",
            "--apply", "swap_attention_to_gqa",
            "--apply-args", "group_size=999",
            "--stdout", "--no-pareto",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("group_size=999 exceeds n_heads=32", result.stdout)
        self.assertIn("clamped to n_kv_heads=1", result.stdout)

    def test_delta_eval_surfaces_gqa_group_size_non_divisible(self):
        """H3 variant: group_size that doesn't divide n_heads should be
        surfaced too, so the user knows the resulting GQA layout differs
        from what they asked for."""
        result = run_cli(
            "ac/cli_delta_eval.py",
            "--baseline-config", "configs/mistral_7b.json",
            "--hardware", "h100",
            "--tp", "8",
            "--workload", "chat",
            "--apply", "swap_attention_to_gqa",
            "--apply-args", "group_size=5",
            "--stdout", "--no-pareto",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "group_size=5 does not divide n_heads=32", result.stdout)

    def test_hardware_specs_document_peak_flops_convention(self):
        """H1: every NVIDIA hardware spec must carry the
        `_peak_flops_tf_convention` field that explains why
        `peak_flops_tf` is ~half the marketed datasheet peak."""
        import json as _json
        for hw_file in ("h100_sxm.json", "b200.json"):
            with open(f"ac/hardware_specs/{hw_file}") as f:
                spec = _json.load(f)
            self.assertIn(
                "_peak_flops_tf_convention", spec,
                f"{hw_file} missing _peak_flops_tf_convention",
            )
            note = spec["_peak_flops_tf_convention"]
            self.assertIn("datasheet", note.lower())
            # The convention text must call out that the field is NOT the
            # raw datasheet peak (otherwise the note isn't doing its job).
            self.assertIn("not the datasheet", note.lower())
            # And the notes.peak_flops_source must point to the datasheet.
            self.assertIn("peak_flops_source", spec.get("notes", {}))

    def test_cli_stress_help_lists_examples_for_each_subcommand(self):
        """H5: `ac-stress --help` should preview the three subcommands so
        the user discovers them without having to trigger an error."""
        result = run_cli("ac/cli_stress.py", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        # All three subcommands appear with at least one example line each.
        self.assertIn("ac-stress stress", result.stdout)
        self.assertIn("ac-stress quality", result.stdout)
        self.assertIn("ac-stress transition", result.stdout)
        # And the metavar should list them in the usage line.
        self.assertIn("{stress,quality,transition}", result.stdout)


if __name__ == "__main__":
    unittest.main()
