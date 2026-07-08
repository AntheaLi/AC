"""Regression tests for architecture/parallelism wiring invariants."""

import copy
import os
import tempfile
import unittest
from unittest.mock import patch

from ac.architecture import parameter_ledger, validate_architecture_views
from ac.implementation_generator import generate_pytorch_implementation
from ac.optimizer import (
    CandidateArch,
    DeploymentConstraints,
    evaluate_candidate,
    generate_moe_candidates,
)
from ac.quality_model import (
    DEFAULT_QUALITY_CONSTANTS,
    _QUALITY_CONSTANTS_CACHE,
    load_quality_constants,
)
from ac.throughput_model import ArchConfig, throughput
from ac.stress import Workload, compute_throughput_stress


def _candidate() -> CandidateArch:
    return CandidateArch(
        d_model=4096,
        n_layers=32,
        n_heads=32,
        d_head=128,
        n_kv_heads=8,
        ffn_dim=11008,
        vocab_size=32000,
        total_params=7_000_000_000,
        total_params_b=7.0,
        active_params=7_000_000_000,
        active_params_b=7.0,
        tp_degree=8,
    )


class ParallelismWiringTests(unittest.TestCase):
    def test_moe_default_search_prefers_ep2_but_ep1_is_legal(self):
        # Wave 21: EP=1 is a legal MoE execution plan (experts TP-sharded
        # across the TP group — the standard vLLM TP-only deployment).
        # Explicit ep_options=[1] must be accepted...
        DeploymentConstraints(target_params_b=1.0, allow_moe=True,
                              ep_options=[1])
        # ...while the DEFAULT search space still prefers EP >= 2 (a
        # search-space prior, not a validity constraint).
        candidates = generate_moe_candidates(
            "h100",
            DeploymentConstraints(
                target_params_b=1.0,
                allow_moe=True,
                tp=8,
                tp_options=[8],
            ),
        )
        self.assertTrue(candidates)
        self.assertTrue(all(c.ep_degree >= 2 for c in candidates))
        direct = ArchConfig(
            d_model=1024,
            n_layers=4,
            n_heads=8,
            d_head=128,
            n_kv_heads=8,
            ffn_dim=2048,
            moe_config={"n_experts": 8, "top_k": 2, "expert_dim": 1024},
        )
        # throughput() at EP=1 must run, produce finite serving numbers,
        # and behave as TP-sharded experts: per-GPU memory strictly larger
        # than the same plan at EP=2 (which halves resident experts), and
        # decode not faster than EP=2 (more expert bytes per rank).
        r1 = throughput(direct, "h100", tp_degree=8, ep_degree=1)
        r2 = throughput(direct, "h100", tp_degree=8, ep_degree=2)
        self.assertGreater(r1.memory_footprint_per_gpu_gb, 0.0)
        self.assertGreater(r1.decode_time_per_token_ms, 0.0)
        self.assertGreater(r1.memory_footprint_per_gpu_gb,
                           r2.memory_footprint_per_gpu_gb)

    def test_cross_node_tp_is_costed_not_rejected(self):
        arch = ArchConfig(
            d_model=8192,
            n_layers=80,
            n_heads=64,
            d_head=128,
            n_kv_heads=8,
            ffn_dim=28672,
            batch_size=8,
            seq_len=8192,
        )
        local = throughput(arch, "h100", tp_degree=8, dp_degree=1)
        cross_node = throughput(arch, "h100", tp_degree=16, dp_degree=1)
        self.assertGreater(
            cross_node.per_layer_breakdown.allreduce_s,
            local.per_layer_breakdown.allreduce_s * 5,
        )
        # Higher TP still receives its memory-sharding benefit.
        self.assertLess(
            cross_node.memory_footprint_per_gpu_gb,
            local.memory_footprint_per_gpu_gb,
        )

    def test_dp_sync_and_fsdp_memory_are_optimizer_inputs(self):
        cand = _candidate()
        common = dict(tp=8, tp_options=[8], serving_batch=8, concurrency=8)
        dp1 = evaluate_candidate(
            cand, "h100", DeploymentConstraints(dp=1, **common)
        )
        dp64 = evaluate_candidate(
            cand, "h100", DeploymentConstraints(dp=64, **common)
        )
        self.assertEqual(dp1.throughput.dp_grad_allreduce_s, 0.0)
        self.assertGreater(dp64.throughput.dp_grad_allreduce_s, 0.0)
        self.assertLess(dp64.training_tps, dp1.training_tps)
        self.assertLess(
            dp64.throughput.training_memory_per_gpu_gb,
            dp1.throughput.training_memory_per_gpu_gb,
        )

    def test_moe_parameter_ledger_shards_experts_over_ep(self):
        cand = _candidate()
        cand.moe = {"n_experts": 64, "top_k": 4, "expert_dim": 2048}
        ledger = parameter_ledger(cand)
        self.assertGreater(ledger.total_params, ledger.active_params)
        self.assertLess(
            ledger.local_total_params(tp=8, pp=1, ep=8),
            ledger.local_total_params(tp=8, pp=1, ep=2),
        )

    def test_stress_training_memory_uses_moe_and_dp_sharding(self):
        arch = ArchConfig(
            d_model=4096,
            n_layers=32,
            n_heads=32,
            d_head=128,
            n_kv_heads=8,
            ffn_dim=11008,
            batch_size=2,
            seq_len=2048,
            moe_config={
                "n_experts": 64,
                "top_k": 4,
                "expert_dim": 2048,
            },
        )
        workload = Workload(
            batch_size=2,
            prefill_seq_len=2048,
            decode_kv_len=2048,
        )
        unsharded = compute_throughput_stress(
            arch, "h100", workload, tp_degree=8, ep_degree=2, dp_degree=1
        )
        sharded = compute_throughput_stress(
            arch, "h100", workload, tp_degree=8, ep_degree=8, dp_degree=64
        )
        self.assertLess(sharded.training_mem, unsharded.training_mem)
        self.assertLess(
            sharded.intermediates["training_optimizer_bytes"],
            unsharded.intermediates["training_optimizer_bytes"],
        )


class ArchitectureFeatureWiringTests(unittest.TestCase):
    def test_nsa_and_yoco_change_evaluated_predictions(self):
        cand = _candidate()
        constraints = DeploymentConstraints(
            dp=1,
            tp=8,
            tp_options=[8],
            context_length=131072,
            prompt_len=131072,
            output_len=512,
            serving_batch=8,
            concurrency=8,
        )
        full = evaluate_candidate(cand, "h100", constraints)

        nsa_cand = copy.deepcopy(cand)
        nsa_cand.attention_type = "nsa"
        nsa_cand.nsa_compress_block_size = 64
        nsa_cand.nsa_compress_block_stride = 16
        nsa_cand.nsa_select_block_size = 64
        nsa_cand.nsa_select_top_k = 16
        nsa_cand.nsa_window_size = 512
        nsa = evaluate_candidate(nsa_cand, "h100", constraints)
        self.assertLess(nsa.throughput.prefill_time_ms, full.throughput.prefill_time_ms)

        yoco_cand = copy.deepcopy(cand)
        yoco_cand.yoco_n_self_attn_layers = 4
        yoco = evaluate_candidate(yoco_cand, "h100", constraints)
        self.assertLess(yoco.memory_per_gpu_gb, full.memory_per_gpu_gb)
        self.assertLess(yoco.serving_tbt_ms, full.serving_tbt_ms)

    def test_phase_view_contract_fails_on_mismatched_architecture(self):
        source = _candidate()
        tput = ArchConfig(
            d_model=source.d_model,
            n_layers=source.n_layers,
            n_heads=source.n_heads,
            d_head=source.d_head,
            n_kv_heads=source.n_kv_heads,
            ffn_dim=source.ffn_dim + 1,
            vocab_size=source.vocab_size,
        )
        with self.assertRaisesRegex(ValueError, "ffn_dim"):
            validate_architecture_views(source, tput)

    def test_unsupported_attention_generation_is_fail_closed(self):
        config = {
            "architecture": {
                "d_model": 64,
                "n_layers": 1,
                "vocab_size": 128,
                "layer_configs": [{
                    "layer_idx": [0],
                    "type": "transformer_block",
                    "attention": {
                        "type": "nsa",
                        "n_heads": 1,
                        "n_kv_heads": 1,
                        "d_head": 64,
                    },
                    "ffn": {"type": "swiglu", "ffn_dim": 128},
                }],
            }
        }
        source = generate_pytorch_implementation(config)
        self.assertIn("no native implementation", source)
        self.assertIn("raise NotImplementedError", source)


class ConfigurationDeterminismTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("AC_QUALITY_DEFAULTS", None)
        _QUALITY_CONSTANTS_CACHE.clear()

    def test_builtin_quality_defaults_do_not_import_yaml(self):
        _QUALITY_CONSTANTS_CACHE.clear()
        real_import = __import__

        def no_yaml(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("simulated missing PyYAML")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_yaml):
            loaded = load_quality_constants()
        self.assertEqual(loaded, DEFAULT_QUALITY_CONSTANTS)

    def test_explicit_missing_quality_override_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "missing.yaml")
            with self.assertRaises(FileNotFoundError):
                load_quality_constants(missing)

    def test_unmodeled_traffic_mix_is_not_accepted_as_metadata(self):
        with self.assertRaisesRegex(ValueError, "not yet a calibrated"):
            DeploymentConstraints(traffic_mix={"chat": 1.0})


if __name__ == "__main__":
    unittest.main()
