"""Wave 18g — per-layer attention heterogeneity (local:global interleave).

Invariants pinned here:
  1. Throughput: decode TBT, prefill time, and KV memory interpolate
     monotonically between full attention and pure SWA as the local
     fraction grows.
  2. Quality: in-band interleaves (global fraction >= 0.125) are at parity
     with full attention; the locality penalty ramps in below the floor and
     is maximal for pure SWA.
  3. Baseline loader: a two-band (swa + full) config yields the right
     n_local_attn_layers / window and global-band shape.
  4. Delta: interleave_local_attention composes and cuts KV without a
     loss penalty at in-band ratios.
  5. Signature: 0 < n_local < n_layers classifies as pattern local_global.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ac.throughput_model import ArchConfig as TArch, throughput
from ac.quality_model import ArchConfig as QArch, estimate_quality
from ac.architecture import architecture_signature


def _tput(n_local, ctx=131072, window=4096):
    a = TArch(d_model=4096, n_layers=32, n_heads=32, d_head=128, n_kv_heads=8,
              ffn_dim=14336, batch_size=8, seq_len=ctx, precision="bf16",
              kv_precision="bf16", local_window=window,
              n_local_attn_layers=n_local)
    return throughput(a, "h100", tp_degree=8, pp_degree=1,
                      decode_kv_len=ctx, prefill_seq_len=ctx, microbatches=8)


class ThroughputInterleaveTests(unittest.TestCase):
    def test_monotone_between_full_and_pure_swa(self):
        rs = {n: _tput(n) for n in (0, 16, 24, 32)}
        tbt = [rs[n].decode_time_per_token_ms for n in (0, 16, 24, 32)]
        ttft = [rs[n].prefill_time_ms for n in (0, 16, 24, 32)]
        mem = [rs[n].memory_footprint_per_gpu_gb for n in (0, 16, 24, 32)]
        for series, name in ((tbt, "tbt"), (ttft, "ttft"), (mem, "mem")):
            for a, b in zip(series, series[1:]):
                self.assertGreater(a, b,
                    f"{name} must strictly decrease with local fraction: {series}")

    def test_half_local_is_midpoint_of_kv_dominated_axes(self):
        full, half, pure = _tput(0), _tput(16), _tput(32)
        mid = 0.5 * (full.decode_time_per_token_ms + pure.decode_time_per_token_ms)
        self.assertAlmostEqual(half.decode_time_per_token_ms, mid,
                               delta=0.15 * mid)
        mid_mem = 0.5 * (full.memory_footprint_per_gpu_gb
                         + pure.memory_footprint_per_gpu_gb)
        self.assertAlmostEqual(half.memory_footprint_per_gpu_gb, mid_mem,
                               delta=0.15 * mid_mem)

    def test_local_layers_not_priced_as_state(self):
        # A pure interleave has no state_config; memory must stay finite and
        # weights-dominated sane (regression: local layers fell into the
        # hybrid estimator's state bucket).
        r = _tput(16)
        self.assertGreater(r.memory_footprint_per_gpu_gb, 3.0)
        self.assertLess(r.memory_footprint_per_gpu_gb, 60.0)


class QualityInterleaveTests(unittest.TestCase):
    def _loss(self, frac, ctx=131072, window=4096):
        kw = dict(d_model=4096, n_layers=32, n_heads=32, d_head=128,
                  n_kv_heads=8, ffn_dim=14336, vocab_size=128256)
        if frac is not None:
            kw.update(local_window=window, local_attention_fraction=frac)
        r = estimate_quality(QArch(**kw), {"training_tokens": int(20e12)},
                             workload_spec={"context_length": ctx})
        return r.predicted_loss

    def test_in_band_parity_and_below_floor_penalty(self):
        base = self._loss(None)
        # Wave 18h: in-band (global fraction >= 0.125) is NEAR-parity, not
        # exact parity. A hard-zero plateau made locality free right up to
        # the recall floor, so the optimizer always rode the cliff edge
        # (max locality, min window) at zero modeled cost. The shallow
        # residual slope keeps the cost small (well under the whole-model
        # SWA penalty) but monotone in the local fraction.
        prev = base
        for frac in (0.5, 0.75, 0.875):
            cur = self._loss(frac)
            self.assertGreater(cur, base,
                msg=f"local frac {frac} must carry a small nonzero cost")
            self.assertLess(cur - base, 0.5 * (self._loss(1.0) - base),
                msg=f"in-band cost at {frac} must stay shallow")
            self.assertGreaterEqual(cur, prev - 1e-9,
                msg="in-band cost must be monotone in local fraction")
            prev = cur
        # Below the floor the penalty ramps in; pure SWA pays the most.
        self.assertGreater(self._loss(0.9375), self._loss(0.875))
        self.assertGreater(self._loss(1.0), self._loss(0.9375))

    def test_whole_model_swa_legacy_semantics_unchanged(self):
        # fraction unset with a window == legacy Mistral-style whole-model
        # SWA — must equal fraction=1.0 exactly.
        kw = dict(d_model=4096, n_layers=32, n_heads=32, d_head=128,
                  n_kv_heads=8, ffn_dim=14336, vocab_size=128256,
                  local_window=4096)
        legacy = estimate_quality(QArch(**kw), {"training_tokens": int(20e12)},
                                  workload_spec={"context_length": 131072}).predicted_loss
        self.assertAlmostEqual(legacy, self._loss(1.0), places=6)


class BaselineLoaderInterleaveTests(unittest.TestCase):
    def _two_band_config(self):
        even = [i for i in range(32) if i % 2 == 0]
        odd = [i for i in range(32) if i % 2 == 1]
        def band(idx, swa):
            att = {"type": "swa" if swa else "full", "n_heads": 32,
                   "n_kv_heads": 8, "d_head": 128, "rope": True,
                   "kv_cache_bits": 16,
                   "precision": {"qk": "bf16", "v": "bf16", "output": "bf16"}}
            if swa:
                att["window_size"] = 4096
            return {"layer_idx": idx, "type": "transformer_block",
                    "attention": att,
                    "ffn": {"type": "swiglu", "ffn_dim": 14336, "precision": "bf16"},
                    "normalization": {"type": "rmsnorm", "eps": 1e-5,
                                      "precision": "bf16"},
                    "residual_dtype": "bf16", "state": None}
        return {
            "schema_version": "0.3",
            "metadata": {"model_name": "two-band-test",
                         "compiler_version": "v0.3.0",
                         "generated_at": "2026-07-02T00:00:00+00:00",
                         "input_hardware": "h100"},
            "parallelism": {"tensor_parallel": 8, "pipeline_parallel": 1,
                            "data_parallel": 8, "expert_parallel": 1,
                            "context_parallel": 1, "cp_method": "ring"},
            "architecture": {"d_model": 4096, "n_layers": 32,
                             "vocab_size": 32000,
                             "positional_encoding": {"type": "rope",
                                                     "base": 1000000},
                             "layer_configs": [band(even, True), band(odd, False)]},
        }

    def test_loader_extracts_interleave(self):
        from ac.baseline import load_baseline_model
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(self._two_band_config(), f)
            path = f.name
        try:
            bm = load_baseline_model(path)
            c = bm.candidate
            self.assertEqual(int(getattr(c, "n_local_attn_layers", 0)), 16)
            self.assertEqual(int(getattr(c, "swa_window", 0)), 4096)
            # Shape from the global band; projection type stays full.
            self.assertEqual(c.attention_type, "full")
            self.assertEqual(c.n_kv_heads, 8)
        finally:
            os.unlink(path)


class DeltaInterleaveTests(unittest.TestCase):
    def test_registry_and_ratio_validation(self):
        from ac.deltas import REGISTRY
        self.assertIn("interleave_local_attention", REGISTRY)
        xf = REGISTRY["interleave_local_attention"]()
        with self.assertRaises(ValueError):
            xf.apply(TArch(d_model=4096, n_layers=32, n_heads=32, d_head=128,
                           n_kv_heads=8, ffn_dim=14336), ratio="0:1")

    def test_readme_mla_state_chain_preserves_local_interleave(self):
        """The documented GPT-OSS MLA+state chain combines three layer
        kinds. It must not erase state placement while rebuilding the
        baseline's local:global interleave.
        """
        from ac.baseline import load_baseline_model
        from ac.deltas import REGISTRY
        from ac.evaluator import arch_to_candidate
        from ac.optimizer import DeploymentConstraints, evaluate_candidate
        from ac.optimizer_bridge import candidate_to_arch

        baseline = load_baseline_model(
            os.path.join(REPO_ROOT, "configs", "gpt_oss_120b.json")
        ).candidate
        arch = candidate_to_arch(
            baseline, batch_size=1, seq_len=2048
        )
        arch = REGISTRY["swap_attention_to_mla"]().apply(
            arch, latent_dim=256
        )
        arch = REGISTRY["add_state_layers"]().apply(
            arch, ratio="1:3"
        )

        self.assertEqual(arch.layer_type_list.count("state"), 9)
        self.assertGreater(
            arch.layer_type_list.count("local_attention"), 0
        )
        self.assertEqual(
            arch.n_local_attn_layers,
            arch.layer_type_list.count("local_attention"),
        )

        candidate = arch_to_candidate(arch, baseline)
        constraints = DeploymentConstraints(
            target_params_b=candidate.active_params_b,
            training_tokens=int(20e12),
            context_length=2048,
            prompt_len=2048,
            serving_batch=1,
            tp=8,
            pp=1,
            dp=8,
        )
        evaluated = evaluate_candidate(candidate, "h100", constraints)
        self.assertTrue(evaluated.meets_constraints)
        self.assertEqual(candidate.n_state_layers, 9)
        self.assertEqual(candidate.n_attention_layers, 27)


class SignatureInterleaveTests(unittest.TestCase):
    def test_pattern_classification(self):
        base = dict(d_model=4096, n_layers=32, n_heads=32, d_head=128,
                    n_kv_heads=8, ffn_dim=14336, batch_size=1, seq_len=8192)
        mixed = TArch(**base, local_window=4096, n_local_attn_layers=16)
        pure = TArch(**base, local_window=4096, n_local_attn_layers=0)
        # Signature reads swa_window; mirror the field the optimizer uses.
        mixed.swa_window = 4096
        pure.swa_window = 4096
        self.assertEqual(architecture_signature(mixed).attention_pattern,
                         "local_global")
        self.assertEqual(architecture_signature(pure).attention_pattern,
                         "local")


if __name__ == "__main__":
    unittest.main()
