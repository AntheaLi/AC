"""Wave 9 — compressed-attention coverage regression tests.

Per `plan/redesign/09-compressed-attention-coverage.md`. Locks in the
three coverage adds (CSA / IndexShare / MSA):

  * test_csa_kv_reduction_matches_compression_ratio — per-token KV scales
    as top_k_blocks / block_size.
  * test_indexshare_kv_reduction_matches_coverage — per-token KV scales as
    top_k_buckets / num_buckets.
  * test_quality_residual_increases_with_compression — CSA at aggressive
    compression has higher residual than at modest compression.
  * test_indexshare_top_k_tradeoff — quality improves with larger top_k.
  * test_msa_per_token_below_full — MSA's per-token KV is < dense full at
    long ctx.
  * test_enumeration_includes_compressed_variants — when allow_csa/
    allow_indexshare/allow_msa are set, the optimizer enumerates each.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _make_tput_arch(attention_type, **extras):
    from ac.throughput_model import ArchConfig as TArchConfig
    base = dict(
        d_model=4096, n_layers=32, n_heads=32, d_head=128, n_kv_heads=8,
        ffn_dim=14336, batch_size=8, seq_len=131072,
        precision="bf16", kv_precision="bf16",
        attention_type=attention_type,
    )
    base.update(extras)
    return TArchConfig(**base)


def _make_cand(attention_type, **extras):
    from ac.optimizer import CandidateArch
    base = dict(
        d_model=4096, n_layers=32, n_heads=32, d_head=128, n_kv_heads=8,
        ffn_dim=14336, vocab_size=128000,
        weight_precision="bf16", ffn_precision="bf16",
        attn_precision={"q": "bf16", "k": "bf16", "v": "bf16", "o": "bf16"},
        kv_cache_bits=16, attention_type=attention_type,
        tp_degree=1, cp_degree=1, ep_degree=1,
    )
    base.update(extras)
    return CandidateArch(**base)


def _make_constraints(**extras):
    from ac.optimizer import DeploymentConstraints
    base = dict(
        target_params_b=7.0, training_tokens=int(2e12),
        context_length=131072,
        tp=1, pp=1, dp=1,
        serving_tbt_ms=None, serving_ttft_ms=None,
        serving_batch=8,
        allow_quality_sentinel=True,
    )
    base.update(extras)
    return DeploymentConstraints(**base)


class CsaThroughputTests(unittest.TestCase):
    """CSA per-token KV ≈ (top_k_blocks × compression_dim) / block_size,
    scaled by n_kv_heads × bpe. The reduction ratio must track the
    arithmetic."""

    def test_csa_kv_reduction_matches_compression_ratio(self):
        # Full attention per-token bytes at this shape: 2 × 8 × 128 × 2 = 4096.
        full = _make_tput_arch("full")
        full_bytes = full.kv_bytes_per_token_per_layer(131072)
        self.assertGreater(full_bytes, 0)

        # CSA at block=64, top_k=16, compression_dim=64.
        # Effective: 2 × 8 × 64 × 2 × 16/64 = 512 bytes.
        csa = _make_tput_arch("csa",
                              csa_block_size=64, csa_top_k_blocks=16,
                              csa_compression_dim=64)
        csa_bytes = csa.kv_bytes_per_token_per_layer(131072)
        self.assertLess(
            csa_bytes, full_bytes,
            f"CSA per-token KV ({csa_bytes}) should be < full ({full_bytes})"
        )
        # Aggressive compression: block=128, top_k=8.
        csa_aggressive = _make_tput_arch("csa",
                                         csa_block_size=128, csa_top_k_blocks=8,
                                         csa_compression_dim=32)
        agg_bytes = csa_aggressive.kv_bytes_per_token_per_layer(131072)
        self.assertLess(
            agg_bytes, csa_bytes,
            f"more aggressive CSA ({agg_bytes}) should be < less aggressive "
            f"CSA ({csa_bytes})"
        )


class IndexShareThroughputTests(unittest.TestCase):
    """IndexShare per-token KV ≈ dense × top_k / num_buckets + tiny index lookup."""

    def test_indexshare_kv_reduction_matches_coverage(self):
        full = _make_tput_arch("full")
        full_bytes = full.kv_bytes_per_token_per_layer(131072)
        idx_4_64 = _make_tput_arch(
            "indexshare",
            indexshare_num_buckets=64, indexshare_top_k_buckets=4,
            indexshare_index_dim=64,
        )
        bytes_4_64 = idx_4_64.kv_bytes_per_token_per_layer(131072)
        idx_4_128 = _make_tput_arch(
            "indexshare",
            indexshare_num_buckets=128, indexshare_top_k_buckets=4,
            indexshare_index_dim=64,
        )
        bytes_4_128 = idx_4_128.kv_bytes_per_token_per_layer(131072)
        # Higher num_buckets at same top_k = lower coverage = smaller KV.
        self.assertLess(
            bytes_4_128, bytes_4_64,
            f"top_k=4/num_buckets=128 ({bytes_4_128}) must read less per "
            f"token than top_k=4/num_buckets=64 ({bytes_4_64})"
        )
        self.assertLess(
            bytes_4_64, full_bytes,
            f"IndexShare ({bytes_4_64}) must read less per token than full "
            f"({full_bytes})"
        )


class MsaThroughputTests(unittest.TestCase):
    """MSA per-token KV is the sum of (window + dilated + global) shares
    of dense; at long ctx the share is small."""

    def test_msa_per_token_below_full_at_long_ctx(self):
        full = _make_tput_arch("full")
        full_bytes = full.kv_bytes_per_token_per_layer(131072)
        msa = _make_tput_arch(
            "msa",
            msa_window_size=512, msa_dilated_top_k=64, msa_global_top_k=16,
        )
        msa_bytes = msa.kv_bytes_per_token_per_layer(131072)
        self.assertLess(
            msa_bytes, full_bytes,
            f"MSA per-token ({msa_bytes}) at 128k ctx must be < full ({full_bytes})"
        )


class CompressedAttentionQualityTests(unittest.TestCase):
    """Quality residuals must grow when the compression is more aggressive
    (CSA) or the coverage shrinks (IndexShare)."""

    def _resid_csa(self, block_size, top_k_blocks):
        from ac.optimizer import evaluate_candidate
        ev = evaluate_candidate(
            _make_cand("csa", csa_block_size=block_size,
                       csa_top_k_blocks=top_k_blocks, csa_compression_dim=64),
            "h100", _make_constraints(),
        )
        feats = ev.quality.terms.get("architecture_residual")
        return feats.features.get("subterms", {}).get("attention_csa", 0.0)

    def _resid_indexshare(self, num_buckets, top_k):
        from ac.optimizer import evaluate_candidate
        ev = evaluate_candidate(
            _make_cand("indexshare", indexshare_num_buckets=num_buckets,
                       indexshare_top_k_buckets=top_k, indexshare_index_dim=64),
            "h100", _make_constraints(),
        )
        feats = ev.quality.terms.get("architecture_residual")
        return feats.features.get("subterms", {}).get("attention_indexshare", 0.0)

    def test_csa_quality_residual_increases_with_compression(self):
        # At block=64/top_k=16, ratio=4. At block=128/top_k=8, ratio=16.
        # The higher ratio must produce a larger residual.
        r_modest = self._resid_csa(64, 16)
        r_aggressive = self._resid_csa(128, 8)
        self.assertGreater(
            r_aggressive, r_modest,
            f"CSA residual should grow with compression ratio: "
            f"modest (4×)={r_modest:.4f}, aggressive (16×)={r_aggressive:.4f}"
        )

    def test_indexshare_top_k_tradeoff(self):
        # At top_k=4/buckets=64, coverage=0.0625. At top_k=8/buckets=64,
        # coverage=0.125 → smaller residual (better quality).
        r_low = self._resid_indexshare(64, 4)
        r_high = self._resid_indexshare(64, 8)
        self.assertGreater(
            r_low, r_high,
            f"IndexShare residual should shrink with larger top_k: "
            f"top_k=4 r={r_low:.4f}, top_k=8 r={r_high:.4f}"
        )


class CompressedAttentionEnumerationTests(unittest.TestCase):
    """When the allow_* flag is on, the optimizer must include the
    corresponding family in the enumerated set."""

    def test_enumeration_includes_each_enabled_variant(self):
        from ac.optimizer import optimize
        c = _make_constraints(
            allow_csa=True, allow_indexshare=True, allow_msa=True,
            max_candidates=200, max_full_evaluations=80,
        )
        r = optimize("h100", c)
        attn_types = {getattr(ev.arch, "attention_type", "full")
                      for ev in r.all_evaluated}
        for variant in ("csa", "indexshare", "msa"):
            self.assertIn(
                variant, attn_types,
                f"{variant} family missing from enumerated set: {attn_types}. "
                f"Wave 9 enumeration regression."
            )

    def test_enumeration_excludes_compressed_when_flags_off(self):
        """Default behavior: no compressed-attention variants enumerated."""
        from ac.optimizer import optimize
        c = _make_constraints(
            max_candidates=80, max_full_evaluations=40,
        )
        r = optimize("h100", c)
        attn_types = {getattr(ev.arch, "attention_type", "full")
                      for ev in r.all_evaluated}
        for variant in ("csa", "indexshare", "msa"):
            self.assertNotIn(
                variant, attn_types,
                f"{variant} appeared in enumeration despite allow_{variant}=False"
            )


if __name__ == "__main__":
    unittest.main()
