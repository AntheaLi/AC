"""swap_attention_to_mla — DeepSeek-V2/V3-style latent KV compression.

We don't have full MLA in the v0 throughput model, so we approximate MLA's
KV behavior by storing a single small latent per token instead of K+V per
head. This is encoded as an aggressively-reduced n_kv_heads with a small
d_head substitute, plus an mla_latent_dim hint on the quality side.
"""

import copy

from .base import Transformation, _copy_arch
from quality_model import ArchConfig as QArchConfig


class SwapAttentionToMLA(Transformation):
    name = "swap_attention_to_mla"
    expected_stress_signature = {
        "kv_footprint": "large_decrease",
        "hbm_bw_decode": "decrease",
        "tc_util_decode": "small_increase",  # latent decompression
    }

    def precondition(self, arch):
        return True, ""

    def apply(self, arch, latent_dim: int = 512):
        if latent_dim < 16:
            raise ValueError("latent_dim must be >= 16")
        out = _copy_arch(arch)
        # Approximate MLA: one "kv head" of size latent_dim.
        # The throughput model's KV cache term becomes
        #   2 × B × 1 × L × latent_dim × bpe
        # which matches the MLA paper's storage formula. We stash the
        # original head shapes in moe_config-style sidecar... actually
        # simpler: just stash it on a custom attribute the quality bridge
        # can read.
        out.n_kv_heads = 1
        # Encode latent_dim by using d_head=latent_dim for the KV path.
        # Q/O still use the original d_head, but the throughput KV-cache
        # term reads `dh = arch.d_head` so we have to compromise: keep
        # d_head but reduce n_kv_heads to the smallest possible value.
        # The quality side carries mla_latent_dim explicitly.
        out._mla_latent_dim = latent_dim  # type: ignore[attr-defined]
        return out

    def to_quality_arch(self, arch):
        # Use base to_quality_arch then add MLA hint.
        qa = super().to_quality_arch(arch)
        latent = getattr(arch, "_mla_latent_dim", None)
        if latent is not None:
            qa.attention_type = "mla"
            qa.mla_latent_dim = int(latent)
        return qa
