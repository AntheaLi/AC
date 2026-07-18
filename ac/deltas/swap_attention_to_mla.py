"""swap_attention_to_mla — DeepSeek-V2/V3-style latent KV compression.

We don't have full MLA in the v0 throughput model, so we approximate MLA's
KV behavior by storing a single small latent per token instead of K+V per
head. This is encoded as an aggressively-reduced n_kv_heads with a small
d_head substitute, plus an mla_latent_dim hint on the quality side.
"""

import copy

from .base import (
    Transformation,
    _copy_arch,
    _record_applied,
    _attention_already_swapped,
    _has_attention_layers,
)
try:
    from ..quality_model import ArchConfig as QArchConfig
except ImportError:
    from quality_model import ArchConfig as QArchConfig


class SwapAttentionToMLA(Transformation):
    name = "swap_attention_to_mla"
    expected_stress_signature = {
        "kv_footprint": "large_decrease",
        "hbm_bw_decode": "decrease",
        "tc_util_decode": "small_increase",  # latent decompression
    }

    def precondition(self, arch):
        if not _has_attention_layers(arch):
            return False, "pure-state baseline has no attention layers to replace"
        attention_type = str(getattr(arch, "attention_type", "full") or "full")
        if attention_type != "full":
            return False, (
                f"baseline attention.type={attention_type!r}; MLA cannot be "
                "stacked over another attention family")
        prior = _attention_already_swapped(arch)
        if prior is not None and prior not in (
                self.name, "interleave_local_attention"):
            return False, (
                f"attention block was already swapped by '{prior}'; "
                f"chaining {self.name} on top would silently overwrite it. "
                "Apply attention swaps to a fresh baseline or pick one swap."
            )
        return True, ""

    def apply(self, arch, latent_dim: int = 512, d_rope: int = 64):
        if latent_dim < 16:
            raise ValueError("latent_dim must be >= 16")
        if int(d_rope) <= 0 or int(d_rope) > int(arch.d_head):
            raise ValueError(
                f"d_rope must be in [1, d_head] (d_head={arch.d_head}), "
                f"got {d_rope}")
        out = _copy_arch(arch)
        # Drive the throughput model's real MLA branch by setting the proper
        # TArchConfig fields. The branch in kv_bytes_per_token_per_layer
        # computes per-token KV bytes as (c_kv + d_rope) * bpe when
        # attention_type == "mla", which matches the DeepSeek-V2/V3 storage
        # formula. We also keep the sidecar so the quality-side bridge
        # (to_quality_arch) can read it without re-deriving from the field.
        out.attention_type = "mla"
        out.mla_kv_latent_dim = int(latent_dim)
        out.mla_rope_head_dim = int(d_rope)
        # Wave 22: stamp d_nope explicitly (split-head convention: keep the
        # baseline's per-head dim, carve d_rope out of it). Leaving it 0 made
        # the parameter ledger drop the K/V up-projections and W_O, so an
        # MLA swap on a 7B shed ~1.3B phantom params (and the quality model
        # priced attention as nearly free).
        _dh = int(getattr(arch, "d_head", 128) or 128)
        out.mla_nope_head_dim = _dh - int(d_rope) if _dh > int(d_rope) else _dh
        # n_kv_heads is no longer the storage axis under MLA, but we leave
        # the value at 1 so any code path that still reads it falls back to
        # a single compressed latent. The MLA throughput branch ignores it.
        out.n_kv_heads = 1
        out._mla_latent_dim = int(latent_dim)  # type: ignore[attr-defined]
        # Clear any prior SWA sidecar so the two flags don't silently coexist.
        if hasattr(out, "_swa_window"):
            try:
                delattr(out, "_swa_window")
            except AttributeError:
                pass
        _record_applied(out, self.name)
        return out

    def to_quality_arch(self, arch):
        # Use base to_quality_arch then add MLA hint.
        qa = super().to_quality_arch(arch)
        latent = getattr(arch, "_mla_latent_dim", None)
        if latent is not None:
            qa.attention_type = "mla"
            qa.mla_latent_dim = int(latent)
            qa.mla_q_latent_dim = getattr(arch, "mla_q_latent_dim", None)
            qa.mla_rope_head_dim = getattr(arch, "mla_rope_head_dim", None)
            qa.mla_nope_head_dim = getattr(arch, "mla_nope_head_dim", None)
        return qa
