"""swap_attention_to_gqa — set n_kv_heads = n_heads / group_size."""

from .base import Transformation, _copy_arch


class SwapAttentionToGQA(Transformation):
    name = "swap_attention_to_gqa"
    expected_stress_signature = {
        "kv_footprint": "decrease",
        "hbm_bw_decode": "decrease",
        "attention_residual": "small_increase",
    }

    def precondition(self, arch):
        if arch.n_heads <= 1:
            return False, "n_heads<=1 — no heads to group"
        return True, ""

    def apply(self, arch, group_size: int = 8):
        if group_size < 1:
            raise ValueError("group_size must be >= 1")
        out = _copy_arch(arch)
        out.n_kv_heads = max(1, arch.n_heads // group_size)
        return out
