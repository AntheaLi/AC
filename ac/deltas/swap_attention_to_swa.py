"""swap_attention_to_swa — sliding-window attention.

Modeled at the stress level by capping the effective KV-cache length at
window_size. The throughput model doesn't have a native SWA branch, so we
override the decode KV length via a sidecar attribute; the stress reporter
reads it when computing kv_footprint and hbm_bw_decode.
"""

from .base import Transformation, _copy_arch


class SwapAttentionToSWA(Transformation):
    name = "swap_attention_to_swa"
    expected_stress_signature = {
        "kv_footprint": "decrease_at_long_context",
        "hbm_bw_decode": "decrease",
        "attention_residual": "small_increase",
    }

    def precondition(self, arch):
        return True, ""

    def apply(self, arch, window_size: int = 4096):
        if window_size < 64:
            raise ValueError("window_size must be >= 64")
        out = _copy_arch(arch)
        # Cap the seq_len used for KV bookkeeping. If the workload's
        # decode_kv_len is smaller than the window, the cap has no effect.
        out._swa_window = window_size  # type: ignore[attr-defined]
        # We also flag attention_type for the quality side.
        return out

    def to_quality_arch(self, arch):
        qa = super().to_quality_arch(arch)
        window = getattr(arch, "_swa_window", None)
        if window is not None:
            qa.attention_type = "swa"
            qa.local_window = int(window)
        return qa
