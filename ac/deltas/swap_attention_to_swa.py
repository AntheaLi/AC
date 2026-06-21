"""swap_attention_to_swa — sliding-window attention.

Modeled at the stress level by capping the effective KV-cache length at
window_size. The throughput model doesn't have a native SWA branch, so we
override the decode KV length via a sidecar attribute; the stress reporter
reads it when computing kv_footprint and hbm_bw_decode.
"""

from .base import (
    Transformation,
    _copy_arch,
    _record_applied,
    _attention_already_swapped,
)


class SwapAttentionToSWA(Transformation):
    name = "swap_attention_to_swa"
    expected_stress_signature = {
        "kv_footprint": "decrease_at_long_context",
        "hbm_bw_decode": "decrease",
        "attention_residual": "small_increase",
    }

    def precondition(self, arch):
        prior = _attention_already_swapped(arch)
        if prior is not None and prior != self.name:
            return False, (
                f"attention block was already swapped by '{prior}'; "
                f"chaining {self.name} on top would silently overwrite it. "
                "Apply attention swaps to a fresh baseline or pick one swap."
            )
        return True, ""

    def apply(self, arch, window_size: int = 4096):
        if window_size < 64:
            raise ValueError("window_size must be >= 64")
        out = _copy_arch(arch)
        # Cap the seq_len used for KV bookkeeping. If the workload's
        # decode_kv_len is smaller than the window, the cap has no effect.
        out._swa_window = window_size  # type: ignore[attr-defined]
        # Clear any prior MLA sidecar so the two flags don't silently coexist.
        if hasattr(out, "_mla_latent_dim"):
            try:
                delattr(out, "_mla_latent_dim")
            except AttributeError:
                pass
        _record_applied(out, self.name)
        return out

    def to_quality_arch(self, arch):
        qa = super().to_quality_arch(arch)
        window = getattr(arch, "_swa_window", None)
        if window is not None:
            qa.attention_type = "swa"
            qa.local_window = int(window)
        return qa
