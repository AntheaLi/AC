"""swap_attention_to_swa — sliding-window attention.

Modeled by preserving the served sequence length and capping each local
layer's attended span at window_size. The sidecar is materialized into the
throughput model's native local-attention layer layout by the bridges.
"""

from .base import (
    Transformation,
    _copy_arch,
    _record_applied,
    _attention_already_swapped,
    _has_attention_layers,
)


class SwapAttentionToSWA(Transformation):
    name = "swap_attention_to_swa"
    expected_stress_signature = {
        "kv_footprint": "decrease_at_long_context",
        "hbm_bw_decode": "decrease",
        "attention_residual": "small_increase",
    }

    def precondition(self, arch):
        if not _has_attention_layers(arch):
            return False, "pure-state baseline has no attention layers to window"
        attention_type = str(getattr(arch, "attention_type", "full") or "full")
        if attention_type != "full":
            return False, (
                f"baseline attention.type={attention_type!r}; whole-model SWA "
                "cannot be stacked over another attention family")
        prior = _attention_already_swapped(arch)
        if prior is not None and prior not in (
                self.name, "swap_attention_to_gqa"):
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
        # Record the retained attention window. The served context remains
        # unchanged; downstream bridges materialize all attention layers as
        # local and apply min(context, window) per layer.
        out._swa_window = window_size  # type: ignore[attr-defined]
        out.local_window = window_size
        out.attention_type = "swa"
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
