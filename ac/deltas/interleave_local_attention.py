"""interleave_local_attention — Wave 18g per-layer attention heterogeneity.

Converts a uniform-attention baseline into a local:global interleave
(GPT-OSS / Gemma-2 / Llama-4 pattern): `ratio` = "L:G" makes L of every
L+G layers sliding-window attention at `window` tokens, the rest stay
global under the baseline's projection (full/GQA or MLA). Unlike
swap_attention_to_swa (whole-model window), the global layers keep the
full-context KV, so long-range recall is preserved while local layers
cut prefill compute and KV memory.
"""

from .base import (
    Transformation,
    _copy_arch,
    _record_applied,
    _attention_already_swapped,
    _has_attention_layers,
)
try:
    from ..architecture import compose_layer_type_list
except ImportError:
    from architecture import compose_layer_type_list


def _parse_ratio(ratio: str):
    parts = str(ratio).split(":")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        raise ValueError(
            f"ratio must be 'L:G' with positive integers (e.g. '3:1'), got {ratio!r}")
    l, g = int(parts[0]), int(parts[1])
    if l < 1 or g < 1:
        raise ValueError(f"ratio sides must both be >= 1, got {ratio!r}")
    return l, g


class InterleaveLocalAttention(Transformation):
    name = "interleave_local_attention"
    expected_stress_signature = {
        "kv_footprint": "decrease_at_long_context",
        "hbm_bw_decode": "decrease",
        "attention_residual": "small_increase_gated_by_global_presence",
    }

    def precondition(self, arch):
        if not _has_attention_layers(arch):
            return False, "pure-state baseline has no attention layers to interleave"
        attention_type = str(getattr(arch, "attention_type", "full") or "full")
        if attention_type not in ("full", "mla"):
            return False, (
                f"baseline attention.type={attention_type!r}; local/global "
                "interleave is calibrated only for full/GQA or MLA globals")
        prior = _attention_already_swapped(arch)
        if prior is not None and prior not in (
                self.name, "swap_attention_to_mla", "swap_attention_to_gqa"):
            return False, (
                f"attention block was already swapped by '{prior}'; "
                f"apply {self.name} to a fresh baseline or pick one swap."
            )
        if (getattr(arch, "_n_local_attn_layers", None)
                or int(getattr(arch, "n_local_attn_layers", 0) or 0) > 0):
            return False, "baseline already has a local:global interleave."
        return True, ""

    def apply(self, arch, ratio: str = "1:1", window: int = 4096):
        window = int(window)
        if window < 64:
            raise ValueError("window must be >= 64")
        l_part, g_part = _parse_ratio(ratio)
        out = _copy_arch(arch)
        n_layers = int(getattr(out, "n_layers", 0) or 0)
        layer_types = list(getattr(out, "layer_type_list", None) or [])
        n_attention = (
            sum(1 for kind in layer_types if kind != "state")
            if len(layer_types) == n_layers else n_layers)
        n_local = int(round(n_attention * l_part / (l_part + g_part)))
        if not (0 < n_local < n_attention):
            raise ValueError(
                f"ratio {ratio} on {n_attention} attention layers leaves no local or no "
                f"global layers; use swap_attention_to_swa for whole-model SWA")
        out._swa_window = window                    # type: ignore[attr-defined]
        out._n_local_attn_layers = n_local          # type: ignore[attr-defined]
        out.local_window = window
        out.n_local_attn_layers = n_local
        out.layer_type_list = compose_layer_type_list(
            out.layer_type_list, n_layers, n_local)
        _record_applied(out, self.name)
        return out

    def to_quality_arch(self, arch):
        qa = super().to_quality_arch(arch)
        window = getattr(arch, "_swa_window", None)
        n_local = getattr(arch, "_n_local_attn_layers", None)
        n_layers = int(getattr(arch, "n_layers", 0) or 0)
        if window and n_local and n_layers:
            # Keep the global layers' projection type; the quality model
            # gates the locality penalty on global presence.
            qa.local_window = int(window)
            qa.local_attention_fraction = float(n_local) / float(n_layers)
        return qa
