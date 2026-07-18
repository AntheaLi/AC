"""swap_attention_to_gqa — set n_kv_heads = n_heads / group_size."""

from .base import (
    Transformation,
    _copy_arch,
    _record_applied,
    _attention_already_swapped,
    _has_attention_layers,
)


class SwapAttentionToGQA(Transformation):
    name = "swap_attention_to_gqa"
    expected_stress_signature = {
        "kv_footprint": "decrease",
        "hbm_bw_decode": "decrease",
        "attention_residual": "small_increase",
    }

    def precondition(self, arch):
        if not _has_attention_layers(arch):
            return False, "pure-state baseline has no attention layers to group"
        attention_type = str(getattr(arch, "attention_type", "full") or "full")
        if attention_type not in ("full", "swa"):
            return False, (
                f"baseline attention.type={attention_type!r}; GQA grouping "
                "only applies to full/GQA attention")
        if arch.n_heads <= 1:
            return False, "n_heads<=1 — no heads to group"
        prior = _attention_already_swapped(arch)
        if prior is not None and prior not in (
                self.name, "interleave_local_attention",
                "swap_attention_to_swa"):
            return False, (
                f"attention block was already swapped by '{prior}'; "
                f"chaining {self.name} on top would silently overwrite the "
                "n_kv_heads chosen by that delta. Apply attention swaps to a "
                "fresh baseline or pick one swap."
            )
        return True, ""

    def apply(self, arch, group_size: int = 8):
        if group_size < 1:
            raise ValueError("group_size must be >= 1")
        if group_size <= arch.n_heads and arch.n_heads % group_size != 0:
            raise ValueError(
                f"group_size={group_size} must divide n_heads={arch.n_heads}; "
                "floor division would create an invalid GQA layout")
        out = _copy_arch(arch)
        raw_kv = arch.n_heads // group_size
        clamped_kv = max(1, raw_kv)
        out.n_kv_heads = clamped_kv
        # Surface a clamp note when group_size exceeds n_heads (so n_kv_heads
        # collapses to 1, i.e. MQA).
        notes = []
        if group_size > arch.n_heads:
            notes.append(
                f"group_size={group_size} exceeds n_heads={arch.n_heads}; "
                f"clamped to n_kv_heads=1 (MQA)."
            )
        if notes:
            prior_notes = list(getattr(out, "_delta_notes", []) or [])
            prior_notes.extend(notes)
            try:
                setattr(out, "_delta_notes", prior_notes)
            except Exception:
                pass
        _record_applied(out, self.name)
        return out
