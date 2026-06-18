"""scale_d_model — ±N along tile-aligned lattice."""

from .base import Transformation, _copy_arch


class ScaleDModel(Transformation):
    name = "scale_d_model"
    expected_stress_signature = {
        "hbm_capacity": "varies",
        "training_mem": "varies",
        "shape_law_loss": "varies",
    }

    def precondition(self, arch):
        return True, ""

    def apply(self, arch, delta: int = 0, align: int = 128):
        """Adjust d_model by delta, rounding to a multiple of `align`."""
        out = _copy_arch(arch)
        new_d = arch.d_model + delta
        new_d = max(align, round(new_d / align) * align)
        out.d_model = new_d
        # Keep d_head aligned to lattice — round n_heads.
        out.n_heads = max(1, new_d // arch.d_head)
        # n_kv_heads bounded by n_heads.
        out.n_kv_heads = min(arch.n_kv_heads, out.n_heads)
        return out
