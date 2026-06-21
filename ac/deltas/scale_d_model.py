"""scale_d_model — ±N along tile-aligned lattice."""

from .base import Transformation, _copy_arch, _record_applied


class ScaleDModel(Transformation):
    name = "scale_d_model"
    expected_stress_signature = {
        "hbm_capacity": "varies",
        "training_mem": "varies",
        "shape_law_loss": "varies",
    }

    def precondition(self, arch):
        return True, ""

    def apply(self, arch, delta: int = 0, align: int = 128,
              scale_ffn: bool = True, ffn_align: int = 128):
        """Adjust d_model by delta, rounding to a multiple of `align`.

        By default, `ffn_dim` is rescaled proportionally so the model
        preserves its FFN-to-d_model capacity ratio (`scale_ffn=True`).
        This avoids the trap where a +1024 d_model bump silently dropped
        the FFN ratio below the published-optimal band and predicted *worse*
        loss despite adding parameters — the original delta scaled only
        d_model and n_heads, leaving ffn_dim untouched.

        Pass `scale_ffn=false` to scale only d_model (legacy behaviour) when
        you specifically want to study the d_model / ffn_dim tradeoff axis
        in isolation.
        """
        out = _copy_arch(arch)
        new_d = arch.d_model + delta
        new_d = max(align, round(new_d / align) * align)
        out.d_model = new_d
        # Keep d_head aligned to lattice — round n_heads.
        out.n_heads = max(1, new_d // arch.d_head)
        # n_kv_heads bounded by n_heads.
        out.n_kv_heads = min(arch.n_kv_heads, out.n_heads)
        # Rescale ffn_dim to preserve the model's MLP-attention ratio. The
        # quality model penalises configs that drift outside the published
        # ~8/3 SwiGLU band (or ~4 dense), so a width-only edit that leaves
        # ffn_dim fixed produces a counterintuitive quality drop.
        if scale_ffn and arch.d_model > 0 and arch.ffn_dim > 0:
            target_ratio = arch.ffn_dim / arch.d_model
            new_ffn = int(round(new_d * target_ratio / ffn_align)) * ffn_align
            out.ffn_dim = max(ffn_align, new_ffn)
        _record_applied(out, self.name)
        return out
