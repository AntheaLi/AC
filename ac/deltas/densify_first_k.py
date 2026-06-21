"""densify_first_k — increase n_dense_ffn_layers (DeepSeek-V3 / Qwen3-MoE style)."""

from .base import Transformation, _copy_arch, _record_applied


class DensifyFirstK(Transformation):
    name = "densify_first_k"
    expected_stress_signature = {
        "all_to_all": "decrease",
        "moe_residual": "small_increase",  # less total expert capacity
    }

    def precondition(self, arch):
        if arch.moe_config is None:
            return False, "no moe_config — densify_first_k requires MoE"
        return True, ""

    def apply(self, arch, k: int = 3):
        if k < 0:
            raise ValueError("k must be >= 0")
        out = _copy_arch(arch)
        out.n_dense_ffn_layers = min(k, arch.n_layers - 1)
        _record_applied(out, self.name)
        return out
