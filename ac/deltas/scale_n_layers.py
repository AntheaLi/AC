"""scale_n_layers — ±N preserving Chinchilla-like depth/width ratio."""

from .base import Transformation, _copy_arch, _record_applied


class ScaleNLayers(Transformation):
    name = "scale_n_layers"
    expected_stress_signature = {
        "training_mem": "varies",
        "shape_law_loss": "varies",
    }

    def precondition(self, arch):
        return True, ""

    def apply(self, arch, delta: int = 0):
        out = _copy_arch(arch)
        out.n_layers = max(1, arch.n_layers + delta)
        if arch.layer_type_list and len(arch.layer_type_list) != out.n_layers:
            # Truncate or pad the layer type list to match.
            ltl = list(arch.layer_type_list)
            if len(ltl) < out.n_layers:
                ltl = ltl + ["attention"] * (out.n_layers - len(ltl))
            else:
                ltl = ltl[:out.n_layers]
            out.layer_type_list = ltl
        _record_applied(out, self.name)
        return out
