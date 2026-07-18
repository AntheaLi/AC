"""scale_n_layers — ±N preserving Chinchilla-like depth/width ratio."""

from .base import Transformation, _copy_arch, _record_applied
try:
    from ..architecture import compose_layer_type_list
    from ..lattice_engine import place_attention_layers
except ImportError:
    from architecture import compose_layer_type_list
    from lattice_engine import place_attention_layers


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
            # Preserve mixer and local/global fractions. Padding with only
            # attention used to turn a 75%-state architecture into 37.5%
            # state when depth doubled; truncation made the result depend on
            # which end happened to contain the state bands.
            old_layout = list(arch.layer_type_list)
            old_state = sum(1 for kind in old_layout if kind == "state")
            old_attention = len(old_layout) - old_state
            old_local = sum(
                1 for kind in old_layout if kind == "local_attention")
            target_state = int(round(
                out.n_layers * old_state / max(1, arch.n_layers)))
            target_state = max(0, min(out.n_layers, target_state))
            target_attention = out.n_layers - target_state
            attention_indices = set(place_attention_layers(
                out.n_layers, target_attention,
                getattr(arch, "placement_strategy", "periodic")
                if getattr(arch, "placement_strategy", "periodic")
                in {"first_periodic_last", "interleaved", "periodic"}
                else "periodic"))
            layout = [
                "attention" if idx in attention_indices else "state"
                for idx in range(out.n_layers)
            ]
            target_local = int(round(
                target_attention * old_local / max(1, old_attention)))
            out.layer_type_list = compose_layer_type_list(
                layout, out.n_layers, target_local)
            out.n_local_attn_layers = target_local
            if target_local > 0:
                out._n_local_attn_layers = target_local  # type: ignore[attr-defined]
        # YOCO's K self-decoder layers are a stack fraction. Keep that
        # fraction stable when the stack depth changes.
        old_yoco = int(getattr(arch, "yoco_n_self_attn_layers", 0) or 0)
        if 0 < old_yoco < arch.n_layers:
            out.yoco_n_self_attn_layers = max(
                1, min(out.n_layers, int(round(
                    out.n_layers * old_yoco / arch.n_layers))))
        _record_applied(out, self.name)
        return out
