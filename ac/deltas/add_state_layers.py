"""add_state_layers — convert a ratio of attention layers to state layers."""

from .base import Transformation, _copy_arch


class AddStateLayers(Transformation):
    name = "add_state_layers"
    expected_stress_signature = {
        "hbm_bw_decode": "large_decrease",
        "kv_footprint": "decrease",
        "state_residual": "increase",
        "sram_tile_fit": "small_increase",
    }

    def precondition(self, arch):
        if arch.n_layers < 2:
            return False, "n_layers<2 — nothing to convert"
        return True, ""

    def apply(self, arch, ratio: str = "1:3", state_type: str = "mamba2",
              d_state: int = 128):
        """Convert layers to state at `ratio` (state:attention).

        ratio: "1:3" → 25% state, 75% attention. "1:1" → 50/50. "all" → all state.
        """
        out = _copy_arch(arch)
        if ratio == "all":
            state_frac = 1.0
        else:
            a, b = ratio.split(":")
            state_frac = int(a) / (int(a) + int(b))
        n_state = int(arch.n_layers * state_frac)
        n_state = max(1, min(arch.n_layers - 1 if state_frac < 1.0 else arch.n_layers,
                             n_state))
        # Interleave state layers throughout the stack rather than block them
        # at the front — matches Jamba / Zamba layout.
        layer_types = ["attention"] * arch.n_layers
        # Place state layers at every other position up to n_state.
        if state_frac >= 1.0:
            layer_types = ["state"] * arch.n_layers
        else:
            step = max(1, arch.n_layers // n_state)
            placed = 0
            for i in range(0, arch.n_layers, step):
                if placed < n_state:
                    layer_types[i] = "state"
                    placed += 1
        out.layer_type_list = layer_types
        out.state_config = dict(arch.state_config or {})
        out.state_config.setdefault("d_state", d_state)
        out.state_config.setdefault("state_expansion", 2)
        out.state_config.setdefault("n_heads", arch.n_heads)
        out.state_config.setdefault("d_head", 64)
        out.state_config.setdefault("state_type", state_type)
        return out
