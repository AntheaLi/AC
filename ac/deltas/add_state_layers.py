"""add_state_layers — convert a ratio of attention layers to state layers."""

from .base import Transformation, _copy_arch, _record_applied


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

    def apply(self, arch, ratio: str = None, state_type: str = "mamba2",
              d_state: int = 128, state_fraction: float = None,
              state_layers: int = None):
        """Convert layers to state. Three accepted ways to specify the count:

          1. ``state_fraction=0.875``  — clearest. Fraction of layers that
             become state. Matches the way the literature describes hybrid
             models ("Jamba is 7/8 state, 1/8 attention").
          2. ``state_layers=28``      — absolute count of state layers.
          3. ``ratio="1:7"``           — **state:attention** ratio.
             "1:7" → 1 state per 7 attention = 12.5% state, 87.5% attention.
             "7:1" → 7 state per 1 attention = 87.5% state, 12.5% attention
             (the Jamba/Zamba-style regime AC's quality model is calibrated to).
             "all" → all layers become state.

        These are alternatives; specify exactly one. The CLI was historically
        ``ratio`` only and the parsing direction (state:attention) is the
        opposite of how the literature usually reads "1:N hybrid", which is a
        documented footgun — prefer ``state_fraction`` for new code.
        """
        # Resolve the three input modes into a single state_fraction. Reject
        # ambiguous combinations early so users get a clear error instead of
        # silent precedence rules.
        spec_count = sum(x is not None for x in (ratio, state_fraction, state_layers))
        if spec_count > 1:
            raise ValueError(
                "add_state_layers: specify exactly one of "
                "`ratio` / `state_fraction` / `state_layers`."
            )
        if spec_count == 0:
            ratio = "1:3"  # historical default

        if state_fraction is not None:
            sf = float(state_fraction)
            if not (0.0 < sf <= 1.0):
                raise ValueError(
                    f"state_fraction must be in (0, 1], got {state_fraction}"
                )
            state_frac = sf
        elif state_layers is not None:
            sl = int(state_layers)
            if not (1 <= sl <= arch.n_layers):
                raise ValueError(
                    f"state_layers must be in [1, {arch.n_layers}], got {state_layers}"
                )
            state_frac = sl / arch.n_layers
        else:
            if ratio == "all":
                state_frac = 1.0
            else:
                try:
                    a, b = ratio.split(":")
                    a, b = int(a), int(b)
                except (ValueError, AttributeError):
                    raise ValueError(
                        f"ratio must be 'all' or positive A:B integers (got {ratio!r})"
                    )
                if a < 0 or b < 0 or (a + b) == 0:
                    raise ValueError(
                        f"ratio must be positive A:B (got {ratio!r})"
                    )
                state_frac = a / (a + b)

        out = _copy_arch(arch)
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
        # Record the resolved interpretation on the candidate so the report
        # echoes "X state layers / Y total → Z% state" — kills the
        # ratio-direction confusion at the source.
        actual_state = sum(1 for lt in layer_types if lt == "state")
        out._state_layer_summary = {  # type: ignore[attr-defined]
            "state_layers": actual_state,
            "attention_layers": arch.n_layers - actual_state,
            "state_fraction": actual_state / arch.n_layers,
            "requested_via": (
                "state_fraction" if state_fraction is not None
                else "state_layers" if state_layers is not None
                else f"ratio={ratio}"
            ),
        }
        _record_applied(out, self.name)
        return out
