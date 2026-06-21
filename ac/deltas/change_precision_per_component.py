"""change_precision_per_component — set FFN / attention / KV cache precision."""

from .base import Transformation, _copy_arch, _record_applied


class ChangePrecisionPerComponent(Transformation):
    name = "change_precision_per_component"
    expected_stress_signature = {
        "hbm_bw_decode": "decrease",
        "hbm_capacity": "decrease",
        "precision_loss": "increase",
    }

    def precondition(self, arch):
        return True, ""

    def apply(self, arch, kv: str = None, weight: str = None,
              activation: str = None):
        """Override per-component precision.

        Throughput model carries `precision` and `kv_precision` at the
        top level; quality model carries finer per-component overrides
        via component_precisions. We set both consistently.
        """
        out = _copy_arch(arch)
        if weight is not None:
            out.precision = weight
        if kv is not None:
            out.kv_precision = kv
        # Activation precision currently has no throughput-side knob — it
        # lives on the quality side, set via to_quality_arch override.
        out._target_activation_precision = activation  # type: ignore[attr-defined]
        _record_applied(out, self.name)
        return out

    def to_quality_arch(self, arch):
        qa = super().to_quality_arch(arch)
        if arch.precision:
            qa.weight_precision = arch.precision
        if arch.kv_precision:
            qa.kv_precision = arch.kv_precision
        act = getattr(arch, "_target_activation_precision", None)
        if act:
            qa.activation_precision = act
        return qa
