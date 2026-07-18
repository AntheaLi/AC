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

        `precision` is the throughput model's dominant matmul tier while
        `weight_precision` preserves the global quality/config identity.
        Activation precision controls activation and communication bytes.
        """
        out = _copy_arch(arch)
        if weight is not None:
            out.precision = weight
            out.weight_precision = weight
        if kv is not None:
            out.kv_precision = kv
        if activation is not None:
            out.activation_precision = activation
            # Backward-compatible sidecar for older callers that inspect it.
            out._target_activation_precision = activation  # type: ignore[attr-defined]
        _record_applied(out, self.name)
        return out

    def to_quality_arch(self, arch):
        qa = super().to_quality_arch(arch)
        if getattr(arch, "weight_precision", None):
            qa.weight_precision = arch.weight_precision
        if arch.kv_precision:
            qa.kv_precision = arch.kv_precision
        act = (getattr(arch, "activation_precision", None)
               or getattr(arch, "_target_activation_precision", None))
        if act:
            qa.activation_precision = act
        return qa
