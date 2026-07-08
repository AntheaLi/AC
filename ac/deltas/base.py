"""
Transformation protocol — base class for the delta library.

Each transformation:
    1. precondition(arch)   — does it make sense for this baseline?
    2. apply(arch, **params) — return a new ArchConfig with the change applied.
                               Must not mutate the input.
    3. expected_stress_signature — documentation dict naming the axes that
                                   typically move, with direction signs.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any, Dict, Optional

try:
    from ..throughput_model import ArchConfig as TArchConfig
    from ..quality_model import ArchConfig as QArchConfig
except ImportError:
    from throughput_model import ArchConfig as TArchConfig
    from quality_model import ArchConfig as QArchConfig


_SIDECAR_ATTRS = (
    "_mla_latent_dim",
    "_swa_window",
    "_n_local_attn_layers",  # Wave 18g: local:global interleave
    "_tp_override", "_pp_override", "_ep_override", "_cp_override",
    "_dp_override",
    # Provenance: the ordered list of delta names already applied to this
    # arch. Used by compose-time precondition checks so a later
    # transformation can refuse to silently overwrite an earlier one.
    "_applied_deltas",
)


def _copy_arch(arch: TArchConfig) -> TArchConfig:
    """Deep-ish copy of an ArchConfig so transformations don't mutate input.

    Preserves the small set of sidecar attributes the delta library uses to
    communicate non-dataclass facts (MLA latent dim, SWA window, parallelism
    overrides, applied-delta provenance) so multi-delta composition does not
    silently drop them.
    """
    fields = asdict(arch)
    # Deep-copy nested dicts/lists so a candidate's moe_config / state_config
    # are independent of the baseline's.
    for k, v in fields.items():
        if isinstance(v, (dict, list)):
            fields[k] = copy.deepcopy(v)
    out = TArchConfig(**fields)
    for attr in _SIDECAR_ATTRS:
        if hasattr(arch, attr):
            setattr(out, attr, copy.deepcopy(getattr(arch, attr)))
    return out


def _record_applied(arch: TArchConfig, name: str) -> None:
    """Append `name` to the arch's applied-deltas trail (in place)."""
    trail = list(getattr(arch, "_applied_deltas", []) or [])
    trail.append(name)
    arch._applied_deltas = trail  # type: ignore[attr-defined]


def _has_applied(arch: TArchConfig, name: str) -> bool:
    return name in (getattr(arch, "_applied_deltas", []) or [])


def _attention_already_swapped(arch: TArchConfig) -> Optional[str]:
    """Return the name of any prior attention-swap delta, or None."""
    trail = getattr(arch, "_applied_deltas", []) or []
    for n in trail:
        if n in ("swap_attention_to_mla",
                 "swap_attention_to_swa",
                 "swap_attention_to_gqa"):
            return n
    # Also check sidecars in case provenance wasn't recorded (older arches).
    if getattr(arch, "_mla_latent_dim", None) is not None:
        return "swap_attention_to_mla"
    if getattr(arch, "_swa_window", None) is not None:
        return "swap_attention_to_swa"
    return None


class Transformation:
    """Abstract base; subclasses implement precondition() and apply()."""

    name: str = "unnamed"
    # Documentation of which stress axes typically move. Not enforced —
    # the actual stress recomputation is the source of truth.
    expected_stress_signature: Dict[str, str] = {}

    def precondition(self, arch: TArchConfig) -> tuple:
        """Return (ok: bool, reason: str). reason='' when ok=True."""
        return True, ""

    def apply(self, arch: TArchConfig, **params: Any) -> TArchConfig:
        raise NotImplementedError

    def to_quality_arch(self, arch: TArchConfig) -> QArchConfig:
        """Coerce a throughput ArchConfig to a quality ArchConfig.

        Quality ArchConfig is a separate dataclass with extra fields
        (attention_type, model_type, mla_latent_dim, etc.). This method
        is overridden by transformations that need to set those quality-side
        signals (e.g. MLA needs mla_latent_dim, state needs model_type).
        """
        # Coerce common fields; ignore the rest. The downstream
        # _coerce_arch_config in quality_model is lenient.
        common_fields = {
            "d_model", "n_layers", "n_heads", "d_head", "n_kv_heads",
            "ffn_dim", "ffn_type", "vocab_size",
        }
        kwargs = {}
        for f in common_fields:
            if hasattr(arch, f):
                kwargs[f] = getattr(arch, f)
        # MoE → mark model_type
        if arch.moe_config is not None:
            kwargs["model_type"] = "moe"
            kwargs["moe_config"] = copy.deepcopy(arch.moe_config)
            kwargs["n_dense_ffn_layers"] = getattr(arch, "n_dense_ffn_layers", 0)
        # State → mark model_type
        if arch.state_config is not None:
            kwargs.setdefault("model_type", "hybrid" if arch.layer_type_list and
                              any(lt == "attention" for lt in arch.layer_type_list)
                              else "state")
            kwargs["state_config"] = copy.deepcopy(arch.state_config)
            # Provide layer counts the quality model expects.
            if arch.layer_type_list:
                kwargs["state_config"].setdefault(
                    "state_layers",
                    sum(1 for lt in arch.layer_type_list if lt == "state"))
                kwargs["state_config"].setdefault(
                    "attention_layers",
                    sum(1 for lt in arch.layer_type_list if lt == "attention"))
                kwargs["state_config"].setdefault("enabled", True)
        return QArchConfig(**kwargs)
