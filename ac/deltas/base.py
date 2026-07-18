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
    # Resolution and precision metadata must survive later transformations
    # in a composed sequence just like topology sidecars do.
    "_target_activation_precision", "_state_layer_summary", "_delta_notes",
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
        return (
            "interleave_local_attention"
            if int(getattr(arch, "_n_local_attn_layers", 0) or 0) > 0
            else "swap_attention_to_swa"
        )
    # Baseline-loaded candidates carry canonical fields rather than delta
    # sidecars. Treat them identically so an entry-point round-trip cannot
    # bypass the overwrite guard.
    if str(getattr(arch, "attention_type", "full") or "full") == "mla":
        return "swap_attention_to_mla"
    if int(getattr(arch, "local_window", 0) or 0) > 0:
        return (
            "interleave_local_attention"
            if int(getattr(arch, "n_local_attn_layers", 0) or 0) > 0
            else "swap_attention_to_swa"
        )
    return None


def _has_attention_layers(arch: TArchConfig) -> bool:
    """Whether an architecture contains at least one attention layer."""
    layer_types = list(getattr(arch, "layer_type_list", None) or [])
    return not layer_types or any(
        kind in ("attention", "local_attention") for kind in layer_types)


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
        kwargs["weight_precision"] = str(
            getattr(arch, "weight_precision", "bf16") or "bf16")
        kwargs["activation_precision"] = str(
            getattr(arch, "activation_precision", "bf16") or "bf16")
        kwargs["kv_precision"] = str(
            getattr(arch, "kv_precision", "bf16") or "bf16")
        component_precisions = {}
        ffn_precision = str(getattr(arch, "precision", "bf16") or "bf16")
        if ffn_precision != kwargs["weight_precision"]:
            for component in ("ffn_up", "ffn_down", "ffn_gate"):
                component_precisions[component] = ffn_precision
        attn_precision = getattr(arch, "attn_precision", None) or {}
        v_precision = attn_precision.get("v", "bf16")
        if v_precision != "bf16":
            component_precisions["qkv_proj"] = v_precision
        qk_precision = next((
            value for value in (
                attn_precision.get("qk"), attn_precision.get("q"),
                attn_precision.get("k"))
            if value and value != "bf16"), "bf16")
        if qk_precision != "bf16":
            component_precisions["qkv_proj"] = qk_precision
        output_precision = attn_precision.get(
            "output", attn_precision.get("o", "bf16"))
        if output_precision != "bf16":
            component_precisions["output_proj"] = output_precision
        if component_precisions:
            kwargs["component_precisions"] = component_precisions
        attention_type = str(
            getattr(arch, "attention_type", "full") or "full")
        local_window = int(
            getattr(arch, "local_window", 0)
            or getattr(arch, "_swa_window", 0)
            or 0)
        n_local = int(
            getattr(arch, "n_local_attn_layers", 0)
            or getattr(arch, "_n_local_attn_layers", 0)
            or 0)
        if local_window > 0 and n_local == 0:
            kwargs["attention_type"] = "swa"
        else:
            kwargs["attention_type"] = (
                "gqa" if attention_type in {"full", "mha", "gqa", "mqa"}
                else attention_type)
        if local_window > 0:
            kwargs["local_window"] = local_window
            if n_local > 0:
                layer_types = list(getattr(arch, "layer_type_list", None) or [])
                n_attention = sum(
                    1 for kind in layer_types if kind != "state")
                if n_attention <= 0:
                    n_attention = int(getattr(arch, "n_layers", 1) or 1)
                kwargs["local_attention_fraction"] = min(
                    1.0, n_local / max(1, n_attention))
        passthrough = {
            "mla_q_latent_dim": "mla_q_latent_dim",
            "mla_rope_head_dim": "mla_rope_head_dim",
            "mla_nope_head_dim": "mla_nope_head_dim",
            "nsa_compress_block_size": "nsa_compress_block_size",
            "nsa_compress_block_stride": "nsa_compress_block_stride",
            "nsa_select_block_size": "nsa_select_block_size",
            "nsa_select_top_k": "nsa_select_top_k",
            "nsa_window_size": "nsa_window_size",
            "csa_block_size": "csa_block_size",
            "csa_top_k_blocks": "csa_top_k_blocks",
            "csa_compression_dim": "csa_compression_dim",
            "indexshare_num_buckets": "indexshare_num_buckets",
            "indexshare_top_k_buckets": "indexshare_top_k_buckets",
            "indexshare_index_dim": "indexshare_index_dim",
            "msa_window_size": "msa_window_size",
            "msa_dilated_top_k": "msa_dilated_top_k",
            "msa_global_top_k": "msa_global_top_k",
            "yoco_n_self_attn_layers": "yoco_n_self_attn_layers",
            "mtp_n_predict_depths": "mtp_n_predict_depths",
            "mtp_depth_n_layers": "mtp_depth_n_layers",
            "mtp_train_loss_weight": "mtp_train_loss_weight",
            "rope_scaling_method": "rope_scaling_method",
            "rope_scaling_factor": "rope_scaling_factor",
            "rope_original_max_position": "rope_original_max_position",
            "sparsity_2_4": "sparsity_2_4",
        }
        for source, target in passthrough.items():
            value = getattr(arch, source, None)
            if value is not None:
                kwargs[target] = copy.deepcopy(value)
        mla_latent = (
            getattr(arch, "mla_kv_latent_dim", None)
            or getattr(arch, "_mla_latent_dim", None))
        if mla_latent is not None:
            kwargs["mla_latent_dim"] = int(mla_latent)
        # MoE → mark model_type
        if arch.moe_config is not None:
            kwargs["model_type"] = "moe"
            kwargs["moe_config"] = copy.deepcopy(arch.moe_config)
            kwargs["n_dense_ffn_layers"] = getattr(arch, "n_dense_ffn_layers", 0)
        # State → mark model_type
        if arch.state_config is not None:
            kwargs.setdefault("model_type", "hybrid" if arch.layer_type_list and
                              any(lt in ("attention", "local_attention")
                                  for lt in arch.layer_type_list)
                              else "state")
            kwargs["state_config"] = copy.deepcopy(arch.state_config)
            # Provide layer counts the quality model expects.
            if arch.layer_type_list:
                kwargs["state_config"].setdefault(
                    "state_layers",
                    sum(1 for lt in arch.layer_type_list if lt == "state"))
                kwargs["state_config"].setdefault(
                    "attention_layers",
                    sum(1 for lt in arch.layer_type_list
                        if lt in ("attention", "local_attention")))
                kwargs["state_config"].setdefault("enabled", True)
        return QArchConfig(**kwargs)
