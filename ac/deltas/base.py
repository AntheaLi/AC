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
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, Optional

# Path bootstrap so callers can import this module from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_V1_STRESS = os.path.dirname(_HERE)
_AC_ROOT = os.path.dirname(_V1_STRESS)
for sub in (_V1_STRESS, os.path.join(_AC_ROOT, "v0-throughput"),
            os.path.join(_AC_ROOT, "v0-quality"),
            os.path.join(_AC_ROOT, "v0-tile-lattice")):
    if sub not in sys.path:
        sys.path.insert(0, sub)

from throughput_model import ArchConfig as TArchConfig  # noqa: E402
from quality_model import ArchConfig as QArchConfig  # noqa: E402


def _copy_arch(arch: TArchConfig) -> TArchConfig:
    """Deep-ish copy of an ArchConfig so transformations don't mutate input."""
    fields = asdict(arch)
    # Deep-copy nested dicts/lists so a candidate's moe_config / state_config
    # are independent of the baseline's.
    for k, v in fields.items():
        if isinstance(v, (dict, list)):
            fields[k] = copy.deepcopy(v)
    return TArchConfig(**fields)


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
