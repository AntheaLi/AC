"""v1-stress transformation library — exports the 10 named transformations."""

from .base import Transformation
from .swap_attention_to_gqa import SwapAttentionToGQA
from .swap_attention_to_mla import SwapAttentionToMLA
from .swap_attention_to_swa import SwapAttentionToSWA
from .interleave_local_attention import InterleaveLocalAttention
from .add_state_layers import AddStateLayers
from .densify_first_k import DensifyFirstK
from .change_moe_topology import ChangeMoeTopology
from .scale_d_model import ScaleDModel
from .scale_n_layers import ScaleNLayers
from .change_precision_per_component import ChangePrecisionPerComponent
from .change_parallelism import ChangeParallelism


REGISTRY = {
    cls.name: cls for cls in (
        InterleaveLocalAttention,  # Wave 18g
        SwapAttentionToGQA,
        SwapAttentionToMLA,
        SwapAttentionToSWA,
        AddStateLayers,
        DensifyFirstK,
        ChangeMoeTopology,
        ScaleDModel,
        ScaleNLayers,
        ChangePrecisionPerComponent,
        ChangeParallelism,
    )
}


def get(name: str) -> Transformation:
    """Look up a transformation by name."""
    if name not in REGISTRY:
        raise KeyError(f"unknown transformation {name!r} (available: "
                       f"{sorted(REGISTRY.keys())})")
    return REGISTRY[name]()


__all__ = [
    "Transformation",
    "REGISTRY",
    "get",
    "SwapAttentionToGQA",
    "SwapAttentionToMLA",
    "SwapAttentionToSWA",
    "AddStateLayers",
    "DensifyFirstK",
    "ChangeMoeTopology",
    "ScaleDModel",
    "ScaleNLayers",
    "ChangePrecisionPerComponent",
    "ChangeParallelism",
]
