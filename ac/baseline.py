"""
Baseline config ingestion for Architecture Compiler v0.5.

This module keeps baseline-aware Pareto modification separate from the
greenfield compiler. v0.5 accepts the compiler JSON schema only; adapters for
HuggingFace configs or live model objects can be layered on later.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from lattice_engine import estimate_params
from optimizer import CandidateArch
from schema import load_config


class BaselineUnsupportedError(ValueError):
    """Raised when a baseline uses a reserved future architecture feature."""


@dataclass
class BaselineModel:
    """Supported dense baseline model extracted from compiler JSON."""

    path: str
    name: str
    config: Dict[str, Any]
    candidate: CandidateArch
    warnings: List[str] = field(default_factory=list)

    @property
    def total_params_b(self) -> float:
        return self.candidate.total_params / 1e9


def _normalize_shared_expert(ffn: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Accept canonical dicts and legacy boolean shared-expert flags."""
    shared = ffn.get("shared_expert")
    if isinstance(shared, dict):
        return {
            "ffn_dim": int(shared["ffn_dim"]),
            "precision": shared.get("precision", ffn.get("precision", "bf16")),
        }
    if shared is True:
        shared_dim = ffn.get("shared_dim", ffn.get("expert_dim", ffn.get("ffn_dim")))
        if shared_dim is None:
            return None
        return {
            "ffn_dim": int(shared_dim),
            "precision": ffn.get("precision", "bf16"),
        }
    return None


def load_baseline_model(path: str) -> BaselineModel:
    """Load a baseline config from compiler JSON.

    v0.5: dense + GQA full attention + SwiGLU only.
    v1-fix: also accepts MoE (`ffn.type == 'moe'`) and MLA (`attention.type ==
    'mla'`) configs. The first uniform-range `transformer_block` layer_config
    is treated as canonical (multi-entry first-K-dense + MoE configs use the
    last/largest entry as canonical and surface the dense prefix as a warning).
    """

    config = load_config(path)
    arch = config["architecture"]
    layer_configs = arch.get("layer_configs", [])

    if not layer_configs:
        raise BaselineUnsupportedError("layer_configs is empty.")

    n_layers = int(arch["n_layers"])
    warnings: List[str] = []

    # v1-fix: support multi-entry layer_configs by picking the dominant entry
    # (largest layer_idx coverage). This lets first-K-dense MoE baselines load.
    if len(layer_configs) == 1:
        lc = layer_configs[0]
        expected_layers = list(range(n_layers))
        if lc.get("layer_idx") != expected_layers:
            raise BaselineUnsupportedError(
                "Single layer_config must cover all layers."
            )
    else:
        lc = max(layer_configs, key=lambda c: len(c.get("layer_idx") or []))
        warnings.append(
            f"Baseline has {len(layer_configs)} layer_configs; using the "
            f"dominant entry ({len(lc.get('layer_idx') or [])} of {n_layers} "
            f"layers). First-K-dense prefix surfaced as metadata only."
        )

    if lc.get("state") is not None:
        raise BaselineUnsupportedError(
            "State/hybrid baselines not supported in baseline modifier mode "
            "(use greenfield --allow-state instead)."
        )

    if lc.get("type") != "transformer_block":
        raise BaselineUnsupportedError(
            f"Unsupported layer type {lc.get('type')!r}; only transformer_block."
        )

    attention = lc.get("attention") or {}
    ffn = lc.get("ffn") or {}

    attn_type = attention.get("type", "full")
    if attn_type not in ("full", "mla"):
        raise BaselineUnsupportedError(
            f"Attention type {attn_type!r} not yet supported for baseline "
            f"loading (supported: full, mla)."
        )

    ffn_type = ffn.get("type", "swiglu")
    if ffn_type not in ("swiglu", "moe"):
        raise BaselineUnsupportedError(
            f"FFN type {ffn_type!r} not supported (supported: swiglu, moe)."
        )

    d_model = int(arch["d_model"])
    n_heads = int(attention["n_heads"])
    d_head = int(attention["d_head"])
    n_kv_heads = int(attention.get("n_kv_heads", n_heads))
    vocab_size = int(arch.get("vocab_size", 32000))

    # FFN dim: dense path uses ffn["ffn_dim"]; MoE path uses expert_dim as the
    # active-per-token FFN compute mass surrogate.
    if ffn_type == "moe":
        ffn_dim = int(ffn.get("expert_dim", ffn.get("ffn_dim", 0)))
        moe_block = {
            "type": "moe",
            "n_experts": int(ffn.get("n_experts", 0)),
            "top_k": int(ffn.get("top_k", 1)),
            "expert_dim": ffn_dim,
            "shared_expert": _normalize_shared_expert(ffn),
            "router": ffn.get("router", {}),
            "capacity_factor": float(ffn.get("capacity_factor", 1.25)),
            "precision": ffn.get("precision", "bf16"),
        }
        warnings.append(
            f"MoE baseline: {ffn.get('n_experts')} experts × top-{ffn.get('top_k')}. "
            f"Modifier scoring uses active-FFN compute mass."
        )
    else:
        ffn_dim = int(ffn["ffn_dim"])
        moe_block = None

    if arch.get("tied_embeddings"):
        warnings.append(
            "Baseline has tied_embeddings=true, but parameter estimator assumes untied."
        )

    attn_precision = attention.get("precision") or {
        "qk": "bf16", "v": "bf16", "output": "bf16",
    }
    ffn_precision = ffn.get("precision", "bf16")
    weight_precision = attn_precision.get("v", "bf16")

    # MLA fields (when present): surfaced into the CandidateArch so downstream
    # quality / throughput models pick up the latent-KV bandwidth term.
    mla_kwargs = {}
    if attn_type == "mla":
        mla_kwargs = dict(
            attention_type="mla",
            mla_kv_latent_dim=int(attention.get("kv_latent_dim", 512)),
            mla_q_latent_dim=int(attention.get("q_latent_dim", 1536)),
            mla_rope_head_dim=int(attention.get("rope_head_dim", 64)),
            mla_nope_head_dim=int(attention.get("nope_head_dim", 128)),
        )

    total_params = estimate_params(
        d_model, n_heads, d_head, ffn_dim, n_layers, n_kv_heads, vocab_size
    )

    # Parallelism block: surface ep / cp into the candidate so MoE memory
    # estimation uses the right per-rank expert count instead of replicating
    # all experts on every GPU. This is the single most common reason an MoE
    # baseline reports infeasible memory (which then pushes the quality model
    # into the INFEASIBLE marker → ~1e6 loss).
    parallelism = config.get("parallelism", {}) or {}
    ep_degree = int(parallelism.get("expert_parallel", 1) or 1)
    cp_degree = int(parallelism.get("context_parallel", 1) or 1)
    cp_method = str(parallelism.get("cp_method", "ring"))

    candidate = CandidateArch(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_head=d_head,
        n_kv_heads=n_kv_heads,
        ffn_dim=ffn_dim,
        vocab_size=vocab_size,
        weight_precision=weight_precision,
        ffn_precision=ffn_precision,
        attn_precision=dict(attn_precision),
        kv_cache_bits=int(attention.get("kv_cache_bits", 16)),
        total_params=total_params,
        total_params_b=round(total_params / 1e9, 2),
        moe=moe_block,
        ep_degree=ep_degree,
        active_params=total_params,
        active_params_b=round(total_params / 1e9, 2),
        moe_style=("fine" if moe_block else "dense"),
        cp_degree=cp_degree,
        cp_method=cp_method,
        **mla_kwargs,
    )

    name = (
        config.get("metadata", {}).get("model_name")
        or config.get("metadata", {}).get("name")
        or os.path.splitext(os.path.basename(path))[0]
    )

    return BaselineModel(
        path=path,
        name=name,
        config=config,
        candidate=candidate,
        warnings=warnings,
    )
