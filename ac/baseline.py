"""
Baseline config ingestion for Architecture Compiler v0.5.

This module keeps baseline-aware Pareto modification separate from the
greenfield compiler. v0.5 accepts the compiler JSON schema only; adapters for
HuggingFace configs or live model objects can be layered on later.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from .lattice_engine import estimate_params
    from .optimizer import CandidateArch
    from .schema import load_config
except ImportError:
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

    if not os.path.exists(path):
        raise BaselineUnsupportedError(
            f"baseline config not found at {path!r}. "
            f"Pass --baseline-config with a path to an AC schema JSON "
            f"(see configs/mistral_7b.json for the reference format)."
        )
    try:
        config = load_config(path)
    except json.JSONDecodeError as e:
        raise BaselineUnsupportedError(
            f"baseline config at {path!r} is not valid JSON: {e}"
        ) from e
    arch = config["architecture"]
    layer_configs = arch.get("layer_configs", [])

    if not layer_configs:
        raise BaselineUnsupportedError("layer_configs is empty.")

    n_layers = int(arch["n_layers"])
    warnings: List[str] = []

    # Wave 18g: per-layer attention heterogeneity. Bands whose attention is
    # sliding-window (type == "swa", or a positive window_size) are LOCAL
    # bands; the rest are GLOBAL. A GPT-OSS / Gemma-2-style alternating
    # stack is expressed as two bands. The local layer count feeds
    # n_local_attn_layers; shape fields come from the dominant GLOBAL band
    # (published interleaves share head geometry across bands).
    def _band_is_local(band: Dict[str, Any]) -> bool:
        att = band.get("attention") or {}
        if str(att.get("type", "full")).lower() == "swa":
            return True
        return int(att.get("window_size", 0) or 0) > 0

    local_bands = [b for b in layer_configs if _band_is_local(b)]
    global_bands = [b for b in layer_configs if not _band_is_local(b)]
    n_local_attn_layers = sum(len(b.get("layer_idx") or []) for b in local_bands)
    local_window = 0
    for b in local_bands:
        att = b.get("attention") or {}
        w = int(att.get("window_size", 0) or 0)
        if w <= 0:
            raise BaselineUnsupportedError(
                "SWA band requires a positive attention.window_size.")
        local_window = max(local_window, w)
    if len({int((b.get("attention") or {}).get("window_size", 0) or 0)
            for b in local_bands}) > 1:
        warnings.append(
            "Multiple SWA window sizes across bands; using the largest "
            "(conservative for KV memory).")

    # v1-fix: support multi-entry layer_configs by picking the dominant entry
    # (largest layer_idx coverage). This lets first-K-dense MoE baselines load.
    # Wave 18g: when local bands exist, the dominant entry is chosen among
    # GLOBAL bands.
    _selectable = global_bands if (local_bands and global_bands) else layer_configs
    if len(layer_configs) == 1:
        lc = layer_configs[0]
        expected_layers = list(range(n_layers))
        if lc.get("layer_idx") != expected_layers:
            raise BaselineUnsupportedError(
                "Single layer_config must cover all layers."
            )
    else:
        lc = max(_selectable, key=lambda c: len(c.get("layer_idx") or []))
        if local_bands and global_bands:
            warnings.append(
                f"Local:global interleave baseline: {n_local_attn_layers} of "
                f"{n_layers} layers are sliding-window (window={local_window}); "
                f"shape read from the dominant global band."
            )
        else:
            warnings.append(
                f"Baseline has {len(layer_configs)} layer_configs; using the "
                f"dominant entry ({len(lc.get('layer_idx') or [])} of {n_layers} "
                f"layers). First-K-dense prefix surfaced as metadata only."
            )
    if local_bands and not global_bands:
        # Whole-model SWA: legacy semantics (swa_window set, n_local == 0),
        # and the dominant (local) band supplies the shape.
        n_local_attn_layers = 0

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
    if attn_type == "swa":
        # Wave 18g: whole-model SWA baseline — GQA projection with a window.
        attn_type = "full"
    if attn_type not in ("full", "mla"):
        raise BaselineUnsupportedError(
            f"Attention type {attn_type!r} not yet supported for baseline "
            f"loading (supported: full, mla, swa)."
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

    # For dense, total_params == active_params and estimate_params is correct.
    # For MoE, estimate_params treats ffn_dim (=expert_dim) as a single dense
    # FFN, which under-counts total by a factor of ~n_experts and slightly
    # over-counts active relative to top_k. Compute both explicitly so the
    # downstream report renders the real model total instead of "one expert
    # worth of FFN".
    active_params = estimate_params(
        d_model, n_heads, d_head, ffn_dim, n_layers, n_kv_heads, vocab_size
    )
    if moe_block is not None:
        # Replace the single-expert FFN mass with top_k experts (active) and
        # n_experts experts (total), plus shared expert if present.
        n_experts = int(moe_block.get("n_experts", 1))
        top_k = max(1, int(moe_block.get("top_k", 1)))
        expert_dim = int(moe_block.get("expert_dim", ffn_dim))
        per_expert_ffn = 3 * d_model * expert_dim
        # Strip the one-expert mass that estimate_params already added.
        active_params = active_params - per_expert_ffn * n_layers
        total_params = active_params + n_experts * per_expert_ffn * n_layers
        active_params = active_params + top_k * per_expert_ffn * n_layers
        # Add shared-expert mass (always-on, counts toward both active and total).
        shared = moe_block.get("shared_expert")
        if isinstance(shared, dict):
            shared_ffn_dim = int(shared.get("ffn_dim", 0))
            if shared_ffn_dim > 0:
                shared_mass = 3 * d_model * shared_ffn_dim * n_layers
                active_params += shared_mass
                total_params += shared_mass
    else:
        total_params = active_params

    # Parallelism block: surface ep / cp into the candidate so MoE memory
    # estimation uses the right per-rank expert count instead of replicating
    # all experts on every GPU. This is the single most common reason an MoE
    # baseline reports infeasible memory (which then pushes the quality model
    # into the INFEASIBLE marker → ~1e6 loss).
    parallelism = config.get("parallelism", {}) or {}
    ep_degree = int(parallelism.get("expert_parallel", 1) or 1)
    cp_degree = int(parallelism.get("context_parallel", 1) or 1)
    cp_method = str(parallelism.get("cp_method", "ring"))
    # Bug fix (Jul 2026): TP and PP from the config's parallelism block were
    # silently dropped — the candidate kept CandidateArch's default and the
    # throughput model evaluated every baseline at TP=1/PP=1 regardless of
    # the declared parallelism (wrong TTFT, no TP all-reduce / cross-node
    # comm in TBT, and per-GPU memory unsharded by TP/PP). 0 = "not
    # declared" → evaluate_candidate falls back to DeploymentConstraints.
    tp_degree = int(parallelism.get("tensor_parallel", 0) or 0)
    pp_degree = int(parallelism.get("pipeline_parallel", 0) or 0)

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
        active_params=active_params,
        active_params_b=round(active_params / 1e9, 2),
        moe_style=("fine" if moe_block else "dense"),
        cp_degree=cp_degree,
        cp_method=cp_method,
        tp_degree=tp_degree,
        pp_degree=pp_degree,
        # Wave 18g: local:global interleave (or whole-model SWA when
        # n_local_attn_layers == 0 and local_window > 0).
        swa_window=int(local_window or 0),
        n_local_attn_layers=int(n_local_attn_layers or 0),
        **mla_kwargs,
    )

    # Wave 20 (loop finding): recompute the param counts through the
    # canonical `architecture.parameter_ledger` now that the candidate
    # carries every relevant field (MLA latents, state config, MoE block).
    # The inline estimate above is MLA-blind — it counted full-MHA QKVO for
    # MLA baselines (mai_thinking_1: 55.46B "active" vs the ledger's
    # 35.27B) — and the consistency gate below was validating metadata
    # against that same blind formula, so mutually-wrong numbers passed.
    try:
        try:
            from ac.architecture import parameter_ledger as _ledger_fn
        except ImportError:
            from architecture import parameter_ledger as _ledger_fn
        _led = _ledger_fn(candidate)
        total_params = int(_led.total_params)
        active_params = int(_led.active_params)
        candidate.total_params = total_params
        candidate.active_params = active_params
        candidate.total_params_b = round(total_params / 1e9, 2)
        candidate.active_params_b = round(active_params / 1e9, 2)
    except Exception:
        pass  # keep the inline estimate if the ledger cannot read the arch

    name = (
        config.get("metadata", {}).get("model_name")
        or config.get("metadata", {}).get("name")
        or os.path.splitext(os.path.basename(path))[0]
    )

    # Wave 18h: ledger-vs-metadata consistency gate. A reference config whose
    # declared parameter counts disagree with what its own architecture block
    # computes poisons every delta-eval / modifier run against it (the
    # gpt_oss_120b fixture shipped for weeks declaring 120B while its
    # architecture block computed 655B). Warn loudly at >2% relative error;
    # the shipped-config CI test (tests/test_reference_config_ledger.py)
    # blocks release on the same threshold.
    declared = (config.get("metadata", {}) or {}).get("params", {}) or {}
    for decl_key, computed in (
        ("total_b", total_params / 1e9),
        ("active_b", active_params / 1e9),
    ):
        decl = declared.get(decl_key)
        if decl is None:
            continue
        try:
            decl_f = float(decl)
        except (TypeError, ValueError):
            continue
        if decl_f <= 0:
            continue
        rel_err = abs(computed - decl_f) / decl_f
        if rel_err > 0.02:
            msg = (
                f"LEDGER MISMATCH: metadata.params.{decl_key}={decl_f:g}B but "
                f"the architecture block computes {computed:.2f}B "
                f"({rel_err * 100:.1f}% off). The computed ledger is used for "
                "all predictions; fix the config metadata or the architecture."
            )
            warnings.append(msg)
            print(f"WARNING: [{name}] {msg}", file=sys.stderr)

    return BaselineModel(
        path=path,
        name=name,
        config=config,
        candidate=candidate,
        warnings=warnings,
    )
