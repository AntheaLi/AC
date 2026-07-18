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
    from .architecture import format_state_attention_ratio
except ImportError:
    from lattice_engine import estimate_params
    from optimizer import CandidateArch
    from schema import load_config
    from architecture import format_state_attention_ratio


class BaselineUnsupportedError(ValueError):
    """Raised when a baseline uses a reserved future architecture feature."""


#: Reference base-model configs bundled inside the wheel (ac/packaged_configs/).
#: PyPI installs do not carry the repo's top-level configs/ directory, so
#: README quickstart paths like ``configs/mistral_7b.json`` would otherwise
#: fail in a clean environment. Keep this tuple in sync with that directory;
#: tests/test_packaged_configs_sync.py guards byte-identity against configs/.
PACKAGED_REFERENCE_CONFIGS = (
    "mistral_7b.json",
    "gpt_oss_120b.json",
    "mai_thinking_1.json",
)


def _resolve_baseline_path(path: str) -> str:
    """Resolve ``path``, falling back to a wheel-bundled reference config.

    The fallback fires only when the requested path does not exist on disk
    AND its basename exactly names a bundled reference config — arbitrary
    missing paths still raise the usual not-found error. A stderr note is
    printed when the fallback fires so the substitution is never silent.
    """
    if os.path.exists(path):
        return path
    base = os.path.basename(path)
    if base not in PACKAGED_REFERENCE_CONFIGS:
        return path
    try:
        from importlib.resources import files

        candidate = files("ac").joinpath(f"packaged_configs/{base}")
        if candidate.is_file():
            resolved = str(candidate)
            print(
                f"[arch-compiler] {path!r} not found locally; using the "
                f"reference config bundled with the installed package.",
                file=sys.stderr,
            )
            return resolved
    except Exception:
        pass
    return path


@dataclass
class BaselineModel:
    """Supported baseline model extracted from compiler JSON."""

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

    Accepts dense, MoE, compressed-attention, local/global, and state/hybrid
    configs emitted by AC. A representative attention/FFN band supplies shared
    geometry while the complete layer layout is retained on CandidateArch.
    """

    path = _resolve_baseline_path(path)
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

    residual_precisions = {
        str(band.get("residual_dtype", "bf16") or "bf16")
        for band in layer_configs
    }
    if len(residual_precisions) != 1:
        raise BaselineUnsupportedError(
            "Multiple residual/activation precisions across layer bands are "
            "not representable by CandidateArch: "
            f"{sorted(residual_precisions)}."
        )
    activation_precision = next(iter(residual_precisions))

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

    transformer_bands = [
        b for b in layer_configs if b.get("type") == "transformer_block"]
    state_bands = [
        b for b in layer_configs if b.get("type") == "state_block"]
    unsupported_bands = [
        b for b in layer_configs
        if b.get("type") not in ("transformer_block", "state_block")]
    if unsupported_bands:
        raise BaselineUnsupportedError(
            f"Unsupported layer type {unsupported_bands[0].get('type')!r}; "
            "supported: transformer_block, state_block."
        )
    if not transformer_bands and not state_bands:
        raise BaselineUnsupportedError(
            "Baseline has no transformer_block or state_block layers.")

    local_bands = [b for b in transformer_bands if _band_is_local(b)]
    global_bands = [b for b in transformer_bands if not _band_is_local(b)]

    # CandidateArch has one shared projection/precision/normalization shape.
    # Local/global bands may differ in attention pattern and window only;
    # arbitrary per-band geometry would otherwise be silently collapsed to
    # whichever band happened to be dominant.
    def _attention_projection_signature(band: Dict[str, Any]) -> tuple:
        attention = band.get("attention") or {}
        precision = attention.get("precision") or {}
        return (
            int(attention.get("n_heads", 0) or 0),
            int(attention.get("n_kv_heads", 0) or 0),
            int(attention.get("d_head", 0) or 0),
            int(attention.get("kv_cache_bits", 16) or 16),
            json.dumps(precision, sort_keys=True),
        )

    projection_signatures = {
        _attention_projection_signature(band) for band in transformer_bands
    }
    if len(projection_signatures) > 1:
        raise BaselineUnsupportedError(
            "Transformer bands use incompatible attention projection or "
            "precision shapes; CandidateArch requires one shared shape.")

    global_attention_types = {
        str((band.get("attention") or {}).get("type", "full"))
        for band in global_bands
    }
    if len(global_attention_types) > 1:
        raise BaselineUnsupportedError(
            "Global attention bands use multiple attention families; "
            "CandidateArch can represent only one global family.")

    normalizations = {
        json.dumps(band.get("normalization") or {}, sort_keys=True)
        for band in layer_configs
    }
    if len(normalizations) > 1:
        raise BaselineUnsupportedError(
            "Layer bands use incompatible normalization configs; "
            "CandidateArch requires one shared normalization.")

    dense_ffn_signatures = {
        json.dumps(band.get("ffn") or {}, sort_keys=True)
        for band in layer_configs
        if (band.get("ffn") or {}).get("type") != "moe"
    }
    moe_ffn_signatures = {
        json.dumps(band.get("ffn") or {}, sort_keys=True)
        for band in layer_configs
        if (band.get("ffn") or {}).get("type") == "moe"
    }
    if len(dense_ffn_signatures) > 1 or len(moe_ffn_signatures) > 1:
        raise BaselineUnsupportedError(
            "Layer bands use incompatible FFN shapes within the dense or "
            "MoE portions; only a uniform FFN with an optional first-K "
            "dense prefix is representable.")
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
        raise BaselineUnsupportedError(
            "Multiple SWA window sizes across bands are not representable "
            "by CandidateArch.")

    # Pick the representative geometry from a global attention band when one
    # exists, then a local attention band, then a state band for pure-state
    # configs. Prefer an MoE band when the architecture contains MoE so the
    # complete expert topology survives a first-K-dense prefix.
    _selectable = global_bands or local_bands or state_bands
    _moe_selectable = [
        b for b in _selectable if (b.get("ffn") or {}).get("type") == "moe"]
    if _moe_selectable:
        _selectable = _moe_selectable
    if len(layer_configs) == 1:
        lc = layer_configs[0]
        expected_layers = list(range(n_layers))
        if lc.get("layer_idx") != expected_layers:
            raise BaselineUnsupportedError(
                "Single layer_config must cover all layers."
            )
    else:
        lc = max(_selectable, key=lambda c: len(c.get("layer_idx") or []))
        if state_bands:
            warnings.append(
                f"State/hybrid baseline: {sum(len(b.get('layer_idx') or []) for b in state_bands)} "
                f"of {n_layers} layers use state blocks."
            )
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

    representative_state = (
        max(state_bands, key=lambda b: len(b.get("layer_idx") or [])).get("state")
        or {}
        if state_bands else {}
    )
    if transformer_bands:
        attention = lc.get("attention") or {}
    else:
        # CandidateArch requires attention geometry even when n_attention=0;
        # use the state projection shape as inert bookkeeping.
        attention = {
            "type": "full",
            "n_heads": int(representative_state.get("n_heads", 1) or 1),
            "n_kv_heads": int(representative_state.get("n_heads", 1) or 1),
            "d_head": int(representative_state.get("d_head", 64) or 64),
            "kv_cache_bits": 16,
            "precision": {
                "qk": "bf16", "v": "bf16", "output": "bf16"},
        }
    moe_bands = [
        b for b in layer_configs if (b.get("ffn") or {}).get("type") == "moe"]
    ffn_lc = (
        max(moe_bands, key=lambda b: len(b.get("layer_idx") or []))
        if moe_bands else lc
    )
    ffn = ffn_lc.get("ffn") or {}

    attn_type = attention.get("type", "full")
    if attn_type == "swa":
        # Wave 18g: whole-model SWA baseline — GQA projection with a window.
        attn_type = "full"
    if attn_type not in ("full", "mla", "nsa", "csa", "indexshare", "msa"):
        raise BaselineUnsupportedError(
            f"Attention type {attn_type!r} not yet supported for baseline "
            f"loading (supported: full, mla, swa, nsa, csa, indexshare, msa)."
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

    layer_type_list = ["attention"] * n_layers
    for band in state_bands:
        for idx in band.get("layer_idx") or []:
            layer_type_list[int(idx)] = "state"
    n_state_layers = sum(1 for value in layer_type_list if value == "state")
    n_attention_layers = n_layers - n_state_layers
    state_config = None
    if state_bands:
        state_specs = [b.get("state") or {} for b in state_bands]
        state_keys = (
            "type", "d_state", "state_expansion", "n_heads", "d_head",
            "state_precision", "precision",
        )
        first_spec = state_specs[0]
        if any(
            any(spec.get(key) != first_spec.get(key) for key in state_keys)
            for spec in state_specs[1:]
        ):
            raise BaselineUnsupportedError(
                "Multiple incompatible state-block shapes are not yet "
                "representable by CandidateArch."
            )
        state_config = {
            "state_type": str(first_spec.get("type", "mamba2")),
            "d_state": int(first_spec.get("d_state", 128) or 128),
            "state_expansion": int(first_spec.get("state_expansion", 2) or 2),
            "n_heads": int(first_spec.get("n_heads", n_heads) or n_heads),
            "d_head": int(first_spec.get("d_head", d_head) or d_head),
            "state_precision": str(
                first_spec.get("state_precision")
                or first_spec.get("precision")
                or state_bands[0].get("residual_dtype", "bf16")
                or "bf16"),
        }
    hybrid_ratio = (
        format_state_attention_ratio(n_state_layers, n_attention_layers)
        if state_config is not None else ""
    )
    predicted = (config.get("metadata", {}).get("predicted", {}) or {})
    placement_strategy = (
        str(arch.get("placement_strategy")
            or predicted.get("placement_strategy")
            or "baseline")
        if state_config is not None else "none"
    )

    # FFN dim is the dense reference width used by dense-prefix and MTP
    # blocks. New configs preserve it at architecture.ffn_dim; older mixed
    # configs can recover it from a dense band. expert_dim is only a final
    # fallback for old all-MoE configs where the value was not serialized.
    if ffn_type == "moe":
        dense_ffn_dims = [
            int((band.get("ffn") or {}).get("ffn_dim", 0) or 0)
            for band in layer_configs
            if (band.get("ffn") or {}).get("type") in ("dense", "swiglu")
        ]
        ffn_dim = int(
            arch.get("ffn_dim", 0)
            or max(dense_ffn_dims or [0])
            or ffn.get("expert_dim", ffn.get("ffn_dim", 0))
        )
        expert_dim = int(ffn.get("expert_dim", ffn_dim))
        moe_block = {
            "type": "moe",
            "n_experts": int(ffn.get("n_experts", 0)),
            "top_k": int(ffn.get("top_k", 1)),
            "expert_dim": expert_dim,
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
    weight_precision = str(
        arch.get("weight_precision")
        or attn_precision.get("v", "bf16")
        or "bf16")

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
    # Wave 35: NSA and compressed/indexer attention baselines round-trip.
    # Greenfield has emitted attention.type="nsa" since v1 (and
    # csa/indexshare/msa since Wave 35); the loader previously rejected
    # its own emitted configs for these families.
    elif attn_type == "nsa":
        mla_kwargs = dict(
            attention_type="nsa",
            nsa_compress_block_size=int(
                attention.get("nsa_compress_block_size", 64)),
            nsa_compress_block_stride=int(
                attention.get("nsa_compress_block_stride", 16)),
            nsa_select_block_size=int(
                attention.get("nsa_select_block_size", 64)),
            nsa_select_top_k=int(attention.get("nsa_select_top_k", 16)),
            nsa_window_size=int(attention.get("nsa_window_size", 512)),
        )
    elif attn_type == "csa":
        mla_kwargs = dict(
            attention_type="csa",
            csa_block_size=int(attention.get("csa_block_size", 64)),
            csa_top_k_blocks=int(attention.get("csa_top_k_blocks", 16)),
            csa_compression_dim=int(attention.get("csa_compression_dim", 64)),
        )
    elif attn_type == "indexshare":
        mla_kwargs = dict(
            attention_type="indexshare",
            indexshare_num_buckets=int(
                attention.get("indexshare_num_buckets", 64)),
            indexshare_top_k_buckets=int(
                attention.get("indexshare_top_k_buckets", 4)),
            indexshare_index_dim=int(
                attention.get("indexshare_index_dim", 64)),
        )
    elif attn_type == "msa":
        mla_kwargs = dict(
            attention_type="msa",
            msa_window_size=int(attention.get("msa_window_size", 512)),
            msa_dilated_top_k=int(attention.get("msa_dilated_top_k", 64)),
            msa_global_top_k=int(attention.get("msa_global_top_k", 16)),
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

    dense_layer_indices = sorted({
        int(idx)
        for band in layer_configs
        if (band.get("ffn") or {}).get("type") != "moe"
        for idx in (band.get("layer_idx") or [])
    }) if moe_block is not None else []
    n_dense_ffn_layers = int(arch.get("n_dense_ffn_layers", 0) or 0)
    if moe_block is not None and n_dense_ffn_layers == 0 and dense_layer_indices:
        prefix = list(range(len(dense_layer_indices)))
        if dense_layer_indices != prefix:
            raise BaselineUnsupportedError(
                "Mixed dense/MoE baseline is not a contiguous first-K-dense prefix."
            )
        n_dense_ffn_layers = len(dense_layer_indices)

    mtp = arch.get("mtp") or {}
    yoco = arch.get("yoco") or {}
    positional = arch.get("positional_encoding") or {}
    rope_scaling = positional.get("scaling") or {}

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
        activation_precision=activation_precision,
        attn_precision=dict(attn_precision),
        kv_cache_bits=int(attention.get("kv_cache_bits", 16)),
        total_params=total_params,
        total_params_b=round(total_params / 1e9, 2),
        moe=moe_block,
        ep_degree=ep_degree,
        n_dense_ffn_layers=n_dense_ffn_layers,
        active_params=active_params,
        active_params_b=round(active_params / 1e9, 2),
        moe_style=("fine" if moe_block else "dense"),
        cp_degree=cp_degree,
        cp_method=cp_method,
        tp_degree=tp_degree,
        pp_degree=pp_degree,
        state_config=state_config,
        layer_type_list=(layer_type_list if state_config is not None else None),
        placement_strategy=placement_strategy,
        n_attention_layers=n_attention_layers if state_config is not None else 0,
        n_state_layers=n_state_layers,
        hybrid_ratio=hybrid_ratio,
        derived_d_state=(
            int(state_config.get("d_state", 0)) if state_config else 0),
        mtp_n_predict_depths=int(mtp.get("n_predict_depths", 0) or 0),
        mtp_depth_n_layers=int(mtp.get("depth_n_layers", 1) or 1),
        mtp_train_loss_weight=float(mtp.get("train_loss_weight", 0.3) or 0.3),
        yoco_n_self_attn_layers=int(
            yoco.get("n_self_attn_layers", 0) or 0),
        yoco_share_pattern=str(
            yoco.get("share_pattern", "single_source") or "single_source"),
        rope_scaling_method=str(rope_scaling.get("method", "none") or "none"),
        rope_scaling_factor=float(rope_scaling.get("factor", 1.0) or 1.0),
        rope_original_max_position=int(
            rope_scaling.get("original_max_position", 8192) or 8192),
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
