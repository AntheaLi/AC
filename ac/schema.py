"""
Architecture Compiler v0 — JSON Config Schema

The schema is the integration contract between the compiler and any consumer
(training framework, inference engine, reference model code).

Versioned from v0. The list-of-configs format in layer_configs is forward-
compatible: v0 emits a single entry spanning all layers; v2 emits entries
with "state_block" type; v6 emits multiple entries with different attention/FFN
settings per layer range.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timezone


# =============================================================================
# Schema version
# =============================================================================

SCHEMA_VERSION = "0.3"
COMPILER_VERSION = "v0.3.0"

# Changelog (oldest -> newest)
SCHEMA_CHANGELOG = [
    ("0.1", "Dense transformer + GQA. layer_configs[i].ffn is dense; moe field reserved (None)."),
    ("0.2", "MoE support. layer_configs[i].ffn is now a dense|moe union. "
            "parallelism.expert_parallel added. layer_configs[i].moe retained for "
            "backward-compat consumers (mirrors ffn when ffn.type == 'moe'); "
            "deprecated and will be removed in 0.3."),
    ("0.3", "State/hybrid support. layer_configs[i].type may be 'state_block'. "
            "state_block layers have a state dict (type='mamba2', d_state, n_heads, d_head) "
            "and ffn (dense or moe) but no attention. build_hybrid_config() emits "
            "heterogeneous layer_configs for hybrid attention/state architectures. "
            "v1-fix Part I: legacy layer_configs[i].moe mirror slot removed; "
            "consumers must read ffn (type-tagged union). "
            "v1-fix Part J: state.type accepts mamba2 + mamba/s4/s5/s6, "
            "sliding_window, delta_net, gated_delta, kda, gla, rwkv7, linear_attention."),
]

# v0.1 and v0.2 configs are accepted by 0.3 readers (forward compatible).
ACCEPTED_SCHEMA_VERSIONS = {"0.1", "0.2", "0.3"}


# =============================================================================
# Sub-schemas
# =============================================================================

@dataclass
class PositionalEncoding:
    type: str = "rope"
    base: int = 500000


@dataclass
class AttentionPrecision:
    qk: str = "bf16"
    v: str = "bf16"
    output: str = "bf16"


@dataclass
class AttentionConfig:
    """Attention block config.

    Supported `type` values:
      "full"  — standard MHA / GQA / MQA (n_heads, n_kv_heads, d_head)
      "mla"   — DeepSeek-V2/V3 Multi-head Latent Attention. KV is stored as
                a single compressed latent (c_kv) per token plus a small
                RoPE'd part (d_rope), so KV cache bytes per token per layer
                drop from `2 * n_kv_heads * d_head * bpe` to
                `(c_kv + d_rope) * bpe`. Q is similarly down-projected
                through `q_latent_dim` and then up-projected to per-head
                queries. See `kv_latent_dim`, `q_latent_dim`,
                `rope_head_dim`, `nope_head_dim`.
      "sliding_window" — reserved (currently routed through state_block)
    """
    type: str = "full"
    n_heads: int = 32
    n_kv_heads: int = 8
    d_head: int = 128
    rope: bool = True
    kv_cache_bits: int = 16
    precision: Optional[Dict[str, str]] = None

    # MLA-specific fields (only meaningful when type == "mla").
    # DeepSeek-V2 defaults: c_kv = 512, c_q = 1536, d_rope = 64, d_nope = 128.
    # When type != "mla" these are ignored.
    kv_latent_dim: Optional[int] = None     # c_kv  — KV latent dimension
    q_latent_dim: Optional[int] = None      # c_q   — Query latent dimension
    rope_head_dim: Optional[int] = None     # d_rope — per-head RoPE'd dimension
    nope_head_dim: Optional[int] = None     # d_nope — per-head non-RoPE dimension

    # v1-fix NSA: Native Sparse Attention (DeepSeek 2025). Three hierarchical
    # branches: compressed (block-summary KV), selected (top-k fine KV),
    # sliding window (local KV). Only meaningful when type == "nsa".
    nsa_compress_block_size: Optional[int] = None   # default 64
    nsa_compress_block_stride: Optional[int] = None # default 16
    nsa_select_block_size: Optional[int] = None     # default 64
    nsa_select_top_k: Optional[int] = None          # default 16 blocks
    nsa_window_size: Optional[int] = None           # default 512

    def __post_init__(self):
        if self.precision is None:
            self.precision = {"qk": "bf16", "v": "bf16", "output": "bf16"}


@dataclass
class FFNConfig:
    """Dense FFN config (v0.1 shape, still valid in 0.2).

    In v0.2, ffn in layer_configs[i] is a tagged union:
      - dense: {"type": "dense" | "swiglu", "ffn_dim": int, "precision": str}
      - moe:   see MoEFFNConfig
    The legacy 'swiglu' tag is treated as dense for backward compat.
    """
    type: str = "swiglu"
    ffn_dim: int = 14336
    precision: str = "bf16"


@dataclass
class SharedExpertConfig:
    """DeepSeek-style always-on expert. Sized like a small dense FFN."""
    ffn_dim: int = 2816
    precision: str = "bf16"


@dataclass
class MoERouterConfig:
    precision: str = "bf16"               # router runs in BF16/FP32 for stability
    load_balance_loss_coef: float = 0.01  # training-time loss coefficient (info only at infer)
    noise_type: Optional[str] = None      # "gumbel" | "jitter" | None


@dataclass
class MoEFFNConfig:
    """MoE FFN config (v0.2+). Lives in layer_configs[i].ffn when type == 'moe'."""
    type: str = "moe"
    n_experts: int = 8
    top_k: int = 2
    expert_dim: int = 14336                       # per-expert FFN width
    shared_expert: Optional[Dict[str, Any]] = None  # SharedExpertConfig or None
    router: Optional[Dict[str, Any]] = None         # MoERouterConfig
    capacity_factor: float = 1.25
    precision: str = "bf16"                       # expert weights precision

    def __post_init__(self):
        if self.router is None:
            self.router = {
                "precision": "bf16",
                "load_balance_loss_coef": 0.01,
                "noise_type": None,
            }


@dataclass
class NormConfig:
    type: str = "rmsnorm"
    eps: float = 1e-5
    precision: str = "bf16"


@dataclass
class LayerConfig:
    """A single layer configuration entry.

    v0: one entry with layer_idx spanning all layers.
    v2+: multiple entries with different types per layer range.
    v6+: multiple entries with different attention/FFN settings.
    """
    layer_idx: List[int] = field(default_factory=list)
    type: str = "transformer_block"
    attention: Optional[Dict[str, Any]] = None
    ffn: Optional[Dict[str, Any]] = None
    normalization: Optional[Dict[str, Any]] = None
    residual_dtype: str = "bf16"
    # v1+ reserved
    moe: Optional[Dict[str, Any]] = None
    # v2+ reserved
    state: Optional[Dict[str, Any]] = None


@dataclass
class Architecture:
    d_model: int = 4096
    n_layers: int = 32
    vocab_size: int = 32000
    tied_embeddings: bool = False
    positional_encoding: Optional[Dict[str, Any]] = None
    layer_configs: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.positional_encoding is None:
            self.positional_encoding = {"type": "rope", "base": 500000}


@dataclass
class Parallelism:
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: int = 1
    expert_parallel: int = 1  # v0.2+: only meaningful if a layer's ffn.type == 'moe'


@dataclass
class PredictedPerformance:
    quality_rank_score: float = 0.0
    training_throughput_tokens_per_sec: float = 0.0
    serving_ttft_ms: float = 0.0
    serving_tbt_ms: float = 0.0
    memory_per_gpu_gb: float = 0.0
    # v4+ reserved
    confidence: Optional[str] = None


@dataclass
class InputConstraints:
    target_params: str = ""
    training_tokens: str = ""
    context_length: int = 8192
    serving_tbt_ms: Optional[float] = None
    serving_ttft_ms: Optional[float] = None
    serving_batch: Optional[int] = None
    hardware: str = ""


@dataclass
class Metadata:
    schema_version: str = SCHEMA_VERSION
    compiler_version: str = COMPILER_VERSION
    generated_at: str = ""
    input_hardware: str = ""
    input_constraints: Optional[Dict[str, Any]] = None
    predicted: Optional[Dict[str, Any]] = None
    search_stats: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if self.input_constraints is None:
            self.input_constraints = {}
        if self.predicted is None:
            self.predicted = {}


@dataclass
class ArchConfig:
    """Top-level architecture config — the full JSON output."""
    metadata: Dict[str, Any] = field(default_factory=dict)
    parallelism: Dict[str, Any] = field(default_factory=dict)
    architecture: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Builder: construct a valid config from optimizer output
# =============================================================================

def build_config(
    # Architecture dimensions
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_head: int,
    n_kv_heads: int,
    ffn_dim: int,
    vocab_size: int = 32000,
    # Precision
    weight_precision: str = "bf16",
    attn_precision: Optional[Dict[str, str]] = None,
    ffn_precision: str = "bf16",
    kv_cache_bits: int = 16,
    # Parallelism
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    ep: int = 1,  # v0.2+: expert parallel (ignored when moe=None)
    cp: int = 1,  # v1-fix CP: context parallel (Ring Attention / Ulysses)
    cp_method: str = "ring",  # "ring" | "ulysses"
    # Metadata
    hardware_name: str = "",
    input_constraints: Optional[Dict[str, Any]] = None,
    predicted: Optional[Dict[str, Any]] = None,
    search_stats: Optional[Dict[str, Any]] = None,
    # Positional encoding
    rope_base: int = 500000,
    # v1-fix RoPE scaling: method ∈ {none, pi, ntk, yarn, longrope}.
    # `scaling_factor` is the context extension ratio (e.g. 4 = 32k → 128k).
    # `original_max_position` is the pretrain context length the model was
    # originally trained at (before any RoPE extension).
    rope_scaling_method: str = "none",
    rope_scaling_factor: float = 1.0,
    rope_original_max_position: int = 8192,
    # v0.2+: optional MoE override for the FFN slot. If provided, the dense
    # ffn_dim / ffn_precision args are still emitted in metadata for reference
    # but the layer's ffn block is the MoE dict.
    moe: Optional[Dict[str, Any]] = None,
    # v1-fix Part B: first-K-dense pattern. When moe is set and
    # n_dense_ffn_layers > 0, the first n_dense_ffn_layers layers use a dense
    # SwiGLU FFN (ffn_dim) and the remaining layers use the MoE block. This
    # mirrors DeepSeek-V3 / Qwen3-MoE conventions where the first 1-3 layers
    # are dense for training stability.
    n_dense_ffn_layers: int = 0,
    # v1-fix MLA: optional MLA attention block. When provided (a dict with
    # at minimum kv_latent_dim and rope_head_dim), the emitted attention
    # block has type="mla" and carries the latent dimensions. When None,
    # the legacy "full" / GQA / MQA block is emitted.
    mla: Optional[Dict[str, Any]] = None,
    # v1-fix NSA: optional Native Sparse Attention block (DeepSeek 2025).
    # Shape: {compress_block_size, compress_block_stride, select_block_size,
    # select_top_k, window_size}.
    nsa: Optional[Dict[str, Any]] = None,
    # v1-fix YOCO: cross-layer KV sharing (Microsoft 2024). Distinct from
    # MLA: MLA *compresses* the per-layer KV; YOCO *shares* KV across
    # layers. Shape: {n_self_attn_layers, share_pattern}. The first K
    # layers each have their own KV; the remaining N-K layers share the
    # K-th layer's KV cache via cross-attention.
    yoco: Optional[Dict[str, Any]] = None,
    # v1-fix MTP: optional Multi-Token Prediction config (DeepSeek-V3 §2.2).
    # When provided, emits architecture.mtp block. Training-time only; at
    # inference these heads are dropped (or used for speculative decode).
    # Shape: {n_predict_depths: int, depth_n_layers: int, share_embeddings:
    # bool, train_loss_weight: float}.
    mtp: Optional[Dict[str, Any]] = None,
) -> dict:
    """Build a complete, validated architecture config dict.

    v0.2: pass moe=<dict> to emit an MoE FFN per layer. The shape of the dict
    must match MoEFFNConfig (type='moe', n_experts, top_k, expert_dim,
    shared_expert, router, capacity_factor, precision).
    """

    if attn_precision is None:
        attn_precision = {"qk": "bf16", "v": weight_precision, "output": weight_precision}

    if moe is not None:
        moe_ffn_block = _normalize_moe_ffn(moe, default_precision=ffn_precision)
    else:
        moe_ffn_block = None
    # v1-fix Part I: the legacy `layer_configs[].moe` mirror slot was deprecated
    # in 0.2 and is removed in 0.3. Consumers must read `ffn` (which is a
    # type-tagged union of dense | moe).

    dense_ffn_block = {
        "type": "swiglu",
        "ffn_dim": ffn_dim,
        "precision": ffn_precision,
    }

    # v1-fix NSA: route through NSA when `nsa` kwarg is provided. Takes
    # precedence over `mla` in the unlikely case both are passed.
    if nsa is not None:
        cbs = int(nsa.get("compress_block_size", NSA_DEFAULTS["compress_block_size"]))
        cbst = int(nsa.get("compress_block_stride", NSA_DEFAULTS["compress_block_stride"]))
        sbs = int(nsa.get("select_block_size", NSA_DEFAULTS["select_block_size"]))
        stk = int(nsa.get("select_top_k", NSA_DEFAULTS["select_top_k"]))
        win = int(nsa.get("window_size", NSA_DEFAULTS["window_size"]))
        attention_block = {
            "type": "nsa",
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "d_head": d_head,
            "rope": True,
            "kv_cache_bits": kv_cache_bits,
            "precision": attn_precision,
            "nsa_compress_block_size": cbs,
            "nsa_compress_block_stride": cbst,
            "nsa_select_block_size": sbs,
            "nsa_select_top_k": stk,
            "nsa_window_size": win,
        }
    # v1-fix MLA: route through MLA block when `mla` kwarg is provided.
    # MLA defaults from MLA_DEFAULTS (DeepSeek-V2/V3 numerics) when individual
    # fields are not explicitly set.
    elif mla is not None:
        c_kv  = int(mla.get("kv_latent_dim",   MLA_DEFAULTS["kv_latent_dim"]))
        c_q   = int(mla.get("q_latent_dim",    MLA_DEFAULTS["q_latent_dim"]))
        d_rope = int(mla.get("rope_head_dim",  MLA_DEFAULTS["rope_head_dim"]))
        d_nope = int(mla.get("nope_head_dim",  MLA_DEFAULTS["nope_head_dim"]))
        attention_block = {
            "type": "mla",
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "d_head": d_head,
            "rope": True,
            "kv_cache_bits": kv_cache_bits,
            "precision": attn_precision,
            "kv_latent_dim": c_kv,
            "q_latent_dim": c_q,
            "rope_head_dim": d_rope,
            "nope_head_dim": d_nope,
        }
    else:
        attention_block = {
            "type": "full",
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "d_head": d_head,
            "rope": True,
            "kv_cache_bits": kv_cache_bits,
            "precision": attn_precision,
        }
    norm_block = {
        "type": "rmsnorm",
        "eps": 1e-5,
        "precision": "bf16",
    }

    # v1-fix Part B: first-K-dense pattern. Two layer_configs entries when
    # n_dense_ffn_layers > 0 and moe is set; otherwise the v0.2 single-entry
    # emission is preserved exactly.
    n_dense = max(0, int(n_dense_ffn_layers))
    if moe is not None and 0 < n_dense < n_layers:
        layer_configs = [
            {
                "layer_idx": list(range(n_dense)),
                "type": "transformer_block",
                "attention": attention_block,
                "ffn": dense_ffn_block,           # first-K layers: dense FFN
                "normalization": norm_block,
                "residual_dtype": "bf16",
                "state": None,
            },
            {
                "layer_idx": list(range(n_dense, n_layers)),
                "type": "transformer_block",
                "attention": attention_block,
                "ffn": moe_ffn_block,             # remainder: MoE FFN
                "normalization": norm_block,
                "residual_dtype": "bf16",
                "state": None,
            },
        ]
        # Compatibility note: build_config historically returned a single
        # layer_config. The single-entry path below preserves that for
        # n_dense == 0 (or no moe). Multi-entry path is used only when we
        # actually need it.
        # `layer_config` (singular) is kept around for the dense-path
        # construction below; the multi-entry list is assembled separately.
        layer_config = None
    else:
        ffn_block = moe_ffn_block if moe is not None else dense_ffn_block
        layer_config = {
            "layer_idx": list(range(n_layers)),
            "type": "transformer_block",
            "attention": attention_block,
            "ffn": ffn_block,
            "normalization": norm_block,
            "residual_dtype": "bf16",
            "state": None,
        }
        layer_configs = [layer_config]

    config = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "compiler_version": COMPILER_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "input_hardware": hardware_name,
            "input_constraints": input_constraints or {},
            "predicted": predicted or {},
            "search_stats": search_stats or {},
        },
        "parallelism": {
            "tensor_parallel": tp,
            "pipeline_parallel": pp,
            "data_parallel": dp,
            "expert_parallel": ep,
            # v1-fix CP: Context Parallelism (Ring Attention / DeepSpeed-Ulysses).
            # Splits the sequence axis across GPUs so long-context training is
            # feasible without holding the full N×N attention matrix on one rank.
            "context_parallel": cp,
            "cp_method": cp_method,
        },
        "architecture": {
            "d_model": d_model,
            "n_layers": n_layers,
            "vocab_size": vocab_size,
            "tied_embeddings": False,
            "positional_encoding": {
                "type": "rope",
                "base": rope_base,
                # v1-fix RoPE scaling: emit the scaling method block when set
                "scaling": ({
                    "method": rope_scaling_method,
                    "factor": rope_scaling_factor,
                    "original_max_position": rope_original_max_position,
                } if rope_scaling_method != "none" else None),
            },
            "layer_configs": layer_configs,
            # v1-fix Part B: surface the dense-prefix count at the architecture
            # level too, so consumers don't have to re-derive it from
            # layer_configs.
            "n_dense_ffn_layers": n_dense if (moe is not None and 0 < n_dense < n_layers) else 0,
            # v1-fix MTP: Multi-Token Prediction heads (training-time architectural
            # decision). Defaults from DeepSeek-V3: 1 extra prediction depth,
            # each depth = 1 transformer block, share token embeddings with
            # the main model, loss-weight 0.3 on the extra prediction.
            **({"mtp": MTP_DEFAULTS.copy() if mtp is True else dict(mtp)} if mtp else {}),
            # v1-fix YOCO: emit when set so consumers can render the
            # cross-layer KV sharing pattern.
            **({"yoco": YOCO_DEFAULTS.copy() if yoco is True else dict(yoco)} if yoco else {}),
        },
    }

    errors = validate_config(config)
    if errors:
        raise ValueError(f"Generated config failed validation: {errors}")

    return config


def _normalize_moe_ffn(moe: Dict[str, Any], default_precision: str = "bf16") -> Dict[str, Any]:
    """Fill in MoE defaults so partial dicts validate. Returns a fresh dict."""
    shared = moe.get("shared_expert")
    if shared is True:
        shared_dim = moe.get("shared_dim", moe.get("expert_dim"))
        shared = {
            "ffn_dim": int(shared_dim),
            "precision": moe.get("precision", default_precision),
        }
    elif not isinstance(shared, dict):
        shared = None

    out = {
        "type": "moe",
        "n_experts": int(moe["n_experts"]),
        "top_k": int(moe["top_k"]),
        "expert_dim": int(moe["expert_dim"]),
        "shared_expert": shared,
        "router": moe.get("router") or {
            "precision": "bf16",
            "load_balance_loss_coef": 0.01,
            "noise_type": None,
        },
        "capacity_factor": float(moe.get("capacity_factor", 1.25)),
        "precision": moe.get("precision", default_precision),
    }
    # Normalize shared_expert dict if present
    if out["shared_expert"] is not None:
        se = out["shared_expert"]
        out["shared_expert"] = {
            "ffn_dim": int(se["ffn_dim"]),
            "precision": se.get("precision", out["precision"]),
        }
    return out


# =============================================================================
# Validator
# =============================================================================

def validate_config(config: dict) -> List[str]:
    """Validate an architecture config dict. Returns list of error strings (empty = valid).

    Accepts schema_version in ACCEPTED_SCHEMA_VERSIONS ({"0.1", "0.2"}).
    For 0.1 configs, parallelism.expert_parallel is optional. For 0.2,
    ffn may be either dense or moe, and parallelism.expert_parallel is
    optional but defaults to 1 (any layer with moe FFN must have it).
    """
    errors = []

    # Top-level keys
    for key in ("schema_version", "metadata", "parallelism", "architecture"):
        if key not in config:
            errors.append(f"Missing top-level key: {key}")

    if errors:
        return errors  # can't check deeper if top-level missing

    # Schema version
    sv = config.get("schema_version", "")
    if not sv:
        errors.append("schema_version is empty")
    elif sv not in ACCEPTED_SCHEMA_VERSIONS:
        errors.append(
            f"schema_version {sv!r} not in accepted set {sorted(ACCEPTED_SCHEMA_VERSIONS)}"
        )

    # Metadata
    meta = config.get("metadata", {})
    for mkey in ("compiler_version", "generated_at", "input_hardware"):
        if mkey not in meta:
            errors.append(f"Missing metadata.{mkey}")

    # Parallelism
    par = config.get("parallelism", {})
    required_par = ("tensor_parallel", "pipeline_parallel", "data_parallel")
    for pkey in required_par:
        if pkey not in par:
            errors.append(f"Missing parallelism.{pkey}")
        elif not isinstance(par[pkey], int) or par[pkey] < 1:
            errors.append(f"parallelism.{pkey} must be a positive integer")
    # expert_parallel is optional in 0.1, optional but typed in 0.2
    if "expert_parallel" in par:
        if not isinstance(par["expert_parallel"], int) or par["expert_parallel"] < 1:
            errors.append("parallelism.expert_parallel must be a positive integer")

    # Architecture
    arch = config.get("architecture", {})
    for akey in ("d_model", "n_layers", "vocab_size", "layer_configs"):
        if akey not in arch:
            errors.append(f"Missing architecture.{akey}")

    # v1-fix MTP: validate Multi-Token Prediction block when present.
    errors.extend(_validate_mtp(arch, "architecture"))
    # v1-fix YOCO: validate cross-layer KV-sharing block when present.
    errors.extend(_validate_yoco(arch, "architecture"))

    any_moe_layer = False  # tracked for EP cross-check

    if "layer_configs" in arch:
        lcs = arch["layer_configs"]
        if not isinstance(lcs, list) or len(lcs) == 0:
            errors.append("architecture.layer_configs must be a non-empty list")
        else:
            for i, lc in enumerate(lcs):
                pfx = f"layer_configs[{i}]"
                if "layer_idx" not in lc:
                    errors.append(f"{pfx}: missing layer_idx")
                if "type" not in lc:
                    errors.append(f"{pfx}: missing type")

                lc_type = lc.get("type", "")
                if lc_type == "transformer_block":
                    if "attention" not in lc or lc["attention"] is None:
                        errors.append(f"{pfx}: transformer_block missing attention")
                    else:
                        attn = lc["attention"]
                        attn_type = attn.get("type", "full")
                        # All attention variants need the per-head shape
                        for ak in ("n_heads", "n_kv_heads", "d_head"):
                            if ak not in attn:
                                errors.append(f"{pfx}.attention: missing {ak}")

                        if "n_heads" in attn and "n_kv_heads" in attn:
                            if attn["n_heads"] % attn["n_kv_heads"] != 0:
                                errors.append(f"{pfx}.attention: n_heads must be divisible by n_kv_heads")

                        # v1-fix MLA: validate full first-class MLA shape.
                        if attn_type == "mla":
                            errors.extend(_validate_mla_attention(attn, pfx))
                        elif attn_type == "nsa":
                            # v1-fix NSA: Native Sparse Attention (DeepSeek 2025)
                            errors.extend(_validate_nsa_attention(attn, pfx))
                        elif attn_type not in ("full", "gqa", "mqa", "mha"):
                            errors.append(
                                f"{pfx}.attention.type={attn_type!r} not recognized "
                                f"(expected one of full | mha | gqa | mqa | mla | nsa)"
                            )

                    if "ffn" not in lc or lc["ffn"] is None:
                        errors.append(f"{pfx}: transformer_block missing ffn")
                    else:
                        ffn = lc["ffn"]
                        ffn_type = ffn.get("type", "swiglu")
                        if ffn_type in ("dense", "swiglu"):
                            if "ffn_dim" not in ffn:
                                errors.append(f"{pfx}.ffn: dense ffn missing ffn_dim")
                        elif ffn_type == "moe":
                            any_moe_layer = True
                            errors.extend(_validate_moe_ffn(ffn, pfx, par))
                        else:
                            errors.append(
                                f"{pfx}.ffn: unknown type {ffn_type!r} "
                                f"(expected 'dense' | 'swiglu' | 'moe')"
                            )

                elif lc_type == "state_block":
                    # v0.3+: state block validation
                    errors.extend(_validate_state_block(lc, pfx))
                    # Check if FFN is MoE for EP tracking
                    ffn = lc.get("ffn")
                    if ffn and isinstance(ffn, dict) and ffn.get("type") == "moe":
                        any_moe_layer = True

            # Check all layers are covered
            if "n_layers" in arch:
                covered = set()
                for lc in lcs:
                    if "layer_idx" in lc:
                        covered.update(lc["layer_idx"])
                expected = set(range(arch["n_layers"]))
                if covered != expected:
                    missing = expected - covered
                    if missing:
                        errors.append(f"layer_configs does not cover layers: {sorted(missing)[:5]}...")

    # If expert_parallel > 1 there must be at least one MoE layer; otherwise
    # EP is meaningless.
    ep = par.get("expert_parallel", 1)
    if ep > 1 and not any_moe_layer:
        errors.append(
            f"parallelism.expert_parallel={ep} > 1 but no layer has ffn.type == 'moe'"
        )

    # Dimension checks (only for transformer_block layers with attention)
    if "d_model" in arch and "layer_configs" in arch and len(arch["layer_configs"]) > 0:
        for lc in arch["layer_configs"]:
            if lc.get("type") == "transformer_block" and lc.get("attention"):
                attn = lc["attention"]
                # v1-fix MLA: for MLA, the per-head d_head can legitimately
                # differ from d_model / n_heads because the attention math
                # operates on the down-projected latent + RoPE split, with
                # n_heads * (d_rope + d_nope) often != d_model. DeepSeek-V2:
                # d_model=5120, n_heads=128, d_head=40 (but effective per-head
                # operating dim = 64+128 = 192). We skip the equality check
                # for MLA and instead validate the MLA-specific shape inside
                # _validate_mla_attention above.
                if attn.get("type") == "mla":
                    break
                if "n_heads" in attn and "d_head" in attn:
                    expected_d = attn["n_heads"] * attn["d_head"]
                    if expected_d != arch["d_model"]:
                        errors.append(
                            f"d_model ({arch['d_model']}) != n_heads ({attn['n_heads']}) x "
                            f"d_head ({attn['d_head']}) = {expected_d}"
                        )
                break  # Only check first transformer_block

    return errors


def _validate_moe_ffn(ffn: Dict[str, Any], pfx: str, parallelism: Dict[str, Any]) -> List[str]:
    """Validate a single MoE FFN block. Returns list of error strings."""
    errors = []
    required = ("n_experts", "top_k", "expert_dim")
    for k in required:
        if k not in ffn:
            errors.append(f"{pfx}.ffn (moe): missing {k}")

    if errors:
        return errors

    n_experts = ffn["n_experts"]
    top_k = ffn["top_k"]
    expert_dim = ffn["expert_dim"]
    if not (isinstance(n_experts, int) and n_experts >= 1):
        errors.append(f"{pfx}.ffn.n_experts must be a positive integer")
    if not (isinstance(top_k, int) and top_k >= 1):
        errors.append(f"{pfx}.ffn.top_k must be a positive integer")
    if not (isinstance(expert_dim, int) and expert_dim >= 1):
        errors.append(f"{pfx}.ffn.expert_dim must be a positive integer")

    if isinstance(n_experts, int) and isinstance(top_k, int):
        if top_k > n_experts:
            errors.append(f"{pfx}.ffn.top_k ({top_k}) > n_experts ({n_experts})")

    # Expert parallel must divide n_experts
    ep = parallelism.get("expert_parallel", 1)
    if isinstance(n_experts, int) and isinstance(ep, int) and ep > 0:
        if n_experts % ep != 0:
            errors.append(
                f"{pfx}.ffn.n_experts ({n_experts}) not divisible by "
                f"parallelism.expert_parallel ({ep})"
            )

    # Shared expert: if present, must have ffn_dim
    shared = ffn.get("shared_expert")
    if shared is not None:
        if not isinstance(shared, dict) or "ffn_dim" not in shared:
            errors.append(f"{pfx}.ffn.shared_expert: missing ffn_dim")
        else:
            sd = shared["ffn_dim"]
            if not (isinstance(sd, int) and sd >= 1):
                errors.append(f"{pfx}.ffn.shared_expert.ffn_dim must be a positive integer")

    # Router (optional, defaulted in builder; just sanity-check if present)
    router = ffn.get("router")
    if router is not None and not isinstance(router, dict):
        errors.append(f"{pfx}.ffn.router must be a dict if present")

    # Capacity factor sanity
    cf = ffn.get("capacity_factor", 1.0)
    if not (isinstance(cf, (int, float)) and cf > 0):
        errors.append(f"{pfx}.ffn.capacity_factor must be a positive number")

    return errors


# v1-fix Part J: supported state-block types. Family resolution lives in
# v0-quality/quality_model._resolve_hybrid_family — keep these strings aligned.
#
#   mamba2, mamba (==mamba1), s4, s5, s6  → family `mamba_sequential`
#   delta_net, gated_delta, kda           → family `gated_delta_or_kda_linear`
#   gla                                   → family `gated_delta_or_kda_linear`
#   rwkv7, linear_attention, retnet       → family `generic_linear_attention`
#   sliding_window                        → family `recurrent_local_attention`
#                                           (d_state field interpreted as window size)
#
# Only `mamba2` has a measured-empirical reference implementation in ac-base/.
# Other families have research-stub references that round-trip shape but are
# not production-tuned. The compiler labels their quality residual confidence
# as "medium" or lower in configs/quality/quality_v1_defaults.yaml.
_SUPPORTED_STATE_TYPES = {
    # Mamba family (selective SSM)
    "mamba2", "mamba", "mamba1", "s4", "s5", "s6",
    # Sliding-window attention (Mistral / Gemma)
    "sliding_window",
    # Delta-rule linear attention family
    "delta_net", "deltanet", "gated_delta", "gated_deltanet", "kda",
    # Gated linear attention
    "gla", "gated_linear_attention",
    # Generic linear attention
    "rwkv7", "rwkv", "linear_attention", "retnet",
}


# v1-fix MTP: Multi-Token Prediction defaults (DeepSeek-V3 §2.2). The MTP
# module is a 1-layer transformer block that predicts the (k+1)-th future
# token from depth k. Multiple MTP heads → multiple future depths.
MTP_DEFAULTS = {
    "enabled": True,
    "n_predict_depths": 1,        # k ∈ {1, 2, 3} typically; DeepSeek-V3 = 1
    "depth_n_layers": 1,          # each MTP depth is 1 transformer block
    "share_embeddings": True,     # share token + position embeddings with main model
    "share_lm_head": True,        # share the LM head weights
    "train_loss_weight": 0.3,     # weight on the MTP prediction loss
    "inference_mode": "drop",     # "drop" | "speculative_decode"
}


def _validate_mtp(arch: Dict[str, Any], pfx: str) -> List[str]:
    """Validate the optional architecture.mtp block."""
    if "mtp" not in arch or arch["mtp"] is None:
        return []
    errors = []
    mtp = arch["mtp"]
    if not isinstance(mtp, dict):
        return [f"{pfx}.mtp must be a dict"]
    if mtp.get("enabled", True):
        n_depths = mtp.get("n_predict_depths", 1)
        if not isinstance(n_depths, int) or n_depths < 1 or n_depths > 8:
            errors.append(f"{pfx}.mtp.n_predict_depths must be 1..8 (got {n_depths!r})")
        d_layers = mtp.get("depth_n_layers", 1)
        if not isinstance(d_layers, int) or d_layers < 1:
            errors.append(f"{pfx}.mtp.depth_n_layers must be a positive int")
        w = mtp.get("train_loss_weight", 0.3)
        if not isinstance(w, (int, float)) or not (0.0 <= w <= 1.0):
            errors.append(f"{pfx}.mtp.train_loss_weight must be in [0, 1]")
        mode = mtp.get("inference_mode", "drop")
        if mode not in ("drop", "speculative_decode"):
            errors.append(
                f"{pfx}.mtp.inference_mode must be 'drop' or 'speculative_decode'"
            )
    return errors


# v1-fix YOCO: You Only Cache Once defaults (Sun et al., Microsoft 2024).
# The first `n_self_attn_layers` layers each have their own KV; the
# remaining N - n_self_attn_layers layers share the K-th layer's KV cache
# via cross-attention. Cuts KV cache by ~N / K compared to dense.
YOCO_DEFAULTS = {
    "enabled": True,
    "n_self_attn_layers": 1,        # only the FIRST layer keeps its own KV
    "share_pattern": "single_source",  # "single_source" | "block_shared"
}


def _validate_yoco(arch: Dict[str, Any], pfx: str) -> List[str]:
    """Validate the optional architecture.yoco block."""
    if "yoco" not in arch or arch["yoco"] is None:
        return []
    errors = []
    yoco = arch["yoco"]
    if not isinstance(yoco, dict):
        return [f"{pfx}.yoco must be a dict"]
    if yoco.get("enabled", True):
        k = yoco.get("n_self_attn_layers", 1)
        if not isinstance(k, int) or k < 1:
            errors.append(f"{pfx}.yoco.n_self_attn_layers must be ≥ 1 (got {k!r})")
        n_layers = arch.get("n_layers", 0)
        if isinstance(n_layers, int) and isinstance(k, int) and k >= n_layers:
            errors.append(
                f"{pfx}.yoco.n_self_attn_layers={k} must be < n_layers={n_layers} "
                f"(at least one layer must share)"
            )
        pat = yoco.get("share_pattern", "single_source")
        if pat not in ("single_source", "block_shared"):
            errors.append(
                f"{pfx}.yoco.share_pattern must be 'single_source' or 'block_shared'"
            )
    return errors


# v1-fix NSA: Native Sparse Attention defaults from DeepSeek 2025.
# Three branches: compressed (block summaries), selected (top-k blocks),
# sliding window (local). Together they give O(B + K + W) attention KV
# instead of O(L), with quality close to dense attention.
NSA_DEFAULTS = {
    "compress_block_size": 64,
    "compress_block_stride": 16,
    "select_block_size": 64,
    "select_top_k": 16,
    "window_size": 512,
}


def _validate_nsa_attention(attn: Dict[str, Any], pfx: str) -> List[str]:
    """Validate NSA-specific fields. Standard n_heads/n_kv_heads/d_head are
    already checked by the calling validator."""
    errors = []
    required = ("nsa_compress_block_size", "nsa_select_top_k", "nsa_window_size")
    for r in required:
        v = attn.get(r)
        if v is None:
            errors.append(f"{pfx}.attention: NSA missing {r}")
        elif not isinstance(v, int) or v < 1:
            errors.append(f"{pfx}.attention.{r} must be a positive int")
    return errors


# v1-fix MLA: defaults from DeepSeek-V2/V3. Used by the builder when MLA
# fields aren't explicitly set.
MLA_DEFAULTS = {
    "kv_latent_dim": 512,    # c_kv
    "q_latent_dim": 1536,    # c_q
    "rope_head_dim": 64,     # d_rope
    "nope_head_dim": 128,    # d_nope
}


def _validate_mla_attention(attn: Dict[str, Any], pfx: str) -> List[str]:
    """Validate MLA-specific fields. n_heads / n_kv_heads / d_head are
    already required by the calling validator; this only checks the
    latent / split-head shape.

    For MLA the per-head d_head conceptually equals d_rope + d_nope, but the
    quality/throughput models read d_head directly so we accept any positive
    integer (DeepSeek-V2 uses d_head=128 internally even though d_rope=64,
    d_nope=128 ⇒ total per-head dim 192). We *do* require kv_latent_dim and
    rope_head_dim because they drive the KV-cache size.
    """
    errors = []
    for required in ("kv_latent_dim", "rope_head_dim"):
        if required not in attn:
            errors.append(f"{pfx}.attention: MLA missing {required}")
        elif not isinstance(attn[required], int) or attn[required] < 1:
            errors.append(f"{pfx}.attention.{required} must be a positive integer")
    # q_latent_dim / nope_head_dim are recommended but optional (default in builder)
    for opt in ("q_latent_dim", "nope_head_dim"):
        if opt in attn and (not isinstance(attn[opt], int) or attn[opt] < 1):
            errors.append(f"{pfx}.attention.{opt} must be a positive integer if present")
    # Sanity: latent dim should be much smaller than n_heads × d_head.
    if "kv_latent_dim" in attn and "n_heads" in attn and "d_head" in attn:
        per_head_kv_uncompressed = 2 * int(attn["n_heads"]) * int(attn["d_head"])
        c_kv = int(attn["kv_latent_dim"])
        if c_kv >= per_head_kv_uncompressed:
            errors.append(
                f"{pfx}.attention.kv_latent_dim={c_kv} >= uncompressed KV size "
                f"{per_head_kv_uncompressed}; MLA needs a true compression."
            )
    return errors


def _validate_state_block(lc: Dict[str, Any], pfx: str) -> List[str]:
    """Validate a single state_block layer config. Returns list of error strings."""
    errors = []

    # state dict is required
    if "state" not in lc or lc["state"] is None:
        errors.append(f"{pfx}: state_block missing state dict")
        return errors

    state = lc["state"]
    if not isinstance(state, dict):
        errors.append(f"{pfx}.state: must be a dict")
        return errors

    # state.type is required
    if "type" not in state:
        errors.append(f"{pfx}.state: missing type")
    else:
        state_type = state["type"]
        # v1-fix Part J: extend the supported state families. Only mamba2 has a
        # measured-empirical reference model; the others have research-stub
        # references in ac-base/ that round-trip shape but are not production-
        # tuned. The schema validator only enforces field presence; the
        # quality model (v0-quality/quality_model._resolve_hybrid_family)
        # routes each type to a calibrated family band.
        if state_type not in _SUPPORTED_STATE_TYPES:
            errors.append(
                f"{pfx}.state.type={state_type!r} is not recognized; "
                f"expected one of {sorted(_SUPPORTED_STATE_TYPES)}"
            )

    # Required state fields. Sliding-window attention (SWA) is a special case:
    # it is fundamentally attention with a bounded window, so d_state is
    # interpreted as the window size and n_heads/d_head must match the
    # surrounding attention configuration.
    state_type = state.get("type")
    is_swa = state_type == "sliding_window"

    for sk in ("d_state", "n_heads", "d_head"):
        if sk not in state:
            errors.append(f"{pfx}.state: missing {sk}")
        elif not isinstance(state[sk], int) or state[sk] < 1:
            errors.append(f"{pfx}.state.{sk} must be a positive integer")

    # Sliding-window-specific check: window size must be sensible.
    if is_swa and state.get("d_state", 0) > 0:
        if state["d_state"] < 64 or state["d_state"] > 32768:
            errors.append(
                f"{pfx}.state.d_state={state['d_state']} for sliding_window is outside "
                f"the supported window range [64, 32768]"
            )

    # FFN is still required for state blocks (state replaces attention, not FFN)
    if "ffn" not in lc or lc["ffn"] is None:
        errors.append(f"{pfx}: state_block missing ffn (state replaces attention, not FFN)")
    else:
        ffn = lc["ffn"]
        ffn_type = ffn.get("type", "swiglu")
        if ffn_type in ("dense", "swiglu"):
            if "ffn_dim" not in ffn:
                errors.append(f"{pfx}.ffn: dense ffn missing ffn_dim")
        elif ffn_type == "moe":
            # Delegate to existing MoE validation; pass empty parallelism for
            # the subset of checks that don't need EP.
            errors.extend(_validate_moe_ffn(ffn, pfx, {}))
        else:
            errors.append(
                f"{pfx}.ffn: unknown type {ffn_type!r} "
                f"(expected 'dense' | 'swiglu' | 'moe')"
            )

    # Attention should be None/absent for state blocks
    if lc.get("attention") is not None:
        errors.append(
            f"{pfx}: state_block should not have attention "
            f"(state replaces attention)"
        )

    return errors


def build_hybrid_config(
    # Architecture dimensions
    d_model: int,
    n_layers: int,
    vocab_size: int = 32000,
    # Attention layer params
    attention_layer_indices: Optional[List[int]] = None,
    n_heads: int = 32,
    d_head: int = 128,
    n_kv_heads: int = 8,
    kv_cache_bits: int = 16,
    attn_precision: Optional[Dict[str, str]] = None,
    # State layer params
    state_layer_indices: Optional[List[int]] = None,
    state_type: str = "mamba2",
    state_d_state: int = 128,
    state_n_heads: int = 32,
    state_d_head: int = 64,
    # FFN params (shared by both attention and state layers)
    ffn_dim: int = 14336,
    ffn_type: str = "swiglu",
    ffn_precision: str = "bf16",
    weight_precision: str = "bf16",
    # MoE override (if not None, used as FFN for all layers)
    moe: Optional[Dict[str, Any]] = None,
    # Parallelism
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    ep: int = 1,
    cp: int = 1,                      # v1-fix CP
    cp_method: str = "ring",
    # v1-fix RoPE scaling
    rope_base: int = 500000,
    rope_scaling_method: str = "none",
    rope_scaling_factor: float = 1.0,
    rope_original_max_position: int = 8192,
    # Metadata
    hardware_name: str = "",
    input_constraints: Optional[Dict[str, Any]] = None,
    predicted: Optional[Dict[str, Any]] = None,
    search_stats: Optional[Dict[str, Any]] = None,
) -> dict:
    """Build a hybrid attention/state architecture config.

    Creates a multi-entry layer_configs list with attention layers
    (transformer_block) and state layers (state_block).

    Args:
        attention_layer_indices: list of layer indices that use attention
        state_layer_indices: list of layer indices that use state (Mamba-2)
        Other args match build_config() semantics.

    Returns:
        Validated config dict.
    """
    if attention_layer_indices is None:
        attention_layer_indices = []
    if state_layer_indices is None:
        state_layer_indices = []

    # Validate coverage
    overlap = set(attention_layer_indices) & set(state_layer_indices)
    if overlap:
        raise ValueError(f"Layer index coverage error: overlapping layers: {sorted(overlap)}")
    all_indices = set(attention_layer_indices) | set(state_layer_indices)
    expected = set(range(n_layers))
    if all_indices != expected:
        missing = expected - all_indices
        extra = all_indices - expected
        issues = []
        if missing:
            issues.append(f"uncovered layers: {sorted(missing)}")
        if extra:
            issues.append(f"extra layers: {sorted(extra)}")
        raise ValueError(f"Layer index coverage error: {'; '.join(issues)}")

    if attn_precision is None:
        attn_precision = {"qk": "bf16", "v": weight_precision, "output": weight_precision}

    # Build FFN block (shared between attention and state layers)
    if moe is not None:
        ffn_block = _normalize_moe_ffn(moe, default_precision=ffn_precision)
    else:
        ffn_block = {
            "type": ffn_type,
            "ffn_dim": ffn_dim,
            "precision": ffn_precision,
        }

    layer_configs = []

    # Attention layer config
    if attention_layer_indices:
        attn_lc = {
            "layer_idx": sorted(attention_layer_indices),
            "type": "transformer_block",
            "attention": {
                "type": "full",
                "n_heads": n_heads,
                "n_kv_heads": n_kv_heads,
                "d_head": d_head,
                "rope": True,
                "kv_cache_bits": kv_cache_bits,
                "precision": attn_precision,
            },
            "ffn": dict(ffn_block),
            "normalization": {
                "type": "rmsnorm",
                "eps": 1e-5,
                "precision": "bf16",
            },
            "residual_dtype": "bf16",
            "state": None,
        }
        layer_configs.append(attn_lc)

    # State layer config
    if state_layer_indices:
        state_lc = {
            "layer_idx": sorted(state_layer_indices),
            "type": "state_block",
            "attention": None,
            "ffn": dict(ffn_block),
            "normalization": {
                "type": "rmsnorm",
                "eps": 1e-5,
                "precision": "bf16",
            },
            "residual_dtype": "bf16",
            "state": {
                "type": state_type,
                "d_state": state_d_state,
                "n_heads": state_n_heads,
                "d_head": state_d_head,
            },
        }
        layer_configs.append(state_lc)

    config = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "compiler_version": COMPILER_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "input_hardware": hardware_name,
            "input_constraints": input_constraints or {},
            "predicted": predicted or {},
            "search_stats": search_stats or {},
        },
        "parallelism": {
            "tensor_parallel": tp,
            "pipeline_parallel": pp,
            "data_parallel": dp,
            "expert_parallel": ep,
            # v1-fix CP: Context Parallelism (Ring Attention / DeepSpeed-Ulysses).
            # Splits the sequence axis across GPUs so long-context training is
            # feasible without holding the full N×N attention matrix on one rank.
            "context_parallel": cp,
            "cp_method": cp_method,
        },
        "architecture": {
            "d_model": d_model,
            "n_layers": n_layers,
            "vocab_size": vocab_size,
            "tied_embeddings": False,
            "positional_encoding": {
                "type": "rope",
                "base": rope_base,
                # v1-fix RoPE scaling: emit the scaling method block when set
                "scaling": ({
                    "method": rope_scaling_method,
                    "factor": rope_scaling_factor,
                    "original_max_position": rope_original_max_position,
                } if rope_scaling_method != "none" else None),
            },
            "layer_configs": layer_configs,
        },
    }

    errors = validate_config(config)
    if errors:
        raise ValueError(f"Generated hybrid config failed validation: {errors}")

    return config


def load_config(path: str) -> dict:
    """Load and validate a config from a JSON file."""
    with open(path) as f:
        config = json.load(f)
    errors = validate_config(config)
    if errors:
        raise ValueError(f"Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    return config


def save_config(config: dict, path: str):
    """Validate and save a config to a JSON file."""
    errors = validate_config(config)
    if errors:
        raise ValueError(f"Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
