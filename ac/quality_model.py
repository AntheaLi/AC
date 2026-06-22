"""
Quality Model v1 — Q(A, D, H, P) -> QualityResult

Ranking function for architecture quality. The v1 backbone is modular:

    loss_proxy = scaling-law spine over active non-embedding compute proxy
               + coupled architecture residual
               + precision residual
               + MoE residual
               + state/hybrid memory residual
               + risk residual
               + optional data-quality adjustment

This is NOT a perplexity predictor. It ranks architectures within a parameter
band and reports uncertainty. The public API keeps the v0 fields used by the
architecture solver (`predicted_loss`, `penalty_breakdown`, confidence bands),
while exposing v1 term-level results for reports and future adapters.
"""

import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Tuple

# Import lattice for param estimation (sibling module)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from lattice_engine import estimate_params

from penalties import (
    shape_penalty,
    gqa_penalty,
    kv_quant_penalty,
    weight_precision_penalty,
    activation_precision_penalty,
    feasibility_penalty,
    PENALTY_REGISTRY,
    INFEASIBLE,
)


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class ArchConfig:
    """Architecture configuration — input to the quality model."""
    d_model: int
    n_layers: int
    n_heads: int
    d_head: int
    n_kv_heads: int
    ffn_dim: int
    ffn_type: str = "swiglu"
    vocab_size: int = 32000

    # Model-family hooks. v0 solver emits dense transformers; the fields are
    # accepted now so future adapters can pass MoE/state/hybrid metadata without
    # changing the public function signature.
    model_type: str = "dense"               # dense | moe | hybrid | state
    attention_type: str = "gqa"             # mha | gqa | mqa | mla
    local_window: Optional[int] = None
    global_frequency: Optional[int] = None
    mla_latent_dim: Optional[int] = None
    kv_compression_ratio: Optional[float] = None
    # v1-fix MLA: full DeepSeek-V2/V3 MLA shape. Only meaningful when
    # attention_type == "mla". `mla_latent_dim` (c_kv) drives the KV
    # cache size; `mla_q_latent_dim` (c_q) drives the down-projection
    # for the query path. `mla_rope_head_dim` (d_rope) and
    # `mla_nope_head_dim` (d_nope) describe the split-head layout.
    mla_q_latent_dim: Optional[int] = None
    mla_rope_head_dim: Optional[int] = None
    mla_nope_head_dim: Optional[int] = None
    # v1-fix NSA: Native Sparse Attention parameters. Only meaningful when
    # attention_type == "nsa".
    nsa_compress_block_size: Optional[int] = None
    nsa_compress_block_stride: Optional[int] = None
    nsa_select_block_size: Optional[int] = None
    nsa_select_top_k: Optional[int] = None
    nsa_window_size: Optional[int] = None
    # v1-fix YOCO: cross-layer KV sharing (Microsoft 2024). When set, only the
    # first `yoco_n_self_attn_layers` layers keep their own KV; the rest read
    # from the K-th layer's KV cache via cross-attention. Different from MLA
    # (which compresses); YOCO shares.
    yoco_n_self_attn_layers: int = 0  # 0 = YOCO off
    # v1-fix MTP: Multi-Token Prediction. Training-time architectural
    # decision (DeepSeek-V3 §2.2). Modeled as a small loss-proxy bonus
    # proportional to n_predict_depths and the train_loss_weight, capped
    # to avoid over-rewarding.
    mtp_n_predict_depths: int = 0
    mtp_depth_n_layers: int = 1
    mtp_train_loss_weight: float = 0.0
    # v1-fix RoPE scaling: positional-encoding extension method.
    # "none" = native (no extension), "pi" = Position Interpolation,
    # "ntk" = NTK-aware, "yarn" = YaRN, "longrope" = LongRoPE.
    # Affects the attn_long_context multiplier in _architecture_residual.
    rope_scaling_method: str = "none"
    rope_scaling_factor: float = 1.0
    rope_original_max_position: int = 8192
    # v1-fix 2:4 sparsity: per-component structured sparsity flags. NVIDIA
    # H100/B200 tensor cores natively support 2:4 (half the values zero).
    # When sparsity is enabled, that component's matmul runs at ~2× throughput
    # with a modest quality cost (~1-3% PPL delta vs dense in published
    # ablations). Components: "ffn_up", "ffn_down", "ffn_gate", "attn_qkv",
    # "attn_o", "embed". Each can independently be 2:4 sparse.
    sparsity_2_4: Optional[Dict[str, bool]] = None

    # Precision per component (default: all BF16)
    weight_precision: str = "bf16"           # global default
    kv_precision: str = "bf16"               # KV cache precision
    activation_precision: str = "bf16"       # activation precision
    component_precisions: Optional[Dict[str, str]] = None  # per-component overrides

    # Future architecture hooks. v1 reports placeholder residuals for these
    # families; the dense v0 solver still searches uniform attention layers.
    layer_types: Optional[List[str]] = None
    moe_config: Optional[dict] = None
    # Supported keys: enabled, n_experts, top_k, expert_dim, shared_dim,
    # granularity, load_balance_prior, shared_expert_ratio.
    state_config: Optional[dict] = None
    # Supported keys: enabled, state_type, d_state, state_layers,
    # attention_layers, pattern.

    # v1-fix Part B: first-K-dense prefix for MoE. When > 0 and moe_config
    # is set, the first n_dense_ffn_layers use dense FFN (ffn_dim); the
    # remaining n_layers - n_dense_ffn_layers use the MoE block. Mirrors
    # DeepSeek-V3 / Qwen3-MoE convention. Ignored when moe_config is None.
    n_dense_ffn_layers: int = 0

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["attention"] * self.n_layers
        if self.component_precisions is None:
            self.component_precisions = {}

    def get_precision(self, component: str) -> str:
        """Get precision for a specific component, with fallback to global."""
        return self.component_precisions.get(component, self.weight_precision)

    @property
    def n_active_params(self) -> int:
        """Active parameter count (params touched per token).
        Dense: active = total. MoE: active experts + shared components.
        """
        if self._moe_enabled:
            return self._estimate_moe_params(active=True)
        return estimate_params(
            self.d_model, self.n_heads, self.d_head, self.ffn_dim,
            self.n_layers, self.n_kv_heads, self.vocab_size
        )

    @property
    def n_total_params(self) -> int:
        """Total parameter count (includes inactive experts for MoE).
        Dense: total = active.
        """
        if self._moe_enabled:
            return self._estimate_moe_params(active=False)
        return self.n_active_params

    @property
    def embedding_params(self) -> int:
        return 2 * self.vocab_size * self.d_model

    @property
    def n_active_non_embedding_params(self) -> int:
        """Active non-embedding parameter proxy used by the scaling spine."""
        return max(1, self.n_active_params - self.embedding_params)

    @property
    def n_total_non_embedding_params(self) -> int:
        return max(1, self.n_total_params - self.embedding_params)

    @property
    def _moe_enabled(self) -> bool:
        return bool(self.moe_config and self.moe_config.get("enabled", True)) or self.model_type == "moe"

    def _estimate_moe_params(self, active: bool) -> int:
        moe = self.moe_config or {}
        n_experts = max(1, int(moe.get("n_experts", 1)))
        top_k = max(1, int(moe.get("top_k", 1)))
        expert_dim = int(moe.get("expert_dim", self.ffn_dim))
        # v1 MoE canonical shape uses nested shared_expert.ffn_dim. Legacy
        # flat shared_dim is honored for backward compat.
        shared_block = moe.get("shared_expert")
        if isinstance(shared_block, dict):
            shared_dim = int(shared_block.get("ffn_dim", 0) or 0)
        else:
            shared_dim = int(moe.get("shared_dim", 0) or 0)

        q_params = self.d_model * self.d_head * self.n_heads
        kv_params = 2 * self.d_model * self.d_head * self.n_kv_heads
        o_params = self.d_head * self.n_heads * self.d_model
        shared_attn_per_layer = q_params + kv_params + o_params
        expert_multiplier = min(top_k, n_experts) if active else n_experts
        expert_ffn = 3 * self.d_model * expert_dim * expert_multiplier
        shared_ffn = 3 * self.d_model * shared_dim if shared_dim > 0 else 0
        moe_per_layer = shared_attn_per_layer + expert_ffn + shared_ffn

        # v1-fix Part B: first-K-dense prefix. n_dense layers use the
        # baseline dense FFN (3 * d_model * ffn_dim); the rest use the MoE
        # block computed above. Both active and total counts split the same
        # way (a dense layer activates all of itself in both regimes).
        n_dense = max(0, min(int(self.n_dense_ffn_layers), self.n_layers))
        dense_ffn_params = 3 * self.d_model * self.ffn_dim
        dense_per_layer = shared_attn_per_layer + dense_ffn_params
        n_moe = self.n_layers - n_dense

        per_layer_sum = dense_per_layer * n_dense + moe_per_layer * n_moe
        embed_params = 2 * self.vocab_size * self.d_model
        norm_params = 2 * self.d_model * self.n_layers
        return per_layer_sum + embed_params + norm_params


@dataclass
class TrainingConfig:
    """Training configuration — input to the quality model."""
    training_tokens: int              # D in Chinchilla
    sequence_length: int = 2048
    hardware: str = "h100"            # for hardware-conditional penalties
    kv_quantization_bits: int = 16    # KV cache bits (16=BF16, 8=INT8, 4=INT4)
    kv_per_channel_scaling: bool = True

    # v0: accepted but unused — flag in docs that Chinchilla coefficients
    # implicitly assume a specific recipe
    data_mixture: Optional[str] = None       # v0: unused
    optimizer_recipe: Optional[str] = None   # v0: unused
    unique_tokens: Optional[int] = None
    compute_budget_flops: Optional[float] = None
    batch_size_tokens: Optional[int] = None
    training_precision: str = "bf16"
    overtraining_ratio: Optional[float] = None
    data_quality: Optional[dict] = None


@dataclass
class PenaltyEntry:
    """A single penalty with its value and metadata."""
    name: str
    value: float
    source: str
    caveat: str = ""
    confidence: str = "high"  # "high", "medium", "low"
    hardware_dependent: bool = False


@dataclass
class TermResult:
    """A v1 quality term with value, uncertainty, and provenance."""
    name: str
    value: float = 0.0          # fractional residual relative to spine
    delta: float = 0.0          # absolute loss-proxy contribution
    uncertainty: float = 0.0    # fractional one-sigma-ish uncertainty
    confidence: str = "high"
    source: str = "compiler_default"
    notes: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityResult:
    """Output of the quality model."""
    # Core predictions
    predicted_loss: float = 0.0             # relative, not absolute
    chinchilla_baseline: float = 0.0        # Chinchilla spine L(N, D)
    total_penalty_fraction: float = 0.0     # sum of all penalty fractions
    total_penalty_absolute: float = 0.0     # total_penalty_fraction × baseline

    # Penalty breakdown
    penalty_breakdown: Dict[str, PenaltyEntry] = field(default_factory=dict)
    dominant_penalty: str = ""              # which penalty contributed most

    # Confidence
    confidence: str = "high"                # "high", "medium", "low"
    confidence_notes: List[str] = field(default_factory=list)

    # Uncertainty intervals (relative to baseline, in %)
    uncertainty_low_pct: float = 0.0        # lower bound of penalty range
    uncertainty_high_pct: float = 0.0       # upper bound of penalty range

    # Architecture info
    n_active_params: int = 0
    n_total_params: int = 0
    spine_active_params: int = 0
    training_tokens: int = 0

    # Chinchilla regime check
    in_chinchilla_regime: bool = True

    # v1 modular quality outputs
    loss_proxy: float = 0.0
    terms: Dict[str, TermResult] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    uncertainty_total: float = 0.0
    uncertainty_breakdown: Dict[str, float] = field(default_factory=dict)
    quality_model_version: str = "quality_v1_modular_backbone"
    benchmark_score_proxy: Optional[float] = None
    benchmark_uncertainty: Optional[float] = None
    benchmark_notes: List[str] = field(default_factory=list)
    eval_predictions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    calibration_warnings: List[str] = field(default_factory=list)


@dataclass
class QualityComparison:
    """Candidate-vs-baseline quality comparison."""
    candidate: QualityResult
    baseline: QualityResult
    delta_abs: float
    delta_pct: float
    uncertainty_low_pct: float
    uncertainty_high_pct: float
    interpretation: str


# =============================================================================
# Chinchilla spine
# =============================================================================
# Source: Hoffmann et al. "Training Compute-Optimal Large Language Models" (2022)
#         Table 2 / Eq. 5
# Caveat: calibrated for MassiveText, AdamW, specific LR schedule.
#         More reliable for relative predictions than absolute loss values.
#         See Gadre et al. (2024) for recipe-dependence analysis.

CHINCHILLA_E = 1.69       # irreducible loss
CHINCHILLA_A = 406.4      # parameter scaling coefficient
CHINCHILLA_ALPHA = 0.34   # parameter scaling exponent
CHINCHILLA_B = 410.7      # data scaling coefficient
CHINCHILLA_BETA = 0.28    # data scaling exponent

# Chinchilla calibration range (approximate)
CHINCHILLA_N_MIN = 70e6    # 70M params
CHINCHILLA_N_MAX = 16e9    # 16B params (extrapolation beyond this)
CHINCHILLA_D_MIN = 5e9     # 5B tokens
CHINCHILLA_D_MAX = 5e12    # 5T tokens

QUALITY_DEFAULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "quality_defaults.yaml",
)

_QUALITY_CONSTANTS_CACHE: Dict[str, Dict[str, Any]] = {}

DEFAULT_QUALITY_CONSTANTS = {
    "spine": {
        "enabled": True,
        "form": "power_law_M_D",
        "E": CHINCHILLA_E,
        "A": CHINCHILLA_A,
        "B": CHINCHILLA_B,
        "alpha": CHINCHILLA_ALPHA,
        "beta": CHINCHILLA_BETA,
        "uncertainty_in_regime": 0.03,
        "uncertainty_out_of_regime": 0.08,
        "source": "Hoffmann_et_al_2022_placeholder_defaults",
    },
    "architecture_residual": {
        "enabled": True,
        "source": "coupled_width_depth_attention_residual",
        "reference_d_model": 4096,
        "reference_d_head": 128,
        "d_head_ref": 128,
        "preferred_d_head": 128,
        "acceptable_d_head": [64, 128, 256],
        "reference_kv_heads_min": 8,
        "reference_mlp_attn_ratio": 2.0,
        "reference_depth_width": 0.0078125,
        # v1-fix: shape_law block consumed by penalties.shape_penalty. Edit
        # these (or override in the YAML) to apply a refit without touching
        # Python source.
        "shape_law": {
            "K_W": 1.80,        # d_opt = K_W × N^gamma_W
            "gamma_W": 0.341,
            "K_D": 0.014,       # l_opt = K_D × N^gamma_D
            "gamma_D": 0.341,
            "C": 0.03,          # penalty strength (2× deviation ≈ 2% PPL)
            "d_min": 256,       # floor on d_opt
            "l_min": 2,         # floor on l_opt
            "fit_source": "v1_fix_2026_06_refit_llama_mistral_qwen_gpt3",
        },
        "weights": {
            "width_depth_legacy_scale": 1.0,
            "mlp_attention": 0.0,
            # v1-fix: head-shape weights bumped from 0.001 to give them real
            # influence on the optimum. Old values made the residual blind to
            # d_head and query-head choices.
            "d_head": 0.01,             # was 0.001
            "query_heads": 0.005,       # was 0.001
            "kv_heads": 0.003,          # was 0.001
            "gqa_sharing": 0.0015,
            "attention_bottleneck": 0.003,
            # v1-fix Part 3: FlashAttention efficiency drops sharply below
            # d_head=64 (kernel doesn't reach roofline). Flat penalty applied
            # whenever d_head < attn_kernel_underrun_threshold.
            "attn_kernel_underrun": 0.003,
            # v1-fix Part (b): attention long-context degradation.
            # Calibrated against: RULER NIAH plateau past 64-128k for dense
            # attention models; lost-in-the-middle (Liu 2024); attention dilution
            # studies. Weight 0.006 yields:
            #   ctx=32k:  0%      (in-band reference)
            #   ctx=128k: 0.83%   (log(4) × 0.006)
            #   ctx=1M:   2.08%   (log(32) × 0.006)
            #   ctx=4M:   2.91%   (log(128) × 0.006)
            # Set to 0 to disable (recovers prior loss-proxy behavior).
            "attn_long_context": 0.006,
            # v1-fix MLA: weight on the latent-compression quality penalty.
            # Calibrated against DeepSeek-V2 §3 ablation where c_kv=512
            # (= 4× d_head=128) is roughly MHA-quality. 0.004 yields a
            # ~0.3% penalty when latent halves to c_kv=2·d_head.
            "mla_compression": 0.004,
            # v1-fix NSA: under-coverage penalty weight.
            "nsa_undercoverage": 0.005,
            # v1-fix 2:4 sparsity: per-share penalty weight. 0.015 yields
            # ~1.5% loss-proxy at 100% sparse params, scaling linearly.
            "sparsity_2_4": 0.015,
            # v1-fix YOCO: weight × share_fraction. 0.012 yields ~1.2%
            # loss-proxy when 31 of 32 layers share (K=1).
            "yoco_sharing": 0.012,
        },
        "attn_kernel_underrun_threshold": 64,
        # v1-fix Part (b): reference context past which dense attention
        # starts paying the long-context degradation. 32k chosen because
        # most attention models still RULER-pass at 32k but degrade past it.
        "attn_long_context_ref_ctx": 32768,
        # v1-fix RoPE scaling: per-method multipliers on attn_long_context.
        # Calibration source: published ablations on YaRN, LongRoPE, NTK.
        "rope_scaling_multipliers": {
            "none":     1.00,
            "pi":       0.85,
            "ntk":      0.65,
            "yarn":     0.45,
            "longrope": 0.40,
        },
        "uncertainty": 0.25,
        "head_uncertainty": 0.35,
    },
    "precision_residual": {
        "enabled": True,
        "source": "component_precision_sensitivity_with_hardware_feasibility",
        "use_component_table": True,
        "default_uncertainty": 0.30,
    },
    "precision_sensitivity": {
        # Fix #12: FP8 weight penalties were ~5-10× too pessimistic vs
        # empirical pretraining results (FP8-LM, NVIDIA Transformer Engine,
        # DeepSeek-V3 trained in FP8 with negligible loss). Lowered to match
        # 2024-2025 reports (≤0.5% absolute loss for FP8 weights on most
        # components). FP4/MXFP4 remain conservative — public training data
        # is still sparse at frontier scale.
        "ffn": {
            "fp8": {"delta": 0.002, "uncertainty": 0.005, "risk": "low"},
            "fp4": {"delta": 0.04, "uncertainty": 0.04, "risk": "medium"},
            "mxfp4": {"delta": 0.012, "uncertainty": 0.020, "risk": "low_medium"},
            "mxfp6": {"delta": 0.004, "uncertainty": 0.012, "risk": "low"},
        },
        "attention_qkv": {
            "fp8": {"delta": 0.004, "uncertainty": 0.008, "risk": "low_medium"},
            "fp4": {"delta": 0.06, "uncertainty": 0.06, "risk": "medium_high"},
            "mxfp4": {"delta": 0.018, "uncertainty": 0.025, "risk": "medium"},
            "mxfp6": {"delta": 0.006, "uncertainty": 0.015, "risk": "low_medium"},
        },
        "attention_o": {
            "fp8": {"delta": 0.003, "uncertainty": 0.007, "risk": "low_medium"},
            "fp4": {"delta": 0.05, "uncertainty": 0.05, "risk": "medium_high"},
            "mxfp4": {"delta": 0.015, "uncertainty": 0.025, "risk": "medium"},
            "mxfp6": {"delta": 0.005, "uncertainty": 0.013, "risk": "low_medium"},
        },
        "qk_logits": {
            "fp8": {"delta": 0.05, "uncertainty": 0.05, "risk": "medium_high"},
            "fp32_accum": {"delta": 0.0, "uncertainty": 0.0, "risk": "low"},
        },
        "router": {
            "fp8": {"delta": 0.04, "uncertainty": 0.05, "risk": "medium_high"},
            "bf16": {"delta": 0.0, "uncertainty": 0.0, "risk": "low"},
        },
        "experts": {
            "int8": {"delta": 0.02, "uncertainty": 0.03, "risk": "medium"},
            "fp8": {"delta": 0.01, "uncertainty": 0.03, "risk": "medium"},
            "fp4": {"delta": 0.06, "uncertainty": 0.06, "risk": "medium_high"},
        },
        "kv_cache": {
            # Fix #12: KIVI INT4 with per-channel scaling is ~0.5–1.5%
            # perplexity loss at 7B-13B (Hooper 2024) — substantially less
            # than the 6% the prior table assumed. INT8 KV is within noise.
            "int8": {"delta": 0.001, "uncertainty": 0.005, "risk": "low"},
            "int4": {"delta": 0.012, "uncertainty": 0.020, "risk": "medium"},
        },
        "lm_head": {
            # Fix #12: lm_head FP8 is more sensitive than FFN FP8 but not
            # 30× more. Numbers below match the FP8-LM ablation table.
            "fp8": {"delta": 0.006, "uncertainty": 0.012, "risk": "low_medium"},
            "fp4": {"delta": 0.05, "uncertainty": 0.06, "risk": "medium_high"},
            "bf16": {"delta": 0.0, "uncertainty": 0.0, "risk": "low"},
        },
    },
    "moe_residual": {
        "enabled": True,
        # v1 calibration sources:
        #   capacity     — Krajewski et al. (2024) + Mixtral 8x7B / DeepSeek-V2
        #                  reported deltas vs same-active dense baselines.
        #   capacity_ratio_cap — saturating cap on log(N_total/N_active); above
        #                  this ratio the bonus stops growing. Without the cap
        #                  the residual spuriously rewards top_k=1 routing.
        #   granularity  — Krajewski granularity term, bonus for G = n_e/top_k
        #                  above the reference (G_ref = 8); negative weight
        #                  means *lower* loss as granularity grows.
        #   routing      — penalty per unit of routing imbalance; defaults to
        #                  0 under the balanced-routing training objective.
        #   shared_exp   — small DeepSeek-V2-ablation bonus for the always-on
        #                  expert presence at fixed N_active.
        #   top_k1       — Switch-style top_k=1 quality penalty vs top_k>=2.
        #   router_fp8   — penalty when router precision is fp8 (instability).
        #   dense_prefix — v1-fix Part B: small per-dense-prefix-layer
        #                  stability bonus, saturating at preferred_prefix.
        #                  Captures the DeepSeek-V3 / Qwen3-MoE observation
        #                  that the first 1-3 dense layers improve training
        #                  stability and downstream quality at fixed
        #                  N_active. Beyond preferred_prefix the model is
        #                  effectively dense and the capacity bonus
        #                  pro-rates it away anyway.
        "source": "moe_capacity_krajewski_plus_mixtral_deepseek_priors",
        "weights": {
            "capacity": 0.05,        # ~6% loss reduction at Mixtral ratio (3.6x)
            "granularity": -0.005,   # small Krajewski bonus above G=8
            "routing": 0.10,         # >0 only when load_balance < 1
            "shared_experts": -0.005, # DeepSeek shared expert bonus
            "top_k1": 0.015,         # Switch-style top_k=1 penalty
            "router_fp8": 0.005,     # fp8 router penalty
            "dense_prefix": -0.003,  # per-layer bonus, applied up to preferred_prefix
        },
        "capacity_ratio_cap": 4.0,
        "reference_granularity": 8.0,
        "preferred_dense_prefix": 3,  # DeepSeek-V3 has 3 dense layers (layers 0-2)
        "uncertainty": 0.40,
    },
    "state_residual": {
        "enabled": True,
        "source": "v2_family_specific_hybrid_ratio",
        "reference_context_length": 8192,
        # --- Family-specific hybrid-ratio band-pass priors ---
        # Each family has (p_low, p_high) for the preferred attention fraction band,
        # and (p_recall_min_general, p_recall_min_recall) for recall-risk.
        "families": {
            "mamba_sequential": {
                "p_low": 0.12, "p_high": 0.28,
                "preferred_p_attn": [0.14, 0.18],
                "p_recall_min_general": 0.14, "p_recall_min_recall": 0.25,
                "confidence": "medium",
            },
            "gated_delta_or_kda_linear": {
                "p_low": 0.12, "p_high": 0.28,
                "preferred_p_attn": [0.14, 0.25],
                "p_recall_min_general": 0.14, "p_recall_min_recall": 0.25,
                "confidence": "medium-high",
            },
            "generic_linear_attention": {
                "p_low": 0.12, "p_high": 0.28,
                "preferred_p_attn": [0.14, 0.25],
                "p_recall_min_general": 0.14, "p_recall_min_recall": 0.25,
                "confidence": "medium",
            },
            "parallel_hybrid_heads": {
                "p_low": 0.10, "p_high": 0.35,
                "preferred_p_attn": [0.10, 0.25],
                "p_recall_min_general": 0.10, "p_recall_min_recall": 0.20,
                "confidence": "medium-low",
            },
            "recurrent_local_attention": {
                "p_low": 0.10, "p_high": 0.35,
                "preferred_p_attn": [0.10, 0.30],
                "p_recall_min_general": 0.10, "p_recall_min_recall": 0.20,
                "confidence": "medium-low",
            },
        },
        # --- Weights for the 5 sub-terms ---
        "w_under": 0.08,       # penalty weight for p_attn below band
        "w_over": 0.04,        # penalty weight for p_attn above band
        "w_recall": 0.06,      # recall-risk weight
        "w_capacity": 0.02,    # state-capacity (compression) weight
        "w_kv_cost": 0.01,     # effective-kv-cost weight
        # --- State capacity parameters ---
        "compression_scale": 0.02,
        "composition_scale": 0.01,
        "d_state_reference": 192,
        "refresh_factor_weight": 0.5,
        # --- Bounds ---
        "min_delta": -0.05,
        "max_delta": 0.30,
        "uncertainty": 0.40,
        "uncertainty_outside_band": 0.55,
    },
    "downstream_head": {
        "enabled": False,
        "input_mode": "loss_proxy",
        "alpha": 0.0,
        "beta": 1.0,
        "gamma": 0.0,
        "uncertainty": 0.80,
    },
    "data_quality": {
        "enabled": False,
        "mode": "none",
        "effective_token_multiplier": 1.0,
        "uncertainty": 0.0,
    },
    "uncertainty": {
        "max_uncertainty": 0.95,
        "additive_interaction_bonus": 0.005,
    },
    "risk_residual": {
        "enabled": True,
        "source": "compiler_attention_and_architecture_risk_prior",
        "uncertainty": 0.02,
    },
    # v1-fix MTP: sample-efficiency bonus per Multi-Token Prediction depth.
    # DeepSeek-V3 reports k=1 yields ~0.6% loss improvement on Hellaswag /
    # MMLU at fixed pretraining FLOPs. Saturates at k=2.
    "mtp_residual": {
        "enabled": True,
        "source": "deepseek_v3_mtp_sample_efficiency",
        "bonus_per_depth": 0.006,        # 0.6% per effective depth at w=0.3
        "depth_saturation_cap": 2,        # diminishing past k=2
    },
}


def chinchilla_loss(n_active: int, training_tokens: int) -> float:
    """
    Chinchilla scaling law: L(N, D) = E + A / N^α + B / D^β

    Source: Hoffmann et al. (2022), Table 2.
    Caveat: coefficients calibrated for MassiveText + AdamW + cosine LR.
    """
    if n_active <= 0 or training_tokens <= 0:
        return CHINCHILLA_E + CHINCHILLA_A + CHINCHILLA_B

    return (
        CHINCHILLA_E
        + CHINCHILLA_A / (n_active ** CHINCHILLA_ALPHA)
        + CHINCHILLA_B / (training_tokens ** CHINCHILLA_BETA)
    )


def _chinchilla_loss_with_constants(n_active: int, training_tokens: int, constants: Dict[str, Any]) -> float:
    spine = constants.get("spine", {})
    if n_active <= 0 or training_tokens <= 0:
        return spine.get("E", CHINCHILLA_E) + spine.get("A", CHINCHILLA_A) + spine.get("B", CHINCHILLA_B)
    return (
        spine.get("E", CHINCHILLA_E)
        + spine.get("A", CHINCHILLA_A) / (n_active ** spine.get("alpha", CHINCHILLA_ALPHA))
        + spine.get("B", CHINCHILLA_B) / (training_tokens ** spine.get("beta", CHINCHILLA_BETA))
    )


def is_in_chinchilla_regime(n_active: int, training_tokens: int) -> Tuple[bool, List[str]]:
    """Check if (N, D) is within Chinchilla's calibration range."""
    notes = []
    in_regime = True

    if n_active < CHINCHILLA_N_MIN:
        notes.append(f"N={n_active/1e6:.0f}M is below Chinchilla's calibration range (70M-16B)")
        in_regime = False
    if n_active > CHINCHILLA_N_MAX:
        notes.append(f"N={n_active/1e9:.1f}B is above Chinchilla's calibration range (70M-16B); extrapolation")
        in_regime = False
    if training_tokens < CHINCHILLA_D_MIN:
        notes.append(f"D={training_tokens/1e9:.0f}B tokens is below Chinchilla's calibration range")
        in_regime = False
    if training_tokens > CHINCHILLA_D_MAX:
        notes.append(f"D={training_tokens/1e12:.1f}T tokens is above Chinchilla's calibration range; extrapolation")
        in_regime = False

    return in_regime, notes


def load_quality_constants(path: Optional[str] = None) -> Dict[str, Any]:
    """Load quality v1 defaults, falling back to built-in defaults if needed."""
    path = path or os.environ.get("AC_QUALITY_DEFAULTS") or QUALITY_DEFAULTS_PATH
    if path in _QUALITY_CONSTANTS_CACHE:
        return _deepcopy_dict(_QUALITY_CONSTANTS_CACHE[path])

    constants = _deepcopy_dict(DEFAULT_QUALITY_CONSTANTS)
    if not os.path.exists(path):
        _QUALITY_CONSTANTS_CACHE[path] = _deepcopy_dict(constants)
        return constants
    try:
        with open(path) as f:
            if str(path).lower().endswith(".json"):
                loaded = json.load(f) or {}
            else:
                import yaml  # type: ignore
                loaded = yaml.safe_load(f) or {}
        if isinstance(loaded, dict):
            constants = _deep_merge(constants, loaded)
            _QUALITY_CONSTANTS_CACHE[path] = _deepcopy_dict(constants)
            return constants
    except Exception:
        # PyYAML is optional in this repo. The in-code defaults mirror the YAML
        # file so validation remains dependency-free.
        _QUALITY_CONSTANTS_CACHE[path] = _deepcopy_dict(constants)
        return constants
    _QUALITY_CONSTANTS_CACHE[path] = _deepcopy_dict(constants)
    return constants


def _deepcopy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _deepcopy_dict(v)
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _safe_log(value: float) -> float:
    return math.log(max(float(value), 1e-9))


def _eval_feature_value(
    feature: str,
    arch: ArchConfig,
    training: TrainingConfig,
    result: QualityResult,
) -> float:
    if feature == "intercept":
        return 1.0
    if feature == "predicted_loss":
        return float(result.predicted_loss)
    if feature == "log_active_params_b":
        return _safe_log(max(result.n_active_params, 1) / 1e9)
    if feature == "log_total_params_b":
        return _safe_log(max(result.n_total_params, 1) / 1e9)
    if feature == "log_training_tokens_t":
        return _safe_log(max(training.training_tokens, 1) / 1e12)
    if feature == "log_context_length":
        return _safe_log(max(training.sequence_length, 1))
    if feature == "is_moe":
        return 1.0 if arch._moe_enabled or arch.model_type == "moe" else 0.0
    if feature == "is_state_or_hybrid":
        return 1.0 if arch.state_config or arch.model_type in {"state", "hybrid"} else 0.0
    return 0.0


def _domain_warning(
    *,
    label: str,
    value: float,
    domain: Dict[str, Any],
) -> Optional[str]:
    rng = domain.get(label)
    if not isinstance(rng, dict):
        return None
    lo = rng.get("min")
    hi = rng.get("max")
    if lo is None or hi is None:
        return None
    if value < float(lo) or value > float(hi):
        return (
            f"Lab calibration extrapolates on {label}: "
            f"{value:.4g} outside [{float(lo):.4g}, {float(hi):.4g}]."
        )
    return None


def _apply_lab_calibration_notes(
    result: QualityResult,
    constants: Dict[str, Any],
    arch: ArchConfig,
    training: TrainingConfig,
) -> None:
    lab = constants.get("lab_calibration", {})
    if not isinstance(lab, dict) or not lab:
        return
    status = lab.get("status")
    if status and status != "production_ready":
        note = f"Lab calibration status is {status}; do not use as sign-off evidence without inspecting the pack."
        result.confidence_notes.append(note)
        result.calibration_warnings.append(note)
    for warning in lab.get("warnings", []) or []:
        text = str(warning)
        result.confidence_notes.append(text)
        result.calibration_warnings.append(text)
    domain = lab.get("domains", {})
    if isinstance(domain, dict):
        checks = (
            ("active_params_b", max(result.n_active_params, 1) / 1e9),
            ("total_params_b", max(result.n_total_params, 1) / 1e9),
            ("training_tokens_t", max(training.training_tokens, 1) / 1e12),
            ("context_length", float(training.sequence_length)),
        )
        for label, value in checks:
            warning = _domain_warning(label=label, value=value, domain=domain)
            if warning:
                result.confidence_notes.append(warning)
                result.calibration_warnings.append(warning)
                if result.confidence == "high":
                    result.confidence = "medium"


def _apply_eval_models(
    result: QualityResult,
    constants: Dict[str, Any],
    arch: ArchConfig,
    training: TrainingConfig,
) -> None:
    eval_models = constants.get("eval_models", {})
    if not isinstance(eval_models, dict):
        return
    models = eval_models.get("evals", {})
    if not isinstance(models, dict) or not models:
        return
    global_status = str(eval_models.get("status", "experimental"))
    if global_status != "validated":
        result.confidence_notes.append(
            f"Eval projections are {global_status}; inspect held-out-family CV before relying on them."
        )
    predictions: Dict[str, Dict[str, Any]] = {}
    for name, model in sorted(models.items()):
        if not isinstance(model, dict):
            continue
        features = model.get("feature_names") or eval_models.get("feature_names") or []
        coefs = model.get("coefficients") or []
        if not features or not coefs or len(features) != len(coefs):
            continue
        x = [_eval_feature_value(str(f), arch, training, result) for f in features]
        score = sum(float(c) * v for c, v in zip(coefs, x))
        uncertainty = float(model.get("uncertainty") or 0.0)
        score_min = model.get("score_min")
        score_max = model.get("score_max")
        outside_training_range = False
        if score_min is not None and score_max is not None:
            span = max(float(score_max) - float(score_min), 1e-9)
            margin = max(span * 0.25, uncertainty)
            if score < float(score_min) - margin or score > float(score_max) + margin:
                outside_training_range = True
        predictions[str(name)] = {
            "score": round(score, 6),
            "uncertainty": round(uncertainty, 6),
            "status": model.get("status", global_status),
            "n": model.get("n", 0),
            "families": model.get("families", []),
            "train_rmse": model.get("train_rmse"),
            "heldout_family_rmse": model.get("heldout_family_rmse"),
            "outside_training_score_range": outside_training_range,
            "notes": list(model.get("warnings", []) or []),
        }
        if outside_training_range:
            warning = f"Eval projection for {name} is outside the calibrated score range."
            result.calibration_warnings.append(warning)
            result.confidence_notes.append(warning)
    result.eval_predictions = predictions


def _safe_log_ratio(value: float, ref: float) -> float:
    if value <= 0 or ref <= 0:
        return 0.0
    return math.log(value / ref)


def _kv_bits_to_policy(bits: int) -> str:
    if bits <= 4:
        return "int4"
    if bits <= 8:
        return "int8"
    return "bf16"


def _term_uncertainty(term: TermResult) -> float:
    if term.uncertainty > 0:
        return term.uncertainty
    if term.value == 0:
        return 0.0
    spread = {"high": 0.20, "medium": 0.30, "low": 0.50}.get(term.confidence, 0.30)
    return abs(term.value) * spread


def _entry_to_term(name: str, entry: PenaltyEntry, baseline: float) -> TermResult:
    return TermResult(
        name=name,
        value=entry.value,
        delta=entry.value * baseline,
        uncertainty=0.0,
        confidence=entry.confidence,
        source=entry.source,
        notes=[entry.caveat] if entry.caveat else [],
    )


def _make_penalty_entry(
    name: str,
    value: float,
    source: str,
    caveat: str = "",
    confidence: str = "high",
    hardware_dependent: bool = False,
) -> PenaltyEntry:
    return PenaltyEntry(
        name=name,
        value=value,
        source=source,
        caveat=caveat,
        confidence=confidence,
        hardware_dependent=hardware_dependent,
    )


# =============================================================================
# Modular quality backbone
# =============================================================================

def _data_quality_filter(training: TrainingConfig, constants: Dict[str, Any]) -> Tuple[int, TermResult, List[str]]:
    defaults = constants.get("data_quality", {})
    policy = _deep_merge(_deepcopy_dict(defaults), training.data_quality or {})
    enabled = bool(policy.get("enabled", False))
    mode = policy.get("mode", "none")
    multiplier = float(policy.get("effective_token_multiplier", 1.0))
    warnings = []
    d_eff = training.training_tokens

    term = TermResult(
        name="data_quality",
        value=0.0,
        uncertainty=0.0,
        confidence="not_applicable",
        source="disabled",
        features={"mode": mode, "effective_token_multiplier": multiplier},
    )

    if not enabled or mode == "none":
        return d_eff, term, warnings

    term.confidence = "low"
    term.source = "quality_v1_data_quality_filter_placeholder"
    term.uncertainty = float(policy.get("uncertainty", 0.0))
    if mode == "effective_tokens":
        d_eff = max(1, int(training.training_tokens * multiplier))
        term.notes.append(
            f"Training tokens adjusted by data-quality multiplier {multiplier:.3f}; "
            "this is a proxy, not a measured data-mixture model."
        )
    elif mode == "uncertainty_only":
        term.notes.append("Data quality only widens uncertainty; loss proxy unchanged.")
    elif mode == "candidate_filter":
        warnings.append("Data-quality candidate filtering hook is reserved; no candidate rejected in v1.")
    else:
        warnings.append(f"Unknown data-quality mode {mode!r}; ignored.")
    return d_eff, term, warnings


def _large_shape_stability_prior(
    arch: ArchConfig,
    n_total: int,
    baseline_loss: float,
) -> TermResult:
    """Mild prior against implausible frontier-scale depth/width shapes.

    Public scaling laws do not pin down 120B-1T internal shapes, but extreme
    skinny/deep or shallow/wide candidates are unstable enough that they should
    not win by accident. Anchors are coarse frontier-family centers, not
    measured optima.

    Shape-stability concerns the *dense forward path* (d_model, n_layers), so
    we anchor on **active** params, not raw n_total. Otherwise MoE edits like
    change_moe_topology that resize expert count would flip the prior on or
    off purely because n_total moved across the threshold, predicting a
    spurious quality change with no architectural cause.
    """
    n_active = max(1, getattr(arch, "n_active_params", n_total) or n_total)
    anchor_n = n_active
    total_b = anchor_n / 1e9
    # v1-fix demo-audit D2: extend the shape prior below 120B. The 120B
    # cutoff was originally rationalized as "scaling laws don't pin down
    # internal shapes here", but in practice the absence of *any* shape
    # signal let the serving-constrained branch select L=5 / L=7 / L=13
    # transformers at 3B-7B targets. We now apply a softer version of the
    # same band check across the whole 1B-1T range, with sub-120B anchors
    # tied to Llama / Mistral / Qwen published shapes.
    anchors = (
        (1.0,    2048.0,  22.0),  # Llama-3 1B / Qwen2-1.5B
        (3.0,    3072.0,  28.0),  # Qwen2-3B
        (7.0,    4096.0,  32.0),  # Llama-3 8B / Mistral-7B
        (13.0,   5120.0,  40.0),  # Llama-2 13B
        (70.0,   8192.0,  80.0),  # Llama-3 70B
        (120.0, 12288.0,  80.0),
        (250.0, 16384.0,  88.0),
        (500.0, 24576.0, 104.0),
        (1000.0,28672.0, 112.0),
    )
    target_log = math.log(max(total_b, 1.0))
    center_b, center_d, center_l = min(
        anchors,
        key=lambda x: abs(math.log(x[0]) - target_log),
    )

    width_ratio = arch.d_model / max(1.0, center_d)
    depth_ratio = arch.n_layers / max(1.0, center_l)
    shape_ratio = (arch.n_layers / max(1.0, arch.d_model)) / (center_l / center_d)

    penalty = 0.0
    reasons: List[str] = []
    if width_ratio < 0.70:
        penalty += 0.040 * math.log(0.70 / max(width_ratio, 1e-6)) ** 2
        reasons.append("width below large-model stability band")
    elif width_ratio > 1.45:
        penalty += 0.020 * math.log(width_ratio / 1.45) ** 2
        reasons.append("width above large-model stability band")

    if depth_ratio < 0.55:
        penalty += 0.030 * math.log(0.55 / max(depth_ratio, 1e-6)) ** 2
        reasons.append("depth below large-model stability band")
    elif depth_ratio > 1.75:
        penalty += 0.025 * math.log(depth_ratio / 1.75) ** 2
        reasons.append("depth above large-model stability band")

    if shape_ratio < 0.50:
        penalty += 0.020 * math.log(0.50 / max(shape_ratio, 1e-6)) ** 2
        reasons.append("depth/width ratio too shallow")
    elif shape_ratio > 2.20:
        penalty += 0.020 * math.log(shape_ratio / 2.20) ** 2
        reasons.append("depth/width ratio too deep")

    # v1-fix demo-audit D2: raise the cap. The previous 6% cap was too low
    # to deselect catastrophic depth/width violations — depth_ratio=19
    # (L=1980 at d=6144 vs anchor L=104) had a raw penalty of ~0.18 that
    # the old `min(0.060, ...)` clipped to 6%. The new cap is 35%, and
    # extreme excursions (>3× off-band) get a hard quadratic term on top
    # so the optimizer can actually see them.
    if depth_ratio > 3.0 or width_ratio < 0.30:
        # Quadratic blow-up for "this is not a transformer anyone would
        # train" territory. Capped at 0.35.
        extreme = 0.0
        if depth_ratio > 3.0:
            extreme += 0.10 * (math.log(depth_ratio / 3.0)) ** 2
        if width_ratio < 0.30:
            extreme += 0.10 * (math.log(0.30 / max(width_ratio, 1e-6))) ** 2
        penalty += extreme
        reasons.append("extreme aspect ratio (depth>3× or width<0.3× of anchor)")

    penalty = min(0.35, max(0.0, penalty))
    if penalty <= 0:
        return TermResult(
            name="large_shape_stability_prior",
            confidence="not_applicable",
            source="within_frontier_shape_band",
            features={
                "target_total_params_b": round(total_b, 3),
                "anchor_total_params_b": center_b,
                "anchor_d_model": int(center_d),
                "anchor_n_layers": int(center_l),
                "width_ratio": round(width_ratio, 4),
                "depth_ratio": round(depth_ratio, 4),
                "depth_width_ratio_vs_anchor": round(shape_ratio, 4),
            },
        )

    confidence = "low" if penalty >= 0.025 else "medium"
    return TermResult(
        name="large_shape_stability_prior",
        value=penalty,
        delta=penalty * baseline_loss,
        uncertainty=penalty * 0.60,
        confidence=confidence,
        source="frontier_scale_depth_width_prior",
        notes=[
            "Coarse shape prior across 1B-1T; verify with training stability sweeps.",
            "; ".join(reasons),
        ],
        features={
            "target_total_params_b": round(total_b, 3),
            "anchor_total_params_b": center_b,
            "anchor_d_model": int(center_d),
            "anchor_n_layers": int(center_l),
            "width_ratio": round(width_ratio, 4),
            "depth_ratio": round(depth_ratio, 4),
            "depth_width_ratio_vs_anchor": round(shape_ratio, 4),
        },
    )


def _architecture_residual(
    arch: ArchConfig,
    n_active: int,
    baseline_loss: float,
    constants: Dict[str, Any],
    workload_spec: Optional[Dict[str, Any]] = None,
) -> Tuple[TermResult, Dict[str, PenaltyEntry]]:
    cfg = constants.get("architecture_residual", {})
    weights = cfg.get("weights", {})

    q_params = arch.d_model * arch.d_head * arch.n_heads
    kv_params = 2 * arch.d_model * arch.d_head * arch.n_kv_heads
    o_params = arch.d_head * arch.n_heads * arch.d_model
    attn_params = max(1, q_params + kv_params + o_params)
    mlp_params = 3 * arch.d_model * arch.ffn_dim
    n_query_heads = arch.n_heads
    n_kv_heads = max(1, arch.n_kv_heads)
    d_head = arch.d_head
    gqa_group = n_query_heads / n_kv_heads
    ffn_ratio = arch.ffn_dim / max(1, arch.d_model)
    mlp_attn = mlp_params / attn_params
    depth_width = arch.n_layers / max(1, arch.d_model)
    d_head_ref = float(cfg.get("d_head_ref", cfg.get("reference_d_head", 128)))
    preferred_d_head = float(cfg.get("preferred_d_head", d_head_ref))
    acceptable_d_head = set(int(x) for x in cfg.get("acceptable_d_head", [64, 128, 256]))
    default_query_heads = max(1, round(arch.d_model / max(1.0, d_head_ref)))
    reference_kv_min = max(1, int(cfg.get("reference_kv_heads_min", 8)))
    kv_reference = min(default_query_heads, reference_kv_min)
    # v1-fix MLA: kv_bytes per token per layer differs sharply for MLA.
    # Standard attention caches K + V per kv-head: 2 × n_kv × d_head × bpe.
    # MLA caches one compressed latent + RoPE'd K: (c_kv + d_rope) × bpe.
    # The user-facing KV-bytes figure in the UI uses this number directly.
    if arch.attention_type == "mla" and arch.mla_latent_dim:
        c_kv = int(arch.mla_latent_dim)
        d_rope = int(arch.mla_rope_head_dim or 0)
        kv_bytes_bf16 = (c_kv + d_rope) * 2  # bf16 bytes
    else:
        kv_bytes_bf16 = 2 * n_kv_heads * d_head * 2

    # v1-fix: pass shape_law constants from YAML so refits don't need code edits
    shape_constants = cfg.get("shape_law", None)
    f_width_depth = (
        float(weights.get("width_depth_legacy_scale", 1.0))
        * shape_penalty(arch.d_model, arch.n_layers, n_active, constants=shape_constants)
    )
    f_mlp_attention_ratio = (
        float(weights.get("mlp_attention", 0.0))
        * _safe_log_ratio(mlp_attn, float(cfg.get("reference_mlp_attn_ratio", 2.0))) ** 2
    )
    f_d_head = float(weights.get("d_head", 0.001)) * _safe_log_ratio(d_head, preferred_d_head) ** 2
    f_query_heads = float(weights.get("query_heads", 0.001)) * _safe_log_ratio(n_query_heads, default_query_heads) ** 2

    # v1-fix C1 (demo audit): two-sided KV-heads penalty with a flat-zero
    # plateau across the "GQA sweet spot" [n_heads/8, n_heads/4]. Before
    # this fix the penalty was one-sided (only fired when n_kv_heads was
    # below a fixed kv_reference=8), so the optimizer was rewarded for
    # adding KV heads — extra KV-cache weights bump total_params, which
    # *lowers* the Chinchilla spine loss for "free". Combined with the
    # weak throughput cost of extra KV heads at TP ≥ n_kv_heads, every
    # greenfield run in the v1 demo corpus selected MHA. The shallow-KV
    # branch (kept) and the new excess-KV branch (added) together produce
    # a U-shaped penalty whose minimum is the plateau [GQA-8, GQA-4].
    #
    # Thresholds:
    #   kv_lower_threshold = max(1, n_query_heads / 8)  — GQA-8
    #   kv_upper_threshold = max(kv_lower_threshold, n_query_heads / 4)  — GQA-4
    #
    # The legacy `kv_reference = min(default_query_heads, 8)` constant is
    # also honoured as a lower floor so existing calibration regressions
    # on small models (where n_heads ≤ 32 and n_heads/8 ≤ 4) keep their
    # original behaviour: kv_lower_threshold = max(kv_reference, …).
    #
    # Calibration: the excess-side weight 0.008 was tuned against the v1
    # demo Mistral-7B-class Pareto frontier so MHA (kv=72) lands 0.02–0.03
    # absolute loss above GQA-4 (kv=18), in line with published Ainslie
    # 2023 / Llama-2-70B / Qwen-3 ablations. The shallow-side weight
    # stays at the legacy 0.001.
    kv_lower_threshold = max(kv_reference, n_query_heads / 8.0)
    kv_upper_threshold = max(kv_lower_threshold, n_query_heads / 4.0)
    f_kv_heads = 0.0
    if n_kv_heads < kv_lower_threshold:
        # Shallow GQA / MQA: per-head group is too aggressive. Quadratic
        # in log(kv_lower_threshold / n_kv_heads).
        f_kv_heads = (
            float(weights.get("kv_heads", 0.001))
            * math.log(kv_lower_threshold / max(1.0, n_kv_heads)) ** 2
        )
    elif n_kv_heads > kv_upper_threshold:
        # Excess KV heads (toward MHA): per-head group is too shallow.
        # Quadratic in log(n_kv_heads / kv_upper_threshold).
        f_kv_heads = (
            float(weights.get("kv_heads_excess", 0.008))
            * math.log(n_kv_heads / kv_upper_threshold) ** 2
        )

    # Compatibility with the original ablation prior, without adding it twice.
    # v1-fix (review): clamp the modeled GQA-sharing term to zero in the
    # GQA-8 / d_model >= 2048 regime so it does not contradict the legacy
    # ablation prior (Llama-2 / Mistral within seed variance). Above that
    # regime the smooth log term still fires.
    legacy_gqa = gqa_penalty(n_query_heads, n_kv_heads, arch.d_model)
    if gqa_group <= 8.0 and arch.d_model >= 2048:
        modeled_gqa = 0.0
    else:
        modeled_gqa = float(weights.get("gqa_sharing", 0.0015)) * math.log(max(1.0, gqa_group))
    f_gqa_sharing = max(legacy_gqa, modeled_gqa)

    attention_bottleneck = False
    bottleneck_note = ""
    query_ratio = n_query_heads / max(1, default_query_heads)
    if arch.d_model >= float(cfg.get("reference_d_model", 4096)) and query_ratio < 0.5:
        attention_bottleneck = True
        bottleneck_note = (
            f"Wide model has {n_query_heads} query heads vs derived default "
            f"{default_query_heads}; attention-slot bottleneck risk."
        )
    f_attention_bottleneck = float(weights.get("attention_bottleneck", 0.003)) if attention_bottleneck else 0.0

    # v1-fix Part 3: FlashAttention/SplashAttention kernels lose tile
    # efficiency sharply below d_head=64 (the kernel can't fill the tile,
    # softmax reduction cost dominates). Fixed penalty rather than a smooth
    # falloff because the cliff is sharp in published benchmarks.
    underrun_threshold = int(cfg.get("attn_kernel_underrun_threshold", 64))
    f_attn_kernel_underrun = (
        float(weights.get("attn_kernel_underrun", 0.003))
        if d_head < underrun_threshold else 0.0
    )

    # v1-fix Part (b) — attention long-context degradation.
    # Symmetric counterpart to the state-residual compression penalty: dense
    # softmax attention also degrades at long context (lost-in-the-middle,
    # Liu et al. 2024; RULER plateaus past 64-128k for non-RAG attention;
    # NeedleInAHaystack accuracy drift). The current quality model treats
    # attention as quality-free at any context, which makes hybrid look
    # over-penalized in the loss comparison at long contexts. This term
    # restores the symmetry.
    #
    # Shape: penalty grows logarithmically past a reference context (default
    # 32k), with linear contribution from each attention layer. Hybrid configs
    # pay it scaled by attention_fraction = n_attention / n_layers, so a
    # 1:3 hybrid pays 25% of what a pure-attention stack pays at the same
    # context. SWA configs can opt out via local_window: a 4k window means
    # attention never sees beyond 4k regardless of `context_length`.
    workload = workload_spec or {}
    workload_context = float(workload.get("context_length", workload.get("ctx", 0)) or 0)
    attn_lc_ref_ctx = float(cfg.get("attn_long_context_ref_ctx", 32768))
    attn_lc_weight = float(weights.get("attn_long_context", 0.006))
    # Attention fraction in [0, 1] — 1 for pure-attention, 0 for pure-state
    state_cfg_for_ctx = getattr(arch, "state_config", None) or {}
    if state_cfg_for_ctx.get("enabled") or getattr(arch, "model_type", "dense") in ("state", "hybrid"):
        n_attn_lc = int(state_cfg_for_ctx.get("attention_layers", arch.n_layers))
        n_state_lc = int(state_cfg_for_ctx.get("state_layers", 0))
        attention_fraction = n_attn_lc / max(1, n_attn_lc + n_state_lc)
    else:
        attention_fraction = 1.0
    # SWA cap: if local_window is set and < workload context, treat the
    # attention as effectively running at the window size (no degradation
    # *beyond* what attention already handles well). A small *positive*
    # SWA-locality penalty is added below to reflect that SWA loses
    # access to tokens outside the window — Mistral's own ablation shows
    # a ~0.5%–2% perplexity hit when workload_context grows past the
    # window, calibrated against window=4k at contexts 8k–32k.
    local_window = float(getattr(arch, "local_window", 0) or 0)
    effective_attn_ctx = workload_context
    if local_window > 0:
        effective_attn_ctx = min(workload_context, local_window)
    f_attention_swa_locality = 0.0
    if local_window > 0 and workload_context > local_window and attention_fraction > 0:
        # weight × attention_fraction × log(ctx / window)
        # weight 0.008 reproduces Mistral's published ~0.55% PPL hit at
        # 8k context with window=4k (log2 ≈ 0.69 × 0.008 = 0.55%).
        swa_locality_weight = float(weights.get("attn_swa_locality", 0.008))
        f_attention_swa_locality = (
            swa_locality_weight
            * attention_fraction
            * math.log(workload_context / max(local_window, 1.0))
        )
    if effective_attn_ctx > attn_lc_ref_ctx and attention_fraction > 0:
        # v1-fix RoPE scaling: extension method multiplier on the long-context
        # attention degradation. Calibrated from published RoPE-extension
        # ablations (YaRN, NTK-aware, LongRoPE):
        #   none     1.0× — native, no extension; full degradation
        #   pi       0.85× — Position Interpolation (Chen et al. 2023)
        #   ntk      0.65× — NTK-aware (NousResearch 2023)
        #   yarn     0.45× — YaRN (Peng et al. 2024)
        #   longrope 0.40× — LongRoPE (Ding et al. 2024); per-dim search
        rope_mult_table = (cfg.get("rope_scaling_multipliers") or {
            "none": 1.0, "pi": 0.85, "ntk": 0.65, "yarn": 0.45, "longrope": 0.40,
        })
        rope_mult = float(rope_mult_table.get(
            str(getattr(arch, "rope_scaling_method", "none")).lower(), 1.0
        ))
        # When the workload exceeds the trained extension range, snap back to
        # full degradation regardless of method.
        rope_factor = float(getattr(arch, "rope_scaling_factor", 1.0) or 1.0)
        rope_orig = float(getattr(arch, "rope_original_max_position", 8192) or 8192)
        extended_max = rope_orig * max(1.0, rope_factor)
        if effective_attn_ctx > extended_max:
            # Past the trained extension — extrapolation penalty kicks in.
            rope_mult = max(rope_mult, 1.0)
        f_attention_long_context = (
            attn_lc_weight
            * attention_fraction
            * math.log(effective_attn_ctx / attn_lc_ref_ctx)
            * rope_mult
        )
    else:
        f_attention_long_context = 0.0

    # v1-fix MLA: small compression-quality penalty for MLA. DeepSeek-V2 §3
    # shows MLA at c_kv ≥ 512 is roughly quality-neutral vs MHA (~0.05 PPL
    # delta on their eval suite). The cost grows when c_kv shrinks far below
    # the per-head working dim.
    #
    # v1-fix demo-audit (June 2026 follow-up): the reference scale for
    # "per-head working dim" used to be `d_head`, but on MLA candidates
    # `d_head` was the dense lattice's leaked value (64/128/256), not the
    # actual MLA Q per-head dim (d_nope + d_rope, typically 192). That made
    # the penalty fire asymmetrically — cell-9 MLA paid 0.28 % at dh=256
    # while cell-5 MLA escaped the penalty entirely at dh=64 because c_kv
    # > 4·64. Use the true MLA per-head dim so the threshold is calibrated
    # against DeepSeek's c_kv≈4·(d_nope+d_rope) sweet spot.
    f_attention_mla = 0.0
    if arch.attention_type == "mla" and arch.mla_latent_dim:
        c_kv = float(arch.mla_latent_dim)
        d_nope_mla = float(getattr(arch, "mla_nope_head_dim", 0) or 0)
        d_rope_mla = float(getattr(arch, "mla_rope_head_dim", 0) or 0)
        # Fall back to d_head only if the MLA dims aren't populated (legacy
        # path / older deltas). New emitter always populates them.
        per_head_mla = (d_nope_mla + d_rope_mla) if (d_nope_mla + d_rope_mla) > 0 else float(d_head)
        c_ref = 4.0 * per_head_mla
        if c_kv < c_ref:
            f_attention_mla = (
                float(weights.get("mla_compression", 0.004))
                * math.log(c_ref / max(1.0, c_kv))
            )

    # v1-fix NSA: small quality penalty for Native Sparse Attention.
    # DeepSeek 2025 reports near-parity vs full attention at L=64k with
    # default branch sizes; the cost grows as top_k shrinks below the
    # block-coverage ratio that the compressed branch can compensate for.
    f_attention_nsa = 0.0
    if arch.attention_type == "nsa" and arch.nsa_window_size:
        # Coverage ratio of the (top-k × block-size) + window over the
        # workload context. When coverage shrinks the model loses information.
        win = float(arch.nsa_window_size or 0)
        stk = float(arch.nsa_select_top_k or 0)
        sbs = float(arch.nsa_select_block_size or 0)
        workload_ctx = float((workload_spec or {}).get("context_length", 65536))
        coverage = (stk * sbs + win) / max(1.0, workload_ctx)
        # Reference coverage ratio = 0.05 (DeepSeek paper). Below that, the
        # selection branch loses recall.
        ref_coverage = float(cfg.get("nsa_reference_coverage", 0.05))
        if coverage < ref_coverage and coverage > 0:
            f_attention_nsa = (
                float(weights.get("nsa_undercoverage", 0.005))
                * math.log(ref_coverage / coverage)
            )

    # v1-fix demo-audit (June 2026 follow-up): MLA / NSA candidates were
    # paying MHA-style GQA penalties because the enumerator emits them with
    # n_kv_heads = n_heads as a carrier value (later zeroed in the JSON
    # serializer for display). That carrier value tripped the "excess KV
    # heads" branch of `f_kv_heads`, adding ~1.5% loss to every MLA pareto
    # entry — enough to push MLA off the frontier in every demo cell despite
    # f_attention_mla itself being small.
    #
    # The MHA/GQA residual subterms (d_head/query-heads/KV-heads/GQA-sharing/
    # attention-bottleneck/kernel-underrun) all assume per-head softmax
    # attention with a per-head KV cache. None of that is the cost model
    # for MLA's compressed-latent KV or NSA's compressed+selected branches.
    # Short-circuit those subterms when attention_type is not "full" and
    # let f_attention_mla / f_attention_nsa carry the architecture residual,
    # calibrated against DeepSeek-V2/V3 and DeepSeek-NSA-2025 ablations.
    is_non_mha = (arch.attention_type in ("mla", "nsa"))
    if is_non_mha:
        f_d_head = 0.0
        f_query_heads = 0.0
        f_kv_heads = 0.0
        f_gqa_sharing = 0.0
        f_attention_bottleneck = 0.0
        f_attn_kernel_underrun = 0.0
        # MHA long-context penalty still applies for MLA (it's still softmax
        # attention over the full token set, just with compressed K/V), but
        # NSA is explicitly a long-context construction and is charged via
        # f_attention_nsa coverage instead.
        if arch.attention_type == "nsa":
            f_attention_long_context = 0.0
            f_attention_swa_locality = 0.0

    f_attention_heads = (
        f_d_head + f_query_heads + f_kv_heads
        + f_gqa_sharing + f_attention_bottleneck + f_attn_kernel_underrun
        + f_attention_long_context + f_attention_mla + f_attention_nsa
        + f_attention_swa_locality
    )

    # v1-fix YOCO: cross-layer KV sharing penalty. Microsoft 2024 paper
    # reports ~1-2% perplexity delta vs full per-layer KV when K = 1 or 2
    # self-attention layers (rest shared). Penalty grows logarithmically
    # with the share fraction (n_layers - K) / n_layers.
    f_yoco = 0.0
    yoco_k = int(getattr(arch, "yoco_n_self_attn_layers", 0) or 0)
    if 0 < yoco_k < arch.n_layers:
        share_fraction = (arch.n_layers - yoco_k) / arch.n_layers
        f_yoco = float(weights.get("yoco_sharing", 0.012)) * share_fraction
    value_extra_yoco = f_yoco

    # v1-fix 2:4 sparsity quality penalty. Published ablations (Pool et al.
    # 2021; Frantar 2023; STEP 2024) show ~1-3% PPL increase when applied to
    # FFN weights with magnitude-based pruning + retraining. Penalty scales
    # with the fraction of params that are 2:4 sparse.
    f_sparsity_2_4 = 0.0
    sparsity = getattr(arch, "sparsity_2_4", None) or {}
    if sparsity:
        # Weight each component by its share of the parameter count
        sparse_param_share = 0.0
        # FFN dominates parameter count; approximate the per-component share
        ffn_share = mlp_params / max(1, mlp_params + attn_params)
        for k in ("ffn_up", "ffn_down", "ffn_gate"):
            if sparsity.get(k):
                sparse_param_share += ffn_share / 3.0
        attn_share = 1.0 - ffn_share
        for k in ("attn_qkv", "attn_o"):
            if sparsity.get(k):
                sparse_param_share += attn_share / 2.0
        if sparse_param_share > 0:
            w = float(weights.get("sparsity_2_4", 0.015))
            f_sparsity_2_4 = w * sparse_param_share
    value_extra_sparsity = f_sparsity_2_4
    value = f_width_depth + f_mlp_attention_ratio + f_attention_heads + value_extra_sparsity + value_extra_yoco

    # v1-fix demo-audit (June 2026 follow-up): for MLA the residual's GQA
    # carrier value is meaningless once the MHA subterms are zeroed above.
    # Report it the same way the JSON schema does (0 = N/A) so users don't
    # see a 64-headed "MHA" feature row on an MLA candidate.
    _features_n_kv_heads = 0 if arch.attention_type == "mla" else n_kv_heads
    _features_gqa_group = 0.0 if arch.attention_type == "mla" else round(gqa_group, 4)

    features = {
        "d_model": arch.d_model,
        "n_layers": arch.n_layers,
        "n_query_heads": n_query_heads,
        "n_query_heads_default": default_query_heads,
        "d_head": d_head,
        "n_kv_heads": _features_n_kv_heads,
        "gqa_group_size": _features_gqa_group,
        "ffn_ratio": round(ffn_ratio, 4),
        "mlp_to_attention_param_ratio": round(mlp_attn, 4),
        "depth_width_ratio": depth_width,
        "attention_param_ratio": round(attn_params / max(1, attn_params + mlp_params), 4),
        "kv_bytes_per_token_per_layer_bf16": kv_bytes_bf16,
        "attention_bottleneck_risk": attention_bottleneck,
        "subterms": {
            "width_depth": round(f_width_depth, 6),
            "mlp_attention_ratio": round(f_mlp_attention_ratio, 6),
            "d_head": round(f_d_head, 6),
            "attn_kernel_underrun": round(f_attn_kernel_underrun, 6),
            "query_heads": round(f_query_heads, 6),
            "kv_heads": round(f_kv_heads, 6),
            "gqa_sharing": round(f_gqa_sharing, 6),
            "attention_bottleneck": round(f_attention_bottleneck, 6),
            # v1-fix Part (b): attention long-context degradation subterm.
            "attention_long_context": round(f_attention_long_context, 6),
            # v1-fix MLA: small compression-quality penalty when c_kv < 4·d_head.
            "attention_mla": round(f_attention_mla, 6),
            # v1-fix NSA: under-coverage penalty when (top_k × bs + window) / L < 0.05.
            "attention_nsa": round(f_attention_nsa, 6),
            # SWA locality: positive residual when workload context exceeds
            # the sliding window. Calibrated to Mistral published numbers.
            "attention_swa_locality": round(f_attention_swa_locality, 6),
            # v1-fix 2:4 sparsity: per-component structured sparsity penalty.
            "sparsity_2_4": round(f_sparsity_2_4, 6),
            # v1-fix YOCO: cross-layer KV sharing penalty.
            "yoco_sharing": round(f_yoco, 6),
        },
        # v1-fix Part (b): expose the inputs that drove the long-context term
        # so the explorer can show "this term contributes X% because attention
        # fraction is Y at context Z".
        "attn_long_context_ref_ctx": int(attn_lc_ref_ctx),
        "attn_long_context_effective_ctx": int(effective_attn_ctx),
        "attention_fraction": round(attention_fraction, 4),
    }

    penalties = {
        "shape": _make_penalty_entry(
            "shape", f_width_depth + f_mlp_attention_ratio,
            "Tay et al. (2021); Hoffmann et al. (2022)",
            caveat="Compatibility alias for width/depth plus MLP-attention residual",
            confidence="medium",
        ),
        "gqa": _make_penalty_entry(
            "gqa", f_gqa_sharing,
            "Touvron et al. (2023) Llama-2; Jiang et al. (2023) Mistral",
            caveat="Compatibility alias for uncertain GQA-sharing residual",
            confidence="medium" if gqa_group > 1 else "high",
        ),
        "attention_heads": _make_penalty_entry(
            "attention_heads", f_attention_heads,
            cfg.get("source", "coupled_width_depth_attention_residual"),
            caveat="Coupled d_head/query-head/KV-head/GQA residual; not a standalone head-count scaling law",
            confidence="medium" if f_attention_heads > 0 else "high",
        ),
    }

    notes = [
        "Query heads are a weak, saturating prior derived from width, not a monotonic quality law.",
        "KV heads are modeled as a direct memory/latency tradeoff; the uncertain quality term is GQA sharing.",
    ]
    if d_head not in acceptable_d_head:
        notes.append(f"d_head={d_head} is outside the conventional set {sorted(acceptable_d_head)}.")
    if gqa_group > 16:
        notes.append("Aggressive GQA/MQA (group_size>16) increases quality uncertainty; validate with ablations before treating as free.")
    if bottleneck_note:
        notes.append(bottleneck_note)

    confidence = "medium" if value > 0 else "high"
    if gqa_group >= 16 or attention_bottleneck:
        confidence = "low"
    uncertainty = (
        abs(f_width_depth + f_mlp_attention_ratio) * float(cfg.get("uncertainty", 0.25))
        + abs(f_attention_heads) * float(cfg.get("head_uncertainty", 0.35))
    )
    return TermResult(
        name="architecture_residual",
        value=value,
        delta=value * baseline_loss,
        uncertainty=uncertainty,
        confidence=confidence,
        source=cfg.get("source", "architecture_residual"),
        notes=notes,
        features=features,
    ), penalties


def _component_table_lookup(constants: Dict[str, Any], component: str, precision: str) -> Tuple[float, float, str]:
    if precision in ("bf16", "fp16", "fp32", "fp32_accum"):
        if precision == "fp32_accum":
            row = constants.get("precision_sensitivity", {}).get(component, {}).get(precision)
            if row:
                return float(row.get("delta", 0.0)), float(row.get("uncertainty", 0.0)), str(row.get("risk", "low"))
        return 0.0, 0.0, "low"
    row = constants.get("precision_sensitivity", {}).get(component, {}).get(precision)
    if row:
        return float(row.get("delta", 0.0)), float(row.get("uncertainty", 0.0)), str(row.get("risk", "medium"))
    return 0.0, 0.0, "unknown"


def _group_precision(arch: ArchConfig, components: List[str]) -> str:
    values = [arch.get_precision(c) for c in components]
    if any(v in ("fp4", "int4") for v in values):
        return "fp4"
    if any(v in ("fp8", "int8") for v in values):
        return "fp8"
    return values[0] if values else arch.weight_precision


def _precision_residual(
    arch: ArchConfig,
    training: TrainingConfig,
    baseline_loss: float,
    constants: Dict[str, Any],
) -> Tuple[TermResult, Dict[str, PenaltyEntry], List[str]]:
    cfg = constants.get("precision_residual", {})
    hw = training.hardware
    warnings = []
    penalties = {}
    total = 0.0
    uncertainty_sq = 0.0
    fp4_seen = False
    infeasible = False
    notes = []

    # KV cache precision: use the configurable v1 table for modeled quality
    # cost, but keep the legacy table as a hardware-feasibility oracle.
    kv_feasible = kv_quant_penalty(
        training.kv_quantization_bits,
        training.kv_per_channel_scaling,
        hw,
    )
    kv_policy = _kv_bits_to_policy(training.kv_quantization_bits)
    kv_value, kv_unc, kv_risk = _component_table_lookup(constants, "kv_cache", kv_policy)
    if kv_feasible is None:
        kv_value = INFEASIBLE
        infeasible = True
        warnings.append(f"KV quantization at {training.kv_quantization_bits} bits is not supported on {hw}.")
    penalties["kv_quant"] = _make_penalty_entry(
        "kv_quant", kv_value,
        "quality_v1 precision_sensitivity.kv_cache; Hooper et al. (2024) KIVI for feasibility",
        caveat=f"risk={kv_risk}; v1 table is placeholder-calibrated",
        confidence="medium" if kv_value > 0 else "high",
        hardware_dependent=True,
    )
    total += kv_value
    uncertainty_sq += kv_unc ** 2

    # Weight precision: model high-level components once instead of summing
    # every projection independently.
    components = ["ffn_up", "ffn_down", "qkv_proj", "output_proj", "output_head", "embedding"]
    if arch.ffn_type == "swiglu":
        components.append("ffn_gate")

    for comp in components:
        prec = arch.get_precision(comp)
        wp = weight_precision_penalty(comp, prec, hw)
        if wp is None:
            infeasible = True
            warnings.append(f"Weight precision {prec} for {comp} is not available on {hw}.")

    ffn_prec = _group_precision(arch, ["ffn_up", "ffn_down", "ffn_gate"])
    qkv_prec = arch.get_precision("qkv_proj")
    out_prec = arch.get_precision("output_proj")
    head_prec = arch.get_precision("output_head")
    embed_prec = arch.get_precision("embedding")

    weight_total = 0.0
    for component, prec in [
        ("ffn", ffn_prec),
        ("attention_qkv", qkv_prec),
        ("attention_o", out_prec),
        ("lm_head", head_prec),
    ]:
        v, u, risk = _component_table_lookup(constants, component, prec)
        weight_total += v
        uncertainty_sq += u ** 2
        if prec in ("fp4", "int4"):
            fp4_seen = True
        if v > 0:
            notes.append(f"{component}={prec} risk={risk}")

    # Embedding sensitivity is still borrowed from the legacy table until the
    # v1 table has an explicit embedding row.
    emb_legacy = weight_precision_penalty("embedding", embed_prec, hw)
    if emb_legacy is None:
        infeasible = True
        emb_legacy = INFEASIBLE
    weight_total += emb_legacy
    if embed_prec in ("fp4", "int4"):
        fp4_seen = True

    if infeasible:
        weight_total += INFEASIBLE

    penalties["weight_precision"] = _make_penalty_entry(
        "weight_precision", weight_total,
        cfg.get("source", "quality_v1 precision_sensitivity"),
        caveat="FP4 and mixed-precision entries are placeholders until measured per-component sweeps exist" if fp4_seen else "",
        confidence="low" if fp4_seen or infeasible else ("medium" if weight_total >= 0.01 else "high"),
        hardware_dependent=True,
    )
    total += weight_total

    act_prec = arch.activation_precision
    if act_prec not in ("bf16", "fp16"):
        act_attn = activation_precision_penalty("attention", act_prec, hw)
        act_ffn = activation_precision_penalty("ffn", act_prec, hw)
        if act_attn is None or act_ffn is None:
            act_total = INFEASIBLE
            infeasible = True
            warnings.append(f"Activation precision {act_prec} is not available on {hw}.")
        else:
            act_total = act_attn + act_ffn
    else:
        act_total = 0.0

    penalties["activation_precision"] = _make_penalty_entry(
        "activation_precision", act_total,
        "Peng et al. (2023) FP8-LM",
        caveat="Activations recomputed each pass; smaller than weight penalties",
        confidence="medium" if act_total > 0 else "high",
        hardware_dependent=True,
    )
    total += act_total

    confidence = "low" if fp4_seen or infeasible else ("medium" if total >= 0.01 else "high")
    return TermResult(
        name="precision_residual",
        value=total,
        delta=total * baseline_loss,
        uncertainty=max(math.sqrt(uncertainty_sq), abs(total) * float(cfg.get("default_uncertainty", 0.30))),
        confidence=confidence,
        source=cfg.get("source", "precision_residual"),
        notes=notes,
        features={
            "ffn": ffn_prec,
            "attention_qkv": qkv_prec,
            "attention_o": out_prec,
            "lm_head": head_prec,
            "embedding": embed_prec,
            "kv_cache": kv_policy,
        },
    ), penalties, warnings


def _moe_residual(
    arch: ArchConfig,
    training: TrainingConfig,
    baseline_loss: float,
    constants: Dict[str, Any],
) -> TermResult:
    """MoE-residual sub-terms (all relative to baseline L(N_active, D)):

      capacity        — Krajewski sparse-capacity benefit ~ log(N_total/N_active)
      granularity     — small Krajewski bonus when G = n_experts/top_k >> G_ref
      top_k1          — Switch-style top_k=1 penalty (~1-2% PPL)
      router_fp8      — penalty when router precision is fp8 (instability)
      shared_experts  — DeepSeek-V2 ablation bonus when shared expert present
      routing         — applied only if load_balance prior < 1 (default 1.0)

    The shape of arch.moe_config accepted here is the canonical v1 nested form
    (shared_expert: {ffn_dim, precision}, router: {precision, ...}). Legacy
    flat keys (shared_dim, router_precision, shared_expert_ratio) are also
    honored for backward compat.
    """
    cfg = constants.get("moe_residual", {})
    enabled = arch._moe_enabled
    if not enabled:
        return TermResult(name="moe_residual", confidence="not_applicable", source="dense_fallback")

    moe = arch.moe_config or {}
    n_active = max(1, arch.n_active_params)
    n_total = max(n_active, arch.n_total_params)
    n_experts = max(1, int(moe.get("n_experts", 1)))
    top_k = max(1, int(moe.get("top_k", 1)))
    active_ratio = n_active / n_total
    expert_activation_ratio = top_k / n_experts

    # shared_ratio: prefer explicit scalar; otherwise derive from nested
    # shared_expert.ffn_dim / dense FFN baseline.
    shared_block = moe.get("shared_expert")
    if isinstance(shared_block, dict):
        shared_dim = int(shared_block.get("ffn_dim", 0) or 0)
    else:
        shared_dim = int(moe.get("shared_dim", 0) or 0)
    has_shared = shared_dim > 0
    if "shared_expert_ratio" in moe:
        shared_ratio = float(moe.get("shared_expert_ratio") or 0.0)
    elif arch.ffn_dim > 0:
        shared_ratio = shared_dim / max(1, arch.ffn_dim)
    else:
        shared_ratio = 0.0

    # Router precision: nested router.precision wins, falls back to legacy flat.
    router_block = moe.get("router")
    if isinstance(router_block, dict):
        router_precision = router_block.get("precision", "bf16")
    else:
        router_precision = moe.get("router_precision", "bf16")

    load_balance = float(moe.get("load_balance_prior", 1.0))
    granularity = float(moe.get("granularity", n_experts / max(1, top_k)))
    ref_g = float(cfg.get("reference_granularity", 8.0))
    weights = cfg.get("weights", {})

    # --- sub-terms ---
    # v1-fix Part B: pro-rate the MoE capacity benefit by the MoE-layer
    # fraction. A model with n_dense dense FFN layers and (n_layers - n_dense)
    # MoE layers gets a fraction (n_moe / n_layers) of the full sparse-
    # capacity bonus, because only the MoE layers add total capacity beyond
    # the active path.
    n_dense_prefix = max(0, min(int(getattr(arch, "n_dense_ffn_layers", 0)), arch.n_layers))
    n_moe_layers = max(0, arch.n_layers - n_dense_prefix)
    moe_layer_fraction = n_moe_layers / max(1, arch.n_layers)

    # Capacity bonus: negative residual (lower loss) when total >> active.
    # Use a soft saturation rather than a hard cap so that expert-count edits
    # (e.g., change_moe_topology halving experts) still produce a non-zero
    # change in the capacity term; Switch (top_k=1) is independently penalised
    # by top_k1_penalty so it does not need the cap to discount its bonus.
    cap_ratio = float(cfg.get("capacity_ratio_cap", 4.0))
    raw_ratio = max(1.0, n_total / n_active)
    # softplus-like saturation: matches log(raw) for raw <= cap, slows above.
    if raw_ratio <= cap_ratio:
        effective_log = math.log(raw_ratio)
    else:
        effective_log = math.log(cap_ratio) + 0.5 * math.log(raw_ratio / cap_ratio)
    capacity_bonus = (
        -float(weights.get("capacity", 0.0))
        * effective_log
        * moe_layer_fraction
    )
    # Granularity: negative weight × positive log-ratio = bonus above ref_g;
    # capped at 0 when granularity <= ref_g (no benefit for coarse).
    granularity_bonus = float(weights.get("granularity", 0.0)) * max(
        0.0, math.log(max(1.0, granularity / ref_g))
    )
    # Routing imbalance (>0 only if load_balance<1).
    routing_penalty = float(weights.get("routing", 0.0)) * max(0.0, 1.0 - load_balance)
    # Shared expert bonus (already negative weight × positive ratio = bonus).
    shared_adjust = float(weights.get("shared_experts", 0.0)) * shared_ratio if has_shared else 0.0
    # Switch-style top_k=1 penalty (positive residual).
    top_k1_penalty = float(weights.get("top_k1", 0.0)) if top_k == 1 else 0.0
    # Router precision penalty (positive residual when router is fp8).
    router_penalty = float(weights.get("router_fp8", 0.0)) if router_precision == "fp8" else 0.0

    # v1-fix Part B: DeepSeek-V3 / Qwen3-MoE stability bonus for a small
    # dense prefix. Linear up to preferred_dense_prefix, flat beyond.
    preferred_prefix = int(cfg.get("preferred_dense_prefix", 3))
    dense_prefix_bonus = (
        float(weights.get("dense_prefix", 0.0))
        * min(n_dense_prefix, preferred_prefix)
    )

    value = (
        capacity_bonus
        + granularity_bonus
        + routing_penalty
        + shared_adjust
        + top_k1_penalty
        + router_penalty
        + dense_prefix_bonus
    )

    notes = [
        "MoE residual: Krajewski sparse-capacity bonus + small top_k/router/shared sub-terms; "
        "still high-uncertainty without measured ablation profiles."
    ]
    if n_experts >= 128 and "load_balance_prior" not in moe:
        notes.append("Very high expert count without measured load-balance prior.")
    if expert_activation_ratio < 0.02:
        notes.append("Very sparse expert activation ratio; routing quality is uncertain.")
    if top_k == 1:
        notes.append("top_k=1 (Switch-style) routing applied a small quality penalty vs top_k>=2.")
    if router_precision == "fp8":
        notes.append("FP8 router precision applied a small instability penalty.")
    if has_shared:
        notes.append(f"Shared expert (ffn_dim={shared_dim}) treated as a small quality bonus.")

    sub_features = {
        "capacity_bonus": round(capacity_bonus, 6),
        "granularity_bonus": round(granularity_bonus, 6),
        "routing_penalty": round(routing_penalty, 6),
        "shared_expert_adjust": round(shared_adjust, 6),
        "top_k1_penalty": round(top_k1_penalty, 6),
        "router_fp8_penalty": round(router_penalty, 6),
        "dense_prefix_bonus": round(dense_prefix_bonus, 6),
        "moe_layer_fraction": round(moe_layer_fraction, 4),
        "n_dense_ffn_layers": n_dense_prefix,
    }

    return TermResult(
        name="moe_residual",
        value=value,
        delta=value * baseline_loss,
        uncertainty=max(float(cfg.get("uncertainty", 0.40)) * max(abs(value), 0.01), 0.0),
        confidence="low",
        source=cfg.get("source", "moe_residual"),
        notes=notes,
        features={
            "N_active": n_active,
            "N_total": n_total,
            "n_experts": n_experts,
            "top_k": top_k,
            "active_ratio": round(active_ratio, 6),
            "expert_activation_ratio": round(expert_activation_ratio, 6),
            "shared_expert_ratio": round(shared_ratio, 6),
            "has_shared_expert": has_shared,
            "router_precision": router_precision,
            "load_balance_prior": load_balance,
            "granularity": granularity,
            "subterms": sub_features,
        },
    )


def _resolve_hybrid_family(state: dict) -> str:
    """Map state_type to a hybrid family key.

    Families:
      mamba_sequential          — Mamba-1/2, S4/S5/S6 sequential SSM
      gated_delta_or_kda_linear — DeltaNet, Gated DeltaNet, KDA, GLA
                                  (delta-rule + gating)
      generic_linear_attention  — RWKV-7, RetNet, generic kernel attention
      parallel_hybrid_heads     — MoH, Hydra, channel-split attention/state
      recurrent_local_attention — Sliding Window Attention (SWA),
                                  local/sliding-window + recurrent state

    v1-fix Part J: added aliases for the four families the user called out
    (Sliding Window Attention, Mamba/Mamba-2, Gated DeltaNet / DeltaNet,
    Kimi Delta Attention). `deltanet`, `gated_deltanet`, and
    `sliding_window` are new spellings; older spellings still resolve.
    """
    st = str(state.get("state_type", state.get("type", "mamba2"))).lower()
    pattern = str(state.get("pattern", state.get("hybrid_pattern", "sequential"))).lower()

    if st in ("mamba2", "mamba", "mamba1", "s4", "s5", "s6"):
        return "mamba_sequential"
    elif st in (
        "gated_delta", "gated_deltanet", "kda",
        "delta_net", "deltanet",
        "gla", "gated_linear_attention",
    ):
        # GLA (Yang et al. 2024) is grouped with the gated delta family
        # because both use delta-rule-style updates with per-head gating.
        return "gated_delta_or_kda_linear"
    elif st in ("linear_attention", "rwkv", "rwkv7", "retnet"):
        return "generic_linear_attention"
    elif st in ("parallel_heads", "moh", "hydra") or pattern == "parallel":
        return "parallel_hybrid_heads"
    elif st in ("sliding_window", "swa", "local_recurrent") or pattern in ("local_global", "local"):
        return "recurrent_local_attention"
    else:
        return "mamba_sequential"  # default


def _compute_p_attn(state: dict, arch: "ArchConfig") -> Tuple[float, str]:
    """Compute normalized attention fraction p_attn and its type.

    For sequential hybrids:
      p_attn = L_attn / (L_attn + L_state)

    For parallel hybrids:
      p_attn = d_attention_mixer / (d_attention_mixer + d_state_mixer)

    Returns (p_attn, ratio_type).
    """
    pattern = str(state.get("pattern", state.get("hybrid_pattern", "sequential"))).lower()
    state_layers = int(state.get("state_layers", 0))
    attention_layers = int(state.get("attention_layers", max(0, arch.n_layers - state_layers)))

    if state_layers <= 0 and arch.model_type in ("state", "hybrid"):
        state_layers = max(1, arch.n_layers // 2)
        attention_layers = arch.n_layers - state_layers

    n_total = max(1, attention_layers + state_layers)

    if pattern == "parallel":
        # Parallel: ratio by dimension
        d_attn = int(state.get("d_attention_mixer", arch.d_model))
        d_state = int(state.get("d_state_mixer", arch.d_model))
        p_attn = d_attn / max(1, d_attn + d_state)
        return p_attn, "parallel"
    elif pattern in ("local_global", "local"):
        # Local/global: ratio by global attention layers
        global_layers = attention_layers
        local_layers = state_layers
        p_global = global_layers / max(1, global_layers + local_layers)
        return p_global, "local_global"
    else:
        # Sequential (default): ratio by layer count
        p_attn = attention_layers / n_total
        return p_attn, "sequential"


def _compute_effective_kv_bytes(
    state: dict, arch: "ArchConfig", context: int,
) -> float:
    """Compute effective KV memory in bytes layer-by-layer.

    KV_bytes = sum_l 2 * bpe * n_kv_heads(l) * d_head(l) * T_eff(l)

    State layers contribute 0 KV bytes. Attention layers contribute full KV.
    Local attention layers use min(context, window_size).
    KV sharing and MLA compression reduce further.
    """
    bpe = 2  # bf16 default
    state_layers = int(state.get("state_layers", 0))
    attention_layers = int(state.get("attention_layers", max(0, arch.n_layers - state_layers)))
    if state_layers <= 0 and arch.model_type in ("state", "hybrid"):
        state_layers = max(1, arch.n_layers // 2)
        attention_layers = arch.n_layers - state_layers

    local_window = arch.local_window
    kv_share = float(state.get("kv_share_factor", 1.0))
    mla_latent = arch.mla_latent_dim
    # v1-fix MLA: the rope split also lives in the KV cache (the q-side RoPE
    # term shares the cached K projection). Add d_rope when set.
    mla_rope = arch.mla_rope_head_dim or 0

    kv_bytes = 0.0
    for _ in range(attention_layers):
        t_eff = context
        if local_window and local_window > 0:
            t_eff = min(context, local_window)
        if kv_share > 1.0:
            t_eff = t_eff / kv_share

        if mla_latent and mla_latent > 0:
            # v1-fix MLA: per-token per-layer KV cache stores ONE compressed
            # latent (c_kv) plus the RoPE'd key (d_rope), shared across all
            # query heads. Not 2× — the K/V down-projection produces a
            # single shared latent (DeepSeek-V2 §3.1 eq. 11–13).
            kv_bytes += bpe * (mla_latent + mla_rope) * t_eff
        else:
            kv_bytes += 2 * bpe * arch.n_kv_heads * arch.d_head * t_eff

    return kv_bytes


def _state_residual(
    arch: ArchConfig,
    workload_spec: Optional[Dict[str, Any]],
    baseline_loss: float,
    constants: Dict[str, Any],
) -> TermResult:
    """State/hybrid quality residual — family-specific 5-term decomposition.

    f_state_or_memory(A, H) =
        f_hybrid_ratio(A, H)       — band-pass penalty on attention fraction
      + f_state_capacity(A)         — compression + composition cost
      + f_effective_kv_cost(A, H)   — per-layer KV memory cost signal
      + f_recall_risk(A, H)         — recall-specific risk from low attention
      + f_family_uncertainty(A)     — extra uncertainty outside supported band

    The hybrid ratio is modeled as a family-specific band-pass prior.
    Inside the preferred band, quality penalty is small and flat.
    """
    cfg = constants.get("state_residual", {})
    state = arch.state_config or {}
    enabled = bool(state.get("enabled", False)) or arch.model_type in ("state", "hybrid")
    if not enabled:
        return TermResult(name="state_residual", confidence="not_applicable", source="disabled")

    workload = workload_spec or {}
    context = int(workload.get("context_length", 8192))
    task_type = str(workload.get("task_type", "general"))

    # --- Resolve family and ratio ---
    hybrid_family = _resolve_hybrid_family(state)
    families_cfg = cfg.get("families", {})
    family_cfg = families_cfg.get(hybrid_family, families_cfg.get("mamba_sequential", {}))

    p_attn, ratio_type = _compute_p_attn(state, arch)

    state_layers = int(state.get("state_layers", 0))
    attention_layers = int(state.get("attention_layers", max(0, arch.n_layers - state_layers)))
    if state_layers <= 0 and arch.model_type in ("state", "hybrid"):
        state_layers = max(1, arch.n_layers // 2)
        attention_layers = arch.n_layers - state_layers
    d_state = int(state.get("d_state", arch.d_model))

    p_low = float(family_cfg.get("p_low", 0.12))
    p_high = float(family_cfg.get("p_high", 0.28))
    w_under = float(cfg.get("w_under", 0.08))
    w_over = float(cfg.get("w_over", 0.04))
    w_recall = float(cfg.get("w_recall", 0.06))
    w_capacity = float(cfg.get("w_capacity", 0.02))
    w_kv_cost = float(cfg.get("w_kv_cost", 0.01))
    compression_scale = float(cfg.get("compression_scale", 0.02))
    composition_scale = float(cfg.get("composition_scale", 0.01))
    d_state_ref = float(cfg.get("d_state_reference", 192))
    refresh_weight = float(cfg.get("refresh_factor_weight", 0.5))

    # =========================================================================
    # Term 1: f_hybrid_ratio — band-pass penalty
    # =========================================================================
    under_band = max(0.0, p_low - p_attn)
    over_band = max(0.0, p_attn - p_high)
    f_hybrid_ratio = w_under * (under_band ** 2) + w_over * (over_band ** 2)

    # =========================================================================
    # Term 2: f_state_capacity — compression + composition
    # =========================================================================
    n_total = max(1, attention_layers + state_layers)
    state_fraction = state_layers / n_total

    # Compression: effective_memory_horizon = d_state * n_state / context
    effective_memory_horizon = (d_state * state_layers) / max(1, context)
    if effective_memory_horizon >= 1.0:
        compression = 0.0
    else:
        compression = compression_scale * (1.0 / effective_memory_horizon - 1.0)

    # Attention-refresh modulation
    log2_ctx = math.log2(max(2.0, context))
    refresh_factor = min(1.0, attention_layers / log2_ctx)
    compression *= (1.0 - refresh_weight * refresh_factor)

    # Composition: small mixing tax
    composition = composition_scale * state_fraction
    composition *= (d_state_ref / max(d_state, 64))
    composition *= (context / 4096.0)

    f_state_capacity = w_capacity * (compression + composition)

    # =========================================================================
    # Term 3: f_effective_kv_cost — KV memory signal (lower = better for serving)
    # =========================================================================
    # Normalized KV cost relative to full-attention baseline
    full_attn_kv = 2 * 2 * arch.n_kv_heads * arch.d_head * context * arch.n_layers
    actual_kv = _compute_effective_kv_bytes(state, arch, context)
    kv_ratio = actual_kv / max(1.0, full_attn_kv)
    # KV cost is a serving benefit signal — lower p_attn reduces KV, which is
    # good for serving but we don't reward it in quality. We only penalize if
    # KV cost is unexpectedly high (i.e. kv_ratio > expected from p_attn).
    expected_kv_ratio = p_attn  # rough expected
    kv_excess = max(0.0, kv_ratio - expected_kv_ratio)
    f_kv_cost = w_kv_cost * kv_excess

    # =========================================================================
    # Term 4: f_recall_risk — separate recall penalty
    # =========================================================================
    if task_type in ("recall_intensive", "long_context_retrieval", "exact_copy", "coding_agent"):
        p_recall_min = float(family_cfg.get("p_recall_min_recall", 0.25))
    else:
        p_recall_min = float(family_cfg.get("p_recall_min_general", 0.14))

    recall_gap = max(0.0, p_recall_min - p_attn)
    f_recall_risk = w_recall * (recall_gap ** 2)

    # =========================================================================
    # Term 5: f_family_uncertainty — extra uncertainty outside supported band
    # =========================================================================
    in_band = p_low <= p_attn <= p_high
    family_confidence = str(family_cfg.get("confidence", "medium"))

    # --- Total ---
    raw = f_hybrid_ratio + f_state_capacity + f_kv_cost + f_recall_risk
    min_delta = float(cfg.get("min_delta", -0.05))
    max_delta = float(cfg.get("max_delta", 0.30))
    value = max(min_delta, min(max_delta, raw))

    # --- Confidence ---
    if not in_band:
        confidence = "low"
    elif family_confidence in ("medium-high", "high"):
        confidence = "medium-high" if attention_layers > 0 else "medium"
    elif family_confidence == "medium-low":
        confidence = "medium-low"
    else:
        confidence = "medium" if attention_layers > 0 else "low"

    # --- Uncertainty ---
    base_unc = float(cfg.get("uncertainty", 0.40))
    outside_unc = float(cfg.get("uncertainty_outside_band", 0.55))
    unc_scale = outside_unc if not in_band else base_unc
    uncertainty = max(unc_scale * max(abs(value), 0.01), 0.0)

    notes = [
        f"Hybrid ratio modeled as {hybrid_family} family band-pass prior "
        f"(p_attn={p_attn:.3f}, band=[{p_low:.2f}, {p_high:.2f}]).",
        f"5-term decomposition: ratio={f_hybrid_ratio:.4f} capacity={f_state_capacity:.4f} "
        f"kv_cost={f_kv_cost:.4f} recall={f_recall_risk:.4f}.",
    ]
    if not in_band:
        notes.append(f"p_attn={p_attn:.3f} is outside preferred band — increased uncertainty.")

    return TermResult(
        name="state_residual",
        value=value,
        delta=value * baseline_loss,
        uncertainty=uncertainty,
        confidence=confidence,
        source=cfg.get("source", "v2_family_specific_hybrid_ratio"),
        notes=notes,
        features={
            "hybrid_family": hybrid_family,
            "p_attn": round(p_attn, 4),
            "ratio_type": ratio_type,
            "p_low": p_low,
            "p_high": p_high,
            "in_band": in_band,
            "d_state": d_state,
            "state_layers": state_layers,
            "attention_layers": attention_layers,
            "state_fraction": round(state_fraction, 4),
            "context_length": context,
            "effective_memory_horizon": round(effective_memory_horizon, 6),
            "f_hybrid_ratio": round(f_hybrid_ratio, 6),
            "f_state_capacity": round(f_state_capacity, 6),
            "f_kv_cost": round(f_kv_cost, 6),
            "f_recall_risk": round(f_recall_risk, 6),
            "kv_ratio": round(kv_ratio, 4),
            "task_type": task_type,
            "family_confidence": family_confidence,
        },
    )


def _risk_residual(
    arch: ArchConfig,
    architecture_term: TermResult,
    baseline_loss: float,
    constants: Dict[str, Any],
) -> TermResult:
    cfg = constants.get("risk_residual", {})
    if not cfg.get("enabled", True):
        return TermResult(name="risk_residual", confidence="not_applicable", source="disabled")

    features = architecture_term.features
    sub = features.get("subterms", {})
    gqa_group = float(features.get("gqa_group_size", 1.0))
    d_head = int(features.get("d_head", arch.d_head))
    bottleneck = bool(features.get("attention_bottleneck_risk", False))
    acceptable = set(int(x) for x in constants.get("architecture_residual", {}).get("acceptable_d_head", [64, 128, 256]))

    notes = []
    uncertainty = 0.0
    # v1-fix demo-audit-2 (Jun 2026): threshold raised from >=8 to >16.
    # Llama-3-70B / Mistral-Large / Qwen-3 all ship with GQA group_size=8
    # (n_q_heads=64, n_kv_heads=8) and group_size=8 is the production
    # standard, not "aggressive". The previous threshold added a 2pp
    # uncertainty penalty to every modern production-shaped 7B/70B,
    # which then leaked into the optimizer as risk_uncertainty_pct=2%
    # on B200 1B even though the chosen config was a normal Llama-3
    # GQA-8 shape. >16 is true MQA-territory (group_size=16 is
    # Falcon-style, >=32 starts approaching pure MQA).
    if gqa_group > 16:
        notes.append("Aggressive GQA sharing (group_size>16) is modeled as uncertainty, not as a precise standalone scaling law.")
        uncertainty += float(cfg.get("uncertainty", 0.02))
    if bottleneck:
        notes.append("Width/head coupling indicates possible attention bottleneck risk.")
        uncertainty += float(cfg.get("uncertainty", 0.02))
    if d_head not in acceptable:
        notes.append("Head dimension is outside the conventional layout set.")
        uncertainty += float(cfg.get("uncertainty", 0.02)) * 0.5

    # f_risk is uncertainty-only in this uncalibrated v1 layer. Architecture
    # risk priors that affect the point estimate live in architecture_residual
    # to avoid double-counting the same width/head coupling.
    value = 0.0
    confidence = "low" if notes else "high"
    return TermResult(
        name="risk_residual",
        value=value,
        delta=value * baseline_loss,
        uncertainty=uncertainty,
        confidence=confidence,
        source=cfg.get("source", "compiler_attention_and_architecture_risk_prior"),
        notes=notes,
        features={
            "gqa_group_size": gqa_group,
            "attention_bottleneck_risk": bottleneck,
            "d_head": d_head,
        },
    )


def _apply_downstream_head(result: QualityResult, constants: Dict[str, Any], workload_spec: Optional[Dict[str, Any]]) -> None:
    cfg = constants.get("downstream_head", {})
    if not cfg.get("enabled", False):
        return
    alpha = float(cfg.get("alpha", 0.0))
    beta = float(cfg.get("beta", 1.0))
    gamma = float(cfg.get("gamma", 0.0))
    context = float((workload_spec or {}).get("context_length", 8192))
    if cfg.get("input_mode", "loss_proxy") == "compute":
        x = math.log(max(1.0, result.n_active_params * result.training_tokens))
    else:
        x = -result.loss_proxy
    correction = math.log(max(1.0, context))
    z = alpha + beta * x + gamma * correction
    result.benchmark_score_proxy = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))
    result.benchmark_uncertainty = float(cfg.get("uncertainty", 0.80))
    result.benchmark_notes.append("Downstream prediction is infrastructure only; do not treat it as benchmark accuracy.")


def estimate_quality(
    config: Any,
    training_spec: Any,
    workload_spec: Optional[Dict[str, Any]] = None,
    constants: Optional[Dict[str, Any]] = None,
    memory_fits: bool = True,
    lattice_aligned: bool = True,
) -> QualityResult:
    """Return absolute quality proxy and uncertainty for an architecture."""
    arch = _coerce_arch_config(config)
    training = _coerce_training_config(training_spec)
    constants = _deep_merge(load_quality_constants(), constants or {})

    d_eff, data_term, dq_warnings = _data_quality_filter(training, constants)
    n_active = arch.n_active_params
    n_total = arch.n_total_params
    n_spine = arch.n_active_non_embedding_params
    L_base = _chinchilla_loss_with_constants(n_spine, d_eff, constants)
    in_regime, regime_notes = is_in_chinchilla_regime(n_spine, d_eff)
    spine_unc = float(constants.get("spine", {}).get(
        "uncertainty_in_regime" if in_regime else "uncertainty_out_of_regime",
        0.03 if in_regime else 0.08,
    ))

    result = QualityResult(
        n_active_params=n_active,
        n_total_params=n_total,
        spine_active_params=n_spine,
        training_tokens=d_eff,
        chinchilla_baseline=L_base,
        in_chinchilla_regime=in_regime,
    )
    result.confidence_notes.extend(regime_notes)
    result.warnings.extend(dq_warnings)

    result.terms["spine"] = TermResult(
        name="spine",
        value=0.0,
        delta=L_base,
        uncertainty=spine_unc,
        confidence="medium" if not in_regime else "high",
        source=constants.get("spine", {}).get("source", "spine"),
        notes=["Scaling-law spine over active non-embedding compute proxy and effective training tokens."],
        features={
            "N_active": n_active,
            "M_active_non_embedding_proxy": n_spine,
            "N_total": n_total,
            "D_effective": d_eff,
            "tokens_per_active_param": d_eff / max(1, n_spine),
            "L_inf": constants.get("spine", {}).get("E", CHINCHILLA_E),
        },
    )
    result.terms["data_quality"] = data_term

    if training.overtraining_ratio is not None:
        result.terms["spine"].features["overtraining_ratio"] = training.overtraining_ratio
        if training.overtraining_ratio > 4:
            result.confidence_notes.append(
                "Candidate is heavily overtrained relative to the base scaling prior; uncertainty widened."
            )
            result.terms["spine"].uncertainty = max(result.terms["spine"].uncertainty, spine_unc * 1.5)

    terms_for_residual = []
    penalties: Dict[str, PenaltyEntry] = {}

    arch_term, arch_penalties = _architecture_residual(
        arch, n_spine, L_base, constants,
        workload_spec=workload_spec,
    )
    result.terms[arch_term.name] = arch_term
    terms_for_residual.append(arch_term)
    penalties.update(arch_penalties)

    precision_term, precision_penalties, precision_warnings = _precision_residual(arch, training, L_base, constants)
    result.terms[precision_term.name] = precision_term
    terms_for_residual.append(precision_term)
    penalties.update(precision_penalties)
    result.warnings.extend(precision_warnings)
    result.confidence_notes.extend(precision_warnings)

    moe_term = _moe_residual(arch, training, L_base, constants)
    result.terms[moe_term.name] = moe_term
    if moe_term.confidence != "not_applicable":
        terms_for_residual.append(moe_term)
        penalties["moe"] = _make_penalty_entry(
            "moe", moe_term.value, moe_term.source,
            caveat="High-uncertainty sparse-capacity residual",
            confidence=moe_term.confidence,
        )

    state_term = _state_residual(arch, workload_spec, L_base, constants)
    result.terms[state_term.name] = state_term
    if state_term.confidence != "not_applicable":
        terms_for_residual.append(state_term)
        penalties["state"] = _make_penalty_entry(
            "state", state_term.value, state_term.source,
            caveat="High-uncertainty state/hybrid residual",
            confidence=state_term.confidence,
        )

    large_shape_term = _large_shape_stability_prior(arch, n_total, L_base)
    result.terms[large_shape_term.name] = large_shape_term
    if large_shape_term.confidence != "not_applicable":
        terms_for_residual.append(large_shape_term)
        penalties["large_shape"] = _make_penalty_entry(
            "large_shape", large_shape_term.value, large_shape_term.source,
            caveat="Coarse frontier-scale depth/width stability prior",
            confidence=large_shape_term.confidence,
        )
        result.warnings.append("Large-model depth/width shape is outside the stability prior band.")

    # v1-fix MTP: Multi-Token Prediction sample-efficiency bonus during
    # training (DeepSeek-V3 §2.2). Modeled as a small negative residual
    # proportional to the n_predict_depths times the training loss weight.
    # DeepSeek-V3 reports ~0.5-1.5% effective perplexity improvement on
    # downstream eval when k=1. The bonus saturates at k=2.
    if arch.mtp_n_predict_depths > 0 and arch.mtp_train_loss_weight > 0:
        mtp_cfg = constants.get("mtp_residual", {})
        bonus_per_depth = float(mtp_cfg.get("bonus_per_depth", 0.006))
        depth_saturation_cap = int(mtp_cfg.get("depth_saturation_cap", 2))
        effective_depth = min(int(arch.mtp_n_predict_depths), depth_saturation_cap)
        mtp_bonus = -bonus_per_depth * effective_depth * float(arch.mtp_train_loss_weight) / 0.3
        mtp_term = TermResult(
            name="mtp_residual",
            value=mtp_bonus,
            delta=mtp_bonus * L_base,
            uncertainty=abs(mtp_bonus) * 0.4,
            confidence="medium",
            source="deepseek_v3_mtp_sample_efficiency",
            notes=[
                f"MTP n_predict_depths={arch.mtp_n_predict_depths} "
                f"(effective={effective_depth}), train_loss_weight={arch.mtp_train_loss_weight}",
                "Training-time only; inference drops MTP heads (or uses for speculative decode).",
            ],
            features={
                "n_predict_depths": arch.mtp_n_predict_depths,
                "effective_depth": effective_depth,
                "train_loss_weight": arch.mtp_train_loss_weight,
                "bonus": round(mtp_bonus, 6),
            },
        )
        result.terms["mtp_residual"] = mtp_term
        terms_for_residual.append(mtp_term)
        penalties["mtp"] = _make_penalty_entry(
            "mtp", mtp_bonus, mtp_term.source,
            caveat="Sample-efficiency bonus from MTP heads at training time",
            confidence="medium",
        )

    risk_term = _risk_residual(arch, arch_term, L_base, constants)
    result.terms[risk_term.name] = risk_term
    if risk_term.confidence != "not_applicable":
        terms_for_residual.append(risk_term)
        penalties["risk"] = _make_penalty_entry(
            "risk", risk_term.value, risk_term.source,
            caveat="Uncertainty-only risk hook for aggressive GQA, attention bottlenecks, and unconventional head dimensions",
            confidence=risk_term.confidence,
        )

    fp = feasibility_penalty(arch.d_head, memory_fits, lattice_aligned)
    feasibility_term = TermResult(
        name="feasibility",
        value=fp,
        delta=fp * L_base,
        uncertainty=0.0,
        confidence="high",
        source="hard_constraint",
    )
    result.terms["feasibility"] = feasibility_term
    terms_for_residual.append(feasibility_term)
    penalties["feasibility"] = _make_penalty_entry("feasibility", fp, "Hard constraint")

    total_frac = sum(t.value for t in terms_for_residual)
    result.total_penalty_fraction = total_frac
    result.total_penalty_absolute = total_frac * L_base
    result.predicted_loss = L_base * (1 + total_frac)
    result.loss_proxy = result.predicted_loss
    result.penalty_breakdown = penalties

    real_penalties = {k: v for k, v in penalties.items() if v.value > 0 and v.value < INFEASIBLE}
    if real_penalties:
        result.dominant_penalty = max(real_penalties, key=lambda k: real_penalties[k].value)
    elif any(v.value >= INFEASIBLE for v in penalties.values()):
        result.dominant_penalty = "feasibility"
    else:
        result.dominant_penalty = "none"

    large_penalties = [k for k, v in penalties.items() if 0.01 <= v.value < INFEASIBLE]
    low_conf = any(
        t.confidence == "low" and abs(t.value) > 0
        for t in terms_for_residual
    ) or any(p.confidence == "low" for p in penalties.values() if p.value > 0)
    if any(v.value >= INFEASIBLE for v in penalties.values()):
        result.confidence = "low"
    elif len(large_penalties) >= 3:
        result.confidence = "low"
        result.confidence_notes.append(
            f"Stacking {len(large_penalties)} residuals >=1% each "
            f"({', '.join(large_penalties)}); additive composition is least reliable here."
        )
    elif len(large_penalties) >= 2 or not in_regime or low_conf:
        result.confidence = "medium"
        if low_conf:
            result.confidence_notes.append("Contains low-confidence residual values.")
    else:
        result.confidence = "high"

    uncertainty_breakdown = {
        "scale_law": result.terms["spine"].uncertainty,
        "architecture": _term_uncertainty(arch_term),
        "precision": _term_uncertainty(precision_term),
        "moe": _term_uncertainty(moe_term),
        "state": _term_uncertainty(state_term),
        "large_shape": _term_uncertainty(large_shape_term),
        "risk": _term_uncertainty(risk_term),
        "data_quality": _term_uncertainty(data_term),
    }
    max_unc = float(constants.get("uncertainty", {}).get("max_uncertainty", 0.95))
    uncertainty_cfg = constants.get("uncertainty", {})
    calibration_multiplier = max(
        0.0, float(uncertainty_cfg.get("calibration_multiplier", 1.0)))
    uncertainty_breakdown = {
        k: v * calibration_multiplier for k, v in uncertainty_breakdown.items()
    }
    result.uncertainty_breakdown = uncertainty_breakdown
    result.uncertainty_total = min(max_unc, math.sqrt(sum(v * v for v in uncertainty_breakdown.values())))

    residual_std = math.sqrt(sum(
        _term_uncertainty(t) ** 2 for t in terms_for_residual
        if t.value < INFEASIBLE
    )) * calibration_multiplier
    interaction_bonus = float(constants.get("uncertainty", {}).get("additive_interaction_bonus", 0.005))
    additive_bonus = interaction_bonus * max(0, len(large_penalties) - 1)
    result.uncertainty_low_pct = round(max(0, total_frac - residual_std - additive_bonus) * 100, 2)
    result.uncertainty_high_pct = round((total_frac + residual_std + additive_bonus) * 100, 2)
    if calibration_multiplier != 1.0:
        source = uncertainty_cfg.get("calibration_source", "lab_auto_calibration")
        result.confidence_notes.append(
            f"Uncertainty scaled by lab calibration multiplier "
            f"{calibration_multiplier:.3f} ({source})."
        )

    _apply_downstream_head(result, constants, workload_spec)
    _apply_lab_calibration_notes(result, constants, arch, training)
    _apply_eval_models(result, constants, arch, training)
    return result


def quality(
    arch: ArchConfig,
    training: TrainingConfig,
    memory_fits: bool = True,
    lattice_aligned: bool = True,
) -> QualityResult:
    """Backward-compatible architecture-solver entry point."""
    return estimate_quality(
        arch,
        training,
        workload_spec={"context_length": training.sequence_length},
        memory_fits=memory_fits,
        lattice_aligned=lattice_aligned,
    )


def compare_quality(
    candidate_config: Any,
    baseline_config: Any,
    training_spec: Any,
    workload_spec: Optional[Dict[str, Any]] = None,
    constants: Optional[Dict[str, Any]] = None,
) -> QualityComparison:
    """Return candidate-vs-baseline quality delta and relative uncertainty."""
    candidate = estimate_quality(candidate_config, training_spec, workload_spec, constants)
    baseline = estimate_quality(baseline_config, training_spec, workload_spec, constants)
    delta_abs = candidate.loss_proxy - baseline.loss_proxy
    delta_pct = 100 * delta_abs / max(1e-12, baseline.loss_proxy)

    cand_arch = _coerce_arch_config(candidate_config)
    base_arch = _coerce_arch_config(baseline_config)
    same_family = cand_arch.model_type == base_arch.model_type
    shared_multiplier = 0.5 if same_family else 1.0
    rel_unc = math.sqrt(candidate.uncertainty_total ** 2 + baseline.uncertainty_total ** 2) * shared_multiplier
    unc_pct = rel_unc * 100
    if abs(delta_pct) <= unc_pct:
        interpretation = "quality-equivalent within modeled uncertainty"
    elif delta_pct < 0:
        interpretation = "candidate improves loss proxy"
    else:
        interpretation = "candidate worsens loss proxy"
    return QualityComparison(
        candidate=candidate,
        baseline=baseline,
        delta_abs=delta_abs,
        delta_pct=delta_pct,
        uncertainty_low_pct=delta_pct - unc_pct,
        uncertainty_high_pct=delta_pct + unc_pct,
        interpretation=interpretation,
    )


def explain_quality(result: QualityResult) -> str:
    """Return markdown-ready explanation of all quality terms."""
    lines = [
        "## Quality Proxy",
        "",
        "The quality proxy uses a scaling-law spine over active non-embedding compute and training tokens, plus modular residuals for coupled architecture variables, precision, MoE, state/memory mechanisms, risk, and data quality.",
        "",
        "`L_quality(A,D,H,P) = L_spine(M_active_non_embedding(A),D) + f_width_depth + f_mlp_attention_ratio + f_attention_heads + f_precision + f_moe + f_state_or_memory + f_risk`",
        "",
        "Query heads are treated as a weak width-derived prior, not as a monotonic quality law. KV heads are treated as a direct memory/latency tradeoff with uncertain GQA-sharing quality risk.",
        "",
        f"- Loss proxy: {result.loss_proxy:.4f}",
        f"- Spine baseline: {result.chinchilla_baseline:.4f}",
        f"- Spine active proxy: {result.spine_active_params/1e9:.3f}B active non-embedding params",
        f"- Residual total: {result.total_penalty_fraction * 100:.2f}%",
        f"- Uncertainty total: ±{result.uncertainty_total * 100:.2f}%",
        f"- Confidence: {result.confidence}",
        "",
        "| Term | Residual | Uncertainty | Confidence | Source |",
        "|---|---:|---:|---|---|",
    ]
    for term in result.terms.values():
        lines.append(
            f"| {term.name} | {term.value * 100:.3f}% | "
            f"{term.uncertainty * 100:.3f}% | {term.confidence} | {term.source} |"
        )
    notes = result.confidence_notes + result.warnings
    if notes:
        lines.append("")
        lines.append("### Notes")
        for note in dict.fromkeys(notes):
            lines.append(f"- {note}")
    return "\n".join(lines)


def _coerce_arch_config(config: Any) -> ArchConfig:
    if isinstance(config, ArchConfig):
        return config
    if isinstance(config, dict):
        return ArchConfig(**config)
    raise TypeError(f"Unsupported architecture config type: {type(config).__name__}")


def _coerce_training_config(training_spec: Any) -> TrainingConfig:
    if isinstance(training_spec, TrainingConfig):
        return training_spec
    if isinstance(training_spec, dict):
        return TrainingConfig(**training_spec)
    raise TypeError(f"Unsupported training config type: {type(training_spec).__name__}")


# =============================================================================
# Convenience: evaluate a known architecture
# =============================================================================

KNOWN_ARCHITECTURES = {
    # --- Dense models ---
    "Llama-2-7B":  {"d_model": 4096, "d_head": 128, "n_heads": 32, "ffn_dim": 11008,
                    "n_layers": 32, "n_kv_heads": 32, "vocab_size": 32000},
    "Llama-2-70B": {"d_model": 8192, "d_head": 128, "n_heads": 64, "ffn_dim": 28672,
                    "n_layers": 80, "n_kv_heads": 8, "vocab_size": 32000},
    "Llama-3-8B":  {"d_model": 4096, "d_head": 128, "n_heads": 32, "ffn_dim": 14336,
                    "n_layers": 32, "n_kv_heads": 8, "vocab_size": 128256},
    "Llama-3-70B": {"d_model": 8192, "d_head": 128, "n_heads": 64, "ffn_dim": 28672,
                    "n_layers": 80, "n_kv_heads": 8, "vocab_size": 128256},
    "Mistral-7B":  {"d_model": 4096, "d_head": 128, "n_heads": 32, "ffn_dim": 14336,
                    "n_layers": 32, "n_kv_heads": 8, "vocab_size": 32000},
    "Gemma-2-9B":  {"d_model": 3584, "d_head": 256, "n_heads": 16, "ffn_dim": 14336,
                    "n_layers": 42, "n_kv_heads": 8, "vocab_size": 256000},
    "Qwen3-8B":    {"d_model": 4096, "d_head": 128, "n_heads": 32, "ffn_dim": 12288,
                    "n_layers": 36, "n_kv_heads": 8, "vocab_size": 151936},
    "Qwen3-32B":   {"d_model": 5120, "d_head": 128, "n_heads": 64, "ffn_dim": 25600,
                    "n_layers": 64, "n_kv_heads": 8, "vocab_size": 151936},
    # --- MoE models (dense-equivalent: ffn_dim = active per-token FFN) ---
    # DeepSeek-V3: 671B total, ~37B active, MLA + MoE (8/256+1 shared)
    "DeepSeek-V3": {"d_model": 7168, "d_head": 128, "n_heads": 128, "ffn_dim": 18432,
                    "n_layers": 61, "n_kv_heads": 128, "vocab_size": 129280},
    # Kimi-K2.5: ~1T total, ~32B active, MLA + MoE (8/384+1 shared)
    "Kimi-K2.5":   {"d_model": 7168, "d_head": 128, "n_heads": 64,  "ffn_dim": 18432,
                    "n_layers": 61, "n_kv_heads": 64,  "vocab_size": 163840},
    # GLM-5.1: 754B total, ~40B active, MLA + MoE (8/256+1 shared)
    "GLM-5.1":     {"d_model": 6144, "d_head": 64,  "n_heads": 64,  "ffn_dim": 18432,
                    "n_layers": 78, "n_kv_heads": 64,  "vocab_size": 154880},
    # GPT-OSS-120B: 116.8B total, ~5.1B active, MoE (4/128)
    "GPT-OSS-120B": {"d_model": 2880, "d_head": 64, "n_heads": 64,  "ffn_dim": 11520,
                    "n_layers": 36, "n_kv_heads": 8,   "vocab_size": 201088},
    # MAI-Base-1: 962B total, ~34.7B active, Latent MoE (8/512)
    "MAI-Base-1":  {"d_model": 6656, "d_head": 128, "n_heads": 80,  "ffn_dim": 24576,
                    "n_layers": 78, "n_kv_heads": 8,   "vocab_size": 141056},
}


def evaluate_known(
    arch_name: str,
    hardware: str = "h100",
    training_tokens: int = 2_000_000_000_000,  # 2T tokens
    weight_precision: str = "bf16",
    kv_bits: int = 16,
) -> QualityResult:
    """Evaluate a known architecture."""
    if arch_name not in KNOWN_ARCHITECTURES:
        raise ValueError(f"Unknown: {arch_name}. Known: {list(KNOWN_ARCHITECTURES.keys())}")

    ka = KNOWN_ARCHITECTURES[arch_name]
    arch = ArchConfig(
        d_model=ka["d_model"], n_layers=ka["n_layers"],
        n_heads=ka["n_heads"], d_head=ka["d_head"],
        n_kv_heads=ka["n_kv_heads"], ffn_dim=ka["ffn_dim"],
        vocab_size=ka["vocab_size"],
        weight_precision=weight_precision,
    )
    training = TrainingConfig(
        training_tokens=training_tokens,
        hardware=hardware,
        kv_quantization_bits=kv_bits,
    )
    return quality(arch, training)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Quality Model v1 — Evaluation")
    print("=" * 70)

    for hw in ["h100", "b200", "tpu_v5p"]:
        print(f"\n{'='*70}")
        print(f"Hardware: {hw}")
        print(f"{'='*70}")

        for name in KNOWN_ARCHITECTURES:
            r = evaluate_known(name, hw, training_tokens=2_000_000_000_000)
            penalties_str = ", ".join(
                f"{k}={v.value:.4f}" for k, v in r.penalty_breakdown.items()
                if v.value > 0 and v.value < INFEASIBLE
            )
            print(f"  {name:15s} | L={r.predicted_loss:.4f} | "
                  f"base={r.chinchilla_baseline:.4f} | "
                  f"penalty={r.total_penalty_fraction:.4f} | "
                  f"dominant={r.dominant_penalty} | conf={r.confidence}")
            if penalties_str:
                print(f"{'':17s}   [{penalties_str}]")

    # Show precision impact
    print(f"\n{'='*70}")
    print("Precision Impact: Llama-3-8B on B200")
    print(f"{'='*70}")
    for prec, kv in [("bf16", 16), ("fp8", 16), ("fp8", 8), ("fp8", 4), ("fp4", 16)]:
        r = evaluate_known("Llama-3-8B", "b200", weight_precision=prec, kv_bits=kv)
        print(f"  weights={prec:4s} kv={kv:2d}bit | L={r.predicted_loss:.4f} | "
              f"penalty={r.total_penalty_fraction:.4f} | conf={r.confidence}")
