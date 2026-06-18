"""
Architecture Compiler v0 — Optimizer

Brute-force search over the lattice-restricted architecture space.
Returns the Pareto frontier in (quality, throughput, memory) and the
argmax under user-specified deployment constraints.

v0 search space: dense transformer with optional GQA, three hardware
targets (H100, B200, TPU v5p), uniform layers, brute-force enumeration.

Extension hooks reserved for:
  - v1: MoE (allow_moe flag, expert enumeration)
  - v2: State mechanisms (allow_state flag, hybrid ratio search)
  - v3: Cross-hardware (compare_hardware mode)
  - v6: Layer heterogeneity (allow_heterogeneous flag, block coordinate descent)
"""

import os
import sys
import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any

# Wire up sibling modules (flat package layout)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from lattice_engine import (
    HARDWARE as LATTICE_HW, compute_lattice, compute_gqa_configs,
    estimate_params, LatticePoint, GQAConfig,
    # v1 MoE additions
    compute_moe_options, default_ep_options, MoEOption,
    # v2 state/hybrid additions
    compute_state_lattice, compute_hybrid_patterns, place_attention_layers,
    HybridPattern,
)
from throughput_model import (
    throughput as throughput_fn, HardwareConfig, ArchConfig as TputArch,
    ThroughputResult,
)
from quality_model import (
    quality as quality_fn, ArchConfig as QualArch, TrainingConfig,
    QualityResult,
)
from schema import build_config, build_hybrid_config, SCHEMA_VERSION
from sram_derivation import derive_d_state, compute_crossover_seq_len


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class DeploymentConstraints:
    """User-specified deployment constraints."""
    target_params_b: float = 7.0        # target param count in billions
    param_tolerance: float = 0.15       # ±fraction around target
    training_tokens: int = 2_000_000_000_000  # D
    context_length: int = 8192
    # Serving constraints
    serving_tbt_ms: Optional[float] = 50.0   # time-between-tokens budget
    serving_ttft_ms: Optional[float] = 500.0  # time-to-first-token budget
    serving_batch: int = 32
    # Parallelism (inputs, not search variables in v0)
    tp: int = 8
    pp: int = 1
    dp: int = 8
    # Workload profile (v0-revision enrichment)
    prompt_len: Optional[int] = None         # if set, overrides context_length for prefill
    output_len: int = 512                    # expected generation length
    concurrency: int = 256                   # concurrent requests
    scheduler: str = "continuous"            # "continuous" | "static" | "chunked"
    traffic_mix: Optional[Dict[str, float]] = None  # e.g. {"short_chat": 0.3, "long_context": 0.5, "rag_prefill_heavy": 0.2}
    # Vocab
    vocab_size: int = 32000
    # Precision search space
    precision_configs: Optional[List[str]] = None  # None = enumerate defaults
    kv_bits_options: Optional[List[int]] = None     # None = [16, 8, 4]
    # v1+ hooks
    allow_moe: bool = False
    allow_state: bool = False
    allow_heterogeneous: bool = False  # TODO v6
    # v2 state/hybrid options
    state_type: str = "mamba2"
    placement_strategies: Optional[List[str]] = None  # None = ["first_periodic_last", "interleaved", "periodic"]
    state_precision: str = "bf16"

    # v1 MoE: when allow_moe=True, target_params_b is interpreted as the
    # N_active budget (per-token compute mass). max_total_params_b caps the
    # MoE total parameter count (memory ceiling); None => derived as 8x
    # active. ep_options and ep_topology control the all-to-all topology;
    # None => use lattice defaults for the hardware target.
    max_total_params_b: Optional[float] = None
    ep_options: Optional[List[int]] = None
    ep_topology: str = "single_axis"
    moe_n_experts_options: Optional[List[int]] = None
    moe_top_k_options: Optional[List[int]] = None
    moe_granularity_targets: Optional[List[float]] = None
    # v1-fix Part B: first-K-dense layer counts to sweep. None → [0] (no
    # dense prefix, pure MoE — original v1-MoE behavior). Adding 1-3 lets the
    # optimizer evaluate DeepSeek-V3 / Qwen3-MoE-style mixed FFN stacks.
    dense_ffn_layer_options: Optional[List[int]] = None

    # v1-fix MLA: enable DeepSeek-V2/V3-style Multi-head Latent Attention
    # candidates. When True, each lattice point also emits an MLA variant
    # with the latent shapes below. MLA dramatically reduces KV cache size
    # (typically 30-60× at long context) and is the dominant attention
    # choice for current frontier MoE models (DeepSeek-V3, Kimi K2, GLM-5).
    allow_mla: bool = False
    mla_kv_latent_options: Optional[List[int]] = None   # default [512]
    mla_q_latent_options: Optional[List[int]] = None    # default [1536]
    mla_rope_head_dim: int = 64
    mla_nope_head_dim: int = 128

    # v1-fix MTP: Multi-Token Prediction (DeepSeek-V3 §2.2). When
    # `allow_mtp=True`, the optimizer enumerates candidates with 0, 1, or 2
    # extra prediction depths. Each depth adds ~8% training compute overhead
    # and a ~0.6% loss-proxy bonus from sample efficiency.
    allow_mtp: bool = False
    mtp_depth_options: Optional[List[int]] = None       # default [0, 1]
    mtp_depth_n_layers: int = 1
    mtp_train_loss_weight: float = 0.3

    # v1-fix CP: Context Parallelism axis. At long context (≥32k), CP enables
    # training that wouldn't fit on a single rank's HBM. Ring Attention
    # streams KV across ranks; Ulysses scatters along the head axis.
    # CP degree multiplies into the world size: world = TP × PP × DP × CP × EP.
    cp: int = 1
    cp_method: str = "ring"           # "ring" | "ulysses"
    cp_options: Optional[List[int]] = None     # default [1, 2, 4, 8]

    # v1-fix RoPE scaling: positional-encoding extension method. When
    # `allow_rope_scaling=True`, the optimizer enumerates rope_scaling_methods
    # at each lattice point so the quality model can compare YaRN vs LongRoPE
    # vs none at long context. Methods: "none" | "pi" | "ntk" | "yarn" | "longrope".
    allow_rope_scaling: bool = False
    rope_scaling_methods: Optional[List[str]] = None
    rope_original_max_position: int = 8192    # training context, before extension

    # Search ergonomics. Defaults preserve exhaustive v0.3 behavior.
    max_candidates: Optional[int] = None       # cap after deterministic dedupe
    progress_every: int = 0                    # stderr update interval; 0 disables

    def __post_init__(self):
        if self.precision_configs is None:
            self.precision_configs = ["all_bf16", "ffn_fp8", "all_fp8"]
        if self.kv_bits_options is None:
            self.kv_bits_options = [16, 8, 4]
        if self.mla_kv_latent_options is None:
            # DeepSeek-V2 (512) and a smaller variant for compression-cost study
            self.mla_kv_latent_options = [512]
        if self.mla_q_latent_options is None:
            self.mla_q_latent_options = [1536]
        if self.mtp_depth_options is None:
            self.mtp_depth_options = [0, 1] if self.allow_mtp else [0]
        if self.cp_options is None:
            # CP makes sense at ≥ 32k contexts. For shorter, keep CP=1 only
            # (matches all existing data / regression).
            self.cp_options = [self.cp] if self.context_length < 32768 else [1, 2, 4, 8]
        if self.rope_scaling_methods is None:
            # Only sweep extension methods when the workload exceeds the
            # native pretrain context. Otherwise pin to "none".
            if not self.allow_rope_scaling or self.context_length <= self.rope_original_max_position:
                self.rope_scaling_methods = ["none"]
            else:
                self.rope_scaling_methods = ["yarn", "ntk", "longrope", "pi"]


@dataclass
class CandidateArch:
    """A candidate architecture with all orthogonal choices resolved."""
    d_model: int
    n_layers: int
    n_heads: int
    d_head: int
    n_kv_heads: int
    ffn_dim: int
    vocab_size: int
    # Precision choices
    weight_precision: str = "bf16"
    ffn_precision: str = "bf16"
    attn_precision: Dict[str, str] = field(default_factory=lambda: {"qk": "bf16", "v": "bf16", "output": "bf16"})
    kv_cache_bits: int = 16
    # Computed (dense: total = active; MoE: total > active)
    total_params: int = 0
    total_params_b: float = 0.0
    # v1 MoE fields (None / 0 for dense candidates).
    moe: Optional[dict] = None        # MoEFFNConfig dict (canonical nested shape)
    ep_degree: int = 1                # expert-parallel degree
    active_params: int = 0            # = total_params for dense
    active_params_b: float = 0.0      # = total_params_b for dense
    moe_style: str = "dense"          # "dense" | "coarse" | "fine"
    # v1-fix Part B: first-K-dense FFN prefix. Only meaningful when moe is set.
    n_dense_ffn_layers: int = 0
    # v2 state/hybrid fields
    state_config: Optional[dict] = None   # {d_state, state_expansion, n_heads, d_head, state_precision}
    layer_type_list: Optional[List[str]] = None  # per-layer "attention"|"state"
    placement_strategy: str = "none"       # "first_periodic_last"|"interleaved"|"periodic"|"none"
    n_attention_layers: int = 0            # 0 = all attention (v0/v1 dense/MoE)
    n_state_layers: int = 0
    hybrid_ratio: str = ""                 # e.g. "1:5.4"
    derived_d_state: int = 0
    crossover_seq_len: float = 0.0
    # v1-fix MLA: when set, this candidate uses MLA attention. The latent
    # dimensions feed both the throughput model (KV-bandwidth term) and
    # the quality model (small compression-quality penalty).
    attention_type: str = "full"           # "full" | "mla"
    mla_kv_latent_dim: int = 0             # c_kv (0 when type=full)
    mla_q_latent_dim: int = 0              # c_q
    mla_rope_head_dim: int = 0             # d_rope
    mla_nope_head_dim: int = 0             # d_nope
    # v1-fix MTP
    mtp_n_predict_depths: int = 0          # 0 = MTP off
    mtp_depth_n_layers: int = 1
    mtp_train_loss_weight: float = 0.3
    # v1-fix CP
    cp_degree: int = 1                     # 1 = no context parallelism
    cp_method: str = "ring"                # "ring" | "ulysses"
    # v1-fix RoPE scaling
    rope_scaling_method: str = "none"
    rope_scaling_factor: float = 1.0
    rope_original_max_position: int = 8192


@dataclass
class EvaluatedCandidate:
    """A candidate that has been evaluated by both throughput and quality models."""
    arch: CandidateArch
    quality: QualityResult
    throughput: ThroughputResult
    # Derived objectives (for Pareto)
    predicted_loss: float = 0.0
    training_tps: float = 0.0       # tokens per second
    serving_tbt_ms: float = 0.0
    memory_per_gpu_gb: float = 0.0
    # Serving regime analysis
    binding_serving_regime: str = ""  # "prefill-heavy" | "decode-heavy" | "mixed"
    binding_reason: str = ""
    # Feasibility
    meets_constraints: bool = True
    constraint_violations: List[str] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Output of the optimizer."""
    optimal: Optional[EvaluatedCandidate] = None
    pareto_frontier: List[EvaluatedCandidate] = field(default_factory=list)
    all_evaluated: List[EvaluatedCandidate] = field(default_factory=list)
    # Stats
    candidates_generated: int = 0
    candidates_feasible: int = 0
    candidates_evaluated: int = 0
    search_time_sec: float = 0.0
    hardware: str = ""
    constraints: Optional[DeploymentConstraints] = None
    # Binding constraints (populated after optimization)
    binding_constraints: List[str] = field(default_factory=list)


# =============================================================================
# Precision config expansion
# =============================================================================

PRECISION_CONFIGS = {
    "all_bf16": {
        "weight_precision": "bf16",
        "ffn_precision": "bf16",
        "attn_precision": {"qk": "bf16", "v": "bf16", "output": "bf16"},
    },
    "ffn_fp8": {
        "weight_precision": "bf16",
        "ffn_precision": "fp8",
        "attn_precision": {"qk": "bf16", "v": "bf16", "output": "bf16"},
    },
    "all_fp8": {
        "weight_precision": "fp8",
        "ffn_precision": "fp8",
        "attn_precision": {"qk": "bf16", "v": "fp8", "output": "fp8"},
    },
    "ffn_fp4": {
        "weight_precision": "fp8",
        "ffn_precision": "fp4",
        "attn_precision": {"qk": "bf16", "v": "fp8", "output": "fp8"},
    },
    "all_fp4": {
        "weight_precision": "fp4",
        "ffn_precision": "fp4",
        "attn_precision": {"qk": "bf16", "v": "fp4", "output": "fp4"},
    },
    # v1-fix microscaling: OCP MX-format variants. MXFP4 uses E2M1 mantissa
    # + 8-bit shared scale per 32-element block, much closer to FP6/FP8
    # quality than plain E2M1. MXFP6 uses E2M3 / E3M2. Blackwell tensor cores
    # natively implement these scales; H100 must emulate (slower).
    "ffn_mxfp4": {
        "weight_precision": "fp8",
        "ffn_precision": "mxfp4",
        "attn_precision": {"qk": "bf16", "v": "fp8", "output": "fp8"},
    },
    "all_mxfp4": {
        "weight_precision": "mxfp4",
        "ffn_precision": "mxfp4",
        "attn_precision": {"qk": "bf16", "v": "mxfp4", "output": "mxfp4"},
    },
    "ffn_mxfp6": {
        "weight_precision": "fp8",
        "ffn_precision": "mxfp6",
        "attn_precision": {"qk": "bf16", "v": "fp8", "output": "fp8"},
    },
    "all_mxfp6": {
        "weight_precision": "mxfp6",
        "ffn_precision": "mxfp6",
        "attn_precision": {"qk": "bf16", "v": "mxfp6", "output": "mxfp6"},
    },
}


def kv_bits_to_precision(kv_bits: int) -> str:
    """Map KV cache bit width to throughput-model precision label."""
    if kv_bits == 16:
        return "bf16"
    if kv_bits == 8:
        return "int8"
    if kv_bits == 4:
        # Throughput model uses fp4 as its 4-bit byte-width label.
        return "fp4"
    return "bf16"


def get_precision_configs_for_hardware(hw_name: str) -> List[str]:
    """Return the valid precision config names for a given hardware.

    v1-fix microscaling: B200 natively supports MXFP4/MXFP6 in the tensor
    cores (OCP MX scales). H100 supports E2M1 FP4 only via emulation;
    Blackwell is the first generation with hardware-accelerated MX.

    v1-fix Trainium: Trn2 supports BF16/FP8; Trn3 adds FP4 + MX formats.
    """
    if hw_name == "b200":
        return ["all_bf16", "ffn_fp8", "all_fp8",
                "ffn_fp4", "all_fp4",
                "ffn_mxfp4", "all_mxfp4",
                "ffn_mxfp6", "all_mxfp6"]
    elif hw_name in ("h100",):
        return ["all_bf16", "ffn_fp8", "all_fp8"]
    elif hw_name in ("trainium2", "trn2"):
        return ["all_bf16", "ffn_fp8", "all_fp8"]
    elif hw_name in ("trainium3", "trn3"):
        return ["all_bf16", "ffn_fp8", "all_fp8",
                "ffn_fp4", "all_fp4",
                "ffn_mxfp4", "all_mxfp4",
                "ffn_mxfp6", "all_mxfp6"]
    else:
        # TPU: BF16 only in v0
        return ["all_bf16"]


# =============================================================================
# Candidate generation
# =============================================================================

def generate_candidates(
    hw_name: str,
    constraints: DeploymentConstraints,
) -> List[CandidateArch]:
    """Generate all candidate architectures from the lattice within the param band."""

    target = constraints.target_params_b * 1e9
    lo = target * (1 - constraints.param_tolerance)
    hi = target * (1 + constraints.param_tolerance)

    lattice_hw = LATTICE_HW.get(hw_name)
    if lattice_hw is None:
        raise ValueError(f"Unknown hardware: {hw_name}. Known: {list(LATTICE_HW.keys())}")

    # Determine which precision to use for lattice computation
    # Use BF16 as the base lattice (most restrictive alignment is fine for v0)
    precision = "bf16"
    if precision not in lattice_hw.tiles:
        precision = list(lattice_hw.tiles.keys())[0]

    lattice = compute_lattice(
        lattice_hw, precision, constraints.tp,
        d_model_min=1024, d_model_max=16384,
        d_head_options=[64, 128, 256],
    )

    # Filter to tile-aligned points only
    aligned = [pt for pt in lattice if pt.tile_aligned]

    # Valid precision configs for this hardware
    hw_prec_configs = get_precision_configs_for_hardware(hw_name)
    prec_configs = [p for p in (constraints.precision_configs or []) if p in hw_prec_configs]
    if not prec_configs:
        prec_configs = hw_prec_configs[:3]  # default to first 3

    candidates = []

    for pt in aligned:
        # Enumerate GQA ratios
        gqa_ratios = [1]  # MHA
        for r in [2, 4, 8, 16]:
            if pt.n_heads % r == 0:
                n_kv = pt.n_heads // r
                if n_kv >= 1 and (n_kv >= constraints.tp or n_kv == 1):
                    gqa_ratios.append(r)
        # MQA
        if pt.n_heads > 1:
            gqa_ratios.append(pt.n_heads)

        gqa_ratios = sorted(set(gqa_ratios))

        for gqa_r in gqa_ratios:
            n_kv_heads = max(1, pt.n_heads // gqa_r)

            # Compute n_layers for target param count
            per_layer_1 = estimate_params(
                pt.d_model, pt.n_heads, pt.d_head, pt.ffn_dim_swiglu,
                1, n_kv_heads, constraints.vocab_size
            )
            embed_params = 2 * constraints.vocab_size * pt.d_model
            per_layer_net = per_layer_1 - embed_params - 2 * pt.d_model  # remove embed + norm
            if per_layer_net <= 0:
                continue

            n_layers_raw = (target - embed_params) / per_layer_net
            # Try a few layer counts around the target
            for n_layers in [max(4, round(n_layers_raw) + delta) for delta in [-2, -1, 0, 1, 2]]:
                total = estimate_params(
                    pt.d_model, pt.n_heads, pt.d_head, pt.ffn_dim_swiglu,
                    n_layers, n_kv_heads, constraints.vocab_size
                )
                if total < lo or total > hi:
                    continue

                # PP divisibility check
                if constraints.pp > 1 and n_layers % constraints.pp != 0:
                    continue

                # Enumerate precision × KV bits × MTP depths
                # v1-fix MTP: when allow_mtp=True the search sweeps depth 0
                # and 1+ (DeepSeek-V3 reports k=1 dominates the cost/benefit
                # tradeoff; we cap at 2 by default).
                mtp_opts = constraints.mtp_depth_options if constraints.allow_mtp else [0]
                cp_opts = constraints.cp_options or [constraints.cp]
                rope_opts = constraints.rope_scaling_methods or ["none"]
                rope_factor = max(1.0, constraints.context_length / max(1, constraints.rope_original_max_position))
                for prec_name in prec_configs:
                    prec = PRECISION_CONFIGS[prec_name]
                    for kv_bits in constraints.kv_bits_options:
                      for mtp_k in mtp_opts:
                       for cp_d in cp_opts:
                        for rope_m in rope_opts:
                          candidates.append(CandidateArch(
                            d_model=pt.d_model,
                            n_layers=n_layers,
                            n_heads=pt.n_heads,
                            d_head=pt.d_head,
                            n_kv_heads=n_kv_heads,
                            ffn_dim=pt.ffn_dim_swiglu,
                            vocab_size=constraints.vocab_size,
                            weight_precision=prec["weight_precision"],
                            ffn_precision=prec["ffn_precision"],
                            attn_precision=dict(prec["attn_precision"]),
                            kv_cache_bits=kv_bits,
                            total_params=total,
                            total_params_b=round(total / 1e9, 2),
                            mtp_n_predict_depths=int(mtp_k),
                            mtp_depth_n_layers=int(constraints.mtp_depth_n_layers),
                            mtp_train_loss_weight=float(constraints.mtp_train_loss_weight),
                            cp_degree=int(cp_d),
                            cp_method=str(constraints.cp_method),
                            rope_scaling_method=str(rope_m),
                            rope_scaling_factor=rope_factor if rope_m != "none" else 1.0,
                            rope_original_max_position=int(constraints.rope_original_max_position),
                        ))

                        # v1-fix MLA: when allow_mla=True, also emit an MLA
                        # variant of this lattice point. The MLA candidate
                        # shares d_model/n_heads/n_layers/ffn_dim with the
                        # legacy candidate but replaces the KV cache shape
                        # with a single compressed latent + RoPE'd key.
                        if getattr(constraints, "allow_mla", False):
                            for c_kv in constraints.mla_kv_latent_options:
                                for c_q in constraints.mla_q_latent_options:
                                    # Sanity: latent should compress KV
                                    uncompressed = 2 * pt.n_heads * pt.d_head
                                    if c_kv >= uncompressed:
                                        continue
                                    candidates.append(CandidateArch(
                                        d_model=pt.d_model,
                                        n_layers=n_layers,
                                        n_heads=pt.n_heads,
                                        d_head=pt.d_head,
                                        n_kv_heads=n_kv_heads,
                                        ffn_dim=pt.ffn_dim_swiglu,
                                        vocab_size=constraints.vocab_size,
                                        weight_precision=prec["weight_precision"],
                                        ffn_precision=prec["ffn_precision"],
                                        attn_precision=dict(prec["attn_precision"]),
                                        kv_cache_bits=kv_bits,
                                        total_params=total,
                                        total_params_b=round(total / 1e9, 2),
                                        # MLA-specific fields
                                        attention_type="mla",
                                        mla_kv_latent_dim=c_kv,
                                        mla_q_latent_dim=c_q,
                                        mla_rope_head_dim=constraints.mla_rope_head_dim,
                                        mla_nope_head_dim=constraints.mla_nope_head_dim,
                                    ))

    return candidates


# =============================================================================
# v1 MoE candidate generation
# =============================================================================

def _moe_active_params_per_layer(
    d_model: int, n_heads: int, d_head: int, n_kv_heads: int, opt: MoEOption,
) -> int:
    """N_active per-layer FFN+attn for a given lattice point and MoE option."""
    attn = (
        d_model * n_heads * d_head        # Q
        + 2 * d_model * n_kv_heads * d_head  # K, V (GQA-reduced)
        + d_model * n_heads * d_head      # O
    )
    # SwiGLU expert: 3 matmuls (up + gate + down) per expert.
    ffn_active = opt.top_k * 3 * d_model * opt.expert_dim
    if opt.shared_dim:
        ffn_active += 3 * d_model * opt.shared_dim
    return attn + ffn_active


def _moe_total_params_per_layer(
    d_model: int, n_heads: int, d_head: int, n_kv_heads: int, opt: MoEOption,
) -> int:
    """N_total per-layer FFN+attn for a given lattice point and MoE option."""
    attn = (
        d_model * n_heads * d_head
        + 2 * d_model * n_kv_heads * d_head
        + d_model * n_heads * d_head
    )
    ffn_total = opt.n_experts * 3 * d_model * opt.expert_dim
    if opt.shared_dim:
        ffn_total += 3 * d_model * opt.shared_dim
    return attn + ffn_total


def generate_moe_candidates(
    hw_name: str,
    constraints: DeploymentConstraints,
) -> List[CandidateArch]:
    """Enumerate MoE candidates that fit the N_active target band and the
    max_total_params_b memory ceiling. Returns CandidateArch instances with
    moe/ep_degree/active_params filled in.

    Skeleton policy:
      - Active-target = constraints.target_params_b (interpreted as N_active).
      - max_total = constraints.max_total_params_b (defaults to 8 × active).
      - Lattice points come from the same compute_lattice call as the dense
        path; n_layers is fit to hit the active band; precision/KV bits are
        sampled from the existing hardware-valid configs.
      - EP options come from constraints.ep_options or lattice defaults.
    """
    if not constraints.allow_moe:
        return []

    target_active = constraints.target_params_b * 1e9
    lo_active = target_active * (1 - constraints.param_tolerance)
    hi_active = target_active * (1 + constraints.param_tolerance)
    max_total = (constraints.max_total_params_b or constraints.target_params_b * 8.0) * 1e9

    lattice_hw = LATTICE_HW.get(hw_name)
    if lattice_hw is None:
        raise ValueError(f"Unknown hardware: {hw_name}. Known: {list(LATTICE_HW.keys())}")

    precision = "bf16"
    if precision not in lattice_hw.tiles:
        precision = list(lattice_hw.tiles.keys())[0]

    lattice = compute_lattice(
        lattice_hw, precision, constraints.tp,
        d_model_min=1024, d_model_max=16384,
        d_head_options=[64, 128, 256],
    )
    aligned = [pt for pt in lattice if pt.tile_aligned]

    hw_prec_configs = get_precision_configs_for_hardware(hw_name)
    prec_configs = [p for p in (constraints.precision_configs or []) if p in hw_prec_configs]
    if not prec_configs:
        prec_configs = hw_prec_configs[:3]

    ep_opts = constraints.ep_options or default_ep_options(hw_name)

    # v1-fix Part B: sweep n_dense_ffn_layers. Default is [0] (pure MoE, the
    # original v1-MoE behavior). [0, 1, 2, 3] covers the common dense-prefix
    # range used by DeepSeek-V3 / Qwen3-MoE / similar.
    dense_ffn_layer_opts = constraints.dense_ffn_layer_options
    if dense_ffn_layer_opts is None:
        dense_ffn_layer_opts = [0]

    candidates: List[CandidateArch] = []

    for pt in aligned:
        # Use the dense lattice's GQA shape sweep (mirrors dense path).
        gqa_ratios = [1]
        for r in [2, 4, 8, 16]:
            if pt.n_heads % r == 0:
                n_kv = pt.n_heads // r
                if n_kv >= 1 and (n_kv >= constraints.tp or n_kv == 1):
                    gqa_ratios.append(r)
        if pt.n_heads > 1:
            gqa_ratios.append(pt.n_heads)
        gqa_ratios = sorted(set(gqa_ratios))

        # MoE options for this lattice point's d_model and ffn_dim baseline.
        moe_opts = compute_moe_options(
            lattice_hw, precision,
            d_model=pt.d_model,
            baseline_ffn_dim=pt.ffn_dim_swiglu,
            ep_degrees=ep_opts,
            n_experts_options=constraints.moe_n_experts_options,
            top_k_options=constraints.moe_top_k_options,
            granularity_targets=tuple(constraints.moe_granularity_targets)
                if constraints.moe_granularity_targets else (1.0, 0.5, 0.25),
        )

        for gqa_r in gqa_ratios:
            n_kv_heads = max(1, pt.n_heads // gqa_r)

            for opt in moe_opts:
                active_per_layer = _moe_active_params_per_layer(
                    pt.d_model, pt.n_heads, pt.d_head, n_kv_heads, opt,
                )
                total_per_layer = _moe_total_params_per_layer(
                    pt.d_model, pt.n_heads, pt.d_head, n_kv_heads, opt,
                )
                if active_per_layer <= 0:
                    continue

                embed = 2 * constraints.vocab_size * pt.d_model
                n_layers_raw = (target_active - embed) / active_per_layer
                if not (1 <= n_layers_raw <= 256):
                    continue

                # v1-fix Part B: per-layer params for dense FFN, used when
                # n_dense > 0 to adjust the active/total counts. Dense FFN
                # contributes the same params to both N_active and N_total
                # (a dense layer activates all of itself).
                dense_ffn_params = 3 * pt.d_model * pt.ffn_dim_swiglu
                attn_params_per_layer = (
                    pt.d_model * pt.n_heads * pt.d_head
                    + 2 * pt.d_model * n_kv_heads * pt.d_head
                    + pt.d_model * pt.n_heads * pt.d_head
                )
                dense_per_layer = attn_params_per_layer + dense_ffn_params

                for n_layers in [max(4, round(n_layers_raw) + d) for d in (-2, -1, 0, 1, 2)]:
                    if constraints.pp > 1 and n_layers % constraints.pp != 0:
                        continue

                    for n_dense in dense_ffn_layer_opts:
                        n_dense_clamped = max(0, min(int(n_dense), n_layers - 1))
                        # Account for the dense prefix in both active and
                        # total counts: replace n_dense MoE layers with dense
                        # ones. (Dense layers are smaller in N_total than
                        # MoE layers but bigger than the MoE active path.)
                        n_moe_l = n_layers - n_dense_clamped
                        active_total = (
                            n_dense_clamped * dense_per_layer
                            + n_moe_l * active_per_layer
                            + embed
                        )
                        if active_total < lo_active or active_total > hi_active:
                            continue
                        total_total = (
                            n_dense_clamped * dense_per_layer
                            + n_moe_l * total_per_layer
                            + embed
                        )
                        if total_total > max_total:
                            continue

                        for prec_name in prec_configs:
                            prec = PRECISION_CONFIGS[prec_name]
                            for kv_bits in constraints.kv_bits_options:
                                # Build the canonical MoE FFN dict (matches W1 schema).
                                shared_block = None
                                if opt.shared_dim:
                                    shared_block = {
                                        "ffn_dim": opt.shared_dim,
                                        "precision": prec["ffn_precision"],
                                    }
                                moe_dict = {
                                    "type": "moe",
                                    "n_experts": opt.n_experts,
                                    "top_k": opt.top_k,
                                    "expert_dim": opt.expert_dim,
                                    "shared_expert": shared_block,
                                    "router": {
                                        "precision": "bf16",
                                        "load_balance_loss_coef": 0.01,
                                        "noise_type": None,
                                    },
                                    "capacity_factor": 1.25 if opt.style == "coarse" else 1.0,
                                    "precision": prec["ffn_precision"],
                                }

                                candidates.append(CandidateArch(
                                    d_model=pt.d_model,
                                    n_layers=n_layers,
                                    n_heads=pt.n_heads,
                                    d_head=pt.d_head,
                                    n_kv_heads=n_kv_heads,
                                    ffn_dim=pt.ffn_dim_swiglu,   # baseline retained for memory subtraction
                                    vocab_size=constraints.vocab_size,
                                    weight_precision=prec["weight_precision"],
                                    ffn_precision=prec["ffn_precision"],
                                    attn_precision=dict(prec["attn_precision"]),
                                    kv_cache_bits=kv_bits,
                                    total_params=total_total,
                                    total_params_b=round(total_total / 1e9, 2),
                                    moe=moe_dict,
                                    ep_degree=opt.ep_degree,
                                    active_params=active_total,
                                    active_params_b=round(active_total / 1e9, 2),
                                    moe_style=opt.style,
                                    n_dense_ffn_layers=n_dense_clamped,
                                ))

    return candidates


# =============================================================================
# v2 State/hybrid candidate generation
# =============================================================================

def generate_state_candidates(
    hw_name: str,
    constraints: DeploymentConstraints,
) -> List[CandidateArch]:
    """Generate hybrid attention/state candidates.

    The search space is:
      dense lattice shape x d_state x n_attn x placement_strategy x precision x kv_bits

    d_state is derived from SRAM via sram_derivation, then tile-aligned candidates
    come from compute_state_lattice. n_attn range is guided by the crossover
    sequence length L*.
    """
    if not constraints.allow_state:
        return []

    target = constraints.target_params_b * 1e9
    lo = target * (1 - constraints.param_tolerance)
    hi = target * (1 + constraints.param_tolerance)

    lattice_hw = LATTICE_HW.get(hw_name)
    if lattice_hw is None:
        raise ValueError(f"Unknown hardware: {hw_name}. Known: {list(LATTICE_HW.keys())}")

    precision = "bf16"
    if precision not in lattice_hw.tiles:
        precision = list(lattice_hw.tiles.keys())[0]

    lattice = compute_lattice(
        lattice_hw, precision, constraints.tp,
        d_model_min=1024, d_model_max=16384,
        d_head_options=[64, 128, 256],
    )
    aligned = [pt for pt in lattice if pt.tile_aligned]

    hw_prec_configs = get_precision_configs_for_hardware(hw_name)
    prec_configs = [p for p in (constraints.precision_configs or []) if p in hw_prec_configs]
    if not prec_configs:
        prec_configs = hw_prec_configs[:3]

    strategies = constraints.placement_strategies or ["first_periodic_last", "interleaved", "periodic"]

    # Get tile-aligned d_state candidates from lattice.
    # Use d_head=64 as the reference (most permissive); hardware with
    # d_head=128 will derive smaller d_state via SRAM budget.
    all_d_state = compute_state_lattice(
        lattice_hw, d_head=64,
        state_precision=constraints.state_precision,
        alpha_state=0.85,
    )
    if not all_d_state:
        return []
    # Sample: keep min, max, and at most 2 intermediates to limit explosion.
    if len(all_d_state) <= 3:
        d_state_candidates = all_d_state
    else:
        d_state_candidates = [all_d_state[0], all_d_state[len(all_d_state)//2], all_d_state[-1]]

    candidates: List[CandidateArch] = []

    for pt in aligned:
        # GQA sweep (same as dense path)
        gqa_ratios = [1]
        for r in [2, 4, 8, 16]:
            if pt.n_heads % r == 0:
                n_kv = pt.n_heads // r
                if n_kv >= 1 and (n_kv >= constraints.tp or n_kv == 1):
                    gqa_ratios.append(r)
        if pt.n_heads > 1:
            gqa_ratios.append(pt.n_heads)
        gqa_ratios = sorted(set(gqa_ratios))

        for gqa_r in gqa_ratios:
            n_kv_heads = max(1, pt.n_heads // gqa_r)

            # Compute n_layers for target param count (same as dense)
            per_layer_1 = estimate_params(
                pt.d_model, pt.n_heads, pt.d_head, pt.ffn_dim_swiglu,
                1, n_kv_heads, constraints.vocab_size
            )
            embed_params = 2 * constraints.vocab_size * pt.d_model
            per_layer_net = per_layer_1 - embed_params - 2 * pt.d_model
            if per_layer_net <= 0:
                continue

            n_layers_raw = (target - embed_params) / per_layer_net

            for n_layers in [max(4, round(n_layers_raw) + delta) for delta in [-2, -1, 0, 1, 2]]:
                total = estimate_params(
                    pt.d_model, pt.n_heads, pt.d_head, pt.ffn_dim_swiglu,
                    n_layers, n_kv_heads, constraints.vocab_size
                )
                if total < lo or total > hi:
                    continue
                if constraints.pp > 1 and n_layers % constraints.pp != 0:
                    continue

                for d_state in d_state_candidates:
                    # Compute crossover sequence length L*
                    L_star = compute_crossover_seq_len(
                        hw_name=hw_name,
                        n_kv_heads=n_kv_heads,
                        d_head=pt.d_head,
                        batch_size=1,
                        kv_precision="bf16",
                        d_state=d_state,
                        state_expansion=2,
                        d_model=pt.d_model,
                        state_precision=constraints.state_precision,
                    )

                    # Derive n_attn range from L* and target context
                    context_length = constraints.context_length
                    quality_floor = max(2, int(math.log2(max(context_length, 1))))

                    if context_length > L_star * 4:
                        # State-heavy: few attention layers
                        suggested_max_attn = max(quality_floor, n_layers // 4)
                    elif context_length > L_star:
                        # Mixed: moderate attention
                        suggested_max_attn = max(quality_floor, n_layers // 2)
                    else:
                        # Attention-favorable: mostly attention
                        suggested_max_attn = n_layers  # pure attention likely wins

                    # Include n_attn=0 for pure state candidate.
                    # Use striding to keep the search space manageable when
                    # the n_attn range is large (e.g., 200+ layers).
                    attn_lo = quality_floor
                    attn_hi = min(suggested_max_attn, n_layers - 1)
                    span = max(1, attn_hi - attn_lo + 1)
                    max_samples = 5  # at most 5 n_attn values per (shape, d_state)
                    stride = max(1, span // max_samples)
                    n_attn_values = [0]  # pure state
                    n_attn_values += list(range(attn_lo, attn_hi + 1, stride))
                    # Always include the endpoints
                    if attn_hi not in n_attn_values:
                        n_attn_values.append(attn_hi)
                    # Deduplicate and sort
                    n_attn_values = sorted(set(n_attn_values))

                    for n_attn in n_attn_values:
                        n_state = n_layers - n_attn

                        for strategy in strategies:
                            attn_indices = place_attention_layers(n_layers, n_attn, strategy)
                            actual_n_attn = len(attn_indices)
                            actual_n_state = n_layers - actual_n_attn

                            # Build layer_type_list
                            attn_set = set(attn_indices)
                            layer_type_list = [
                                "attention" if i in attn_set else "state"
                                for i in range(n_layers)
                            ]

                            # Hybrid ratio string
                            if actual_n_attn > 0 and actual_n_state > 0:
                                ratio = actual_n_state / actual_n_attn
                                hybrid_ratio = f"1:{ratio:.1f}"
                            elif actual_n_attn == 0:
                                hybrid_ratio = "pure_state"
                            else:
                                hybrid_ratio = "pure_attention"

                            state_cfg = {
                                "d_state": d_state,
                                "state_expansion": 2,
                                "n_heads": pt.n_heads,
                                "d_head": pt.d_head,
                                "state_precision": constraints.state_precision,
                                # v1-fix UI: carry the actual SSM/linear-attention family
                                # through to result_to_config and the quality model.
                                # Previously state_cfg only carried the numeric precision,
                                # and downstream code read state_cfg["state_precision"]
                                # as if it were the family name — silently coercing every
                                # Part-J family to mamba_sequential.
                                "state_type": constraints.state_type,
                            }

                            for prec_name in prec_configs:
                                prec = PRECISION_CONFIGS[prec_name]
                                for kv_bits in constraints.kv_bits_options:
                                    candidates.append(CandidateArch(
                                        d_model=pt.d_model,
                                        n_layers=n_layers,
                                        n_heads=pt.n_heads,
                                        d_head=pt.d_head,
                                        n_kv_heads=n_kv_heads,
                                        ffn_dim=pt.ffn_dim_swiglu,
                                        vocab_size=constraints.vocab_size,
                                        weight_precision=prec["weight_precision"],
                                        ffn_precision=prec["ffn_precision"],
                                        attn_precision=dict(prec["attn_precision"]),
                                        kv_cache_bits=kv_bits,
                                        total_params=total,
                                        total_params_b=round(total / 1e9, 2),
                                        # State fields
                                        state_config=state_cfg,
                                        layer_type_list=layer_type_list,
                                        placement_strategy=strategy,
                                        n_attention_layers=actual_n_attn,
                                        n_state_layers=actual_n_state,
                                        hybrid_ratio=hybrid_ratio,
                                        derived_d_state=d_state,
                                        crossover_seq_len=L_star,
                                    ))

    return candidates


# =============================================================================
# v1-fix Part D: Combined MoE × hybrid-state candidate generation
# =============================================================================

def generate_moe_hybrid_candidates(
    hw_name: str,
    constraints: DeploymentConstraints,
) -> List[CandidateArch]:
    """Generate candidates that are *both* MoE and hybrid attention/state.

    This is the cartesian product of the MoE and state search spaces:
      lattice point × GQA × MoE option × state d_state × n_attn × placement
        × precision × kv_bits

    Active and total parameter accounting handles three layer types:
      - attention + MoE FFN (standard MoE layer)
      - state + MoE FFN     (Mamba layer with routed FFN; e.g. Jamba-MoE)
      - attention + dense FFN if first-K-dense (currently disabled in this
        generator to keep the search space bounded — combined-mode dense
        prefix is a v1-fix Part D' follow-on).

    Requires both allow_moe and allow_state to be True. Returns [] otherwise.
    The optimizer adds these candidates *in addition to* pure-MoE and
    pure-hybrid candidates so the Pareto frontier compares all four families
    (dense, MoE, hybrid, MoE+hybrid) on equal footing.
    """
    if not (constraints.allow_moe and constraints.allow_state):
        return []

    target_active = constraints.target_params_b * 1e9
    lo_active = target_active * (1 - constraints.param_tolerance)
    hi_active = target_active * (1 + constraints.param_tolerance)
    max_total = (constraints.max_total_params_b or constraints.target_params_b * 8.0) * 1e9

    lattice_hw = LATTICE_HW.get(hw_name)
    if lattice_hw is None:
        raise ValueError(f"Unknown hardware: {hw_name}. Known: {list(LATTICE_HW.keys())}")

    precision = "bf16"
    if precision not in lattice_hw.tiles:
        precision = list(lattice_hw.tiles.keys())[0]

    lattice = compute_lattice(
        lattice_hw, precision, constraints.tp,
        d_model_min=1024, d_model_max=16384,
        d_head_options=[64, 128, 256],
    )
    aligned = [pt for pt in lattice if pt.tile_aligned]

    hw_prec_configs = get_precision_configs_for_hardware(hw_name)
    prec_configs = [p for p in (constraints.precision_configs or []) if p in hw_prec_configs]
    if not prec_configs:
        prec_configs = hw_prec_configs[:3]

    ep_opts = constraints.ep_options or default_ep_options(hw_name)
    strategies = constraints.placement_strategies or ["first_periodic_last", "interleaved", "periodic"]

    # State lattice — same downsampling as generate_state_candidates.
    all_d_state = compute_state_lattice(
        lattice_hw, d_head=64,
        state_precision=constraints.state_precision,
        alpha_state=0.85,
    )
    if not all_d_state:
        return []
    if len(all_d_state) <= 2:
        d_state_candidates = all_d_state
    else:
        # Keep the search small — middle d_state is the most representative.
        d_state_candidates = [all_d_state[len(all_d_state) // 2]]

    candidates: List[CandidateArch] = []

    for pt in aligned:
        # GQA sweep
        gqa_ratios = [1]
        for r in [2, 4, 8, 16]:
            if pt.n_heads % r == 0:
                n_kv = pt.n_heads // r
                if n_kv >= 1 and (n_kv >= constraints.tp or n_kv == 1):
                    gqa_ratios.append(r)
        if pt.n_heads > 1:
            gqa_ratios.append(pt.n_heads)
        gqa_ratios = sorted(set(gqa_ratios))

        # MoE options
        moe_opts = compute_moe_options(
            lattice_hw, precision,
            d_model=pt.d_model,
            baseline_ffn_dim=pt.ffn_dim_swiglu,
            ep_degrees=ep_opts,
            n_experts_options=constraints.moe_n_experts_options,
            top_k_options=constraints.moe_top_k_options,
            granularity_targets=tuple(constraints.moe_granularity_targets)
                if constraints.moe_granularity_targets else (1.0,),
        )

        for gqa_r in gqa_ratios:
            n_kv_heads = max(1, pt.n_heads // gqa_r)

            # Attention-layer per-layer params (shared with dense path).
            attn_params_per_layer = (
                pt.d_model * pt.n_heads * pt.d_head
                + 2 * pt.d_model * n_kv_heads * pt.d_head
                + pt.d_model * pt.n_heads * pt.d_head
            )
            # State-layer per-layer params (Mamba-2 style, approximate):
            # in_proj + B_proj + C_proj + out_proj + dt_proj ≈ comparable to
            # attention but no KV. Use a 1.0× attention proxy as a coarse
            # placeholder — the throughput model handles the actual cost.
            state_params_per_layer = attn_params_per_layer

            for opt in moe_opts:
                # Per-layer FFN params: MoE active vs total.
                ffn_active_per_layer = opt.top_k * 3 * pt.d_model * opt.expert_dim
                if opt.shared_dim:
                    ffn_active_per_layer += 3 * pt.d_model * opt.shared_dim
                ffn_total_per_layer = opt.n_experts * 3 * pt.d_model * opt.expert_dim
                if opt.shared_dim:
                    ffn_total_per_layer += 3 * pt.d_model * opt.shared_dim

                # Build the canonical MoE FFN dict.
                shared_block = None
                if opt.shared_dim:
                    shared_block = {
                        "ffn_dim": opt.shared_dim,
                        "precision": "bf16",  # filled per prec_name below
                    }

                # Initial n_layers estimate uses an average per-layer cost
                # (attention + MoE active). The exact split shifts when we
                # vary n_attn but the embed-discounted target is the same.
                embed = 2 * constraints.vocab_size * pt.d_model
                avg_per_layer = attn_params_per_layer + ffn_active_per_layer
                n_layers_raw = (target_active - embed) / avg_per_layer
                if not (4 <= n_layers_raw <= 200):
                    continue

                for n_layers in [max(4, round(n_layers_raw) + d) for d in (-1, 0, 1)]:
                    if constraints.pp > 1 and n_layers % constraints.pp != 0:
                        continue

                    for d_state in d_state_candidates:
                        # Crossover sequence length and suggested n_attn.
                        L_star = compute_crossover_seq_len(
                            hw_name=hw_name,
                            n_kv_heads=n_kv_heads,
                            d_head=pt.d_head,
                            batch_size=1,
                            kv_precision="bf16",
                            d_state=d_state,
                            state_expansion=2,
                            d_model=pt.d_model,
                            state_precision=constraints.state_precision,
                        )

                        context_length = constraints.context_length
                        quality_floor = max(2, int(math.log2(max(context_length, 1))))
                        if context_length > L_star * 4:
                            suggested_max_attn = max(quality_floor, n_layers // 4)
                        elif context_length > L_star:
                            suggested_max_attn = max(quality_floor, n_layers // 2)
                        else:
                            suggested_max_attn = n_layers

                        # Sparse sweep over n_attn to keep combinatorics manageable.
                        attn_lo = quality_floor
                        attn_hi = min(suggested_max_attn, n_layers - 1)
                        if attn_hi < attn_lo:
                            continue
                        attn_values = sorted({attn_lo, (attn_lo + attn_hi) // 2, attn_hi})

                        for n_attn in attn_values:
                            n_state = n_layers - n_attn

                            # Per-layer-type param computation.
                            active_total = (
                                n_attn * (attn_params_per_layer + ffn_active_per_layer)
                                + n_state * (state_params_per_layer + ffn_active_per_layer)
                                + embed
                            )
                            if active_total < lo_active or active_total > hi_active:
                                continue
                            total_total = (
                                n_attn * (attn_params_per_layer + ffn_total_per_layer)
                                + n_state * (state_params_per_layer + ffn_total_per_layer)
                                + embed
                            )
                            if total_total > max_total:
                                continue

                            for strategy in strategies:
                                attn_indices = place_attention_layers(n_layers, n_attn, strategy)
                                actual_n_attn = len(attn_indices)
                                actual_n_state = n_layers - actual_n_attn

                                attn_set = set(attn_indices)
                                layer_type_list = [
                                    "attention" if i in attn_set else "state"
                                    for i in range(n_layers)
                                ]
                                if actual_n_attn > 0 and actual_n_state > 0:
                                    hybrid_ratio = f"1:{actual_n_state / actual_n_attn:.1f}"
                                elif actual_n_attn == 0:
                                    hybrid_ratio = "pure_state"
                                else:
                                    hybrid_ratio = "pure_attention"

                                state_cfg = {
                                    "d_state": d_state,
                                    "state_expansion": 2,
                                    "n_heads": pt.n_heads,
                                    "d_head": pt.d_head,
                                    "state_precision": constraints.state_precision,
                                    # v1-fix UI: see note on the dense-state branch above.
                                    "state_type": constraints.state_type,
                                }

                                for prec_name in prec_configs:
                                    prec = PRECISION_CONFIGS[prec_name]
                                    for kv_bits in constraints.kv_bits_options:
                                        local_shared = None
                                        if shared_block is not None:
                                            local_shared = {
                                                "ffn_dim": shared_block["ffn_dim"],
                                                "precision": prec["ffn_precision"],
                                            }
                                        moe_dict = {
                                            "type": "moe",
                                            "n_experts": opt.n_experts,
                                            "top_k": opt.top_k,
                                            "expert_dim": opt.expert_dim,
                                            "shared_expert": local_shared,
                                            "router": {
                                                "precision": "bf16",
                                                "load_balance_loss_coef": 0.01,
                                                "noise_type": None,
                                            },
                                            "capacity_factor": 1.25 if opt.style == "coarse" else 1.0,
                                            "precision": prec["ffn_precision"],
                                        }

                                        candidates.append(CandidateArch(
                                            d_model=pt.d_model,
                                            n_layers=n_layers,
                                            n_heads=pt.n_heads,
                                            d_head=pt.d_head,
                                            n_kv_heads=n_kv_heads,
                                            ffn_dim=pt.ffn_dim_swiglu,
                                            vocab_size=constraints.vocab_size,
                                            weight_precision=prec["weight_precision"],
                                            ffn_precision=prec["ffn_precision"],
                                            attn_precision=dict(prec["attn_precision"]),
                                            kv_cache_bits=kv_bits,
                                            total_params=total_total,
                                            total_params_b=round(total_total / 1e9, 2),
                                            # MoE
                                            moe=moe_dict,
                                            ep_degree=opt.ep_degree,
                                            active_params=active_total,
                                            active_params_b=round(active_total / 1e9, 2),
                                            moe_style=opt.style,
                                            # State
                                            state_config=state_cfg,
                                            layer_type_list=layer_type_list,
                                            placement_strategy=strategy,
                                            n_attention_layers=actual_n_attn,
                                            n_state_layers=actual_n_state,
                                            hybrid_ratio=hybrid_ratio,
                                            derived_d_state=d_state,
                                            crossover_seq_len=L_star,
                                        ))

    return candidates


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_candidate(
    cand: CandidateArch,
    hw_name: str,
    constraints: DeploymentConstraints,
) -> EvaluatedCandidate:
    """Evaluate a single candidate with throughput and quality models."""

    # --- Throughput ---
    tput_arch = TputArch(
        d_model=cand.d_model,
        n_layers=cand.n_layers,
        n_heads=cand.n_heads,
        d_head=cand.d_head,
        n_kv_heads=cand.n_kv_heads,
        ffn_dim=cand.ffn_dim,
        vocab_size=cand.vocab_size,
        precision=cand.ffn_precision,  # dominant precision for throughput
        kv_precision=kv_bits_to_precision(cand.kv_cache_bits),
        # v1: MoE branch fires when moe_config is set on the throughput's ArchConfig
        moe_config=cand.moe,
        # v1-fix Part B: first-K-dense prefix
        n_dense_ffn_layers=cand.n_dense_ffn_layers,
    )

    # v2: wire state/hybrid fields to throughput model
    if cand.state_config is not None:
        tput_arch.state_config = cand.state_config
        tput_arch.layer_type_list = cand.layer_type_list

    # v1-fix MLA: wire MLA fields to the throughput model so the decode
    # KV bandwidth term uses the latent shape instead of 2 × n_kv × d_head.
    if cand.attention_type == "mla":
        tput_arch.attention_type = "mla"
        tput_arch.mla_kv_latent_dim = cand.mla_kv_latent_dim
        tput_arch.mla_q_latent_dim = cand.mla_q_latent_dim
        tput_arch.mla_rope_head_dim = cand.mla_rope_head_dim
        tput_arch.mla_nope_head_dim = cand.mla_nope_head_dim

    # v1-fix MTP: wire MTP depths so the training compute term picks up the
    # per-depth overhead.
    if cand.mtp_n_predict_depths > 0:
        tput_arch.mtp_n_predict_depths = int(cand.mtp_n_predict_depths)
        tput_arch.mtp_depth_n_layers = int(cand.mtp_depth_n_layers)
    # v1-fix CP: wire CP into the throughput model's training step.
    if cand.cp_degree > 1:
        tput_arch.cp_degree = int(cand.cp_degree)
        tput_arch.cp_method = str(cand.cp_method)

    tput = throughput_fn(
        tput_arch, hw_name,
        tp_degree=constraints.tp,
        pp_degree=constraints.pp,
        microbatches=constraints.dp,
        decode_kv_len=constraints.prompt_len or constraints.context_length,
        ep_degree=cand.ep_degree,
        ep_topology=constraints.ep_topology,
    )

    # --- Quality ---
    # Map precision configs to quality model format
    component_precs = {}
    if cand.ffn_precision != cand.weight_precision:
        for comp in ("ffn_up", "ffn_down", "ffn_gate"):
            component_precs[comp] = cand.ffn_precision
    if cand.attn_precision.get("v", "bf16") != "bf16":
        component_precs["qkv_proj"] = cand.attn_precision.get("v", "bf16")
        component_precs["output_proj"] = cand.attn_precision.get("output", "bf16")

    # Determine model_type
    if cand.state_config is not None:
        if cand.n_attention_layers == 0:
            model_type = "state"
        else:
            model_type = "hybrid"
    elif cand.moe is not None:
        model_type = "moe"
    else:
        model_type = "dense"

    # Build quality state_config in the format expected by quality model
    qual_state_config = None
    if cand.state_config is not None:
        # v1-fix UI: read the actual SSM family from state_cfg["state_type"]
        # (added in this round). Previous code read state_cfg["state_precision"]
        # which is the numeric precision, not the family name.
        qual_state_config = {
            "enabled": True,
            "state_type": cand.state_config.get(
                "state_type",
                cand.state_config.get("state_precision", "mamba2"),
            ),
            "d_state": cand.state_config["d_state"],
            "state_layers": cand.n_state_layers,
            "attention_layers": cand.n_attention_layers,
            "pattern": cand.placement_strategy,
        }

    qual_arch = QualArch(
        d_model=cand.d_model,
        n_layers=cand.n_layers,
        n_heads=cand.n_heads,
        d_head=cand.d_head,
        n_kv_heads=cand.n_kv_heads,
        ffn_dim=cand.ffn_dim,
        vocab_size=cand.vocab_size,
        weight_precision=cand.weight_precision,
        component_precisions=component_precs if component_precs else None,
        # v1: MoE fires the quality model's _moe_residual hook.
        moe_config=cand.moe,
        model_type=model_type,
        state_config=qual_state_config,
        # v1-fix Part B: first-K-dense prefix
        n_dense_ffn_layers=cand.n_dense_ffn_layers,
        # v1-fix MLA: thread MLA shape into the quality model so the
        # compression-quality subterm and the KV-bytes feature fire.
        attention_type=("mla" if cand.attention_type == "mla" else "gqa"),
        mla_latent_dim=(cand.mla_kv_latent_dim if cand.attention_type == "mla" else None),
        mla_q_latent_dim=(cand.mla_q_latent_dim if cand.attention_type == "mla" else None),
        mla_rope_head_dim=(cand.mla_rope_head_dim if cand.attention_type == "mla" else None),
        mla_nope_head_dim=(cand.mla_nope_head_dim if cand.attention_type == "mla" else None),
        # v1-fix MTP: quality model's mtp_residual bonus
        mtp_n_predict_depths=int(cand.mtp_n_predict_depths),
        mtp_depth_n_layers=int(cand.mtp_depth_n_layers),
        mtp_train_loss_weight=float(cand.mtp_train_loss_weight),
        # v1-fix RoPE scaling
        rope_scaling_method=str(cand.rope_scaling_method),
        rope_scaling_factor=float(cand.rope_scaling_factor),
        rope_original_max_position=int(cand.rope_original_max_position),
    )

    qual = quality_fn(
        qual_arch,
        TrainingConfig(
            training_tokens=constraints.training_tokens,
            hardware=hw_name,
            kv_quantization_bits=cand.kv_cache_bits,
        ),
        memory_fits=(tput.memory_footprint_per_gpu_gb < _get_hbm_gb(hw_name)),
        lattice_aligned=True,
    )

    # --- Constraint checking ---
    violations = []
    meets = True

    if constraints.serving_tbt_ms is not None and tput.decode_time_per_token_ms > constraints.serving_tbt_ms:
        violations.append(f"TBT {tput.decode_time_per_token_ms:.1f}ms > {constraints.serving_tbt_ms}ms budget")
        meets = False

    if constraints.serving_ttft_ms is not None and tput.prefill_time_ms > constraints.serving_ttft_ms:
        violations.append(f"TTFT {tput.prefill_time_ms:.1f}ms > {constraints.serving_ttft_ms}ms budget")
        meets = False

    hbm = _get_hbm_gb(hw_name)
    if tput.memory_footprint_per_gpu_gb > hbm:
        violations.append(f"Memory {tput.memory_footprint_per_gpu_gb:.1f}GB > {hbm}GB HBM")
        meets = False

    # Infeasible quality (precision not supported on this hardware)
    if qual.predicted_loss > 1e4:
        violations.append("Quality model reports infeasible (precision not supported)")
        meets = False

    # Determine binding serving regime
    prefill_ms = tput.prefill_time_ms
    decode_ms = tput.decode_time_per_token_ms
    output_len = constraints.output_len if hasattr(constraints, 'output_len') else 512
    total_decode_ms = decode_ms * output_len
    if prefill_ms > total_decode_ms * 1.5:
        regime = "prefill-heavy"
        reason = f"Prefill {prefill_ms:.0f}ms dominates total decode {total_decode_ms:.0f}ms"
    elif total_decode_ms > prefill_ms * 1.5:
        regime = "decode-heavy"
        reason = (f"Decode {total_decode_ms:.0f}ms ({output_len} tokens × {decode_ms:.1f}ms) "
                  f"dominates prefill {prefill_ms:.0f}ms")
    else:
        regime = "mixed"
        reason = f"Prefill {prefill_ms:.0f}ms and decode {total_decode_ms:.0f}ms are comparable"

    return EvaluatedCandidate(
        arch=cand,
        quality=qual,
        throughput=tput,
        predicted_loss=qual.predicted_loss,
        training_tps=tput.training_throughput_tokens_per_sec,
        serving_tbt_ms=tput.decode_time_per_token_ms,
        memory_per_gpu_gb=tput.memory_footprint_per_gpu_gb,
        binding_serving_regime=regime,
        binding_reason=reason,
        meets_constraints=meets,
        constraint_violations=violations,
    )


def _get_hbm_gb(hw_name: str) -> float:
    """Get HBM capacity for a hardware target."""
    hbm_map = {
        "h100": 80,
        "b200": 192,
        "tpu_v5e": 16,
        "tpu_v5p": 95,
        "trainium2": 96,
        "trn2": 96,
        "trainium3": 192,
        "trn3": 192,
    }
    return hbm_map.get(hw_name, 80)


# =============================================================================
# Pareto frontier
# =============================================================================

def is_dominated(a: EvaluatedCandidate, b: EvaluatedCandidate) -> bool:
    """Return True if b dominates a (b is better or equal in all objectives, strictly better in at least one).

    v1 adds N_total as a 5th axis so an MoE candidate doesn't auto-dominate a
    same-active dense at the same training tput, TBT, and memory just because
    of the sparse-capacity quality bonus — its higher total parameter count
    is a real cost (storage, serving cluster footprint). For dense candidates
    total_params == active_params, so this axis is inert in dense-only
    searches and preserves the v0 frontier exactly.
    """
    objs_a = (a.predicted_loss, -a.training_tps, a.serving_tbt_ms,
              a.memory_per_gpu_gb, a.arch.total_params)
    objs_b = (b.predicted_loss, -b.training_tps, b.serving_tbt_ms,
              b.memory_per_gpu_gb, b.arch.total_params)

    at_least_one_better = False
    for oa, ob in zip(objs_a, objs_b):
        if ob > oa:  # b is worse in this objective
            return False
        if ob < oa:  # b is better in this objective
            at_least_one_better = True

    return at_least_one_better


def compute_pareto_frontier(candidates: List[EvaluatedCandidate]) -> List[EvaluatedCandidate]:
    """Compute the 4D Pareto frontier via non-dominated sorting."""
    if not candidates:
        return []

    frontier = []
    for c in candidates:
        dominated = False
        for other in candidates:
            if other is c:
                continue
            if is_dominated(c, other):
                dominated = True
                break
        if not dominated:
            frontier.append(c)

    # Sort by predicted loss (primary)
    frontier.sort(key=lambda x: x.predicted_loss)
    return frontier


# =============================================================================
# Main optimizer
# =============================================================================

def _identify_binding_constraints(
    optimal: Optional[EvaluatedCandidate],
    constraints: DeploymentConstraints,
    hw_name: str,
    all_evaluated: List[EvaluatedCandidate],
    feasible: List[EvaluatedCandidate],
) -> List[str]:
    """Identify which constraints are binding (active) for the optimal solution."""
    if optimal is None:
        return ["no_feasible_solution"]

    binding = []
    tput = optimal.throughput
    hbm = _get_hbm_gb(hw_name)

    # TBT constraint: binding if within 20% of budget
    if constraints.serving_tbt_ms is not None:
        ratio = tput.decode_time_per_token_ms / constraints.serving_tbt_ms
        if ratio > 0.8:
            binding.append("decode_tbt_latency")

    # TTFT constraint: binding if within 20% of budget
    if constraints.serving_ttft_ms is not None:
        ratio = tput.prefill_time_ms / constraints.serving_ttft_ms
        if ratio > 0.8:
            binding.append("prefill_ttft_latency")

    # Memory: binding if within 15% of HBM
    mem_ratio = tput.memory_footprint_per_gpu_gb / hbm
    if mem_ratio > 0.85:
        binding.append("hbm_capacity")

    # TP divisibility: check if better architectures were eliminated by TP
    if constraints.tp > 1:
        # If many candidates failed feasibility due to TP, it's binding
        tp_violations = sum(1 for e in all_evaluated
                           if not e.meets_constraints
                           and any("TP" in v or "tp" in v.lower() for v in e.constraint_violations))
        if tp_violations > len(all_evaluated) * 0.1:
            binding.append("tp_divisibility")

    # Tile efficiency: check if the optimal uses a non-ideal tile utilization
    if tput.per_layer_breakdown and tput.per_layer_breakdown.bottleneck == "compute":
        binding.append("compute_bound")
    elif tput.per_layer_breakdown and tput.per_layer_breakdown.bottleneck == "memory":
        binding.append("memory_bandwidth")

    # Decode KV bandwidth: check if decode is bandwidth-bound
    if optimal.binding_serving_regime == "decode-heavy":
        binding.append("decode_kv_bandwidth")

    if not binding:
        binding.append("none_identified")

    return binding


def optimize(
    hw_name: str,
    constraints: DeploymentConstraints,
) -> OptimizationResult:
    """
    Main entry point. Brute-force search over the lattice-restricted space.

    Returns the Pareto frontier and the argmax (lowest predicted loss among
    feasible candidates meeting all deployment constraints).
    """
    t0 = time.time()

    # 1. Generate candidates (dense + optional MoE + optional state/hybrid +
    #    optional combined MoE×state when both flags are on).
    candidates = generate_candidates(hw_name, constraints)
    if constraints.allow_moe:
        candidates = candidates + generate_moe_candidates(hw_name, constraints)
    if constraints.allow_state:
        candidates = candidates + generate_state_candidates(hw_name, constraints)
    if constraints.allow_moe and constraints.allow_state:
        # v1-fix Part D: combined MoE + hybrid-state search (Jamba-MoE pattern).
        # Adds MoE×hybrid candidates alongside the pure-MoE and pure-hybrid
        # ones so the Pareto frontier compares all four families on equal
        # footing.
        candidates = candidates + generate_moe_hybrid_candidates(hw_name, constraints)

    # 2. Deduplicate (same arch dimensions + precision can appear from multiple lattice points)
    seen = set()
    unique = []
    for c in candidates:
        # MoE candidates carry a moe dict; include its identifying fields in
        # the key so an MoE variant isn't deduped against a dense one with
        # the same skeleton.
        if c.moe is not None:
            moe_key = (
                c.moe["n_experts"], c.moe["top_k"], c.moe["expert_dim"],
                (c.moe.get("shared_expert") or {}).get("ffn_dim"),
                c.ep_degree,
                # v1-fix Part B: n_dense distinguishes mixed-FFN variants
                c.n_dense_ffn_layers,
            )
        else:
            moe_key = None
        # State candidates: include state-identifying fields so a hybrid
        # variant isn't deduped against a dense/MoE with the same skeleton.
        if c.state_config is not None:
            state_key = (
                c.state_config.get("d_state"),
                c.n_attention_layers,
                c.n_state_layers,
                c.placement_strategy,
            )
        else:
            state_key = None
        key = (c.d_model, c.n_layers, c.n_heads, c.d_head, c.n_kv_heads,
               c.ffn_dim, c.weight_precision, c.ffn_precision,
               tuple(sorted(c.attn_precision.items())), c.kv_cache_bits,
               moe_key, state_key)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    candidates = unique
    if constraints.max_candidates is not None and constraints.max_candidates > 0:
        cap = int(constraints.max_candidates)
        if len(candidates) > cap:
            step = len(candidates) / cap
            candidates = [candidates[min(int(i * step), len(candidates) - 1)]
                          for i in range(cap)]

    # 3. Evaluate all candidates
    evaluated = []
    total = len(candidates)
    for idx, cand in enumerate(candidates, start=1):
        try:
            ev = evaluate_candidate(cand, hw_name, constraints)
            evaluated.append(ev)
        except Exception:
            # Skip candidates that cause errors (e.g., missing hardware spec)
            continue
        if constraints.progress_every and idx % constraints.progress_every == 0:
            print(
                f"[arch-compiler] evaluated {idx:,}/{total:,} candidates "
                f"({len(evaluated):,} scored)",
                file=sys.stderr,
            )

    # 4. Filter to feasible
    feasible = [e for e in evaluated if e.meets_constraints]

    # 5. Pareto frontier (over feasible only)
    pareto = compute_pareto_frontier(feasible)

    # 6. Argmax: lowest predicted loss among feasible, ties broken by training throughput
    optimal = None
    if feasible:
        optimal = min(feasible, key=lambda x: (x.predicted_loss, -x.training_tps))

    elapsed = time.time() - t0

    # 7. Identify binding constraints
    binding = _identify_binding_constraints(optimal, constraints, hw_name, evaluated, feasible)

    return OptimizationResult(
        optimal=optimal,
        pareto_frontier=pareto,
        all_evaluated=evaluated,
        candidates_generated=len(candidates),
        candidates_feasible=len(feasible),
        candidates_evaluated=len(evaluated),
        search_time_sec=round(elapsed, 2),
        hardware=hw_name,
        constraints=constraints,
        binding_constraints=binding,
    )


def result_to_config(
    result: OptimizationResult,
    nsa: Optional[Dict[str, Any]] = None,
    yoco: Optional[Dict[str, Any]] = None,
) -> Optional[dict]:
    """Convert the optimal candidate from an OptimizationResult to a JSON config dict.

    NSA and YOCO are not (yet) candidate-enumerated by the search. The two
    optional kwargs let the caller stamp an NSA or YOCO block onto the
    emitted config (the schema, quality model, and throughput model all
    pick them up). When `nsa` is set, the candidate's attention_type is
    forced to NSA on emission (NSA precedes MLA in build_config).
    """
    if result.optimal is None:
        return None

    opt = result.optimal
    c = opt.arch
    terms = getattr(opt.quality, "terms", {})
    arch_term = terms.get("architecture_residual")
    precision_term = terms.get("precision_residual")
    risk_term = terms.get("risk_residual")

    input_constraints = {
        "target_params": f"{result.constraints.target_params_b}B",
        "training_tokens": f"{result.constraints.training_tokens / 1e12:.1f}T",
        "context_length": result.constraints.context_length,
        "serving_tbt_ms": result.constraints.serving_tbt_ms,
        "serving_ttft_ms": result.constraints.serving_ttft_ms,
        "serving_batch": result.constraints.serving_batch,
        "allow_moe": result.constraints.allow_moe,
        "max_total_params_b": result.constraints.max_total_params_b,
        "allow_state": result.constraints.allow_state,
    }

    predicted = {
        "quality_rank_score": round(-opt.predicted_loss, 4),
        "predicted_loss": round(opt.predicted_loss, 4),
        "training_throughput_tokens_per_sec": round(opt.training_tps),
        "serving_tbt_ms": round(opt.serving_tbt_ms, 1),
        "serving_ttft_ms": round(opt.throughput.prefill_time_ms, 1),
        "memory_per_gpu_gb": round(opt.memory_per_gpu_gb, 1),
        "active_params_b": c.active_params_b or c.total_params_b,
        "total_params_b": c.total_params_b,
        "moe_style": c.moe_style,
        "ep_degree": c.ep_degree,
        "confidence": opt.quality.confidence,
        "scaling_spine_loss": round(opt.quality.chinchilla_baseline, 4),
        "spine_active_params_b": round(getattr(opt.quality, "spine_active_params", 0) / 1e9, 3),
        "total_residual_pct": round(opt.quality.total_penalty_fraction * 100, 2),
        "architecture_residual_pct": round((arch_term.value if arch_term else 0.0) * 100, 3),
        "precision_residual_pct": round((precision_term.value if precision_term else 0.0) * 100, 3),
        "risk_uncertainty_pct": round((risk_term.uncertainty if risk_term else 0.0) * 100, 3),
        "total_penalty_pct": round(opt.quality.total_penalty_fraction * 100, 2),
        "dominant_penalty": opt.quality.dominant_penalty,
        "uncertainty_low_pct": round(opt.quality.uncertainty_low_pct, 2),
        "uncertainty_high_pct": round(opt.quality.uncertainty_high_pct, 2),
        "uncertainty_total_pct": round(getattr(opt.quality, "uncertainty_total", 0.0) * 100, 2),
        "uncertainty_breakdown": {
            k: round(v * 100, 3)
            for k, v in getattr(opt.quality, "uncertainty_breakdown", {}).items()
        },
        "quality_model_version": getattr(opt.quality, "quality_model_version", "quality_v0"),
        "quality_terms": {
            k: {
                "value_pct": round(v.value * 100, 4),
                "uncertainty_pct": round(v.uncertainty * 100, 4),
                "confidence": v.confidence,
                "source": v.source,
                "notes": v.notes,
                "features": v.features,
            }
            for k, v in getattr(opt.quality, "terms", {}).items()
            if v.confidence != "not_applicable" or abs(v.value) > 0 or v.uncertainty > 0
        },
        "binding_serving_regime": opt.binding_serving_regime,
        "binding_constraints": result.binding_constraints,
    }

    # v2: add state/hybrid metadata to predicted block
    if c.state_config is not None:
        predicted["hybrid_ratio"] = c.hybrid_ratio
        predicted["placement_strategy"] = c.placement_strategy
        predicted["n_attention_layers"] = c.n_attention_layers
        predicted["n_state_layers"] = c.n_state_layers
        predicted["derived_d_state"] = c.derived_d_state
        predicted["crossover_seq_len"] = round(c.crossover_seq_len, 1)

    search_stats = {
        "candidates_generated": result.candidates_generated,
        "candidates_feasible": result.candidates_feasible,
        "pareto_size": len(result.pareto_frontier),
        "search_time_sec": result.search_time_sec,
    }

    # v2: use build_hybrid_config when the winner is a hybrid/state candidate
    if c.state_config is not None and c.layer_type_list is not None:
        attn_indices = [i for i, lt in enumerate(c.layer_type_list) if lt == "attention"]
        state_indices = [i for i, lt in enumerate(c.layer_type_list) if lt == "state"]
        return build_hybrid_config(
            d_model=c.d_model,
            n_layers=c.n_layers,
            vocab_size=c.vocab_size,
            attention_layer_indices=attn_indices,
            n_heads=c.n_heads,
            d_head=c.d_head,
            n_kv_heads=c.n_kv_heads,
            kv_cache_bits=c.kv_cache_bits,
            attn_precision=c.attn_precision,
            state_layer_indices=state_indices,
            # v1-fix UI: state_cfg now carries an explicit state_type field
            # (Part J families pass through end-to-end). Older state_cfg dicts
            # without the field still fall back to mamba2.
            state_type=c.state_config.get("state_type", "mamba2"),
            state_d_state=c.state_config["d_state"],
            state_n_heads=c.state_config.get("n_heads", c.n_heads),
            state_d_head=c.state_config.get("d_head", c.d_head),
            ffn_dim=c.ffn_dim,
            ffn_precision=c.ffn_precision,
            weight_precision=c.weight_precision,
            moe=c.moe,
            tp=result.constraints.tp,
            pp=result.constraints.pp,
            dp=result.constraints.dp,
            ep=c.ep_degree,
            hardware_name=result.hardware,
            input_constraints=input_constraints,
            predicted=predicted,
            search_stats=search_stats,
        )

    # v1-fix MLA: pass the MLA block to build_config when the winning
    # candidate is MLA. build_config(mla=…) emits attention.type="mla" +
    # latent fields.
    mla_kw = None
    if c.attention_type == "mla":
        mla_kw = {
            "kv_latent_dim":  c.mla_kv_latent_dim,
            "q_latent_dim":   c.mla_q_latent_dim,
            "rope_head_dim":  c.mla_rope_head_dim,
            "nope_head_dim":  c.mla_nope_head_dim,
        }
    # v1-fix MTP: pass the MTP block when the candidate has prediction depths.
    mtp_kw = None
    if c.mtp_n_predict_depths > 0:
        mtp_kw = {
            "enabled": True,
            "n_predict_depths": int(c.mtp_n_predict_depths),
            "depth_n_layers": int(c.mtp_depth_n_layers),
            "share_embeddings": True,
            "share_lm_head": True,
            "train_loss_weight": float(c.mtp_train_loss_weight),
            "inference_mode": "drop",
        }
    return build_config(
        d_model=c.d_model,
        n_layers=c.n_layers,
        n_heads=c.n_heads,
        d_head=c.d_head,
        n_kv_heads=c.n_kv_heads,
        ffn_dim=c.ffn_dim,
        vocab_size=c.vocab_size,
        weight_precision=c.weight_precision,
        attn_precision=c.attn_precision,
        ffn_precision=c.ffn_precision,
        kv_cache_bits=c.kv_cache_bits,
        tp=result.constraints.tp,
        pp=result.constraints.pp,
        dp=result.constraints.dp,
        ep=c.ep_degree,
        cp=int(c.cp_degree),
        cp_method=str(c.cp_method),
        rope_scaling_method=str(c.rope_scaling_method),
        rope_scaling_factor=float(c.rope_scaling_factor),
        rope_original_max_position=int(c.rope_original_max_position),
        moe=c.moe,
        n_dense_ffn_layers=c.n_dense_ffn_layers,
        mla=mla_kw,
        nsa=nsa,
        yoco=yoco,
        mtp=mtp_kw,
        hardware_name=result.hardware,
        input_constraints=input_constraints,
        predicted=predicted,
        search_stats=search_stats,
    )


def result_to_pareto_csv(result: OptimizationResult) -> str:
    """Convert the Pareto frontier to CSV format. v1 adds MoE columns; dense
    rows leave them blank/0 so existing parsers degrade cleanly."""
    lines = [
        "rank,d_model,n_layers,n_heads,d_head,n_kv_heads,ffn_dim,"
        "weight_prec,ffn_prec,kv_bits,active_params_B,total_params_B,"
        "moe_style,n_experts,top_k,expert_dim,ep,"
        "predicted_loss,training_tps,serving_tbt_ms,memory_gb,confidence"
    ]
    for i, ev in enumerate(result.pareto_frontier):
        c = ev.arch
        if c.moe is not None:
            n_experts = c.moe["n_experts"]
            top_k = c.moe["top_k"]
            expert_dim = c.moe["expert_dim"]
        else:
            n_experts = top_k = expert_dim = 0
        active = c.active_params_b or c.total_params_b
        lines.append(
            f"{i+1},{c.d_model},{c.n_layers},{c.n_heads},{c.d_head},{c.n_kv_heads},"
            f"{c.ffn_dim},{c.weight_precision},{c.ffn_precision},{c.kv_cache_bits},"
            f"{active},{c.total_params_b},"
            f"{c.moe_style},{n_experts},{top_k},{expert_dim},{c.ep_degree},"
            f"{ev.predicted_loss:.4f},{ev.training_tps:.0f},"
            f"{ev.serving_tbt_ms:.1f},{ev.memory_per_gpu_gb:.1f},{ev.quality.confidence}"
        )
    return "\n".join(lines)
