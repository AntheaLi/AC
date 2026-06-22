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
    estimate_params, estimate_mla_total_params, estimate_mla_per_layer_params,
    LatticePoint, GQAConfig,
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

VALID_ROPE_SCALING_METHODS = {"none", "pi", "ntk", "yarn", "longrope"}
VALID_PLACEMENT_STRATEGIES = {
    "first_periodic_last",
    "interleaved",
    "periodic",
}

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
    # Objective profile for greenfield selection. Pareto always keeps the
    # full frontier, but `optimal` uses this weighted objective. Including
    # TTFT prevents long-context runs from optimizing decode TBT while
    # silently accepting huge cold-prefill latency.
    # v1-fix demo-audit: stay on "research_quality" by default (loss-dominant),
    # but rely on the uncertainty-band tiebreak in
    # `build_display_sort_key` to break ties on memory/TBT/-tps within ~25%
    # of the model's own uncertainty. "balanced" reweights throughput so
    # aggressively that it accepts 8–10% loss regressions for throughput
    # gains, which is too far on a quality model with ~3% uncertainty.
    objective_profile: str = "research_quality"
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
        self._validate_positive_inputs()
        if self.placement_strategies is not None:
            cleaned = [str(v).strip() for v in self.placement_strategies if str(v).strip()]
            if not cleaned:
                raise ValueError("placement_strategies must contain at least one strategy")
            bad = [v for v in cleaned if v not in VALID_PLACEMENT_STRATEGIES]
            if bad:
                raise ValueError(
                    f"Unknown placement strategy value(s): {bad}. "
                    f"Supported: {', '.join(sorted(VALID_PLACEMENT_STRATEGIES))}"
                )
            self.placement_strategies = cleaned
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
            self.cp_options = [self.cp]
        elif any(int(v) <= 0 for v in self.cp_options):
            raise ValueError("cp_options values must be > 0")
        if self.rope_scaling_methods is None:
            # Only sweep extension methods when the workload exceeds the
            # native pretrain context. Otherwise pin to "none".
            if not self.allow_rope_scaling or self.context_length <= self.rope_original_max_position:
                self.rope_scaling_methods = ["none"]
            else:
                self.rope_scaling_methods = ["yarn", "ntk", "longrope", "pi"]
        else:
            cleaned = [str(v).strip().lower() for v in self.rope_scaling_methods if str(v).strip()]
            if not cleaned:
                raise ValueError("rope_scaling_methods must contain at least one method")
            bad = [v for v in cleaned if v not in VALID_ROPE_SCALING_METHODS]
            if bad:
                raise ValueError(
                    f"Unknown rope scaling method value(s): {bad}. "
                    f"Supported: {', '.join(sorted(VALID_ROPE_SCALING_METHODS))}"
                )
            if not self.allow_rope_scaling and any(v != "none" for v in cleaned):
                raise ValueError(
                    "rope_scaling_methods with a non-'none' method require allow_rope_scaling=True"
                )
            self.rope_scaling_methods = cleaned
        if self.objective_profile not in OBJECTIVE_PROFILES:
            raise ValueError(
                f"Unknown objective_profile {self.objective_profile!r}; "
                f"expected one of {sorted(OBJECTIVE_PROFILES)}"
            )

    def _validate_positive_inputs(self) -> None:
        checks = {
            "target_params_b": self.target_params_b,
            "training_tokens": self.training_tokens,
            "context_length": self.context_length,
            "serving_batch": self.serving_batch,
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "vocab_size": self.vocab_size,
            "output_len": self.output_len,
            "concurrency": self.concurrency,
            "cp": self.cp,
            "rope_original_max_position": self.rope_original_max_position,
        }
        for name, value in checks.items():
            if value is None or value <= 0:
                raise ValueError(f"{name} must be > 0")
        optional_positive = {
            "serving_tbt_ms": self.serving_tbt_ms,
            "serving_ttft_ms": self.serving_ttft_ms,
            "prompt_len": self.prompt_len,
            "max_total_params_b": self.max_total_params_b,
            "max_candidates": self.max_candidates,
        }
        for name, value in optional_positive.items():
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.param_tolerance < 0:
            raise ValueError("param_tolerance must be >= 0")
        if self.progress_every < 0:
            raise ValueError("progress_every must be >= 0")
        for name in ("moe_n_experts_options", "moe_top_k_options",
                     "ep_options", "mla_kv_latent_options",
                     "mla_q_latent_options"):
            values = getattr(self, name)
            if values is not None and any(int(v) <= 0 for v in values):
                raise ValueError(f"{name} values must be > 0")
        for name in ("dense_ffn_layer_options", "mtp_depth_options"):
            values = getattr(self, name)
            if values is not None and any(int(v) < 0 for v in values):
                raise ValueError(f"{name} values must be >= 0")


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
    attention_type: str = "full"           # "full" | "mla" | "swa"
    mla_kv_latent_dim: int = 0             # c_kv (0 when type=full)
    mla_q_latent_dim: int = 0              # c_q
    mla_rope_head_dim: int = 0             # d_rope
    mla_nope_head_dim: int = 0             # d_nope
    # v1-fix SWA: when set, KV cache is capped at min(seq_len, window_size)
    # per token; decode TBT reads the smaller cache; prefill compute is
    # O(N*W) instead of O(N^2); quality model adds a small residual when
    # the workload context exceeds the window.
    swa_window: int = 0                    # 0 = no sliding window (full attention)
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


# Selection profiles mirror the web UI tradeoff presets. Weights sum to 1.0;
# lower score is better. `ttft` is deliberately separate from decode TBT.
OBJECTIVE_PROFILES: Dict[str, Dict[str, float]] = {
    "balanced":      {"loss": 0.30, "tbt": 0.15, "ttft": 0.10, "tps": 0.15, "mem": 0.15, "params": 0.15},
    "quality":       {"loss": 0.90, "tbt": 0.02, "ttft": 0.01, "tps": 0.03, "mem": 0.02, "params": 0.02},
    "research_quality": {"loss": 1.00, "tbt": 0.0, "ttft": 0.0, "tps": 0.0, "mem": 0.0, "params": 0.0},
    "loss_only":     {"loss": 1.00, "tbt": 0.0, "ttft": 0.0, "tps": 0.0, "mem": 0.0, "params": 0.0},
    "latency":       {"loss": 0.15, "tbt": 0.30, "ttft": 0.25, "tps": 0.10, "mem": 0.10, "params": 0.10},
    "serving_cost":  {"loss": 0.15, "tbt": 0.15, "ttft": 0.10, "tps": 0.05, "mem": 0.30, "params": 0.25},
    "training_cost": {"loss": 0.20, "tbt": 0.05, "ttft": 0.05, "tps": 0.45, "mem": 0.10, "params": 0.15},
}


def _d_model_max_for_target(target_params_b: float) -> int:
    """Allow frontier-scale widths without exploding small-model search."""
    if target_params_b >= 900:
        return 32768
    if target_params_b >= 650:
        return 28672
    if target_params_b >= 300:
        return 24576
    return 16384


def _candidate_metric(ev: EvaluatedCandidate, key: str) -> float:
    if key == "loss":
        return ev.predicted_loss
    if key == "tbt":
        return ev.serving_tbt_ms
    if key == "ttft":
        return ev.throughput.prefill_time_ms
    if key == "tps":
        return ev.training_tps
    if key == "mem":
        return ev.memory_per_gpu_gb
    if key == "params":
        return float(ev.arch.total_params)
    raise KeyError(key)


def _aspect_ratio_prior_penalty(ev: EvaluatedCandidate, pool_size: int = 16) -> float:
    """Penalty against d_model/n_layers ratios outside the published
    frontier-lab band.

    Empirical d_model/n_layers from the 14-model reference set spans roughly
    80–140 (Qwen3-32B ~80, Llama-3-70B ~102, DeepSeek-V3 ~117, Mistral
    7B/Llama 7-8B ~128, Mistral-Large-123B ~140). Outside that corridor we
    add a penalty proportional to log-distance from the band; inside the
    corridor the penalty is zero.

    v1-fix demo-audit D3: the cap is now adaptive. When the feasible
    Pareto frontier is rich (>= 8 candidates), this stays a 3% tiebreaker
    as before — it never overrides a real Pareto move. When the frontier
    collapses to one or two points (e.g., H100 750B at PP=1, TP=32 where
    HBM forces a single feasible lattice point), the cap relaxes to 25%
    so a 19× depth/width violation is no longer a 3% nudge against
    nothing.
    """
    L = max(1, int(ev.arch.n_layers))
    d = max(1, int(ev.arch.d_model))
    ratio = d / L
    lo, hi = 80.0, 160.0
    if lo <= ratio <= hi:
        return 0.0
    import math
    if ratio < lo:
        dist = math.log(lo / ratio)
    else:
        dist = math.log(ratio / hi)
    # Base penalty 1% per natural-log unit outside the band.
    raw = 0.01 * dist
    # Sparse-frontier multiplier: when the optimizer has no real Pareto
    # signal to break the tie on, this prior becomes the only voice
    # against unshippable shapes. Scale linearly between 1× (≥8 candidates)
    # and 8× (1 candidate).
    sparse_mult = max(1.0, min(8.0, 8.0 / max(pool_size, 1)))

    # v1-fix demo-audit-2 (Jun 2026): two-tier penalty.
    # Mild violations (≤2x outside the band, log-dist ≤ ~0.69) stay capped
    # at the previous 3%*sparse_mult so the prior remains a soft tiebreaker
    # for borderline choices like 4096x40 (ratio=102) vs 4608x33 (ratio=140).
    # Severe violations (>2x outside the band — i.e. ratio < 40 or > 320) grow
    # *quadratically* with no cap so unshippable shapes like 11776x4
    # (ratio=2944, log-dist=2.9) cost ~9% even on a rich frontier and
    # 25-50% on sparse frontiers. The previous flat 3%/24% cap meant a
    # 73x band violation cost the same as a 2x violation, which is what let
    # the H100 7B serving_20ms Pareto fill up with wide-shallow monsters.
    SOFT_DIST = math.log(2.0)  # ≈ 0.693
    if dist <= SOFT_DIST:
        cap = 0.03 * sparse_mult
        return min(cap, raw * sparse_mult)
    soft_part = 0.01 * SOFT_DIST  # the penalty earned up to the soft boundary
    hard_dist = dist - SOFT_DIST
    # Quadratic growth past the soft boundary: each additional ln-unit costs
    # 4% (vs 1% in the linear regime). At log-dist 2.9 (ratio 2944, i.e.
    # 11776x4) this is ~0.01*0.69 + 0.04*(2.2)^2 ≈ 0.20 = 20% raw before
    # sparse_mult; well above any reasonable predicted-loss delta on the
    # Pareto so the optimizer will only pick a wide-shallow if it really
    # has no other choice.
    hard_part = 0.04 * (hard_dist ** 2)
    return (soft_part + hard_part) * sparse_mult


def _objective_score(
    ev: EvaluatedCandidate,
    pool: List[EvaluatedCandidate],
    profile: str,
) -> float:
    """Weighted %-delta from best-on-axis over the feasible frontier."""
    weights = OBJECTIVE_PROFILES[profile]
    best: Dict[str, float] = {}
    for key in weights:
        vals = [_candidate_metric(x, key) for x in pool]
        best[key] = max(vals) if key == "tps" else min(vals)

    score = 0.0
    for key, weight in weights.items():
        if weight <= 0:
            continue
        v = _candidate_metric(ev, key)
        b = best[key]
        if b == 0:
            continue
        delta = (b - v) / abs(b) if key == "tps" else (v - b) / abs(b)
        score += weight * max(0.0, delta)
    # Aspect-ratio prior, calibrated to the 14-model published frontier-lab
    # reference set. Adaptive cap based on frontier richness so a singleton
    # frontier doesn't silently ship a pathological aspect ratio.
    score += _aspect_ratio_prior_penalty(ev, pool_size=len(pool))
    return score


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


def _kv_heads_compatible_with_tp(n_kv_heads: int, tp_degree: int) -> bool:
    """Return whether KV heads have a clear TP placement.

    `n_kv_heads == 1` is the replicated MQA-style case. Otherwise require an
    even shard across TP ranks so emitted configs do not rely on an unstated
    uneven KV-head placement policy.
    """
    n_kv_heads = int(n_kv_heads)
    tp_degree = max(1, int(tp_degree))
    return n_kv_heads == 1 or (n_kv_heads >= tp_degree and n_kv_heads % tp_degree == 0)


def _gqa_ratios_for_point(pt: LatticePoint, constraints: DeploymentConstraints) -> List[int]:
    """Enumerate GQA ratios whose KV-head count is TP-placeable.

    v1-fix C1 (demo audit follow-up): the previous version iterated a
    fixed list of GQA ratios [2, 4, 8, 16] and accepted them only when
    BOTH n_heads % r == 0 AND (n_heads / r) was TP-divisible. For
    "non-round" n_heads counts (e.g. n_heads=72 at TP=8) none of the
    fixed ratios produced a TP-compatible n_kv, so the lattice emitted
    only MHA (kv=72) and MQA (kv=1) — even though n_kv=24 (group=3)
    and n_kv=8 (group=9) are both valid and TP-divisible. Combined
    with the one-sided KV-heads quality term, this guaranteed MHA
    selection on every greenfield run.

    The fix considers a small, fixed *candidate target* set of GQA
    n_kv values — the canonical published targets at the published
    n_heads/4, n_heads/8 (Llama-3 / Qwen-3 / Mistral / Gemma-2 style)
    plus the legacy [2,4,8,16] ratios for round head counts — and
    keeps the ones that are TP-placeable. That stays bounded at ~6-8
    candidates per lattice point (no perf regression from the old
    [2,4,8,16] sweep) while restoring the non-round GQA targets that
    the demo audit named.
    """
    nh = int(pt.n_heads)
    tp = int(constraints.tp)
    ratios: set = set()
    ratios.add(1)  # MHA
    if nh > 1:
        ratios.add(nh)  # MQA / replicated single KV head

    # Legacy round ratios (preserved so existing calibration snapshots
    # don't shift).
    for r in (2, 4, 8, 16):
        if nh % r == 0:
            n_kv = nh // r
            if _kv_heads_compatible_with_tp(n_kv, tp):
                ratios.add(r)

    # Demo-audit fix: also try GQA targets that come from "natural"
    # n_kv values for non-round head counts. Each candidate n_kv is
    # promoted to a ratio iff (a) it divides nh evenly, (b) the
    # resulting ratio is at least 2, and (c) the n_kv is TP-placeable.
    candidate_n_kvs = (
        # n_heads-derived (every published GQA-N model lands on one of
        # these): n_kv = nh/4, nh/8, nh/16 → group sizes 4, 8, 16.
        nh // 2, nh // 3, nh // 4, nh // 6, nh // 8,
        # Common production absolute n_kv counts: Llama-3 family uses
        # nkv=8; DeepSeek-V2/V3-style GQA-1 uses nkv=128, etc.
        8, 16, 24, 32, 64,
    )
    for n_kv in candidate_n_kvs:
        if n_kv <= 0 or n_kv > nh:
            continue
        if nh % n_kv != 0:
            continue
        r = nh // n_kv
        if r < 2:
            continue
        if _kv_heads_compatible_with_tp(n_kv, tp):
            ratios.add(r)

    return sorted(ratios)


def _search_option_lists(constraints: DeploymentConstraints) -> Tuple[List[int], List[int], List[str], float]:
    """Shared MTP/CP/RoPE option lists for every architecture family."""
    mtp_opts = constraints.mtp_depth_options if constraints.allow_mtp else [0]
    cp_opts = constraints.cp_options or [constraints.cp]
    rope_opts = constraints.rope_scaling_methods or ["none"]
    rope_factor = max(
        1.0,
        constraints.context_length / max(1, constraints.rope_original_max_position),
    )
    return list(mtp_opts), list(cp_opts), list(rope_opts), float(rope_factor)


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
        d_model_min=1024, d_model_max=_d_model_max_for_target(constraints.target_params_b),
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
    mtp_opts, cp_opts, rope_opts, rope_factor = _search_option_lists(constraints)

    for pt in aligned:
        gqa_ratios = _gqa_ratios_for_point(pt, constraints)

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
            # v1-fix demo-audit D1: cap n_layers_raw to a sane band before
            # enumerating. Without this cap, narrow lattice points combined
            # with very large param targets at PP=1 produce n_layers_raw in
            # the 1000-2000 range, and the optimizer happily emits 1980-layer
            # "transformers" because the shape-stability penalty downstream
            # is capped at 6%. The MoE branch already does `1 <= n_layers_raw
            # <= 256`; we match that here. 256 is comfortably above every
            # published frontier model (Llama-3-405B L=126, DeepSeek-V3 L=61,
            # GPT-3 175B L=96) — anything beyond this band is the lattice
            # exploiting depth to chase param count, not a real architecture.
            if not (1 <= n_layers_raw <= 256):
                continue
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
                        #
                        # v1-fix demo-audit D4: only emit MLA once per
                        # lattice point — NOT once per GQA ratio. MLA's
                        # KV cost is a function of (c_kv, d_rope), not
                        # n_kv_heads, so emitting it N times for N GQA
                        # ratios produced N identical MLA candidates that
                        # only differed in the (downstream-meaningless)
                        # n_kv_heads tag, polluting the Pareto sample.
                        if getattr(constraints, "allow_mla", False) and gqa_r == gqa_ratios[0]:
                            # v1-fix demo-audit (June 2026 follow-up): MLA
                            # candidates previously inherited the dense MHA
                            # `total` and `d_head` from the surrounding loop.
                            # That made (a) `total_params` count an MHA Q+K+V+O
                            # block (~26% too large at low-n_heads MLA shapes
                            # like cell 9), (b) the `d_head` field describe a
                            # matmul that doesn't exist in MLA (DeepSeek-V2/V3
                            # uses a per-head Q dim of d_nope + d_rope, not
                            # the lattice's dh), and (c) the Chinchilla
                            # baseline use the inflated N, artificially
                            # Pareto-favoring MLA. Fix: snap d_head to
                            # (d_nope + d_rope), recompute total_params via
                            # the true MLA layout, and re-solve n_layers for
                            # the MLA shape so the candidate hits the param
                            # target band (it almost never does at the dense
                            # n_layers because MLA attention is smaller).
                            d_nope_mla = int(constraints.mla_nope_head_dim)
                            d_rope_mla = int(constraints.mla_rope_head_dim)
                            dh_mla = d_nope_mla + d_rope_mla
                            for c_kv in constraints.mla_kv_latent_options:
                                for c_q in constraints.mla_q_latent_options:
                                    # Sanity: latent should compress KV vs
                                    # the MLA per-head KV (2 * dh_mla per head),
                                    # not vs the dense lattice's KV.
                                    uncompressed_mla = 2 * pt.n_heads * dh_mla
                                    if c_kv >= uncompressed_mla:
                                        continue
                                    # Solve n_layers for THIS MLA shape so
                                    # the candidate hits the param target.
                                    per_layer_mla = estimate_mla_per_layer_params(
                                        pt.d_model, pt.n_heads,
                                        c_kv, c_q, d_nope_mla, d_rope_mla,
                                        pt.ffn_dim_swiglu,
                                    )
                                    embed_p = 2 * constraints.vocab_size * pt.d_model
                                    if per_layer_mla <= 0:
                                        continue
                                    n_layers_mla_raw = (target - embed_p) / per_layer_mla
                                    if not (1 <= n_layers_mla_raw <= 256):
                                        continue
                                    for dL in (-2, -1, 0, 1, 2):
                                        n_layers_mla = max(4, round(n_layers_mla_raw) + dL)
                                        if constraints.pp > 1 and n_layers_mla % constraints.pp != 0:
                                            continue
                                        mla_total = estimate_mla_total_params(
                                            pt.d_model, pt.n_heads, n_layers_mla,
                                            c_kv, c_q, d_nope_mla, d_rope_mla,
                                            pt.ffn_dim_swiglu, constraints.vocab_size,
                                        )
                                        if mla_total < lo or mla_total > hi:
                                            continue
                                        candidates.append(CandidateArch(
                                            d_model=pt.d_model,
                                            n_layers=n_layers_mla,
                                            n_heads=pt.n_heads,
                                            # Snap d_head to the real MLA per-head Q dim.
                                            d_head=dh_mla,
                                            n_kv_heads=n_kv_heads,
                                            ffn_dim=pt.ffn_dim_swiglu,
                                            vocab_size=constraints.vocab_size,
                                            weight_precision=prec["weight_precision"],
                                            ffn_precision=prec["ffn_precision"],
                                            attn_precision=dict(prec["attn_precision"]),
                                            kv_cache_bits=kv_bits,
                                            total_params=mla_total,
                                            total_params_b=round(mla_total / 1e9, 2),
                                            # MLA-specific fields
                                            attention_type="mla",
                                            mla_kv_latent_dim=c_kv,
                                            mla_q_latent_dim=c_q,
                                            mla_rope_head_dim=d_rope_mla,
                                            mla_nope_head_dim=d_nope_mla,
                                            mtp_n_predict_depths=int(mtp_k),
                                            mtp_depth_n_layers=int(constraints.mtp_depth_n_layers),
                                            mtp_train_loss_weight=float(constraints.mtp_train_loss_weight),
                                            cp_degree=int(cp_d),
                                            cp_method=str(constraints.cp_method),
                                            rope_scaling_method=str(rope_m),
                                            rope_scaling_factor=rope_factor if rope_m != "none" else 1.0,
                                            rope_original_max_position=int(constraints.rope_original_max_position),
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
        d_model_min=1024, d_model_max=_d_model_max_for_target(constraints.target_params_b),
        d_head_options=[64, 128, 256],
    )
    aligned = [pt for pt in lattice if pt.tile_aligned]

    hw_prec_configs = get_precision_configs_for_hardware(hw_name)
    prec_configs = [p for p in (constraints.precision_configs or []) if p in hw_prec_configs]
    if not prec_configs:
        prec_configs = hw_prec_configs[:3]

    ep_opts = constraints.ep_options or default_ep_options(hw_name)
    mtp_opts, cp_opts, rope_opts, rope_factor = _search_option_lists(constraints)

    # v1-fix Part B: sweep n_dense_ffn_layers. Default is [0] (pure MoE, the
    # original v1-MoE behavior). [0, 1, 2, 3] covers the common dense-prefix
    # range used by DeepSeek-V3 / Qwen3-MoE / similar.
    dense_ffn_layer_opts = constraints.dense_ffn_layer_options
    if dense_ffn_layer_opts is None:
        dense_ffn_layer_opts = [0]

    candidates: List[CandidateArch] = []

    for pt in aligned:
        # Use the dense lattice's GQA shape sweep (mirrors dense path).
        gqa_ratios = _gqa_ratios_for_point(pt, constraints)

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
                if opt.top_k >= opt.n_experts:
                    continue
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
                                for mtp_k in mtp_opts:
                                    for cp_d in cp_opts:
                                        for rope_m in rope_opts:
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
                                                mtp_n_predict_depths=int(mtp_k),
                                                mtp_depth_n_layers=int(constraints.mtp_depth_n_layers),
                                                mtp_train_loss_weight=float(constraints.mtp_train_loss_weight),
                                                cp_degree=int(cp_d),
                                                cp_method=str(constraints.cp_method),
                                                rope_scaling_method=str(rope_m),
                                                rope_scaling_factor=rope_factor if rope_m != "none" else 1.0,
                                                rope_original_max_position=int(constraints.rope_original_max_position),
                                                moe=moe_dict,
                                                ep_degree=opt.ep_degree,
                                                active_params=active_total,
                                                active_params_b=round(active_total / 1e9, 2),
                                                moe_style=opt.style,
                                                n_dense_ffn_layers=n_dense_clamped,
                                            ))
                                            if getattr(constraints, "allow_mla", False):
                                                # v1-fix demo-audit (June 2026
                                                # follow-up): MoE+MLA emission
                                                # had the same MHA-inheritance
                                                # bug as the dense-MLA path —
                                                # `total_total` was the MoE-MHA
                                                # total, and `d_head` was the
                                                # lattice's. For MoE we keep
                                                # n_layers because the FFN
                                                # branch (experts) dominates
                                                # the per-layer count, so the
                                                # MoE-MHA→MoE-MLA delta is
                                                # comparatively small. Just
                                                # recompute total_total against
                                                # the real MLA attention block
                                                # and snap d_head.
                                                d_nope_mla = int(constraints.mla_nope_head_dim)
                                                d_rope_mla = int(constraints.mla_rope_head_dim)
                                                dh_mla = d_nope_mla + d_rope_mla
                                                # MLA attention params (no FFN).
                                                mla_attn_per_layer = estimate_mla_per_layer_params(
                                                    pt.d_model, pt.n_heads,
                                                    0, 0, d_nope_mla, d_rope_mla,
                                                    0,
                                                )  # placeholder for c_kv/c_q below
                                                for c_kv in constraints.mla_kv_latent_options:
                                                    for c_q in constraints.mla_q_latent_options:
                                                        uncompressed_mla = 2 * pt.n_heads * dh_mla
                                                        if c_kv >= uncompressed_mla:
                                                            continue
                                                        # Replace the MHA attn block in total_total
                                                        # with the true MLA block. total_total
                                                        # already accounts for embeddings, norms
                                                        # and the MoE FFN; we subtract the dense
                                                        # MHA attention contribution and add the
                                                        # MLA attention contribution.
                                                        mha_attn_per_layer = (
                                                            2 * pt.d_model * pt.d_model           # Q + O
                                                            + 2 * pt.d_model * n_kv_heads * pt.d_head  # K + V
                                                        )
                                                        mla_attn_block = estimate_mla_per_layer_params(
                                                            pt.d_model, pt.n_heads,
                                                            c_kv, c_q, d_nope_mla, d_rope_mla,
                                                            0,                                    # no FFN here
                                                        ) - 2 * pt.d_model                        # strip norm double-count
                                                        delta_per_layer = mla_attn_block - mha_attn_per_layer
                                                        mla_total_total = total_total + delta_per_layer * n_layers
                                                        # MoE block only gates on active-params
                                                        # (already satisfied for the parent) and
                                                        # a max_total cap. The MLA swap doesn't
                                                        # change active per-token compute, so we
                                                        # only need to re-check max_total.
                                                        if mla_total_total > max_total:
                                                            continue
                                                        candidates.append(CandidateArch(
                                                            d_model=pt.d_model,
                                                            n_layers=n_layers,
                                                            n_heads=pt.n_heads,
                                                            # Snap d_head to MLA per-head Q dim
                                                            d_head=dh_mla,
                                                            n_kv_heads=n_kv_heads,
                                                            ffn_dim=pt.ffn_dim_swiglu,
                                                            vocab_size=constraints.vocab_size,
                                                            weight_precision=prec["weight_precision"],
                                                            ffn_precision=prec["ffn_precision"],
                                                            attn_precision=dict(prec["attn_precision"]),
                                                            kv_cache_bits=kv_bits,
                                                            total_params=mla_total_total,
                                                            total_params_b=round(mla_total_total / 1e9, 2),
                                                            mtp_n_predict_depths=int(mtp_k),
                                                            mtp_depth_n_layers=int(constraints.mtp_depth_n_layers),
                                                            mtp_train_loss_weight=float(constraints.mtp_train_loss_weight),
                                                            cp_degree=int(cp_d),
                                                            cp_method=str(constraints.cp_method),
                                                            rope_scaling_method=str(rope_m),
                                                            rope_scaling_factor=rope_factor if rope_m != "none" else 1.0,
                                                            rope_original_max_position=int(constraints.rope_original_max_position),
                                                            moe=moe_dict,
                                                            ep_degree=opt.ep_degree,
                                                            active_params=active_total,
                                                            active_params_b=round(active_total / 1e9, 2),
                                                            moe_style=opt.style,
                                                            n_dense_ffn_layers=n_dense_clamped,
                                                            attention_type="mla",
                                                            mla_kv_latent_dim=c_kv,
                                                            mla_q_latent_dim=c_q,
                                                            mla_rope_head_dim=d_rope_mla,
                                                            mla_nope_head_dim=d_nope_mla,
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
        d_model_min=1024, d_model_max=_d_model_max_for_target(constraints.target_params_b),
        d_head_options=[64, 128, 256],
    )
    aligned = [pt for pt in lattice if pt.tile_aligned]

    hw_prec_configs = get_precision_configs_for_hardware(hw_name)
    prec_configs = [p for p in (constraints.precision_configs or []) if p in hw_prec_configs]
    if not prec_configs:
        prec_configs = hw_prec_configs[:3]

    strategies = constraints.placement_strategies or ["first_periodic_last", "interleaved", "periodic"]
    mtp_opts, cp_opts, rope_opts, rope_factor = _search_option_lists(constraints)

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
        gqa_ratios = _gqa_ratios_for_point(pt, constraints)

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
            # v1-fix demo-audit D1: same depth band as the dense path.
            if not (1 <= n_layers_raw <= 256):
                continue

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
        d_model_min=1024, d_model_max=_d_model_max_for_target(constraints.target_params_b),
        d_head_options=[64, 128, 256],
    )
    aligned = [pt for pt in lattice if pt.tile_aligned]

    hw_prec_configs = get_precision_configs_for_hardware(hw_name)
    prec_configs = [p for p in (constraints.precision_configs or []) if p in hw_prec_configs]
    if not prec_configs:
        prec_configs = hw_prec_configs[:3]

    ep_opts = constraints.ep_options or default_ep_options(hw_name)
    strategies = constraints.placement_strategies or ["first_periodic_last", "interleaved", "periodic"]
    mtp_opts, cp_opts, rope_opts, rope_factor = _search_option_lists(constraints)

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
        gqa_ratios = _gqa_ratios_for_point(pt, constraints)

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
                if opt.top_k >= opt.n_experts:
                    continue
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
                                        for mtp_k in mtp_opts:
                                            for cp_d in cp_opts:
                                                for rope_m in rope_opts:
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
                                                        mtp_n_predict_depths=int(mtp_k),
                                                        mtp_depth_n_layers=int(constraints.mtp_depth_n_layers),
                                                        mtp_train_loss_weight=float(constraints.mtp_train_loss_weight),
                                                        cp_degree=int(cp_d),
                                                        cp_method=str(constraints.cp_method),
                                                        rope_scaling_method=str(rope_m),
                                                        rope_scaling_factor=rope_factor if rope_m != "none" else 1.0,
                                                        rope_original_max_position=int(constraints.rope_original_max_position),
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
                                                    if getattr(constraints, "allow_mla", False):
                                                        for c_kv in constraints.mla_kv_latent_options:
                                                            for c_q in constraints.mla_q_latent_options:
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
                                                                    total_params=total_total,
                                                                    total_params_b=round(total_total / 1e9, 2),
                                                                    mtp_n_predict_depths=int(mtp_k),
                                                                    mtp_depth_n_layers=int(constraints.mtp_depth_n_layers),
                                                                    mtp_train_loss_weight=float(constraints.mtp_train_loss_weight),
                                                                    cp_degree=int(cp_d),
                                                                    cp_method=str(constraints.cp_method),
                                                                    rope_scaling_method=str(rope_m),
                                                                    rope_scaling_factor=rope_factor if rope_m != "none" else 1.0,
                                                                    rope_original_max_position=int(constraints.rope_original_max_position),
                                                                    moe=moe_dict,
                                                                    ep_degree=opt.ep_degree,
                                                                    active_params=active_total,
                                                                    active_params_b=round(active_total / 1e9, 2),
                                                                    moe_style=opt.style,
                                                                    state_config=state_cfg,
                                                                    layer_type_list=layer_type_list,
                                                                    placement_strategy=strategy,
                                                                    n_attention_layers=actual_n_attn,
                                                                    n_state_layers=actual_n_state,
                                                                    hybrid_ratio=hybrid_ratio,
                                                                    derived_d_state=d_state,
                                                                    crossover_seq_len=L_star,
                                                                    attention_type="mla",
                                                                    mla_kv_latent_dim=c_kv,
                                                                    mla_q_latent_dim=c_q,
                                                                    mla_rope_head_dim=constraints.mla_rope_head_dim,
                                                                    mla_nope_head_dim=constraints.mla_nope_head_dim,
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
    prefill_len = max(1, int(constraints.prompt_len or constraints.context_length))
    tput_arch = TputArch(
        d_model=cand.d_model,
        n_layers=cand.n_layers,
        n_heads=cand.n_heads,
        d_head=cand.d_head,
        n_kv_heads=cand.n_kv_heads,
        ffn_dim=cand.ffn_dim,
        vocab_size=cand.vocab_size,
        batch_size=max(1, int(constraints.serving_batch)),
        seq_len=max(1, int(constraints.context_length)),
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

    # v1-fix SWA: wire the sliding window into the throughput model so
    # both decode KV reads and prefill compute scale with the window
    # instead of the full sequence length. Without this, swap_attention_to_swa
    # was effectively a label-only delta on every observable metric.
    swa_window = int(getattr(cand, "swa_window", 0) or 0)
    if swa_window > 0:
        tput_arch.local_window = swa_window
        tput_arch.attention_type = "swa"

    # v1-fix MTP: wire MTP depths so the training compute term picks up the
    # per-depth overhead.
    if cand.mtp_n_predict_depths > 0:
        tput_arch.mtp_n_predict_depths = int(cand.mtp_n_predict_depths)
        tput_arch.mtp_depth_n_layers = int(cand.mtp_depth_n_layers)
    # v1-fix CP: wire CP into the throughput model's training step.
    if cand.cp_degree > 1:
        tput_arch.cp_degree = int(cand.cp_degree)
        tput_arch.cp_method = str(cand.cp_method)

    # SWA cap on the throughput call: a sliding-window candidate's decode
    # reads a windowed KV cache, so decode_kv_len must be capped at the
    # window. We deliberately do NOT cap prefill_seq_len: prefill compute
    # is dominated by the FFN term (linear in seq) at short prompts, and
    # capping seq_len would shrink the FFN cost as well as the attention
    # cost. The net effect is that SWA's decode wins and KV-cache wins
    # are booked correctly, while prefill TTFT is left as a conservative
    # upper bound (real SWA prefill is O(N·W) instead of O(N²); we model
    # it as O(N²)). Caveat surfaced in the report's topology notes.
    effective_decode_len = prefill_len
    if swa_window > 0:
        effective_decode_len = min(effective_decode_len, swa_window)

    tput = throughput_fn(
        tput_arch, hw_name,
        tp_degree=constraints.tp,
        pp_degree=constraints.pp,
        microbatches=constraints.dp,
        decode_kv_len=effective_decode_len,
        prefill_seq_len=prefill_len,
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
        attention_type=(
            "mla" if cand.attention_type == "mla"
            else ("swa" if (cand.attention_type == "swa" or swa_window > 0) else "gqa")
        ),
        mla_latent_dim=(cand.mla_kv_latent_dim if cand.attention_type == "mla" else None),
        mla_q_latent_dim=(cand.mla_q_latent_dim if cand.attention_type == "mla" else None),
        mla_rope_head_dim=(cand.mla_rope_head_dim if cand.attention_type == "mla" else None),
        mla_nope_head_dim=(cand.mla_nope_head_dim if cand.attention_type == "mla" else None),
        # v1-fix SWA: thread the window so the quality model adds the small
        # SWA-locality residual when workload_context exceeds the window.
        local_window=(swa_window if swa_window > 0 else None),
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
            sequence_length=max(1, int(constraints.context_length)),
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

    v1 adds TTFT and N_total as axes so an MoE candidate doesn't auto-dominate a
    same-active dense at the same training tput, TBT, and memory just because
    of the sparse-capacity quality bonus — its higher total parameter count
    is a real cost (storage, serving cluster footprint). TTFT is separate from
    decode TBT because long-prefill architectures can otherwise look Pareto
    clean despite being unusable for cold 1M-context serving.
    """
    objs_a = (a.predicted_loss, -a.training_tps, a.serving_tbt_ms,
              a.throughput.prefill_time_ms, a.memory_per_gpu_gb, a.arch.total_params)
    objs_b = (b.predicted_loss, -b.training_tps, b.serving_tbt_ms,
              b.throughput.prefill_time_ms, b.memory_per_gpu_gb, b.arch.total_params)

    at_least_one_better = False
    for oa, ob in zip(objs_a, objs_b):
        if ob > oa:  # b is worse in this objective
            return False
        if ob < oa:  # b is better in this objective
            at_least_one_better = True

    return at_least_one_better


def compute_pareto_frontier(candidates: List[EvaluatedCandidate]) -> List[EvaluatedCandidate]:
    """Compute the Pareto frontier via non-dominated sorting."""
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
        attn_key = (
            c.attention_type,
            c.mla_kv_latent_dim,
            c.mla_q_latent_dim,
            c.mla_rope_head_dim,
            c.mla_nope_head_dim,
        )
        long_context_key = (
            c.cp_degree,
            c.cp_method,
            c.rope_scaling_method,
            round(float(c.rope_scaling_factor), 6),
            c.rope_original_max_position,
        )
        mtp_key = (
            c.mtp_n_predict_depths,
            c.mtp_depth_n_layers,
            round(float(c.mtp_train_loss_weight), 6),
        )
        key = (c.d_model, c.n_layers, c.n_heads, c.d_head, c.n_kv_heads,
               c.ffn_dim, c.weight_precision, c.ffn_precision,
               tuple(sorted(c.attn_precision.items())), c.kv_cache_bits,
               moe_key, state_key, attn_key, long_context_key, mtp_key)
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

    # 6. Pick the displayed optimum from the Pareto surface using the selected
    # tradeoff preset. This keeps "optimal" aligned with latency/cost profiles
    # instead of always snapping to the lowest loss point.
    #
    # Uncertainty-aware tiebreak: bucket predicted_loss to a fraction of the
    # quality model's own ±%-uncertainty band so that two candidates whose
    # quality differs by less than the noise floor are *not* split by the
    # 6th decimal of predicted_loss. Inside a bucket the next keys (prefill,
    # TBT, memory) decide, which avoids the previous behaviour of picking a
    # config that was statistically indistinguishable from a much cheaper
    # neighbour purely because it had the absolute argmin loss.
    optimal = None
    if feasible:
        scoring_pool = pareto if pareto else feasible
        # Build the SAME sort key used by both the picker and the CSV
        # writer. Anchoring both paths to one builder is what guarantees
        # that `rank=1` in pareto.csv always agrees with `selected=True`.
        display_sort_key = build_display_sort_key(scoring_pool, constraints)
        optimal = min(scoring_pool, key=display_sort_key)

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


def build_display_sort_key(scoring_pool, constraints):
    """Single source of truth for the displayed-Pareto ordering.

    Both the picker (which produces `selected=True`) and the pareto.csv
    writer (which produces the `rank` column) call this. Any divergence
    between them will desynchronize rank=1 from selected=True, which is
    a footgun visible in every README example. Don't re-implement the
    key in either site; extend this builder.
    """
    # Pure-loss profiles: argmin(predicted_loss) with an aspect-ratio prior
    # and a deterministic secondary key. v1-fix demo-audit: prior versions
    # did NOT collapse losses inside a noise band, which let argmin pick
    # configurations 0.05% better on a 3% uncertainty signal while being
    # 2–5× worse on TBT/memory. We now bucket loss by ~25% of the model's
    # own median uncertainty before tiebreaking on memory/tbt/-tps — the
    # same uncertainty-aware tiebreak the balanced profile uses, but with
    # loss still strictly dominant via the bucket index.
    profile_name = constraints.objective_profile if constraints else "balanced"
    if profile_name in ("research_quality", "loss_only"):
        pool_size = len(scoring_pool)
        TIEBREAK_K_LOSS = 0.25
        _pool_best = min(x.predicted_loss for x in scoring_pool)
        _u_sorted = sorted(
            float(getattr(x.quality, "uncertainty_total", 0.0) or 0.0)
            for x in scoring_pool
        )
        _u_med = _u_sorted[len(_u_sorted) // 2] if _u_sorted else 0.0
        _BAND = max(_u_med, 0.005) * max(_pool_best, 0.01) * TIEBREAK_K_LOSS

        def _loss_key(x):
            prior = _aspect_ratio_prior_penalty(x, pool_size=pool_size)
            adj_loss = x.predicted_loss * (1.0 + prior)
            bucket = (
                max(0, int((adj_loss - _pool_best) / _BAND))
                if _BAND > 0 else 0
            )
            return (
                bucket,
                x.memory_per_gpu_gb,
                x.serving_tbt_ms,
                -x.training_tps,
                round(adj_loss, 6),
                float(x.arch.total_params),
            )

        return _loss_key

    # General profiles: uncertainty-aware noise bucket on both the
    # objective score and on raw loss, then lexicographic on (memory,
    # tbt, prefill, -tps, loss).
    #
    # k=0.25 means "treat loss as equal within ~quarter of the
    # uncertainty band". This is conservative: real measurement noise
    # plus optimizer variance routinely exceed this, so we never trade a
    # meaningful quality win away.
    TIEBREAK_K = 0.25

    # Pool-wide bucket denominator (same for every candidate, so noisier
    # candidates do not get a free shift toward bucket 0). Median is more
    # robust to outliers than mean.
    pool_best_loss = min(x.predicted_loss for x in scoring_pool)
    _pool_u_sorted = sorted(
        float(getattr(x.quality, "uncertainty_total", 0.0) or 0.0)
        for x in scoring_pool
    )
    _u_median = _pool_u_sorted[len(_pool_u_sorted) // 2] if _pool_u_sorted else 0.0
    POOL_BAND = max(_u_median, 0.005) * max(pool_best_loss, 0.01) * TIEBREAK_K

    def _quality_bucket(ev) -> int:
        if POOL_BAND <= 0:
            return 0
        return max(0, int((ev.predicted_loss - pool_best_loss) / POOL_BAND))

    weights = OBJECTIVE_PROFILES.get(profile_name, {})
    loss_weight = float(weights.get("loss", 0.0))
    # Pool-wide objective-score band so two close candidates collapse to
    # the same score bucket and get tiebroken on throughput/memory.
    _pool_scores = {
        id(x): _objective_score(x, scoring_pool, profile_name)
        for x in scoring_pool
    }
    _best_score = min(_pool_scores.values()) if _pool_scores else 0.0
    SCORE_BAND = (
        loss_weight * TIEBREAK_K * max(_u_median, 0.01)
        if loss_weight > 0 else POOL_BAND
    )

    def _score_bucket(ev) -> int:
        if SCORE_BAND <= 0:
            return 0
        score = _pool_scores.get(id(ev), 0.0)
        return max(0, int((score - _best_score) / SCORE_BAND))

    def _general_key(x):
        return (
            _score_bucket(x),
            _quality_bucket(x),
            # Within a quality bucket, prefer faster, smaller, cheaper.
            x.memory_per_gpu_gb,
            x.serving_tbt_ms,
            x.throughput.prefill_time_ms,
            -x.training_tps,
            # Deterministic final tiebreak on raw loss.
            x.predicted_loss,
        )

    return _general_key


def _same_candidate(a: EvaluatedCandidate, b: EvaluatedCandidate) -> bool:
    ca = a.arch
    cb = b.arch
    return (
        ca.d_model == cb.d_model
        and ca.n_layers == cb.n_layers
        and ca.n_heads == cb.n_heads
        and ca.d_head == cb.d_head
        and ca.n_kv_heads == cb.n_kv_heads
        and ca.ffn_dim == cb.ffn_dim
        and ca.weight_precision == cb.weight_precision
        and ca.ffn_precision == cb.ffn_precision
        and ca.kv_cache_bits == cb.kv_cache_bits
        and ca.moe_style == cb.moe_style
        and ca.ep_degree == cb.ep_degree
        and ca.attention_type == cb.attention_type
        and ca.mla_kv_latent_dim == cb.mla_kv_latent_dim
        and ca.cp_degree == cb.cp_degree
        and ca.rope_scaling_method == cb.rope_scaling_method
        and ca.mtp_n_predict_depths == cb.mtp_n_predict_depths
    )


def _candidate_summary(ev: EvaluatedCandidate) -> Dict[str, Any]:
    c = ev.arch
    return {
        "d_model": c.d_model,
        "n_layers": c.n_layers,
        "n_heads": c.n_heads,
        "d_head": c.d_head,
        "n_kv_heads": c.n_kv_heads,
        "attention_type": c.attention_type,
        "ffn_precision": c.ffn_precision,
        "kv_bits": c.kv_cache_bits,
        "moe_style": c.moe_style,
        "ep_degree": c.ep_degree,
        "active_params_b": c.active_params_b or c.total_params_b,
        "total_params_b": c.total_params_b,
        "predicted_loss": round(ev.predicted_loss, 6),
        "training_tps": round(ev.training_tps),
        "serving_tbt_ms": round(ev.serving_tbt_ms, 3),
        "serving_ttft_ms": round(ev.throughput.prefill_time_ms, 3),
        "memory_per_gpu_gb": round(ev.memory_per_gpu_gb, 3),
        "confidence": ev.quality.confidence,
    }


def _loss_interval(ev: EvaluatedCandidate) -> Tuple[float, float]:
    unc = max(0.0, float(getattr(ev.quality, "uncertainty_total", 0.0)))
    low = max(0.0, ev.predicted_loss * (1.0 - unc))
    high = ev.predicted_loss * (1.0 + unc)
    return low, high


_FAMILY_AXES = (
    "d_model", "n_layers", "n_heads", "d_head", "n_kv_heads",
    "ffn_dim", "ffn_precision", "kv_cache_bits", "attention_type",
    "moe_style",
)


def _contender_summary(ev: EvaluatedCandidate) -> Dict[str, Any]:
    """Compact one-row summary of a contending candidate for the family view."""
    arch = ev.arch
    active = getattr(arch, "active_params_b", 0.0) or 0.0
    total = getattr(arch, "total_params_b", 0.0) or 0.0
    # Dense candidates only populate total_params_b; fall back to it when
    # active is missing or zero so the table doesn't show 0.0 everywhere.
    if not active and total:
        active = total
    return {
        "d_model": getattr(arch, "d_model", None),
        "n_layers": getattr(arch, "n_layers", None),
        "n_heads": getattr(arch, "n_heads", None),
        "d_head": getattr(arch, "d_head", None),
        "n_kv_heads": getattr(arch, "n_kv_heads", None),
        "ffn_dim": getattr(arch, "ffn_dim", None),
        "ffn_precision": getattr(arch, "ffn_precision", None),
        "kv_cache_bits": getattr(arch, "kv_cache_bits", None),
        "attention_type": getattr(arch, "attention_type",
                                    getattr(arch, "attn_type", "full")),
        "moe_style": getattr(arch, "moe_style", "dense"),
        "active_params_b": active,
        "predicted_loss": round(float(ev.predicted_loss), 6),
        "training_tps": int(ev.training_tps),
        "serving_tbt_ms": round(float(ev.serving_tbt_ms), 2),
        "memory_per_gpu_gb": round(float(ev.memory_per_gpu_gb), 2),
    }


def _contending_family(
    opt: EvaluatedCandidate,
    contenders: List[EvaluatedCandidate],
    top_n: int = 8,
    contender_reasons: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Compute a "statistically-indistinguishable family" view of the
    contenders: the axes along which they actually vary, plus a top-N row
    table that lets a pretrain lead see the shape of the indecision.
    """
    if not contenders:
        return {
            "varying_axes": [],
            "row_count": 0,
            "members": [],
        }
    opt_row = _contender_summary(opt)
    rows = []
    for ev in contenders:
        r = _contender_summary(ev)
        if contender_reasons is not None:
            r["tied_via"] = contender_reasons.get(id(ev), "loss")
        rows.append(r)
    varying = []
    for axis in _FAMILY_AXES:
        values = {opt_row.get(axis)} | {r.get(axis) for r in rows}
        if len(values) > 1:
            varying.append(axis)
    # Surface the contenders sorted by closeness to the optimum's predicted
    # loss so the top of the table is the strongest competitor.
    rows.sort(key=lambda r: abs(float(r["predicted_loss"])
                                  - float(opt_row["predicted_loss"])))
    return {
        "varying_axes": varying,
        "row_count": len(rows),
        "members": rows[:top_n],
        "selected": opt_row,
    }


def _intervals_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    """Return True iff two closed intervals overlap."""
    return a_lo <= b_hi and b_lo <= a_hi


def _throughput_intervals(ev: EvaluatedCandidate, k: float = 1.0) -> Dict[str, tuple]:
    """Return ±k*sigma intervals for each throughput-side metric on a
    candidate. Pulls the propagated sigmas from the ThroughputResult; falls
    back to 0-width intervals when sigma isn't available."""
    t = ev.throughput
    sig_tps = float(getattr(t, "training_throughput_sigma_tps", 0.0) or 0.0)
    sig_tbt = float(getattr(t, "decode_time_sigma_ms", 0.0) or 0.0)
    sig_pre = float(getattr(t, "prefill_time_sigma_ms", 0.0) or 0.0)
    return {
        "training_tps": (ev.training_tps - k * sig_tps,
                          ev.training_tps + k * sig_tps),
        "serving_tbt_ms": (ev.serving_tbt_ms - k * sig_tbt,
                            ev.serving_tbt_ms + k * sig_tbt),
        "prefill_time_ms": (t.prefill_time_ms - k * sig_pre,
                             t.prefill_time_ms + k * sig_pre),
    }


def _is_throughput_contender(
    opt: EvaluatedCandidate,
    ev: EvaluatedCandidate,
    k: float = 1.0,
) -> bool:
    """True when ev is within ±k*sigma of opt on any throughput metric. The
    interval test only counts as a contender if ev is also not strictly
    dominated on the other axes (we don't want to call every slower model a
    'contender'); we approximate that by requiring the candidate to be at
    least as good as the optimum on at least one throughput axis after the
    sigma adjustment."""
    opt_iv = _throughput_intervals(opt, k)
    ev_iv = _throughput_intervals(ev, k)
    overlapping = 0
    at_least_as_good = False
    for axis, (a_lo, a_hi) in opt_iv.items():
        b_lo, b_hi = ev_iv[axis]
        if _intervals_overlap(a_lo, a_hi, b_lo, b_hi):
            overlapping += 1
            # "At least as good" depends on axis direction.
            if axis == "training_tps":
                if b_hi >= a_lo:
                    at_least_as_good = True
            else:
                if b_lo <= a_hi:
                    at_least_as_good = True
    return overlapping > 0 and at_least_as_good


def compute_confidence_envelope(
    result: OptimizationResult, opt: EvaluatedCandidate
) -> Dict[str, Any]:
    """Public wrapper around the loss-CI overlap analysis.

    Exposed so the CLI (`cli_compile.py`) can surface a non-robust pick
    as a `WARNING:` without re-implementing the contender accounting.
    The private alias `_confidence_envelope` is retained for backward
    compat with existing callers inside this module.
    """
    return _confidence_envelope(result, opt)


def compute_contending_family_full(
    result: OptimizationResult,
    opt: EvaluatedCandidate,
    top_n: int = 32,
) -> Dict[str, Any]:
    """Return the full contending-family snapshot (up to `top_n` rows).

    Embedded `confidence_envelope.contending_family.members` carries
    only the top 5 rows so the emitted config stays small. Downstream
    tooling that needs the broader view (notebooks, dashboards,
    auto-calibrate) should read the sidecar JSON the CLI writes, which
    is produced by this function.
    """
    opt_low, opt_high = _loss_interval(opt)
    contenders = []
    contender_reasons = {}
    for ev in result.all_evaluated:
        if ev is opt or not ev.meets_constraints:
            continue
        ev_low, ev_high = _loss_interval(ev)
        loss_contender = _intervals_overlap(opt_low, opt_high, ev_low, ev_high)
        thr_contender = _is_throughput_contender(opt, ev)
        if loss_contender or thr_contender:
            contenders.append(ev)
            if loss_contender and thr_contender:
                contender_reasons[id(ev)] = "both"
            elif loss_contender:
                contender_reasons[id(ev)] = "loss"
            else:
                contender_reasons[id(ev)] = "throughput"
    return _contending_family(
        opt, contenders, top_n=top_n, contender_reasons=contender_reasons,
    )


def _confidence_envelope(result: OptimizationResult, opt: EvaluatedCandidate) -> Dict[str, Any]:
    opt_low, opt_high = _loss_interval(opt)
    contenders = []
    contender_reasons = {}  # id(ev) -> "loss" | "throughput" | "both"
    for ev in result.all_evaluated:
        if ev is opt or not ev.meets_constraints:
            continue
        low, high = _loss_interval(ev)
        loss_contender = low <= opt_high and ev.predicted_loss <= opt_high
        # Throughput-side tie: candidate is within ±1σ of opt on any of
        # {TPS, TBT, prefill}. Lets the contending-family view reflect the
        # fact that the predicted throughput numbers carry ±20% uncertainty
        # that the deterministic sort previously ignored.
        thr_contender = (loss_contender or
                          ev.predicted_loss <= opt_high * 1.005) and \
                         _is_throughput_contender(opt, ev, k=1.0)
        if loss_contender or thr_contender:
            contenders.append((ev, low, high))
            tag = ("both" if (loss_contender and thr_contender)
                    else ("loss" if loss_contender else "throughput"))
            contender_reasons[id(ev)] = tag
    best_other_low = min((low for _, low, _ in contenders), default=None)
    target_coverage = None
    try:
        from quality_model import load_quality_constants
        target_coverage = (
            load_quality_constants()
            .get("uncertainty", {})
            .get("calibration_target_coverage")
        )
    except Exception:
        target_coverage = None
    # Cap the embedded family at top_n=5 to keep the inline metadata
    # small (under a kilobyte even for wide pareto fronts). The full
    # top_n=32 view is emitted to a sidecar file by the CLI when the
    # envelope is non-robust; downstream tooling that needs all rows
    # should read the sidecar, not parse this dict. Forensics-mode
    # callers can bump the inline cap via AC_CONTENDING_FAMILY_INLINE=N.
    inline_top_n = 5
    try:
        _env_cap = os.environ.get("AC_CONTENDING_FAMILY_INLINE")
        if _env_cap is not None:
            parsed_cap = int(_env_cap)
            if parsed_cap > 0:
                inline_top_n = parsed_cap
    except (ValueError, TypeError):
        # Malformed env value → fall back to the default cap; the env
        # var is a forensics knob, not a correctness lever, so we don't
        # raise.
        pass
    family = _contending_family(opt, [ev for ev, _, _ in contenders],
                                  top_n=inline_top_n,
                                  contender_reasons=contender_reasons)
    # Throughput-uncertainty fields, exposed so the report renderer can show
    # the propagated sigmas alongside the point estimates.
    t = opt.throughput
    throughput_uncertainty = {
        "training_tps_sigma": round(float(getattr(t, "training_throughput_sigma_tps", 0.0) or 0.0), 1),
        "serving_tbt_sigma_ms": round(float(getattr(t, "decode_time_sigma_ms", 0.0) or 0.0), 2),
        "prefill_time_sigma_ms": round(float(getattr(t, "prefill_time_sigma_ms", 0.0) or 0.0), 2),
        "efficiency_bucket": getattr(t, "efficiency_bucket", ""),
    }
    return {
        "loss_low": round(opt_low, 6),
        "loss_high": round(opt_high, 6),
        "uncertainty_total_pct": round(float(getattr(opt.quality, "uncertainty_total", 0.0)) * 100, 3),
        "target_coverage": target_coverage,
        "robust_to_loss_uncertainty": not contenders,
        "contending_candidates": len(contenders),
        "contending_family": family,
        "throughput_uncertainty": throughput_uncertainty,
        "best_contender_loss_low": (
            round(best_other_low, 6) if best_other_low is not None else None
        ),
    }


def _selection_diagnostics(result: OptimizationResult, opt: EvaluatedCandidate) -> Dict[str, Any]:
    pareto = result.pareto_frontier or []
    profile = result.constraints.objective_profile if result.constraints else "balanced"
    best_loss = min(pareto, key=lambda ev: ev.predicted_loss) if pareto else opt
    selected_rank = None
    best_loss_rank = None
    for idx, ev in enumerate(pareto, start=1):
        if selected_rank is None and _same_candidate(ev, opt):
            selected_rank = idx
        if best_loss_rank is None and _same_candidate(ev, best_loss):
            best_loss_rank = idx
    selected_score = (
        _objective_score(opt, pareto, profile) if pareto else 0.0
    )
    best_loss_score = (
        _objective_score(best_loss, pareto, profile) if pareto else 0.0
    )
    loss_gap_pct = 0.0
    if best_loss.predicted_loss > 0:
        loss_gap_pct = (opt.predicted_loss - best_loss.predicted_loss) / best_loss.predicted_loss * 100.0

    warnings = []
    if loss_gap_pct > 1e-6:
        warnings.append(
            f"Selected point is {loss_gap_pct:.2f}% worse in predicted loss than the best-loss Pareto point."
        )
    if opt.quality.confidence == "low" and best_loss.quality.confidence != "low":
        warnings.append(
            "Selected point has low quality confidence while the best-loss Pareto point does not."
        )
    if result.constraints and result.constraints.allow_moe and opt.arch.moe is None:
        moe_points = [ev for ev in pareto if ev.arch.moe is not None]
        if moe_points and min(ev.predicted_loss for ev in moe_points) < opt.predicted_loss:
            warnings.append(
                "MoE search was enabled but the selected point is dense while a lower-loss MoE point exists on the Pareto frontier."
            )
    return {
        "objective_profile": profile,
        "selected_pareto_rank": selected_rank,
        "selected_objective_score": round(selected_score, 8),
        "best_loss_pareto_rank": best_loss_rank,
        "best_loss_objective_score": round(best_loss_score, 8),
        "loss_gap_vs_best_pct": round(loss_gap_pct, 4),
        "selected": _candidate_summary(opt),
        "best_loss": _candidate_summary(best_loss),
        "warnings": warnings,
    }


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
    selection_diag = _selection_diagnostics(result, opt)

    # v1-fix E (demo audit): include hardware + parallelism so a downstream
    # reviewer can re-derive what was compiled without re-running the
    # pipeline. The previous emitted block recorded only workload knobs
    # (params, tokens, context, serving) and architecture flags; the
    # hardware lived only implicitly inside the calibrated efficiency
    # numbers, so a reader reading the config in isolation could not tell
    # whether this was an H100 or B200 run. We also surface TP/PP/DP/CP
    # so the throughput-per-replica vs aggregate split makes sense
    # downstream.
    input_constraints = {
        "hardware": result.hardware,
        "tp": result.constraints.tp,
        "pp": result.constraints.pp,
        "dp": result.constraints.dp,
        "cp": getattr(result.constraints, "cp", 1),
        "cp_method": getattr(result.constraints, "cp_method", "ring"),
        "target_params": f"{result.constraints.target_params_b}B",
        "training_tokens": f"{result.constraints.training_tokens / 1e12:.1f}T",
        "context_length": result.constraints.context_length,
        "serving_tbt_ms": result.constraints.serving_tbt_ms,
        "serving_ttft_ms": result.constraints.serving_ttft_ms,
        "serving_batch": result.constraints.serving_batch,
        "prompt_len": result.constraints.prompt_len,
        "output_len": result.constraints.output_len,
        "scheduler": result.constraints.scheduler,
        "objective_profile": result.constraints.objective_profile,
        "allow_moe": result.constraints.allow_moe,
        "max_total_params_b": result.constraints.max_total_params_b,
        "allow_state": result.constraints.allow_state,
    }

    # Fix #2: training throughput is per TP replica (one TP group). The
    # aggregate across DP × PP replicas is the user-facing cluster number.
    # Both are emitted so downstream tools and the user can pick the one they
    # need without re-deriving the unit. The legacy field is kept for back-
    # compat and aliased to the per-replica value.
    dp_degree = max(1, int(getattr(result.constraints, "dp", 1) or 1))
    pp_degree = max(1, int(getattr(result.constraints, "pp", 1) or 1))
    tp_degree = max(1, int(getattr(result.constraints, "tp", 1) or 1))
    cp_degree = max(1, int(getattr(c, "cp_degree", 1) or 1))
    per_replica_tps = round(opt.training_tps)
    # Aggregate scales linearly with DP. PP pipelines a single replica, so it
    # doesn't multiply throughput; we still expose pp for the reader.
    aggregate_tps = round(opt.training_tps * dp_degree)
    # GPU counts. A "replica" is one TP × PP × CP group; the cluster has DP
    # such replicas. Per-GPU TPS is the only number that's comparable across
    # different parallelism layouts — without it, a CP=4 run looks "4×
    # faster" than CP=1 just because the replica grew 4× in GPUs.
    gpus_per_replica = tp_degree * pp_degree * cp_degree
    total_gpus = gpus_per_replica * dp_degree
    per_gpu_tps = round(opt.training_tps / max(1, gpus_per_replica))

    predicted = {
        "quality_rank_score": round(-opt.predicted_loss, 4),
        "predicted_loss": round(opt.predicted_loss, 4),
        # Per-TP-replica throughput. Same value across DP — DP scales the
        # aggregate, not the per-replica rate. The per-GPU number is the
        # apples-to-apples comparison across parallelism choices.
        "training_throughput_tokens_per_sec": per_replica_tps,
        "training_throughput_tokens_per_sec_per_replica": per_replica_tps,
        "training_throughput_tokens_per_sec_per_gpu": per_gpu_tps,
        "aggregate_training_throughput_tokens_per_sec": aggregate_tps,
        "training_throughput_unit": (
            f"tokens/sec per TP×PP×CP replica ({tp_degree}×{pp_degree}×{cp_degree}"
            f" = {gpus_per_replica} GPUs/replica); per-GPU = {per_gpu_tps} tok/s; "
            f"aggregate over DP={dp_degree} replicas "
            f"({total_gpus} total GPUs) = {aggregate_tps} tok/s"
        ),
        "serving_tbt_ms": round(opt.serving_tbt_ms, 1),
        "serving_ttft_ms": round(opt.throughput.prefill_time_ms, 1),
        "prefill_model": {
            "prompt_len": int(result.constraints.prompt_len or result.constraints.context_length),
            "cold_prefill": True,
            "prefix_cache_hit_rate": 0.0,
            "scheduler": result.constraints.scheduler,
            "chunk_size": 65536 if result.constraints.scheduler == "chunked" else None,
            "chunking_changes_total_ttft": False,
            "context_parallel_degree": int(c.cp_degree),
            "context_parallel_method": str(c.cp_method),
        },
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
        "confidence_envelope": _confidence_envelope(result, opt),
        "selection_diagnostics": selection_diag,
        "selection_warnings": selection_diag["warnings"],
        "calibration_warnings": list(getattr(opt.quality, "calibration_warnings", [])),
        "eval_predictions": getattr(opt.quality, "eval_predictions", {}),
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
        "rank,selected,objective_profile,objective_score,d_model,n_layers,n_heads,d_head,n_kv_heads,ffn_dim,"
        "weight_prec,ffn_prec,kv_bits,active_params_B,total_params_B,"
        "attention_type,mla_kv_latent,mla_q_latent,cp,cp_method,rope_method,rope_factor,mtp_depth,"
        "state_layers,attention_layers,placement_strategy,"
        "moe_style,n_experts,top_k,expert_dim,ep,"
        "predicted_loss,loss_ci_low,loss_ci_high,uncertainty_total_pct,"
        "training_tps,serving_tbt_ms,serving_ttft_ms,memory_gb,confidence"
    ]
    # Use the *exact* same sort key the picker used to choose `selected`.
    # This is the only way to guarantee rank=1 == selected=True. See
    # build_display_sort_key for the definition.
    if result.constraints and result.pareto_frontier:
        sort_key = build_display_sort_key(result.pareto_frontier, result.constraints)
        sorted_frontier = sorted(result.pareto_frontier, key=sort_key)
    else:
        sorted_frontier = list(result.pareto_frontier)
    for i, ev in enumerate(sorted_frontier):
        c = ev.arch
        if c.moe is not None:
            n_experts = c.moe["n_experts"]
            top_k = c.moe["top_k"]
            expert_dim = c.moe["expert_dim"]
        else:
            n_experts = top_k = expert_dim = 0
        active = c.active_params_b or c.total_params_b
        objective_score = (
            _objective_score(ev, result.pareto_frontier, result.constraints.objective_profile)
            if result.constraints else 0.0
        )
        loss_low, loss_high = _loss_interval(ev)
        lines.append(
            f"{i+1},{_same_candidate(ev, result.optimal) if result.optimal else False},"
            f"{result.constraints.objective_profile if result.constraints else ''},"
            f"{objective_score:.8f},"
            f"{c.d_model},{c.n_layers},{c.n_heads},{c.d_head},{c.n_kv_heads},"
            f"{c.ffn_dim},{c.weight_precision},{c.ffn_precision},{c.kv_cache_bits},"
            f"{active},{c.total_params_b},"
            f"{c.attention_type},{c.mla_kv_latent_dim},{c.mla_q_latent_dim},"
            f"{c.cp_degree},{c.cp_method},{c.rope_scaling_method},"
            f"{c.rope_scaling_factor:.4f},{c.mtp_n_predict_depths},"
            f"{c.n_state_layers},{c.n_attention_layers},{c.placement_strategy},"
            f"{c.moe_style},{n_experts},{top_k},{expert_dim},{c.ep_degree},"
            f"{ev.predicted_loss:.4f},{loss_low:.4f},{loss_high:.4f},"
            f"{getattr(ev.quality, 'uncertainty_total', 0.0) * 100:.2f},"
            f"{ev.training_tps:.0f},"
            f"{ev.serving_tbt_ms:.1f},{ev.throughput.prefill_time_ms:.1f},"
            f"{ev.memory_per_gpu_gb:.1f},{ev.quality.confidence}"
        )
    return "\n".join(lines)
