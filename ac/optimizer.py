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

import copy as _copy
import os
import math
import sys
import time
from dataclasses import dataclass, field, replace as _dc_replace
from typing import List, Dict, Optional, Tuple, Callable, Any

try:
    from .lattice_engine import (
        HARDWARE as LATTICE_HW, compute_lattice, compute_gqa_configs,
        estimate_params, estimate_mla_total_params, estimate_mla_per_layer_params,
        LatticePoint, GQAConfig, compute_moe_options, default_ep_options,
        MoEOption, compute_state_lattice, compute_hybrid_patterns,
        place_attention_layers, HybridPattern,
    )
    from .throughput_model import (
        throughput as throughput_fn, HardwareConfig, ArchConfig as TputArch,
        ThroughputResult,
    )
    from .quality_model import (
        quality as quality_fn, ArchConfig as QualArch, TrainingConfig,
        QualityResult, chinchilla_loss,
    )
    from .penalties import shape_penalty
    from .schema import build_config, build_hybrid_config, SCHEMA_VERSION
    from .sram_derivation import derive_d_state, compute_crossover_seq_len
    from .architecture import (
        architecture_fingerprint, compose_layer_type_list, parameter_ledger,
        validate_architecture_views, format_state_attention_ratio,
    )
except ImportError:
    # Direct-file execution remains supported for the legacy CLI entrypoints.
    from lattice_engine import (
        HARDWARE as LATTICE_HW, compute_lattice, compute_gqa_configs,
        estimate_params, estimate_mla_total_params, estimate_mla_per_layer_params,
        LatticePoint, GQAConfig, compute_moe_options, default_ep_options,
        MoEOption, compute_state_lattice, compute_hybrid_patterns,
        place_attention_layers, HybridPattern,
    )
    from throughput_model import (
        throughput as throughput_fn, HardwareConfig, ArchConfig as TputArch,
        ThroughputResult,
    )
    from quality_model import (
        quality as quality_fn, ArchConfig as QualArch, TrainingConfig,
        QualityResult, chinchilla_loss,
    )
    from penalties import shape_penalty
    from schema import build_config, build_hybrid_config, SCHEMA_VERSION
    from sram_derivation import derive_d_state, compute_crossover_seq_len
    from architecture import (
        architecture_fingerprint, compose_layer_type_list, parameter_ledger,
        validate_architecture_views, format_state_attention_ratio,
    )


# =============================================================================
# Data classes
# =============================================================================

VALID_ROPE_SCALING_METHODS = {"none", "pi", "ntk", "yarn", "longrope"}
VALID_PLACEMENT_STRATEGIES = {
    "first_periodic_last",
    "interleaved",
    "periodic",
}


def _filter_ep_options_by_dp(
    raw_ep_options,
    dp,
    source: str = "hardware default",
) -> List[int]:
    """Clamp candidate EP degrees to what the training-throughput model can price.

    Wave 24 P0 fix. The Wave-19 training-TPS math assumes MoE lays EP over
    the DP dimension (each EP rank processes its own microbatch), so
    ``EP > DP`` describes a layout the model cannot price honestly.
    Enumerating those candidates and then patching the picker with a
    post-hoc ``WARNING: picked EP=X exceeds DP=Y`` was leaving fabricated
    TPS numbers on the winning row.

    EP=1 is legal: experts remain TP-sharded and serving-memory feasibility
    decides whether the full local expert set fits. This keeps greenfield
    behavior aligned with the throughput, baseline, and delta paths.

    Two DP regimes:

    * ``DP >= 2`` — real training layout, ``EP`` must divide ``DP``. If the user
      explicitly supplied ``--ep-options`` and NOTHING survives, raise;
      silently disabling the MoE search is worse than a loud error.
    * ``DP <= 1`` — matrix-probe / single-cell scoring path. The training
      DP dimension is unspecified, so we do not cap by DP. Callers that
      still want an upper bound can constrain ``--ep-options`` themselves.
    """
    dp_int = max(0, int(dp) if dp is not None else 0)
    raw = list(raw_ep_options or [])
    if dp_int <= 1:
        return [int(v) for v in raw if int(v) >= 1]
    survivors = [
        int(v) for v in raw
        if int(v) >= 1 and int(v) <= dp_int and dp_int % int(v) == 0
    ]
    if raw and not survivors and source == "user-supplied ep_options":
        raise ValueError(
            f"--ep-options {sorted(int(v) for v in raw)} incompatible with "
            f"--dp {dp_int}: the training-throughput model requires "
            f"EP must divide DP (EP overlays the DP dimension). Change "
            f"--dp or --ep-options."
        )
    return survivors

# v1-fix sanity gate (June 2026): if the picked optimum's predicted_loss
# exceeds this multiple of its Chinchilla baseline, treat the cell as
# "no feasible solution" instead of returning a candidate with leaked
# INFEASIBLE-sentinel inflation. The quality model uses INFEASIBLE=1e6
# as a hard-violation marker (penalties.py); when that sentinel sneaks
# into total_penalty_fraction the displayed loss balloons to ~1e6×base.
# Real penalty fractions on feasible candidates are << 1, so a 10× cap
# gives a wide safety margin without affecting any plausibly-shipping
# architecture.
_SENTINEL_LOSS_MULT = 10.0

@dataclass
class DeploymentConstraints:
    """User-specified deployment constraints."""
    target_params_b: float = 7.0        # target param count in billions
    param_tolerance: float = 0.15       # ±fraction around target
    training_tokens: int = 20_000_000_000_000  # central frontier scenario
    unique_training_tokens: Optional[int] = None
    pretraining_context_length: int = 8192
    quality_model_version: str = "effective_capacity_v2"
    context_length: int = 8192
    # Serving constraints
    serving_tbt_ms: Optional[float] = 50.0   # time-between-tokens budget
    serving_ttft_ms: Optional[float] = 500.0  # time-to-first-token budget
    serving_batch: int = 32
    # Parallelism (inputs, not search variables in v0)
    tp: int = 8
    pp: int = 1
    dp: int = 8
    # Optional cluster-size floor. When set, candidate DP is derived from
    # ceil(cluster / (TP x PP x CP)) and rounded up to a multiple of EP.
    # This keeps TP/CP search from silently buying more GPUs while leaving
    # a stale scalar DP attached to every candidate.
    training_cluster_gpus: Optional[int] = None
    # Optional hard ceiling on the evaluated training world
    # (TP x PP x CP x DP). Set this equal to training_cluster_gpus when the
    # deployment has an exact, fixed-size training cluster rather than a
    # minimum scale target.
    max_training_cluster_gpus: Optional[int] = None
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
    # Wave 18h: optional vocab sweep axis. When set (e.g. [32000, 128256]),
    # every candidate generator runs once per vocab size; the quality model's
    # vocab_residual gives the axis an interior optimum (undersized vocab
    # loses data efficiency, oversized vocab displaces non-embedding params).
    vocab_options: Optional[List[int]] = None
    # Wave 18h: strict quality ordering. When True, the picker and pareto.csv
    # rank by point-estimate (shape-prior-adjusted) loss with NO uncertainty
    # noise-band bucketing — the top row is the argmin-loss candidate, and any
    # throughput/memory tiebreak applies only to exact loss ties.
    strict_quality: bool = False
    # Precision search space
    precision_configs: Optional[List[str]] = None  # None = enumerate defaults
    kv_bits_options: Optional[List[int]] = None     # None = [16, 8, 4]
    # v1+ hooks
    allow_moe: bool = False
    allow_state: bool = False
    # v1-fix Wave 3 (Jun 2026): when True, the quality model's INFEASIBLE=1e6
    # sentinel is demoted to a warning so the optimizer surfaces the
    # least-bad candidate. Use for outside-coverage corners (e.g. 1B @ 2M,
    # 750B @ 2M, 1000B @ 128k+) where every candidate hits the sentinel
    # but the user still wants a "best effort, marked uncovered" answer.
    allow_quality_sentinel: bool = False
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

    # Wave 9 (Jun 2026): compressed-attention variants. When True, the dense
    # generator emits one extra candidate per lattice point for each enabled
    # variant. Default off so existing callers don't change behavior.
    # See plan/redesign/09-compressed-attention-coverage.md.
    allow_csa: bool = False
    csa_block_size_options: Optional[List[int]] = None        # default [64, 128]
    csa_top_k_options: Optional[List[int]] = None             # default [8, 16, 32]
    csa_compression_dim: int = 64
    allow_indexshare: bool = False
    indexshare_num_buckets_options: Optional[List[int]] = None  # default [64, 128]
    indexshare_top_k_options: Optional[List[int]] = None        # default [4, 8]
    indexshare_index_dim: int = 64
    allow_msa: bool = False
    msa_window_options: Optional[List[int]] = None              # default [512, 1024]
    msa_dilated_top_k_options: Optional[List[int]] = None       # default [64]
    msa_global_top_k_options: Optional[List[int]] = None        # default [16]

    # Evaluated architecture transforms. These replace the old CLI-only
    # post-search stamps: every candidate is transformed before evaluation.
    force_nsa: bool = False
    nsa_compress_block_size: int = 64
    nsa_compress_block_stride: int = 16
    nsa_select_block_size: int = 64
    nsa_select_top_k: int = 16
    nsa_window_size: int = 512
    force_yoco: bool = False
    yoco_n_self_attn_layers: int = 1
    yoco_share_pattern: str = "single_source"

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
    # Training world = TP × PP × DP × CP; EP partitions DP. A serving
    # instance spans TP × PP × CP × EP because serving has no DP replicas.
    cp: int = 1
    cp_method: str = "ring"           # "ring" | "ulysses"
    cp_options: Optional[List[int]] = None     # default [1, 2, 4, 8]

    # Wave 4 TP search: TP is now a search variable. When tp_options is
    # None, the enumerator uses [tp] as a single-element list (back-compat
    # with the v0/v1 behavior). When tp_options is a list, every candidate
    # generator sweeps it; each emitted CandidateArch records its tp_degree
    # so the Pareto frontier ranks across TP alongside the other axes.
    #
    # The grid driver (`scripts/_generator_payload.py`) populates
    # tp_options from `context_aware_parallelism`: typically
    # [base_tp, 2*base_tp, 4*base_tp] capped at the NVLink island size,
    # so small-model long-context cells can pick a higher TP and surface
    # a fits-in-HBM solution.
    tp_options: Optional[List[int]] = None

    # Wave 8 (Jun 2026) — PP/EP/DP as search variables (feedback-review #6).
    # pp_options: pipeline-parallel degrees to search. Defaults to [pp].
    #   When > 1 value, the optimizer enumerates per-generator. Mirrors the
    #   tp_options pattern. Caveats: PP multiplies the candidate count
    #   per-generator (use len([1,2,4]) = 3× as a budget for the planner).
    # ep_options: EP search is ALREADY wired via compute_moe_options(ep_degrees=)
    #   inside generate_moe_candidates / generate_moe_hybrid_candidates. EP=1
    #   is excluded for MoE paths by default_ep_options(for_moe=True). No
    #   normalization needed here — handled at the call site.
    # dp_options: DP is intentionally NOT a search variable. DP is derived
    #   from cluster_size / (tp × pp): given a fixed cluster and chosen
    #   (tp, pp), dp is determined. Surface as a read-only field for
    #   diagnostics; do not iterate.
    pp_options: Optional[List[int]] = None

    # v1-fix RoPE scaling: positional-encoding extension method. When
    # `allow_rope_scaling=True`, the optimizer enumerates rope_scaling_methods
    # at each lattice point so the quality model can compare YaRN vs LongRoPE
    # vs none at long context. Methods: "none" | "pi" | "ntk" | "yarn" | "longrope".
    allow_rope_scaling: bool = False
    rope_scaling_methods: Optional[List[str]] = None
    rope_original_max_position: int = 8192    # training context, before extension

    # Wave 18g (Jul 2026): per-layer attention heterogeneity sweep.
    # When `allow_local_global=True`, the search emits interleaved variants
    # of every attention-only candidate (dense-FFN and MoE, GQA-global and
    # MLA-global): `n_local_attn_layers` layers of sliding-window attention
    # at each window in `local_window_options`, at each local:global ratio
    # in `local_global_ratio_options` ("3:1" = 3 local per 1 global —
    # Llama-4; "1:1" = alternating — GPT-OSS / Gemma-2).
    allow_local_global: bool = False
    local_window_options: Optional[List[int]] = None       # default [1024, 4096]
    local_global_ratio_options: Optional[List[str]] = None  # default ["1:1","3:1","7:1"]

    # Wave 10A (Jun 2026): explicit training micro-batch so callers can pin
    # the training-step batch independent of serving_batch. Default None →
    # throughput model picks `max(8, arch.training_micro_batch)` as before.
    # See plan/redesign/10-optimizer-self-consistency.md Change A.
    training_micro_batch: Optional[int] = None
    pipeline_microbatches: int = 1

    # Search ergonomics. Defaults preserve exhaustive v0.3 behavior.
    max_candidates: Optional[int] = None       # cap after deterministic dedupe
    progress_every: int = 0                    # stderr update interval; 0 disables

    # Wave 8b (Jun 2026) — Two-stage candidate evaluation. When set, after
    # enumerate+dedupe the optimizer cheaply ranks every candidate by the
    # `_cheap_quality_rank` proxy (chinchilla baseline × shape penalty —
    # microseconds per candidate) and only the top-N survive to the full
    # `evaluate_candidate` cost (throughput + quality residuals + Pareto).
    # This decouples cost control from arch selection: every shape competes
    # fairly in the cheap stage, then the cap is a pure perf knob.
    #
    # Default None preserves the exhaustive v1 behavior. Set to ~500 once
    # `tests/test_two_stage_evaluation.py` confirms cheap-rank monotonicity
    # against full evaluate; reduce to ~200 in production cells.
    max_full_evaluations: Optional[int] = None
    # Wave 34: cheap local refinement. After the capped pool is fully
    # evaluated, the search revisits the UNCAPPED enumeration and
    # full-evaluates up to this many lattice neighbors of the per-class
    # Pareto leaders (same structural class, |d_model step| <= 512,
    # |n_layers delta| <= 3). 0 disables. Only active when the
    # max_candidates cap actually dropped candidates — with an uncapped
    # search every neighbor was already evaluated.
    local_refine_budget: int = 96

    def __post_init__(self):
        self._validate_positive_inputs()
        if self.scheduler not in {"continuous", "static", "chunked"}:
            raise ValueError(
                "scheduler must be 'continuous', 'static', or 'chunked'"
            )
        if self.traffic_mix is not None:
            raise ValueError(
                "traffic_mix is not yet a calibrated performance-model input; "
                "split the workload into separate optimize() calls instead"
            )
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
        # Wave 4: tp_options mirrors cp_options. Default to [tp] when the
        # caller didn't supply a list — preserves the existing single-TP
        # behavior for every consumer that still passes scalar `tp`.
        if self.tp_options is None:
            self.tp_options = [self.tp]
        else:
            cleaned_tp = [int(v) for v in self.tp_options if int(v) > 0]
            if not cleaned_tp:
                raise ValueError("tp_options must contain at least one positive value")
            # Deterministic ascending order; dedupe.
            seen = set()
            ordered = []
            for v in cleaned_tp:
                if v not in seen:
                    seen.add(v)
                    ordered.append(v)
            self.tp_options = sorted(ordered)
        # Wave 8: pp_options mirrors tp_options. Default to [pp] for back-compat.
        if self.pp_options is None:
            self.pp_options = [self.pp]
        else:
            cleaned_pp = [int(v) for v in self.pp_options if int(v) > 0]
            if not cleaned_pp:
                raise ValueError("pp_options must contain at least one positive value")
            seen = set()
            ordered = []
            for v in cleaned_pp:
                if v not in seen:
                    seen.add(v)
                    ordered.append(v)
            self.pp_options = sorted(ordered)
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
        # Wave 18g: normalize local:global sweep options.
        if self.allow_local_global:
            if self.local_window_options is None:
                self.local_window_options = [1024, 4096]
            self.local_window_options = [int(w) for w in self.local_window_options if int(w) > 0]
            if not self.local_window_options:
                raise ValueError("local_window_options must contain at least one positive window")
            if self.local_global_ratio_options is None:
                self.local_global_ratio_options = ["1:1", "3:1", "7:1"]
            cleaned_ratios = []
            for r in self.local_global_ratio_options:
                # Idempotent: dataclasses.replace() re-runs __post_init__, so
                # already-normalized (L, G) pairs must pass through unchanged.
                if isinstance(r, (tuple, list)) and len(r) == 2:
                    l, g = int(r[0]), int(r[1])
                else:
                    s = str(r).strip()
                    parts = s.split(":")
                    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
                        raise ValueError(
                            f"Bad local:global ratio {r!r}; expected 'L:G' with positive "
                            f"integers, e.g. '3:1' (3 local layers per global layer)")
                    l, g = int(parts[0]), int(parts[1])
                if l < 1 or g < 1:
                    raise ValueError(f"Bad local:global ratio {r!r}; both sides must be >= 1")
                cleaned_ratios.append((l, g))
            self.local_global_ratio_options = cleaned_ratios
        if self.objective_profile not in OBJECTIVE_PROFILES:
            raise ValueError(
                f"Unknown objective_profile {self.objective_profile!r}; "
                f"expected one of {sorted(OBJECTIVE_PROFILES)}"
            )

    def _validate_positive_inputs(self) -> None:
        checks = {
            "target_params_b": self.target_params_b,
            "training_tokens": self.training_tokens,
            "pretraining_context_length": self.pretraining_context_length,
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
            "training_micro_batch": self.training_micro_batch,
            "unique_training_tokens": self.unique_training_tokens,
            "training_cluster_gpus": self.training_cluster_gpus,
            "max_training_cluster_gpus": self.max_training_cluster_gpus,
        }
        for name, value in optional_positive.items():
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.param_tolerance < 0:
            raise ValueError("param_tolerance must be >= 0")
        if self.progress_every < 0:
            raise ValueError("progress_every must be >= 0")
        if self.pipeline_microbatches <= 0:
            raise ValueError("pipeline_microbatches must be > 0")
        if self.unique_training_tokens is not None \
                and self.unique_training_tokens > self.training_tokens:
            raise ValueError(
                "unique_training_tokens cannot exceed training_tokens"
            )
        if (self.training_cluster_gpus is not None
                and self.max_training_cluster_gpus is not None
                and self.training_cluster_gpus
                > self.max_training_cluster_gpus):
            raise ValueError(
                "training_cluster_gpus cannot exceed "
                "max_training_cluster_gpus"
            )
        if self.quality_model_version not in {
            "effective_capacity_v2",
            "legacy_residual_v1",
        }:
            raise ValueError(
                "quality_model_version must be 'effective_capacity_v2' "
                "or 'legacy_residual_v1'"
            )
        if self.force_nsa:
            for name in (
                "nsa_compress_block_size", "nsa_compress_block_stride",
                "nsa_select_block_size", "nsa_select_top_k",
                "nsa_window_size",
            ):
                if int(getattr(self, name)) <= 0:
                    raise ValueError(f"{name} must be > 0 when force_nsa=True")
        if self.force_yoco:
            if int(self.yoco_n_self_attn_layers) <= 0:
                raise ValueError("yoco_n_self_attn_layers must be > 0")
            if self.yoco_share_pattern != "single_source":
                raise ValueError(
                    "yoco_share_pattern must be 'single_source'; "
                    "other sharing topologies are not calibrated"
                )
        for name in ("moe_n_experts_options", "moe_top_k_options",
                     "ep_options", "mla_kv_latent_options",
                     "mla_q_latent_options"):
            values = getattr(self, name)
            if values is not None and any(int(v) <= 0 for v in values):
                raise ValueError(f"{name} values must be > 0")
        # Wave 21: EP=1 is a legal MoE plan (experts TP-sharded across the
        # TP group, vLLM-style). Explicit ep_options may include 1; the
        # DEFAULT search space still prefers EP >= 2 via
        # default_ep_options(for_moe=True), which is a search-space prior,
        # not a validity constraint.
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
    activation_precision: str = "bf16"
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
    hybrid_ratio: str = ""                 # exact state:attention ratio
    derived_d_state: int = 0
    crossover_seq_len: float = 0.0
    # v1-fix MLA: when set, this candidate uses MLA attention. The latent
    # dimensions feed both the throughput model (KV-bandwidth term) and
    # the quality model (small compression-quality penalty).
    attention_type: str = "full"           # "full" | "mla" | "nsa" | "swa" | "csa" | "indexshare" | "msa"
    mla_kv_latent_dim: int = 0             # c_kv (0 when type=full)
    mla_q_latent_dim: int = 0              # c_q
    mla_rope_head_dim: int = 0             # d_rope
    mla_nope_head_dim: int = 0             # d_nope
    # v1-fix SWA: when set, KV cache is capped at min(seq_len, window_size)
    # per token; decode TBT reads the smaller cache; prefill compute is
    # O(N*W) instead of O(N^2); quality model adds a small residual when
    # the workload context exceeds the window.
    swa_window: int = 0                    # 0 = no sliding window (full attention)
    # Wave 18g: per-layer attention heterogeneity — local:global interleave.
    # When > 0, this many layers use sliding-window attention at
    # `swa_window` (GQA projection) while the remaining layers are global
    # (full/GQA or MLA per `attention_type`), spread evenly through the
    # stack. n_local_attn_layers == 0 with swa_window > 0 keeps the legacy
    # whole-model SWA semantics.
    n_local_attn_layers: int = 0
    # Evaluated NSA and YOCO parameters.
    nsa_compress_block_size: int = 0
    nsa_compress_block_stride: int = 0
    nsa_select_block_size: int = 0
    nsa_select_top_k: int = 0
    nsa_window_size: int = 0
    yoco_n_self_attn_layers: int = 0
    yoco_share_pattern: str = "single_source"
    sparsity_2_4: Optional[Dict[str, bool]] = None
    # Wave 9 (Jun 2026): compressed-attention variants. See
    # plan/redesign/09-compressed-attention-coverage.md. Each variant
    # carries its own config dict; the throughput model reads them via
    # the kv_bytes_per_token_per_layer helper.
    csa_block_size: int = 0                # 0 → unused; common: 64, 128
    csa_top_k_blocks: int = 0              # 0 → unused; common: 8, 16, 32
    csa_compression_dim: int = 0           # 0 → defaults to d_head
    indexshare_num_buckets: int = 0
    indexshare_top_k_buckets: int = 0
    indexshare_index_dim: int = 0
    msa_window_size: int = 0
    msa_dilated_top_k: int = 0
    msa_global_top_k: int = 0
    # v1-fix MTP
    mtp_n_predict_depths: int = 0          # 0 = MTP off
    mtp_depth_n_layers: int = 1
    mtp_train_loss_weight: float = 0.3
    # v1-fix CP
    cp_degree: int = 1                     # 1 = no context parallelism
    cp_method: str = "ring"                # "ring" | "ulysses"
    # Wave 4 TP search: TP is now per-candidate (the enumerator sweeps
    # `constraints.tp_options`). Default 0 means "unset" — evaluate_candidate
    # falls back to constraints.tp. The candidate generators below always set
    # this explicitly to the current tp in their sweep, and the baseline
    # loader sets it from the config's parallelism block. (Bug fix Jul 2026:
    # the old default of 1 was indistinguishable from an explicit TP=1, so
    # the constraints.tp fallback in evaluate_candidate could never fire and
    # baseline-config evaluations silently ran the throughput model at TP=1 —
    # wrong TTFT, wrong TP all-reduce/cross-node comm in TBT, wrong per-GPU
    # memory.)
    tp_degree: int = 0
    # Wave 8 (Jun 2026) — PP search. Carries the per-candidate pp degree.
    # Same semantics as tp_degree: 0 = unset → constraints.pp fallback;
    # generators and the baseline loader set it explicitly.
    pp_degree: int = 0
    # Effective data-parallel degree used for evaluation. Search generators
    # leave this at zero; evaluate_candidate derives it from either scalar DP
    # or DeploymentConstraints.training_cluster_gpus.
    dp_degree: int = 0
    # v1-fix RoPE scaling
    rope_scaling_method: str = "none"
    rope_scaling_factor: float = 1.0
    rope_original_max_position: int = 8192

    def __post_init__(self) -> None:
        """Reject YOCO topology labels the evaluator cannot distinguish."""
        if (int(self.yoco_n_self_attn_layers or 0) > 0
                and self.yoco_share_pattern != "single_source"):
            raise ValueError(
                "yoco_share_pattern must be 'single_source'; "
                "other sharing topologies are not calibrated"
            )


def _candidate_training_replica_gpus(
    cand: CandidateArch,
    constraints: Optional[DeploymentConstraints] = None,
) -> int:
    tp = int(getattr(cand, "tp_degree", 0) or 0)
    if tp <= 0:
        tp = int(getattr(constraints, "tp", 1) or 1)
    pp = int(getattr(cand, "pp_degree", 0) or 0)
    if pp <= 0:
        pp = int(getattr(constraints, "pp", 1) or 1)
    cp = max(1, int(getattr(cand, "cp_degree", 1) or 1))
    return max(1, tp * pp * cp)


def _effective_candidate_dp(
    cand: CandidateArch,
    constraints: DeploymentConstraints,
) -> int:
    """Resolve DP without changing the requested training-cluster floor."""
    cluster = getattr(constraints, "training_cluster_gpus", None)
    if cluster is None:
        return max(1, int(getattr(constraints, "dp", 1) or 1))
    replica_gpus = _candidate_training_replica_gpus(cand, constraints)
    minimum_dp = max(1, math.ceil(int(cluster) / replica_gpus))
    ep = (
        max(1, int(getattr(cand, "ep_degree", 1) or 1))
        if getattr(cand, "moe", None) else 1
    )
    return max(ep, math.ceil(minimum_dp / ep) * ep)


def _evaluated_training_replica_gpus(ev: "EvaluatedCandidate") -> int:
    return _candidate_training_replica_gpus(ev.arch)


def _evaluated_training_tps_per_gpu(ev: "EvaluatedCandidate") -> float:
    return float(ev.training_tps) / _evaluated_training_replica_gpus(ev)


def _evaluated_serving_instance_gpus(ev: "EvaluatedCandidate") -> int:
    ep = max(1, int(getattr(ev.arch, "ep_degree", 1) or 1))
    return _evaluated_training_replica_gpus(ev) * ep


@dataclass
class GuardResult:
    """Wave 13 (Jun 2026): one named feasibility guard's outcome."""
    name: str
    triggered: bool
    fails_feasibility: bool
    is_warning: bool
    message: str = ""
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class Feasibility:
    """Wave 13 (Jun 2026): structured feasibility report for one candidate.

    Replaces the previously-scattered guards in evaluate_candidate:
      - memory_extreme_overflow (mem > 10x HBM)
      - precision_unsupported (qual.predicted_loss > 1e4 sentinel)
      - quality_sentinel_tripped (post-pick loss > 10x baseline)
      - tbt_budget_warning (soft — does not fail)
      - ttft_budget_warning (soft — does not fail)
      - hbm_spill_warning (soft — annotates spill tier)

    Back-compat: EvaluatedCandidate.meets_constraints still set from
    feasibility.is_feasible. Old code paths reading meets_constraints
    continue to work; new code paths can read structured guards.
    """
    is_feasible: bool
    guards: Dict[str, GuardResult] = field(default_factory=dict)

    @property
    def violated_guards(self) -> List[str]:
        return [n for n, g in self.guards.items() if g.fails_feasibility and g.triggered]

    @property
    def warning_guards(self) -> List[str]:
        return [n for n, g in self.guards.items() if g.is_warning and g.triggered]


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
    serving_request_latency_ms: float = 0.0
    memory_per_gpu_gb: float = 0.0
    # Serving regime analysis
    binding_serving_regime: str = ""  # "prefill-heavy" | "decode-heavy" | "mixed"
    binding_reason: str = ""
    # Feasibility
    meets_constraints: bool = True
    constraint_violations: List[str] = field(default_factory=list)
    # Wave 13 (Jun 2026): structured feasibility report. When set, every
    # named guard reports its trigger state. meets_constraints above
    # equals feasibility.is_feasible by construction.
    feasibility: Optional[Feasibility] = None
    # Wave 18a (Jun 2026): factorized architecture signature. Populated
    # lazily via `.signature` (see below) so existing constructors don't
    # need to pass it. When present, downstream consumers (Pareto records,
    # decision diagnostics, calibration ingestion, family stratification)
    # must classify by this signature rather than by handwritten string
    # matches on `arch.moe` / `arch.state_config`.
    architecture_signature_cached: Optional[Any] = None

    @property
    def signature(self):
        """Return the factorized architecture signature (Wave 18a).

        Cached on first access so repeated calls (Pareto, report,
        stratification) share one derivation.  Any code that previously
        read a handwritten family label should call `.signature` instead —
        the legacy label is available via `.signature.legacy_family` for
        the schema-migration cycle.
        """
        if self.architecture_signature_cached is None:
            from ac.architecture import architecture_signature
            self.architecture_signature_cached = architecture_signature(self.arch)
        return self.architecture_signature_cached


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
    evaluation_failures: int = 0
    evaluation_failure_reasons: Dict[str, int] = field(default_factory=dict)
    # v1-fix Wave 5 follow-up (Jun 2026): raw enumeration size BEFORE the
    # max_candidates cap. Use this as the "did the search space explode"
    # signal — candidates_generated is post-cap and therefore bounded by
    # constraints.max_candidates (typically 400 in production), so it
    # can't distinguish "small search" from "huge enumeration trimmed
    # to 400." Downstream consumers (canonical-shape pin in
    # _generator_payload) gate on this field.
    candidates_enumerated_raw: int = 0
    search_time_sec: float = 0.0
    hardware: str = ""
    constraints: Optional[DeploymentConstraints] = None
    # Binding constraints (populated after optimization)
    binding_constraints: List[str] = field(default_factory=list)
    # Wave 18d (Jun 2026): confidence-aware decision. When present, downstream
    # renderers should prefer this over the legacy `optimal` field, which
    # remains a loss-sorted compatibility projection for one transition
    # version. `decision.status` distinguishes winner / unresolved /
    # out_of_domain / no_physically_feasible_candidate / model_validation_failure.
    decision: Optional[Any] = None  # ac.decision.DecisionAssessment


# Selection profiles mirror the web UI tradeoff presets. Weights sum to 1.0;
# lower score is better. `ttft` is deliberately separate from decode TBT.
OBJECTIVE_PROFILES: Dict[str, Dict[str, float]] = {
    "balanced":      {"loss": 0.30, "tbt": 0.10, "ttft": 0.05, "e2e": 0.10, "tps": 0.15, "mem": 0.10, "train_mem": 0.05, "params": 0.15},
    "quality":       {"loss": 0.90, "tbt": 0.015, "ttft": 0.005, "e2e": 0.01, "tps": 0.03, "mem": 0.02, "params": 0.02},
    "research_quality": {"loss": 1.00, "tbt": 0.0, "ttft": 0.0, "tps": 0.0, "mem": 0.0, "params": 0.0},
    "loss_only":     {"loss": 1.00, "tbt": 0.0, "ttft": 0.0, "tps": 0.0, "mem": 0.0, "params": 0.0},
    "latency":       {"loss": 0.15, "tbt": 0.20, "ttft": 0.15, "e2e": 0.25, "tps": 0.05, "mem": 0.10, "params": 0.10},
    "serving_cost":  {"loss": 0.15, "tbt": 0.10, "ttft": 0.05, "e2e": 0.15, "tps": 0.05, "mem": 0.30, "params": 0.20},
    "training_cost": {"loss": 0.20, "tbt": 0.05, "ttft": 0.05, "tps": 0.45, "mem": 0.0, "train_mem": 0.10, "params": 0.15},
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
    if key == "e2e":
        return ev.serving_request_latency_ms
    if key == "tps":
        return _evaluated_training_tps_per_gpu(ev)
    if key == "mem":
        return ev.memory_per_gpu_gb
    if key == "train_mem":
        return float(getattr(
            ev.throughput, "training_memory_per_gpu_gb", 0.0) or 0.0)
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

    --- Division of labor (Wave 12 follow-up, Jun 2026) ---
    `penalties.shape_penalty` (in quality_model) is the canonical mechanism
    for "pathological shapes lose on the Pareto frontier" — its quadratic-
    in-log-ratio formula at C=0.08 puts 27-54% loss on shapes like
    11776x4, which makes them dominated on quality at the Pareto level.
    This picker prior is now scoped to a different role: **tiebreaking
    among Pareto-equivalent survivors** (same loss to within the
    uncertainty noise band). Without this prior, two candidates with
    identical Pareto score but different aspect ratios would tiebreak
    arbitrarily; with it, the picker prefers the one in the [80,160] band.

    Wave 12 follow-up: cap reduced from 25% to 10% because shape_penalty
    now carries the heavy lifting at C=0.08 (was 0.05). The sparse-frontier
    safety net stays — when only one candidate remains, this prior is
    still the only voice against unshippable shapes — but it can't
    override a real Pareto move.
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
    # Wave 12 follow-up: cap reduced from 8x → 4x. Combined with the
    # 0.01-base penalty, this gives 1% at rich-frontier (≥8 candidates)
    # and 4% at single-candidate (vs 25% before). shape_penalty (C=0.08)
    # does the heavy lifting on pathological shapes at the Pareto level.
    sparse_mult = max(1.0, min(4.0, 4.0 / max(pool_size, 1)))

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
    if hw_name in ("b200", "gb200_nvl72"):
        # gb200_nvl72 (Gate-2 Task C) is B200 silicon — same precision set.
        return ["all_bf16", "ffn_fp8", "all_fp8",
                "ffn_fp4", "all_fp4",
                "ffn_mxfp4", "all_mxfp4",
                "ffn_mxfp6", "all_mxfp6"]
    elif hw_name in ("h100", "h800"):
        # h800 (Gate-2 Task C) is H100 silicon with reduced NVLink — same set.
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

    When KV heads are fewer than TP ranks, replicate each KV head across an
    equal-size TP subgroup. Otherwise shard KV heads evenly across TP ranks.
    """
    n_kv_heads = int(n_kv_heads)
    tp_degree = max(1, int(tp_degree))
    if n_kv_heads >= tp_degree:
        return n_kv_heads % tp_degree == 0
    return tp_degree % n_kv_heads == 0


def _gqa_ratios_for_point(
    pt: LatticePoint,
    constraints: DeploymentConstraints,
    tp_degree: Optional[int] = None,
) -> List[int]:
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
    # Wave 4 TP search: prefer the per-call tp_degree so the GQA placement
    # check matches the lattice the candidate was generated against. Fall
    # back to constraints.tp when the caller hasn't yet been threaded
    # through (preserves back-compat for any external callers).
    tp = int(tp_degree if tp_degree is not None else constraints.tp)
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

def _stratified_candidate_cap(candidates: list, cap: int,
                              constraints: "DeploymentConstraints" = None) -> list:
    """Deterministic cap that preserves structural-variant representation.

    Wave 32 fix: the previous cap was a plain even-stride over the deduped
    enumeration ORDER. Variant classes are generated in contiguous blocks
    (per-generator, per-shape loops), so a class occupying a block shorter
    than the stride gets skipped almost entirely: with allow_moe +
    allow_state + compressed attention at 7B/128k, csa/indexshare/msa were
    47% of the 419k enumeration but got 13 of the 400 capped slots — the
    grid could then never surface them regardless of merit (the same bug
    class Wave 8b fixed for max_full_evaluations with family
    stratification).

    Bucket by the cheap structural class (attention_type, has_moe,
    has_state), give every bucket an equal share of the cap (remainder to
    the largest buckets), and select WITHIN each bucket. Deterministic
    for a fixed enumeration order.

    Wave 34 (two-stage default): when `constraints` is provided, the
    within-bucket selection is cheap-rank-GUIDED instead of a blind
    stride — 70% of each bucket's share goes to the best candidates by
    `_cheap_quality_rank` (O(microseconds) loss proxy over the COMPLETE
    deduped enumeration, so every supported possibility is scored before
    anything is dropped), and the remaining 30% is an even stride over
    the rest of the bucket. The stride tail is deliberate: the cheap
    rank is a pure loss proxy that knows nothing about throughput or
    memory, and a bucket selected purely by cheap loss would strip the
    fast/low-memory end of the Pareto surface that budget-bound picks
    (--serving-tbt) and the memory tiebreak need. Without constraints
    the old pure-stride behavior is preserved.
    """
    if cap <= 0 or len(candidates) <= cap:
        return candidates
    buckets: dict = {}
    for c in candidates:
        key = (
            getattr(c, "attention_type", "full") or "full",
            bool(getattr(c, "moe", None)),
            bool(getattr(c, "state_config", None)),
        )
        buckets.setdefault(key, []).append(c)
    # Deterministic bucket order: SMALLEST first (ties by key repr), so a
    # bucket that can't fill its equal share returns the slack to the
    # buckets processed after it — the largest buckets absorb the surplus
    # and the output always reaches min(cap, len(candidates)).
    ordered = sorted(buckets.items(), key=lambda kv: (len(kv[1]), repr(kv[0])))
    out: list = []
    remaining = cap
    for i, (key, bucket) in enumerate(ordered):
        slots_left = len(ordered) - i
        share = max(1, remaining // max(1, slots_left))
        take = min(len(bucket), share, remaining)
        if take <= 0:
            continue
        if len(bucket) <= take:
            picked = bucket
        elif constraints is not None:
            # Two-stage: cheap-rank the WHOLE bucket (complete coverage —
            # every candidate is scored before anything is dropped), then
            # 40% of the share to the cheap-rank top and 60% to an even
            # stride over the bucket in ENUMERATION order.
            #
            # Wave 34 post-probe: the first cut used 70% cheap-rank and
            # strided only the leftovers. That let the cheap rank make
            # within-bucket shape decisions it is not qualified to make —
            # it is blind to MoE granularity (and throughput entirely), so
            # MoE buckets filled up with capacity-ratio-maximizing shapes
            # the full evaluator penalizes, and two cli_smoke pins
            # (dense-vs-MoE argmin picks) flipped. Same lesson as Wave
            # 8b: cheap signals guarantee REPRESENTATION, they must not
            # pick winners. The stride now runs over the whole bucket so
            # the capped pool approximately contains the old stride
            # sample, and the cheap-rank tranche is additive on top; the
            # local-refinement stage is where shape optimization happens.
            try:
                ranked = sorted(
                    bucket,
                    key=lambda c: _cheap_quality_rank(
                        c, constraints.training_tokens,
                        constraints.quality_model_version))
            except Exception:
                ranked = bucket  # cheap rank must never lose candidates
            n_top = max(1, int(take * 0.4))
            n_top = min(n_top, take)
            picked = list(ranked[:n_top])
            picked_ids = set(id(c) for c in picked)
            n_stride = take - len(picked)
            if n_stride > 0:
                step = len(bucket) / n_stride
                for j in range(n_stride):
                    c = bucket[min(int(j * step), len(bucket) - 1)]
                    if id(c) in picked_ids:
                        continue
                    picked.append(c)
                    picked_ids.add(id(c))
                # Backfill collisions from the cheap-rank order.
                k = n_top
                while len(picked) < take and k < len(ranked):
                    c = ranked[k]; k += 1
                    if id(c) not in picked_ids:
                        picked.append(c)
                        picked_ids.add(id(c))
        else:
            step = len(bucket) / take
            picked = [bucket[min(int(j * step), len(bucket) - 1)]
                      for j in range(take)]
        out.extend(picked)
        remaining -= len(picked)
        if remaining <= 0:
            break
    return out


def _select_refinement_neighbors(
    leader_archs: list,
    precap_pool: list,
    evaluated: list,
    budget: int,
) -> list:
    """Wave 34: pick unevaluated lattice neighbors of the Pareto leaders.

    A capped search decides each class's shape from a subsample; this
    second stage densifies the neighborhood of what stage one found. For
    every leader we admit candidates from the PRE-CAP deduped pool (so
    every shape came from a real generator and the parameter ledger is
    trustworthy) that share the leader's structural class and sit within
    one lattice step (|d_model| <= 512, |n_layers| <= 3), ordered by
    shape distance, round-robin across leaders, capped at `budget`.
    """
    if budget <= 0 or not leader_archs or not precap_pool:
        return []

    def _class(c):
        return (
            getattr(c, "attention_type", "full") or "full",
            bool(getattr(c, "moe", None)),
            bool(getattr(c, "state_config", None)),
        )

    def _ident(c):
        return (
            c.d_model, c.n_layers, c.n_heads, c.d_head, c.n_kv_heads,
            c.ffn_dim, c.vocab_size, c.weight_precision, c.ffn_precision,
            c.activation_precision,
            c.kv_cache_bits, _class(c),
            int(getattr(c, "tp_degree", 0) or 0),
            int(getattr(c, "cp_degree", 0) or 0),
            int(getattr(c, "ep_degree", 0) or 0),
        )

    seen = {_ident(c) for c in evaluated}
    per_leader: list = []
    for lead in leader_archs:
        lc = _class(lead)
        near = []
        for c in precap_pool:
            if _class(c) != lc:
                continue
            dd = abs(int(c.d_model) - int(lead.d_model))
            dl = abs(int(c.n_layers) - int(lead.n_layers))
            if dd > 512 or dl > 3:
                continue
            k = _ident(c)
            if k in seen:
                continue
            near.append(((dd, dl, abs(int(c.ffn_dim) - int(lead.ffn_dim))), c))
        near.sort(key=lambda t: t[0])
        per_leader.append([c for _score, c in near])

    out: list = []
    taken = set()
    i = 0
    while len(out) < budget:
        progressed = False
        for lst in per_leader:
            if i < len(lst):
                c = lst[i]
                k = _ident(c)
                if k not in taken:
                    taken.add(k)
                    out.append(c)
                    if len(out) >= budget:
                        break
                progressed = True
        if not progressed:
            break
        i += 1
    return out


def _enumeration_pool_cap(constraints: "DeploymentConstraints") -> int:
    """Per-family retention bound for candidate enumeration (Wave 18f).

    Shared by the four generators and the per-family stride in optimize()
    so enumeration memory and the sampling density stay consistent.
    """
    mc = int(constraints.max_candidates or 0)
    # An explicit final cap is a speed/memory knob. The old 100k floor made
    # `--max-candidates 400` and `--max-candidates 1000` retain 100k rows PER
    # family before the final cap, so all-family searches consumed gigabytes
    # and minutes despite asking for a small search. A 10x deterministic
    # oversample leaves ample material for stratified cheap-rank selection
    # while making the requested bound operational. Uncapped searches keep
    # the historical 200k-per-family retention budget.
    return max(1_000, 10 * mc) if mc > 0 else 200_000


class _BoundedCandidateList(list):
    """Streaming, deterministic decimation for candidate enumeration.

    Perf-fix Wave 18f (Jul 2026): with allow_moe + allow_state + rope/cp
    sweeps at long context, a single family generator materializes 10^6-10^7
    CandidateArch objects (13B @ 128k: generate_moe_candidates alone
    exceeded the sandbox's memory budget and the process was OOM-killed
    before --max-candidates applied). This list subclass bounds retention:
    appends behave normally until 2×cap items are held, then the list is
    decimated in place by a factor of 2 — the same even-stride sampling the
    top-level max_candidates cap uses — and subsequent appends retain every
    2nd, 4th, ... produced item. Memory is bounded at <2×cap entries while
    coverage of the enumeration order stays approximately uniform, and the
    result is deterministic for a fixed enumeration order.
    """

    def __init__(self, cap: int):
        super().__init__()
        self._cap = max(1, int(cap))
        self._keep_every = 1   # retain every N-th produced item
        self._produced = 0     # total items offered via append()

    def append(self, item) -> None:  # type: ignore[override]
        self._produced += 1
        if self._produced % self._keep_every:
            return
        super().append(item)
        if len(self) >= 2 * self._cap:
            self[:] = self[::2]
            self._keep_every *= 2


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

    # Valid precision configs for this hardware
    hw_prec_configs = get_precision_configs_for_hardware(hw_name)
    prec_configs = [p for p in (constraints.precision_configs or []) if p in hw_prec_configs]
    if not prec_configs:
        prec_configs = hw_prec_configs[:3]  # default to first 3

    candidates: List[CandidateArch] = _BoundedCandidateList(_enumeration_pool_cap(constraints))
    mtp_opts, cp_opts, rope_opts, rope_factor = _search_option_lists(constraints)
    # Wave 4: TP is a search variable. tp_options is normalized to [tp]
    # when the caller didn't supply a list, so the legacy single-TP path
    # falls out for free (one pass through the outer loop).
    tp_opts = list(constraints.tp_options or [constraints.tp])

    for tp_d in tp_opts:
      lattice = compute_lattice(
          lattice_hw, precision, tp_d,
          d_model_min=1024, d_model_max=_d_model_max_for_target(constraints.target_params_b),
          d_head_options=[64, 128, 256],
      )

      # Filter to tile-aligned points only
      aligned = [pt for pt in lattice if pt.tile_aligned]
      # Wave 5 follow-up (Jun 2026): bound the lattice point count to
      # keep the inner-loop candidate cross-product in check. Dense has
      # the lightest multiplier so its budget is the largest.
      aligned = _filter_lattice_to_budget(
          aligned, constraints.target_params_b,
          budget=_generation_lattice_budget(constraints, "dense"), family="dense")

      for pt in aligned:
        gqa_ratios = _generation_gqa_ratios(pt, constraints, tp_degree=tp_d)

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
            for n_layers in [
                max(4, round(n_layers_raw) + delta)
                for delta in _generation_layer_deltas(constraints)
            ]:
                total = estimate_params(
                    pt.d_model, pt.n_heads, pt.d_head, pt.ffn_dim_swiglu,
                    n_layers, n_kv_heads, constraints.vocab_size
                )
                if total < lo or total > hi:
                    continue

                # PP divisibility check
                if min(constraints.pp_options) > 1 and n_layers % min(constraints.pp_options) != 0:
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
                            tp_degree=int(tp_d),
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
                                    for dL in _generation_layer_deltas(constraints):
                                        n_layers_mla = max(4, round(n_layers_mla_raw) + dL)
                                        if min(constraints.pp_options) > 1 and n_layers_mla % min(constraints.pp_options) != 0:
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
                                            tp_degree=int(tp_d),
                                        ))

                        # Wave 9 (Jun 2026): compressed-attention emissions.
                        # Only emit once per lattice point (gqa_r == gqa_ratios[0])
                        # so we don't pollute the Pareto with N copies that
                        # differ only in the meaningless GQA ratio. See
                        # plan/redesign/09-compressed-attention-coverage.md.
                        if gqa_r == gqa_ratios[0]:
                            _w9_common = dict(
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
                                tp_degree=int(tp_d),
                            )
                            if getattr(constraints, "allow_csa", False):
                                csa_blocks = constraints.csa_block_size_options or [64, 128]
                                csa_top_ks = constraints.csa_top_k_options or [8, 16, 32]
                                for _bs in csa_blocks:
                                    for _tk in csa_top_ks:
                                        if _tk >= _bs:
                                            continue
                                        candidates.append(CandidateArch(
                                            attention_type="csa",
                                            csa_block_size=int(_bs),
                                            csa_top_k_blocks=int(_tk),
                                            csa_compression_dim=int(constraints.csa_compression_dim),
                                            **_w9_common,
                                        ))
                            if getattr(constraints, "allow_indexshare", False):
                                idx_buckets = constraints.indexshare_num_buckets_options or [64, 128]
                                idx_top_ks = constraints.indexshare_top_k_options or [4, 8]
                                for _nb in idx_buckets:
                                    for _tk in idx_top_ks:
                                        if _tk >= _nb:
                                            continue
                                        candidates.append(CandidateArch(
                                            attention_type="indexshare",
                                            indexshare_num_buckets=int(_nb),
                                            indexshare_top_k_buckets=int(_tk),
                                            indexshare_index_dim=int(constraints.indexshare_index_dim),
                                            **_w9_common,
                                        ))
                            if getattr(constraints, "allow_msa", False):
                                msa_wins = constraints.msa_window_options or [512, 1024]
                                msa_dils = constraints.msa_dilated_top_k_options or [64]
                                msa_globs = constraints.msa_global_top_k_options or [16]
                                for _w in msa_wins:
                                    for _d in msa_dils:
                                        for _g in msa_globs:
                                            candidates.append(CandidateArch(
                                                attention_type="msa",
                                                msa_window_size=int(_w),
                                                msa_dilated_top_k=int(_d),
                                                msa_global_top_k=int(_g),
                                                **_w9_common,
                                            ))

    # Wave 8 PP expansion (Jun 2026): duplicate each candidate per valid
    # pp_degree from constraints.pp_options. Single-pp callers see no
    # change (pp_options=[pp], expansion is a no-op rename). Multi-pp
    # callers get one candidate per (shape × pp_d) combination where
    # n_layers is divisible by pp_d.
    pp_opts_w8 = list(constraints.pp_options or [constraints.pp])
    if len(pp_opts_w8) == 1:
        for _c in candidates:
            _c.pp_degree = int(pp_opts_w8[0])
    else:
        _expanded = []
        for _cand in candidates:
            for _pp_d in pp_opts_w8:
                if _pp_d > 1 and (_cand.n_layers % _pp_d) != 0:
                    continue
                _new = _copy.copy(_cand)
                _new.pp_degree = int(_pp_d)
                _expanded.append(_new)
        candidates = _expanded

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


# =============================================================================
# Lattice-filter trigger (Wave 5 follow-up, Jun 2026)
# =============================================================================
#
# Each generator does: aligned = [lattice points that tile-align], then loops
# over (aligned × gqa_ratios × ep_options × moe_configs × cp_options × ...).
# The cross product blows up at param targets in the 3B–13B range, where
# `aligned` is largest AND the inner combinatorial factors are still large.
# Concretely we measured 2.5M candidates at 7B+8k with allow_moe+allow_state,
# producing ~1-2 GB of Python objects before evaluation.
#
# Fix: when the aligned lattice exceeds a soft budget, keep only the K points
# closest to the Chinchilla-shape baseline for the target parameter count.
# The kept points are the ones with the best (d_model, n_layers) match to a
# well-trained Llama/DeepSeek-style architecture at that scale — i.e. the
# points the Pareto frontier would survive on anyway. Distant lattice points
# (very tall-thin or short-fat shapes) are statistically dominated by
# neighbors and would be pruned by the Pareto, so we skip them at the source.

# Wave 5 follow-up — fairness redesign (Jun 2026).
#
# Previous scheme: lattice budgets were asymmetric (dense=60, moe=8, state=15,
# moe_hybrid=4) to keep total candidate cardinality bounded. This created a
# structural bias: MoE/MoE-hybrid kept only ~15-24% of lattice points (vs
# 100% for dense / 88% for state), so the optimizer at small params + high
# TP literally had fewer distinct (d_model, d_head) shapes to choose from
# for MoE than for hybrid. The winner converged on the Chinchilla-anchor
# shape because the filter kept only the 8 points closest to it.
#
# New scheme: equalize lattice budgets so every path gets the same shape
# diversity, and bound total cardinality via a per-generator inner-loop cap
# (see `_INNER_LOOP_TARGET` below). The cap applies to the largest single
# combinatorial dimension per path (moe_opts for MoE, strategies × types
# for state). Stride-sampling preserves dimensional diversity inside each
# kept set.
#
# The remaining inequality is honest: paths with intrinsically larger
# inner cross-products (MoE-hybrid) still produce more candidates than
# state-only paths, but the *shape* search is now identical across paths.
# Shape choices dominate quality differences; inner-loop choices (which
# EP, which top_k) are finer-grained and the Pareto picks similar answers
# whether the inner sample is 80 or 240 options per shape.
_LATTICE_BUDGET = {
    "dense":      20,   # was 60 — equalizing for shape-diversity fairness
    "moe":        20,   # was 8  — now equal to dense
    "state":      20,   # was 15
    "moe_hybrid": 20,   # was 4
}

# Per-generator inner-loop cap on the LARGEST single combinatorial dimension.
# Applied via _stride_sample() so dimensional diversity is preserved inside
# each kept set. Targeted so `lattice × inner ≈ 100K-200K candidates per
# generator` (the empirical threshold where Python heap stays bounded).
#
#   dense:      inner is small (~30 per pt), no cap needed.
#   moe:        moe_opts is the dominant dimension (~180-240 per pt).
#               Cap to 80 → 20 × 80 = 1600 base × (gqa × precision × cp × ...)
#                                  ≈ 160K candidates per generator.
#   state:      placement_strategies × state_types is ~15 per pt.
#               No cap needed; dimension is intrinsically small.
#   moe_hybrid: moe_opts × strategies × types. Cap moe_opts to 40
#               (half of MoE's because moe_hybrid multiplies state).
#               20 × 40 × ~6 = 4800 base × (gqa × precision × cp × ...)
#                            ≈ 240K candidates per generator.
_INNER_LOOP_TARGET = {
    "dense":      None,   # no inner cap
    "moe":         80,    # cap moe_opts per lattice point
    "state":      None,   # placement × types already small
    "moe_hybrid":  40,    # cap moe_opts for moe_hybrid (state inner stays)
}


def _stride_sample(items, target):
    """Deterministic stride-sample of `items` down to `target` while preserving
    coverage of the input order. Used by the inner-loop cap to keep N diverse
    candidates from a list of M, where M > N.

    Identical to the stride used by max_candidates at the top-level optimize()
    cap (around line 2466) — chosen for reproducibility (no RNG) and because
    it preserves the natural ordering of the source list (which generators
    already sort by structural diversity).
    """
    if target is None or len(items) <= target:
        return list(items)
    if target <= 0:
        return []
    step = len(items) / target
    return [items[min(int(i * step), len(items) - 1)] for i in range(target)]


def _endpoint_sample(items, target):
    """Deterministically sample a sequence while retaining both endpoints."""
    values = list(items)
    if target is None or len(values) <= target:
        return values
    if target <= 0:
        return []
    if target == 1:
        return [values[len(values) // 2]]
    return [
        values[round(i * (len(values) - 1) / (target - 1))]
        for i in range(target)
    ]


def _generation_lattice_budget(
    constraints: DeploymentConstraints,
    family: str,
) -> int:
    """Return the lattice budget, honoring a private bounded-search hint."""
    requested = int(getattr(constraints, "_source_lattice_budget", 0) or 0)
    return min(_LATTICE_BUDGET[family], requested) if requested else _LATTICE_BUDGET[family]


def _generation_gqa_ratios(
    pt: LatticePoint,
    constraints: DeploymentConstraints,
    tp_degree: int,
) -> List[int]:
    ratios = _gqa_ratios_for_point(pt, constraints, tp_degree=tp_degree)
    cap = int(getattr(constraints, "_source_gqa_cap", 0) or 0)
    return _endpoint_sample(ratios, cap) if cap else ratios


def _generation_layer_deltas(constraints: DeploymentConstraints) -> List[int]:
    deltas = [-2, -1, 0, 1, 2]
    cap = int(getattr(constraints, "_source_layer_delta_cap", 0) or 0)
    return _endpoint_sample(deltas, cap) if cap else deltas


def _generation_moe_options(
    options: List[MoEOption],
    constraints: DeploymentConstraints,
    family: str,
    baseline_ffn_dim: int,
) -> List[MoEOption]:
    """Bound MoE source work without silently deleting a requested axis.

    The initial endpoint sample controls cardinality. Coverage repair then
    guarantees that every emitted style, EP degree, expert count, top-k, and
    fine-grained active-capacity ratio still reaches candidate construction.
    The repaired set can exceed the nominal target by a few rows; that is
    intentional because a hard cap that erases an axis is a worse contract.
    """
    target = _INNER_LOOP_TARGET[family]
    source_target = int(getattr(constraints, "_source_moe_option_cap", 0) or 0)
    if source_target:
        target = min(int(target or source_target), source_target)
    if target is None or len(options) <= target:
        return list(options)

    selected = _endpoint_sample(options, target)
    selected_ids = {id(opt) for opt in selected}

    def _granularity(opt: MoEOption):
        if opt.style != "fine":
            return None
        return round(opt.active_ffn_equivalent / max(1, baseline_ffn_dim), 3)

    projections = (
        lambda opt: opt.style,
        lambda opt: opt.ep_degree,
        lambda opt: opt.n_experts,
        lambda opt: opt.top_k,
        _granularity,
    )
    for project in projections:
        represented = {project(opt) for opt in selected}
        for value in dict.fromkeys(project(opt) for opt in options):
            if value in represented:
                continue
            match = next(opt for opt in options if project(opt) == value)
            if id(match) not in selected_ids:
                selected.append(match)
                selected_ids.add(id(match))
            represented.add(value)
    return selected


def _source_generation_slices(
    hw_name: str,
    constraints: DeploymentConstraints,
) -> Tuple[List[DeploymentConstraints], bool]:
    """Build coverage-preserving common-axis slices for a capped search.

    Candidate retention alone does not bound runtime: the generators still
    construct every member of TP x precision x KV x MTP x CP x RoPE before
    the bounded list discards rows. When that common product is large and the
    caller supplied ``max_candidates``, sample deterministic Latin-style
    tuples at the source. Every individual value on every axis occurs in at
    least one tuple. Small products and uncapped searches retain the historic
    exhaustive behavior.
    """
    vocabs = [
        int(v)
        for v in dict.fromkeys(
            constraints.vocab_options or [constraints.vocab_size]
        )
    ]
    mc = int(constraints.max_candidates or 0)

    def _vocab_only_slices() -> List[DeploymentConstraints]:
        return [
            constraints
            if vocab == constraints.vocab_size
            else _dc_replace(constraints, vocab_size=vocab)
            for vocab in vocabs
        ]

    if mc <= 0:
        return _vocab_only_slices(), False

    hw_precisions = get_precision_configs_for_hardware(hw_name)
    precisions = [
        p for p in (constraints.precision_configs or []) if p in hw_precisions
    ]
    if not precisions:
        precisions = hw_precisions[:3]
    mtp_opts, cp_opts, rope_opts, _ = _search_option_lists(constraints)
    axes = [
        vocabs,
        list(constraints.tp_options or [constraints.tp]),
        precisions,
        list(constraints.kv_bits_options),
        mtp_opts,
        cp_opts,
        rope_opts,
    ]
    total = math.prod(len(axis) for axis in axes)
    max_axis = max(len(axis) for axis in axes)
    # A source tuple still expands over shapes, GQA, depth, and architecture
    # families. Budget roughly one tuple per 64 requested final candidates.
    budget = min(total, max(max_axis, math.ceil(mc / 64)))
    if total <= max(32, budget):
        return _vocab_only_slices(), False

    def _combo_at(flat_index: int) -> tuple:
        values = [None] * len(axes)
        remainder = int(flat_index)
        for axis_index in range(len(axes) - 1, -1, -1):
            axis = axes[axis_index]
            values[axis_index] = axis[remainder % len(axis)]
            remainder //= len(axis)
        return tuple(values)

    # The rotated tuples cover every value of every axis in max_axis rows.
    combos: List[tuple] = []
    seen = set()
    for i in range(max_axis):
        combo = tuple(
            axis[(i + axis_index) % len(axis)]
            for axis_index, axis in enumerate(axes)
        )
        if combo not in seen:
            seen.add(combo)
            combos.append(combo)

    # Fill the remaining budget with an even traversal of the full product.
    for i in range(budget):
        combo = _combo_at(min(int(i * total / budget), total - 1))
        if combo not in seen:
            seen.add(combo)
            combos.append(combo)
        if len(combos) >= budget:
            break
    if len(combos) < budget:
        for flat_index in range(total):
            combo = _combo_at(flat_index)
            if combo not in seen:
                seen.add(combo)
                combos.append(combo)
            if len(combos) >= budget:
                break

    slices: List[DeploymentConstraints] = []
    for vocab, tp, precision, kv_bits, mtp, cp, rope in combos:
        sliced = _dc_replace(
            constraints,
            vocab_size=int(vocab),
            vocab_options=[int(vocab)],
            tp=int(tp),
            tp_options=[int(tp)],
            precision_configs=[str(precision)],
            kv_bits_options=[int(kv_bits)],
            mtp_depth_options=[int(mtp)],
            cp=int(cp),
            cp_options=[int(cp)],
            rope_scaling_methods=[str(rope)],
        )
        # Private generation hints deliberately do not change the public
        # constraint schema or serialized recipe identity.
        sliced._source_lattice_budget = 4
        sliced._source_gqa_cap = 3
        sliced._source_layer_delta_cap = 3
        sliced._source_moe_option_cap = 8
        slices.append(sliced)
    return slices, True


# Wave 8 follow-up — family-aware Chinchilla anchor (Jun 2026).
#
# Empirical multipliers per architecture family vs the dense baseline:
#   dense       — baseline; the Llama/DeepSeek empirical fit
#   moe         — narrower + deeper (FFN width absorbed by experts; depth stays).
#                 Llama-style fit to MoE active params overshoots d_model and
#                 undershoots n_layers; scale d down ~15%, scale L up ~15%.
#   hybrid      — wider + shallower. State layers benefit from depth budget
#                 freed up by attention compression; published Jamba/Samba
#                 shapes are 1.1-1.2x dense d_model and 0.85-0.95x n_layers.
#   moe_hybrid  — between MoE and hybrid; modest depth bonus from state.
_FAMILY_ANCHOR_SCALE = {
    "dense":      (1.00, 1.00),   # baseline
    "moe":        (0.85, 1.15),   # narrower + deeper
    "hybrid":     (1.15, 0.90),   # wider + shallower
    "moe_hybrid": (1.00, 1.05),   # near-baseline + slightly deeper
}


def _chinchilla_reference_shape(
    target_params_b: float,
    family: str = "dense",
) -> Tuple[int, int]:
    """Approximate (d_model, n_layers) for a Llama/DeepSeek-style architecture
    at the given parameter target. Used as the anchor for the lattice filter.

    Empirical fit to Llama-1/2/3 + DeepSeek-V2/V3 + Qwen-2.5 published
    architectures: d_model ≈ 1024 × N_B^0.33, n_layers ≈ 16 × N_B^0.34.

    Wave 8 follow-up (Jun 2026): accepts a `family` parameter so MoE / hybrid /
    MoE-hybrid filters pull toward shapes that actually suit them. Without
    this, the dense anchor systematically biased MoE-hybrid toward shapes that
    don't reflect Jamba/DeepSeek-V3 / Samba published architectures.
    """
    # Wave 18h: use the SAME fitted shape law as penalties.shape_penalty
    # (d_opt = K_W × N^gamma_W with N in raw params). The previous ad-hoc
    # fit (1024 × N_B^0.33) put the 7B anchor at d≈1965 while the quality
    # model's own optimum was d≈4104 — the lattice filter pulled candidates
    # toward shapes the quality model then penalized.
    try:
        from .penalties import (
            SHAPE_K_W, SHAPE_GAMMA_W, SHAPE_K_D, SHAPE_GAMMA_D,
        )
    except ImportError:
        from penalties import (
            SHAPE_K_W, SHAPE_GAMMA_W, SHAPE_K_D, SHAPE_GAMMA_D,
        )
    n_raw = max(0.5, float(target_params_b)) * 1e9
    base_d = SHAPE_K_W * (n_raw ** SHAPE_GAMMA_W)
    base_L = SHAPE_K_D * (n_raw ** SHAPE_GAMMA_D)
    scale_d, scale_L = _FAMILY_ANCHOR_SCALE.get(family, (1.0, 1.0))
    d_model = int(round(base_d * scale_d))
    n_layers = int(round(base_L * scale_L))
    return d_model, max(2, n_layers)


def _filter_lattice_to_budget(
    aligned: List,
    target_params_b: float,
    budget: int = 30,
    family: str = "dense",
) -> List:
    """When `aligned` exceeds `budget`, return the `budget` points closest
    to the Chinchilla reference shape for the target parameter count.

    Distance metric: log-space Euclidean over (d_model, n_layers via depth
    proxy d_head). Log-space because architectures vary geometrically.

    No-op when len(aligned) <= budget.
    """
    if len(aligned) <= budget:
        return aligned
    ref_d, _ref_L = _chinchilla_reference_shape(target_params_b, family=family)
    # The lattice points carry d_model and d_head but not n_layers (n_layers
    # is fit per-point inside the inner loop). Use d_head as the depth proxy:
    # higher d_head implies fewer attention heads at the same d_model, which
    # correlates with shallower depth in the param-budget calculation.
    import math as _m
    def dist(pt):
        dm = max(1, pt.d_model)
        dh = max(1, pt.d_head)
        return (_m.log2(dm / ref_d)) ** 2 + 0.5 * (_m.log2(dh / 128)) ** 2
    ranked = sorted(aligned, key=dist)
    # Wave 18h: with the FFN-stride fix the lattice is a dense 64-step width
    # grid, so the `budget` nearest points would all sit within a few % of
    # ref_d and the Pareto would lose genuine width diversity (throughput
    # often prefers wider-than-anchor shapes). Keep the nearest half, then
    # stride-sample the remaining distance-sorted tail so the kept set still
    # spans the full width range.
    near = ranked[: max(1, budget // 2)]
    tail = ranked[max(1, budget // 2):]
    n_tail_keep = budget - len(near)
    if n_tail_keep > 0 and tail:
        step = max(1, len(tail) // n_tail_keep)
        near = near + tail[::step][:n_tail_keep]
    return near


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

    hw_prec_configs = get_precision_configs_for_hardware(hw_name)
    prec_configs = [p for p in (constraints.precision_configs or []) if p in hw_prec_configs]
    if not prec_configs:
        prec_configs = hw_prec_configs[:3]

    # v1-fix Wave 3 follow-up (Jun 2026): drop EP=1 from MoE/hybrid
    # candidate enumeration. EP=1 makes each rank hold the full set of
    # experts, which at any realistic MoE size blows past HBM and was
    # being silently caught via the memory→INFEASIBLE back-door. With
    # the back-door now removed (Wave 3 trace_post_filter fix), we have
    # to exclude EP=1 here at the source so the search doesn't waste
    # candidates on a configuration that's never workable.
    #
    # Wave 24 (P0): also cap EP at the requested DP degree. The Wave-19
    # training-throughput math prices MoE with EP laying over DP (each EP
    # rank routes its own microbatch), which is only physical when
    # EP <= DP. Previously the enumerator would happily pick EP=16 with
    # DP=4 on B200 (default_ep_options runs up to the NVLink-domain cap
    # of 72), and cli_compile.py would confess with a post-hoc WARNING
    # while still emitting the training-TPS number computed under the
    # violated assumption. Filtering here makes the guard unreachable.
    ep_opts = _filter_ep_options_by_dp(
        constraints.ep_options or default_ep_options(hw_name, for_moe=True),
        dp=(1 if constraints.training_cluster_gpus is not None
            else constraints.dp),
        source=("user-supplied ep_options"
                if constraints.ep_options is not None
                else "hardware default"),
    )
    mtp_opts, cp_opts, rope_opts, rope_factor = _search_option_lists(constraints)
    # Wave 4: TP search. Same loop pattern as the dense path.
    tp_opts = list(constraints.tp_options or [constraints.tp])

    # v1-fix Part B: sweep n_dense_ffn_layers. Default is [0] (pure MoE, the
    # original v1-MoE behavior). [0, 1, 2, 3] covers the common dense-prefix
    # range used by DeepSeek-V3 / Qwen3-MoE / similar.
    dense_ffn_layer_opts = constraints.dense_ffn_layer_options
    if dense_ffn_layer_opts is None:
        dense_ffn_layer_opts = [0]

    candidates: List[CandidateArch] = _BoundedCandidateList(_enumeration_pool_cap(constraints))

    for tp_d in tp_opts:
      lattice = compute_lattice(
          lattice_hw, precision, tp_d,
          d_model_min=1024, d_model_max=_d_model_max_for_target(constraints.target_params_b),
          d_head_options=[64, 128, 256],
      )
      aligned = [pt for pt in lattice if pt.tile_aligned]
      # Wave 5 follow-up (Jun 2026): when the aligned lattice is huge,
      # drop distant-from-Chinchilla points to bound the candidate
      # cross-product. Threshold is conservative — the kept K points
      # always include the ones a sane Pareto would survive on.
      aligned = _filter_lattice_to_budget(
          aligned, constraints.target_params_b,
          budget=_generation_lattice_budget(constraints, "moe"), family="moe")

      for pt in aligned:
        # Use the dense lattice's GQA shape sweep (mirrors dense path).
        gqa_ratios = _generation_gqa_ratios(pt, constraints, tp_degree=tp_d)

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
        # Wave 5 follow-up fairness redesign: stride-sample moe_opts down
        # to _INNER_LOOP_TARGET["moe"] so MoE doesn't blow past its
        # generator-cap on total cardinality. Stride preserves diversity
        # in n_experts × top_k × expert_dim × ep ordering.
        moe_opts = _generation_moe_options(
            moe_opts, constraints, "moe", pt.ffn_dim_swiglu,
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

                for n_layers in [
                    max(4, round(n_layers_raw) + d)
                    for d in _generation_layer_deltas(constraints)
                ]:
                    if min(constraints.pp_options) > 1 and n_layers % min(constraints.pp_options) != 0:
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
                                                tp_degree=int(tp_d),
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
                                                            tp_degree=int(tp_d),
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

    # Wave 8 PP expansion (Jun 2026): duplicate each candidate per valid
    # pp_degree from constraints.pp_options. Single-pp callers see no
    # change (pp_options=[pp], expansion is a no-op rename). Multi-pp
    # callers get one candidate per (shape × pp_d) combination where
    # n_layers is divisible by pp_d.
    pp_opts_w8 = list(constraints.pp_options or [constraints.pp])
    if len(pp_opts_w8) == 1:
        for _c in candidates:
            _c.pp_degree = int(pp_opts_w8[0])
    else:
        _expanded = []
        for _cand in candidates:
            for _pp_d in pp_opts_w8:
                if _pp_d > 1 and (_cand.n_layers % _pp_d) != 0:
                    continue
                _new = _copy.copy(_cand)
                _new.pp_degree = int(_pp_d)
                _expanded.append(_new)
        candidates = _expanded

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

    hw_prec_configs = get_precision_configs_for_hardware(hw_name)
    prec_configs = [p for p in (constraints.precision_configs or []) if p in hw_prec_configs]
    if not prec_configs:
        prec_configs = hw_prec_configs[:3]

    strategies = constraints.placement_strategies or ["first_periodic_last", "interleaved", "periodic"]
    mtp_opts, cp_opts, rope_opts, rope_factor = _search_option_lists(constraints)
    # Wave 4: TP search.
    tp_opts = list(constraints.tp_options or [constraints.tp])

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

    candidates: List[CandidateArch] = _BoundedCandidateList(_enumeration_pool_cap(constraints))

    for tp_d in tp_opts:
      lattice = compute_lattice(
          lattice_hw, precision, tp_d,
          d_model_min=1024, d_model_max=_d_model_max_for_target(constraints.target_params_b),
          d_head_options=[64, 128, 256],
      )
      aligned = [pt for pt in lattice if pt.tile_aligned]
      # Wave 5 follow-up (Jun 2026): when the aligned lattice is huge,
      # drop distant-from-Chinchilla points to bound the candidate
      # cross-product. Threshold is conservative — the kept K points
      # always include the ones a sane Pareto would survive on.
      aligned = _filter_lattice_to_budget(
          aligned, constraints.target_params_b,
          budget=_generation_lattice_budget(constraints, "state"), family="state")

      for pt in aligned:
        # GQA sweep (same as dense path)
        gqa_ratios = _generation_gqa_ratios(pt, constraints, tp_degree=tp_d)

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

            for n_layers in [
                max(4, round(n_layers_raw) + delta)
                for delta in _generation_layer_deltas(constraints)
            ]:
                total = estimate_params(
                    pt.d_model, pt.n_heads, pt.d_head, pt.ffn_dim_swiglu,
                    n_layers, n_kv_heads, constraints.vocab_size
                )
                if total < lo or total > hi:
                    continue
                if min(constraints.pp_options) > 1 and n_layers % min(constraints.pp_options) != 0:
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

                        # Placement, RoPE, KV layout, and attention-style CP
                        # are undefined for a pure-state stack. Canonicalize
                        # those axes instead of generating duplicate Mamba
                        # candidates that claim benefits from LongRoPE or
                        # Ulysses despite having zero attention layers.
                        strategy_choices = strategies if n_attn > 0 else ["none"]
                        for strategy in strategy_choices:
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
                            hybrid_ratio = format_state_attention_ratio(
                                actual_n_state, actual_n_attn)

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
                                        state_cp_opts = cp_opts if actual_n_attn > 0 else [1]
                                        state_rope_opts = rope_opts if actual_n_attn > 0 else ["none"]
                                        for cp_d in state_cp_opts:
                                            for rope_m in state_rope_opts:
                                                candidates.append(CandidateArch(
                                                    d_model=pt.d_model,
                                                    n_layers=n_layers,
                                                    n_heads=pt.n_heads,
                                                    d_head=pt.d_head,
                                                    n_kv_heads=(
                                                        n_kv_heads if actual_n_attn > 0
                                                        else pt.n_heads),
                                                    ffn_dim=pt.ffn_dim_swiglu,
                                                    vocab_size=constraints.vocab_size,
                                                    weight_precision=prec["weight_precision"],
                                                    ffn_precision=prec["ffn_precision"],
                                                    attn_precision=dict(prec["attn_precision"]),
                                                    kv_cache_bits=(
                                                        kv_bits if actual_n_attn > 0
                                                        else 16),
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
                                                    tp_degree=int(tp_d),
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

    # Wave 8 PP expansion (Jun 2026): duplicate each candidate per valid
    # pp_degree from constraints.pp_options. Single-pp callers see no
    # change (pp_options=[pp], expansion is a no-op rename). Multi-pp
    # callers get one candidate per (shape × pp_d) combination where
    # n_layers is divisible by pp_d.
    pp_opts_w8 = list(constraints.pp_options or [constraints.pp])
    if len(pp_opts_w8) == 1:
        for _c in candidates:
            _c.pp_degree = int(pp_opts_w8[0])
    else:
        _expanded = []
        for _cand in candidates:
            for _pp_d in pp_opts_w8:
                if _pp_d > 1 and (_cand.n_layers % _pp_d) != 0:
                    continue
                _new = _copy.copy(_cand)
                _new.pp_degree = int(_pp_d)
                _expanded.append(_new)
        candidates = _expanded

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

    hw_prec_configs = get_precision_configs_for_hardware(hw_name)
    prec_configs = [p for p in (constraints.precision_configs or []) if p in hw_prec_configs]
    if not prec_configs:
        prec_configs = hw_prec_configs[:3]

    # MoE-hybrid path: EP=1 excluded (same reasoning as MoE-only path above).
    # Wave 24: EP <= DP cap — see MoE-only path for full rationale.
    ep_opts = _filter_ep_options_by_dp(
        constraints.ep_options or default_ep_options(hw_name, for_moe=True),
        dp=(1 if constraints.training_cluster_gpus is not None
            else constraints.dp),
        source=("user-supplied ep_options"
                if constraints.ep_options is not None
                else "hardware default"),
    )
    strategies = constraints.placement_strategies or ["first_periodic_last", "interleaved", "periodic"]
    mtp_opts, cp_opts, rope_opts, rope_factor = _search_option_lists(constraints)
    # Wave 4: TP search.
    tp_opts = list(constraints.tp_options or [constraints.tp])

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

    candidates: List[CandidateArch] = _BoundedCandidateList(_enumeration_pool_cap(constraints))

    for tp_d in tp_opts:
      lattice = compute_lattice(
          lattice_hw, precision, tp_d,
          d_model_min=1024, d_model_max=_d_model_max_for_target(constraints.target_params_b),
          d_head_options=[64, 128, 256],
      )
      aligned = [pt for pt in lattice if pt.tile_aligned]
      # MoE-hybrid generator: the tightest budget because this generator
      # multiplies both the MoE and state cross-products per lattice point.
      aligned = _filter_lattice_to_budget(
          aligned, constraints.target_params_b,
          budget=_generation_lattice_budget(constraints, "moe_hybrid"), family="moe_hybrid")

      for pt in aligned:
        # GQA sweep
        gqa_ratios = _generation_gqa_ratios(pt, constraints, tp_degree=tp_d)

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
        # Wave 5 fairness redesign: tighter cap for moe_hybrid (40 vs MoE's 80)
        # because moe_hybrid multiplies state inner cross-product.
        moe_opts = _generation_moe_options(
            moe_opts, constraints, "moe_hybrid", pt.ffn_dim_swiglu,
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
                    if min(constraints.pp_options) > 1 and n_layers % min(constraints.pp_options) != 0:
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

                            strategy_choices = strategies if n_attn > 0 else ["none"]
                            for strategy in strategy_choices:
                                attn_indices = place_attention_layers(n_layers, n_attn, strategy)
                                actual_n_attn = len(attn_indices)
                                actual_n_state = n_layers - actual_n_attn

                                attn_set = set(attn_indices)
                                layer_type_list = [
                                    "attention" if i in attn_set else "state"
                                    for i in range(n_layers)
                                ]
                                hybrid_ratio = format_state_attention_ratio(
                                    actual_n_state, actual_n_attn)

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
                                            state_cp_opts = cp_opts if actual_n_attn > 0 else [1]
                                            state_rope_opts = rope_opts if actual_n_attn > 0 else ["none"]
                                            for cp_d in state_cp_opts:
                                                for rope_m in state_rope_opts:
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
                                                        n_kv_heads=(
                                                            n_kv_heads if actual_n_attn > 0
                                                            else pt.n_heads),
                                                        ffn_dim=pt.ffn_dim_swiglu,
                                                        vocab_size=constraints.vocab_size,
                                                        weight_precision=prec["weight_precision"],
                                                        ffn_precision=prec["ffn_precision"],
                                                        attn_precision=dict(prec["attn_precision"]),
                                                        kv_cache_bits=(
                                                            kv_bits if actual_n_attn > 0
                                                            else 16),
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
                                                        tp_degree=int(tp_d),
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
                                                    if (actual_n_attn > 0
                                                            and getattr(constraints, "allow_mla", False)):
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
                                                                    tp_degree=int(tp_d),
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

    # Wave 8 PP expansion (Jun 2026): duplicate each candidate per valid
    # pp_degree from constraints.pp_options. Single-pp callers see no
    # change (pp_options=[pp], expansion is a no-op rename). Multi-pp
    # callers get one candidate per (shape × pp_d) combination where
    # n_layers is divisible by pp_d.
    pp_opts_w8 = list(constraints.pp_options or [constraints.pp])
    if len(pp_opts_w8) == 1:
        for _c in candidates:
            _c.pp_degree = int(pp_opts_w8[0])
    else:
        _expanded = []
        for _cand in candidates:
            for _pp_d in pp_opts_w8:
                if _pp_d > 1 and (_cand.n_layers % _pp_d) != 0:
                    continue
                _new = _copy.copy(_cand)
                _new.pp_degree = int(_pp_d)
                _expanded.append(_new)
        candidates = _expanded

    return candidates


def _estimate_n_active_params(cand: "CandidateArch") -> int:
    """Wave 8b (Jun 2026): cheap O(1) active-param estimate for cheap-rank.

    Uses the standard transformer formula (embed + attn-QKVO + SwiGLU).
    For MoE candidates, only the *active* experts (top_k) contribute, so
    the proxy is conservative — fine for ranking, not for reporting.

    NOT the canonical n_active_params (that's in schema.ArchConfig). This
    is a lightweight proxy for sorting in `_cheap_quality_rank`; the
    Pareto-stage full evaluate gets the real number.
    """
    d = max(1, int(getattr(cand, "d_model", 0)))
    L = max(1, int(getattr(cand, "n_layers", 0)))
    n_h = max(1, int(getattr(cand, "n_heads", 0)))
    n_kv = max(1, int(getattr(cand, "n_kv_heads", n_h) or n_h))
    d_h = max(1, int(getattr(cand, "d_head", 0)))
    ffn = max(1, int(getattr(cand, "ffn_dim", 0)))
    vocab = max(1, int(getattr(cand, "vocab_size", 0)))
    # Attention QKVO: Q is full (d × n_h × d_h), K/V follow GQA (d × n_kv × d_h
    # each), O projects back (n_h × d_h × d).
    attn = d * (n_h + 2 * n_kv) * d_h + (n_h * d_h * d)
    # SwiGLU FFN: 3 × (d × ffn).
    ffn_params = 3 * d * ffn
    # MoE: scale ffn by top_k (active experts), not n_experts.
    moe = getattr(cand, "moe", None)
    if moe and isinstance(moe, dict):
        top_k = int(moe.get("top_k", 1) or 1)
        # Active FFN params = top_k × per-expert. Per-expert ≈ ffn_params /
        # n_experts in a typical fine-grained MoE. We approximate active-FFN
        # as top_k × ffn_params (i.e. each expert ≈ baseline FFN size for
        # ranking purposes — exact MoE math is in n_active_params).
        # For the cheap rank we keep this simple.
        ffn_active = top_k * ffn_params
    else:
        ffn_active = ffn_params
    embed = vocab * d
    return int(embed + L * (attn + ffn_active))


def _cheap_quality_rank(
    cand: "CandidateArch",
    training_tokens: int,
    quality_model_version: str = "effective_capacity_v2",
) -> float:
    """Wave 8b (Jun 2026, fixed Jun 2026 post-probe): O(microseconds) quality
    proxy for two-stage eval.

    Returns L_base × (1 + shape_penalty + family_bonus). Lower is better.
    Used to prune the candidate set to `constraints.max_full_evaluations`
    before paying the full `evaluate_candidate` cost.

    POST-PROBE FIX: the original version (chinchilla_baseline × shape_penalty
    only) was family-blind. MoE candidates carry an essentially negative
    `capacity_bonus = -0.05 × log(N_total/N_active) × moe_layer_fraction`
    in the FULL evaluator that the cheap rank didn't account for. Result:
    MoE candidates were systematically pruned away at moderate scales
    (e.g. 120B target → 13500 MoE enumerated, 0 in top-40 cheap-ranked,
    every cell falsely showed dense=moe in the displayed matrix). The fix
    is to add a family-conditional adjustment mirroring the dominant
    residual term — kept extremely cheap (one log, one multiply).

    State / hybrid candidates also get a small approximation of the
    state-capacity vs full-attention KV-cost gap; the actual state
    residual depends on context-length-aware recall band-pass, which we
    skip here (the full evaluator catches it).
    """
    n_active = _estimate_n_active_params(cand)
    n_spine = n_active
    moe = getattr(cand, "moe", None)
    if quality_model_version == "effective_capacity_v2" \
            and moe and isinstance(moe, dict):
        n_total = max(
            n_active,
            int(float(getattr(cand, "total_params_b", 0.0) or 0.0) * 1e9),
        )
        ratio = max(1.0, n_total / max(1, n_active))
        tokens_per_total = int(training_tokens) / max(1, n_total)
        utilization = tokens_per_total / (tokens_per_total + 20.0)
        n_spine = int(n_active * (ratio ** (0.25 * utilization)))
    L_base = chinchilla_loss(n_spine, int(training_tokens))
    sp = shape_penalty(
        int(cand.d_model), int(cand.n_layers), n_active,
    )
    # Family-conditional capacity bonus (mirrors quality_model._moe_residual).
    # MoE: capacity = -0.05 × log_sat(N_total/N_active) × moe_layer_fraction
    # Saturation cap = 4.0 matches `capacity_ratio_cap` in defaults.
    family_bonus = 0.0
    if quality_model_version == "legacy_residual_v1" \
            and moe and isinstance(moe, dict):
        import math as _math
        n_experts = max(1, int(moe.get("n_experts", 1)))
        top_k = max(1, int(moe.get("top_k", 1)))
        # Ratio approximation: n_total/n_active ≈ n_experts/top_k for the FFN
        # portion. (Attention is shared across experts so the actual ratio
        # is slightly lower than this; close enough for ranking.)
        raw_ratio = n_experts / top_k
        cap_ratio = 4.0
        if raw_ratio <= cap_ratio:
            eff_log = _math.log(max(1.0, raw_ratio))
        else:
            eff_log = _math.log(cap_ratio) + 0.5 * _math.log(raw_ratio / cap_ratio)
        n_dense_prefix = max(0, int(getattr(cand, "n_dense_ffn_layers", 0) or 0))
        n_layers = max(1, int(cand.n_layers))
        moe_layer_fraction = max(0.0, (n_layers - n_dense_prefix) / n_layers)
        family_bonus += -0.05 * eff_log * moe_layer_fraction
    # State/hybrid: small fixed bonus reflecting the typical mid-context-length
    # KV-cost savings. Real state_residual is context-aware; this is a
    # plausibility offset so hybrid candidates aren't pruned out before the
    # full evaluator gets to weigh them. Tuned to ~-0.01 (1% loss-equivalent
    # at typical operating points; less than MoE's ~6% bonus by intent —
    # state mechanisms have higher uncertainty in cheap-rank terms).
    state_cfg = getattr(cand, "state_config", None)
    if state_cfg and quality_model_version == "legacy_residual_v1":
        family_bonus += -0.01
    return L_base * (1.0 + max(-0.5, sp + family_bonus))


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_candidate(
    cand: CandidateArch,
    hw_name: str,
    constraints: DeploymentConstraints,
) -> EvaluatedCandidate:
    """Evaluate a single candidate with throughput and quality models."""

    # Canonicalize public parameter counts before any objective/report reads
    # them. Generators use fast family-specific estimates for pruning, but
    # state/MLA/MoE compositions can differ by several percent from those
    # approximations. The shared ledger is already the source of truth for
    # quality and throughput views; CandidateArch must expose the same totals.
    ledger = parameter_ledger(cand)
    cand.total_params = int(ledger.total_params)
    cand.active_params = int(ledger.active_params)
    cand.total_params_b = round(cand.total_params / 1e9, 6)
    cand.active_params_b = round(cand.active_params / 1e9, 6)
    effective_dp = _effective_candidate_dp(cand, constraints)
    cand.dp_degree = effective_dp

    # --- Throughput ---
    prefill_len = max(1, int(constraints.prompt_len or constraints.context_length))
    # Scheduler-aware effective decode batch. AC models service time rather
    # than queueing delay; concurrency can only fill up to serving_batch.
    effective_serving_batch = max(
        1, min(int(constraints.serving_batch), int(constraints.concurrency))
    )
    tput_arch = TputArch(
        d_model=cand.d_model,
        n_layers=cand.n_layers,
        n_heads=cand.n_heads,
        d_head=cand.d_head,
        n_kv_heads=cand.n_kv_heads,
        ffn_dim=cand.ffn_dim,
        vocab_size=cand.vocab_size,
        batch_size=effective_serving_batch,
        seq_len=max(1, int(constraints.context_length)),
        precision=cand.ffn_precision,  # dominant precision for throughput
        weight_precision=cand.weight_precision,
        activation_precision=cand.activation_precision,
        attn_precision=_copy.deepcopy(cand.attn_precision),
        kv_precision=kv_bits_to_precision(cand.kv_cache_bits),
        # v1: MoE branch fires when moe_config is set on the throughput's ArchConfig
        moe_config=cand.moe,
        # v1-fix Part B: first-K-dense prefix
        n_dense_ffn_layers=cand.n_dense_ffn_layers,
        dp_degree=effective_dp,
        training_micro_batch=int(
            constraints.training_micro_batch
            if constraints.training_micro_batch is not None else 8
        ),
        pipeline_microbatches=int(constraints.pipeline_microbatches),
        serving_scheduler=str(constraints.scheduler),
        serving_concurrency=int(constraints.concurrency),
        serving_output_len=int(constraints.output_len),
        prefill_chunk_size=65536,
    )
    # Wave 10A (Jun 2026): pin training micro-batch on tput_arch so the
    # throughput model's `train_arch = replace(arch, batch_size=max(8,
    # arch.training_micro_batch))` picks up the caller's explicit value.
    # When constraints.training_micro_batch is None, the existing default
    # (8) applies — preserving v1 back-compat.
    _train_mb = getattr(constraints, "training_micro_batch", None)
    if _train_mb is not None and int(_train_mb) > 0:
        tput_arch.training_micro_batch = int(_train_mb)

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
    n_local_attn = int(getattr(cand, "n_local_attn_layers", 0) or 0)
    if n_local_attn > 0 and swa_window > 0:
        # Wave 18g: local:global interleave. The throughput model builds a
        # per-layer type list from n_local_attn_layers and prices local
        # layers at min(len, window) for decode KV and S x W for prefill.
        # attention_type stays that of the GLOBAL layers (full/GQA or MLA);
        # do NOT cap decode length globally — only the local layers cap.
        tput_arch.local_window = swa_window
        tput_arch.n_local_attn_layers = n_local_attn
        # Materialise one combined layout. Re-running __post_init__ here used
        # to rebuild a local/global list from scratch and silently erase state
        # layers on composed local+state candidates.
        tput_arch.layer_type_list = compose_layer_type_list(
            cand.layer_type_list, cand.n_layers, n_local_attn
        )
    elif swa_window > 0:
        tput_arch.local_window = swa_window
        tput_arch.attention_type = "swa"
        # Wave 40: whole-model SWA is priced through the same per-layer
        # local-attention path the Wave-18g interleave uses (prefill
        # attention S x min(S, W), windowed KV) instead of the legacy
        # O(N^2) prefill upper bound. Before this, the SAME physical model
        # cost 15x more TTFT expressed as attention_type="swa" than as a
        # 32/32 local interleave (64.4s vs 4.2s at 512k/window 4096) —
        # the optimizer compared the two encodings on inconsistent bases.
        # compose_layer_type_list preserves state layers and clamps the
        # local count to the available attention slots.
        tput_arch.n_local_attn_layers = cand.n_layers
        tput_arch.layer_type_list = compose_layer_type_list(
            cand.layer_type_list, cand.n_layers, cand.n_layers
        )

    # Wave 9 (Jun 2026): compressed-attention variants. Wire each variant's
    # config to the throughput model so the kv_bytes_per_token_per_layer
    # helper picks up the correct per-token KV cost at decode.
    if cand.attention_type == "csa" and cand.csa_top_k_blocks > 0:
        tput_arch.attention_type = "csa"
        tput_arch.csa_block_size = int(cand.csa_block_size)
        tput_arch.csa_top_k_blocks = int(cand.csa_top_k_blocks)
        tput_arch.csa_compression_dim = int(cand.csa_compression_dim
                                            or cand.d_head)
    elif cand.attention_type == "indexshare" and cand.indexshare_top_k_buckets > 0:
        tput_arch.attention_type = "indexshare"
        tput_arch.indexshare_num_buckets = int(cand.indexshare_num_buckets)
        tput_arch.indexshare_top_k_buckets = int(cand.indexshare_top_k_buckets)
        tput_arch.indexshare_index_dim = int(cand.indexshare_index_dim or 64)
    elif cand.attention_type == "msa":
        tput_arch.attention_type = "msa"
        tput_arch.msa_window_size = int(cand.msa_window_size or 512)
        tput_arch.msa_dilated_top_k = int(cand.msa_dilated_top_k or 64)
        tput_arch.msa_global_top_k = int(cand.msa_global_top_k or 16)
    elif cand.attention_type == "nsa":
        tput_arch.attention_type = "nsa"
        tput_arch.nsa_compress_block_size = int(
            cand.nsa_compress_block_size or 64
        )
        tput_arch.nsa_compress_block_stride = int(
            cand.nsa_compress_block_stride or 16
        )
        tput_arch.nsa_select_block_size = int(
            cand.nsa_select_block_size or 64
        )
        tput_arch.nsa_select_top_k = int(cand.nsa_select_top_k or 16)
        tput_arch.nsa_window_size = int(cand.nsa_window_size or 512)
    if cand.yoco_n_self_attn_layers > 0:
        tput_arch.yoco_n_self_attn_layers = int(
            cand.yoco_n_self_attn_layers
        )

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
    # cost. Wave 40: whole-model SWA prefill is no longer the legacy
    # O(N^2) upper bound — it routes through the per-layer local-attention
    # path (S x min(S, W)) via the all-local layer_type_list composed
    # above, identical to the Wave-18g interleave pricing.
    effective_decode_len = prefill_len
    # Wave 18g: only whole-model SWA caps the decode length globally. In a
    # local:global interleave the global layers still read the full KV; the
    # local layers' window cap is applied per-layer inside the throughput
    # model's heterogeneous path.
    if swa_window > 0 and n_local_attn == 0:
        effective_decode_len = min(effective_decode_len, swa_window)

    # Wave 4: TP is now a per-candidate field. Fall back to constraints.tp
    # when the candidate does not carry an explicit TP (tp_degree=0 sentinel;
    # older CandidateArch construction paths / test fixtures) so scalar-tp
    # callers see the legacy behavior. Bug fix (Jul 2026): the sentinel is 0,
    # not 1 — an explicit TP=1 candidate must stay TP=1, while an unset field
    # must inherit the deployment's TP so the throughput model prices TTFT,
    # TP all-reduce (incl. cross-node) in TBT, and per-GPU memory against the
    # actual parallelism instead of TP=1.
    _cand_tp = int(getattr(cand, "tp_degree", 0) or 0)
    if _cand_tp <= 0:
        _cand_tp = int(constraints.tp)
    # PP: same sentinel semantics. A searched candidate carries its explicit
    # pp_degree (generators always stamp it, even when it is 1, so a searched
    # PP=1 candidate is never re-attached to constraints.pp); an unset
    # pp_degree (0) — e.g. a baseline config that omits pipeline_parallel —
    # falls back to the deployment constraints.
    _cand_pp = int(getattr(cand, "pp_degree", 0) or 0)
    if _cand_pp <= 0:
        _cand_pp = max(1, int(getattr(constraints, "pp", 1) or 1))
    tput = throughput_fn(
        tput_arch, hw_name,
        tp_degree=_cand_tp,
        pp_degree=_cand_pp,
        microbatches=int(constraints.pipeline_microbatches),
        dp_degree=effective_dp,
        decode_kv_len=effective_decode_len,
        prefill_seq_len=prefill_len,
        training_seq_len=max(
            1, int(constraints.pretraining_context_length)),
        ep_degree=cand.ep_degree,
        ep_topology=constraints.ep_topology,
    )

    # --- Quality ---
    # Map precision configs to quality model format
    component_precs = {}
    if cand.ffn_precision != cand.weight_precision:
        for comp in ("ffn_up", "ffn_down", "ffn_gate"):
            component_precs[comp] = cand.ffn_precision
    # Wave 40: each attn_precision key is read independently. Previously
    # "output" was consumed only when "v" was quantized, and "qk" was never
    # consumed at all — fp8 qk-logits or a lone fp8 output projection were
    # silent quality no-ops (same class as the Wave-38 sparsity_2_4 bug).
    # Both key schemas are accepted ({qk,v,output} canonical; {q,k,v,o} as
    # used by docs/invariant-probing-playbook.md) instead of silently
    # dropping unknown keys.
    _ap = cand.attn_precision or {}
    _attn_v = _ap.get("v", "bf16")
    _attn_out = _ap.get("output", _ap.get("o", "bf16"))
    # qk logits are quantized if ANY of qk/q/k is quantized ("bf16" is
    # truthy, so an `or` chain would let {"q": "bf16", "k": "fp8"} slip
    # through as bf16).
    _attn_qk = next((v for v in (_ap.get("qk"), _ap.get("q"), _ap.get("k"))
                     if v and v != "bf16"), "bf16")
    if _attn_v != "bf16":
        component_precs["qkv_proj"] = _attn_v
    if _attn_out != "bf16":
        component_precs["output_proj"] = _attn_out
    if _attn_qk != "bf16":
        component_precs["qk_logits"] = _attn_qk

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
            # Keep the state projection shape in the quality view. The
            # canonical parameter ledger reads these fields; omitting them
            # silently substituted d_head=64 and made otherwise-valid
            # hybrids fail the architecture-view invariant whenever their
            # state head width differed from that default.
            "state_expansion": cand.state_config.get("state_expansion", 2),
            "n_heads": cand.state_config.get("n_heads", cand.n_heads),
            "d_head": cand.state_config.get("d_head", cand.d_head),
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
        activation_precision=cand.activation_precision,
        component_precisions=component_precs if component_precs else None,
        # v1: MoE fires the quality model's _moe_residual hook.
        moe_config=cand.moe,
        model_type=model_type,
        state_config=qual_state_config,
        # v1-fix Part B: first-K-dense prefix
        n_dense_ffn_layers=cand.n_dense_ffn_layers,
        # v1-fix MLA + Wave 9 compressed-attention: thread attention type
        # and per-variant config to the quality model so the right residual
        # subterm fires. Wave 9 adds csa / indexshare / msa; each carries
        # its own config dict.
        attention_type=(
            "mla" if cand.attention_type == "mla"
            else "nsa" if cand.attention_type == "nsa"
            else "csa" if cand.attention_type == "csa"
            else "indexshare" if cand.attention_type == "indexshare"
            else "msa" if cand.attention_type == "msa"
            # Wave 18g: a local:global interleave keeps the GLOBAL layers'
            # projection type; only whole-model SWA (n_local == 0) reports
            # as "swa" to the quality model.
            else ("swa" if (cand.attention_type == "swa"
                            or (swa_window > 0 and n_local_attn == 0)) else "gqa")
        ),
        mla_latent_dim=(cand.mla_kv_latent_dim if cand.attention_type == "mla" else None),
        mla_q_latent_dim=(cand.mla_q_latent_dim if cand.attention_type == "mla" else None),
        mla_rope_head_dim=(cand.mla_rope_head_dim if cand.attention_type == "mla" else None),
        mla_nope_head_dim=(cand.mla_nope_head_dim if cand.attention_type == "mla" else None),
        nsa_compress_block_size=(cand.nsa_compress_block_size if cand.attention_type == "nsa" else None),
        nsa_compress_block_stride=(cand.nsa_compress_block_stride if cand.attention_type == "nsa" else None),
        nsa_select_block_size=(cand.nsa_select_block_size if cand.attention_type == "nsa" else None),
        nsa_select_top_k=(cand.nsa_select_top_k if cand.attention_type == "nsa" else None),
        nsa_window_size=(cand.nsa_window_size if cand.attention_type == "nsa" else None),
        # Wave 9 compressed-attention configs (only one is populated at a time).
        csa_block_size=(cand.csa_block_size if cand.attention_type == "csa" else None),
        csa_top_k_blocks=(cand.csa_top_k_blocks if cand.attention_type == "csa" else None),
        csa_compression_dim=(cand.csa_compression_dim if cand.attention_type == "csa" else None),
        indexshare_num_buckets=(cand.indexshare_num_buckets if cand.attention_type == "indexshare" else None),
        indexshare_top_k_buckets=(cand.indexshare_top_k_buckets if cand.attention_type == "indexshare" else None),
        indexshare_index_dim=(cand.indexshare_index_dim if cand.attention_type == "indexshare" else None),
        msa_window_size=(cand.msa_window_size if cand.attention_type == "msa" else None),
        msa_dilated_top_k=(cand.msa_dilated_top_k if cand.attention_type == "msa" else None),
        msa_global_top_k=(cand.msa_global_top_k if cand.attention_type == "msa" else None),
        # v1-fix SWA: thread the window so the quality model adds the small
        # SWA-locality residual when workload_context exceeds the window.
        local_window=(swa_window if swa_window > 0 else None),
        # Wave 18g: fraction of attention layers that are local. The quality
        # model gates the locality penalty on global-layer presence and only
        # caps effective attention context for whole-model SWA.
        local_attention_fraction=(
            (n_local_attn / max(1, cand.n_layers - int(getattr(cand, "n_state_layers", 0) or 0)))
            if (n_local_attn > 0 and swa_window > 0) else None
        ),
        # v1-fix MTP: quality model's mtp_residual bonus
        mtp_n_predict_depths=int(cand.mtp_n_predict_depths),
        mtp_depth_n_layers=int(cand.mtp_depth_n_layers),
        mtp_train_loss_weight=float(cand.mtp_train_loss_weight),
        # v1-fix RoPE scaling
        rope_scaling_method=str(cand.rope_scaling_method),
        rope_scaling_factor=float(cand.rope_scaling_factor),
        rope_original_max_position=int(cand.rope_original_max_position),
        yoco_n_self_attn_layers=int(cand.yoco_n_self_attn_layers),
        # Wave 38: sparsity_2_4 was carried on CandidateArch and read
        # inside _architecture_residual (line ~2127) but never threaded
        # into the QualArch view — so the FFN/attn 2:4 sparsity quality
        # penalty (0.015 x sparse_param_share, ~1% loss for full-FFN
        # 2:4) was a silent no-op. The throughput model's 2x speedup
        # WAS wired via ArchConfig, so recommending 2:4 sparsity gave
        # a "free" 2x TPS with zero quality cost. Property test: see
        # docs/invariant-probing-playbook.md #E.
        sparsity_2_4=getattr(cand, "sparsity_2_4", None),
    )
    validate_architecture_views(cand, tput_arch, qual_arch)

    # v1-fix Wave 3 follow-up (Jun 2026): the `memory_fits` flag was the
    # back-door HBM hard cap. When False, `feasibility_penalty` in
    # penalties.py injected INFEASIBLE=1e6 into the quality loss, which
    # then tripped the >1e4 cull in _check_feasibility — effectively
    # restoring the hard memory cap that Wave 2a Step 2a.3 tried to
    # remove. Now that Wave 2a Step 2a.2's HBM-spill computes a
    # continuous TBT penalty for moderate memory overflow, only EXTREME
    # overflow (>50× HBM) should be treated as truly infeasible — that
    # bound catches genuinely-broken configs (e.g. MoE without EP, where
    # all experts try to land on one GPU) without restoring the cap on
    # moderate spill candidates we want to surface with a TBT penalty.
    _hbm_gb = _get_hbm_gb(hw_name)
    _mem_gb = tput.memory_footprint_per_gpu_gb
    # 10× threshold: NVLink intra-node pool can absorb ~7× HBM (gpus_per_node-1
    # peers). Past ~10× we exhaust intra-node and hit PCIe/IB, which is
    # 50-100× slower than HBM — at that point the spill multiplier becomes
    # so large the candidate is effectively infeasible anyway, so flagging
    # it via the quality-side INFEASIBLE marker still catches the right
    # set of broken configs (MoE without EP, 1000B on single-GPU TP, etc.)
    # without restoring the cap on candidates that genuinely benefit from
    # the moderate-spill TBT penalty.
    _extreme_overflow = _mem_gb > 10.0 * _hbm_gb
    qual = quality_fn(
        qual_arch,
        TrainingConfig(
            training_tokens=constraints.training_tokens,
            unique_tokens=constraints.unique_training_tokens,
            sequence_length=max(
                1, int(constraints.pretraining_context_length)
            ),
            hardware=hw_name,
            kv_quantization_bits=cand.kv_cache_bits,
            quality_model_version=constraints.quality_model_version,
        ),
        memory_fits=(not _extreme_overflow),
        lattice_aligned=True,
        workload_spec={
            "context_length": max(1, int(constraints.context_length)),
            "task_type": "general",
            "traffic_mix": constraints.traffic_mix or {},
        },
    )

    # --- Constraint checking ---
    # v1-fix Wave 2a Step 2a.3 (Jun 2026): the hard TBT/TTFT/HBM cutoffs
    # are gone. Serving cost is now continuous — TBT, TTFT, and the
    # HBM-spill penalty are part of the Pareto objective so the optimizer
    # can trade quality against serving cost instead of pruning. The
    # remaining hard violations are only physics-level: precision not
    # supported on this hardware. Memory overflow becomes a TBT penalty
    # via throughput_model.py's hbm_spill / spill_tier fields.
    violations = []
    meets = True

    # A minimum training-cluster target and a maximum are deliberately
    # separate constraints. The former derives candidate-specific DP and may
    # round upward for a legal EP overlay; the latter is a hard physical cap.
    # Keeping this guard in the shared evaluator prevents generator, Python,
    # and future CLI entry points from disagreeing about the same topology.
    _training_replica_gpus = _candidate_training_replica_gpus(cand, constraints)
    _training_world_gpus = _training_replica_gpus * effective_dp
    _max_training_world = getattr(
        constraints, "max_training_cluster_gpus", None)
    if (_max_training_world is not None
            and _training_world_gpus > int(_max_training_world)):
        violations.append(
            f"Training world {_training_world_gpus} GPUs > "
            f"hard cap {int(_max_training_world)} GPUs"
        )
        meets = False

    # Keep TBT/TTFT budgets as *soft warnings* if the user explicitly set
    # them (so opting back into the v0 behavior is possible from Python),
    # but don't fail feasibility on them. Default is None so the grid
    # driver no longer touches these.
    if constraints.serving_tbt_ms is not None and tput.decode_time_per_token_ms > constraints.serving_tbt_ms:
        violations.append(f"TBT {tput.decode_time_per_token_ms:.1f}ms > {constraints.serving_tbt_ms}ms soft budget (continuous: not a feasibility cut)")
    if constraints.serving_ttft_ms is not None and tput.prefill_time_ms > constraints.serving_ttft_ms:
        violations.append(f"TTFT {tput.prefill_time_ms:.1f}ms > {constraints.serving_ttft_ms}ms soft budget (continuous: not a feasibility cut)")
    # Memory: noted as warning when spilling, but no longer infeasible.
    if getattr(tput, "spill_tier", "fits") != "fits":
        violations.append(
            f"HBM spill {tput.hbm_spill_gb:.1f}GB via {tput.spill_tier} (TBT penalty {tput.decode_time_per_token_ms / max(0.001, tput.tbt_ms_no_spill):.2f}×)")

    # Infeasible quality (precision not supported on this hardware) — the
    # only remaining hard cut.
    # v1-fix Wave 3 (Jun 2026): when constraints.allow_quality_sentinel is
    # set (driven by the AC_QUALITY_SENTINEL_SOFT env var), demote the
    # sentinel-loss cull to a warning so the optimizer surfaces the
    # *least-bad* candidate instead of declaring the cell empty. Used by
    # the H100/cells matrix to populate quality-outside-coverage corners
    # (1B@2M, 750B@2M, 1000B@128k+ etc.) with a "best effort, model
    # uncovered" answer rather than a dash.
    import os as _os
    _soft_sentinel = (
        getattr(constraints, "allow_quality_sentinel", False)
        or _os.environ.get("AC_QUALITY_SENTINEL_SOFT") == "1"
    )
    if qual.predicted_loss > 1e4:
        if _soft_sentinel:
            violations.append("Quality model reports infeasible (precision not supported) — surfaced as soft warning under allow_quality_sentinel")
        else:
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

    # Wave 13 (Jun 2026): assemble structured Feasibility report from
    # the same signals that drove `meets` + `violations` above. The named
    # guards make it easy for downstream code (Pareto, picker, justify)
    # to ask "which specific guard failed" rather than parsing strings.
    _feas_guards = {
        "memory_extreme_overflow": GuardResult(
            name="memory_extreme_overflow",
            triggered=_extreme_overflow,
            fails_feasibility=False,  # routed via quality sentinel below
            is_warning=_extreme_overflow,
            message=(f"Memory {tput.memory_footprint_per_gpu_gb:.1f}GB > "
                     f"10x HBM ({_hbm_gb}GB); routes to quality sentinel."
                     if _extreme_overflow else ""),
            metric_value=tput.memory_footprint_per_gpu_gb,
            threshold=10.0 * _hbm_gb,
        ),
        "quality_sentinel_tripped": GuardResult(
            name="quality_sentinel_tripped",
            triggered=(qual.predicted_loss > 1e4),
            fails_feasibility=(qual.predicted_loss > 1e4) and not _soft_sentinel,
            is_warning=(qual.predicted_loss > 1e4) and _soft_sentinel,
            message=("Quality model returned INFEASIBLE sentinel (precision "
                     "not supported, or extreme memory overflow routed here)"
                     if qual.predicted_loss > 1e4 else ""),
            metric_value=qual.predicted_loss,
            threshold=1e4,
        ),
        "training_cluster_cap": GuardResult(
            name="training_cluster_cap",
            triggered=(
                _max_training_world is not None
                and _training_world_gpus > int(_max_training_world)
            ),
            fails_feasibility=(
                _max_training_world is not None
                and _training_world_gpus > int(_max_training_world)
            ),
            is_warning=False,
            message=(
                f"Training world {_training_world_gpus} GPUs exceeds hard "
                f"cap {int(_max_training_world)} GPUs"
                if (_max_training_world is not None
                    and _training_world_gpus > int(_max_training_world))
                else ""
            ),
            metric_value=float(_training_world_gpus),
            threshold=(
                float(_max_training_world)
                if _max_training_world is not None else None
            ),
        ),
        "tbt_budget_warning": GuardResult(
            name="tbt_budget_warning",
            triggered=(constraints.serving_tbt_ms is not None
                       and tput.decode_time_per_token_ms > constraints.serving_tbt_ms),
            fails_feasibility=False,
            is_warning=(constraints.serving_tbt_ms is not None
                        and tput.decode_time_per_token_ms > constraints.serving_tbt_ms),
            message=(f"TBT {tput.decode_time_per_token_ms:.1f}ms > "
                     f"{constraints.serving_tbt_ms}ms (soft)"
                     if (constraints.serving_tbt_ms is not None
                         and tput.decode_time_per_token_ms > constraints.serving_tbt_ms)
                     else ""),
            metric_value=tput.decode_time_per_token_ms,
            threshold=constraints.serving_tbt_ms,
        ),
        "ttft_budget_warning": GuardResult(
            name="ttft_budget_warning",
            triggered=(constraints.serving_ttft_ms is not None
                       and tput.prefill_time_ms > constraints.serving_ttft_ms),
            fails_feasibility=False,
            is_warning=(constraints.serving_ttft_ms is not None
                        and tput.prefill_time_ms > constraints.serving_ttft_ms),
            message=(f"TTFT {tput.prefill_time_ms:.1f}ms > "
                     f"{constraints.serving_ttft_ms}ms (soft)"
                     if (constraints.serving_ttft_ms is not None
                         and tput.prefill_time_ms > constraints.serving_ttft_ms)
                     else ""),
            metric_value=tput.prefill_time_ms,
            threshold=constraints.serving_ttft_ms,
        ),
        "hbm_spill_warning": GuardResult(
            name="hbm_spill_warning",
            triggered=(getattr(tput, "spill_tier", "fits") != "fits"),
            fails_feasibility=False,
            is_warning=(getattr(tput, "spill_tier", "fits") != "fits"),
            message=(f"HBM spill {getattr(tput, 'hbm_spill_gb', 0.0):.1f}GB via "
                     f"{getattr(tput, 'spill_tier', '?')}"
                     if (getattr(tput, "spill_tier", "fits") != "fits") else ""),
            metric_value=getattr(tput, "hbm_spill_gb", 0.0),
            threshold=None,
        ),
    }
    _feas = Feasibility(is_feasible=meets, guards=_feas_guards)

    return EvaluatedCandidate(
        arch=cand,
        quality=qual,
        throughput=tput,
        predicted_loss=qual.predicted_loss,
        training_tps=tput.training_throughput_tokens_per_sec,
        serving_tbt_ms=tput.decode_time_per_token_ms,
        serving_request_latency_ms=tput.serving_request_latency_ms,
        memory_per_gpu_gb=tput.memory_footprint_per_gpu_gb,
        binding_serving_regime=regime,
        binding_reason=reason,
        meets_constraints=meets,
        constraint_violations=violations,
        feasibility=_feas,
    )


def _get_hbm_gb(hw_name: str) -> float:
    """Get HBM capacity for a hardware target."""
    hbm_map = {
        "h100": 80,
        "h800": 80,             # Gate-2 Task C: H100 silicon, same 80 GB HBM3
        "b200": 192,
        "gb200_nvl72": 192,     # Gate-2 Task C: B200 silicon, 192 GB HBM3e
        "tpu_v5e": 16,
        "tpu_v5p": 95,
        "trainium2": 96,
        "trn2": 96,
        "trainium3": 192,
        "trn3": 192,
    }
    return hbm_map.get(hw_name, 80)


def _prefer_training_fit(
    candidates: List[EvaluatedCandidate], hw_name: str,
) -> List[EvaluatedCandidate]:
    """Prefer physically trainable plans, falling back only if none fit.

    Serving spill is explicitly modeled, but AC has no training-spill or
    optimizer-offload model. A candidate whose training memory exceeds HBM
    therefore cannot honestly beat a fitting candidate on a tiny loss prior.
    Keep best-effort behavior for genuinely impossible cells by returning the
    original pool when every candidate overflows.
    """
    hbm = float(_get_hbm_gb(hw_name))
    fitting = [
        ev for ev in candidates
        if float(getattr(
            ev.throughput, "training_memory_per_gpu_gb", 0.0) or 0.0)
        <= hbm + 1e-9
    ]
    return fitting or candidates


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
    # Wave 14 (Jun 2026): training_memory_per_gpu_gb added as 6th axis
    # between memory_per_gpu_gb (inference) and total_params. Two candidates
    # with identical inference cost but different training cost — the
    # cheaper-to-train one Pareto-dominates. Surfaces "fits at TP=8 inference
    # but needs 240 GB during training" architectures for objective profiles
    # that care about training cost. Safe default: training_mem comes from
    # ThroughputResult.training_memory_per_gpu_gb (FSDP/ZeRO-3-sharded);
    # for candidates that don't compute it (e.g. older fixtures), 0.0 is
    # neutral on the axis.
    train_mem_a = float(getattr(a.throughput, "training_memory_per_gpu_gb", 0.0) or 0.0)
    train_mem_b = float(getattr(b.throughput, "training_memory_per_gpu_gb", 0.0) or 0.0)
    objs_a = (a.predicted_loss, -_evaluated_training_tps_per_gpu(a),
              a.serving_tbt_ms,
              a.throughput.prefill_time_ms, a.memory_per_gpu_gb,
              train_mem_a, a.arch.total_params,
              _evaluated_serving_instance_gpus(a))
    objs_b = (b.predicted_loss, -_evaluated_training_tps_per_gpu(b),
              b.serving_tbt_ms,
              b.throughput.prefill_time_ms, b.memory_per_gpu_gb,
              train_mem_b, b.arch.total_params,
              _evaluated_serving_instance_gpus(b))

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

    # TP divisibility: check if better architectures were eliminated by TP.
    # Wave 4: read the *picked* TP from the optimum (it's a candidate field
    # now), not constraints.tp (which may be 1 even when the optimizer
    # actually selected a larger TP from constraints.tp_options).
    _opt_tp = int(getattr(optimal.arch, "tp_degree", 0) or 0) or int(constraints.tp)
    if _opt_tp > 1:
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


def _expand_compressed_variants(
    candidates: List["CandidateArch"],
    constraints: "DeploymentConstraints",
    max_expansion_per_type: int = 25_000,
) -> List["CandidateArch"]:
    """Wave 33 (Jul 2026): compressed-attention x MoE / state composition.

    The Wave 9 CSA/IndexShare/MSA emissions live only in the dense
    generator, so an allow_csa search could never produce a
    "sparse-attention MoE" or a "sparse-global hybrid" — the exact stacks
    that compete with MLA+MoE+state at long context. The evaluator
    already scores the combinations correctly (throughput branches per
    attention_type independently of FFN/state; the quality model fires
    the compression residual alongside moe/state residuals), so the gap
    was pure enumeration.

    This expander runs at the family-combine point: for every
    full-attention candidate that carries MoE and/or state layers, emit
    one copy per allowed compressed family at that family's DEFAULT
    config (first option of each list). One config per type keeps the
    expansion at <=3x the eligible subset; an even-stride subsample caps
    the eligible base at `max_expansion_per_type` so enumeration memory
    stays bounded. Compressed attention leaves the parameter ledger
    unchanged (same convention as the Wave 9 dense emissions), so
    dataclasses.replace preserves ledger consistency.
    """
    want_csa = bool(getattr(constraints, "allow_csa", False))
    want_idx = bool(getattr(constraints, "allow_indexshare", False))
    want_msa = bool(getattr(constraints, "allow_msa", False))
    if not (want_csa or want_idx or want_msa):
        return candidates

    eligible = [
        c for c in candidates
        if (getattr(c, "attention_type", "full") or "full") == "full"
        and (getattr(c, "moe", None) or getattr(c, "state_config", None))
        # Local:global interleave candidates already mix attention spans;
        # stacking a second sparse pattern on their global layers is not
        # a modeled composition.
        and not int(getattr(c, "n_local_attn_layers", 0) or 0)
    ]
    if not eligible:
        return candidates
    if len(eligible) > max_expansion_per_type:
        step = len(eligible) / max_expansion_per_type
        eligible = [eligible[min(int(i * step), len(eligible) - 1)]
                    for i in range(max_expansion_per_type)]

    from dataclasses import replace as _dr
    out = list(candidates)
    if want_csa:
        _bs = int((constraints.csa_block_size_options or [64])[0])
        _tk = int((constraints.csa_top_k_options or [16])[0])
        _cd = int(getattr(constraints, "csa_compression_dim", 64) or 64)
        for c in eligible:
            out.append(_dr(c, attention_type="csa", csa_block_size=_bs,
                           csa_top_k_blocks=_tk, csa_compression_dim=_cd))
    if want_idx:
        _nb = int((constraints.indexshare_num_buckets_options or [64])[0])
        _tk = int((constraints.indexshare_top_k_options or [4])[0])
        _id = int(getattr(constraints, "indexshare_index_dim", 64) or 64)
        for c in eligible:
            out.append(_dr(c, attention_type="indexshare",
                           indexshare_num_buckets=_nb,
                           indexshare_top_k_buckets=_tk,
                           indexshare_index_dim=_id))
    if want_msa:
        _w = int((constraints.msa_window_options or [512])[0])
        _dk = int((constraints.msa_dilated_top_k_options or [64])[0])
        _gk = int((constraints.msa_global_top_k_options or [16])[0])
        for c in eligible:
            out.append(_dr(c, attention_type="msa", msa_window_size=_w,
                           msa_dilated_top_k=_dk, msa_global_top_k=_gk))
    return out


def _apply_forced_architecture_features(
    candidates: List[CandidateArch],
    constraints: DeploymentConstraints,
) -> List[CandidateArch]:
    """Apply CLI-requested NSA/YOCO transforms before any evaluation.

    Historically these transforms were stamped only onto the output schema,
    leaving the attached predictions for a different full-attention model.
    This function makes the requested architecture part of candidate identity,
    throughput, quality, Pareto ranking, and final serialization.
    """
    if not constraints.force_nsa and not constraints.force_yoco:
        return candidates
    out: List[CandidateArch] = []
    for cand in candidates:
        if constraints.force_yoco:
            k = int(constraints.yoco_n_self_attn_layers)
            if k >= cand.n_layers:
                continue
            cand.yoco_n_self_attn_layers = k
            cand.yoco_share_pattern = str(constraints.yoco_share_pattern)
        if constraints.force_nsa:
            cand.attention_type = "nsa"
            cand.mla_kv_latent_dim = 0
            cand.mla_q_latent_dim = 0
            cand.mla_rope_head_dim = 0
            cand.mla_nope_head_dim = 0
            cand.nsa_compress_block_size = int(
                constraints.nsa_compress_block_size
            )
            cand.nsa_compress_block_stride = int(
                constraints.nsa_compress_block_stride
            )
            cand.nsa_select_block_size = int(
                constraints.nsa_select_block_size
            )
            cand.nsa_select_top_k = int(constraints.nsa_select_top_k)
            cand.nsa_window_size = int(constraints.nsa_window_size)
        out.append(cand)
    return out


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
    evaluation_failure_reasons: Dict[str, int] = {}

    def _record_evaluation_failure(exc: Exception) -> None:
        message = " ".join(str(exc).split())
        key = f"{type(exc).__name__}: {message or '<no message>'}"
        if len(key) > 240:
            key = key[:237] + "..."
        evaluation_failure_reasons[key] = (
            evaluation_failure_reasons.get(key, 0) + 1
        )

    # 1. Generate candidates (dense + optional MoE + optional state/hybrid +
    #    optional combined MoE×state when both flags are on).
    #
    # Perf-fix Wave 18f (Jul 2026): bound each family pool at the source.
    # With allow_moe + allow_state + rope/cp sweeps at long context the raw
    # cross-product reaches 10^6-10^7 CandidateArch objects (7B @ 2M ctx
    # with cp-options 1,4,16,64: 3.45M candidates, >6 GB once the four
    # family lists were concatenated) — the process died on memory before
    # --max-candidates ever applied. Each family list is therefore strided
    # down to a bounded, deterministic pool immediately after generation,
    # mirroring the exact stride the top-level max_candidates cap uses.
    # Per-family (rather than union) striding also guarantees no family is
    # crowded out of the pool by a combinatorially larger sibling, which
    # matches the Wave 8b stratified-prune philosophy.
    _mc = int(constraints.max_candidates or 0)
    _family_pool_cap = _enumeration_pool_cap(constraints)

    def _expand_local_global(fam: List["CandidateArch"]) -> List["CandidateArch"]:
        """Wave 18g: stream local:global interleave variants into a bounded
        pool. For every attention-only candidate (no state layers, no
        NSA/YOCO/whole-model-SWA), emit one variant per (ratio × window):
        n_local of the layers become sliding-window attention, the rest stay
        global under the candidate's projection (full/GQA or MLA). Streaming
        through _BoundedCandidateList keeps memory bounded regardless of the
        multiplier."""
        if not constraints.allow_local_global:
            return fam
        out = _BoundedCandidateList(_family_pool_cap)
        for c in fam:
            out.append(c)
            if (getattr(c, "n_state_layers", 0)
                    or getattr(c, "swa_window", 0)
                    or c.attention_type not in ("full", "mla")
                    or getattr(c, "yoco_n_self_attn_layers", 0)):
                continue
            for (l_part, g_part) in constraints.local_global_ratio_options:
                n_local = int(round(c.n_layers * l_part / (l_part + g_part)))
                if not (0 < n_local < c.n_layers):
                    continue
                for w in constraints.local_window_options:
                    if w >= constraints.context_length:
                        continue  # window >= ctx is just full attention
                    v = _copy.copy(c)
                    v.n_local_attn_layers = n_local
                    v.swa_window = int(w)
                    out.append(v)
        return list(out)

    def _bounded_family(gen_fn) -> List["CandidateArch"]:
        generation_slices, _source_sliced = _source_generation_slices(
            hw_name, constraints,
        )
        fam: List["CandidateArch"] = []
        for vc in generation_slices:
            fam = fam + gen_fn(hw_name, vc)
        if len(fam) > _family_pool_cap:
            step = len(fam) / _family_pool_cap
            fam = [fam[min(int(i * step), len(fam) - 1)]
                   for i in range(_family_pool_cap)]
        return _expand_local_global(fam)

    candidates = _bounded_family(generate_candidates)
    if constraints.allow_moe:
        candidates = candidates + _bounded_family(generate_moe_candidates)
    if constraints.allow_state:
        candidates = candidates + _bounded_family(generate_state_candidates)
    if constraints.allow_moe and constraints.allow_state:
        # v1-fix Part D: combined MoE + hybrid-state search (Jamba-MoE pattern).
        # Adds MoE×hybrid candidates alongside the pure-MoE and pure-hybrid
        # ones so the Pareto frontier compares all four families on equal
        # footing.
        candidates = candidates + _bounded_family(generate_moe_hybrid_candidates)
    # Wave 33: compressed-attention x MoE / state composition.
    candidates = _expand_compressed_variants(candidates, constraints)
    candidates = _apply_forced_architecture_features(candidates, constraints)

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
        # Perf-fix Wave 18f: state_type / state precision belong in the key
        # too — a mamba2 hybrid and a gla hybrid with the same layer split
        # are different architectures (different residual family, different
        # throughput kernel) and must not collide.
        if c.state_config is not None:
            state_key = (
                c.state_config.get(
                    "state_type", c.state_config.get("type")),
                c.state_config.get("d_state"),
                c.state_config.get("state_expansion"),
                c.state_config.get("n_heads"),
                c.state_config.get("d_head"),
                c.state_config.get(
                    "state_precision", c.state_config.get("precision")),
                tuple(c.layer_type_list or ()),
                c.n_attention_layers,
                c.n_state_layers,
                c.placement_strategy,
            )
        else:
            state_key = None
        # Wave 37 (probe-caught): the key must carry EVERY attention-variant
        # config field. Without the compressed/NSA fields, sweeping
        # --csa-block-sizes 64,128 x --csa-top-k-blocks 8,16 emitted four
        # configs per shape and dedupe silently collapsed them to one —
        # the sweep flags were dead on arrival.
        attn_key = (
            c.attention_type,
            c.mla_kv_latent_dim,
            c.mla_q_latent_dim,
            c.mla_rope_head_dim,
            c.mla_nope_head_dim,
            int(getattr(c, "csa_block_size", 0) or 0),
            int(getattr(c, "csa_top_k_blocks", 0) or 0),
            int(getattr(c, "csa_compression_dim", 0) or 0),
            int(getattr(c, "indexshare_num_buckets", 0) or 0),
            int(getattr(c, "indexshare_top_k_buckets", 0) or 0),
            int(getattr(c, "indexshare_index_dim", 0) or 0),
            int(getattr(c, "msa_window_size", 0) or 0),
            int(getattr(c, "msa_dilated_top_k", 0) or 0),
            int(getattr(c, "msa_global_top_k", 0) or 0),
            int(getattr(c, "nsa_compress_block_size", 0) or 0),
            int(getattr(c, "nsa_compress_block_stride", 0) or 0),
            int(getattr(c, "nsa_select_block_size", 0) or 0),
            int(getattr(c, "nsa_select_top_k", 0) or 0),
            int(getattr(c, "nsa_window_size", 0) or 0),
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
        # Wave 4: tp_degree is part of the dedupe key. Two otherwise-identical
        # candidates at TP=4 vs TP=8 must NOT collide — they map to different
        # lattices, KV/HBM placement, and throughput numbers.
        #
        # Wave 8 follow-up (Jun 2026): sparsity_2_4, yoco_n_self_attn_layers,
        # and swa_window also belong in the key. 2:4 sparsity changes matmul
        # throughput by 2× on supported hardware (real difference in TBT and
        # training_tps); YOCO changes KV cache by K/n_layers; SWA window
        # changes both prefill compute and decode KV reads. Without these
        # fields in the key, two structurally distinct candidates collide
        # and the second is silently dropped.
        sparsity_key = None
        if getattr(c, "sparsity_2_4", None):
            sp = c.sparsity_2_4
            sparsity_key = tuple(sorted(
                (k, bool(v)) for k, v in (sp.items() if isinstance(sp, dict) else [])
            ))
        yoco_key = int(getattr(c, "yoco_n_self_attn_layers", 0) or 0)
        swa_key = (int(getattr(c, "swa_window", 0) or 0),
                   int(getattr(c, "n_local_attn_layers", 0) or 0))  # Wave 18g
        # Perf-fix Wave 18f: dedupe on the structural tuple key, NOT on
        # architecture_fingerprint(c). The fingerprint path deep-copies the
        # whole dataclass (dataclasses.asdict) and JSON-encodes it per
        # candidate (~0.04-0.1 ms each). With allow_moe + allow_state +
        # rope/cp sweeps the raw enumeration reaches ~10^5-10^6 candidates,
        # so fingerprint-dedupe alone cost 40-120 s before --max-candidates
        # was ever applied — the advertised cap did not bound runtime. The
        # tuple below carries every identity axis the fingerprint carried
        # for enumerated candidates (shape, precision, KV, MoE, state, MLA,
        # long-context, MTP, sparsity/YOCO/SWA, TP) at ~1 µs per candidate.
        # architecture_fingerprint remains in use for report joins where a
        # content hash of arbitrary arch objects is genuinely needed.
        key = (
            c.d_model, c.n_layers, c.n_heads, c.d_head, c.n_kv_heads,
            c.ffn_dim, c.vocab_size,
            c.weight_precision, c.ffn_precision, c.activation_precision,
            tuple(sorted((c.attn_precision or {}).items())),
            c.kv_cache_bits,
            moe_key, state_key, attn_key, long_context_key, mtp_key,
            sparsity_key, yoco_key, swa_key,
            int(getattr(c, "tp_degree", 0) or 0),
            int(getattr(c, "pp_degree", 0) or 0),
            int(getattr(c, "ep_degree", 0) or 0),
        )
        if key not in seen:
            seen.add(key)
            unique.append(c)
    candidates = unique
    # v1-fix Wave 5 follow-up (Jun 2026): record the raw enumeration size
    # BEFORE the max_candidates cap. This is what downstream consumers
    # need to know "was the search space large enough that the optimizer's
    # per-cell pick is structurally unreliable" — candidates_generated
    # alone is bounded by the cap and so can't answer the question.
    _candidates_enumerated_raw = len(candidates)
    _precap_candidates = candidates  # Wave 34: kept for local refinement
    if constraints.max_candidates is not None and constraints.max_candidates > 0:
        # Wave 32/34: stratified + cheap-rank-guided — see _stratified_candidate_cap.
        candidates = _stratified_candidate_cap(
            candidates, int(constraints.max_candidates), constraints)

    # Wave 8b (Jun 2026, stratified-by-family post-probe): Two-stage prune.
    # Cap full evaluations to `constraints.max_full_evaluations` while
    # GUARANTEEING fair per-family representation in the surviving set.
    # The original cap-by-pure-cheap-rank had a discovered bug: even with
    # the family-conditional bonus, MoE/hybrid candidates either dominated
    # (at large scale, where their bonus exceeds dense's shape advantage)
    # or got swamped (at small scale, where the converse holds). Either
    # way, the cheap rank was making the family-selection decision the
    # full evaluator is supposed to make.
    #
    # Stratified fix: bucket candidates by family, take top (cap / n_families)
    # from each bucket by cheap rank, then combine. Now every family gets
    # equal full-evaluation budget; the full evaluator decides who wins.
    if (constraints.max_full_evaluations is not None
            and constraints.max_full_evaluations > 0
            and len(candidates) > int(constraints.max_full_evaluations)):
        cap_n = int(constraints.max_full_evaluations)
        try:
            # Wave 18a: family bucketing now flows through the canonical
            # architecture_signature helper, so stratification uses the same
            # classification rules as decision diagnostics / calibration
            # ingestion.  A candidate whose `moe` field is a placeholder
            # dict (e.g. empty or top_k=1) correctly buckets as `dense` here.
            from ac.architecture import architecture_signature as _arch_sig
            def _family_key(c):
                # Wave 34: stratify by (family, attention_type), not family
                # alone. Compressed-attention variants (csa/indexshare/msa)
                # all classify as family "dense"; with family-only buckets
                # the attention-blind cheap rank ties them against their
                # full-attention siblings and sort stability drops whole
                # variant classes (msa vanished from an allow_msa search).
                _at = getattr(c, "attention_type", "full") or "full"
                try:
                    return (_arch_sig(c).legacy_family, _at)
                except Exception:
                    # Defensive fallback: architecture_signature raises on
                    # missing shape fields; keep the search running rather
                    # than dropping the candidate entirely.
                    has_moe = bool(getattr(c, "moe", None))
                    has_state = bool(getattr(c, "state_config", None))
                    if has_moe and has_state: return ("moe_hybrid", _at)
                    if has_moe:                return ("moe", _at)
                    if has_state:              return ("hybrid", _at)
                    return ("dense", _at)
            buckets: dict = {}
            for c in candidates:
                buckets.setdefault(_family_key(c), []).append(c)
            n_fams = max(1, len(buckets))
            per_fam = max(1, cap_n // n_fams)
            survived = []
            for fam, fam_cands in buckets.items():
                fam_cands.sort(
                    key=lambda c: _cheap_quality_rank(
                        c, constraints.training_tokens,
                        constraints.quality_model_version))
                survived.extend(fam_cands[:per_fam])
            # If we underspent the budget (e.g. small families), top up with
            # the next-best across the union by cheap rank, deduped.
            if len(survived) < cap_n:
                # Perf-fix Wave 18f: membership by object identity. The old
                # `c not in survived` list-scan invoked dataclass __eq__
                # against up to cap_n entries per candidate — O(N × cap)
                # deep comparisons on a 10^5-10^6 candidate list.
                _survived_ids = {id(c) for c in survived}
                remaining = [c for c in candidates if id(c) not in _survived_ids]
                remaining.sort(
                    key=lambda c: _cheap_quality_rank(
                        c, constraints.training_tokens,
                        constraints.quality_model_version))
                survived.extend(remaining[:cap_n - len(survived)])
            candidates = survived[:cap_n]
        except Exception:
            # Cheap-rank should never fail; if it does, fall back to
            # full evaluation rather than dropping candidates silently.
            pass

    # 3. Evaluate all candidates
    evaluated = []
    total = len(candidates)
    # Wave 47: name the search size up front when it is large enough that
    # the user will be waiting on it, and say which knobs make it faster.
    # Keyed on progress_every so --quiet (progress_every=0) suppresses it.
    if constraints.progress_every and total >= 2 * constraints.progress_every:
        _uncapped = (constraints.max_candidates is None
                     and constraints.max_full_evaluations is None)
        _hint = (" (uncapped search; --max-candidates or "
                 "--max-full-evaluations trades coverage for speed)"
                 if _uncapped else "")
        print(
            f"[arch-compiler] {total:,} candidates to fully evaluate{_hint}",
            file=sys.stderr,
        )
    for idx, cand in enumerate(candidates, start=1):
        try:
            ev = evaluate_candidate(cand, hw_name, constraints)
            evaluated.append(ev)
        except Exception as exc:
            _record_evaluation_failure(exc)
            continue
        if constraints.progress_every and idx % constraints.progress_every == 0:
            print(
                f"[arch-compiler] evaluated {idx:,}/{total:,} candidates "
                f"({len(evaluated):,} scored)",
                file=sys.stderr,
            )

    # 3b. Wave 34: cheap local refinement. When the cap dropped candidates,
    # densify the lattice neighborhood of the provisional per-class leaders
    # with shapes from the PRE-CAP pool, then fold them into the same
    # evaluated set. No-op for uncapped searches.
    _refine_budget = int(getattr(constraints, "local_refine_budget", 0) or 0)
    # max_full_evaluations is a HARD contract on total full evaluations
    # (pinned by test_two_stage_evaluation): refinement must fit in the
    # remaining headroom, which after stage one fills the cap is zero —
    # raise the cap if you want refinement under two-stage pruning.
    if constraints.max_full_evaluations is not None             and constraints.max_full_evaluations > 0:
        _refine_budget = min(
            _refine_budget,
            max(0, int(constraints.max_full_evaluations) - len(evaluated)))
    if (_refine_budget > 0 and evaluated
            and len(_precap_candidates) > len(candidates)):
        _by_class: dict = {}
        for e in evaluated:
            k = (getattr(e.arch, "attention_type", "full") or "full",
                 bool(getattr(e.arch, "moe", None)),
                 bool(getattr(e.arch, "state_config", None)))
            cur = _by_class.get(k)
            if cur is None or e.predicted_loss < cur.predicted_loss:
                _by_class[k] = e
        _leaders = [e.arch for e in _by_class.values()]
        _extra = _select_refinement_neighbors(
            _leaders, _precap_candidates,
            [e.arch for e in evaluated], _refine_budget)
        for cand in _extra:
            try:
                evaluated.append(evaluate_candidate(cand, hw_name, constraints))
            except Exception as exc:
                _record_evaluation_failure(exc)
                continue

    evaluation_failures = sum(evaluation_failure_reasons.values())
    if candidates and not evaluated and evaluation_failures:
        dominant, count = max(
            evaluation_failure_reasons.items(), key=lambda item: item[1]
        )
        raise ValueError(
            f"all {evaluation_failures} candidate evaluation(s) failed; "
            f"most common ({count}x): {dominant}"
        )

    # 4. Filter to feasible
    feasible = [e for e in evaluated if e.meets_constraints]

    # 5. Pareto frontier. When at least one candidate fits training HBM,
    # exclude overflow plans from winner selection: training spill/offload is
    # not modeled. If every plan overflows, retain the best-effort frontier
    # and let the CLI's explicit no-fit warning carry the result.
    selection_feasible = _prefer_training_fit(feasible, hw_name)
    pareto = compute_pareto_frontier(selection_feasible)

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
    if selection_feasible:
        scoring_pool = pareto if pareto else selection_feasible
        # Build the SAME sort key used by both the picker and the CSV
        # writer. Anchoring both paths to one builder is what guarantees
        # that `rank=1` in pareto.csv always agrees with `selected=True`.
        display_sort_key = build_display_sort_key(scoring_pool, constraints)
        optimal = min(scoring_pool, key=display_sort_key)

    # v1-fix sanity gate (June 2026): never surface a candidate whose
    # predicted_loss carries the INFEASIBLE-sentinel inflation. Even though
    # evaluate_candidate sets meets_constraints=False when
    # predicted_loss > 1e4 (line ~1957), a few code paths (re-batched
    # serializer reruns, hardware-fallback evaluators) historically picked
    # up an INFEASIBLE-tainted candidate as "best of the bad lot" and let
    # loss ~ 2e6 leak into the public grid. The audit caught 19.8% of
    # feasible rows showing loss in the millions because of this — the
    # filter below guarantees that a returned `optimal` is bounded by a
    # small multiple of the Chinchilla baseline (i.e. total_penalty_fraction
    # is bounded), turning any silent leak into an honest "no feasible
    # solution" instead.
    import os as _os2
    _soft_sentinel_optimal = (
        getattr(constraints, "allow_quality_sentinel", False)
        or _os2.environ.get("AC_QUALITY_SENTINEL_SOFT") == "1"
    )
    if optimal is not None and not _soft_sentinel_optimal:
        try:
            base = float(optimal.quality.chinchilla_baseline)
            loss = float(optimal.predicted_loss)
            if base > 0 and loss > _SENTINEL_LOSS_MULT * base:
                optimal = None
        except (AttributeError, TypeError, ValueError):
            # Defensive: if the quality fields are absent, treat as no result.
            if optimal is None or not getattr(optimal, "meets_constraints", False):
                optimal = None

    elapsed = time.time() - t0

    # 7. Identify binding constraints
    binding = _identify_binding_constraints(optimal, constraints, hw_name, evaluated, feasible)

    # Wave 18d (Jun 2026): confidence-aware decision assessment.
    # `optimal` remains as the legacy loss-sorted single-winner projection
    # for one transition version; `decision` is the new source of truth for
    # matrix rendering and honours the pre-calibration unique-winner rule
    # (advantage > max(5%, combined_quality_uncertainty_pct)).
    try:
        from ac.decision import assess_decision as _assess_decision
        # Assess over the *feasible* set — Wave 18c physical guards already
        # excluded infeasible candidates upstream. Pass the Pareto-selected
        # subset first so the assessment focuses on the operating-point
        # contenders, falling back to feasible if Pareto is empty.
        _decision_pool = pareto if pareto else selection_feasible
        _decision = _assess_decision(_decision_pool)
    except Exception:
        _decision = None

    return OptimizationResult(
        optimal=optimal,
        pareto_frontier=pareto,
        all_evaluated=evaluated,
        candidates_generated=len(candidates),
        candidates_feasible=len(feasible),
        candidates_evaluated=len(evaluated),
        evaluation_failures=evaluation_failures,
        evaluation_failure_reasons=dict(evaluation_failure_reasons),
        candidates_enumerated_raw=_candidates_enumerated_raw,
        search_time_sec=round(elapsed, 2),
        hardware=hw_name,
        constraints=constraints,
        binding_constraints=binding,
        decision=_decision,
    )


# =============================================================================
# Wave 6 (Jun 2026): optimize_across_contexts — principled shape coherence
# =============================================================================
#
# Why this entry point exists:
#   The per-cell `optimize()` above picks an independent architecture for every
#   (hw, params, family, ctx) tuple. Adjacent ctxs in the same row can land on
#   structurally different shapes (e.g., 13B MoE picks d=6144×L=55 at 8k and
#   d=6144×L=30 at 128k), which makes the displayed grid look incoherent and
#   forced us to add a post-hoc canonical-shape pin (see
#   scripts/_generator_payload.py:_pin_canonical_shape_per_family).
#
#   This Wave-6 entry point makes "one architecture serves the whole row" a
#   first-class operation: enumerate candidates ONCE at a reference ctx (default
#   128k — production deployment context), then evaluate every candidate at
#   every ctx in the row jointly. The displayed optimum is chosen from a joint
#   Pareto frontier across all ctxs, using ctx-weighted scoring.
#
#   The existing `optimize()` is preserved unchanged for back-compat — every
#   existing call site keeps working. The grid driver can opt in to the new
#   path via AC_GRID_MULTI_CTX=1 (see scripts/_generator_payload.py wiring).
#
# Design tradeoffs (see plan/redesign/06-principled-pin.md for full context):
#   - Joint feasibility: a candidate is "row-feasible" iff it meets constraints
#     at the reference_ctx AND at >=60% of the ctxs in the list. This prevents
#     spurious infeasibility from a single hard corner.
#   - Joint Pareto: a candidate is dominated iff some other candidate is no-
#     worse at every (ctx, objective) pair and strictly better at at least one.
#     Frontier surface is larger than per-cell — we cap at 200 candidates by
#     weighted score to keep downstream payloads bounded.
#   - Scoring: weighted-sum of normalized per-ctx losses (default uniform
#     across ctxs; pass `ctx_weights` to bias toward deployment-realistic
#     contexts).


@dataclass
class MultiCtxResult:
    """Result of optimize_across_contexts: one architecture, evaluated at
    every requested context length.

    Fields:
      optimal: the architecture picked by the joint scoring. None when no
        candidate is row-feasible.
      pareto_frontier: candidates on the joint Pareto frontier across ctxs.
      per_ctx_metrics: ctx -> EvaluatedCandidate (the OPTIMUM evaluated at
        that ctx). Same `optimal.arch` object across all ctxs; only the
        throughput/quality results differ per ctx.
      per_ctx_all_evaluated: ctx -> list of EvaluatedCandidate (every
        candidate at every ctx). Memory-heavy; not always populated.
      candidates_enumerated_raw: pre-cap enumeration size at the reference
        ctx. Used by the canonical-shape pin's gate.
      candidates_feasible_per_ctx: ctx -> count of candidates feasible at
        that ctx (before joint feasibility filter).
      reference_ctx: the ctx used to plan parallelism and enumerate.
      ctx_weights: the weights used to pick the optimum (one per ctx).
    """
    optimal: Optional["EvaluatedCandidate"] = None
    pareto_frontier: List["EvaluatedCandidate"] = field(default_factory=list)
    per_ctx_metrics: Dict[int, "EvaluatedCandidate"] = field(default_factory=dict)
    per_ctx_all_evaluated: Dict[int, List["EvaluatedCandidate"]] = field(default_factory=dict)
    candidates_enumerated_raw: int = 0
    candidates_feasible_per_ctx: Dict[int, int] = field(default_factory=dict)
    evaluation_failures: int = 0
    evaluation_failure_reasons: Dict[str, int] = field(default_factory=dict)
    search_time_sec: float = 0.0
    hardware: str = ""
    constraints: Optional[DeploymentConstraints] = None
    reference_ctx: int = 0
    ctx_weights: Dict[int, float] = field(default_factory=dict)


def _enumerate_and_dedupe(
    hw_name: str,
    constraints: DeploymentConstraints,
) -> Tuple[List["CandidateArch"], int]:
    """Run all candidate generators and dedupe. Returns (candidates, raw_count)
    where raw_count is the pre-cap dedup'd size. Lifted from optimize() so
    optimize_across_contexts can share the enumeration logic without re-evaluating
    per-ctx for every candidate.
    """
    # Wave 18h: vocab sweep — run each generator once per vocab option. The
    # generators read constraints.vocab_size for the parameter ledger, so a
    # shallow copy per vocab is the single wiring point for the whole axis.
    vocab_list = [
        int(v) for v in (constraints.vocab_options or [constraints.vocab_size])
    ]
    candidates: List["CandidateArch"] = []
    for _vocab in dict.fromkeys(vocab_list):
        vc = (
            constraints
            if _vocab == constraints.vocab_size
            else _dc_replace(constraints, vocab_size=_vocab)
        )
        candidates = candidates + generate_candidates(hw_name, vc)
        if vc.allow_moe:
            candidates = candidates + generate_moe_candidates(hw_name, vc)
        if vc.allow_state:
            candidates = candidates + generate_state_candidates(hw_name, vc)
        if vc.allow_moe and vc.allow_state:
            candidates = candidates + generate_moe_hybrid_candidates(hw_name, vc)
    # Wave 33: compressed-attention x MoE / state composition.
    candidates = _expand_compressed_variants(candidates, constraints)
    candidates = _apply_forced_architecture_features(candidates, constraints)

    # Dedupe (same key shape as in optimize())
    # Perf-fix Wave 18f/18g: this loop had the same bug optimize()'s dedupe
    # had — it built the cheap structural sub-keys and then hashed the whole
    # dataclass through architecture_fingerprint (asdict + json + sha256,
    # ~0.04-0.1 ms/candidate) anyway. Dedupe on the structural tuple.
    seen = set()
    unique = []
    for c in candidates:
        if c.moe is not None:
            moe_key = (c.moe["n_experts"], c.moe["top_k"], c.moe["expert_dim"],
                       (c.moe.get("shared_expert") or {}).get("ffn_dim"),
                       c.ep_degree, c.n_dense_ffn_layers)
        else:
            moe_key = None
        if c.state_config is not None:
            state_key = (c.state_config.get(
                             "state_type", c.state_config.get("type")),
                         c.state_config.get("d_state"),
                         c.state_config.get("state_expansion"),
                         c.state_config.get("n_heads"),
                         c.state_config.get("d_head"),
                         c.state_config.get(
                             "state_precision", c.state_config.get("precision")),
                         tuple(c.layer_type_list or ()),
                         c.n_attention_layers,
                         c.n_state_layers, c.placement_strategy)
        else:
            state_key = None
        # Wave 37: same attention-variant completeness as optimize()'s key.
        attn_key = (
            c.attention_type, c.mla_kv_latent_dim, c.mla_q_latent_dim,
            c.mla_rope_head_dim, c.mla_nope_head_dim,
            int(getattr(c, "csa_block_size", 0) or 0),
            int(getattr(c, "csa_top_k_blocks", 0) or 0),
            int(getattr(c, "csa_compression_dim", 0) or 0),
            int(getattr(c, "indexshare_num_buckets", 0) or 0),
            int(getattr(c, "indexshare_top_k_buckets", 0) or 0),
            int(getattr(c, "indexshare_index_dim", 0) or 0),
            int(getattr(c, "msa_window_size", 0) or 0),
            int(getattr(c, "msa_dilated_top_k", 0) or 0),
            int(getattr(c, "msa_global_top_k", 0) or 0),
            int(getattr(c, "nsa_compress_block_size", 0) or 0),
            int(getattr(c, "nsa_compress_block_stride", 0) or 0),
            int(getattr(c, "nsa_select_block_size", 0) or 0),
            int(getattr(c, "nsa_select_top_k", 0) or 0),
            int(getattr(c, "nsa_window_size", 0) or 0),
        )
        long_context_key = (c.cp_degree, c.cp_method, c.rope_scaling_method,
                            round(float(c.rope_scaling_factor), 6),
                            c.rope_original_max_position)
        mtp_key = (c.mtp_n_predict_depths, c.mtp_depth_n_layers,
                   round(float(c.mtp_train_loss_weight), 6))
        # Wave 8 follow-up: include sparsity_2_4, yoco, swa_window so
        # structurally distinct candidates don't collide in the dedupe.
        sparsity_key = None
        if getattr(c, "sparsity_2_4", None):
            sp = c.sparsity_2_4
            sparsity_key = tuple(sorted(
                (k, bool(v)) for k, v in (sp.items() if isinstance(sp, dict) else [])
            ))
        yoco_key = int(getattr(c, "yoco_n_self_attn_layers", 0) or 0)
        swa_key = (int(getattr(c, "swa_window", 0) or 0),
                   int(getattr(c, "n_local_attn_layers", 0) or 0))  # Wave 18g
        key = (
            c.d_model, c.n_layers, c.n_heads, c.d_head, c.n_kv_heads,
            c.ffn_dim, c.vocab_size,
            c.weight_precision, c.ffn_precision, c.activation_precision,
            tuple(sorted((c.attn_precision or {}).items())),
            c.kv_cache_bits,
            moe_key, state_key, attn_key, long_context_key, mtp_key,
            sparsity_key, yoco_key, swa_key,
            int(getattr(c, "tp_degree", 0) or 0),
            int(getattr(c, "pp_degree", 0) or 0),
            int(getattr(c, "ep_degree", 0) or 0),
        )
        if key not in seen:
            seen.add(key)
            unique.append(c)
    raw_count = len(unique)

    # Apply max_candidates cap (same logic as optimize()).
    # Wave 32/34: stratified + cheap-rank-guided — see _stratified_candidate_cap.
    precap = unique
    if constraints.max_candidates is not None and constraints.max_candidates > 0:
        unique = _stratified_candidate_cap(
            unique, int(constraints.max_candidates), constraints)
    return unique, raw_count, precap


def _is_jointly_dominated(
    a_evals: List["EvaluatedCandidate"],
    b_evals: List["EvaluatedCandidate"],
) -> bool:
    """Return True iff b dominates a across all ctxs jointly. Mirrors
    is_dominated but compares per-ctx vectors instead of scalars."""
    at_least_one_strictly_better = False
    for ev_a, ev_b in zip(a_evals, b_evals):
        # Same 6 objectives as is_dominated()
        obj_a = (ev_a.predicted_loss,
                 -_evaluated_training_tps_per_gpu(ev_a),
                 ev_a.serving_tbt_ms,
                 ev_a.throughput.prefill_time_ms, ev_a.memory_per_gpu_gb,
                 float(getattr(ev_a.throughput, "training_memory_per_gpu_gb", 0.0) or 0.0),
                 ev_a.arch.total_params,
                 _evaluated_serving_instance_gpus(ev_a))
        obj_b = (ev_b.predicted_loss,
                 -_evaluated_training_tps_per_gpu(ev_b),
                 ev_b.serving_tbt_ms,
                 ev_b.throughput.prefill_time_ms, ev_b.memory_per_gpu_gb,
                 float(getattr(ev_b.throughput, "training_memory_per_gpu_gb", 0.0) or 0.0),
                 ev_b.arch.total_params,
                 _evaluated_serving_instance_gpus(ev_b))
        for oa, ob in zip(obj_a, obj_b):
            if ob > oa:
                return False  # b worse than a on this objective at this ctx
            if ob < oa:
                at_least_one_strictly_better = True
    return at_least_one_strictly_better


def _compute_joint_pareto(
    per_cand_per_ctx: List[List["EvaluatedCandidate"]],
) -> List[int]:
    """Indices of candidates on the joint Pareto frontier across ctxs.

    per_cand_per_ctx[i] is the list of EvaluatedCandidate for candidate i,
    one per ctx in the ctx_list order.
    """
    n = len(per_cand_per_ctx)
    frontier = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j: continue
            if _is_jointly_dominated(per_cand_per_ctx[i], per_cand_per_ctx[j]):
                dominated = True
                break
        if not dominated:
            frontier.append(i)
    return frontier


def _meets_declared_budgets(ev: "EvaluatedCandidate",
                            constraints: "DeploymentConstraints") -> bool:
    """True iff the candidate satisfies every serving budget the user
    actually declared (TBT / TTFT). Budgets are soft guards by design
    (fails_feasibility=False — extreme corners should degrade to a
    marked best-effort answer, not an empty one), but Wave 33 makes them
    binding at PICK time: a candidate violating a declared budget loses
    to any candidate that meets it, and only if every candidate violates
    do we fall back to ranking among violators."""
    tbt = getattr(constraints, "serving_tbt_ms", None)
    if tbt is not None and float(ev.serving_tbt_ms) > float(tbt):
        return False
    ttft = getattr(constraints, "serving_ttft_ms", None)
    if ttft is not None and float(ev.throughput.prefill_time_ms) > float(ttft):
        return False
    return True


def _pick_joint_optimum(
    pareto_indices: List[int],
    per_cand_per_ctx: List[List["EvaluatedCandidate"]],
    ctx_list: List[int],
    ctx_weights: Dict[int, float],
    constraints: Optional["DeploymentConstraints"] = None,
) -> Optional[int]:
    """Score each Pareto candidate by weighted-sum of normalized per-ctx losses.
    Returns the index of the picked candidate (an index into the original
    candidates list, NOT into pareto_indices).

    Wave 33: when serving budgets are declared, candidates that meet them
    at the reference ctx (index 0 weight ordering aside — we check every
    ctx and require a majority) outrank all violators regardless of loss;
    ranking falls back to all candidates when nobody meets the budgets.
    """
    if not pareto_indices:
        return None

    if constraints is not None and (
            getattr(constraints, "serving_tbt_ms", None) is not None
            or getattr(constraints, "serving_ttft_ms", None) is not None):
        ok = [i for i in pareto_indices
              if sum(1 for ev in per_cand_per_ctx[i]
                     if _meets_declared_budgets(ev, constraints))
              > len(per_cand_per_ctx[i]) // 2]
        if ok:
            pareto_indices = ok

    # Normalize losses per-ctx: divide by the min loss at that ctx so each ctx
    # contributes on the same scale.
    per_ctx_min_loss = []
    for k, ctx in enumerate(ctx_list):
        losses = [per_cand_per_ctx[i][k].predicted_loss for i in pareto_indices]
        per_ctx_min_loss.append(min(losses) if losses else 1.0)

    weights = [ctx_weights.get(ctx, 1.0) for ctx in ctx_list]
    total_w = sum(weights) or 1.0

    scores = {}
    for i in pareto_indices:
        score = 0.0
        for k, ctx in enumerate(ctx_list):
            min_l = per_ctx_min_loss[k] or 1.0
            score += weights[k] * (per_cand_per_ctx[i][k].predicted_loss / min_l)
        scores[i] = score / total_w
    best_score = min(scores.values())

    # Wave 34: candidates whose weighted score sits inside the same noise
    # band as the best are quality-equivalent; break the tie with the
    # canonical display sort key at the reference ctx (heaviest-weighted
    # ctx when weights differ) instead of raw argmin. This keeps the
    # multi-ctx pick consistent with optimize()'s picker — the two paths
    # used to diverge on kv-bits/memory once the pool got dense enough
    # for exact-loss near-ties to appear. Band = 0.5% relative, matching
    # the uncalibrated tiebreak cap in build_display_sort_key.
    _BAND_REL = 0.005 * 0.25 * 2.0  # 0.25 x capped 2% uncertainty
    in_band = [i for i in pareto_indices
               if scores[i] <= best_score * (1.0 + _BAND_REL)]
    if len(in_band) == 1:
        return in_band[0]
    ref_k = max(range(len(ctx_list)),
                key=lambda k: ctx_weights.get(ctx_list[k], 1.0))
    pool = [per_cand_per_ctx[i][ref_k] for i in in_band]
    try:
        key = build_display_sort_key(pool, constraints)
        ranked = sorted(zip(pool, in_band), key=lambda t: key(t[0]))
        return ranked[0][1]
    except Exception:
        return min(in_band, key=lambda i: scores[i])


def optimize_across_contexts(
    hw_name: str,
    constraints: DeploymentConstraints,
    ctx_list: List[int],
    reference_ctx: Optional[int] = None,
    ctx_weights: Optional[Dict[int, float]] = None,
    joint_feasibility_min_ratio: float = 0.6,
) -> MultiCtxResult:
    """Pick ONE architecture and evaluate it at every ctx in ctx_list.

    See module-level comment block above for full design rationale.

    Args:
      hw_name: target hardware.
      constraints: deployment constraints. context_length is overridden per
        ctx in ctx_list; all other fields (tp_options, cp_options, allow_moe,
        etc.) apply uniformly.
      ctx_list: list of contexts to evaluate at. The reference ctx (used for
        enumeration + parallelism planning) defaults to 128k if present, else
        the median of ctx_list.
      reference_ctx: override for the reference ctx. Must be in ctx_list.
      ctx_weights: optional dict {ctx: weight} for the weighted-sum scoring.
        Defaults to uniform.
      joint_feasibility_min_ratio: a candidate must be feasible at the
        reference ctx AND at least `ratio * len(ctx_list)` ctxs total.
        Default 0.6 (so a 5-ctx row requires feasibility at 3+ ctxs).

    Returns:
      MultiCtxResult with one `optimal` architecture and per-ctx metrics.
    """
    t0 = time.time()
    evaluation_failure_reasons: Dict[str, int] = {}

    def _record_evaluation_failure(exc: Exception) -> None:
        message = " ".join(str(exc).split())
        key = f"{type(exc).__name__}: {message or '<no message>'}"
        if len(key) > 240:
            key = key[:237] + "..."
        evaluation_failure_reasons[key] = (
            evaluation_failure_reasons.get(key, 0) + 1
        )

    # Reference ctx selection.
    if reference_ctx is None:
        if 131072 in ctx_list:
            reference_ctx = 131072
        else:
            reference_ctx = sorted(ctx_list)[len(ctx_list) // 2]
    if reference_ctx not in ctx_list:
        raise ValueError(
            f"reference_ctx={reference_ctx} not in ctx_list={ctx_list}")

    # Default uniform weights.
    if ctx_weights is None:
        ctx_weights = {ctx: 1.0 for ctx in ctx_list}

    # A one-cell multi-context request has no joint trade-off to solve. Route
    # it through the canonical optimizer so candidate pruning, Pareto
    # selection, uncertainty tiebreaks, and evaluator-failure reporting cannot
    # drift between two public entry points for the same problem.
    if len(ctx_list) == 1:
        only_ctx = int(ctx_list[0])
        single_constraints = _copy.copy(constraints)
        single_constraints.context_length = only_ctx
        single = optimize(hw_name, single_constraints)
        per_ctx_metrics = (
            {only_ctx: single.optimal} if single.optimal is not None else {}
        )
        return MultiCtxResult(
            optimal=single.optimal,
            pareto_frontier=list(single.pareto_frontier),
            per_ctx_metrics=per_ctx_metrics,
            per_ctx_all_evaluated={only_ctx: list(single.all_evaluated)},
            candidates_enumerated_raw=single.candidates_enumerated_raw,
            candidates_feasible_per_ctx={
                only_ctx: single.candidates_feasible},
            evaluation_failures=single.evaluation_failures,
            evaluation_failure_reasons=dict(
                single.evaluation_failure_reasons),
            search_time_sec=single.search_time_sec,
            hardware=hw_name,
            constraints=single_constraints,
            reference_ctx=only_ctx,
            ctx_weights=dict(ctx_weights),
        )

    # Step 1: enumerate candidates ONCE at the reference ctx. The lattice
    # filter (in each generator) and the dedup logic both run inside
    # _enumerate_and_dedupe.
    ref_constraints = _copy.copy(constraints)
    ref_constraints.context_length = int(reference_ctx)
    candidates, raw_count, _precap_pool = _enumerate_and_dedupe(hw_name, ref_constraints)

    # Wave 8b (Jun 2026, stratified-by-family): same fairness rule as
    # `optimize()` — bucket by family, take top per-family by cheap rank,
    # then combine. In multi-ctx the savings compound (every cand evals
    # at every ctx).
    if (constraints.max_full_evaluations is not None
            and constraints.max_full_evaluations > 0
            and len(candidates) > int(constraints.max_full_evaluations)):
        cap_n = int(constraints.max_full_evaluations)
        try:
            # Wave 18a: family bucketing now flows through the canonical
            # architecture_signature helper, so stratification uses the same
            # classification rules as decision diagnostics / calibration
            # ingestion.  A candidate whose `moe` field is a placeholder
            # dict (e.g. empty or top_k=1) correctly buckets as `dense` here.
            from ac.architecture import architecture_signature as _arch_sig
            def _family_key(c):
                # Wave 34: stratify by (family, attention_type), not family
                # alone. Compressed-attention variants (csa/indexshare/msa)
                # all classify as family "dense"; with family-only buckets
                # the attention-blind cheap rank ties them against their
                # full-attention siblings and sort stability drops whole
                # variant classes (msa vanished from an allow_msa search).
                _at = getattr(c, "attention_type", "full") or "full"
                try:
                    return (_arch_sig(c).legacy_family, _at)
                except Exception:
                    # Defensive fallback: architecture_signature raises on
                    # missing shape fields; keep the search running rather
                    # than dropping the candidate entirely.
                    has_moe = bool(getattr(c, "moe", None))
                    has_state = bool(getattr(c, "state_config", None))
                    if has_moe and has_state: return ("moe_hybrid", _at)
                    if has_moe:                return ("moe", _at)
                    if has_state:              return ("hybrid", _at)
                    return ("dense", _at)
            buckets: dict = {}
            for c in candidates:
                buckets.setdefault(_family_key(c), []).append(c)
            n_fams = max(1, len(buckets))
            per_fam = max(1, cap_n // n_fams)
            survived = []
            for fam, fam_cands in buckets.items():
                fam_cands.sort(
                    key=lambda c: _cheap_quality_rank(
                        c, constraints.training_tokens,
                        constraints.quality_model_version))
                survived.extend(fam_cands[:per_fam])
            if len(survived) < cap_n:
                # Perf-fix Wave 18f: membership by object identity. The old
                # `c not in survived` list-scan invoked dataclass __eq__
                # against up to cap_n entries per candidate — O(N × cap)
                # deep comparisons on a 10^5-10^6 candidate list.
                _survived_ids = {id(c) for c in survived}
                remaining = [c for c in candidates if id(c) not in _survived_ids]
                remaining.sort(
                    key=lambda c: _cheap_quality_rank(
                        c, constraints.training_tokens,
                        constraints.quality_model_version))
                survived.extend(remaining[:cap_n - len(survived)])
            candidates = survived[:cap_n]
        except Exception:
            pass

    # Step 2: evaluate every candidate at every ctx. Per-candidate list of
    # EvaluatedCandidate, in ctx_list order.
    per_cand_per_ctx: List[List["EvaluatedCandidate"]] = []
    per_ctx_all_evaluated: Dict[int, List["EvaluatedCandidate"]] = {ctx: [] for ctx in ctx_list}

    def _eval_row(cand) -> None:
        cand_evals = []
        for ctx in ctx_list:
            ctx_constraints = _copy.copy(constraints)
            ctx_constraints.context_length = int(ctx)
            try:
                ev = evaluate_candidate(cand, hw_name, ctx_constraints)
            except Exception as exc:
                _record_evaluation_failure(exc)
                ev = None
            if ev is None:
                # Skip candidates that fail to evaluate at any ctx — they're
                # not row-comparable.
                return
            cand_evals.append(ev)
        for k, ctx in enumerate(ctx_list):
            per_ctx_all_evaluated[ctx].append(cand_evals[k])
        per_cand_per_ctx.append(cand_evals)

    for cand in candidates:
        _eval_row(cand)

    evaluation_failures = sum(evaluation_failure_reasons.values())
    if candidates and not per_cand_per_ctx and evaluation_failures:
        dominant, count = max(
            evaluation_failure_reasons.items(), key=lambda item: item[1]
        )
        raise ValueError(
            f"all multi-context candidate rows failed evaluation; "
            f"{evaluation_failures} cell failure(s), most common "
            f"({count}x): {dominant}"
        )

    # Step 2b: Wave 34 cheap local refinement (same design as optimize()):
    # when the cap dropped candidates, densify the lattice neighborhood of
    # the provisional per-class leaders — leaders judged at the REFERENCE
    # ctx — with shapes from the pre-cap pool, evaluated at every ctx.
    _refine_budget = int(getattr(constraints, "local_refine_budget", 0) or 0)
    # Same hard-contract rule as optimize(): max_full_evaluations bounds
    # TOTAL full evaluations per ctx, refinement included.
    if constraints.max_full_evaluations is not None             and constraints.max_full_evaluations > 0:
        _refine_budget = min(
            _refine_budget,
            max(0, int(constraints.max_full_evaluations)
                - len(per_cand_per_ctx)))
    _ref_k = ctx_list.index(reference_ctx)
    if (_refine_budget > 0 and per_cand_per_ctx
            and len(_precap_pool) > len(candidates)):
        _by_class: dict = {}
        for cand_evals in per_cand_per_ctx:
            ev = cand_evals[_ref_k]
            k = (getattr(ev.arch, "attention_type", "full") or "full",
                 bool(getattr(ev.arch, "moe", None)),
                 bool(getattr(ev.arch, "state_config", None)))
            cur = _by_class.get(k)
            if cur is None or ev.predicted_loss < cur.predicted_loss:
                _by_class[k] = ev
        _extra = _select_refinement_neighbors(
            [e.arch for e in _by_class.values()], _precap_pool,
            [row[_ref_k].arch for row in per_cand_per_ctx], _refine_budget)
        for cand in _extra:
            _eval_row(cand)

    # Step 3: joint feasibility. A candidate is row-feasible iff feasible at
    # the reference ctx AND at >= joint_feasibility_min_ratio of the ctxs.
    ref_ctx_idx = ctx_list.index(reference_ctx)
    min_feasible_ctxs = max(1, int(round(joint_feasibility_min_ratio * len(ctx_list))))
    row_feasible_idx: List[int] = []
    for i, cand_evals in enumerate(per_cand_per_ctx):
        n_feas = sum(1 for ev in cand_evals if ev.meets_constraints)
        if cand_evals[ref_ctx_idx].meets_constraints and n_feas >= min_feasible_ctxs:
            row_feasible_idx.append(i)

    per_ctx_feasible_count = {
        ctx: sum(1 for ev in per_ctx_all_evaluated[ctx] if ev.meets_constraints)
        for ctx in ctx_list
    }

    # Step 4: joint Pareto frontier.
    if row_feasible_idx:
        feasible_per_cand_per_ctx = [per_cand_per_ctx[i] for i in row_feasible_idx]
        pareto_local_idx = _compute_joint_pareto(feasible_per_cand_per_ctx)
        pareto_global_idx = [row_feasible_idx[k] for k in pareto_local_idx]
    else:
        pareto_global_idx = []

    # Step 5: pick the displayed optimum via weighted scoring.
    optimal_idx = _pick_joint_optimum(
        pareto_global_idx, per_cand_per_ctx, ctx_list, ctx_weights,
        constraints=constraints)

    optimal_ev_per_ctx: Dict[int, "EvaluatedCandidate"] = {}
    if optimal_idx is not None:
        for k, ctx in enumerate(ctx_list):
            optimal_ev_per_ctx[ctx] = per_cand_per_ctx[optimal_idx][k]

    # Wave 5 sentinel gate (same as optimize()) — applied at the reference ctx.
    import os as _os_local
    soft_sentinel = (getattr(constraints, "allow_quality_sentinel", False)
                     or _os_local.environ.get("AC_QUALITY_SENTINEL_SOFT") == "1")
    if optimal_idx is not None and not soft_sentinel:
        ref_ev = optimal_ev_per_ctx[reference_ctx]
        try:
            base = float(ref_ev.quality.chinchilla_baseline)
            loss = float(ref_ev.predicted_loss)
            if base > 0 and loss > _SENTINEL_LOSS_MULT * base:
                optimal_idx = None
                optimal_ev_per_ctx = {}
        except (AttributeError, TypeError, ValueError):
            pass

    optimal_ev = optimal_ev_per_ctx.get(reference_ctx) if optimal_idx is not None else None
    pareto_evs = [per_cand_per_ctx[i][ref_ctx_idx] for i in pareto_global_idx]

    return MultiCtxResult(
        optimal=optimal_ev,
        pareto_frontier=pareto_evs,
        per_ctx_metrics=optimal_ev_per_ctx,
        per_ctx_all_evaluated=per_ctx_all_evaluated,
        candidates_enumerated_raw=raw_count,
        candidates_feasible_per_ctx=per_ctx_feasible_count,
        evaluation_failures=evaluation_failures,
        evaluation_failure_reasons=dict(evaluation_failure_reasons),
        search_time_sec=round(time.time() - t0, 2),
        hardware=hw_name,
        constraints=constraints,
        reference_ctx=int(reference_ctx),
        ctx_weights=dict(ctx_weights),
    )


def _quality_pack_loaded() -> bool:
    """True when a lab calibration pack is resolvable (AC_QUALITY_DEFAULTS)."""
    _p = os.environ.get("AC_QUALITY_DEFAULTS")
    return bool(_p) and os.path.exists(_p)


# Wave 19 (P1-4): maximum relative loss uncertainty the tiebreak band may be
# derived from when NO calibration pack is loaded. Uncalibrated uncertainty
# is ±8%, which made the "quality-equivalent" bucket ~2% of loss — at 20T
# that is months of training compute silently spent on a throughput tiebreak.
# 2% × the 0.25 band factor caps the pre-calibration spend at ~0.5%.
_UNCALIBRATED_TIEBREAK_UNC_CAP = 0.02


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
    strict = bool(getattr(constraints, "strict_quality", False)) if constraints else False

    # Wave 33: declared serving budgets are BOUNDS — a candidate that
    # violates --serving-tbt / --serving-ttft must lose to any candidate
    # that meets them, regardless of loss. Budgets stay soft at the
    # feasibility layer (extreme corners degrade to a marked best-effort
    # answer instead of an empty one), so the bound is enforced here at
    # ranking time. If NO candidate in the pool meets the budgets the
    # bucket is constant and ranking proceeds among violators unchanged.
    _budgets_declared = constraints is not None and (
        getattr(constraints, "serving_tbt_ms", None) is not None
        or getattr(constraints, "serving_ttft_ms", None) is not None)
    if _budgets_declared:
        _any_ok = any(_meets_declared_budgets(x, constraints)
                      for x in scoring_pool)
        if _any_ok:
            def _budget_bucket(x) -> int:
                return 0 if _meets_declared_budgets(x, constraints) else 1
        else:
            def _budget_bucket(x) -> int:
                return 0
    else:
        def _budget_bucket(x) -> int:
            return 0
    _any_no_spill = any(
        getattr(getattr(x, "throughput", None), "spill_tier", "fits")
        == "fits"
        for x in scoring_pool
    )

    def _spill_bucket(x) -> int:
        if not _any_no_spill:
            return 0
        tier = getattr(
            getattr(x, "throughput", None), "spill_tier", "fits"
        )
        return 0 if tier == "fits" else 1

    if profile_name in ("research_quality", "loss_only"):
        pool_size = len(scoring_pool)
        # Wave 18h --strict-quality: no noise-band bucketing. Rank by the
        # shape-prior-adjusted loss point estimate; throughput/memory only
        # breaks EXACT loss ties.
        if strict:
            def _strict_loss_key(x):
                prior = _aspect_ratio_prior_penalty(x, pool_size=pool_size)
                adj_loss = x.predicted_loss * (1.0 + prior)
                return (
                    _budget_bucket(x),
                    round(adj_loss, 9),
                    _spill_bucket(x),
                    _evaluated_serving_instance_gpus(x),
                    x.memory_per_gpu_gb,
                    x.serving_tbt_ms,
                    -_evaluated_training_tps_per_gpu(x),
                    float(x.arch.total_params),
                )
            return _strict_loss_key
        TIEBREAK_K_LOSS = 0.25
        _pool_best = min(x.predicted_loss for x in scoring_pool)
        _u_sorted = sorted(
            float(getattr(x.quality, "uncertainty_total", 0.0) or 0.0)
            for x in scoring_pool
        )
        _u_med = _u_sorted[len(_u_sorted) // 2] if _u_sorted else 0.0
        # Wave 19 (P1-4): pre-calibration, the ±8% uncertainty is a fiction
        # for tiebreak purposes — cap the band input so rank-1 can spend at
        # most ~0.5% predicted loss on throughput/memory. A loaded pack's
        # fitted uncertainty governs unchanged.
        if not _quality_pack_loaded():
            _u_med = min(_u_med, _UNCALIBRATED_TIEBREAK_UNC_CAP)
        _BAND = max(_u_med, 0.005) * max(_pool_best, 0.01) * TIEBREAK_K_LOSS

        # Wave 23: bucket memory into a coarse band before it gates
        # TBT/-TPS. The prior sort tuple keyed on raw memory_per_gpu_gb,
        # so two loss-equivalent candidates would tiebreak on a
        # sub-display memory difference (e.g. 8.2771 GB vs 8.2806 GB —
        # both display as "8.3 GB") — throwing away meaningful decode-TBT
        # and training-TPS wins to save ~0.04% memory. The band is
        # max(0.1 GB, 2% of the pool's max memory), which is well below
        # anything a user reads off the report but large enough to absorb
        # arithmetic drift between candidates that would display the same
        # rounded value.
        _pool_mem_max = max(
            (float(x.memory_per_gpu_gb) for x in scoring_pool
             if math.isfinite(float(x.memory_per_gpu_gb))),
            default=0.0,
        )
        _MEM_BAND = max(0.1, 0.02 * _pool_mem_max)

        def _mem_bucket(x) -> int:
            m = float(x.memory_per_gpu_gb)
            if not math.isfinite(m) or _MEM_BAND <= 0:
                return 0
            return int(m / _MEM_BAND)

        def _loss_key(x):
            prior = _aspect_ratio_prior_penalty(x, pool_size=pool_size)
            adj_loss = x.predicted_loss * (1.0 + prior)
            bucket = (
                max(0, int((adj_loss - _pool_best) / _BAND))
                if _BAND > 0 else 0
            )
            return (
                _budget_bucket(x),
                bucket,
                _spill_bucket(x),
                _evaluated_serving_instance_gpus(x),
                _mem_bucket(x),
                x.serving_tbt_ms,
                -_evaluated_training_tps_per_gpu(x),
                x.memory_per_gpu_gb,
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
    # Wave 19 (P1-4): same pre-calibration cap as the loss-first profiles.
    if not _quality_pack_loaded():
        _u_median = min(_u_median, _UNCALIBRATED_TIEBREAK_UNC_CAP)
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
        if strict:
            # Wave 18h --strict-quality: exact score/loss ordering, no bands.
            return (
                _budget_bucket(x),
                round(_pool_scores.get(id(x), 0.0), 12),
                x.predicted_loss,
                _spill_bucket(x),
                _evaluated_serving_instance_gpus(x),
                x.memory_per_gpu_gb,
                x.serving_tbt_ms,
                x.throughput.prefill_time_ms,
                -_evaluated_training_tps_per_gpu(x),
            )
        return (
            _budget_bucket(x),
            _score_bucket(x),
            _quality_bucket(x),
            _spill_bucket(x),
            _evaluated_serving_instance_gpus(x),
            # Within a quality bucket, prefer faster, smaller, cheaper.
            x.memory_per_gpu_gb,
            x.serving_tbt_ms,
            x.throughput.prefill_time_ms,
            -_evaluated_training_tps_per_gpu(x),
            # Deterministic final tiebreak on raw loss.
            x.predicted_loss,
        )

    return _general_key


def _same_candidate(a: EvaluatedCandidate, b: EvaluatedCandidate) -> bool:
    return architecture_fingerprint(a.arch) == architecture_fingerprint(b.arch)


def _candidate_summary(ev: EvaluatedCandidate) -> Dict[str, Any]:
    c = ev.arch
    return {
        "d_model": c.d_model,
        "n_layers": c.n_layers,
        "n_heads": c.n_heads,
        "d_head": c.d_head,
        "n_kv_heads": c.n_kv_heads,
        "attention_type": c.attention_type,
        "weight_precision": c.weight_precision,
        "ffn_precision": c.ffn_precision,
        "activation_precision": c.activation_precision,
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
    "ffn_dim", "weight_precision", "ffn_precision", "activation_precision",
    "kv_cache_bits", "attention_type",
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
        "weight_precision": getattr(arch, "weight_precision", None),
        "ffn_precision": getattr(arch, "ffn_precision", None),
        "activation_precision": getattr(arch, "activation_precision", None),
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
    replica_gpus = _evaluated_training_replica_gpus(ev)
    tps_per_gpu = _evaluated_training_tps_per_gpu(ev)
    sig_tps = (
        float(getattr(t, "training_throughput_sigma_tps", 0.0) or 0.0)
        / replica_gpus
    )
    sig_tbt = float(getattr(t, "decode_time_sigma_ms", 0.0) or 0.0)
    sig_pre = float(getattr(t, "prefill_time_sigma_ms", 0.0) or 0.0)
    return {
        "training_tps": (tps_per_gpu - k * sig_tps,
                          tps_per_gpu + k * sig_tps),
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


def _collect_contenders(
    result: OptimizationResult, opt: EvaluatedCandidate,
) -> Tuple[List[EvaluatedCandidate], Dict[int, str], Optional[float]]:
    """Wave 27: single source of truth for the contending-family test.

    Pre-Wave-27 the paired-sigma rule (added in Wave 18h) only lived
    inside `_confidence_envelope`. `compute_contending_family_full`
    (the sidecar) and `justification.render_arch_justification`'s
    markdown table both re-derived contenders with the older naïve
    `_loss_interval` overlap. On the default H100 7B run that put the
    three "same idea" surfaces at three different counts — CLI warning
    said 9, markdown said 34, sidecar row_count said 157 — and the
    author's own comment on the markdown claimed it was the tightest
    view when in fact it was the widest. Fold the paired-sigma rule
    into a shared helper so every surface reports the same number and
    the same short list of contenders.

    Returns (contenders, reasons, best_other_low). `best_other_low`
    is the tightest ev_low across contenders, used by the envelope's
    `best_contender_loss_low` field; callers that don't need it can
    ignore.
    """
    opt_low, opt_high = _loss_interval(opt)
    try:
        from ac.quality_model import paired_loss_uncertainty as _paired_unc
    except Exception:  # pragma: no cover - package layout fallback
        try:
            from quality_model import paired_loss_uncertainty as _paired_unc
        except Exception:
            _paired_unc = None

    def _paired_sigma_abs(ev):
        """Paired decision sigma vs the optimum, or None for legacy path."""
        if _paired_unc is None:
            return None
        qa = getattr(opt, "quality", None)
        qb = getattr(ev, "quality", None)
        if not (getattr(qa, "terms", None) and getattr(qb, "terms", None)):
            return None
        try:
            p = _paired_unc(qa, qb)
        except Exception:
            return None
        if not p.get("enabled", True):
            return None
        return float(p["sigma_abs"])

    def _is_loss_contender(ev) -> bool:
        sig = _paired_sigma_abs(ev)
        if sig is None:
            low, _high = _loss_interval(ev)
            return low <= opt_high and ev.predicted_loss <= opt_high
        delta = abs(float(ev.predicted_loss) - float(opt.predicted_loss))
        return delta <= sig

    eligible = [
        ev for ev in result.all_evaluated
        if getattr(ev, "meets_constraints", False)
    ]
    eligible = _prefer_training_fit(eligible, result.hardware)
    contenders: List[EvaluatedCandidate] = []
    contender_reasons: Dict[int, str] = {}
    best_other_low: Optional[float] = None
    for ev in eligible:
        if ev is opt:
            continue
        low, _high = _loss_interval(ev)
        loss_contender = _is_loss_contender(ev)
        # Throughput-side tie: candidate is within ±1σ of opt on any of
        # {TPS, TBT, prefill}. Lets the contending-family view reflect
        # the fact that the predicted throughput numbers carry ±20%
        # uncertainty that the deterministic sort previously ignored.
        # Wave 18h: the throughput-tie loss gate also uses the paired
        # sigma. Previously it admitted anything within opt's FULL +8%
        # CI (opt_high x 1.005) that tied on a throughput axis, which
        # kept the contender count pinned at the pre-pairing level.
        _sig = _paired_sigma_abs(ev)
        _thr_loss_gate = (
            ev.predicted_loss <= opt_high * 1.005 if _sig is None
            else abs(float(ev.predicted_loss) - float(opt.predicted_loss))
                 <= max(_sig, 0.005 * float(opt.predicted_loss)))
        thr_contender = (loss_contender or _thr_loss_gate) and \
                         _is_throughput_contender(opt, ev, k=1.0)
        if loss_contender or thr_contender:
            contenders.append(ev)
            if best_other_low is None or low < best_other_low:
                best_other_low = low
            if loss_contender and thr_contender:
                contender_reasons[id(ev)] = "both"
            elif loss_contender:
                contender_reasons[id(ev)] = "loss"
            else:
                contender_reasons[id(ev)] = "throughput"
    return contenders, contender_reasons, best_other_low


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
    contenders, contender_reasons, _ = _collect_contenders(result, opt)
    return _contending_family(
        opt, contenders, top_n=top_n, contender_reasons=contender_reasons,
    )


def _confidence_envelope(result: OptimizationResult, opt: EvaluatedCandidate) -> Dict[str, Any]:
    opt_low, opt_high = _loss_interval(opt)
    contenders, contender_reasons, best_other_low = _collect_contenders(result, opt)
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
    family = _contending_family(opt, contenders,
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
    # Ranks in the emitted diagnostics must use the same display ordering as
    # pareto.csv and the picker. Iterating the optimizer's discovery-order
    # frontier made JSON say selected_pareto_rank=2 while CSV correctly put
    # selected=True at rank=1.
    if pareto and result.constraints:
        ranked_pareto = sorted(
            pareto, key=build_display_sort_key(pareto, result.constraints)
        )
    else:
        ranked_pareto = list(pareto)
    selected_rank = None
    best_loss_rank = None
    for idx, ev in enumerate(ranked_pareto, start=1):
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
    """Convert the evaluated optimal candidate to a matching JSON config.

    ``nsa``/``yoco`` kwargs are retained for API compatibility but may not
    introduce unevaluated transforms. Candidate fields are authoritative.
    """
    if result.optimal is None:
        return None

    opt = result.optimal
    c = opt.arch
    evaluated_nsa = None
    if c.attention_type == "nsa":
        evaluated_nsa = {
            "compress_block_size": int(c.nsa_compress_block_size),
            "compress_block_stride": int(c.nsa_compress_block_stride),
            "select_block_size": int(c.nsa_select_block_size),
            "select_top_k": int(c.nsa_select_top_k),
            "window_size": int(c.nsa_window_size),
        }
    evaluated_yoco = None
    if c.yoco_n_self_attn_layers > 0:
        evaluated_yoco = {
            "enabled": True,
            "n_self_attn_layers": int(c.yoco_n_self_attn_layers),
            "share_pattern": str(c.yoco_share_pattern),
        }
    if nsa is not None and evaluated_nsa is None:
        raise ValueError(
            "Cannot stamp NSA onto an unevaluated result; set "
            "DeploymentConstraints(force_nsa=True)"
        )
    if nsa is not None and evaluated_nsa is not None and nsa != evaluated_nsa:
        raise ValueError(
            "Requested NSA block differs from the evaluated candidate"
        )
    if yoco is not None and evaluated_yoco is None:
        raise ValueError(
            "Cannot stamp YOCO onto an unevaluated result; set "
            "DeploymentConstraints(force_yoco=True)"
        )
    if yoco is not None and evaluated_yoco is not None and yoco != evaluated_yoco:
        raise ValueError(
            "Requested YOCO block differs from the evaluated candidate"
        )
    nsa = evaluated_nsa
    yoco = evaluated_yoco
    # Wave 35: emit the evaluated compressed/indexer attention block.
    # Before this, a csa/indexshare/msa winner was emitted as
    # attention.type="full" — the config a user would train did not match
    # the candidate the search evaluated (same bug class as the Wave 19
    # local:global finding).
    evaluated_compressed = None
    if c.attention_type == "csa":
        evaluated_compressed = {
            "type": "csa",
            "csa_block_size": int(c.csa_block_size or 64),
            "csa_top_k_blocks": int(c.csa_top_k_blocks or 16),
            "csa_compression_dim": int(c.csa_compression_dim or 64),
        }
    elif c.attention_type == "indexshare":
        evaluated_compressed = {
            "type": "indexshare",
            "indexshare_num_buckets": int(c.indexshare_num_buckets or 64),
            "indexshare_top_k_buckets": int(c.indexshare_top_k_buckets or 4),
            "indexshare_index_dim": int(c.indexshare_index_dim or 64),
        }
    elif c.attention_type == "msa":
        evaluated_compressed = {
            "type": "msa",
            "msa_window_size": int(c.msa_window_size or 512),
            "msa_dilated_top_k": int(c.msa_dilated_top_k or 64),
            "msa_global_top_k": int(c.msa_global_top_k or 16),
        }
    terms = getattr(opt.quality, "terms", {})
    arch_term = terms.get("architecture_residual")
    precision_term = terms.get("precision_residual")
    risk_term = terms.get("risk_residual")
    effective_capacity_term = terms.get("effective_capacity")
    effective_data_term = terms.get("effective_data")
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
    # Wave 4: when TP was a search variable, the constraints carried a list
    # `tp_options` and the *picked* TP lives on the winning candidate. Surface
    # the candidate's tp_degree as the cell's tp; expose tp_options as a
    # sibling field so a downstream reviewer can see the search space the
    # optimizer was given.
    _tp_for_emit = int(getattr(c, "tp_degree", 0) or 0) or int(result.constraints.tp)
    _pp_for_emit = max(1, int(getattr(c, "pp_degree", 1) or 1))
    _dp_for_emit = max(
        1,
        int(getattr(c, "dp_degree", 0) or 0)
        or int(result.constraints.dp),
    )
    input_constraints = {
        "hardware": result.hardware,
        "tp": _tp_for_emit,
        "tp_options": list(result.constraints.tp_options or [result.constraints.tp]),
        "pp": _pp_for_emit,
        "pp_options": list(result.constraints.pp_options or [result.constraints.pp]),
        "dp": _dp_for_emit,
        "training_cluster_gpus": result.constraints.training_cluster_gpus,
        "max_training_cluster_gpus": (
            result.constraints.max_training_cluster_gpus
        ),
        "ep": int(getattr(c, "ep_degree", 1) or 1),
        "ep_options": list(result.constraints.ep_options or []),
        "cp": int(getattr(c, "cp_degree", 1) or 1),
        "cp_options": list(result.constraints.cp_options or [result.constraints.cp]),
        "cp_method": getattr(result.constraints, "cp_method", "ring"),
        "target_params": f"{result.constraints.target_params_b}B",
        "training_tokens": f"{result.constraints.training_tokens / 1e12:.1f}T",
        "unique_training_tokens": (
            f"{result.constraints.unique_training_tokens / 1e12:.1f}T"
            if result.constraints.unique_training_tokens is not None
            else None
        ),
        "pretraining_context_length": result.constraints.pretraining_context_length,
        "quality_model_version": result.constraints.quality_model_version,
        "context_length": result.constraints.context_length,
        "serving_tbt_ms": result.constraints.serving_tbt_ms,
        "serving_ttft_ms": result.constraints.serving_ttft_ms,
        "serving_batch": result.constraints.serving_batch,
        "prompt_len": result.constraints.prompt_len,
        "output_len": result.constraints.output_len,
        "scheduler": result.constraints.scheduler,
        "concurrency": result.constraints.concurrency,
        "training_micro_batch": result.constraints.training_micro_batch or 8,
        "pipeline_microbatches": result.constraints.pipeline_microbatches,
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
    #
    # v1-fix Wave 1 Step 1.2 (Jun 2026): per-replica TPS now already nets
    # out the DP gradient reduce-scatter + weight all-gather cost (folded
    # into train_step_s in throughput_model.py). That makes the linear
    # `× dp_degree` aggregation below honest — at dp=1024 the per-replica
    # TPS is materially lower than at dp=1 for the same shape, so the
    # cluster number isn't inflated.
    dp_degree = _dp_for_emit
    pp_degree = max(1, _pp_for_emit)
    # Wave 4: TP is per-candidate. Fall back to constraints.tp for
    # back-compat when the candidate didn't record one.
    tp_degree = max(1, int(getattr(c, "tp_degree", 0) or 0)
                       or int(getattr(result.constraints, "tp", 1) or 1))
    cp_degree = max(1, int(getattr(c, "cp_degree", 1) or 1))
    per_replica_tps = round(opt.training_tps)
    # Aggregate scales linearly with DP. PP pipelines a single replica, so it
    # doesn't multiply throughput; we still expose pp for the reader.
    aggregate_tps = round(opt.training_tps * dp_degree)
    # GPU counts. A "replica" is one TP × PP × CP group; the cluster has DP
    # such replicas. Per-GPU TPS is the only number that's comparable across
    # different parallelism layouts — without it, a CP=4 run looks "4×
    # faster" than CP=1 just because the replica grew 4× in GPUs.
    #
    # Wave 19 (P0-1): EP is NOT part of the training replica. In training,
    # EP lays over the DP dimension — every EP rank processes its own
    # microbatch (the throughput model books M × top_k routed tokens per
    # rank accordingly) — so counting ep into gpus_per_replica divided MoE
    # per-GPU throughput by an extra ep× (the "MoE 20× slower than dense"
    # release-review finding). A SERVING instance does span tp×pp×cp×ep
    # GPUs sharing one batch; that count is exposed separately.
    ep_degree = max(1, int(getattr(c, "ep_degree", 1) or 1))
    gpus_per_replica = tp_degree * pp_degree * cp_degree
    serving_instance_gpus = gpus_per_replica * ep_degree
    total_gpus = gpus_per_replica * dp_degree
    per_gpu_tps = round(opt.training_tps / max(1, gpus_per_replica))

    predicted = {
        "evaluated_architecture_hash": architecture_fingerprint(c),
        "quality_rank_score": round(-opt.predicted_loss, 4),
        "predicted_loss": round(opt.predicted_loss, 4),
        # Per-TP-replica throughput. Same value across DP — DP scales the
        # aggregate, not the per-replica rate. The per-GPU number is the
        # apples-to-apples comparison across parallelism choices.
        "training_throughput_tokens_per_sec": per_replica_tps,
        "training_throughput_tokens_per_sec_per_replica": per_replica_tps,
        "training_throughput_tokens_per_sec_per_gpu": per_gpu_tps,
        "aggregate_training_throughput_tokens_per_sec": aggregate_tps,
        "training_throughput_tokens_per_sec_per_serving_gpu": round(
            opt.training_tps / max(1, serving_instance_gpus)
        ),
        "serving_instance_gpus": serving_instance_gpus,
        "training_throughput_unit": (
            f"tokens/sec per TP×PP×CP training replica "
            f"({tp_degree}×{pp_degree}×{cp_degree}"
            f" = {gpus_per_replica} GPUs/replica; EP={ep_degree} lays over "
            f"DP and does not add replica GPUs); per-GPU = {per_gpu_tps} "
            f"tok/s; aggregate over DP={dp_degree} replicas "
            f"({total_gpus} total GPUs) = {aggregate_tps} tok/s; a serving "
            f"instance spans {serving_instance_gpus} GPUs (TP×PP×CP×EP)"
        ),
        "serving_tbt_ms": round(opt.serving_tbt_ms, 1),
        "serving_ttft_ms": round(opt.throughput.prefill_time_ms, 1),
        "serving_request_latency_ms": round(
            opt.serving_request_latency_ms, 1
        ),
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
        "training_memory_per_gpu_gb": round(
            float(getattr(opt.throughput, "training_memory_per_gpu_gb", 0.0)),
            1,
        ),
        "parallelism_costs": {
            "tp_allreduce_per_layer_ms": round(
                1000.0 * float(
                    getattr(opt.throughput.per_layer_breakdown, "allreduce_s", 0.0)
                    if opt.throughput.per_layer_breakdown else 0.0
                ),
                4,
            ),
            "ep_alltoall_per_layer_ms": round(
                1000.0 * float(
                    getattr(opt.throughput.per_layer_breakdown, "alltoall_s", 0.0)
                    if opt.throughput.per_layer_breakdown else 0.0
                ),
                4,
            ),
            "dp_gradient_sync_ms": round(
                1000.0 * float(
                    getattr(opt.throughput, "dp_grad_allreduce_s", 0.0)
                ),
                4,
            ),
            "pp_training_transfer_ms": round(
                1000.0 * float(
                    getattr(opt.throughput, "pp_training_comm_s", 0.0)
                ),
                4,
            ),
            "pp_prefill_transfer_ms": round(
                1000.0 * float(
                    getattr(opt.throughput, "pp_prefill_comm_s", 0.0)
                ),
                4,
            ),
            "pp_decode_transfer_ms": round(
                1000.0 * float(
                    getattr(opt.throughput, "pp_decode_comm_s", 0.0)
                ),
                4,
            ),
            "pipeline_bubble_fraction": round(
                float(getattr(opt.throughput, "bubble_fraction", 0.0)),
                6,
            ),
        },
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
        "pretraining_loss_proxy": round(
            float(getattr(opt.quality, "pretraining_loss_proxy", opt.predicted_loss)),
            6,
        ),
        "task_adjusted_loss_proxy": round(
            float(getattr(opt.quality, "task_adjusted_loss_proxy", opt.predicted_loss)),
            6,
        ),
        "effective_capacity_delta": round(
            float(getattr(effective_capacity_term, "delta", 0.0)), 8
        ),
        "effective_data_delta": round(
            float(getattr(effective_data_term, "delta", 0.0)), 8
        ),
        "effective_params_b": round(
            float(getattr(opt.quality, "spine_effective_params", 0)) / 1e9,
            6,
        ),
        "effective_training_tokens_t": round(
            float(getattr(opt.quality, "training_tokens", 0)) / 1e12,
            6,
        ),
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
        "candidates_evaluated": result.candidates_evaluated,
        "evaluation_failures": int(
            getattr(result, "evaluation_failures", 0) or 0),
        "evaluation_failure_reasons": dict(
            getattr(result, "evaluation_failure_reasons", {}) or {}),
        "candidates_enumerated_raw": result.candidates_enumerated_raw,
        "candidates_feasible": result.candidates_feasible,
        "pareto_size": len(result.pareto_frontier),
        "search_time_sec": result.search_time_sec,
    }

    # Build feature blocks before choosing the dense vs hybrid schema path.
    # Both paths must serialize the exact architecture that was evaluated.
    mla_kw = None
    if c.attention_type == "mla":
        mla_kw = {
            "kv_latent_dim": c.mla_kv_latent_dim,
            "q_latent_dim": c.mla_q_latent_dim,
            "rope_head_dim": c.mla_rope_head_dim,
            "nope_head_dim": c.mla_nope_head_dim,
        }
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
    local_global_kw = (
        {
            "n_local_layers": int(
                getattr(c, "n_local_attn_layers", 0) or 0),
            "window_size": int(getattr(c, "swa_window", 0) or 0),
        }
        if (int(getattr(c, "n_local_attn_layers", 0) or 0) > 0
            and int(getattr(c, "swa_window", 0) or 0) > 0)
        else None
    )
    swa_kw = (
        {"window_size": int(getattr(c, "swa_window", 0) or 0)}
        if (int(getattr(c, "swa_window", 0) or 0) > 0
            and int(getattr(c, "n_local_attn_layers", 0) or 0) == 0)
        else None
    )

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
            state_expansion=c.state_config.get("state_expansion", 2),
            state_n_heads=c.state_config.get("n_heads", c.n_heads),
            state_d_head=c.state_config.get("d_head", c.d_head),
            state_precision=c.state_config.get("state_precision", "bf16"),
            placement_strategy=c.placement_strategy,
            hybrid_ratio=c.hybrid_ratio,
            ffn_dim=c.ffn_dim,
            ffn_precision=c.ffn_precision,
            weight_precision=c.weight_precision,
            activation_precision=c.activation_precision,
            moe=c.moe,
            tp=tp_degree,
            pp=pp_degree,
            dp=dp_degree,
            ep=c.ep_degree,
            cp=cp_degree,
            cp_method=str(c.cp_method),
            rope_scaling_method=str(c.rope_scaling_method),
            rope_scaling_factor=float(c.rope_scaling_factor),
            rope_original_max_position=int(c.rope_original_max_position),
            hardware_name=result.hardware,
            input_constraints=input_constraints,
            predicted=predicted,
            search_stats=search_stats,
            n_dense_ffn_layers=c.n_dense_ffn_layers,
            mla=mla_kw,
            swa=swa_kw,
            nsa=nsa,
            compressed=evaluated_compressed,
            yoco=yoco,
            mtp=mtp_kw,
            local_global=local_global_kw,
        )

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
        activation_precision=c.activation_precision,
        kv_cache_bits=c.kv_cache_bits,
        tp=tp_degree,
        pp=pp_degree,
        dp=dp_degree,
        ep=c.ep_degree,
        cp=int(c.cp_degree),
        cp_method=str(c.cp_method),
        rope_scaling_method=str(c.rope_scaling_method),
        rope_scaling_factor=float(c.rope_scaling_factor),
        rope_original_max_position=int(c.rope_original_max_position),
        moe=c.moe,
        n_dense_ffn_layers=c.n_dense_ffn_layers,
        mla=mla_kw,
        swa=swa_kw,
        nsa=nsa,
        compressed=evaluated_compressed,
        yoco=yoco,
        mtp=mtp_kw,
        # Wave 19 (L1): emit the searched local:global interleave. Without
        # this the emitted config silently trained pure full attention.
        local_global=local_global_kw,
        hardware_name=result.hardware,
        input_constraints=input_constraints,
        predicted=predicted,
        search_stats=search_stats,
    )


def result_to_pareto_csv(result: OptimizationResult) -> str:
    """Convert the Pareto frontier to CSV format. v1 adds MoE columns; dense
    rows leave them blank/0 so existing parsers degrade cleanly."""
    # Wave 4: surface per-candidate tp so two pareto rows that differ only by
    # the picked TP are distinguishable in the emitted CSV.
    lines = [
        "rank,selected,objective_profile,objective_score,d_model,n_layers,n_heads,d_head,n_kv_heads,ffn_dim,"
        "weight_prec,ffn_prec,activation_prec,kv_bits,active_params_B,total_params_B,architecture_family,"
        "attention_type,mla_kv_latent,mla_q_latent,"
        "nsa_select_top_k,nsa_window,csa_block,csa_top_k,csa_dim,"
        "indexshare_buckets,indexshare_top_k,indexshare_dim,"
        "msa_window,msa_dilated_top_k,msa_global_top_k,"
        "yoco_self_attn_layers,yoco_share_pattern,"
        "cp,cp_method,rope_method,rope_factor,mtp_depth,tp,pp,dp,"
        "state_layers,attention_layers,placement_strategy,"
        "local_attn_layers,local_window,"
        "moe_style,n_experts,top_k,expert_dim,ep,vocab_size,"
        "predicted_loss,loss_ci_low,loss_ci_high,uncertainty_total_pct,"
        "training_tps,training_tps_per_gpu,serving_tbt_ms,serving_ttft_ms,"
        "training_memory_gb,memory_gb,confidence"
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
        if c.moe is not None and c.state_config is not None:
            architecture_family = "moe_hybrid"
        elif c.moe is not None:
            architecture_family = "moe"
        elif c.state_config is not None:
            architecture_family = "hybrid"
        else:
            architecture_family = "dense"
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
            f"{c.ffn_dim},{c.weight_precision},{c.ffn_precision},"
            f"{c.activation_precision},{c.kv_cache_bits},"
            f"{active},{c.total_params_b},{architecture_family},"
            f"{c.attention_type},{c.mla_kv_latent_dim},{c.mla_q_latent_dim},"
            f"{int(getattr(c, 'nsa_select_top_k', 0) or 0)},"
            f"{int(getattr(c, 'nsa_window_size', 0) or 0)},"
            f"{int(getattr(c, 'csa_block_size', 0) or 0)},"
            f"{int(getattr(c, 'csa_top_k_blocks', 0) or 0)},"
            f"{int(getattr(c, 'csa_compression_dim', 0) or 0)},"
            f"{int(getattr(c, 'indexshare_num_buckets', 0) or 0)},"
            f"{int(getattr(c, 'indexshare_top_k_buckets', 0) or 0)},"
            f"{int(getattr(c, 'indexshare_index_dim', 0) or 0)},"
            f"{int(getattr(c, 'msa_window_size', 0) or 0)},"
            f"{int(getattr(c, 'msa_dilated_top_k', 0) or 0)},"
            f"{int(getattr(c, 'msa_global_top_k', 0) or 0)},"
            f"{int(getattr(c, 'yoco_n_self_attn_layers', 0) or 0)},"
            f"{str(getattr(c, 'yoco_share_pattern', '') or '')},"
            f"{c.cp_degree},{c.cp_method},{c.rope_scaling_method},"
            f"{c.rope_scaling_factor:.4f},{c.mtp_n_predict_depths},"
            f"{int(getattr(c, 'tp_degree', 1) or 1)},"
            f"{int(getattr(c, 'pp_degree', 1) or 1)},"
            f"{int(getattr(c, 'dp_degree', 0) or 0) or int(getattr(result.constraints, 'dp', 1) or 1)},"
            f"{c.n_state_layers},{(c.n_layers if c.state_config is None and c.n_attention_layers == 0 else c.n_attention_layers)},{c.placement_strategy},"
            f"{int(getattr(c, 'n_local_attn_layers', 0) or 0)},"
            f"{int(getattr(c, 'swa_window', 0) or 0)},"
            f"{c.moe_style},{n_experts},{top_k},{expert_dim},{c.ep_degree},"
            f"{int(getattr(c, 'vocab_size', 0) or 0)},"
            f"{ev.predicted_loss:.4f},{loss_low:.4f},{loss_high:.4f},"
            f"{getattr(ev.quality, 'uncertainty_total', 0.0) * 100:.2f},"
            f"{ev.training_tps:.0f},"
            # Wave 19 (P0-1): per-GPU over the TRAINING replica (tp×pp×cp;
            # EP lays over DP) — the only cross-family-comparable number.
            f"{ev.training_tps / max(1, (int(getattr(c, 'tp_degree', 1) or 1) * max(1, int(getattr(c, 'pp_degree', 1) or 1)) * max(1, int(getattr(c, 'cp_degree', 1) or 1)))):.0f},"
            f"{ev.serving_tbt_ms:.1f},{ev.throughput.prefill_time_ms:.1f},"
            f"{float(getattr(ev.throughput, 'training_memory_per_gpu_gb', 0.0) or 0.0):.1f},"
            f"{ev.memory_per_gpu_gb:.1f},{ev.quality.confidence}"
        )
    return "\n".join(lines)
