"""
Penalty Table v0 — Published-ablation quality corrections

Each penalty function returns a fractional relative loss increase (e.g., 0.005 = 0.5%).
The quality model multiplies these by the Chinchilla baseline and sums them.

Every penalty has:
  - A source citation
  - The model scale at which it was measured
  - Known caveats
  - A hardware argument where the penalty is hardware-conditional

Convention: return 0.0 for no penalty, a positive float for a quality cost,
None if the configuration is infeasible on the target hardware (optimizer skips).
"""

import math
from typing import Optional


# =============================================================================
# Shape penalty
# =============================================================================
# v1-fix refit (June 2026): re-fitted against Llama-2-{7B,13B,70B},
# Llama-3-{8B,70B}, Mistral-7B, Qwen3-{8B,32B}, GPT-3-175B by minimizing
# log-residuals on (d_model, n_layers) vs N_active. The previous coefficients
# (from "Chinchilla appendix (approximate)") put d_opt(7B)=7945 and
# l_opt(7B)=76, which biased the optimizer toward deep-narrow architectures
# that no practitioner builds. The refit values put d_opt(7B)≈4060 and
# l_opt(7B)≈32, matching Llama-3-8B / Mistral-7B / Qwen3-8B.
#
# The constants below are the *defaults*. The optimizer / quality model
# always reads from a constants dict (loaded from
# configs/quality/quality_v1_defaults.yaml) so future re-fits can be applied
# by editing the YAML — no code change required. To refit:
#
#   1. Collect (N_active, d_model, n_layers) pairs from a target model
#      family.
#   2. Run a two-line linear regression on log(N) vs log(d_model) for K_W,
#      gamma_W; same for n_layers → K_D, gamma_D.
#   3. Update `architecture_residual.shape_law.{K_W, gamma_W, K_D, gamma_D}`
#      in `configs/quality/quality_v1_defaults.yaml`.
#
# The module-level SHAPE_* constants below are the in-code fallback when no
# YAML is loaded; they should track the YAML.

# Optimal shape coefficients — v1-fix refit
# d_model_optimal ≈ K_W × N^γ_W
# n_layers_optimal ≈ K_D × N^γ_D
SHAPE_K_W = 1.80      # width coefficient
SHAPE_GAMMA_W = 0.341  # width exponent
SHAPE_K_D = 0.014     # depth coefficient
SHAPE_GAMMA_D = 0.341  # depth exponent

# Penalty strength: previously 0.03 (calibrated so 2× deviation ≈ 2% PPL),
# but the v1 demo audit caught the optimizer picking pathological shapes
# (d_model=6144 with L=15 at 7B params) when the SLO got tight — the
# 2% penalty was easily eaten by the wide-shallow throughput win.
# Re-calibrated to 0.05, consistent with Tay et al. 2021 ("Scale
# Efficiently") and Levine et al. 2020 figures of ~5-8% PPL at 0.5×
# depth — i.e. wide-shallow degeneracy hurts more than the v0 fit said.
# Wave 12 follow-up (Jun 2026): bumped from 0.05 → 0.08 after calibration
# validation showed reference architectures stay under 1.3% penalty while
# pathological shapes (e.g. 11776×4) move from 27% → 43%. The bump
# strengthens shape_penalty's role as the canonical "pathological shapes
# lose on the Pareto frontier" mechanism; the picker prior in
# optimizer._aspect_ratio_prior_penalty is now scoped to tiebreaks among
# Pareto-equivalent survivors only (cap reduced from 25% to 10%).
SHAPE_C = 0.08

# Lower bounds (sanity floors)
SHAPE_D_MIN = 256
SHAPE_L_MIN = 2


def shape_penalty(
    d_model: int,
    n_layers: int,
    n_active_params: int,
    constants: Optional[dict] = None,
) -> float:
    """
    Quadratic penalty for deviating from the empirically-fit optimal
    width/depth aspect ratio at a given parameter count.

    `constants` accepts the `architecture_residual.shape_law` block (a dict
    with keys K_W, gamma_W, K_D, gamma_D, C, d_min, l_min). When omitted,
    the module-level SHAPE_* defaults are used. The optimizer should pass
    `constants` from the loaded YAML so a refit doesn't require code edits.

    Source for v1-fix refit (June 2026): regression on practitioner dense
    architectures Llama-2/3, Mistral, Qwen3, GPT-3 (see file header).
    """
    if n_active_params <= 0 or d_model <= 0 or n_layers <= 0:
        return 0.0

    if constants is None:
        constants = {}
    K_W = float(constants.get("K_W", SHAPE_K_W))
    gamma_W = float(constants.get("gamma_W", SHAPE_GAMMA_W))
    K_D = float(constants.get("K_D", SHAPE_K_D))
    gamma_D = float(constants.get("gamma_D", SHAPE_GAMMA_D))
    C = float(constants.get("C", SHAPE_C))
    d_min = float(constants.get("d_min", SHAPE_D_MIN))
    l_min = float(constants.get("l_min", SHAPE_L_MIN))

    d_opt = K_W * (n_active_params ** gamma_W)
    l_opt = K_D * (n_active_params ** gamma_D)

    # Clamp optimal values to a sensible floor
    d_opt = max(d_opt, d_min)
    l_opt = max(l_opt, l_min)

    log_d_ratio = math.log(d_model / d_opt)
    log_l_ratio = math.log(n_layers / l_opt)

    return C * (log_d_ratio ** 2 + log_l_ratio ** 2)


# =============================================================================
# GQA penalty
# =============================================================================
# Source: Touvron et al. "Llama 2" (2023), GQA ablation table (Section 3.2);
#         Jiang et al. "Mistral 7B" (2023) — uses GQA-8, no reported degradation.
# Measured at: 7B–70B parameter range
# Caveat: ablations used Meta's training recipe (AdamW, cosine LR, 2T tokens).
#         Transfer to other recipes is plausible but unvalidated.

def gqa_penalty(
    n_heads: int,
    n_kv_heads: int,
    d_model: int,
) -> float:
    """
    Quality cost of grouped-query attention relative to MHA.

    GQA-8 at d_model ≥ 2048 is within seed variance per Llama-2 ablations.
    MQA (n_kv_heads=1) has small but measurable cost.

    Source: Touvron et al. (2023) Llama-2, GQA ablation; Jiang et al. (2023) Mistral 7B.
    """
    if n_kv_heads >= n_heads:
        return 0.0  # MHA

    ratio = n_heads / n_kv_heads

    if ratio <= 8 and d_model >= 2048:
        # GQA-8 at reasonable width: within seed variance
        return 0.0
    elif ratio <= 8:
        # GQA-8 at narrow width: small cost
        return 0.002
    elif ratio <= 16:
        # GQA-16: moderate cost
        return 0.005
    else:
        # MQA or extreme grouping
        return 0.008


# =============================================================================
# KV cache quantization penalty
# =============================================================================
# Source: Hooper et al. "KIVI: A Tuning-Free Asymmetric 2bit Quantization for
#         KV Cache" (2024); Liu et al. "KIVI" NeurIPS 2024.
# Measured at: 7B–70B on H100 (per-channel/per-token INT4 KV)
# Caveat: KIVI numbers measured on GPU with custom Triton kernels.
#         TPU per-channel KV scaling is less mature — conservative penalty.
#         INT2 KV is experimental; penalty is an extrapolation.

def kv_quant_penalty(
    kv_bits: int,
    has_per_channel_scaling: bool = True,
    hardware: str = "h100",
) -> Optional[float]:
    """
    Quality cost of KV cache quantization.

    INT8 KV is within noise on all hardware.
    INT4 KV with per-channel scaling (KIVI recipe) costs ~1% on GPU, ~2% on TPU.
    INT4 without per-channel scaling costs ~3%.
    INT2 is experimental and not recommended.

    Source: Hooper et al. (2024) KIVI.
    Returns None if the configuration is infeasible.
    """
    if kv_bits >= 16:
        return 0.0
    elif kv_bits == 8:
        return 0.0  # INT8 KV within noise on all hardware
    elif kv_bits == 4:
        if has_per_channel_scaling:
            if hardware in ("h100", "b200"):
                return 0.010  # KIVI INT4 mid-range: 0.5-1.5% PPL
            elif hardware in ("tpu_v5p", "tpu_v5e"):
                return 0.020  # per-channel scaling less mature on TPU
            else:
                return 0.015  # unknown hardware, conservative
        else:
            return 0.030  # naive INT4 much worse
    elif kv_bits <= 3:
        return 0.050  # INT2/INT3: experimental, large cost
    else:
        return 0.0


# =============================================================================
# Wave 10B (Jun 2026): hw-blind quality functions + hw-conditional feasibility
# =============================================================================
# Per plan/redesign/10-optimizer-self-consistency.md Change B: quality should
# not depend on hardware once precision is feasible. Split the historical
# kv_quant_penalty / weight_precision_penalty into:
#   - `_precision_supported(prec, hw)` — feasibility check (hw-conditional).
#   - `kv_quant_quality(...)` — quality penalty, hw-blind.
#   - `weight_precision_quality(...)` — quality penalty, hw-blind.
#
# Bug D fix: same arch + same precision should produce identical predicted_loss
# across H100 / B200 / TPU, because quality is an arch+data function, not a hw
# function. The hw-conditional KV / FP8-on-TPU uplifts that used to live in
# the quality penalty now live in feasibility; if a hw doesn't support a
# precision, `_precision_supported` returns False and the candidate is culled.


def precision_supported(precision: str, hardware: str = "h100") -> bool:
    """Wave 10B: True iff `precision` is natively supported on `hardware`.

    Pure hw-conditional check; no quality term. Quality residuals call the
    hw-blind `*_quality` variants below; feasibility code culls candidates
    where this returns False.

    Wave 21: this is the NATIVE-COMPUTE check (tensor-core / MXU datapath).
    Weight STORAGE formats have looser requirements — see
    `weight_storage_supported`.
    """
    if precision in ("bf16", "fp16"):
        return True
    return hardware in PRECISION_HARDWARE_SUPPORT.get(precision, set())


# Wave 21: weight-only quantized STORAGE formats are deployable on any
# supported hardware — the weights are dequantized to a native compute type
# at the register/shared-memory boundary (GPT-OSS mxfp4 experts served on
# H100 via bf16-compute dequant kernels; GPTQ/AWQ int4 on Ampere+; fp8
# weight-only on pre-Hopper parts). The old single support table conflated
# storage with compute and marked e.g. mxfp4-weights-on-H100 INFEASIBLE,
# which sentineled the quality model on real, shipping deployments (the
# gpt-oss-120b trust anchor). The throughput model already prices
# non-native formats correctly by construction: `peak_flops_s` falls back
# to the bf16 compute rate while `bytes_per_elem` charges the reduced
# storage bytes — exactly weight-only-quant serving physics.
WEIGHT_STORAGE_UNIVERSAL = {"fp8", "fp4", "mxfp4", "mxfp6", "int8", "int4"}


def weight_storage_supported(precision: str, hardware: str = "h100") -> bool:
    """True iff `precision` is a legal WEIGHT-STORAGE format on `hardware`.

    Weight-only quantization needs no native tensor-core datapath, so all
    narrow float/int storage formats are feasible everywhere; anything
    else falls back to the native-compute table.
    """
    if precision in ("bf16", "fp16"):
        return True
    if precision in WEIGHT_STORAGE_UNIVERSAL:
        return True
    return hardware in PRECISION_HARDWARE_SUPPORT.get(precision, set())


def kv_quant_quality(kv_bits: int, has_per_channel_scaling: bool = True) -> float:
    """Wave 10B: hw-blind KV quantization quality penalty.

    Uses the H100/B200 KIVI numbers as canonical (these are the most
    rigorously measured; the historic TPU 2× uplift was a precaution under
    "per-channel scaling less mature on TPU", which is a feasibility
    concern, not a quality concern). Same arch + same precision now gives
    the same loss across hardware.
    """
    if kv_bits >= 8:
        return 0.0
    if kv_bits == 4:
        return 0.010 if has_per_channel_scaling else 0.030
    if kv_bits <= 3:
        return 0.050
    return 0.0


def weight_precision_quality(component: str, precision: str) -> float:
    """Wave 10B: hw-blind weight-precision quality penalty.

    Uses the canonical per-component table; drops the historic 1.5× TPU
    FP8 uplift (which was a feasibility / "less mature" hedge, not a real
    PPL delta on the same training run).
    """
    if precision in ("bf16", "fp16"):
        return 0.0
    key = (component, precision)
    p = WEIGHT_PRECISION_PENALTIES.get(key)
    if p is not None:
        return p
    if precision == "fp8":
        return 0.005
    if precision == "fp4":
        return 0.015
    return 0.005


def activation_precision_quality(component: str, precision: str) -> float:
    """Wave 10B: hw-blind activation-precision quality penalty."""
    if precision in ("bf16", "fp16"):
        return 0.0
    return ACTIVATION_PRECISION_PENALTIES.get((component, precision), 0.003)


# =============================================================================
# Weight precision penalty (per component, per hardware)
# =============================================================================
# Source: Peng et al. "FP8-LM: Training FP8 Large Language Models" (2023);
#         NVIDIA Transformer Engine documentation (2024);
#         NVIDIA Blackwell MXFP4 launch materials (2025-2026);
#         Early MXFP4 training papers (late 2025 / 2026).
# Measured at: 7B–175B for FP8; 7B–70B for FP4 (sparse data)
# Caveat: FP4 penalties are LOW-CONFIDENCE. Derived from early MXFP4 literature
#         where per-component breakdowns are sparse. v4 measured profiling
#         will produce reliable FP4 numbers.

# Component sensitivity ranking (lower = more tolerant):
# FFN > output_proj > qkv_proj > output_head > embedding

WEIGHT_PRECISION_PENALTIES = {
    # (component, precision) -> fractional penalty
    # FP8 (E4M3 weights / E5M2 gradients)
    ("ffn_up", "fp8"):      0.001,   # FFN tolerates FP8 very well
    ("ffn_down", "fp8"):    0.001,
    ("ffn_gate", "fp8"):    0.001,
    ("qkv_proj", "fp8"):    0.003,   # attention weights slightly more sensitive
    ("output_proj", "fp8"): 0.003,
    ("output_head", "fp8"): 0.005,   # final projection somewhat sensitive
    ("embedding", "fp8"):   0.010,   # embedding is sensitive; usually kept higher

    # FP4 (MXFP4 with microscaling) — B200 only
    ("ffn_up", "fp4"):      0.005,   # FFN tolerates FP4 with microscaling
    ("ffn_down", "fp4"):    0.005,
    ("ffn_gate", "fp4"):    0.005,
    ("qkv_proj", "fp4"):    0.012,   # attention more sensitive to FP4
    ("output_proj", "fp4"): 0.012,
    ("output_head", "fp4"): 0.020,   # final projection FP4-sensitive
    ("embedding", "fp4"):   0.030,   # embedding rarely run at FP4
}

# Hardware availability for each precision
PRECISION_HARDWARE_SUPPORT = {
    "bf16": {"h100", "b200", "tpu_v5e", "tpu_v5p", "trainium2", "trn2", "trainium3", "trn3"},
    "fp16": {"h100", "b200", "tpu_v5e", "tpu_v5p", "trainium2", "trn2", "trainium3", "trn3"},
    "fp8":  {"h100", "b200", "trainium2", "trn2", "trainium3", "trn3"},
    "fp4":  {"b200", "trainium3", "trn3"},
    # v1-fix microscaling: OCP MX formats. Blackwell + Trainium 3.
    "mxfp4": {"b200", "trainium3", "trn3"},
    "mxfp6": {"b200", "trainium3", "trn3"},
    "int8": {"h100", "b200", "tpu_v5e", "tpu_v5p", "trainium2", "trn2", "trainium3", "trn3"},
}


def weight_precision_penalty(
    component: str,
    precision: str,
    hardware: str = "h100",
) -> Optional[float]:
    """
    Quality cost of running a specific component at reduced precision.

    Returns None if the precision is not supported on the target hardware
    (optimizer should skip this candidate).

    Source: Peng et al. (2023) FP8-LM; NVIDIA Transformer Engine docs;
            NVIDIA MXFP4 launch materials (2025-2026).
    Caveat: FP4 penalties are low-confidence (early literature, sparse data).
    """
    # BF16/FP16: zero penalty everywhere
    if precision in ("bf16", "fp16"):
        return 0.0

    # Check hardware support
    supported_hw = PRECISION_HARDWARE_SUPPORT.get(precision, set())
    if hardware not in supported_hw:
        return None  # not available; optimizer skips

    # FP8 on TPU: partial support, add extra uncertainty
    if precision == "fp8" and hardware in ("tpu_v5p", "tpu_v5e"):
        # TPU v5p has partial FP8; treat as low-confidence with higher penalty
        base = WEIGHT_PRECISION_PENALTIES.get((component, "fp8"), 0.005)
        return base * 1.5  # 50% penalty uplift for TPU FP8 uncertainty

    # Lookup
    key = (component, precision)
    penalty = WEIGHT_PRECISION_PENALTIES.get(key)
    if penalty is None:
        # Unknown component — use conservative default
        if precision == "fp8":
            return 0.005
        elif precision == "fp4":
            return 0.015
        else:
            return 0.005

    return penalty


# =============================================================================
# Activation precision penalty
# =============================================================================
# Source: Peng et al. (2023) FP8-LM; NVIDIA Transformer Engine docs.
# Measured at: 7B–175B
# Caveat: activations are recomputed each forward pass and don't compound
#         across training, so penalties are smaller than weight penalties.

ACTIVATION_PRECISION_PENALTIES = {
    ("attention", "fp8"):   0.002,
    ("ffn", "fp8"):         0.001,
    ("attention", "fp4"):   0.008,  # FP4 activations newer, less validated
    ("ffn", "fp4"):         0.004,
}


def activation_precision_penalty(
    component: str,
    precision: str,
    hardware: str = "h100",
) -> Optional[float]:
    """
    Quality cost of reduced-precision activations.
    Smaller than weight penalties since activations are recomputed each pass.

    Source: Peng et al. (2023) FP8-LM.
    """
    if precision in ("bf16", "fp16"):
        return 0.0

    supported_hw = PRECISION_HARDWARE_SUPPORT.get(precision, set())
    if hardware not in supported_hw:
        return None

    key = (component, precision)
    return ACTIVATION_PRECISION_PENALTIES.get(key, 0.003)


# =============================================================================
# Feasibility penalty
# =============================================================================
# Not from a paper — this is a hard constraint.
# Returns +inf if the architecture is infeasible.

INFEASIBLE = 1e6  # effectively +∞ for the optimizer


def feasibility_penalty(
    d_head: int,
    memory_fits: bool = True,
    lattice_aligned: bool = True,
) -> float:
    """
    Hard penalty for infeasible architectures.

    Checks:
      - d_head within FlashAttention range [32, 256]
      - Memory footprint fits in HBM (flag from throughput model)
      - Dimensions are lattice-aligned (defensive check)

    Not from a paper — hard constraint for the optimizer.
    """
    if not memory_fits:
        return INFEASIBLE
    if not lattice_aligned:
        return INFEASIBLE
    if d_head < 32 or d_head > 256:
        return INFEASIBLE
    return 0.0


# =============================================================================
# Summary of all penalties (for model card / documentation)
# =============================================================================

PENALTY_REGISTRY = {
    "shape": {
        "function": "shape_penalty",
        "source": "Tay et al. 'Scale Efficiently' (2021); Hoffmann et al. (2022) Appendix",
        "measured_scale": "100M-10B params",
        "caveat": "Parametric form is heuristic (quadratic in log-space). "
                  "Coefficient hand-set for 2× deviation ≈ 2% PPL. Weakest penalty in table.",
        "hardware_dependent": False,
    },
    "gqa": {
        "function": "gqa_penalty",
        "source": "Touvron et al. 'Llama 2' (2023), GQA ablation; Jiang et al. 'Mistral 7B' (2023)",
        "measured_scale": "7B-70B params",
        "caveat": "Ablations used Meta's training recipe. Transfer to other recipes unvalidated.",
        "hardware_dependent": False,
    },
    "kv_quant": {
        "function": "kv_quant_penalty",
        "source": "Hooper et al. 'KIVI' (2024), NeurIPS 2024",
        "measured_scale": "7B-70B on H100",
        "caveat": "GPU numbers measured with custom Triton kernels. "
                  "TPU per-channel scaling less mature — conservative penalty applied.",
        "hardware_dependent": True,
    },
    "weight_precision": {
        "function": "weight_precision_penalty",
        "source": "Peng et al. 'FP8-LM' (2023); NVIDIA Transformer Engine docs; "
                  "NVIDIA MXFP4 launch materials (2025-2026)",
        "measured_scale": "7B-175B for FP8; 7B-70B for FP4 (sparse data)",
        "caveat": "FP4 penalties LOW-CONFIDENCE. Early MXFP4 literature, sparse per-component data. "
                  "v4 measured profiling will replace these.",
        "hardware_dependent": True,
    },
    "activation_precision": {
        "function": "activation_precision_penalty",
        "source": "Peng et al. 'FP8-LM' (2023); NVIDIA Transformer Engine docs",
        "measured_scale": "7B-175B",
        "caveat": "Activations recomputed each pass; penalties smaller than weight penalties.",
        "hardware_dependent": True,
    },
    "feasibility": {
        "function": "feasibility_penalty",
        "source": "Hard constraint (not from a paper)",
        "measured_scale": "N/A",
        "caveat": "Returns +inf for infeasible architectures.",
        "hardware_dependent": False,
    },
}
