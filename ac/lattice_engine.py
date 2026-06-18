"""
Tile-Aligned Architecture Lattice Engine

Computes the discrete set of efficient architecture dimensions for
hardware-aware neural network design across NVIDIA H100, NVIDIA B200,
and Google TPU v5e at each precision level and tensor parallelism degree.
"""

import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

# =============================================================================
# Hardware Specifications
# =============================================================================

@dataclass
class TileSpec:
    """Tensor core / MXU tile specification for a given precision."""
    precision: str
    # Instruction-level tile (m, n, k)
    inst_m: int
    inst_n: int
    inst_k: int
    # CTA-level tile (m, n, k) — the practical alignment unit
    cta_m: int
    cta_n: int
    cta_k: int

@dataclass
class HardwareSpec:
    """Full hardware platform specification."""
    name: str
    short_name: str
    n_sms: int  # Number of SMs (NVIDIA) or cores (TPU)
    sram_per_sm_bytes: int  # Shared memory per SM
    tiles: Dict[str, TileSpec]  # precision -> TileSpec


# --- NVIDIA H100 SXM ---
H100 = HardwareSpec(
    name="NVIDIA H100 SXM",
    short_name="h100",
    n_sms=132,
    sram_per_sm_bytes=228 * 1024,  # 228 KB shared memory per SM
    tiles={
        "tf32": TileSpec("tf32", 16, 8, 8, 64, 64, 8),
        "bf16": TileSpec("bf16", 16, 8, 16, 64, 64, 64),
        "fp8":  TileSpec("fp8",  16, 8, 32, 64, 64, 128),
        "int8": TileSpec("int8", 16, 8, 32, 64, 64, 128),
    }
)

# --- NVIDIA B200 (Blackwell) ---
B200 = HardwareSpec(
    name="NVIDIA B200",
    short_name="b200",
    n_sms=160,  # estimated
    sram_per_sm_bytes=256 * 1024,  # estimated 256 KB
    tiles={
        "bf16": TileSpec("bf16", 16, 8, 16, 128, 128, 64),
        "fp8":  TileSpec("fp8",  16, 8, 32, 128, 128, 128),
        "int8": TileSpec("int8", 16, 8, 32, 128, 128, 128),
        "fp4":  TileSpec("fp4",  16, 8, 64, 128, 128, 256),
    }
)

# --- Google TPU v5e ---
TPU_V5E = HardwareSpec(
    name="Google TPU v5e",
    short_name="tpu_v5e",
    n_sms=1,  # TPU uses MXU, not SMs — wave quantization N/A in same way
    sram_per_sm_bytes=32 * 1024 * 1024,  # 32 MB HBM per chip used differently
    tiles={
        "bf16": TileSpec("bf16", 128, 128, 128, 128, 128, 128),
        "int8": TileSpec("int8", 128, 128, 128, 128, 128, 128),
    }
)

# --- Google TPU v5p ---
TPU_V5P = HardwareSpec(
    name="Google TPU v5p",
    short_name="tpu_v5p",
    n_sms=1,  # TPU uses MXU, not SMs — wave quantization N/A in same way
    sram_per_sm_bytes=95 * 1024 * 1024,  # 95 GB HBM3 per chip (used as VMEM proxy)
    tiles={
        "bf16": TileSpec("bf16", 128, 128, 128, 128, 128, 128),
        "int8": TileSpec("int8", 128, 128, 128, 128, 128, 128),
    }
)

# v1-fix Trainium: AWS Trainium 2 / Trainium 3. NCv3/v4 are systolic-array
# matmul units with 128×128 dispatch tile, closer to TPU MXU than NVIDIA
# wmma. Trn3 adds an FP4 path matching B200's tile shape.
TRAINIUM2 = HardwareSpec(
    name="AWS Trainium 2",
    short_name="trainium2",
    n_sms=8,
    sram_per_sm_bytes=96 * 1024 * 1024,
    tiles={
        "bf16": TileSpec("bf16", 128, 128, 64, 128, 128, 64),
        "fp8":  TileSpec("fp8",  128, 128, 128, 128, 128, 128),
        "int8": TileSpec("int8", 128, 128, 128, 128, 128, 128),
    }
)

TRAINIUM3 = HardwareSpec(
    name="AWS Trainium 3",
    short_name="trainium3",
    n_sms=16,
    sram_per_sm_bytes=192 * 1024 * 1024,
    tiles={
        "bf16": TileSpec("bf16", 128, 128, 64, 128, 128, 64),
        "fp8":  TileSpec("fp8",  128, 128, 128, 128, 128, 128),
        "fp4":  TileSpec("fp4",  128, 128, 256, 128, 128, 256),
        "int8": TileSpec("int8", 128, 128, 128, 128, 128, 128),
    }
)

HARDWARE = {
    "h100": H100, "b200": B200,
    "tpu_v5e": TPU_V5E, "tpu_v5p": TPU_V5P,
    "trainium2": TRAINIUM2, "trn2": TRAINIUM2,
    "trainium3": TRAINIUM3, "trn3": TRAINIUM3,
}

# =============================================================================
# Alignment Utilities
# =============================================================================

def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)

def round_up_to(value: int, multiple: int) -> int:
    """Round up to nearest multiple."""
    return ((value + multiple - 1) // multiple) * multiple

def round_nearest_to(value: int, multiple: int) -> int:
    """Round to nearest multiple."""
    lower = (value // multiple) * multiple
    upper = lower + multiple
    return lower if (value - lower) <= (upper - value) else upper

def tile_utilization(dim: int, tile: int) -> float:
    """Compute utilization for a single dimension against a tile size."""
    if dim <= 0:
        return 0.0
    n_tiles = math.ceil(dim / tile)
    return dim / (n_tiles * tile)

def matmul_tile_utilization(M: int, N: int, K: int, tile: TileSpec) -> float:
    """Compute tile utilization for a matmul shape."""
    util_m = tile_utilization(M, tile.cta_m)
    util_n = tile_utilization(N, tile.cta_n)
    util_k = tile_utilization(K, tile.cta_k)
    return util_m * util_n * util_k

def wave_efficiency(M: int, N: int, tile: TileSpec, n_sms: int) -> float:
    """Compute wave quantization efficiency."""
    if n_sms <= 1:
        return 1.0  # TPU doesn't have SM-based wave quantization
    n_tiles_m = math.ceil(M / tile.cta_m)
    n_tiles_n = math.ceil(N / tile.cta_n)
    total_tiles = n_tiles_m * n_tiles_n
    if total_tiles == 0:
        return 0.0
    n_waves = math.ceil(total_tiles / n_sms)
    return total_tiles / (n_waves * n_sms)

# =============================================================================
# Lattice Computation
# =============================================================================

@dataclass
class LatticePoint:
    """A single point on the tile-aligned architecture lattice."""
    d_model: int
    d_head: int
    n_heads: int
    ffn_dim_swiglu: int
    ffn_dim_dense: int
    tile_aligned: bool
    tile_util_qkv: float
    tile_util_attn: float
    tile_util_ffn: float
    tile_util_harmonic: float
    wave_eff_2048: float
    wave_eff_8192: float

@dataclass
class GQAConfig:
    """A valid GQA configuration."""
    d_model: int
    d_head: int
    n_heads: int
    n_kv_heads: int
    gqa_ratio: int  # n_heads / n_kv_heads
    kv_proj_dim_per_gpu: int
    tile_aligned: bool

@dataclass
class MoEConfig:
    """A valid MoE expert dimension."""
    d_model: int
    n_experts: int
    expert_ffn_dim: int
    total_ffn_equivalent: int
    tile_aligned: bool

@dataclass
class StateConfig:
    """A valid state mechanism dimension."""
    d_state: int
    d_head: int
    tile_aligned: bool
    sram_bytes: int  # per-head state size in bytes


def get_alignment_stride(tile: TileSpec, tp: int) -> dict:
    """Get the alignment stride for each dimension given tile spec and TP."""
    return {
        "d_model_k": lcm(tile.cta_k, tp),      # d_model as K dimension
        "d_model_n": lcm(tile.cta_n, tp),      # d_model as N dimension
        "d_model":   lcm(lcm(tile.cta_k, tile.cta_n), tp),  # both
        "n_dim":     tile.cta_n,                 # generic N alignment
        "k_dim":     tile.cta_k,                 # generic K alignment
        "heads_tp":  tp,                         # n_heads divisible by TP
    }


def compute_lattice(
    hw: HardwareSpec,
    precision: str,
    tp: int,
    d_model_min: int = 1024,
    d_model_max: int = 16384,
    d_head_options: List[int] = None,
) -> List[LatticePoint]:
    """
    Compute the tile-aligned architecture lattice for a given
    (hardware, precision, TP_degree) combination.
    """
    if precision not in hw.tiles:
        return []

    tile = hw.tiles[precision]
    if d_head_options is None:
        d_head_options = [32, 64, 96, 128, 256]

    # d_model must be aligned to both K and N tile dims, and divisible by TP
    d_model_stride = lcm(lcm(tile.cta_k, tile.cta_n), tp)

    results = []

    d_model = round_up_to(d_model_min, d_model_stride)
    while d_model <= d_model_max:
        for d_head in d_head_options:
            # d_head must be tile-aligned in both K and N
            if d_head % tile.cta_k != 0 or d_head % tile.cta_n != 0:
                # Relax to instruction-level alignment for d_head
                if d_head % tile.inst_k != 0 or d_head % tile.inst_n != 0:
                    continue
                # Flag as instruction-aligned but not CTA-aligned
                cta_aligned_dhead = False
            else:
                cta_aligned_dhead = True

            # n_heads = d_model / d_head
            if d_model % d_head != 0:
                continue
            n_heads = d_model // d_head

            # n_heads must be divisible by TP
            if n_heads % tp != 0:
                continue

            heads_per_gpu = n_heads // tp

            # QKV projection: N = 3 * d_head * heads_per_gpu
            qkv_n = 3 * d_head * heads_per_gpu
            qkv_aligned = (qkv_n % tile.cta_n == 0)

            # Attention: d_head as K and N
            attn_k_aligned = (d_head % tile.cta_k == 0)
            attn_n_aligned = (d_head % tile.cta_n == 0)

            # Output projection: K = d_head * heads_per_gpu
            out_k = d_head * heads_per_gpu
            out_k_aligned = (out_k % tile.cta_k == 0)

            # FFN dim (SwiGLU: 8/3 ratio)
            ffn_stride = lcm(tile.cta_n, tp)
            ffn_raw_swiglu = int(d_model * 8 / 3)
            ffn_dim_swiglu = round_nearest_to(ffn_raw_swiglu, ffn_stride)
            if ffn_dim_swiglu < ffn_stride:
                ffn_dim_swiglu = ffn_stride

            # FFN dim (dense: 4x ratio)
            ffn_raw_dense = d_model * 4
            ffn_dim_dense = round_nearest_to(ffn_raw_dense, ffn_stride)
            if ffn_dim_dense < ffn_stride:
                ffn_dim_dense = ffn_stride

            # FFN alignment checks
            ffn_per_gpu_swiglu = ffn_dim_swiglu // tp
            ffn_per_gpu_dense = ffn_dim_dense // tp
            ffn_n_aligned = (ffn_per_gpu_swiglu % tile.cta_n == 0)
            ffn_k_aligned = (ffn_per_gpu_swiglu % tile.cta_k == 0)

            # Overall tile alignment
            all_aligned = all([
                d_model % tile.cta_k == 0,
                d_model % tile.cta_n == 0,
                qkv_aligned,
                attn_k_aligned or (d_head % tile.inst_k == 0),
                attn_n_aligned or (d_head % tile.inst_n == 0),
                out_k_aligned,
                ffn_n_aligned,
                ffn_k_aligned,
            ])

            # Compute utilization scores
            M_2048 = 2048   # batch*seq = 2048
            M_8192 = 8192

            # QKV matmul util
            util_qkv = matmul_tile_utilization(M_2048, qkv_n, d_model, tile)

            # Attention score util (batch*heads is M, but d_head is K and seq is N)
            # For scoring, use d_head alignment as proxy
            util_attn_k = tile_utilization(d_head, tile.cta_k)
            util_attn_n = tile_utilization(d_head, tile.cta_n)
            util_attn = util_attn_k * util_attn_n

            # FFN util (using SwiGLU dim)
            util_ffn_up = matmul_tile_utilization(M_2048, ffn_per_gpu_swiglu, d_model, tile)
            util_ffn_down = matmul_tile_utilization(M_2048, d_model, ffn_per_gpu_swiglu, tile)
            util_ffn = (util_ffn_up * util_ffn_down) ** 0.5

            # Harmonic mean of utilizations
            utils = [u for u in [util_qkv, util_attn, util_ffn] if u > 0]
            if utils:
                harmonic = len(utils) / sum(1/u for u in utils)
            else:
                harmonic = 0.0

            # Wave efficiency
            wave_2048 = wave_efficiency(M_2048, ffn_per_gpu_swiglu, tile, hw.n_sms)
            wave_8192 = wave_efficiency(M_8192, ffn_per_gpu_swiglu, tile, hw.n_sms)

            results.append(LatticePoint(
                d_model=d_model,
                d_head=d_head,
                n_heads=n_heads,
                ffn_dim_swiglu=ffn_dim_swiglu,
                ffn_dim_dense=ffn_dim_dense,
                tile_aligned=all_aligned,
                tile_util_qkv=round(util_qkv, 4),
                tile_util_attn=round(util_attn, 4),
                tile_util_ffn=round(util_ffn, 4),
                tile_util_harmonic=round(harmonic, 4),
                wave_eff_2048=round(wave_2048, 4),
                wave_eff_8192=round(wave_8192, 4),
            ))

        d_model += d_model_stride

    return results


def compute_gqa_configs(
    hw: HardwareSpec,
    precision: str,
    tp: int,
    d_model: int,
    d_head: int,
    n_heads: int,
) -> List[GQAConfig]:
    """Compute valid GQA (n_heads, n_kv_heads) pairs for a lattice point."""
    if precision not in hw.tiles:
        return []

    tile = hw.tiles[precision]
    results = []

    for n_kv_heads in range(1, n_heads + 1):
        # n_heads must be divisible by n_kv_heads (even group sizes)
        if n_heads % n_kv_heads != 0:
            continue

        # n_kv_heads must be divisible by TP (or handle n_kv_heads < TP specially)
        if n_kv_heads >= tp:
            if n_kv_heads % tp != 0:
                continue
            kv_heads_per_gpu = n_kv_heads // tp
        else:
            # n_kv_heads < TP: each GPU gets a replicated copy (valid in some frameworks)
            kv_heads_per_gpu = 1

        # KV projection dim per GPU: 2 * d_head * kv_heads_per_gpu
        kv_proj_per_gpu = 2 * d_head * kv_heads_per_gpu
        kv_aligned = (kv_proj_per_gpu % tile.cta_n == 0) or (kv_proj_per_gpu % tile.inst_n == 0)

        gqa_ratio = n_heads // n_kv_heads

        results.append(GQAConfig(
            d_model=d_model,
            d_head=d_head,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            gqa_ratio=gqa_ratio,
            kv_proj_dim_per_gpu=kv_proj_per_gpu,
            tile_aligned=kv_aligned,
        ))

    return results


# =============================================================================
# v1 MoE option enumeration
# =============================================================================

# Default n_experts / top_k axes searched by the v1 optimizer. Picked to span
# the published design space: Mixtral (8/2), Llama-4-style (16/2), and the
# DeepSeek-V2/V3 fine-grained regime (64..256, top_k up to 8).
DEFAULT_N_EXPERTS = [8, 16, 32, 64, 128, 256]
DEFAULT_TOP_K = [1, 2, 4, 8]


# Default expert-parallel options per hardware. NVLink "domain size" determines
# the largest EP that still talks over NVLink — beyond that, all-to-all
# traverses inter-node fabric and the throughput model applies a large penalty.
#   H100  — 8-GPU NVLink island (DGX H100 / HGX H100)
#   B200  — NVL72 rack-scale NVLink fabric: up to 72 GPUs at full BW
#   TPU v5p — 3D torus ICI, EP along a single torus axis; cap at 16 to keep
#             the cross-axis traversal heuristic well-defined
#   TPU v5e — small chip count per host; EP up to 8
NVLINK_DOMAIN = {
    "h100": 8,
    "b200": 72,
    "tpu_v5p": 16,   # treated as "axis" rather than NVLink, but same role
    "tpu_v5e": 8,
}


def default_ep_options(hw_name: str) -> List[int]:
    """Powers of 2 up to the hardware's NVLink/axis domain size."""
    cap = NVLINK_DOMAIN.get(hw_name, 8)
    opts = [1]
    e = 2
    while e <= cap:
        opts.append(e)
        e *= 2
    # B200 NVL72 has a non-power-of-two cap that's worth exposing as well.
    if cap not in opts:
        opts.append(cap)
    return opts


@dataclass
class MoEOption:
    """A single MoE configuration choice for one (lattice point, EP) combination.

    style: 'coarse' (Mixtral-like) | 'fine' (DeepSeek-like)
    expert_dim is rounded to the precision's CTA-N tile and to the EP degree
    (so each rank holds n_experts/ep complete experts).
    shared_dim is the always-on expert's FFN width or None.
    """
    d_model: int
    n_experts: int
    top_k: int
    expert_dim: int
    shared_dim: Optional[int]
    style: str          # 'coarse' | 'fine'
    ep_degree: int
    precision: str
    tile_aligned: bool
    # Active vs total FFN-equivalent widths (for downstream param accounting).
    active_ffn_equivalent: int    # = top_k * expert_dim + shared_dim (if any)
    total_ffn_equivalent: int     # = n_experts * expert_dim + shared_dim (if any)


def compute_moe_options(
    hw: HardwareSpec,
    precision: str,
    d_model: int,
    baseline_ffn_dim: int,
    ep_degrees: Optional[List[int]] = None,
    n_experts_options: Optional[List[int]] = None,
    top_k_options: Optional[List[int]] = None,
    granularity_targets: Tuple[float, ...] = (1.0, 0.5, 0.25),
) -> List[MoEOption]:
    """Enumerate MoE configurations for a given (lattice point, hardware) pair.

    Two `style` shapes are emitted per `(n_experts, top_k, ep)`:

      - coarse: expert_dim ≈ baseline_ffn_dim, shared_dim = None  (Mixtral)
      - fine:   expert_dim ≈ baseline_ffn_dim × top_k / n_experts × g,
                shared_dim ≈ baseline_ffn_dim / 4                  (DeepSeek)

    The fine-grained scale factor `g ∈ granularity_targets` controls how much
    active capacity the MoE has relative to the dense FFN it replaces. The
    default sweep (1.0, 0.5, 0.25) covers iso-active, half-active, and
    quarter-active regimes.

    Pruning rules:
      - n_experts % ep_degree == 0
      - top_k <= n_experts
      - expert_dim aligned to the precision's CTA-N tile
      - expert_dim >= cta_n (no degenerate single-tile experts)
      - coarse style is only emitted when n_experts >= 4 (Mixtral floor)
      - fine style is only emitted when n_experts / top_k >= 8 (Krajewski
        granularity regime where the law is reasonably calibrated)
    """
    if precision not in hw.tiles:
        return []
    tile = hw.tiles[precision]

    if ep_degrees is None:
        ep_degrees = default_ep_options(hw.short_name)
    if n_experts_options is None:
        n_experts_options = DEFAULT_N_EXPERTS
    if top_k_options is None:
        top_k_options = DEFAULT_TOP_K

    out: List[MoEOption] = []
    seen = set()  # dedupe by (n_experts, top_k, expert_dim, shared_dim, ep, style)

    for ep in ep_degrees:
        for n_experts in n_experts_options:
            if n_experts % ep != 0:
                continue
            for top_k in top_k_options:
                if top_k > n_experts:
                    continue

                # --- Coarse (Mixtral-style) ---
                # expert_dim ≈ baseline_ffn_dim, tile-aligned.
                if n_experts >= 4:
                    expert_dim = round_nearest_to(baseline_ffn_dim, tile.cta_n)
                    if expert_dim < tile.cta_n:
                        expert_dim = tile.cta_n
                    aligned = (expert_dim % tile.cta_n == 0)
                    key = (n_experts, top_k, expert_dim, None, ep, "coarse")
                    if key not in seen:
                        seen.add(key)
                        out.append(MoEOption(
                            d_model=d_model,
                            n_experts=n_experts,
                            top_k=top_k,
                            expert_dim=expert_dim,
                            shared_dim=None,
                            style="coarse",
                            ep_degree=ep,
                            precision=precision,
                            tile_aligned=aligned,
                            active_ffn_equivalent=top_k * expert_dim,
                            total_ffn_equivalent=n_experts * expert_dim,
                        ))

                # --- Fine-grained (DeepSeek-style) ---
                # Only emit when the granularity ratio is in the calibrated
                # regime (n_experts / top_k >= 8 is the Krajewski floor).
                if n_experts / top_k < 8:
                    continue

                shared_raw = baseline_ffn_dim // 4
                shared_dim = round_nearest_to(shared_raw, tile.cta_n)
                if shared_dim < tile.cta_n:
                    shared_dim = tile.cta_n

                for g in granularity_targets:
                    # Active FFN-equivalent ≈ g × baseline_ffn_dim (minus shared)
                    target_active = max(0, int(g * baseline_ffn_dim) - shared_dim)
                    if target_active <= 0:
                        continue
                    expert_dim_raw = target_active // top_k
                    expert_dim = round_nearest_to(expert_dim_raw, tile.cta_n)
                    if expert_dim < tile.cta_n:
                        expert_dim = tile.cta_n
                    aligned = (expert_dim % tile.cta_n == 0)

                    key = (n_experts, top_k, expert_dim, shared_dim, ep, "fine")
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(MoEOption(
                        d_model=d_model,
                        n_experts=n_experts,
                        top_k=top_k,
                        expert_dim=expert_dim,
                        shared_dim=shared_dim,
                        style="fine",
                        ep_degree=ep,
                        precision=precision,
                        tile_aligned=aligned,
                        active_ffn_equivalent=top_k * expert_dim + shared_dim,
                        total_ffn_equivalent=n_experts * expert_dim + shared_dim,
                    ))

    return out


def compute_moe_configs(
    hw: HardwareSpec,
    precision: str,
    tp: int,
    d_model: int,
    expert_counts: List[int] = None,
) -> List[MoEConfig]:
    """Compute valid MoE expert FFN dimensions for a lattice point."""
    if precision not in hw.tiles:
        return []

    tile = hw.tiles[precision]
    if expert_counts is None:
        expert_counts = [8, 16, 64, 128]

    ffn_stride = lcm(tile.cta_n, tp)
    results = []

    for n_experts in expert_counts:
        # Typical: expert_ffn_dim ≈ (d_model * 8/3) / n_experts (SwiGLU shared capacity)
        raw_expert_ffn = int(d_model * 8 / 3 / n_experts)
        if raw_expert_ffn < tile.cta_n:
            raw_expert_ffn = tile.cta_n

        expert_ffn = round_nearest_to(raw_expert_ffn, tile.cta_n)
        if expert_ffn < tile.cta_n:
            expert_ffn = tile.cta_n

        aligned = (expert_ffn % tile.cta_n == 0)
        total_equiv = expert_ffn * n_experts

        results.append(MoEConfig(
            d_model=d_model,
            n_experts=n_experts,
            expert_ffn_dim=expert_ffn,
            total_ffn_equivalent=total_equiv,
            tile_aligned=aligned,
        ))

    return results


def compute_state_configs(
    hw: HardwareSpec,
    precision: str,
    d_head_options: List[int] = None,
    max_d_state: int = 512,
    bytes_per_element: int = 2,  # BF16
) -> List[StateConfig]:
    """Compute valid d_state values for state mechanisms."""
    if precision not in hw.tiles:
        return []

    tile = hw.tiles[precision]
    if d_head_options is None:
        d_head_options = [64, 128]

    if precision == "fp8":
        bytes_per_element = 1
    elif precision == "int8":
        bytes_per_element = 1
    elif precision in ("bf16", "tf32"):
        bytes_per_element = 2
    elif precision == "fp4":
        bytes_per_element = 1  # approximate

    results = []
    d_state = tile.cta_n
    while d_state <= max_d_state:
        if d_state % tile.cta_n == 0:
            for d_head in d_head_options:
                if d_head % tile.cta_k != 0 and d_head % tile.inst_k != 0:
                    continue
                sram_bytes = d_state * d_head * bytes_per_element
                results.append(StateConfig(
                    d_state=d_state,
                    d_head=d_head,
                    tile_aligned=True,
                    sram_bytes=sram_bytes,
                ))
        d_state += tile.cta_n

    return results



# =============================================================================
# v2 State lattice and hybrid pattern computation
# =============================================================================

# Per-hardware concurrent-heads and fast-memory parameters for SRAM derivation.
# These are inlined here so the lattice engine remains self-contained (no
# dependency on the v1-sram-state-hybrid module at import time).
_STATE_HW_PARAMS = {
    "h100":    {"fast_mem_bytes": 228 * 1024, "h_concurrent": 8,  "tier": "sram"},
    "b200":    {"fast_mem_bytes": 256 * 1024, "h_concurrent": 8,  "tier": "sram"},
    "tpu_v5e": {"fast_mem_bytes": 16 * 1024 * 1024, "h_concurrent": 32, "tier": "vmem"},
    "tpu_v5p": {"fast_mem_bytes": 32 * 1024 * 1024, "h_concurrent": 64, "tier": "vmem"},
}

# GPU d_state candidates: multiples of 32 up to 256 that are common tile-friendly values
_GPU_D_STATE_CANDIDATES = [64, 96, 128, 160, 192, 224, 256]

# Quality saturation cap (beyond this, quality improvements are marginal)
_QUALITY_SATURATION_CAP = 256


def _derive_d_state_max_from_hw(
    hw: HardwareSpec,
    d_head: int,
    state_precision: str = "bf16",
    alpha_state: float = 0.85,
) -> int:
    """Derive max d_state from hardware SRAM/VMEM capacity."""
    params = _STATE_HW_PARAMS.get(hw.short_name)
    if params is None:
        # Fallback: use the lattice engine's sram_per_sm_bytes
        fast_mem = hw.sram_per_sm_bytes
        h_conc = 8
    else:
        fast_mem = params["fast_mem_bytes"]
        h_conc = params["h_concurrent"]

    bpe = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1, "int8": 1, "fp4": 1}.get(state_precision, 2)
    denominator = h_conc * d_head * bpe
    if denominator <= 0:
        return 0
    return int(fast_mem * alpha_state / denominator)


def compute_state_lattice(
    hw: HardwareSpec,
    d_head: int,
    state_precision: str = "bf16",
    alpha_state: float = 0.85,
) -> List[int]:
    """
    Compute tile-aligned d_state values that fit in SRAM for a given hardware.

    GPU: returns values from {64, 96, 128, 160, 192, 224, 256} that are <= d_state_max.
    TPU: returns multiples of 128 up to the quality saturation cap (256).

    Args:
        hw: Hardware specification
        d_head: head dimension for state
        state_precision: precision for state storage
        alpha_state: SRAM budget fraction for state

    Returns:
        Sorted list of valid d_state values
    """
    d_state_max = _derive_d_state_max_from_hw(hw, d_head, state_precision, alpha_state)
    params = _STATE_HW_PARAMS.get(hw.short_name, {})
    tier = params.get("tier", "sram")

    if tier == "vmem":
        # TPU: multiples of 128 up to quality cap
        cap = min(d_state_max, _QUALITY_SATURATION_CAP)
        result = []
        d = 128
        while d <= cap:
            result.append(d)
            d += 128
        return result
    else:
        # GPU: from candidate set, filtered by d_state_max
        return [d for d in _GPU_D_STATE_CANDIDATES if d <= d_state_max]


@dataclass
class HybridPattern:
    """A hybrid attention/state layer placement pattern."""
    n_attn: int
    n_state: int
    n_total: int
    placement_strategy: str
    attention_indices: List[int]
    state_indices: List[int]


def place_attention_layers(n_total: int, n_attn: int, strategy: str) -> List[int]:
    """
    Determine which layer indices should be attention layers.

    Strategies:
        first_periodic_last: layer 0 + last layer + evenly spaced interior
        interleaved: alternating attention/state
        periodic: 1 attention per N state (Jamba-style)

    Args:
        n_total: total number of layers
        n_attn: number of attention layers
        strategy: placement strategy name

    Returns:
        Sorted list of attention layer indices
    """
    if n_attn <= 0:
        return []
    if n_attn >= n_total:
        return list(range(n_total))

    if strategy == "first_periodic_last":
        if n_attn == 1:
            return [0]
        if n_attn == 2:
            return [0, n_total - 1]
        # Layer 0 + last layer + evenly spaced interior
        attn = {0, n_total - 1}
        remaining = n_attn - 2
        if remaining > 0:
            # Evenly space in the interior [1, n_total-2]
            interior_len = n_total - 2
            if remaining >= interior_len:
                attn.update(range(1, n_total - 1))
            else:
                for i in range(remaining):
                    idx = 1 + int((i + 0.5) * interior_len / remaining)
                    idx = min(idx, n_total - 2)
                    attn.add(idx)
        return sorted(attn)

    elif strategy == "interleaved":
        # Alternating: attention at positions where (index * n_attn / n_total) crosses an integer
        attn = []
        n_state = n_total - n_attn
        if n_attn <= n_state:
            # Fewer attention: space them out
            for i in range(n_total):
                # Place attention at positions that are evenly distributed
                if len(attn) < n_attn:
                    # Next attention position
                    target = int((len(attn) + 0.5) * n_total / n_attn)
                    if i >= target or (n_total - i) <= (n_attn - len(attn)):
                        attn.append(i)
        else:
            # More attention than state: invert (place state, everything else is attention)
            state_set = set()
            n_st = n_state
            for i in range(n_total):
                if len(state_set) < n_st:
                    target = int((len(state_set) + 0.5) * n_total / n_st)
                    if i >= target or (n_total - i) <= (n_st - len(state_set)):
                        state_set.add(i)
            attn = [i for i in range(n_total) if i not in state_set]
        return sorted(attn)

    elif strategy == "periodic":
        # Jamba-style: 1 attention per period (period = n_total / n_attn)
        period = n_total / max(1, n_attn)
        attn = []
        for i in range(n_attn):
            idx = int(i * period)
            idx = min(idx, n_total - 1)
            attn.append(idx)
        return sorted(set(attn))

    else:
        raise ValueError(f"Unknown placement strategy: {strategy!r}. "
                        f"Supported: first_periodic_last, interleaved, periodic")


def compute_hybrid_patterns(
    n_layers: int,
    n_attn_range: range,
    strategies: Optional[List[str]] = None,
) -> List[HybridPattern]:
    """
    Compute all hybrid attention/state patterns for given parameters.

    Args:
        n_layers: total number of layers
        n_attn_range: range of attention layer counts to try
        strategies: list of placement strategies (default: all three)

    Returns:
        List of HybridPattern objects
    """
    if strategies is None:
        strategies = ["first_periodic_last", "interleaved", "periodic"]

    results = []
    for n_attn in n_attn_range:
        if n_attn < 0 or n_attn > n_layers:
            continue
        n_state = n_layers - n_attn
        for strategy in strategies:
            attn_indices = place_attention_layers(n_layers, n_attn, strategy)
            # Adjust n_attn to actual count (periodic strategy may deduplicate)
            actual_n_attn = len(attn_indices)
            state_indices = [i for i in range(n_layers) if i not in set(attn_indices)]
            actual_n_state = len(state_indices)
            results.append(HybridPattern(
                n_attn=actual_n_attn,
                n_state=actual_n_state,
                n_total=n_layers,
                placement_strategy=strategy,
                attention_indices=attn_indices,
                state_indices=state_indices,
            ))

    return results


def compute_cross_precision_intersection(
    hw: HardwareSpec,
    prec_a: str,
    prec_b: str,
    tp: int,
    d_model_min: int = 1024,
    d_model_max: int = 16384,
) -> List[int]:
    """Compute d_model values valid for both precision levels simultaneously."""
    if prec_a not in hw.tiles or prec_b not in hw.tiles:
        return []

    tile_a = hw.tiles[prec_a]
    tile_b = hw.tiles[prec_b]

    # d_model must satisfy both alignment constraints
    stride_a = lcm(lcm(tile_a.cta_k, tile_a.cta_n), tp)
    stride_b = lcm(lcm(tile_b.cta_k, tile_b.cta_n), tp)
    combined_stride = lcm(stride_a, stride_b)

    results = []
    d = round_up_to(d_model_min, combined_stride)
    while d <= d_model_max:
        results.append(d)
        d += combined_stride

    return results


# =============================================================================
# Calculator: efficient_configs()
# =============================================================================

@dataclass
class ArchConfig:
    """A complete architecture configuration from the calculator."""
    d_model: int
    n_heads: int
    d_head: int
    ffn_dim: int
    n_kv_heads: Optional[int]
    n_layers: int
    total_params_b: float
    tile_util_harmonic: float
    wave_eff_2048: float
    wave_eff_8192: float
    composite_score: float


def estimate_params(d_model, n_heads, d_head, ffn_dim, n_layers, n_kv_heads=None, vocab_size=32000):
    """Estimate total parameter count for a transformer."""
    if n_kv_heads is None:
        n_kv_heads = n_heads

    # QKV projections
    q_params = d_model * d_head * n_heads
    kv_params = 2 * d_model * d_head * n_kv_heads
    # Output projection
    o_params = d_head * n_heads * d_model
    # FFN (SwiGLU: up, gate, down)
    ffn_params = 3 * d_model * ffn_dim
    # Per layer
    per_layer = q_params + kv_params + o_params + ffn_params
    # Embedding + output head
    embed_params = 2 * vocab_size * d_model
    # Layer norms (small)
    norm_params = 2 * d_model * n_layers

    total = per_layer * n_layers + embed_params + norm_params
    return total


def efficient_configs(
    hardware: str,
    precision: str,
    tp_degree: int,
    target_params_b: float,
    min_d_model: int = 1024,
    max_d_model: int = 16384,
    d_head: int = 128,
    gqa_group_size: Optional[int] = None,
    ffn_type: str = "swiglu",
    vocab_size: int = 32000,
    min_layers: int = 4,
    max_layers: int = 200,
    top_k: int = 10,
) -> List[ArchConfig]:
    """
    Find efficient architecture configurations for a target parameter count.

    Returns the top-k configurations sorted by composite score.
    """
    hw = HARDWARE[hardware]
    target_params = target_params_b * 1e9

    lattice = compute_lattice(hw, precision, tp_degree, min_d_model, max_d_model, [d_head])

    configs = []
    for pt in lattice:
        if not pt.tile_aligned:
            continue

        ffn_dim = pt.ffn_dim_swiglu if ffn_type == "swiglu" else pt.ffn_dim_dense

        # Determine n_kv_heads
        if gqa_group_size is not None and gqa_group_size > 1:
            n_kv_heads = pt.n_heads // gqa_group_size
            if n_kv_heads < 1:
                continue
            if n_kv_heads < tp_degree and n_kv_heads != 1:
                continue
        else:
            n_kv_heads = pt.n_heads

        # Compute n_layers for target param count
        per_layer = estimate_params(pt.d_model, pt.n_heads, pt.d_head, ffn_dim, 1, n_kv_heads, vocab_size)
        per_layer -= 2 * vocab_size * pt.d_model  # remove embedding from per-layer
        per_layer -= 2 * pt.d_model  # remove norm from per-layer
        embed_params = 2 * vocab_size * pt.d_model

        if per_layer <= 0:
            continue

        n_layers_raw = (target_params - embed_params) / per_layer
        n_layers = max(1, round(n_layers_raw))

        # Filter by practical depth range
        if n_layers < min_layers or n_layers > max_layers:
            continue

        actual_params = per_layer * n_layers + embed_params + 2 * pt.d_model * n_layers
        params_b = actual_params / 1e9

        # Param proximity score (closer to target = better)
        param_ratio = min(params_b, target_params_b) / max(params_b, target_params_b)

        # Composite score: weight tile utilization highest, then wave efficiency, then param proximity
        composite = (
            0.50 * pt.tile_util_harmonic +
            0.15 * pt.wave_eff_2048 +
            0.10 * pt.wave_eff_8192 +
            0.25 * param_ratio
        )

        configs.append(ArchConfig(
            d_model=pt.d_model,
            n_heads=pt.n_heads,
            d_head=pt.d_head,
            ffn_dim=ffn_dim,
            n_kv_heads=n_kv_heads,
            n_layers=n_layers,
            total_params_b=round(params_b, 2),
            tile_util_harmonic=pt.tile_util_harmonic,
            wave_eff_2048=pt.wave_eff_2048,
            wave_eff_8192=pt.wave_eff_8192,
            composite_score=round(composite, 4),
        ))

    # Sort by composite score descending
    configs.sort(key=lambda c: c.composite_score, reverse=True)
    return configs[:top_k]


# =============================================================================
# Full Table Generation (JSON export for the HTML calculator)
# =============================================================================

def generate_all_tables() -> dict:
    """Generate all lattice tables as a JSON-serializable dict."""
    output = {
        "hardware": {},
        "cross_precision": {},
        "state_configs": {},
    }

    for hw_name, hw in HARDWARE.items():
        output["hardware"][hw_name] = {
            "name": hw.name,
            "n_sms": hw.n_sms,
            "precisions": {}
        }

        for prec, tile in hw.tiles.items():
            output["hardware"][hw_name]["precisions"][prec] = {
                "tile": {
                    "inst": [tile.inst_m, tile.inst_n, tile.inst_k],
                    "cta": [tile.cta_m, tile.cta_n, tile.cta_k],
                },
                "tp_configs": {}
            }

            for tp in [1, 2, 4, 8]:
                lattice = compute_lattice(hw, prec, tp)
                points = []
                for pt in lattice:
                    points.append({
                        "d_model": pt.d_model,
                        "d_head": pt.d_head,
                        "n_heads": pt.n_heads,
                        "ffn_swiglu": pt.ffn_dim_swiglu,
                        "ffn_dense": pt.ffn_dim_dense,
                        "aligned": pt.tile_aligned,
                        "util_qkv": pt.tile_util_qkv,
                        "util_attn": pt.tile_util_attn,
                        "util_ffn": pt.tile_util_ffn,
                        "util_harmonic": pt.tile_util_harmonic,
                        "wave_2048": pt.wave_eff_2048,
                        "wave_8192": pt.wave_eff_8192,
                    })
                output["hardware"][hw_name]["precisions"][prec]["tp_configs"][str(tp)] = points

            # State configs per precision
            state_key = f"{hw_name}_{prec}"
            state = compute_state_configs(hw, prec)
            output["state_configs"][state_key] = [
                {"d_state": s.d_state, "d_head": s.d_head, "sram_bytes": s.sram_bytes}
                for s in state
            ]

        # Cross-precision intersection
        prec_list = list(hw.tiles.keys())
        for i in range(len(prec_list)):
            for j in range(i + 1, len(prec_list)):
                pa, pb = prec_list[i], prec_list[j]
                for tp in [1, 2, 4, 8]:
                    key = f"{hw_name}_{pa}_{pb}_tp{tp}"
                    vals = compute_cross_precision_intersection(hw, pa, pb, tp)
                    output["cross_precision"][key] = vals

    return output


# =============================================================================
# Known Architecture Validation
# =============================================================================

KNOWN_ARCHITECTURES = {
    # --- Dense models ---
    "Llama-2-7B":   {"d_model": 4096, "d_head": 128, "n_heads": 32, "ffn_dim": 11008, "n_layers": 32},
    "Llama-2-13B":  {"d_model": 5120, "d_head": 128, "n_heads": 40, "ffn_dim": 13824, "n_layers": 40},
    "Llama-2-70B":  {"d_model": 8192, "d_head": 128, "n_heads": 64, "ffn_dim": 28672, "n_layers": 80},
    "Llama-3-8B":   {"d_model": 4096, "d_head": 128, "n_heads": 32, "ffn_dim": 14336, "n_layers": 32},
    "Llama-3-70B":  {"d_model": 8192, "d_head": 128, "n_heads": 64, "ffn_dim": 28672, "n_layers": 80},
    "Mistral-7B":   {"d_model": 4096, "d_head": 128, "n_heads": 32, "ffn_dim": 14336, "n_layers": 32},
    "Gemma-2-9B":   {"d_model": 3584, "d_head": 256, "n_heads": 16, "ffn_dim": 14336, "n_layers": 42},
    "Qwen3-8B":     {"d_model": 4096, "d_head": 128, "n_heads": 32, "ffn_dim": 12288, "n_layers": 36},
    "Qwen3-32B":    {"d_model": 5120, "d_head": 128, "n_heads": 64, "ffn_dim": 25600, "n_layers": 64},
    # --- MoE models (dense-equivalent: attention dims exact, ffn_dim = active per-token FFN) ---
    # DeepSeek-V3: 671B total, ~37B active, MLA + MoE (8/256+1 shared), 128 attn heads
    "DeepSeek-V3":  {"d_model": 7168, "d_head": 128, "n_heads": 128, "ffn_dim": 18432, "n_layers": 61},
    # Kimi-K2.5: ~1T total, ~32B active, MLA + MoE (8/384+1 shared), DeepSeek-V3 family
    "Kimi-K2.5":    {"d_model": 7168, "d_head": 128, "n_heads": 64,  "ffn_dim": 18432, "n_layers": 61},
    # GLM-5.1: 754B total, ~40B active, MLA + MoE (8/256+1 shared)
    "GLM-5.1":      {"d_model": 6144, "d_head": 64,  "n_heads": 64,  "ffn_dim": 18432, "n_layers": 78},
    # GPT-OSS-120B: 116.8B total, ~5.1B active, MoE (4/128), d_model!=n_heads*d_head
    "GPT-OSS-120B": {"d_model": 2880, "d_head": 64,  "n_heads": 64,  "ffn_dim": 11520, "n_layers": 36},
    # MAI-Base-1: 962B total, ~34.7B active, Latent MoE (8/512), d_model!=n_heads*d_head
    "MAI-Base-1":   {"d_model": 6656, "d_head": 128, "n_heads": 80,  "ffn_dim": 24576, "n_layers": 78},
}


def validate_known_architectures(hardware: str = "h100", precision: str = "bf16"):
    """Check tile alignment of known architectures."""
    hw = HARDWARE[hardware]
    tile = hw.tiles[precision]

    results = []
    for name, arch in KNOWN_ARCHITECTURES.items():
        d = arch["d_model"]
        dh = arch["d_head"]
        nh = arch["n_heads"]
        ffn = arch["ffn_dim"]

        checks = {
            "d_model % cta_k": d % tile.cta_k == 0,
            "d_model % cta_n": d % tile.cta_n == 0,
            "d_head % cta_k":  dh % tile.cta_k == 0,
            "d_head % cta_n":  dh % tile.cta_n == 0,
            "ffn % cta_n":     ffn % tile.cta_n == 0,
        }
        all_pass = all(checks.values())
        results.append({"name": name, "checks": checks, "all_aligned": all_pass})

    return results


if __name__ == "__main__":
    # Generate all tables and write JSON
    print("Computing tile-aligned architecture lattice...")
    tables = generate_all_tables()

    with open("lattice_data.json", "w") as f:
        json.dump(tables, f, indent=2)
    print(f"Wrote lattice_data.json")

    # Validate known architectures
    print("\nValidating known architectures (H100 BF16):")
    for r in validate_known_architectures():
        status = "✓" if r["all_aligned"] else "✗"
        fails = [k for k, v in r["checks"].items() if not v]
        fail_str = f" (fails: {', '.join(fails)})" if fails else ""
        print(f"  {status} {r['name']}{fail_str}")

    # Example calculator usage
    print("\nExample: 7B configs on H100 BF16 TP=1:")
    configs = efficient_configs("h100", "bf16", 1, 7.0)
    for c in configs[:5]:
        print(f"  d={c.d_model} h={c.n_heads} dh={c.d_head} ffn={c.ffn_dim} "
              f"L={c.n_layers} params={c.total_params_b}B util={c.tile_util_harmonic:.3f} "
              f"score={c.composite_score:.4f}")
