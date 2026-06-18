"""
Throughput Model v0 — T(A, H, lattice) -> ThroughputResult

Analytical end-to-end throughput model for dense transformers (with optional GQA)
on NVIDIA H100, NVIDIA B200, Google TPU v5e, and Google TPU v5p.

Sits on top of the tile-aligned lattice: queries tile efficiency per matmul,
never modifies the lattice. Designed for millisecond-scale evaluation so the
optimizer can call it thousands of times per second.

Extension hooks reserved for v1+: MoE, state layers, heterogeneous pipelines.
"""

import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Import lattice engine (sibling module in the same package)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from lattice_engine import (
    HardwareSpec as LatticeHardwareSpec,
    TileSpec,
    HARDWARE as LATTICE_HARDWARE,
    matmul_tile_utilization,
    wave_efficiency,
    compute_lattice,
    compute_gqa_configs,
    estimate_params,
    KNOWN_ARCHITECTURES,
)


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class ArchConfig:
    """Architecture configuration — input to the throughput model."""
    d_model: int
    n_layers: int
    n_heads: int
    d_head: int
    n_kv_heads: int          # = n_heads for MHA, < n_heads for GQA/MQA
    ffn_dim: int
    ffn_type: str = "swiglu"  # "swiglu" | "dense"
    vocab_size: int = 32000
    batch_size: int = 1
    seq_len: int = 2048
    precision: str = "bf16"
    kv_precision: str = "bf16"  # KV cache precision (can differ for quantized KV)

    # v1 MoE hook (v0.1: always None; v0.2+: dict with the MoE FFN block).
    # Shape matches schema MoEFFNConfig:
    #   {n_experts, top_k, expert_dim, shared_expert, router,
    #    capacity_factor, precision}
    layer_types: Optional[List[str]] = None   # v0: always ['attention'] * n_layers
    moe_config: Optional[dict] = None

    # v1 MoE: worst-case routing imbalance multiplier. 1.0 = balanced (training
    # objective). Set > 1.0 to stress-test concentration during planning.
    worst_case_imbalance_factor: float = 1.0

    # v1-fix Part B: first-K-dense prefix when moe_config is set. Throughput
    # treats per-layer FFN cost as a weighted average:
    #   per-stack FFN time =
    #     n_dense_ffn_layers * dense_ffn_time +
    #     (n_layers - n_dense_ffn_layers) * moe_ffn_time
    # All-to-all volume also pro-rates by MoE-layer fraction.
    n_dense_ffn_layers: int = 0

    # v2 state/hybrid fields
    state_config: Optional[dict] = None
    # Keys: d_state, state_expansion, n_heads, d_head, state_precision
    layer_type_list: Optional[List[str]] = None
    # Per-layer type: "attention" or "state". When None, all attention.

    # v1-fix MLA: DeepSeek-V2/V3 Multi-head Latent Attention. When
    # `attention_type == "mla"`, the per-token KV cache stores ONE compressed
    # latent (c_kv) plus the RoPE'd key (d_rope), regardless of n_kv_heads.
    # This dramatically cuts decode KV bandwidth at long context. Quality
    # cost is captured by `attention_mla` in the architecture residual.
    attention_type: str = "full"          # "full" | "mla" | "nsa"
    mla_kv_latent_dim: Optional[int] = None     # c_kv
    mla_q_latent_dim: Optional[int] = None      # c_q
    mla_rope_head_dim: Optional[int] = None     # d_rope
    mla_nope_head_dim: Optional[int] = None     # d_nope
    # v1-fix NSA: Native Sparse Attention (DeepSeek 2025). Three hierarchical
    # branches sum to per-token KV: B = ceil(L / stride) compressed blocks +
    # K × bs selected tokens + W sliding-window tokens.
    nsa_compress_block_size: Optional[int] = None
    nsa_compress_block_stride: Optional[int] = None
    nsa_select_block_size: Optional[int] = None
    nsa_select_top_k: Optional[int] = None
    nsa_window_size: Optional[int] = None

    # v1-fix MTP: Multi-Token Prediction training overhead. Inference path
    # is unchanged (heads dropped); training path pays per-depth FLOPs.
    mtp_n_predict_depths: int = 0
    mtp_depth_n_layers: int = 1
    mtp_inference_mode: str = "drop"      # "drop" | "speculative_decode"

    # v1-fix CP: Context Parallelism. Splits the sequence axis across CP
    # ranks. At training time, attention compute and KV-cache memory both
    # divide by cp; the all-gather comm cost is roughly
    #   (cp-1)/cp × n_layers × B × S × d_model × bpe / NVLink_bw
    # per training step.
    cp_degree: int = 1                    # 1 = no CP
    cp_method: str = "ring"               # "ring" | "ulysses"

    # v1-fix 2:4 structured sparsity. Per-component flags driving the
    # tensor-core sparse path. NVIDIA H100/B200 give 2× matmul throughput
    # on 2:4-sparsified weights. Other vendors (TPU, Trainium) fall back
    # to dense, so the speedup is hardware-conditional.
    sparsity_2_4: Optional[Dict[str, bool]] = None

    # v1-fix YOCO: cross-layer KV sharing. Only the first K layers each
    # carry their own KV; the remaining N-K layers cross-attend to the
    # K-th layer's KV. KV cache shrinks to K/N of dense, and decode
    # bandwidth too (modulo the cross-attention read pattern).
    yoco_n_self_attn_layers: int = 0  # 0 = YOCO off

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["attention"] * self.n_layers
        if self.layer_type_list is None:
            self.layer_type_list = ["attention"] * self.n_layers

    def kv_bytes_per_token_per_layer(self, context_length: int = 0) -> int:
        """v1-fix MLA/NSA: per-token per-layer KV cache bytes.

        For MLA, KV is ONE compressed latent (c_kv) + RoPE'd K (d_rope)
        shared across all heads. For NSA, KV is the sum of three branches:
        compressed (ceil(L/stride) blocks × d_head), selected (top-k blocks ×
        block-size × d_head), and sliding window (W × d_head). The full
        L-dependent sum is folded into `kv_bytes_per_layer` below by callers
        that pass `context_length`; for the per-token figure, we report the
        amortized per-token cost using the effective NSA budget.
        For non-MLA/non-NSA, K and V are each stored per-kv-head:
        2 × n_kv_heads × d_head × bpe.
        """
        bpe_map = {"bf16": 2, "fp16": 2, "fp8": 1, "fp4": 0.5,
                   "int8": 1, "int4": 0.5, "tf32": 4, "fp32": 4,
                   # v1-fix microscaling: MX formats include the shared scale
                   # (1 byte per 32 elements ≈ 0.03 byte/elem) but the elem
                   # body is the same as E2M1 / E2M3 / E3M2 (4/6 bits).
                   "mxfp4": 0.53, "mxfp6": 0.78}
        bpe = bpe_map.get(self.kv_precision, 2)
        if self.attention_type == "mla" and self.mla_kv_latent_dim:
            c_kv = int(self.mla_kv_latent_dim)
            d_rope = int(self.mla_rope_head_dim or 0)
            return int((c_kv + d_rope) * bpe)
        if self.attention_type == "nsa" and self.nsa_window_size:
            # NSA's *amortized* per-token KV: stored once but read multiple
            # times. For decode-time read bandwidth, the effective budget is
            # (compressed_blocks + selected_tokens + window) attended *per
            # query token*. We use a context-aware compact form here when
            # the caller supplies L; otherwise approximate by the steady-state
            # NSA budget at L=64k (DeepSeek paper).
            L = int(context_length or 65536)
            cbs = int(self.nsa_compress_block_size or 64)
            cbst = int(self.nsa_compress_block_stride or 16)
            sbs = int(self.nsa_select_block_size or 64)
            stk = int(self.nsa_select_top_k or 16)
            win = int(self.nsa_window_size or 512)
            # Total bytes read per query token (averaged over heads):
            n_compress = max(1, (L + cbst - 1) // cbst)
            n_select = stk * sbs
            effective_tokens = min(L, n_compress + n_select + win)
            # Per-token *cache footprint* (write-side) is still the full L
            # per kv-head; but bandwidth-relevant figure is effective_tokens
            # divided by L. To match the existing kv_per_layer math
            # (per_token × L = total), we shrink per-token bytes so that
            # per_token × L = effective_tokens × per_kv_head_bytes_per_tok.
            per_kv_head = 2 * self.n_kv_heads * self.d_head * bpe
            return int(per_kv_head * effective_tokens / max(1, L))
        return int(2 * self.n_kv_heads * self.d_head * bpe)


@dataclass
class HardwareConfig:
    """Hardware configuration loaded from JSON spec files."""
    vendor: str
    accelerator_family: str
    accelerator_name: str
    compute_units: int
    compute_unit_type: str
    hbm_capacity_gb: float
    hbm_bandwidth_tb_s: float
    peak_flops_tf: Dict[str, float]
    bytes_per_element: Dict[str, float]
    supported_precisions: List[str]
    fused_attention_efficiency: Dict[str, float]
    fused_attention_kernel: str
    interconnect: dict
    gpus_per_node: int = 8
    chips_per_host: int = 4

    # Fabric domain size: how many ranks share full-bandwidth NVLink (or single-axis ICI).
    # 8 for DGX-H100/HGX-H100/DGX-B200, 72 for NVL72, 16 for TPU v5p single torus axis,
    # 8 for TPU v5e. Optional in JSON — falls back to family-based inference when absent.
    nvlink_domain_size: Optional[int] = None

    # Calibration constants — account for real-world overheads not modeled analytically
    calibration: dict = field(default_factory=lambda: {
        "kernel_launch_overhead_us": 5.0,
        "kernels_per_layer": 12,
        "optimizer_bytes_per_param": 12,    # AdamW: 4 (fp32 master) + 4 (m) + 4 (v) = 12 bytes
        "training_system_efficiency": 0.55, # accounts for data loading, optimizer, mem mgmt
        "decode_system_efficiency": 0.42,   # accounts for kernel launch dominance at small batch
        "prefill_system_efficiency": 0.60,
    })

    @classmethod
    def from_json(cls, path: str) -> "HardwareConfig":
        with open(path) as f:
            d = json.load(f)
        calibration = d.get("calibration", {})
        return cls(
            vendor=d["vendor"],
            accelerator_family=d["accelerator_family"],
            accelerator_name=d["accelerator_name"],
            compute_units=d["compute_units"],
            compute_unit_type=d["compute_unit_type"],
            hbm_capacity_gb=d["hbm_capacity_gb"],
            hbm_bandwidth_tb_s=d["hbm_bandwidth_tb_s"],
            peak_flops_tf=d["peak_flops_tf"],
            bytes_per_element=d["bytes_per_element"],
            supported_precisions=d["supported_precisions"],
            fused_attention_efficiency=d["fused_attention_efficiency"],
            fused_attention_kernel=d["fused_attention_kernel"],
            interconnect=d["interconnect"],
            gpus_per_node=d.get("gpus_per_node", d.get("chips_per_host", 8)),
            chips_per_host=d.get("chips_per_host", d.get("gpus_per_node", 4)),
            nvlink_domain_size=d.get("nvlink_domain_size"),
            calibration=calibration,
        )

    @property
    def hbm_bandwidth_bytes_s(self) -> float:
        return self.hbm_bandwidth_tb_s * 1e12

    def peak_flops_s(self, precision: str) -> float:
        """Peak FLOPS in raw FLOPS (not teraflops)."""
        return self.peak_flops_tf.get(precision, self.peak_flops_tf.get("bf16", 0)) * 1e12

    def bytes_per_elem(self, precision: str) -> float:
        return self.bytes_per_element.get(precision, 2)

    def interconnect_bw_bytes_s(self, tp_degree: int) -> float:
        """Effective interconnect bandwidth for TP all-reduce."""
        if tp_degree <= 1:
            return float("inf")
        ic = self.interconnect
        per_node = self.gpus_per_node if self.vendor == "nvidia" else self.chips_per_host
        if tp_degree <= per_node:
            return ic["intra_node_bw_gb_s"] * 1e9
        else:
            return ic["inter_node_bw_gb_s"] * 1e9


@dataclass
class LayerBreakdown:
    """Per-layer time breakdown in seconds.

    v0.2: extended with MoE-specific terms (alltoall_s, expert_load_s,
    load_balance_factor). Dense layers report zeros for these. The
    bottleneck enum now includes "alltoall" and "expert_load".
    """
    compute_s: float = 0.0
    memory_s: float = 0.0
    communication_s: float = 0.0
    total_s: float = 0.0
    bottleneck: str = "compute"

    # Sub-operation detail
    qkv_proj_s: float = 0.0
    attention_s: float = 0.0
    out_proj_s: float = 0.0
    ffn_up_s: float = 0.0
    ffn_down_s: float = 0.0
    membound_ops_s: float = 0.0
    allreduce_s: float = 0.0

    # v1+ MoE-specific terms (zero for dense layers)
    alltoall_s: float = 0.0           # expert dispatch + combine
    expert_load_s: float = 0.0        # decode-phase expert weight loading
    shared_expert_s: float = 0.0      # DeepSeek-style always-on expert compute
    load_balance_factor: float = 1.0  # routing imbalance multiplier applied to expert compute


@dataclass
class ThroughputResult:
    """Output of the throughput model."""
    # Training
    training_time_per_step_s: float = 0.0
    training_throughput_tokens_per_sec: float = 0.0

    # Inference — prefill
    prefill_time_ms: float = 0.0

    # Inference — decode (per generated token, at a given KV cache length)
    decode_time_per_token_ms: float = 0.0
    decode_kv_cache_length: int = 0

    # Memory
    memory_footprint_per_gpu_gb: float = 0.0

    # Breakdown
    per_layer_breakdown: Optional[LayerBreakdown] = None  # training
    prefill_layer_breakdown: Optional[LayerBreakdown] = None
    decode_layer_breakdown: Optional[LayerBreakdown] = None
    bottleneck: str = "compute"

    # Parallelism
    tp_degree: int = 1
    pp_degree: int = 1
    bubble_fraction: float = 0.0

    # Metadata
    hardware_name: str = ""
    precision: str = ""


# =============================================================================
# Hardware spec loader
# =============================================================================

_SPEC_DIR = os.environ.get(
    "AC_HARDWARE_SPEC_DIR",
    os.path.join(os.path.dirname(__file__), "hardware_specs"),
)
_CALIBRATION_DIR = os.environ.get(
    "AC_CALIBRATION_DIR",
    os.path.join(os.path.dirname(__file__), "calibration"),
)


# =============================================================================
# Calibration layer — optional measured efficiency overrides
# =============================================================================

@dataclass
class CalibrationTable:
    """Measured kernel efficiencies that override analytic estimates."""
    gemm_efficiencies: Dict[str, float] = field(default_factory=dict)
    attention_latencies: Dict[str, float] = field(default_factory=dict)
    decode_kv_latencies: Dict[str, float] = field(default_factory=dict)
    allreduce_latencies: Dict[str, float] = field(default_factory=dict)
    source: str = "analytic"

    @classmethod
    def from_json(cls, path: str) -> "CalibrationTable":
        with open(path) as f:
            d = json.load(f)
        table = cls(source=d.get("source", "public_estimate"))
        for entry in d.get("gemm_shapes", []):
            key = f"{entry['m_bucket']}x{entry['n_bucket']}x{entry['k_bucket']}_{entry.get('precision','bf16')}"
            table.gemm_efficiencies[key] = entry["efficiency"]
        for entry in d.get("attention_prefill", []):
            key = f"b{entry['batch']}_s{entry['seq_len']}_h{entry['n_heads']}_d{entry['d_head']}"
            if entry.get("latency_ms", 0) > 0:
                table.attention_latencies[key] = entry["latency_ms"]
        for entry in d.get("decode_kv", []):
            key = f"b{entry['batch']}_c{entry['context']}_kv{entry['n_kv_heads']}_d{entry['d_head']}_{entry.get('kv_dtype','bf16')}"
            if entry.get("latency_ms", 0) > 0:
                table.decode_kv_latencies[key] = entry["latency_ms"]
        for entry in d.get("all_reduce", []):
            key = f"tp{entry['tp']}_msg{entry['message_size_mb']}mb"
            if entry.get("latency_ms", 0) > 0:
                table.allreduce_latencies[key] = entry["latency_ms"]
        return table

    def lookup_gemm_efficiency(self, M: int, N: int, K: int, precision: str) -> Optional[float]:
        """Find closest matching GEMM efficiency from calibration data."""
        best_key = None
        best_dist = float("inf")
        for key, eff in self.gemm_efficiencies.items():
            parts = key.split("_")
            dims = parts[0].split("x")
            prec = parts[1] if len(parts) > 1 else "bf16"
            if prec != precision:
                continue
            m, n, k = int(dims[0]), int(dims[1]), int(dims[2])
            dist = abs(math.log(max(M,1)/max(m,1))) + abs(math.log(max(N,1)/max(n,1))) + abs(math.log(max(K,1)/max(k,1)))
            if dist < best_dist:
                best_dist = dist
                best_key = key
        if best_key is not None and best_dist < 2.0:
            return self.gemm_efficiencies[best_key]
        return None


_CALIBRATION_CACHE: Dict[str, Optional[CalibrationTable]] = {}

def load_calibration(hw_name: str) -> Optional[CalibrationTable]:
    """Load calibration table for a hardware target, if available."""
    if hw_name in _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE[hw_name]
    path = os.path.join(_CALIBRATION_DIR, f"{hw_name}_calibration.json")
    if os.path.exists(path):
        table = CalibrationTable.from_json(path)
        _CALIBRATION_CACHE[hw_name] = table
        return table
    _CALIBRATION_CACHE[hw_name] = None
    return None


def load_hardware(name: str) -> HardwareConfig:
    """Load a hardware config by short name.

    v1-fix Trainium: AWS Trainium 2 / Trainium 3 added as fourth-vendor
    hardware targets. Trn2 ships today (re:Invent 2024); Trn3 numbers are
    public-estimate from the AWS roadmap.
    """
    mapping = {
        "h100": "h100_sxm.json",
        "b200": "b200.json",
        "tpu_v5e": "tpu_v5e.json",
        "tpu_v5p": "tpu_v5p.json",
        "trainium2": "trainium2.json",
        "trn2": "trainium2.json",          # alias
        "trainium3": "trainium3.json",
        "trn3": "trainium3.json",          # alias
    }
    filename = mapping.get(name)
    if filename is None:
        raise ValueError(f"Unknown hardware: {name}. Supported: {list(mapping.keys())}")
    return HardwareConfig.from_json(os.path.join(_SPEC_DIR, filename))


# =============================================================================
# Lattice integration — tile efficiency lookup
# =============================================================================

def get_tile_efficiency(
    M: int, N: int, K: int,
    precision: str,
    lattice_hw: LatticeHardwareSpec,
) -> float:
    """
    Query the lattice for tile efficiency of a matmul (M, N, K) at given precision.
    Returns combined tile utilization × wave efficiency as the effective efficiency.
    """
    if precision not in lattice_hw.tiles:
        # Fall back to bf16 if precision not in lattice
        precision = "bf16"
    if precision not in lattice_hw.tiles:
        return 0.8  # conservative fallback

    tile = lattice_hw.tiles[precision]
    tile_util = matmul_tile_utilization(M, N, K, tile)
    wave_eff = wave_efficiency(M, N, tile, lattice_hw.n_sms)
    return tile_util * wave_eff


# =============================================================================
# Per-operation cost functions
# =============================================================================

def _matmul_cost(
    M: int, N: int, K: int,
    precision: str,
    hw: HardwareConfig,
    lattice_hw: LatticeHardwareSpec,
    tp_degree: int = 1,
    calibration: Optional[CalibrationTable] = None,
) -> Tuple[float, float, str]:
    """
    Roofline cost for a single matmul.
    Returns (time_s, flops, bottleneck_type).

    If a calibration table is provided and has a matching GEMM shape,
    uses measured efficiency instead of analytic tile efficiency.
    """
    bpe = hw.bytes_per_elem(precision)
    flops = 2 * M * N * K

    # Try calibration first, fall back to lattice tile efficiency
    eff = None
    if calibration is not None:
        eff = calibration.lookup_gemm_efficiency(M, N, K, precision)
    if eff is None:
        eff = get_tile_efficiency(M, N, K, precision, lattice_hw)
    eff = max(eff, 0.1)  # floor to avoid division by zero

    t_compute = flops / (hw.peak_flops_s(precision) * eff)

    # Memory traffic: weights + activations
    weight_bytes = N * K * bpe
    act_bytes = (M * K + M * N) * bpe
    t_memory = (weight_bytes + act_bytes) / hw.hbm_bandwidth_bytes_s

    t = max(t_compute, t_memory)
    bottleneck = "compute" if t_compute >= t_memory else "memory"
    return t, flops, bottleneck


def _attention_cost(
    B: int, S: int, n_heads: int, d_head: int, n_kv_heads: int,
    precision: str,
    hw: HardwareConfig,
    tp_degree: int = 1,
) -> float:
    """
    Fused attention cost (FlashAttention / Splash Attention).
    Uses analytical IO model from Dao 2022.
    """
    heads_per_gpu = n_heads // tp_degree
    kv_heads_per_gpu = max(1, n_kv_heads // tp_degree)
    bpe = hw.bytes_per_elem(precision)

    # Compute: QK^T and softmax×V
    # With GQA, each KV head serves (n_heads / n_kv_heads) query heads
    flops_attn = 2 * B * heads_per_gpu * S * S * d_head * 2

    # HBM traffic under fusion: Q, K, V read + O written (no S×S materialization)
    hbm_bytes = B * S * d_head * bpe * (heads_per_gpu + 2 * kv_heads_per_gpu + heads_per_gpu)

    fused_eff = hw.fused_attention_efficiency.get(precision,
                hw.fused_attention_efficiency.get("bf16", 0.75))

    t_compute = flops_attn / (hw.peak_flops_s(precision) * fused_eff)
    t_memory = hbm_bytes / hw.hbm_bandwidth_bytes_s
    return max(t_compute, t_memory)


def _membound_ops_cost(
    B: int, S: int, d_model: int,
    precision: str,
    hw: HardwareConfig,
    n_norms: int = 3,  # pre-attn norm, pre-FFN norm, post-FFN norm/residual
) -> float:
    """
    Cost of memory-bound operations per layer: norms, residuals, activations.
    These are bandwidth-bound — just data movement.
    """
    bpe = hw.bytes_per_elem(precision)
    # Each norm: read + write = 2 × B × S × d_model
    # Residual connections: similar traffic
    # Activation functions: read + write
    total_bytes = n_norms * 2 * B * S * d_model * bpe
    return total_bytes / hw.hbm_bandwidth_bytes_s


def _allreduce_cost(
    B: int, S: int, d_model: int,
    precision: str,
    hw: HardwareConfig,
    tp_degree: int,
    n_allreduces: int = 2,  # one after attention, one after FFN
) -> float:
    """
    TP all-reduce cost per layer (no compute-communication overlap in v0).
    GPU: NVLink ring/tree. TPU: ICI.
    """
    if tp_degree <= 1:
        return 0.0

    bpe = hw.bytes_per_elem(precision)
    per_allreduce_bytes = 2 * B * S * d_model * bpe

    link_bw = hw.interconnect_bw_bytes_s(tp_degree)

    if hw.vendor == "nvidia":
        # Ring all-reduce: 2 × (P-1)/P × bytes / BW
        t_per = 2 * (tp_degree - 1) / tp_degree * per_allreduce_bytes / link_bw
    else:
        # TPU ICI: simpler model — 2 × bytes / effective_BW
        t_per = 2 * per_allreduce_bytes / link_bw

    return n_allreduces * t_per


# =============================================================================
# v1 MoE cost helpers
# =============================================================================

def _nvlink_domain_size(hw: HardwareConfig) -> int:
    """How many ranks share full-bandwidth NVLink (or single-axis ICI).

    Prefers the explicit `nvlink_domain_size` field from the hardware spec JSON
    (added in v1-fix part G so DGX-B200 vs NVL72 can be distinguished without
    code edits). Falls back to family-based inference for older specs that
    haven't been migrated.
    """
    if hw.nvlink_domain_size is not None and hw.nvlink_domain_size > 0:
        return hw.nvlink_domain_size
    fam = hw.accelerator_family.lower()
    if "blackwell" in fam:
        return 72       # legacy fallback: assume NVL72; DGX-B200 should set spec field to 8
    if "hopper" in fam:
        return 8        # DGX/HGX H100
    if fam.startswith("tpu"):
        return 16       # single ICI torus axis
    return hw.gpus_per_node


def _moe_alltoall_cost(
    volume_bytes: float,
    ep_degree: int,
    hw: HardwareConfig,
    ep_topology: str = "single_axis",
) -> float:
    """Time for the two all-to-alls in one MoE layer (dispatch + combine).

    volume_bytes is the *total* per-layer volume (both all-to-alls included).

    NVLink (NVIDIA): ring all-to-all at ~67% of peak link BW within the NVLink
    domain (8 on H100, 72 on B200 NVL72). Beyond the domain, the inter-node
    fabric is used and is dramatically slower; the formula below uses the
    inter-node BW directly because it is already much smaller than NVLink,
    so the cross-domain penalty is captured by the BW switch.

    TPU ICI 3D torus: effective single-axis BW at ~80% of peak ICI BW. If
    ep_topology=='cross_axis', traversal across torus axes inflates cost by
    a factor of 2.5 (default). The optimizer should prefer single-axis EP.
    """
    if ep_degree <= 1:
        # EP=1 still pays a small dispatch/combine kernel cost even though
        # there's no cross-rank traffic. Approximate as 1 us per layer.
        return 1e-6

    if hw.vendor == "nvidia":
        nvlink_domain = _nvlink_domain_size(hw)
        if ep_degree <= nvlink_domain:
            link_bw = hw.interconnect["intra_node_bw_gb_s"] * 1e9
            effective_bw = 0.67 * link_bw  # ring all-to-all efficiency
        else:
            # Beyond NVLink domain: use inter-node BW directly (much smaller).
            inter_bw = hw.interconnect["inter_node_bw_gb_s"] * 1e9
            effective_bw = 0.67 * inter_bw
        return (ep_degree - 1) / ep_degree * volume_bytes / max(effective_bw, 1.0)

    # TPU (or any non-NVIDIA): 3D torus / ICI
    link_bw = hw.interconnect["intra_node_bw_gb_s"] * 1e9  # v5p: intra == inter
    if ep_topology == "single_axis":
        effective_bw = 0.8 * link_bw
    else:
        effective_bw = 0.8 * link_bw / 2.5  # cross-axis penalty
    return (ep_degree - 1) / ep_degree * volume_bytes / max(effective_bw, 1.0)


def _moe_ffn_cost(
    B: int, S: int, d_model: int,
    moe_cfg: dict,
    ep_degree: int,
    tp_degree: int,
    activation_precision: str,
    expert_precision: str,
    hw: HardwareConfig,
    lattice_hw: LatticeHardwareSpec,
    phase: str,                          # "training" | "prefill" | "decode"
    calibration: Optional[CalibrationTable] = None,
    ep_topology: str = "single_axis",
    imbalance: float = 1.0,
) -> Dict[str, float]:
    """Compute the per-layer FFN time for an MoE layer.

    Returns a dict with keys:
      compute_s        — sum of expert and shared-expert matmul times
      shared_expert_s  — shared-expert portion (subset of compute_s)
      alltoall_s       — dispatch + combine communication time
      expert_load_s    — decode-phase HBM bandwidth to stream top_k experts
                          (= 0 for training/prefill where weights are reused)
      load_balance_factor — imbalance multiplier actually applied

    The dense-equivalent FFN matmul cost is *replaced* by this function when
    arch.moe_config is set; the caller should not add the dense ffn_up_s and
    ffn_down_s terms in that case.
    """
    n_experts = int(moe_cfg["n_experts"])
    top_k = int(moe_cfg["top_k"])
    expert_dim = int(moe_cfg["expert_dim"])
    capacity = float(moe_cfg.get("capacity_factor", 1.0))
    shared_block = moe_cfg.get("shared_expert")
    shared_dim = int(shared_block["ffn_dim"]) if shared_block else 0

    bpe_act = hw.bytes_per_elem(activation_precision)
    bpe_exp = hw.bytes_per_elem(expert_precision)

    expert_dim_per_rank = max(expert_dim // max(tp_degree, 1), 1)
    shared_dim_per_rank = max(shared_dim // max(tp_degree, 1), 1) if shared_dim else 0

    # --- Compute: per-rank tokens routed to local experts ---
    # Balanced routing: each rank holds n_experts/ep experts and receives
    # M * top_k * capacity / ep tokens. Decode treats S=1 here.
    S_eff = S if phase != "decode" else 1
    M = B * S_eff

    tokens_per_rank = max(1, int(M * top_k * capacity / max(ep_degree, 1)))
    tokens_per_rank = int(tokens_per_rank * imbalance)

    # SwiGLU expert: three matmuls (up, gate, down) per active expert per token.
    # In this analytic model we charge the *aggregate* matmul shape at the
    # tokens_per_rank batch; this implicitly assumes routed tokens are
    # contiguous-batched on each rank (the standard EP implementation).
    t_up, _, _ = _matmul_cost(
        tokens_per_rank, expert_dim_per_rank, d_model,
        expert_precision, hw, lattice_hw, tp_degree, calibration,
    )
    t_gate, _, _ = _matmul_cost(
        tokens_per_rank, expert_dim_per_rank, d_model,
        expert_precision, hw, lattice_hw, tp_degree, calibration,
    )
    t_down, _, _ = _matmul_cost(
        tokens_per_rank, d_model, expert_dim_per_rank,
        expert_precision, hw, lattice_hw, tp_degree, calibration,
    )
    expert_compute_s = t_up + t_gate + t_down

    # Shared expert (DeepSeek-style): always-on, sees every token (M, not
    # tokens_per_rank). Replicated across EP; sharded across TP.
    shared_compute_s = 0.0
    if shared_dim_per_rank > 0:
        t_su, _, _ = _matmul_cost(
            M, shared_dim_per_rank, d_model,
            expert_precision, hw, lattice_hw, tp_degree, calibration,
        )
        t_sg, _, _ = _matmul_cost(
            M, shared_dim_per_rank, d_model,
            expert_precision, hw, lattice_hw, tp_degree, calibration,
        )
        t_sd, _, _ = _matmul_cost(
            M, d_model, shared_dim_per_rank,
            expert_precision, hw, lattice_hw, tp_degree, calibration,
        )
        shared_compute_s = t_su + t_sg + t_sd

    compute_s = expert_compute_s + shared_compute_s

    # --- Decode-phase expert weight loading ---
    # At decode, each token activates top_k experts. Their weights must be
    # streamed from HBM (caches don't help across tokens because routing
    # changes). This is the *bandwidth* term that makes MoE economical for
    # serving relative to a same-total-param dense.
    expert_load_s = 0.0
    if phase == "decode":
        # Per token: top_k experts, three matmuls' worth of weights, sharded
        # across TP only (each rank holds 1/ep of the experts; on a hit, the
        # rank streams its TP-shard of those experts).
        per_token_expert_bytes = top_k * 3 * d_model * expert_dim_per_rank * bpe_exp
        total_expert_load_bytes = B * per_token_expert_bytes
        if shared_dim_per_rank > 0:
            total_expert_load_bytes += B * 3 * d_model * shared_dim_per_rank * bpe_exp
        expert_load_s = total_expert_load_bytes / hw.hbm_bandwidth_bytes_s
        # In decode the load typically *exceeds* the compute term — take the
        # max so the layer time reflects the bandwidth wall.
        compute_s = max(compute_s, expert_load_s)

    # --- All-to-all (dispatch + combine) ---
    # Two all-to-alls per MoE layer; each moves top_k × d_model bytes per
    # token (with capacity_factor inflation for dispatch overflow).
    volume = 2 * M * top_k * d_model * bpe_act * capacity
    alltoall_s = _moe_alltoall_cost(volume, ep_degree, hw, ep_topology)

    return {
        "compute_s": compute_s,
        "shared_expert_s": shared_compute_s,
        "alltoall_s": alltoall_s,
        "expert_load_s": expert_load_s,
        "load_balance_factor": imbalance,
    }


# =============================================================================
# v2 State layer cost
# =============================================================================

def _state_layer_cost(
    arch: ArchConfig,
    hw: HardwareConfig,
    lattice_hw: LatticeHardwareSpec,
    tp_degree: int = 1,
    phase: str = "training",
    calibration: Optional[CalibrationTable] = None,
) -> LayerBreakdown:
    """
    Compute per-layer cost for a Mamba-2 state layer.

    State replaces attention (QKV proj + attention + output proj).
    FFN still runs normally (handled separately by caller).

    Key insight: at decode, state is SRAM-resident. No KV cache,
    no L-dependent term. Only pay weight loading + small compute.

    Training/prefill: matmul-based compute (structured SSM duality) + weight loading
    Decode: weight loading only (state is SRAM-resident)

    Returns a LayerBreakdown with the state-attention-replacement cost
    in the qkv_proj_s, attention_s, and out_proj_s fields.
    """
    sc = arch.state_config
    if sc is None:
        raise ValueError("state_config is required for state layer cost computation")

    B = arch.batch_size
    S = arch.seq_len if phase != "decode" else 1
    d = arch.d_model
    prec = arch.precision
    bpe = hw.bytes_per_elem(prec)

    d_state = int(sc.get("d_state", 128))
    state_expansion = int(sc.get("state_expansion", 2))
    state_n_heads = int(sc.get("n_heads", arch.n_heads))
    state_d_head = int(sc.get("d_head", 64))
    state_prec = sc.get("state_precision", prec)
    state_bpe = hw.bytes_per_elem(state_prec)

    heads_per_gpu = state_n_heads // tp_degree
    M = B * S

    breakdown = LayerBreakdown()

    # Mamba-2 structured SSM: the computation involves:
    # 1. Input projection: (d_model) -> (state_expansion * d_model)
    #    which includes B, C, delta projections
    # 2. SSM scan: structured state space computation
    # 3. Output projection: (d_model) -> (d_model)
    #
    # We model this as matmul-equivalent projections.

    # Input projection: (M, d) -> (M, state_expansion * d / tp)
    in_proj_N = state_expansion * d // tp_degree
    t_in_proj, _, _ = _matmul_cost(M, in_proj_N, d, prec, hw, lattice_hw, tp_degree, calibration)

    # SSM scan compute (structured state space duality):
    # In training/prefill, this is a matmul-like operation
    # Compute: 2 * B * S * heads_per_gpu * d_state * state_d_head FLOPs
    if phase != "decode":
        ssm_flops = 2 * M * heads_per_gpu * d_state * state_d_head
        eff = get_tile_efficiency(M, d_state, state_d_head, state_prec, lattice_hw)
        eff = max(eff, 0.1)
        t_ssm = ssm_flops / (hw.peak_flops_s(state_prec) * eff)
        # SSM also has memory traffic for the state matrices
        ssm_bytes = M * (heads_per_gpu * d_state * state_d_head) * state_bpe * 2
        t_ssm_mem = ssm_bytes / hw.hbm_bandwidth_bytes_s
        t_ssm = max(t_ssm, t_ssm_mem)
    else:
        # Decode: state is SRAM-resident, no KV cache, no L-dependent term
        # Only pay weight loading for the SSM parameters
        # State update is a small vector operation (d_state x d_head per head)
        ssm_weight_bytes = heads_per_gpu * d_state * state_d_head * state_bpe
        t_ssm = ssm_weight_bytes / hw.hbm_bandwidth_bytes_s
        # Plus a tiny compute term for the state update
        ssm_flops = 2 * B * heads_per_gpu * d_state * state_d_head
        t_ssm_compute = ssm_flops / max(hw.peak_flops_s(state_prec), 1.0)
        t_ssm = max(t_ssm, t_ssm_compute)

    # Output projection: (M, d / tp) -> (M, d)
    out_K = d // tp_degree
    t_out, _, _ = _matmul_cost(M, d, out_K, prec, hw, lattice_hw, tp_degree, calibration)

    if phase == "decode":
        # Decode: dominated by weight loading. No KV cache. No L-dependent term.
        # Input projection weight load
        in_proj_weight_bytes = d * in_proj_N * state_bpe
        t_in_proj_decode = in_proj_weight_bytes / hw.hbm_bandwidth_bytes_s
        # Small compute for the projection
        t_in_proj_decode = max(t_in_proj_decode, t_in_proj)

        # Output projection weight load
        out_weight_bytes = out_K * d * state_bpe
        t_out_decode = out_weight_bytes / hw.hbm_bandwidth_bytes_s
        t_out_decode = max(t_out_decode, t_out)

        breakdown.qkv_proj_s = t_in_proj_decode
        breakdown.attention_s = t_ssm
        breakdown.out_proj_s = t_out_decode
    else:
        breakdown.qkv_proj_s = t_in_proj
        breakdown.attention_s = t_ssm
        breakdown.out_proj_s = t_out

    return breakdown


def compute_crossover_seq_len(
    arch: ArchConfig,
    hw: HardwareConfig,
    tp_degree: int = 1,
) -> float:
    """
    Compute L* where state decode cost equals attention decode cost.

    At decode:
    - Attention: KV cache load = 2 * B * n_kv_heads_per_gpu * L * d_head * kv_bpe / HBM_BW
    - State: weight load (L-independent) = state_weight_bytes / HBM_BW

    L* = state_weight_bytes / (2 * B * n_kv_heads_per_gpu * d_head * kv_bpe)

    Above L*, state layers are cheaper at decode time.
    """
    sc = arch.state_config
    if sc is None:
        return float("inf")

    d = arch.d_model
    B = arch.batch_size
    kv_bpe = {"bf16": 2, "fp8": 1, "fp4": 0.5, "int8": 1}.get(arch.kv_precision, 2)
    state_prec = sc.get("state_precision", arch.precision)
    state_bpe = {"bf16": 2, "fp8": 1, "fp4": 0.5, "int8": 1}.get(state_prec, 2)

    d_state = int(sc.get("d_state", 128))
    state_expansion = int(sc.get("state_expansion", 2))
    state_n_heads = int(sc.get("n_heads", arch.n_heads))
    state_d_head = int(sc.get("d_head", 64))

    heads_per_gpu = state_n_heads // tp_degree
    kv_heads_per_gpu = max(1, arch.n_kv_heads // tp_degree)

    # State weight bytes per layer (loaded at decode):
    # Input projection: d * state_expansion * d / tp
    # SSM parameters: heads_per_gpu * d_state * state_d_head
    # Output projection: d/tp * d
    in_proj_bytes = d * (state_expansion * d // tp_degree) * state_bpe
    ssm_bytes = heads_per_gpu * d_state * state_d_head * state_bpe
    out_proj_bytes = (d // tp_degree) * d * state_bpe
    total_state_bytes = in_proj_bytes + ssm_bytes + out_proj_bytes

    # Attention KV cache load per unit of L:
    # 2 * B * kv_heads_per_gpu * d_head * kv_bpe (per increment of L by 1)
    # v1-fix MLA: for MLA, the per-token KV is the latent + d_rope (not
    # 2× n_kv × d_head), and it's not sharded across TP because the latent
    # is shared across heads. Use the model's helper for accuracy.
    if arch.attention_type == "mla" and arch.mla_kv_latent_dim:
        kv_load_per_L = B * arch.kv_bytes_per_token_per_layer()
    else:
        kv_load_per_L = 2 * B * kv_heads_per_gpu * arch.d_head * kv_bpe

    if kv_load_per_L <= 0:
        return float("inf")

    # Also need attention weight bytes to compare fairly:
    # QKV proj weights: d * (heads_per_gpu + 2*kv_heads_per_gpu) * d_head * bpe
    # Output proj weights: heads_per_gpu * d_head * d * bpe
    attn_heads_per_gpu = arch.n_heads // tp_degree
    attn_bpe = {"bf16": 2, "fp8": 1, "fp4": 0.5, "int8": 1}.get(arch.precision, 2)
    qkv_weight_bytes = d * (attn_heads_per_gpu + 2 * kv_heads_per_gpu) * arch.d_head * attn_bpe
    out_weight_bytes = attn_heads_per_gpu * arch.d_head * d * attn_bpe
    attn_fixed_bytes = qkv_weight_bytes + out_weight_bytes

    # L* where: attn_fixed_bytes + kv_load_per_L * L = total_state_bytes
    # So: L* = (total_state_bytes - attn_fixed_bytes) / kv_load_per_L
    # If total_state_bytes < attn_fixed_bytes, state is always cheaper (L*=0)
    if total_state_bytes <= attn_fixed_bytes:
        return 0.0

    return (total_state_bytes - attn_fixed_bytes) / kv_load_per_L


# =============================================================================
# Memory footprint estimator
# =============================================================================

def estimate_memory_per_gpu(
    arch: ArchConfig,
    tp_degree: int = 1,
    pp_degree: int = 1,
    include_kv_cache: bool = True,
    kv_cache_len: int = 2048,
    ep_degree: int = 1,
) -> float:
    """Estimate memory footprint per GPU in bytes.

    Dense path matches v0 behavior.

    MoE path: subtracts the dense-FFN contribution implicit in the v0
    estimate_params call and replaces it with the MoE expert + shared
    contribution sharded by both TP and EP:

        expert_weights_per_gpu = n_experts * 3 * d_model * expert_dim
                                  * bpe_expert
                                  / (tp_degree * ep_degree)
        shared_weights_per_gpu = 3 * d_model * shared_dim
                                  * bpe_expert / tp_degree   (replicated over EP)
    """
    bpe = {"bf16": 2, "fp8": 1, "fp4": 0.5, "int8": 1, "tf32": 4}.get(arch.precision, 2)
    kv_bpe = {"bf16": 2, "fp8": 1, "fp4": 0.5, "int8": 1}.get(arch.kv_precision, 2)

    # Baseline model parameters (dense estimate).
    total_params = estimate_params(
        arch.d_model, arch.n_heads, arch.d_head, arch.ffn_dim,
        arch.n_layers, arch.n_kv_heads, arch.vocab_size
    )
    layers_per_stage = arch.n_layers // max(pp_degree, 1)
    params_this_stage = total_params * layers_per_stage / arch.n_layers
    model_bytes = params_this_stage * bpe / tp_degree

    # MoE FFN adjustment: swap the dense-FFN contribution for the MoE one.
    if arch.moe_config is not None:
        moe_cfg = arch.moe_config
        n_experts = int(moe_cfg["n_experts"])
        expert_dim = int(moe_cfg["expert_dim"])
        shared_block = moe_cfg.get("shared_expert")
        shared_dim = int(shared_block["ffn_dim"]) if shared_block else 0
        expert_prec = moe_cfg.get("precision", arch.precision)
        bpe_exp = {"bf16": 2, "fp8": 1, "fp4": 0.5, "int8": 1, "tf32": 4}.get(expert_prec, 2)

        # v1-fix Part B: only the MoE-layer subset of the stack gets the
        # dense→MoE swap. The first n_dense_ffn_layers keep their dense FFN
        # bytes (already baked in by estimate_params and not subtracted).
        n_dense = max(0, min(int(getattr(arch, "n_dense_ffn_layers", 0)), arch.n_layers))
        n_moe = max(0, arch.n_layers - n_dense)
        moe_layers_per_stage = max(0, layers_per_stage - int(n_dense * layers_per_stage / max(1, arch.n_layers)))

        # Remove the dense-FFN bytes for the MoE subset only.
        dense_ffn_params_per_layer = 3 * arch.d_model * arch.ffn_dim
        dense_ffn_bytes_moe_subset = dense_ffn_params_per_layer * moe_layers_per_stage * bpe / tp_degree
        model_bytes -= dense_ffn_bytes_moe_subset

        # Add MoE expert weights only for the MoE subset (sharded by TP and EP).
        expert_params_per_layer = n_experts * 3 * arch.d_model * expert_dim
        expert_bytes = (expert_params_per_layer * moe_layers_per_stage * bpe_exp
                        / (tp_degree * max(ep_degree, 1)))
        model_bytes += expert_bytes

        # Add shared-expert weights (MoE subset only; sharded by TP, replicated across EP).
        if shared_dim > 0:
            shared_params_per_layer = 3 * arch.d_model * shared_dim
            shared_bytes = shared_params_per_layer * moe_layers_per_stage * bpe_exp / tp_degree
            model_bytes += shared_bytes

    # KV cache
    kv_bytes = 0
    if include_kv_cache:
        # Per layer: 2 (K+V) × n_kv_heads × d_head × seq_len × batch × bytes
        # v1-fix MLA: when type=mla, KV is a single compressed latent + RoPE
        # key (not 2× n_kv_heads × d_head). kv_bytes_per_token_per_layer
        # returns the right per-token quantity.
        kv_per_token = arch.kv_bytes_per_token_per_layer(kv_cache_len)
        kv_per_layer = kv_per_token * kv_cache_len * arch.batch_size
        # v1-fix YOCO: cross-layer KV sharing — only K layers carry their
        # own KV. Cuts kv_bytes by K/n_layers in the dense memory estimator.
        yoco_k = int(getattr(arch, "yoco_n_self_attn_layers", 0) or 0)
        if 0 < yoco_k < arch.n_layers:
            effective_kv_layers = max(1, int(yoco_k * layers_per_stage / arch.n_layers + 0.5))
        else:
            effective_kv_layers = layers_per_stage
        kv_bytes = kv_per_layer * effective_kv_layers / tp_degree

    # Activations (rough: batch x seq x d_model x ~10 tensors per layer)
    act_bytes = arch.batch_size * arch.seq_len * arch.d_model * bpe * 10

    return model_bytes + kv_bytes + act_bytes


def estimate_memory_per_gpu_hybrid(
    arch: ArchConfig,
    tp_degree: int = 1,
    pp_degree: int = 1,
    include_kv_cache: bool = True,
    kv_cache_len: int = 2048,
    ep_degree: int = 1,
) -> float:
    """Estimate memory footprint for hybrid attention/state architectures.

    KV cache is only allocated for attention layers.
    State layers contribute SSM projection weights but no KV cache.
    """
    layer_types = arch.layer_type_list or (["attention"] * arch.n_layers)
    n_attn_layers = sum(1 for lt in layer_types if lt == "attention")
    n_state_layers = sum(1 for lt in layer_types if lt == "state")

    bpe = {"bf16": 2, "fp8": 1, "fp4": 0.5, "int8": 1, "tf32": 4}.get(arch.precision, 2)
    kv_bpe = {"bf16": 2, "fp8": 1, "fp4": 0.5, "int8": 1}.get(arch.kv_precision, 2)

    layers_per_stage = arch.n_layers // max(pp_degree, 1)

    # Count attention and state layers in this pipeline stage
    # (Simplified: assume uniform distribution across stages)
    attn_frac = n_attn_layers / max(1, arch.n_layers)
    state_frac = n_state_layers / max(1, arch.n_layers)
    attn_layers_this_stage = int(attn_frac * layers_per_stage + 0.5)
    state_layers_this_stage = layers_per_stage - attn_layers_this_stage

    # Attention layer weights per layer
    q_params = arch.d_model * arch.d_head * arch.n_heads
    kv_params = 2 * arch.d_model * arch.d_head * arch.n_kv_heads
    o_params = arch.d_head * arch.n_heads * arch.d_model
    ffn_params = 3 * arch.d_model * arch.ffn_dim
    attn_per_layer = q_params + kv_params + o_params + ffn_params
    attn_bytes = attn_per_layer * attn_layers_this_stage * bpe / tp_degree

    # State layer weights per layer
    sc = arch.state_config or {}
    d_state = int(sc.get("d_state", 128))
    state_expansion = int(sc.get("state_expansion", 2))
    state_n_heads = int(sc.get("n_heads", arch.n_heads))
    state_d_head = int(sc.get("d_head", 64))
    state_prec = sc.get("state_precision", arch.precision)
    state_bpe = {"bf16": 2, "fp8": 1, "fp4": 0.5, "int8": 1, "tf32": 4}.get(state_prec, 2)

    # State replaces attention: input proj + SSM params + output proj + FFN
    state_proj_params = arch.d_model * (state_expansion * arch.d_model)
    ssm_params = state_n_heads * d_state * state_d_head
    state_out_params = arch.d_model * arch.d_model
    state_per_layer = state_proj_params + ssm_params + state_out_params + ffn_params
    state_bytes = state_per_layer * state_layers_this_stage * state_bpe / tp_degree

    # Embedding
    embed_params = 2 * arch.vocab_size * arch.d_model
    embed_bytes = embed_params * bpe / tp_degree

    # Norm params
    norm_bytes = 2 * arch.d_model * layers_per_stage * bpe

    model_bytes = attn_bytes + state_bytes + embed_bytes + norm_bytes

    # KV cache: only for attention layers
    kv_bytes = 0
    if include_kv_cache:
        # v1-fix MLA: MLA caches one compressed latent + d_rope key, not
        # 2× n_kv × d_head. The helper handles both attention types.
        kv_per_token = arch.kv_bytes_per_token_per_layer(kv_cache_len)
        kv_per_layer = kv_per_token * kv_cache_len * arch.batch_size
        # v1-fix YOCO: cross-layer KV sharing — only K layers keep their own
        # KV. Cuts kv_bytes by K/n_layers.
        yoco_k = int(getattr(arch, "yoco_n_self_attn_layers", 0) or 0)
        if 0 < yoco_k < arch.n_layers:
            effective_kv_layers = max(1, int(yoco_k * layers_per_stage / arch.n_layers + 0.5))
        else:
            effective_kv_layers = attn_layers_this_stage
        kv_bytes = kv_per_layer * effective_kv_layers / tp_degree

    # Activations
    act_bytes = arch.batch_size * arch.seq_len * arch.d_model * bpe * 10

    return model_bytes + kv_bytes + act_bytes


# =============================================================================
# Core throughput function
# =============================================================================

def compute_layer_time(
    arch: ArchConfig,
    hw: HardwareConfig,
    lattice_hw: LatticeHardwareSpec,
    tp_degree: int = 1,
    phase: str = "training",  # "training" | "prefill" | "decode"
    kv_cache_len: int = 0,    # for decode phase
    calibration: Optional[CalibrationTable] = None,
    ep_degree: int = 1,                # v1: expert-parallel degree (ignored if dense)
    ep_topology: str = "single_axis",  # v1: TPU torus axis layout
    layer_type: str = "attention",     # v2: "attention" | "state"
) -> LayerBreakdown:
    """
    Compute per-layer time for a given phase.

    Training / prefill: batch x seq matmuls, fused attention.
    Decode: batch x 1 matmuls, KV cache load dominates.

    v2: layer_type="state" branches to state layer cost (Mamba-2).
    State layers have NO KV cache, NO L-dependent decode term.
    """
    B = arch.batch_size
    S = arch.seq_len if phase != "decode" else 1
    d = arch.d_model
    dh = arch.d_head
    nh = arch.n_heads
    nkv = arch.n_kv_heads
    ffn = arch.ffn_dim
    prec = arch.precision

    heads_per_gpu = nh // tp_degree
    kv_heads_per_gpu = max(1, nkv // tp_degree)

    # v2: branch on layer type
    if layer_type == "state" and arch.state_config is not None:
        # State layer: compute state cost (replaces attention) + FFN
        state_bd = _state_layer_cost(arch, hw, lattice_hw, tp_degree, phase, calibration)
        breakdown = LayerBreakdown()
        breakdown.qkv_proj_s = state_bd.qkv_proj_s
        breakdown.attention_s = state_bd.attention_s
        breakdown.out_proj_s = state_bd.out_proj_s

        # FFN still runs (state replaces attention, not FFN)
        M = B * S
        if arch.moe_config is None:
            ffn_per_gpu = ffn // tp_degree
            if arch.ffn_type == "swiglu":
                t_up, _, _ = _matmul_cost(M, ffn_per_gpu, d, prec, hw, lattice_hw, tp_degree, calibration)
                t_gate, _, _ = _matmul_cost(M, ffn_per_gpu, d, prec, hw, lattice_hw, tp_degree, calibration)
                breakdown.ffn_up_s = t_up + t_gate
            else:
                t_up, _, _ = _matmul_cost(M, ffn_per_gpu, d, prec, hw, lattice_hw, tp_degree, calibration)
                breakdown.ffn_up_s = t_up
            t_down, _, _ = _matmul_cost(M, d, ffn_per_gpu, prec, hw, lattice_hw, tp_degree, calibration)
            breakdown.ffn_down_s = t_down
        else:
            expert_prec = arch.moe_config.get("precision", prec)
            moe_cost = _moe_ffn_cost(
                B=B, S=S, d_model=d,
                moe_cfg=arch.moe_config,
                ep_degree=ep_degree,
                tp_degree=tp_degree,
                activation_precision=prec,
                expert_precision=expert_prec,
                hw=hw, lattice_hw=lattice_hw,
                phase=phase,
                calibration=calibration,
                ep_topology=ep_topology,
                imbalance=getattr(arch, "worst_case_imbalance_factor", 1.0),
            )

            # v1-fix Part B: blend dense + MoE FFN costs by layer-mix
            # fraction when n_dense_ffn_layers > 0.
            n_dense = max(0, min(int(getattr(arch, "n_dense_ffn_layers", 0)), arch.n_layers))
            n_moe = max(0, arch.n_layers - n_dense)
            if n_dense > 0 and n_moe > 0:
                ffn_per_gpu = ffn // tp_degree
                t_up_d, _, _ = _matmul_cost(M, ffn_per_gpu, d, prec, hw, lattice_hw, tp_degree, calibration)
                t_gate_d, _, _ = _matmul_cost(M, ffn_per_gpu, d, prec, hw, lattice_hw, tp_degree, calibration)
                t_down_d, _, _ = _matmul_cost(M, d, ffn_per_gpu, prec, hw, lattice_hw, tp_degree, calibration)
                dense_layer_s = t_up_d + t_gate_d + t_down_d
                w_d = n_dense / arch.n_layers
                w_m = n_moe / arch.n_layers
                breakdown.ffn_up_s = w_m * moe_cost["compute_s"] + w_d * dense_layer_s
                breakdown.ffn_down_s = 0.0
                breakdown.shared_expert_s = w_m * moe_cost["shared_expert_s"]
                breakdown.alltoall_s = w_m * moe_cost["alltoall_s"]
                breakdown.expert_load_s = w_m * moe_cost["expert_load_s"]
                breakdown.load_balance_factor = moe_cost["load_balance_factor"]
            else:
                breakdown.ffn_up_s = moe_cost["compute_s"]
                breakdown.ffn_down_s = 0.0
                breakdown.shared_expert_s = moe_cost["shared_expert_s"]
                breakdown.alltoall_s = moe_cost["alltoall_s"]
                breakdown.expert_load_s = moe_cost["expert_load_s"]
                breakdown.load_balance_factor = moe_cost["load_balance_factor"]

        # Membound ops and allreduce
        breakdown.membound_ops_s = _membound_ops_cost(B, S if phase != "decode" else 1, d, prec, hw)
        breakdown.allreduce_s = _allreduce_cost(B, S if phase != "decode" else 1, d, prec, hw, tp_degree)

        # Aggregate
        compute_total = (breakdown.qkv_proj_s + breakdown.attention_s +
                        breakdown.out_proj_s + breakdown.ffn_up_s + breakdown.ffn_down_s)
        memory_total = breakdown.membound_ops_s
        comm_total = breakdown.allreduce_s + breakdown.alltoall_s

        breakdown.compute_s = compute_total
        breakdown.memory_s = memory_total
        breakdown.communication_s = comm_total
        breakdown.total_s = compute_total + memory_total + comm_total

        if compute_total >= max(memory_total, comm_total):
            breakdown.bottleneck = "compute"
        elif comm_total >= memory_total:
            breakdown.bottleneck = "communication"
        else:
            breakdown.bottleneck = "memory"

        return breakdown

    # --- Original attention layer path (v0 behavior, unchanged) ---
    breakdown = LayerBreakdown()

    # Effective M dimension for matmuls
    M = B * S  # tokens in this step

    # --- QKV projection ---
    # Q: (M, d) × (d, nh*dh/tp) -> three such matmuls (Q, K, V separately or fused)
    # Fused: (M, d) × (d, (nh + 2*nkv)*dh / tp)
    qkv_N = (heads_per_gpu + 2 * kv_heads_per_gpu) * dh
    t_qkv, _, _ = _matmul_cost(M, qkv_N, d, prec, hw, lattice_hw, tp_degree, calibration)
    breakdown.qkv_proj_s = t_qkv

    # --- Attention ---
    if phase == "decode":
        # Decode attention: query (B,1,nh,dh) × keys (B,L,nkv,dh)
        # This is KV-cache-bandwidth-bound, not compute-bound
        kv_bpe = {"bf16": 2, "fp8": 1, "fp4": 0.5, "int8": 1}.get(arch.kv_precision, 2)
        L = kv_cache_len

        # KV cache load per layer: K + V, each (B, nkv_per_gpu, L, dh).
        # v1-fix MLA: MLA caches a single shared latent (c_kv + d_rope) per
        # token, NOT 2 × n_kv × d_head. The latent is not sharded across TP
        # because it's shared across query heads.
        if arch.attention_type == "mla" and arch.mla_kv_latent_dim:
            kv_bytes_per_layer = B * L * arch.kv_bytes_per_token_per_layer(L)
        elif arch.attention_type == "nsa" and arch.nsa_window_size:
            kv_bytes_per_layer = B * L * arch.kv_bytes_per_token_per_layer(L)
        else:
            kv_bytes_per_layer = 2 * B * kv_heads_per_gpu * L * dh * kv_bpe
        # v1-fix YOCO: only K of N layers actually read their own KV during
        # decode; the rest cross-attend to a single shared cache. Amortize
        # the bandwidth cost across the stack so decode TBT shrinks by K/N.
        yoco_k = int(getattr(arch, "yoco_n_self_attn_layers", 0) or 0)
        if 0 < yoco_k < arch.n_layers:
            kv_bytes_per_layer *= yoco_k / arch.n_layers
        t_kv_load = kv_bytes_per_layer / hw.hbm_bandwidth_bytes_s

        # Small compute: (B, heads_per_gpu, 1, dh) × (B, heads_per_gpu, dh, L)
        # GQA: each kv head serves multiple q heads, but compute is still small
        attn_flops = 2 * B * heads_per_gpu * 1 * L * dh * 2
        fused_eff = hw.fused_attention_efficiency.get(prec,
                    hw.fused_attention_efficiency.get("bf16", 0.75))
        t_attn_compute = attn_flops / (hw.peak_flops_s(prec) * fused_eff)
        breakdown.attention_s = max(t_kv_load, t_attn_compute)
    else:
        # Training / prefill: full S×S attention
        breakdown.attention_s = _attention_cost(
            B, arch.seq_len, nh, dh, nkv, prec, hw, tp_degree
        )

    # --- Output projection ---
    # (M, heads_per_gpu * dh) × (heads_per_gpu * dh, d)
    out_K = heads_per_gpu * dh
    t_out, _, _ = _matmul_cost(M, d, out_K, prec, hw, lattice_hw, tp_degree, calibration)
    breakdown.out_proj_s = t_out

    # v1-fix 2:4 sparsity: per-component speedup factor. NVIDIA tensor cores
    # give 2× on 2:4-sparsified matmuls; TPU/Trainium have no native sparse
    # path so the speedup is gated by vendor.
    sparsity = getattr(arch, "sparsity_2_4", None) or {}
    sparse_speedup_nvidia = (hw.vendor == "nvidia")
    def _sparse_factor(component: str) -> float:
        if sparsity.get(component) and sparse_speedup_nvidia:
            return 0.5
        return 1.0
    breakdown.qkv_proj_s *= _sparse_factor("attn_qkv")
    breakdown.out_proj_s *= _sparse_factor("attn_o")

    # --- FFN: dense or MoE branch ---
    if arch.moe_config is None:
        # Dense path (v0 behavior, unchanged).
        ffn_per_gpu = ffn // tp_degree
        if arch.ffn_type == "swiglu":
            # SwiGLU: two parallel projections (up + gate), each (M, d) → (M, ffn/tp)
            t_up, _, _ = _matmul_cost(M, ffn_per_gpu, d, prec, hw, lattice_hw, tp_degree, calibration)
            t_gate, _, _ = _matmul_cost(M, ffn_per_gpu, d, prec, hw, lattice_hw, tp_degree, calibration)
            breakdown.ffn_up_s = t_up * _sparse_factor("ffn_up") + t_gate * _sparse_factor("ffn_gate")
        else:
            t_up, _, _ = _matmul_cost(M, ffn_per_gpu, d, prec, hw, lattice_hw, tp_degree, calibration)
            breakdown.ffn_up_s = t_up * _sparse_factor("ffn_up")

        # --- FFN down ---
        t_down, _, _ = _matmul_cost(M, d, ffn_per_gpu, prec, hw, lattice_hw, tp_degree, calibration)
        breakdown.ffn_down_s = t_down * _sparse_factor("ffn_down")
    else:
        # MoE path. ffn_up_s carries the expert-compute total; ffn_down_s is
        # left at 0 (the down matmul is rolled into compute_s by _moe_ffn_cost).
        # The shared-expert compute is recorded separately for inspection.
        expert_prec = arch.moe_config.get("precision", prec)
        moe_cost = _moe_ffn_cost(
            B=B, S=S, d_model=d,
            moe_cfg=arch.moe_config,
            ep_degree=ep_degree,
            tp_degree=tp_degree,
            activation_precision=prec,
            expert_precision=expert_prec,
            hw=hw, lattice_hw=lattice_hw,
            phase=phase,
            calibration=calibration,
            ep_topology=ep_topology,
            imbalance=getattr(arch, "worst_case_imbalance_factor", 1.0),
        )

        # v1-fix Part B: blend dense + MoE FFN costs by layer-mix fraction
        # when n_dense_ffn_layers > 0. The dense FFN compute is computed
        # here at the same B*S; the per-stack throughput multiplies by
        # n_layers downstream, so blending here gives the right total cost
        # for a mixed stack.
        n_dense = max(0, min(int(getattr(arch, "n_dense_ffn_layers", 0)), arch.n_layers))
        n_moe = max(0, arch.n_layers - n_dense)
        if n_dense > 0 and n_moe > 0:
            ffn_per_gpu = ffn // tp_degree
            t_up_d, _, _ = _matmul_cost(M, ffn_per_gpu, d, prec, hw, lattice_hw, tp_degree, calibration)
            t_gate_d, _, _ = _matmul_cost(M, ffn_per_gpu, d, prec, hw, lattice_hw, tp_degree, calibration)
            t_down_d, _, _ = _matmul_cost(M, d, ffn_per_gpu, prec, hw, lattice_hw, tp_degree, calibration)
            dense_layer_s = t_up_d + t_gate_d + t_down_d
            w_d = n_dense / arch.n_layers
            w_m = n_moe / arch.n_layers
            breakdown.ffn_up_s = w_m * moe_cost["compute_s"] + w_d * dense_layer_s
            breakdown.ffn_down_s = 0.0
            breakdown.shared_expert_s = w_m * moe_cost["shared_expert_s"]
            breakdown.alltoall_s = w_m * moe_cost["alltoall_s"]
            breakdown.expert_load_s = w_m * moe_cost["expert_load_s"]
            breakdown.load_balance_factor = moe_cost["load_balance_factor"]
        else:
            breakdown.ffn_up_s = moe_cost["compute_s"]
            breakdown.ffn_down_s = 0.0
            breakdown.shared_expert_s = moe_cost["shared_expert_s"]
            breakdown.alltoall_s = moe_cost["alltoall_s"]
            breakdown.expert_load_s = moe_cost["expert_load_s"]
            breakdown.load_balance_factor = moe_cost["load_balance_factor"]

    # --- Memory-bound ops ---
    breakdown.membound_ops_s = _membound_ops_cost(B, S if phase != "decode" else 1, d, prec, hw)

    # --- Communication ---
    breakdown.allreduce_s = _allreduce_cost(B, S if phase != "decode" else 1, d, prec, hw, tp_degree)

    # --- Aggregate ---
    compute_total = (breakdown.qkv_proj_s + breakdown.attention_s +
                     breakdown.out_proj_s + breakdown.ffn_up_s + breakdown.ffn_down_s)
    memory_total = breakdown.membound_ops_s
    # MoE all-to-all is part of communication; expert weight loading is already
    # folded into ffn_up_s (compute_s = max(compute, expert_load) in decode).
    comm_total = breakdown.allreduce_s + breakdown.alltoall_s

    breakdown.compute_s = compute_total
    breakdown.memory_s = memory_total
    breakdown.communication_s = comm_total

    # v0: no compute-communication overlap
    breakdown.total_s = compute_total + memory_total + comm_total

    # MoE-aware bottleneck identification.
    if arch.moe_config is not None and phase == "decode" and breakdown.expert_load_s > 0:
        # If expert loading is the dominant term inside ffn_up_s, surface that.
        if breakdown.expert_load_s >= 0.5 * breakdown.ffn_up_s and breakdown.ffn_up_s >= 0.5 * compute_total:
            breakdown.bottleneck = "expert_load"
        elif breakdown.alltoall_s > max(compute_total, memory_total) * 0.5:
            breakdown.bottleneck = "alltoall"
        elif compute_total >= max(memory_total, comm_total):
            breakdown.bottleneck = "compute"
        elif comm_total >= memory_total:
            breakdown.bottleneck = "communication"
        else:
            breakdown.bottleneck = "memory"
    else:
        if arch.moe_config is not None and breakdown.alltoall_s > max(compute_total, memory_total) * 0.5:
            breakdown.bottleneck = "alltoall"
        elif compute_total >= max(memory_total, comm_total):
            breakdown.bottleneck = "compute"
        elif comm_total >= memory_total:
            breakdown.bottleneck = "communication"
        else:
            breakdown.bottleneck = "memory"

    return breakdown


def compute_heterogeneous_layer_times(
    arch: ArchConfig,
    hw: HardwareConfig,
    lattice_hw: LatticeHardwareSpec,
    tp_degree: int = 1,
    phase: str = "training",
    kv_cache_len: int = 0,
    calibration: Optional[CalibrationTable] = None,
    ep_degree: int = 1,
    ep_topology: str = "single_axis",
) -> List[LayerBreakdown]:
    """
    Compute per-layer costs for a heterogeneous (hybrid) architecture.

    Returns a list of LayerBreakdown, one per layer, respecting each layer's
    type (attention or state).
    """
    layer_types = arch.layer_type_list or (["attention"] * arch.n_layers)
    results = []
    for i in range(arch.n_layers):
        lt = layer_types[i] if i < len(layer_types) else "attention"
        bd = compute_layer_time(
            arch, hw, lattice_hw, tp_degree, phase,
            kv_cache_len=kv_cache_len,
            calibration=calibration,
            ep_degree=ep_degree,
            ep_topology=ep_topology,
            layer_type=lt,
        )
        results.append(bd)
    return results


def throughput(
    arch: ArchConfig,
    hardware: str,         # "h100", "b200", "tpu_v5e", "tpu_v5p"
    tp_degree: int = 1,
    pp_degree: int = 1,
    microbatches: int = 1,
    decode_kv_len: int = 1024,
    lattice_hw_override: str = None,
    ep_degree: int = 1,                # v1 MoE: expert-parallel degree
    ep_topology: str = "single_axis",  # v1 MoE: TPU torus axis layout
) -> ThroughputResult:
    """
    T(A, H, lattice) -> ThroughputResult

    Main entry point. Computes training throughput, prefill time, and
    decode time per token for a given (architecture, hardware) pair.

    Args:
        arch: Architecture configuration.
        hardware: Hardware target name.
        tp_degree: Tensor parallelism degree.
        pp_degree: Pipeline parallelism degree.
        microbatches: Number of microbatches for pipeline parallelism.
        decode_kv_len: KV cache length for decode phase estimate.
        lattice_hw_override: Override lattice hardware name (if different from throughput hw).
    """
    hw = load_hardware(hardware)
    cal_table = load_calibration(hardware)

    # Map throughput hardware name to lattice hardware name
    lattice_hw_name = lattice_hw_override or hardware
    if lattice_hw_name not in LATTICE_HARDWARE:
        # Fallback: try closest match
        if "tpu" in lattice_hw_name:
            lattice_hw_name = "tpu_v5p" if "v5p" in lattice_hw_name else "tpu_v5e"
        else:
            lattice_hw_name = "h100"
    lattice_hw = LATTICE_HARDWARE[lattice_hw_name]

    result = ThroughputResult(
        hardware_name=hw.accelerator_name,
        precision=arch.precision,
        tp_degree=tp_degree,
        pp_degree=pp_degree,
        decode_kv_cache_length=decode_kv_len,
    )

    layers_per_stage = arch.n_layers // max(pp_degree, 1)
    cal = hw.calibration

    # Per-layer kernel launch overhead (real systems pay ~5-8μs per kernel launch)
    kernel_overhead_per_layer_s = (
        cal.get("kernel_launch_overhead_us", 5.0) *
        cal.get("kernels_per_layer", 12) * 1e-6
    )

    # Check if we have a heterogeneous (hybrid) architecture
    is_hybrid = (arch.layer_type_list is not None and
                 any(lt == "state" for lt in arch.layer_type_list) and
                 arch.state_config is not None)

    # Kernel launch overhead across all layers
    kernel_overhead_total = kernel_overhead_per_layer_s * layers_per_stage

    if is_hybrid:
        # --- Heterogeneous path: sum per-layer costs ---
        train_layers = compute_heterogeneous_layer_times(
            arch, hw, lattice_hw, tp_degree, "training",
            calibration=cal_table, ep_degree=ep_degree, ep_topology=ep_topology,
        )
        # Sum costs for layers in this pipeline stage
        # Simplified: use first layers_per_stage layers
        raw_train_s = sum(bd.total_s for bd in train_layers[:layers_per_stage])
        layer_train = train_layers[0]  # Representative for breakdown

        prefill_layers = compute_heterogeneous_layer_times(
            arch, hw, lattice_hw, tp_degree, "prefill",
            calibration=cal_table, ep_degree=ep_degree, ep_topology=ep_topology,
        )
        raw_prefill_s = sum(bd.total_s for bd in prefill_layers[:layers_per_stage])
        layer_prefill = prefill_layers[0]

        decode_layers = compute_heterogeneous_layer_times(
            arch, hw, lattice_hw, tp_degree, "decode",
            kv_cache_len=decode_kv_len, calibration=cal_table,
            ep_degree=ep_degree, ep_topology=ep_topology,
        )
        raw_decode_s = sum(bd.total_s for bd in decode_layers[:layers_per_stage])
        layer_decode = decode_layers[0]
    else:
        # --- Uniform path (v0 behavior) ---
        layer_train = compute_layer_time(arch, hw, lattice_hw, tp_degree, "training",
                                         calibration=cal_table,
                                         ep_degree=ep_degree, ep_topology=ep_topology)
        raw_train_s = layer_train.total_s * layers_per_stage

        layer_prefill = compute_layer_time(arch, hw, lattice_hw, tp_degree, "prefill",
                                           calibration=cal_table,
                                           ep_degree=ep_degree, ep_topology=ep_topology)
        raw_prefill_s = layer_prefill.total_s * layers_per_stage

        layer_decode = compute_layer_time(
            arch, hw, lattice_hw, tp_degree, "decode",
            kv_cache_len=decode_kv_len, calibration=cal_table,
            ep_degree=ep_degree, ep_topology=ep_topology,
        )
        raw_decode_s = layer_decode.total_s * layers_per_stage

    # Optimizer step: read params + gradients, write params + optimizer states
    # AdamW: ~12 bytes per param (fp32 master weights + m + v)
    total_params = estimate_params(
        arch.d_model, arch.n_heads, arch.d_head, arch.ffn_dim,
        arch.n_layers, arch.n_kv_heads, arch.vocab_size
    )
    params_per_gpu = total_params / tp_degree / max(pp_degree, 1)
    opt_bytes = params_per_gpu * cal.get("optimizer_bytes_per_param", 12)
    optimizer_step_s = opt_bytes / hw.hbm_bandwidth_bytes_s

    # Pipeline bubble
    if pp_degree > 1:
        M_micro = max(microbatches, 1)
        bubble = (pp_degree - 1) / (M_micro + pp_degree - 1)
    else:
        bubble = 0.0
    result.bubble_fraction = bubble

    # Training: forward + backward ~ 3x forward compute
    sys_eff_train = cal.get("training_system_efficiency", 0.55)
    train_step_s = (raw_train_s * 3.0 + kernel_overhead_total * 3 + optimizer_step_s) * (1 + bubble) / sys_eff_train

    # v1-fix MTP: extra training compute from MTP heads (DeepSeek-V3 §2.2).
    # Each MTP depth is a small transformer block run on the same batch,
    # roughly costing `(mtp_layers / n_layers)` of one forward+backward pass.
    # Conservative estimate: 8% per depth, capped at 20% total.
    mtp_depths = int(getattr(arch, "mtp_n_predict_depths", 0) or 0)
    mtp_layers = int(getattr(arch, "mtp_depth_n_layers", 1) or 1)
    if mtp_depths > 0:
        per_depth_overhead = min(0.20, mtp_layers / max(1, arch.n_layers))
        mtp_overhead = min(0.20, mtp_depths * per_depth_overhead)
        train_step_s *= (1 + mtp_overhead)

    # v1-fix CP: Context Parallelism — split sequence across `cp_degree` ranks.
    # Attention compute and KV memory both shrink by 1/cp; the all-gather/
    # ring-attention comm cost is (cp-1)/cp × n_layers × B × S × d_model × bpe.
    # Ulysses is roughly 2× cheaper in comm than Ring (head scatter vs ring KV).
    cp = max(1, int(getattr(arch, "cp_degree", 1) or 1))
    if cp > 1:
        # Compute & memory share of the seq-parallel work
        bpe_act = hw.bytes_per_elem(arch.precision)
        seq_bytes_per_layer = arch.batch_size * arch.seq_len * arch.d_model * bpe_act
        comm_factor = (cp - 1) / cp
        cp_method_factor = 0.5 if getattr(arch, "cp_method", "ring") == "ulysses" else 1.0
        cp_comm_bytes = comm_factor * arch.n_layers * seq_bytes_per_layer * cp_method_factor
        # All-gather over NVLink within the CP group
        nvlink_bw = hw.interconnect["intra_node_bw_gb_s"] * 1e9
        cp_comm_s = cp_comm_bytes / max(1.0, nvlink_bw)
        # CP reduces the attention compute proportionally; sequence-parallel
        # FLOP savings are folded into raw_train_s here.
        train_step_s = train_step_s / cp + cp_comm_s
    tokens_per_step = arch.batch_size * arch.seq_len
    result.training_time_per_step_s = train_step_s
    result.training_throughput_tokens_per_sec = tokens_per_step / train_step_s if train_step_s > 0 else 0

    # --- Prefill (inference) ---
    sys_eff_prefill = cal.get("prefill_system_efficiency", 0.60)
    raw_prefill = raw_prefill_s + kernel_overhead_total
    result.prefill_time_ms = raw_prefill / sys_eff_prefill * 1000

    # --- Decode ---
    sys_eff_decode = cal.get("decode_system_efficiency", 0.42)
    raw_decode = raw_decode_s + kernel_overhead_total
    result.decode_time_per_token_ms = raw_decode / sys_eff_decode * 1000

    # --- Memory ---
    if is_hybrid:
        mem_bytes = estimate_memory_per_gpu_hybrid(
            arch, tp_degree, pp_degree,
            include_kv_cache=True, kv_cache_len=decode_kv_len,
            ep_degree=ep_degree,
        )
    else:
        mem_bytes = estimate_memory_per_gpu(
            arch, tp_degree, pp_degree,
            include_kv_cache=True, kv_cache_len=decode_kv_len,
            ep_degree=ep_degree,
        )
    result.memory_footprint_per_gpu_gb = mem_bytes / (1024**3)

    # Per-layer breakdowns for all phases
    result.per_layer_breakdown = layer_train
    result.prefill_layer_breakdown = layer_prefill
    result.decode_layer_breakdown = layer_decode
    result.bottleneck = layer_train.bottleneck

    return result


# =============================================================================
# Convenience: evaluate a known architecture
# =============================================================================

def evaluate_known(
    arch_name: str,
    hardware: str = "h100",
    precision: str = "bf16",
    tp_degree: int = 1,
    pp_degree: int = 1,
    batch_size: int = 1,
    seq_len: int = 2048,
    decode_kv_len: int = 1024,
) -> ThroughputResult:
    """Evaluate a known architecture (e.g., 'Llama-2-7B') on a hardware target."""
    if arch_name not in KNOWN_ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {arch_name}. Known: {list(KNOWN_ARCHITECTURES.keys())}")

    ka = KNOWN_ARCHITECTURES[arch_name]

    # Determine n_kv_heads (GQA info for known architectures)
    gqa_map = {
        "Llama-2-7B": None,   # MHA
        "Llama-2-13B": None,  # MHA
        "Llama-2-70B": 8,     # GQA
        "Llama-3-8B": 8,      # GQA
        "Llama-3-70B": 8,     # GQA
        "Mistral-7B": 8,      # GQA
        "Gemma-2-9B": 8,      # GQA
        "Qwen3-8B": 8,        # GQA
        "Qwen3-32B": 8,       # GQA
        # MoE models (dense-equivalent n_kv_heads)
        "DeepSeek-V3": 128,   # MLA (all heads act as KV via latent decompression)
        "Kimi-K2.5": 64,      # MLA
        "GLM-5.1": 64,        # MLA
        "GPT-OSS-120B": 8,    # GQA
        "MAI-Base-1": 8,      # GQA
    }
    n_kv = gqa_map.get(arch_name) or ka["n_heads"]

    vocab_map = {
        "Llama-2-7B": 32000, "Llama-2-13B": 32000, "Llama-2-70B": 32000,
        "Llama-3-8B": 128256, "Llama-3-70B": 128256,
        "Mistral-7B": 32000,
        "Gemma-2-9B": 256000,
        "Qwen3-8B": 151936, "Qwen3-32B": 151936,
        "DeepSeek-V3": 129280, "Kimi-K2.5": 163840,
        "GLM-5.1": 154880, "GPT-OSS-120B": 201088,
        "MAI-Base-1": 141056,
    }

    arch = ArchConfig(
        d_model=ka["d_model"],
        n_layers=ka["n_layers"],
        n_heads=ka["n_heads"],
        d_head=ka["d_head"],
        n_kv_heads=n_kv,
        ffn_dim=ka["ffn_dim"],
        ffn_type="swiglu",
        vocab_size=vocab_map.get(arch_name, 32000),
        batch_size=batch_size,
        seq_len=seq_len,
        precision=precision,
        kv_precision=precision,
    )
    return throughput(arch, hardware, tp_degree, pp_degree, decode_kv_len=decode_kv_len)


# =============================================================================
# Validation harness
# =============================================================================

# Published reference throughput numbers (approximate, from public benchmarks)
# Format: (arch_name, hardware, tp, batch, seq, tokens/sec or tok/s/gpu)
REFERENCE_TRAINING = {
    # H100 training throughput — tokens/sec/GPU at given batch×seq
    ("Llama-2-7B", "h100", 1, 4, 2048):   {"tokens_per_sec_gpu": 3800, "source": "Meta training infra / MLPerf approx"},
    ("Llama-3-8B", "h100", 1, 4, 2048):   {"tokens_per_sec_gpu": 3500, "source": "Meta release / community benchmarks"},
    ("Mistral-7B", "h100", 1, 4, 2048):   {"tokens_per_sec_gpu": 3600, "source": "Community benchmarks / Mistral docs"},
}

REFERENCE_DECODE = {
    # H100 decode throughput — tokens/sec at batch=1, seq=1, kv_len=1024
    ("Llama-2-7B", "h100", 1, 1, 1024):    {"tokens_per_sec": 85, "source": "vLLM benchmarks H100"},
    ("Llama-3-8B", "h100", 1, 1, 1024):    {"tokens_per_sec": 75, "source": "vLLM / TensorRT-LLM benchmarks"},
    ("Mistral-7B", "h100", 1, 1, 1024):    {"tokens_per_sec": 80, "source": "vLLM benchmarks"},
}


def run_validation(verbose: bool = True) -> dict:
    """
    Run validation against reference throughput numbers.
    Returns dict of results with predicted/measured ratios.
    """
    results = {"training": [], "decode": []}

    # Training validation
    for (arch_name, hw, tp, batch, seq), ref in REFERENCE_TRAINING.items():
        try:
            r = evaluate_known(arch_name, hw, "bf16", tp, 1, batch, seq)
            predicted = r.training_throughput_tokens_per_sec
            measured = ref["tokens_per_sec_gpu"]
            ratio = predicted / measured if measured > 0 else 0
            error_pct = abs(ratio - 1.0) * 100
            status = "PASS" if error_pct <= 25 else "FAIL"

            entry = {
                "arch": arch_name, "hardware": hw, "tp": tp,
                "batch": batch, "seq": seq,
                "predicted": round(predicted, 1),
                "measured": measured,
                "ratio": round(ratio, 3),
                "error_pct": round(error_pct, 1),
                "status": status,
                "source": ref["source"],
            }
            results["training"].append(entry)

            if verbose:
                print(f"  [TRAIN {status}] {arch_name} on {hw} TP={tp} B={batch} S={seq}: "
                      f"predicted={predicted:.0f} measured={measured} ratio={ratio:.3f} error={error_pct:.1f}%")
        except Exception as e:
            if verbose:
                print(f"  [ERROR] {arch_name} on {hw}: {e}")

    # Decode validation
    for (arch_name, hw, tp, batch, kv_len), ref in REFERENCE_DECODE.items():
        try:
            r = evaluate_known(arch_name, hw, "bf16", tp, 1, batch, 1, decode_kv_len=kv_len)
            predicted_tbt = r.decode_time_per_token_ms
            predicted_tps = 1000 / predicted_tbt if predicted_tbt > 0 else 0
            measured = ref["tokens_per_sec"]
            ratio = predicted_tps / measured if measured > 0 else 0
            error_pct = abs(ratio - 1.0) * 100
            status = "PASS" if error_pct <= 25 else "FAIL"

            entry = {
                "arch": arch_name, "hardware": hw, "tp": tp,
                "batch": batch, "kv_len": kv_len,
                "predicted_tps": round(predicted_tps, 1),
                "predicted_tbt_ms": round(predicted_tbt, 2),
                "measured_tps": measured,
                "ratio": round(ratio, 3),
                "error_pct": round(error_pct, 1),
                "status": status,
                "source": ref["source"],
            }
            results["decode"].append(entry)

            if verbose:
                print(f"  [DECODE {status}] {arch_name} on {hw} TP={tp} B={batch} KV={kv_len}: "
                      f"predicted={predicted_tps:.0f} tok/s measured={measured} ratio={ratio:.3f} error={error_pct:.1f}%")
        except Exception as e:
            if verbose:
                print(f"  [ERROR] {arch_name} on {hw}: {e}")

    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    print("=" * 70)
    print("Throughput Model v0 — Validation Run")
    print("=" * 70)

    # Run all known architectures across all hardware
    for hw_name in ["h100", "b200", "tpu_v5e", "tpu_v5p"]:
        print(f"\n{'='*70}")
        print(f"Hardware: {hw_name}")
        print(f"{'='*70}")

        for arch_name in KNOWN_ARCHITECTURES:
            try:
                r = evaluate_known(arch_name, hw_name, "bf16", tp_degree=1, batch_size=1, seq_len=2048)
                decode_tps = 1000 / r.decode_time_per_token_ms if r.decode_time_per_token_ms > 0 else 0
                print(f"  {arch_name:20s} | train={r.training_throughput_tokens_per_sec:8.0f} tok/s | "
                      f"prefill={r.prefill_time_ms:7.1f}ms | decode={decode_tps:6.0f} tok/s | "
                      f"mem={r.memory_footprint_per_gpu_gb:5.1f}GB | bottleneck={r.bottleneck}")
            except Exception as e:
                print(f"  {arch_name:20s} | ERROR: {e}")

    print(f"\n{'='*70}")
    print("Validation against reference numbers (H100):")
    print(f"{'='*70}")
    run_validation(verbose=True)
