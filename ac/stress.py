"""
Throughput Stress Reporter — 10-axis named utilization vector.

Reads intermediate quantities that the v0 throughput model already computes
(LayerBreakdown fields + arch shape) and exposes them as a fixed-schema
StressVector. Formulas mirror those in v0-throughput/throughput_model.py;
do NOT change formulas here without changing them there first.

Axes (per instruction §2.1):
    hbm_bw_decode   — (KV_load + weight_load + state_load) / (HBM_BW × decode_step_time)
    hbm_bw_prefill  — (weight_load + activation_traffic) / (HBM_BW × prefill_time)
    hbm_capacity    — (weights + KV + activations) / HBM_capacity
    kv_footprint    — KV_cache_steady_state / HBM_capacity
    tc_util_prefill — prefill_FLOPS_useful / (peak_FLOPS × prefill_time)
    tc_util_decode  — decode_FLOPS_useful / (peak_FLOPS × decode_time)
    sram_tile_fit   — 1 - achievable_tile / ideal_tile (dominant matmul)
    all_reduce      — TP_all_reduce_bytes / (NVLink_BW × layer_time)
    all_to_all      — EP_all_to_all_bytes / (NVLink_BW × layer_time)
    training_mem    — (weights + activations + grad + optimizer) / HBM_capacity

Severity bands (per §2.2):
    [0, 0.7)    relaxed
    [0.7, 0.9)  loaded
    [0.9, 1.0)  pressured
    [1.0, 1.2)  binding
    [1.2, ∞)    violated
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# --- repo path bootstrap so we can import sibling v0 modules -----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from throughput_model import (  # noqa: E402
    ArchConfig,
    HardwareConfig,
    LayerBreakdown,
    ThroughputResult,
    compute_layer_time,
    compute_heterogeneous_layer_times,
    estimate_memory_per_gpu,
    estimate_memory_per_gpu_hybrid,
    load_calibration,
    load_hardware,
    throughput,
)
from lattice_engine import (  # noqa: E402
    HARDWARE as LATTICE_HARDWARE,
    estimate_params,
    matmul_tile_utilization,
)


def _lattice_hw_for(hw_name: str):
    """Pick the lattice spec for a throughput-style hardware name."""
    if hw_name in LATTICE_HARDWARE:
        return LATTICE_HARDWARE[hw_name]
    return LATTICE_HARDWARE["h100"]


# =============================================================================
# Bands
# =============================================================================

_BANDS = [
    (0.7, "relaxed"),
    (0.9, "loaded"),
    (1.0, "pressured"),
    (1.2, "binding"),
    (float("inf"), "violated"),
]


def severity_band(v: float) -> str:
    """Map a utilization fraction to a named severity band."""
    for cutoff, name in _BANDS:
        if v < cutoff:
            return name
    return "violated"


# Axes whose severity ≥ "pressured" count as binding for the ranker (§5.2).
PRESSURED_OR_WORSE = {"pressured", "binding", "violated"}


# =============================================================================
# Workload spec
# =============================================================================

@dataclass
class Workload:
    """Workload context for a stress computation.

    A single architecture has different stress profiles at batch=1 chat vs.
    batch=64 batch inference. Always carry the workload with the vector.
    """
    batch_size: int = 1
    prefill_seq_len: int = 2048
    decode_kv_len: int = 2048
    phase: str = "decode"  # "prefill" | "decode" | "training" | "serving_mixed"

    def workload_id(self) -> str:
        """Human-readable workload identifier, e.g. ``decode-b1-prefill2048-kv4096``.

        Used both as a stable key for KG entries and as the user-facing label
        in ac-stress output. Previously a hex hash, which carried no
        information for a single-shot CLI invocation; the slug form lets the
        reader verify the workload without parsing JSON.
        """
        return (
            f"{self.phase}-b{int(self.batch_size)}"
            f"-prefill{int(self.prefill_seq_len)}"
            f"-kv{int(self.decode_kv_len)}"
        )


# =============================================================================
# StressVector dataclass
# =============================================================================

# Ordered list of the 10 axes — fixed in v1.
STRESS_AXES = (
    "hbm_bw_decode",
    "hbm_bw_prefill",
    "hbm_capacity",
    "kv_footprint",
    "tc_util_prefill",
    "tc_util_decode",
    "sram_tile_fit",
    "all_reduce",
    "all_to_all",
    "training_mem",
)


def active_axes_for_phase(phase: str) -> tuple:
    """Axes considered binding for the requested workload phase."""
    phase_key = str(phase or "decode").lower()
    if phase_key == "decode":
        return (
            "hbm_bw_decode",
            "hbm_capacity",
            "kv_footprint",
            "tc_util_decode",
            "sram_tile_fit",
            "all_reduce",
            "all_to_all",
        )
    if phase_key == "prefill":
        return (
            "hbm_bw_prefill",
            "hbm_capacity",
            "tc_util_prefill",
            "sram_tile_fit",
            "all_reduce",
            "all_to_all",
        )
    if phase_key == "training":
        return (
            "hbm_bw_prefill",
            "hbm_capacity",
            "tc_util_prefill",
            "sram_tile_fit",
            "all_reduce",
            "all_to_all",
            "training_mem",
        )
    if phase_key == "serving_mixed":
        return (
            "hbm_bw_decode",
            "hbm_bw_prefill",
            "hbm_capacity",
            "kv_footprint",
            "tc_util_prefill",
            "tc_util_decode",
            "sram_tile_fit",
            "all_reduce",
            "all_to_all",
        )
    return STRESS_AXES


@dataclass
class StressVector:
    schema_version: int = 1

    # Context (per instruction §8 data model)
    arch_name: str = ""
    hardware_id: str = ""
    workload_id: str = ""
    phase: str = "decode"

    # The 10 axes (utilization fractions, canonical 0–1, values >1 allowed)
    hbm_bw_decode: float = 0.0
    hbm_bw_prefill: float = 0.0
    hbm_capacity: float = 0.0
    kv_footprint: float = 0.0
    tc_util_prefill: float = 0.0
    tc_util_decode: float = 0.0
    sram_tile_fit: float = 0.0
    all_reduce: float = 0.0
    all_to_all: float = 0.0
    training_mem: float = 0.0

    # Derived (cached for query convenience per §8)
    binding_axes: List[str] = field(default_factory=list)

    # Diagnostics — raw numerators/denominators for inspection
    intermediates: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def axis_value(self, axis: str) -> float:
        if axis not in STRESS_AXES:
            raise KeyError(f"unknown stress axis: {axis!r}")
        return float(getattr(self, axis))

    def band(self, axis: str) -> str:
        return severity_band(self.axis_value(axis))

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["bands"] = {a: self.band(a) for a in STRESS_AXES}
        d["active_axes"] = list(active_axes_for_phase(self.phase))
        return d

    def pretty(self) -> str:
        """Human-readable summary for CLI use.

        Inactive-axis handling: a phase-inactive axis (e.g. `training_mem`
        when phase=decode) gets shown as `inactive` instead of its raw
        band. Previously we printed `binding (117% — over) [inactive for
        decode]`, which is internally contradictory: an axis cannot be
        binding AND inactive simultaneously, and a quick scanner will
        latch onto "binding" before reading the bracket. Demote inactive
        rows to the `inactive` label and tuck the underlying number into
        a parenthetical so power users can still see what would happen if
        the axis became live.
        """
        rows = []
        active_axes = set(active_axes_for_phase(self.phase))
        for axis in STRESS_AXES:
            v = self.axis_value(axis)
            b = self.band(axis)
            if axis not in active_axes:
                # Demote to `inactive`; surface the raw band only as
                # parenthetical context, never as the headline word.
                shown_band = f"inactive ({b} {v*100:.0f}% if active)"
            else:
                shown_band = b
                if v >= 1.0:
                    shown_band = f"{shown_band} ({v*100:.0f}% of peak — over)"
            marker = "*" if axis in self.binding_axes else " "
            rows.append(f"  {marker} {axis:18s} {v:6.3f}  {shown_band}")
        binding = ", ".join(self.binding_axes) or "(none)"
        return (
            f"StressVector  arch={self.arch_name}  hw={self.hardware_id}  "
            f"phase={self.phase}  workload={self.workload_id}\n"
            + "\n".join(rows)
            + f"\n  binding: {binding}"
            + "\n  bands: relaxed [0,0.7), loaded [0.7,0.9), "
              "pressured [0.9,1.0), binding [1.0,1.2), violated [1.2,∞), "
              "inactive (axis off this phase)"
        )


# =============================================================================
# Volume / FLOPS helpers (formulas mirror throughput_model.py)
# =============================================================================

_BPE_TABLE = {"bf16": 2, "fp16": 2, "fp8": 1, "fp4": 0.5,
              "int8": 1, "int4": 0.5, "tf32": 4}


def _bpe(prec: str) -> float:
    return _BPE_TABLE.get(prec, 2)


def _kv_cache_bytes_total(arch: ArchConfig, kv_len: int, tp_degree: int = 1) -> float:
    """Steady-state KV cache bytes per rank, summed across all attention layers.

    Mirror of throughput_model.estimate_memory_per_gpu line 920:
        kv_per_layer = 2 × n_kv_heads × d_head × kv_len × batch × kv_bpe
    summed across layers, sharded by TP.

    For hybrid models, only attention layers contribute KV. State layers add 0.
    """
    kv_bpe = _bpe(arch.kv_precision)
    n_attn_layers = arch.n_layers
    if arch.layer_type_list:
        n_attn_layers = sum(1 for lt in arch.layer_type_list if lt == "attention")
    kv_heads_per_gpu = max(1, math.ceil(arch.n_kv_heads / max(1, tp_degree)))
    per_layer = 2 * kv_heads_per_gpu * arch.d_head * kv_len * arch.batch_size * kv_bpe
    return per_layer * n_attn_layers


def _kv_load_bytes_per_decode_step(arch: ArchConfig, kv_len: int,
                                   tp_degree: int = 1) -> float:
    """Bytes streamed from HBM for KV cache during ONE decode step.

    Mirror of throughput_model.compute_layer_time decode branch line 1150:
        kv_bytes_per_layer = 2 × B × kv_heads_per_gpu × L × d_head × kv_bpe
    summed across attention layers.

    Identical to total KV-cache footprint per rank for the decode-step
    bandwidth model because the model must read the full cache for every
    generated token. Returns the same value as _kv_cache_bytes_total.
    """
    return _kv_cache_bytes_total(arch, kv_len, tp_degree)


def _weight_bytes_per_gpu(arch: ArchConfig, tp_degree: int = 1,
                          pp_degree: int = 1, ep_degree: int = 1) -> float:
    """Model weight bytes loaded from HBM per decode step (per rank).

    Reuses the memory estimator from throughput_model.py to stay in sync
    with the dense / MoE / hybrid accounting. Subtracts KV + activations to
    get just weights.
    """
    is_hybrid = bool(arch.layer_type_list and
                     any(lt == "state" for lt in arch.layer_type_list)
                     and arch.state_config is not None)
    if is_hybrid:
        with_kv = estimate_memory_per_gpu_hybrid(
            arch, tp_degree=tp_degree, pp_degree=pp_degree,
            include_kv_cache=False, ep_degree=ep_degree,
        )
    else:
        with_kv = estimate_memory_per_gpu(
            arch, tp_degree=tp_degree, pp_degree=pp_degree,
            include_kv_cache=False, ep_degree=ep_degree,
        )
    # Subtract the activations component (last term inside estimate_memory_per_gpu)
    bpe = _bpe(arch.precision)
    act_bytes = arch.batch_size * arch.seq_len * arch.d_model * bpe * 10
    return max(with_kv - act_bytes, 0.0)


def _activation_bytes(arch: ArchConfig) -> float:
    """Activation memory (matches throughput_model line 925).

        act_bytes = batch × seq × d_model × bpe × 10
    """
    return arch.batch_size * arch.seq_len * arch.d_model * _bpe(arch.precision) * 10


def _state_load_bytes_per_decode_step(arch: ArchConfig, tp_degree: int = 1) -> float:
    """State-layer weight load per decode step.

    Mirror of throughput_model._state_layer_cost decode branch lines 753-770.
    Only the per-layer weight load is L-independent; SSM parameters fit in
    SRAM and are streamed once per decode step.
    """
    if not arch.state_config or not arch.layer_type_list:
        return 0.0
    n_state_layers = sum(1 for lt in arch.layer_type_list if lt == "state")
    if n_state_layers == 0:
        return 0.0
    sc = arch.state_config
    d = arch.d_model
    d_state = int(sc.get("d_state", 128))
    state_expansion = int(sc.get("state_expansion", 2))
    state_n_heads = int(sc.get("n_heads", arch.n_heads))
    state_d_head = int(sc.get("d_head", 64))
    state_prec = sc.get("state_precision", arch.precision)
    bpe = _bpe(state_prec)
    heads_per_gpu = max(1, state_n_heads // max(1, tp_degree))
    in_proj_bytes = d * (state_expansion * d // max(1, tp_degree)) * bpe
    ssm_bytes = heads_per_gpu * d_state * state_d_head * bpe
    out_proj_bytes = (d // max(1, tp_degree)) * d * bpe
    return (in_proj_bytes + ssm_bytes + out_proj_bytes) * n_state_layers


def _decode_flops_useful(arch: ArchConfig, kv_len: int, tp_degree: int = 1) -> float:
    """Total useful FLOPs in one decode step.

    Roofline numerator. Mirror of throughput_model:
      - QKV matmul: 2 × M × (Q+K+V dim) × d_model
      - Attention compute (line 1155): 2 × B × heads_per_gpu × 1 × L × d_head × 2
      - Output proj: 2 × M × d × heads_per_gpu × d_head
      - FFN: 2 × M × ffn × d (×2 for SwiGLU up+gate) + 2 × M × d × ffn
    summed across layers.
    """
    B = arch.batch_size
    M = B  # decode: S=1, so M = B × 1
    d = arch.d_model
    nh = arch.n_heads
    nkv = arch.n_kv_heads
    dh = arch.d_head
    ffn = arch.ffn_dim
    heads_per_gpu = nh // max(1, tp_degree)
    kv_heads_per_gpu = max(1, math.ceil(nkv / max(1, tp_degree)))
    ffn_per_gpu = ffn // max(1, tp_degree)
    n_attn_layers = arch.n_layers
    if arch.layer_type_list:
        n_attn_layers = sum(1 for lt in arch.layer_type_list if lt == "attention")

    # QKV
    qkv_N = (heads_per_gpu + 2 * kv_heads_per_gpu) * dh
    qkv_flops = 2 * M * qkv_N * d
    # Attention
    attn_flops = 2 * B * heads_per_gpu * 1 * kv_len * dh * 2
    # Output proj
    out_flops = 2 * M * d * (heads_per_gpu * dh)
    # FFN (SwiGLU = up+gate+down)
    if arch.ffn_type == "swiglu":
        ffn_flops = 2 * (2 * M * ffn_per_gpu * d) + 2 * M * d * ffn_per_gpu
    else:
        ffn_flops = 2 * M * ffn_per_gpu * d + 2 * M * d * ffn_per_gpu
    per_attn_layer = qkv_flops + attn_flops + out_flops + ffn_flops
    return per_attn_layer * n_attn_layers


def _prefill_flops_useful(arch: ArchConfig, tp_degree: int = 1) -> float:
    """Useful FLOPs in one prefill step.

    Same matmul shapes as decode, but M = B × S not B × 1. Attention scales
    as S² (instead of L × 1).
    """
    B = arch.batch_size
    S = arch.seq_len
    M = B * S
    d = arch.d_model
    nh = arch.n_heads
    nkv = arch.n_kv_heads
    dh = arch.d_head
    ffn = arch.ffn_dim
    heads_per_gpu = nh // max(1, tp_degree)
    kv_heads_per_gpu = max(1, math.ceil(nkv / max(1, tp_degree)))
    ffn_per_gpu = ffn // max(1, tp_degree)
    n_attn_layers = arch.n_layers
    if arch.layer_type_list:
        n_attn_layers = sum(1 for lt in arch.layer_type_list if lt == "attention")

    qkv_N = (heads_per_gpu + 2 * kv_heads_per_gpu) * dh
    qkv_flops = 2 * M * qkv_N * d
    # _attention_cost line 414: 2 × B × heads_per_gpu × S × S × d_head × 2
    attn_flops = 2 * B * heads_per_gpu * S * S * dh * 2
    out_flops = 2 * M * d * (heads_per_gpu * dh)
    if arch.ffn_type == "swiglu":
        ffn_flops = 2 * (2 * M * ffn_per_gpu * d) + 2 * M * d * ffn_per_gpu
    else:
        ffn_flops = 2 * M * ffn_per_gpu * d + 2 * M * d * ffn_per_gpu
    per_attn_layer = qkv_flops + attn_flops + out_flops + ffn_flops
    return per_attn_layer * n_attn_layers


def _all_reduce_bytes_per_layer(arch: ArchConfig, tp_degree: int,
                                phase: str) -> float:
    """TP all-reduce bytes per layer.

    Mirror of throughput_model._allreduce_cost line 460:
        per_allreduce_bytes = 2 × B × S × d_model × bpe
    and there are 2 all-reduces per layer (after attention, after FFN).

    For ring all-reduce, the volume sent over the wire per rank is
    2 × (P-1)/P × bytes. We return the **wire-side** volume so the
    denominator is link-bandwidth × time and the ratio is what fraction
    of link capacity the all-reduce uses.
    """
    if tp_degree <= 1:
        return 0.0
    S = arch.seq_len if phase != "decode" else 1
    bpe = _bpe(arch.precision)
    per_allreduce_bytes = 2 * arch.batch_size * S * arch.d_model * bpe
    # 2 all-reduces per layer, ring efficiency
    return 2 * 2 * (tp_degree - 1) / tp_degree * per_allreduce_bytes


def _all_to_all_bytes_per_layer(arch: ArchConfig, ep_degree: int,
                                phase: str) -> float:
    """MoE all-to-all bytes per MoE layer (dispatch + combine).

    Mirror of throughput_model._moe_ffn_cost line 653:
        volume = 2 × M × top_k × d_model × bpe_act × capacity_factor
    """
    if not arch.moe_config or ep_degree <= 1:
        return 0.0
    S = arch.seq_len if phase != "decode" else 1
    M = arch.batch_size * S
    moe = arch.moe_config
    top_k = int(moe.get("top_k", 2))
    capacity = float(moe.get("capacity_factor", 1.0))
    bpe = _bpe(arch.precision)
    per_layer = 2 * M * top_k * arch.d_model * bpe * capacity
    # The all-to-all volume on the wire is (ep-1)/ep × per_layer (per ring round).
    return (ep_degree - 1) / max(1, ep_degree) * per_layer


def _link_bw_bytes_s(hw: HardwareConfig, n_ranks: int) -> float:
    """Effective interconnect bandwidth for the dominant link.

    Mirror of throughput_model.HardwareConfig.interconnect_bw_bytes_s line 163.
    """
    if n_ranks <= 1:
        return float("inf")
    ic = hw.interconnect
    domain = hw.nvlink_domain_size or hw.gpus_per_node
    if n_ranks <= (domain or 8):
        return ic["intra_node_bw_gb_s"] * 1e9
    return ic["inter_node_bw_gb_s"] * 1e9


# =============================================================================
# SRAM tile-fit axis (lattice integration)
# =============================================================================

def _dominant_matmul_shape(arch: ArchConfig, phase: str,
                           tp_degree: int = 1) -> tuple:
    """Pick the matmul shape that dominates time in a given phase.

    Heuristic: in prefill / training, the FFN down-projection dominates
    (M=B×S, N=d, K=ffn). In decode, the QKV projection is small enough that
    its tile fit matters more. We return one representative (M,N,K) per phase.
    """
    B = arch.batch_size
    S = arch.seq_len if phase != "decode" else 1
    M = B * S
    d = arch.d_model
    ffn = arch.ffn_dim
    nh = arch.n_heads
    nkv = arch.n_kv_heads
    dh = arch.d_head
    heads_per_gpu = nh // max(1, tp_degree)
    kv_heads_per_gpu = max(1, math.ceil(nkv / max(1, tp_degree)))
    ffn_per_gpu = ffn // max(1, tp_degree)
    if phase == "decode":
        qkv_N = (heads_per_gpu + 2 * kv_heads_per_gpu) * dh
        return (M, qkv_N, d)
    return (M, d, ffn_per_gpu)


def _sram_tile_fit(arch: ArchConfig, hw_name: str, phase: str,
                   tp_degree: int = 1) -> float:
    """Compute 1 - tile_utilization for the dominant matmul shape.

    A value of 0 means perfect tile fit; 1.0 means the kernel uses none of
    the tile (extreme padding). Sourced from lattice_engine.matmul_tile_utilization.
    """
    lattice_hw_name = hw_name if hw_name in LATTICE_HARDWARE else "h100"
    lhw = LATTICE_HARDWARE[lattice_hw_name]
    precision = arch.precision if arch.precision in lhw.tiles else "bf16"
    tile = lhw.tiles.get(precision)
    if tile is None:
        return 0.0
    M, N, K = _dominant_matmul_shape(arch, phase, tp_degree)
    util = matmul_tile_utilization(M, N, K, tile)
    return max(0.0, 1.0 - util)


# =============================================================================
# Main entry point
# =============================================================================

def _raw_layer_times(arch: ArchConfig, hw: HardwareConfig, lhw,
                     tp_degree: int, ep_degree: int, cal_table) -> tuple:
    """Sum raw per-layer kernel times across the stack for decode and prefill.

    Returns (decode_step_s, prefill_step_s) before kernel-launch overhead and
    before the system-efficiency divide that throughput_model.throughput
    applies at the end.
    """
    is_hybrid = bool(arch.layer_type_list
                     and any(lt == "state" for lt in arch.layer_type_list)
                     and arch.state_config is not None)
    if is_hybrid:
        decode_layers = compute_heterogeneous_layer_times(
            arch, hw, lhw, tp_degree, "decode",
            kv_cache_len=arch.seq_len, calibration=cal_table,
            ep_degree=ep_degree,
        )
        prefill_layers = compute_heterogeneous_layer_times(
            arch, hw, lhw, tp_degree, "prefill",
            calibration=cal_table, ep_degree=ep_degree,
        )
        decode_s = sum(bd.total_s for bd in decode_layers)
        prefill_s = sum(bd.total_s for bd in prefill_layers)
    else:
        lbd_decode = compute_layer_time(
            arch, hw, lhw, tp_degree, "decode",
            kv_cache_len=arch.seq_len, calibration=cal_table,
            ep_degree=ep_degree,
        )
        lbd_prefill = compute_layer_time(
            arch, hw, lhw, tp_degree, "prefill",
            calibration=cal_table, ep_degree=ep_degree,
        )
        decode_s = lbd_decode.total_s * arch.n_layers
        prefill_s = lbd_prefill.total_s * arch.n_layers
    return max(decode_s, 1e-9), max(prefill_s, 1e-9)


def compute_throughput_stress(
    arch: ArchConfig,
    hardware: str,
    workload: Optional[Workload] = None,
    tp_degree: int = 1,
    pp_degree: int = 1,
    ep_degree: int = 1,
    arch_name: str = "",
    _throughput_result: Optional[ThroughputResult] = None,
) -> StressVector:
    """Compute the 10-axis StressVector for (arch, hardware, workload).

    Calls into throughput_model.throughput() to get the timings, then
    computes byte/FLOP volumes from arch shape (formulas mirrored from
    throughput_model with citations in helper docstrings).

    Pass _throughput_result if you've already called throughput() to avoid
    redundant work; the function will use it instead of recomputing.
    """
    workload = workload or Workload(
        batch_size=arch.batch_size,
        prefill_seq_len=arch.seq_len,
        decode_kv_len=arch.seq_len,
        phase="decode",
    )

    # SWA cap: if the candidate uses sliding-window attention, KV reads
    # during decode and the per-token attention cost during prefill both
    # see the windowed length, not the full sequence length. We apply the
    # cap here so the throughput call and every downstream byte/FLOP
    # calculation in this function uses the effective length.
    # Two sources, in order: `local_window` (canonical, set by quality
    # ArchConfig and by baseline-loaded SWA configs) and `_swa_window`
    # (sidecar set by the SwapAttentionToSWA delta — needed because the
    # throughput ArchConfig has no `local_window` field).
    swa_window = int(
        getattr(arch, "local_window", 0)
        or getattr(arch, "_swa_window", 0)
        or 0
    )
    effective_decode_kv = workload.decode_kv_len
    effective_prefill_seq = workload.prefill_seq_len
    if swa_window > 0:
        if effective_decode_kv:
            effective_decode_kv = min(effective_decode_kv, swa_window)
        if effective_prefill_seq:
            effective_prefill_seq = min(effective_prefill_seq, swa_window)
        # Replace the workload object (shallow) so byte/FLOP helpers see
        # the capped values without mutating the caller's instance.
        workload = Workload(
            batch_size=workload.batch_size,
            prefill_seq_len=effective_prefill_seq,
            decode_kv_len=effective_decode_kv,
            phase=workload.phase,
        )

    hw = load_hardware(hardware)

    # Reset the arch's batch/seq to workload context for the stress calc.
    # This is non-mutating: we make a shallow copy.
    arch = _arch_with_workload(arch, workload)

    if _throughput_result is None:
        tput = throughput(
            arch, hardware,
            tp_degree=tp_degree, pp_degree=pp_degree,
            decode_kv_len=effective_decode_kv,
            prefill_seq_len=effective_prefill_seq,
            ep_degree=ep_degree,
        )
    else:
        tput = _throughput_result

    # --- Raw kernel times (pre-system-efficiency) ---
    # Stress vectors measure architectural pressure on the actual kernel
    # path, not the user-facing wall-clock time. See ac-stress-build.md
    # decisions log (2026-06-16) for rationale.
    cal_table = load_calibration(hardware)
    lhw = _lattice_hw_for(hardware)
    decode_s, prefill_s = _raw_layer_times(
        arch, hw, lhw, tp_degree, ep_degree, cal_table,
    )
    hbm_bw = hw.hbm_bandwidth_bytes_s
    hbm_cap = hw.hbm_capacity_gb * (1024 ** 3)
    peak_flops_bf16 = hw.peak_flops_s(arch.precision)

    # --- Byte volumes (per rank) ---
    kv_bytes = _kv_cache_bytes_total(arch, workload.decode_kv_len, tp_degree)
    weight_bytes = _weight_bytes_per_gpu(arch, tp_degree, pp_degree, ep_degree)
    state_bytes_decode = _state_load_bytes_per_decode_step(arch, tp_degree)
    act_bytes = _activation_bytes(arch)

    # --- FLOPs ---
    decode_flops = _decode_flops_useful(arch, workload.decode_kv_len, tp_degree)
    prefill_flops = _prefill_flops_useful(arch, tp_degree)

    # --- Communication ---
    ar_per_layer = _all_reduce_bytes_per_layer(arch, tp_degree, "decode")
    a2a_per_layer = _all_to_all_bytes_per_layer(arch, ep_degree, "decode")
    decode_layers = tput.decode_layer_breakdown.total_s if tput.decode_layer_breakdown else (decode_s / max(1, arch.n_layers))
    link_bw_tp = _link_bw_bytes_s(hw, tp_degree)
    link_bw_ep = _link_bw_bytes_s(hw, ep_degree)

    # --- Training memory (weights + grads + opt state + activations × ckpt factor) ---
    # AdamW: optimizer_bytes_per_param ≈ 12. Grads ≈ 1 × weights (in bf16).
    opt_bpe = hw.calibration.get("optimizer_bytes_per_param", 12)
    total_params = estimate_params(
        arch.d_model, arch.n_heads, arch.d_head, arch.ffn_dim,
        arch.n_layers, arch.n_kv_heads, arch.vocab_size,
    )
    params_per_gpu = total_params / max(1, tp_degree) / max(1, pp_degree)
    grad_bytes = params_per_gpu * _bpe(arch.precision)
    opt_bytes = params_per_gpu * opt_bpe
    train_act_bytes = act_bytes * 4  # checkpointed activation factor

    # --- Compute the 10 ratios ---
    decode_traffic = kv_bytes + weight_bytes + state_bytes_decode
    prefill_traffic = weight_bytes + act_bytes  # prefill streams weights once + activations
    sv = StressVector(
        arch_name=arch_name,
        hardware_id=hardware,
        workload_id=workload.workload_id(),
        phase=workload.phase,
        hbm_bw_decode=decode_traffic / max(hbm_bw * decode_s, 1.0),
        hbm_bw_prefill=prefill_traffic / max(hbm_bw * prefill_s, 1.0),
        hbm_capacity=(weight_bytes + kv_bytes + act_bytes) / max(hbm_cap, 1.0),
        kv_footprint=kv_bytes / max(hbm_cap, 1.0),
        tc_util_prefill=prefill_flops / max(peak_flops_bf16 * prefill_s, 1.0),
        tc_util_decode=decode_flops / max(peak_flops_bf16 * decode_s, 1.0),
        sram_tile_fit=_sram_tile_fit(arch, hardware, "prefill", tp_degree),
        all_reduce=(ar_per_layer / max(link_bw_tp * decode_layers, 1.0))
                    if tp_degree > 1 else 0.0,
        all_to_all=(a2a_per_layer / max(link_bw_ep * decode_layers, 1.0))
                    if (arch.moe_config and ep_degree > 1) else 0.0,
        training_mem=(weight_bytes + grad_bytes + opt_bytes + train_act_bytes)
                      / max(hbm_cap, 1.0),
    )

    sv.intermediates = {
        "kv_bytes": kv_bytes,
        "weight_bytes": weight_bytes,
        "state_bytes_decode": state_bytes_decode,
        "act_bytes": act_bytes,
        "decode_flops": decode_flops,
        "prefill_flops": prefill_flops,
        "decode_step_s": decode_s,
        "prefill_step_s": prefill_s,
        "hbm_bw_bytes_s": hbm_bw,
        "hbm_cap_bytes": hbm_cap,
        "peak_flops_s": peak_flops_bf16,
        "ar_per_layer_bytes": ar_per_layer,
        "a2a_per_layer_bytes": a2a_per_layer,
        "link_bw_tp_bytes_s": link_bw_tp,
        "link_bw_ep_bytes_s": link_bw_ep,
        "decode_layer_time_s": decode_layers,
        "grad_bytes": grad_bytes,
        "opt_bytes": opt_bytes,
        "train_act_bytes": train_act_bytes,
    }
    active_axes = active_axes_for_phase(workload.phase)
    sv.binding_axes = [a for a in active_axes
                       if severity_band(getattr(sv, a)) in PRESSURED_OR_WORSE]
    return sv


def _arch_with_workload(arch: ArchConfig, wl: Workload) -> ArchConfig:
    """Return a shallow copy of `arch` with batch/seq overridden by workload.

    The throughput model takes (B, S) from the arch object; we don't want
    callers to have to mutate their config before computing stress.
    """
    seq_for_phase = wl.prefill_seq_len if wl.phase == "prefill" else wl.decode_kv_len
    # Build a new ArchConfig with same fields, overriding B/S only.
    fields = asdict(arch)
    fields["batch_size"] = wl.batch_size
    fields["seq_len"] = seq_for_phase if wl.phase in ("prefill", "training") else arch.seq_len
    return ArchConfig(**fields)


# =============================================================================
# Convenience for known architectures
# =============================================================================

def compute_stress_for_known(arch_name: str, hardware: str,
                             workload: Optional[Workload] = None,
                             **kwargs) -> StressVector:
    """Wraps evaluate_known + compute_throughput_stress for quick experiments."""
    from throughput_model import evaluate_known  # local import to avoid cycle
    workload = workload or Workload()
    tput = evaluate_known(
        arch_name, hardware,
        tp_degree=kwargs.get("tp_degree", 1),
        pp_degree=kwargs.get("pp_degree", 1),
        batch_size=workload.batch_size,
        seq_len=workload.prefill_seq_len,
        decode_kv_len=workload.decode_kv_len,
    )
    # Re-derive the arch from KNOWN_ARCHITECTURES then compute stress
    from lattice_engine import KNOWN_ARCHITECTURES
    ka = KNOWN_ARCHITECTURES[arch_name]
    gqa_map = {
        "Llama-3-70B": 8, "Llama-3-8B": 8, "Mistral-7B": 8, "Llama-2-70B": 8,
    }
    arch = ArchConfig(
        d_model=ka["d_model"], n_layers=ka["n_layers"],
        n_heads=ka["n_heads"], d_head=ka["d_head"],
        n_kv_heads=gqa_map.get(arch_name, ka["n_heads"]),
        ffn_dim=ka["ffn_dim"], ffn_type="swiglu",
        batch_size=workload.batch_size,
        seq_len=workload.decode_kv_len,
    )
    return compute_throughput_stress(
        arch, hardware, workload=workload,
        tp_degree=kwargs.get("tp_degree", 1),
        pp_degree=kwargs.get("pp_degree", 1),
        ep_degree=kwargs.get("ep_degree", 1),
        arch_name=arch_name,
        _throughput_result=tput,
    )
