"""
SRAM-Parameterized State Dimension Derivation

The core v2 contribution: d_state is derived from hardware SRAM capacity,
not treated as a tunable hyperparameter.

General formula:
  d_state_max = (fast_mem_capacity * alpha_state) / (h_concurrent * d_head * bytes(precision))

Per hardware:
  H100: SRAM 228KB/SM, h_concurrent=8 -> d_state_max ~192 (BF16, d_head=64)
  B200:  SRAM ~256KB/SM, h_concurrent=8 -> d_state_max ~217 -> rounds to 192 or 224
  TPU v5p: VMEM 32MB/core, h_concurrent=64 -> d_state_max ~3494 -> capped at 256 (quality saturation)
"""

import json
import math
import os
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Import lattice engine (sibling module)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from lattice_engine import HardwareSpec, HARDWARE as LATTICE_HARDWARE

# Hardware spec JSON directory (sibling)
_SPEC_DIR = os.path.join(_HERE, "hardware_specs")

# =============================================================================
# Hardware-specific SRAM / fast-memory constants
# =============================================================================

# These are loaded from JSON where available, or fall back to built-in defaults.
# sram_per_sm_kb: NVIDIA shared memory per SM in KB
# vmem_per_core_mb: TPU VMEM per core in MB
# h_concurrent_heads: number of state heads processed concurrently in SRAM
# fast_memory_tier: "sram" for GPU shared memory, "vmem" for TPU VMEM

_HW_STATE_PARAMS = {
    "h100": {
        "sram_per_sm_kb": 228,
        "h_concurrent_heads": 8,
        "fast_memory_tier": "sram",
    },
    "b200": {
        "sram_per_sm_kb": 256,
        "h_concurrent_heads": 8,
        "fast_memory_tier": "sram",
    },
    "tpu_v5p": {
        "vmem_per_core_mb": 32,
        "h_concurrent_heads": 64,
        "fast_memory_tier": "vmem",
    },
    "tpu_v5e": {
        "vmem_per_core_mb": 16,
        "h_concurrent_heads": 32,
        "fast_memory_tier": "vmem",
    },
}


def _load_hw_state_params(hw_name: str) -> dict:
    """Load state-relevant hardware params, merging JSON overrides with defaults."""
    defaults = _HW_STATE_PARAMS.get(hw_name, {})
    # Try to load from JSON spec
    mapping = {
        "h100": "h100_sxm.json",
        "b200": "b200.json",
        "tpu_v5p": "tpu_v5p.json",
        "tpu_v5e": "tpu_v5e.json",
    }
    filename = mapping.get(hw_name)
    if filename:
        path = os.path.join(_SPEC_DIR, filename)
        if os.path.exists(path):
            with open(path) as f:
                spec = json.load(f)
            # Merge any state-related fields from JSON
            for key in ("sram_per_sm_kb", "h_concurrent_heads", "fast_memory_tier",
                        "vmem_per_core_mb"):
                if key in spec:
                    defaults[key] = spec[key]
    return defaults


def _precision_bytes(precision: str) -> int:
    """Return bytes per element for a given precision."""
    return {
        "bf16": 2,
        "fp16": 2,
        "fp32": 4,
        "fp8": 1,
        "int8": 1,
        "fp4": 1,  # conservative (0.5 actual, but alignment needs 1)
    }.get(precision, 2)


def _fast_mem_bytes(hw_name: str) -> int:
    """Return fast memory capacity in bytes for one compute unit."""
    params = _load_hw_state_params(hw_name)
    tier = params.get("fast_memory_tier", "sram")
    if tier == "vmem":
        vmem_mb = params.get("vmem_per_core_mb", 32)
        return vmem_mb * 1024 * 1024
    else:
        sram_kb = params.get("sram_per_sm_kb", 228)
        return sram_kb * 1024


def _h_concurrent(hw_name: str) -> int:
    """Return number of concurrent state heads per compute unit."""
    params = _load_hw_state_params(hw_name)
    return params.get("h_concurrent_heads", 8)


# =============================================================================
# Core derivation functions
# =============================================================================

def derive_d_state_max(
    hw_spec: HardwareSpec,
    d_head: int,
    state_precision: str = "bf16",
    alpha_state: float = 0.85,
) -> int:
    """
    Derive the maximum d_state that fits in SRAM for the given hardware.

    Formula:
      d_state_max = floor(fast_mem_capacity * alpha_state / (h_concurrent * d_head * bytes_per_elem))

    Args:
        hw_spec: LatticeHardwareSpec with sram_per_sm_bytes
        d_head: dimension per head
        state_precision: precision for state storage
        alpha_state: fraction of SRAM budget allocated to state (default 0.85)

    Returns:
        Maximum d_state as integer
    """
    hw_name = hw_spec.short_name
    fast_mem = _fast_mem_bytes(hw_name)
    h_conc = _h_concurrent(hw_name)
    bpe = _precision_bytes(state_precision)

    denominator = h_conc * d_head * bpe
    if denominator <= 0:
        return 0

    d_state_raw = int(fast_mem * alpha_state / denominator)
    return max(0, d_state_raw)


def apply_quality_saturation_cap(d_state_raw: int, cap: int = 256) -> int:
    """
    Cap d_state at quality saturation point.

    For hardware with very large fast memory (e.g., TPU VMEM), the raw
    d_state_max can be very large. Beyond ~256, quality improvements
    saturate while compute costs continue to grow.

    Args:
        d_state_raw: uncapped d_state value
        cap: quality saturation cap (default 256)

    Returns:
        min(d_state_raw, cap)
    """
    return min(d_state_raw, cap)


def derive_d_state(
    hw_name: str,
    d_head: int,
    state_precision: str = "bf16",
    alpha_state: float = 0.85,
) -> int:
    """
    Convenience function: derive d_state from hardware name.

    Loads the hardware spec, derives raw d_state_max, applies quality
    saturation cap, and returns the final value.

    Args:
        hw_name: hardware name (h100, b200, tpu_v5p, tpu_v5e)
        d_head: dimension per head
        state_precision: precision for state storage
        alpha_state: fraction of SRAM budget allocated to state

    Returns:
        Derived d_state (capped)
    """
    if hw_name not in LATTICE_HARDWARE:
        raise ValueError(f"Unknown hardware: {hw_name}. Supported: {list(LATTICE_HARDWARE.keys())}")

    hw_spec = LATTICE_HARDWARE[hw_name]
    raw = derive_d_state_max(hw_spec, d_head, state_precision, alpha_state)
    return apply_quality_saturation_cap(raw)


def compute_crossover_seq_len(
    hw_name: str,
    n_kv_heads: int,
    d_head: int,
    batch_size: int,
    kv_precision: str,
    d_state: int,
    state_expansion: int,
    d_model: int,
    state_precision: str,
) -> float:
    """
    Compute L* where state decode cost equals attention decode cost.

    At decode time:
    - Attention decode cost per layer: KV cache load = 2 * B * n_kv_heads * L * d_head * kv_bpe / HBM_BW
    - State decode cost per layer: weight load = state_weights / HBM_BW
      where state_weights = state_expansion * d_model * d_state * state_bpe
      (the SSM in/out projections that replace QKV + output proj)

    L* = state_weight_bytes / (2 * B * n_kv_heads * d_head * kv_bpe)

    At L < L*, attention is cheaper (short sequences).
    At L > L*, state is cheaper (long sequences, the key benefit).

    Args:
        hw_name: hardware name (for future HBM BW asymmetry, currently unused)
        n_kv_heads: number of KV heads in the attention layer
        d_head: dimension per head
        batch_size: decode batch size
        kv_precision: KV cache precision
        d_state: state dimension
        state_expansion: expansion factor for SSM projections (typically 2)
        d_model: model width
        state_precision: state weight precision

    Returns:
        L* crossover sequence length (float)
    """
    kv_bpe = _precision_bytes(kv_precision)
    state_bpe = _precision_bytes(state_precision)

    # State decode cost: load the SSM projection weights
    # Mamba-2 has input projection (d_model -> state_expansion * d_state)
    # and output projection (d_state -> d_model), plus discretization params
    # Simplified: total_state_weight_bytes ~ state_expansion * d_model * d_state * bpe
    state_weight_bytes = state_expansion * d_model * d_state * state_bpe

    # Attention decode cost per token: load KV cache
    # kv_load_per_L = 2 * batch_size * n_kv_heads * d_head * kv_bpe
    kv_load_per_L = 2 * batch_size * n_kv_heads * d_head * kv_bpe

    if kv_load_per_L <= 0:
        return float("inf")

    L_star = state_weight_bytes / kv_load_per_L
    return L_star


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SRAM-Parameterized State Dimension Derivation")
    print("=" * 70)

    for hw_name in ["h100", "b200", "tpu_v5p", "tpu_v5e"]:
        print(f"\n--- {hw_name} ---")
        for d_head in [64, 128]:
            raw = derive_d_state_max(LATTICE_HARDWARE[hw_name], d_head)
            capped = derive_d_state(hw_name, d_head)
            print(f"  d_head={d_head}: raw_max={raw}, capped={capped}")

    print(f"\n--- Crossover sequence lengths ---")
    for hw_name in ["h100", "b200"]:
        L_star = compute_crossover_seq_len(
            hw_name=hw_name,
            n_kv_heads=8,
            d_head=128,
            batch_size=1,
            kv_precision="bf16",
            d_state=128,
            state_expansion=2,
            d_model=4096,
            state_precision="bf16",
        )
        print(f"  {hw_name}: L* = {L_star:.0f}")
