"""
Optimizer bridge — convert v1-ac-solver's CandidateArch + DeploymentConstraints
to a (throughput_model.ArchConfig, Workload) pair so we can compute a stress
vector for it.

The conversion is centralized here so v1-ac-solver/modifier.py can integrate
stress relief without growing extra surface area. Any future change to
CandidateArch (e.g. v2 MoE / state extensions) gets one corresponding edit
to this bridge, not to every caller.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from throughput_model import ArchConfig  # noqa: E402

from stress import StressVector, Workload, compute_throughput_stress  # noqa: E402


_KV_BITS_TO_PRECISION = {16: "bf16", 8: "int8", 4: "fp4"}


def _kv_precision(kv_bits: int) -> str:
    return _KV_BITS_TO_PRECISION.get(int(kv_bits), "bf16")


def candidate_to_arch(candidate, *, batch_size: int = 1, seq_len: int = 2048) -> ArchConfig:
    """Coerce an optimizer.CandidateArch to throughput_model.ArchConfig.

    Maps:
        kv_cache_bits → kv_precision (16→bf16, 8→int8, 4→fp4)
        ffn_precision → precision (for the FFN tier; matmul cost path reads this)
        moe (dict)    → moe_config
        state_config  → state_config
        layer_type_list → layer_type_list
    """
    return ArchConfig(
        d_model=candidate.d_model,
        n_layers=candidate.n_layers,
        n_heads=candidate.n_heads,
        d_head=candidate.d_head,
        n_kv_heads=candidate.n_kv_heads,
        ffn_dim=candidate.ffn_dim,
        ffn_type="swiglu",
        vocab_size=candidate.vocab_size,
        batch_size=batch_size,
        seq_len=seq_len,
        precision=getattr(candidate, "weight_precision", "bf16") or "bf16",
        kv_precision=_kv_precision(getattr(candidate, "kv_cache_bits", 16)),
        moe_config=getattr(candidate, "moe", None),
        n_dense_ffn_layers=getattr(candidate, "n_dense_ffn_layers", 0),
        state_config=getattr(candidate, "state_config", None),
        layer_type_list=getattr(candidate, "layer_type_list", None),
    )


def stress_for_candidate(
    candidate,
    *,
    hardware: str,
    tp_degree: int,
    pp_degree: int = 1,
    ep_degree: int = 1,
    context_length: int,
    serving_batch: int = 1,
    prefill_seq_len: Optional[int] = None,
    phase: str = "decode",
    arch_name: str = "",
) -> StressVector:
    """Compute a StressVector for one optimizer candidate.

    `context_length` is the workload's decode KV length (what the user will
    be serving). `prefill_seq_len` defaults to the same value.
    """
    arch = candidate_to_arch(candidate, batch_size=serving_batch,
                              seq_len=context_length)
    wl = Workload(
        batch_size=serving_batch,
        prefill_seq_len=prefill_seq_len or context_length,
        decode_kv_len=context_length,
        phase=phase,
    )
    return compute_throughput_stress(
        arch, hardware, wl,
        tp_degree=max(1, int(tp_degree)),
        pp_degree=max(1, int(pp_degree)),
        ep_degree=max(1, int(ep_degree)),
        arch_name=arch_name,
    )


# =============================================================================
# Helpers for modifier integration
# =============================================================================

def stress_relief_vs(baseline: StressVector, candidate: StressVector) -> dict:
    """Compute the bookkeeping that modifier.py needs for ranking.

    Returns a dict with:
        relieved_binding_axes  — axes that were pressured/binding/violated in
                                 baseline and dropped to loaded/relaxed in candidate
        new_binding_axes       — axes that weren't pressured in baseline but
                                 are now
        relief_score           — Σ relief on baseline-binding axes - 0.5 × new pressure
        severe_regression      — True if any axis jumped from <pressured to violated
    """
    from stress import PRESSURED_OR_WORSE, STRESS_AXES, severity_band

    relieved = []
    new_binding = []
    relief = 0.0
    severe = False
    for axis in STRESS_AXES:
        bv = getattr(baseline, axis)
        cv = getattr(candidate, axis)
        b_band = severity_band(bv)
        c_band = severity_band(cv)
        if b_band in PRESSURED_OR_WORSE and c_band not in PRESSURED_OR_WORSE:
            relieved.append(axis)
        if b_band not in PRESSURED_OR_WORSE and c_band in PRESSURED_OR_WORSE:
            new_binding.append(axis)
        if b_band in PRESSURED_OR_WORSE:
            relief += max(0.0, bv - cv)
        if b_band not in PRESSURED_OR_WORSE and c_band == "violated":
            severe = True
    # Penalty for new pressure (less than half-weight on the new pressure value).
    for axis in new_binding:
        relief -= 0.5 * getattr(candidate, axis)
    return {
        "relieved_binding_axes": relieved,
        "new_binding_axes": new_binding,
        "relief_score": relief,
        "severe_regression": severe,
    }
