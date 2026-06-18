"""
Delta engine — apply named transformations and compute Transition objects.

Given a baseline architecture A and a list of (Transformation, params), it:
    1. Applies each transformation to A → A_i
    2. Computes baseline + candidate StressVectors and QualityStressVectors
    3. Builds a Transition recording deltas, binding-axis flips, feasibility
    4. Returns the list of Transitions

Stress-aware ranking on top of this lives in `rank.py` (Phase C).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from throughput_model import ArchConfig as TArchConfig  # noqa: E402
from quality_model import (  # noqa: E402
    TrainingConfig,
)

from stress import StressVector, Workload, compute_throughput_stress  # noqa: E402
from quality_stress import compute_quality_stress  # noqa: E402
from transition import Transition  # noqa: E402
from deltas import REGISTRY, get as get_transformation  # noqa: E402
from deltas.base import Transformation  # noqa: E402


def _arch_hash(arch: TArchConfig) -> str:
    """Stable short hash for an architecture (for KG keys)."""
    public = {k: v for k, v in asdict(arch).items() if not k.startswith("_")}
    blob = json.dumps(public, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def _parallelism_for(arch: TArchConfig, defaults: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Read parallelism overrides set by ChangeParallelism, fall back to defaults."""
    tp = getattr(arch, "_tp_override", defaults[0])
    pp = getattr(arch, "_pp_override", defaults[1])
    ep = getattr(arch, "_ep_override", defaults[2])
    return tp, pp, ep


def apply_transition(
    baseline: TArchConfig,
    transformation: Transformation,
    params: Optional[Dict[str, Any]] = None,
    *,
    hardware: str = "h100",
    workload: Optional[Workload] = None,
    tp_degree: int = 1,
    pp_degree: int = 1,
    ep_degree: int = 1,
    training: Optional[TrainingConfig] = None,
    baseline_name: str = "",
    _baseline_stress: Optional[StressVector] = None,
) -> Transition:
    """Apply one transformation to the baseline and produce a Transition.

    Pass `_baseline_stress` if you've already computed it to avoid recomputing
    for each candidate in a batch — see `apply_transitions`.
    """
    params = params or {}
    workload = workload or Workload(
        batch_size=baseline.batch_size,
        prefill_seq_len=baseline.seq_len,
        decode_kv_len=baseline.seq_len,
        phase="decode",
    )
    training = training or TrainingConfig(training_tokens=2_000_000_000_000)

    # --- precondition ---
    ok, reason = transformation.precondition(baseline)
    if not ok:
        return Transition(
            transformation_name=transformation.name,
            transformation_params=dict(params),
            baseline_architecture_id=_arch_hash(baseline),
            hardware_id=hardware,
            workload_id=workload.workload_id(),
            feasible=False,
            reason_if_infeasible=f"precondition_failed: {reason}",
        )

    # --- apply ---
    try:
        candidate = transformation.apply(baseline, **params)
    except Exception as e:
        return Transition(
            transformation_name=transformation.name,
            transformation_params=dict(params),
            baseline_architecture_id=_arch_hash(baseline),
            hardware_id=hardware,
            workload_id=workload.workload_id(),
            feasible=False,
            reason_if_infeasible=f"apply_raised: {type(e).__name__}: {e}",
        )

    # --- read parallelism overrides from the candidate ---
    c_tp, c_pp, c_ep = _parallelism_for(candidate, (tp_degree, pp_degree, ep_degree))

    # --- stress vectors ---
    if _baseline_stress is None:
        b_stress = compute_throughput_stress(
            baseline, hardware, workload,
            tp_degree=tp_degree, pp_degree=pp_degree, ep_degree=ep_degree,
            arch_name=baseline_name or "baseline",
        )
    else:
        b_stress = _baseline_stress
    try:
        c_stress = compute_throughput_stress(
            candidate, hardware, workload,
            tp_degree=c_tp, pp_degree=c_pp, ep_degree=c_ep,
            arch_name=f"{baseline_name or 'baseline'}+{transformation.name}",
        )
    except Exception as e:
        return Transition(
            transformation_name=transformation.name,
            transformation_params=dict(params),
            baseline_architecture_id=_arch_hash(baseline),
            candidate_architecture_id=_arch_hash(candidate),
            hardware_id=hardware,
            workload_id=workload.workload_id(),
            baseline_stress=b_stress,
            feasible=False,
            reason_if_infeasible=f"stress_compute_failed: {type(e).__name__}: {e}",
        )

    # --- quality vectors (best-effort; failures don't block stress) ---
    b_q = c_q = None
    try:
        b_qarch = transformation.to_quality_arch(baseline)
        b_q = compute_quality_stress(b_qarch, training,
                                      workload_spec={"context_length": workload.decode_kv_len},
                                      arch_name=baseline_name or "baseline")
    except Exception:
        pass
    try:
        c_qarch = transformation.to_quality_arch(candidate)
        c_q = compute_quality_stress(c_qarch, training,
                                      workload_spec={"context_length": workload.decode_kv_len},
                                      arch_name=f"{baseline_name or 'baseline'}+{transformation.name}")
    except Exception:
        pass

    t = Transition(
        transformation_name=transformation.name,
        transformation_params=dict(params),
        baseline_architecture_id=_arch_hash(baseline),
        candidate_architecture_id=_arch_hash(candidate),
        hardware_id=hardware,
        workload_id=workload.workload_id(),
        baseline_stress=b_stress,
        candidate_stress=c_stress,
        baseline_quality=b_q,
        candidate_quality=c_q,
        feasible=True,
    )
    t.compute_deltas()
    return t


def apply_transitions(
    baseline: TArchConfig,
    transforms: List[Tuple[str, Dict[str, Any]]],
    *,
    hardware: str = "h100",
    workload: Optional[Workload] = None,
    tp_degree: int = 1,
    pp_degree: int = 1,
    ep_degree: int = 1,
    training: Optional[TrainingConfig] = None,
    baseline_name: str = "",
) -> List[Transition]:
    """Apply a batch of transformations, sharing the baseline stress computation."""
    workload = workload or Workload(
        batch_size=baseline.batch_size,
        prefill_seq_len=baseline.seq_len,
        decode_kv_len=baseline.seq_len,
        phase="decode",
    )
    # Compute baseline stress once.
    b_stress = compute_throughput_stress(
        baseline, hardware, workload,
        tp_degree=tp_degree, pp_degree=pp_degree, ep_degree=ep_degree,
        arch_name=baseline_name or "baseline",
    )
    out: List[Transition] = []
    for name, params in transforms:
        try:
            xf = get_transformation(name)
        except KeyError as e:
            out.append(Transition(
                transformation_name=name,
                transformation_params=dict(params or {}),
                baseline_architecture_id=_arch_hash(baseline),
                hardware_id=hardware,
                workload_id=workload.workload_id(),
                feasible=False,
                reason_if_infeasible=f"unknown_transformation: {e}",
            ))
            continue
        t = apply_transition(
            baseline, xf, params or {},
            hardware=hardware, workload=workload,
            tp_degree=tp_degree, pp_degree=pp_degree, ep_degree=ep_degree,
            training=training, baseline_name=baseline_name,
            _baseline_stress=b_stress,
        )
        out.append(t)
    return out


# =============================================================================
# Stress-aware ranker (instruction §5.2)
# =============================================================================

def rank_transitions(transitions: List[Transition]) -> List[Transition]:
    """Order transitions by binding-stress relief minus new-pressure penalty.

    Algorithm (instruction §5.2):
      1. (already done at construction time) Each Transition carries
         baseline_stress.binding_axes.
      2. Compute relief_score = Σ relief_k - 0.5 × Σ new_pressured_axis_value.
      3. Drop transitions where any axis went from < pressured to violated.
      4. Sort by score descending.
    """
    feasible = [t for t in transitions
                if t.feasible and not t.has_severe_regression()]
    feasible.sort(key=lambda t: t.relief_score(), reverse=True)
    return feasible
