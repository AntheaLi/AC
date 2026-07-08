"""
Stress-aware ranking — the optimizer-facing entry point.

This module is the integration seam called out in instruction §6.3. It does
two things the optimizer needs:

    annotate_candidates(candidates, hardware, workload)
        For each (architecture, throughput_result, quality_result) tuple
        the optimizer already emits, attach a StressVector + QualityStressVector
        without re-running the throughput / quality models.

    stress_relief_frontier(baseline, candidates)
        Return the subset of candidates that improve at least one binding
        axis without violating another. This is the "stress-aware Pareto
        frontier" called out in §5.2.

The optimizer's own search algorithm is NOT modified here — this is a
post-processing pass over what it already produces, preserving the §7.1
constraint that the optimizer stays as-is.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from throughput_model import ArchConfig  # noqa: E402
from quality_model import TrainingConfig  # noqa: E402

from stress import (  # noqa: E402
    PRESSURED_OR_WORSE,
    STRESS_AXES,
    StressVector,
    Workload,
    compute_throughput_stress,
    severity_band,
)
from quality_stress import compute_quality_stress  # noqa: E402


def annotate_candidate(
    arch: ArchConfig,
    hardware: str,
    workload: Workload,
    tp_degree: int = 1,
    pp_degree: int = 1,
    ep_degree: int = 1,
    training: Optional[TrainingConfig] = None,
    arch_name: str = "",
) -> Dict[str, Any]:
    """Compute stress + quality-stress vectors for one candidate and bundle them.

    Return shape (used by the web explorer and the optimizer's KG writer):

        {
            "arch_name": ...,
            "stress": <StressVector serialized>,
            "quality": <QualityStressVector serialized>,
            "binding_axes": [...],
        }
    """
    sv = compute_throughput_stress(
        arch, hardware, workload,
        tp_degree=tp_degree, pp_degree=pp_degree, ep_degree=ep_degree,
        arch_name=arch_name,
    )
    try:
        # to_quality_arch only knows about Transformation-derived configs;
        # for the optimizer's direct candidates we coerce via field copy.
        from deltas.base import Transformation
        q_arch = Transformation().to_quality_arch(arch)
        training = training or TrainingConfig(training_tokens=20_000_000_000_000)
        qsv = compute_quality_stress(
            q_arch, training,
            workload_spec={"context_length": workload.decode_kv_len},
            arch_name=arch_name,
        )
        quality_dict = qsv.as_dict()
    except Exception as e:
        quality_dict = {"error": f"{type(e).__name__}: {e}"}
    return {
        "arch_name": arch_name,
        "stress": sv.as_dict(),
        "quality": quality_dict,
        "binding_axes": list(sv.binding_axes),
    }


def stress_relief_frontier(
    baseline: ArchConfig,
    candidates: List[ArchConfig],
    *,
    hardware: str = "h100",
    workload: Optional[Workload] = None,
    tp_degree: int = 1,
    pp_degree: int = 1,
    ep_degree: int = 1,
    baseline_name: str = "baseline",
    candidate_names: Optional[List[str]] = None,
) -> List[Tuple[str, ArchConfig, StressVector, float]]:
    """Rank candidate architectures by binding-stress relief vs baseline.

    Returns a list of (name, arch, stress_vector, relief_score) sorted by
    relief_score descending. Candidates that violate a previously-OK axis
    are filtered out (per instruction §5.2 step 3).
    """
    workload = workload or Workload(
        batch_size=baseline.batch_size,
        prefill_seq_len=baseline.seq_len,
        decode_kv_len=baseline.seq_len,
        phase="decode",
    )
    b_stress = compute_throughput_stress(
        baseline, hardware, workload,
        tp_degree=tp_degree, pp_degree=pp_degree, ep_degree=ep_degree,
        arch_name=baseline_name,
    )

    out = []
    names = candidate_names or [f"candidate_{i}" for i in range(len(candidates))]
    for name, cand in zip(names, candidates):
        try:
            c_stress = compute_throughput_stress(
                cand, hardware, workload,
                tp_degree=tp_degree, pp_degree=pp_degree, ep_degree=ep_degree,
                arch_name=name,
            )
        except Exception:
            continue
        # Severe-regression filter.
        regressed = False
        for axis in STRESS_AXES:
            bv = getattr(b_stress, axis)
            cv = getattr(c_stress, axis)
            if severity_band(bv) not in PRESSURED_OR_WORSE \
               and severity_band(cv) == "violated":
                regressed = True
                break
        if regressed:
            continue
        # relief = Σ (bv - cv) over baseline-binding axes, minus penalty
        # for any new pressured axes.
        relief = 0.0
        for axis in b_stress.binding_axes:
            relief += max(0.0, getattr(b_stress, axis) - getattr(c_stress, axis))
        for axis in c_stress.binding_axes:
            if axis not in b_stress.binding_axes:
                relief -= 0.5 * getattr(c_stress, axis)
        out.append((name, cand, c_stress, relief))
    out.sort(key=lambda x: x[3], reverse=True)
    return out
