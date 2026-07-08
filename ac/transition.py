"""
Transition — output dataclass for the delta engine.

A Transition records the (baseline, candidate) pair plus the delta stress
and quality vectors. Per instruction §8, this is what the KG persists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from .stress import (
        StressVector, STRESS_AXES, active_axes_for_phase, severity_band,
        PRESSURED_OR_WORSE,
    )
    from .quality_stress import QualityStressVector, QUALITY_STRESS_AXES
except ImportError:
    from stress import (
        StressVector, STRESS_AXES, active_axes_for_phase, severity_band,
        PRESSURED_OR_WORSE,
    )
    from quality_stress import QualityStressVector, QUALITY_STRESS_AXES


@dataclass
class Transition:
    """Result of applying one transformation to a baseline architecture."""
    schema_version: int = 1

    # Identifying info
    baseline_architecture_id: str = ""
    candidate_architecture_id: str = ""
    hardware_id: str = ""
    workload_id: str = ""

    transformation_name: str = ""
    transformation_params: Dict[str, Any] = field(default_factory=dict)

    # Vectors
    baseline_stress: Optional[StressVector] = None
    candidate_stress: Optional[StressVector] = None
    baseline_quality: Optional[QualityStressVector] = None
    candidate_quality: Optional[QualityStressVector] = None

    # Derived deltas (Δ = candidate - baseline; positive = candidate is higher)
    delta_stress: Dict[str, float] = field(default_factory=dict)
    delta_quality: Dict[str, float] = field(default_factory=dict)
    delta_quality_total: float = 0.0

    # Feasibility flag — false when the candidate fails Schema 0.3 validation
    # or violates a hard hardware constraint (e.g. tp_degree not divisible).
    feasible: bool = True
    reason_if_infeasible: str = ""

    # Binding-axis bookkeeping (instruction §5.2)
    relieved_binding_axes: List[str] = field(default_factory=list)
    new_binding_axes: List[str] = field(default_factory=list)

    computed_at: str = field(default_factory=lambda:
                              datetime.now(timezone.utc).isoformat())

    # ------------------------------------------------------------------

    def compute_deltas(self) -> None:
        """Populate delta_stress, delta_quality, and binding-axis bookkeeping
        from baseline + candidate vectors. Idempotent."""
        if self.baseline_stress and self.candidate_stress:
            for axis in STRESS_AXES:
                bv = getattr(self.baseline_stress, axis)
                cv = getattr(self.candidate_stress, axis)
                self.delta_stress[axis] = cv - bv
            self.relieved_binding_axes = [
                a for a in self.baseline_stress.binding_axes
                if severity_band(getattr(self.candidate_stress, a))
                   not in PRESSURED_OR_WORSE
            ]
            self.new_binding_axes = [
                a for a in self.candidate_stress.binding_axes
                if a not in self.baseline_stress.binding_axes
            ]
        if self.baseline_quality and self.candidate_quality:
            total = 0.0
            for axis in QUALITY_STRESS_AXES:
                bv = self.baseline_quality.axis_value(axis)
                cv = self.candidate_quality.axis_value(axis)
                d = cv - bv
                self.delta_quality[axis] = d
                total += d
            self.delta_quality_total = total

    # ------------------------------------------------------------------

    def relief_score(self) -> float:
        """Signed stress relief over baseline's binding axes (instruction §5.2).

        Higher = more relief on what was binding. Used by the ranker.

        Signed, not clamped: a candidate that pushes an already-binding
        axis further into violation must rank *below* a neutral candidate.
        The old `max(0.0, bv - cv)` clamp let e.g. MLA-at-TP>=latent-width
        (which replicates the latent KV per rank and worsens hbm_capacity)
        tie at 0.0 with a genuinely neutral change — and the axis could
        never appear in `new_binding_axes` because it was already binding
        at baseline, so no penalty fired anywhere.
        """
        if not self.baseline_stress or not self.candidate_stress:
            return 0.0
        s = 0.0
        for axis in self.baseline_stress.binding_axes:
            bv = getattr(self.baseline_stress, axis)
            cv = getattr(self.candidate_stress, axis)
            s += bv - cv
        # Penalize newly pressured axes that weren't pressured before.
        for axis in self.new_binding_axes:
            cv = getattr(self.candidate_stress, axis)
            s -= 0.5 * cv
        return s

    def has_severe_regression(self) -> bool:
        """True if the candidate pushed any axis from < pressured to violated.

        Used to filter candidates per instruction §5.2 step 3.
        """
        if not self.baseline_stress or not self.candidate_stress:
            return False
        for axis in active_axes_for_phase(self.baseline_stress.phase):
            bv = getattr(self.baseline_stress, axis)
            cv = getattr(self.candidate_stress, axis)
            if severity_band(bv) not in PRESSURED_OR_WORSE \
               and severity_band(cv) == "violated":
                return True
        return False

    def as_dict(self) -> Dict[str, Any]:
        d = {
            "transformation_name": self.transformation_name,
            "transformation_params": self.transformation_params,
            "feasible": self.feasible,
            "reason_if_infeasible": self.reason_if_infeasible,
            "delta_stress": dict(self.delta_stress),
            "delta_quality": dict(self.delta_quality),
            "delta_quality_total": self.delta_quality_total,
            "relieved_binding_axes": list(self.relieved_binding_axes),
            "new_binding_axes": list(self.new_binding_axes),
            "relief_score": self.relief_score(),
            "computed_at": self.computed_at,
        }
        if self.baseline_stress:
            d["baseline_stress"] = self.baseline_stress.as_dict()
        if self.candidate_stress:
            d["candidate_stress"] = self.candidate_stress.as_dict()
        if self.baseline_quality:
            d["baseline_quality"] = self.baseline_quality.as_dict()
        if self.candidate_quality:
            d["candidate_quality"] = self.candidate_quality.as_dict()
        return d
