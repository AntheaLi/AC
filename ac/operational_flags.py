"""Wave 18c — Physical feasibility vs. operational assessment.

Per ``plan/redesign/18c-context-workloads-and-operational-flags.md``. Provides:

    * `PhysicalStatus` — enum with the four values 18c requires
      (``feasible``, ``infeasible``, ``unsupported``, ``model_validation_failure``).
    * `OperationalFlag` — canonical strings for the six default flags
      the spec enumerates.
    * `OperationalAssessment` — dataclass carrying `flags`, `thresholds`,
      and `measured` blocks with a compact ``.summary()`` accessor.
    * `ExtendedFeasibility` — combines a physical status with an
      operational assessment. Downstream (18b, 18d) consume this and NEVER
      remove a candidate purely for operational reasons.
    * `evaluate_operational_flags(metrics, workload, hardware)` — evaluates
      the six default flags against a `CandidateMetrics` instance plus its
      `ServingWorkloadSpec` and (optionally) a `HardwareConfig`.

Design principles:

    1. Only physical guards affect optimizer feasibility. Slow, costly,
       low-batch, and oversized-replica candidates remain physically feasible
       and stay in the frontier so downstream 18b can compare them.
    2. HBM spill is physical only if the workload's ``allowed_memory_tiers``
       excludes ``hbm+spill``. If the workload permits spill, the candidate
       is physically feasible with a spill warning.
    3. ``cross_node_collective_risk`` uses the hardware's actual NVLink
       domain (via ``hw.nvlink_domain_size or hw.gpus_per_node``) rather
       than a hard-coded 8. B200 NVL72 (72), TPU v5p (16), and future
       hardware get the right threshold automatically.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Enums
# =============================================================================


class PhysicalStatus(str, enum.Enum):
    """Physical status per 18c spec §Feasibility interface."""
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNSUPPORTED = "unsupported"
    MODEL_VALIDATION_FAILURE = "model_validation_failure"


class OperationalFlag(str, enum.Enum):
    """The six default operational flags per 18c spec §Feasibility interface."""
    EXTREME_TBT = "extreme_tbt"
    EXTREME_TTFT = "extreme_ttft"
    LOW_BATCH_EFFICIENCY = "low_batch_efficiency"
    OVERSIZED_REPLICA = "oversized_replica"
    HIGH_GPU_SECONDS_PER_REQUEST = "high_gpu_seconds_per_request"
    CROSS_NODE_COLLECTIVE_RISK = "cross_node_collective_risk"
    # Additional annotation-only tag for spill routing (physical when
    # workload disallows spill, operational when workload permits spill).
    HBM_SPILL = "hbm_spill"


# =============================================================================
# Default thresholds
# =============================================================================


# Per spec §Feasibility interface:
#   extreme_tbt:                    > 200 ms
#   extreme_ttft:                   > 30 s interactive, > 600 s cold ingestion
#   low_batch_efficiency:           serving batch < 4
#   oversized_replica:              > 512 GPUs interactive, > 4096 cold/offline
#   high_gpu_seconds_per_request:   above scenario budget (caller sets)
#   cross_node_collective_risk:     TP, EP, or CP exceeds local domain
DEFAULT_THRESHOLDS: Dict[str, float] = {
    OperationalFlag.EXTREME_TBT.value:                200.0,     # ms
    OperationalFlag.EXTREME_TTFT.value:                30_000.0, # ms (interactive)
    OperationalFlag.EXTREME_TTFT.value + "_cold":     600_000.0, # ms (cold ingestion)
    OperationalFlag.LOW_BATCH_EFFICIENCY.value:            4.0,
    OperationalFlag.OVERSIZED_REPLICA.value:             512.0,  # GPUs (interactive)
    OperationalFlag.OVERSIZED_REPLICA.value + "_cold":  4096.0,  # GPUs (cold/offline)
    OperationalFlag.HIGH_GPU_SECONDS_PER_REQUEST.value:   10.0,  # generous default
}


# =============================================================================
# OperationalAssessment
# =============================================================================


@dataclass
class OperationalAssessment:
    """Operational-flag report per 18c spec §Feasibility interface."""
    flags: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    measured: Dict[str, float] = field(default_factory=dict)

    def has_flag(self, name: str) -> bool:
        return name in self.flags

    def summary(self) -> str:
        if not self.flags:
            return "no operational flags"
        return ", ".join(sorted(self.flags))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "flags": list(self.flags),
            "thresholds": dict(self.thresholds),
            "measured": dict(self.measured),
        }


@dataclass
class ExtendedFeasibility:
    """Physical + operational report for one candidate under one workload.

    ``physical_status`` alone drives optimizer feasibility. ``operational``
    NEVER changes physical status; it only annotates.
    """
    physical_status: PhysicalStatus = PhysicalStatus.FEASIBLE
    physical_message: str = ""
    operational: OperationalAssessment = field(default_factory=OperationalAssessment)

    @property
    def is_feasible(self) -> bool:
        return self.physical_status == PhysicalStatus.FEASIBLE

    def as_dict(self) -> Dict[str, Any]:
        return {
            "physical_status": self.physical_status.value,
            "physical_message": self.physical_message,
            "operational": self.operational.as_dict(),
        }


# =============================================================================
# Flag evaluation
# =============================================================================


def _hw_local_domain(hardware: Any) -> Optional[int]:
    """Return the hardware's NVLink/local domain, or None if unknown."""
    if hardware is None:
        return None
    nvlink = getattr(hardware, "nvlink_domain_size", None)
    if nvlink and int(nvlink) > 0:
        return int(nvlink)
    gpus_per_node = getattr(hardware, "gpus_per_node", None)
    if gpus_per_node and int(gpus_per_node) > 0:
        return int(gpus_per_node)
    chips_per_host = getattr(hardware, "chips_per_host", None)
    if chips_per_host and int(chips_per_host) > 0:
        return int(chips_per_host)
    return None


def evaluate_operational_flags(
    metrics: Any,
    workload: Any,
    *,
    hardware: Any = None,
    thresholds: Optional[Dict[str, float]] = None,
    gpu_seconds_budget: Optional[float] = None,
) -> OperationalAssessment:
    """Evaluate the six default flags against a metrics + workload pair.

    ``metrics`` should be a `CandidateMetrics` (18b) or any object with
    the same attributes (``tbt_ms``, ``ttft_ms``, ``replica_gpus``,
    ``serving_gpu_seconds_per_request``, ``topology.tp/ep/cp``).

    ``workload`` should be a `ServingWorkloadSpec` (Wave 18c) or None; the
    cold-ingestion path uses ``workload.is_cold_ingestion`` to switch
    thresholds. When ``workload`` is None, the interactive thresholds apply.

    ``hardware`` is optional. When provided (a `HardwareConfig` or similar
    with ``nvlink_domain_size``/``gpus_per_node``), the cross-node collective
    check uses the hardware's actual local domain.

    ``thresholds`` overrides any subset of the defaults; unspecified keys
    fall back to ``DEFAULT_THRESHOLDS``.
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    if gpu_seconds_budget is not None:
        th[OperationalFlag.HIGH_GPU_SECONDS_PER_REQUEST.value] = float(gpu_seconds_budget)

    is_cold = bool(getattr(workload, "is_cold_ingestion", False))
    ttft_key = (OperationalFlag.EXTREME_TTFT.value + ("_cold" if is_cold else ""))
    replica_key = (OperationalFlag.OVERSIZED_REPLICA.value + ("_cold" if is_cold else ""))

    # Measured metrics.
    tbt_ms = float(getattr(metrics, "tbt_ms", 0.0) or 0.0)
    ttft_ms = float(getattr(metrics, "ttft_ms", 0.0) or 0.0)
    replica_gpus = int(getattr(metrics, "replica_gpus", 1) or 1)
    gpu_s_req = float(getattr(metrics, "serving_gpu_seconds_per_request", 0.0) or 0.0)
    serving_batch = int(getattr(getattr(metrics, "topology", None),
                                "serving_batch",
                                int(getattr(workload, "serving_batch", 1))) or 1)

    # Collect flags.
    flags: List[str] = []
    measured: Dict[str, float] = {}
    active_thresholds: Dict[str, float] = {}

    # extreme_tbt
    tbt_thresh = th[OperationalFlag.EXTREME_TBT.value]
    measured[OperationalFlag.EXTREME_TBT.value] = tbt_ms
    active_thresholds[OperationalFlag.EXTREME_TBT.value] = tbt_thresh
    if tbt_ms > tbt_thresh:
        flags.append(OperationalFlag.EXTREME_TBT.value)

    # extreme_ttft (interactive vs cold thresholds)
    ttft_thresh = th[ttft_key]
    measured[OperationalFlag.EXTREME_TTFT.value] = ttft_ms
    active_thresholds[OperationalFlag.EXTREME_TTFT.value] = ttft_thresh
    if ttft_ms > ttft_thresh:
        flags.append(OperationalFlag.EXTREME_TTFT.value)

    # low_batch_efficiency
    low_batch_thresh = th[OperationalFlag.LOW_BATCH_EFFICIENCY.value]
    measured[OperationalFlag.LOW_BATCH_EFFICIENCY.value] = serving_batch
    active_thresholds[OperationalFlag.LOW_BATCH_EFFICIENCY.value] = low_batch_thresh
    if serving_batch < low_batch_thresh:
        flags.append(OperationalFlag.LOW_BATCH_EFFICIENCY.value)

    # oversized_replica (interactive vs cold thresholds)
    replica_thresh = th[replica_key]
    measured[OperationalFlag.OVERSIZED_REPLICA.value] = replica_gpus
    active_thresholds[OperationalFlag.OVERSIZED_REPLICA.value] = replica_thresh
    if replica_gpus > replica_thresh:
        flags.append(OperationalFlag.OVERSIZED_REPLICA.value)

    # high_gpu_seconds_per_request
    cost_thresh = th[OperationalFlag.HIGH_GPU_SECONDS_PER_REQUEST.value]
    measured[OperationalFlag.HIGH_GPU_SECONDS_PER_REQUEST.value] = gpu_s_req
    active_thresholds[OperationalFlag.HIGH_GPU_SECONDS_PER_REQUEST.value] = cost_thresh
    if gpu_s_req > cost_thresh:
        flags.append(OperationalFlag.HIGH_GPU_SECONDS_PER_REQUEST.value)

    # cross_node_collective_risk
    local_domain = _hw_local_domain(hardware)
    topology = getattr(metrics, "topology", None)
    tp = int(getattr(topology, "tp", 1) or 1)
    ep = int(getattr(topology, "ep", 1) or 1)
    cp = int(getattr(topology, "cp", 1) or 1)
    max_local = max(tp, ep, cp)
    measured[OperationalFlag.CROSS_NODE_COLLECTIVE_RISK.value] = max_local
    if local_domain is not None:
        active_thresholds[OperationalFlag.CROSS_NODE_COLLECTIVE_RISK.value] = local_domain
        if max_local > local_domain:
            flags.append(OperationalFlag.CROSS_NODE_COLLECTIVE_RISK.value)

    return OperationalAssessment(
        flags=flags,
        thresholds=active_thresholds,
        measured=measured,
    )


# =============================================================================
# HBM-spill routing
# =============================================================================


def hbm_spill_physical(
    memory_per_gpu_gb: float,
    hbm_gb: float,
    allowed_memory_tiers: Tuple[str, ...],
) -> Tuple[bool, bool]:
    """Determine whether HBM spill routes to physical infeasibility.

    Returns ``(spills, is_physical_infeasible)``.

    Per spec §Feasibility interface: "HBM spill is physical if the requested
    memory tiers exclude spill; otherwise it remains a costly operational
    condition." When spill is permitted, the caller should attach an
    operational ``hbm_spill`` flag; when spill is not permitted, the caller
    should mark physical status ``INFEASIBLE``.
    """
    from ac.serving_workload import MEMORY_TIER_HBM_SPILL
    spills = float(memory_per_gpu_gb) > float(hbm_gb) * 1.0
    if not spills:
        return (False, False)
    spill_permitted = MEMORY_TIER_HBM_SPILL in allowed_memory_tiers
    return (True, not spill_permitted)


# =============================================================================
# Extended feasibility construction
# =============================================================================


def build_extended_feasibility(
    metrics: Any,
    workload: Any,
    *,
    hardware: Any = None,
    hbm_gb: Optional[float] = None,
    physical_status: PhysicalStatus = PhysicalStatus.FEASIBLE,
    physical_message: str = "",
    thresholds: Optional[Dict[str, float]] = None,
    gpu_seconds_budget: Optional[float] = None,
) -> ExtendedFeasibility:
    """Build an `ExtendedFeasibility` combining physical status and operational assessment.

    If ``hbm_gb`` is provided and the candidate's ``memory_per_gpu_gb`` exceeds
    it, the spill routing (physical vs operational) is applied according to
    the workload's ``allowed_memory_tiers``.

    Physical guards passed in via ``physical_status``/``physical_message`` (from
    Wave 13's Feasibility.violated_guards, unsupported attention encoding,
    etc.) short-circuit: they override the FEASIBLE default. The operational
    assessment is still populated so a reviewer can see what a physically
    infeasible candidate would have looked like on the operational axes.
    """
    assessment = evaluate_operational_flags(
        metrics, workload, hardware=hardware,
        thresholds=thresholds, gpu_seconds_budget=gpu_seconds_budget,
    )

    # HBM spill: physical when spill is not an allowed memory tier.
    if hbm_gb is not None and hbm_gb > 0:
        mem = float(getattr(metrics, "memory_per_gpu_gb", 0.0) or 0.0)
        tiers = tuple(getattr(workload, "allowed_memory_tiers", ()))
        spills, physical_infeas = hbm_spill_physical(mem, hbm_gb, tiers)
        if spills:
            assessment.flags.append(OperationalFlag.HBM_SPILL.value)
            assessment.measured[OperationalFlag.HBM_SPILL.value] = mem
            assessment.thresholds[OperationalFlag.HBM_SPILL.value] = hbm_gb
            if physical_infeas and physical_status == PhysicalStatus.FEASIBLE:
                physical_status = PhysicalStatus.INFEASIBLE
                physical_message = (
                    f"HBM spill required ({mem:.1f} GB > {hbm_gb:.1f} GB HBM) "
                    f"and workload disallows spill tier"
                )

    return ExtendedFeasibility(
        physical_status=physical_status,
        physical_message=physical_message,
        operational=assessment,
    )
