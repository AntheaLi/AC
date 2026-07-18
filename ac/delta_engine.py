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
from dataclasses import asdict, replace as dc_replace
from typing import Any, Dict, List, Optional, Tuple

try:
    from .throughput_model import ArchConfig as TArchConfig
    from .quality_model import TrainingConfig
    from .stress import StressVector, Workload, compute_throughput_stress
    from .quality_stress import compute_quality_stress
    from .transition import Transition
    from .deltas import REGISTRY, get as get_transformation
    from .deltas.base import Transformation
except ImportError:
    from throughput_model import ArchConfig as TArchConfig
    from quality_model import TrainingConfig
    from stress import StressVector, Workload, compute_throughput_stress
    from quality_stress import compute_quality_stress
    from transition import Transition
    from deltas import REGISTRY, get as get_transformation
    from deltas.base import Transformation


_KV_PRECISION_TO_BITS = {
    "bf16": 16, "fp16": 16, "fp32": 32, "tf32": 32,
    "fp8": 8, "int8": 8, "fp4": 4, "int4": 4,
    "mxfp4": 4, "mxfp6": 6, "mxfp8": 8, "nvfp4": 4,
}


def _training_for_arch(training: TrainingConfig, arch: TArchConfig) -> TrainingConfig:
    """Wave 18h: thread the arch's KV-cache precision into the quality-side
    TrainingConfig.

    The KV-quantization quality penalty lives on
    `TrainingConfig.kv_quantization_bits`, not on the quality ArchConfig —
    so before this fix, baseline and candidate quality vectors were both
    computed with the CALLER's TrainingConfig and a KV-precision delta
    (change_precision_per_component:kv=int4) was invisible to the residual
    decomposition: the summary said "Quality cost: negligible" while the
    evaluator's metric table showed +1.2% predicted loss for the same swap.
    """
    bits = _KV_PRECISION_TO_BITS.get(
        str(getattr(arch, "kv_precision", "bf16")).lower(), 16
    )
    if bits == int(getattr(training, "kv_quantization_bits", 16)):
        return training
    return dc_replace(training, kv_quantization_bits=bits)


def _arch_hash(arch: TArchConfig) -> str:
    """Stable short hash for an architecture (for KG keys)."""
    public = {k: v for k, v in asdict(arch).items() if not k.startswith("_")}
    blob = json.dumps(public, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def _parallelism_for(
    arch: TArchConfig,
    defaults: Tuple[int, int, int, int, int],
) -> Tuple[int, int, int, int, int]:
    """Read parallelism overrides set by ChangeParallelism, fall back to defaults."""
    tp = getattr(arch, "_tp_override", defaults[0])
    pp = getattr(arch, "_pp_override", defaults[1])
    ep = getattr(arch, "_ep_override", defaults[2])
    dp = getattr(arch, "_dp_override", defaults[3])
    cp = getattr(arch, "_cp_override", defaults[4])
    return tp, pp, ep, dp, cp


def _structural_execution_error(
    arch: TArchConfig,
    *,
    tp: int,
    pp: int,
    ep: int,
    dp: int,
    cp: int,
) -> str:
    """Return a schema/execution-plan mismatch, or an empty string.

    Delta evaluation used to pass arbitrary parallelism and head layouts
    straight into the analytic models. That priced candidates greenfield
    generation and config validation would never emit. Keep this central so
    every current and future transformation receives the same guard.
    """
    for name, value in (("tp", tp), ("pp", pp), ("ep", ep),
                        ("dp", dp), ("cp", cp)):
        if int(value) < 1:
            return f"{name} must be >= 1"

    n_heads = int(getattr(arch, "n_heads", 0) or 0)
    n_kv = int(getattr(arch, "n_kv_heads", 0) or 0)
    d_model = int(getattr(arch, "d_model", 0) or 0)
    ffn_dim = int(getattr(arch, "ffn_dim", 0) or 0)
    n_layers = int(getattr(arch, "n_layers", 0) or 0)
    if n_kv < 1 or n_heads % n_kv != 0:
        return f"n_kv_heads={n_kv} must divide n_heads={n_heads}"
    if d_model % tp != 0 or n_heads % tp != 0:
        return (
            f"TP={tp} must divide d_model={d_model} and n_heads={n_heads}")
    if n_kv < tp and tp % n_kv != 0:
        return (
            f"TP={tp} is incompatible with n_kv_heads={n_kv}; when KV "
            "heads are fewer than TP, TP must divide into KV groups evenly")
    if ffn_dim % tp != 0:
        return f"TP={tp} must divide ffn_dim={ffn_dim}"
    if pp > 1 and n_layers % pp != 0:
        return f"PP={pp} must divide n_layers={n_layers}"

    moe = getattr(arch, "moe_config", None)
    if moe:
        n_experts = int(moe.get("n_experts", 0) or 0)
        expert_dim = int(moe.get("expert_dim", ffn_dim) or ffn_dim)
        if dp > 1 and (ep > dp or dp % ep != 0):
            return f"EP={ep} must divide DP={dp} when EP overlays DP"
        if n_experts % ep != 0:
            return f"EP={ep} must divide n_experts={n_experts}"
        if expert_dim % tp != 0:
            return f"TP={tp} must divide MoE expert_dim={expert_dim}"
    elif ep != 1:
        return f"EP={ep} is meaningless for a dense FFN; use EP=1"

    layer_types = list(getattr(arch, "layer_type_list", None) or [])
    has_attention = not layer_types or any(
        kind in ("attention", "local_attention") for kind in layer_types)
    if cp > 1 and not has_attention:
        return "CP>1 is unsupported for a pure-state stack"
    if (cp > 1 and str(getattr(arch, "cp_method", "ring")) == "ulysses"
            and n_heads % cp != 0):
        return f"Ulysses CP={cp} must divide n_heads={n_heads}"
    return ""


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
    dp_degree: int = 1,
    cp_degree: int = 1,
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
    if training is None:
        training = TrainingConfig(
            training_tokens=20_000_000_000_000,
            hardware=hardware,
        )
    elif training.hardware != hardware:
        # `hardware` is the execution target for this transition. Keeping a
        # caller-supplied TrainingConfig's default h100 here made valid B200
        # activation formats trip the quality sentinel in the stress summary
        # while the evaluator's B200 metric panel called them feasible.
        training = dc_replace(training, hardware=hardware)

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
    c_tp, c_pp, c_ep, c_dp, c_cp = _parallelism_for(
        candidate, (tp_degree, pp_degree, ep_degree, dp_degree, cp_degree)
    )
    structural_error = _structural_execution_error(
        candidate, tp=c_tp, pp=c_pp, ep=c_ep, dp=c_dp, cp=c_cp)
    if structural_error:
        return Transition(
            transformation_name=transformation.name,
            transformation_params=dict(params),
            baseline_architecture_id=_arch_hash(baseline),
            candidate_architecture_id=_arch_hash(candidate),
            hardware_id=hardware,
            workload_id=workload.workload_id(),
            feasible=False,
            reason_if_infeasible=(
                f"structural_validation_failed: {structural_error}"),
        )

    # --- stress vectors ---
    if _baseline_stress is None:
        b_stress = compute_throughput_stress(
            baseline, hardware, workload,
            tp_degree=tp_degree, pp_degree=pp_degree, ep_degree=ep_degree,
            dp_degree=dp_degree, cp_degree=cp_degree,
            arch_name=baseline_name or "baseline",
        )
    else:
        b_stress = _baseline_stress
    try:
        c_stress = compute_throughput_stress(
            candidate, hardware, workload,
            tp_degree=c_tp, pp_degree=c_pp, ep_degree=c_ep,
            dp_degree=c_dp, cp_degree=c_cp,
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
        b_q = compute_quality_stress(b_qarch, _training_for_arch(training, baseline),
                                      workload_spec={"context_length": workload.decode_kv_len},
                                      arch_name=baseline_name or "baseline")
    except Exception:
        pass
    try:
        c_qarch = transformation.to_quality_arch(candidate)
        c_q = compute_quality_stress(c_qarch, _training_for_arch(training, candidate),
                                      workload_spec={"context_length": workload.decode_kv_len},
                                      arch_name=f"{baseline_name or 'baseline'}+{transformation.name}")
    except Exception:
        pass

    # QualityStressVector preserves the raw residual even though QualityResult
    # caps its displayed loss at 10x baseline. Fail closed here so direct
    # transition callers cannot report an unsupported activation precision as
    # feasible merely because the old predicted-loss sentinel was capped.
    if c_q is not None and c_q.total_residual >= 1000.0:
        return Transition(
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
            feasible=False,
            reason_if_infeasible=(
                "quality_validation_failed: candidate quality model returned "
                "the INFEASIBLE sentinel; check precision/hardware support"
            ),
        )

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
    dp_degree: int = 1,
    cp_degree: int = 1,
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
        dp_degree=dp_degree, cp_degree=cp_degree,
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
            dp_degree=dp_degree, cp_degree=cp_degree,
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
