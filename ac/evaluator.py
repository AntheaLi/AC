"""
Delta Influence Evaluator — core composition.

`evaluate_delta(baseline, hw, workload, delta_name, delta_args)` returns a
DeltaEvaluation with:
    - metric deltas    (TBT, TTFT, training TPS, mem, KV, predicted loss + axes)
    - stress deltas    (10-axis StressVector for baseline + candidate)
    - quality deltas   (7-axis QualityStressVector when available)
    - Pareto position  (relative to baseline-conditioned modifier frontier)
    - justification    (1-3 sentence Markdown summary from justify_transition)

This module is pure composition over existing pieces. It does not add new
physics; it isolates and reports one specific delta's effect.
"""

from __future__ import annotations

import copy
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# --- repo path bootstrap --------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# v0/v1 imports
from optimizer import (  # noqa: E402
    CandidateArch,
    DeploymentConstraints,
    EvaluatedCandidate,
    evaluate_candidate,
)
from throughput_model import ArchConfig as TArchConfig  # noqa: E402
from lattice_engine import estimate_params  # noqa: E402

# v1-stress imports
from stress import (  # noqa: E402
    StressVector,
    Workload,
    STRESS_AXES,
    PRESSURED_OR_WORSE,
    compute_throughput_stress,
    severity_band,
)
from transition import Transition  # noqa: E402
from delta_engine import apply_transition  # noqa: E402
from deltas import REGISTRY, get as get_transformation  # noqa: E402
from justify_transition import justify  # noqa: E402
from optimizer_bridge import candidate_to_arch, stress_relief_vs  # noqa: E402


# =============================================================================
# Public dataclasses
# =============================================================================

@dataclass
class MetricDelta:
    """One scalar metric: baseline value, candidate value, signed delta, % change.

    `direction` is "improves" / "worsens" / "neutral" according to the
    `lower_is_better` flag (TBT, mem → lower better; throughput TPS → higher
    better). Quality loss is lower-is-better.
    """
    name: str
    baseline: float
    candidate: float
    delta: float
    pct_change: float
    direction: str = "neutral"   # "improves" | "worsens" | "neutral"
    lower_is_better: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeltaEvaluation:
    """Full quantitative evaluation of one delta against one baseline."""

    # Identification
    baseline_name: str = ""
    hardware: str = ""
    delta_name: str = ""
    delta_args: Dict[str, Any] = field(default_factory=dict)

    # Feasibility (precondition + apply succeeded; candidate scorable)
    feasible: bool = True
    reason_if_infeasible: str = ""

    # Architecture changes (field-level diff)
    field_changes: List[Dict[str, Any]] = field(default_factory=list)

    # Raw metric deltas (baseline -> candidate)
    metrics: Dict[str, MetricDelta] = field(default_factory=dict)

    # Stress influence
    stress_baseline: Optional[Dict[str, Any]] = None     # StressVector.as_dict()
    stress_candidate: Optional[Dict[str, Any]] = None
    binding_axes_baseline: List[str] = field(default_factory=list)
    binding_axes_relieved: List[str] = field(default_factory=list)
    binding_axes_introduced: List[str] = field(default_factory=list)
    stress_relief_score: float = 0.0
    severe_stress_regression: bool = False

    # Quality residual deltas (7 axes, when both vectors available)
    quality_delta: Dict[str, float] = field(default_factory=dict)
    quality_delta_total: float = 0.0

    # Pareto position (filled by pareto_position.classify_position)
    pareto_position: str = ""           # see PARETO_POSITION_KIND
    pareto_distance: float = 0.0
    pareto_frontier_size: int = 0
    pareto_axes: List[str] = field(default_factory=list)
    pareto_dominated_count: int = 0
    pareto_dominates_count: int = 0

    # Narrative
    justification: str = ""

    def as_dict(self) -> Dict[str, Any]:
        d = {
            "baseline_name": self.baseline_name,
            "hardware": self.hardware,
            "delta_name": self.delta_name,
            "delta_args": dict(self.delta_args),
            "feasible": self.feasible,
            "reason_if_infeasible": self.reason_if_infeasible,
            "field_changes": list(self.field_changes),
            "metrics": {k: v.as_dict() for k, v in self.metrics.items()},
            "stress_baseline": self.stress_baseline,
            "stress_candidate": self.stress_candidate,
            "binding_axes_baseline": list(self.binding_axes_baseline),
            "binding_axes_relieved": list(self.binding_axes_relieved),
            "binding_axes_introduced": list(self.binding_axes_introduced),
            "stress_relief_score": self.stress_relief_score,
            "severe_stress_regression": self.severe_stress_regression,
            "quality_delta": dict(self.quality_delta),
            "quality_delta_total": self.quality_delta_total,
            "pareto_position": self.pareto_position,
            "pareto_distance": self.pareto_distance,
            "pareto_frontier_size": self.pareto_frontier_size,
            "pareto_axes": list(self.pareto_axes),
            "pareto_dominated_count": self.pareto_dominated_count,
            "pareto_dominates_count": self.pareto_dominates_count,
            "justification": self.justification,
        }
        return d


# =============================================================================
# ArchConfig ↔ CandidateArch bridge
# =============================================================================

_KV_PREC_TO_BITS = {"bf16": 16, "fp16": 16, "int8": 8, "fp4": 4, "int4": 4}


def _kv_bits_from_precision(prec: str) -> int:
    return _KV_PREC_TO_BITS.get(prec, 16)


def arch_to_candidate(arch: TArchConfig, base: CandidateArch) -> CandidateArch:
    """Reverse bridge: throughput ArchConfig → optimizer CandidateArch.

    Fields not present on ArchConfig (vocab, attn_precision per-component) are
    copied from `base`. Recomputes `total_params` so downstream evaluation
    reflects the candidate shape.

    Sidecar attrs on `arch` (_tp_override, _mla_latent_dim, _swa_window) are
    not part of CandidateArch; callers must thread overrides through the
    `tp_override` argument to evaluate_delta separately.
    """
    cand = CandidateArch(
        d_model=arch.d_model,
        n_layers=arch.n_layers,
        n_heads=arch.n_heads,
        d_head=arch.d_head,
        n_kv_heads=arch.n_kv_heads,
        ffn_dim=arch.ffn_dim,
        vocab_size=arch.vocab_size,
        weight_precision=base.weight_precision,
        ffn_precision=arch.precision or base.ffn_precision,
        attn_precision=copy.deepcopy(base.attn_precision),
        kv_cache_bits=_kv_bits_from_precision(arch.kv_precision),
        moe=copy.deepcopy(arch.moe_config) if arch.moe_config else None,
        ep_degree=getattr(base, "ep_degree", 1),
        n_dense_ffn_layers=getattr(arch, "n_dense_ffn_layers", 0),
        state_config=copy.deepcopy(arch.state_config) if arch.state_config else None,
        layer_type_list=(
            list(arch.layer_type_list)
            if arch.layer_type_list and any(
                lt != "attention" for lt in arch.layer_type_list)
            else None
        ),
    )
    # Recompute param count from new shape.
    cand.total_params = estimate_params(
        cand.d_model, cand.n_heads, cand.d_head,
        cand.ffn_dim, cand.n_layers, cand.n_kv_heads, cand.vocab_size,
    )
    cand.total_params_b = round(cand.total_params / 1e9, 2)
    # State/MoE bookkeeping
    if cand.state_config is not None and cand.layer_type_list:
        cand.n_attention_layers = sum(
            1 for lt in cand.layer_type_list if lt == "attention")
        cand.n_state_layers = sum(
            1 for lt in cand.layer_type_list if lt == "state")
    if cand.moe is not None:
        cand.moe_style = "fine"   # heuristic; evaluate_candidate doesn't need this
    return cand


def _arch_changes(baseline: CandidateArch,
                   candidate: CandidateArch) -> List[Dict[str, Any]]:
    """Field-level diff for the report."""
    changes = []
    fields_to_diff = (
        "d_model", "n_layers", "n_heads", "d_head", "n_kv_heads",
        "ffn_dim", "vocab_size", "weight_precision", "ffn_precision",
        "kv_cache_bits", "total_params_b",
    )
    for f in fields_to_diff:
        b = getattr(baseline, f, None)
        c = getattr(candidate, f, None)
        if b != c:
            changes.append({"field": f, "baseline": b, "candidate": c})
    # MoE / state structural shifts
    if (baseline.moe is None) != (candidate.moe is None):
        changes.append({"field": "moe_enabled",
                        "baseline": baseline.moe is not None,
                        "candidate": candidate.moe is not None})
    if (baseline.state_config is None) != (candidate.state_config is None):
        changes.append({"field": "state_enabled",
                        "baseline": baseline.state_config is not None,
                        "candidate": candidate.state_config is not None})
    if baseline.attn_precision != candidate.attn_precision:
        changes.append({"field": "attn_precision",
                        "baseline": dict(baseline.attn_precision),
                        "candidate": dict(candidate.attn_precision)})
    return changes


# =============================================================================
# Metric computation
# =============================================================================

def _metric(name: str, baseline: float, candidate: float,
            lower_is_better: bool = True) -> MetricDelta:
    delta = candidate - baseline
    if abs(baseline) > 1e-9:
        pct = (delta / abs(baseline)) * 100.0
    else:
        pct = 0.0
    if abs(delta) < 1e-9:
        direction = "neutral"
    elif (delta < 0 and lower_is_better) or (delta > 0 and not lower_is_better):
        direction = "improves"
    else:
        direction = "worsens"
    return MetricDelta(
        name=name,
        baseline=float(baseline),
        candidate=float(candidate),
        delta=float(delta),
        pct_change=float(pct),
        direction=direction,
        lower_is_better=lower_is_better,
    )


def _evaluated_metrics(base_ev: EvaluatedCandidate,
                       cand_ev: EvaluatedCandidate,
                       base_kv_gb: float,
                       cand_kv_gb: float) -> Dict[str, MetricDelta]:
    """Compute the standard metric panel from two EvaluatedCandidate."""
    m: Dict[str, MetricDelta] = {}
    m["predicted_loss"] = _metric(
        "predicted_loss", base_ev.predicted_loss, cand_ev.predicted_loss,
        lower_is_better=True)
    m["serving_tbt_ms"] = _metric(
        "serving_tbt_ms", base_ev.serving_tbt_ms, cand_ev.serving_tbt_ms,
        lower_is_better=True)
    m["prefill_time_ms"] = _metric(
        "prefill_time_ms",
        base_ev.throughput.prefill_time_ms,
        cand_ev.throughput.prefill_time_ms,
        lower_is_better=True)
    m["training_tps"] = _metric(
        "training_tps", base_ev.training_tps, cand_ev.training_tps,
        lower_is_better=False)
    m["memory_per_gpu_gb"] = _metric(
        "memory_per_gpu_gb", base_ev.memory_per_gpu_gb,
        cand_ev.memory_per_gpu_gb, lower_is_better=True)
    m["kv_cache_gb"] = _metric(
        "kv_cache_gb", base_kv_gb, cand_kv_gb, lower_is_better=True)
    m["total_params_b"] = _metric(
        "total_params_b",
        base_ev.arch.total_params_b, cand_ev.arch.total_params_b,
        lower_is_better=True)  # neutral — direction signal less meaningful
    # Quality-term decomposition (when available)
    base_q_terms = getattr(base_ev.quality, "terms", {}) or {}
    cand_q_terms = getattr(cand_ev.quality, "terms", {}) or {}
    for term_name in sorted(set(base_q_terms) | set(cand_q_terms)):
        b_term = base_q_terms.get(term_name)
        c_term = cand_q_terms.get(term_name)
        b_val = float(getattr(b_term, "value", 0.0) or 0.0) if b_term else 0.0
        c_val = float(getattr(c_term, "value", 0.0) or 0.0) if c_term else 0.0
        m[f"quality_{term_name}"] = _metric(
            f"quality_{term_name}", b_val, c_val, lower_is_better=True)
    return m


def _kv_cache_gb_for_cand(cand: CandidateArch,
                          constraints: DeploymentConstraints,
                          tp: int) -> float:
    """Mirror modifier._kv_cache_gb logic so KV diagnostic stays consistent."""
    bits_to_bpe = {16: 2, 8: 1, 4: 0.5}
    kv_bpe = bits_to_bpe.get(int(cand.kv_cache_bits), 2)
    context = constraints.prompt_len or constraints.context_length or 2048
    batch = 1
    layers_per_stage = cand.n_layers // max(constraints.pp, 1)
    kv_heads_per_gpu = max(1, math.ceil(cand.n_kv_heads / max(tp, 1)))
    bytes_total = (
        batch * context * layers_per_stage * 2
        * kv_heads_per_gpu * cand.d_head * kv_bpe
    )
    return bytes_total / (1024 ** 3)


# =============================================================================
# Core entry point
# =============================================================================

def evaluate_delta(
    baseline_candidate: CandidateArch,
    hardware: str,
    constraints: DeploymentConstraints,
    delta_name: str,
    delta_args: Optional[Dict[str, Any]] = None,
    *,
    baseline_name: str = "baseline",
    tp_override: Optional[int] = None,
    include_pareto: bool = True,
    pareto_classifier=None,
) -> DeltaEvaluation:
    """Evaluate one named transformation against a baseline.

    Parameters
    ----------
    baseline_candidate : CandidateArch
        The architecture to evaluate the delta against. Typically obtained
        from `baseline.load_baseline_model(path).candidate`.
    hardware : str
        Hardware name ("h100", "b200", "tpu_v5p").
    constraints : DeploymentConstraints
        Carries tp/pp/context_length/serving_batch — all of which feed the
        throughput model and stress vector computation.
    delta_name : str
        One of the names in v1-stress/deltas/REGISTRY (e.g. "swap_attention_to_gqa").
    delta_args : dict, optional
        Parameters passed to the transformation's `apply` method.
    baseline_name : str
        Human-readable name for the baseline (passed through to the
        StressVector / justification text).
    tp_override : int, optional
        Override the candidate's tp degree (when the delta is
        `change_parallelism`, the candidate's tp comes from the sidecar
        `_tp_override`; pass that here as well so evaluate_candidate uses it).
    include_pareto : bool
        When True (default), also compute the candidate's Pareto position
        against a baseline-conditioned modifier frontier. Set False for
        cheap repeated calls (e.g. inside a round-trip test).
    pareto_classifier : callable, optional
        Custom (baseline_record, candidate_record, frontier) →
        (position_str, distance, axes) function. When None, the default
        from pareto_position.classify_position is used. Injectable so
        evaluator.py can avoid importing pareto_position at top level
        (it imports back via this module).
    """
    delta_args = dict(delta_args or {})
    constraints = copy.deepcopy(constraints)

    # 1) Resolve and apply the transformation
    try:
        xf = get_transformation(delta_name)
    except KeyError as e:
        return DeltaEvaluation(
            baseline_name=baseline_name,
            hardware=hardware,
            delta_name=delta_name,
            delta_args=delta_args,
            feasible=False,
            reason_if_infeasible=str(e),
        )

    base_arch = candidate_to_arch(
        baseline_candidate,
        batch_size=max(1, int(getattr(constraints, "serving_batch", 1) or 1)),
        seq_len=int(constraints.prompt_len or constraints.context_length or 2048),
    )

    workload = Workload(
        batch_size=base_arch.batch_size,
        prefill_seq_len=int(constraints.prompt_len or constraints.context_length or 2048),
        decode_kv_len=int(constraints.context_length or 2048),
        phase="decode",
    )

    transition: Transition = apply_transition(
        base_arch, xf, params=delta_args,
        hardware=hardware,
        workload=workload,
        tp_degree=int(constraints.tp),
        pp_degree=int(getattr(constraints, "pp", 1) or 1),
        ep_degree=int(getattr(baseline_candidate, "ep_degree", 1) or 1),
        baseline_name=baseline_name,
    )

    ev = DeltaEvaluation(
        baseline_name=baseline_name,
        hardware=hardware,
        delta_name=delta_name,
        delta_args=delta_args,
    )

    if not transition.feasible:
        ev.feasible = False
        ev.reason_if_infeasible = transition.reason_if_infeasible
        ev.justification = justify(transition)
        return ev

    # 2) Compose the candidate ArchConfig back into a CandidateArch so we can
    #    drive evaluate_candidate (which expects CandidateArch).
    candidate_arch_t: TArchConfig = xf.apply(base_arch, **delta_args)
    candidate_cand: CandidateArch = arch_to_candidate(
        candidate_arch_t, baseline_candidate)

    # Effective TP: if the delta is change_parallelism, the candidate carries
    # an _tp_override sidecar; thread that into evaluate_candidate.
    effective_tp = (
        int(tp_override)
        if tp_override is not None
        else int(getattr(candidate_arch_t, "_tp_override",
                          getattr(base_arch, "_tp_override", constraints.tp)))
    )

    base_constraints = copy.deepcopy(constraints)
    base_constraints.tp = int(constraints.tp)
    cand_constraints = copy.deepcopy(constraints)
    cand_constraints.tp = effective_tp

    # 3) Evaluate baseline + candidate
    try:
        base_ev = evaluate_candidate(baseline_candidate, hardware, base_constraints)
        cand_ev = evaluate_candidate(candidate_cand, hardware, cand_constraints)
    except Exception as e:
        ev.feasible = False
        ev.reason_if_infeasible = f"evaluate_candidate_failed: {type(e).__name__}: {e}"
        ev.justification = justify(transition)
        # Still surface the stress story we computed.
        if transition.baseline_stress is not None:
            ev.stress_baseline = transition.baseline_stress.as_dict()
        if transition.candidate_stress is not None:
            ev.stress_candidate = transition.candidate_stress.as_dict()
        return ev

    # 4) Metric panel
    base_kv = _kv_cache_gb_for_cand(baseline_candidate, base_constraints,
                                     int(constraints.tp))
    cand_kv = _kv_cache_gb_for_cand(candidate_cand, cand_constraints, effective_tp)
    ev.metrics = _evaluated_metrics(base_ev, cand_ev, base_kv, cand_kv)

    # 5) Field changes
    ev.field_changes = _arch_changes(baseline_candidate, candidate_cand)

    # 6) Stress influence (transition already computed all of this)
    if transition.baseline_stress is not None:
        ev.stress_baseline = transition.baseline_stress.as_dict()
        ev.binding_axes_baseline = list(transition.baseline_stress.binding_axes)
    if transition.candidate_stress is not None:
        ev.stress_candidate = transition.candidate_stress.as_dict()
    ev.binding_axes_relieved = list(transition.relieved_binding_axes)
    ev.binding_axes_introduced = list(transition.new_binding_axes)
    ev.stress_relief_score = float(transition.relief_score())
    ev.severe_stress_regression = bool(transition.has_severe_regression())

    # 7) Quality decomposition
    ev.quality_delta = dict(transition.delta_quality)
    ev.quality_delta_total = float(transition.delta_quality_total)

    # 8) Justification (1-3 sentences)
    ev.justification = justify(transition)

    # 9) Pareto position
    if include_pareto:
        # Import here to avoid a top-level circular import on package init.
        from pareto_position import classify_position
        clf = pareto_classifier or classify_position
        try:
            position_str, distance, axes, dominated, dominates, frontier_size = clf(
                base_ev=base_ev, cand_ev=cand_ev,
                baseline_candidate=baseline_candidate,
                hardware=hardware, constraints=constraints,
            )
            ev.pareto_position = position_str
            ev.pareto_distance = float(distance)
            ev.pareto_axes = list(axes)
            ev.pareto_dominated_count = int(dominated)
            ev.pareto_dominates_count = int(dominates)
            ev.pareto_frontier_size = int(frontier_size)
        except Exception as e:
            ev.pareto_position = "unknown"
            ev.pareto_distance = 0.0
            ev.pareto_axes = []
            ev.pareto_frontier_size = 0
            # Surface as a note in justification
            ev.justification += f"\n[pareto classification failed: {type(e).__name__}: {e}]"

    return ev


# =============================================================================
# Convenience: compose multiple deltas
# =============================================================================

def evaluate_delta_sequence(
    baseline_candidate: CandidateArch,
    hardware: str,
    constraints: DeploymentConstraints,
    deltas: List[Tuple[str, Dict[str, Any]]],
    *,
    baseline_name: str = "baseline",
    include_pareto: bool = True,
) -> DeltaEvaluation:
    """Apply a sequence of transformations in order, then evaluate the
    composed candidate as a single delta against the original baseline.

    The intermediate transformations don't get their own evaluation here —
    the report is the *cumulative* effect of the sequence. This is what the
    modifier does internally; this helper just exposes it.
    """
    if not deltas:
        # No-op: synthesize a feasible "(none)" DeltaEvaluation with
        # baseline self-comparison so callers get a stable shape back.
        base_constraints = copy.deepcopy(constraints)
        base_constraints.tp = int(constraints.tp)
        base_ev_eval = evaluate_candidate(
            baseline_candidate, hardware, base_constraints)
        base_kv = _kv_cache_gb_for_cand(
            baseline_candidate, base_constraints, int(constraints.tp))
        ev_noop = DeltaEvaluation(
            baseline_name=baseline_name,
            hardware=hardware,
            delta_name="(none)",
            delta_args={},
            feasible=True,
            field_changes=[],
            metrics=_evaluated_metrics(base_ev_eval, base_ev_eval,
                                        base_kv, base_kv),
            justification="No deltas applied.",
        )
        return ev_noop

    base_arch = candidate_to_arch(
        baseline_candidate,
        batch_size=max(1, int(getattr(constraints, "serving_batch", 1) or 1)),
        seq_len=int(constraints.prompt_len or constraints.context_length or 2048),
    )
    cumulative = base_arch
    for name, args in deltas:
        xf = get_transformation(name)
        ok, reason = xf.precondition(cumulative)
        if not ok:
            ev = DeltaEvaluation(
                baseline_name=baseline_name, hardware=hardware,
                delta_name=name, delta_args=dict(args or {}),
                feasible=False,
                reason_if_infeasible=f"precondition_failed: {reason}",
            )
            return ev
        cumulative = xf.apply(cumulative, **(args or {}))
    composed_cand = arch_to_candidate(cumulative, baseline_candidate)

    # Re-use evaluate_delta machinery by faking a single "delta" — but that
    # requires the transformation library to know about the composed form.
    # Cleaner: do the panel directly here.
    workload = Workload(
        batch_size=base_arch.batch_size,
        prefill_seq_len=int(constraints.prompt_len or constraints.context_length or 2048),
        decode_kv_len=int(constraints.context_length or 2048),
        phase="decode",
    )
    base_stress = compute_throughput_stress(
        base_arch, hardware, workload,
        tp_degree=int(constraints.tp),
        pp_degree=int(getattr(constraints, "pp", 1) or 1),
        ep_degree=int(getattr(baseline_candidate, "ep_degree", 1) or 1),
        arch_name=baseline_name,
    )
    cand_tp = int(getattr(cumulative, "_tp_override", constraints.tp))
    cand_pp = int(getattr(cumulative, "_pp_override",
                          getattr(constraints, "pp", 1) or 1))
    cand_ep = int(getattr(cumulative, "_ep_override",
                          getattr(baseline_candidate, "ep_degree", 1) or 1))
    cand_stress = compute_throughput_stress(
        cumulative, hardware, workload,
        tp_degree=cand_tp, pp_degree=cand_pp, ep_degree=cand_ep,
        arch_name=baseline_name + "+composed",
    )

    base_constraints = copy.deepcopy(constraints)
    base_constraints.tp = int(constraints.tp)
    cand_constraints = copy.deepcopy(constraints)
    cand_constraints.tp = cand_tp
    base_ev = evaluate_candidate(baseline_candidate, hardware, base_constraints)
    cand_ev = evaluate_candidate(composed_cand, hardware, cand_constraints)
    base_kv = _kv_cache_gb_for_cand(baseline_candidate, base_constraints,
                                     int(constraints.tp))
    cand_kv = _kv_cache_gb_for_cand(composed_cand, cand_constraints, cand_tp)

    ev = DeltaEvaluation(
        baseline_name=baseline_name,
        hardware=hardware,
        delta_name="+".join(name for name, _ in deltas),
        delta_args={"sequence": [{"name": n, "args": dict(a or {})}
                                   for n, a in deltas]},
        feasible=True,
        field_changes=_arch_changes(baseline_candidate, composed_cand),
        metrics=_evaluated_metrics(base_ev, cand_ev, base_kv, cand_kv),
        stress_baseline=base_stress.as_dict(),
        stress_candidate=cand_stress.as_dict(),
        binding_axes_baseline=list(base_stress.binding_axes),
    )
    rel = stress_relief_vs(base_stress, cand_stress)
    ev.binding_axes_relieved = list(rel["relieved_binding_axes"])
    ev.binding_axes_introduced = list(rel["new_binding_axes"])
    ev.stress_relief_score = float(rel["relief_score"])
    ev.severe_stress_regression = bool(rel["severe_regression"])

    # Build a synthetic Transition just to reuse justify() prose.
    syn_transition = Transition(
        transformation_name=ev.delta_name,
        transformation_params=ev.delta_args,
        baseline_stress=base_stress,
        candidate_stress=cand_stress,
        feasible=True,
    )
    syn_transition.compute_deltas()
    ev.justification = justify(syn_transition)
    ev.quality_delta = dict(syn_transition.delta_quality)
    ev.quality_delta_total = float(syn_transition.delta_quality_total)

    if include_pareto:
        from pareto_position import classify_position
        try:
            position_str, distance, axes, dominated, dominates, frontier_size = (
                classify_position(
                    base_ev=base_ev, cand_ev=cand_ev,
                    baseline_candidate=baseline_candidate,
                    hardware=hardware, constraints=constraints,
                )
            )
            ev.pareto_position = position_str
            ev.pareto_distance = float(distance)
            ev.pareto_axes = list(axes)
            ev.pareto_dominated_count = int(dominated)
            ev.pareto_dominates_count = int(dominates)
            ev.pareto_frontier_size = int(frontier_size)
        except Exception as e:
            ev.pareto_position = "unknown"
            ev.justification += (f"\n[pareto classification failed: "
                                  f"{type(e).__name__}: {e}]")

    return ev
