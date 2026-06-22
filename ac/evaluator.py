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
from penalties import INFEASIBLE  # noqa: E402


# Any predicted_loss at or above this threshold is treated as the
# INFEASIBLE sentinel leaking through from the quality-model penalty
# stack (see ac/penalties.py:INFEASIBLE). The 0.5 factor gives margin
# against small residuals that may be added on top of the marker before
# it reaches the metrics table.
_INFEASIBLE_LOSS_THRESHOLD = 0.5 * INFEASIBLE


def _loss_is_infeasible_sentinel(value: Any) -> bool:
    """Return True if `value` looks like the INFEASIBLE marker."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(v):
        return True
    return v >= _INFEASIBLE_LOSS_THRESHOLD


def _infeasibility_check(
    base_ev: EvaluatedCandidate,
    cand_ev: EvaluatedCandidate,
) -> Optional[str]:
    """Detect baseline/candidate carrying the INFEASIBLE sentinel.

    The quality model returns `INFEASIBLE` (1e6) when its inputs are
    structurally invalid (e.g. an MoE config with `expert_parallel`
    missing puts every expert on every rank). That marker used to leak
    straight through to the report and showed up as a delta of
    ~-1.96e6 with a misleading "improves" direction. Surface it as a
    real infeasibility instead.
    """
    base_bad = _loss_is_infeasible_sentinel(getattr(base_ev, "predicted_loss", None))
    cand_bad = _loss_is_infeasible_sentinel(getattr(cand_ev, "predicted_loss", None))
    if not (base_bad or cand_bad):
        return None
    parts: List[str] = []
    if base_bad:
        parts.append(
            "baseline_loss=INFEASIBLE — the baseline config itself failed "
            "the quality-model feasibility check (common cause: MoE "
            "config without parallelism.expert_parallel set, which puts "
            "all experts on every rank). Re-check the baseline config "
            "before interpreting the delta."
        )
    if cand_bad:
        parts.append(
            "candidate_loss=INFEASIBLE — the post-delta config failed "
            "the quality-model feasibility check. Inspect the field "
            "changes and re-run with revised --apply-args."
        )
    return " ".join(parts)


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
        d = asdict(self)
        # Fix #4: emit JSON-safe sentinel for the undefined-relative-change
        # case (baseline ~0, delta non-zero). math.inf survives Python's
        # json.dumps as `Infinity`, which is *not* strict JSON and breaks
        # downstream consumers (jq, R jsonlite, Pandas read_json). Convert
        # to None so the field is unambiguously "undefined" but valid JSON.
        pct = d.get("pct_change", 0.0)
        if isinstance(pct, float) and not math.isfinite(pct):
            d["pct_change"] = None
        return d


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

    # Fix #3: Resolved workload (batch, prompt_len, context, scheduler, TP/PP/DP)
    # — emitted into the report and JSON so a reader can verify which
    # workload the predictions describe, and tell when a delta-eval and a
    # greenfield emission were apples-to-apples.
    resolved_workload: Optional[Dict[str, Any]] = None

    # Delta-specific resolution notes — currently used by `add_state_layers`
    # to surface the resolved state_fraction so the user can verify which
    # CLI input mode produced what layer count (kills the ratio direction
    # confusion). Other deltas may populate this as more user-facing
    # interpretation helpers are added.
    delta_summary: Optional[Dict[str, Any]] = None

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
            "resolved_workload": dict(self.resolved_workload or {}),
            "delta_summary": dict(self.delta_summary or {}),
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
        # Thread MLA fields through so the throughput model's real MLA
        # branch fires after a swap_attention_to_mla delta. Without this,
        # the candidate evaluator silently saw type=full / latent=0 and the
        # delta paid the quality cost without booking the compute/memory
        # benefit.
        attention_type=getattr(arch, "attention_type", "full") or "full",
        mla_kv_latent_dim=int(getattr(arch, "mla_kv_latent_dim", 0) or 0),
        mla_q_latent_dim=int(getattr(arch, "mla_q_latent_dim", 0) or 0),
        mla_rope_head_dim=int(getattr(arch, "mla_rope_head_dim", 0) or 0),
        mla_nope_head_dim=int(getattr(arch, "mla_nope_head_dim", 0) or 0),
        # SWA: read both the canonical local_window and the sidecar
        # _swa_window so candidates produced either by the delta engine
        # (sidecar) or by a baseline loader that already populated the
        # field both reach the throughput / KV / quality models.
        swa_window=int(
            getattr(arch, "_swa_window", 0)
            or getattr(arch, "local_window", 0)
            or 0
        ),
    )
    # Normalise attention_type label when SWA window is set but the upstream
    # arch left attention_type="full" (delta.to_quality_arch sets type=swa
    # but a baseline-loaded SWA config may carry just the window).
    if cand.swa_window > 0 and cand.attention_type == "full":
        cand.attention_type = "swa"
    # Recompute param count from new shape. For dense models this is direct;
    # for MoE we need to add (n_experts - 1) expert masses on top of the
    # single-expert mass that estimate_params produces, plus any shared expert.
    dense_like_params = estimate_params(
        cand.d_model, cand.n_heads, cand.d_head,
        cand.ffn_dim, cand.n_layers, cand.n_kv_heads, cand.vocab_size,
    )
    if cand.moe is not None:
        n_experts = int(cand.moe.get("n_experts", 1))
        top_k = max(1, int(cand.moe.get("top_k", 1)))
        expert_dim = int(cand.moe.get("expert_dim", cand.ffn_dim))
        per_expert_ffn = 3 * cand.d_model * expert_dim
        # Strip the one-expert FFN mass that estimate_params added.
        attention_and_other = dense_like_params - per_expert_ffn * cand.n_layers
        cand.total_params = attention_and_other + n_experts * per_expert_ffn * cand.n_layers
        cand.active_params = attention_and_other + top_k * per_expert_ffn * cand.n_layers
        shared = cand.moe.get("shared_expert")
        if isinstance(shared, dict):
            shared_ffn_dim = int(shared.get("ffn_dim", 0))
            if shared_ffn_dim > 0:
                shared_mass = 3 * cand.d_model * shared_ffn_dim * cand.n_layers
                cand.total_params += shared_mass
                cand.active_params += shared_mass
        cand.active_params_b = round(cand.active_params / 1e9, 2)
    else:
        cand.total_params = dense_like_params
        cand.active_params = dense_like_params
        cand.active_params_b = round(cand.active_params / 1e9, 2)
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


_SIDECAR_LABELS = {
    "_mla_latent_dim": "attention.mla_latent_dim",
    "_swa_window": "attention.sliding_window",
    "_tp_override": "parallelism.tensor_parallel",
    "_pp_override": "parallelism.pipeline_parallel",
    "_ep_override": "parallelism.expert_parallel",
    "_cp_override": "parallelism.context_parallel",
}

# Fix #6: when a parallelism sidecar is set on the candidate, we report it as
# a `parallelism.tensor_parallel` (etc.) change. The baseline doesn't carry a
# sidecar — its parallelism lives in the DeploymentConstraints or the
# CandidateArch's cp_degree/ep_degree fields — so the diff used to render
# `baseline: None`. We try multiple attribute lookups before giving up.
_SIDECAR_BASELINE_ATTRS = {
    "_tp_override": ("tp_degree", "tensor_parallel", "tp"),
    "_pp_override": ("pp_degree", "pipeline_parallel", "pp"),
    "_ep_override": ("ep_degree", "expert_parallel", "ep"),
    "_cp_override": ("cp_degree", "context_parallel", "cp"),
}


def _append_sidecar_changes(
    changes: List[Dict[str, Any]],
    baseline_arch: Any,
    candidate_arch: Any,
    baseline_constraints: Any = None,
) -> None:
    """Surface non-dataclass sidecar attributes (MLA latent, SWA window, etc.)
    in the field-level diff. Without this, attention swaps that only set a
    sidecar would report 'no structural changes' in the user-facing report.

    `baseline_constraints` is the optional DeploymentConstraints used to
    evaluate the baseline; we read TP/PP/CP/EP from it as the last fallback
    when the candidate arch carries no canonical attribute. (Fix #6.)
    """
    for attr, label in _SIDECAR_LABELS.items():
        b = getattr(baseline_arch, attr, None)
        c = getattr(candidate_arch, attr, None)
        # Fix #6: if the baseline has no sidecar, walk the canonical attribute
        # candidates on the arch, then on the constraints, before falling back
        # to None.
        if b is None and attr in _SIDECAR_BASELINE_ATTRS:
            for cand_attr in _SIDECAR_BASELINE_ATTRS[attr]:
                canonical = getattr(baseline_arch, cand_attr, None)
                if canonical is not None:
                    b = canonical
                    break
            if b is None and baseline_constraints is not None:
                for cand_attr in _SIDECAR_BASELINE_ATTRS[attr]:
                    canonical = getattr(baseline_constraints, cand_attr, None)
                    if canonical is not None:
                        b = canonical
                        break
        # B2 fix: do the same fallback for the candidate. Before this fix,
        # the candidate sidecar would be None whenever the delta didn't
        # explicitly override parallelism, and the diff would render an
        # incorrect "parallelism.tensor_parallel: 8 -> None" row even
        # though the candidate was *evaluated* at the baseline TP=8. We
        # walk the candidate's own canonical attributes first; only if
        # those are also absent do we inherit from the baseline (which
        # mirrors the runtime behaviour — the modifier/delta path inherits
        # baseline parallelism when nothing is overridden).
        if c is None and attr in _SIDECAR_BASELINE_ATTRS:
            for cand_attr in _SIDECAR_BASELINE_ATTRS[attr]:
                canonical = getattr(candidate_arch, cand_attr, None)
                if canonical is not None:
                    c = canonical
                    break
            if c is None:
                # Inherit baseline. If the baseline itself resolved to
                # None we leave c=None (truly unspecified on both sides);
                # the `b != c` check below will then no-op as expected.
                c = b
        if b != c:
            changes.append({"field": label, "baseline": b, "candidate": c})
    # Also emit attention.type when an attention-swap sidecar lands.
    base_type = "full"
    if getattr(baseline_arch, "_mla_latent_dim", None) is not None:
        base_type = "mla"
    elif getattr(baseline_arch, "_swa_window", None) is not None:
        base_type = "swa"
    cand_type = "full"
    if getattr(candidate_arch, "_mla_latent_dim", None) is not None:
        cand_type = "mla"
    elif getattr(candidate_arch, "_swa_window", None) is not None:
        cand_type = "swa"
    if base_type != cand_type:
        changes.append({"field": "attention.type",
                        "baseline": base_type, "candidate": cand_type})
    # Provenance trail (which deltas were applied, in order). Useful even
    # if the field-level shape diff comes out small.
    trail = list(getattr(candidate_arch, "_applied_deltas", []) or [])
    base_trail = list(getattr(baseline_arch, "_applied_deltas", []) or [])
    if trail and trail != base_trail:
        changes.append({"field": "applied_deltas",
                        "baseline": base_trail or None,
                        "candidate": trail})


def _arch_changes(baseline: CandidateArch,
                   candidate: CandidateArch) -> List[Dict[str, Any]]:
    """Field-level diff for the report.

    B1 fix: previously this diff only inspected a hand-picked list of
    top-level scalars (`d_model`, `n_layers`, …) plus three enable-flags
    (`moe_enabled`, `state_enabled`, `attn_precision`). That hid the
    structural effect of every delta that operates inside `layer_configs` or
    on a sub-field — `densify_first_k`, `change_moe_topology`,
    `add_state_layers`, `swap_attention_to_mla`, `swap_attention_to_swa`,
    `change_parallelism` (EP/CP). The throughput model *did* see those
    changes (metrics moved), but the report's diff table didn't show them,
    which in turn fired the misleading "structurally identical" callout in
    report.py.

    The fix here is to extend the diff to cover every CandidateArch field
    that any registered delta can move. We keep the original `fields_to_diff`
    untouched so existing test snapshots are preserved.
    """
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

    # B1 fix: attention type + MLA/SWA shape parameters. The MLA/SWA
    # *sidecar* attributes are emitted by _append_sidecar_changes, but the
    # canonical CandidateArch fields (set by the modifier or a chained
    # delta) also need to be visible.
    attn_shape_fields = (
        ("attention_type", "attention.type"),
        ("mla_kv_latent_dim", "attention.mla_kv_latent_dim"),
        ("mla_q_latent_dim", "attention.mla_q_latent_dim"),
        ("mla_rope_head_dim", "attention.mla_rope_head_dim"),
        ("mla_nope_head_dim", "attention.mla_nope_head_dim"),
        ("swa_window", "attention.sliding_window"),
    )
    for src, label in attn_shape_fields:
        b = getattr(baseline, src, None)
        c = getattr(candidate, src, None)
        # Treat 0 and None as equivalent (the canonical "absent" marker
        # used by MLA/SWA fields when the attention type is "full").
        b_norm = None if b in (0, None) else b
        c_norm = None if c in (0, None) else c
        if b_norm != c_norm:
            changes.append({"field": label, "baseline": b, "candidate": c})

    # B1 fix: first-K-dense FFN prefix (densify_first_k is the canonical
    # case where every top-level scalar is unchanged).
    if getattr(baseline, "n_dense_ffn_layers", 0) != getattr(
            candidate, "n_dense_ffn_layers", 0):
        changes.append({
            "field": "n_dense_ffn_layers",
            "baseline": getattr(baseline, "n_dense_ffn_layers", 0),
            "candidate": getattr(candidate, "n_dense_ffn_layers", 0),
        })

    # B1 fix: MoE topology. The previous diff only fired when MoE was
    # enabled/disabled; reshapes (n_experts, top_k, expert_dim,
    # capacity_factor, shared_expert) silently moved metrics without
    # appearing in the diff table.
    if (baseline.moe is None) != (candidate.moe is None):
        changes.append({"field": "moe_enabled",
                        "baseline": baseline.moe is not None,
                        "candidate": candidate.moe is not None})
    elif baseline.moe is not None and candidate.moe is not None:
        moe_topology_fields = (
            "n_experts", "top_k", "expert_dim",
            "capacity_factor", "shared_expert", "router",
        )
        for f in moe_topology_fields:
            b = baseline.moe.get(f) if isinstance(baseline.moe, dict) else None
            c = candidate.moe.get(f) if isinstance(candidate.moe, dict) else None
            if b != c:
                changes.append({
                    "field": f"moe.{f}",
                    "baseline": b,
                    "candidate": c,
                })

    # Expert-parallel degree on the canonical arch (set by modifier sweeps
    # or `change_parallelism`). Sidecar overrides are also handled by
    # _append_sidecar_changes; this catches the case where the canonical
    # value differs.
    if getattr(baseline, "ep_degree", 1) != getattr(candidate, "ep_degree", 1):
        changes.append({
            "field": "parallelism.expert_parallel",
            "baseline": getattr(baseline, "ep_degree", 1),
            "candidate": getattr(candidate, "ep_degree", 1),
        })

    # B1 fix: state/hybrid shape. add_state_layers changes
    # `n_state_layers`, `n_attention_layers`, `hybrid_ratio`,
    # `placement_strategy`, and `state_config.type` without touching any
    # of the original scalar fields_to_diff.
    if (baseline.state_config is None) != (candidate.state_config is None):
        changes.append({"field": "state_enabled",
                        "baseline": baseline.state_config is not None,
                        "candidate": candidate.state_config is not None})
    if isinstance(baseline.state_config, dict) and isinstance(
            candidate.state_config, dict):
        for f in ("type", "d_state", "state_expansion",
                  "n_heads", "d_head", "state_precision"):
            b = baseline.state_config.get(f)
            c = candidate.state_config.get(f)
            if b != c:
                changes.append({
                    "field": f"state.{f}",
                    "baseline": b,
                    "candidate": c,
                })
    state_layout_fields = (
        ("n_state_layers", "state.n_layers"),
        ("n_attention_layers", "attention.n_layers"),
        ("hybrid_ratio", "state.hybrid_ratio"),
        ("placement_strategy", "state.placement_strategy"),
    )
    for src, label in state_layout_fields:
        b = getattr(baseline, src, None)
        c = getattr(candidate, src, None)
        # The "none" placement_strategy default and 0 attention/state
        # layer counts are dataclass defaults, not user-supplied values.
        # Don't emit ghost diffs when both sides hold the default.
        b_norm = None if b in (0, "", "none", None) else b
        c_norm = None if c in (0, "", "none", None) else c
        if b_norm != c_norm:
            changes.append({"field": label, "baseline": b, "candidate": c})

    # CP method (set by `change_parallelism` when cp != 1).
    if getattr(baseline, "cp_method", "ring") != getattr(
            candidate, "cp_method", "ring"):
        changes.append({
            "field": "parallelism.cp_method",
            "baseline": getattr(baseline, "cp_method", "ring"),
            "candidate": getattr(candidate, "cp_method", "ring"),
        })

    # MTP shape — meaningful for any chained delta that touches
    # `mtp_n_predict_depths` even though no delta in REGISTRY currently
    # writes it (forward-compatible).
    if getattr(baseline, "mtp_n_predict_depths", 0) != getattr(
            candidate, "mtp_n_predict_depths", 0):
        changes.append({
            "field": "mtp.n_predict_depths",
            "baseline": getattr(baseline, "mtp_n_predict_depths", 0),
            "candidate": getattr(candidate, "mtp_n_predict_depths", 0),
        })

    if baseline.attn_precision != candidate.attn_precision:
        changes.append({"field": "attn_precision",
                        "baseline": dict(baseline.attn_precision),
                        "candidate": dict(candidate.attn_precision)})
    return changes


# Aliases that collapse to a single conceptual field. The right-hand value
# is the *canonical* label we want the user to see; whichever row arrives
# first (which is the canonical _arch_changes row, since that helper runs
# before _append_sidecar_changes) wins, and any subsequent alias is dropped.
#
# `attention.mla_latent_dim` is the legacy sidecar label for what is really
# the c_kv dimension; `attention.mla_kv_latent_dim` is the canonical name
# (matches `CandidateArch.mla_kv_latent_dim` and the schema field). They
# describe the same number.
_FIELD_ALIASES = {
    "attention.mla_latent_dim": "attention.mla_kv_latent_dim",
}


def _dedupe_field_changes(
    changes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Collapse duplicate / aliased field rows produced when the canonical
    arch diff and the sidecar diff both touch the same conceptual field.

    Before this pass the report could show, for a single
    `swap_attention_to_mla` call:

        | `attention.mla_kv_latent_dim` | `0`    | `256` |
        | `attention.mla_latent_dim`    | `None` | `256` |
        | `attention.type`              | `full` | `mla` |
        | `attention.type`              | `full` | `mla` |

    All four rows describe the same two facts (MLA was switched on,
    c_kv=256). We keep the first occurrence per canonical field name
    (where "canonical" applies the _FIELD_ALIASES map) and drop the rest,
    preserving order otherwise.
    """
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for ch in changes:
        raw_field = ch.get("field")
        canonical = _FIELD_ALIASES.get(raw_field, raw_field)
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(ch)
    return out


# =============================================================================
# Metric computation
# =============================================================================

def _metric(name: str, baseline: float, candidate: float,
            lower_is_better: bool = True) -> MetricDelta:
    delta = candidate - baseline
    # Fix #4: when baseline is ~0 and there is a real move, returning
    # pct=0.0 is misleading — downstream dashboards will read "no change"
    # despite a non-zero delta. Distinguish three cases:
    #   * baseline non-zero        : normal relative change
    #   * baseline ~0, delta ~0    : truly neutral, pct = 0.0
    #   * baseline ~0, delta != 0  : undefined relative change; we store
    #                                math.inf (signed) so it surfaces as a
    #                                sentinel in JSON instead of silently
    #                                reading "no change".
    if abs(baseline) > 1e-9:
        pct = (delta / abs(baseline)) * 100.0
    elif abs(delta) < 1e-9:
        pct = 0.0
    else:
        pct = math.copysign(math.inf, delta)
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
    """Mirror modifier._kv_cache_gb logic so KV diagnostic stays consistent.

    Three correctness branches that earlier versions missed:

    - **MLA**: storage per token per layer is `(c_kv + d_rope) * bpe`, not
      `2 * n_kv_heads * d_head * bpe`. Required so swap_attention_to_mla
      books the cache shrink.
    - **SWA**: per-token cache is bounded by the sliding window, so the
      effective sequence length is `min(context, swa_window)`. Without
      this branch, swap_attention_to_swa was a complete no-op on KV.
    - **State-hybrid**: only attention layers hold a KV cache; state-mixer
      layers carry their own (much smaller) recurrent state. We multiply
      the per-layer KV cost by the attention-layer fraction so
      add_state_layers ratio=1:3 cuts KV by ~25%, not 0%.
    """
    bits_to_bpe = {16: 2, 8: 1, 4: 0.5}
    kv_bpe = bits_to_bpe.get(int(cand.kv_cache_bits), 2)
    context = constraints.prompt_len or constraints.context_length or 2048
    # SWA cap. Both `swa_window` (canonical) and the historical
    # `local_window` field are honoured.
    window = int(
        getattr(cand, "swa_window", 0)
        or getattr(cand, "local_window", 0)
        or 0
    )
    if window > 0:
        context = min(context, window)
    batch = 1
    layers_per_stage = cand.n_layers // max(constraints.pp, 1)
    # State-hybrid: only attention layers carry KV.
    n_attn = int(getattr(cand, "n_attention_layers", 0) or 0)
    n_state = int(getattr(cand, "n_state_layers", 0) or 0)
    if n_attn + n_state > 0:
        attn_frac = n_attn / max(1, n_attn + n_state)
        attn_layers_per_stage = layers_per_stage * attn_frac
    else:
        attn_layers_per_stage = layers_per_stage

    if getattr(cand, "attention_type", "full") == "mla" \
            and int(getattr(cand, "mla_kv_latent_dim", 0) or 0) > 0:
        c_kv = int(cand.mla_kv_latent_dim)
        d_rope = int(getattr(cand, "mla_rope_head_dim", 0) or 0)
        per_token = (c_kv + d_rope) * kv_bpe
        bytes_total = batch * context * attn_layers_per_stage * per_token
    else:
        kv_heads_per_gpu = max(1, math.ceil(cand.n_kv_heads / max(tp, 1)))
        bytes_total = (
            batch * context * attn_layers_per_stage * 2
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

    # 3b) Gate INFEASIBLE-sentinel leak before building the metric panel.
    # See _infeasibility_check docstring for the motivating bug.
    infeas_reason = _infeasibility_check(base_ev, cand_ev)
    if infeas_reason is not None:
        ev.feasible = False
        ev.reason_if_infeasible = infeas_reason
        ev.justification = (
            "Infeasible: " + infeas_reason
            + "\n\nDelta-influence numbers are NOT reported because one or "
              "both sides of the comparison hit the quality-model's "
              "INFEASIBLE marker. Fix the config and re-run."
        )
        if transition.baseline_stress is not None:
            ev.stress_baseline = transition.baseline_stress.as_dict()
            ev.binding_axes_baseline = list(transition.baseline_stress.binding_axes)
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
    # Surface sidecar attribute changes (MLA latent, SWA window, parallelism
    # overrides) so multi-delta composition is visible in the report.
    _append_sidecar_changes(
        ev.field_changes, base_arch, candidate_arch_t,
        baseline_constraints=base_constraints)
    # B1 follow-up: the canonical and sidecar paths can both emit a row for
    # the same conceptual field (e.g. `attention.type` is set by both
    # CandidateArch.attention_type and the `_mla_latent_dim` sidecar; the
    # MLA latent appears as `attention.mla_kv_latent_dim` from the
    # canonical diff and as the legacy `attention.mla_latent_dim` from
    # the sidecar diff). Collapse the alias and dedupe so the report
    # shows one row per concept.
    ev.field_changes = _dedupe_field_changes(ev.field_changes)
    # Forward delta-resolution sidecars (e.g. add_state_layers writes
    # `_state_layer_summary` so the user can see the *resolved* layer count
    # regardless of which CLI input mode they used).
    summary = getattr(candidate_arch_t, "_state_layer_summary", None)
    if summary:
        ev.delta_summary = dict(summary)

    # Forward free-form delta clamp / coercion notes (e.g. swap_attention_to_gqa
    # clamping a too-large `group_size` to MQA writes `_delta_notes` so the
    # user sees what the engine actually applied, not just what they asked
    # for). These flow into `delta_summary["clamp_notes"]` and are surfaced
    # by report.render_topology_notes.
    clamp_notes = getattr(candidate_arch_t, "_delta_notes", None)
    if clamp_notes:
        notes = list(clamp_notes)
        if notes:
            current = dict(ev.delta_summary or {})
            current.setdefault("clamp_notes", []).extend(notes)
            ev.delta_summary = current

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

    # Same INFEASIBLE-sentinel gate as the single-delta path: don't let the
    # quality-model marker leak into the metrics table as a fake ~-1e6 delta.
    infeas_reason = _infeasibility_check(base_ev, cand_ev)
    if infeas_reason is not None:
        ev = DeltaEvaluation(
            baseline_name=baseline_name,
            hardware=hardware,
            delta_name="+".join(name for name, _ in deltas),
            delta_args={"sequence": [{"name": n, "args": dict(a or {})}
                                       for n, a in deltas]},
            feasible=False,
            reason_if_infeasible=infeas_reason,
            stress_baseline=base_stress.as_dict(),
            stress_candidate=cand_stress.as_dict(),
            binding_axes_baseline=list(base_stress.binding_axes),
        )
        ev.justification = (
            "Infeasible: " + infeas_reason
            + "\n\nDelta-influence numbers are NOT reported because one or "
              "both sides of the comparison hit the quality-model's "
              "INFEASIBLE marker. Fix the config and re-run."
        )
        return ev

    composed_field_changes = _arch_changes(baseline_candidate, composed_cand)
    _append_sidecar_changes(
        composed_field_changes, base_arch, cumulative,
        baseline_constraints=base_constraints)
    # Same dedup as the single-delta path (see comment above).
    composed_field_changes = _dedupe_field_changes(composed_field_changes)
    # Forward the delta-resolution summary from add_state_layers (if any)
    # through the chained path too.
    composed_summary = getattr(cumulative, "_state_layer_summary", None)
    ev = DeltaEvaluation(
        baseline_name=baseline_name,
        hardware=hardware,
        delta_name="+".join(name for name, _ in deltas),
        delta_args={"sequence": [{"name": n, "args": dict(a or {})}
                                   for n, a in deltas]},
        feasible=True,
        field_changes=composed_field_changes,
        delta_summary=(dict(composed_summary) if composed_summary else None),
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
