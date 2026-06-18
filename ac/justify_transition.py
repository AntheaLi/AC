"""
Justification engine — turn a Transition into a human-readable explanation.

Per instruction §6.4, output should look like:
    "Selected MLA over MHA because baseline was decode-bandwidth-bound
    (HBM-BW-decode=0.94 binding). MLA reduces KV footprint by 6×, dropping
    decode-bandwidth utilization to 0.46. Quality cost: +0.008 attention
    residual, which is well below the binding threshold elsewhere."

Three sentences max: (1) baseline binding state, (2) what the transformation
relieved, (3) quality cost decomposition.
"""

from __future__ import annotations

from typing import Optional

from stress import severity_band, PRESSURED_OR_WORSE
from transition import Transition


# Pretty names for stress axes when surfaced in prose.
_AXIS_PRETTY = {
    "hbm_bw_decode": "HBM-BW-decode",
    "hbm_bw_prefill": "HBM-BW-prefill",
    "hbm_capacity": "HBM-capacity",
    "kv_footprint": "KV-footprint",
    "tc_util_prefill": "TC-util-prefill",
    "tc_util_decode": "TC-util-decode",
    "sram_tile_fit": "SRAM-tile-fit",
    "all_reduce": "all-reduce",
    "all_to_all": "all-to-all",
    "training_mem": "training-memory",
}

_QUALITY_PRETTY = {
    "shape_law_loss": "shape-law",
    "attention_residual": "attention",
    "moe_residual": "MoE",
    "state_residual": "state-residual",
    "precision_loss": "precision",
    "fa_underrun": "FA-underrun",
    "context_extrapolation": "context-extrapolation",
}

# Pretty names for transformations.
_TRANSFORM_PRETTY = {
    "swap_attention_to_gqa": "GQA",
    "swap_attention_to_mla": "MLA",
    "swap_attention_to_swa": "SWA",
    "add_state_layers": "state layers",
    "densify_first_k": "first-K dense FFN",
    "change_moe_topology": "MoE topology change",
    "scale_d_model": "d_model rescale",
    "scale_n_layers": "n_layers rescale",
    "change_precision_per_component": "per-component precision change",
    "change_parallelism": "parallelism change",
}


def _pretty_axis(axis: str) -> str:
    return _AXIS_PRETTY.get(axis, axis)


def _pretty_quality(axis: str) -> str:
    return _QUALITY_PRETTY.get(axis, axis)


def _pretty_transform(name: str) -> str:
    return _TRANSFORM_PRETTY.get(name, name)


def _format_binding_state(transition: Transition) -> str:
    """Sentence 1 — why we're considering this transition."""
    b = transition.baseline_stress
    if b is None or not b.binding_axes:
        return "Baseline has no binding stresses; transformation is exploratory."
    parts = []
    for axis in b.binding_axes:
        v = getattr(b, axis)
        parts.append(f"{_pretty_axis(axis)}={v:.2f} {severity_band(v)}")
    return ("Baseline binding stresses: " + ", ".join(parts) + ".")


def _format_relief(transition: Transition) -> str:
    """Sentence 2 — what the transformation actually did to stress."""
    if transition.candidate_stress is None:
        return ""
    pretty_name = _pretty_transform(transition.transformation_name)
    if not transition.relieved_binding_axes and not transition.new_binding_axes:
        # Find the single largest absolute delta to mention.
        if transition.delta_stress:
            axis = max(transition.delta_stress.keys(),
                       key=lambda a: abs(transition.delta_stress[a]))
            d = transition.delta_stress[axis]
            direction = "drops" if d < 0 else "rises"
            return (f"Applying {pretty_name} {direction} {_pretty_axis(axis)} "
                    f"by {abs(d):.2f}; no binding axis was relieved.")
        return f"Applying {pretty_name} produces no measurable stress change."
    relieved_clause = ""
    if transition.relieved_binding_axes:
        moves = []
        for axis in transition.relieved_binding_axes:
            bv = getattr(transition.baseline_stress, axis)
            cv = getattr(transition.candidate_stress, axis)
            moves.append(f"{_pretty_axis(axis)} {bv:.2f}→{cv:.2f}")
        relieved_clause = "relieves " + "; ".join(moves)
    new_clause = ""
    if transition.new_binding_axes:
        moves = []
        for axis in transition.new_binding_axes:
            cv = getattr(transition.candidate_stress, axis)
            moves.append(f"{_pretty_axis(axis)}={cv:.2f}")
        new_clause = ", introduces new pressure on " + ", ".join(moves)
    joiner = " " if not relieved_clause else " "
    return f"Applying {pretty_name} {relieved_clause}{new_clause}.".replace("  ", " ")


def _format_quality_cost(transition: Transition) -> str:
    """Sentence 3 — what it cost in quality."""
    if not transition.delta_quality:
        return ""
    # Show the two axes with largest absolute delta.
    sorted_axes = sorted(transition.delta_quality.items(),
                         key=lambda kv: abs(kv[1]), reverse=True)
    nonzero = [(a, d) for a, d in sorted_axes if abs(d) > 1e-6]
    if not nonzero:
        return "Quality cost: negligible across all axes."
    top = nonzero[:2]
    parts = []
    for axis, d in top:
        sign = "+" if d >= 0 else ""
        parts.append(f"{sign}{d:.4f} {_pretty_quality(axis)}")
    total = transition.delta_quality_total
    sign = "+" if total >= 0 else ""
    return (f"Quality cost: {', '.join(parts)} ({sign}{total:.4f} total "
            f"residual change).")


def justify(transition: Transition) -> str:
    """Render a 1-3 sentence justification for a Transition."""
    if not transition.feasible:
        return (f"Transformation {_pretty_transform(transition.transformation_name)} "
                f"infeasible: {transition.reason_if_infeasible}")
    sentences = [
        _format_binding_state(transition),
        _format_relief(transition),
        _format_quality_cost(transition),
    ]
    return " ".join(s for s in sentences if s)


def justify_batch(transitions, top_n: int = 3) -> str:
    """Format a multi-line justification for a ranked list."""
    lines = []
    for i, t in enumerate(transitions[:top_n], start=1):
        lines.append(f"{i}. {justify(t)}")
    return "\n".join(lines)
