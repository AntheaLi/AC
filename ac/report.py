"""
Report — render a DeltaEvaluation as Markdown / JSON / Pareto CSV.

The Markdown layout intentionally mirrors the `v1-ac-solver/baseline_delta.md`
"## Stress-Conditioned Relief" section so the two reports compose well.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, List

from evaluator import DeltaEvaluation, MetricDelta
from stress import STRESS_AXES


# =============================================================================
# Pretty names
# =============================================================================

_METRIC_PRETTY = {
    "predicted_loss":    "Predicted loss",
    "serving_tbt_ms":    "Serving TBT (ms)",
    "prefill_time_ms":   "Prefill / TTFT (ms)",
    "training_tps":      "Training TPS (tok/s)",
    "memory_per_gpu_gb": "Memory / GPU (GB)",
    "kv_cache_gb":       "KV cache (GB)",
    "total_params_b":    "Total params (B)",
}

_STRESS_AXIS_PRETTY = {
    "hbm_bw_decode":   "HBM bandwidth (decode)",
    "hbm_bw_prefill":  "HBM bandwidth (prefill)",
    "hbm_capacity":    "HBM capacity",
    "kv_footprint":    "KV footprint",
    "tc_util_prefill": "Tensor-core util (prefill)",
    "tc_util_decode":  "Tensor-core util (decode)",
    "sram_tile_fit":   "SRAM tile fit",
    "all_reduce":      "All-reduce traffic",
    "all_to_all":      "All-to-all traffic",
    "training_mem":    "Training memory",
}

_POSITION_PRETTY = {
    "DOMINATES_BASELINE":   "Strictly dominates baseline",
    "DOMINATED_BY_BASELINE": "Dominated by baseline",
    "EXPANDS_FRONTIER":      "Expands the Pareto frontier",
    "ON_FRONTIER":           "On the Pareto frontier",
    "INTERIOR":              "Interior trade-off (Pareto-dominated)",
    "EQUIVALENT":            "Equivalent to baseline",
    "UNKNOWN":               "Unknown (classification failed)",
    "":                      "Not evaluated",
}


# =============================================================================
# Markdown
# =============================================================================

def _fmt_signed(v: float, suffix: str = "") -> str:
    if v >= 0:
        return f"+{v:.3f}{suffix}"
    return f"{v:.3f}{suffix}"


def _fmt_pct(v: float) -> str:
    return f"{v:+.2f}%"


def render_metric_table(metrics: Dict[str, MetricDelta],
                        keys: List[str] = None) -> str:
    """Render the canonical metric panel as a Markdown table."""
    keys = keys or [
        "predicted_loss",
        "serving_tbt_ms",
        "prefill_time_ms",
        "training_tps",
        "memory_per_gpu_gb",
        "kv_cache_gb",
        "total_params_b",
    ]
    lines = ["| Metric | Baseline | Candidate | Δ | Δ% | Direction |",
             "|---|---:|---:|---:|---:|---|"]
    for k in keys:
        md = metrics.get(k)
        if md is None:
            continue
        pretty = _METRIC_PRETTY.get(k, k)
        lines.append(
            f"| {pretty} | {md.baseline:.3f} | {md.candidate:.3f} | "
            f"{_fmt_signed(md.delta)} | {_fmt_pct(md.pct_change)} | "
            f"{md.direction} |"
        )
    return "\n".join(lines)


def render_quality_table(metrics: Dict[str, MetricDelta]) -> str:
    """Quality-term decomposition (filters to metrics starting with quality_)."""
    qk = sorted(k for k in metrics if k.startswith("quality_"))
    if not qk:
        return ""
    lines = ["| Quality term | Baseline | Candidate | Δ | Δ% |",
             "|---|---:|---:|---:|---:|"]
    for k in qk:
        md = metrics[k]
        if abs(md.baseline) < 1e-9 and abs(md.candidate) < 1e-9:
            continue
        pretty = k.replace("quality_", "")
        lines.append(
            f"| {pretty} | {md.baseline:.5f} | {md.candidate:.5f} | "
            f"{_fmt_signed(md.delta)} | {_fmt_pct(md.pct_change)} |"
        )
    if len(lines) == 2:
        return ""
    return "\n".join(lines)


def render_stress_section(ev: DeltaEvaluation) -> str:
    """Stress-vector card with binding-axis bookkeeping."""
    if not ev.stress_baseline or not ev.stress_candidate:
        return "_Stress vector not available (delta engine returned no vectors)._"
    rows = ["| Stress axis | Baseline | Candidate | Δ | Baseline band | Candidate band |",
            "|---|---:|---:|---:|---|---|"]
    base = ev.stress_baseline
    cand = ev.stress_candidate
    base_bands = base.get("bands", {})
    cand_bands = cand.get("bands", {})
    for axis in STRESS_AXES:
        bv = float(base.get(axis, 0.0))
        cv = float(cand.get(axis, 0.0))
        if abs(bv) < 1e-9 and abs(cv) < 1e-9:
            continue
        delta = cv - bv
        pretty = _STRESS_AXIS_PRETTY.get(axis, axis)
        marker_b = base_bands.get(axis, "")
        marker_c = cand_bands.get(axis, "")
        rows.append(
            f"| {pretty} | {bv:.3f} | {cv:.3f} | {_fmt_signed(delta)} | "
            f"{marker_b} | {marker_c} |"
        )
    body = "\n".join(rows)
    if ev.binding_axes_baseline:
        body += "\n\n**Baseline binding axes:** " + ", ".join(
            _STRESS_AXIS_PRETTY.get(a, a) for a in ev.binding_axes_baseline)
    if ev.binding_axes_relieved:
        body += "\n\n**Relieved by delta:** " + ", ".join(
            _STRESS_AXIS_PRETTY.get(a, a) for a in ev.binding_axes_relieved)
    if ev.binding_axes_introduced:
        body += "\n\n**Newly pressured:** " + ", ".join(
            _STRESS_AXIS_PRETTY.get(a, a) for a in ev.binding_axes_introduced)
    body += f"\n\n**Stress relief score:** {ev.stress_relief_score:+.3f}"
    if ev.severe_stress_regression:
        body += "\n\n> Warning: severe stress regression; at least one axis jumped from <pressured to violated."
    return body


def render_pareto_section(ev: DeltaEvaluation) -> str:
    pretty_pos = _POSITION_PRETTY.get(ev.pareto_position, ev.pareto_position)
    sign = "+" if ev.pareto_distance >= 0 else ""
    if not ev.pareto_position:
        return "_Pareto position not evaluated for this run._"
    lines = [
        f"**Verdict:** {pretty_pos}",
        f"**Signed distance to frontier (normalized):** {sign}{ev.pareto_distance:.4f}",
        f"**Frontier size:** {ev.pareto_frontier_size} candidates",
        f"**Frontier points dominated by this delta:** {ev.pareto_dominates_count}",
        f"**Frontier points that dominate this delta:** {ev.pareto_dominated_count}",
    ]
    if ev.pareto_axes:
        lines.append("**Axes that moved:** " + ", ".join(
            _METRIC_PRETTY.get(a, a) for a in ev.pareto_axes))
    return "\n".join(lines)


def render_field_changes(ev: DeltaEvaluation) -> str:
    if not ev.field_changes:
        return "_No structural changes (sidecar-only delta)._"
    lines = ["| Field | Baseline | Candidate |", "|---|---|---|"]
    for ch in ev.field_changes:
        lines.append(
            f"| `{ch['field']}` | `{ch['baseline']}` | `{ch['candidate']}` |"
        )
    return "\n".join(lines)


def render_topology_notes(ev: DeltaEvaluation) -> str:
    """Explain topology cases where a structural delta has no local metric move."""
    notes: List[str] = []
    kv_change = next(
        (ch for ch in ev.field_changes if ch.get("field") == "n_kv_heads"),
        None,
    )
    kv_metric = ev.metrics.get("kv_cache_gb")
    tbt_metric = ev.metrics.get("serving_tbt_ms")
    if kv_change and kv_metric and abs(kv_metric.delta) < 1e-9:
        notes.append(
            "KV heads changed, but modeled per-GPU KV cache stayed flat because "
            "the current TP/KV placement assumes at least one KV head resident "
            "per rank. This is expected when TP is greater than or equal to the "
            "candidate KV-head count; use a different KV-sharding policy or lower "
            "TP to realize per-rank KV-cache savings."
        )
        if tbt_metric and abs(tbt_metric.delta) < 1e-9:
            notes.append(
                "Decode TBT is neutral for the same reason: local decode "
                "bandwidth still reads one KV head per rank."
            )
    if not notes:
        return ""
    return "\n".join(f"- {note}" for note in notes)


def render_markdown(ev: DeltaEvaluation) -> str:
    """Render a single-page Markdown report for one DeltaEvaluation."""
    args_str = ", ".join(f"{k}={v}" for k, v in ev.delta_args.items()) or "(no args)"
    head = [
        f"# Delta Influence — `{ev.delta_name}`",
        "",
        f"**Baseline:** {ev.baseline_name}  ",
        f"**Hardware:** {ev.hardware}  ",
        f"**Delta args:** {args_str}",
        "",
    ]
    if not ev.feasible:
        return "\n".join(head + [
            f"## Infeasible",
            "",
            f"`{ev.reason_if_infeasible}`",
            "",
            ev.justification or "",
        ])

    sections = [
        "## Summary",
        "",
        ev.justification or "_No narrative available._",
        "",
        "## Field-level changes",
        "",
        render_field_changes(ev),
        "",
        "## Evaluation metrics",
        "",
        render_metric_table(ev.metrics),
        "",
        "## Topology notes",
        "",
        render_topology_notes(ev) or "_No topology caveats for this delta._",
        "",
        "## Quality residual decomposition",
        "",
        render_quality_table(ev.metrics) or "_No quality-term deltas were reported._",
        "",
        "## Stress influence",
        "",
        render_stress_section(ev),
        "",
        "## Pareto position",
        "",
        render_pareto_section(ev),
        "",
    ]
    return "\n".join(head + sections)


# =============================================================================
# JSON
# =============================================================================

def render_json(ev: DeltaEvaluation, *, indent: int = 2) -> str:
    """Stable JSON dump of the full DeltaEvaluation."""
    return json.dumps(ev.as_dict(), indent=indent, sort_keys=True, default=str)


# =============================================================================
# Three-row Pareto CSV
# =============================================================================

def render_pareto_csv(ev: DeltaEvaluation) -> str:
    """Compact three-row CSV: baseline, candidate, signed delta.

    Schema mirrors `v1-ac-solver/modifier.modifier_pareto_to_csv` for the
    columns we emit so downstream consumers can read this with the same
    parser.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    columns = [
        "row", "kind",
        "predicted_loss", "serving_tbt_ms", "prefill_time_ms",
        "training_tps", "memory_per_gpu_gb", "kv_cache_gb",
        "total_params_b",
        # stress axes (5 representative ones — same selection as
        # modifier_pareto_to_csv)
        "stress_hbm_bw_decode", "stress_kv_footprint",
        "stress_hbm_capacity", "stress_training_mem", "stress_all_reduce",
        "binding_axes",
    ]
    writer.writerow(columns)

    def _metric_pair(key: str):
        md = ev.metrics.get(key)
        if md is None:
            return (0.0, 0.0)
        return (md.baseline, md.candidate)

    def _stress_axis(side: Dict[str, Any], axis: str) -> float:
        if not side:
            return 0.0
        return float(side.get(axis, 0.0))

    base_row = ["1", "baseline"]
    cand_row = ["2", "candidate"]
    delta_row = ["3", "delta"]

    for key in ("predicted_loss", "serving_tbt_ms", "prefill_time_ms",
                "training_tps", "memory_per_gpu_gb", "kv_cache_gb",
                "total_params_b"):
        b, c = _metric_pair(key)
        base_row.append(round(b, 4))
        cand_row.append(round(c, 4))
        delta_row.append(round(c - b, 4))

    for axis in ("hbm_bw_decode", "kv_footprint", "hbm_capacity",
                 "training_mem", "all_reduce"):
        bv = _stress_axis(ev.stress_baseline or {}, axis)
        cv = _stress_axis(ev.stress_candidate or {}, axis)
        base_row.append(round(bv, 4))
        cand_row.append(round(cv, 4))
        delta_row.append(round(cv - bv, 4))

    base_row.append("|".join(ev.binding_axes_baseline))
    cand_row.append("|".join(ev.stress_candidate.get("binding_axes", [])
                              if ev.stress_candidate else []))
    delta_row.append("|".join(ev.binding_axes_relieved
                                + [f"+{a}" for a in ev.binding_axes_introduced]))

    writer.writerow(base_row)
    writer.writerow(cand_row)
    writer.writerow(delta_row)
    return output.getvalue()


# =============================================================================
# Batch (sequence of evaluations) renderer
# =============================================================================

def render_markdown_multi(evs: List[DeltaEvaluation]) -> str:
    """Render multiple DeltaEvaluations as one Markdown document with TOC."""
    parts = ["# Delta Influence — Batch Report", ""]
    if not evs:
        parts.append("_No deltas were evaluated._")
        return "\n".join(parts)
    parts.append("## Contents")
    parts.append("")
    for i, ev in enumerate(evs, 1):
        feasible_marker = "" if ev.feasible else " — *infeasible*"
        parts.append(f"{i}. `{ev.delta_name}` — "
                     f"{_POSITION_PRETTY.get(ev.pareto_position, ev.pareto_position)}{feasible_marker}")
    parts.append("")
    for ev in evs:
        parts.append("---")
        parts.append("")
        parts.append(render_markdown(ev))
        parts.append("")
    return "\n".join(parts)
