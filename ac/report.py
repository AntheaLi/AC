"""
Report — render a DeltaEvaluation as Markdown / JSON / Pareto CSV.

The Markdown layout intentionally mirrors the `v1-ac-solver/baseline_delta.md`
"## Stress-Conditioned Relief" section so the two reports compose well.
"""

from __future__ import annotations

import csv
import io
import json
import math
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
    # Fix #4: handle the inf sentinel produced when baseline is ~0 and
    # delta is non-zero. "+inf%" is uglier than "(no baseline)".
    try:
        if not math.isfinite(v):
            return "(no baseline)"
    except (TypeError, ValueError):
        pass
    return f"{v:+.2f}%"


def _fmt_pct_md(md) -> str:
    """Render a MetricDelta's percent change in the most readable form.

    - For very small baselines (sub-1e-6, or where the move dwarfs the
      baseline by 20×), `+12345%` is mathematically defined but operationally
      meaningless. Render as a multiplicative ratio (`5.4× larger`,
      `0.02× smaller`) when both signs are positive so the reader can see the
      scale at a glance.
    - For ordinary cases, fall through to the usual ±%.
    - Only return n/a as a last resort (e.g. baseline is exactly zero with
      same-sign candidate that can't be rendered as a ratio either).
    """
    try:
        base = float(md.baseline)
        cand = float(md.candidate)
    except Exception:
        return _fmt_pct(getattr(md, "pct_change", 0.0))
    delta = abs(cand - base)
    # Quality residuals sometimes sit at ~1e-7. Surface the *signed move*
    # (e.g. "(no baseline, +0.021)") rather than a bare "n/a" so the reader
    # sees direction and magnitude even when a ratio is undefined.
    near_zero_baseline = abs(base) < 1e-6
    blown_up = delta > 0 and abs(base) > 0 and abs(base) < 0.05 * delta
    if near_zero_baseline:
        if delta < 1e-9:
            return "n/a"
        signed = cand - base
        return f"(no baseline, {signed:+.3g})"
    if blown_up:
        # Render as a clean ratio when both numbers have the same sign so it
        # parses as "candidate is N× the baseline". Otherwise fall back to
        # n/a.
        if base != 0 and (base > 0) == (cand > 0):
            ratio = cand / base
            if ratio >= 1.0:
                return f"{ratio:.2g}× larger"
            return f"{ratio:.2g}× of baseline"
        return "n/a"
    return _fmt_pct(md.pct_change)


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
            f"{_fmt_signed(md.delta)} | {_fmt_pct_md(md)} | "
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
            f"{_fmt_signed(md.delta)} | {_fmt_pct_md(md)} |"
        )
    if len(lines) == 2:
        return ""
    return "\n".join(lines)


def _stress_band_label(vector: Dict[str, Any], axis: str) -> str:
    band = (vector.get("bands", {}) or {}).get(axis, "")
    active_axes = set(vector.get("active_axes") or STRESS_AXES)
    if axis not in active_axes and band in {"pressured", "binding", "violated"}:
        phase = vector.get("phase", "this phase")
        return f"{band} (inactive for {phase})"
    return band


def render_stress_section(ev: DeltaEvaluation) -> str:
    """Stress-vector card with binding-axis bookkeeping."""
    if not ev.stress_baseline or not ev.stress_candidate:
        return "_Stress vector not available (delta engine returned no vectors)._"
    rows = ["| Stress axis | Baseline | Candidate | Δ | Baseline band | Candidate band |",
            "|---|---:|---:|---:|---|---|"]
    base = ev.stress_baseline
    cand = ev.stress_candidate
    for axis in STRESS_AXES:
        bv = float(base.get(axis, 0.0))
        cv = float(cand.get(axis, 0.0))
        if abs(bv) < 1e-9 and abs(cv) < 1e-9:
            continue
        delta = cv - bv
        pretty = _STRESS_AXIS_PRETTY.get(axis, axis)
        marker_b = _stress_band_label(base, axis)
        marker_c = _stress_band_label(cand, axis)
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
    if not ev.pareto_position:
        return "_Pareto position not evaluated for this run._"
    # Degenerate case: the position classifier produced an empty frontier and
    # zero dominated/dominating counts. Reporting "On the Pareto frontier"
    # here is vacuously true but misleading — there is nothing to compare
    # against. Surface the situation honestly instead.
    frontier_empty = (
        int(ev.pareto_frontier_size) == 0
        and int(ev.pareto_dominates_count) == 0
        and int(ev.pareto_dominated_count) == 0
    )
    if frontier_empty:
        lines = [
            "**Verdict:** _Not classified — the local Pareto frontier was empty._",
            "",
            "The classifier had no other candidates to compare against, so no "
            "dominance verdict could be reached. This usually means the local "
            "neighborhood around the baseline produced no feasible alternatives "
            "under the current TP/PP/CP and constraint settings; try "
            "`--no-pareto` to skip the classification, or widen the modifier "
            "sweep upstream to populate the comparison frontier.",
        ]
        if ev.pareto_axes:
            lines.append("")
            lines.append("**Axes that moved:** " + ", ".join(
                _METRIC_PRETTY.get(a, a) for a in ev.pareto_axes))
        return "\n".join(lines)

    pretty_pos = _POSITION_PRETTY.get(ev.pareto_position, ev.pareto_position)
    sign = "+" if ev.pareto_distance >= 0 else ""
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

    # 0) Delta resolution echo. When a delta has multiple input modes (e.g.
    # add_state_layers accepts ratio / state_fraction / state_layers), echo
    # the *resolved* interpretation so the user can verify the layer count
    # without parsing the CLI convention themselves. This is the cure for
    # the "ratio=1:7 actually means 12.5% state, not 87.5%" footgun.
    summary = getattr(ev, "delta_summary", None) or {}

    # Surface free-form clamp / coercion notes (e.g. GQA group_size clamped
    # to n_heads, n_heads not divisible by group_size). These reflect what
    # the delta engine actually applied, not what the user asked for, so we
    # show them first.
    clamp_notes = summary.get("clamp_notes") or []
    for note in clamp_notes:
        notes.append(str(note))

    if summary and "state_fraction" in summary:
        sf = float(summary.get("state_fraction", 0.0))
        n_state = int(summary.get("state_layers", 0))
        n_attn = int(summary.get("attention_layers", 0))
        n_total = n_state + n_attn
        requested_via = summary.get("requested_via", "")
        notes.append(
            f"Resolved: {n_state}/{n_total} layers are state mixers "
            f"({sf*100:.1f}% state, {(1-sf)*100:.1f}% attention). "
            f"Requested via `{requested_via}`. AC's quality model is "
            "calibrated to Jamba/Zamba-style hybrids with 72-88% state; "
            "use `state_fraction=...` if the CLI's `state:attention` ratio "
            "convention is unfamiliar."
        )

    # Filter to *substantive* field changes: ignore sidecar parallelism
    # echoes (tp/pp/cp moved from <int> to None) and the always-present
    # `applied_deltas` provenance row, since they're book-keeping and
    # never reflect a real architectural edit.
    SIDECAR_ECHO_FIELDS = {
        "parallelism.tensor_parallel",
        "parallelism.pipeline_parallel",
        "parallelism.context_parallel",
        "parallelism.expert_parallel",
    }
    substantive_changes = [
        ch for ch in ev.field_changes
        if ch.get("field") not in SIDECAR_ECHO_FIELDS
        and ch.get("field") != "applied_deltas"
    ]

    # 1) No-op delta callout. When the delta produced a candidate whose
    # arch fields are identical to the baseline (e.g. `swap_attention_to_gqa
    # group_size=4` on a baseline that is already GQA(32/8)), every metric
    # delta will read ~0 — the user can't tell whether the model is wrong
    # or whether the requested edit was already in place. Surface this
    # explicitly so the user knows to revisit the arguments.
    #
    # B1 fix: in addition to "no substantive field-level changes", require
    # that the evaluated metrics are also flat before claiming a no-op.
    # Previously the callout fired purely on the structural-diff side, so
    # for any delta whose effect lives inside layer_configs / sub-fields
    # (densify_first_k, change_moe_topology, add_state_layers, …) the
    # report could show real metric movement above the table while the
    # note below claimed "every metric reads ~0". That contradiction is
    # the single most user-confusing defect in the v0.3 report; the
    # combined gate fixes it without changing behaviour for the genuine
    # no-op case (e.g. group_size=4 on a GQA(32/8) baseline), where the
    # metrics really are flat.
    METRICS_TO_GATE = (
        "predicted_loss",
        "serving_tbt_ms",
        "prefill_time_ms",
        "training_tps",
        "memory_per_gpu_gb",
        "kv_cache_gb",
        "total_params_b",
    )
    def _is_flat(metric_name: str) -> bool:
        m = ev.metrics.get(metric_name)
        if m is None:
            return True
        # Relative threshold: 0.05% of baseline magnitude, with a small
        # absolute floor so tiny absolute baselines (KV cache in
        # gigabytes for short-context configs) don't trip the gate.
        abs_floor = 1e-6
        rel_floor = 5e-4 * max(abs(m.baseline), abs(m.candidate))
        return abs(m.delta) <= max(abs_floor, rel_floor)
    metrics_are_flat = all(_is_flat(name) for name in METRICS_TO_GATE)

    if not substantive_changes and metrics_are_flat:
        args_hint = ""
        if ev.delta_args:
            args_hint = " (args: " + ", ".join(
                f"{k}={v}" for k, v in ev.delta_args.items()
            ) + ")"
        notes.append(
            f"Delta `{ev.delta_name}`{args_hint} produced a candidate that is "
            "structurally identical to the baseline — every metric reads ~0 "
            "because the requested edit was already in place. Try different "
            "args (e.g. a smaller group_size for GQA, a smaller window for "
            "SWA, or a different latent_dim for MLA)."
        )
    elif not substantive_changes and not metrics_are_flat:
        # The structural diff didn't surface the change but the metrics
        # moved. With the B1 fix to `_arch_changes` this should be very
        # rare; emit a softer note so the user knows the delta did
        # something even if the per-field rendering is empty.
        notes.append(
            f"Delta `{ev.delta_name}` moved evaluated metrics but the "
            "field-level diff above is empty — the change lives inside a "
            "sub-field the report does not surface. Inspect the raw "
            "evaluation JSON if you need the exact shape change."
        )

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

    # Fix #1: MLA per-GPU KV "growth" caveat. The MLA branch in the
    # throughput model stores a single shared (c_kv + d_rope) latent per
    # token that intentionally is NOT sharded across TP ranks — that's the
    # whole point of MLA's KV compression, and it's how DeepSeek-V2/V3
    # ships it. A consequence is that when the baseline already has a
    # small, well-sharded GQA KV (e.g. Mistral 7B at TP=4: 8 KV heads ÷
    # TP=4 = 2 heads × 128 dh × 2 B/elem = 1024 B/tok/layer), the MLA
    # latent (512 + 64) × 2 B = 1152 B/tok/layer can be LARGER than the
    # baseline on a *per-GPU* basis, even though aggregate / unsharded
    # KV shrinks. Without this note, a reader sees "MLA worsens KV by
    # +12%" and concludes the model is broken; with this note they see
    # that they are comparing a sharded baseline to an unsharded MLA
    # latent and that the headline MLA savings show up at larger batches,
    # higher TP, or longer context where the sharded baseline saturates.
    mla_swap = next(
        (ch for ch in ev.field_changes
         if ch.get("field") in ("attention.type", "attention_type")
         and str(ch.get("candidate", "")).lower() == "mla"),
        None,
    )
    if mla_swap and kv_metric is not None:
        notes.append(
            "MLA stores ONE shared compressed latent (`c_kv + d_rope`) per "
            "token, which is *not* sharded across TP ranks (this is the "
            "intended behaviour, matching DeepSeek-V2/V3). On a baseline "
            "whose GQA KV is already small after TP sharding, the per-GPU "
            "KV figure can stay flat or grow under MLA even as the aggregate "
            "(un-sharded) KV shrinks. The headline MLA savings appear at "
            "larger batch sizes, higher TP, or long context — where the "
            "sharded GQA baseline saturates against the one-KV-head-per-rank "
            "floor. Compare the candidate against an *aggregate* KV figure "
            "(per replica, not per GPU) to see the true MLA benefit."
        )

    # 2) SWA prefill caveat. Sliding-window attention reduces prefill
    # compute from O(N²) to O(N·W), but our throughput model bills full
    # N² attention compute for SWA at prefill so the FFN-dominated short-
    # prompt cost stays correct. Flag this so users know prefill TTFT is
    # a conservative upper bound for SWA, not an optimistic estimate.
    swa_change = next(
        (ch for ch in ev.field_changes if ch.get("field") == "attention.sliding_window"),
        None,
    )
    prefill_metric = ev.metrics.get("prefill_time_ms")
    if (swa_change and prefill_metric and
            abs(prefill_metric.delta) < 1e-3 and prefill_metric.baseline > 0):
        notes.append(
            "SWA prefill TTFT is reported as a *conservative upper bound*: "
            "the throughput model bills full O(N²) attention compute at "
            "prefill so the FFN-dominated short-prompt cost stays correct. "
            "Real FlashAttention-2 sliding-window kernels achieve O(N·W); "
            "expect actual TTFT to be lower than reported when context >> window."
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
    # Fix #3: surface the resolved workload at the top of the report so a
    # reader can tell which preset was used and how it maps to the
    # greenfield / modifier numbers (which use their own defaults).
    rw = ev.resolved_workload or {}
    if rw:
        head.extend([
            "**Resolved workload** (all predictions below use these settings):",
            "",
            f"- preset: `{rw.get('workload_preset', '?')}`  ",
            f"- serving_batch: {rw.get('serving_batch')}  ",
            f"- prompt_len: {rw.get('prompt_len')}  context_length: {rw.get('context_length')}  ",
            f"- TP / PP / DP: {rw.get('tp')} / {rw.get('pp')} / {rw.get('dp')}  ",
            f"- TBT budget: {rw.get('serving_tbt_ms_budget')} ms",
            "",
        ])
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
