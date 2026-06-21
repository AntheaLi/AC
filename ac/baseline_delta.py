"""
Markdown reports for baseline-aware Pareto modification.
"""

from typing import Iterable, List

from modifier import ModifierRecord, ModifierResult, modifier_score, quality_risk_pct


def _modifier_scope_warning(result: ModifierResult) -> List[str]:
    """Fix #5: announce up front which kinds of moves the modifier *cannot*
    explore for this baseline. Earlier versions buried this in caveats at the
    bottom, so users with MoE / MLA / state baselines saw "1 evaluated, 1
    feasible, 1 Pareto = baseline" without understanding why.

    Reports:
      - moe         : baseline is MoE; only dense-side local moves can fire.
      - mla         : baseline uses MLA; head/precision sweeps stay constant.
      - hybrid_state: baseline has state layers; placement/ratio sweeps are
                      reserved hooks.
    """
    bm = result.baseline_model
    cand = getattr(bm, "candidate", None) if bm else None
    if cand is None:
        return []

    # CandidateArch uses `moe` (not `moe_config`) and surfaces MLA via
    # `mla_kv_latent_dim > 0` rather than an `attention_type` string.
    is_moe = (getattr(cand, "moe", None) is not None
              or getattr(cand, "moe_config", None) is not None
              or str(getattr(cand, "moe_style", "")).lower() in {"fine", "shared", "first_k_dense"})
    attn_type = str(getattr(cand, "attention_type", "")).lower()
    mla_latent = int(getattr(cand, "mla_kv_latent_dim", 0) or 0)
    is_mla = attn_type == "mla" or mla_latent > 0
    layer_type_list = getattr(cand, "layer_type_list", None) or []
    state_cfg = getattr(cand, "state_config", None)
    is_hybrid = bool(state_cfg) or any(str(t).lower() == "state" for t in layer_type_list)

    if not (is_moe or is_mla or is_hybrid):
        return []

    flags = []
    if is_moe:
        flags.append(
            "**MoE baseline detected.** The local modifier currently sweeps "
            "dense-side moves (width / depth / FFN precision / KV bits). It "
            "does NOT sweep `n_experts`, `top_k`, expert-parallel degree, or "
            "first-K-dense prefix. Use `ac-delta-eval --apply change_moe_topology` "
            "or `--apply densify_first_k` to evaluate those."
        )
    if is_mla:
        flags.append(
            "**MLA baseline detected.** Head-count and KV-bit sweeps that "
            "would re-allocate the attention shape don't fire on MLA blocks. "
            "Use `ac-delta-eval --apply swap_attention_to_mla` with a "
            "different `latent_dim` to compare."
        )
    if is_hybrid:
        flags.append(
            "**State / hybrid baseline detected.** State ratio, placement "
            "strategy, and state-block precision are not enumerated by the "
            "local modifier. Use `ac-delta-eval --apply add_state_layers` for "
            "ratio sweeps."
        )

    lines = [
        "## Modifier Scope",
        "",
        "Before reading the results below, note which moves this modifier did "
        "**not** enumerate for your baseline:",
        "",
    ]
    for flag in flags:
        lines.append(f"- {flag}")
    lines.append("")
    return lines


def generate_baseline_delta_report(result: ModifierResult, top_n: int = 8) -> str:
    """Generate baseline_delta.md for modifier mode."""

    baseline = result.baseline
    selected = result.selected
    lines: List[str] = []
    lines.append("# Baseline Delta Report\n")
    lines.append(f"**Baseline**: {result.baseline_model.name if result.baseline_model else 'baseline'}")
    lines.append(f"**Hardware**: {result.hardware.upper()}")
    if result.allow_quality_spending:
        lines.append(f"**Selection mode**: quality-spending allowed up to +{result.quality_risk_budget_pct:.2f}% relative loss proxy")
    else:
        lines.append("**Selection mode**: same-quality / model-preserving moves only")
    lines.append("")

    # Fix #5: surface the modifier's limitations at the *top* of the report
    # before the user starts reading per-candidate tables. The earlier
    # placement of this caveat (at the bottom of the report) caused users with
    # MoE/MLA/state baselines to mistake "1 candidate evaluated" for a real
    # search result.
    lines.extend(_modifier_scope_warning(result))

    lines.append("## Baseline Status\n")
    lines.extend(_metrics_block("Baseline", baseline, baseline))
    lines.append("")
    if result.baseline_risk_aware_pareto_optimal:
        lines.append("- Risk-aware Pareto status: baseline remains on the frontier because lower-risk/no-change is itself an objective.")
    else:
        lines.append("- Risk-aware Pareto status: baseline is dominated by at least one local modification.")
    if result.baseline_performance_dominated:
        lines.append("- Performance-only status: baseline is dominated by local variants when risk is excluded from the objective axes.")
    else:
        lines.append("- Performance-only status: no local variant strictly dominates the baseline across quality, latency, throughput, memory, and KV footprint.")
    if baseline.evaluated.constraint_violations:
        lines.append("- Baseline constraint violations: " + "; ".join(baseline.evaluated.constraint_violations))
    lines.append("")

    lines.append("## Selected Modification\n")
    if selected.is_baseline:
        if result.allow_quality_spending:
            lines.append("No local modification beat the baseline inside the configured quality-risk budget. The baseline config is retained.")
        else:
            lines.append("No modeled same-quality deployment modification beat the baseline. The baseline architecture is retained.")
    else:
        lines.append(f"Selected change: **{selected.change_summary}**")
        lines.append(f"Move class: **{selected.move_class}**")
        lines.append(f"Quality-preserving: **{selected.quality_preserving}**")
        lines.append(f"Risk: **{selected.risk_label}** (score {selected.risk_score})")
        lines.extend(_metrics_block("Selected", selected, baseline))
    lines.append("")

    lines.append("## Same-Quality Hardware-Fit Modifications\n")
    top_same_quality = [
        r for r in result.pareto_frontier
        if not r.is_baseline and r.quality_preserving
    ][:top_n]
    if top_same_quality:
        lines.extend(_candidate_table(top_same_quality, baseline))
    else:
        lines.append("No non-baseline same-quality modeled deployment variant is on the current Pareto frontier.")
        lines.append("")
        lines.append("Reserved same-quality hardware-fit hooks for the next modifier layer:")
        lines.append("- Tensor/pipeline/data parallel placement search without changing weights or layer shapes.")
        lines.append("- GQA-aware head sharding and KV-group placement across TP ranks.")
        lines.append("- Paged KV cache block size, allocator locality, and scheduler residency policy.")
        lines.append("- Static shape buckets / CUDA graph capture for decode and prefill.")
        lines.append("- Fused BF16 kernels, tensor-core weight swizzles, and sequence-parallel activation layout.")
        lines.append("- Chunked prefill scheduling for long-context prompts.")
    lines.append("")

    lines.append("## Optional Quality-Spending Modifications\n")
    top = [
        r for r in result.pareto_frontier
        if not r.is_baseline and not r.quality_preserving
    ][:top_n]
    if top:
        lines.append("These candidates change architecture or numerics and should be treated as retraining/calibration options, not same-quality hardware-fit edits.\n")
        lines.extend(_candidate_table(top, baseline))
    else:
        lines.append("No quality-spending local variant is on the risk-aware Pareto frontier.")
    lines.append("")

    lines.append("## Baseline-Dominating Candidates\n")
    if result.performance_dominating:
        lines.append("These variants dominate the baseline on performance/resource axes before accounting for risk as a separate objective.\n")
        lines.extend(_candidate_table(result.performance_dominating[:top_n], baseline))
    else:
        lines.append("No local candidate strictly dominates the baseline on performance/resource axes.")
    lines.append("")

    lines.append("## Near-Dominating Candidates\n")
    if result.near_dominating:
        if result.allow_quality_spending:
            lines.append("These variants improve at least one resource axis while staying inside the quality-risk budget.\n")
        else:
            lines.append("These variants are same-quality/model-preserving and improve at least one resource axis.\n")
        lines.extend(_candidate_table(result.near_dominating[:top_n], baseline))
    else:
        lines.append("No additional near-dominating variants found inside the risk budget.")
    lines.append("")

    lines.append("## Decode KV Bandwidth\n")
    lines.append(_kv_bandwidth_note(result))
    lines.append("")

    if result.baseline_model and result.baseline_model.warnings:
        lines.append("## Baseline Ingestion Notes\n")
        for warning in result.baseline_model.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("## Caveats\n")
    lines.append("- Same-quality mode means the learned model topology and numerical precision are unchanged in the modifier schema.")
    lines.append("- Quality values are proxy estimates for ranking nearby candidates, not measured perplexity.")
    lines.append("- Risk labels are heuristic placeholders until empirical per-component sensitivity and coupling data are available.")
    lines.append("- MoE, state/hybrid, MLA, and heterogeneous layer edits are reserved hooks and are not selected by the current local modifier mode.")
    return "\n".join(lines)


def generate_modifier_shadow_report(result: ModifierResult, top_n: int = 12) -> str:
    """Generate shadow_prices.md from local modifier perturbations."""

    baseline = result.baseline
    lines: List[str] = []
    lines.append("# Shadow Price Report\n")
    lines.append("Local shadow prices are estimated by directly evaluating nearby architecture perturbations from the baseline.")
    lines.append("")

    lines.append("## Binding Constraints\n")
    lines.append(f"- Serving regime: {baseline.evaluated.binding_serving_regime} — {baseline.evaluated.binding_reason}")
    if baseline.evaluated.throughput.per_layer_breakdown:
        lines.append(f"- Training-layer bottleneck proxy: {baseline.evaluated.throughput.per_layer_breakdown.bottleneck}")
    if baseline.evaluated.constraint_violations:
        for violation in baseline.evaluated.constraint_violations:
            lines.append(f"- Baseline violation: {violation}")
    else:
        lines.append("- Baseline meets configured serving and memory constraints.")
    lines.append("")

    lines.append("## Marginal Tradeoffs\n")
    records = [r for r in result.all_records if not r.is_baseline]
    records.sort(key=lambda r: -modifier_score(r, baseline))
    lines.extend(_shadow_table(records[:top_n], baseline, result.quality_risk_budget_pct))
    lines.append("")

    lines.append("## Interpretation\n")
    lines.append("- Positive throughput deltas mean the variant is faster than the baseline.")
    lines.append("- Positive loss-proxy deltas mean expected quality risk increased.")
    lines.append("- Accepted/rejected is based on feasibility plus the configured relative loss-proxy budget.")
    return "\n".join(lines)


def generate_modifier_justification(result: ModifierResult, top_n: int = 6) -> str:
    """Generate justification.md for modifier mode."""

    selected = result.selected
    baseline = result.baseline
    c = selected.evaluated.arch
    lines: List[str] = []
    lines.append("# Modifier Justification\n")
    lines.append("This run used the baseline-aware Pareto modifier ability. The original greenfield compiler remains available when no baseline config is supplied.")
    lines.append("Default modifier selection is same-quality: it only selects changes that preserve the learned model topology and numerics.")
    lines.append("")
    lines.append("## Selected Config\n")
    lines.append(f"- Change: {selected.change_summary}")
    lines.append(f"- d_model={c.d_model}, layers={c.n_layers}, heads={c.n_heads}, d_head={c.d_head}, kv_heads={c.n_kv_heads}, ffn_dim={c.ffn_dim}")
    lines.append(f"- FFN precision={c.ffn_precision}, KV cache={c.kv_cache_bits}-bit, TP={selected.tp}")
    lines.append(f"- Relative loss-proxy delta: {quality_risk_pct(selected, baseline):+.3f}%")
    lines.append(f"- Quality-preserving: {selected.quality_preserving}")
    lines.append(f"- Move class: {selected.move_class}")
    lines.append(f"- Risk label: {selected.risk_label}")
    lines.append("")
    lines.append("## Why This Moved The Baseline\n")
    if selected.is_baseline:
        lines.append("The baseline was retained because no local modification improved resource use inside the configured quality-risk budget.")
    else:
        for change in selected.changes:
            lines.append(f"- {change.label}: {_reason_for_change(change.field)}")
    lines.append("")
    lines.append("## Pareto Context\n")
    lines.append(f"- Local candidates evaluated: {result.candidates_evaluated}")
    lines.append(f"- Feasible candidates: {len(result.feasible_records)}")
    lines.append(f"- Risk-aware Pareto frontier size: {len(result.pareto_frontier)}")
    if result.performance_dominating:
        lines.append(f"- Performance-dominating variants found: {len(result.performance_dominating)}")
    lines.append("")
    lines.append("## Representative Alternatives\n")
    alternatives = [r for r in result.pareto_frontier if r is not selected and not r.is_baseline][:top_n]
    if alternatives:
        lines.extend(_candidate_table(alternatives, baseline))
    else:
        lines.append("No additional non-baseline Pareto alternatives found.")
    lines.append("")
    lines.append("## Uncertainty\n")
    q = selected.evaluated.quality
    lines.append(f"- Confidence: {q.confidence}")
    lines.append(f"- Total residual: {q.total_penalty_fraction * 100:.2f}%")
    lines.append(f"- Uncertainty interval: +{q.uncertainty_low_pct:.2f}% to +{q.uncertainty_high_pct:.2f}% residual range")
    lines.append("- Treat this as an architecture-ranking signal, not a final perplexity prediction.")
    return "\n".join(lines)


def _metrics_block(title: str, rec: ModifierRecord, baseline: ModifierRecord) -> List[str]:
    ev = rec.evaluated
    return [
        f"- {title} loss proxy: {ev.predicted_loss:.4f} ({quality_risk_pct(rec, baseline):+.3f}% vs baseline)",
        f"- {title} training throughput: {ev.training_tps:,.0f} tok/s ({_pct_delta(ev.training_tps, baseline.evaluated.training_tps):+.1f}%)",
        f"- {title} TBT: {ev.serving_tbt_ms:.2f} ms ({_pct_delta_lower_is_better(ev.serving_tbt_ms, baseline.evaluated.serving_tbt_ms):+.1f}% faster)",
        f"- {title} TTFT: {ev.throughput.prefill_time_ms:.2f} ms ({_pct_delta_lower_is_better(ev.throughput.prefill_time_ms, baseline.evaluated.throughput.prefill_time_ms):+.1f}% faster)",
        f"- {title} memory/GPU: {ev.memory_per_gpu_gb:.2f} GB ({_pct_delta_lower_is_better(ev.memory_per_gpu_gb, baseline.evaluated.memory_per_gpu_gb):+.1f}% lower)",
        f"- {title} modeled KV cache/GPU: {rec.kv_cache_gb:.3f} GB ({_pct_delta_lower_is_better(rec.kv_cache_gb, baseline.kv_cache_gb):+.1f}% lower)",
    ]


def _candidate_table(records: Iterable[ModifierRecord], baseline: ModifierRecord) -> List[str]:
    lines = [
        "| Change | Risk | Loss Risk | TBT Improvement | Train TPS Improvement | Mem Improvement | Modeled KV Improvement | Reason |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rec in records:
        ev = rec.evaluated
        lines.append(
            "| "
            f"{rec.change_summary} | "
            f"{rec.risk_label} | "
            f"{quality_risk_pct(rec, baseline):+.3f}% | "
            f"{_fmt_pct_lower_is_better(ev.serving_tbt_ms, baseline.evaluated.serving_tbt_ms)} | "
            f"{_fmt_pct_higher_is_better(ev.training_tps, baseline.evaluated.training_tps)} | "
            f"{_fmt_pct_lower_is_better(ev.memory_per_gpu_gb, baseline.evaluated.memory_per_gpu_gb)} | "
            f"{_fmt_pct_lower_is_better(rec.kv_cache_gb, baseline.kv_cache_gb)} | "
            f"{_reason_for_record(rec)} |"
        )
    return lines


def _shadow_table(
    records: Iterable[ModifierRecord],
    baseline: ModifierRecord,
    quality_risk_budget_pct: float,
) -> List[str]:
    lines = [
        "| Perturbation | Feasible | Throughput Impact | Quality Proxy Impact | Decision |",
        "|---|---:|---:|---:|---|",
    ]
    for rec in records:
        ev = rec.evaluated
        throughput_impact = (
            f"TBT {_pct_delta_lower_is_better(ev.serving_tbt_ms, baseline.evaluated.serving_tbt_ms):+.1f}%, "
            f"train {_pct_delta(ev.training_tps, baseline.evaluated.training_tps):+.1f}%"
        )
        loss = quality_risk_pct(rec, baseline)
        decision = _decision_for_record(rec, baseline, quality_risk_budget_pct)
        lines.append(
            f"| {rec.change_summary} | {ev.meets_constraints} | {throughput_impact} | {loss:+.3f}% | {decision} |"
        )
    return lines


def _kv_bandwidth_note(result: ModifierResult) -> str:
    baseline = result.baseline
    selected = result.selected
    base_arch = baseline.evaluated.arch
    sel_arch = selected.evaluated.arch
    base_group = base_arch.n_heads // base_arch.n_kv_heads
    sel_group = sel_arch.n_heads // sel_arch.n_kv_heads
    if selected.is_baseline:
        return (
            f"Baseline uses {base_arch.n_kv_heads} KV heads (GQA group size {base_group}) "
            f"and {base_arch.kv_cache_bits}-bit KV. No accepted local change reduced KV bandwidth "
            "inside the risk budget."
        )
    return (
        f"Baseline uses {base_arch.n_kv_heads} KV heads (GQA group size {base_group}) "
        f"and {base_arch.kv_cache_bits}-bit KV. Selected config uses {sel_arch.n_kv_heads} KV heads "
        f"(group size {sel_group}) and {sel_arch.kv_cache_bits}-bit KV. The estimated per-GPU KV "
        f"cache footprint changes from {baseline.kv_cache_gb:.3f} GB to {selected.kv_cache_gb:.3f} GB "
        "under the current single-stream decode memory proxy."
    )


def _decision_for_record(
    rec: ModifierRecord,
    baseline: ModifierRecord,
    quality_risk_budget_pct: float,
) -> str:
    if not rec.evaluated.meets_constraints:
        return "rejected: " + "; ".join(rec.evaluated.constraint_violations[:2])
    if quality_risk_pct(rec, baseline) > quality_risk_budget_pct:
        return "rejected: outside quality-risk budget"
    if not _record_improves_resource(rec, baseline):
        return "rejected: worsens or does not improve resource axes"
    if quality_risk_pct(rec, baseline) > 0:
        return f"accepted within risk budget ({rec.risk_label})"
    if rec.evaluated.serving_tbt_ms < baseline.evaluated.serving_tbt_ms:
        return "accepted: improves decode latency with no loss-proxy increase"
    return "neutral"


def _record_improves_resource(rec: ModifierRecord, baseline: ModifierRecord) -> bool:
    ev = rec.evaluated
    base = baseline.evaluated
    return (
        ev.serving_tbt_ms < base.serving_tbt_ms
        or ev.throughput.prefill_time_ms < base.throughput.prefill_time_ms
        or ev.training_tps > base.training_tps
        or ev.memory_per_gpu_gb < base.memory_per_gpu_gb
        or rec.kv_cache_gb < baseline.kv_cache_gb
    )


def _reason_for_record(rec: ModifierRecord) -> str:
    fields = {ch.field for ch in rec.changes}
    if "kv_cache_bits" in fields or "n_kv_heads" in fields:
        return "changes KV bandwidth/capacity pressure"
    if "ffn_precision" in fields:
        return "uses lower-precision FFN matmuls"
    if "tp" in fields:
        return "changes tensor-parallel communication/compute split"
    if "n_layers" in fields:
        return "trades depth quality proxy against latency"
    if "ffn_dim" in fields:
        return "changes FFN capacity and tile shape"
    return "local architecture perturbation"


def _reason_for_change(field: str) -> str:
    return {
        "kv_cache_bits": "reduces KV cache bytes and decode KV bandwidth pressure, with a heuristic quantization residual.",
        "n_kv_heads": "trades KV bandwidth and memory against the GQA quality proxy.",
        "ffn_precision": "uses lower-precision FFN matmuls for throughput/memory gains with a precision residual.",
        "tp": "changes tensor-parallel sharding and all-reduce pressure.",
        "n_layers": "trades depth against latency, memory, and scaling-law shape residual.",
        "ffn_dim": "adjusts FFN capacity while keeping the dimension tile-friendly.",
    }.get(field, "local baseline-relative architecture change.")


def _pct_delta(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old * 100


def _pct_delta_lower_is_better(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (old - new) / old * 100


def _fmt_pct_lower_is_better(new: float, old: float) -> str:
    """Like _pct_delta_lower_is_better but renders very large worsenings as a
    multiplicative ratio (`3.0× larger`) instead of `-200.0%`, which readers
    routinely misinterpret as "the metric got 2× smaller".
    """
    if old == 0:
        return "+0.0%"
    pct = (old - new) / old * 100
    # Only switch to ratio form when the metric got dramatically worse and the
    # percentage would otherwise read as a misleading negative > 100%.
    if pct < -100 and new > 0 and old > 0:
        ratio = new / old
        return f"{ratio:.2g}× larger"
    return f"{pct:+.1f}%"


def _fmt_pct_higher_is_better(new: float, old: float) -> str:
    """As above but for metrics where 'higher is better' (e.g., training TPS).
    Renders giant improvements as a ratio (`5.4× higher`) instead of `+440%`.
    """
    if old == 0:
        return "+0.0%"
    pct = (new - old) / old * 100
    if pct > 200 and new > 0 and old > 0:
        ratio = new / old
        return f"{ratio:.2g}× higher"
    return f"{pct:+.1f}%"
