"""
Architecture Compiler v0 — Justification Generator

Generates a markdown justification document from an OptimizationResult.
One section per non-trivial design decision, each citing the constraint
or penalty that drove it.
"""

from typing import Optional
from optimizer import OptimizationResult, EvaluatedCandidate
from shadow_prices import ShadowPriceReport


# Wave 28: human-readable names for the state-mixer families. The
# justification used to hard-code "Mamba-2 structured SSM" for every
# hybrid pick, so a --state-type gated_deltanet run emitted a config
# whose layer_configs said `"type": "gated_deltanet"` while the
# markdown told the reader they had picked Mamba-2.
_STATE_FAMILY_DISPLAY = {
    "mamba2": "Mamba-2 structured SSM",
    "mamba": "Mamba-1 selective SSM",
    "gla": "Gated Linear Attention (GLA)",
    "kda": "Kimi Delta Attention (KDA)",
    "gated_deltanet": "Gated DeltaNet",
    "deltanet": "DeltaNet",
    "rwkv7": "RWKV-7",
    "retnet": "RetNet",
    "swa": "sliding-window attention",
    "sliding_window": "sliding-window attention",
    "linear_attention": "linear attention",
}


def _state_family_display(state_config) -> str:
    stype = str((state_config or {}).get("state_type", "mamba2") or "mamba2")
    return _STATE_FAMILY_DISPLAY.get(stype, stype)


def generate_justification(
    result: OptimizationResult,
    shadow_report: Optional[ShadowPriceReport] = None,
) -> str:
    """Generate markdown justification from optimizer output."""

    if result.optimal is None:
        return _no_solution(result)

    opt = result.optimal
    c = opt.arch
    q = opt.quality
    t = opt.throughput
    con = result.constraints

    lines = []
    lines.append("# Architecture Justification\n")

    # Target summary
    lines.append(f"**Target**: {con.target_params_b}B active parameters, "
                 f"{con.training_tokens/1e12:.1f}T training tokens, "
                 f"{result.hardware.upper()} (TP={con.tp} PP={con.pp} DP={con.dp}), "
                 f"context={con.context_length}.")
    # Surface the param tolerance band so users can see why active_params_b
    # may not match --params exactly.
    tol = float(getattr(con, "param_tolerance", 0.0) or 0.0)
    if tol > 0:
        target_b = float(con.target_params_b)
        lo_b = target_b * (1 - tol)
        hi_b = target_b * (1 + tol)
        actual_b = (
            getattr(opt.arch, "active_params_b", 0.0)
            or getattr(opt.arch, "total_params_b", 0.0)
            or target_b
        )
        lines.append(
            f"Parameter tolerance: ±{tol*100:.0f}% "
            f"(accepts shapes in [{lo_b:.2f}B, {hi_b:.2f}B]; "
            f"selected: {float(actual_b):.2f}B). "
            f"Override with `--param-tolerance 0.05` for a tighter match."
        )
        # v1-fix D (demo audit): when the selected active-params lands a
        # non-trivial fraction above the target, surface *why*. The
        # shape-law prior often prefers a slightly wider d_model than the
        # exact-target shape; the user typed `--params 1` and got 1.14B,
        # which looks like a bug. Naming the mechanism (shape-law /
        # lattice quantisation) and the override (`--param-tolerance`)
        # closes the loop without making the optimizer's default tighter.
        drift_pct = (float(actual_b) - target_b) / max(1e-9, target_b) * 100.0
        if abs(drift_pct) >= 5.0:
            direction = "above" if drift_pct > 0 else "below"
            lines.append(
                f"Active-params landed {abs(drift_pct):.1f}% {direction} target "
                f"({float(actual_b):.2f}B vs {target_b:.2f}B requested). "
                "The shape-law penalty preferred this shape over the exact-target "
                "candidate within the lattice quantisation; tighten with "
                "`--param-tolerance 0.05` if you need a closer match, or pass "
                "`--param-band 0.02` to force a hard band."
            )
    budget_parts = []
    if con.serving_tbt_ms is not None:
        budget_parts.append(f"TBT <= {con.serving_tbt_ms}ms")
    if con.serving_ttft_ms is not None:
        budget_parts.append(f"TTFT <= {con.serving_ttft_ms}ms")
    if budget_parts:
        lines.append(
            f"Serving soft budget(s): {', '.join(budget_parts)}, "
            f"batch={con.serving_batch}. These are warning/selection signals, "
            "not hard feasibility cuts."
        )
    lines.append("")

    # Predicted performance
    lines.append("## Predicted Performance\n")
    sig_tps = float(getattr(opt.throughput, "training_throughput_sigma_tps", 0.0) or 0.0)
    sig_tbt = float(getattr(opt.throughput, "decode_time_sigma_ms", 0.0) or 0.0)
    sig_pre = float(getattr(opt.throughput, "prefill_time_sigma_ms", 0.0) or 0.0)
    tps_suffix = f" (±{sig_tps:,.0f})" if sig_tps > 0 else ""
    tbt_suffix = f" (±{sig_tbt:.1f}ms)" if sig_tbt > 0 else ""
    pre_suffix = f" (±{sig_pre:.1f}ms)" if sig_pre > 0 else ""
    # Fix #2: be explicit about the throughput unit. The base number is
    # per-TP-replica; the aggregate scales by DP.
    dp_replicas = max(1, int(getattr(con, "dp", 1) or 1))
    agg_tps = opt.training_tps * dp_replicas
    lines.append(
        f"- **Training throughput (per TP replica, TP={con.tp})**: "
        f"{opt.training_tps:,.0f} tokens/sec{tps_suffix}"
    )
    lines.append(
        f"- **Aggregate training throughput (× DP={dp_replicas})**: "
        f"{agg_tps:,.0f} tokens/sec"
    )
    lines.append(f"- **Serving TBT**: {opt.serving_tbt_ms:.1f}ms{tbt_suffix}"
                 + (f" ({_pct_under(opt.serving_tbt_ms, con.serving_tbt_ms)} under soft budget)"
                    if con.serving_tbt_ms else ""))
    # Wave 19 (P0-2): always state the prompt length the TTFT was computed
    # for — an unqualified TTFT next to a long --context invited reading it
    # as full-context prefill when a shorter prompt was assumed.
    _ttft_prompt = int(getattr(con, "prompt_len", 0) or
                       getattr(con, "context_length", 0) or 0)
    # Wave 29: attribute the serving-stack floor so the TTFT is not read
    # as pure compute (and the excluded queueing term is named).
    _ttft_ovh = float(getattr(t, "ttft_serving_overhead_ms", 0.0) or 0.0)
    _ttft_ovh_txt = (
        f"; includes {_ttft_ovh:.1f}ms serving-stack floor "
        f"(tokenize/schedule/detokenize — excludes load-dependent queueing)"
        if _ttft_ovh > 0 else ""
    )
    lines.append(f"- **Serving TTFT**: {t.prefill_time_ms:.1f}ms{pre_suffix}"
                 f" (cold prefill of a {_ttft_prompt:,}-token prompt"
                 f"{_ttft_ovh_txt})"
                 + (f" ({_pct_under(t.prefill_time_ms, con.serving_ttft_ms)} under soft budget)"
                    if con.serving_ttft_ms else ""))
    lines.append(f"- **Memory per GPU**: {opt.memory_per_gpu_gb:.1f} GB")
    lines.append(f"- **Predicted loss**: {opt.predicted_loss:.4f} "
                 f"(scaling-law spine: {q.chinchilla_baseline:.4f}, "
                 f"total residual: {q.total_penalty_fraction*100:.2f}%)")
    lines.append(f"- **Confidence**: {q.confidence}")
    lines.append("")

    if getattr(q, "terms", None):
        lines.append("## Quality Proxy Backbone\n")
        if getattr(q, "quality_model_version", "") == "effective_capacity_v2":
            lines.append("The quality proxy uses a scaling-law spine over effective sparse capacity and effective training data. Serving-context effects are a separate task-utility delta relative to the pretraining context; the legacy active-parameter residual model remains selectable.")
        else:
            lines.append("The selected legacy quality proxy uses a scaling-law spine over active non-embedding parameters and training tokens, plus additive architecture residuals.")
        lines.append("The compiler treats query heads as a weak, saturating architecture prior derived from width, not as a monotonic quality law. KV heads are treated as a direct memory/latency tradeoff with uncertain GQA-sharing quality risk.")
        lines.append(f"- **Model version**: {getattr(q, 'quality_model_version', 'quality_v0')}")
        lines.append(f"- **Spine active proxy**: {getattr(q, 'spine_active_params', 0)/1e9:.3f}B active non-embedding params")
        lines.append(f"- **Spine effective proxy**: {getattr(q, 'spine_effective_params', 0)/1e9:.3f}B effective non-embedding params")
        lines.append(f"- **Pretraining loss proxy**: {getattr(q, 'pretraining_loss_proxy', q.predicted_loss):.4f}")
        lines.append(f"- **Total uncertainty**: ±{getattr(q, 'uncertainty_total', 0.0) * 100:.2f}%")
        for name in ("effective_capacity", "effective_data", "architecture_residual", "precision_residual", "moe_residual", "state_residual", "context_utility", "risk_residual", "data_quality"):
            term = q.terms.get(name)
            if not term or (term.confidence == "not_applicable" and abs(term.value) == 0):
                continue
            lines.append(f"- **{name}**: {term.value*100:.2f}% residual, "
                         f"±{term.uncertainty*100:.2f}% uncertainty, "
                         f"confidence={term.confidence}")
        lines.append("")

    # Design decisions
    lines.append("## Design Decisions\n")

    # d_model
    lines.append(f"### d_model = {c.d_model}\n")
    lines.append(f"Lattice constraint: must be divisible by TP={con.tp} and "
                 f"tile-aligned for BF16 CTA tiles on {result.hardware.upper()}. "
                 f"d_model={c.d_model} lies on the lattice with n_heads={c.n_heads} × "
                 f"d_head={c.d_head} = {c.n_heads * c.d_head}.")
    _add_alternatives(lines, result, "d_model", c.d_model)
    lines.append("")

    # n_layers
    # For MoE the Chinchilla aspect-ratio prior is over *active* (per-token
    # compute) params, not total — total includes sparse expert mass that
    # doesn't scale width/depth optimum the way dense parameters do.
    is_moe = (getattr(c, "moe_style", "dense") != "dense")
    aspect_ratio_basis_b = c.active_params_b if is_moe else c.total_params_b
    basis_label = "active" if is_moe else "total"
    lines.append(f"### n_layers = {c.n_layers}\n")
    lines.append(
        f"Chinchilla-derived aspect ratio target for {aspect_ratio_basis_b}B "
        f"{basis_label} parameters is approximately {c.n_layers} layers at "
        f"d_model={c.d_model}."
    )
    if con.pp > 1:
        lines.append(f"Divisible by PP={con.pp} for pipeline parallelism.")
    lines.append("")

    # Attention layout (MHA / GQA / MQA / MLA)
    attn_type = str(getattr(c, "attention_type", "full")).lower()
    if attn_type == "mla":
        # Multi-head Latent Attention — n_kv_heads is a shape-bookkeeping
        # field, NOT a GQA ratio. Surface MLA explicitly so the report does
        # not silently describe MLA as GQA.
        c_kv = int(getattr(c, "mla_kv_latent_dim", 0) or 0)
        c_q = int(getattr(c, "mla_q_latent_dim", 0) or 0)
        lines.append(f"### Attention: MLA (Multi-head Latent Attention)\n")
        lines.append(
            f"DeepSeek-V2/V3-style MLA selected: KV-latent c_kv={c_kv}, "
            f"Q-latent c_q={c_q}, n_heads={c.n_heads}, d_head={c.d_head}. "
            f"KV cache stores the c_kv latent (not n_kv_heads × d_head), "
            f"giving the bulk of the decode-bandwidth relief."
        )
        mla_pen = q.penalty_breakdown.get("attention_mla") if hasattr(q, "penalty_breakdown") else None
        if mla_pen and getattr(mla_pen, "value", 0) != 0:
            lines.append(
                f"Coupled MLA quality residual: {mla_pen.value*100:.2f}% "
                f"(uncertain low-rank attention prior)."
            )
        lines.append("")
    elif attn_type == "nsa":
        # Wave 28: NSA runs used to fall into the GQA branch below and
        # render "### n_kv_heads = 8 (GQA-4)" as the only attention
        # decision — the justification never mentioned that every layer
        # attends through NSA's compressed/selected/window branches.
        lines.append("### Attention: NSA (Native Sparse Attention)\n")
        lines.append(
            f"NSA selected: compress_block={getattr(c, 'nsa_compress_block_size', 0)} "
            f"(stride {getattr(c, 'nsa_compress_block_stride', 0)}), "
            f"select_block={getattr(c, 'nsa_select_block_size', 0)} × "
            f"top-{getattr(c, 'nsa_select_top_k', 0)}, "
            f"local window={getattr(c, 'nsa_window_size', 0)}; "
            f"n_heads={c.n_heads}, n_kv_heads={c.n_kv_heads}, d_head={c.d_head}. "
            "Prefill/decode attention cost scales with the compressed + "
            "selected + window token set rather than full context."
        )
        lines.append("")
    else:
        gqa_ratio = c.n_heads // c.n_kv_heads if c.n_kv_heads > 0 else 1
        if c.n_kv_heads == c.n_heads:
            gqa_label = "MHA"
        elif c.n_kv_heads == 1:
            gqa_label = "MQA"
        else:
            gqa_label = f"GQA-{gqa_ratio}"

        lines.append(f"### n_kv_heads = {c.n_kv_heads} ({gqa_label})\n")

        if c.n_kv_heads == c.n_heads:
            lines.append("Multi-head attention (MHA) selected. No KV sharing.")
        else:
            # Find the GQA penalty
            gqa_pen = q.penalty_breakdown.get("gqa")
            gqa_val = gqa_pen.value if gqa_pen else 0
            lines.append(f"{gqa_label} reduces KV cache by {gqa_ratio}x vs MHA. "
                         f"Coupled GQA-sharing residual: {gqa_val*100:.2f}% "
                         f"with uncertainty from KV-head sharing.")
            if gqa_ratio <= 8 and c.d_model >= 2048:
                lines.append("GQA-8 at d_model ≥ 2048 is within seed variance per published ablations.")
        lines.append("")

    # Weight precision
    lines.append(f"### Weight precision: FFN={c.ffn_precision}, "
                 f"attention={c.attn_precision.get('v', 'bf16')}\n")
    wp = q.penalty_breakdown.get("weight_precision")
    if wp and wp.value > 0:
        lines.append(f"Quality model penalty: {wp.value*100:.2f}% relative PPL "
                     f"(source: {wp.source}).")
    else:
        lines.append("BF16 baseline — no weight precision penalty.")
    if c.ffn_precision != "bf16":
        lines.append(f"FFN at {c.ffn_precision.upper()} is well-tolerated "
                     f"(~0.1% relative PPL per FP8-LM); throughput gain outweighs quality cost.")
    lines.append("")

    # KV cache
    lines.append(f"### KV cache: {c.kv_cache_bits}-bit\n")
    kv_pen = q.penalty_breakdown.get("kv_quant")
    if kv_pen and kv_pen.value > 0:
        lines.append(f"KV quantization penalty: {kv_pen.value*100:.2f}% "
                     f"(source: {kv_pen.source}).")
    else:
        lines.append("Full-precision KV cache — no quantization penalty.")
    lines.append("")

    # State/hybrid architecture (v2)
    if c.n_state_layers > 0:
        lines.append(f"### Hybrid architecture: {c.n_attention_layers} attention + "
                     f"{c.n_state_layers} state layers\n")
        lines.append(f"State mechanism: {_state_family_display(c.state_config)} with "
                     f"d_state={c.derived_d_state} (SRAM-derived).")
        lines.append(f"Placement strategy: {c.placement_strategy}.")
        lines.append(f"Hybrid ratio: {c.hybrid_ratio} "
                     f"(attention:state layers).")
        if c.crossover_seq_len > 0:
            lines.append(f"Decode cost crossover L* = {c.crossover_seq_len:.0f}: "
                         f"above this sequence length, state layers are cheaper "
                         f"than attention layers at decode time.")
        lines.append(f"State layers have NO KV cache — decode cost is "
                     f"L-independent, and state is SRAM-resident.")
        if c.state_config:
            lines.append(f"State config: n_heads={c.state_config.get('n_heads', 'N/A')}, "
                         f"d_head={c.state_config.get('d_head', 'N/A')}, "
                         f"precision={c.state_config.get('state_precision', 'bf16')}.")
        state_pen = q.penalty_breakdown.get("state")
        if state_pen and abs(state_pen.value) > 0:
            lines.append(f"State/hybrid quality residual: {state_pen.value*100:.2f}% "
                         f"(confidence: {state_pen.confidence}).")
        lines.append("")

    # YOCO KV sharing (Wave 28: a --yoco pick emitted architecture.yoco
    # in the config — and halved the KV budget — without the
    # justification mentioning YOCO once).
    _yoco_k = int(getattr(c, "yoco_n_self_attn_layers", 0) or 0)
    if _yoco_k > 0:
        _yoco_pat = str(getattr(c, "yoco_share_pattern", "single_source"))
        lines.append(f"### YOCO KV sharing: {_yoco_k} self-attention "
                     f"KV-producer layer(s), pattern={_yoco_pat}\n")
        lines.append(
            "YOCO (You Only Cache Once): the remaining layers cross-attend "
            "to the shared KV cache produced by the "
            f"{'first layer' if _yoco_k == 1 else f'first {_yoco_k} layers'}, "
            "so the KV footprint is that of "
            f"{_yoco_k} layer(s) instead of {c.n_layers}. The quality cost "
            "is carried in the architecture residual (yoco_sharing "
            "subterm); treat it as a low-confidence prior — published "
            "YOCO ablations are thin above 3B."
        )
        lines.append("")

    # Multi-token prediction (Wave 28: mtp=2 picks were invisible here.)
    _mtp_k = int(getattr(c, "mtp_n_predict_depths", 0) or 0)
    if _mtp_k > 0:
        lines.append(f"### Multi-token prediction: depth {_mtp_k}\n")
        lines.append(
            f"MTP head predicts {_mtp_k} future token(s) per position "
            f"({int(getattr(c, 'mtp_depth_n_layers', 1) or 1)} extra "
            f"layer(s) per depth, train loss weight "
            f"{float(getattr(c, 'mtp_train_loss_weight', 0.0) or 0.0):g}). "
            "Training-time auxiliary loss; the serving numbers here do "
            "NOT assume speculative-decoding acceptance gains."
        )
        lines.append("")

    # Search stats
    lines.append("## Search Statistics\n")
    lines.append(f"- Candidates generated: {result.candidates_generated:,}")
    lines.append(f"- Candidates evaluated: {result.candidates_evaluated:,}")
    lines.append(f"- Feasible candidates: {result.candidates_feasible:,}")
    lines.append(f"- Pareto frontier size: {len(result.pareto_frontier)}")
    lines.append(f"- Search time: {result.search_time_sec:.1f}s")
    lines.append("")

    # Contending family — candidates statistically indistinguishable from
    # the selected one under the quality model's uncertainty band. Surfaces
    # the *axes* along which they vary so a pretrain lead can see what's
    # actually being chosen between, instead of treating a single picked
    # config as deterministic.
    #
    # Wave 27: route through `compute_contending_family_full` so this
    # markdown table, the CLI `WARNING: N contending candidate(s)`
    # count, the emitted config's
    # `confidence_envelope.contending_candidates`, and the
    # `<config>_contending_family.json` sidecar all agree. Pre-Wave-27
    # this section re-derived contenders with the naïve `_loss_interval`
    # overlap rule (counting shared model error against the pick),
    # which on a default H100 7B run reported 34 markdown contenders vs
    # 9 in the CLI warning — a 3.8x inflation a researcher reads as
    # "the pick is fragile" even when the paired-sigma count says
    # otherwise.
    try:
        try:
            from optimizer import compute_contending_family_full  # type: ignore
        except Exception:  # pragma: no cover - package layout fallback
            from ac.optimizer import compute_contending_family_full  # type: ignore
        # Cap the markdown table at 8 rows for readability; the sidecar
        # JSON carries up to 32 for programmatic consumers.
        family = compute_contending_family_full(result, opt, top_n=8)
    except Exception:
        family = {"row_count": 0, "members": [], "varying_axes": []}
    if family.get("row_count", 0) > 0:
        lines.append("## Contending Family\n")
        # Wave 27: unified scope across CLI warning, config, sidecar and
        # this markdown table. The count includes candidates within
        # paired-sigma of the pick on loss OR within ±1σ on any
        # throughput metric (subject to a paired-sigma loss gate so
        # slower-and-losier candidates aren't counted).
        lines.append(
            f"The selected configuration is statistically indistinguishable "
            f"from **{family['row_count']} other feasible candidate(s)** "
            f"inside the paired quality-model uncertainty band (loss, or "
            f"within ±1σ on throughput). Treat the \"selected\" pick as one "
            f"element of this family, not a deterministic answer. (The "
            f"companion sidecar `<config>_contending_family.json` carries "
            f"up to the top 32 members with the same scope.)\n"
        )
        axes = family.get("varying_axes") or []
        if axes:
            lines.append(
                "**Axes along which the family varies:** "
                + ", ".join(f"`{a}`" for a in axes)
                + "\n"
            )
        members = family.get("members", [])
        if members:
            lines.append(
                "| pick | d_model | n_layers | n_kv_heads | ffn_prec | "
                "kv_bits | active_B | pred_loss | TBT (ms) | mem (GB) |"
            )
            lines.append(
                "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"
            )
            sel_row = family.get("selected") or {}
            sel_key = (sel_row.get("d_model"), sel_row.get("n_layers"),
                       sel_row.get("n_kv_heads"), sel_row.get("ffn_precision"),
                       sel_row.get("kv_cache_bits"))

            # Wave 29: uniform numeric formatting. The raw dict values
            # carried whatever rounding their producer used (2.04154 in
            # one row, 2.041368 in the next), which reads as false
            # precision differences inside one table.
            def _fmt(v, nd):
                try:
                    return f"{float(v):.{nd}f}"
                except (TypeError, ValueError):
                    return str(v)

            def _family_row(tag, r):
                return (
                    f"| {tag} | {r.get('d_model')} | "
                    f"{r.get('n_layers')} | {r.get('n_kv_heads')} | "
                    f"{r.get('ffn_precision')} | "
                    f"{r.get('kv_cache_bits')} | "
                    f"{_fmt(r.get('active_params_b'), 2)} | "
                    f"{_fmt(r.get('predicted_loss'), 5)} | "
                    f"{_fmt(r.get('serving_tbt_ms'), 2)} | "
                    f"{_fmt(r.get('memory_per_gpu_gb'), 2)} |"
                )

            # Selected row first
            lines.append(_family_row("**selected**", sel_row))
            for r in members:
                key = (r.get("d_model"), r.get("n_layers"), r.get("n_kv_heads"),
                       r.get("ffn_precision"), r.get("kv_cache_bits"))
                if key == sel_key:
                    continue
                lines.append(_family_row("contender", r))
            lines.append("")
    else:
        lines.append("## Contending Family\n")
        lines.append(
            "The selected configuration is robust to the quality model's "
            "current uncertainty band — no other feasible candidate falls "
            "inside its loss interval. The selection should be treated as a "
            "single deterministic pick at this confidence level."
        )
        lines.append("")

    # Shadow prices
    if shadow_report and shadow_report.prices:
        lines.append("## Shadow Prices\n")
        lines.append("What happens if you relax each constraint:\n")
        for sp in shadow_report.prices:
            sign = "+" if sp.delta_loss_pct >= 0 else ""
            lines.append(f"- **{sp.perturbation_desc}**: {sign}{sp.delta_loss_pct:.2f}% "
                         f"predicted quality change. {sp.interpretation}")
        lines.append("")

    # Caveats
    lines.append("## Caveats\n")
    lines.append("- Quality predictions are *relative* within this parameter band "
                 "(compiler scaling-law spine + modular residuals; not validated against absolute PPL).")
    lines.append("- Throughput predictions assume calibrated system efficiency factors "
                 "(training ~37%, decode ~42% on H100); actual performance may vary ±20%.")
    if any(p.confidence == "low" for p in q.penalty_breakdown.values() if p.value > 0):
        lines.append("- Some penalty values (e.g., FP4) are low-confidence — "
                     "derived from early literature with sparse per-component data.")
    lines.append("- KV quantization penalties calibrated from KIVI on 7B-scale models; "
                 "transfer to other model families assumed.")
    if q.confidence_notes:
        for note in q.confidence_notes:
            lines.append(f"- {note}")
    lines.append("")

    return "\n".join(lines)


def _pct_under(actual: float, budget: Optional[float]) -> str:
    if budget is None or budget == 0:
        return "N/A"
    pct = ((budget - actual) / budget) * 100
    return f"{pct:.0f}%"


def _add_alternatives(lines, result, dim_name, chosen_value):
    """Add a note about alternative values seen on the Pareto frontier."""
    alt_values = set()
    for ev in result.pareto_frontier[:10]:
        val = getattr(ev.arch, dim_name, None)
        if val is not None and val != chosen_value:
            alt_values.add(val)
    if alt_values:
        alts = sorted(alt_values)[:3]
        lines.append(f"Pareto alternatives: {', '.join(str(a) for a in alts)}.")


def _no_solution(result: OptimizationResult) -> str:
    """Generate justification when no feasible solution was found."""
    con = result.constraints
    lines = [
        "# Architecture Justification\n",
        "## No Feasible Solution Found\n",
        f"No architecture in the search space satisfies all constraints for "
        f"{con.target_params_b}B on {result.hardware}.\n",
        f"- Candidates generated: {result.candidates_generated:,}",
        f"- Candidates evaluated: {result.candidates_evaluated:,}",
        f"- Feasible: 0\n",
        "**Suggestions**:",
        "- Relax serving TBT/TTFT budgets",
        "- Increase param tolerance",
        "- Reduce context length",
        "- Try a different hardware target",
        "",
    ]
    return "\n".join(lines)


def generate_assumptions() -> str:
    """Generate assumptions.md content."""
    return """# Architecture Compiler v0 — Assumptions

## Quality Proxy

- No training sweeps were performed. The default `effective_capacity_v2` proxy uses effective sparse capacity and discounted effective data in the scaling-law spine. `legacy_residual_v1` preserves the previous active-parameter spine and fractional capacity bonus.
- The quality proxy predicts relative expected loss among nearby architecture candidates; it is not an absolute perplexity predictor.
- Architecture residuals couple width/depth, MLP-to-attention allocation, d_head, query heads, KV heads, and GQA sharing. Query heads are a weak width-derived prior, not a monotonic quality law; KV heads carry the direct memory/latency tradeoff and GQA-sharing uncertainty.
- Precision residuals use a configurable per-component sensitivity table plus hardware feasibility checks from the legacy penalty table.
- MoE routing/stability and state/hybrid residuals are explicit high-uncertainty hooks. Sparse capacity itself is represented by `N_effective` in v2.
- Residual composition is additive. In practice, residuals interact (e.g., GQA + FP4 KV may be worse than the sum). The coupling matrix is deferred to a later calibrated model.
- State/hybrid residuals (v2) model two penalties: compression (effective memory horizon vs context) and composition (state fraction × d_state ratio × context scaling). Recall-intensive tasks incur a 3x compression penalty multiplier. Quality saturation caps d_state at 256 regardless of hardware capacity.
- Uncertainty intervals are approximate. They capture residual confidence levels, scaling-law regime uncertainty, and placeholder uncertainty for uncalibrated hooks; data distribution effects are disabled by default.

## MoE Quality Residual and Effective Capacity

- The MoE residual is calibrated against Krajewski et al. (2024) plus published Mixtral 8×7B and DeepSeek-V2/V3 priors. It is not fit against measured loss deltas from training runs the compiler can run itself.
- In `effective_capacity_v2`, inactive expert capacity enters through `N_effective`; the legacy percentage capacity bonus is disabled. The old `-0.05 × log(N_total/N_active)` behavior is retained only by `legacy_residual_v1`.
- Granularity bonus: `-0.005 × log(max(G/8, 1))` where `G = N_total / top_k`. Applied above the Krajewski reference granularity of 8.
- Shared-expert adjust: `-0.005 × shared_ratio` (DeepSeek-V2 shared-expert ablation).
- Top-k=1 (Switch) penalty: `+0.015` — Switch vs ≥top_k=2 ablations in the Mixture-of-Experts literature.
- Router-fp8 penalty: `+0.005` when router precision is fp8 (DeepSeekMoE router-stability ablation).
- Routing-imbalance penalty: `+0.10 × max(0, 1 - load_balance)`. **Defaults to zero** because v1 assumes balanced routing under a load-balance loss. Production deployments with degraded balance (e.g., LB loss removed, no dropless routing) should set `load_balance` explicitly.
- The MoE residual is currently a smooth function of (`n_experts`, `top_k`, `expert_dim`, `shared_dim`). It does NOT model expert collapse, dropless-vs-dropping scheduler tradeoffs, or token-choice vs expert-choice routing. These are out of scope for v1.
- First-K-dense FFN patterns change active/total parameter ledgers directly and retain a small stability residual (`-0.003` per dense prefix layer, up to 3 layers).

## State / Hybrid Quality Residual (v2)

- Only **Mamba-2** has a measured-empirical reference implementation in `ac-base/` (PyTorch and JAX). Other state families (Mamba-1, Gated DeltaNet, KDA, GLA, RWKV-7, sliding-window, generic linear attention) are validated by the schema and routed to the correct family in the quality residual, but their reference models are research-stub: shapes are correct and `verify_forward()` round-trips, but they are not production-tuned and should not be used to benchmark wall-clock throughput.
- The family-specific decomposition (`f_hybrid_ratio`, `f_state_capacity`, `f_kv_cost`, `f_recall_risk`, `f_family_uncertainty`) uses the band-pass priors in `quality_defaults.yaml:state_residual.families`. DeepSeek-V3 is treated as MLA+MoE, not as a state-space hybrid. These priors should be re-fit when matched sweeps land.
- Family resolution rules (`_resolve_hybrid_family`):
  - `mamba_sequential` — Mamba-1/2, S4/S5/S6.
  - `gated_delta_or_kda_linear` — DeltaNet, Gated DeltaNet, Kimi Delta Attention (KDA), GLA (when used with gated linear attention).
  - `generic_linear_attention` — RWKV-7, RetNet, generic linear attention; also the fallback when a per-family band hasn't been calibrated.
  - `parallel_hybrid_heads` — MoH-style parallel splits where attention/state mixers run in parallel within a layer rather than alternating across layers.
  - `recurrent_local_attention` — Sliding Window Attention (SWA), local-attention + recurrent state hybrids.
- `recall_intensive` workloads pay a 3× compression-penalty multiplier (heuristic from RULER / passkey ablations); other workloads pay 1×.
- The 5-term residual is uncertainty-heavy on purpose: family confidence ranges from `medium` (mamba_sequential, generic_linear_attention) to `medium-low` (parallel_hybrid_heads, recurrent_local_attention).

## Throughput Model

- Throughput is estimated via analytic roofline modeling: per-operation cost = max(compute_time, memory_time).
- Kernel calibration, when available, overrides analytic GEMM efficiency estimates. If calibration data is absent, public spec estimates are used (source: "public_estimate").
- No compute-communication overlap is assumed. In production systems, overlap can improve throughput by 10-30%.
- Pipeline bubble fraction uses the simple (pp-1)/(M+pp-1) formula. Interleaved schedules are not modeled.

## Serving Model

- Serving workload is modeled using three canonical regimes (prefill-heavy, decode-heavy, mixed). This does not capture continuous batching scheduler dynamics.
- KV cache memory is computed assuming static allocation for the full context length. PagedAttention-style dynamic allocation is not modeled.
- Latency predictions are single-request estimates. Queuing effects under concurrency are not modeled in v0.

## Search

- Brute-force enumeration over the lattice. No gradient-based or Bayesian optimization. Tractable because the lattice constrains the space to ~1K-9K candidates per configuration.
- Parallelism (TP, PP, DP) is treated as a fixed input, not a search variable. Joint architecture-parallelism search is deferred to v3.

## Hardware

- Hardware specs are from public documentation. Actual performance may vary based on firmware, driver version, cooling, and system configuration.
- Results are intended for architecture ranking, not final production deployment validation.
"""


def generate_model_card() -> str:
    """Generate model_card.md content for the compiler itself."""
    return """# Architecture Compiler v0 — Model Card

## Intended Use

Dense-transformer architecture ranking under explicit training and serving constraints. Designed to produce internally consistent, falsifiable architecture recommendations with hardware-derived justifications.

## Not Intended For

- Final production architecture selection without training validation.
- Absolute perplexity prediction.
- Cross-architecture-family comparisons (e.g., dense vs. MoE).

## Hardware Scope

NVIDIA H100 SXM (primary, with calibration), NVIDIA B200, Google TPU v5p, Google TPU v5e (analytic only).

## Model Scope

Decoder-only Transformers with MHA/GQA/MQA. Supports dense (v0), MoE (v1), and hybrid attention/state architectures (v2). v2 adds Mamba-2 structured SSM layers with SRAM-derived d_state and hardware-derived hybrid ratios. v1-fix Part J extends the schema validator to accept additional state families — Mamba-1/S4/S5/S6, Sliding Window Attention, DeltaNet, Gated DeltaNet, Kimi Delta Attention (KDA), GLA, RWKV-7, and generic linear attention — with reference-model stubs in PyTorch and JAX.

## Quality Model

Relative expected-loss proxy based on a configurable compiler scaling-law spine plus modular residuals for coupled architecture variables, precision, MoE, state/memory mechanisms, risk, and optional data quality. Dense v0 continuity is preserved with compatibility aliases for shape and GQA, but the architecture residual models width/depth, MLP-attention ratio, d_head, query heads, KV heads, and GQA sharing together.

## Throughput Model

Analytic roofline with optional lightweight kernel calibration. Covers GEMM, fused attention, KV cache bandwidth, TP all-reduce, and PP bubble. System efficiency factors calibrated against published benchmarks.

## Known Failure Modes

- May mis-rank architectures when actual kernel efficiency differs significantly from calibration data.
- May underestimate quality loss from aggressive GQA at large group sizes (>8).
- Additive residual composition breaks down when stacking 3+ large quality risks.
- Data quality defaults to disabled and optimizer stability/training dynamics are not modeled.
- MoE quality terms are hooks with moderate uncertainty calibrated from published routing ablations (Krajewski + Mixtral + DeepSeek priors). Routing imbalance defaults to zero — production deployments with degraded balance must set `load_balance` explicitly.
- MoE all-to-all cost assumes ring-AllToAll efficiency of 0.67 of peak NVLink BW within the NVLink domain and 0.80 of single-axis ICI BW on TPU; cross-axis ICI multiplies by 2.5. These efficiencies are heuristics, not measured.
- State/hybrid quality residuals (v2) model compression and composition penalties with medium confidence; the recall-intensive task multiplier (3x) is empirically motivated but not sweep-calibrated.
- Only Mamba-2 has a measured-empirical reference implementation in `ac-base/`. Other state families (Mamba-1, Gated DeltaNet, KDA, GLA, RWKV-7, sliding-window, generic linear attention) have research-stub references that round-trip shape but are not production-tuned — do not use them to benchmark wall-clock throughput.
- Does not model RL post-training workloads or production scheduler details.
- Does not model compute-communication overlap.

## Validation Target

Within 25% throughput prediction error on known public baselines (Llama-2/3, Mistral). Stable pairwise ranking among architectures within the same parameter band.

## Minimum Claim

Under the stated workload and quality proxy assumptions, the compiler finds a dense architecture that improves predicted serving throughput over a hand-copied Llama-style baseline at similar expected loss. The result is a falsifiable architecture recommendation with explicit hardware-derived justifications, not a claim of global optimality.
"""
