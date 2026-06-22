"""
Architecture Compiler v0 — Shadow Price Computation

Computes the marginal quality improvement per unit of constraint relaxation.
Uses constraint perturbation: re-run the optimizer with each constraint
shifted by a fixed amount, report the change in optimal Q.

v0: hard-coded list of ~6 perturbations. Each re-run takes seconds.
Total shadow-price computation adds ~30 seconds to run time.

Not via formal Lagrangians — constraint perturbation is empirically
equivalent at v0 fidelity, simpler, and easier to explain.
"""

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from optimizer import (
    optimize, OptimizationResult, DeploymentConstraints,
    evaluate_candidate, CandidateArch, EvaluatedCandidate,
    OBJECTIVE_PROFILES,
)


@dataclass
class ShadowPrice:
    """A single shadow price entry."""
    constraint: str           # name of the perturbed constraint
    original_value: str       # original constraint value
    perturbed_value: str      # perturbed constraint value
    perturbation_desc: str    # human-readable description
    original_loss: float      # optimal predicted loss at original
    perturbed_loss: float     # optimal predicted loss after perturbation
    delta_loss: float         # absolute change
    delta_loss_pct: float     # percentage change
    interpretation: str       # one-line explanation


@dataclass
class ArchDimShadowPrice:
    """Shadow price for a single architecture dimension perturbation."""
    dimension: str           # e.g. "n_layers", "d_model", "n_kv_heads"
    change_desc: str         # e.g. "+1 layer", "+256 d_model"
    base_value: int          # original value
    perturbed_value: int     # new value
    # Quality impact
    delta_loss_pct: float    # change in predicted loss (%)
    # Throughput impact
    delta_train_tps_pct: float    # change in training TPS (%)
    delta_tbt_pct: float          # change in decode TBT (%)
    delta_mem_pct: float          # change in memory (%)
    # Decision
    decision: str            # "accepted" | "rejected" | "neutral"
    reason: str              # why accepted/rejected
    feasible: bool = True    # does perturbed arch meet constraints?


@dataclass
class ShadowPriceReport:
    """Full shadow price report."""
    prices: List[ShadowPrice] = field(default_factory=list)
    arch_dim_prices: List[ArchDimShadowPrice] = field(default_factory=list)
    base_loss: float = 0.0
    hardware: str = ""


# =============================================================================
# Perturbation definitions
# =============================================================================

def _compute_one(
    hw_name: str,
    base_constraints: DeploymentConstraints,
    base_loss: float,
    perturbed_constraints: DeploymentConstraints,
    constraint_name: str,
    original_str: str,
    perturbed_str: str,
    perturbation_desc: str,
) -> ShadowPrice:
    """Run optimizer with perturbed constraints and compute shadow price."""
    result = optimize(hw_name, perturbed_constraints)
    perturbed_loss = result.optimal.predicted_loss if result.optimal else base_loss
    delta = perturbed_loss - base_loss
    delta_pct = (delta / base_loss) * 100 if base_loss > 0 else 0

    if abs(delta_pct) < 0.01:
        interp = "No meaningful quality change — this constraint is not binding."
    elif delta < 0:
        interp = f"Relaxing this constraint improves predicted quality by {abs(delta_pct):.2f}%."
    else:
        interp = f"Tightening this constraint would worsen predicted quality by {delta_pct:.2f}%."

    return ShadowPrice(
        constraint=constraint_name,
        original_value=original_str,
        perturbed_value=perturbed_str,
        perturbation_desc=perturbation_desc,
        original_loss=round(base_loss, 4),
        perturbed_loss=round(perturbed_loss, 4),
        delta_loss=round(delta, 4),
        delta_loss_pct=round(delta_pct, 2),
        interpretation=interp,
    )


def _decide_alignment(
    ev,
    base_loss: float,
    base_tps: float,
    base_tbt: float,
    base_mem: float,
    d_loss_pct: float,
    d_tps_pct: float,
    d_tbt_pct: float,
    d_mem_pct: float,
    constraints,
    base_arch=None,
):
    """Decide accepted / rejected / neutral by mimicking the optimizer's
    weighted-score sort, then collapsing within the model's own uncertainty
    band.

    Returns (decision, reason).

    Rationale: the prior shadow_prices module used two independent
    decision rules ("d_loss < -0.05% and d_tbt < 5% → accepted",
    "d_loss < 0.3% and any of TPS↑/TBT↓/MEM↓ > 5% → accepted") that were
    both inconsistent with the optimizer's actual sort key. That made
    arch_dim_shadow_prices report e.g. "Enable FP8 FFN — accepted" while
    the optimizer kept bf16 FFN, which is the exact contradiction the
    demo audit surfaced. Here we evaluate moves with the same weights
    the picker uses, so the labels track the actual selection.
    """
    if not getattr(ev, "meets_constraints", True):
        return "rejected", "; ".join(getattr(ev, "constraint_violations", [])[:2]) or "infeasible"

    profile = getattr(constraints, "objective_profile", "balanced") if constraints else "balanced"
    weights = OBJECTIVE_PROFILES.get(profile, OBJECTIVE_PROFILES.get("balanced", {}))
    w_loss = float(weights.get("loss", 1.0))
    w_tbt = float(weights.get("tbt", 0.0))
    w_tps = float(weights.get("tps", 0.0))
    w_mem = float(weights.get("mem", 0.0))
    w_params = float(weights.get("params", 0.0))
    w_ttft = float(weights.get("ttft", 0.0))

    # Weighted score delta. Lower is better, so we expect:
    #   Δloss positive → bad; Δtps positive → good (subtract); Δtbt
    #   positive → bad; Δmem positive → bad.
    # All deltas are in %; scale loss term by ~10× because predicted-loss
    # uncertainty is ~3% while throughput-side measurement noise is ~30%,
    # i.e. one loss-percent moves the picker ten times more than one
    # throughput-percent. This matches the empirical sort key behaviour.
    score = (
        w_loss * 10.0 * d_loss_pct
        + w_tbt * d_tbt_pct
        + w_mem * d_mem_pct
        - w_tps * d_tps_pct
    )

    # Uncertainty band — collapse within the model's own noise.
    unc_pct = float(getattr(getattr(ev, "quality", None), "uncertainty_total", 0.03) or 0.03) * 100
    band = max(0.5, 0.25 * unc_pct * w_loss * 10.0)

    if score < -band:
        if d_loss_pct < -0.05:
            reason = (
                f"Improves quality by {abs(d_loss_pct):.2f}% with acceptable "
                f"throughput/memory cost (weighted score Δ={score:+.2f})"
            )
        else:
            reason = (
                f"Improves on throughput/memory at acceptable quality cost "
                f"(Δloss={d_loss_pct:+.2f}%, weighted score Δ={score:+.2f})"
            )
        return "accepted", reason
    if score > band:
        if d_loss_pct > 0.05:
            reason = (
                f"Worsens quality by {d_loss_pct:.2f}% beyond what throughput "
                f"gains compensate (weighted score Δ={score:+.2f})"
            )
        else:
            reason = (
                f"Net worse under the active profile "
                f"(Δloss={d_loss_pct:+.2f}%, weighted score Δ={score:+.2f})"
            )
        return "rejected", reason
    return "neutral", (
        f"Within the {unc_pct:.1f}% uncertainty band on quality "
        f"(weighted score Δ={score:+.2f})"
    )


def compute_shadow_prices(
    hw_name: str,
    constraints: DeploymentConstraints,
    base_result: OptimizationResult,
) -> ShadowPriceReport:
    """
    Compute shadow prices by perturbation for the standard set of constraints.

    Perturbations:
      1. TBT budget +20%
      2. TTFT budget +20%
      3. Memory budget (relax param tolerance to +25%)
      4. Training tokens 2× (does this change the optimal arch?)
      5. Context length halved
      6. Wider param band (±25% instead of ±15%)
    """
    if base_result.optimal is None:
        return ShadowPriceReport(hardware=hw_name)

    base_loss = base_result.optimal.predicted_loss
    prices = []

    # 1. TBT budget +20%
    if constraints.serving_tbt_ms is not None:
        pc = copy.deepcopy(constraints)
        pc.serving_tbt_ms = constraints.serving_tbt_ms * 1.2
        prices.append(_compute_one(
            hw_name, constraints, base_loss, pc,
            "serving_tbt_ms",
            f"{constraints.serving_tbt_ms}ms",
            f"{pc.serving_tbt_ms}ms",
            "+20% TBT budget",
        ))

    # 2. TTFT budget +20%
    if constraints.serving_ttft_ms is not None:
        pc = copy.deepcopy(constraints)
        pc.serving_ttft_ms = constraints.serving_ttft_ms * 1.2
        prices.append(_compute_one(
            hw_name, constraints, base_loss, pc,
            "serving_ttft_ms",
            f"{constraints.serving_ttft_ms}ms",
            f"{pc.serving_ttft_ms}ms",
            "+20% TTFT budget",
        ))

    # 3. Wider param tolerance
    pc = copy.deepcopy(constraints)
    pc.param_tolerance = 0.25
    prices.append(_compute_one(
        hw_name, constraints, base_loss, pc,
        "param_tolerance",
        f"±{constraints.param_tolerance*100:.0f}%",
        f"±{pc.param_tolerance*100:.0f}%",
        "Widen parameter band to ±25%",
    ))

    # 4. 2× training tokens
    pc = copy.deepcopy(constraints)
    pc.training_tokens = constraints.training_tokens * 2
    prices.append(_compute_one(
        hw_name, constraints, base_loss, pc,
        "training_tokens",
        f"{constraints.training_tokens/1e12:.1f}T",
        f"{pc.training_tokens/1e12:.1f}T",
        "2× training tokens",
    ))

    # 5. Halved context length
    pc = copy.deepcopy(constraints)
    pc.context_length = constraints.context_length // 2
    prices.append(_compute_one(
        hw_name, constraints, base_loss, pc,
        "context_length",
        f"{constraints.context_length}",
        f"{pc.context_length}",
        "Half context length (reduces KV cache pressure)",
    ))

    # 6. No TBT constraint (remove serving constraint entirely)
    if constraints.serving_tbt_ms is not None:
        pc = copy.deepcopy(constraints)
        pc.serving_tbt_ms = None
        pc.serving_ttft_ms = None
        prices.append(_compute_one(
            hw_name, constraints, base_loss, pc,
            "serving_constraints",
            f"TBT≤{constraints.serving_tbt_ms}ms, TTFT≤{constraints.serving_ttft_ms}ms",
            "None",
            "Remove all serving constraints (training-only optimization)",
        ))

    # Architecture-dimension shadow prices
    arch_dim = compute_arch_dim_shadow_prices(hw_name, constraints, base_result)

    return ShadowPriceReport(
        prices=prices,
        arch_dim_prices=arch_dim,
        base_loss=round(base_loss, 4),
        hardware=hw_name,
    )


def compute_arch_dim_shadow_prices(
    hw_name: str,
    constraints: DeploymentConstraints,
    base_result: OptimizationResult,
) -> List[ArchDimShadowPrice]:
    """
    Compute architecture-dimension shadow prices.

    For each dimension of the optimal architecture, perturb by a small amount
    and measure the marginal quality/throughput/memory impact.

    Perturbations:
      - +1/-1 layer
      - +256/-256 d_model
      - +1/-1 KV head (if GQA)
      - +512/-512 FFN dimension
      - Enable FP8 FFN (if currently BF16)
      - Enable INT8 KV cache (if currently 16-bit)
      - Enable INT4 KV cache (if currently 16-bit)
    """
    if base_result.optimal is None:
        return []

    opt = base_result.optimal
    base = opt.arch
    base_loss = opt.predicted_loss
    base_tps = opt.training_tps
    base_tbt = opt.serving_tbt_ms
    base_mem = opt.memory_per_gpu_gb

    results = []

    # Define perturbations as (field, delta, description)
    perturbations = [
        ("n_layers", +1, "+1 layer"),
        ("n_layers", -1, "-1 layer"),
        ("d_model", +256, "+256 d_model"),
        ("d_model", -256, "-256 d_model"),
        ("ffn_dim", +512, "+512 FFN dim"),
        ("ffn_dim", -512, "-512 FFN dim"),
    ]

    # KV head perturbations (only if GQA)
    if base.n_kv_heads < base.n_heads:
        if base.n_kv_heads > 1:
            perturbations.append(("n_kv_heads", -1, "-1 KV head"))
        if base.n_kv_heads < base.n_heads:
            perturbations.append(("n_kv_heads", +1, "+1 KV head"))

    for dim, delta, desc in perturbations:
        new_val = getattr(base, dim) + delta
        if new_val <= 0:
            continue

        # Build perturbed candidate
        cand = CandidateArch(
            d_model=base.d_model + (delta if dim == "d_model" else 0),
            n_layers=base.n_layers + (delta if dim == "n_layers" else 0),
            n_heads=base.n_heads,
            d_head=base.d_head,
            n_kv_heads=base.n_kv_heads + (delta if dim == "n_kv_heads" else 0),
            ffn_dim=base.ffn_dim + (delta if dim == "ffn_dim" else 0),
            vocab_size=base.vocab_size,
            weight_precision=base.weight_precision,
            ffn_precision=base.ffn_precision,
            attn_precision=copy.deepcopy(base.attn_precision),
            kv_cache_bits=base.kv_cache_bits,
        )

        # Validate basic constraints
        if cand.n_kv_heads > cand.n_heads or cand.n_kv_heads < 1:
            continue
        if cand.n_heads % cand.n_kv_heads != 0:
            continue
        if dim == "d_model" and cand.d_model % (cand.n_heads * cand.d_head) != 0:
            # d_model must equal n_heads * d_head; skip if violated
            # Instead, adjust n_heads to match
            if cand.d_model % cand.d_head != 0:
                continue
            cand.n_heads = cand.d_model // cand.d_head
            if cand.n_heads % cand.n_kv_heads != 0:
                continue

        try:
            ev = evaluate_candidate(cand, hw_name, constraints)
        except Exception:
            continue

        d_loss = ((ev.predicted_loss - base_loss) / base_loss) * 100
        d_tps = ((ev.training_tps - base_tps) / base_tps) * 100 if base_tps > 0 else 0
        d_tbt = ((ev.serving_tbt_ms - base_tbt) / base_tbt) * 100 if base_tbt > 0 else 0
        d_mem = ((ev.memory_per_gpu_gb - base_mem) / base_mem) * 100 if base_mem > 0 else 0

        # v1-fix demo-audit: align "decision" with the optimizer's actual
        # sort key under the active profile, instead of an independent rule
        # that contradicts what got picked. The optimizer minimizes a
        # weighted score of (loss, tbt, ttft, tps, mem, params); a move is
        # "accepted" only if it would improve that weighted score by more
        # than the quality model's uncertainty band. Otherwise the user
        # sees "+96% TPS — accepted" while the picker stays on bf16 FFN.
        decision, reason = _decide_alignment(
            ev, base_loss, base_tps, base_tbt, base_mem,
            d_loss, d_tps, d_tbt, d_mem,
            constraints, base_arch=base,
        )

        results.append(ArchDimShadowPrice(
            dimension=dim,
            change_desc=desc,
            base_value=getattr(base, dim),
            perturbed_value=new_val,
            delta_loss_pct=round(d_loss, 2),
            delta_train_tps_pct=round(d_tps, 2),
            delta_tbt_pct=round(d_tbt, 2),
            delta_mem_pct=round(d_mem, 2),
            decision=decision,
            reason=reason,
            feasible=ev.meets_constraints,
        ))

    # Precision perturbations
    prec_perturbations = []
    if base.ffn_precision == "bf16":
        prec_perturbations.append(("ffn_precision", "fp8", "Enable FP8 FFN"))
    if base.kv_cache_bits == 16:
        prec_perturbations.append(("kv_cache_bits", 8, "Enable INT8 KV cache"))
        prec_perturbations.append(("kv_cache_bits", 4, "Enable INT4 KV cache"))
    elif base.kv_cache_bits == 8:
        prec_perturbations.append(("kv_cache_bits", 4, "Downgrade to INT4 KV cache"))

    for dim, new_val, desc in prec_perturbations:
        cand = CandidateArch(
            d_model=base.d_model, n_layers=base.n_layers,
            n_heads=base.n_heads, d_head=base.d_head,
            n_kv_heads=base.n_kv_heads, ffn_dim=base.ffn_dim,
            vocab_size=base.vocab_size,
            weight_precision=base.weight_precision,
            ffn_precision=new_val if dim == "ffn_precision" else base.ffn_precision,
            attn_precision=copy.deepcopy(base.attn_precision),
            kv_cache_bits=new_val if dim == "kv_cache_bits" else base.kv_cache_bits,
        )

        try:
            ev = evaluate_candidate(cand, hw_name, constraints)
        except Exception:
            continue

        d_loss = ((ev.predicted_loss - base_loss) / base_loss) * 100
        d_tps = ((ev.training_tps - base_tps) / base_tps) * 100 if base_tps > 0 else 0
        d_tbt = ((ev.serving_tbt_ms - base_tbt) / base_tbt) * 100 if base_tbt > 0 else 0
        d_mem = ((ev.memory_per_gpu_gb - base_mem) / base_mem) * 100 if base_mem > 0 else 0

        # v1-fix demo-audit: route precision perturbations through the same
        # decision rule as architectural perturbations, so the report is
        # internally consistent. The previous "fp8 always accepted if d_loss
        # < 0.3% and any throughput axis improves >5%" rule was orthogonal
        # to the optimizer's score and is exactly what made the report tell
        # users "Enable FP8 FFN — accepted" while shipping bf16 FFN.
        decision, reason = _decide_alignment(
            ev, base_loss, base_tps, base_tbt, base_mem,
            d_loss, d_tps, d_tbt, d_mem,
            constraints, base_arch=base,
        )

        results.append(ArchDimShadowPrice(
            dimension=dim,
            change_desc=desc,
            base_value=getattr(base, dim) if hasattr(base, dim) else 0,
            perturbed_value=new_val if isinstance(new_val, int) else 0,
            delta_loss_pct=round(d_loss, 2),
            delta_train_tps_pct=round(d_tps, 2),
            delta_tbt_pct=round(d_tbt, 2),
            delta_mem_pct=round(d_mem, 2),
            decision=decision,
            reason=reason,
            feasible=ev.meets_constraints,
        ))

    return results


def shadow_prices_to_json(report: ShadowPriceReport) -> dict:
    """Convert shadow price report to JSON-serializable dict."""
    return {
        "base_loss": report.base_loss,
        "hardware": report.hardware,
        "prices": [
            {
                "constraint": p.constraint,
                "original_value": p.original_value,
                "perturbed_value": p.perturbed_value,
                "perturbation_desc": p.perturbation_desc,
                "original_loss": p.original_loss,
                "perturbed_loss": p.perturbed_loss,
                "delta_loss": p.delta_loss,
                "delta_loss_pct": p.delta_loss_pct,
                "interpretation": p.interpretation,
            }
            for p in report.prices
        ],
        "arch_dim_prices": [
            {
                "dimension": a.dimension,
                "change": a.change_desc,
                "base_value": a.base_value,
                "perturbed_value": a.perturbed_value,
                "delta_loss_pct": a.delta_loss_pct,
                "delta_train_tps_pct": a.delta_train_tps_pct,
                "delta_tbt_pct": a.delta_tbt_pct,
                "delta_mem_pct": a.delta_mem_pct,
                "decision": a.decision,
                "reason": a.reason,
                "feasible": a.feasible,
            }
            for a in report.arch_dim_prices
        ],
    }
