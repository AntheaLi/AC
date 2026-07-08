"""Wave 18h — ladder planning: the experiment plan AC generates, priced by
AC's own throughput model. No training is run; the output is a *quote*.

Given a target architecture decision (two candidate families at a target
scale/tokens/context on a hardware target), the planner:

  1. Scores both arms at target scale and computes the PAIRED decision
     sigma (correlated errors cancel — see quality_model.
     paired_loss_uncertainty). If |Δloss| already exceeds z·σ, the decision
     is resolvable from priors and the plan is empty.
  2. Otherwise identifies the residual terms that dominate the paired
     sigma — those are what a ladder must constrain.
  3. Proposes μP-style scaled-down paired runs. Each ladder pair measures
     the differing terms with noise σ_run = √2 · run_noise_floor (two real
     runs) inflated by a scale-transfer factor that widens with the
     N-distance to the target (architecture residual deltas transfer well
     across ~1 decade of scale; the widening encodes the residual risk that
     they don't).
  4. Greedily adds the run with the best marginal posterior shrink per
     GPU-day until the decision resolves at z sigma, the budget runs out,
     or marginal information vanishes. GPU-days come from AC's throughput
     model for the actual arm architectures at ladder scale.
  5. Emits plan.md, plan.json, and measurement-template JSONL rows in
     ac-auto-calibrate's ingestion format, so executing the plan feeds
     straight back into a calibration pack.

Scale-transfer model (documented, deliberately simple):
    sigma_effective(run at N_l, target N_t)
        = sqrt(2)·floor · (1 + kappa · log10(N_t / N_l))
with floor = paired_decision.run_noise_floor_pct and kappa = 0.5 by
default: a pair 1 decade below target is ~1.5x noisier as evidence about
the target than a pair at target scale; 2 decades ~2x.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from .quality_model import (
        ArchConfig as QArch, estimate_quality, paired_loss_uncertainty,
        load_quality_constants, run_noise_floor_pct)
    from .throughput_model import ArchConfig as TArch, throughput
except ImportError:  # pragma: no cover
    from quality_model import (
        ArchConfig as QArch, estimate_quality, paired_loss_uncertainty,
        load_quality_constants, run_noise_floor_pct)
    from throughput_model import ArchConfig as TArch, throughput


# Canonical dense shapes by active-parameter size (billions). Chosen to be
# tile-aligned and within published aspect-ratio practice; the planner's
# job is information accounting, not shape optimization, so one sane shape
# per size is enough.
_CANON_SHAPES = {
    0.25: dict(d_model=1024, n_layers=16, n_heads=8, d_head=128, ffn_dim=4096),
    0.5:  dict(d_model=1536, n_layers=18, n_heads=12, d_head=128, ffn_dim=6144),
    1.0:  dict(d_model=2048, n_layers=22, n_heads=16, d_head=128, ffn_dim=8192),
    3.0:  dict(d_model=3072, n_layers=26, n_heads=24, d_head=128, ffn_dim=12288),
    7.0:  dict(d_model=4096, n_layers=32, n_heads=32, d_head=128, ffn_dim=14336),
    13.0: dict(d_model=5120, n_layers=40, n_heads=40, d_head=128, ffn_dim=17920),
}

ARM_FAMILIES = ("dense", "moe", "hybrid", "moe_hybrid", "local_global", "mla")


def _nearest_shape(params_b: float) -> Dict[str, int]:
    key = min(_CANON_SHAPES, key=lambda k: abs(math.log(k / max(1e-6, params_b))))
    return dict(_CANON_SHAPES[key])


def _arm_arch(family: str, params_b: float, vocab_size: int = 128256,
              context: int = 8192) -> QArch:
    """Representative architecture for a family at a given active size."""
    s = _nearest_shape(params_b)
    kw: Dict[str, Any] = dict(
        d_model=s["d_model"], n_layers=s["n_layers"], n_heads=s["n_heads"],
        d_head=s["d_head"], n_kv_heads=8, ffn_dim=s["ffn_dim"],
        vocab_size=vocab_size)
    if family == "moe":
        kw.update(model_type="moe", ffn_dim=s["ffn_dim"] // 8,
                  moe_config={"n_experts": 64, "top_k": 8,
                              "expert_dim": s["ffn_dim"] // 8})
    elif family == "hybrid":
        n_state = int(round(s["n_layers"] * 0.75))
        kw.update(model_type="hybrid",
                  state_config={"enabled": True, "state_type": "mamba2",
                                "d_state": 128, "state_layers": n_state,
                                "attention_layers": s["n_layers"] - n_state})
    elif family == "moe_hybrid":
        n_state = int(round(s["n_layers"] * 0.75))
        kw.update(model_type="hybrid", ffn_dim=s["ffn_dim"] // 8,
                  moe_config={"n_experts": 64, "top_k": 8,
                              "expert_dim": s["ffn_dim"] // 8},
                  state_config={"enabled": True, "state_type": "mamba2",
                                "d_state": 128, "state_layers": n_state,
                                "attention_layers": s["n_layers"] - n_state})
    elif family == "local_global":
        kw.update(local_window=4096, local_attention_fraction=0.75)
    elif family == "mla":
        kw.update(attention_type="mla", mla_latent_dim=512,
                  mla_q_latent_dim=1536, mla_rope_head_dim=64)
    elif family != "dense":
        raise ValueError(f"unknown arm family {family!r}; "
                        f"supported: {ARM_FAMILIES}")
    return QArch(**kw)


def _arm_tarch(family: str, params_b: float, context: int) -> TArch:
    s = _nearest_shape(params_b)
    kw: Dict[str, Any] = dict(
        d_model=s["d_model"], n_layers=s["n_layers"], n_heads=s["n_heads"],
        d_head=s["d_head"], n_kv_heads=8, ffn_dim=s["ffn_dim"],
        batch_size=8, seq_len=min(context, 8192),
        precision="bf16", kv_precision="bf16")
    if family in ("moe", "moe_hybrid"):
        kw.update(ffn_dim=s["ffn_dim"] // 8,
                  moe_config={"n_experts": 64, "top_k": 8,
                              "expert_dim": s["ffn_dim"] // 8,
                              "precision": "bf16"})
    if family in ("hybrid", "moe_hybrid"):
        n_state = int(round(s["n_layers"] * 0.75))
        kw.update(state_config={"d_state": 128, "state_expansion": 2,
                                "n_heads": s["n_heads"], "d_head": 64},
                  layer_type_list=(["state"] * n_state
                                   + ["attention"] * (s["n_layers"] - n_state)))
    if family == "local_global":
        kw.update(local_window=4096,
                  n_local_attn_layers=int(round(s["n_layers"] * 0.75)))
    if family == "mla":
        kw.update(attention_type="mla", mla_kv_latent_dim=512,
                  mla_q_latent_dim=1536, mla_rope_head_dim=64)
    return TArch(**kw)


def _gpu_days(family: str, params_b: float, tokens: float, hardware: str,
              tp: int = 8) -> float:
    """Price one training run with AC's own throughput model."""
    try:
        arch = _arm_tarch(family, params_b, 8192)
        # Wave 19 (P2-b): MoE arms must be priced at a legal EP (EP >= 2);
        # EP lays over DP so it does not change the per-replica GPU count.
        ep = 8 if (getattr(arch, "moe_config", None)) else 1
        r = throughput(arch, hardware, tp_degree=tp, pp_degree=1,
                       decode_kv_len=8192, prefill_seq_len=8192,
                       microbatches=8, ep_degree=ep)
        tps = float(r.training_throughput_tokens_per_sec or 0.0)
        if tps > 0:
            return tokens / tps / 86400.0 * tp
    except Exception:
        pass
    # Fallback: 6ND at 40% MFU of the effective 495 TF bf16 H100 baseline.
    flops = 6.0 * params_b * 1e9 * tokens
    return flops / (495e12 * 0.40) / 86400.0


@dataclass
class LadderRun:
    size_b: float
    tokens_t: float
    arm_a: str
    arm_b: str
    gpu_days_pair: float
    transfer_factor: float
    sigma_evidence_pct: float     # effective noise of this pair as evidence
    marginal_sigma_after_pct: float
    cumulative_gpu_days: float


@dataclass
class LadderPlan:
    resolvable_now: bool
    delta_pct: float
    sigma_prior_pct: float
    sigma_floor_pct: float
    z: float
    driving_terms: List[Tuple[str, float]]
    runs: List[LadderRun] = field(default_factory=list)
    sigma_final_pct: float = 0.0
    resolves: bool = False
    verdict: str = ""


def plan_ladder(
    arm_a: str,
    arm_b: str,
    target_params_b: float,
    target_tokens_t: float,
    context: int = 8192,
    hardware: str = "h100",
    ladder_sizes_b: Optional[List[float]] = None,
    seeds_per_point: int = 1,
    max_gpu_days: Optional[float] = None,
    z: float = 2.0,
    kappa: float = 0.5,
    ladder_tokens_per_param_cap: float = 100.0,
) -> LadderPlan:
    # Wave 20 (feedback #5): read the floor from the shared accessor so the
    # planner and the greenfield picker can never codify different values.
    floor_pct = run_noise_floor_pct()
    ladder_sizes_b = ladder_sizes_b or [0.5, 1.0, 3.0]
    tokens_per_param = target_tokens_t * 1e12 / (target_params_b * 1e9)
    # Wave 19 (P2-b): cap the rung over-training ratio. Pricing every rung
    # at the TARGET's tokens/param (e.g. 2857 t/p for 7B@20T) made a
    # 0.5-3B ladder cost ~40k GPU-days — two orders of magnitude beyond any
    # real mu-P-style ladder practice (rungs at 20-100 t/p, transfer the
    # exponents). The cap is a design default, not a physics claim; runs
    # above it carry a transfer-risk annotation instead of an absurd bill.
    tokens_per_param_capped = min(tokens_per_param, ladder_tokens_per_param_cap)
    regime_capped = tokens_per_param_capped < tokens_per_param

    # 1. Target-scale decision under priors.
    tr = {"training_tokens": int(target_tokens_t * 1e12)}
    w = {"context_length": int(context)}
    qa = estimate_quality(_arm_arch(arm_a, target_params_b), tr, workload_spec=w)
    qb = estimate_quality(_arm_arch(arm_b, target_params_b), tr, workload_spec=w)
    p = paired_loss_uncertainty(qa, qb)
    delta_pct = abs(qb.predicted_loss - qa.predicted_loss) \
        / max(1e-9, qa.predicted_loss) * 100.0
    sigma_prior_pct = p["sigma_rel"] * 100.0
    driving = sorted(
        ((t, d["sigma_pair"] * 100.0) for t, d in p["per_term"].items()
         if d["sigma_pair"] > 1e-6),
        key=lambda kv: -kv[1])

    plan = LadderPlan(
        resolvable_now=delta_pct > z * sigma_prior_pct,
        delta_pct=round(delta_pct, 3),
        sigma_prior_pct=round(sigma_prior_pct, 3),
        sigma_floor_pct=floor_pct,
        z=z,
        driving_terms=[(t, round(s, 3)) for t, s in driving[:5]],
    )
    if plan.resolvable_now:
        plan.resolves = True
        plan.sigma_final_pct = plan.sigma_prior_pct
        plan.verdict = (
            f"Resolvable from priors: |Δ|={delta_pct:.2f}% > "
            f"{z:.0f}σ={z * sigma_prior_pct:.2f}%. No runs needed.")
        return plan

    # Floor check BEFORE proposing runs: if the predicted gap sits below
    # z × run-noise floor, no ladder of this design can resolve it — the
    # honest plan is zero runs plus the advice to decide on other axes.
    if delta_pct <= z * floor_pct:
        plan.sigma_final_pct = plan.sigma_prior_pct
        plan.resolves = False
        plan.verdict = (
            f"UNRESOLVABLE BY THIS LADDER: |Δ|={delta_pct:.2f}% is below "
            f"the {z:.0f}× run-noise floor ({z * floor_pct:.2f}%). The two "
            f"arms are quality-equivalent at the resolution any experiment "
            f"of this shape can reach — decide on throughput/memory/cost, "
            f"or change the experiment (more seeds per point averages the "
            f"floor down by 1/sqrt(seeds); a longer target-scale run "
            f"tightens it directly). No runs proposed.")
        return plan

    # 2. Candidate ladder pairs (size × seeds), priced by throughput.
    candidates: List[Dict[str, Any]] = []
    for size in sorted(ladder_sizes_b):
        run_tokens = tokens_per_param_capped * size * 1e9  # capped regime (see above)
        gpu_days_pair = (_gpu_days(arm_a, size, run_tokens, hardware)
                         + _gpu_days(arm_b, size, run_tokens, hardware))
        transfer = 1.0 + kappa * math.log10(
            max(1.0, target_params_b / size))
        if regime_capped:
            # Rung trained at a lower tokens/param than the target: the
            # data-regime mismatch adds transfer risk on top of the scale
            # gap. Half a kappa per decade of token-ratio distance.
            transfer *= 1.0 + 0.5 * kappa * math.log10(
                max(1.0, tokens_per_param / tokens_per_param_capped))
        sigma_evidence = math.sqrt(2.0) * floor_pct * transfer
        for _seed in range(max(1, seeds_per_point)):
            candidates.append(dict(
                size_b=size, tokens_t=run_tokens / 1e12,
                gpu_days=gpu_days_pair, transfer=transfer,
                sigma_evidence=sigma_evidence))

    # 3. Greedy accumulation. The measurable quantity is the sum of the
    # differing-term contributions — the same components that make up the
    # paired sigma above the run-noise floor. Evidence shrinks that
    # reducible component; the floor is irreducible.
    reducible_var = max(0.0, (sigma_prior_pct ** 2) - (floor_pct ** 2))
    inv_var_prior = (1.0 / reducible_var) if reducible_var > 0 else float("inf")
    inv_var = inv_var_prior
    cum_days = 0.0
    for cand in sorted(candidates, key=lambda c: (
            # information per GPU-day, best first
            -(1.0 / c["sigma_evidence"] ** 2) / max(1e-9, c["gpu_days"]))):
        if max_gpu_days is not None and cum_days + cand["gpu_days"] > max_gpu_days:
            continue
        inv_var_new = inv_var + 1.0 / (cand["sigma_evidence"] ** 2)
        sigma_after = math.sqrt(1.0 / inv_var_new + floor_pct ** 2)
        sigma_before = math.sqrt(1.0 / inv_var + floor_pct ** 2)
        if sigma_before - sigma_after < 0.005:  # <0.005% marginal shrink
            continue
        inv_var = inv_var_new
        cum_days += cand["gpu_days"]
        plan.runs.append(LadderRun(
            size_b=cand["size_b"], tokens_t=round(cand["tokens_t"], 3),
            arm_a=arm_a, arm_b=arm_b,
            gpu_days_pair=round(cand["gpu_days"], 1),
            transfer_factor=round(cand["transfer"], 2),
            sigma_evidence_pct=round(cand["sigma_evidence"], 3),
            marginal_sigma_after_pct=round(sigma_after, 3),
            cumulative_gpu_days=round(cum_days, 1)))
        if delta_pct > z * sigma_after:
            break

    plan.sigma_final_pct = round(
        math.sqrt(1.0 / inv_var + floor_pct ** 2), 3) if inv_var != float("inf") \
        else floor_pct
    plan.resolves = delta_pct > z * plan.sigma_final_pct
    if plan.resolves:
        plan.verdict = (
            f"Resolvable with {len(plan.runs)} paired run(s), "
            f"~{plan.runs[-1].cumulative_gpu_days:.0f} GPU-days total: "
            f"σ {sigma_prior_pct:.2f}% → {plan.sigma_final_pct:.2f}%, "
            f"|Δ|={delta_pct:.2f}% > {z:.0f}σ.")
    else:
        floor_bound = z * floor_pct
        if delta_pct <= floor_bound:
            plan.verdict = (
                f"UNRESOLVABLE BY THIS LADDER: |Δ|={delta_pct:.2f}% is below "
                f"the {z:.0f}× run-noise floor ({floor_bound:.2f}%). The two "
                f"arms are equivalent at the resolution any experiment of "
                f"this shape can reach — decide on throughput/memory, or "
                f"reduce seed noise (more seeds, longer runs).")
        else:
            plan.verdict = (
                f"Ladder shrinks σ to {plan.sigma_final_pct:.2f}% but "
                f"|Δ|={delta_pct:.2f}% still < {z:.0f}σ. Add target-scale "
                f"runs (transfer factor 1.0) or raise the budget.")
    return plan


def describe_arm(family: str) -> str:
    """Wave 19 (P2-b): one-line statement of the topology an arm assumes.

    plan-ladder previously said "--arm-b moe" without ever stating WHICH
    MoE it priced — the reader had no way to know the plan assumed
    64 experts x top-8. Surfaced in plan.md and plan.json.
    """
    if family == "moe":
        return ("MoE: 64 experts x top-8, expert_dim = dense_ffn/8, "
                "no shared expert, EP=8 (over DP)")
    if family == "hybrid":
        return "Hybrid: 75% state layers (mamba2-class, d_state=128) + 25% attention"
    if family == "moe_hybrid":
        return ("MoE-hybrid: 64x top-8 experts + 75% state layers "
                "(combination of the two arms above)")
    if family == "local_global":
        return "Local:global 3:1 interleave, window 4096"
    if family == "mla":
        return "MLA: c_kv=512, c_q=1536, d_rope=64"
    return "Dense GQA-8 SwiGLU transformer"


def render_plan_md(plan: LadderPlan, arm_a: str, arm_b: str,
                   target_params_b: float, target_tokens_t: float,
                   context: int, hardware: str) -> str:
    lines = [
        f"# Ladder plan — {arm_a} vs {arm_b} @ "
        f"{target_params_b:g}B / {target_tokens_t:g}T, ctx {context:,}, {hardware}",
        "",
        f"Predicted |Δloss|: **{plan.delta_pct:.2f}%** | paired prior σ: "
        f"**{plan.sigma_prior_pct:.2f}%** | resolution bar: {plan.z:.0f}σ | "
        f"run-noise floor: {plan.sigma_floor_pct:.2f}%",
        "",
        "Arm assumptions (the plan prices THESE topologies):",
        "",
        f"- `{arm_a}`: {describe_arm(arm_a)}",
        f"- `{arm_b}`: {describe_arm(arm_b)}",
        "",
        "Driving uncertainty terms (paired — shared errors already cancelled):",
        "",
    ]
    for t, s in plan.driving_terms:
        lines.append(f"- `{t}`: {s:.2f}%")
    lines.append("")
    if plan.resolvable_now:
        lines.append(f"**{plan.verdict}**")
        return "\n".join(lines)
    lines.append("## Proposed runs (greedy, information per GPU-day)")
    lines.append("")
    lines.append("| # | size | tokens | arms | GPU-days (pair) | transfer | σ after |")
    lines.append("|---|---|---|---|---:|---:|---:|")
    for i, r in enumerate(plan.runs, 1):
        lines.append(
            f"| {i} | {r.size_b:g}B | {r.tokens_t:g}T | {r.arm_a} vs {r.arm_b} "
            f"| {r.gpu_days_pair:g} | {r.transfer_factor:.2f}x "
            f"| {r.marginal_sigma_after_pct:.2f}% |")
    lines.append("")
    lines.append(f"**{plan.verdict}**")
    lines.append("")
    lines.append(
        "Execute each row as TWO runs (both arms, same data order & seed), "
        "then feed the measurement JSONL back through `ac-auto-calibrate "
        "fit` / `fit-pairs`. Ladder points use the target's tokens-per-"
        "param ratio so the data-sufficiency regime matches the target.")
    return "\n".join(lines)


def write_measurement_templates(plan: LadderPlan, arm_a: str, arm_b: str,
                                hardware: str, context: int,
                                out_path: str) -> None:
    rows = []
    for i, r in enumerate(plan.runs, 1):
        for arm in (arm_a, arm_b):
            rows.append({
                "id": f"ladder_{i}_{arm}_{r.size_b:g}b",
                "hardware": hardware,
                "architecture_family": arm,
                "model_type": arm,
                "active_params_b": r.size_b,
                "total_params_b": None,
                "training_tokens": r.tokens_t,
                "context_length": context,
                "predicted_loss": None,
                "observed_loss": "<FILL AFTER RUN>",
                "pair_id": f"ladder_{i}",
                "notes": "Wave 18h ladder-plan template; keep seed and "
                         "data order identical across the pair.",
            })
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run_plan_ladder(args) -> int:
    plan = plan_ladder(
        arm_a=args.arm_a, arm_b=args.arm_b,
        target_params_b=float(args.params),
        target_tokens_t=float(args.tokens),
        context=int(args.context), hardware=args.hardware,
        ladder_sizes_b=([float(x) for x in str(args.ladder).split(",")]
                        if args.ladder else None),
        seeds_per_point=int(args.seeds),
        max_gpu_days=(float(args.max_gpu_days)
                      if args.max_gpu_days is not None else None),
        z=float(args.z),
        ladder_tokens_per_param_cap=float(
            getattr(args, "ladder_tokens_per_param", None) or 100.0))
    os.makedirs(args.out, exist_ok=True)
    md = render_plan_md(plan, args.arm_a, args.arm_b, float(args.params),
                        float(args.tokens), int(args.context), args.hardware)
    with open(os.path.join(args.out, "plan.md"), "w") as f:
        f.write(md)
    with open(os.path.join(args.out, "plan.json"), "w") as f:
        json.dump({
            "arm_a": args.arm_a, "arm_b": args.arm_b,
            "arm_assumptions": {args.arm_a: describe_arm(args.arm_a),
                                 args.arm_b: describe_arm(args.arm_b)},
            "target_params_b": float(args.params),
            "target_tokens_t": float(args.tokens),
            "context": int(args.context), "hardware": args.hardware,
            "delta_pct": plan.delta_pct,
            "sigma_prior_pct": plan.sigma_prior_pct,
            "sigma_final_pct": plan.sigma_final_pct,
            "z": plan.z, "resolvable_now": plan.resolvable_now,
            "resolves": plan.resolves, "verdict": plan.verdict,
            "driving_terms": plan.driving_terms,
            "runs": [vars(r) for r in plan.runs],
        }, f, indent=2)
    write_measurement_templates(
        plan, args.arm_a, args.arm_b, args.hardware, int(args.context),
        os.path.join(args.out, "ladder_measurements_template.jsonl"))
    print(plan.verdict)
    print(f"Wrote {os.path.join(args.out, 'plan.md')}")
    return 0
